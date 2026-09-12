"""Positive control: does the trust gate ADMIT and KEEP genuinely-good (real) data?

The multi-seed run found EEGNet's gate reverts ~100% of the time. That could mean either
(a) synthetic genuinely doesn't help EEGNet, or (b) the gate/pipeline is faulty for EEGNet.
This script falsifies (b): it replaces the synthetic generator with an ORACLE that emits
REAL held-out ictal windows, and checks the gate's behaviour.

Design (leakage-safe, event-level):
  * base detector trains on 50% of each fold's seizure EVENTS (scarcity 0.5),
  * the "oracle pool" = the ictal windows of the OTHER 50% of events -- real seizures the
    base never saw, same training patients (no val/test leakage),
  * run real_only(0.5) / ungated / gated(q=0.90, 0.50) with the oracle pool.

Expected if the pipeline is SOUND: the teacher scores real held-out ictal highly ->
admitted at a high rate; the augmented model improves -> the gate does NOT revert.
If the gate refuses/reverts REAL seizures (esp. for EEGNet), that is a genuine design
fault. A negative-control (Gaussian noise pool) is included to confirm the gate rejects junk.

NOTE: uses the full train/val/test pipeline (fp16 window cache), so run it when the main
multi-seed job is NOT running (shared 62 GB container cgroup would OOM otherwise).

PHASE 3 (2026-09-02). This is the ceiling experiment for the whole project: if REAL held-out
ictal from other training patients does not help patient-independent detection, no generator
that matches the real cross-patient distribution can, and the negative result is about the task
rather than about the WGAN. It therefore runs at the SAME n as the main grids by default --
folds x seeds x all three detectors -- is resumable, takes ``--splits`` so it can run on the
Phase 3 fold files, and emits the validation columns the free analysis needs
(``scripts/analyze_validation_selection.py``). ``--with-simple-baselines`` adds class_weighted
and classical_aug at the same scarcity so the registered reference exists for these cells too.
Interpretation rules are fixed in PREREGISTRATION_PHASE3.md Sec 4 before the run.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import zarr

from chbmit.splits import Split
from chbmit.scarcity import select_seizure_events, apply_scarcity_to_windows
from chbmit.datasets import (negative_sample, prefetch_windows, clear_window_cache,
                             materialize_windows)
from synthetic.trust_gate import TrustGateConfig
from experiments.training import (
    run_cell, CellSpec, TrainConfig, UNGATED_SYNTHETIC, GATED_SYNTHETIC)
from experiments.environment import record_environment

# Defaults for a local checkout. The processed store now lives in GCS
# (gs://chbmit-bench-2486a474/processed_chbmit_real/) and is reached through Vertex's /gcs mount,
# so --data-root / --device exist to point this script at it without editing the source.
DATA_ROOT = Path("data/processed_chbmit_real")
RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
BASE_FRAC = 0.5   # base detector trains on 50% of events; other 50% is the oracle pool


def _evt(res):
    """Same row schema as run_multiseed_downstream._evt, so the two CSVs analyse alike."""
    em = res.get("event_metrics", {}) or {}
    g = res.get("gate")
    ts = res.get("threshold_selection", {}) or {}
    aug = (res.get("gated_model_metrics", {}) or {}).get("event_metrics", em) if res.get(
        "reverted_to_real_only") else em
    return {"event_f1": em.get("event_f1"), "event_sensitivity": em.get("event_sensitivity"),
            "event_precision": em.get("event_precision"), "fp_per_24h": em.get("fp_per_24h"),
            "val_event_f1": ts.get("validation_event_f1"),
            "val_fp_per_24h": ts.get("validation_fp_per_24h"),
            "val_auprc": ts.get("validation_auprc"),
            "aug_event_f1": aug.get("event_f1"), "aug_fp_per_24h": aug.get("fp_per_24h"),
            "aug_event_sensitivity": aug.get("event_sensitivity"),
            "n_synth_injected": res.get("n_synth_injected"),
            "reverted_to_real_only": res.get("reverted_to_real_only"),
            "gate_admission_rate": (g or {}).get("admission_rate"),
            "gate_n_admitted": (g or {}).get("n_admitted"),
            "gate_reason": (g or {}).get("reason"),
            "gate_reference": (g or {}).get("reference"),
            "gate_selection": (g or {}).get("selection"),
            "gate_min_admitted": (g or {}).get("min_admitted")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # Full-n defaults (Phase 3): the same cells as the main grids, all three detectors.
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 2024])
    ap.add_argument("--detectors", nargs="*", default=["eegnet", "lct", "tcn"])
    ap.add_argument("--qs", nargs="*", type=float, default=[0.90, 0.50])
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="injection ratio for the oracle arms, n_synth = ratio * n_train_pos")
    ap.add_argument("--base-frac", type=float, default=BASE_FRAC,
                    help="event-level scarcity of the base detector; the complement is the oracle pool")
    ap.add_argument("--with-simple-baselines", action="store_true",
                    help="also run class_weighted and classical_aug at the base scarcity, so the "
                         "registered reference exists for these cells")
    ap.add_argument("--splits", default=str(RES / "splits/splits_seed42.json"),
                    help="split file (see scripts/make_phase3_splits.py)")
    ap.add_argument("--tag", default="", help="suffix for positive_control_gate<tag>.csv")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--with-noise-control", action="store_true")  # add Gaussian-noise negative control
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="directory holding eeg.zarr and processed_index.csv "
                         "(e.g. /gcs/chbmit-bench-2486a474/processed_chbmit_real)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # The oracle pool IS real held-out ictal, so calibrating the admission threshold on real
    # TRAINING ictal is the right reference for this experiment specifically -- the whole point
    # is "does the teacher rate unseen real seizures as highly as the ones it memorised". That is
    # the opposite of the main grids, where the same reference disabled the gate
    # (DECISION_GATE_2.md Q6), so it is set explicitly here rather than inherited from the
    # TrustGateConfig default, which is now "pool".
    ap.add_argument("--gate-reference", default="real_ictal", choices=["real_ictal", "pool"],
                    help="admission reference (default: real_ictal -- correct HERE because the "
                         "candidate pool is itself real ictal)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    store = str(data_root / "eeg.zarr")
    base_frac = float(args.base_frac)
    t0 = time.time()
    index_df = pd.read_csv(data_root / "processed_index.csv")
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(args.splits))
    print(f"[splits] {args.splits}: {sp.get('n_folds')} folds, "
          f"val_strategy={sp.get('val_strategy', 'carve')}", flush=True)
    fs = int(zarr.open_group(store, mode="r").attrs["sampling_rate"])
    cfg = TrainConfig(epochs=args.epochs, device=args.device, num_workers=0)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"positive_control_gate{args.tag}.csv"
    record_environment(OUT / f"run_environment_positive_control{args.tag}.json")

    # Resume: a (fold, seed, detector) block is complete when every arm of this invocation is
    # present. Computed from the grid, never a literal (see run_multiseed_downstream.n_conds).
    n_expected = (2 + len(args.qs) + (2 if args.with_simple_baselines else 0)
                  + (1 if args.with_noise_control else 0))
    rows, done_blocks = [], set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        for (fo, se, det), grp in prev.groupby(["fold", "seed", "detector"]):
            if len(grp) >= n_expected:
                rows.extend(grp.to_dict("records"))
                done_blocks.add((int(fo), int(se), str(det)))
        print(f"[resume] {len(done_blocks)} complete blocks loaded from {out_csv} "
              f"({n_expected} rows per block)", flush=True)

    for fold in args.folds:
        if all((fold, s, d) in done_blocks for s in args.seeds for d in args.detectors):
            print(f"[{time.time()-t0:5.0f}s] fold {fold} already complete; skipping", flush=True)
            continue
        f = sp["folds"][fold]
        split = Split(fold=fold, train_groups=f["train_groups"],
                      val_groups=f["val_groups"], test_groups=f["test_groups"])
        clear_window_cache()
        for grp in ("val_groups", "test_groups"):
            ids = set(index_df[index_df["group"].isin(f[grp])]["file_id"])
            prefetch_windows(win[win["file_id"].isin(ids)], store, workers=16, dtype="float16")
        train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
        train_windows = win[win["file_id"].isin(train_ids) & (~win["excluded"])]
        train_events = ev[ev["group"].isin(split.train_groups)]
        print(f"[{time.time()-t0:5.0f}s] fold {fold}: val/test prefetched", flush=True)

        for seed in args.seeds:
            if all((fold, seed, d) in done_blocks for d in args.detectors):
                continue
            # Event-level split: base events (matches run_cell's internal selection at the same
            # fraction) vs the complement, whose ictal windows are the real "oracle" pool.
            base_events = select_seizure_events(train_events, base_frac, seed,
                                                restrict_groups=split.train_groups)
            all_events = select_seizure_events(train_events, 1.0, seed,
                                               restrict_groups=split.train_groups)
            pool_events = all_events - base_events
            pool_win = train_windows[(train_windows["label"] == 1)
                                     & (train_windows["seizure_event_id"].isin(pool_events))]
            prefetch_windows(pool_win, store, workers=16, dtype="float16")
            real_pool, _ = materialize_windows(pool_win, store)
            print(f"[{time.time()-t0:5.0f}s] f{fold} s{seed}: base={len(base_events)}ev "
                  f"pool={len(pool_events)}ev -> {len(real_pool)} real held-out ictal windows",
                  flush=True)
            if len(real_pool) == 0:
                print("   (no held-out ictal; skipping seed)"); continue

            def oracle(n, s, _pool=real_pool):
                rng = np.random.default_rng(s)
                idx = rng.choice(len(_pool), size=n, replace=len(_pool) < n)
                return _pool[idx].astype("float32")

            def noise(n, s, _shape=real_pool.shape[1:]):
                return np.random.default_rng(s).standard_normal((n, *_shape)).astype("float32")

            # Prefetch the BASE TRAINING table too. Without this only val/test and the oracle
            # pool are cached, so run_cell reads every training window over the network on
            # every epoch (~65 ms/window) and the job sits at GPU 0% for hours. This rebuilds
            # exactly the table run_cell assembles internally -- same scarcity fraction, same
            # events, same seeded negative_sample -- so no wrong rows are cached.
            base_scarce = apply_scarcity_to_windows(train_windows, base_events)
            base_train = negative_sample(base_scarce, ratio=cfg.background_to_seizure_ratio,
                                         exclude_seconds=cfg.exclude_seconds_around_seizure,
                                         seed=seed)
            prefetch_windows(base_train, store, workers=16, dtype="float16")
            print(f"[{time.time()-t0:5.0f}s] f{fold} s{seed}: base train prefetched "
                  f"({len(base_train)} windows, {int((base_train['label'] == 1).sum())} ictal)",
                  flush=True)

            for det in args.detectors:
                if (fold, seed, det) in done_blocks:
                    continue

                def _spec(cond, **kw):
                    return CellSpec(fold=fold, seed=seed, scarcity_fraction=base_frac,
                                    detector=det, condition=cond, **kw)
                base = {"fold": fold, "seed": seed, "detector": det, "base_frac": base_frac,
                        "ratio": args.ratio, "n_oracle_pool": int(len(real_pool))}

                teacher = run_cell(_spec("real_only"), index_df, win, ev, store, split,
                                   cfg=cfg, return_model=True)
                tmodel = teacher.pop("model", None)
                rows.append({**base, "pool": "real_base", "condition": "real_only", "q": None, **_evt(teacher)})

                if args.with_simple_baselines:
                    for simple in ("class_weighted", "classical_aug"):
                        r = run_cell(_spec(simple), index_df, win, ev, store, split, cfg=cfg)
                        rows.append({**base, "pool": "real_base", "condition": simple, "q": None, **_evt(r)})

                r = run_cell(_spec(UNGATED_SYNTHETIC, generator="oracle_real", synthetic_ratio=args.ratio),
                             index_df, win, ev, store, split, cfg=cfg, synthetic_provider=oracle)
                rows.append({**base, "pool": "real_oracle", "condition": "ungated", "q": None, **_evt(r)})
                for q in args.qs:
                    r = run_cell(_spec(GATED_SYNTHETIC, generator="oracle_real", synthetic_ratio=args.ratio),
                                 index_df, win, ev, store, split, cfg=cfg, synthetic_provider=oracle,
                                 teacher_model=tmodel, teacher_result=copy.deepcopy(teacher),
                                 gate_cfg=TrustGateConfig(q=q, reference=args.gate_reference))
                    rows.append({**base, "pool": "real_oracle", "condition": "gated", "q": q, **_evt(r)})

                if args.with_noise_control:
                    r = run_cell(_spec(GATED_SYNTHETIC, generator="noise", synthetic_ratio=1.0),
                                 index_df, win, ev, store, split, cfg=cfg, synthetic_provider=noise,
                                 teacher_model=tmodel, teacher_result=copy.deepcopy(teacher),
                                 gate_cfg=TrustGateConfig(q=0.50, reference=args.gate_reference))
                    rows.append({**base, "pool": "noise", "condition": "gated", "q": 0.50, **_evt(r)})

                done_blocks.add((fold, seed, det))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"[{time.time()-t0:5.0f}s] DONE f{fold} s{seed} {det}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== POSITIVE CONTROL: gate behaviour on REAL held-out ictal ===", flush=True)
    show = ["fold", "seed", "detector", "pool", "condition", "q", "event_f1",
            "aug_event_f1", "gate_admission_rate", "reverted_to_real_only", "gate_reason"]
    pd.set_option("display.width", 240, "display.max_rows", 200)
    print(df[show].to_string(index=False), flush=True)
    print(f"\n[{time.time()-t0:5.0f}s] wrote {out_csv}", flush=True)
    clear_window_cache()


if __name__ == "__main__":
    main()
