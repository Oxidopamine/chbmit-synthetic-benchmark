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

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
DEVICE = "cuda"
BASE_FRAC = 0.5   # base detector trains on 50% of events; other 50% is the oracle pool


def _evt(res):
    em = res.get("event_metrics", {}) or {}
    g = res.get("gate")
    aug = (res.get("gated_model_metrics", {}) or {}).get("event_metrics", em) if res.get(
        "reverted_to_real_only") else em
    return {"event_f1": em.get("event_f1"), "fp_per_24h": em.get("fp_per_24h"),
            "aug_event_f1": aug.get("event_f1"), "aug_fp_per_24h": aug.get("fp_per_24h"),
            "reverted_to_real_only": res.get("reverted_to_real_only"),
            "gate_admission_rate": (g or {}).get("admission_rate"),
            "gate_n_admitted": (g or {}).get("n_admitted"),
            "gate_reason": (g or {}).get("reason")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    ap.add_argument("--detectors", nargs="*", default=["eegnet", "tcn"])
    ap.add_argument("--qs", nargs="*", type=float, default=[0.90, 0.50])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--with-noise-control", action="store_true")  # add Gaussian-noise negative control
    args = ap.parse_args()

    t0 = time.time()
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])
    cfg = TrainConfig(epochs=args.epochs, device=DEVICE, num_workers=0)
    out_csv = OUT / "positive_control_gate.csv"

    rows = []
    for fold in args.folds:
        f = sp["folds"][fold]
        split = Split(fold=fold, train_groups=f["train_groups"],
                      val_groups=f["val_groups"], test_groups=f["test_groups"])
        clear_window_cache()
        for grp in ("val_groups", "test_groups"):
            ids = set(index_df[index_df["group"].isin(f[grp])]["file_id"])
            prefetch_windows(win[win["file_id"].isin(ids)], STORE, workers=16, dtype="float16")
        train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
        train_windows = win[win["file_id"].isin(train_ids) & (~win["excluded"])]
        train_events = ev[ev["group"].isin(split.train_groups)]
        print(f"[{time.time()-t0:5.0f}s] fold {fold}: val/test prefetched", flush=True)

        for seed in args.seeds:
            # Event-level split: base events (matches run_cell's internal 0.5 selection)
            # vs the complement, whose ictal windows are the real "oracle" pool.
            base_events = select_seizure_events(train_events, BASE_FRAC, seed,
                                                restrict_groups=split.train_groups)
            all_events = select_seizure_events(train_events, 1.0, seed,
                                               restrict_groups=split.train_groups)
            pool_events = all_events - base_events
            pool_win = train_windows[(train_windows["label"] == 1)
                                     & (train_windows["seizure_event_id"].isin(pool_events))]
            prefetch_windows(pool_win, STORE, workers=16, dtype="float16")
            real_pool, _ = materialize_windows(pool_win, STORE)
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

            for det in args.detectors:
                def _spec(cond, **kw):
                    return CellSpec(fold=fold, seed=seed, scarcity_fraction=BASE_FRAC,
                                    detector=det, condition=cond, **kw)
                base = {"fold": fold, "seed": seed, "detector": det}

                teacher = run_cell(_spec("real_only"), index_df, win, ev, STORE, split,
                                   cfg=cfg, return_model=True)
                tmodel = teacher.pop("model", None)
                rows.append({**base, "pool": "real_base", "condition": "real_only", "q": None, **_evt(teacher)})

                r = run_cell(_spec(UNGATED_SYNTHETIC, generator="oracle_real", synthetic_ratio=1.0),
                             index_df, win, ev, STORE, split, cfg=cfg, synthetic_provider=oracle)
                rows.append({**base, "pool": "real_oracle", "condition": "ungated", "q": None, **_evt(r)})
                for q in args.qs:
                    r = run_cell(_spec(GATED_SYNTHETIC, generator="oracle_real", synthetic_ratio=1.0),
                                 index_df, win, ev, STORE, split, cfg=cfg, synthetic_provider=oracle,
                                 teacher_model=tmodel, teacher_result=copy.deepcopy(teacher),
                                 gate_cfg=TrustGateConfig(q=q))
                    rows.append({**base, "pool": "real_oracle", "condition": "gated", "q": q, **_evt(r)})

                if args.with_noise_control:
                    r = run_cell(_spec(GATED_SYNTHETIC, generator="noise", synthetic_ratio=1.0),
                                 index_df, win, ev, STORE, split, cfg=cfg, synthetic_provider=noise,
                                 teacher_model=tmodel, teacher_result=copy.deepcopy(teacher),
                                 gate_cfg=TrustGateConfig(q=0.50))
                    rows.append({**base, "pool": "noise", "condition": "gated", "q": 0.50, **_evt(r)})

                pd.DataFrame(rows).to_csv(out_csv, index=False)
                rr = rows[-1]
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
