"""Downstream gated help/harm experiment with the band-limited WGAN generator (fold 0).

The paper's open question: once we use a spectrally-realistic generator (WGAN-GP) and
strip its out-of-band junk (band-limiting), does the ADMITTED synthetic actually help
patient-independent seizure detection -- or is the trust gate's conservatism warranted?

This runs the real Tier-B pipeline (run_cell: train -> val threshold selection ->
event-level test metrics) on fold 0 for conditions:
  * real_only              -- teacher / baseline
  * ungated_synthetic_aug  -- inject band-limited WGAN synthetic directly (naive)
  * trust_gated (q=0.90)   -- pre-registered gate (expected ~fail-closed: admits ~0.1%)
  * trust_gated (q=0.50)   -- SENSITIVITY axis: at the real-ictal median threshold the
                              band-limited WGAN admits ~17%, so this is where "does
                              admitted synthetic help?" can actually be answered.

The generator is the persisted fold-0 WGAN-GP; its output is band-limited to 0.5-40 Hz
before injection/admission. The processed store is on a network fs, so we PREFETCH the
fold's train/val/test windows into the in-process cache once (per-file bandwidth reads),
after which the whole pipeline runs from RAM with num_workers=0.

Reports event-F1 / event-sensitivity / FP-per-24h per condition + gate admission/decision.
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
from chbmit.datasets import negative_sample, prefetch_windows, clear_window_cache
from synthetic.wgan_gp_provider import WGANConfig
from synthetic.train_provider import build_provider
from synthetic.band_limit import band_limit_windows
from synthetic.trust_gate import TrustGateConfig
from experiments.training import (
    run_cell, CellSpec, TrainConfig, UNGATED_SYNTHETIC, GATED_SYNTHETIC)

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
DEVICE = "cuda"
DETECTORS = ["eegnet", "lct", "tcn"]
GATE_QS = [0.90, 0.50]
SEED, FRAC = 42, 1.0


def _evt(res):
    """Pull the headline event metrics out of a run_cell result dict.

    ``event_f1``/``fp_per_24h`` are the condition's EFFECTIVE output (for a reverted
    gated cell that is the real-only teacher). ``aug_*`` are the model actually trained
    on the (admitted) synthetic -- what tells us whether the synthetic helped -- taken
    from ``gated_model_metrics`` when the gate reverted, else from ``event_metrics``.
    """
    em = res.get("event_metrics", {}) or {}
    g = res.get("gate")
    aug = (res.get("gated_model_metrics", {}) or {}).get("event_metrics", em) if res.get(
        "reverted_to_real_only") else em
    return {
        "event_f1": em.get("event_f1"), "event_sensitivity": em.get("event_sensitivity"),
        "event_precision": em.get("event_precision"), "fp_per_24h": em.get("fp_per_24h"),
        "aug_event_f1": aug.get("event_f1"), "aug_fp_per_24h": aug.get("fp_per_24h"),
        "aug_event_sensitivity": aug.get("event_sensitivity"),
        "reverted_to_real_only": res.get("reverted_to_real_only"),
        "gate_admission_rate": (g or {}).get("admission_rate"),
        "gate_n_admitted": (g or {}).get("n_admitted"),
        "gate_reason": (g or {}).get("reason"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--detectors", nargs="*", default=DETECTORS)
    ap.add_argument("--qs", nargs="*", type=float, default=GATE_QS)
    ap.add_argument("--tag", default="")  # output suffix for smoke runs
    args = ap.parse_args()
    detectors, gate_qs = args.detectors, args.qs
    out_csv = OUT / f"downstream_gated_bandlimited{args.tag}.csv"

    t0 = time.time()
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    f0 = sp["folds"][0]
    split = Split(fold=0, train_groups=f0["train_groups"],
                  val_groups=f0["val_groups"], test_groups=f0["test_groups"])
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])
    cfg = TrainConfig(epochs=args.epochs, device=DEVICE, num_workers=0)

    # --- Prefetch fold-0 windows into RAM (train real table + all val + all test) ---
    train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
    val_ids = set(index_df[index_df["group"].isin(split.val_groups)]["file_id"])
    test_ids = set(index_df[index_df["group"].isin(split.test_groups)]["file_id"])
    train_windows = win[win["file_id"].isin(train_ids) & (~win["excluded"])]
    selected = select_seizure_events(ev[ev["group"].isin(split.train_groups)], FRAC, SEED,
                                     restrict_groups=split.train_groups)
    scarce = apply_scarcity_to_windows(train_windows, selected)
    train_table = negative_sample(scarce, ratio=cfg.background_to_seizure_ratio,
                                  exclude_seconds=cfg.exclude_seconds_around_seizure, seed=SEED)
    val_windows = win[win["file_id"].isin(val_ids)]
    test_windows = win[win["file_id"].isin(test_ids)]
    for name, tbl in [("train", train_table), ("val", val_windows), ("test", test_windows)]:
        n = prefetch_windows(tbl, STORE, workers=16)
        print(f"[{time.time()-t0:6.0f}s] prefetched {name} ({len(tbl)} win); cache={n}", flush=True)

    # --- Band-limited WGAN provider (persisted fold-0 generator) ---
    wgan = build_provider("wgan_gp", WGANConfig(min_ictal_windows=256, device=DEVICE, seed=SEED))
    wgan._load_state(RES / "generators" / "wgan_fold0")
    wgan.fitted = True

    def bl_wgan(n, s):
        return band_limit_windows(wgan.generate(n, seed=s), fs=fs)

    rows = []
    for det in detectors:
        def _spec(cond, **kw):
            return CellSpec(fold=0, seed=SEED, scarcity_fraction=FRAC, detector=det,
                            condition=cond, **kw)

        # real_only teacher/baseline
        teacher_res = run_cell(_spec("real_only"), index_df, win, ev, STORE, split,
                               cfg=cfg, return_model=True)
        teacher_model = teacher_res.pop("model", None)
        rows.append({"detector": det, "condition": "real_only", "q": None, **_evt(teacher_res)})
        print(f"[{time.time()-t0:6.0f}s] {det} real_only  F1={_evt(teacher_res)['event_f1']}", flush=True)

        # ungated (naive) band-limited WGAN
        r = run_cell(_spec(UNGATED_SYNTHETIC, generator="wgan_gp_bandlimited", synthetic_ratio=1.0),
                     index_df, win, ev, STORE, split, cfg=cfg, synthetic_provider=bl_wgan)
        rows.append({"detector": det, "condition": "ungated", "q": None, **_evt(r)})
        print(f"[{time.time()-t0:6.0f}s] {det} ungated    F1={_evt(r)['event_f1']}", flush=True)

        # gated at each q (fresh teacher copy so state isn't mutated across runs)
        for q in gate_qs:
            r = run_cell(_spec(GATED_SYNTHETIC, generator="wgan_gp_bandlimited", synthetic_ratio=1.0),
                         index_df, win, ev, STORE, split, cfg=cfg, synthetic_provider=bl_wgan,
                         teacher_model=teacher_model,
                         teacher_result=copy.deepcopy(teacher_res),
                         gate_cfg=TrustGateConfig(q=q))
            e = _evt(r)
            rows.append({"detector": det, "condition": "gated", "q": q, **e})
            print(f"[{time.time()-t0:6.0f}s] {det} gated q={q} F1={e['event_f1']} "
                  f"admit={e['gate_admission_rate']} reverted={e['reverted_to_real_only']}", flush=True)

        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\n=== DOWNSTREAM GATED (band-limited WGAN, fold 0) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\n[{time.time()-t0:6.0f}s] wrote {out_csv}", flush=True)
    clear_window_cache()


if __name__ == "__main__":
    main()
