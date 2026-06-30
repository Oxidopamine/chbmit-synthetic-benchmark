"""Validate the full pipeline on a REAL CHB-MIT slice (chb01 + chb02).

Runs manifest -> channel audit -> preprocess -> windows -> splits and one real
real-only training cell (train one patient group, test the other) to confirm the
data-handling and evaluation code works against real EDF/summary formats.
Only the EDFs present on disk are used; missing files are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chbmit.make_manifest import build_manifest, summarize_manifest
from chbmit.channel_audit import audit_channels
from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig, summarize_windows
from experiments.prepare import prepare_dataset
from experiments.training import CellSpec, TrainConfig, run_cell

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_chbmit"
PROC = ROOT / "data" / "processed_chbmit_real"
RES = ROOT / "results_chbmit_synthetic" / "real_validation"


def main():
    print("== manifest ==")
    man = build_manifest(RAW)
    print(json.dumps(summarize_manifest(man), indent=2)[:1200])

    print("\n== channel audit ==")
    audit = audit_channels(man[man["included"]], max_dropped_fraction=0.10)
    print(json.dumps({k: audit.summary()[k] for k in
                      ["accepted", "n_files_total", "n_files_kept",
                       "dropped_seizure_event_fraction", "final_channel_set"]}, indent=2))

    print("\n== prepare (preprocess + windows + splits) ==")
    n_groups = int(man[man["included"]]["group"].nunique())
    n_folds = max(3, min(5, n_groups))  # need >=3 groups for patient-independent train/val/test
    prepared = prepare_dataset(
        RAW, PROC, RES, n_folds=n_folds, seed=42,
        pre_cfg=PreprocessConfig(target_sampling_rate=256),
        win_cfg=WindowingConfig(sampling_rate=256),
    )
    print("channels:", prepared.channels)
    print("windows:", json.dumps(summarize_windows(prepared.windows_df), indent=2))
    print("events:", len(prepared.events_df))
    for s in prepared.splits:
        print(f"  fold {s.fold}: train={s.train_groups} val={s.val_groups} test={s.test_groups}")

    # Pick a fold whose train and test both contain a seizure.
    print("\n== real real_only cell (eegnet) ==")
    spec = CellSpec(fold=0, seed=42, scarcity_fraction=1.0, detector="eegnet",
                    condition="real_only")
    res = run_cell(spec, prepared.index_df, prepared.windows_df, prepared.events_df,
                   prepared.store, prepared.splits[0],
                   cfg=TrainConfig(epochs=8, batch_size=64, monitor_max_neg_per_pos=20))
    print("selected_threshold:", res["selected_threshold"])
    print("window_metrics:", json.dumps(res["window_metrics"], indent=2))
    print("event_metrics:", json.dumps(res["event_metrics"], indent=2, default=float))
    print("n_train_windows:", res["n_train_windows"], "n_train_pos:", res["n_train_pos"])


if __name__ == "__main__":
    main()
