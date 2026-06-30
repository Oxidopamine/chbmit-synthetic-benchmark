"""Tier A: development run (plan Section 18).

5 (or fewer) balanced patient-group folds, 1 seed, EEGNet + LCT, conditions
real_only / class_weighted / classical_aug, NO synthetic. Goal: prove the whole
pipeline end-to-end and produce real-only patient-independent event-level
results (the first reportable milestone, Section 23).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chbmit.config import load_config
from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig
from experiments.aggregate_results import save_results
from experiments.prepare import prepare_dataset
from experiments.training import CellSpec, TrainConfig, run_cell

TIER_A_DETECTORS = ["eegnet", "lct"]
TIER_A_CONDITIONS = ["real_only", "class_weighted", "classical_aug"]


def main():
    ap = argparse.ArgumentParser(description="Tier A dev run")
    ap.add_argument("--config", default="configs/chbmit_synthetic.yaml")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--detectors", nargs="*", default=TIER_A_DETECTORS)
    ap.add_argument("--reuse-processed", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_root = args.raw_root or cfg.resolve_path("paths.raw_chbmit")
    processed_dir = args.processed_dir or cfg.resolve_path("paths.processed_chbmit")
    results_dir = Path(args.results_dir or cfg.resolve_path("paths.results"))
    seed = cfg.get("seeds.dev", [42])[0]

    pre = PreprocessConfig(
        target_sampling_rate=cfg.get("preprocessing.target_sampling_rate", 256),
        bandpass_low_hz=cfg.get("preprocessing.bandpass_low_hz", 0.5),
        bandpass_high_hz=cfg.get("preprocessing.bandpass_high_hz", 40.0),
        notch_hz=cfg.get("preprocessing.notch_hz", None),
    )
    win = WindowingConfig(
        window_seconds=cfg.get("windowing.window_seconds", 4),
        stride_seconds=cfg.get("windowing.stride_seconds", 2),
        sampling_rate=pre.target_sampling_rate,
    )
    prepared = prepare_dataset(
        raw_root, processed_dir, results_dir, n_folds=args.folds, seed=seed,
        include_chb24=cfg.get("manifest.include_chb24", True),
        pre_cfg=pre, win_cfg=win, reuse_processed=args.reuse_processed,
    )
    if not prepared.audit_accepted:
        print("WARNING: channel audit gate not met; using reduced montage.")

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=cfg.get("classifier_training.batch_size", 64),
        learning_rate=cfg.get("classifier_training.learning_rate", 3e-4),
        weight_decay=cfg.get("classifier_training.weight_decay", 1e-4),
        early_stopping_patience=cfg.get("classifier_training.early_stopping_patience", 12),
        background_to_seizure_ratio=cfg.get("negative_sampling.background_to_seizure_ratio", 5),
        exclude_seconds_around_seizure=cfg.get("negative_sampling.exclude_seconds_around_seizure", 60),
    )

    results = []
    for fold in range(len(prepared.splits)):
        for detector in args.detectors:
            for condition in TIER_A_CONDITIONS:
                spec = CellSpec(fold=fold, seed=seed, scarcity_fraction=1.0,
                                detector=detector, condition=condition)
                print(f"[Tier A] fold={fold} {detector} {condition}")
                res = run_cell(spec, prepared.index_df, prepared.windows_df,
                               prepared.events_df, prepared.store,
                               prepared.splits[fold], cfg=train_cfg)
                results.append(res)

    path = save_results(results, results_dir / "tables", "tierA_dev")
    print(f"wrote {path}")
    print(json.dumps({"n_cells": len(results)}, indent=2))


if __name__ == "__main__":
    main()
