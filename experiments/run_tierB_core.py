"""Tier B: core conference experiment (plan Section 18).

5 balanced patient-group folds x 3 seeds; detectors EEGNet + LCT + TCN; scarcity
1.0 / 0.5 / 0.25; conditions real_only, class_weighted, balanced_sampler,
classical_aug, synthetic_aug; generators WGAN-GP and cVAE at synthetic ratio 1.0;
window + SzCORE event + patient metrics with paired deltas per generator x
detector cell. CLI flags allow running subsets (this machine is CPU-only; the
full grid is intended for a GPU/cloud box).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chbmit.config import load_config
from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig
from experiments.aggregate_results import save_results
from experiments.grid import GridSpec, run_grid
from experiments.prepare import prepare_dataset
from experiments.training import TrainConfig
from synthetic.cvae_provider import CVAEConfig
from synthetic.wgan_gp_provider import WGANConfig


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Tier B core run")
    ap.add_argument("--config", default="configs/chbmit_synthetic.yaml")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of fold indices")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--scarcity", type=float, nargs="*", default=None)
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--gen-epochs", type=int, default=300)
    ap.add_argument("--min-ictal", type=int, default=256)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--reuse-processed", action="store_true")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    raw_root = args.raw_root or cfg.resolve_path("paths.raw_chbmit")
    processed_dir = args.processed_dir or cfg.resolve_path("paths.processed_chbmit")
    results_dir = Path(args.results_dir or cfg.resolve_path("paths.results"))

    pre = PreprocessConfig(
        target_sampling_rate=cfg.get("preprocessing.target_sampling_rate", 256),
        bandpass_low_hz=cfg.get("preprocessing.bandpass_low_hz", 0.5),
        bandpass_high_hz=cfg.get("preprocessing.bandpass_high_hz", 40.0),
        notch_hz=cfg.get("preprocessing.notch_hz", None),
    )
    win = WindowingConfig(window_seconds=cfg.get("windowing.window_seconds", 4),
                          stride_seconds=cfg.get("windowing.stride_seconds", 2),
                          sampling_rate=pre.target_sampling_rate)
    prepared = prepare_dataset(raw_root, processed_dir, results_dir, n_folds=args.n_folds,
                               seed=cfg.base_seed, include_chb24=cfg.get("manifest.include_chb24", True),
                               pre_cfg=pre, win_cfg=win, reuse_processed=args.reuse_processed)

    grid = GridSpec(
        detectors=args.detectors or cfg.get("detectors.required", ["eegnet", "lct", "tcn"]),
        conditions=cfg.get("augmentation.core_methods"),
        scarcity_fractions=args.scarcity or cfg.get("scarcity.fractions", [1.0, 0.5, 0.25]),
        seeds=args.seeds or cfg.get("seeds.core", [42, 123, 2024]),
        generators=args.generators or cfg.get("generators.in_house", ["wgan_gp", "cvae"]),
        synthetic_ratio=cfg.get("generators.core_synthetic_ratios", [1.0])[0],
        folds=args.folds,
        do_quality=not args.no_quality,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=cfg.get("classifier_training.batch_size", 64),
        learning_rate=cfg.get("classifier_training.learning_rate", 3e-4),
        weight_decay=cfg.get("classifier_training.weight_decay", 1e-4),
        early_stopping_patience=cfg.get("classifier_training.early_stopping_patience", 12),
        background_to_seizure_ratio=cfg.get("negative_sampling.background_to_seizure_ratio", 5),
        exclude_seconds_around_seizure=cfg.get("negative_sampling.exclude_seconds_around_seizure", 60),
    )
    gen_configs = {
        "wgan_gp": WGANConfig(epochs=args.gen_epochs, min_ictal_windows=args.min_ictal),
        "cvae": CVAEConfig(epochs=max(100, args.gen_epochs // 2), min_ictal_windows=args.min_ictal),
    }

    out = run_grid(prepared, grid, train_cfg=train_cfg, gen_configs=gen_configs,
                   results_dir=results_dir)
    save_results(out["results"], results_dir / "tables", "tierB_core")
    if out["quality"]:
        (results_dir / "tables").mkdir(parents=True, exist_ok=True)
        (results_dir / "tables" / "tierB_quality.json").write_text(
            json.dumps(out["quality"], indent=2, default=float), encoding="utf-8")
    print(json.dumps({"n_cells": len(out["results"]),
                      "n_quality": len(out["quality"])}, indent=2))


if __name__ == "__main__":
    main()
