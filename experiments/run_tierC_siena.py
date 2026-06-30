"""Tier C: Siena cross-dataset verification (plan Section 18).

Re-runs the core comparison on Siena (reduced): 1 seed, scarcity 1.0 and 0.25,
conditions real_only / classical_aug / synthetic_aug, detectors EEGNet + LCT +
TCN, generator = the in-house generator that performed best on CHB-MIT. Goal:
show the CHB-MIT trend is not dataset-specific. Verification, not a full grid.

Uses the Siena manifest adapter so the identical pipeline runs unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chbmit.config import load_config
from chbmit.preprocess_edf import PreprocessConfig
from chbmit.siena import build_siena_manifest
from chbmit.window_metadata import WindowingConfig
from experiments.aggregate_results import save_results
from experiments.grid import GridSpec, run_grid
from experiments.prepare import prepare_dataset
from experiments.training import TrainConfig
from synthetic.cvae_provider import CVAEConfig
from synthetic.wgan_gp_provider import WGANConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier C Siena verification run")
    ap.add_argument("--config", default="configs/siena_verification.yaml")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--generator", default="cvae")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--gen-epochs", type=int, default=200)
    ap.add_argument("--min-ictal", type=int, default=256)
    ap.add_argument("--reuse-processed", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    raw_root = args.raw_root or cfg.resolve_path("paths.raw_siena")
    processed_dir = args.processed_dir or cfg.resolve_path("paths.processed_siena")
    results_dir = Path(args.results_dir or cfg.resolve_path("paths.results"))

    pre = PreprocessConfig(target_sampling_rate=cfg.get("preprocessing.target_sampling_rate", 256))
    win = WindowingConfig(sampling_rate=pre.target_sampling_rate)
    prepared = prepare_dataset(
        raw_root, processed_dir, results_dir, n_folds=args.n_folds, seed=cfg.base_seed,
        pre_cfg=pre, win_cfg=win, reuse_processed=args.reuse_processed,
        manifest_builder=build_siena_manifest,
    )

    grid = GridSpec(
        detectors=cfg.get("detectors.required", ["eegnet", "lct", "tcn"]),
        conditions=["real_only", "classical_aug", "synthetic_aug"],
        scarcity_fractions=[1.0, 0.25],
        seeds=[cfg.get("seeds.dev", [42])[0]],
        generators=[args.generator],
    )
    train_cfg = TrainConfig(epochs=args.epochs)
    gen_configs = {
        "cvae": CVAEConfig(epochs=args.gen_epochs, min_ictal_windows=args.min_ictal),
        "wgan_gp": WGANConfig(epochs=args.gen_epochs, min_ictal_windows=args.min_ictal),
    }
    out = run_grid(prepared, grid, train_cfg=train_cfg, gen_configs=gen_configs,
                   results_dir=results_dir)
    save_results(out["results"], results_dir / "tables", "tierC_siena")
    print(json.dumps({"n_cells": len(out["results"]),
                      "audit_accepted": prepared.audit_accepted}, indent=2))


if __name__ == "__main__":
    main()
