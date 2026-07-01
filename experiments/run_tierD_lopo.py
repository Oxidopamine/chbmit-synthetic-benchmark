"""Tier D: leave-one-patient-group-out verification (plan Section 18).

LOPO is balanced_group_kfold with n_folds = number of groups, so each patient
group is the test set exactly once. 1 seed; EEGNet + LCT + TCN; scarcity 1.0 and
0.25; conditions real_only, classical_aug, ungated_synthetic_aug,
trust_gated_synthetic_aug with the best in-house generator. Goal: show the main
harm/mitigation trend survives the stricter LOPO protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chbmit.config import load_config
from chbmit.make_manifest import build_manifest
from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig
from experiments.aggregate_results import save_results
from experiments.grid import GridSpec, run_grid
from experiments.prepare import prepare_dataset
from experiments.training import TrainConfig
from synthetic.cvae_provider import CVAEConfig
from synthetic.wgan_gp_provider import WGANConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier D LOPO run")
    ap.add_argument("--config", default="configs/chbmit_synthetic.yaml")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--generator", default="cvae")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--gen-epochs", type=int, default=200)
    ap.add_argument("--min-ictal", type=int, default=256)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--reuse-processed", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    raw_root = args.raw_root or cfg.resolve_path("paths.raw_chbmit")
    processed_dir = args.processed_dir or cfg.resolve_path("paths.processed_chbmit")
    results_dir = Path(args.results_dir or cfg.resolve_path("paths.results"))

    # Number of groups -> n_folds for LOPO.
    manifest = build_manifest(raw_root, include_chb24=cfg.get("manifest.include_chb24", True))
    n_groups = manifest[manifest["included"]]["group"].nunique()

    pre = PreprocessConfig(target_sampling_rate=cfg.get("preprocessing.target_sampling_rate", 256))
    win = WindowingConfig(sampling_rate=pre.target_sampling_rate)
    prepared = prepare_dataset(raw_root, processed_dir, results_dir, n_folds=n_groups,
                               seed=cfg.base_seed, pre_cfg=pre, win_cfg=win,
                               reuse_processed=args.reuse_processed)

    grid = GridSpec(
        detectors=cfg.get("detectors.required", ["eegnet", "lct", "tcn"]),
        conditions=["real_only", "classical_aug", "ungated_synthetic_aug",
                    "trust_gated_synthetic_aug"],
        scarcity_fractions=[1.0, 0.25],
        seeds=[cfg.get("seeds.dev", [42])[0]],
        generators=[args.generator],
        folds=args.folds,
    )
    train_cfg = TrainConfig(epochs=args.epochs)
    gen_configs = {
        "cvae": CVAEConfig(epochs=args.gen_epochs, min_ictal_windows=args.min_ictal),
        "wgan_gp": WGANConfig(epochs=args.gen_epochs, min_ictal_windows=args.min_ictal),
    }
    out = run_grid(prepared, grid, train_cfg=train_cfg, gen_configs=gen_configs,
                   results_dir=results_dir)
    save_results(out["results"], results_dir / "tables", "tierD_lopo")
    print(json.dumps({"n_groups": int(n_groups), "n_cells": len(out["results"])}, indent=2))


if __name__ == "__main__":
    main()
