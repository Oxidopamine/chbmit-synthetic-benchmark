"""Dataset preparation driver: raw EDFs -> analysis-ready artifacts.

Runs manifest -> channel audit -> preprocess (zarr) -> window/event tables ->
balanced patient-group splits, writing artifacts under the results dir and
returning the in-memory tables the tier runners consume. This is the single
entry point shared by every tier so the leakage-safe pipeline is identical.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from chbmit.channel_audit import audit_channels, write_audit
from chbmit.make_manifest import build_manifest, write_manifest
from chbmit.preprocess_edf import PreprocessConfig, preprocess_dataset
from chbmit.splits import Split, make_splits, save_splits
from chbmit.window_metadata import WindowingConfig, build_window_table, save_window_tables


@dataclass
class PreparedDataset:
    index_df: pd.DataFrame
    windows_df: pd.DataFrame
    events_df: pd.DataFrame
    splits: List[Split]
    store: str
    channels: List[str]
    audit_accepted: bool


def prepare_dataset(
    raw_root: str | Path,
    processed_dir: str | Path,
    results_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
    include_chb24: bool = True,
    pre_cfg: Optional[PreprocessConfig] = None,
    win_cfg: Optional[WindowingConfig] = None,
    max_dropped_fraction: float = 0.10,
    reuse_processed: bool = False,
    manifest_builder=None,
) -> PreparedDataset:
    raw_root = Path(raw_root)
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    pre_cfg = pre_cfg or PreprocessConfig()
    win_cfg = win_cfg or WindowingConfig(sampling_rate=pre_cfg.target_sampling_rate)

    builder = manifest_builder or build_manifest
    manifest = builder(raw_root, include_chb24=include_chb24)
    write_manifest(manifest, results_dir / "manifests")
    audit = audit_channels(manifest, max_dropped_fraction=max_dropped_fraction)
    write_audit(audit, results_dir / "manifests")

    index_path = processed_dir / "processed_index.csv"
    if not (reuse_processed and index_path.exists()):
        preprocess_dataset(manifest, raw_root, processed_dir, audit.final_channel_set, pre_cfg)
    index_df = pd.read_csv(index_path)

    windows_df, events_df = build_window_table(index_df, win_cfg)
    save_window_tables(windows_df, events_df, results_dir / "windows")

    splits = make_splits(index_df, n_folds=n_folds, seed=seed)
    save_splits(splits, results_dir / "splits", seed=seed)

    return PreparedDataset(
        index_df=index_df, windows_df=windows_df, events_df=events_df,
        splits=splits, store=str(processed_dir / "eeg.zarr"),
        channels=list(audit.final_channel_set), audit_accepted=audit.accepted,
    )
