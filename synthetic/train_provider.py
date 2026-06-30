"""Fit a synthetic provider on a fold's exact training subset (plan Rule 4).

Reproduces the same event-level scarcity selection and ictal-window subset the
classifier uses, materializes those windows in normalized training space, and
fits the provider with full provenance. The fitted provider is reused across
detectors for a given (fold, seed, scarcity, generator) so the generator never
depends on the downstream detector.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from chbmit.datasets import materialize_windows
from chbmit.scarcity import (
    apply_scarcity_to_windows,
    save_selected_events,
    scarcity_coverage_report,
    select_seizure_events,
)
from synthetic.cvae_provider import CVAEConfig, CVAEProvider
from synthetic.precomputed_provider import PrecomputedProvider
from synthetic.provider_base import SyntheticProvider
from synthetic.wgan_gp_provider import WGANConfig, WGANGPProvider

_PROVIDERS = {
    "wgan_gp": WGANGPProvider,
    "cvae": CVAEProvider,
    "precomputed": PrecomputedProvider,
}


def build_provider(name: str, config=None) -> SyntheticProvider:
    if name not in _PROVIDERS:
        raise KeyError(f"unknown provider '{name}'; available: {sorted(_PROVIDERS)}")
    if config is None:
        return _PROVIDERS[name]()
    return _PROVIDERS[name](config)


def fit_provider_for_cell(
    provider: SyntheticProvider,
    index_df,
    windows_df,
    events_df,
    store: str,
    split,
    fold: int,
    seed: int,
    scarcity_fraction: float,
    normalize_method: str = "per_window_channel_zscore",
    eps: float = 1e-6,
    save_events_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Fit ``provider`` on the fold's scarcity-selected training ictal windows."""
    train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
    train_windows = windows_df[windows_df["file_id"].isin(train_ids) & (~windows_df["excluded"])]

    train_events = events_df[events_df["group"].isin(split.train_groups)]
    selected = select_seizure_events(
        train_events, scarcity_fraction, seed, restrict_groups=split.train_groups
    )
    scarce = apply_scarcity_to_windows(train_windows, selected)
    ictal_table = scarce[scarce["label"] == 1]

    X, y = materialize_windows(ictal_table, store, normalize_method, eps)
    train_metadata = {
        "fold_id": fold,
        "source_train_patient_groups": list(split.train_groups),
        "source_train_seizure_events": sorted(int(e) for e in selected),
        "channels": _channels(store),
        "window_samples": int(X.shape[2]) if X.size else None,
        "normalization_protocol": normalize_method,
    }
    if save_events_dir is not None:
        save_selected_events(selected, events_df, save_events_dir, fold, seed, scarcity_fraction)

    provider.fit(X, y, train_metadata)
    coverage = scarcity_coverage_report(selected, train_events, split.train_groups)
    return {
        "provider": provider,
        "n_ictal": int(len(X)),
        "real_ictal": X,  # real training ictal windows (for quality checks)
        "selected_events": sorted(int(e) for e in selected),
        "skipped": getattr(provider, "skipped", False),
        "fitted": provider.fitted,
        "coverage": coverage,
    }


def _channels(store: str):
    import zarr

    return list(zarr.open_group(str(store), mode="r").attrs["channels"])
