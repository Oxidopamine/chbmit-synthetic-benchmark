"""Precomputed synthetic provider / loader (plan Section 8, Provider D).

Loads synthetic windows generated outside the pipeline while enforcing
fold-specific, train-only provenance (Rule 8). The required metadata fields must
be present or loading fails, so untraceable synthetic data can never enter
training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from synthetic.provider_base import ProviderMetadata, SyntheticProvider

REQUIRED_META_FIELDS = [
    "fold_id", "source_train_patient_groups", "source_train_seizure_events",
    "provider_name", "paradigm", "synthetic_ratio", "generation_seed",
    "channels", "window_samples", "normalization_protocol",
]


class PrecomputedProvider(SyntheticProvider):
    name = "precomputed"
    paradigm = "patient_independent"

    def __init__(self):
        super().__init__()
        self.windows: Optional[np.ndarray] = None

    def fit(self, train_windows, train_labels, train_metadata, config=None):
        raise NotImplementedError("PrecomputedProvider loads windows; it is not fitted")

    def load_npz(self, path: str | Path) -> "PrecomputedProvider":
        """Load ``windows`` ``(N, C, T)`` and provenance metadata from an .npz."""
        data = np.load(path, allow_pickle=True)
        if "windows" not in data:
            raise ValueError("precomputed npz must contain a 'windows' array")
        meta = dict(data["metadata"].item()) if "metadata" in data else {}
        missing = [f for f in REQUIRED_META_FIELDS if f not in meta]
        if missing:
            raise ValueError(f"precomputed provider missing provenance fields: {missing}")
        self.windows = np.asarray(data["windows"], dtype="float32")
        self.paradigm = meta.get("paradigm", self.paradigm)
        self.name = meta.get("provider_name", self.name)
        self.metadata = ProviderMetadata(
            name=self.name, paradigm=self.paradigm, fold_id=meta["fold_id"],
            source_train_patient_groups=list(meta["source_train_patient_groups"]),
            source_train_seizure_events=list(meta["source_train_seizure_events"]),
            channels=list(meta["channels"]), window_samples=meta["window_samples"],
            normalization_protocol=meta["normalization_protocol"],
            synthetic_ratio=meta["synthetic_ratio"], generation_seed=meta["generation_seed"],
            n_train_ictal_windows=meta.get("n_train_ictal_windows"),
        )
        self.fitted = True
        return self

    def generate(self, n: int, class_label: str = "seizure", seed: int = 42) -> np.ndarray:
        if self.windows is None:
            raise RuntimeError("PrecomputedProvider has no loaded windows")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.windows), size=n)
        return self.windows[idx].astype("float32")
