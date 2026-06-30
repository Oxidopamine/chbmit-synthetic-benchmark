"""Common synthetic-provider interface and provenance (plan Section 8, Rule 8).

Every provider (GAN, VAE, external, precomputed) implements the same contract
and fits on TRAINING patient groups only (Rule 3), emitting windows in the same
normalized ``(C, T)`` training space as the classifier (Rule 6). Each provider
records the fold and the exact training patient groups and seizure events it was
derived from, so synthetic windows are fully traceable.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProviderMetadata:
    name: str
    paradigm: str  # "patient_independent" | "patient_specific"
    fold_id: Optional[int] = None
    source_train_patient_groups: List[str] = field(default_factory=list)
    source_train_seizure_events: List[int] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    window_samples: Optional[int] = None
    normalization_protocol: str = "per_window_channel_zscore"
    synthetic_ratio: Optional[float] = None
    generation_seed: Optional[int] = None
    n_train_ictal_windows: Optional[int] = None
    extra: Dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class SyntheticProvider(ABC):
    """Abstract provider. ``name`` and ``paradigm`` are set by subclasses."""

    name: str = "base"
    paradigm: str = "patient_independent"

    def __init__(self):
        self.metadata: Optional[ProviderMetadata] = None
        self.fitted: bool = False

    @abstractmethod
    def fit(self, train_windows: np.ndarray, train_labels: np.ndarray,
            train_metadata: dict, config: Optional[dict] = None) -> "SyntheticProvider":
        """Fit on training ictal windows ``(N, C, T)`` (Rule 3)."""

    @abstractmethod
    def generate(self, n: int, class_label: str = "seizure", seed: int = 42) -> np.ndarray:
        """Generate ``n`` synthetic ``(C, T)`` windows in normalized training space."""

    def _build_metadata(self, train_metadata: dict, n_ictal: int) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            paradigm=self.paradigm,
            fold_id=train_metadata.get("fold_id"),
            source_train_patient_groups=list(train_metadata.get("source_train_patient_groups", [])),
            source_train_seizure_events=list(train_metadata.get("source_train_seizure_events", [])),
            channels=list(train_metadata.get("channels", [])),
            window_samples=train_metadata.get("window_samples"),
            normalization_protocol=train_metadata.get("normalization_protocol",
                                                      "per_window_channel_zscore"),
            n_train_ictal_windows=int(n_ictal),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.metadata is not None:
            (path / "provider_metadata.json").write_text(
                json.dumps(self.metadata.as_dict(), indent=2), encoding="utf-8"
            )
        self._save_state(path)

    def load(self, path: str | Path) -> "SyntheticProvider":
        path = Path(path)
        meta_path = path / "provider_metadata.json"
        if meta_path.exists():
            self.metadata = ProviderMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
        self._load_state(path)
        self.fitted = True
        return self

    # subclasses override these for their own serialized state
    def _save_state(self, path: Path) -> None:  # pragma: no cover - optional
        pass

    def _load_state(self, path: Path) -> None:  # pragma: no cover - optional
        pass


def ictal_subset(train_windows: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Select ictal (positive) windows for generator fitting."""
    labels = np.asarray(train_labels).astype(int)
    return np.asarray(train_windows, dtype="float32")[labels == 1]
