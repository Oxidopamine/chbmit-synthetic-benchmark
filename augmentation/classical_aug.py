"""Classical EEG augmentation (plan Section 9, ``classical_aug``).

Conservative perturbations applied to ictal windows in TRAINING ONLY, in the
normalized window space (per-window z-scored, so std ~ 1):

* Gaussian jitter (std 0.02)
* amplitude scaling in [0.9, 1.1]
* small time shift (<= 64 samples)
* channel dropout (p 0.10, <= 2 channels)

The augmenter is label-aware: only positive (ictal) windows are perturbed,
matching "perturbations on ictal windows only". Negatives pass through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ClassicalAugConfig:
    jitter_std: float = 0.02
    amplitude_scale: Tuple[float, float] = (0.9, 1.1)
    max_time_shift_samples: int = 64
    channel_dropout_p: float = 0.10
    max_dropped_channels: int = 2


class ClassicalAugment:
    """Callable ``(x[, label]) -> x`` applying conservative perturbations."""

    def __init__(self, cfg: Optional[ClassicalAugConfig] = None, seed: int = 0,
                 ictal_only: bool = True):
        self.cfg = cfg or ClassicalAugConfig()
        self.rng = np.random.default_rng(seed)
        self.ictal_only = ictal_only

    def __call__(self, x: np.ndarray, label: Optional[int] = None) -> np.ndarray:
        if self.ictal_only and label is not None and label != 1:
            return x
        x = np.array(x, dtype="float32", copy=True)
        C, T = x.shape
        # amplitude scaling (per window)
        lo, hi = self.cfg.amplitude_scale
        x *= float(self.rng.uniform(lo, hi))
        # time shift (circular roll)
        if self.cfg.max_time_shift_samples > 0:
            shift = int(self.rng.integers(-self.cfg.max_time_shift_samples,
                                          self.cfg.max_time_shift_samples + 1))
            if shift != 0:
                x = np.roll(x, shift, axis=-1)
        # gaussian jitter
        if self.cfg.jitter_std > 0:
            x += self.rng.normal(0.0, self.cfg.jitter_std, size=x.shape).astype("float32")
        # channel dropout
        if self.cfg.channel_dropout_p > 0 and self.cfg.max_dropped_channels > 0:
            if self.rng.random() < self.cfg.channel_dropout_p:
                k = int(self.rng.integers(1, self.cfg.max_dropped_channels + 1))
                drop = self.rng.choice(C, size=min(k, C), replace=False)
                x[drop, :] = 0.0
        return x
