"""Balanced (oversampling) training sampler (plan Section 9, ``balanced_sampler``).

Oversamples ictal windows in TRAINING ONLY via a WeightedRandomSampler so each
class is drawn with equal probability. Validation/test are never resampled.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels, num_samples: int = None) -> WeightedRandomSampler:
    labels = np.asarray(labels).astype(int)
    class_count = np.bincount(labels, minlength=2).astype(float)
    class_count[class_count == 0] = 1.0
    per_class_w = 1.0 / class_count
    weights = per_class_w[labels]
    n = num_samples if num_samples is not None else len(labels)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(n), replacement=True,
    )
