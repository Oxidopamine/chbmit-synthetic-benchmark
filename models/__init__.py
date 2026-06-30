"""Detector architectures and a small build registry (plan Section 7).

All detectors share one contract:
    input  : (N, C, T) float32   (C = montage channels, T = window samples)
    output : (N,) raw logit      (use sigmoid for the positive-class score)

This keeps the training loop and evaluation identical across architectures
(Tier B requirement: identical data, splits, and budgets).
"""
from __future__ import annotations

from typing import Callable, Dict

from models.eegnet import EEGNet
from models.lct_wrapper import LCT
from models.tcn import TCN

_REGISTRY: Dict[str, Callable] = {
    "eegnet": EEGNet,
    "lct": LCT,
    "tcn": TCN,
}


def available_models():
    return sorted(_REGISTRY)


def build_model(name: str, n_channels: int, n_samples: int, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {available_models()}")
    return _REGISTRY[name](n_channels=n_channels, n_samples=n_samples, **kwargs)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
