"""Class-weighting / focal-loss baseline (plan Section 9, ``class_weighted``)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_weight_from_labels(labels) -> torch.Tensor:
    """BCE ``pos_weight`` = (#neg / #pos), clipped to a sane range."""
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    w = (n_neg / n_pos) if n_pos > 0 else 1.0
    return torch.tensor(float(np.clip(w, 1.0, 100.0)), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Binary focal loss with logits (gamma=2.0 default, Section 9)."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def make_loss(kind: str, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
    """Return a ``loss(logits, targets)`` callable for the chosen condition."""
    if kind == "focal":
        return FocalLoss(gamma=gamma, pos_weight=pos_weight)
    if kind == "weighted_bce":
        def loss(logits, targets):
            return F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=pos_weight,
            )
        return loss
    if kind == "bce":
        def loss(logits, targets):
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        return loss
    raise ValueError(f"unknown loss kind: {kind}")
