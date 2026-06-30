"""EEGNet detector (anchor model, plan Section 7, Detector 1).

Compact depthwise-separable temporal+spatial CNN (EEGNet-8,2 style) adapted to
the canonical ``(N, C, T)`` input and a single-logit binary head. Kept close to
the literature configuration; not over-optimized.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, F1: int = 8, D: int = 2,
                 F2: int = 16, kern_length: int = 64, dropout: float = 0.25):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv across all channels.
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            # Separable conv = depthwise (1,16) + pointwise (1,1).
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        feat = self._feature_dim(n_channels, n_samples)
        self.head = nn.Linear(feat, 1)

    def _feature_dim(self, n_channels: int, n_samples: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.block2(self.block1(dummy))
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, T) -> (N, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        x = torch.flatten(x, 1)
        return self.head(x).squeeze(-1)
