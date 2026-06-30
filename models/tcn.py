"""Temporal Convolutional Network detector (plan Section 7, Detector 3).

Stacked dilated causal 1-D convolutions give a large receptive field with no
attention and no recurrence - architecturally distinct from EEGNet (compact
conv) and LCT (attention). Fully in-house, no external dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:  # prefer the non-deprecated parametrization API
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - older torch
    from torch.nn.utils import weight_norm


class _Chomp1d(nn.Module):
    """Trim right padding to keep convolutions causal."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, channels=(32, 32, 64, 64),
                 kernel: int = 7, dropout: float = 0.2):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        layers = []
        prev = n_channels
        for i, ch in enumerate(channels):
            layers.append(_TemporalBlock(prev, ch, kernel, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, T) -> features over time -> global average pool
        h = self.tcn(x)
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)
