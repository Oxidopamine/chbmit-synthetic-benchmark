"""Lightweight Convolution Transformer (LCT) detector (plan Section 7, Detector 2).

A practical transformer-style cross-patient multichannel detector (NOT a SOTA
claim). A strided convolutional stem turns the canonical ``(N, C, T)`` input
into a short token sequence; a few transformer-encoder layers with learned
positional embeddings model temporal context; attention pooling feeds a
single-logit head. The ``(N, C, T) -> token`` adaptation is handled here so the
training loop stays architecture-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvStem(nn.Module):
    """Two strided conv blocks: (N, C, T) -> (N, d_model, T') tokens over time."""

    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LCT(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1,
                 max_tokens: int = 512):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.stem = _ConvStem(n_channels, d_model)
        n_tokens = self._n_tokens(n_channels, n_samples)
        self.pos = nn.Parameter(torch.zeros(1, max(n_tokens, 1), d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _n_tokens(self, n_channels: int, n_samples: int) -> int:
        with torch.no_grad():
            out = self.stem(torch.zeros(1, n_channels, n_samples))
            return int(out.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).transpose(1, 2)  # (N, T', d_model)
        tokens = tokens + self.pos[:, : tokens.shape[1], :]
        h = self.encoder(tokens)
        w = torch.softmax(self.attn_pool(h), dim=1)  # attention pooling over tokens
        pooled = (w * h).sum(dim=1)
        return self.head(pooled).squeeze(-1)
