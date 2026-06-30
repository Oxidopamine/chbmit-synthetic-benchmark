"""Window normalization protocols (plan Section 12).

The main protocol is per-window per-channel z-score: stateless, leakage-free,
and stable across models. It removes absolute amplitude information (documented
trade-off). The same function is applied to real, synthetic, train, val and
test windows (Rule 6) so synthetic providers emit data in the same space.

All functions operate on the last axis (time) and broadcast over leading dims,
so they work on a single ``(C, T)`` window or a ``(N, C, T)`` batch.
"""
from __future__ import annotations

from typing import Optional, Tuple


def normalize_window(x, method: str = "per_window_channel_zscore", eps: float = 1e-6,
                     stats: Optional[Tuple] = None):
    """Normalize ``x`` (``..., C, T``) per the named protocol.

    ``stats`` (mean, std) arrays are required only for ``train_channel_zscore``.
    """
    import numpy as np

    x = np.asarray(x, dtype="float32")
    if method == "per_window_channel_zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        return ((x - mu) / (sd + eps)).astype("float32")
    if method == "train_channel_zscore":
        if stats is None:
            raise ValueError("train_channel_zscore requires (mean, std) stats")
        mu, sd = stats
        mu = np.asarray(mu, dtype="float32").reshape(*([1] * (x.ndim - 2)), -1, 1)
        sd = np.asarray(sd, dtype="float32").reshape(*([1] * (x.ndim - 2)), -1, 1)
        return ((x - mu) / (sd + eps)).astype("float32")
    if method == "none":
        return x
    raise ValueError(f"unknown normalization method: {method}")
