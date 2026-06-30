"""Lazy datasets and training-set construction (plan Sections 10, 11, 12).

* ``WindowDataset`` reads ``(C, T)`` windows on demand from the zarr store and
  applies the configured normalization (Rule 6). An optional ``transform``
  supports classical augmentation in normalized space.
* ``ArrayDataset`` wraps already-normalized in-memory windows (real or
  synthetic) so they can be concatenated into a training loader.
* ``materialize_windows`` pulls a (usually small, e.g. ictal) window subset into
  RAM as ``(N, C, T)`` arrays for generator fitting and quality checks.
* ``negative_sample`` builds a training window table at the configured
  background:seizure ratio, excluding near-seizure background (Section 10).

The continuous per-file timelines stay in zarr; evaluation reads them in order.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from chbmit.normalize import normalize_window


@lru_cache(maxsize=8)
def _open_signals(store_path: str):
    import zarr

    root = zarr.open_group(store_path, mode="r")
    return root["signals"]


def read_window(store_path: str, file_id: str, start: int, end: int) -> np.ndarray:
    """Read a single ``(C, T)`` window slice from the zarr store."""
    signals = _open_signals(str(store_path))
    return np.asarray(signals[file_id][:, start:end], dtype="float32")


class WindowDataset(Dataset):
    """Lazy windows from zarr, normalized on read."""

    def __init__(self, table, store_path: str, normalize_method: str = "per_window_channel_zscore",
                 eps: float = 1e-6, transform: Optional[Callable] = None,
                 transform_takes_label: bool = False):
        self.records = table.reset_index(drop=True)
        self.store_path = str(store_path)
        self.normalize_method = normalize_method
        self.eps = eps
        self.transform = transform
        self.transform_takes_label = transform_takes_label
        self.file_ids = self.records["file_id"].to_numpy()
        self.starts = self.records["start_sample"].to_numpy()
        self.ends = self.records["end_sample"].to_numpy()
        self.labels = self.records["label"].to_numpy().astype("int64")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        label = int(self.labels[i])
        x = read_window(self.store_path, self.file_ids[i], int(self.starts[i]), int(self.ends[i]))
        x = normalize_window(x, self.normalize_method, self.eps)
        if self.transform is not None:
            x = self.transform(x, label) if self.transform_takes_label else self.transform(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype="float32")), label


class ArrayDataset(Dataset):
    """Already-normalized in-memory windows (e.g. synthetic ictal samples)."""

    def __init__(self, x: np.ndarray, y, transform: Optional[Callable] = None):
        self.x = np.asarray(x, dtype="float32")
        self.y = np.asarray(y).astype("int64")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype="float32")), int(self.y[i])


def materialize_windows(table, store_path: str, normalize_method: str = "per_window_channel_zscore",
                        eps: float = 1e-6):
    """Load a window subset into RAM as ``(X (N,C,T), y (N,))``."""
    xs, ys = [], []
    for _, r in table.iterrows():
        x = read_window(store_path, r["file_id"], int(r["start_sample"]), int(r["end_sample"]))
        xs.append(normalize_window(x, normalize_method, eps))
        ys.append(int(r["label"]))
    if not xs:
        return np.empty((0, 0, 0), dtype="float32"), np.empty((0,), dtype="int64")
    return np.stack(xs).astype("float32"), np.asarray(ys, dtype="int64")


def negative_sample(train_windows, ratio: float = 5.0, exclude_seconds: float = 60.0,
                    seed: int = 42):
    """Build a training table: all positives + sampled eligible negatives.

    Eligible negatives are background windows at least ``exclude_seconds`` from
    any seizure (``seconds_from_nearest_seizure``), avoiding ambiguous peri-ictal
    background (Section 10).
    """
    pos = train_windows[train_windows["label"] == 1]
    neg_pool = train_windows[
        (train_windows["label"] == 0)
        & (train_windows["seconds_from_nearest_seizure"] >= exclude_seconds)
        & (~train_windows["excluded"])
    ]
    n_neg = int(round(ratio * len(pos)))
    if len(neg_pool) > n_neg:
        neg = neg_pool.sample(n=n_neg, random_state=seed)
    else:
        neg = neg_pool
    out = (
        __import__("pandas")
        .concat([pos, neg])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return out


def make_loader(dataset, batch_size: int = 64, shuffle: bool = False,
                sampler=None, num_workers: int = 0):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=num_workers, drop_last=False,
    )
