"""Alarm post-processing: per-window scores -> alarm events (plan Section 15).

Pipeline per EDF (never across recording gaps):

    threshold -> k-of-n persistence filter -> merge consecutive positives
    -> merge alarms separated by < min_alarm_gap

Each active window covers ``[center - window/2, center + window/2]``; runs of
active windows union into alarm intervals. Operates on a single file's ordered
windows; callers iterate per file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class PostprocConfig:
    k: int = 3
    n: int = 5
    min_alarm_gap_seconds: float = 60.0
    window_seconds: float = 4.0


def k_of_n(pred: np.ndarray, k: int, n: int) -> np.ndarray:
    """Causal k-of-n persistence: window i active if >=k of trailing n preds are 1."""
    pred = np.asarray(pred, dtype=int)
    if len(pred) == 0:
        return pred.astype(bool)
    csum = np.cumsum(pred)
    active = np.zeros(len(pred), dtype=bool)
    for i in range(len(pred)):
        lo = i - n + 1
        s = csum[i] - (csum[lo - 1] if lo - 1 >= 0 else 0)
        active[i] = s >= k
    return active


def _merge_intervals(intervals: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def windows_to_alarms(
    center_times: Sequence[float],
    scores: Sequence[float],
    threshold: float,
    cfg: PostprocConfig,
) -> List[Tuple[float, float]]:
    """Convert ordered per-window scores for one file into alarm intervals."""
    center_times = np.asarray(center_times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(center_times)
    center_times, scores = center_times[order], scores[order]

    pred = (scores >= threshold).astype(int)
    active = k_of_n(pred, cfg.k, cfg.n)

    half = cfg.window_seconds / 2.0
    raw = [(float(t - half), float(t + half)) for t, a in zip(center_times, active) if a]
    # Runs of active windows overlap (stride < window), so a 0-gap merge unions
    # them; then merge alarms separated by < min_alarm_gap.
    runs = _merge_intervals(raw, gap=1e-9)
    return _merge_intervals(runs, gap=cfg.min_alarm_gap_seconds)
