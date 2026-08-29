"""Threshold selection on validation patients only (plan Section 16).

Primary objective: maximize validation event F1; tie-break on lower FP/24h.
Thresholds are never tuned on test patients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from evaluation.event_metrics_szcore import (
    FilePrediction,
    PostprocConfig,
    SzCoreParams,
    sweep_thresholds,
)


@dataclass
class ThresholdSelection:
    selected_threshold: float
    selection_metric: str
    validation_event_f1: float
    validation_fp_per_24h: float
    persistence_k: int
    persistence_n: int
    min_alarm_gap_seconds: float
    # Threshold-free ranking quality on validation. Defaulted so the dataclass stays
    # constructible by older callers; NaN simply means "not computed".
    validation_auprc: float = float("nan")

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compute_validation_auprc(val_predictions: List[FilePrediction]) -> float:
    """Window-level average precision on validation — a statistic with no operating point.

    ``validation_event_f1`` above is a MAX over a 19-point threshold sweep, so the fail-closed
    rule compares two *selected maxima* at margin 0 (``the verification record`` §4.2) — the same
    selected-maximum defect the analysis was corrected for. The parent method instead compares a
    threshold-free statistic at margin 0.01. Emitting this per arm lets that comparison be made
    in ANALYSIS, without a second grid: both the augmented and real-only models already exist in
    every cell, so re-deciding admission is a re-selection among trained arms — but only if the
    statistic was written down at run time.

    Labels follow the sweep's own convention: a window is positive when its center time falls
    inside a reference event. Returns NaN when validation has only one class.
    """
    ys, ss = [], []
    for p in val_predictions:
        t = np.asarray(p.center_times, dtype=float)
        if not t.size:
            continue
        lab = np.zeros(t.shape, dtype=bool)
        for a, b in (p.ref_events or []):
            lab |= (t >= float(a)) & (t <= float(b))
        ys.append(lab)
        ss.append(np.asarray(p.scores, dtype=float))
    if not ys:
        return float("nan")
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if y.all() or not y.any():          # single-class validation => AP undefined
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y.astype(int), s))


def select_threshold(
    val_predictions: List[FilePrediction],
    postproc: Optional[PostprocConfig] = None,
    params: Optional[SzCoreParams] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> ThresholdSelection:
    postproc = postproc or PostprocConfig()
    params = params or SzCoreParams()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    sweep = sweep_thresholds(val_predictions, thresholds, postproc, params)

    def key(row):
        f1 = row["event_f1"] if row["event_f1"] == row["event_f1"] else -1.0
        fp = row["fp_per_24h"] if row["fp_per_24h"] == row["fp_per_24h"] else 1e18
        # maximize F1, then minimize FP/24h, then prefer higher threshold (fewer FAs)
        return (f1, -fp, row["threshold"])

    best = max(sweep, key=key)
    return ThresholdSelection(
        selected_threshold=float(best["threshold"]),
        selection_metric="validation_event_f1",
        validation_event_f1=float(best["event_f1"]) if best["event_f1"] == best["event_f1"] else float("nan"),
        validation_fp_per_24h=float(best["fp_per_24h"]) if best["fp_per_24h"] == best["fp_per_24h"] else float("nan"),
        persistence_k=postproc.k,
        persistence_n=postproc.n,
        min_alarm_gap_seconds=postproc.min_alarm_gap_seconds,
        validation_auprc=compute_validation_auprc(val_predictions),
    )
