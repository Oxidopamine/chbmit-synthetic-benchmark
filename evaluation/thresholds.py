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

    def as_dict(self) -> dict:
        return self.__dict__.copy()


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
    )
