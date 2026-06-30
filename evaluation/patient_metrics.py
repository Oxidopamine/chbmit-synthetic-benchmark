"""Patient-level aggregation (plan Section 15).

Compute window metrics per test patient, then aggregate (mean +- SD and
median + IQR) so high-window-count patients do not dominate pooled results.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from evaluation.window_metrics import window_metrics


def per_patient_window_metrics(
    patient_to_arrays: Dict[str, tuple], threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """``{patient: (y_true, y_score)}`` -> ``{patient: window_metrics}``."""
    return {
        pat: window_metrics(y_true, y_score, threshold)
        for pat, (y_true, y_score) in patient_to_arrays.items()
    }


def aggregate_patient_metrics(per_patient: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    """Aggregate per-patient metric dicts into mean/SD/median/IQR per metric."""
    if not per_patient:
        return {}
    keys = [k for k in next(iter(per_patient.values())) if k not in {"tp", "fp", "fn", "tn"}]
    out: Dict[str, dict] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_patient.values()], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out[k] = {"mean": float("nan"), "sd": float("nan"),
                      "median": float("nan"), "iqr": [float("nan"), float("nan")], "n": 0}
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "median": float(np.median(vals)),
            "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
            "n": int(vals.size),
        }
    return out
