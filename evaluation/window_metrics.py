"""Window-level (sample-based) metrics (plan Section 15).

AUPRC is essential (ictal windows are rare); we also report AUROC, balanced
accuracy, sensitivity/specificity/precision/recall, macro F1 and Cohen's kappa.
Threshold-dependent metrics use a supplied operating threshold (default 0.5).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def brier_score(y_true, y_score) -> float:
    """Mean squared error of the predicted probability (lower is better).

    A proper scoring rule, so it moves with both calibration and discrimination. Added
    because the parent method's most uncomfortable result is a calibration one -- *ungated*
    augmentation was its best-calibrated arm (Brier 0.359 vs 0.390 real-only) -- and this
    benchmark could not observe that trade-off at all without it.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, y_score, n_bins: int = 15) -> float:
    """Equal-width-bin ECE: sum over bins of (bin share) x |empirical rate - mean score|.

    Scores are assumed to be probabilities; values outside [0, 1] are clipped for binning
    only, so an out-of-range score still contributes its true error to the bin gap.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(np.clip(p, 0.0, 1.0), edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def window_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out: Dict[str, float] = {}
    # Ranking metrics need both classes present.
    both = len(np.unique(y_true)) == 2
    out["auroc"] = float(roc_auc_score(y_true, y_score)) if both else float("nan")
    out["auprc"] = float(average_precision_score(y_true, y_score)) if both else float("nan")
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred)) if both else float("nan")
    # Calibration -- threshold-free, computed from the stored scores.
    out["brier"] = brier_score(y_true, y_score)
    out["ece"] = expected_calibration_error(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["precision"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    out["recall"] = out["sensitivity"]
    out["tp"], out["fp"], out["fn"], out["tn"] = int(tp), int(fp), int(fn), int(tn)
    return out


def best_threshold_by_f1(y_true, y_score, grid: Optional[np.ndarray] = None) -> float:
    """Window-level F1-optimal threshold (utility; event F1 selection is primary)."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0, 1.0], np.quantile(y_score, np.linspace(0, 1, 50))]))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
