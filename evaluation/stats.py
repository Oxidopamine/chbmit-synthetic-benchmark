"""Paired statistics and tail-risk across folds/patients (v5.3 Sec 0, 6; plan Sec 17).

Headline is the bootstrap 95% CI on paired deltas; the paired Wilcoxon signed-rank
p-value is reported as exploratory (small n). Deltas are paired by fold/patient group:
``ungated_synthetic_aug - real_only`` etc. Because the v5.3 thesis is that *mean* window
metrics conceal event-level harm, every paired delta is also reported with tail-risk:
harm rate (fraction of folds past a pre-registered harm threshold), worst-fold delta, and
CVaR (mean of the worst ``alpha`` fraction).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def _clean(values: Sequence[float]) -> np.ndarray:
    return np.asarray([v for v in np.ravel(np.asarray(values, dtype=float)) if v == v], dtype=float)


def cvar(values: Sequence[float], alpha: float = 0.10,
         higher_is_better: bool = True) -> float:
    """Conditional value at risk: mean of the worst ``alpha`` fraction of ``values``.

    For a higher-is-better metric (event-F1) the worst tail is the lowest deltas; for a
    lower-is-better metric (FP/24h) it is the highest deltas.
    """
    vals = _clean(values)
    if vals.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(alpha * vals.size)))
    s = np.sort(vals)
    tail = s[:k] if higher_is_better else s[-k:]
    return float(np.mean(tail))


def worst_delta(values: Sequence[float], higher_is_better: bool = True) -> float:
    """Worst single paired delta (min for higher-is-better, max for lower-is-better)."""
    vals = _clean(values)
    if vals.size == 0:
        return float("nan")
    return float(np.min(vals)) if higher_is_better else float(np.max(vals))


def harm_rate(values: Sequence[float], threshold: float,
              higher_is_better: bool = True) -> float:
    """Fraction of paired deltas that count as HARM under the pre-registered threshold.

    higher_is_better (event-F1): harm if delta < threshold (a negative threshold).
    lower_is_better (FP/24h):    harm if delta > threshold (a positive threshold).
    """
    vals = _clean(values)
    if vals.size == 0:
        return float("nan")
    harmed = (vals < threshold) if higher_is_better else (vals > threshold)
    return float(np.mean(harmed))


@dataclass
class PairedResult:
    n: int
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]
    worst_delta: Optional[float] = None
    cvar: Optional[float] = None
    harm_rate: Optional[float] = None
    harm_threshold: Optional[float] = None
    higher_is_better: Optional[bool] = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def bootstrap_ci(values: Sequence[float], ci: float = 0.95, n_boot: int = 10000,
                 seed: int = 42) -> tuple:
    vals = np.asarray([v for v in values if v == v], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo = (1 - ci) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, 100 - lo))


def paired_delta(a: Sequence[float], b: Sequence[float], ci: float = 0.95,
                 seed: int = 42, higher_is_better: bool = True,
                 harm_threshold: Optional[float] = None,
                 cvar_alpha: float = 0.10) -> PairedResult:
    """Paired delta a-b with bootstrap CI, exploratory Wilcoxon, and tail-risk.

    ``higher_is_better`` orients the tail-risk metrics (worst-fold delta, CVaR, harm rate)
    so the same call works for event-F1 (True) and FP/24h (False). ``harm_threshold`` is
    the pre-registered harm cut-off; when ``None`` the harm rate is left undefined.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    deltas = (a - b)[mask]
    stat, p = _wilcoxon(deltas)
    lo, hi = bootstrap_ci(deltas, ci=ci, seed=seed)
    return PairedResult(
        n=int(deltas.size),
        mean_delta=float(np.mean(deltas)) if deltas.size else float("nan"),
        median_delta=float(np.median(deltas)) if deltas.size else float("nan"),
        ci_low=lo, ci_high=hi, wilcoxon_stat=stat, wilcoxon_p=p,
        worst_delta=worst_delta(deltas, higher_is_better),
        cvar=cvar(deltas, cvar_alpha, higher_is_better),
        harm_rate=(harm_rate(deltas, harm_threshold, higher_is_better)
                   if harm_threshold is not None else None),
        harm_threshold=harm_threshold,
        higher_is_better=higher_is_better,
    )


def _wilcoxon(deltas: np.ndarray):
    nz = deltas[deltas != 0]
    if nz.size < 1:
        return None, None
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return None, None


def paired_delta_table(
    metric_by_condition: Dict[str, Dict[str, float]],
    reference_conditions: Sequence[str] = ("real_only", "classical_aug", "class_weighted"),
    target_condition: str = "ungated_synthetic_aug",
    ci: float = 0.95,
    higher_is_better: bool = True,
    harm_threshold: Optional[float] = None,
    cvar_alpha: float = 0.10,
) -> Dict[str, dict]:
    """For ``target - ref`` over shared groups, build paired deltas (with tail-risk) per ref.

    ``metric_by_condition[condition] = {group: metric_value}``.
    """
    out: Dict[str, dict] = {}
    target = metric_by_condition.get(target_condition, {})
    for ref in reference_conditions:
        ref_map = metric_by_condition.get(ref, {})
        groups = sorted(set(target) & set(ref_map))
        a = [target[g] for g in groups]
        b = [ref_map[g] for g in groups]
        out[f"{target_condition}-{ref}"] = {
            "groups": groups,
            **paired_delta(a, b, ci=ci, higher_is_better=higher_is_better,
                           harm_threshold=harm_threshold, cvar_alpha=cvar_alpha).as_dict(),
        }
    return out
