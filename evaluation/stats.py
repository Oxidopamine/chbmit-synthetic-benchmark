"""Paired statistics across folds/patients (plan Section 17).

Headline is the bootstrap 95% CI on paired deltas; the paired Wilcoxon
signed-rank p-value is reported as exploratory (small n). Deltas are paired by
fold/patient group: ``synthetic_aug - real_only`` etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class PairedResult:
    n: int
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]

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
                 seed: int = 42) -> PairedResult:
    """Paired delta a-b with bootstrap CI and exploratory Wilcoxon."""
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
    target_condition: str = "synthetic_aug",
    ci: float = 0.95,
) -> Dict[str, dict]:
    """For ``target - ref`` over shared groups, build paired deltas per reference.

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
            **paired_delta(a, b, ci=ci).as_dict(),
        }
    return out
