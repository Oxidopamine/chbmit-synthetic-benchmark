"""SzCORE-aligned event-level metrics (plan Section 15).

Uses the ESL-EPFL ``timescoring`` library (SzCORE scoring) with its standard
defaults: any-overlap detection, 30 s/60 s start/end tolerance, 90 s event
merge, 5 min max event duration. Each EDF is scored independently (never merging
alarms across recording gaps, Rule 7); per-file ``tp/fp/refTrue`` are summed per
subject and metrics computed per subject, then averaged across subjects
(SzCORE convention).

Latency (first-alarm minus onset) is computed separately since timescoring does
not expose it; we report median + IQR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.postprocessing import PostprocConfig, windows_to_alarms


@dataclass
class SzCoreParams:
    toleranceStart: float = 30.0
    toleranceEnd: float = 60.0
    minOverlap: float = 0.0
    maxEventDuration: float = 5 * 60.0
    minDurationBetweenEvents: float = 90.0


@dataclass
class FilePrediction:
    file_id: str
    patient: str
    duration_sec: float
    center_times: Sequence[float]
    scores: Sequence[float]
    ref_events: List[Tuple[float, float]]  # (start_sec, end_sec)


def _score_file(ref_events, hyp_events, duration_sec, params: SzCoreParams, fs: int = 10):
    from timescoring.annotations import Annotation
    from timescoring.scoring import EventScoring

    # SzCORE EventScoring operates at 10 Hz precision; build masks at that rate
    # directly (25x smaller than 256 Hz) since finer resolution is discarded.
    num = max(1, round(duration_sec * fs))
    ref = Annotation([tuple(map(float, e)) for e in ref_events], fs, num)
    hyp = Annotation([tuple(map(float, e)) for e in hyp_events], fs, num)
    p = EventScoring.Parameters(
        toleranceStart=params.toleranceStart, toleranceEnd=params.toleranceEnd,
        minOverlap=params.minOverlap, maxEventDuration=params.maxEventDuration,
        minDurationBetweenEvents=params.minDurationBetweenEvents,
    )
    es = EventScoring(ref, hyp, p)
    return int(es.tp), int(es.fp), int(es.refTrue)


def _latencies_for_file(ref_events, hyp_alarms, tol_start: float) -> List[float]:
    out: List[float] = []
    alarms = sorted(hyp_alarms)
    for (onset, end) in ref_events:
        lo = onset - tol_start
        first = next((a for a in alarms if a[1] >= lo and a[0] <= end), None)
        if first is not None:
            out.append(first[0] - onset)
    return out


def _metrics_from_counts(tp, fp, ref_true, duration_sec) -> Dict[str, float]:
    sens = (tp / ref_true) if ref_true > 0 else float("nan")
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    if sens >= 0 and prec >= 0 and (sens + prec) > 0:
        f1 = 2 * sens * prec / (sens + prec)
    else:
        f1 = 0.0
    fp_24h = (fp / duration_sec * 86400.0) if duration_sec > 0 else float("nan")
    return {
        "event_sensitivity": sens,
        "event_precision": prec,
        "event_f1": f1,
        "fp_per_24h": fp_24h,
        "fa_per_hour": (fp_24h / 24.0) if fp_24h == fp_24h else float("nan"),
        "tp": int(tp), "fp": int(fp), "ref_true": int(ref_true),
        "duration_hours": duration_sec / 3600.0,
    }


def evaluate_event_level(
    predictions: List[FilePrediction],
    threshold: float,
    postproc: Optional[PostprocConfig] = None,
    params: Optional[SzCoreParams] = None,
) -> Dict[str, object]:
    """Score a set of files at one operating threshold; aggregate per subject."""
    postproc = postproc or PostprocConfig()
    params = params or SzCoreParams()

    per_patient: Dict[str, dict] = {}
    for fp_pred in predictions:
        alarms = windows_to_alarms(fp_pred.center_times, fp_pred.scores, threshold, postproc)
        tp, fp, ref_true = _score_file(fp_pred.ref_events, alarms, fp_pred.duration_sec, params)
        lat = _latencies_for_file(fp_pred.ref_events, alarms, params.toleranceStart)
        acc = per_patient.setdefault(
            fp_pred.patient,
            {"tp": 0, "fp": 0, "ref_true": 0, "duration_sec": 0.0, "latencies": []},
        )
        acc["tp"] += tp
        acc["fp"] += fp
        acc["ref_true"] += ref_true
        acc["duration_sec"] += fp_pred.duration_sec
        acc["latencies"].extend(lat)

    patient_metrics = {
        pat: _metrics_from_counts(a["tp"], a["fp"], a["ref_true"], a["duration_sec"])
        for pat, a in per_patient.items()
    }
    all_lat = [l for a in per_patient.values() for l in a["latencies"]]
    agg = _aggregate_over_patients(patient_metrics, all_lat)
    return {"per_patient": patient_metrics, "aggregate": agg, "threshold": threshold}


def _aggregate_over_patients(patient_metrics: Dict[str, dict], latencies: List[float]) -> dict:
    def mean_ignore_nan(key):
        vals = [m[key] for m in patient_metrics.values() if m[key] == m[key]]
        return float(np.mean(vals)) if vals else float("nan")

    tot_tp = sum(m["tp"] for m in patient_metrics.values())
    tot_fp = sum(m["fp"] for m in patient_metrics.values())
    tot_ref = sum(m["ref_true"] for m in patient_metrics.values())
    tot_hours = sum(m["duration_hours"] for m in patient_metrics.values())
    lat = np.asarray(latencies, dtype=float)
    return {
        "event_sensitivity": mean_ignore_nan("event_sensitivity"),
        "event_precision": mean_ignore_nan("event_precision"),
        "event_f1": mean_ignore_nan("event_f1"),
        "fp_per_24h": mean_ignore_nan("fp_per_24h"),
        "fa_per_hour": mean_ignore_nan("fa_per_hour"),
        "pooled_sensitivity": (tot_tp / tot_ref) if tot_ref else float("nan"),
        "pooled_fp_per_24h": (tot_fp / tot_hours * 24.0) if tot_hours else float("nan"),
        "median_latency_seconds": float(np.median(lat)) if lat.size else float("nan"),
        "latency_iqr": (
            [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))]
            if lat.size else [float("nan"), float("nan")]
        ),
        "n_patients": len(patient_metrics),
        "total_ref_events": int(tot_ref),
        "total_false_alarms": int(tot_fp),
    }


def sweep_thresholds(
    predictions: List[FilePrediction],
    thresholds: Sequence[float],
    postproc: Optional[PostprocConfig] = None,
    params: Optional[SzCoreParams] = None,
) -> List[dict]:
    """Aggregate event metrics over a grid of thresholds (for selection / curves)."""
    rows = []
    for t in thresholds:
        res = evaluate_event_level(predictions, float(t), postproc, params)
        row = {"threshold": float(t)}
        row.update(res["aggregate"])
        rows.append(row)
    return rows


def sensitivity_at_fa_budgets(
    predictions: List[FilePrediction],
    budgets_fa_per_hour: Sequence[float] = (0.5, 1.0, 2.0),
    n_thresholds: int = 41,
    postproc: Optional[PostprocConfig] = None,
    params: Optional[SzCoreParams] = None,
) -> Dict[float, float]:
    """Max event sensitivity achievable at or below each FA/hour budget."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    sweep = sweep_thresholds(predictions, thresholds, postproc, params)
    out: Dict[float, float] = {}
    for budget in budgets_fa_per_hour:
        feasible = [r for r in sweep if r["fa_per_hour"] == r["fa_per_hour"]
                    and r["fa_per_hour"] <= budget]
        out[float(budget)] = (
            float(max(r["event_sensitivity"] for r in feasible
                      if r["event_sensitivity"] == r["event_sensitivity"]))
            if feasible else float("nan")
        )
    return out
