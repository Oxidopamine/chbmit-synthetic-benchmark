"""Validate scoring on KNOWN cases BEFORE any model (plan Section 23 step 7)."""
import numpy as np

from evaluation.event_metrics_szcore import (
    FilePrediction,
    evaluate_event_level,
    sensitivity_at_fa_budgets,
)
from evaluation.postprocessing import PostprocConfig, k_of_n, windows_to_alarms
from evaluation.stats import bootstrap_ci, paired_delta
from evaluation.window_metrics import window_metrics


def _centers(duration=60.0, stride=2.0, win=4.0):
    return np.arange(win / 2, duration - win / 2 + 1e-9, stride)


def _scores_for_interval(centers, lo, hi):
    return ((centers >= lo) & (centers < hi)).astype(float)


# --- post-processing primitives ----------------------------------------
def test_k_of_n_persistence():
    pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    active = k_of_n(pred, k=3, n=5)
    # First time 3 of trailing 5 are positive is at index 8 (windows 6,7,8).
    assert active[8]
    assert not active[:7].any()


def test_isolated_positive_makes_no_alarm():
    centers = _centers()
    scores = np.zeros_like(centers)
    scores[5] = 1.0  # single positive window
    alarms = windows_to_alarms(centers, scores, 0.5, PostprocConfig(k=3, n=5))
    assert alarms == []


def test_alarm_merging_within_min_gap():
    centers = _centers(duration=120.0)
    scores = np.zeros_like(centers)
    # two bursts ~30 s apart; min_alarm_gap=60 should merge them
    for c in (20, 22, 24):
        scores[np.argmin(np.abs(centers - c))] = 1.0
    for c in (50, 52, 54):
        scores[np.argmin(np.abs(centers - c))] = 1.0
    alarms = windows_to_alarms(centers, scores, 0.5, PostprocConfig(k=3, n=5, min_alarm_gap_seconds=60))
    assert len(alarms) == 1


# --- event-level scoring on known cases --------------------------------
def _pred(file_id, patient, scores, ref_events, duration=60.0):
    return FilePrediction(file_id, patient, duration, _centers(duration), scores, ref_events)


def test_perfect_detection():
    centers = _centers()
    scores = _scores_for_interval(centers, 20, 40)
    pred = _pred("f1", "chb01", scores, [(20.0, 40.0)])
    res = evaluate_event_level([pred], threshold=0.5, postproc=PostprocConfig(k=3, n=5))
    agg = res["aggregate"]
    assert agg["pooled_sensitivity"] == 1.0
    assert res["per_patient"]["chb01"]["fp"] == 0
    assert res["per_patient"]["chb01"]["event_sensitivity"] == 1.0
    assert 0.0 <= agg["median_latency_seconds"] <= 10.0


def test_missed_seizure():
    centers = _centers()
    scores = np.zeros_like(centers)
    pred = _pred("f1", "chb01", scores, [(20.0, 40.0)])
    res = evaluate_event_level([pred], threshold=0.5, postproc=PostprocConfig(k=3, n=5))
    assert res["per_patient"]["chb01"]["event_sensitivity"] == 0.0
    assert res["per_patient"]["chb01"]["fp"] == 0


def test_false_alarm_on_seizure_free_file():
    centers = _centers()
    scores = _scores_for_interval(centers, 30, 40)  # spurious burst, no real seizure
    pred = _pred("f1", "chb01", scores, [])
    res = evaluate_event_level([pred], threshold=0.5, postproc=PostprocConfig(k=3, n=5))
    pm = res["per_patient"]["chb01"]
    assert pm["fp"] >= 1
    assert pm["fp_per_24h"] > 0
    # 1 FA in a 60 s file -> 1440 FP/24h
    assert abs(res["aggregate"]["pooled_fp_per_24h"] - 1440.0) < 1e-6


def test_sensitivity_at_fa_budgets_monotone():
    centers = _centers()
    scores = _scores_for_interval(centers, 20, 40) * 0.9 + 0.05
    pred = _pred("f1", "chb01", scores, [(20.0, 40.0)])
    out = sensitivity_at_fa_budgets([pred], budgets_fa_per_hour=(0.5, 1.0, 2.0))
    # sensitivity at a looser budget is >= sensitivity at a stricter budget
    vals = [out[0.5], out[1.0], out[2.0]]
    vals = [v if v == v else 0.0 for v in vals]
    assert vals[2] >= vals[1] >= vals[0] - 1e-9


# --- window metrics & stats --------------------------------------------
def test_window_metrics_known():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = window_metrics(y_true, y_score, threshold=0.5)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_paired_delta_and_bootstrap():
    a = [0.8, 0.7, 0.9, 0.85]
    b = [0.6, 0.65, 0.7, 0.75]
    r = paired_delta(a, b)
    assert r.n == 4
    assert r.mean_delta > 0
    assert r.ci_low <= r.mean_delta <= r.ci_high
    lo, hi = bootstrap_ci([1.0, 1.0, 1.0])
    assert lo == hi == 1.0
