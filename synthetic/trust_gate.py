"""Event-level fail-closed trust gate for synthetic ictal augmentation (v5.3 Sec 3.1, 5).

This is an ADAPTATION of fail-closed trust-gated augmentation (TGA: Choi, Yip, Choi & Park,
*npj Digital Medicine* 9(1) 634, 2026, ``10.1038/s41746-026-02778-0``, PMID 42185473) to
the seizure-specific, event-level setting. It is *not* proposed here as novel; the
contributions of this benchmark are the seizure-specific event-level reformulation of
the admission/fail-closed criteria and the harm characterization the gate is meant to
address (see ``PREREGISTRATION.md``). The gate has two stages:

1. **Admission (window-level).** Score every candidate synthetic ictal window with the
   real-only detector teacher (its seizure-class probability) and admit only windows
   whose teacher confidence is at least the ``q``-quantile of a reference distribution.
   High ``q`` => stricter admission.

   The reference is the candidate pool itself (``reference="pool"``, the rank cut TGA actually
   publishes, and the DEFAULT here) or the *real* training ictal windows
   (``reference="real_ictal"``, as this benchmark was originally built). **Nothing here computes
   a manifold distance** -- the criterion is the teacher detector's seizure confidence and
   nothing else. TGA's covariance-manifold audit is a separate, unimplemented component; do not
   describe this gate as enforcing it. The real-ictal reference is badly conditioned in
   practice, because the teacher saturates on real ictal: q 0.50 -> 0.90 moves the threshold by
   0.037 and changes admission 174x, q = 0.99 admits nothing at all (``the verification record``
   §2.1), and in the live grids it admitted 6 windows against a target of 251 -- i.e. it
   disabled the mechanism (``DECISION_GATE_2.md`` Q6). It is retained to reproduce Phases 1-2
   and for the positive control, never as a default.

2. **Fail-closed selection (event-level).** After training the augmented detector on
   real + admitted synthetic, compare its VALIDATION event-F1 / FP-24h to the real-only
   teacher. Admit the augmented model only if it improves validation event-F1 by at
   least ``admit_margin_event_f1`` AND does not inflate validation FP/24h beyond
   ``fp24h_safety_slack``. Otherwise revert to the real-only model (fail-closed).

The teacher is the real-only model itself (v5.3 Sec 5.1), so the gate adds only cheap
inference plus one already-trained augmented model -- no extra teacher training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class TrustGateConfig:
    """Pre-registered trust-gate hyper-parameters (see ``PREREGISTRATION.md``)."""

    q: float = 0.90                     # core validation-selected admission quantile
    oversample: int = 6                 # candidate pool = oversample x target synthetic count
    admit_margin_event_f1: float = 0.0  # gated val event-F1 must beat teacher by >= this
    fp24h_safety_slack: float = 0.25    # gated val FP/24h must not exceed teacher by > this
    min_admitted: int = 1               # below this, treat as "nothing admitted" (still trains real-only-like)
    score_batch_size: int = 256
    # Admission reference distribution. "pool" is the published TGA rule (a rank cut on the
    # candidate pool itself) and is the default since 2026-08-31: admitted =
    # min(oversample*(1-q), 1) * n_synth EXACTLY, so q is direct dose control.
    # "real_ictal" is what this benchmark originally built (quantile of the teacher's confidence
    # on REAL training ictal windows, which the teacher has MEMORISED). It admits 6 windows
    # against a target of 251 and 23 against 752 -- the admitted count is decoupled from the
    # request, and no ratio ladder moves it. It is kept ONLY to reproduce Phases 1-2 and for the
    # positive control, where a real-ictal candidate pool makes it the right reference.
    # See the verification record Sec 2.1 and DECISION_GATE_2.md Q6.
    reference: str = "pool"             # "pool" (published) | "real_ictal" (Phases 1-2)
    # Which windows to keep once the admitted COUNT is fixed. "random" is the matched-volume
    # control the parent paper runs: same number of windows, drawn uniformly from the pool.
    selection: str = "teacher"          # "teacher" | "random"
    selection_seed: int = 0             # RNG seed for selection="random"


def score_windows(model, X: np.ndarray, device: str = "cpu",
                  batch_size: int = 256) -> np.ndarray:
    """Teacher seizure-class probability for each ``(C, T)`` window in ``X`` (N, C, T)."""
    X = np.asarray(X, dtype="float32")
    if X.size == 0:
        return np.empty((0,), dtype="float32")
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy().reshape(-1))
    return np.concatenate(out).astype("float32")


def admission_threshold(reference_scores: np.ndarray, q: float) -> float:
    """``q``-quantile of teacher confidence on the reference (real ictal) windows."""
    ref = np.asarray([s for s in np.ravel(reference_scores) if s == s], dtype=float)
    if ref.size == 0:
        return 0.0  # no reference => admit everything (fail-closed selector still guards)
    return float(np.quantile(ref, float(q)))


def admit_indices(synth_scores: np.ndarray, threshold: float,
                  max_keep: Optional[int] = None) -> np.ndarray:
    """Indices of synthetic windows admitted (teacher score >= threshold).

    When more than ``max_keep`` qualify, keep the highest-confidence ones so the injected
    count matches the ungated condition for a fair paired comparison.
    """
    scores = np.asarray(synth_scores, dtype=float)
    keep = np.where(scores >= threshold)[0]
    if keep.size == 0:
        return keep
    order = keep[np.argsort(-scores[keep], kind="stable")]
    if max_keep is not None and order.size > max_keep:
        order = order[:max_keep]
    return np.sort(order)


@dataclass
class GateAdmission:
    threshold: float
    n_pool: int
    n_admitted: int
    n_injected: int
    admitted_index: np.ndarray
    admission_rate: float


def run_admission(
    teacher_model,
    pool_windows: np.ndarray,
    real_ictal_windows: Optional[np.ndarray],
    cfg: TrustGateConfig,
    target_count: int,
    device: str = "cpu",
) -> GateAdmission:
    """Stage 1: score the candidate pool, calibrate ``q`` on real ictal, admit top windows."""
    pool_scores = score_windows(teacher_model, pool_windows, device, cfg.score_batch_size)
    if cfg.reference == "pool":
        ref_scores = pool_scores
    elif real_ictal_windows is not None and len(real_ictal_windows):
        ref_scores = score_windows(teacher_model, real_ictal_windows, device, cfg.score_batch_size)
    else:  # fall back to calibrating on the pool itself when no real ictal is available
        ref_scores = pool_scores
    thr = admission_threshold(ref_scores, cfg.q)
    idx = admit_indices(pool_scores, thr, max_keep=target_count)

    # Matched-volume random control: keep the SAME number of windows the teacher would have
    # admitted, but draw them uniformly from the pool. Isolates selection quality from dose.
    if cfg.selection == "random" and idx.size:
        rng = np.random.default_rng(cfg.selection_seed)
        idx = np.sort(rng.choice(len(pool_scores), size=int(idx.size), replace=False))

    n_pool = int(len(pool_scores))
    return GateAdmission(
        threshold=thr,
        n_pool=n_pool,
        n_admitted=int(len(idx)),
        n_injected=int(len(idx)),
        admitted_index=idx,
        admission_rate=(float(len(idx)) / n_pool) if n_pool else 0.0,
    )


@dataclass
class GateDecision:
    admitted: bool
    reverted: bool
    reason: str


def fail_closed_decision(
    gated_val_event_f1: float,
    gated_val_fp_per_24h: float,
    teacher_val_event_f1: float,
    teacher_val_fp_per_24h: float,
    cfg: TrustGateConfig,
    n_admitted: int = 1,
) -> GateDecision:
    """Stage 2: admit the augmented model only if it clears the event-F1 margin under
    the FP/24h safety constraint; otherwise fail closed (revert to real-only)."""
    if n_admitted < cfg.min_admitted:
        return GateDecision(False, True, "no_synthetic_admitted")
    if gated_val_event_f1 != gated_val_event_f1:  # NaN
        return GateDecision(False, True, "gated_val_event_f1_nan")

    f1_ok = gated_val_event_f1 >= (teacher_val_event_f1 + cfg.admit_margin_event_f1)
    # FP/24h safety constraint; if the teacher's val FP is undefined, do not block on it.
    if teacher_val_fp_per_24h != teacher_val_fp_per_24h or \
            gated_val_fp_per_24h != gated_val_fp_per_24h:
        fp_ok = True
    else:
        fp_ok = gated_val_fp_per_24h <= (teacher_val_fp_per_24h + cfg.fp24h_safety_slack)

    if f1_ok and fp_ok:
        return GateDecision(True, False, "admitted")
    if not f1_ok:
        return GateDecision(False, True, "val_event_f1_below_margin")
    return GateDecision(False, True, "val_fp24h_exceeds_safety_slack")
