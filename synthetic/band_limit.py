"""Band-limit synthetic ictal windows to the real acquisition passband.

The spectral-gap diagnostic (scripts/gen_spectral_gap_diag.py) showed that both
generators reproduce the in-band (0.5-40 Hz) ictal spectrum reasonably -- the WGAN-GP
nearly perfectly -- but emit a broadband floor above 40 Hz where the real signal,
having been zero-phase band-passed in preprocessing, has ~10 orders of magnitude less
power. That out-of-band energy is the single largest real-vs-synthetic discrepancy and
almost certainly the strongest cue the real-only teacher detector keys on when it
scores synthetic windows as non-seizure (driving trust-gate admission to 0).

This applies the SAME zero-phase Butterworth band-pass used in preprocessing
(chbmit.preprocess_edf._design_bandpass) to synthetic windows, then re-applies the
per-window/per-channel z-score so the output stays in the normalized training space the
detectors expect. Use it to wrap a provider's ``generate`` before injection/admission.
"""
from __future__ import annotations

import numpy as np

from chbmit.preprocess_edf import _design_bandpass


def band_limit_windows(X: np.ndarray, fs: int = 256, low: float = 0.5,
                       high: float = 40.0, order: int = 4,
                       restandardize: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Zero-phase band-pass ``(N, C, T)`` synthetic windows and re-standardize.

    ``restandardize`` re-applies per-window/per-channel z-score (the pipeline's
    ``per_window_channel_zscore``) after filtering, since band-passing changes each
    channel's scale and the detectors consume z-scored input.
    """
    from scipy.signal import sosfiltfilt

    X = np.asarray(X, dtype="float64")
    if X.ndim != 3:
        raise ValueError(f"expected (N, C, T), got shape {X.shape}")
    sos = _design_bandpass(float(fs), low, high, order)
    Y = sosfiltfilt(sos, X, axis=-1)
    if restandardize:
        mu = Y.mean(axis=-1, keepdims=True)
        sd = Y.std(axis=-1, keepdims=True)
        Y = (Y - mu) / (sd + eps)
    return np.ascontiguousarray(Y, dtype="float32")
