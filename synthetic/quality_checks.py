"""Synthetic-data quality characterization (plan Section 20).

Classifier gains are not enough; we characterize the synthetic windows directly:
PSD (Welch), waveform examples, amplitude distributions, cross-channel
correlation (Frobenius distance), MMD on PSD features, nearest-neighbour
memorization vs REAL TRAINING ictal windows only, and a real-vs-synthetic
discriminator AUC (very high AUC => obvious artifacts). Run per generator so
quality differences are visible alongside downstream effects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def welch_psd(X: np.ndarray, fs: int = 256, nperseg: int = 256):
    """Average Welch PSD over windows. ``X`` (N, C, T) -> (freqs, (C, F))."""
    from scipy.signal import welch

    f, p = welch(X, fs=fs, nperseg=min(nperseg, X.shape[-1]), axis=-1)
    return f, p.mean(axis=0)  # average over windows -> (C, F)


def psd_log_distance(real: np.ndarray, synth: np.ndarray, fs: int = 256) -> float:
    _, pr = welch_psd(real, fs)
    _, ps = welch_psd(synth, fs)
    return float(np.mean(np.abs(np.log(pr + 1e-12) - np.log(ps + 1e-12))))


def cross_channel_corr_frobenius(real: np.ndarray, synth: np.ndarray) -> float:
    def avg_corr(X):
        mats = []
        for w in X:
            c = np.corrcoef(w)
            mats.append(np.nan_to_num(c))
        return np.mean(mats, axis=0)

    cr, cs = avg_corr(real), avg_corr(synth)
    return float(np.linalg.norm(cr - cs, ord="fro"))


def _psd_features(X: np.ndarray, fs: int = 256) -> np.ndarray:
    from scipy.signal import welch

    _, p = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    feats = np.log(p + 1e-12).reshape(len(X), -1)  # (N, C*F)
    return feats


def mmd_rbf(A: np.ndarray, B: np.ndarray, gamma: Optional[float] = None) -> float:
    from sklearn.metrics.pairwise import rbf_kernel

    A = np.asarray(A, dtype="float64")
    B = np.asarray(B, dtype="float64")
    if gamma is None:
        from scipy.spatial.distance import pdist

        med = np.median(pdist(np.vstack([A, B])[: min(500, len(A) + len(B))]) ** 2)
        gamma = 1.0 / (med + 1e-12)
    Kxx = rbf_kernel(A, A, gamma=gamma)
    Kyy = rbf_kernel(B, B, gamma=gamma)
    Kxy = rbf_kernel(A, B, gamma=gamma)
    m, n = len(A), len(B)
    return float(Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n))


def nearest_neighbor_memorization(synth: np.ndarray, real_train: np.ndarray,
                                  max_pairs: int = 2000) -> Dict[str, float]:
    """Distance from each synthetic window to its closest REAL TRAINING window."""
    s = synth.reshape(len(synth), -1).astype("float64")
    r = real_train.reshape(len(real_train), -1).astype("float64")
    if len(s) > max_pairs:
        s = s[np.random.default_rng(0).choice(len(s), max_pairs, replace=False)]
    # Chunked exact-Euclidean nearest neighbour via ||s-r||^2 = |s|^2 + |r|^2 - 2 s.r^T.
    # Only forms (chunk, n_real) matrices, never the (chunk, n_real, D) broadcast that
    # would blow up memory for high-dimensional (C*T) windows.
    r_sq = np.einsum("ij,ij->i", r, r)
    mins = []
    for i in range(0, len(s), 256):
        sc = s[i:i + 256]
        d2 = np.einsum("ij,ij->i", sc, sc)[:, None] + r_sq[None, :] - 2.0 * (sc @ r.T)
        np.maximum(d2, 0.0, out=d2)
        mins.append(np.sqrt(d2.min(axis=1)))
    mins = np.concatenate(mins)
    return {"nn_dist_mean": float(mins.mean()), "nn_dist_min": float(mins.min()),
            "nn_dist_p05": float(np.percentile(mins, 5))}


def discriminator_auc(real: np.ndarray, synth: np.ndarray, fs: int = 256) -> float:
    """5-fold CV AUC of a logistic discriminator on PSD features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    Xr, Xs = _psd_features(real, fs), _psd_features(synth, fs)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xs))])
    n_splits = int(min(5, np.bincount(y.astype(int)).min()))
    if n_splits < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=200)
    return float(np.mean(cross_val_score(clf, X, y, cv=n_splits, scoring="roc_auc")))


def run_quality_suite(real: np.ndarray, synth: np.ndarray, fs: int, out_dir: str | Path,
                      provider_name: str, make_figures: bool = True) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "provider": provider_name,
        "n_real": int(len(real)),
        "n_synth": int(len(synth)),
        "psd_log_distance": psd_log_distance(real, synth, fs),
        "cross_channel_corr_frobenius": cross_channel_corr_frobenius(real, synth),
        "mmd_psd": mmd_rbf(_psd_features(real, fs), _psd_features(synth, fs)),
        "discriminator_auc": discriminator_auc(real, synth, fs),
        **nearest_neighbor_memorization(synth, real),
    }
    (out_dir / f"quality_{provider_name}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    if make_figures:
        try:
            _quality_figures(real, synth, fs, out_dir, provider_name)
        except Exception as e:  # figures are best-effort
            metrics["figure_error"] = str(e)
    return metrics


def _quality_figures(real, synth, fs, out_dir, provider_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, pr = welch_psd(real, fs)
    _, ps = welch_psd(synth, fs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(f, pr.mean(0), label="real")
    ax[0].semilogy(f, ps.mean(0), label="synthetic")
    ax[0].set_title(f"Mean PSD ({provider_name})")
    ax[0].set_xlabel("Hz"); ax[0].legend()
    ax[1].plot(real[0, 0], label="real ch0")
    ax[1].plot(synth[0, 0], label="synthetic ch0", alpha=0.8)
    ax[1].set_title("Example waveform (channel 0)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"quality_{provider_name}.png", dpi=120)
    plt.close(fig)
