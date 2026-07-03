"""Spectral-gap diagnostic: WHY is real-vs-synth log-PSD perfectly separable?

Both generators produce discriminator AUC = 1.00 on log-PSD features, yet the
WGAN-GP has healthy diversity (ratio ~0.94) and realistic amplitude -- so the
failure is spectral SHAPE, not collapse. This localizes the gap: it computes the
mean Welch PSD of real vs WGAN vs cVAE ictal windows per frequency and the
per-frequency |log-PSD| gap that a logistic discriminator keys on.

Hypothesis: the stride-2 ConvTranspose1d up-sampling stack injects spurious
high-frequency (checkerboard) power that real 0.5-40 Hz band-passed EEG lacks,
making the classes linearly separable regardless of low-band realism.

Trains both generators on fold 0 to a representative budget (WGAN MMD plateaus by
~300 ep; cVAE run at beta=0.01, its best-diversity setting). Persists the WGAN
generator for reuse, writes per-frequency arrays + a 2-panel figure.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import zarr

from chbmit.splits import Split
from synthetic.cvae_provider import CVAEConfig
from synthetic.wgan_gp_provider import WGANConfig
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.quality_checks import welch_psd, discriminator_auc

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
FIGDIR = OUT / "figures"
WGAN_EPOCHS = 400
CVAE_EPOCHS = 400


def _mean_psd_over_ch(X, fs):
    """Welch PSD averaged over windows then channels -> (freqs, psd[F])."""
    f, p = welch_psd(X, fs)          # p: (C, F)
    return f, p.mean(axis=0)         # (F,)


def main():
    t0 = time.time()
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    f0 = sp["folds"][0]
    split = Split(fold=0, train_groups=f0["train_groups"],
                  val_groups=f0["val_groups"], test_groups=f0["test_groups"])
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])

    probe = build_provider("wgan_gp", WGANConfig(epochs=0, min_ictal_windows=256,
                                                 device="cuda", seed=42))
    info = fit_provider_for_cell(probe, index_df, win, ev, STORE, split, 0, 42, 1.0)
    real = info["real_ictal"]
    n = min(len(real), 500)
    print(f"[{time.time()-t0:5.0f}s] real ready n={len(real)} shape={real.shape}", flush=True)

    # WGAN-GP
    wgan = build_provider("wgan_gp", WGANConfig(epochs=WGAN_EPOCHS, min_ictal_windows=256,
                                                device="cuda", seed=42))
    fit_provider_for_cell(wgan, index_df, win, ev, STORE, split, 0, 42, 1.0)
    wgan_synth = wgan.generate(n, seed=42)
    gdir = RES / "generators" / "wgan_fold0"
    gdir.mkdir(parents=True, exist_ok=True)
    wgan._save_state(gdir)
    print(f"[{time.time()-t0:5.0f}s] wgan trained+saved", flush=True)

    # cVAE at beta=0.01 (best-diversity setting from the beta sweep)
    cvae = build_provider("cvae", CVAEConfig(epochs=CVAE_EPOCHS, beta=0.01,
                                             min_ictal_windows=256, device="cuda", seed=42))
    fit_provider_for_cell(cvae, index_df, win, ev, STORE, split, 0, 42, 1.0)
    cvae_synth = cvae.generate(n, seed=42)
    print(f"[{time.time()-t0:5.0f}s] cvae trained", flush=True)

    f, pr = _mean_psd_over_ch(real, fs)
    _, pw = _mean_psd_over_ch(wgan_synth, fs)
    _, pc = _mean_psd_over_ch(cvae_synth, fs)
    gap_w = np.abs(np.log(pr + 1e-12) - np.log(pw + 1e-12))
    gap_c = np.abs(np.log(pr + 1e-12) - np.log(pc + 1e-12))

    # Discriminator AUC restricted to the 0.5-40 Hz analysis band vs full band,
    # to test whether separability lives in the out-of-band (>40 Hz) region.
    def band_auc(synth, lo=0.5, hi=40.0):
        from scipy.signal import welch
        _, pr_ = welch(real, fs=fs, nperseg=min(256, real.shape[-1]), axis=-1)
        ff, ps_ = welch(synth, fs=fs, nperseg=min(256, synth.shape[-1]), axis=-1)
        m = (ff >= lo) & (ff <= hi)
        Xr = np.log(pr_[..., m] + 1e-12).reshape(len(real), -1)
        Xs = np.log(ps_[..., m] + 1e-12).reshape(len(synth), -1)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        X = np.vstack([Xr, Xs]); y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
        return float(np.mean(cross_val_score(LogisticRegression(max_iter=200), X, y,
                                             cv=5, scoring="roc_auc")))

    summary = {
        "fs": fs, "n_synth": n,
        "wgan_full_discAUC": round(discriminator_auc(real, wgan_synth, fs), 3),
        "wgan_inband_discAUC": round(band_auc(wgan_synth), 3),
        "cvae_full_discAUC": round(discriminator_auc(real, cvae_synth, fs), 3),
        "cvae_inband_discAUC": round(band_auc(cvae_synth), 3),
        "wgan_outband_frac_power": round(float(pw[f > 40].sum() / pw.sum()), 4),
        "cvae_outband_frac_power": round(float(pc[f > 40].sum() / pc.sum()), 4),
        "real_outband_frac_power": round(float(pr[f > 40].sum() / pr.sum()), 4),
        "wgan_gap_inband_mean": round(float(gap_w[(f >= 0.5) & (f <= 40)].mean()), 3),
        "wgan_gap_outband_mean": round(float(gap_w[f > 40].mean()), 3),
        "cvae_gap_inband_mean": round(float(gap_c[(f >= 0.5) & (f <= 40)].mean()), 3),
    }
    np.savez(OUT / "spectral_gap.npz", freqs=f, psd_real=pr, psd_wgan=pw, psd_cvae=pc,
             gap_wgan=gap_w, gap_cvae=gap_c)
    (OUT / "spectral_gap.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogy(f, pr, label="real", lw=2, color="k")
    ax[0].semilogy(f, pw, label="WGAN-GP", lw=1.5)
    ax[0].semilogy(f, pc, label="cVAE (beta=0.01)", lw=1.5)
    ax[0].axvspan(0.5, 40, color="green", alpha=0.06, label="0.5-40 Hz band")
    ax[0].set_title("Mean PSD, real vs synthetic (fold 0 ictal)")
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("power"); ax[0].legend(fontsize=8)
    ax[1].plot(f, gap_w, label="|log-PSD| gap WGAN")
    ax[1].plot(f, gap_c, label="|log-PSD| gap cVAE")
    ax[1].axvspan(0.5, 40, color="green", alpha=0.06)
    ax[1].set_title("Per-frequency log-PSD gap (what the discriminator keys on)")
    ax[1].set_xlabel("Hz"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "spectral_gap.png", dpi=130)
    plt.close(fig)

    print(f"\n[{time.time()-t0:5.0f}s] === SPECTRAL GAP SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT/'spectral_gap.json'}, .npz, and {FIGDIR/'spectral_gap.png'}", flush=True)


if __name__ == "__main__":
    main()
