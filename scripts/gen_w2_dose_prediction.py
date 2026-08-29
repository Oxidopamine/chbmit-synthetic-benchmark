"""Predict the per-fold synthetic injection optimum from W2(real ictal, synthetic). CPU-only.

WHY THIS EXISTS. `reports/LITERATURE_AUDIT_2026-08-29.md` Sec 4.2 flags that the grid's dose
coverage has a hole exactly where the parent method operates (r in [0.05, 0.30]), and that
Shidani et al., *Beyond Real Data: Synthetic Data Through The Lens Of Regularization*
(arXiv:2510.08095) Thm 3.1 gives a generalization bound that is U-SHAPED in the synthetic mixing
ratio: too little synthetic leaves variance high, too much lets distributional mismatch dominate,
and for a fixed W2(p_x, p'_x) an optimal mixing parameter exists. Running the full ladder to find
that optimum empirically costs a large grid; measuring W2 and PREDICTING where it sits costs a
couple of CPU-hours off checkpoints already in git.

CORRECTION TO THE AUDIT'S PREMISE. Sec 4.2 proposes reading W2 off "the cached critics", since a
WGAN-GP critic is a W2 estimator. **The critics were never saved** -- `WGANGPProvider._save_state`
persists only the generator (`generator.pt`), so no cached critic exists for any of the 9 cells.
This script therefore estimates W2 *empirically*, by sliced Wasserstein between real ictal windows
and windows drawn from the cached generator. That is a distributional estimate rather than the
critic's dual estimate; it needs no critic, no training, and no GPU, and it is the same quantity
the theorem is stated in terms of.

WHAT IS AND IS NOT PREDICTED. Two outputs, with very different evidential status:

  1. **W2 per (fold, seed)** -- measured. No free parameters.
  2. **r\\* per fold** -- model-dependent. The bound's constants (the Lipschitz-ish factor on the
     W2 term, and the scale of the stability term) are NOT identified by anything measurable here,
     so an absolute r\\* cannot be claimed. The script reports r\\* across a sweep of the one
     ratio that matters, kappa, and states it as conditional.

The claim worth putting in a paper is the one that survives not knowing kappa: **r\\* is monotone
decreasing in W2 at every kappa**, so ranking folds by W2 ranks them by optimal dose in reverse.
That ordering is falsifiable against the mini-grid without estimating a single constant, which is
exactly what makes it worth running before the grid rather than after.

Usage (CPU, on any machine that can see the store -- run it in us-central1 against the bucket
rather than downloading 51 GiB):

    CHBMIT_STORE=/gcs/chbmit-bench-2486a474/processed_chbmit_real/eeg.zarr \\
    CHBMIT_PROC=/gcs/chbmit-bench-2486a474/processed_chbmit_real/processed_index.csv \\
    python3 scripts/gen_w2_dose_prediction.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import zarr

from chbmit.splits import Split
from synthetic.wgan_gp_provider import WGANConfig
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.band_limit import band_limit_windows

STORE = os.environ.get("CHBMIT_STORE", "data/processed_chbmit_real/eeg.zarr")
PROC = os.environ.get("CHBMIT_PROC", "data/processed_chbmit_real/processed_index.csv")
RES = Path(os.environ.get("CHBMIT_RESULTS", "results_chbmit_synthetic/real_validation"))
OUT = RES / "analysis_tierB"
GEN_DIR = RES / "generators"
FRAC = 1.0


def sliced_w2(X: np.ndarray, Y: np.ndarray, n_proj: int = 512, seed: int = 0,
              block: int = 64) -> float:
    """Sliced Wasserstein-2 distance between two window sets, each ``(N, C, T)``.

    Projects both onto random unit directions and averages the 1-D W2, which is exact in 1-D
    (sort, then compare quantiles). Sample sizes need not match: when they differ, both marginals
    are resampled onto a common quantile grid, which is the standard empirical-quantile estimator
    rather than a pairing heuristic. Projections are batched so the (d, n_proj) basis never
    exceeds a few hundred MB.
    """
    Xf = X.reshape(len(X), -1).astype("float64", copy=False)
    Yf = Y.reshape(len(Y), -1).astype("float64", copy=False)
    if Xf.shape[1] != Yf.shape[1]:
        raise ValueError(f"dimension mismatch: real {Xf.shape[1]} vs synthetic {Yf.shape[1]}")
    d = Xf.shape[1]
    rng = np.random.default_rng(seed)
    nq = min(len(Xf), len(Yf))
    q = np.linspace(0.0, 1.0, nq)
    total, done = 0.0, 0
    while done < n_proj:
        k = min(block, n_proj - done)
        V = rng.normal(size=(d, k))
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        a = np.sort(Xf @ V, axis=0)          # (n_x, k)
        b = np.sort(Yf @ V, axis=0)          # (n_y, k)
        if len(a) != nq:
            a = np.quantile(a, q, axis=0)
        if len(b) != nq:
            b = np.quantile(b, q, axis=0)
        total += float(np.sum(np.mean((a - b) ** 2, axis=0)))
        done += k
    return float(np.sqrt(total / n_proj))


def predicted_optimum(w2: float, n_real: int, kappa: float, r_max: float = 1.0) -> float:
    """Optimal injection ratio under a bias-variance form consistent with Thm 3.1's U-shape.

    This is NOT the paper's expression -- its constants are not reproduced here. It is the
    standard decomposition that *produces* the U-shape the theorem describes, so the qualitative
    dependence can be read off without the paper's constants.

    Weight ``lam`` on synthetic, ``1 - lam`` on real. A generator can emit unlimited samples, so
    the synthetic variance term vanishes and what remains is real-sample variance shrinking in
    ``lam`` against synthetic bias growing in it:

        f(lam) = (1 - lam)^2 * V / n_real  +  lam^2 * (xi * w2)^2

    which is genuinely U-shaped, and differentiating gives a closed form:

        lam* = (V / n_real) / (V / n_real + xi^2 w2^2)
        r*   = lam* / (1 - lam*) = kappa / (n_real * w2^2),   kappa := V / xi^2

    So ``kappa`` is the single unidentified constant, and **r\\* is inversely proportional to
    w2^2** -- strictly decreasing in the mismatch, at every ``kappa``. That monotonicity is what
    the fold ordering rests on and it survives not knowing ``kappa``; the absolute value does not.

    An earlier form here used a ``1/sqrt`` stability term. It was wrong: its interior critical
    point is a maximum, so it returned r* = 0 for every input. Kept as a note because the failure
    is silent -- a bound has to be checked for an interior *minimum*, not assumed to have one.
    """
    w2 = float(w2)
    if not np.isfinite(w2) or w2 <= 0 or n_real <= 0:
        return float("nan")
    return float(np.clip(kappa / (n_real * w2 * w2), 0.0, r_max))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 2024])
    ap.add_argument("--device", default="cpu", help="cpu is the point; cuda only if handy")
    ap.add_argument("--n-proj", type=int, default=512, help="random projections for sliced W2")
    ap.add_argument("--n-synth", type=int, default=0,
                    help="synthetic windows to draw (0 = match the real ictal count)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])

    rows = []
    for fold in args.folds:
        f = sp["folds"][fold]
        split = Split(fold=fold, train_groups=f["train_groups"],
                      val_groups=f["val_groups"], test_groups=f["test_groups"])
        for seed in args.seeds:
            gdir = GEN_DIR / f"wgan_f{fold}_s{seed}"
            if not (gdir / "generator.pt").exists():
                print(f"  [skip] no cached generator at {gdir}", flush=True)
                continue

            # Real ictal windows exactly as the pipeline builds them. epochs=0 => no training,
            # this is only a materialisation pass.
            probe = build_provider("wgan_gp", WGANConfig(epochs=0, min_ictal_windows=256,
                                                         device=args.device, seed=seed))
            info = fit_provider_for_cell(probe, index_df, win, ev, STORE, split, fold, seed, FRAC)
            real = np.asarray(info["real_ictal"])
            if real.ndim != 3 or not len(real):
                print(f"  [skip] f{fold} s{seed}: no real ictal windows", flush=True)
                continue

            wgan = build_provider("wgan_gp", WGANConfig(epochs=0, min_ictal_windows=256,
                                                        device=args.device, seed=seed))
            wgan._load_state(gdir)
            wgan.fitted = True
            n_synth = args.n_synth or len(real)
            # Band-limited, matching what the downstream grid actually injects -- W2 against the
            # raw generator output would describe a distribution no experiment ever used.
            synth = band_limit_windows(wgan.generate(n_synth, seed=seed), fs=fs)
            synth = np.asarray(synth)

            w2 = sliced_w2(real, synth, n_proj=args.n_proj, seed=seed)
            # Scale-free companion: the same statistic between two halves of the REAL windows is
            # the noise floor this W2 must be read against.
            h = len(real) // 2
            w2_real_self = (sliced_w2(real[:h], real[h:2 * h], n_proj=args.n_proj, seed=seed)
                            if h >= 8 else float("nan"))
            rows.append({
                "fold": fold, "seed": seed, "n_real": int(len(real)),
                "n_synth": int(len(synth)), "w2_sliced": w2,
                "w2_real_self": w2_real_self,
                "w2_ratio": (w2 / w2_real_self) if w2_real_self == w2_real_self and
                            w2_real_self > 0 else float("nan"),
            })
            print(f"[{time.time()-t0:6.0f}s] f{fold} s{seed}: n_real={len(real)} "
                  f"W2={w2:.4f}  real-vs-real floor={w2_real_self:.4f}  "
                  f"ratio={rows[-1]['w2_ratio']:.2f}", flush=True)

    if not rows:
        print("No cells produced a W2 estimate.")
        return
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"w2_dose_prediction{args.tag}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== W2(real ictal, band-limited synthetic), per fold ===")
    per_fold = df.groupby("fold").agg(w2=("w2_sliced", "mean"), w2_sd=("w2_sliced", "std"),
                                      n_real=("n_real", "median")).reset_index()
    print(per_fold.to_string(index=False))

    # r* = kappa / (n * w2^2) has one unidentified constant, so absolute values mean nothing on
    # their own. Anchoring is more useful than an arbitrary kappa sweep: pin the MEDIAN fold's
    # optimum at each rung of the parent method's own ladder, then report where the other folds
    # land. That answers the operational question ("where do the rungs go, and do folds differ
    # enough to matter?") without pretending the constant is known.
    med_w2 = float(per_fold["w2"].median())
    med_n = float(per_fold["n_real"].median())
    print("\n=== predicted optimal injection ratio r*, anchored on the median fold ===")
    print("    r* = kappa / (n_real * W2^2); kappa is unidentified, so each column fixes it by")
    print("    pinning the median fold at that rung. ABSOLUTE values are conditional;")
    print("    the SPREAD across folds and the ORDERING are not.")
    tab = {"fold": per_fold["fold"].tolist(), "W2": per_fold["w2"].round(4).tolist()}
    for anchor in (0.05, 0.10, 0.20, 0.30):
        k = anchor * med_n * med_w2 * med_w2          # kappa that puts the median fold at anchor
        tab[f"r* | median={anchor:.2f}"] = [
            round(predicted_optimum(w, int(n), k), 3)
            for w, n in zip(per_fold["w2"], per_fold["n_real"])]
    print(pd.DataFrame(tab).to_string(index=False))
    spread = per_fold["w2"].max() / max(per_fold["w2"].min(), 1e-12)
    print(f"\n  W2 spread across folds: {spread:.2f}x  =>  r* spread {spread**2:.2f}x "
          f"(r* goes as 1/W2^2)")
    if spread < 1.15:
        print("  Folds are near-identical in W2. A per-fold dose is not worth it; use one ladder.")
    else:
        print("  Folds differ enough that a single shared rung is a compromise -- worth saying so.")

    order = per_fold.sort_values("w2", ascending=False)["fold"].tolist()
    print(f"\n=== the falsifiable prediction ===")
    print(f"  folds by W2, largest first: {order}")
    print(f"  => predicted optimal dose, SMALLEST first: {order}")
    print("  r* is monotone decreasing in W2 at every kappa, so this ordering holds without")
    print("  estimating any constant. Test it against the mini-grid: the fold with the largest")
    print("  W2 should peak at the lowest ratio. If the ordering inverts, the bound does not")
    print("  describe this setting and the dose story should be reported empirically only.")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
