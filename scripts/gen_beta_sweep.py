"""Confirm the cVAE collapse is KL-driven (posterior collapse) and test the fix.

The epoch sweep showed persistent mode collapse (diversity ratio ~0.001) that
worsens with training -> classic VAE posterior collapse under beta=1.0. This
sweeps the KL weight (beta) at a fixed epoch budget, no code change required
(CVAEConfig already has a `beta` field). If lowering beta restores synthetic
diversity and drops discriminator AUC below 1.0, the generator is fixable and
"strengthen the generator" is tractable via KL down-weighting / annealing.
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
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.quality_checks import run_quality_suite

STORE = "/tmp/proc_local/processed_chbmit_real/eeg.zarr"
PROC = "/tmp/proc_local/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
BETA_GRID = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0]
EPOCHS = 400


def _diversity(x, k=200, seed=0):
    f = x.reshape(len(x), -1).astype("float64")
    if len(f) > k:
        f = f[np.random.default_rng(seed).choice(len(f), k, replace=False)]
    g = np.einsum("ij,ij->i", f, f)
    d2 = g[:, None] + g[None, :] - 2.0 * (f @ f.T)
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(len(f), k=1)
    return float(np.sqrt(d2[iu]).mean())


def main():
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    f0 = json.load(open(RES / "splits/splits_seed42.json"))["folds"][0]
    split = Split(fold=0, train_groups=f0["train_groups"],
                  val_groups=f0["val_groups"], test_groups=f0["test_groups"])
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])

    rows = []
    for beta in BETA_GRID:
        t0 = time.time()
        prov = build_provider("cvae", CVAEConfig(epochs=EPOCHS, beta=beta,
                                                 min_ictal_windows=256,
                                                 device="cuda", seed=42))
        info = fit_provider_for_cell(prov, index_df, win, ev, STORE, split, 0, 42, 1.0)
        real = info["real_ictal"]
        synth = prov.generate(min(len(real), 500), seed=42)
        qm = run_quality_suite(real, synth, fs, "/tmp/qbeta", f"cvae_b{beta}",
                               make_figures=False)
        rd, sd = _diversity(real), _diversity(synth)
        row = {"beta": beta, "fit_sec": round(time.time() - t0, 1),
               "discriminator_auc": round(qm["discriminator_auc"], 3),
               "mmd_psd": round(qm["mmd_psd"], 3),
               "synth_diversity": round(sd, 2),
               "diversity_ratio": round(sd / rd, 3) if rd else float("nan")}
        rows.append(row)
        print(f"[beta={beta:<6}] discAUC={row['discriminator_auc']:.3f}  "
              f"mmd={row['mmd_psd']:.3f}  synth_div={row['synth_diversity']:.2f}  "
              f"div_ratio={row['diversity_ratio']:.3f}", flush=True)

    out = RES / "analysis_tierB" / "generator_beta_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n=== SUMMARY (real_diversity ~192) ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
