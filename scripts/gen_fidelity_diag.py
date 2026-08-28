"""Generator fidelity diagnostic: does the cVAE improve with more training?

Tier B showed the trust gate admitted 0/45 synthetic because the cVAE was
low-fidelity (real-vs-synth discriminator AUC = 1.00). This sweeps cVAE training
epochs on a single fold and measures whether fidelity improves at all, so we can
decide if "strengthen the generator" is tractable or a dead end.

Metrics per epoch budget:
  * discriminator_auc  -> 0.5 = indistinguishable (ideal), 1.0 = trivially fake
  * mmd_psd            -> spectral distribution gap (lower better)
  * nn_dist_mean       -> mean nearest-neighbour distance to real (memorisation check)
  * synth_diversity / real_diversity -> mode-collapse ratio (near 1.0 = healthy)
Runs on the local /tmp zarr copy on GPU. Fold 0, scarcity 1.0.
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

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
EPOCH_GRID = [150, 400, 800, 1500]


def _diversity(x: np.ndarray, k: int = 200, seed: int = 0) -> float:
    """Mean pairwise L2 distance among up to k flattened samples."""
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
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    f0 = sp["folds"][0]
    split = Split(fold=0, train_groups=f0["train_groups"],
                  val_groups=f0["val_groups"], test_groups=f0["test_groups"])
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])

    rows = []
    for ep in EPOCH_GRID:
        t0 = time.time()
        prov = build_provider("cvae", CVAEConfig(epochs=ep, min_ictal_windows=256,
                                                 device="cuda", seed=42))
        info = fit_provider_for_cell(prov, index_df, win, ev, STORE, split, 0, 42, 1.0)
        fit_s = time.time() - t0
        real = info["real_ictal"]
        synth = prov.generate(min(len(real), 500), seed=42)
        qm = run_quality_suite(real, synth, fs, "/tmp/qsweep", f"cvae_ep{ep}",
                               make_figures=False)
        rd, sd = _diversity(real), _diversity(synth)
        row = {
            "epochs": ep, "fit_sec": round(fit_s, 1), "n_real": len(real),
            "discriminator_auc": round(qm["discriminator_auc"], 3),
            "mmd_psd": round(qm["mmd_psd"], 3),
            "nn_dist_mean": round(qm["nn_dist_mean"], 2),
            "real_diversity": round(rd, 2), "synth_diversity": round(sd, 2),
            "diversity_ratio": round(sd / rd, 3) if rd else float("nan"),
        }
        rows.append(row)
        print(f"[ep={ep:>4}] {fit_s:5.0f}s  discAUC={row['discriminator_auc']:.3f}  "
              f"mmd={row['mmd_psd']:.3f}  nn={row['nn_dist_mean']:.1f}  "
              f"div_ratio={row['diversity_ratio']:.3f}", flush=True)

    out = RES / "analysis_tierB" / "generator_fidelity_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n=== SUMMARY ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
