"""WGAN-GP generator fidelity diagnostic (companion to gen_fidelity_diag.py).

The cVAE was confirmed a fidelity dead end: the real-vs-synth discriminator stays
at AUC = 1.00 across every training-epoch budget (posterior collapse worsens with
more training) and every KL-beta (lowering beta restores diversity but not spectral
realism). The report's #1 next step is the WGAN-GP axis. This trains the WGAN-GP
provider's *exact* generator/critic on fold 0 (same z-scored real ictal windows) and
snapshots fidelity at several epoch checkpoints along a SINGLE run -- WGAN-GP costs
~20 s/epoch (gradient-penalty double-backward x n_critic), so re-fitting from scratch
per budget the way the cVAE diag did would take ~16 h. Checkpointing one run gives the
whole trajectory at the cost of the longest single fit.

Answers the live question: can a stronger generator drive the discriminator toward
0.5 (clearing the trust gate's fidelity bar) where the cVAE structurally could not?

Metrics per checkpoint (identical to the cVAE diag):
  * discriminator_auc  -> 0.5 = indistinguishable (ideal), 1.0 = trivially fake
  * mmd_psd            -> spectral distribution gap (lower better)
  * nn_dist_mean       -> mean nearest-neighbour distance to real (memorisation check)
  * synth_diversity / real_diversity -> mode-collapse ratio (near 1.0 = healthy)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import zarr

from chbmit.splits import Split
from synthetic.wgan_gp_provider import (
    WGANConfig, WGANGPProvider, _Generator, _Critic, _fix_length)
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.quality_checks import run_quality_suite

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
CHECKPOINTS = [50, 150, 300, 450, 600]  # cumulative epochs; single run, no refits


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
    device = "cuda"
    cfg = WGANConfig(epochs=CHECKPOINTS[-1], min_ictal_windows=256, device=device, seed=42)

    # Materialise fold-0 real ictal windows exactly as the pipeline does (0-epoch fit).
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    f0 = sp["folds"][0]
    split = Split(fold=0, train_groups=f0["train_groups"],
                  val_groups=f0["val_groups"], test_groups=f0["test_groups"])
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])
    probe = build_provider("wgan_gp", WGANConfig(epochs=0, min_ictal_windows=256,
                                                 device=device, seed=42))
    info = fit_provider_for_cell(probe, index_df, win, ev, STORE, split, 0, 42, 1.0)
    real = info["real_ictal"]
    n_channels, n_samples = real.shape[1], real.shape[2]
    real_div = _diversity(real)
    print(f"n_real={len(real)}  shape={real.shape}  real_diversity={real_div:.2f}", flush=True)

    # Build the provider's exact modules and WGAN-GP optimisers.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    t0 = max(1, n_samples // cfg.base_time_divisor)
    G = _Generator(cfg.latent_dim, n_channels, t0).to(device)
    critic = _Critic(n_channels).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
    optC = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
    prov = WGANGPProvider(cfg)  # thin wrapper for .generate() at each checkpoint
    prov.G, prov.n_channels, prov.n_samples, prov.fitted = G, n_channels, n_samples, True

    data = torch.from_numpy(real).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=cfg.batch_size,
        shuffle=True, drop_last=True)

    def _gp(real_b, fake_b):
        n = real_b.shape[0]
        eps = torch.rand(n, 1, 1, device=device)
        inter = (eps * real_b + (1 - eps) * fake_b).requires_grad_(True)
        d = critic(inter)
        grads = torch.autograd.grad(d, inter, torch.ones_like(d),
                                    create_graph=True, retain_graph=True)[0]
        return ((grads.reshape(n, -1).norm(2, dim=1) - 1) ** 2).mean()

    rows = []
    start = time.time()
    done = 0
    for target in CHECKPOINTS:
        for epoch in range(done, target):
            for (real_b,) in loader:
                real_b = real_b.to(device)
                n = real_b.shape[0]
                for _ in range(cfg.n_critic):
                    z = torch.randn(n, cfg.latent_dim, device=device)
                    fake = _fix_length(G(z), n_samples).detach()
                    optC.zero_grad()
                    loss_c = (critic(fake).mean() - critic(real_b).mean()
                              + cfg.gp_lambda * _gp(real_b, fake))
                    loss_c.backward()
                    optC.step()
                z = torch.randn(n, cfg.latent_dim, device=device)
                fake = _fix_length(G(z), n_samples)
                optG.zero_grad()
                (-critic(fake).mean()).backward()
                optG.step()
        done = target

        G.eval()
        synth = prov.generate(min(len(real), 500), seed=42)
        G.train()
        qm = run_quality_suite(real, synth, fs, "/tmp/qsweep_wgan",
                               f"wgan_ep{target}", make_figures=False)
        sd = _diversity(synth)
        row = {
            "epochs": target, "elapsed_sec": round(time.time() - start, 1),
            "n_real": len(real),
            "discriminator_auc": round(qm["discriminator_auc"], 3),
            "mmd_psd": round(qm["mmd_psd"], 3),
            "nn_dist_mean": round(qm["nn_dist_mean"], 2),
            "real_diversity": round(real_div, 2), "synth_diversity": round(sd, 2),
            "diversity_ratio": round(sd / real_div, 3) if real_div else float("nan"),
        }
        rows.append(row)
        print(f"[ep={target:>4}] {row['elapsed_sec']:6.0f}s  "
              f"discAUC={row['discriminator_auc']:.3f}  mmd={row['mmd_psd']:.3f}  "
              f"nn={row['nn_dist_mean']:.1f}  div_ratio={row['diversity_ratio']:.3f}",
              flush=True)
        out = RES / "analysis_tierB" / "generator_wgan_fidelity_sweep.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)  # incremental save

    print("\n=== SUMMARY (WGAN-GP, single checkpointed run) ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
