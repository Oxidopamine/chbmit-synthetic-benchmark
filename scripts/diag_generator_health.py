"""Diagnose the generative pipeline: is the injected synthetic ictal data actually usable?

Motivated by Phase 1 fold-0 results where every synthetic arm lost to a class-weighted loss.
Before accepting that as a finding we need to rule out defects in what is being injected.
This runs on CPU against a cached WGAN checkpoint -- it trains nothing and changes nothing.

Checks, in order of how badly each would break augmentation:

1. MODE COLLAPSE. Injecting n_train_pos near-identical windows would actively harm training.
   Reports mean pairwise distance between synthetic windows as a ratio of the same statistic
   on real ictal windows. Ratio << 1 means collapse. (The cVAE scored ~0.001 here.)
2. NEAREST-NEIGHBOUR MEMORISATION. The opposite failure: if synthetic windows are copies of
   training windows, augmentation adds nothing. Compares synth->real NN distance against
   real->real NN distance.
3. AMPLITUDE / SCALE. Synthetic must occupy the same normalised space the detector expects
   (per-window per-channel z-score => mean~0, std~1 per channel).
4. SPECTRUM. Band-power by canonical EEG band, real vs synthetic, plus out-of-band leakage
   above 40 Hz (the defect band_limit.py exists to fix).
5. CROSS-CHANNEL STRUCTURE. Real EEG has strong inter-channel correlation from volume
   conduction; white-ish per-channel noise would not.

Usage: python3 scripts/diag_generator_health.py [fold] [seed] [n_synth]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from chbmit.splits import Split
from chbmit.scarcity import select_seizure_events, apply_scarcity_to_windows
from chbmit.datasets import materialize_windows, prefetch_windows
from synthetic.wgan_gp_provider import WGANConfig
from synthetic.train_provider import build_provider
from synthetic.band_limit import band_limit_windows

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
GEN_DIR = RES / "generators"


def pairwise_mean_dist(X, k=300, seed=0):
    """Mean pairwise Euclidean distance on a random subsample of flattened windows."""
    rng = np.random.default_rng(seed)
    F = X.reshape(len(X), -1).astype("float64")
    if len(F) > k:
        F = F[rng.choice(len(F), k, replace=False)]
    g = np.einsum("ij,ij->i", F, F)
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (F @ F.T), 0.0)
    iu = np.triu_indices(len(F), k=1)
    return float(np.sqrt(d2[iu]).mean())


def nn_dist(A, B, k=300, seed=0):
    """Mean nearest-neighbour distance from A into B (excluding self-matches by construction)."""
    rng = np.random.default_rng(seed)
    FA = A.reshape(len(A), -1).astype("float64")
    FB = B.reshape(len(B), -1).astype("float64")
    if len(FA) > k:
        FA = FA[rng.choice(len(FA), k, replace=False)]
    if len(FB) > k:
        FB = FB[rng.choice(len(FB), k, replace=False)]
    ga, gb = np.einsum("ij,ij->i", FA, FA), np.einsum("ij,ij->i", FB, FB)
    d2 = np.maximum(ga[:, None] + gb[None, :] - 2.0 * (FA @ FB.T), 0.0)
    return float(np.sqrt(d2.min(axis=1)).mean())


def bandpower(X, fs=256, k=200, seed=0):
    from scipy.signal import welch
    rng = np.random.default_rng(seed)
    Z = X[rng.choice(len(X), min(k, len(X)), replace=False)]
    f, P = welch(Z, fs=fs, nperseg=min(256, Z.shape[-1]), axis=-1)
    P = P.mean(axis=(0, 1))
    bands = {"delta 0.5-4": (0.5, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13),
             "beta 13-30": (13, 30), "gamma 30-40": (30, 40), "OUT-OF-BAND >40": (40, 128)}
    return {n: float(P[(f >= lo) & (f < hi)].sum()) for n, (lo, hi) in bands.items()}


def xchan_corr(X, k=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = X[rng.choice(len(X), min(k, len(X)), replace=False)]
    cs = []
    for w in Z:
        c = np.corrcoef(w)
        cs.append(np.abs(c[np.triu_indices(len(c), k=1)]).mean())
    return float(np.mean(cs))


def main():
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_synth = int(sys.argv[3]) if len(sys.argv) > 3 else 2508

    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    f = sp["folds"][fold]
    split = Split(fold=fold, train_groups=f["train_groups"], val_groups=f["val_groups"],
                  test_groups=f["test_groups"])

    train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
    tw = win[win["file_id"].isin(train_ids) & (~win["excluded"])]
    sel = select_seizure_events(ev[ev["group"].isin(split.train_groups)], 1.0, seed,
                                restrict_groups=split.train_groups)
    ictal = apply_scarcity_to_windows(tw, sel)
    ictal = ictal[ictal["label"] == 1]
    print(f"fold {fold} seed {seed}: {len(ictal)} real ictal training windows")
    prefetch_windows(ictal, STORE, workers=16, dtype="float16")
    real, _ = materialize_windows(ictal, STORE)
    print(f"real shape {real.shape}")

    gdir = GEN_DIR / f"wgan_f{fold}_s{seed}"
    wgan = build_provider("wgan_gp", WGANConfig(epochs=300, min_ictal_windows=256,
                                                device="cpu", seed=seed))
    wgan._load_state(gdir)
    wgan.fitted = True
    raw = wgan.generate(n_synth, seed=seed)
    bl = band_limit_windows(raw, fs=256)
    print(f"synthetic shape {bl.shape} (from {gdir})\n")

    print("=" * 72)
    print("1. MODE COLLAPSE  (mean pairwise distance; ratio << 1 => collapsed)")
    dr, ds, db = pairwise_mean_dist(real), pairwise_mean_dist(raw), pairwise_mean_dist(bl)
    print(f"   real                {dr:10.2f}")
    print(f"   synthetic raw       {ds:10.2f}   ratio {ds/dr:.3f}")
    print(f"   synthetic band-lim  {db:10.2f}   ratio {db/dr:.3f}")

    print("\n2. MEMORISATION  (NN distance synth->real vs real->real; << 1 => copying)")
    rr, sr = nn_dist(real, real, seed=1), nn_dist(bl, real)
    print(f"   real->real NN {rr:10.2f}   synth->real NN {sr:10.2f}   ratio {sr/rr:.3f}")

    print("\n3. AMPLITUDE / SCALE  (per-window per-channel z-score => mean~0 std~1)")
    for nm, X in (("real", real), ("synth raw", raw), ("synth band-lim", bl)):
        print(f"   {nm:<16} mean {X.mean():+.4f}  std {X.std():.4f}  "
              f"chan-std med {np.median(X.std(axis=-1)):.4f}  |max| {np.abs(X).max():.1f}")

    print("\n4. SPECTRUM  (mean band power; OUT-OF-BAND is the band_limit target)")
    br, bs, bb = bandpower(real), bandpower(raw), bandpower(bl)
    print(f"   {'band':<18}{'real':>12}{'synth raw':>12}{'synth BL':>12}{'BL/real':>10}")
    for k in br:
        r = bb[k] / br[k] if br[k] else float("nan")
        print(f"   {k:<18}{br[k]:>12.4f}{bs[k]:>12.4f}{bb[k]:>12.4f}{r:>10.2f}")

    print("\n5. CROSS-CHANNEL STRUCTURE  (mean |corr| between channels)")
    print(f"   real {xchan_corr(real):.3f}   synth raw {xchan_corr(raw):.3f}   "
          f"synth band-lim {xchan_corr(bl):.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
