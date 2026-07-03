"""Re-test trust-gate admission with the two fixes: band-limited output + WGAN generator.

Tier B's gate reverted 100% because STAGE-1 admission was 0.0 -- the real-only teacher
detector scored every synthetic window below the q=0.90 quantile of its confidence on
real ictal (i.e. it did not recognise the synthetic windows as seizures). The gate never
used discriminator-AUC; that was only a reported quality diagnostic. So the real lever is
generator-side. This isolates stage-1 admission (the bottleneck) and asks: does a
spectrally-realistic WGAN-GP, with its out-of-band junk removed by band-limiting to the
real 0.5-40 Hz passband, produce windows the teacher recognises well enough to admit?

Fold 0, scarcity 1.0. Teacher = real-only EEGNet trained exactly as in Tier B. Compares
teacher-score distributions + admission rate for: cVAE(beta=0.01, best-diversity),
WGAN-GP raw, WGAN-GP band-limited. WGAN generator is loaded from the persisted checkpoint.
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

import torch

from concurrent.futures import ThreadPoolExecutor

from chbmit.splits import Split
from chbmit.scarcity import select_seizure_events, apply_scarcity_to_windows
from chbmit.datasets import negative_sample, read_window, normalize_window
from models import build_model
from synthetic.cvae_provider import CVAEConfig
from synthetic.wgan_gp_provider import WGANConfig
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.band_limit import band_limit_windows
from synthetic.trust_gate import TrustGateConfig, run_admission, score_windows

STORE = "data/processed_chbmit_real/eeg.zarr"
PROC = "data/processed_chbmit_real/processed_index.csv"
RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
DEVICE = "cuda"


def parallel_materialize(table, store, workers=48,
                         normalize="per_window_channel_zscore", eps=1e-6):
    """Thread-parallel version of chbmit.datasets.materialize_windows.

    The processed store is on a network filesystem (~65 ms/window latency), so serial
    reads of ~15k windows take ~14 min. Reads are latency-bound and release the GIL, so
    a thread pool hides the latency (order-of-magnitude speedup). Returns (X, y) in the
    ORIGINAL table order.
    """
    rows = list(table.itertuples(index=False))

    def _load(r):
        x = read_window(store, r.file_id, int(r.start_sample), int(r.end_sample))
        return normalize_window(x, normalize, eps), int(r.label)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_load, rows))
    X = np.stack([a for a, _ in out]).astype("float32")
    y = np.asarray([b for _, b in out], dtype="int64")
    return X, y


def _summ(name, scores):
    q = np.quantile(scores, [0.5, 0.9, 0.99])
    return (f"{name:22s} n={len(scores):5d}  mean={scores.mean():.3f}  "
            f"med={q[0]:.3f}  p90={q[1]:.3f}  p99={q[2]:.3f}  max={scores.max():.3f}")


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

    # 1. Real-only EEGNet teacher (fold 0, scarcity 1.0). Trained in-memory on the
    #    real_only training set only -- the teacher's role in the gate is purely to SCORE
    #    windows, so we skip the (huge, zarr-bound) val/test event evaluation that run_cell
    #    does. Recipe matches the real_only condition: plain BCE, Adam lr 3e-4, wd 1e-4.
    seed = 42
    train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
    train_windows = win[win["file_id"].isin(train_ids) & (~win["excluded"])]
    selected = select_seizure_events(
        ev[ev["group"].isin(split.train_groups)], 1.0, seed, restrict_groups=split.train_groups)
    scarce = apply_scarcity_to_windows(train_windows, selected)
    train_table = negative_sample(scarce, ratio=5.0, exclude_seconds=60.0, seed=seed)
    Xtr, ytr = parallel_materialize(train_table, STORE)
    print(f"[{time.time()-t0:5.0f}s] teacher train set: {Xtr.shape}, pos={int(ytr.sum())}", flush=True)

    torch.manual_seed(seed); np.random.seed(seed)
    n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
    teacher = build_model("eegnet", n_channels=n_ch, n_samples=n_t).to(DEVICE)
    opt = torch.optim.Adam(teacher.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xg = torch.from_numpy(Xtr).to(DEVICE)
    yg = torch.from_numpy(ytr.astype("float32")).to(DEVICE)
    bs, epochs = 64, 60
    teacher.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xg), device=DEVICE)
        tot = 0.0
        for i in range(0, len(Xg), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(teacher(Xg[idx]), yg[idx])
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"[{time.time()-t0:5.0f}s]   teacher ep{ep:>3} bce={tot/len(Xg):.4f}", flush=True)
    del Xg, yg
    # Sanity: teacher must separate real ictal from background on its own train set.
    tr_sc = score_windows(teacher, Xtr, DEVICE)
    print(f"[{time.time()-t0:5.0f}s] teacher trained; train ictal mean score="
          f"{tr_sc[ytr == 1].mean():.3f} vs bg {tr_sc[ytr == 0].mean():.3f}", flush=True)

    # 2. Real fold-0 training ictal windows = the admission reference (same set the gate
    #    calibrates the q-quantile on). Reuse the positives already materialised above.
    real = Xtr[ytr == 1]
    real_scores = tr_sc[ytr == 1]
    n_synth = len(real)                      # synthetic_ratio 1.0
    pool_n = TrustGateConfig().oversample * n_synth

    # 3. Candidate generators.
    wgan = build_provider("wgan_gp", WGANConfig(min_ictal_windows=256, device=DEVICE, seed=42))
    wgan._load_state(RES / "generators" / "wgan_fold0")  # restores G, n_channels, n_samples
    wgan.fitted = True                                    # _load_state doesn't set this
    wgan_raw = wgan.generate(pool_n, seed=42)
    wgan_bl = band_limit_windows(wgan_raw, fs=fs)
    print(f"[{time.time()-t0:5.0f}s] wgan pools ready ({len(wgan_raw)})", flush=True)

    cvae = build_provider("cvae", CVAEConfig(epochs=400, beta=0.01,
                                             min_ictal_windows=256, device=DEVICE, seed=42))
    fit_provider_for_cell(cvae, index_df, win, ev, STORE, split, 0, 42, 1.0)
    cvae_raw = cvae.generate(pool_n, seed=42)
    print(f"[{time.time()-t0:5.0f}s] cvae pool ready", flush=True)

    pools = {
        "cvae_b0.01": cvae_raw,
        "wgan_raw": wgan_raw,
        "wgan_bandlimited": wgan_bl,
    }
    gcfg = TrustGateConfig()  # q=0.90, oversample=6 (matches Tier B)
    print("\n=== teacher seizure-confidence distributions ===", flush=True)
    print(_summ("REAL ictal (ref)", real_scores), flush=True)

    # Score each pool once; then admission is just fraction >= a real-quantile threshold.
    scores = {name: score_windows(teacher, pool, DEVICE) for name, pool in pools.items()}
    for name in pools:
        print(_summ(name, scores[name]), flush=True)

    # Headline table at the pre-registered q=0.90.
    rows = []
    for name, pool in pools.items():
        adm = run_admission(teacher, pool, real, gcfg, target_count=n_synth, device=DEVICE)
        rows.append({
            "generator": name, "n_pool": adm.n_pool,
            "admission_threshold_q90_real": round(adm.threshold, 4),
            "n_admitted": adm.n_admitted, "admission_rate": round(adm.admission_rate, 4),
            "synth_score_mean": round(float(scores[name].mean()), 4),
            "synth_score_p99": round(float(np.quantile(scores[name], 0.99)), 4),
        })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "gate_admission_retest.csv", index=False)

    # Sweep the admission quantile q: how conservative is the gate, and what q admits
    # the band-limited WGAN? Admission propensity = fraction of pool >= the q-quantile of
    # real ictal scores (uncapped, i.e. before the target-count cap).
    Q_GRID = [0.5, 0.75, 0.90, 0.95, 0.99]
    sweep = []
    for q in Q_GRID:
        thr = float(np.quantile(real_scores, q))
        row = {"q": q, "threshold": round(thr, 4)}
        for name in pools:
            row[name] = round(float((scores[name] >= thr).mean()), 4)
        sweep.append(row)
    sdf = pd.DataFrame(sweep)
    sdf.to_csv(OUT / "gate_admission_qsweep.csv", index=False)

    meta = {"teacher_train_ictal_score_mean": round(float(real_scores.mean()), 4),
            "teacher_train_bg_score_mean": round(float(tr_sc[ytr == 0].mean()), 4),
            "q_preregistered": gcfg.q, "oversample": gcfg.oversample,
            "admission_threshold_q90": round(float(np.quantile(real_scores, gcfg.q)), 4)}
    (OUT / "gate_admission_retest.json").write_text(json.dumps(meta, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = {"cvae_b0.01": "tab:orange", "wgan_raw": "tab:green",
                  "wgan_bandlimited": "tab:blue"}
        for name in pools:
            ax[0].hist(scores[name], bins=60, range=(0, 1), density=True, alpha=0.5,
                       label=name, color=colors[name])
        ax[0].hist(real_scores, bins=60, range=(0, 1), density=True, histtype="step",
                   color="k", lw=2, label="REAL ictal")
        ax[0].axvline(meta["admission_threshold_q90"], color="red", ls="--",
                      label=f"q0.90 thr={meta['admission_threshold_q90']:.3f}")
        ax[0].set_title("Teacher seizure-confidence (fold 0)")
        ax[0].set_xlabel("teacher P(seizure)"); ax[0].legend(fontsize=7)
        for name in pools:
            ax[1].plot(Q_GRID, [r[name] for r in sweep], "o-", label=name, color=colors[name])
        ax[1].axvline(0.90, color="red", ls="--", alpha=0.6, label="pre-registered q")
        ax[1].set_title("Admission propensity vs admission quantile q")
        ax[1].set_xlabel("q (quantile of real ictal scores)")
        ax[1].set_ylabel("fraction of pool admitted"); ax[1].legend(fontsize=7)
        fig.tight_layout()
        (OUT / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "figures" / "gate_admission.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"figure error: {e}", flush=True)

    print("\n=== ADMISSION RE-TEST SUMMARY (fold 0, eegnet teacher, q=0.90) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print("\n=== ADMISSION-vs-q SWEEP (fraction of pool admitted) ===", flush=True)
    print(sdf.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"\n[{time.time()-t0:5.0f}s] wrote {OUT/'gate_admission_retest.csv'}, "
          f"{OUT/'gate_admission_qsweep.csv'}, figures/gate_admission.png", flush=True)


if __name__ == "__main__":
    main()
