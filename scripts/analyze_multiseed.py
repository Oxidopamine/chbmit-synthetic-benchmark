"""Summarise the multi-seed x multi-fold downstream experiment into publishable numbers.

Reads `analysis_tierB/downstream_gated_multiseed.csv` (written incrementally by
run_multiseed_downstream.py) and, PER DETECTOR, answers the paper's questions with paired
statistics across the (fold, seed) cells:

  * Does NAIVE (ungated) synthetic help or harm?  -> paired delta vs real_only + Wilcoxon.
  * Does the GATE deliver on its promise (never worse than real_only)?  -> effective delta.
  * Does the ADMITTED synthetic actually help?  -> augmented-model delta vs real_only,
    and a count of genuine "upside" cells (gate admitted AND effective F1 improved).
  * What does the gate DO?  -> mean admission rate, revert rate.

Works on partial data (only fully-complete (fold,seed,detector) blocks are used), so it can
be run for an interim read while the experiment is still going. Bootstrap 95% CIs; Wilcoxon
signed-rank paired test (falls back gracefully when n is tiny).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
CSV = OUT / "downstream_gated_multiseed.csv"
CONDS_EXPECTED = 4  # real_only, ungated, gated q0.90, gated q0.50


def _boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _wilcoxon(delta):
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def main():
    if not CSV.exists():
        print(f"No results yet at {CSV}"); return
    df = pd.read_csv(CSV)

    # Keep only complete (fold,seed,detector) blocks.
    keep = []
    for key, g in df.groupby(["fold", "seed", "detector"]):
        if len(g) >= CONDS_EXPECTED:
            keep.append(g)
    if not keep:
        print("No complete blocks yet."); return
    df = pd.concat(keep, ignore_index=True)

    def cell(sub, cond, q=None):
        m = sub[(sub["condition"] == cond) & (sub["q"].isna() if q is None else (sub["q"] == q))]
        return m.iloc[0] if len(m) else None

    print(f"Complete (fold,seed,detector) blocks: {len(df)//CONDS_EXPECTED}")
    print(f"Folds: {sorted(df.fold.unique())}  Seeds: {sorted(df.seed.unique())}\n")

    summary_rows = []
    for det in sorted(df.detector.unique()):
        sub = df[df.detector == det]
        pairs = sub.groupby(["fold", "seed"])
        real, ung, g9_eff, g9_aug, g5_eff, g5_aug = [], [], [], [], [], []
        adm9, adm5, rev9, rev5, upside = [], [], [], [], []
        for (fo, se), blk in pairs:
            r = cell(blk, "real_only")
            u = cell(blk, "ungated")
            g9 = cell(blk, "gated", 0.90)
            g5 = cell(blk, "gated", 0.50)
            if r is None:
                continue
            rf = r["event_f1"]
            real.append(rf)
            if u is not None:
                ung.append(u["event_f1"])
            if g9 is not None:
                g9_eff.append(g9["event_f1"]); g9_aug.append(g9["aug_event_f1"])
                adm9.append(g9["gate_admission_rate"]); rev9.append(bool(g9["reverted_to_real_only"]))
                if (not bool(g9["reverted_to_real_only"])) and g9["event_f1"] > rf:
                    upside.append((fo, se, "q0.90", rf, g9["event_f1"]))
            if g5 is not None:
                g5_eff.append(g5["event_f1"]); g5_aug.append(g5["aug_event_f1"])
                adm5.append(g5["gate_admission_rate"]); rev5.append(bool(g5["reverted_to_real_only"]))
                if (not bool(g5["reverted_to_real_only"])) and g5["event_f1"] > rf:
                    upside.append((fo, se, "q0.50", rf, g5["event_f1"]))

        real = np.array(real, dtype=float)
        n = len(real)
        print(f"===== {det.upper()}  (n = {n} fold x seed cells) =====")
        print(f"  real_only event-F1: {real.mean():.3f}  95%CI {_boot_ci(real)}")

        def report(name, vals):
            vals = np.array(vals, dtype=float)
            if len(vals) != n or n == 0:
                print(f"  {name}: (incomplete)"); return None
            delta = vals - real
            lo, hi = _boot_ci(delta)
            p = _wilcoxon(delta)
            helped = int((delta > 0).sum()); hurt = int((delta < 0).sum())
            print(f"  {name}: F1 {vals.mean():.3f}  |  delta vs real {delta.mean():+.3f} "
                  f"[95%CI {lo:+.3f},{hi:+.3f}]  Wilcoxon p={p:.3f}  (helped {helped}/{n}, hurt {hurt}/{n})")
            return {"detector": det, "comparison": name, "n": n, "mean_delta": delta.mean(),
                    "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p, "helped": helped, "hurt": hurt}

        r1 = report("ungated (naive)          ", ung)
        r2 = report("gated q0.90 [effective]  ", g9_eff)
        r3 = report("gated q0.90 [augmented]  ", g9_aug)
        r4 = report("gated q0.50 [effective]  ", g5_eff)
        r5 = report("gated q0.50 [augmented]  ", g5_aug)
        for r in (r1, r2, r3, r4, r5):
            if r:
                summary_rows.append(r)
        if adm9:
            print(f"  gate q0.90: mean admission {np.nanmean(adm9):.3f}, revert rate {np.mean(rev9):.2f}")
        if adm5:
            print(f"  gate q0.50: mean admission {np.nanmean(adm5):.3f}, revert rate {np.mean(rev5):.2f}")
        if upside:
            print(f"  UPSIDE cells (gate admitted AND effective F1 improved): {len(upside)}")
            for u in upside:
                print(f"     fold {u[0]} seed {u[1]} {u[2]}: {u[3]:.3f} -> {u[4]:.3f}")
        else:
            print("  UPSIDE cells: 0 (gate never both admitted and improved)")
        print()

    if summary_rows:
        outp = OUT / "multiseed_summary.csv"
        pd.DataFrame(summary_rows).to_csv(outp, index=False)
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()
