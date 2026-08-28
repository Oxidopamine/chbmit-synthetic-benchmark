"""Summarise the multi-seed x multi-fold downstream experiment into publishable numbers.

Reads ``analysis_tierB/downstream_gated<tag>.csv`` (written incrementally by
run_multiseed_downstream.py) and reports, PER DETECTOR, paired statistics over the
(fold, seed) cells. Four things distinguish this from the original version, all required by
``the verification record``:

(a) **The registered reference.** ``PREREGISTRATION.md`` Sec 3 pairs against "the best simple
    baseline per (fold, seed)" -- the best of {real_only, class_weighted, classical_aug} -- not
    against real_only. Deltas are reported against BOTH so the substitution is visible. The
    reference arm is chosen by event-F1 and then supplies *both* metrics, so the reference is
    always one real training run rather than a per-metric best-of mixture.

(b) **Corrected statistics.** Wilcoxon alone treats 9 cells sharing 3 splits as 9 independent
    observations. Reported alongside it: the Nadeau-Bengio corrected resampled t-test, a
    fold-level t-test on seed-averaged deltas (n = folds, disjoint test groups), and the
    explicit comparison count so Bonferroni is visible rather than implied.

(c) **The safety-benefit frontier** -- one row per deployable policy (always-revert / gated /
    admit-always / ungated / random-gated), with mean delta event-F1, mean and worst delta
    FP/24h, and the pre-registered harm rate.

(d) **The tail analysis** -- gated cells split by the gate's own admit/revert decision, tested
    with a within-fold permutation test and a Fisher exact on the tail event delta FP/24h > +20.
    This is the mechanism finding: the gate is a false-alarm tail controller, not an F1 selector.

Works on partial data (only fully-complete (fold,seed,detector) blocks are used), so it can be
run for an interim read while the experiment is still going.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from evaluation.stats import bootstrap_ci, cvar, worst_delta

RES = Path("results_chbmit_synthetic/real_validation")
OUT = RES / "analysis_tierB"
SPLITS = RES / "splits/splits_seed42.json"

# Rows in a COMPLETE (fold,seed,detector) block for the Phase 1 grid: real_only,
# class_weighted, classical_aug, ungated, gated q0.90, gated q0.50, random_gated q0.90.
# Legacy files (the pre-Phase-1 grid) have 4 -- pass --conds-expected 4 to read those.
CONDS_EXPECTED = 7

# Pre-registered harm definition (PREREGISTRATION.md Sec 3; configs/chbmit_synthetic.yaml).
HARM_DELTA_EVENT_F1 = -0.01
HARM_DELTA_FP24H = 0.25
CVAR_ALPHA = 0.10
TAIL_FP24H = 20.0          # tail event for the Fisher exact test in (d)
SIMPLE_BASELINES = ["real_only", "class_weighted", "classical_aug"]


# --------------------------------------------------------------------------- statistics

def test_train_ratio(splits: dict, folds) -> float:
    """n_test / n_train in patient GROUPS, averaged over the folds actually run."""
    rs = [len(splits["folds"][int(i)]["test_groups"]) / len(splits["folds"][int(i)]["train_groups"])
          for i in folds]
    return float(np.mean(rs)) if rs else float("nan")


def nadeau_bengio(delta, ratio: float):
    """Nadeau-Bengio corrected resampled t-test: variance scaled by (1/n + n_test/n_train).

    The uncorrected paired t-test assumes the cells are independent. They are not -- the seeds
    within a fold reuse the identical split -- so the naive variance understates the true one.
    Returns (t, p); (nan, nan) where undefined (n < 2, zero variance, or a bad ratio).
    """
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2 or not np.isfinite(ratio) or ratio < 0:
        return float("nan"), float("nan")
    var = (1.0 / n + ratio) * float(d.var(ddof=1))
    if not np.isfinite(var) or var <= 0:
        return float("nan"), float("nan")
    from scipy.stats import t as tdist
    t = float(d.mean() / np.sqrt(var))
    return t, float(2 * tdist.sf(abs(t), df=n - 1))


def nadeau_bengio_ci(delta, ratio: float, ci: float = 0.95):
    """The corrected 95% interval that goes with ``nadeau_bengio``.

    Reported instead of the bootstrap CI wherever a significance claim is made: the bootstrap
    resamples the same dependent cells and so inherits their optimism.
    """
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2 or not np.isfinite(ratio) or ratio < 0:
        return float("nan"), float("nan")
    var = (1.0 / n + ratio) * float(d.var(ddof=1))
    if not np.isfinite(var) or var <= 0:
        return float("nan"), float("nan")
    from scipy.stats import t as tdist
    half = float(tdist.ppf(0.5 + ci / 2, df=n - 1) * np.sqrt(var))
    return float(d.mean() - half), float(d.mean() + half)


def fold_level_t(delta, folds):
    """Paired t-test on seed-averaged deltas: one observation per fold, test groups disjoint."""
    d = np.asarray(delta, dtype=float)
    f = np.asarray(folds)
    means = np.array([np.nanmean(d[f == u]) for u in np.unique(f)], dtype=float)
    means = means[~np.isnan(means)]
    if means.size < 2 or np.allclose(means, means[0]):
        return float("nan"), float("nan"), int(means.size)
    from scipy.stats import ttest_1samp
    r = ttest_1samp(means, 0.0)
    return float(r.statistic), float(r.pvalue), int(means.size)


def wilcoxon_p(delta) -> float:
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 3 or np.allclose(d, 0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def prereg_harm(d_f1, d_fp) -> float:
    """Pre-registered harm rate: fraction of cells with dF1 < -0.01 OR dFP/24h > +0.25.

    A NaN on one axis is treated as "no harm on that axis"; a cell that is NaN on both is
    dropped rather than counted as safe.
    """
    f1 = np.asarray(d_f1, dtype=float)
    fp = np.asarray(d_fp, dtype=float)
    keep = ~(np.isnan(f1) & np.isnan(fp))
    if not keep.any():
        return float("nan")
    harmed = ((np.nan_to_num(f1, nan=0.0) < HARM_DELTA_EVENT_F1) |
              (np.nan_to_num(fp, nan=0.0) > HARM_DELTA_FP24H))
    return float(harmed[keep].mean())


def permutation_within_fold(values, groups, folds, n_perm: int = 20000, seed: int = 0):
    """Two-sided permutation test on a difference of group means, shuffling WITHIN each fold.

    ``groups`` is boolean (True = group A). Shuffling inside folds holds fold composition
    fixed, so the test asks whether the split is explained by fold identity alone. Folds with
    no label variation contribute no permutation freedom, which is the honest behaviour.
    Returns (observed_difference, p) with the +1/+1 correction so p is never 0.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=bool)
    f = np.asarray(folds)
    ok = ~np.isnan(v)
    v, g, f = v[ok], g[ok], f[ok]
    if v.size < 3 or g.all() or (~g).all():
        return float("nan"), float("nan")
    obs = float(v[g].mean() - v[~g].mean())
    rng = np.random.default_rng(seed)
    idx_by_fold = [np.where(f == u)[0] for u in np.unique(f)]
    count = 0
    for _ in range(n_perm):
        perm = g.copy()
        for idx in idx_by_fold:
            perm[idx] = rng.permutation(g[idx])
        if perm.all() or (~perm).all():
            continue
        if abs(v[perm].mean() - v[~perm].mean()) >= abs(obs) - 1e-12:
            count += 1
    return obs, float((count + 1) / (n_perm + 1))


def fisher_tail(values, groups, threshold: float = TAIL_FP24H):
    """Fisher exact on the 2x2 of (group) x (value > threshold). Returns ((a,b,c,d), p)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=bool)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    a = int((v[g] > threshold).sum()); b = int(g.sum() - a)
    c = int((v[~g] > threshold).sum()); d = int((~g).sum() - c)
    try:
        from scipy.stats import fisher_exact
        p = float(fisher_exact([[a, b], [c, d]])[1])
    except Exception:
        p = float("nan")
    return (a, b, c, d), p


def mannwhitney_p(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if x.size < 1 or y.size < 1:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- data access

def cell(blk, cond, q=None):
    """The single row for one condition in one (fold,seed,detector) block, or None."""
    if q is None:
        m = blk[(blk["condition"] == cond) & (blk["q"].isna())]
    else:
        m = blk[(blk["condition"] == cond) & np.isclose(blk["q"].fillna(-1.0), q)]
    return m.iloc[0] if len(m) else None


def reference_row(blk, available):
    """The registered reference: the best simple baseline in this block, chosen by event-F1.

    The chosen ARM supplies both event-F1 and FP/24h, so the reference is one real training
    run and not a per-metric best-of mixture (which no deployable system corresponds to).
    """
    rows = [cell(blk, c) for c in available]
    rows = [r for r in rows if r is not None and r["event_f1"] == r["event_f1"]]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["event_f1"]))


# Deployable policies. Each maps a block -> (event_f1, fp_per_24h) actually obtained if that
# policy were deployed. "effective" columns are post-gate (reverted cells carry real_only's
# test metrics); "aug_" columns are always the augmented model, i.e. the gate ignored.
POLICIES = [
    ("always_revert (= real_only)", "real_only", None, "eff"),
    ("gated q0.90",                 "gated",     0.90, "eff"),
    ("gated q0.50",                 "gated",     0.50, "eff"),
    ("random_gated q0.90",          "random_gated", 0.90, "eff"),
    ("admit_always q0.90 pool",     "gated",     0.90, "aug"),
    ("admit_always random pool",    "random_gated", 0.90, "aug"),
    ("ungated (all synthetic)",     "ungated",   None, "eff"),
]


def policy_metrics(blk, cond, q, mode):
    """(event_f1, fp_per_24h) for one policy in one block, or (nan, nan) if absent."""
    r = cell(blk, cond, q)
    if r is None:
        return float("nan"), float("nan")
    if mode == "aug":
        return float(r["aug_event_f1"]), float(r["aug_fp_per_24h"])
    return float(r["event_f1"]), float(r["fp_per_24h"])


# --------------------------------------------------------------------------- reporting

def describe(delta, folds, ratio, name, detector, ref_name, n_comparisons=None):
    """Print and return one paired-comparison row with corrected statistics."""
    d = np.asarray(delta, dtype=float)
    n = int((~np.isnan(d)).sum())
    lo, hi = bootstrap_ci(d)
    p_w = wilcoxon_p(d)
    t_nb, p_nb = nadeau_bengio(d, ratio)
    nb_lo, nb_hi = nadeau_bengio_ci(d, ratio)
    t_fl, p_fl, n_fl = fold_level_t(d, folds)
    helped = int((d > 0).sum()); hurt = int((d < 0).sum())
    print(f"  {name:<32s} d {np.nanmean(d):+.3f} NB95 [{nb_lo:+.3f},{nb_hi:+.3f}]  "
          f"p_wil {p_w:.3f}  p_NB {p_nb:.3f}  p_fold {p_fl:.3f}  (+{helped}/-{hurt} of {n})")
    row = {"detector": detector, "comparison": name, "reference": ref_name, "n": n,
           "mean_delta": float(np.nanmean(d)) if n else float("nan"),
           "ci_lo": lo, "ci_hi": hi, "nb_ci_lo": nb_lo, "nb_ci_hi": nb_hi, "p_wilcoxon": p_w,
           "t_nadeau_bengio": t_nb, "p_nadeau_bengio": p_nb,
           "t_fold_level": t_fl, "p_fold_level": p_fl, "n_folds": n_fl,
           "helped": helped, "hurt": hurt,
           "worst_delta": worst_delta(d, higher_is_better=True),
           "cvar_10pct": cvar(d, CVAR_ALPHA, higher_is_better=True)}
    if n_comparisons:
        row["n_comparisons"] = n_comparisons
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_multiseed", help="reads downstream_gated<tag>.csv")
    ap.add_argument("--csv", default=None, help="explicit CSV path (overrides --tag)")
    ap.add_argument("--conds-expected", type=int, default=CONDS_EXPECTED,
                    help="rows in a complete (fold,seed,detector) block; 4 for legacy files")
    ap.add_argument("--n-perm", type=int, default=20000)
    args = ap.parse_args()

    csv = Path(args.csv) if args.csv else OUT / f"downstream_gated{args.tag}.csv"
    if not csv.exists():
        print(f"No results at {csv}")
        return
    df = pd.read_csv(csv)
    splits = json.load(open(SPLITS)) if SPLITS.exists() else {"folds": []}

    sizes = df.groupby(["fold", "seed", "detector"]).size()
    keep = [g for _, g in df.groupby(["fold", "seed", "detector"])
            if len(g) >= args.conds_expected]
    print(f"=== {csv.name} ===")
    print(f"block sizes present: {dict(sizes.value_counts().sort_index())}  "
          f"(complete = {args.conds_expected} rows)")
    if not keep:
        print("No complete blocks. If this is a legacy 4-condition file, pass --conds-expected 4.")
        return
    df = pd.concat(keep, ignore_index=True)

    present = [c for c in SIMPLE_BASELINES if c in set(df["condition"])]
    missing = [c for c in SIMPLE_BASELINES if c not in present]
    folds_run = sorted(df["fold"].unique())
    ratio = test_train_ratio(splits, folds_run) if splits["folds"] else float("nan")

    print(f"complete blocks: {len(df) // args.conds_expected}   "
          f"folds {folds_run}  seeds {sorted(df['seed'].unique())}  "
          f"detectors {sorted(df['detector'].unique())}")
    # When only real_only is present the two references coincide; reporting both would double
    # the comparison count and make the Bonferroni threshold look twice as strict as it is.
    degraded = present == ["real_only"]
    print(f"registered reference = best of {present} per (fold, seed)")
    if missing:
        print(f"  !! MISSING registered baselines: {missing}. The 'registered' reference below "
              f"degrades to the best of what is present -- deltas are NOT the pre-registered "
              f"comparison (PREREGISTRATION.md Sec 3).")
    print(f"Nadeau-Bengio n_test/n_train over folds {folds_run}: {ratio:.4f}\n")

    summary, frontier, tail_rows = [], [], []

    for det in sorted(df["detector"].unique()):
        sub = df[df["detector"] == det]
        blocks = [(fo, se, blk) for (fo, se), blk in sub.groupby(["fold", "seed"])]
        n_cells = len(blocks)
        print(f"===== {det.upper()}  (n = {n_cells} fold x seed cells) =====")

        folds_v = np.array([fo for fo, _, _ in blocks])
        ref_reg = [reference_row(blk, present) for _, _, blk in blocks]
        ref_real = [cell(blk, "real_only") for _, _, blk in blocks]

        def ref_vals(rows, col):
            return np.array([float(r[col]) if r is not None else np.nan for r in rows])

        reg_f1, reg_fp = ref_vals(ref_reg, "event_f1"), ref_vals(ref_reg, "fp_per_24h")
        real_f1, real_fp = ref_vals(ref_real, "event_f1"), ref_vals(ref_real, "fp_per_24h")
        refs = ([("real_only", real_f1, real_fp)] if degraded else
                [("registered", reg_f1, reg_fp), ("real_only", real_f1, real_fp)])
        chosen = pd.Series([r["condition"] if r is not None else "?" for r in ref_reg])
        print(f"  real_only event-F1 {np.nanmean(real_f1):.3f} | "
              f"registered-reference event-F1 {np.nanmean(reg_f1):.3f} "
              f"(reference arm chosen: {dict(chosen.value_counts())})")

        # ---- (a)+(b) paired deltas against BOTH references, corrected statistics ----
        for label, cond, q, mode in POLICIES[1:]:      # skip always_revert (it IS real_only)
            vals = np.array([policy_metrics(blk, cond, q, mode) for _, _, blk in blocks])
            if np.isnan(vals[:, 0]).all():
                continue
            f1, fp = vals[:, 0], vals[:, 1]
            for ref_name, rf1, rfp in refs:
                row = describe(f1 - rf1, folds_v, ratio, f"{label} vs {ref_name}", det, ref_name)
                row["mean_delta_fp24h"] = float(np.nanmean(fp - rfp))
                row["worst_delta_fp24h"] = worst_delta(fp - rfp, higher_is_better=False)
                row["prereg_harm_rate"] = prereg_harm(f1 - rf1, fp - rfp)
                summary.append(row)

        # ---- (c) safety-benefit frontier ----
        ref_label = "real_only" if degraded else "registered reference"
        print(f"\n  --- safety-benefit frontier ({det}, vs {ref_label}) ---")
        print(f"  {'policy':<28s} {'dF1':>7s} {'dFP/24h':>9s} {'worst dFP':>10s} {'harm':>6s}")
        for label, cond, q, mode in POLICIES:
            vals = np.array([policy_metrics(blk, cond, q, mode) for _, _, blk in blocks])
            if np.isnan(vals[:, 0]).all():
                continue
            d_f1, d_fp = vals[:, 0] - reg_f1, vals[:, 1] - reg_fp
            d_f1r, d_fpr = vals[:, 0] - real_f1, vals[:, 1] - real_fp
            hr = prereg_harm(d_f1, d_fp)
            print(f"  {label:<28s} {np.nanmean(d_f1):+7.3f} {np.nanmean(d_fp):+9.2f} "
                  f"{worst_delta(d_fp, higher_is_better=False):+10.2f} {hr:6.2f}")
            frontier.append({
                "detector": det, "policy": label, "n": int((~np.isnan(d_f1)).sum()),
                "mean_delta_event_f1": float(np.nanmean(d_f1)),
                "mean_delta_fp24h": float(np.nanmean(d_fp)),
                "worst_delta_fp24h": worst_delta(d_fp, higher_is_better=False),
                "cvar10_delta_event_f1": cvar(d_f1, CVAR_ALPHA, higher_is_better=True),
                "prereg_harm_rate": hr,
                "mean_delta_event_f1_vs_real_only": float(np.nanmean(d_f1r)),
                "mean_delta_fp24h_vs_real_only": float(np.nanmean(d_fpr)),
                "prereg_harm_rate_vs_real_only": prereg_harm(d_f1r, d_fpr),
            })

        # ---- rows for (d): every gated-family cell, tagged with the gate's own decision ----
        for cond, q in [("gated", 0.90), ("gated", 0.50), ("random_gated", 0.90)]:
            for i, (fo, se, blk) in enumerate(blocks):
                r = cell(blk, cond, q)
                if r is None:
                    continue
                tail_rows.append({
                    "detector": det, "fold": fo, "seed": se, "arm": f"{cond} q{q}",
                    "reverted": bool(r["reverted_to_real_only"]),
                    "reason": r.get("gate_reason"),
                    "n_admitted": r.get("gate_n_admitted"),
                    "delta_fp24h_aug": float(r["aug_fp_per_24h"]) - real_fp[i],
                    "delta_f1_aug": float(r["aug_event_f1"]) - real_f1[i],
                    "delta_fp24h_aug_vs_reg": float(r["aug_fp_per_24h"]) - reg_fp[i],
                    "delta_f1_aug_vs_reg": float(r["aug_event_f1"]) - reg_f1[i],
                })
        print()

    # ------------------------------------------------------------------ (d) tail analysis
    tail = pd.DataFrame(tail_rows)
    if len(tail):
        print("=== (d) what the gate controls: admitted vs reverted cells ===")
        print("    (all gated-family cells pooled; delta = AUGMENTED model - real_only)")
        adm = tail[~tail["reverted"]]
        rev = tail[tail["reverted"]]
        for nm, grp in (("admitted", adm), ("reverted", rev)):
            if not len(grp):
                continue
            print(f"  {nm:<9s} n={len(grp):3d}  mean dFP/24h {grp['delta_fp24h_aug'].mean():+8.2f}  "
                  f"median {grp['delta_fp24h_aug'].median():+8.2f}  "
                  f"worst {grp['delta_fp24h_aug'].max():+8.2f}  "
                  f"mean dF1 {grp['delta_f1_aug'].mean():+.3f}")
        if len(adm) and len(rev):
            g = (~tail["reverted"]).to_numpy()
            obs, p_perm = permutation_within_fold(tail["delta_fp24h_aug"].to_numpy(), g,
                                                  tail["fold"].to_numpy(), n_perm=args.n_perm)
            (a, b, c, d), p_f = fisher_tail(tail["delta_fp24h_aug"].to_numpy(), g)
            print("  all p-values below are TWO-SIDED (halve them for the one-sided form; "
                  "the verification record Sec 3.3 quotes the one-sided values)")
            print(f"  dFP/24h  Mann-Whitney p = "
                  f"{mannwhitney_p(adm['delta_fp24h_aug'], rev['delta_fp24h_aug']):.4f}   "
                  f"within-fold permutation p = {p_perm:.4f} (obs diff {obs:+.2f})")
            print(f"  tail dFP/24h > +{TAIL_FP24H:.0f}: admitted {a}/{a + b}, "
                  f"reverted {c}/{c + d}  Fisher p = {p_f:.4f}")
            print(f"  dF1      Mann-Whitney p = "
                  f"{mannwhitney_p(adm['delta_f1_aug'], rev['delta_f1_aug']):.4f}  "
                  f"(the axis the gate does NOT reliably select on)")
        if "reason" in tail:
            print("\n  by the gate's stated reason:")
            for reason, grp in tail.groupby("reason"):
                print(f"    {str(reason):<32s} n={len(grp):3d}  "
                      f"mean dFP/24h {grp['delta_fp24h_aug'].mean():+8.2f}  "
                      f"max {grp['delta_fp24h_aug'].max():+8.2f}")

    # --------------------------------------------------------------------------- outputs
    # Outputs are named after, and written beside, the CSV that produced them -- so pointing
    # --csv at a scratch file cannot overwrite the real analysis under the default tag.
    outdir = csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    tag = (csv.stem[len("downstream_gated"):] if csv.stem.startswith("downstream_gated")
           else "_" + csv.stem)
    stem = f"analysis{tag}"
    written = []
    if summary:
        s = pd.DataFrame(summary)
        s["n_comparisons"] = len(s)
        s["bonferroni_p_wilcoxon"] = np.minimum(1.0, s["p_wilcoxon"] * len(s))
        s.to_csv(outdir / f"{stem}_summary.csv", index=False)
        written.append(outdir / f"{stem}_summary.csv")
        print(f"\n=== multiple comparisons: {len(s)} paired tests reported ===")
        print("  Bonferroni threshold for alpha=0.05: p < "
              f"{0.05 / len(s):.4f}; comparisons meeting it: "
              f"{int((s['p_wilcoxon'] < 0.05 / len(s)).sum())} on Wilcoxon, "
              f"{int((s['p_nadeau_bengio'] < 0.05 / len(s)).sum())} on Nadeau-Bengio")
        surv = s[(s["p_nadeau_bengio"] < 0.05)]
        print(f"  comparisons with uncorrected-for-multiplicity p_NB < 0.05: {len(surv)}"
              + ("" if not len(surv) else
                 "\n" + surv[["detector", "comparison", "mean_delta",
                              "p_nadeau_bengio"]].to_string(index=False)))
    for name, obj in (("frontier", frontier), ("gate_tail", tail_rows)):
        if obj:
            pd.DataFrame(obj).to_csv(outdir / f"{stem}_{name}.csv", index=False)
            written.append(outdir / f"{stem}_{name}.csv")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
