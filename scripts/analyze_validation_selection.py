"""Phase 3, free part: everything the committed Phase 2 CSVs can answer without a GPU.

Four analyses, all TCN-only because only the Phase 2 grids (``_p2``, ``_p3``) carry the
validation columns (``val_event_f1``, ``val_fp_per_24h``, ``val_auprc``) for every arm. The
Phase 1 CSV (``_v2``) does not, so EEGNet and LCT wait for the seeded baseline re-run.

(1) **The registered reference, chosen on validation.** ``PREREGISTRATION.md`` Sec 3 pairs against
    the best simple baseline per (fold, seed). Every "vs registered" number published so far chose
    that arm on TEST event-F1 (``analyze_multiseed.py``), which DECISION_GATE_2.md measured as a
    +0.035 inflation. Here the arm is chosen on validation event-F1 and scored on test, and both
    versions are reported side by side.

(2) **The validation-selection competitor.** Does the trust gate do anything that plain model
    selection on validation does not? Policies that need no admission machinery:
      * ``valsel4``            -- argmax validation event-F1 over {real_only, class_weighted,
                                  classical_aug, ungated}; deploy that arm.
      * ``valsel4_fpguard``    -- same, restricted to arms whose validation FP/24h is within the
                                  gate's own safety slack (+0.25) of real_only.
      * ``ungated_failclosed`` -- the gate's fail-closed rule applied to the UNGATED arm: deploy it
                                  iff it beats real_only on validation event-F1 and stays inside the
                                  FP slack, else revert to real_only. This is "the gate minus its
                                  admission stage".
      * ``gated -> valsel3``   -- the gate's own decision, but reverting to the validation-selected
                                  best simple baseline instead of to real_only. The earlier
                                  "-> best-baseline" frontier row chose that fallback on TEST and is
                                  therefore not quotable; this one is.
    Each is compared cell-by-cell with the gate as deployed, with Wilcoxon, Nadeau-Bengio and
    fold-level statistics, and with the tail metrics the gate is credited with controlling.

(3) **Harm rate as a curve against the margin.** The registered harm thresholds (dF1 < -0.01,
    dFP/24h > +0.25) are far below every noise floor this project has measured. Rather than a point
    harm rate, this sweeps the margin and reports harm rate as a function of it, alongside a NULL
    curve built from the 27 same-specification re-run pairs (TCN baselines in ``_v2`` vs ``_p2``,
    identical except for weight-initialisation seeding). A harm rate is only readable where it
    separates from the null curve. NOTE: that null is the DIFFERENT-INITIALISATION floor. The
    floor that applies to a same-seed paired delta (same init, different synthetic draw) has not
    been measured and is the first GPU item in PREREGISTRATION_PHASE3.md.

(4) **Power design for Phase 3.** Nadeau-Bengio standard error and minimum detectable effect at
    80% power for candidate fold x seed designs over the 23 CHB-MIT groups. rho = n_test/n_train
    is computed from the split geometry, so the table shows why folds, not seeds, are the lever.

Everything is read from committed CSVs. No GPU, no data store, no torch.

    python scripts/analyze_validation_selection.py            # writes analysis_tierB/phase3_free/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from evaluation.stats import cvar, worst_delta
from scripts.analyze_multiseed import (
    fold_level_t, nadeau_bengio, nadeau_bengio_ci, wilcoxon_p, test_train_ratio,
    HARM_DELTA_EVENT_F1, HARM_DELTA_FP24H, TAIL_FP24H, CVAR_ALPHA,
)

RES = Path("results_chbmit_synthetic/real_validation")
A = RES / "analysis_tierB"
SPLITS = RES / "splits/splits_seed42.json"
OUT_DEFAULT = A / "phase3_free"

SIMPLE = ["real_only", "class_weighted", "classical_aug"]
FP_SLACK = 0.25          # TrustGateConfig.fp24h_safety_slack, mirrored (torch-free script)
F1_MARGIN = 0.0          # TrustGateConfig.admit_margin_event_f1
N_GROUPS = 23            # CHB-MIT patient groups after the chb01/chb21 merge
VAL_FRACTION = 0.20      # configs/chbmit_synthetic.yaml splitting.validation_fraction_within_train

GRIDS = [
    # (label, csv, ratio rung, core gated q for the harm curve)
    ("p2_r010", "downstream_gated_p2.csv", 0.10, 0.90),
    ("p2_r030", "downstream_gated_p2.csv", 0.30, 0.90),
    ("p3_r100", "downstream_gated_p3.csv", 1.00, 0.95),
]


# --------------------------------------------------------------------------- cell access

def _one(blk: pd.DataFrame, cond: str, q=None, ratio=None):
    m = blk[blk["condition"] == cond]
    m = m[m["q"].isna()] if q is None else m[np.isclose(m["q"].fillna(-1.0), q)]
    if ratio is not None and "ratio" in m.columns and m["ratio"].notna().any():
        m = m[np.isclose(m["ratio"], ratio)]
    if len(m) > 1:
        raise ValueError(f"{len(m)} rows for condition={cond} q={q} ratio={ratio} in one block")
    return m.iloc[0] if len(m) else None


def _tm(row):
    """(test event-F1, test FP/24h) as deployed."""
    return (float(row["event_f1"]), float(row["fp_per_24h"])) if row is not None else (np.nan, np.nan)


def _vm(row):
    """(validation event-F1, validation FP/24h) of the model whose test metrics the row carries."""
    return (float(row["val_event_f1"]), float(row["val_fp_per_24h"])) if row is not None else (np.nan, np.nan)


def _passes_failclosed(cand_val, ref_val):
    """The gate's stage-2 rule: non-inferior validation event-F1 AND FP/24h inside the slack."""
    (cf1, cfp), (rf1, rfp) = cand_val, ref_val
    if not (np.isfinite(cf1) and np.isfinite(rf1) and np.isfinite(cfp) and np.isfinite(rfp)):
        return False
    return (cf1 >= rf1 + F1_MARGIN) and (cfp <= rfp + FP_SLACK)


def _argmax_val(cands):
    """cands: list of (name, row). Highest validation event-F1 wins; ties go to the earliest
    entry (callers put real_only first, so a tie never promotes a synthetic arm)."""
    best, best_v = None, -np.inf
    for name, row in cands:
        if row is None:
            continue
        v = float(row["val_event_f1"])
        if np.isfinite(v) and v > best_v:
            best, best_v = (name, row), v
    return best


def cell_policies(blk: pd.DataFrame, ratio: float, qs):
    """Every deployable policy's (test F1, test FP) in one (fold, seed) block, plus bookkeeping."""
    ro = _one(blk, "real_only")
    cw = _one(blk, "class_weighted")
    ca = _one(blk, "classical_aug")
    ug = _one(blk, "ungated", ratio=ratio)
    simple = [("real_only", ro), ("class_weighted", cw), ("classical_aug", ca)]

    out, chosen = {}, {}
    out["real_only"] = _tm(ro)
    out["class_weighted"] = _tm(cw)
    out["classical_aug"] = _tm(ca)
    out["ungated"] = _tm(ug)

    # Registered reference, two ways.
    by_test = max([s for s in simple if s[1] is not None], key=lambda s: float(s[1]["event_f1"]))
    by_val = _argmax_val(simple)
    out["best_simple_TESTselected"] = _tm(by_test[1])
    out["best_simple_valselected"] = _tm(by_val[1])
    chosen["ref_test"] = by_test[0]
    chosen["ref_val"] = by_val[0]

    # Validation-selection competitors.
    v4 = _argmax_val(simple + [("ungated", ug)])
    out["valsel4"] = _tm(v4[1])
    chosen["valsel4"] = v4[0]
    guarded = [(n, r) for n, r in simple + [("ungated", ug)]
               if r is not None and float(r["val_fp_per_24h"]) <= float(ro["val_fp_per_24h"]) + FP_SLACK]
    v4g = _argmax_val(guarded) or ("real_only", ro)
    out["valsel4_fpguard"] = _tm(v4g[1])
    chosen["valsel4_fpguard"] = v4g[0]
    ufc = ug if (ug is not None and _passes_failclosed(_vm(ug), _vm(ro))) else ro
    out["ungated_failclosed"] = _tm(ufc)
    chosen["ungated_failclosed"] = "ungated" if ufc is ug else "real_only"

    for q in qs:
        g = _one(blk, "gated", q=q, ratio=ratio)
        r = _one(blk, "random_gated", q=q, ratio=ratio)
        out[f"gated q{q:g}"] = _tm(g)
        out[f"random_gated q{q:g}"] = _tm(r)
        if g is not None:
            out[f"admit_always q{q:g}"] = (float(g["aug_event_f1"]), float(g["aug_fp_per_24h"]))
            reverted = bool(g["reverted_to_real_only"])
            out[f"gated q{q:g} -> valsel3"] = _tm(by_val[1]) if reverted else (
                float(g["aug_event_f1"]), float(g["aug_fp_per_24h"]))
            # valsel5: the gate's deployed arm competes on its own validation numbers.
            v5 = _argmax_val(simple + [("ungated", ug), (f"gated q{q:g}", g)])
            out[f"valsel5 q{q:g}"] = _tm(v5[1])
            chosen[f"valsel5 q{q:g}"] = v5[0]
    return out, chosen


# --------------------------------------------------------------------------- statistics

def margin_grids():
    """Margin sweeps for the harm curves. The registered thresholds (0.01, 0.25) are included
    explicitly so the curve can be read off at the point the pre-registration fixed."""
    f1 = np.unique(np.round(np.concatenate([np.linspace(0.0, 0.30, 61), [abs(HARM_DELTA_EVENT_F1)]]), 4))
    fp = np.unique(np.round(np.concatenate([np.linspace(0.0, 80.0, 81), [HARM_DELTA_FP24H]]), 2))
    return f1, fp


def paired_row(d_f1, d_fp, folds, rho, label, ref):
    d = np.asarray(d_f1, float); dp = np.asarray(d_fp, float)
    n = int(np.isfinite(d).sum())
    t_nb, p_nb = nadeau_bengio(d, rho)
    lo, hi = nadeau_bengio_ci(d, rho)
    _, p_fl, n_fl = fold_level_t(d, folds)
    return {
        "comparison": label, "reference": ref, "n": n,
        "mean_delta_event_f1": float(np.nanmean(d)) if n else np.nan,
        "median_delta_event_f1": float(np.nanmedian(d)) if n else np.nan,
        "nb_ci_lo": lo, "nb_ci_hi": hi,
        "p_wilcoxon": wilcoxon_p(d), "p_nadeau_bengio": p_nb, "p_fold_level": p_fl,
        "n_folds": n_fl, "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
        "identical": int(np.isclose(d, 0).sum()),
        "mean_delta_fp24h": float(np.nanmean(dp)) if n else np.nan,
        "worst_delta_fp24h": worst_delta(dp, higher_is_better=False),
        "tail_fp24h_gt20": int((dp > TAIL_FP24H).sum()),
        "cvar10_delta_event_f1": cvar(d, CVAR_ALPHA, higher_is_better=True),
        "prereg_harm_rate": float(np.mean((d < HARM_DELTA_EVENT_F1) | (dp > HARM_DELTA_FP24H))) if n else np.nan,
    }


def harm_curve(d_f1, d_fp, arm, ref, f1_grid, fp_grid):
    """Harm rate as a function of the margin, one axis at a time and jointly."""
    d = np.asarray(d_f1, float); dp = np.asarray(d_fp, float)
    rows = []
    for m in f1_grid:
        rows.append({"arm": arm, "reference": ref, "axis": "event_f1", "margin": float(m),
                     "harm_rate": float(np.mean(d < -m))})
    for m in fp_grid:
        rows.append({"arm": arm, "reference": ref, "axis": "fp24h", "margin": float(m),
                     "harm_rate": float(np.mean(dp > m))})
    # Joint sweep along the diagonal: F1 margin m, FP margin scaled so both reach their grid
    # maxima together. Reported so the registered OR-rule has a curve of its own.
    for m_f1, m_fp in zip(f1_grid, np.interp(f1_grid, [f1_grid[0], f1_grid[-1]],
                                             [fp_grid[0], fp_grid[-1]])):
        rows.append({"arm": arm, "reference": ref, "axis": "joint", "margin": float(m_f1),
                     "margin_fp24h": float(m_fp),
                     "harm_rate": float(np.mean((d < -m_f1) | (dp > m_fp)))})
    return rows


def null_floor(v2: pd.DataFrame, p2: pd.DataFrame, f1_grid, fp_grid):
    """The different-initialisation floor: same (fold, seed, detector, condition), two runs.

    ``_v2`` ran before ``torch.manual_seed`` preceded ``build_model``; ``_p2`` after. For the
    three pool-free baselines that is the only difference, so each pair is one draw of pure
    run-to-run noise. Harm is one-sided, and the ordering of the two runs is arbitrary, so the
    null harm rate at margin m is P(delta < -m) under the symmetrised distribution
    = 0.5 * P(|delta| > m).
    """
    k = ["fold", "seed", "detector", "condition"]
    a = v2[v2.condition.isin(SIMPLE) & (v2.detector == "tcn")][k + ["event_f1", "fp_per_24h"]]
    b = p2[p2.condition.isin(SIMPLE) & (p2.detector == "tcn")][k + ["event_f1", "fp_per_24h"]]
    m = a.merge(b, on=k, suffixes=("_v2", "_p2"))
    m["d_f1"] = m.event_f1_v2 - m.event_f1_p2
    m["d_fp"] = m.fp_per_24h_v2 - m.fp_per_24h_p2
    sig = m.groupby("condition").agg(sigma_f1=("d_f1", lambda s: float(np.std(s, ddof=1))),
                                     max_abs_d_f1=("d_f1", lambda s: float(np.abs(s).max())),
                                     sigma_fp=("d_fp", lambda s: float(np.std(s, ddof=1))),
                                     max_abs_d_fp=("d_fp", lambda s: float(np.abs(s).max())),
                                     n=("d_f1", "size")).reset_index()
    rows = []
    for m_ in f1_grid:
        rows.append({"arm": "NULL same-spec re-run", "reference": "null", "axis": "event_f1",
                     "margin": float(m_), "harm_rate": float(0.5 * np.mean(np.abs(m.d_f1) > m_))})
    for m_ in fp_grid:
        rows.append({"arm": "NULL same-spec re-run", "reference": "null", "axis": "fp24h",
                     "margin": float(m_), "harm_rate": float(0.5 * np.mean(np.abs(m.d_fp) > m_))})
    for m_f1, m_fp in zip(f1_grid, np.interp(f1_grid, [f1_grid[0], f1_grid[-1]],
                                             [fp_grid[0], fp_grid[-1]])):
        # Symmetrised joint: each pair contributes both orderings.
        both_f1 = np.concatenate([m.d_f1, -m.d_f1]); both_fp = np.concatenate([m.d_fp, -m.d_fp])
        rows.append({"arm": "NULL same-spec re-run", "reference": "null", "axis": "joint",
                     "margin": float(m_f1), "margin_fp24h": float(m_fp),
                     "harm_rate": float(np.mean((both_f1 < -m_f1) | (both_fp > m_fp)))})
    return m, sig, rows


def power_design(sd_values: dict, splits: dict):
    """NB standard error and 80%-power MDE for candidate designs over the 23 groups."""
    from scipy.stats import t as tdist
    designs = []
    # As run: folds 0-2 of the committed 5-fold partition, rho from the actual group counts.
    rho3 = test_train_ratio(splits, [0, 1, 2])
    designs.append(("3 folds x 3 seeds (as run)", 3, 3, rho3))
    for k, seeds in [(5, 1), (5, 3), (10, 1), (10, 3), (23, 1), (23, 2), (23, 3)]:
        n_test = N_GROUPS / k
        rest = N_GROUPS - n_test
        n_val = max(1.0, round(VAL_FRACTION * rest))
        n_train = rest - n_val
        designs.append((f"{k} folds x {seeds} seed{'s' if seeds > 1 else ''}"
                        + (" (leave-one-group-out)" if k == N_GROUPS else ""),
                        k, seeds, n_test / n_train))
    rows = []
    for sd_name, sd in sd_values.items():
        for label, k, seeds, rho in designs:
            n = k * seeds
            se = sd * np.sqrt(1.0 / n + rho)
            mde = (tdist.ppf(0.975, n - 1) + tdist.ppf(0.80, n - 1)) * se
            rows.append({"sd_source": sd_name, "sd": sd, "design": label, "folds": k,
                         "seeds": seeds, "n_cells": n, "rho": rho, "se_nb": se,
                         "mde_80pct": mde, "se_floor_inf_seeds": sd * np.sqrt(rho)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def analyse_grid(label, csv, ratio, q_core, rho, out: Path):
    df = pd.read_csv(A / csv)
    df = df[df.detector == "tcn"]
    qs = sorted({float(q) for q in df[df.condition == "gated"]["q"].dropna().unique()})
    blocks = [(fo, se, blk) for (fo, se), blk in df.groupby(["fold", "seed"])]
    folds = np.array([fo for fo, _, _ in blocks])

    pol_rows, chosen_rows = [], []
    for fo, se, blk in blocks:
        p, c = cell_policies(blk, ratio, qs)
        for name, (f1, fp) in p.items():
            pol_rows.append({"grid": label, "fold": fo, "seed": se, "policy": name,
                             "event_f1": f1, "fp_per_24h": fp})
        chosen_rows.append({"grid": label, "fold": fo, "seed": se, **c})
    pol = pd.DataFrame(pol_rows)
    chosen = pd.DataFrame(chosen_rows)
    wide_f1 = pol.pivot_table(index=["fold", "seed"], columns="policy", values="event_f1").sort_index()
    wide_fp = pol.pivot_table(index=["fold", "seed"], columns="policy", values="fp_per_24h").sort_index()
    folds = wide_f1.index.get_level_values("fold").to_numpy()

    print(f"\n{'=' * 78}\n{label}: {csv} ratio {ratio}  (TCN, n = {len(wide_f1)} cells)\n{'=' * 78}")

    # (1) reference selection
    print("\n(1) Registered reference: chosen on TEST vs chosen on VALIDATION")
    print(f"    test-selected  mean event-F1 {wide_f1['best_simple_TESTselected'].mean():.3f}  "
          f"arms {dict(chosen.ref_test.value_counts())}")
    print(f"    val-selected   mean event-F1 {wide_f1['best_simple_valselected'].mean():.3f}  "
          f"arms {dict(chosen.ref_val.value_counts())}")
    agree = int((chosen.ref_test == chosen.ref_val).sum())
    infl = wide_f1['best_simple_TESTselected'].mean() - wide_f1['best_simple_valselected'].mean()
    print(f"    same arm in {agree}/{len(chosen)} cells; test-selection inflation {infl:+.3f}")

    # (2) policies vs the two references + the head-to-head against the gate
    frontier, comps = [], []
    refs = {"real_only": "real_only", "registered (val-selected)": "best_simple_valselected"}
    for ref_label, ref_pol in refs.items():
        rf1, rfp = wide_f1[ref_pol].to_numpy(), wide_fp[ref_pol].to_numpy()
        for pname in wide_f1.columns:
            if pname == ref_pol:
                continue
            d_f1 = wide_f1[pname].to_numpy() - rf1
            d_fp = wide_fp[pname].to_numpy() - rfp
            row = paired_row(d_f1, d_fp, folds, rho, pname, ref_label)
            row.update({"grid": label, "mean_event_f1": float(wide_f1[pname].mean()),
                        "mean_fp24h": float(wide_fp[pname].mean())})
            frontier.append(row)
    fr = pd.DataFrame(frontier)
    show = fr[fr.reference == "registered (val-selected)"].set_index("comparison")
    order = [c for c in ["real_only", "class_weighted", "classical_aug", "ungated",
                         "ungated_failclosed", "valsel4", "valsel4_fpguard"]
             + [c for c in show.index if c.startswith(("gated", "random_gated", "admit_always", "valsel5"))]
             if c in show.index]
    print("\n(2) Deployable policies vs the VALIDATION-selected registered reference")
    print(f"    {'policy':<28s} {'F1':>6s} {'dF1':>7s} {'p_NB':>6s} {'FP/24h':>7s} {'dFP':>7s} "
          f"{'worstdFP':>9s} {'tail>20':>7s} {'harm':>5s}")
    for c in order:
        r = show.loc[c]
        print(f"    {c:<28s} {r.mean_event_f1:6.3f} {r.mean_delta_event_f1:+7.3f} {r.p_nadeau_bengio:6.3f} "
              f"{r.mean_fp24h:7.2f} {r.mean_delta_fp24h:+7.2f} {r.worst_delta_fp24h:+9.2f} "
              f"{int(r.tail_fp24h_gt20):7d} {r.prereg_harm_rate:5.2f}")

    print("\n    Head-to-head with the gate as deployed (positive = competitor better):")
    for q in qs:
        g = f"gated q{q:g}"
        for comp in ["valsel4", "valsel4_fpguard", "ungated_failclosed", f"gated q{q:g} -> valsel3",
                     f"valsel5 q{q:g}", "best_simple_valselected", "class_weighted"]:
            if comp not in wide_f1.columns:
                continue
            d_f1 = wide_f1[comp].to_numpy() - wide_f1[g].to_numpy()
            d_fp = wide_fp[comp].to_numpy() - wide_fp[g].to_numpy()
            row = paired_row(d_f1, d_fp, folds, rho, f"{comp} vs {g}", g)
            row["grid"] = label
            comps.append(row)
            print(f"      {comp:<26s} vs {g:<14s} dF1 {row['mean_delta_event_f1']:+.3f} "
                  f"NB95 [{row['nb_ci_lo']:+.3f},{row['nb_ci_hi']:+.3f}] p_wil {row['p_wilcoxon']:.3f} "
                  f"p_NB {row['p_nadeau_bengio']:.3f}  (+{row['better']}/-{row['worse']}/={row['identical']})  "
                  f"dFP {row['mean_delta_fp24h']:+.2f}")
    cm = pd.DataFrame(comps)
    print(f"    {len(cm)} head-to-head tests in this grid; Bonferroni alpha 0.05 -> p < {0.05 / max(1, len(cm)):.4f}")
    print("    arms chosen by valsel4:", dict(chosen.valsel4.value_counts()),
          "| ungated_failclosed deployed ungated in",
          int((chosen.ungated_failclosed == "ungated").sum()), "cells")

    # (3) harm curves
    f1_grid, fp_grid = margin_grids()
    curve = []
    arms = ["ungated", f"gated q{q_core:g}", f"random_gated q{q_core:g}", "class_weighted",
            "valsel4", "ungated_failclosed"]
    for ref_label, ref_pol in refs.items():
        for arm in arms:
            if arm not in wide_f1.columns or arm == ref_pol:
                continue
            d_f1 = wide_f1[arm].to_numpy() - wide_f1[ref_pol].to_numpy()
            d_fp = wide_fp[arm].to_numpy() - wide_fp[ref_pol].to_numpy()
            for r in harm_curve(d_f1, d_fp, arm, ref_label, f1_grid, fp_grid):
                r["grid"] = label
                curve.append(r)
    cv = pd.DataFrame(curve)

    out.mkdir(parents=True, exist_ok=True)
    pol.to_csv(out / f"policies_{label}.csv", index=False)
    chosen.to_csv(out / f"chosen_arms_{label}.csv", index=False)
    fr.to_csv(out / f"frontier_valref_{label}.csv", index=False)
    cm.to_csv(out / f"head_to_head_{label}.csv", index=False)
    cv.to_csv(out / f"harm_curve_{label}.csv", index=False)
    return wide_f1, wide_fp, fr, cm, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out)
    splits = json.load(open(SPLITS))
    rho = test_train_ratio(splits, [0, 1, 2])
    print(f"Nadeau-Bengio rho (n_test/n_train, folds 0-2) = {rho:.4f}")

    sd_sources = {}
    all_fr, all_cm, all_cv = [], [], []
    for label, csv, ratio, q_core in GRIDS:
        wf1, wfp, fr, cm, cv = analyse_grid(label, csv, ratio, q_core, rho, out)
        all_fr.append(fr); all_cm.append(cm); all_cv.append(cv)
        if label == "p3_r100":
            sd_sources["valsel4 - gated q0.95 (p3)"] = float(np.std(wf1["valsel4"] - wf1["gated q0.95"], ddof=1))
            sd_sources["ungated - real_only (p3, r=1.0)"] = float(np.std(wf1["ungated"] - wf1["real_only"], ddof=1))
        if label == "p2_r030":
            sd_sources["ungated - real_only (p2, r=0.30)"] = float(np.std(wf1["ungated"] - wf1["real_only"], ddof=1))
    sd_sources["DG2 reference sd 0.094"] = 0.094

    # (3b) the null floor
    v2 = pd.read_csv(A / "downstream_gated_v2.csv")
    p2 = pd.read_csv(A / "downstream_gated_p2.csv")
    f1_grid, fp_grid = margin_grids()
    pairs, sig, null_rows = null_floor(v2, p2, f1_grid, fp_grid)
    print(f"\n{'=' * 78}\n(3) Different-initialisation floor: {len(pairs)} same-spec pairs (_v2 vs _p2, TCN)\n{'=' * 78}")
    print(sig.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    nl = pd.DataFrame(null_rows)
    for m in (0.01, 0.05, 0.10, 0.15, 0.20):
        r = nl[(nl.axis == "event_f1") & np.isclose(nl.margin, m)].harm_rate.iloc[0]
        print(f"    null harm rate at dF1 margin {m:.2f}: {r:.2f}")
    for m in (0.25, 5.0, 10.0, 20.0, 40.0):
        r = nl[(nl.axis == "fp24h") & np.isclose(nl.margin, m)].harm_rate.iloc[0]
        print(f"    null harm rate at dFP/24h margin {m:5.2f}: {r:.2f}")
    print("    (registered margins are 0.01 and 0.25: at those the null flags roughly half of all\n"
          "     same-specification re-runs as 'harm')")
    pairs.to_csv(out / "null_floor_pairs.csv", index=False)
    sig.to_csv(out / "null_floor_sigma.csv", index=False)
    nl.to_csv(out / "harm_curve_null.csv", index=False)

    # (4) power design
    pw = power_design(sd_sources, splits)
    print(f"\n{'=' * 78}\n(4) Phase 3 power design (Nadeau-Bengio, 80% power, two-sided alpha 0.05)\n{'=' * 78}")
    ref = pw[pw.sd_source == "DG2 reference sd 0.094"]
    print(f"    {'design':<40s} {'n':>4s} {'rho':>6s} {'se_NB':>6s} {'MDE':>6s} {'floor':>6s}")
    for _, r in ref.iterrows():
        print(f"    {r.design:<40s} {int(r.n_cells):4d} {r.rho:6.3f} {r.se_nb:6.3f} {r.mde_80pct:6.3f} {r.se_floor_inf_seeds:6.3f}")
    print("    sd sources available:", {k: round(v, 3) for k, v in sd_sources.items()})
    pw.to_csv(out / "power_design.csv", index=False)

    pd.concat(all_fr).to_csv(out / "frontier_valref_all.csv", index=False)
    pd.concat(all_cm).to_csv(out / "head_to_head_all.csv", index=False)
    pd.concat(all_cv + [nl]).to_csv(out / "harm_curve_all.csv", index=False)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
