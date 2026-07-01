"""Analysis: harm characterization, gen x detector deltas, tables and key figures.

v5.3 leads with HARM. The headline artifacts are:
* harm/tail-risk tables: paired deltas of each synthetic condition (ungated, gated) vs
  the simple baselines, per (generator, detector, scarcity), reporting harm rate,
  worst-fold delta, and CVaR alongside mean/median + bootstrap 95% CIs (plan Sec 17);
* the gen x detector heatmap of d(event F1) / d(FP/24h) (Figure 8);
* a gate-behavior summary (admission rate, fail-closed revert rate, gated-vs-ungated);
* performance vs training fraction per detector/condition (Figure 9).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.stats import paired_delta

BASELINES = ["real_only", "class_weighted", "balanced_sampler", "classical_aug"]
SYNTHETIC_CONDITIONS = ["ungated_synthetic_aug", "trust_gated_synthetic_aug"]

# Pre-registered harm thresholds (mirror configs/chbmit_synthetic.yaml -> harm:).
HARM_DELTA_EVENT_F1 = -0.01
HARM_DELTA_FP24H = 0.25


def _harm_threshold(metric: str, higher_is_better: bool) -> Optional[float]:
    if metric == "event_f1":
        return HARM_DELTA_EVENT_F1
    if metric in ("fp_per_24h", "fa_per_hour"):
        return HARM_DELTA_FP24H
    return None


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[~df.get("condition", pd.Series(dtype=str)).isna()] if "condition" in df else df


def generator_detector_deltas(df: pd.DataFrame, metric: str = "event_f1",
                              higher_is_better: bool = True,
                              target_condition: str = "ungated_synthetic_aug") -> pd.DataFrame:
    """Paired delta of a synthetic condition vs best baseline per (generator, detector, scarcity).

    Reports tail-risk (harm rate at the pre-registered threshold, worst-fold delta, CVaR)
    so event-level harm hidden by the mean is visible (v5.3 Sec 0, 6).
    """
    rows: List[dict] = []
    syn = df[df["condition"] == target_condition]
    base = df[df["condition"].isin(BASELINES)]
    harm_thr = _harm_threshold(metric, higher_is_better)
    for (detector, frac), bsub in base.groupby(["detector", "scarcity_fraction"]):
        # best baseline per (fold, seed): max (or min) metric across baseline conditions
        agg = "max" if higher_is_better else "min"
        best = (bsub.groupby(["fold", "seed"])[metric].agg(agg)
                .rename("baseline").reset_index())
        for generator, ssub in syn[(syn["detector"] == detector)
                                   & (syn["scarcity_fraction"] == frac)].groupby("generator"):
            merged = ssub.merge(best, on=["fold", "seed"], how="inner")
            if merged.empty:
                continue
            r = paired_delta(merged[metric].to_numpy(), merged["baseline"].to_numpy(),
                             higher_is_better=higher_is_better, harm_threshold=harm_thr)
            rows.append({
                "target_condition": target_condition,
                "detector": detector, "generator": generator, "scarcity_fraction": frac,
                "metric": metric, "n": r.n, "mean_delta": r.mean_delta,
                "median_delta": r.median_delta, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "worst_delta": r.worst_delta, "cvar": r.cvar, "harm_rate": r.harm_rate,
                "harm_threshold": r.harm_threshold, "wilcoxon_p": r.wilcoxon_p,
            })
    return pd.DataFrame(rows)


def harm_table(df: pd.DataFrame, metric: str = "event_f1",
               higher_is_better: bool = True) -> pd.DataFrame:
    """Harm/tail-risk for BOTH synthetic conditions vs best baseline, stacked."""
    parts = [generator_detector_deltas(df, metric, higher_is_better, target)
             for target in SYNTHETIC_CONDITIONS if (df["condition"] == target).any()]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def gate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (detector, generator, scarcity) fail-closed behavior of the trust gate."""
    g = df[df["condition"] == "trust_gated_synthetic_aug"]
    if g.empty or "gate_reverted" not in g.columns:
        return pd.DataFrame()
    rows: List[dict] = []
    for (detector, generator, frac), sub in g.groupby(["detector", "generator", "scarcity_fraction"]):
        rows.append({
            "detector": detector, "generator": generator, "scarcity_fraction": frac,
            "n_cells": int(len(sub)),
            "revert_rate": float(np.nanmean(sub["gate_reverted"].astype(float))),
            "mean_admission_rate": float(np.nanmean(pd.to_numeric(
                sub.get("gate_admission_rate"), errors="coerce"))),
            "mean_delta_val_event_f1": float(np.nanmean(pd.to_numeric(
                sub.get("gate_delta_val_event_f1"), errors="coerce"))),
        })
    return pd.DataFrame(rows)


def gated_vs_ungated(df: pd.DataFrame, metric: str = "event_f1",
                     higher_is_better: bool = True) -> pd.DataFrame:
    """Paired delta gated - ungated per (generator, detector, scarcity): the gate's net effect."""
    rows: List[dict] = []
    gated = df[df["condition"] == "trust_gated_synthetic_aug"]
    ungated = df[df["condition"] == "ungated_synthetic_aug"]
    keys = ["fold", "seed", "detector", "generator", "scarcity_fraction"]
    if gated.empty or ungated.empty:
        return pd.DataFrame()
    for (detector, generator, frac), gsub in gated.groupby(["detector", "generator", "scarcity_fraction"]):
        usub = ungated[(ungated["detector"] == detector) & (ungated["generator"] == generator)
                       & (ungated["scarcity_fraction"] == frac)]
        merged = gsub.merge(usub, on=keys, how="inner", suffixes=("_gated", "_ungated"))
        if merged.empty:
            continue
        r = paired_delta(merged[f"{metric}_gated"].to_numpy(),
                         merged[f"{metric}_ungated"].to_numpy(), higher_is_better=higher_is_better)
        rows.append({
            "detector": detector, "generator": generator, "scarcity_fraction": frac,
            "metric": metric, "n": r.n, "mean_delta": r.mean_delta, "median_delta": r.median_delta,
            "ci_low": r.ci_low, "ci_high": r.ci_high, "worst_delta": r.worst_delta, "cvar": r.cvar,
        })
    return pd.DataFrame(rows)


def save_delta_tables(df: pd.DataFrame, out_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for metric, hib in [("event_f1", True), ("fp_per_24h", False),
                        ("event_sensitivity", True), ("win_auprc", True)]:
        if metric not in df.columns:
            continue
        d = harm_table(df, metric=metric, higher_is_better=hib)
        if d.empty:
            d = generator_detector_deltas(df, metric=metric, higher_is_better=hib)
        p = out_dir / f"delta_{metric}.csv"
        d.to_csv(p, index=False)
        paths[metric] = p
    gv = gated_vs_ungated(df, "event_f1", True)
    if not gv.empty:
        p = out_dir / "gated_vs_ungated_event_f1.csv"
        gv.to_csv(p, index=False)
        paths["gated_vs_ungated_event_f1"] = p
    gs = gate_summary(df)
    if not gs.empty:
        p = out_dir / "gate_summary.csv"
        gs.to_csv(p, index=False)
        paths["gate_summary"] = p
    return paths


def figure_gen_detector_heatmap(df: pd.DataFrame, out_path: str | Path,
                                metric: str = "event_f1", scarcity: Optional[float] = None,
                                target_condition: str = "ungated_synthetic_aug"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = generator_detector_deltas(df, metric=metric,
                                       higher_is_better=(metric != "fp_per_24h"),
                                       target_condition=target_condition)
    if scarcity is not None:
        deltas = deltas[deltas["scarcity_fraction"] == scarcity]
    if deltas.empty:
        return None
    pivot = deltas.pivot_table(index="detector", columns="generator",
                               values="mean_delta", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 2, 1.0 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.nanmax(np.abs(pivot.values)), vmax=np.nanmax(np.abs(pivot.values)))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"d({metric}): {target_condition} - best baseline"
                 + (f" @frac={scarcity}" if scarcity is not None else ""))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def figure_harm_rate_heatmap(df: pd.DataFrame, out_path: str | Path,
                             metric: str = "event_f1"):
    """Heatmap of harm rate (fraction of folds past the pre-registered threshold)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hib = metric != "fp_per_24h"
    d = harm_table(df, metric=metric, higher_is_better=hib)
    if d.empty or d["harm_rate"].isna().all():
        return None
    d = d.assign(row=d["target_condition"] + " / " + d["detector"].astype(str))
    pivot = d.pivot_table(index="row", columns="generator", values="harm_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.8 * len(pivot.columns) + 2.5, 0.7 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"harm rate on {metric} (frac of folds past pre-registered threshold)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def figure_perf_vs_fraction(df: pd.DataFrame, out_path: str | Path, metric: str = "event_f1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = sorted(df["detector"].dropna().unique())
    fig, axes = plt.subplots(1, max(1, len(detectors)),
                             figsize=(4 * max(1, len(detectors)), 4), squeeze=False)
    for ax, det in zip(axes[0], detectors):
        sub = df[df["detector"] == det]
        for cond, csub in sub.groupby("condition"):
            label = cond
            if cond in SYNTHETIC_CONDITIONS and "generator" in csub:
                for gen, gsub in csub.groupby("generator"):
                    g = gsub.groupby("scarcity_fraction")[metric].mean().sort_index()
                    ax.plot(g.index, g.values, marker="o", label=f"{cond}/{gen}")
                continue
            g = csub.groupby("scarcity_fraction")[metric].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", label=label)
        ax.set_title(det); ax.set_xlabel("training fraction"); ax.set_ylabel(metric)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run_analysis(results_csv: str | Path, out_dir: str | Path) -> Dict[str, object]:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    df = load_results(results_csv)
    tables = save_delta_tables(df, out_dir / "tables")
    figs = {}
    figs["fig8_event_f1"] = figure_gen_detector_heatmap(
        df, out_dir / "figures" / "fig8_gen_detector_event_f1.png", "event_f1")
    figs["fig8_fp24h"] = figure_gen_detector_heatmap(
        df, out_dir / "figures" / "fig8_gen_detector_fp24h.png", "fp_per_24h")
    figs["fig_harm_event_f1"] = figure_harm_rate_heatmap(
        df, out_dir / "figures" / "fig_harm_rate_event_f1.png", "event_f1")
    figs["fig9_perf_vs_fraction"] = figure_perf_vs_fraction(
        df, out_dir / "figures" / "fig9_perf_vs_fraction.png", "event_f1")
    return {"tables": {k: str(v) for k, v in tables.items()},
            "figures": {k: (str(v) if v else None) for k, v in figs.items()}}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Analyze tier results")
    ap.add_argument("results_csv")
    ap.add_argument("--out-dir", default="results_chbmit_synthetic")
    args = ap.parse_args()
    print(run_analysis(args.results_csv, args.out_dir))
