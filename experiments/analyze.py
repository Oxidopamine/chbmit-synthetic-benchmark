"""Analysis: paired generator x detector deltas, tables and key figures.

Builds the headline artifacts (plan Sections 17, 19, 26):
* paired deltas of ``synthetic_aug`` vs the BEST simple baseline, per
  (generator, detector, scarcity), with bootstrap 95% CIs (pairing on fold+seed);
* Figure 8: generator x detector heatmap of d(event F1) and d(FP/24h);
* Figure 9: performance vs training fraction, per detector/condition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.stats import paired_delta

BASELINES = ["real_only", "class_weighted", "balanced_sampler", "classical_aug"]


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[~df.get("condition", pd.Series(dtype=str)).isna()] if "condition" in df else df


def generator_detector_deltas(df: pd.DataFrame, metric: str = "event_f1",
                              higher_is_better: bool = True) -> pd.DataFrame:
    """Paired delta of synthetic_aug vs best baseline per (generator, detector, scarcity)."""
    rows: List[dict] = []
    syn = df[df["condition"] == "synthetic_aug"]
    base = df[df["condition"].isin(BASELINES)]
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
            r = paired_delta(merged[metric].to_numpy(), merged["baseline"].to_numpy())
            rows.append({
                "detector": detector, "generator": generator, "scarcity_fraction": frac,
                "metric": metric, "n": r.n, "mean_delta": r.mean_delta,
                "median_delta": r.median_delta, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "wilcoxon_p": r.wilcoxon_p,
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
        d = generator_detector_deltas(df, metric=metric, higher_is_better=hib)
        p = out_dir / f"delta_{metric}.csv"
        d.to_csv(p, index=False)
        paths[metric] = p
    return paths


def figure_gen_detector_heatmap(df: pd.DataFrame, out_path: str | Path,
                                metric: str = "event_f1", scarcity: Optional[float] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = generator_detector_deltas(df, metric=metric,
                                       higher_is_better=(metric != "fp_per_24h"))
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
    ax.set_title(f"d({metric}): synthetic_aug - best baseline"
                 + (f" @frac={scarcity}" if scarcity is not None else ""))
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
            if cond == "synthetic_aug" and "generator" in csub:
                for gen, gsub in csub.groupby("generator"):
                    g = gsub.groupby("scarcity_fraction")[metric].mean().sort_index()
                    ax.plot(g.index, g.values, marker="o", label=f"synthetic/{gen}")
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
