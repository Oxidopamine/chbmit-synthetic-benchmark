"""Generate the three figures for reports/PREPRINT_DRAFT.md from committed result CSVs.

Run it with:

    python3 scripts/make_preprint_figures.py --out reports/figures

(Historical note: this script carried a "never executed" warning because Windows Smart App Control
blocked matplotlib's `kiwisolver` extension on the authoring machine. That is no longer the case --
verified 2026-08-31, the figures render locally.)

Figures, in the order they appear in the manuscript:

  fig1_realized_vs_requested  (§4.2) -- the central finding. Requested injection count against
      what the gate actually admitted, under both admission references. The real_ictal points sit
      on the floor irrespective of request; the pool points sit on the identity line. This is the
      one figure that should survive if only one can.
  fig2_dose_response          (§4.3) -- event-F1 against realized injection ratio for the ungated
      arm, which receives the requested dose exactly, with per-cell points behind the mean.
      Monotone, no interior optimum.
  fig3_tail_control           (§4.5) -- distribution of ΔFP/24h for admitted vs reverted cells,
      with the +20 tail threshold marked, split by arm so the dose-stratification argument is
      visible rather than asserted.

Every number plotted is read from the committed CSVs; nothing is recomputed or smoothed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

A = Path("results_chbmit_synthetic/real_validation/analysis_tierB")

# Colour-blind safe, print-safe. Deliberately not a default matplotlib cycle.
TEACHER, RANDOM, REAL = "#0072B2", "#D55E00", "#444444"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.figsize": (5.5, 3.4),
    })


def _n_pos_by_fold(p2: pd.DataFrame) -> dict:
    """Real ictal training windows per fold. Read, never hardcoded -- it differs by fold.

    Preferred source is ``w2_dose_prediction.csv``, whose ``n_real`` column is the exact count
    the generator was fit on (2508 / 2406 / 2643). Falls back to reconstructing it from the
    gated arm's own bookkeeping -- n_admitted / admission_rate is the candidate pool size, which
    the driver builds as oversample * ratio * n_pos -- which lands within 2 windows but inherits
    the rounding in the stored admission_rate.
    """
    w2 = A / "w2_dose_prediction.csv"
    if w2.exists():
        d = pd.read_csv(w2)
        return d.groupby("fold").n_real.median().to_dict()
    g = p2[(p2.condition == "gated") & (p2.gate_admission_rate > 0)].copy()
    g["n_pos"] = g.gate_n_admitted / g.gate_admission_rate / (6 * g.ratio)
    return g.groupby("fold").n_pos.median().round().to_dict()


def fig1_realized_vs_requested(out: Path):
    """Requested vs admitted count, both references. The paper's central claim, in one panel."""
    import matplotlib.pyplot as plt
    p2 = pd.read_csv(A / "downstream_gated_p2.csv")     # real_ictal reference
    p3 = pd.read_csv(A / "downstream_gated_p3.csv")     # pool reference
    # Real ictal count per FOLD, not a single pooled constant: it is 2508 / 2406 / 2643 for folds
    # 0 / 1 / 2, so using fold 0's value everywhere put a ~5% error in the requested-count axis.
    # Recovered from the ungated arm, which receives exactly ratio * n_pos windows.
    n_pos = _n_pos_by_fold(p2)

    g2 = p2[(p2.condition == "gated") & p2.gate_n_admitted.notna()]
    req2 = g2.ratio * g2.fold.map(n_pos)
    g3 = p3[(p3.condition == "gated") & p3.gate_n_admitted.notna()]
    # under pool, requested = min(oversample*(1-q),1)*n_synth with n_synth = ratio*n_pos
    req3 = np.minimum(6 * (1 - g3["q"]), 1.0) * g3.ratio * g3.fold.map(n_pos)

    fig, ax = plt.subplots()
    lim = [5, 3000]
    ax.plot(lim, lim, ls="--", c=REAL, lw=1, zorder=1, label="identity (admitted = requested)")
    ax.scatter(req2, g2.gate_n_admitted, s=26, c=RANDOM, alpha=.85, zorder=3,
               label='reference = "real_ictal" (as first built)')
    ax.scatter(req3, g3.gate_n_admitted, s=26, c=TEACHER, marker="^", alpha=.85, zorder=3,
               label='reference = "pool" (as published)')
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("requested synthetic windows"); ax.set_ylabel("windows actually admitted")
    ax.set_title("The gate admits what it admits, not what it is asked for")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(out / "fig1_realized_vs_requested.pdf"); plt.close(fig)


def fig2_dose_response(out: Path):
    """Event-F1 against realized dose for the ungated arm (which receives the full request)."""
    import matplotlib.pyplot as plt
    p2 = pd.read_csv(A / "downstream_gated_p2.csv")
    p3 = pd.read_csv(A / "downstream_gated_p3.csv")
    k = ["fold", "seed"]
    pts = [(0.0, p2[p2.condition == "real_only"].set_index(k).sort_index().event_f1)]
    for r in (0.10, 0.30):
        m = p2[(p2.condition == "ungated") & np.isclose(p2.ratio, r)]
        pts.append((r, m.set_index(k).sort_index().event_f1))
    # r = 1.0 comes from the _p3 grid, which shares _p2's code version and is bit-identical to
    # the Phase 1 ungated arm in all 9 cells -- so the ladder is a single within-code comparison.
    pts.append((1.0, p3[p3.condition == "ungated"].set_index(k).sort_index().event_f1))

    fig, ax = plt.subplots()
    for r, s in pts:                                   # per-cell points behind the mean
        ax.scatter(np.full(len(s), r) + np.random.default_rng(0).normal(0, .008, len(s)),
                   s.values, s=12, c=REAL, alpha=.35, zorder=2)
    xs = [r for r, _ in pts]; ys = [s.mean() for _, s in pts]
    ax.plot(xs, ys, "-o", c=TEACHER, lw=2, ms=6, zorder=3, label="mean over 9 cells")
    ax.axvspan(0.05, 0.30, color=TEACHER, alpha=.07, zorder=1)
    ax.text(0.175, ax.get_ylim()[1], "source method's\noperating band", ha="center", va="top",
            fontsize=7.5, color=TEACHER)
    ax.set_xlabel("realized injection ratio  r = n_synthetic / n_real_ictal")
    ax.set_ylabel("test event-F1")
    ax.set_title("No interior optimum inside the band the source method uses")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(out / "fig2_dose_response.pdf"); plt.close(fig)


def fig3_tail_control(out: Path):
    """ΔFP/24h by the gate's own decision, split by arm so dose stratification is visible."""
    import matplotlib.pyplot as plt
    t = pd.read_csv(A / "analysis_v2_gate_tail.csv")
    arms = sorted(t.arm.unique())
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        for j, (lab, sub) in enumerate((("admitted", t[(t.arm == arm) & ~t.reverted]),
                                        ("reverted", t[(t.arm == arm) & t.reverted]))):
            x = i + (j - 0.5) * 0.45
            ax.scatter(x + rng.normal(0, .05, len(sub)), sub.delta_fp24h_aug, s=20,
                       c=(TEACHER if j == 0 else RANDOM), alpha=.8,
                       label=lab if i == 0 else None)
            ax.hlines(sub.delta_fp24h_aug.median(), x - .18, x + .18,
                      color=(TEACHER if j == 0 else RANDOM), lw=2)
    ax.axhline(20, ls=":", c=REAL, lw=1)
    ax.text(len(arms) - .5, 22, "tail threshold +20 FP/24 h", ha="right", fontsize=7.5, c=REAL)
    ax.axhline(0, ls="-", c=REAL, lw=.6, alpha=.5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylabel("Δ FP/24 h vs real_only")
    ax.set_title("The gate's decision separates the false-alarm tail, within every arm")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(out / "fig3_tail_control.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _style()
    for fn in (fig1_realized_vs_requested, fig2_dose_response, fig3_tail_control):
        fn(out)
        print(f"wrote {out}/{fn.__name__.split('_', 1)[0]}*.pdf")


if __name__ == "__main__":
    main()
