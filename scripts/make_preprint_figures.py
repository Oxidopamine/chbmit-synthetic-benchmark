"""Generate the result figures for the README and reports/PREPRINT_DRAFT.md from committed CSVs.

Run it with:

    python3 scripts/make_preprint_figures.py --out reports/figures

Each figure is written three ways:

  fig<N>_<name>-light.png   README embed, light GitHub theme
  fig<N>_<name>-dark.png    README embed, dark GitHub theme
  fig<N>_<name>.pdf         manuscript embed (vector, light tokens)

The README pairs the two PNGs with ``<picture><source media="(prefers-color-scheme: dark)" ...>``
so a dark-theme reader is not served black axis text on a black page. The dark variant is a
SELECTED palette stepped for the dark surface, not an automatic inversion.

Figures, in the order they appear:

  fig1_realized_vs_requested  -- the central Phase 2 finding. Requested injection count against
      what the gate actually admitted, under both admission references. Log-log with equal aspect
      so the identity line really is at 45 degrees and "on the line" is readable. The real_ictal
      points sit on a floor irrespective of request; the pool points sit on the identity line.
  fig2_dose_response          -- event-F1 against realized injection ratio for the ungated arm,
      which receives the requested dose exactly, with per-cell points behind the mean. Monotone,
      no interior optimum inside the source method's operating band.
  fig3_tail_control           -- every raw cell of delta FP/24h for admitted vs reverted, split by
      arm so the dose-stratification argument is visible rather than asserted. A strip plot, not a
      violin: at n=5..22 per group a smoothed density would invent shape the data cannot support.

Every number plotted is read from the committed CSVs; nothing is recomputed or smoothed.

Design constraints this file follows deliberately, so please keep them if you edit it:
  * Palette is validated for colour-vision deficiency in BOTH themes (see PALETTE below). Do not
    add a fourth categorical hue without re-validating -- a scatter is an all-pairs form and only
    three slots clear the all-pairs floors.
  * Identity is never colour alone: every multi-series figure also separates by marker shape or
    by position, so the figures survive greyscale and CVD.
  * Sizes are given in CSS px by the design reference and converted once here (1 px = 0.75 pt).
  * Reference rules (identity, threshold, zero) are SOLID and distinguished by weight and colour.
    Dashes are reserved for data series, so a dashed rule can never be mistaken for data.
  * No twin axes anywhere. Two units never share one frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

A = Path("results_chbmit_synthetic/real_validation/analysis_tierB")

PX = 0.75  # 1 CSS px in matplotlib points

# Colour-vision-safe in both themes, validated with the design reference's checker:
# light all-pairs worst CVD dE 9.2 / normal 24.0; dark 9.4 / 20.9. Slot 1 = blue, slot 2 = orange.
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8a86",
        "grid": "#e8e8e6", "s1": "#2a78d6", "s2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#8a8a86",
        "grid": "#33332f", "s1": "#3987e5", "s2": "#d95926",
    },
}


def _style(t: dict):
    """rcParams for one theme. Every colour that matplotlib defaults to black is set explicitly."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        # Pin the bundled face so any machine reproduces the same bytes.
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.figsize": (5.5, 3.4),
        # Theme tokens.
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "savefig.edgecolor": "none",
        "text.color": t["ink"], "axes.labelcolor": t["ink"], "axes.titlecolor": t["ink"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"], "axes.linewidth": PX,
        "grid.color": t["grid"], "grid.linewidth": PX, "grid.linestyle": "-",
        "xtick.major.width": PX, "ytick.major.width": PX,
        "xtick.minor.width": PX, "ytick.minor.width": PX,
    })


def _save(fig, out: Path, stem: str, theme: str):
    """PNG for the README in this theme; PDF once, from the light pass, for the manuscript."""
    fig.savefig(out / f"{stem}-{theme}.png", metadata={"Software": None})
    if theme == "light":
        fig.savefig(out / f"{stem}.pdf")


def _assert_in_limits(lim, series_list):
    """Fail loudly if any plotted point falls outside the axes.

    A clipped point is invisible and silent: the reader sees a figure that looks complete and is
    not. This is checked rather than eyeballed because the one time it happened, the dropped
    points were the three smallest admitted counts -- the floor that the figure exists to show.
    """
    lo, hi = lim
    for s in series_list:
        v = np.asarray(s, dtype=float)
        v = v[~np.isnan(v)]
        bad = v[(v < lo) | (v > hi)]
        if len(bad):
            raise ValueError(f"{len(bad)} point(s) outside axes limits {lim}: {sorted(bad)[:8]}")


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


def fig1_realized_vs_requested(out: Path, t: dict, theme: str):
    """Requested vs admitted count, both references. The paper's central claim, in one panel."""
    import matplotlib.pyplot as plt
    p2 = pd.read_csv(A / "downstream_gated_p2.csv")     # real_ictal reference
    p3 = pd.read_csv(A / "downstream_gated_p3.csv")     # pool reference
    # Real ictal count per FOLD, not a single pooled constant: it is 2508 / 2406 / 2643 for folds
    # 0 / 1 / 2, so using fold 0's value everywhere put a ~5% error in the requested-count axis.
    n_pos = _n_pos_by_fold(p2)

    g2 = p2[(p2.condition == "gated") & p2.gate_n_admitted.notna()]
    req2 = g2.ratio * g2.fold.map(n_pos)
    g3 = p3[(p3.condition == "gated") & p3.gate_n_admitted.notna()]
    # under pool, requested = min(oversample*(1-q),1)*n_synth with n_synth = ratio*n_pos
    req3 = np.minimum(6 * (1 - g3["q"]), 1.0) * g3.ratio * g3.fold.map(n_pos)

    # Wider than the default: equal aspect plus a tight bbox crops the unused width off a square
    # axes, so HEIGHT is what widens it: at (5.5, 4.4) the saved PNG came out 1261 px, under the
    # 1400 px floor for a ~880 px README column.
    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    # Limits must contain every point. The real_ictal floor reaches 1 admitted window, and an
    # earlier lower bound of 3 silently dropped three cells off the bottom of the axes -- which is
    # the one failure this figure cannot afford, since the floor IS the finding.
    lim = [0.7, 2000]
    _assert_in_limits(lim, [req2, g2.gate_n_admitted, req3, g3.gate_n_admitted])
    # Identity: solid, recessive, under every point, directly labelled -- never a legend entry.
    ax.plot(lim, lim, ls="-", c=t["muted"], lw=PX, zorder=0)
    ax.annotate("requested = admitted", xy=(430, 430), xycoords="data",
                xytext=(-4, 5), textcoords="offset points", rotation=45,
                rotation_mode="anchor", ha="center", va="bottom", fontsize=7.5, color=t["ink2"])
    # Two series, separated by hue AND marker shape, each dot carrying a surface-coloured ring.
    ax.scatter(req2, g2.gate_n_admitted, s=36, marker="o", c=t["s2"],
               linewidths=1.5 * PX, edgecolors=t["surface"], zorder=3,
               label='reference = "real_ictal"  (as first built)')
    ax.scatter(req3, g3.gate_n_admitted, s=36, marker="^", c=t["s1"],
               linewidths=1.5 * PX, edgecolors=t["surface"], zorder=3,
               label='reference = "pool"  (as published)')
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")     # so "on the line" means what it looks like
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("requested synthetic windows")
    ax.set_ylabel("windows actually admitted")
    ax.set_title("The gate admits what it admits, not what it is asked for")
    ax.legend(loc="upper left")
    _save(fig, out, "fig1_realized_vs_requested", theme)
    plt.close(fig)


def fig2_dose_response(out: Path, t: dict, theme: str):
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
    # The source method's ladder band: a 10% wash, no edge, under everything.
    ax.axvspan(0.05, 0.30, facecolor=t["s1"], alpha=0.10, edgecolor="none", zorder=0)
    rng = np.random.default_rng(0)
    for r, s in pts:                                   # per-cell points behind the mean
        ax.scatter(np.full(len(s), r) + rng.normal(0, .008, len(s)), s.values,
                   s=16, c=t["muted"], alpha=.55, linewidths=0, zorder=2)
    xs = [r for r, _ in pts]
    ys = [s.mean() for _, s in pts]
    ax.plot(xs, ys, "-o", c=t["s1"], lw=1.5, ms=6, zorder=3,
            solid_capstyle="round", solid_joinstyle="round",
            markeredgewidth=1.5 * PX, markeredgecolor=t["surface"])
    # One series: no legend box. Direct labels carry both the mean line and the raw cells.
    ax.annotate("mean of 9 cells", xy=(xs[-1], ys[-1]), xytext=(-6, 10),
                textcoords="offset points", ha="right", fontsize=7.5, color=t["ink2"])
    ax.annotate("one point per (fold, seed) cell", xy=(0.30, min(pts[2][1])), xytext=(8, -4),
                textcoords="offset points", ha="left", fontsize=7.5, color=t["ink2"])
    ax.margins(y=0.14)                       # headroom so no label sits against the frame
    ax.text(0.175, ax.get_ylim()[1], "source method's\noperating band", ha="center", va="top",
            fontsize=7.5, color=t["ink2"])
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("realized injection ratio   r = n_synthetic / n_real_ictal")
    ax.set_ylabel("test event-F1")
    ax.set_title("No interior optimum inside the band the source method uses")
    _save(fig, out, "fig2_dose_response", theme)
    plt.close(fig)


def fig3_tail_control(out: Path, t: dict, theme: str):
    """Delta FP/24h by the gate's own decision, split by arm so dose stratification is visible."""
    import matplotlib.pyplot as plt
    tail = pd.read_csv(A / "analysis_v2_gate_tail.csv")
    arms = sorted(tail.arm.unique())
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    rng = np.random.default_rng(0)
    groups = (("admitted", False, t["s1"], "o"), ("reverted", True, t["s2"], "^"))
    for i, arm in enumerate(arms):
        for j, (lab, rev, colour, mark) in enumerate(groups):
            sub = tail[(tail.arm == arm) & (tail.reverted == rev)]
            x = i + (j - 0.5) * 0.45
            ax.scatter(x + rng.normal(0, .05, len(sub)), sub.delta_fp24h_aug,
                       s=36, marker=mark, c=colour, linewidths=1.5 * PX,
                       edgecolors=t["surface"], zorder=3,
                       label=f"{lab} (n={len(tail[tail.reverted == rev])})" if i == 0 else None)
            ax.hlines(sub.delta_fp24h_aug.median(), x - .18, x + .18,
                      color=colour, lw=1.5, zorder=4)
    # Reference rules: solid, weight-and-colour coded, never dashed.
    ax.axhline(20, ls="-", c=t["ink2"], lw=PX, zorder=1)
    ax.axhline(0, ls="-", c=t["grid"], lw=PX, zorder=1)
    # Exactly three text marks, all of them counts or thresholds the caption asserts.
    ax.text(-0.45, 21.5, "+20 FP/24 h", ha="left", va="bottom",
            fontsize=7.5, color=t["ink2"])
    # Every number here is derived. An earlier version hardcoded "0 of 21" and "of 60" and
    # counted threshold crossings across BOTH groups, so an admitted cell above +20 would have
    # been silently attributed to the reverted count while the text still read "0 of 21".
    adm, rev = tail[~tail.reverted], tail[tail.reverted]
    ax.text(0.02, 0.97,
            f"above the threshold:  {(adm.delta_fp24h_aug > 20).sum()} of {len(adm)} admitted,  "
            f"{(rev.delta_fp24h_aug > 20).sum()} of {len(rev)} reverted",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=t["ink"])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Δ FP/24 h  vs real_only")
    ax.set_title("The gate's decision separates the false-alarm tail, within every arm")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _save(fig, out, "fig3_tail_control", theme)
    plt.close(fig)


FIGURES = (fig1_realized_vs_requested, fig2_dose_response, fig3_tail_control)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--themes", default="light,dark",
                    help="comma-separated subset of light,dark")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in [s.strip() for s in args.themes.split(",") if s.strip()]:
        tokens = PALETTE[theme]
        _style(tokens)
        for fn in FIGURES:
            fn(out, tokens, theme)
            stem = fn.__name__
            extra = " + .pdf" if theme == "light" else ""
            print(f"wrote {out}/{stem}-{theme}.png{extra}")


if __name__ == "__main__":
    main()
