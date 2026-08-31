"""Recompute the reported scarcity-axis numbers from tierB_core.csv, and preview Decision gate 1 Q1.

Run bare:  python3 scripts/verify_reported_numbers.py

Two jobs. (1) Falsify the reported real_only-vs-best-simple-baseline gap -- the numbers that
carry the whole "deviation 1 threatens the paper" argument -- by recomputing them from the CSV
rather than re-reading the claim. (2) Preview Decision gate 1 Q1 with the only data in the repo that has
the registered simple baselines and a synthetic arm in the same cells.

CAVEATS on part 2, which must travel with any quotation of it: cVAE era (the cVAE was crippled
by the 288x KL bug), 5 folds x seed 42, and the gated arm is the cVAE gate
that failed closed almost always. It cannot settle Q1 -- only Phase 1 can. It shows direction.

No GPU, no data store; reads results_chbmit_synthetic/real_validation/tables/tierB_core.csv.
"""
import numpy as np
import pandas as pd

d = pd.read_csv("results_chbmit_synthetic/real_validation/tables/tierB_core.csv")
S1 = d[d.scarcity_fraction == 1.0]

print("=" * 78)
print("Table 1 -- real_only vs BEST SIMPLE BASELINE at scarcity 1.0")
print("  reported: TCN 0.157 -> 0.320 (+0.163) | LCT 0.186 -> 0.296 (+0.110) |")
print("              EEGNet 0.168 -> 0.182 (+0.015)")
print("=" * 78)
simple = ["real_only", "class_weighted", "classical_aug"]
print(f"{'det':<8}{'real_only':>11}{'best simple':>13}{'gap':>9}   per-fold best arm")
for det in ("tcn", "lct", "eegnet"):
    sub = S1[(S1.detector == det) & (S1.condition.isin(simple))]
    piv = sub.pivot_table(index="fold", columns="condition", values="event_f1")
    ro = piv["real_only"].mean()
    best_per_fold = piv[simple].max(axis=1)
    arm = piv[simple].idxmax(axis=1)
    print(f"{det:<8}{ro:>11.3f}{best_per_fold.mean():>13.3f}"
          f"{best_per_fold.mean() - ro:>+9.3f}   {list(arm)}")

print()
print("=" * 78)
print("Table 2 -- per-condition delta vs real_only at scarcity 1.0")
print("  reported TCN: class_weighted +0.112 (5/5), classical_aug +0.115 (5/5),")
print("             ungated(cVAE) +0.022 (2/5);  LCT: +0.077 (4/5), +0.009 (2/5), +0.054 (4/5);")
print("             EEGNet: -0.034 (2/5), -0.018 (1/5), +0.096 (3/5)")
print("=" * 78)
conds = ["class_weighted", "classical_aug", "ungated_synthetic_aug"]
print(f"{'det':<8}" + "".join(f"{c:>26}" for c in conds))
for det in ("tcn", "lct", "eegnet"):
    piv = S1[S1.detector == det].pivot_table(index="fold", columns="condition",
                                             values="event_f1")
    cells = []
    for c in conds:
        dl = piv[c] - piv["real_only"]
        cells.append(f"{dl.mean():+.3f} ({int((dl > 0).sum())}/{len(dl)})")
    print(f"{det:<8}" + "".join(f"{x:>26}" for x in cells))

print()
print("Side-by-side: 'class_weighted TCN reaches event-F1 0.269'")
cw = S1[(S1.detector == "tcn") & (S1.condition == "class_weighted")]["event_f1"]
ro = S1[(S1.detector == "tcn") & (S1.condition == "real_only")]["event_f1"]
print(f"  class_weighted TCN mean event-F1 = {cw.mean():.3f}   real_only = {ro.mean():.3f}")

print()
print("class_weighted TCN 'cuts false alarms sharply (11.8 -> 0.8, 74.0 -> 12.3,")
print("             18.4 -> 6.5 FP/24h)'")
fp = S1[(S1.detector == "tcn") & (S1.condition.isin(["real_only", "class_weighted"]))] \
    .pivot_table(index="fold", columns="condition", values="fp_per_24h")
print(fp.round(2).to_string())

print()
print("=" * 78)
print("Check (a) -- parameter counts constant across cells")
print("  reported EEGNet 1,905 | LCT 120,834 | TCN 129,441 (EEGNet 68x smaller than TCN)")
print("=" * 78)
for det in ("eegnet", "lct", "tcn"):
    u = d[d.detector == det]["n_params"].unique()
    print(f"  {det:<8} {u}  (constant: {len(u) == 1})")
print(f"  TCN / EEGNet ratio = "
      f"{d[d.detector=='tcn'].n_params.iloc[0] / d[d.detector=='eegnet'].n_params.iloc[0]:.1f}x")

print()
print("=" * 78)
print("Check (b) -- ungated cVAE delta across ALL scarcities")
print("  reported EEGNet +0.048 (11/15), TCN -0.005 (6/15), LCT +0.013 (9/15)")
print("=" * 78)
for det in ("eegnet", "tcn", "lct"):
    piv = d[d.detector == det].pivot_table(index=["fold", "scarcity_fraction"],
                                           columns="condition", values="event_f1")
    dl = (piv["ungated_synthetic_aug"] - piv["real_only"]).dropna()
    print(f"  {det:<8} {dl.mean():+.3f} ({int((dl > 0).sum())}/{len(dl)})")

print()
print("=" * 78)
print("Are the simple baselines really generator-independent?")
print("=" * 78)
sb = d[d.condition.isin(["class_weighted", "classical_aug"])]
print(f"  generator column values: {sb.generator.fillna('(empty)').unique()}")
print(f"  n_synth_injected values: {sb.n_synth_injected.unique()}")



HARM_F1, HARM_FP = -0.01, 0.25
SIMPLE = ["real_only", "class_weighted", "classical_aug"]
SYNTH = ["ungated_synthetic_aug", "trust_gated_synthetic_aug"]


def harm(df1, dfp):
    return float(((df1 < HARM_F1) | (dfp > HARM_FP)).mean())


for frac in (1.0, 0.5, 0.25):
    S = d[d.scarcity_fraction == frac]
    print("=" * 92)
    print(f"scarcity {frac}   (n = 5 folds, seed 42, cVAE)")
    print("=" * 92)
    print(f"{'det':<8}{'arm':<26}{'dF1 vs real':>13}{'dF1 vs REG':>12}"
          f"{'dFP vs REG':>12}{'harm REG':>10}{'harm real':>11}")
    for det in ("eegnet", "lct", "tcn"):
        sub = S[S.detector == det]
        f1 = sub.pivot_table(index="fold", columns="condition", values="event_f1")
        fp = sub.pivot_table(index="fold", columns="condition", values="fp_per_24h")
        if not set(SIMPLE).issubset(f1.columns):
            continue
        # registered reference: best simple baseline per fold, chosen by event-F1, that ARM
        # supplying both metrics.
        arm = f1[SIMPLE].idxmax(axis=1)
        reg_f1 = pd.Series([f1.loc[i, arm[i]] for i in f1.index], index=f1.index)
        reg_fp = pd.Series([fp.loc[i, arm[i]] for i in fp.index], index=fp.index)
        for cond in SYNTH:
            if cond not in f1.columns:
                continue
            d_real = f1[cond] - f1["real_only"]
            d_reg = f1[cond] - reg_f1
            dfp_reg = fp[cond] - reg_fp
            dfp_real = fp[cond] - fp["real_only"]
            print(f"{det:<8}{cond:<26}{d_real.mean():>+13.3f}{d_reg.mean():>+12.3f}"
                  f"{dfp_reg.mean():>+12.2f}{harm(d_reg, dfp_reg):>10.2f}"
                  f"{harm(d_real, dfp_real):>11.2f}")
    print()

print("=" * 92)
print("POOLED over all scarcities and folds (n = 15 per detector), the headline question:")
print("  does the SYNTHETIC arm beat the best simple baseline, or only real_only?")
print("=" * 92)
for det in ("eegnet", "lct", "tcn"):
    sub = d[d.detector == det]
    f1 = sub.pivot_table(index=["fold", "scarcity_fraction"], columns="condition",
                         values="event_f1")
    arm = f1[SIMPLE].idxmax(axis=1)
    reg = pd.Series([f1.loc[i, arm[i]] for i in f1.index], index=f1.index)
    for cond in SYNTH:
        dr = (f1[cond] - f1["real_only"]).dropna()
        dg = (f1[cond] - reg).dropna()
        print(f"  {det:<8}{cond:<26} vs real_only {dr.mean():+.3f} "
              f"({int((dr > 0).sum())}/{len(dr)})    vs REGISTERED {dg.mean():+.3f} "
              f"({int((dg > 0).sum())}/{len(dg)})")
