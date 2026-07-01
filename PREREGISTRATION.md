# Pre-Registration — Synthetic Ictal Augmentation Harm & Trust-Gate Benchmark

This file fixes the analysis decisions **before the held-out TEST patients are scored**,
per the v5.3 reframe (`chbmit_synthetic_augmentation_reframe_v5_3.md`, §6). Declaring the
harm definition, the fail-closed margins, and the statistics plan in advance is what lets a
null or negative outcome (e.g. the gate not transferring) read as a *finding* rather than a
*failure*. Do not edit the thresholds below after looking at test results.

The machine-readable copy of these constants lives in
[`configs/chbmit_synthetic.yaml`](configs/chbmit_synthetic.yaml) under the `trust_gate:` and
`harm:` blocks; the harm thresholds are also mirrored in
[`experiments/analyze.py`](experiments/analyze.py) (`HARM_DELTA_EVENT_F1`, `HARM_DELTA_FP24H`).

## 1. Primary object of study

Harm to **patient-independent, event-level** seizure detection from training-time synthetic
ictal augmentation, measured on held-out patients. The trust gate is an *adapted* mitigation
(from TGA, bioRxiv 2026), not a proposed contribution.

## 2. Conditions (core grid)

`real_only`, `class_weighted`, `classical_aug`, `ungated_synthetic_aug`,
`trust_gated_synthetic_aug`. Core generator: cVAE (WGAN-GP is an appendix sensitivity axis).
Split: 5 balanced patient-group folds × 3 seeds. Scarcity: 1.0 (context), 0.5, 0.25.
Synthetic ratio: 1.0×. Detectors: EEGNet, LCT, TCN.

## 3. Pre-registered HARM definition

A paired delta (synthetic condition − reference) on a single (fold, seed) is **harm** if:

| Metric | Harm if | Constant |
|---|---|---|
| event-F1 (higher better) | Δ event-F1 `< −0.01` | `delta_event_f1_threshold = -0.01` |
| FP/24h (lower better) | Δ FP/24h `> +0.25` | `delta_fp24h_threshold = +0.25` |

References for pairing: `real_only`, `class_weighted`, `classical_aug` (the best simple
baseline per (fold, seed) is used for the headline gen×detector deltas).

**Tail-risk reporting (always, alongside the mean):** harm rate (fraction of folds past the
threshold above), worst-fold delta, and CVaR at `cvar_alpha = 0.10` (mean of the worst 10%).
These exist because the central v5.3 thesis is that *mean window metrics conceal event-level
harm* — so a non-negative mean delta does **not** discharge the harm question.

## 4. Trust gate — admission (window level)

The detector teacher is the `real_only` model itself (reused, never retrained; §5.1). A
candidate synthetic window is **admitted** iff the teacher's seizure-class probability for it
is `≥` the `q`-quantile of the teacher's probabilities on the **real training ictal** windows.

- Candidate pool size = `oversample × target_count`, `oversample = 6`.
- Core admission quantile: `q_core = 0.90` (single, fixed for the core grid so the gated
  condition trains **one** augmented model per cell — §5.3).
- Appendix sweep only: `q_grid = {0.75, 0.90, 0.99}`.
- When more than `target_count` windows qualify, keep the highest-confidence ones so the
  injected count matches `ungated_synthetic_aug` (fair paired comparison).

## 5. Trust gate — fail-closed selection (event level)

After training the augmented detector on real + admitted synthetic, **admit the augmented
model** iff BOTH hold on the **validation** patients (never test):

1. `val_event_F1(gated) ≥ val_event_F1(real_only) + admit_margin_event_f1`,
   `admit_margin_event_f1 = 0.0` (must be non-inferior in event-F1), **and**
2. `val_FP/24h(gated) ≤ val_FP/24h(real_only) + fp24h_safety_slack`,
   `fp24h_safety_slack = 0.25` (false-alarm safety constraint).

Otherwise **fail closed**: report the `real_only` model's TEST metrics under the gated spec
(`reverted_to_real_only = True`). If `< min_admitted = 1` synthetic windows are admitted, the
gate also fails closed. The gate decides on validation **only**; this bounds validation
downside by construction, which is exactly why Q3/Outcome interpretation is framed around
**test-set transfer and gain sacrificed**, not "gated beats ungated."

## 6. Statistics plan

- Headline: bootstrap 95% CI on paired deltas (10000 resamples, pairing on fold+seed).
- Exploratory: paired Wilcoxon signed-rank (reported as exploratory given small n).
- Tail-risk: harm rate, worst-fold delta, CVaR(0.10) on the same paired deltas.
- Gate behavior: revert rate, mean admission rate, gated−ungated paired delta.

## 7. Outcome interpretation (registered before test)

A–E exactly as v5.3 §6 (`Outcome A`…`Outcome E`). Each is a publishable result; Outcome E
(validation-selected gating fails to transfer to test, or trades most of the gain for safety)
is registered in advance as an honest limitation finding, not a failure.

## 8. Claim discipline

The gate is **adapted / reformulated**, never "proposed" or "novel." No clinical-deployment or
"clinically realistic synthetic EEG" claims. No "synthetic augmentation generally helps / does
not help" claim. Generator-specific claims only when a single provider is used.
