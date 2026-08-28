# Decision Gate 1 — Phase 1 results

**Date:** 2026-08-29 · **Grid:** `downstream_gated_v2.csv`, 27 complete blocks
(3 detectors × 3 folds × 3 seeds × 7 conditions = 189 rows) · **Run:** 3 Vertex AI Spot A100
jobs, 13.1 h wall-clock, no preemptions.

Reproduce: `python3 scripts/analyze_multiseed.py --tag _v2`

**Stop point.** `the working brief` §8 says to report here and not auto-continue to Phase 2.

---

## Headline

Against the reference the pre-registration actually names — *the best simple baseline per
(fold, seed)* — **synthetic ictal augmentation does not help any detector, and the trust gate
does not recover the loss.** Measured against `real_only`, the arm the previous headline used,
every effect is ≈ 0. The entire apparent effect in this benchmark is a consequence of the
reference choice.

The registered reference is materially stronger than `real_only` on every detector:

| detector | `real_only` | registered reference | gap | winning arm (of 9 cells) |
|---|---|---|---|---|
| EEGNet | 0.225 | **0.298** | +0.073 | real_only 5, class_weighted 2, classical_aug 2 |
| LCT | 0.312 | **0.413** | +0.101 | class_weighted 3, real_only 3, classical_aug 3 |
| TCN | 0.293 | **0.428** | +0.135 | **class_weighted 6**, real_only 2, classical_aug 1 |

No single baseline dominates, so this is genuinely "best of three per cell" and not one
baseline in disguise.

---

## Q1 — Does synthetic beat the best simple baseline, or only `real_only`?

**Only `real_only`, and barely.** Δevent-F1 for the core gated arm:

| detector | vs `real_only` | vs **registered** | cells better |
|---|---|---|---|
| EEGNet | −0.007 | **−0.080** | 1 of 9 |
| LCT | +0.006 | **−0.095** | **0 of 9** |
| TCN | +0.008 | **−0.128** | **0 of 9** |

Ungated is no better (−0.063 / −0.096 / −0.161 vs registered). The strongest single statement:
**for LCT and TCN, not one of nine cells beat the registered reference under any gated arm.**

This confirms `the verification record` §5.1 at n = 9 per detector, on a rebuilt data pipeline and a
detector (LCT) that was absent from the original grid.

## Q2 — Does `random_gated` match `gated`?

**Yes. Admission adds nothing over dose.** At matched volume (identical `n_admitted`, verified
per cell), teacher-confidence selection and a uniform random draw are indistinguishable:

| detector | gated q0.90 | random_gated q0.90 | difference |
|---|---|---|---|
| EEGNet | −0.080 | −0.074 | 0.006 |
| LCT | −0.095 | −0.086 | 0.009 |
| TCN | −0.128 | −0.124 | 0.004 |

All three gaps are an order of magnitude below the between-cell spread. This is the control the
parent method ran and left unresolved; here it resolves **against** admission quality.

## Q3 — Does LCT behave like TCN (capacity) or EEGNet (architecture)?

**Neither — the heterogeneity itself is gone.** All three detectors now behave identically:
≈ 0 against `real_only`, −0.046 to −0.153 against the registered reference. LCT (121 k params,
attention) patterns with TCN (129 k, dilated conv) *and* with EEGNet (1.9 k) simultaneously.

The old "TCN benefits, EEGNet is neutral" contrast — which motivated the capacity-vs-architecture
question — **was an artifact of measuring against the weakest baseline.** There is no benefit
whose heterogeneity needs explaining. The three-family design earned its keep by showing that.

## Q4 — Does the tail-control finding replicate?

**Yes, and more strongly than before.** Pooled over all 81 gated-family cells, split by the
gate's own admit/revert decision (Δ = augmented model − `real_only`):

| | n | mean ΔFP/24h | median | worst | mean Δevent-F1 |
|---|---|---|---|---|---|
| **admitted** | 21 | **−22.58** | −26.34 | **+15.28** | +0.031 |
| **reverted** | 60 | **+10.90** | +13.83 | **+62.63** | −0.025 |

- ΔFP/24h separation: Mann–Whitney **p < 0.0001**, within-fold permutation **p < 0.0001** (two-sided)
- Tail event ΔFP/24h > +20: **0 of 21 admitted, 23 of 60 reverted**, Fisher **p = 0.0004**
- Δevent-F1 separation: **p = 0.092** — weak, as before

The asymmetry is the finding, now at n = 81 versus n = 36 in the original grid: **the gate is a
false-alarm tail controller and not an event-F1 selector.** This is the one claim from
`the verification record` §3.3 that survives Phase 1 untouched, and it is independent of the reference
choice — it compares admitted against reverted cells, never against a baseline.

---

## The fallback defect, and its fix, at full n

`_apply_fail_closed` reverts to the `real_only` teacher while `PREREGISTRATION.md` §3 defines
harm against the best simple baseline, so every revert gives away the baseline gap. Re-scoring
the *same admission decisions* with a registered-baseline fallback (no retraining — reverting is
a deterministic choice among arms already computed):

| detector | gated q0.90 as built | → best-baseline fallback | harm rate |
|---|---|---|---|
| EEGNet | −0.080 | **−0.016** | 0.44 → **0.11** |
| LCT | −0.095 | **−0.037** | 0.67 → **0.22** |
| TCN | −0.128 | **−0.007** | 0.78 → **0.11** |

For TCN the fixed policy also beats `real_only` by **+0.128 event-F1 in 7 of 9 cells**
(fold-level p = 0.021) — i.e. most of what the gate was supposed to deliver was being thrown
away at the revert step. Worst-case ΔFP/24h for TCN drops from +20.96 to **+8.08**.

**Defect 2 remains open:** `fail_closed_decision` still compares the augmented model against
`real_only`'s validation metrics, so it *admits* models that beat `real_only` but lose to the
registered baseline. That changes which cells admit, so it cannot be fixed in analysis — it is
the Phase 2 experiment.

---

## Statistics, stated honestly

**48 paired comparisons** are reported in this grid. Bonferroni at α = 0.05 requires
p < 0.0010. **Nothing meets it, on either test.** Under Nadeau–Bengio correction for fold
dependence, **zero comparisons reach p < 0.05**; the minimum is TCN gated q0.50 vs registered at
**p_NB = 0.054** (Wilcoxon would have said 0.016).

So the defensible claims are **direction and count**, not significance:

- 0 of 9 cells beat the registered reference for LCT and TCN under the core gated arm
- `random_gated` matches `gated` to within 0.009 on all three detectors
- the tail-control separation clears Bonferroni comfortably (Fisher p = 0.0004, permutation p < 0.0001)

The tail result is the only one in the grid strong enough to survive multiplicity correction.

---

## Cross-run comparability — do not compare `_v2` against the old CSV cell-by-cell

The rebuilt pipeline does not reproduce the old run's per-cell numbers (tcn f0 s42 `real_only`:
old 0.310, new 0.245) despite byte-identical splits and matching `n_train_pos`. The cause is
environment — different GPU, different torch/cuDNN, no deterministic-algorithm flags — not data.
A second hazard was measured directly: raising `--num-workers` from 0 to 8 is 2.16× faster and
changes 6 of 7 conditions (`real_only` 0.0938 → 0.1605). **Per-cell results in this codebase are
sensitive to execution environment, not just to seed.** Every comparison in this report is
within-run and therefore unaffected.

---

## What this means for the paper

The benefit claim is gone, and it was never there: it was an artifact of the reference. What
remains is stronger and cleaner than a contested effect size —

1. **A pre-registered negative result with a controlled explanation.** Synthetic ictal
   augmentation does not beat a one-line class-weighted loss on patient-independent seizure
   detection, on three detector families, at n = 9 each, with a matched-volume control showing
   admission quality contributes nothing.
2. **A mechanistic finding that survives everything.** The gate governs the false-alarm tail
   (0 of 21 admitted cells above +20 FP/24h vs 23 of 60 blocked; Fisher p = 0.0004) and does not
   select for event-F1 (p = 0.092).
3. **A concrete governance lesson.** A fail-closed gate must fall back to the *best available*
   model, not the weakest. Fixing that alone moves TCN from −0.128 to −0.007 against the
   registered reference and cuts its harm rate from 0.78 to 0.11.

The generator was audited and is not the culprit: no mode collapse (diversity ratio 0.911 of
real), no memorisation (NN ratio 1.156), correct normalised space, band-limiting effective
(out-of-band power 0.0093 → 0.0005). See `the execution log`, "DEBUG PASS".
