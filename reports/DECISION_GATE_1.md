# Decision Gate 1 — Phase 1 results

**Date:** 2026-08-29 · **Grid:** `downstream_gated_v2.csv`, 27 complete blocks
(3 detectors × 3 folds × 3 seeds × 7 conditions = 189 rows) · **Run:** 3 Vertex AI Spot A100
jobs, 13.1 h wall-clock, no preemptions.

Reproduce: `python3 scripts/analyze_multiseed.py --tag _v2`

**Stop point.** Report here; do not auto-continue to Phase 2.

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

This confirms the earlier scarcity-axis result at n = 9 per detector, on a rebuilt data pipeline
and a detector (LCT) that was absent from the original grid.

## Q2 — Does `random_gated` match `gated`?

> **RETRACTED — see CORRECTION 2 at the end of this file.** The table below is scored *after*
> the fail-closed revert, so 18 of 27 cells are bit-identical because both arms reverted to the
> same `real_only` model, not because admission did nothing. On the augmented models the gaps
> are 2.6–4.1× larger, the FP/24h axis was never tested, and the control was run only at the q
> where the gate injects ~3% of the intended dose. **Do not quote the 0.009.**

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
false-alarm tail controller and not an event-F1 selector.** This is the one pre-Phase-1 claim
that survives Phase 1 untouched, and it is independent of the reference
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
- ~~`random_gated` matches `gated` to within 0.009 on all three detectors~~ **RETRACTED, see
  CORRECTION 2** — that 0.009 is an artifact of scoring after the revert
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
   detection, on three detector families, at n = 9 each. ~~with a matched-volume control showing
   admission quality contributes nothing~~ — **that half is withdrawn by CORRECTION 2**; the
   matched-volume control was run at a ~3% dose and cannot support a claim about admission
   quality in either direction.
2. **A mechanistic finding that survives everything.** The gate governs the false-alarm tail
   (0 of 21 admitted cells above +20 FP/24h vs 23 of 60 blocked; Fisher p = 0.0004) and does not
   select for event-F1 (p = 0.092).
3. **A concrete governance lesson.** A fail-closed gate must fall back to the *best available*
   model, not the weakest. Fixing that alone moves TCN from −0.128 to −0.007 against the
   registered reference and cuts its harm rate from 0.78 to 0.11.

The generator was audited and is not the culprit: no mode collapse (diversity ratio 0.911 of
real), no memorisation (NN ratio 1.156), correct normalised space, band-limiting effective
(out-of-band power 0.0093 → 0.0005).

---

# CORRECTION (appended 2026-08-29, same day)

**The Q1 headline above overstates the negative. The registered best-of-3 reference is inflated
by selection bias, and this analysis selected it on TEST performance.**

Mean event-F1 per arm (n = 9 cells):

| detector | real_only | class_weighted | classical_aug | **best-of-3** |
|---|---|---|---|---|
| eegnet | 0.225 | 0.205 | 0.197 | **0.298** |
| lct | 0.312 | 0.299 | 0.298 | **0.413** |
| tcn | 0.293 | **0.364** | 0.225 | **0.428** |

For eegnet and lct the best-of-3 reference exceeds **every individual baseline** by 0.07-0.10.
No baseline is that good; that gap is the max-of-three-noisy-estimates bias. It is the same
selected-maximum defect identified in the gate's own fail-closed selector, reproduced here in
the analysis that was auditing it.

**Against single pre-specified baselines** (gated q0.90, Δevent-F1, cells better of 9):

| detector | vs real_only | vs class_weighted | vs classical_aug | vs best-of-3 |
|---|---|---|---|---|
| eegnet | −0.007 (1/9) | **+0.013 (5/9)** | **+0.021 (6/9)** | −0.080 (1/9) |
| lct | +0.006 (1/9) | **+0.019 (4/9)** | **+0.020 (5/9)** | −0.095 (0/9) |
| tcn | +0.008 (1/9) | **−0.063 (2/9)** | **+0.076 (6/9)** | −0.128 (0/9) |

### Corrected Q1

**Synthetic ictal augmentation is at parity with the simple baselines** — it neither helps nor
meaningfully harms. The largest single loss is **TCN vs `class_weighted` (−0.063, 2 of 9)**, where
`class_weighted` really does reach 0.364 against `real_only`'s 0.293. The "0 of 9 cells better"
statement holds only against the biased best-of-3 reference and must not be quoted without it.

> **Amended.** This was written as "the one genuine loss". Drop that phrasing: cross-family
> comparisons are not initialisation-controlled (unseeded-init defect; `README.md`
> Known issue #1) and the run-to-run floor at 80 epochs was never measured, so −0.063 cannot be
> separated from noise. The direction is defensible; the magnitude is not.

### What is unaffected

**Q2** (`random_gated` matches `gated` within 0.009) and **Q4** (tail control, Fisher p = 0.0004)
compare arms against each other, never against a baseline, so neither depends on the reference
choice. Both stand exactly as reported. Q4 remains the only result surviving multiplicity
correction.

> **Amended by CORRECTION 2.** Correct for Q4, and correct for Q2 *as far as the reference
> choice goes* — but Q2 does not stand. It fails for an unrelated reason (it is scored after the
> fail-closed revert, tested on the wrong metric, and run at a ~3% dose). Only Q4 is unaffected.

Q3's conclusion also stands but for a simpler reason than stated above: with all detectors at
parity there is still no benefit whose heterogeneity needs explaining.

### Consequence for the pre-registration — declare this

`PREREGISTRATION.md` §3 registers the harm reference as "the best simple baseline per
(fold, seed)". That is a **selected maximum**, and it is optimistically biased by ~0.07-0.10
event-F1 in this grid. Two fixes, both declarable:

1. Choose the reference arm on **validation**, report its **test** metrics — never select on the
   metric being reported. This analysis violated that and must be re-run that way; the CSV does
   not currently carry per-arm validation metrics, so it needs a small emit change.
2. Report against **each baseline separately** as the primary analysis, with best-of-3 as a
   clearly-labelled secondary.

This is a genuine methodological finding in its own right: **baseline selection swings the
apparent effect by more than the effect itself** (±0.13 vs an effect of ~0.01-0.06 here).

---

# CORRECTION 2 (appended 2026-08-29, same day)

**The Q2 null is confounded and underpowered. "Admission adds nothing over dose" is not
supported by this grid, and CORRECTION 1's claim that Q2 is unaffected was right about the
reference choice and wrong about everything else.** Reproduce with
`python3 scripts/analyze_multiseed.py --tag _v2`, section **(e)**; machine-readable in
`analysis_v2_admission.csv`.

## The control itself is sound — this is not a bug

Audited first, because a broken control would be the cheaper explanation:

- **Identical candidate pool.** `experiments/training.py` draws it as
  `synthetic_provider(6 * n_synth, spec.seed)` and both arms pass the same `spec.seed`, so the
  two arms score the *same* generated windows.
- **Identical dose.** `gate_n_admitted` matches between the arms in **27 of 27 cells**
  (`n_dose_mismatch = 0` in the new output). The volume really is matched.
- **The random draw is not secretly the teacher's draw.** Expected overlap equals the admission
  rate, which peaks at **4.0%**; the two admitted sets are effectively disjoint.
- **The draws are independent across detectors** despite sharing `selection_seed=seed`
  (numpy's `choice(replace=False)` at different `size` does not produce nested prefixes —
  checked directly: 0% and 4% overlap, both at chance).
- **The pairing is init-controlled.** Per `README.md` "Known issues" #1, weight initialisation is
  unseeded and the condition perturbs the RNG stream, which breaks pairing for deltas against
  `real_only` — but the gated-family arms draw *identical* pools and so advance the stream
  identically. `gated` vs `random_gated` is therefore one of the few genuinely paired
  comparisons in this grid.

The arms are comparable — pool-matched, dose-matched, seed-matched and init-matched, differing
**only** in which windows were admitted. The defects are in **how the comparison was scored** and
**where it was run**.

## Defect 1 — scored after the fail-closed revert, so most cells are forced ties

Q2 paired the arms on the post-revert `event_f1`. When both arms fail closed they revert to the
**same** `real_only` model, so those cells are bit-identical for reasons that have nothing to do
with admission. Of 27 cells, **19 tie and 18 of those are forced by a shared revert**
(EEGNet 7, LCT 5, TCN 6). Two-thirds of the sample is pinned to zero before any evidence enters.

Scored on the augmented model instead (`aug_*`, pre-revert — the arms as actually trained):

| detector | Q2 as reported (post-revert) | valid contrast (pre-revert) | factor |
|---|---|---|---|
| EEGNet | 0.006 | **0.023** | 4.1× |
| LCT | 0.009 | **0.024** | 2.6× |
| TCN | 0.004 | **0.016** | 4.0× |

Per cell, pooled over the 26 cells that received any synthetic at all, mean \|difference\| is
**0.059 event-F1**, range −0.252 to +0.160. The two selections do not produce the same models.
The mean is near zero because the differences are large and **sign-inconsistent** — a much
weaker statement than "indistinguishable within 0.009".

## Defect 2 — tested only on event-F1, the axis Q4 says the gate does not act on

Q4 concludes the gate is "a false-alarm tail controller and **not** an event-F1 selector". Q2
then tested admission quality on event-F1 alone. On FP/24h (pre-revert, cells with
`n_admitted > 0`; **negative = teacher selection better**):

| detector | Δ event-F1 (teacher − random) | Δ FP/24h (teacher − random) | mean \|Δ\| FP/24h |
|---|---|---|---|
| EEGNet | −0.026 (p_wil 0.047) | **+5.00** (p_wil 0.008) | 5.00 |
| LCT | +0.024 (p_wil 0.734) | **−16.32** (p_wil 0.250) | 28.01 |
| TCN | −0.016 (p_wil 1.000) | **−9.91** (p_wil 0.129) | 14.30 |

Pooled mean \|Δ\| is **16.2 FP/24h**. The sign is inconsistent — random is better for EEGNet,
teacher selection is better for LCT and TCN — so this is *not* a win for admission. But it is
not "nothing" either, and the original analysis never looked at this axis.

**(e) is its own comparison family: 12 paired tests, Bonferroni threshold p < 0.0042, and
nothing meets it** (EEGNet FP/24h at p_wil 0.008 is closest and fails). No significance is
claimed here in either direction.

## Defect 3 — run only at the q where the dose is a rounding error

The gate admits a small fraction of the volume it was asked to inject (`n_synth` = the real
ictal count ≈ 2508 windows per cell):

| detector | median admitted | share of intended dose |
|---|---|---|
| EEGNet | 8 | **0.3%** |
| TCN | 79 | **3.1%** |
| LCT | 328 | **13.1%** |

EEGNet cells inject 0, 2, 2, 6, 8, 14, 19, 47 and 115 windows. **No selection rule can be
distinguished from another at a dose of 2 windows in 2508.** And `random_gated` was run *only*
at q = 0.90, the arm with the smallest dose. At **q = 0.50 the gate admits the full dose**
(median 2508 — `max_keep` binds, so the teacher is making a real top-17% selection of the pool),
and there is **no matched-volume control there at all**. The control was run in the one
configuration where it could not detect anything.

## Corrected Q2

**The grid does not license "admission adds nothing over dose."** What it licenses is narrower
and more mechanical:

> At q = 0.90 the admission stage is **inert because it admits ~3% of the intended dose**. That
> is a fact about threshold calibration — the real-ictal reference and the teacher's saturation
> on real ictal, already flagged in `synthetic/trust_gate.py` and
> `README.md` "Known issues" #2 — not a finding about whether admission quality can matter.

Known issue #2 is the direct mechanism here: the threshold is calibrated on the teacher's *own
training positives*, which it has memorised, so the reference distribution is pinned near 1.0 and
almost nothing in the synthetic pool clears it. The tiny dose is not incidental to the null — it
*is* the null. Switching `TrustGateConfig.reference` to `"pool"` (the published TGA rank cut,
already implemented) is the other way to give the control something to measure.

A separate *a priori* point survives regardless of the data: admitting the **highest**
teacher-confidence windows selects exactly the examples the teacher already classifies
correctly — the lowest-gradient, most redundant windows in the pool. A confidence-max rule is
expected to add little. Even a fully powered version of this test would be evaluating one
poorly-motivated admission rule, not the concept of admission.

## What must be run

`random_gated` at **q = 0.50**, where the dose is full. The driver now runs a matched-volume
control for **every** q in `--qs` (new `--random-qs`, defaulting to `--qs`), making the Phase 1
grid 8 conditions per block instead of 7. **Staged, not run — it needs a GPU and the data
store, neither of which is currently available.**

**It requires re-running whole blocks, not just the new arm.** The control is matched only if it
shares the teacher that fixed its admitted count, so bolting a new `random_gated q0.50` row onto
the existing `_v2` rows — trained by a different teacher in a different environment — would
produce an unmatched control, the same class of defect this correction is about. `n_conds()`
therefore counts 8, and legacy 7-row blocks are correctly *not* treated as complete.

## Bookkeeping

The frontier gained `admit_always q0.50 pool` (the teacher-side counterpart that was missing),
so the primary family is **54 paired tests, not 48**, and the Bonferroni threshold moves from
p < 0.0010 to **p < 0.0009**. **Nothing meets it on either test, exactly as before** — no
conclusion in this report changes. `admit_always random pool` is now labelled
`admit_always random pool q0.90`, since q is no longer implicit.

## What still stands

**Q1, Q3 and Q4 are untouched.** Q4 in particular — the tail-control result — compares admitted
against reverted cells within the same arm and never uses `random_gated`.

### Q4 also survives the circularity objection (new, and it strengthens the claim)

With Q1 at parity, Q2 unresolved and Q3 vacuous, **Q4 now carries the paper**, so it was worth
attacking directly. The obvious reviewer objection: the gate admits partly on *validation*
FP/24h, so of course admitted cells show better *test* FP/24h — selection on the outcome.

The reason breakdown refutes it. Of 60 reverts, only **19** fired on
`val_fp24h_exceeds_safety_slack`; **39** fired on `val_event_f1_below_margin` and 2 on
`no_synthetic_admitted`. Dropping every cell selected on validation FP/24h and re-testing on the
remainder — cells rejected on a **different metric** — the separation not only holds, it is
stronger than the pooled version:

| | n | mean ΔFP/24h | tail > +20 |
|---|---|---|---|
| admitted | 21 | **−22.58** | **0 / 21** |
| reverted on val event-F1 | 41 | +10.45 | 16 / 41 |

Mann–Whitney **p = 0.000046**, within-fold permutation **p < 0.0001**, Fisher on the tail
**p = 0.000494**. Both clear the primary family's Bonferroni threshold (p < 0.0009).

So the gate's **event-F1 admission criterion predicts test false-alarm inflation** — a
cross-metric, cross-split prediction that selection-on-the-outcome cannot produce. This is the
form of Q4 to quote when challenged, and section (d) of `analyze_multiseed.py` now computes it
automatically.

The claim to retire everywhere it appears: *`random_gated` matches `gated` to within 0.009,
therefore admission adds nothing.* It appears in the Q2 section above, in the "defensible
claims" list, in CORRECTION 1's "What is unaffected" and in `README.md`; all have been marked.


---

# CORRECTION 3 (appended 2026-08-31, from the full project audit)

**"the run-to-run floor at 80 epochs was never measured" is no longer true, and the amendment
above is now too weak.** The floor has been measured, and it is larger than the effect this report
declines to claim.

Phase 2's `_p2` grid re-ran all three simple baselines for TCN at identical settings with the
initialisation fix in place (`c32705e`). The only behavioural change between `_v2` and `_p2` is
that one `torch.manual_seed` line, so the paired difference between the two grids **is** the
run-to-run floor for the affected arms:

| baseline (TCN, 9 cells) | `_v2` (this report) | `_p2` re-run | σ of paired difference |
|---|---|---|---|
| `real_only` | 0.293 | 0.283 | 0.099 |
| `class_weighted` | **0.364** | **0.271** | **0.159** |
| `classical_aug` | 0.225 | 0.255 | 0.171 |
| `ungated` | 0.268 | 0.268 | 0.000 — bit-identical in 9/9 |

**−0.063 is therefore withdrawn as a magnitude.** Against the re-run baselines the same gated arm
scores **+0.030 (3/9)**, and the best-of-3 gap moves from −0.128 (0/9) to −0.059 (2/9). Neither
reading is significant on any correction, so **this report's conclusion — parity — is unchanged**;
what changes is that the sentence "`class_weighted` really does reach 0.364" describes one draw,
not a measurement.

The `ungated` row also explains why only some arms were affected: `WGANGPProvider.generate` seeds
torch and then samples on the GPU, leaving the CPU stream that `build_model` draws from at exactly
`manual_seed(seed)`, so every pool-drawing arm was already seeded by accident. Within-family
contrasts in this grid were init-controlled all along; only comparisons against the three pool-free
baselines were not.

Full working: the 2026-08-31 audit, §2.
