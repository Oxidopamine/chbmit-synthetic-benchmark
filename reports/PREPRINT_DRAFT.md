# Measure what was injected, not what was requested: a leakage-safe replication of trust-gated synthetic EEG augmentation on CHB-MIT

**Abdullah R. Alotaibi**

**Status:** DRAFT v0.4, 2026-08-31. All sections are drafted and all three figures are rendered
(`reports/figures/`). Remaining **[TODO]** markers are bibliographic details requiring source
verification. Every number in Results is reproducible from this repository; provenance for each is
given in §8.

> **v0.4 changes, from a full project audit.** (1) §4.1's −0.063 is withdrawn as a
> measurement. Re-running the three simple baselines under the Phase 2 initialisation fix moves
> `class_weighted` by 0.093 and flips the comparison to +0.030; the run-to-run floor
> (σ = 0.099–0.171) is now reported instead, and added to §4.6 as a fourth pitfall. The
> abstract is rewritten accordingly. (2) Deviations 3 and 6 were overstated: the registered 5-fold
> and 3-scarcity design *was* run at Tier B, the negative result holds at every rung, and §7 now
> reports it. (3) The +0.163 in deviation 1 is re-attributed to the grid it came from. (4) The
> reproduction command for §4.2–§4.3 in §8 was broken and is fixed. (5) Figures rendered.
>
> **v0.2 changes, from an internal review.** (1) The claim that admission quality yields *equal
> deployments* was an equivalence conclusion drawn from a non-significant difference — the exact
> inference this paper criticises in §4.6. It is restated in §4.4 with a TOST and a confidence
> interval, and weakened accordingly. (2) The title no longer asserts the published method is
> unevaluable; the divergence that disabled the gate was **ours**, and §4.2 now says so and answers
> the obvious objection. (3) §4.5 now states plainly that the tail-control result **replicates the
> source method's own central claim** rather than presenting it as a discovery.

---

## Abstract

Generative augmentation of rare ictal EEG is widely reported to improve seizure detection, and
fail-closed "trust gating" has been proposed to make it safe under subject shift. We replicate
trust-gated augmentation on CHB-MIT under patient-independent splits, scored at the event level
with SzCORE conventions and pre-registered harm.

Augmentation is at parity with simple baselines, beating none of them on any detector by a margin
this design can establish. The comparison is bounded from below rather than from above: two
independent runs of the *identical* baseline specification, differing only in weight-initialisation
seeding, differ by 0.099–0.171 event-F1 (σ over 9 cells), so a run-to-run floor larger than any
effect reported in this literature sits under every cross-family number here — ours included. This
parity is not a dose artefact: across the source method's operating band (r ∈ [0.05, 0.30]) the
dose–performance curve is monotone with no interior optimum, and it holds at every scarcity level
of the registered grid.

More consequentially, the gate as we first built it *cannot inject a dose at all*. Calibrating
admission on real ictal windows — which the teacher has memorised — admits 6 windows against a
target of 251 and 23 against 752, whatever is requested, and the source method's K_min = 200
safeguard is met in 0 of 9 cells. The divergence was ours, documented in advance as a design
choice, and invisible in every pipeline output. The published pool rank cut restores exact dose
control.

At a real dose, teacher admission yields better *models* than a matched random draw (+0.072
event-F1, 8 of 9 cells), but this fails fold-dependence correction and is undetectable after the
fail-closed stage — which we decline to call equivalence, the study bounding it only at ±0.121.
Tail control replicates, surviving multiplicity correction, a circularity objection and dose
stratification, and holding even for *randomly selected* synthetic.

We quantify five evaluation pitfalls that changed our own conclusions, including that the
Nadeau–Bengio variance floor is set by fold count — **additional seeds cannot buy fold-corrected
power**, and at three folds no number of replicates resolves effects below ≈0.13 event-F1 — and
that an unmeasured weight-initialisation floor can exceed the effect being reported.

**Keywords:** seizure detection, synthetic data, generative augmentation, patient-independent
validation, negative results, evaluation methodology

---

## 1. Introduction

Seizures are rare events. In our processed CHB-MIT corpus, ictal windows are 0.319 % of the data
(5,563 of 1,746,447). That imbalance makes generative augmentation of the positive class an
attractive proposition, and a substantial literature reports gains from it.

Two properties make most of those reports difficult to act on clinically.

**Window-level metrics conceal event-level harm.** A model can improve balanced accuracy or AUROC
while inflating false alarms per day — the quantity that determines whether a detector is
deployable at all. We score every arm with `timescoring` under SzCORE conventions [6] and report
FP/24 h alongside event-F1 throughout.

**Patient-independent validation is rare.** Splitting windows rather than patients leaks subject
identity and inflates every number. Here patient groups are partitioned *before* windowing,
generators are fit on training patients only, and provenance is enforced in code and tested.

Against this background, Choi et al. [1] propose trust-gated augmentation: score candidate synthetic
windows with a teacher detector, admit only those clearing a confidence threshold, train an
augmented model, and *fail closed* — revert to the real-only model unless the augmented one
improves a validation criterion. The framing is explicitly one of governance rather than accuracy.

This paper asks three questions. Does synthetic augmentation help, measured honestly against
simple baselines? Does the gate's *admission* rule do anything a random draw of the same size
would not? And what, mechanically, does the gate control?

Our contribution is not a better detector. It is (i) a leakage-safe, pre-registered,
event-level replication that answers those questions, (ii) the finding that a plausible-looking
divergence in how the admission threshold is calibrated silently disables the entire mechanism,
and (iii) a set of quantified evaluation pitfalls, three of which invalidated our own earlier
conclusions before we caught them.

## 2. Relationship to prior work

The fail-closed trust gate is not our invention. It is adapted from [1], whose preprint [2] we
also consulted for implementation detail:

> Choi, D.; Yip, C.; Choi, A.; Park, J. (2026). *Trust-gated synthetic EEG augmentation reduces
> performance drops when generalizing to new patients.* npj Digital Medicine 9(1), art. 634.
> doi:10.1038/s41746-026-02778-0, PMID 42185473.

TGA contains no seizure content; our contribution includes the seizure-specific, event-level
reformulation of the admission and fail-closed criteria. TGA also runs a matched-volume
random-gating control and reports it as mixed — in the published numbers, two of three
comparisons favour *random* gating, while the paper concludes that curation adds safety. Resolving
that control at a defensible dose is one of our aims.

### 2.1 Generative augmentation for seizure detection

The closest prior work is GP-EEG [3], which evaluates four generators as augmentation on CHB-MIT
and Siena under leave-one-patient-out splits with EEGNet. Its reported deltas against baseline are
instructive:

| method | CHB-MIT ΔF1 | ΔRecall | Siena ΔF1 |
|---|---|---|---|
| COSCI-GAN | −10.52 | −17.23 | −0.54 |
| TimeVAE | −14.23 | −22.31 | −12.70 |
| ImagenTime | −3.70 | −8.16 | −0.27 |
| GP-EEG | +2.75 | +2.26 | +5.11 |

Two things follow. First, **harm from synthetic ictal augmentation on CHB-MIT is already
published**, with the signature we also observe — precision up, recall down hard. Our negative
result is therefore not novel in direction, and we do not claim it as such. What we add is
*characterisation*: pre-registered harm thresholds, tail-risk reporting, and an explicit
false-alarm axis.

Second, the gap GP-EEG leaves is precisely the set we fill. It reports **no non-generative
baselines** — no class weighting, no oversampling, no classical augmentation; **sample-level
metrics only**, on 1024-sample segments, with no event-level scoring; and **no FP/day**, hence no
operating-regime comparison and no tail-risk statement. Its own +2.75 F1 headline is a
sample-level number with no class-weighting comparison. Our §4.1 result bears directly on how such
a number should be read: for TCN the *same* gated arm scores +0.008 against `real_only`, −0.063
against `class_weighted` and +0.076 against `classical_aug` — a **0.139 span** attributable purely
to which baseline is chosen, larger than any effect under discussion in this literature. §4.1 adds
a second span of the same order from re-running one baseline under a different initialisation seed.
An augmentation delta reported against a single unnamed baseline, with no floor measured, is
consistent with almost any underlying truth.

Because GP-EEG established CHB-MIT and Siena as the expected pairing, evaluation on Siena is a
comparability requirement rather than an extension. We have not met it (§6).

### 2.2 Baselines, dose, and selection

Two literatures frame our methodological results. *Stronger Baseline Models* [5] names class
balancing via inverse-frequency weighting as a required clinical-ML baseline and documents cases
where simple models match or beat complex ones; our finding that a one-line class-weighted loss
beats the full generative pipeline is an instance of a documented, citable pattern rather than an
idiosyncrasy of this corpus. Related work on how proposed-method win rates fall as baseline counts
rise [7] makes the same point from the other direction.

On dose, Shidani et al. [4] give a stability-based generalisation bound whose risk term is
**U-shaped in the synthetic mixing ratio**: too little synthetic leaves variance high, too much
lets distributional mismatch dominate, and for fixed W₂(real, synthetic) an optimal mixing
parameter exists. This is why a single uncontrolled injection ratio cannot support a negative
result, and why §4.3 samples the band rather than a point.

Finally, the observation that random selection is a strong baseline against learned selection is
established in the coreset and data-pruning literature [8]. Our §4.4 is a seizure-specific,
event-level instance of that comparison, and we read our result in that light rather than as a
surprise.

## 3. Methods

### 3.1 Data

CHB-MIT Scalp EEG [9] (PhysioNet), retrieved from the AWS Open Data mirror.

| property | value |
|---|---|
| patients / groups | 24 / 23 (chb21 is chb01 re-recorded, grouped with it) |
| recordings | 686 EDF files, 676 in manifest, 673 retained after channel audit |
| duration | 3,505,100 s = 973.6 h |
| seizure events | 198 annotated; 185 survive windowing |
| montage | 18 bipolar channels, common to all retained files |
| sampling rate | 256 Hz; band-pass 0.5–40 Hz, zero-phase Butterworth (order 4) |
| windows | 4 s at 2 s stride → 1,746,447 windows, 5,563 ictal (0.319 %) |
| normalisation | per-window, per-channel z-score |

Files whose montage cannot supply the 18-channel set are dropped by an explicit audit, which also
reports the fraction of seizure events lost (6.6 %). Peri-ictal background within 60 s of a
seizure is excluded from training negatives.

### 3.2 Splits, detectors, generator

Patient groups are partitioned before windowing; test groups are disjoint across folds. Three
folds were run (train/val/test groups 14/5/4, 13/6/4, 14/4/5), three seeds each, giving n = 9
paired cells per condition per detector.

Three detector families spanning two orders of magnitude in capacity: **EEGNet** [11] (1,905
parameters), **LCT** (120,834), **TCN** (129,441).

The generator is a WGAN-GP [12] fit per (fold, seed) on that fold's training ictal windows only, with
outputs band-limited to the acquisition passband before injection. Band-limiting reduces
out-of-band power from 0.0093 to 0.0005.

### 3.3 The gate

**Admission.** Score each candidate synthetic window with the real-only teacher detector's
seizure probability; admit those at or above the q-quantile of a reference distribution, capped at
the target injection count. The reference is either the *real ictal training windows*
(`real_ictal`) or the *candidate pool itself* (`pool`, the rank cut TGA publishes). §4.2 shows
this choice is decisive.

**Fail-closed selection.** Compare the augmented model's validation event-F1 and FP/24 h against
the teacher's; deploy the augmented model only if it improves event-F1 without inflating FP/24 h
beyond a slack of 0.25. Otherwise revert to the real-only model.

We emphasise what the gate does *not* do: nothing in this implementation computes a covariance-
manifold distance. The admission criterion is teacher confidence and nothing else.

### 3.4 Metrics, harm, and statistics

Event-level scoring uses `timescoring` under SzCORE conventions [6] (toleranceStart 30 s,
toleranceEnd 60 s, minOverlap 0, maxEventDuration 300 s, minDurationBetweenEvents 90 s).

**Harm** is fixed before test scoring as a paired delta with Δevent-F1 < −0.01 **or**
ΔFP/24 h > +0.25, reported with harm rate, worst-cell delta, and CVaR at α = 0.10.

**Statistics.** We report the Wilcoxon signed-rank test conventional in this literature *and* the
corrections it requires, because nine cells sharing three splits are not nine independent
observations: the Nadeau–Bengio corrected resampled t-test [10], a fold-level t-test on seed-averaged
deltas, and an explicit comparison count so Bonferroni is visible rather than implied. Where
Wilcoxon and Nadeau–Bengio disagree, we report the corrected result.

## 4. Results

### 4.1 Synthetic augmentation does not beat a simple baseline

Mean test event-F1, n = 9 cells per detector (Phase 1):

| detector | real_only | class_weighted | classical_aug | ungated | gated q0.90 |
|---|---|---|---|---|---|
| EEGNet | 0.225 | 0.205 | 0.197 | 0.234 | 0.218 |
| LCT | 0.312 | 0.299 | 0.298 | 0.317 | 0.318 |
| TCN | 0.293 | **0.364** | 0.225 | 0.268 | 0.301 |

Against single pre-specified baselines (gated q0.90, Δevent-F1, cells better of 9):

| detector | vs real_only | vs class_weighted | vs classical_aug |
|---|---|---|---|
| EEGNet | −0.007 (1/9) | +0.013 (5/9) | +0.021 (6/9) |
| LCT | +0.006 (1/9) | +0.019 (4/9) | +0.020 (5/9) |
| TCN | +0.008 (1/9) | **−0.063 (2/9)** | +0.076 (6/9) |

Synthetic augmentation is at **parity** with the simple baselines. The largest single apparent loss
is TCN against `class_weighted`, where a focal loss with a positive-class weight — two lines,
requiring no generator — reaches 0.364 event-F1 at 9.8 FP/24 h versus real-only's 0.293 at 16.9,
better on both axes.

**We now know that −0.063 is not a measurement.** The Phase 1 grid ran before the
initialisation fix (§6, item 6), and the arms that draw no synthetic pool — precisely the three
simple baselines — reached model construction with an unseeded RNG stream. Phase 2 re-ran all three
for TCN under identical settings with the fix in place, giving a rare direct read on the run-to-run
floor:

| baseline (TCN, 9 cells) | Phase 1 (unseeded) | Phase 2 re-run (seeded) | σ of the paired difference |
|---|---|---|---|
| `real_only` | 0.293 | 0.283 | 0.099 |
| `class_weighted` | **0.364** | **0.271** | **0.159** |
| `classical_aug` | 0.225 | 0.255 | 0.171 |
| `ungated` (draws a pool) | 0.268 | 0.268 | **0.000** — identical in 9/9 cells |

Scoring the same gated arm against the re-run baselines moves TCN's headline comparison from
−0.063 (2/9 cells) to **+0.030 (3/9)**, and the registered-reference gap from −0.128 (0/9) to
−0.059 (2/9). Neither reading is significant on any correction, so the conclusion — parity — is
unchanged. What the two readings establish jointly is the bound: **two draws of the identical
baseline differ by 0.093, which brackets the 0.063 the first reading appeared to show.** No
cross-family effect of that size can be claimed from a design of this width, by us or by the
literature we are comparing against.

The `ungated` row is why the floor is measurable at all. Arms that draw a synthetic pool call
`WGANGPProvider.generate`, which seeds `torch` and then samples on the GPU, leaving the *CPU*
generator — the one model construction draws from — at exactly `manual_seed(seed)`. Those arms were
therefore already seeded by accident, and the Phase 2 fix is a bit-exact no-op for them. Only the
pool-free arms were exposed. The consequence is bounded and specific: **within-family comparisons
(gated vs random_gated vs ungated) were always initialisation-controlled; comparisons against the
simple baselines were not, in Phase 1 only.**

### 4.2 The gate cannot inject a dose

Under `reference="real_ictal"`, the admitted count is decoupled from the requested count:

| target r | target windows | admitted | realized r |
|---|---|---|---|
| 0.10 | 251 | **6** | 0.0024 |
| 0.30 | 752 | **23** | 0.0092 |

Tripling the request moved the admitted count from 6 windows to 23; the admission rate is
0.4–0.5 % of the pool regardless of target (**Figure 1**). The cause is that the threshold is a quantile of the
teacher's confidence on **real ictal training windows, which the teacher has memorised**, so it
sits near 1.0 and almost no generated window clears it.

The published alternative behaves entirely differently. With `reference="pool"`, admitted =
min(oversample·(1−q), 1) · n_synth **exactly**, so q = 1 − r/oversample targets any realized ratio.
Confirmed live: q = 0.9917 → 126 admitted (predicted 125); q = 0.9500 → 755 (predicted 752),
matched cell-for-cell between teacher and random selection in 9 of 9 cells.

Compounding this, TGA publishes a minimum acceptance of K_min = 200; we used 1. Under the
real-ictal reference **0 of 9 cells** reach 200 admitted windows, so the source method's own
safeguard would have rejected every cell.

This is the paper's central methodological finding: a divergence that looked like a
reparameterisation in the design table **disabled the mechanism**, and it did so silently, because
the gate continued to produce plausible outputs.

**We should meet the obvious objection directly: this was our divergence, not the source
method's.** A reader may reasonably respond that we misimplemented TGA, found our
misimplementation did not work, and wrote a paper about it. Three things distinguish this from
that reading.

First, the divergence was **documented in advance as a deliberate design choice**, not discovered
as a bug: calibrating on real ictal windows is a defensible reading of "admit windows the teacher
finds as convincing as real seizures", and it appears in our implementation-fidelity table from
the outset. It is the kind of substitution practitioners make routinely when adapting a method to
a new domain.

Second, **the failure was undetectable from any output the pipeline produced.** Admission rates,
event-F1, FP/24 h, harm rates and gate decisions were all well-formed and plausible for an entire
experimental phase. Nothing short of comparing the *realized* injected count against the
*requested* one exposes it — and no standard reporting template asks for that comparison.

Third, the consequence is not that our numbers were slightly off but that **an entire class of
question became unanswerable without our noticing**: the matched-volume control cannot separate
admission quality from dose when the dose is 6 windows, and we published a null from it before
catching the cause (§4.6, pitfall 2).

The transferable claim is therefore not "TGA does not work". It is that gated augmentation
pipelines have a failure mode in which a threshold-calibration choice silently reduces the
intervention to nothing, and that **reporting realized rather than requested quantities is the
cheap diagnostic that catches it.**

### 4.3 Dose is not the confound

A referee could object that our negative result was measured at an injection ratio far outside the
source method's operating range. It was: our first grids sampled only r ≈ 0.032 and r = 1.00, and
the source method's validation ladder selects r ≤ 0.30. Since the theoretical dose–performance
bound is U-shaped, this was a real confound.

Sampling the band directly (`ungated`, which receives the requested dose exactly; TCN, n = 9):

| arm | event-F1 | Δ vs real_only | cells better |
|---|---|---|---|
| real_only (r = 0) | 0.283 | — | — |
| ungated r = 0.10 | 0.264 | −0.019 | 5/9 |
| ungated r = 0.30 | 0.217 | −0.065 | 4/9 |
| ungated r = 1.00 | 0.268 | −0.015 | 5/9 |

Monotone through the source method's own band, **no interior optimum**, nothing significant
(**Figure 2**). The negative result does not depend on dose.

### 4.4 Admission quality: better models, equal deployments

With pool-relative admission the matched-volume control is well-posed: identical pool, identical
admitted count, identical initialisation, at doses of 126 and 755 windows.

On the augmented models, teacher selection is clearly better:

| dose | metric | teacher − random | cells | Wilcoxon | **Nadeau–Bengio** | fold-level |
|---|---|---|---|---|---|---|
| r ≈ 0.30 | event-F1 | **+0.072** | 8/9 | 0.020 | **0.140** | 0.103 |
| r ≈ 0.30 | FP/24 h | −22.3 | 6/9 | 0.098 | 0.394 | 0.242 |
| r ≈ 0.05 | event-F1 | +0.043 | 6/9 | 0.164 | 0.379 | 0.160 |
| r ≈ 0.05 | FP/24 h | −31.2 | 5/9 | 0.359 | 0.498 | 0.130 |

**The effect does not survive correction.** Bonferroni for this family (8 tests) requires
p < 0.0063; nothing meets it, and the headline row collapses under the fold-dependence correction.
We report direction and count; we do not claim significance.

The policy comparison is more interesting, and needs stating carefully:

| arm | pre-revert event-F1 | deployed event-F1 | reverted |
|---|---|---|---|
| gated q = 0.95 | **0.277** | 0.300 | **2/9** |
| random q = 0.95 | **0.205** | 0.307 | **6/9** |

Teacher admission builds better models (0.277 vs 0.205) and passes validation three times as
often (2/9 reverts vs 6/9). After the fail-closed stage, we **cannot detect a difference** between
the deployed policies: paired difference −0.007, naive 95 % CI [−0.079, +0.065], Nadeau–Bengio
95 % CI **[−0.148, +0.134]**.

**We are careful not to read this as equivalence, which would repeat the error we criticise in
§4.6.** A two one-sided tests procedure establishes equivalence only at margins of ±0.065
(naive) or **±0.121** (Nadeau–Bengio); at the pre-registered harm threshold of 0.01 the TOST
p-value is 0.48. The equivalence bound the data supports is therefore *wider than the +0.072
model-level effect we decline to claim above*. The minimum effect detectable at 80 % power is
0.100 event-F1 naive and **0.196** under fold-dependence correction.

The honest statement is therefore narrow: **teacher admission's model-level advantage does not
visibly survive the fail-closed stage, but this study is far too small to establish that the
advantage is erased.** The mechanism we propose — that reverting the random arm's failures to
real-only recovers most of what curation buys — is consistent with the reverting counts (2/9 vs
6/9) and with the fallback result in §4.6, but it is a **hypothesis this design cannot confirm**.

### 4.4.1 Why more seeds would not have helped

It is natural to assume the fix is more replicates. It is not, and the reason generalises beyond
this study. The Nadeau–Bengio standard error is `sd·√(1/n + ρ)` with ρ = n_test/n_train, so the
second term **does not shrink with the number of cells**. At our observed sd = 0.094 and ρ = 0.317:

| cells | se (NB) | smallest equivalence margin at 80 % power |
|---|---|---|
| 9 | 0.061 | 0.168 |
| 36 | 0.055 | 0.140 |
| 120 | 0.053 | 0.134 |
| ∞ | **0.053** | **0.131** |

Seeds buy precision in the point estimate and nothing in the corrected inference. **The binding
constraint is the number of folds — that is, disjoint test groups** — because ρ is fixed by the
split structure:

| design | se (NB) | margin |
|---|---|---|
| 3 folds × 3 seeds (this study) | 0.061 | 0.168 |
| 5 folds × 3 seeds | 0.047 | 0.125 |
| 10 folds × 3 seeds | 0.033 | 0.084 |

Any patient-independent EEG study reporting fold-corrected inference from three folds is subject
to this floor regardless of how many seeds it averages. We flag it because the instinct to buy
power with seeds is cheap and, for this class of inference, ineffective.

### 4.5 What the gate controls: the false-alarm tail (a replication, not a discovery)

**We state the status of this result before reporting it.** The source method's title is about
governing tail risk under subject shift, and its harm framing — with what probability does
augmentation harm subject-disjoint generalisation by a clinically meaningful margin — is the same
object as this section. What follows is therefore **not a new finding**. Its value is that it is,
to our knowledge, the **first independent replication** of that claim, the first at the **event
level**, and the first with an explicit **FP/24 h** axis on a rare-event clinical task. Readers
holding the source paper should weigh it as corroboration, and novelty in this manuscript should
be sought in §4.2, §4.4.1 and §4.6 instead.

Pooling all 81 gated-family cells and splitting by the gate's own admit/revert decision
(Δ against real_only, augmented model; **Figure 3**):

| | n | mean ΔFP/24 h | median | worst | mean Δevent-F1 |
|---|---|---|---|---|---|
| admitted | 21 | **−22.58** | −26.34 | +15.28 | +0.031 |
| reverted | 60 | **+10.90** | +13.83 | +62.63 | −0.025 |

Mann–Whitney p < 0.0001; within-fold permutation p < 0.0001; tail ΔFP/24 h > +20 is **0 of 21 vs
23 of 60**, Fisher p = 0.0004. On Δevent-F1 the separation is weak (p = 0.092). **The asymmetry is
the finding: the gate governs the false-alarm tail and does not select for event-F1.**

We subjected this to two attacks.

**Circularity.** The gate admits partly on validation FP/24 h, so better test FP/24 h could be
selection on the outcome. It is not: only 19 of 60 reverts fired on the FP criterion, while 39
fired on validation *event-F1*. Restricting to cells rejected on that different metric, the
separation strengthens — admitted 0/21 in the tail vs 16/41, Mann–Whitney p = 0.000046, Fisher
p = 0.000494, both clearing Bonferroni.

**Dose.** §4.2 shows admitted counts span 6 to 2,508, raising the possibility that admitted cells
are simply low-dose ones where the augmented model is nearly real-only. They are not: dose does
not differ between groups (median 577 admitted vs 136 reverted, p = 0.197 — and the sign is
*opposite* to the confound), dose does not predict ΔFP/24 h at all (Spearman −0.010, p = 0.932),
and the separation holds within each arm (`gated q0.5` p = 0.0008; `random_gated q0.9` p = 0.0008).

That last stratum matters: **the separation holds for randomly selected synthetic.** The gate's
decision stage predicts the false-alarm tail regardless of how the windows were chosen. Combined
with §4.4, two independent lines of evidence say the same thing — the gate's value lies in its
decision and fallback, not in what it admits.

### 4.6 Four evaluation pitfalls, quantified

1. **Selecting the comparison baseline on test inflates it by +0.035 event-F1.** A "best of three
   simple baselines per cell" reference scores 0.359 when chosen on test and 0.325 when chosen on
   validation and scored on test; validation and test pick the same arm in only 5 of 9 cells. We
   made this error ourselves and corrected it.

2. **Scoring a matched-volume control after the fail-closed revert destroys it.** When both arms
   fail closed they revert to the *same* real-only model, so those cells are bit-identical by
   construction. In our Phase 1 grid this affected 18 of 27 cells, driving the measured difference
   to 0.009 where the pre-revert difference was 2.6–4.1× larger. We made this error too.

3. **The selector statistic flips most admission decisions.** The event-F1 rule would admit 9 of 9
   cells on validation; the threshold-free AUPRC rule at the source method's margin of 0.01 admits
   1 of 9 and 0 of 9. The two disagree on 8 of 9 and 9 of 9 cells. Which statistic the fail-closed
   rule compares is not an implementation detail.

4. **An unmeasured run-to-run floor makes every small cross-family delta uninterpretable.** Re-running
   three baseline conditions under identical settings, changing nothing but weight-initialisation
   seeding, moved them by σ = 0.099–0.171 event-F1 and one of them by 0.093 in the mean (§4.1). That
   is larger than every augmentation effect reported in the papers we compare against, and larger
   than the effect we ourselves nearly reported. The cost of measuring it is one repeated arm; the
   cost of not measuring it is that a null and a real effect are indistinguishable. **Report the
   floor before reporting the effect.** We made this error too — it is the third of the four here
   that changed one of our own conclusions.

### 4.7 Generator fidelity

Sliced Wasserstein distance between real ictal windows and band-limited generator samples, against
a real-vs-real floor computed on disjoint halves of the real windows:

| fold | W₂ | floor | ratio |
|---|---|---|---|
| 0 | 0.465 | 0.072 | 6.4× |
| 1 | 0.450 | 0.073 | 6.1× |
| 2 | 0.445 | 0.071 | 6.3× |

W₂/floor ≈ 6.3 is interpretable and varies across cells, where the discriminator-AUC fidelity
metric used earlier saturated at 1.000 in all 11 sweep rows and could not grade anything. It also
supplies the mechanism for §4.4: at q = 0.95 the teacher keeps the top 5 % of a pool sitting 6.3×
further from real ictal than real ictal sits from itself, so it is discarding bad windows rather
than selecting redundant ones.

**Reported as a negative:** we had hoped to predict per-fold dose optima from W₂ and test the
ordering against the grid. Fold-to-fold W₂ spread is only 1.04×, implying an r* spread of 1.09×,
far below what n = 9 resolves. The prediction is not testable here.

## 5. Discussion

### 5.1 What survives

The governance framing survives our replication; the accuracy framing does not. Nothing here
suggests generative augmentation of ictal EEG is worth its complexity against a class-weighted
loss on this corpus, and the harm signature we observe matches what is already published for other
generators on the same data [3]. But the gate does something real and narrow: it bounds the
false-alarm tail, and it does so through its decision stage rather than its admission rule.

That suggests a practical consequence, which we state as a hypothesis rather than a result. If the
fallback is strong, the gate's value may lie in *rejecting* rather than in *curating*, and a
system builder would do better to invest in the fallback target than in generator fidelity or
admission thresholds. Three strands point this way: the admission advantage is not visible after
the fail-closed stage (§4.4); switching the fallback from real-only to the best simple baseline is
worth +0.048 event-F1 at r = 0.30, larger than any admission effect that survived correction; and
the tail-control separation holds even for randomly selected synthetic (§4.5), which places the
mechanism in the decision stage.

We stress that §4.4 cannot establish the first strand, only fail to refute it. A study powered for
equivalence — which, by §4.4.1, means **more folds, not more seeds** — is required before this
becomes a recommendation rather than a conjecture.

### 5.2 Silent mechanism failure

The result we expect to travel furthest is §4.2, and it is not about seizures. A gated
augmentation pipeline has a failure mode in which a threshold-calibration choice reduces the
intervention to approximately nothing while every observable output remains well-formed. Admission
rates, event-F1, FP/24 h, harm rates and gate decisions were all plausible for an entire
experimental phase during which the gate injected 6 windows where 251 were requested.

What makes this dangerous is that the failure is invisible to the reporting conventions of the
field. No standard results table asks for the realized intervention magnitude. A paper reporting
"trust-gated augmentation at q = 0.90" is, on its face, complete — and gives the reader no way to
discover that the gate admitted 0.4 % of what it was asked to. We were the authors, we had written
the gate, and we did not notice for a phase.

The remedy is cheap: **report realized quantities, not requested ones.** For augmentation that
means the injected count per cell, not the ratio parameter; for selection methods it means the
size and composition of the selected set, not the threshold. We would go further and suggest that
any paper claiming an intervention had no effect should be required to demonstrate that the
intervention actually occurred at the intended magnitude — a null from an intervention that did
not happen is not a null about the intervention.

Our own §4.6 pitfall 2 is the same lesson in the statistical register: a control scored after a
fallback stage measures the fallback, not the control.

### 5.3 On negative results and their reference class

Our headline negative — synthetic augmentation does not beat class weighting — is only as strong
as the baseline it is measured against, and we found that baseline choice swings the apparent
effect by more than the effect. That cuts both ways. It disciplines our own claim, but it also
implies that positive results in this literature reported against a single weak baseline, without
class weighting, are not interpretable at the precision they are quoted to. GP-EEG's +2.75 F1 on
CHB-MIT [3] is a sample-level number with no class-weighting comparison; we make no claim about
whether it would survive one, only that the question is open and cheap to answer.

The deeper problem is that this field's standard design — three to five patient-independent folds,
several seeds — has a hard inferential floor (§4.4.1) that is widely unacknowledged. Studies
average over seeds and report Wilcoxon tests as though replicates were independent. At three
folds, effects below roughly 0.13 event-F1 are not resolvable no matter how many seeds are run.
Most reported effects in this area are smaller than that.

### 5.4 What we would do differently

Three things, in order of value. Run more **folds** rather than more seeds. Report the **realized**
intervention magnitude alongside the requested one. And choose the comparison baseline on
**validation**, reporting its test score — selecting it on test inflated ours by +0.035, which is
comparable to the effects being debated.

The broader lesson concerns silent mechanism failure. Our admission reference diverged from the
published rank cut in a way that read, in a design table, as a minor reparameterisation. It
disabled the gate. The system continued to run, produce plausible metrics, and support a full
analysis — for an entire experimental phase — while injecting 6 windows where 251 were requested.
Only measuring the realized quantity, rather than the requested one, exposed it.

## 6. Limitations

1. **Single dataset.** CHB-MIT only; pediatric, and 2.2× denser in seizures than the Dianalund
   corpus used by the SzCORE challenge. Siena is specified but not run.
2. **Underpowered.** n = 9 per detector from 3 folds × 3 seeds. Only the tail-control result
   survives correction for fold dependence and multiplicity.
3. **Three folds of five** in the headline (WGAN) grids; the registered design is five. The
   earlier cVAE grid did run all five — see §7, deviation 3.
4. **Narrow validation panel.** Only 8 of 23 patient groups ever serve as validation across all
   five folds, and five of those appear in 4 of 5 folds, because the split routine
   deterministically carves the most seizure-rich patients. In the three folds the headline grids
   actually run it is narrower still — **7 of 23**, with chb01 and chb13 in all three. Both the
   admission threshold and the fail-closed decision live entirely on validation, so the gate has
   been evaluated on one near-fixed panel.
5. **Scarcity fixed at 1.0 in the WGAN grids.** The registered 0.5 and 0.25 rungs — where
   augmentation has most to offer — were run only with the cVAE at one seed (§7, deviation 6).
   The negative result holds there too, but it has not been replicated with the headline
   generator.
6. **Unseeded initialisation in Phase 1.** Model construction preceded seeding. Arms drawing a
   synthetic pool were seeded incidentally by the generator (§4.1) and are unaffected; the three
   pool-free baselines were not, so Phase 1's cross-family comparisons are not
   initialisation-controlled and its baseline rows are not reproducible. Fixed for Phase 2, which
   re-ran all three for TCN — the re-run is what makes the floor in §4.1 measurable. EEGNet and
   LCT baselines have not been re-run.
7. **Phase 2 ran one detector** (TCN) for budget reasons; the cross-detector replication of the
   dose curve and the admission contrast is absent. See `README.md` "Budget-constrained scope".
8. **Unconditional generators**; only the ictal phase is generated, which forecloses the
   label-consistency half of the published admission rule.

## 7. Declared pre-registration deviations

Our principal methodological asset is pre-registration, so deviations from it must be declared
rather than absorbed. Six were identified by internal audit *after* the first grid was scored;
their status after Phases 1 and 2 is below. We were not aware of them at the time of the initial
analysis, which is itself part of the record.

| # | axis | registered | as run | status |
|---|---|---|---|---|
| **1** | **Harm reference** | best simple baseline per (fold, seed) — max of real_only, class_weighted, classical_aug | `real_only` only in the multi-seed grid; the other two ran at Tier B but not there | **Remedied** in Phase 1, which added both. Worth **+0.163** event-F1 for TCN in the Tier B grid (5 folds, seed 42, cVAE) where it was first measured, **+0.135** in Phase 1 itself, and **+0.076** against Phase 2's re-seeded baselines. All three select the best arm on *test* and so are inflated by roughly +0.035 (§4.6, pitfall 1) |
| 2 | Detectors | eegnet, lct, tcn | eegnet, tcn (LCT dropped) | **Remedied** in Phase 1. LCT is a detector where the simple baseline also wins |
| 3 | Folds | 5 | all 5 at Tier B (cVAE, seed 42); 0, 1, 2 in the WGAN grids | **Partly stands.** The registered 5-fold design was executed, but only with the appendix generator at one seed. The headline grids trade folds for seeds — which §4.4.1 shows was the wrong trade: fold count, not seed count, binds fold-corrected inference |
| 4 | Core generator | cVAE core; WGAN-GP appendix | band-limited WGAN-GP as headline | **Stands.** Confirmatory → exploratory. Justified by a measured fidelity dead-end in the cVAE, but a deviation nonetheless |
| 5 | Admission quantile | q_core 0.90; grid {0.75, 0.90, 0.99} | {0.90, 0.50}, then {0.9917, 0.9500} under the pool reference | **Superseded.** §4.2 shows q is not comparable across admission references: the same q admits 6 windows under `real_ictal` and 1,505 under `pool` |
| 6 | Scarcity | 1.0, 0.5, 0.25 | all three at Tier B (cVAE, 5 folds, seed 42); 1.0 only in the WGAN grids | **Largely remedied.** The registered scarcity axis was run and the negative result holds at every rung — see below. Not replicated with the band-limited WGAN or at 3 seeds |

**Deviations 3 and 6, resolved against the Tier B grid.** These two were previously declared as
standing omissions. They are not: `tables/tierB_core.csv` holds the registered design — 5 folds
× 3 scarcity levels × 3 detectors × 5 conditions — run with the cVAE at seed 42, and
`scripts/verify_reported_numbers.py` reproduces it. Δ event-F1 against the registered
best-simple-baseline reference:

| scarcity | TCN ungated | TCN gated | LCT ungated | EEGNet ungated |
|---|---|---|---|---|
| 1.00 | −0.140 | −0.163 | −0.057 | +0.081 |
| 0.50 | −0.106 | −0.103 | −0.127 | −0.045 |
| 0.25 | −0.149 | −0.114 | −0.014 | −0.028 |

**The negative result holds at every scarcity level, including the two where augmentation should
have had its best case**, and across all five registered folds. That is a stronger statement than
this paper previously made about itself. What remains genuinely un-run is the *combination*:
scarcity 0.5 / 0.25 with the band-limited WGAN at three seeds.

Two further departures arose during the work and are declared here for completeness: Phase 2 ran a
**single detector** (TCN) for budget reasons, so its dose and admission results are not replicated
across architectures; and Phase 1 was run with **unseeded weight initialisation** (§6, item 6),
which breaks pairing for its cross-family comparisons and was fixed only for Phase 2.

We also record two corrections to our own analysis, both made before publication and both
material: the harm reference was initially selected on *test* rather than validation, inflating it
by +0.035 event-F1 (§4.6); and the matched-volume control was initially scored *after* the
fail-closed revert, which made 18 of 27 cells identical by construction and produced a null we
briefly believed (§4.6).

## 8. Reproducibility

All results reproduce from this repository without a GPU except the detector grids.

**Figures.** `scripts/make_preprint_figures.py --out reports/figures` generates all three from
the committed CSVs; the rendered PDFs are committed alongside. (Earlier drafts carried a warning
that this had never been executed, because Windows Smart App Control blocked matplotlib on the
authoring machine. That is no longer true.)

| result | data | command |
|---|---|---|
| §4.1, §4.5 | `downstream_gated_v2.csv` | `analyze_multiseed.py --tag _v2` |
| §4.1 (re-seeded baselines) | `downstream_gated_p2.csv` | `analyze_multiseed.py --csv … --conds-expected 9 --ratio 0.30` |
| §4.2, §4.3 | `downstream_gated_p2.csv` | `analyze_multiseed.py --csv … --conds-expected 9 --ratio 0.10` |
| §4.4 | `downstream_gated_p3.csv` | `analyze_multiseed.py --csv … --conds-expected 8` |
| §4.7 | `w2_dose_prediction.csv` | `gen_w2_dose_prediction.py --device cpu` |
| §7 (scarcity) | `tables/tierB_core.csv` | `scripts/verify_reported_numbers.py` |

The two `--ratio` rungs write to distinct `analysis_p2_r01_*` / `analysis_p2_r03_*` outputs. Until
2026-08-31 the `--ratio` filter also dropped the ratio-less baseline rows, so both commands above
returned an all-NaN report; the numbers in §4.2–§4.3 were correct but were not produced by the
command as printed. Fixed in `scripts/analyze_multiseed.py`.

Full experimental record, including two corrections to our own analysis:
`reports/DECISION_GATE_1.md`, `reports/DECISION_GATE_2.md`. Execution log: `the execution log`.

## References

Verification status for each is recorded in `reports/LITERATURE_AUDIT_2026-08-29.md` §6.
Items marked **[unverified]** are cited for a qualitative pattern only and were confirmed at
search level; they must be checked before any number is quoted from them.

[1] Choi, D.; Yip, C.; Choi, A.; Park, J. (2026). *Trust-gated synthetic EEG augmentation reduces
performance drops when generalizing to new patients.* npj Digital Medicine 9(1), art. 634.
doi:10.1038/s41746-026-02778-0. PMID 42185473. *(Verified: Crossref API and Europe PMC.)*

[2] Choi, D. et al. Preprint of [1]. bioRxiv, doi:10.64898/2026.01.26.701638v1.

[3] Moutonnet, M.; Corneck, ?; Tobar, F.; Mandic, D. (2026). *Synthesizing Epileptic Seizures:
Gaussian Processes for EEG Generation.* arXiv:2601.21752. **[TODO: complete author initials.]**

[4] Shidani, A.; Farghly, T.; Sun, ?; Ganjgahi, H.; Deligiannidis, G. *Beyond Real Data: Synthetic
Data Through The Lens Of Regularization.* arXiv:2510.08095. **[TODO: complete author initials.]**

[5] *Stronger Baseline Models — A Key Requirement for Aligning Machine Learning Research with
Clinical Utility.* arXiv:2409.12116. **[TODO: author list.]**

[6] Dan, J. et al. *SzCORE: A seizure community open-source research evaluation framework.*
arXiv:2505.18191. **[TODO: confirm author list and the v2 title, which differs from v1.]**

[7] *Meaningless comparisons lead to false optimism in medical machine learning.*
arXiv:1707.06289. **[unverified]**

[8] Coreset and data-pruning literature on random selection as a strong baseline, e.g.
arXiv:2210.15809. **[unverified]**

[9] Shoeb, A. (2009). *Application of machine learning to epileptic seizure onset detection and
treatment.* PhD thesis, MIT. CHB-MIT Scalp EEG Database, PhysioNet.
**[TODO: add the PhysioNet/PhysioBank citation alongside the thesis.]**

[10] Nadeau, C.; Bengio, Y. (2003). *Inference for the generalization error.* Machine Learning
52(3), 239–281. **[TODO: verify pagination.]**

[11] Lawhern, V. J. et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based
brain–computer interfaces.* Journal of Neural Engineering 15(5), 056013.
**[TODO: verify; also add citations for the LCT and TCN detector architectures.]**

[12] Gulrajani, I. et al. (2017). *Improved training of Wasserstein GANs.* NeurIPS.
**[TODO: verify.]**
