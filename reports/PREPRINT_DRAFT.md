# What does a fail-closed trust gate add to validation model selection? A leakage-safe, event-level replication of trust-gated synthetic ictal EEG augmentation on CHB-MIT

**Abdullah R. Alotaibi**

**Status:** DRAFT v0.5, 2026-09-02. Rewritten from v0.4 as a single account of the final design and
results. The correction history that earlier drafts carried inline is in
`reports/SUPPLEMENT_S1_corrections.md`; every number below is reproducible from this repository
by the commands in §8; bibliographic verification is in `reports/REFERENCES_VERIFIED_2026-09-02.md`.
Results marked **[Phase 3]** are pre-registered in `PREREGISTRATION_PHASE3.md` and not yet run.

---

## Abstract

Generative augmentation of rare ictal EEG is widely reported to improve seizure detection, and a
fail-closed "trust gate" [1] has been proposed to make it safe under subject shift: a real-data
teacher admits only synthetic windows it finds convincing, and the augmented model is deployed only
if it beats the real-only model on validation. We replicate trust-gated augmentation for
patient-independent seizure detection on CHB-MIT, scored at the event level under SzCORE
conventions, with pre-registered harm reporting and three detector families.

Synthetic augmentation is at parity with simple baselines on every detector, at every injection
ratio in the source method's operating band, and at every registered scarcity level. With the
published pool rank cut restored, teacher admission builds better augmented models than a matched
random draw (+0.072 event-F1, 8 of 9 cells) but the advantage does not survive fold-dependence
correction and is not visible after the fail-closed stage.

The gate's deployed behaviour is reproduced by validation model selection with no admission
stage. Choosing, per cell, the arm with the best validation event-F1 among real-only,
class-weighted, classically augmented and ungated-synthetic models, subject to the gate's own
false-alarm guard, lands within 0.03 event-F1 of the gate in all three Phase 2 grids and lowers
mean test false alarms by 8 to 18 per day, with no cell above the +20 FP/24 h tail threshold. The
tail control that replicates from the source method is therefore a property of validation
selection with a false-alarm constraint, not of synthetic curation.

Three evaluation findings travel beyond this dataset. The registered harm thresholds sit below
the measured noise floor: identical re-runs differing only in weight initialisation are flagged
as harm half the time, so harm must be reported as a curve against the margin. Selecting the
comparison baseline on test inflates it by +0.035 event-F1. And Nadeau–Bengio corrected inference
is floored by fold count, so three folds cannot resolve effects below about 0.13 event-F1 however
many seeds are run; a leave-one-group-out design over the 23 patient groups reaches 0.09 at one
seed. Phase 3, pre-registered here, runs that design together with the ceiling experiment the
question needs: whether real held-out ictal from other patients helps at all.

**Keywords:** seizure detection, synthetic data, generative augmentation, patient-independent
validation, model selection, negative results, evaluation methodology

---

## 1. Introduction

Seizures are rare. In our processed CHB-MIT corpus, ictal windows are 0.319 % of the data (5,563 of
1,746,447), which makes generative augmentation of the positive class an attractive idea, and a
substantial literature reports gains from it. Two properties make most of those reports hard to
act on clinically. Window-level metrics conceal event-level harm: a model can raise AUROC while
inflating false alarms per day, the number that decides whether a detector is deployable.
And patient-independent validation is rare: splitting windows rather than patients leaks subject
identity into every number.

Against this background Choi et al. [1] propose trust-gated augmentation (TGA). A teacher
detector trained on real data scores candidate synthetic windows and admits only those above a
confidence rank cut; the augmented model is deployed only if it improves a validation criterion,
and otherwise the system *fails closed* to the real-only model. The framing is explicitly one of
governance rather than accuracy, and the paper's own evidence is on balanced tasks (chronic pain,
motor imagery) with AUROC and a harm rate.

This paper asks three questions of that mechanism in a rare-event clinical setting. Does synthetic
augmentation help, measured against simple baselines chosen the way a practitioner would choose
them? Does the gate's *admission* stage do anything a random draw of the same size would not? And
does the gate as a whole do anything that validation model selection over already-trained
models would not?

Our contribution is not a better detector. It is (i) a leakage-safe, pre-registered, event-level
replication that answers those questions on CHB-MIT with three detector families; (ii) the
observation that the gate's deployed behaviour, including the false-alarm tail control that
replicates from [1], is matched by validation selection with the same false-alarm guard and no
synthetic curation at all; and (iii) three quantified evaluation results, on the harm margin, the
choice of reference and the fold floor of corrected inference, that apply to any study of this
kind.

## 2. Relationship to prior work

The fail-closed gate is not our invention. It is adapted from [1] (preprint [2], under a different
title). TGA contains no seizure content; our contribution includes the seizure-specific,
event-level reformulation of its admission and fail-closed criteria. TGA also runs a
matched-volume random-gating control and reports it as mixed: in its published numbers two of
three comparisons favour random gating, while the paper concludes that curation adds safety.

**Generative augmentation for seizure detection.** The closest prior work is GP-EEG [3], which
evaluates four generators as augmentation on CHB-MIT and Siena under leave-one-patient-out splits
with EEGNet and reports, against baseline, ΔF1 of −10.52 (COSCI-GAN), −14.23 (TimeVAE), −3.70
(ImagenTime) and +2.75 (GP-EEG) on CHB-MIT. Harm from synthetic ictal augmentation on this corpus
is therefore already published, with the signature we also observe. What GP-EEG leaves open is
exactly our contribution set: it reports no non-generative baselines, sample-level metrics only,
and no false-alarm rate. Because it established CHB-MIT and Siena as the expected pairing, Siena is
a comparability requirement we have not yet met (§6).

**Baselines, dose and selection.** Wolfrath et al. [5] name class balancing as a required clinical
baseline and document simple models matching complex ones; our finding that a class-weighted loss
is the strongest arm on TCN is an instance of a documented pattern. DeMasi et al. [8] show how
proposed-method win rates fall as baselines are added. Shidani et al. [4] give a
stability-based bound in which risk is U-shaped in the synthetic mixing ratio, which is why a
negative result cannot rest on a single dose (§4.2). Random selection as a strong baseline for
learned data selection is established in the coreset literature [9]; §4.4 is a seizure-specific,
event-level instance.

**Statistics.** Dietterich [22] and Nadeau and Bengio [14] show that resampled folds are not
independent observations and that the corrected variance has a term that does not shrink with
replicates; Bouckaert and Frank [21] show the replicability consequences. §4.4.1 applies this to
patient-group folds.

## 3. Methods

### 3.1 Data

CHB-MIT Scalp EEG [10, 11] from PhysioNet [12, 13], retrieved from the AWS Open Data mirror.

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

Patient groups are partitioned before windowing into five folds balanced on seizure count and
duration; test groups are disjoint across folds. The headline grids ran folds 0–2 at three seeds,
giving n = 9 paired (fold, seed) cells per condition per detector. Validation was carved from the
training pool by seizure richness, which we now know produced a near-fixed validation panel (7 of
23 groups across the three folds; §6). Phase 3 replaces this with a rotating panel and a
leave-one-group-out design (`scripts/make_phase3_splits.py`).

Three detector families spanning two orders of magnitude in capacity: EEGNet [15] (1,905
parameters), a lightweight convolution–transformer in the spirit of [18] (LCT, 120,834) and a
temporal convolutional network [17] (TCN, 129,441).

The generator is a WGAN-GP [16] fit per (fold, seed) on that fold's training ictal windows only,
with outputs band-limited to the acquisition passband before injection (out-of-band power 0.0093
→ 0.0005). Its fidelity, by sliced Wasserstein distance to real ictal, is 6.3 times the
real-vs-real floor (§4.7).

### 3.3 The gate and the policies compared with it

**Admission.** Each candidate synthetic window is scored by the real-only teacher's seizure
probability and admitted if at or above the q-quantile of a reference distribution, capped at the
target count. TGA's reference is the candidate pool itself (a rank cut). Our first grids used the
real training ictal windows as the reference; because the teacher has memorised those, the
threshold sits near 1.0 and the admitted count decouples from the request (§4.3). The pool
reference was restored for the Phase 2b grid and is the default.

**Fail-closed selection.** The augmented model is deployed only if its validation event-F1 is at
least the teacher's and its validation FP/24 h is within +0.25 of the teacher's; otherwise the
real-only model is deployed. Nothing in this implementation computes a manifold distance.

**Matched-volume control.** `random_gated` admits exactly as many windows as its gated sibling
from the same pool, drawn uniformly.

**Validation-selection policies (no admission stage).** To ask what the gate adds, we compare it
with policies that only *choose* among models already trained in the cell:

| policy | rule |
|---|---|
| `valsel4` | deploy the arm with the highest validation event-F1 among real_only, class_weighted, classical_aug, ungated |
| `valsel4_fpguard` | same, restricted to arms whose validation FP/24 h is within the gate's own +0.25 slack of real_only |
| `ungated_failclosed` | the gate's fail-closed rule applied to the ungated arm, no admission |
| `gated → valsel3` | the gate's decision, but reverting to the validation-selected best simple baseline instead of real_only |

All are computable from the Phase 2 CSVs, which carry validation event-F1, FP/24 h and AUPRC for
every arm (`scripts/analyze_validation_selection.py`).

### 3.4 Metrics, harm, statistics

Event scoring uses `timescoring` [20] under SzCORE conventions [6, 7] (toleranceStart 30 s,
toleranceEnd 60 s, minOverlap 0, maxEventDuration 300 s, minDurationBetweenEvents 90 s). The
operating threshold and a fixed 3-of-5 persistence filter are selected on validation and applied
once to test.

**Reference.** The pre-registered reference is the best of real_only, class_weighted and
classical_aug per cell. We choose that arm on *validation* event-F1 and score it on test. Choosing
it on test, as our earlier analyses did, inflates it by +0.035 event-F1 (§4.5).

**Harm.** `PREREGISTRATION.md` fixed harm as Δevent-F1 < −0.01 or ΔFP/24 h > +0.25. Those margins
turn out to lie below the measured noise floor (§4.6), so harm is reported as a curve against the
margin, with a null curve from identical re-runs, and the point rate at the registered margin is
kept only for continuity.

**Statistics.** Nine cells sharing three splits are not nine independent observations. Every
paired comparison reports the Nadeau–Bengio corrected t-test and interval [14] with ρ =
n_test/n_train = 0.317 from the split geometry, a fold-level t-test on seed-averaged deltas, the
Wilcoxon signed-rank test conventional in this literature, and the comparison count. Where they
disagree we report the corrected result.

## 4. Results

All numbers are held-out patient timelines, n = 9 cells per detector unless stated. Phase 1
(all three detectors, real-ictal admission reference, r = 1.0), Phase 2a (TCN, r ∈ {0.10, 0.30},
real-ictal reference) and Phase 2b (TCN, r = 1.0, pool reference, q ∈ {0.95, 0.9917}) are the
three grids.

### 4.1 Synthetic augmentation is at parity with simple baselines

Mean test event-F1, Phase 1:

| detector | real_only | class_weighted | classical_aug | ungated | gated q0.90 |
|---|---|---|---|---|---|
| EEGNet | 0.225 | 0.205 | 0.197 | 0.234 | 0.218 |
| LCT | 0.312 | 0.299 | 0.298 | 0.317 | 0.318 |
| TCN | 0.293 | 0.364 | 0.225 | 0.268 | 0.301 |

No synthetic arm beats any baseline by a margin this design can establish, on any detector. The
cross-family rows carry a caveat: Phase 1's three pool-free baselines ran before weight
initialisation was seeded, and re-running them for TCN under identical settings moved them by
σ = 0.099 (real_only), 0.159 (class_weighted) and 0.171 (classical_aug) event-F1, while the
pool-drawing arms were seeded incidentally and reproduce bit-for-bit. The TCN comparison against
class_weighted therefore reads −0.063 in Phase 1 and +0.030 against the re-run baselines; neither
is a measurement. EEGNet and LCT baselines await the seeded re-run **[Phase 3]**.

The parity holds at every registered scarcity level (1.0, 0.5, 0.25) and across all five folds in
the earlier cVAE grid (§7, deviations 3 and 6).

### 4.2 Dose is not the confound

Phase 1 sampled only r ≈ 0.032 and r = 1.00; the source method's validation ladder selects
r ≤ 0.30. Sampling the band with the ungated arm, which receives the requested dose exactly
(TCN): event-F1 0.283 at r = 0, 0.264 at 0.10, 0.217 at 0.30, 0.268 at 1.00. Monotone through
the band, no interior optimum, nothing significant (**Figure 2**).

### 4.3 Report realized, not requested, dose

Under the real-ictal admission reference the gate admitted 6 windows against a request of 251 and
23 against 752 (0.4–0.5 % of the pool regardless of target), and the source method's K_min = 200
safeguard was met in 0 of 9 cells. With the pool reference, admitted = min(oversample·(1−q), 1)·
n_synth exactly, confirmed live at 126 and 755 windows (**Figure 1**). The divergence was ours,
documented in advance as a design choice, and invisible in every pipeline output for a full
experimental phase. The transferable lesson is one sentence: a null from an intervention that did
not occur is not a null about the intervention, so the injected count belongs in every table.

### 4.4 Admission quality: better models, no detectable difference after fail-closed

With the pool reference the matched-volume control is well posed: identical pool, identical count,
identical initialisation. On the augmented models, teacher selection beats a random draw by +0.072
event-F1 (8 of 9 cells) at r ≈ 0.30 and −22.3 FP/24 h; Wilcoxon p = 0.020 but Nadeau–Bengio
p = 0.140, fold-level 0.103, and the family's Bonferroni threshold is 0.0063. Direction and count
are reportable; significance is not.

After the fail-closed stage the deployed policies differ by −0.007 (teacher 0.300, random 0.307),
NB 95 % CI [−0.148, +0.134]. A TOST procedure establishes equivalence only at ±0.121, wider than
the +0.072 above, so this is a failure to detect, not equivalence.

#### 4.4.1 Why more seeds would not have helped

The Nadeau–Bengio standard error is sd·√(1/n + ρ) with ρ = n_test/n_train fixed by the split
geometry, so its floor sd·√ρ = 0.053 at three folds cannot be bought down with seeds. The 80 %
power minimum detectable effect for candidate designs over the 23 CHB-MIT groups
(`phase3_free/power_design.csv`, sd = 0.094):

| design | n cells | ρ | se (NB) | MDE |
|---|---|---|---|---|
| 3 folds × 3 seeds (as run) | 9 | 0.317 | 0.061 | 0.196 |
| 5 folds × 3 seeds | 15 | 0.319 | 0.058 | 0.176 |
| 10 folds × 3 seeds | 30 | 0.138 | 0.039 | 0.113 |
| 23 folds × 1 seed (leave-one-group-out) | 23 | 0.056 | 0.030 | **0.087** |

One seed of leave-one-group-out beats three seeds of three folds by more than a factor of two.
This is the Phase 3 design.

### 4.5 The gate versus validation model selection

This is the section the paper turns on. **Table 1** shows every deployable policy in the Phase 2b
grid (TCN, r = 1.0, pool reference, 9 cells), with deltas against real_only, and **Figure 5**
shows the cells.

**Table 1.** Deployed test metrics, Phase 2b. Tail = cells with ΔFP/24 h > +20 against real_only.

| policy | needs admission | event-F1 | FP/24 h | worst ΔFP vs real_only | tail |
|---|---|---|---|---|---|
| real_only | – | 0.283 | 39.1 | – | – |
| class_weighted | – | 0.271 | 14.4 | +55.9 | 1 |
| ungated (r = 1.0) | – | 0.268 | 39.0 | +71.5 | 3 |
| admit-always q0.95 (admission, no fail-closed) | yes | 0.277 | 14.7 | +5.4 | 0 |
| **trust gate q0.95, as deployed** | yes | **0.300** | **13.8** | 0.0 | 0 |
| random admission q0.95, as deployed | yes | 0.307 | 27.3 | 0.0 | 0 |
| ungated_failclosed | no | 0.341 | 22.4 | 0.0 | 0 |
| valsel4 | no | 0.291 | 13.5 | +46.7 | 1 |
| **valsel4_fpguard** | no | **0.328** | **5.8** | 0.0 | 0 |

Head-to-head against the gate as deployed, paired by cell:

| competitor vs gate | grid | Δ event-F1 | NB 95 % CI | p_NB | Δ FP/24 h | cells above +20 vs gate |
|---|---|---|---|---|---|---|
| valsel4_fpguard | 2a, r = 0.10 | −0.005 | [−0.19, +0.19] | 0.96 | −10.7 | 0 |
| valsel4_fpguard | 2a, r = 0.30 | +0.043 | [−0.13, +0.22] | 0.58 | −13.0 | 0 |
| valsel4_fpguard | 2b, r = 1.0 | +0.028 | [−0.19, +0.25] | 0.78 | −8.0 | 0 |
| ungated_failclosed | 2b, r = 1.0 | +0.042 | [−0.08, +0.16] | 0.43 | +8.6 | 1 |
| gated → valsel3 | 2b, r = 1.0 | −0.003 | [−0.02, +0.01] | 0.62 | +0.3 | 0 |

Three things follow. First, validation selection with the gate's own false-alarm guard is never
detectably worse than the gate on event-F1 and is lower on false alarms in every grid, by 8 to 13
per day on average, with no cell in the tail. Second, the fail-closed rule applied to the ungated
arm, with no admission at all, reaches the highest deployed event-F1 in the grid (0.341), though
at the cost of one tail cell. Third, changing the gate's fallback target changes almost nothing
here, because in this grid the gate rarely reverts.

None of the event-F1 differences approaches significance; the false-alarm differences are
consistent in sign across three grids and nine cells each but are likewise unpowered. The claim
we make is bounded: **nothing the gate does at deployment has been shown to exceed what
validation selection over already-trained arms does, and on the false-alarm axis selection is
ahead in every grid.** A design that can distinguish them is registered (§4.4.1) **[Phase 3]**.

A caveat that must travel with Table 1: any policy that can select the reference arm coincides
with it in some cells, so its harm rate against that reference is partly zero by construction. The
gate reverting to real_only is the same phenomenon. The head-to-head rows are the non-circular
comparison.

### 4.6 Harm as a curve, not a point

The registered harm margins (−0.01 event-F1, +0.25 FP/24 h) were fixed before any floor had been
measured. Two floors are now measurable. Twenty-seven pairs of identical specification, differing
only in weight-initialisation seeding (TCN baselines, Phase 1 vs Phase 2), give σ = 0.10–0.17
event-F1 and 10–37 FP/24 h. At the registered margins that null flags **50 %** of identical
re-runs as harm on event-F1 and **46 %** on FP/24 h. The floor that applies to a same-seed paired
delta (same initialisation, different synthetic draw) is not yet measured and is the first Phase 3
run.

**Figure 4** therefore reports harm as a function of the margin, with the null curve overlaid.
Against real_only in Phase 2b, the ungated arm sits *at* the null across the whole event-F1 range
and *above* it on false alarms up to +70 FP/24 h; the class-weighted arm sits below the null on
false alarms beyond +5 FP/24 h; the gate as deployed has no false-alarm harm at any margin, which
is what reverting to the reference guarantees. A harm rate is quotable only where a curve separates
from the null, and at the registered margins none does.

### 4.7 Tail control replicates, and is not specific to the gate

Pooling the 81 Phase 1 gated-family cells by the gate's own admit/revert decision, admitted cells
show ΔFP/24 h of −22.6 (mean, augmented model vs real_only) against +10.9 for reverted cells;
0 of 21 admitted cells exceed +20 FP/24 h against 23 of 60 reverted (Fisher p = 0.0004, Mann–
Whitney and within-fold permutation p < 0.0001). The separation survives dropping every cell
reverted on the FP criterion (the rest were reverted on validation event-F1), survives dose
stratification within arm, and holds for randomly admitted synthetic (**Figure 3**). This
replicates the source method's central claim at the event level and on a false-alarm axis.

It is, however, the statement that validation performance predicts test performance. §4.5 shows
the same tail control is obtained by validation selection with the same false-alarm guard and no
synthetic data in the loop. The mechanism is the decision stage, not the admission stage.

### 4.8 Generator fidelity

Sliced Wasserstein distance between real ictal and band-limited generator samples is 6.4, 6.1 and
6.3 times the real-vs-real floor in folds 0–2. The discriminator-AUC metric used earlier saturated
at 1.000 in every sweep row. The fold-to-fold spread (1.04×) is too small to test any dose-optimum
prediction at n = 9.

## 5. Discussion

**What survives.** The governance framing of [1] survives replication in a rare-event clinical
task; the accuracy framing does not. Nothing here suggests generative augmentation of ictal EEG is
worth its complexity against a class-weighted loss on this corpus, and its harm signature matches
what is already published for other generators [3]. The gate bounds the false-alarm tail. But so
does choosing among already-trained models on validation with the same false-alarm constraint,
and in every grid that choice also lowers the mean false-alarm rate.

**A hypothesis, stated as one.** If the fallback is strong, the gate's value lies in *rejecting*
rather than *curating*, and a system builder would do better to invest in the candidate set and
the selection rule than in generator fidelity or admission thresholds. Three strands point this
way: the admission advantage is not visible after the fail-closed stage; the fail-closed rule
applied to the ungated arm is the best deployed policy in Phase 2b; and validation selection with
an FP guard matches the gate's tail control while the gate's admission stage adds nothing
detectable. Phase 3's Q-B test is designed to confirm or refute this at an MDE of 0.087.

**The ceiling question.** Every result above is about one generator of measured low fidelity. The
question a referee will ask first is whether *any* augmentation could help this task. Phase 3's
positive control injects real held-out ictal from other training patients as if it were synthetic.
If that does not move patient-independent event-F1, parity is a property of the task and generator
work is not the next step; if it does, fidelity is the binding constraint. The interpretation
rules are fixed in `PREREGISTRATION_PHASE3.md` §4 before the run.

**Three evaluation results that travel.** Report realized intervention magnitudes, not requested
ones (§4.3). Choose the comparison baseline on validation and report the inflation if it was ever
chosen on test (+0.035 here). Report harm as a curve against the margin with the re-run null
overlaid, and design for folds rather than seeds (§4.4.1, §4.6).

## 6. Limitations

1. **Single dataset.** CHB-MIT only, paediatric, 2.2× denser in seizures than the SzCORE
   challenge corpus. Siena is specified but not run.
2. **Underpowered.** n = 9 per detector from 3 folds × 3 seeds; the fold-corrected MDE is 0.196
   event-F1. Only the tail-control separation survives correction, and §4.7 explains what it
   measures.
3. **Narrow validation panel.** Only 7 of 23 groups ever served as validation in the folds run,
   with chb01 and chb13 in all three, because the split routine carved the most seizure-rich
   groups deterministically. Every gate decision and every validation-selection policy in this
   paper was evaluated on that near-fixed panel. The Phase 3 split files rotate it.
4. **Operating regime.** Arms typically run at 10–40 FP/24 h because the threshold maximises
   validation event-F1; the SzCORE challenge leaders sit near 1–2 per day. Results at a fixed
   false-alarm cap require per-window scores that the grids did not persist.
5. **One generator, low fidelity.** 6.3× the real-vs-real floor. The parity result is about this
   generator until the positive control bounds the task.
6. **Phase 1 baselines not initialisation-controlled** for EEGNet and LCT; TCN's were re-run.
7. **Phase 2 ran one detector** for budget reasons; §4.5 is TCN only.
8. **Unconditional generator**; only the ictal class is generated.

## 7. Declared pre-registration deviations

Our principal methodological asset is pre-registration, so deviations are declared, not absorbed.

| # | axis | registered | as run | status |
|---|---|---|---|---|
| 1 | harm reference | best simple baseline per cell | real_only only in the first multi-seed grid; both baselines added in Phase 1; arm chosen on test until 2026-09-02, now on validation | remedied; test-selection inflation +0.035 reported |
| 2 | detectors | EEGNet, LCT, TCN | LCT dropped from the first grid | remedied in Phase 1 |
| 3 | folds | 5 | all 5 at Tier B (cVAE, seed 42); 0–2 in the WGAN grids | partly stands; Phase 3 moves to 23 |
| 4 | core generator | cVAE core, WGAN-GP appendix | band-limited WGAN-GP as headline | stands (measured cVAE fidelity dead-end) |
| 5 | admission quantile | 0.90; grid {0.75, 0.90, 0.99} | {0.90, 0.50}, then {0.9917, 0.95} under the pool reference | superseded; q is not comparable across references |
| 6 | scarcity | 1.0, 0.5, 0.25 | all three at Tier B; 1.0 in the WGAN grids | largely remedied; parity holds at every rung |
| 7 | harm margin | −0.01 / +0.25 | reported, but shown to lie below the noise floor | curve reported; Phase 3 registers a floor-based margin |
| 8 | validation panel | not registered | near-fixed panel (7 of 23 groups) | Phase 3 rotates |

The correction history behind these rows, including two analyses we published and later withdrew,
is in `reports/SUPPLEMENT_S1_corrections.md`.

## 8. Reproducibility

All results reproduce from this repository without a GPU except the detector grids.

| result | command |
|---|---|
| §4.1, §4.7 | `python scripts/analyze_multiseed.py --tag _v2` |
| §4.1 re-seeded, §4.2, §4.3 | `python scripts/analyze_multiseed.py --csv …/downstream_gated_p2.csv --conds-expected 9 --ratio 0.10` (and `0.30`) |
| §4.4 | `python scripts/analyze_multiseed.py --csv …/downstream_gated_p3.csv --conds-expected 8` |
| §4.4.1, §4.5, §4.6 | `python scripts/analyze_validation_selection.py` → `analysis_tierB/phase3_free/` |
| §4.8 | `python scripts/gen_w2_dose_prediction.py --device cpu` |
| §7 scarcity | `python scripts/verify_reported_numbers.py` |
| figures | `python scripts/make_preprint_figures.py --out reports/figures` |
| Phase 3 splits | `python scripts/make_phase3_splits.py` |

`analyze_multiseed.py` selects the registered reference on test and says so in its output; it is
kept for Phase 1, whose CSV lacks validation columns. The environment that reproduces the analysis
is pinned in `requirements.lock.txt`; every Phase 3 driver writes `run_environment<tag>.json`
beside its CSV. Continuous integration runs the tests, both analysis scripts, the split-file
regeneration and the figure render on every push.

## References

Verified 2026-09-02 against Crossref, arXiv and publisher registries; see
`reports/REFERENCES_VERIFIED_2026-09-02.md` for per-field status.

1. Choi D, Yip C, Choi A, Park J. Trust-gated synthetic EEG augmentation reduces performance drops when generalizing to new patients. *npj Digit Med*. 2026;9(1):634. doi:10.1038/s41746-026-02778-0
2. Choi D, Yip C, Choi A, Park J. Fail closed trust gated synthetic augmentation governs tail risk under subject shift in EEG. *bioRxiv*. 2026. doi:10.64898/2026.01.26.701638
3. Moutonnet N, Corneck J, Tobar F, Mandic D. Synthesizing epileptic seizures: Gaussian processes for EEG generation. arXiv:2601.21752. 2026.
4. Shidani A, Farghly T, Sun Y, Ganjgahi H, Deligiannidis G. Beyond real data: synthetic data through the lens of regularization. arXiv:2510.08095. 2025.
5. Wolfrath N, Wolfrath J, Hu H, Banerjee A, Kothari AN. Stronger baseline models: a key requirement for aligning machine learning research with clinical utility. arXiv:2409.12116. 2024.
6. Dan J, Shahbazinia A, Kechris C, Atienza D. Quantifying the generalization gap in seizure detection: a large-scale empirical benchmark via the SzCORE challenge. arXiv:2505.18191. 2025 (v2 2026).
7. Dan J, Pale U, Amirshahi A, et al. SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of electroencephalography-based automated seizure detection algorithms. *Epilepsia*. 2025;66(S3):14–24. doi:10.1111/epi.18113
8. DeMasi O, Kording K, Recht B. Meaningless comparisons lead to false optimism in medical machine learning. *PLoS ONE*. 2017;12(9):e0184604. doi:10.1371/journal.pone.0184604
9. Zheng H, Liu R, Lai F, Prakash A. Coverage-centric coreset selection for high pruning rates. *Proc ICLR 2023*. arXiv:2210.15809
10. Shoeb AH. Application of machine learning to epileptic seizure onset detection and treatment [PhD thesis]. MIT; 2009. http://hdl.handle.net/1721.1/54669
11. Guttag J. CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. 2010. doi:10.13026/C2K01R
12. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*. 2000;101(23):e215. doi:10.1161/01.CIR.101.23.e215
13. Pollard T, Moody BE, Lehman L-wH, et al. PhysioNet as a global platform for biomedical research. *Nat Health*. 2026;1(8):792–795. doi:10.1038/s44360-026-00096-z
14. Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn*. 2003;52(3):239–281. doi:10.1023/A:1024068626366
15. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *J Neural Eng*. 2018;15(5):056013. doi:10.1088/1741-2552/aace8c
16. Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved training of Wasserstein GANs. *Adv Neural Inf Process Syst 30*. 2017:5767–5777.
17. Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. 2018.
18. Rukhsar S, Tiwari AK. Lightweight convolution transformer for cross-patient seizure detection in multi-channel EEG signals. *Comput Methods Programs Biomed*. 2023;242:107856. doi:10.1016/j.cmpb.2023.107856
19. Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. *Proc IEEE ICCV*. 2017:2999–3007. doi:10.1109/ICCV.2017.324
20. Embedded Systems Laboratory, EPFL. timescoring [software]. https://github.com/esl-epfl/timescoring
21. Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. *PAKDD 2004*, LNCS 3056. Springer; 2004:3–12. doi:10.1007/978-3-540-24775-3_3
22. Dietterich TG. Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Comput*. 1998;10(7):1895–1923. doi:10.1162/089976698300017197
