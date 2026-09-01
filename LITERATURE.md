# LITERATURE.md — external evidence, benchmarking context, and positioning

Compiled 2026-08-28 from a targeted literature search. Purpose: place this benchmark's
results in the published landscape, verify the prior work it adapts, and convert several
open methodological questions into citable, solved problems.

**Provenance.** Sources marked **[full text]** were fetched and read in full. Sources marked
**[summary]** come from search-result summaries only and should be verified against the
primary text before citation in a manuscript.

---

## 0. What this changes

| Area | Before | After |
|---|---|---|
| Absolute performance | "event-F1 0.19–0.31 looks low vs. literature's 0.94" | matches the winner of an independent 28-algorithm challenge (0.32); the 0.94 claims are leakage-inflated |
| TGA novelty claim | asserted | **verified** — TGA contains no seizure/epilepsy/event-level content (but it is peer-reviewed prior art, §10.1) |
| TGA fidelity | assumed faithful | **four spec divergences**, two decisive (quantile reference distribution; `min_admitted`) |
| "Gate is over-conservative" | tune `q` empirically | diagnosed as **a substituted admission rule + teacher miscalibration**; `q = 0.99` is unreachable in this code |
| n=9 dependence problem | open criticism | **Nadeau–Bengio corrected resampled t-test** |
| Venue | specialist journal / NeurIPS D&B | NeurIPS **Evaluations & Datasets** track explicitly invites this framing |

---

## 1. Benchmark context: these numbers are realistic, not weak

### 1.1 The SzCORE Challenge — the reference point this project should be citing

**[full text]** *Quantifying the Generalization Gap in Seizure Detection: A Large-Scale
Empirical Benchmark via the SzCORE Challenge* — arXiv:2505.18191 (report from the 2025 AI in
Epilepsy and Neurological Disorders Conference). 28 submitted algorithms evaluated on a
strictly held-out **private** dataset: 65 subjects, 4,360 hours, expert-neurophysiologist
annotated.

| | winner ("Sz Transformer") | top-5 range | all 28 submissions |
|---|---|---|---|
| **Event F1** | **0.32** | — | 0.32 → 0.00 |
| Sensitivity | 0.37 | 0.30 – 0.58 | 0.99 → 0.00 |
| Precision | 0.29 | 0.19 – 0.29 | 0.29 → 0.00 |
| FP / day | 1.34 | 1.34 – 14 | 1 → 290 |

Authors' conclusion: *"nearly all algorithms strongly overestimated their performance
compared to the results obtained on this independent test set."*

**Comparison to this project** (`analysis_tierB/downstream_gated_multiseed.csv`, n=9 cells):

| | event-F1 |
|---|---|
| SzCORE challenge winner | 0.32 |
| this project, TCN `real_only` (mean) | 0.191 |
| this project, EEGNet `real_only` (mean) | 0.218 |
| this project, TCN gated q0.90 augmented (mean) | 0.275 |
| this project, best single cell (fold 2 / seed 42) | 0.410 |

**Directly comparable.** `evaluation/event_metrics_szcore.py:24-29` uses exactly the published
SzCORE tolerances — `toleranceStart=30 s`, `toleranceEnd=60 s`, `minOverlap=0` (any-overlap),
`maxEventDuration=300 s`, `minDurationBetweenEvents=90 s` — scored through the same
`timescoring` library the challenge used.

> **Action:** add this comparison to `reports/comprehensive_report.md`. It is the single
> cheapest credibility gain available and the report currently omits it.

### 1.2 Why the CHB-MIT literature reports 0.94+ and why it should be discounted

**[full text]** *Epileptic seizure detection using CHB-MIT dataset: The overlooked
perspectives* — PMC11286169. Documents three failure modes in the CHB-MIT literature:

1. **Subject-level leakage.** Random segment-wise train/test splits put the same patient on
   both sides. (This project splits on patient groups before windowing — `chbmit/splits.py`.)
2. **Selective record use.** Most studies use only seizure-bearing records, hiding the true
   imbalance (344:1 in their accounting). (This project scores continuous real timelines.)
3. **Segment vs. event confusion.** Reported "accuracy" is usually random-segment
   classification, not detection of complete seizure events.

Corrected for all three, they obtain **72.63–75.34 % sensitivity at 4.79–5.32 false
detections per hour ≈ 115–128 FP/24h**.

**Implication for the FP/24h finding.** The false-alarm inflation in the augmented
conditions (mean +9.4/24h, worst +40.7/24h) must be reported — it is pre-registered harm —
but the *absolute* FP/24h levels here (2–73) are competitive with, and often better than,
this properly-evaluated reference. The honest framing is a **sensitivity / false-alarm
trade**, not a safety failure.

---

## 2. Prior work: TGA verified — seizure/event-level novelty holds, implementation diverges

**[full text]** *Fail closed trust gated synthetic augmentation governs tail risk under
subject shift in EEG.*

**Citation metadata verified against the bioRxiv details API** (`api.biorxiv.org/details/biorxiv/10.64898/2026.01.26.701638`):

```
title:     Fail closed trust gated synthetic augmentation governs tail risk under subject shift in EEG
authors:   Choi, D.; Yip, C.; Choi, A.; Park, J.
date:      2026-01-28      version: 1      category: bioinformatics
published: NA              server: bioRxiv
```

> **SUPERSEDED — see §10.1.** Cite the peer-reviewed *npj Digital Medicine* version of record
> (`10.1038/s41746-026-02778-0`, 2026-05-25; PMID 42185473). The bioRxiv `published: NA` field
> is **stale** — it is not evidence of preprint status. The preprint DOI
> `10.64898/2026.01.26.701638` is also real and resolves; the npj article is the version of
> record. §10.1 below corrects the earlier claim here.

Datasets: PainMunich (n=189 — 101 chronic-pain, 88 controls, resting-state) and BCI IV-2a
(n=9, motor imagery). Metric: **AUROC only**, at window and participant level. Harm defined
as ΔAUROC < −0.01 vs. paired real-only baseline.

**The paper contains no mention of seizure, epilepsy, or event-level metrics.** The
seizure-specific, event-level reformulation of admission and fail-closed criteria claimed in
`PREREGISTRATION.md` §1 and `README.md` is therefore **a genuine contribution**.

### 2.1 Four divergences from the source method

| | TGA (source) | this implementation | severity |
|---|---|---|---|
| **quantile reference distribution** | **appears to be the synthetic candidate pool** (rank-based: keep the top `1−q`) | **the real training-ictal score distribution** (`run_admission` → `admission_threshold(ref_scores, q)`) | **high** — see §2.1.1 |
| admission criteria | label-consistency **AND** confidence quantile | confidence quantile only (`synthetic/trust_gate.py::admit_indices`) | low — see §3 |
| **minimum admitted** | **< 200 candidates → augmentation disabled** | **`min_admitted = 1`** | **high** |
| selector margin | AUROC + 0.01 | event-F1 + 0.0 | low (this is *less* conservative) |
| `q` swept | {0.70, 0.80, 0.90, 0.95, 0.99} | {0.90, 0.50} | medium |

#### 2.1.1 Evidence that TGA's quantile is pool-relative, not real-relative

TGA reports a **median of 4,200 windows kept at `q = 0.70` and 200 at `q = 0.99`**. Against a
candidate pool of roughly 15,000 (this project's pool is exactly 15,048 = `oversample 6 ×
n_synth 2,508`), a *pool-relative* rule "keep the top `1−q` fraction" predicts:

| q | predicted kept (top `1−q` of ~15,000) | TGA reported |
|---|---|---|
| 0.70 | ~4,500 | **4,200** |
| 0.99 | ~150 | **200** |

The match is close enough that TGA's `q` is almost certainly a **rank cut on the synthetic
pool**, not a quantile of teacher scores on real ictal windows.

This repo calibrates on real training-ictal scores instead, which is why `q = 0.99` yields an
admission threshold of exactly **1.0** and admits **zero** windows for *every* generator tried
(`gate_admission_qsweep.csv`) — i.e. **the source paper's best-performing setting is
unreachable in this implementation.** This is the root cause of the "over-conservative gate"
finding, and it is a substitution, not a property of trust-gated augmentation.

> **Action:** re-implement admission as a rank cut on the candidate pool and re-run the
> `q` sweep. This is the same fix §3 arrives at from the calibration direction — two
> independent lines of evidence converging on the same change.

**The `min_admitted` divergence is decisive.** The headline TCN q=0.90 cells admit
**22, 26, 58, 70, 83, 98, 143, 144, 201** synthetic windows against a target of 2,508. Under
TGA's own floor of 200, **eight of nine cells would have had augmentation disabled entirely.**
The +0.083 event-F1 result is therefore carried almost entirely by cells the source method
would have refused to run.

> **Action:** re-run the headline with `min_admitted = 200` and report the outcome, or state
> and justify the departure explicitly. A reviewer familiar with TGA will find this
> immediately.

Note also that `q = 0.50` lies far below anything TGA tested (their floor was 0.70), and it
was not in this project's own pre-registered grid `{0.75, 0.90, 0.99}`.

### 2.2 Two findings that support this project

- **TGA found harm non-monotonic in `q`, with lowest pooled harm at `q = 0.80`** (0.34 vs.
  0.44 ungated). This independently corroborates the "the gate is mis-tuned, not
  fundamentally wrong" reading — and see §3 for the likely mechanism.
- **TGA's ungated harm rates were 0.44 – 0.56.** This project's ungated harm rates are
  0.22–0.44 (event-F1) and 0.55–0.67 (FP/24h). **This benchmark independently replicates
  TGA's core phenomenon in a new modality (scalp seizure EEG), a new task (event detection),
  and a new metric family (event-F1 / FP-24h).** That is a robust standalone contribution
  that does not depend on the contested +0.083, and the current report underplays it.

### 2.3 TGA's own roadmap is this project's Tier C

TGA's stated next step: *"frozen-policy external clinical validation: select and freeze the
governance policy… on a development cohort, then evaluate the same policy on an independent
clinical EEG cohort without post hoc tuning."*

That is precisely the Siena / TUH plan. Positioning it as executing the prior work's
explicitly-stated roadmap is stronger than presenting it as a generalization check.

---

## 3. Diagnosis: gate conservatism is teacher miscalibration, not conservatism

From this project's own `analysis_tierB/gate_admission_qsweep.csv` and
`gate_admission_retest.json`:

```
teacher score on real train ictal:  mean 0.8507   (background mean 0.0486)

q       admission threshold    admitted (wgan_bandlimited)
0.50    0.9612                 17.4 %
0.75    0.9921                  2.2 %
0.90    0.9985                  0.10 %
0.95    0.9994                  0.01 %
0.99    1.0000                  0 %
```

Every quantile of the teacher's real-ictal score distribution is compressed into
**[0.96, 1.0]**. Moving `q` from 0.50 → 0.90 shifts the threshold by **0.037** but changes
admission by **174×**. `q` is a catastrophically ill-conditioned control knob.

This is textbook modern-network overconfidence — **[summary]** Guo et al., *On Calibration of
Modern Neural Networks* (ICML 2017): deep networks are systematically overconfident, so
quantiles of raw sigmoid/softmax output carry little information. It also plausibly explains
TGA's non-monotonic-in-`q` result (§2.2).

**Recommended fix, in preference order:**

1. **Temperature-scale the teacher** on the validation patients (single parameter, fit by
   NLL) *before* computing the admission quantile.
2. **Gate on rank rather than absolute probability** — admit the top-k synthetic windows by
   teacher score, making `k` (not `q`) the control.
3. Only then sweep the margin / fail-closed slack.

This converts the generator-fidelity next step from empirical knob-twiddling into a diagnosed,
citable correction.

**Refuted hypothesis (recorded so it is not re-investigated):** the admission threshold at
`q = 0.50` is 0.9612, far above 0.5, so `q = 0.50` does **not** admit label-inconsistent
windows. TGA's label-consistency criterion is implicitly satisfied here, which is why its
absence from `admit_indices` is low-severity.

---

## 4. Statistics: the n=9 dependence problem has a named fix

The 9 cells are 3 folds × 3 seeds; seeds within a fold share held-out patients, so the paired
deltas are not independent and both the Wilcoxon test and the i.i.d. bootstrap are
mis-specified.

- **[summary]** **Nadeau & Bengio corrected resampled t-test** — inflates the variance
  estimate by `1/n + n_test/n_train` to account for overlapping training sets across
  resampling runs. Exactly the correction this design needs. Implementations: the R package
  [`correctR`](https://hendersontrent.github.io/correctR/); a Python version exists as a
  public gist. Recommended in Weka; the standard advice is 10×10-fold CV with this correction.
- **[full text]** *The role of data partitioning on the performance of EEG-based deep
  learning models in supervised cross-subject analysis* — arXiv:2505.13021. Finds that
  partitioning choice produces performance differences **that can exceed those attributed to
  architecture**, and recommends (i) nested leave-N-subjects-out, (ii) reporting seeds **and**
  fold specifications, (iii) explicit sensitivity analysis across seeds.

> **Action:** this is the citation that reframes the seed-only control run as *current best
> practice* rather than as fixing a bug. It also independently supports the run-to-run
> instability observed between `downstream_gated_bandlimited.csv` and
> `downstream_gated_multiseed.csv` at fold 0 / seed 42.

---

## 5. Second-dataset intelligence (Tier C / Tier D)

**[summary]** Expected cross-dataset degradation:

| train → test | reported AUC |
|---|---|
| CHB-MIT → CHB-MIT (within) | 0.904 ± 0.059 |
| **CHB-MIT → TUSZ** | **0.615 ± 0.039** |
| TUSZ → CHB-MIT | 0.762 ± 0.175 |
| → Siena, no domain adaptation | ~0.701 ± 0.025 |

Siena is adult (20–71 y), 512 Hz, 29 channels — a substantial shift from pediatric CHB-MIT.

**Directly relevant warning:** at least one multi-dataset study reports that **adding Siena
improved seizure detection rate but substantially increased false positives per day.** Given
that FP/24h inflation is this project's key unreported harm axis, that interaction must be
pre-registered *before* Tier C is run.

**Practical:** the SzCORE benchmark already publishes CHB-MIT, Siena, TUH (train/dev/eval)
and SeizeIT1 in a standardized converted format with a defined ML task. Reusing it likely
makes `chbmit/siena.py` redundant and guarantees comparability. See also **[summary]**
*PySeizure: a single machine learning classifier framework to detect seizures in diverse
datasets* — arXiv:2508.07253.

---

## 6. Generator evaluation: retire discriminator-AUC

The observed saturation of real-vs-synthetic discriminator AUC at 1.0 is a known dead end.
**[summary]** Current practice for generative time-series / EEG evaluation:

- **TSTR (train-on-synthetic, test-on-real)** — the standard functional utility metric.
- **Channel-covariance Frobenius distance** — probes multichannel spatial coupling, the
  failure mode ("off-manifold" synthetic) that TGA identifies as driving harm.
- **ACF-based distance** — temporal dependence structure.
- 1-NN accuracy in learned feature space as a pragmatic C2ST variant.

This project already has the strongest form of evaluation — downstream utility on real
held-out patients. Adding TSTR and covariance distance, and dropping discriminator AUC, is a
straightforward upgrade to `synthetic/quality_checks.py`.

---

## 7. Venue and publication norms

- **[summary]** **NeurIPS 2026 Evaluations & Datasets track** explicitly welcomes *"negative
  results, critical analyses, methodological analyses"* and states that a submission *"need
  not beat a baseline"* — its stated purpose is to deepen understanding of evaluation
  practice. This fits the harm-characterization + gate-conservatism framing better than the
  "significant TCN benefit" framing, and does not depend on the contested p-value.
- **[summary]** Pre-registration precedent in ML, to cite so `PREREGISTRATION.md` reads as
  methodologically literate rather than idiosyncratic: the NeurIPS pre-registration workshop
  (2020 pilot, 2021 workshop); *Pre-registration for Predictive Modeling* (arXiv:2311.18807);
  *Position: Embracing Negative Results in Machine Learning* (arXiv:2406.03980).

---

## 8. Prioritized actions arising from this review

| # | Action | Cost | Why |
|---|---|---|---|
| 1 | **Re-implement admission as a rank cut on the candidate pool** (top `1−q`), then re-run the `q` sweep | ~half day | §2.1.1 + §3 converge here; makes `q = 0.99` reachable and fixes the 174× ill-conditioned knob at the root |
| 2 | Add the SzCORE Challenge comparison to `reports/comprehensive_report.md` | ~1 h | Turns "our numbers look low" into "our numbers match the best independently-validated benchmark" |
| 3 | Re-run the headline under TGA's `min_admitted = 200` | 1 GPU-night | The result a knowledgeable reviewer will demand; 8/9 headline cells fall below this floor |
| 3b | Temperature-scale the teacher (if retaining a real-relative threshold) | ~half day | Secondary to #1; only needed if the real-ictal calibration is kept deliberately |
| 4 | Apply the Nadeau–Bengio correction to the 3×3 design | ~1 h | Keeps the analysis honest without discarding the effect |
| 5 | Lead the narrative with the TGA replication, not the +0.083 | writing | Robust, genuinely novel, independent of nine correlated cells |
| 6 | Pre-register the FP/24h axis before Tier C | ~1 h | Published evidence says adding Siena inflates FP/day |
| 7 | Swap discriminator-AUC for TSTR + covariance distance | ~1 day | Current standard; the saturated metric is uninformative |

---

## 9. Reference list

| Ref | Verified | Use |
|---|---|---|
| SzCORE Challenge — arXiv:2505.18191 | **[full text]** | Benchmark comparison (§1.1) |
| SzCORE framework — arXiv:2402.13005; *Epilepsia* 10.1111/epi.18113 | **[summary]** | Scoring standard, tolerances |
| CHB-MIT overlooked perspectives — PMC11286169 | **[full text]** | Why literature numbers are inflated (§1.2) |
| TGA — **npj Digit. Med. 9(1) 634, 10.1038/s41746-026-02778-0** (preprint: bioRxiv 10.64898/2026.01.26.701638) | **[full text + Crossref/EuropePMC]** | Prior work; novelty & divergences (§2) |
| EEG data partitioning — arXiv:2505.13021 | **[full text]** | Seed/fold variance practice (§4) |
| Nadeau & Bengio; `correctR` | **[summary]** | Dependent-fold correction (§4) |
| Guo et al., calibration (ICML 2017) | **[summary]** | Teacher miscalibration (§3) |
| PySeizure — arXiv:2508.07253 | **[summary]** | Multi-dataset framework (§5) |
| NeurIPS Evaluations & Datasets track (2026) | **[summary]** | Venue (§7) |
| Pre-registration for Predictive Modeling — arXiv:2311.18807 | **[summary]** | Pre-registration precedent (§7) |
| Embracing Negative Results in ML — arXiv:2406.03980 | **[summary]** | Framing (§7) |

---

## 10. Reconciliation with `reports/related_work.md`

Two untracked files covering similar ground were already present in the working tree when
this review was written. They **disagree with each other** on TGA's publication status, and
one of them is wrong in a way that would damage a submission.

### 10.1 The citation dispute — RESOLVED AGAINST THIS FILE (corrected 2026-08-28)

> **This section was wrong. The correction below supersedes it.** An earlier draft of §10.1
> asserted that TGA is preprint-only and instructed the reader not to cite the *npj Digital
> Medicine* version. **Do not follow that instruction.** `reports/related_work_v2.md` is correct.

**Cite the peer-reviewed version of record:**

> Choi, D.; Yip, C.; Choi, A.; Park, J. (2026). *Trust-gated synthetic EEG augmentation reduces
> performance drops when generalizing to new patients.* **npj Digital Medicine 9(1), art. 634**,
> `10.1038/s41746-026-02778-0`, published **2026-05-25**. PMID **42185473**.

Verified independently at two authoritative sources:

| Check | Result |
|---|---|
| **Crossref** `api.crossref.org/works/10.1038/s41746-026-02778-0` | **Resolves.** npj Digital Medicine 9(1), art. 634, Springer, published 2026-05-25, CC BY-NC-ND 4.0. Authors Choi, Yip, Choi, Park. |
| **Europe PMC** REST search by DOI | **1 hit.** Same title/journal/authors, 2026, PMID 42185473. |
| bioRxiv details API for `10.64898/2026.01.26.701638` | Returns the record with `published: NA` — the preprint exists, and its `published` field is stale. |

**Why the earlier check failed, and the method lesson.** The two negatives that produced the
wrong conclusion were both weak: a title search that missed, and `doi.org/10.1038/...`
redirecting to a Nature authentication wall (an auth wall is not evidence of absence). The
bioRxiv `published` field is frequently stale and **is not evidence of preprint status**.
Resolve DOIs through Crossref and Europe PMC before asserting that anything is unpublished.

The earlier claim that the bioRxiv DOI prefix `10.64898` is invalid is *also* wrong — the
preprint is real. Both records exist; the npj article is the version of record. The
`github.com/danielchoi0315/TGA-repo` code link asserted for the method remains unverified.

**Consequence for positioning:** TGA is unambiguously prior art in a high-visibility venue.
"Propose"/"novel" framing for the gate is off the table, and implementation fidelity to the
published specification becomes a live referee question.

### 10.2 Its analytical claims — one overstated, one probably right

- **"The cells the gate reverted had a *higher* mean benefit than the cells it kept."**
  Checked against `downstream_gated_multiseed.csv`:

  | detector / q | kept (n, mean Δ) | reverted (n, mean Δ) | claim holds? |
  |---|---|---|---|
  | TCN q=0.90 | 4, +0.076 | 5, +0.089 | **yes** (headline condition) |
  | TCN q=0.50 | 2, +0.078 | 7, +0.039 | no |
  | EEGNet q=0.90 | 1, +0.101 | 8, −0.017 | no |
  | EEGNet q=0.50 | 2, +0.096 | 7, −0.042 | no |

  True in the headline cell only, on n=4 vs n=5. Stated generally in that file; it is not
  general.

- **"The admission threshold is calibrated on the wrong distribution."** Independently
  supported — see §2.1.1. This is the most valuable claim in either file.

### 10.3 Recommendation

Superseded. `reports/related_work_v2.md` is the positioning document of record (it carries
the correct citation); `reports/related_work.md` is retained only as a search log with
provenance. The implementation audit behind §10.1–§10.2 stands: its citation correction is
**right** (§10.1 above) and its code-level findings were independently re-verified against the
repository; only its "dose-response" reading needs the qualification in §10.2.
