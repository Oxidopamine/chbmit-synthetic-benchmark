# Does Synthetic Ictal EEG Help or Harm Seizure Detection?
### A Leakage-Safe, Harm-First Benchmark on CHB-MIT — Comprehensive Report

*Machine Learning for Biomedical Signals. Report date: 3 July 2026.*
*Status: baselines + core experiment complete; generator-fidelity, gate-mechanism, and downstream gated experiments completed on fold 0 (single seed). Multi-seed / multi-dataset validation pending.*

---

## How to read this report

This document is written to be understood by a reader who is **not** a specialist in
machine learning or epilepsy. Technical terms are defined the first time they appear,
and a **glossary** at the end collects them all. Each results section states first, in
plain language, *what question it answers and what we found*, then gives the numbers and
a discussion. Nothing here requires you to take a claim on faith: every number is
produced by scripts in this repository from the raw public data.

---

## 1. Executive summary (plain language)

An epilepsy monitor is a computer that watches a patient's brain-wave recording (an
**EEG**) and sounds an alarm when a seizure begins. Training such a detector is hard
because seizures are *rare*: in our data, only about **0.3%** of the recording is
seizure. A model shown mostly non-seizure data tends either to miss seizures or to cry
wolf constantly.

A popular idea to fix this is **synthetic data augmentation**: train a second AI (a
"generator") to manufacture fake but realistic seizure snippets, and add them to the
training set to give the detector more seizures to learn from. Much of the published
literature reports that this helps. This project asks a more skeptical question, in a
deliberately honest experimental setup:

> **Does adding synthetic seizures actually help a detector recognise seizures in
> patients it has never seen — or can it quietly cause harm? And if it can harm, can a
> simple safety mechanism prevent the harm while keeping any benefit?**

Our answers, in one paragraph:

1. **Naive synthetic augmentation is unreliable and can cause real harm.** In our core
   experiment it helped some detectors and hurt others; in the worst cases it degraded
   performance substantially and multiplied false alarms.
2. **A "fail-closed trust gate" — a gatekeeper that only admits synthetic data if it
   provably does not worsen validation performance — removes the downside.** When the
   synthetic data is poor, the gate declines it and the system safely falls back to
   using only real data.
3. **Making the generator better is genuinely hard, and *why* it is hard is subtle.**
   Two very different generator designs both fail a naive "realism" test — but for
   different reasons, and the standard realism *metrics* turn out to be misleading.
4. **With a stronger generator whose output is cleaned up, we found the first case where
   the gate admitted synthetic data and the detector improved** (on one detector, one
   data split). This is a promising but fragile signal that must be confirmed across
   more random seeds and patients before it can be believed.

---

## 2. Background: the problem and why it is hard

### 2.1 What the detector is doing

An EEG records tiny electrical voltages from electrodes on the scalp — in our data,
**18 channels** (electrode pairs) sampled **256 times per second**. The detector reads
this stream in short **windows** (here, 4-second windows taken every 2 seconds) and, for
each window, outputs a probability that a seizure is occurring.

### 2.2 The central difficulty: class imbalance

Because seizures are rare, the training data is extremely lopsided — 99.7% "background"
(non-seizure), 0.3% "ictal" (seizure; *ictal* is the medical term for the seizure state).
A model can score 99.7% "accuracy" by never predicting a seizure at all, which is useless.
This is the **class-imbalance** problem.

### 2.3 Two things the augmentation literature usually gets wrong

**(a) Data leakage.** EEG from the *same patient* is highly recognisable. If windows from
one patient appear in **both** the training set and the test set, the model can succeed by
recognising the patient rather than the seizure. Reported accuracy is then inflated and
does not reflect performance on a genuinely new patient. Avoiding this requires a
**patient-independent** (also called **leave-patients-out**) protocol: no patient is ever
split across training and test. Much of the literature does not enforce this.

**(b) Silent harm.** A generator can produce fake seizures that are unrealistic or too
similar to each other. Training on them can *degrade* the detector — and often the damage
is uneven, hurting some patients while the average still looks acceptable. Averages hide
this. We therefore report **tail risk**: the worst cases, not just the mean.

### 2.4 The idea we test: a fail-closed trust gate

Rather than trusting synthetic data blindly, we borrow a safety principle from
engineering: **fail closed**. A gatekeeper ("trust gate") admits synthetic seizures into
training **only if** the resulting detector demonstrably does not do worse on a
held-out **validation** set (and does not raise false alarms beyond a small
pre-registered tolerance). If those conditions are not met, it admits nothing and reverts
to a detector trained on real data only. In the worst case the gate does *nothing* — but
it should never make things worse.

---

## 3. Research questions

- **Q1 — Baseline reality.** How well do standard detectors perform on unseen patients
  using only real data, under honest event-level scoring?
- **Q2 — Help or harm.** Does synthetic augmentation improve or degrade detection versus
  simple, safe baselines?
- **Q3 — Mitigation.** Does the fail-closed trust gate prevent the harmful cases while
  preserving gains?
- **Q4 — Can we make it help?** If the standard generator is too weak to clear the gate,
  can a stronger generator (better realism, cleaned-up output) produce synthetic data
  that the gate admits *and* that actually improves detection?

Questions Q1–Q3 were addressed by the baseline and core experiments (Sections 6–7).
Question Q4 is the focus of the new work reported here (Sections 8–10).

---

## 4. Data

We use the **CHB-MIT Scalp EEG Database** (Children's Hospital Boston / MIT), a public
dataset on PhysioNet. After a channel-consistency audit that keeps a fixed 18-channel
bipolar montage (a standard set of electrode pairings):

- **24 patients** (treated as **23 leakage-safe groups**; patients chb01 and chb21 are the
  same individual and are merged so they can never be split across train and test),
- **673 recordings**, **198 annotated seizures**, ~**974 hours** of EEG,
- signals **band-pass filtered to 0.5–40 Hz** (a standard clinical frequency range) with a
  zero-phase filter, and resampled to **256 Hz**,
- windowed into **1,746,447** four-second windows, of which only **5,563 (0.32%)** are
  seizure.

**Band-pass filtering** removes very slow drifts (below 0.5 Hz) and high-frequency noise
(above 40 Hz). This detail becomes important later (Section 8): real EEG has essentially
*no* power above 40 Hz because the filter removed it, and a good generator must reproduce
that.

---

## 5. Methods

### 5.1 Leakage-safe splitting

We use **grouped 5-fold cross-validation** at the patient level. The patients are divided
into 5 "folds"; in each fold, some patients are used for training, some for validation
(tuning), and some for testing (final scoring), with **zero patient overlap** between the
three. We verified programmatically that every patient group is tested exactly once and
never leaks across the split.

- **Training set** — patients the detector learns from.
- **Validation set** — held-out patients used to *tune* decisions (e.g. the alarm
  threshold, and the trust gate's admit/reject choice). Never used for final scoring.
- **Test set** — held-out patients used *only* for the final reported numbers.

### 5.2 Detectors (the models being trained)

Three neural-network detectors spanning a wide range of size, so conclusions are not tied
to one architecture:

- **EEGNet** — a very small network (~1,900 parameters). "Parameters" are the tunable
  numbers a network learns; more parameters means more capacity but higher overfitting
  risk.
- **LCT** — a mid-size convolutional-transformer (~121,000 parameters).
- **TCN** — a temporal convolutional network (a model specialised for time series).

### 5.3 Generators (the models that fabricate synthetic seizures)

Both are trained **patient-independently** — only on each fold's *training* patients — so
synthetic data never carries information about test patients.

- **cVAE (conditional Variational Autoencoder).** A network that learns to compress real
  seizure windows into a compact "latent code" and reconstruct them, after which it can
  sample new codes to generate new windows. VAEs are stable but tend to produce **blurry,
  over-smoothed** output.
- **WGAN-GP (Wasserstein GAN with Gradient Penalty).** A **generative adversarial
  network**: a generator and a critic play a game — the generator tries to fool the critic
  into thinking its fakes are real, the critic tries to tell them apart. GANs can produce
  sharper output but are trickier to train.

### 5.4 How we score detectors: event-level, harm-first

Scoring at the level of individual 4-second windows is misleading clinically. What matters
is whether the detector catches the **seizure event** and how often it **falsely alarms**.
We use **SzCORE**-aligned **event-level** metrics:

- **Event sensitivity** — fraction of real seizures the detector catches (higher is
  better).
- **Event precision** — of the alarms it raises, the fraction that are real seizures.
- **Event-F1** — a single balanced score combining sensitivity and precision (0 = useless,
  1 = perfect). This is our headline metric.
- **FP/24h — false alarms per 24 hours** — how often it cries wolf per day (lower is
  better; critical clinically, because alarm fatigue makes a monitor unusable).

We also report **tail risk** (worst-fold outcomes, harm rates), not just averages, because
silent harm hides in the tail.

For reference, **window AUROC** (area under the ROC curve at the window level) measures raw
discrimination ability at the window level; it often looks healthy even when event-level
performance is poor — which is exactly why we score at the event level.

### 5.5 The trust gate, precisely

The gate has two stages:

1. **Admission (window level).** Score every candidate synthetic window with the
   **real-only detector as a "teacher"**: how confident is the real-trained detector that
   this window is a seizure? Admit a synthetic window only if the teacher's confidence
   exceeds the **q-th quantile** of the teacher's confidence on *real* seizure windows.
   `q` is a strictness dial: `q = 0.90` means "a synthetic window must look at least as
   seizure-like to the teacher as the top 10% of real seizures." Higher `q` = stricter.
   *(Note: the gate keys on the teacher's confidence, not on any separate "realism"
   score — a point that matters in Section 9.)*
2. **Fail-closed selection (event level).** Train the detector on real + admitted
   synthetic, then compare its **validation** event-F1 and false-alarm rate to the
   real-only detector. Keep the augmented model only if it does not worsen validation
   event-F1 and does not raise validation false alarms beyond a small pre-registered
   tolerance. Otherwise **revert** to the real-only model.

All thresholds are **pre-registered** (fixed in advance, in `PREREGISTRATION.md`) so we
cannot tune them after seeing results.

---

## 6. Results — Q1: real-only baselines (Tier A)

**Plain-language finding:** even with real data only, detecting seizures in *new* patients
is genuinely hard — far from solved — and the difficulty varies a lot from patient to
patient.

Mean ± standard deviation over 5 folds (each fold tests entirely held-out patients):

| Detector / condition | Event-F1 | Event Sens. | Event Prec. | FP / 24h | Win AUROC |
|---|---|---|---|---|---|
| **EEGNet** real_only | 0.179 ± 0.12 | 0.596 | 0.165 | 60.6 | 0.768 |
| EEGNet class_weighted | 0.117 ± 0.11 | 0.578 | 0.129 | 64.6 | 0.808 |
| EEGNet classical_aug | 0.206 ± 0.13 | 0.569 | 0.188 | 45.4 | 0.784 |
| **LCT** real_only | 0.246 ± 0.11 | 0.609 | 0.282 | 43.8 | 0.800 |
| LCT class_weighted | 0.295 ± 0.04 | 0.577 | 0.287 | 34.1 | 0.805 |
| LCT classical_aug | 0.230 ± 0.12 | 0.720 | 0.196 | 58.7 | 0.808 |

*("class_weighted" = penalise missing seizures more heavily during training;
"classical_aug" = simple label-preserving signal augmentations, e.g. jitter/scaling — not
generative.)*

**Discussion.** Event-F1 of 0.12–0.30 with 34–65 false alarms per day is modest, even
though window-AUROC looks healthy (~0.77–0.81). This gap is the whole reason we score at
the event level. The large between-fold spread (±0.11) reflects genuine per-patient
difficulty and motivates reporting tail risk.

---

## 7. Results — Q2 & Q3: help/harm and the gate (Tier B core)

**Plain-language finding:** naive synthetic augmentation was a coin-flip — it helped one
detector, was neutral for another, and hurt the third (with a jump in false alarms). The
trust gate, faced with a weak generator, **declined all synthetic data 100% of the time**
and safely reproduced real-only performance. A cheap, safe baseline (class weighting)
often beat synthetic augmentation.

All 225 model-training cells completed (5 folds × 3 detectors × 3 scarcity levels × 5
conditions), plus 15 generator fits. Values are mean event-F1 pooled over folds and
scarcity levels, single seed (42), with the **cVAE** generator.

**Table 7a — mean event-F1 by detector × condition (higher is better):**

| Detector | real_only | class_weighted | classical_aug | ungated synth | trust-gated synth |
|---|---|---|---|---|---|
| EEGNet | 0.206 | 0.190 | 0.164 | **0.253** | 0.206 |
| LCT | 0.212 | **0.267** | 0.188 | 0.226 | 0.212 |
| TCN | 0.167 | **0.237** | 0.241 | 0.162 | 0.167 |

**Table 7b — mean false alarms / 24h (lower is better):**

| Detector | real_only | class_weighted | classical_aug | ungated synth | trust-gated synth |
|---|---|---|---|---|---|
| EEGNet | 43.8 | 60.5 | 43.4 | **42.3** | 43.8 |
| LCT | 47.8 | 34.9 | 69.8 | **36.3** | 47.8 |
| TCN | 61.8 | 48.0 | 39.9 | **82.2** | 61.8 |

**Three findings.**

- **Silent harm is real.** Naive (ungated) synthetic helped EEGNet (0.253 vs 0.206) but
  hurt TCN (0.162 vs 0.167, with false alarms rising 61.8 → 82.2/day). In the worst fold,
  ungated synthetic cost up to −0.43 event-F1 versus the best baseline. This is exactly
  the uneven, hidden harm the project warns about.
- **The safety net works.** The gate **failed closed 100% of the time** — across all 45
  gated cells it admitted **zero** synthetic windows, so gated performance is identical to
  real-only. No synthetic-induced harm got through.
- **A simple baseline often wins.** Plain class weighting was best for LCT (0.267) and
  strong for TCN (0.237), beating naive synthetic augmentation. Synthetic data has to earn
  its place.

**The open question this raised.** The gate declined *everything* because the cVAE was
low-fidelity. So the gate removed the downside but delivered no upside. The natural next
question (Q4): **can a better generator produce synthetic data good enough to clear the
gate and actually help?** Everything below (Sections 8–10) investigates that.

---

## 8. Results — Q4, part 1: why is the generator's output "obviously fake"?

**Plain-language finding:** we measured *how* fake the synthetic seizures look, and made
two discoveries. First, the WGAN-GP generator actually reproduces the realistic frequency
content of seizures in the physiological band (0.5–40 Hz) **almost perfectly** — far
better than the cVAE. Second, the standard "realism score" everyone uses **saturates**: it
screams "fake!" with equal certainty for a nearly-perfect generator and a terrible one, so
it cannot tell them apart. The real, fixable flaw is that both generators add spurious
high-frequency noise above 40 Hz that real (filtered) EEG does not have.

### 8.1 The metrics we used

To judge a generator's output we compare its synthetic windows to real seizure windows on:

- **Discriminator AUC** — train a simple classifier to tell real from synthetic. If it
  scores **0.5**, real and synthetic are indistinguishable (ideal). If it scores **1.0**,
  synthetic is trivially recognisable as fake.
- **MMD (Maximum Mean Discrepancy) on frequency features** — a distance between the real
  and synthetic distributions of power across frequencies; lower is closer.
- **Diversity ratio** — average spread among synthetic windows ÷ spread among real ones.
  Near **1.0** is healthy; near **0** means **mode collapse** (the generator outputs
  near-identical windows).
- **Power spectrum (PSD)** — how a signal's energy is distributed across frequencies. Real
  EEG has characteristic spectral shape; a good generator should match it.

### 8.2 Training the WGAN longer does not fix "fakeness" — but it does not collapse

Sweeping WGAN-GP training length on fold 0 (single checkpointed run):

| epochs | discriminator AUC | MMD (freq) | diversity ratio |
|---|---|---|---|
| 50  | 1.000 | 1.203 | 1.003 |
| 150 | 1.000 | 1.174 | 0.969 |
| 300 | 1.000 | 1.163 | 0.939 |
| 450 | 1.000 | 1.149 | 0.936 |
| 600 | 1.000 | 1.139 | 0.936 |

Compare the cVAE (its diversity **collapses** to ~0.001 — it outputs almost the same
window every time):

| cVAE epochs | discriminator AUC | MMD (freq) | diversity ratio |
|---|---|---|---|
| 150 | 1.000 | 1.324 | 0.002 |
| 1500 | 1.000 | 1.338 | 0.001 |

**Discussion.** The discriminator AUC is pinned at **1.000** for both generators at every
setting — apparently "both are equally, totally fake." But the WGAN's diversity is healthy
(~0.94) while the cVAE's has collapsed (~0.001), and the WGAN's MMD is lower and improving.
So the AUC is hiding a real quality difference. The WGAN's problem is *not* mode collapse.

### 8.3 The spectral-gap diagnostic: where the fakeness lives

We trained both generators (fold 0) and compared the **average power spectrum** of real
vs synthetic seizure windows, frequency by frequency
(`analysis_tierB/figures/spectral_gap.png`).

| quantity | WGAN-GP | cVAE (β=0.01) | real |
|---|---|---|---|
| in-band (0.5–40 Hz) mean log-PSD gap to real | **0.157** | 2.66 | — |
| out-of-band (>40 Hz) mean log-PSD gap to real | ~12.4 | ~9 | — |
| fraction of power above 40 Hz | 0.0072 | 0.0049 | 0.0022 |
| discriminator AUC using **only** 0.5–40 Hz features | 1.000 | 1.000 | — |

**Discussion — three results here, each important:**

1. **The WGAN reproduces the physiological seizure spectrum almost perfectly.** Its
   in-band gap (0.157) is ~17× smaller than the cVAE's (2.66). In the figure, the WGAN's
   spectrum sits right on top of the real spectrum across the whole 0.5–40 Hz band; the
   cVAE is visibly wrong. So in the frequency range that matters clinically, the WGAN is a
   genuinely good generator.
2. **Both generators fail to reproduce the filter roll-off above 40 Hz.** Real EEG was
   band-pass filtered, so it has almost no power above 40 Hz (its spectrum falls ~10 orders
   of magnitude). Both generators instead emit a flat "floor" of high-frequency noise. This
   out-of-band mismatch is enormous (gap ~12) and is the single largest real-vs-synthetic
   discrepancy. **Crucially, this is fixable:** just filter the synthetic output the same
   way the real data was filtered (Section 10).
3. **The discriminator AUC is saturated and cannot grade quality.** Even restricted to the
   in-band features (where the WGAN is nearly perfect), the AUC is still 1.000. Why? A
   classifier with many frequency features and thousands of examples can exploit *tiny but
   consistent* differences to separate the two distributions perfectly — even when they are
   nearly identical. So "AUC = 1.0" does **not** mean "grossly unrealistic"; it just means
   "separable." This is a genuine **methodological lesson**: common synthetic-fidelity
   metrics can be uninformative for judging *how* good a generator is.

---

## 9. Results — Q4, part 2: what the gate actually reacts to

**Plain-language finding:** we discovered that the gate never used the "realism score"
(discriminator AUC) at all — that was only a diagnostic we reported. The gate's real
gatekeeping is whether the **real-trained detector recognises the synthetic windows as
seizures**. We then measured that recognition directly, and found the fixes from Section 8
make the synthetic windows much more recognisable as seizures.

### 9.1 A correction to how we understood the gate

In the core experiment we had loosely attributed the gate's refusals to the low
"discriminator AUC" realism score. Reading the implementation carefully, that is **not**
what the gate keys on. The gate reverted 100% of the time because its **stage-1 admission
rate was 0**: the real-only "teacher" detector scored **every** synthetic window below the
confidence quantile it assigns real seizures. In plain terms: *the detector trained on
real seizures did not recognise the fake seizures as seizures*. The fix therefore has to
be generator-side (make the fakes more recognisable), not a change to the realism metric.

### 9.2 Measuring recognisability, with the two fixes

We trained a faithful real-only EEGNet "teacher" on fold 0 (it cleanly separates real
seizures, mean confidence **0.850**, from background, **0.048**), then measured the
teacher's seizure-confidence on synthetic windows from three generators. **Fix 1** =
band-limit the WGAN output to 0.5–40 Hz (remove the out-of-band noise from Section 8).
**Fix 2** = use the WGAN instead of the cVAE.

| generator | teacher's mean seizure-confidence | admission rate at q=0.90 |
|---|---|---|
| cVAE (β=0.01) | 0.475 | 0.0% |
| WGAN raw | 0.666 | 1.1% |
| **WGAN band-limited** | **0.774** | 0.1% |
| *real seizures (reference)* | *0.850* | — |

And how admission responds to the strictness dial `q`
(`analysis_tierB/gate_admission_qsweep.csv`; figure `figures/gate_admission.png`):

| q (strictness) | threshold | cVAE | WGAN raw | WGAN band-limited |
|---|---|---|---|---|
| 0.50 | 0.961 | 1.2% | 21.8% | **17.4%** |
| 0.75 | 0.992 | 0.03% | 6.6% | 2.2% |
| 0.90 (pre-registered) | 0.9985 | 0.0% | 1.1% | **0.1%** |

**Discussion.** Recognisability rises monotonically: cVAE (0.475) < WGAN-raw (0.666) <
WGAN-band-limited (0.774) → approaching real (0.850). Both fixes work as intended: the
band-limited WGAN's fake seizures look *nearly as seizure-like to the detector as real
ones*. However, the **pre-registered strictness (q = 0.90) is very demanding** — its
threshold (0.9985) sits in the extreme top tail of real-seizure confidence — so at that
setting almost nothing is admitted. Relaxing to `q = 0.50` (the median real-seizure
confidence) admits a substantial ~17% of the band-limited WGAN's windows, enough to
actually augment training. (A subtlety: band-limiting *raised* average recognisability but
slightly *lowered* admission at the extreme q = 0.90, because the strict threshold rewards
a few very-high-confidence outliers that the raw WGAN happens to produce. This mirrors the
metric-saturation lesson: the admission rate at extreme strictness is not a clean measure
of overall quality.)

---

## 10. Results — Q4, part 3: does admitted synthetic data actually help?

**Plain-language finding:** we ran the full detector-training-and-scoring pipeline on fold
0 with the improved (band-limited WGAN) generator, across all three detectors. Injecting
synthetic data *naively* harmed **all three** detectors — sometimes badly. The gate
correctly blocked the harm where the synthetic data was bad. And in **one** case (the TCN
detector at the pre-registered strictness) the gate **admitted** the synthetic data and the
detector **improved** on both accuracy and false alarms — the first genuine "upside" case.
This is encouraging but is a single-seed, single-split result and could be noise.

Definitions for the table: **"effective"** columns are what the system actually delivers
(for a reverted gated cell, that is the real-only model). **"aug"** columns are the model
that was actually trained on the (admitted) synthetic data — this tells us whether the
synthetic *would* help, regardless of the gate's decision.

| detector | condition | q | effective F1 | FP/24h | **aug F1** | admitted windows | gate decision |
|---|---|---|---|---|---|---|---|
| **EEGNet** | real_only | — | 0.380 | 2.3 | 0.380 | — | — |
| | ungated (naive) | — | 0.256 | 7.3 | 0.256 | — | injected all |
| | gated | 0.90 | 0.380 | 2.3 | 0.099 | 25 (0.2%) | **reverted** |
| | gated | 0.50 | 0.380 | 2.3 | 0.225 | 2508 (17%) | **reverted** |
| **LCT** | real_only | — | 0.130 | 33.2 | 0.130 | — | — |
| | ungated (naive) | — | 0.092 | 32.1 | 0.092 | — | injected all |
| | gated | 0.90 | 0.130 | 33.2 | 0.106 | 202 (1.3%) | **reverted** |
| | gated | 0.50 | 0.130 | 33.2 | 0.078 | 2508 (17%) | **reverted** |
| **TCN** | real_only | — | 0.174 | 12.4 | 0.174 | — | — |
| | ungated (naive) | — | 0.097 | 69.5 | 0.097 | — | injected all |
| | **gated** | **0.90** | **0.204** | **8.3** | **0.204** | 126 (0.8%) | **ADMITTED** |
| | gated | 0.50 | 0.174 | 12.4 | 0.229 | 2508 (17%) | reverted |

**Discussion.**

- **Naive injection harms every detector.** EEGNet 0.380 → 0.256; LCT 0.130 → 0.092; TCN
  0.174 → 0.097 — and TCN's false alarms explode (12 → 69 per day). Even with the *good*
  (spectrally-realistic, diverse, band-limited) generator, dumping synthetic data into
  training uncritically is harmful. This strongly reinforces the project's central warning.
- **The gate prevents harm, and its decision generalises.** For EEGNet and LCT the
  augmented models are worse (the "aug F1" columns are below real_only), the gate reverts
  based on **validation**, and that revert is *correct* — it protects **test** performance.
  This confirms the fail-closed mechanism does what it promises.
- **The first upside case: TCN at the pre-registered q = 0.90.** The gate admitted a
  small, high-confidence set of band-limited WGAN windows (126 windows), the augmented
  model beat the teacher on validation (so it was **not** reverted), and on the held-out
  **test** set it improved on *both* axes: event-F1 **0.174 → 0.204** and false alarms
  **12.4 → 8.3 per day**. This is the paper's dream cell — a good-enough generator producing
  synthetic data that clears the *pre-registered* gate and safely helps.
- **The gate is conservative, sometimes to a fault.** At q = 0.50, the TCN augmented model
  *would* have helped on test (0.229 > 0.174), but the gate reverted because the gain did
  not show up on validation. So the gate both catches harm (good) and can occasionally
  reject benefit (a cost). Both directions appear within a single detector.

---

## 11. Overall discussion — what we have learned

1. **Synthetic ictal augmentation should be treated as a hazard, not a free lunch.** Across
   the core experiment and the new downstream experiment, naive injection is unreliable and
   frequently harmful — even when the generator is spectrally realistic and diverse. The
   "silent harm" this project set out to expose is real and reproducible.
2. **The fail-closed trust gate is an effective safety mechanism.** It converts an
   unpredictable, occasionally-harmful intervention into one that is *never worse than doing
   nothing*, and its validation-based decisions generalise correctly to held-out test
   patients. For a clinical setting, "never worse" is a valuable guarantee.
3. **Making the generator good enough is the crux, and standard fidelity metrics can
   mislead.** The discriminator-AUC "realism score" saturates and cannot distinguish a
   nearly-perfect generator from a poor one; a diversity metric distinguishes collapse from
   health; the spectral analysis localises the true, *fixable* defect (out-of-band noise).
   The practical lesson for the field: **do not judge synthetic-EEG fidelity by a single
   saturating score.**
4. **With a better, cleaned-up generator, benefit is possible.** The TCN q = 0.90 cell is
   the first demonstration that a generator can produce synthetic seizures that clear the
   pre-registered safety gate *and* improve a real detector. It reframes the project's story
   from purely cautionary toward "safe *and*, under the right conditions, beneficial."

---

## 12. Limitations (read before believing anything positive)

- **Single seed, single fold for the deep dives (Sections 8–10).** The TCN upside (+0.03
  event-F1) is comfortably within the between-fold noise observed in the baselines (±0.11
  standard deviation). **It is a signal to chase, not a result to claim.** It must be
  reproduced across multiple random seeds and multiple patient folds before it can be
  believed.
- **One dataset (CHB-MIT), which is small and dated.** Generality to other populations and
  recording setups is untested. A second dataset (Siena is scoped; the larger TUH EEG corpus
  would be stronger) is essential for any strong claim.
- **The trust gate is adapted from prior work**, not novel; the contribution is its
  seizure-specific, event-level reformulation and the harm-first evaluation around it.
- **The improved generator was validated in the frequency domain**, which does not capture
  every aspect of realism (e.g. fine temporal morphology, cross-channel relationships beyond
  what we checked).

---

## 13. Conclusions and next steps

**Conclusion.** On unseen patients, seizure detection is far from solved; naive synthetic
augmentation does not reliably help and can quietly harm; a fail-closed trust gate removes
that downside; and — for the first time in this project — a stronger, spectrally-realistic,
band-limited generator produced synthetic data that cleared the pre-registered gate and
improved one detector. Whether that benefit is real or noise is the pivotal open question.

**Decisive next experiment.** Repeat the downstream gated experiment across **multiple
random seeds and all patient folds** (the pre-registration specifies three seeds), and
report the gated-vs-real_only difference with confidence intervals and a paired
significance test. This single experiment determines which paper this becomes:

- If a TCN-style upside **survives** replication → a positive result: *a good-enough
  generator plus a fail-closed gate can make synthetic augmentation safe and beneficial.*
- If it **evaporates** under seeds → a clean, rigorous cautionary result: *even a
  near-realistic generator does not reliably help, and the gate's conservatism is warranted.*

Both outcomes are honest and publishable. Secondary next steps: a second dataset (Siena/TUH)
for generality, and stricter leave-one-patient-out validation.

**A note on the engineering that made this feasible.** The processed EEG store lives on a
network filesystem with high per-read latency, which made the full training/validation/test
pipeline impractically slow. We added an opt-in, per-file **parallel prefetch cache** so the
pipeline runs from RAM (~40× faster; inert unless activated), and we **band-limit** synthetic
output using the exact preprocessing filter. Both are small, reusable additions that make the
multi-seed/multi-fold confirmation runs tractable.

---

## 14. Glossary

- **AUROC / window AUROC** — a 0.5-to-1.0 score for how well the detector separates seizure
  from non-seizure windows; 0.5 = chance, 1.0 = perfect.
- **Band-pass filter / band-limit** — keep only frequencies in a chosen range (here
  0.5–40 Hz), removing slow drift and high-frequency noise.
- **Class imbalance** — one class (seizure) is vastly rarer than the other (background).
- **cVAE** — conditional Variational Autoencoder; a stable but blur-prone generator.
- **Discriminator AUC** — how easily a classifier tells real from synthetic; 0.5 = ideal
  (indistinguishable), 1.0 = trivially fake.
- **Diversity ratio / mode collapse** — spread of synthetic samples vs real; near 0 means
  the generator repeats itself (collapse).
- **Event-F1 / sensitivity / precision** — event-level accuracy measures (Section 5.4).
- **Fail-closed** — a safety default: if not proven safe, do nothing.
- **FP/24h** — false alarms per 24 hours (lower is better).
- **Ictal** — the seizure state; **inter-ictal** = between seizures (background).
- **Leakage / patient-independent** — never letting one patient appear in both training and
  test; required for honest evaluation.
- **MMD** — Maximum Mean Discrepancy; a distance between two distributions (lower = closer).
- **Parameters** — the tunable numbers a network learns; a proxy for model capacity.
- **PSD (power spectrum)** — how signal energy is distributed across frequencies.
- **Quantile (q)** — the gate's strictness dial; the fraction of real seizures a synthetic
  window must out-score to be admitted.
- **Scarcity** — deliberately sub-sampling real seizures (to 100/50/25%) to study
  data-limited regimes.
- **Tail risk** — the worst-case outcomes, not the average.
- **WGAN-GP** — Wasserstein GAN with Gradient Penalty; a sharper but trickier generator.

---

## 15. Reproducibility and artifacts

All numbers derive from scripts in this repository, run on the public CHB-MIT dataset with
fixed seeds and pre-registered thresholds.

- Preprocessing: `scripts/run_preprocess.py`
- Core grid (Tiers A/B): `experiments/run_tierA_dev.py`, `experiments/run_tierB_core.py`
- Generator fidelity: `scripts/gen_wgan_fidelity_diag.py`, `scripts/gen_beta_sweep.py`
- Spectral-gap diagnostic: `scripts/gen_spectral_gap_diag.py`
  → `analysis_tierB/spectral_gap.{json,npz}`, `figures/spectral_gap.png`
- Gate admission re-test + q-sweep: `scripts/gate_admission_retest.py`
  → `analysis_tierB/gate_admission_retest.csv`, `gate_admission_qsweep.csv`,
  `figures/gate_admission.png`
- Downstream gated experiment: `scripts/run_downstream_gated.py`
  → `analysis_tierB/downstream_gated_bandlimited.csv`
- Band-limiting: `synthetic/band_limit.py`; window cache: `chbmit/datasets.py`
  (`prefetch_windows`)
- Trust gate: `synthetic/trust_gate.py`; pre-registered thresholds: `PREREGISTRATION.md`

*Bracketed author/course fields in the formal write-up are placeholders to complete before
submission. Metrics follow the SzCORE event-scoring convention. Deep-dive results
(Sections 8–10) are fold-0, single-seed and require multi-seed/multi-fold confirmation.*
