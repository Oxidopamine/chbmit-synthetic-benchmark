# Does Synthetic Ictal EEG Help or Harm Seizure Detection?
### A Leakage-Safe, Harm-First Benchmark on CHB-MIT — Comprehensive Report

> ## SUPERSEDED IN PART — read this first (2026-08-28)
>
> A full audit retired two of this report's headline claims. **Do not quote them.**
>
> 1. *"+0.083 event-F1, p = 0.008, statistically significant"* (this file at the Executive
>    Summary item 4, the multi-seed section, and the findings list). The direction is real
>    (8/9 cells) but the significance is not: **Nadeau-Bengio p = 0.135**, 95% CI
>    **[-0.032, +0.198]**, fold-level p = 0.088, and nothing survives Bonferroni over the
>    comparisons actually reported. It is also measured against `real_only`, whereas
>    `PREREGISTRATION.md` §3 registers *the best simple baseline per (fold, seed)* — worth
>    +0.163 event-F1 for TCN, about twice the claimed effect.
> 2. *"The gate is over-conservative"* / *"loosening the gate is the clear next lever"*. The
>    gate is a **false-alarm tail controller** with a weak event-F1 selector: admitted cells
>    stay within +6.07 FP/24h, blocked ones reach +99.88 (Fisher p = 0.0163 two-sided). The
>    lever is the admission *reference* and the *selector*, not q.
>
> Also: this report's harm rates are all measured against `real_only`, and a reverted cell *is*
> `real_only`, so its harm is zero **by construction**. Against the registered reference the
> same cells score harm 1.00 for TCN and LCT.
>
> Authority: `README.md`, then `reports/DECISION_GATE_1.md`.

*Machine Learning for Biomedical Signals. Report date: 4 July 2026.*
*Status: baselines + core experiment complete; generator-fidelity, gate-mechanism, single-fold and **multi-seed × multi-fold (n=9) downstream experiments complete**; a real-data positive-control run is in progress. Single dataset (CHB-MIT); second-dataset and leave-one-patient-out validation pending.*

---

## How to read this report

This document is written to be understood by a reader who is **not** a specialist in
machine learning or epilepsy. Technical terms are defined the first time they appear, and
a **glossary** (§15) collects them all. Each results section states first, in plain
language, *what question it answers and what we found*, then gives the numbers and a
discussion. Every number is produced by scripts in this repository from the public data.

---

## 1. Executive summary (plain language)

An epilepsy monitor is a computer that watches a patient's brain-wave recording (an
**EEG**) and alarms when a seizure begins. Training such a detector is hard because
seizures are *rare*: only ~**0.3%** of our data is seizure. A model shown mostly
non-seizure data tends either to miss seizures or to alarm constantly.

A popular remedy is **synthetic data augmentation**: train a generator to manufacture
fake-but-realistic seizure snippets and add them to training. Much of the literature
reports this helps. This project asks a more skeptical, safety-first question:

> **Does adding synthetic seizures actually help detection in patients the model has never
> seen — or can it quietly cause harm? If it can harm, can a simple gatekeeper prevent the
> harm while keeping any benefit? And can a strong-enough generator make it genuinely
> beneficial?**

Our answers:

1. **Naive synthetic augmentation is unreliable and can cause real harm** — it helped some
   detectors and hurt others, sometimes multiplying false alarms.
2. **A "fail-closed trust gate" removes the downside.** When synthetic data is poor, the
   gate declines it and the system safely reverts to real-data-only performance.
3. **Making the generator good is hard, and standard "realism" metrics mislead.** Two very
   different generators both fail a naive realism test — but for different reasons, and the
   usual fidelity score *saturates* (can't tell a near-perfect generator from a poor one).
4. **With a stronger, cleaned-up generator we found a statistically significant benefit for
   one detector.** Across 9 patient-split × random-seed cells, admitting our improved
   synthetic data improved the TCN detector's accuracy by **+0.083 event-F1 (p = 0.008,
   8/9 cells)** — the first solid evidence that synthetic augmentation can be made *safe and
   beneficial*.
5. **But the gate, as pre-registered, is over-conservative** — it rejects more than half the
   genuinely-beneficial cases, so the benefit it actually *delivers* (+0.034) is smaller than
   the benefit that *exists* (+0.083). Loosening the gate is the clear next lever.

**One-line takeaway:** naive synthetic augmentation is a hazard; a fail-closed gate makes it
safe; and a good-enough generator can make it *beneficial* (significantly so for TCN) — the
remaining work is tuning the gate to capture that benefit and confirming it on more data.

---

## 2. Background: the problem and why it is hard

### 2.1 What the detector does

An EEG records tiny scalp voltages — here **18 channels** at **256 samples/second**. The
detector reads the stream in **4-second windows** (every 2 seconds) and outputs, per window,
a probability that a seizure is occurring.

### 2.2 Class imbalance

Seizures are rare, so training data is ~99.7% "background" vs 0.3% "ictal" (*ictal* = the
seizure state). A model can score 99.7% "accuracy" by never predicting a seizure — useless.

### 2.3 Two things the literature usually gets wrong

**(a) Data leakage.** EEG from the *same patient* is recognisable. If one patient's windows
appear in both training and test, reported accuracy is inflated and doesn't reflect a new
patient. Avoiding this needs a **patient-independent** protocol.

**(b) Silent harm.** A generator can produce unrealistic or low-diversity "seizures";
training on them can *degrade* the detector, often unevenly across patients while the average
looks fine. We therefore report **tail risk** (worst cases), not just averages.

### 2.4 The fail-closed trust gate

Instead of trusting synthetic data blindly, a gatekeeper admits synthetic seizures **only if**
the resulting detector demonstrably does not do worse on held-out **validation** data (within a
small pre-registered tolerance). Otherwise it admits nothing and reverts to a real-data-only
detector. In the worst case the gate does *nothing* — but should never make things worse.

---

## 3. Research questions

- **Q1 — Baseline reality.** How well do detectors do on unseen patients with real data only?
- **Q2 — Help or harm.** Does synthetic augmentation improve or degrade detection vs simple
  safe baselines?
- **Q3 — Mitigation.** Does the fail-closed gate prevent harm while preserving gains?
- **Q4 — Can we make it help?** With a stronger, cleaned-up generator, does admitted synthetic
  data actually *improve* detection — and does the gate capture that benefit?

---

## 4. Data

**CHB-MIT Scalp EEG Database** (PhysioNet). After a channel audit fixing an 18-channel bipolar
montage: **24 patients** (23 leakage-safe groups; chb01/chb21 merged), **198 seizures**,
~**974 h**, band-pass filtered **0.5–40 Hz** (zero-phase), resampled to **256 Hz**, windowed
into **1,746,447** windows of which **5,563 (0.32%)** are seizure.

The 0.5–40 Hz filter matters later (§8): real EEG has essentially no power above 40 Hz, and a
good generator must reproduce that.

---

## 5. Methods

### 5.1 Leakage-safe splitting
**Grouped 5-fold cross-validation** at the patient level — training/validation/test patients
never overlap; verified programmatically. (Validation tunes decisions; test is used only for
final numbers.)

### 5.2 Detectors
**EEGNet** (~1,900 params), **LCT** (~121k, convolutional-transformer), **TCN** (temporal
convolutional network) — spanning a wide capacity range so conclusions aren't tied to one model.

### 5.3 Generators (trained patient-independently on each fold's training patients only)
- **cVAE** — conditional Variational Autoencoder; stable but tends to blur/over-smooth.
- **WGAN-GP** — Wasserstein GAN with gradient penalty; sharper but trickier to train.

### 5.4 Scoring: event-level, harm-first
**Event-F1** (headline, 0–1), **event sensitivity/precision**, and **false alarms per 24 h
(FP/24h)** — plus tail-risk (worst-fold, harm rates). Window-AUROC is reported for reference.

### 5.5 The trust gate
**(1) Admission:** score each candidate synthetic window with the real-only detector as a
"teacher"; admit only those whose teacher seizure-confidence exceeds the **q-quantile** of the
teacher's confidence on *real* seizures (higher q = stricter). **(2) Fail-closed selection:**
train on real + admitted synthetic; keep the augmented model only if it doesn't worsen
validation event-F1 / false alarms, else **revert** to real-only. All thresholds
**pre-registered**.

---

## 6. Results — Q1: real-only baselines

Cross-patient detection is genuinely hard even with real data (event-F1 0.12–0.30, 34–65
FP/24h), and varies a lot between patients (±0.11) — motivating tail-risk reporting.

| Detector / condition | Event-F1 | Sens. | Prec. | FP/24h | Win AUROC |
|---|---|---|---|---|---|
| EEGNet real_only | 0.179 ± 0.12 | 0.596 | 0.165 | 60.6 | 0.768 |
| EEGNet class_weighted | 0.117 ± 0.11 | 0.578 | 0.129 | 64.6 | 0.808 |
| EEGNet classical_aug | 0.206 ± 0.13 | 0.569 | 0.188 | 45.4 | 0.784 |
| LCT real_only | 0.246 ± 0.11 | 0.609 | 0.282 | 43.8 | 0.800 |
| LCT class_weighted | 0.295 ± 0.04 | 0.577 | 0.287 | 34.1 | 0.805 |
| LCT classical_aug | 0.230 ± 0.12 | 0.720 | 0.196 | 58.7 | 0.808 |

---

## 7. Results — Q2 & Q3: core help/harm and the gate (Tier B, cVAE, 1 seed)

Naive synthetic augmentation was a **coin-flip** (helped EEGNet, neutral for LCT, hurt TCN with
rising false alarms). Faced with the low-fidelity cVAE, the gate **declined all synthetic 100%**
and reproduced real-only performance. A cheap baseline (class weighting) often beat synthetic
augmentation. Mean event-F1 (pooled over folds/scarcity):

| Detector | real_only | class_weighted | classical_aug | ungated synth | trust-gated |
|---|---|---|---|---|---|
| EEGNet | 0.206 | 0.190 | 0.164 | **0.253** | 0.206 |
| LCT | 0.212 | **0.267** | 0.188 | 0.226 | 0.212 |
| TCN | 0.167 | **0.237** | 0.241 | 0.162 | 0.167 |

This raised **Q4**: the gate removed downside but delivered no upside because the cVAE was too
weak. Can a stronger generator clear the gate *and* help?

---

## 8. Results — Q4(a): why the generator's output looks "fake"

We measured *how* fake the synthetic seizures are, using: **discriminator AUC** (0.5 = ideal /
indistinguishable, 1.0 = trivially fake), **MMD** (frequency-distribution distance, lower
better), **diversity ratio** (near 1 healthy; near 0 = **mode collapse**), and the **power
spectrum**.

**Finding 1 — training longer doesn't fix "fakeness," but the WGAN doesn't collapse.** Across
50→600 epochs the discriminator AUC stays pinned at **1.000** for both generators, yet the
WGAN's diversity is healthy (~0.94) while the cVAE's collapses to ~0.001. So AUC hides a real
quality difference.

**Finding 2 — the WGAN reproduces the physiological spectrum almost perfectly.** In the 0.5–40 Hz
band the WGAN's mean log-PSD gap to real is **0.157** vs the cVAE's **2.66** (~17× closer); its
spectrum sits on top of the real one. See `analysis_tierB/figures/spectral_gap.png`.

**Finding 3 — both miss the >40 Hz filter roll-off.** Real EEG has almost no power above 40 Hz
(it was filtered); both generators emit a broadband floor there — the single largest
real-vs-synthetic discrepancy, and **fixable** by filtering the synthetic output (§10).

**Finding 4 — the discriminator AUC is saturated and cannot grade fidelity.** Even restricted to
in-band features it reads 1.000 for both, because a classifier with many features exploits tiny
consistent offsets. *Methodological lesson: do not judge synthetic-EEG fidelity by a single
saturating score.*

---

## 9. Results — Q4(b): what the gate actually reacts to

**Correction to our earlier understanding:** the gate does **not** use the discriminator-AUC
"realism score" (that was only a diagnostic). It keys on whether the **real-trained teacher
detector recognises the synthetic windows as seizures**. In the core experiment the gate
reverted because its admission rate was 0 — the teacher scored every synthetic window below the
confidence it assigns real seizures. The fix must therefore be generator-side.

We trained a real-only EEGNet teacher (cleanly separating real ictal 0.850 from background
0.048) and measured its seizure-confidence on synthetic windows, with two fixes: **(1)
band-limit** the WGAN output to 0.5–40 Hz; **(2)** use the WGAN instead of the cVAE.

| generator | teacher's mean seizure-confidence | admission @ q=0.90 |
|---|---|---|
| cVAE (β=0.01) | 0.475 | 0.0% |
| WGAN raw | 0.666 | 1.1% |
| **WGAN band-limited** | **0.774** | 0.1% |
| *real seizures (reference)* | *0.850* | — |

Recognisability rises monotonically cVAE < WGAN-raw < WGAN-band-limited → real. The
pre-registered q=0.90 is very strict (threshold 0.9985, the extreme tail of real-seizure
confidence), so little is admitted there; relaxing to q=0.50 admits ~17% of band-limited WGAN
windows — enough to actually augment training.

---

## 10. Results — Q4(c): single-fold downstream (fold 0)

Running the full pipeline on fold 0 with the band-limited WGAN: **naive injection harmed all
three detectors** (e.g. TCN false alarms 12→69/day). The gate protected EEGNet and LCT
(reverted → real-only). And **one upside cell appeared** — TCN at the pre-registered q=0.90 was
admitted and improved (event-F1 0.174→0.204, FP 12.4→8.3). That single-seed signal motivated the
confirmation run below.

---

## 11. Results — Q4(d): multi-seed × multi-fold confirmation (n = 9) — the decisive experiment

**Plain-language finding:** repeating the downstream experiment across **3 folds × 3 seeds =
9 patient-split × seed cells** per detector, the TCN upside **held and reached statistical
significance**: admitting the band-limited WGAN synthetic improved TCN by **+0.083 event-F1
(paired p = 0.008, 8/9 cells)**. But the gate, as pre-registered, is **over-conservative** and
delivers only part of that benefit. EEGNet is neutral/heterogeneous (not systematically harmed).

*Definitions:* **"augmented"** = the model actually trained on the gate-admitted synthetic
(measures whether the synthetic *helps*). **"effective"** = what the gate *delivers* (for a
reverted cell, that is the real-only model). Δ is paired vs real_only within each fold×seed;
95% CIs are bootstrap; the test is a Wilcoxon signed-rank across the 9 cells.

**TCN (n = 9):**

| condition | mean F1 | Δ vs real_only [95% CI] | helped | Wilcoxon p |
|---|---|---|---|---|
| real_only | 0.191 | — | — | — |
| ungated (naive) | 0.251 | +0.060 [−0.03, +0.14] | 7/9 | 0.16 |
| **gated q0.90 — augmented** | 0.275 | **+0.083 [+0.037, +0.132]** | **8/9** | **0.008** |
| gated q0.90 — effective (deployed) | 0.225 | +0.034 [−0.003, +0.079] | 3/9 | 0.25 |
| gated q0.50 — augmented | 0.239 | +0.048 [−0.011, +0.111] | 6/9 | 0.25 |

**EEGNet (n = 9):**

| condition | mean F1 | Δ vs real_only [95% CI] | helped | Wilcoxon p |
|---|---|---|---|---|
| real_only | 0.218 | — | — | — |
| ungated (naive) | 0.252 | +0.034 [−0.04, +0.13] | 5/9 | 0.73 |
| gated q0.90 — augmented | 0.214 | −0.004 [−0.10, +0.08] | 5/9 | 0.91 |
| gated q0.90 — effective | 0.229 | +0.011 [0.00, +0.03] | 1/9 | — |

**Three findings.**

- **The synthetic genuinely helps TCN — significantly.** The augmented TCN model improved by
  **+0.083 event-F1, 8/9 cells, p = 0.008**, with a 95% CI that excludes zero. A good-enough
  generator (spectrally realistic, diverse, band-limited) produces synthetic seizures that
  measurably improve a real detector. This is the project's central positive result.
- **The gate is over-conservative — it leaves benefit on the table.** For TCN it **reverted 56%**
  of cells at q=0.90, so the *delivered* benefit (+0.034, not significant) is much smaller than
  the benefit that *exists* (+0.083). The fail-closed gate as pre-registered trades real benefit
  for its safety guarantee. Loosening the admission/margin criteria is the clear next lever.
- **Benefit is detector-specific; EEGNet is neutral, not harmed.** EEGNet's augmented delta is
  ≈0 (coin-flip, 5/9 helped) and it even has upside cells (e.g. fold 2: 0.245→0.346). The earlier
  impression that "EEGNet is refused 100%" was a two-fold artifact; with all folds it is
  heterogeneous, and the gate's 100%-safe behaviour (effective ≥ real_only) still holds.

*(A real-data **positive control** — feeding genuine held-out ictal windows through the gate to
confirm it admits/keeps clearly-good data and thus is not faulty for EEGNet — is running; results
to be appended. A Gaussian-noise negative control confirms the gate rejects junk.)*

---

## 12. Overall discussion

1. **Synthetic ictal augmentation is a hazard, not a free lunch** — naive injection is
   unreliable and frequently harmful, even with a good generator.
2. **The fail-closed gate is an effective safety mechanism** — it converts an unpredictable,
   occasionally-harmful intervention into one that is *never worse than doing nothing*, and its
   validation-based decisions generalise to held-out test patients.
3. **Standard fidelity metrics mislead** — the discriminator-AUC saturates; diversity and
   spectral analysis are needed to see real quality differences and locate fixable defects.
4. **A good-enough generator delivers real, significant benefit (TCN, +0.083, p = 0.008)** — the
   "safe *and* beneficial" case, not just "safe."
5. **The current gate under-delivers that benefit** by being too conservative — a concrete,
   actionable finding: the safety/benefit trade-off is tunable via the admission quantile and
   fail-closed margin.

---

## 13. Limitations

- **One dataset (CHB-MIT), small and dated.** Generality is untested; a second dataset (Siena is
  scoped; the larger TUH corpus would be stronger) is essential for a strong claim.
- **n = 9 and a single significant comparison.** The TCN augmented result (p = 0.008) survives a
  modest multiple-comparison correction (~6 comparisons → ~0.05) but should be reported with that
  caution; it needs replication.
- **The significant effect is on the *augmented* model, not the deployed gate output.** The
  deployed (effective) benefit is smaller and not yet significant because the gate is
  conservative.
- **High run-to-run nondeterminism** (GPU + tiny event counts) makes single cells noisy — the
  reason we average over seeds and folds.
- **The trust gate is adapted from prior work**, not novel; the contribution is the
  seizure-specific, event-level reformulation, the harm characterisation, and the demonstration
  that a good generator can clear it and help.

---

## 14. Conclusions and next steps

**Conclusion.** On unseen patients, seizure detection is far from solved; naive synthetic
augmentation is unreliable and can harm; a fail-closed trust gate removes that downside; and,
with a spectrally-realistic, band-limited WGAN-GP generator, admitting synthetic seizures
produced a **statistically significant improvement for the TCN detector (+0.083 event-F1,
p = 0.008)**. The gate as pre-registered is over-conservative and captures only part of this
benefit.

**Next steps, in priority order:**
1. **Tune the gate to capture the demonstrated benefit** — relax the admission quantile /
   fail-closed margin and measure the safety↔benefit trade-off (the delivered +0.034 should move
   toward the available +0.083 without reintroducing harm).
2. **Second dataset (Siena / TUH)** for generality — the single biggest lever for publishability.
3. **Complete the real-data positive control** (in progress) and report it, plus the noise
   negative control.
4. **Leave-one-patient-out** validation for a stricter generalisation test.

**Engineering enablers (this round).** The processed store lives on a network filesystem; we
added an opt-in **full-file signal cache** with parallel prefetch so the pipeline runs from RAM
(~40× faster; float16 storage to fit a 62 GB container memory limit), and we **band-limit**
synthetic output with the exact preprocessing filter. Both are small, reusable additions that
made the multi-seed confirmation tractable.

---

## 15. Glossary

- **AUROC / window AUROC** — 0.5–1.0 score for window-level separation of seizure vs non-seizure.
- **Band-pass / band-limit** — keep only 0.5–40 Hz, removing drift and high-frequency noise.
- **Class imbalance** — seizures vastly rarer than background.
- **cVAE / WGAN-GP** — the two generators (autoencoder vs adversarial network).
- **Discriminator AUC** — how easily real is told from synthetic; 0.5 ideal, 1.0 trivially fake.
- **Diversity ratio / mode collapse** — synthetic spread vs real; near 0 = generator repeats itself.
- **Event-F1 / sensitivity / precision** — event-level accuracy measures.
- **Fail-closed** — if not proven safe, do nothing.
- **FP/24h** — false alarms per 24 hours (lower better).
- **Ictal** — the seizure state.
- **Leakage / patient-independent** — never sharing a patient across train and test.
- **MMD** — Maximum Mean Discrepancy; distance between two distributions.
- **augmented vs effective** — model trained on admitted synthetic vs what the gate actually
  delivers (real-only if it reverts).
- **PSD** — power spectrum (energy vs frequency).
- **Quantile q** — gate strictness dial (fraction of real seizures a synthetic window must
  out-score to be admitted).
- **Tail risk** — worst-case outcomes, not the average.
- **Wilcoxon signed-rank test** — a paired significance test used across the fold×seed cells.

---

## 16. Reproducibility and artifacts

All numbers derive from scripts in this repository, on public CHB-MIT, with fixed seeds and
pre-registered thresholds.

- Preprocessing: `scripts/run_preprocess.py`
- Core grid (Tiers A/B): `experiments/run_tierA_dev.py`, `experiments/run_tierB_core.py`
- Generator fidelity / spectral gap: `scripts/gen_wgan_fidelity_diag.py`,
  `scripts/gen_beta_sweep.py`, `scripts/gen_spectral_gap_diag.py`
- Gate admission re-test + q-sweep: `scripts/gate_admission_retest.py`
- Single-fold downstream: `scripts/run_downstream_gated.py`
- **Multi-seed downstream (n=9): `scripts/run_multiseed_downstream.py`**;
  analysis: `scripts/analyze_multiseed.py` → `analysis_tierB/downstream_gated_multiseed.csv`,
  `analysis_tierB/multiseed_summary.csv`
- Positive control: `scripts/positive_control_gate.py`
- Band-limiting: `synthetic/band_limit.py`; window cache: `chbmit/datasets.py`
  (`prefetch_windows`); trust gate: `synthetic/trust_gate.py`; pre-registration:
  `PREREGISTRATION.md`

*Author/course fields to be completed before submission. Metrics follow the SzCORE convention.
The n=9 downstream results are single-dataset and require second-dataset / LOPO confirmation.*
