# Related Work — deep literature search (2026-08-28)

Search run against the project as described in `the project notes` / `reports/comprehensive_report.md`.
Each entry marked **[V]** = citation metadata verified from the publisher/API; **[S]** = from a search
result or third-party summary only — verify before citing.

---

## 0. Verification checklist from POSITIONING.md — resolved

| Item | Status |
|---|---|
| Confirm TGA preprint authors/title/venue/status | **DONE** — see §1. Still an unreviewed bioRxiv v1 preprint; cite as preprint. |
| Search for concurrent seizure-specific trust-gated / harm-aware augmentation | **DONE** — none found. The niche is open (see §8). |
| Verify geometry-preserving / Riemannian VAE citation | **DONE** — it is a *BSc thesis*, MI-BCI only. See §5. Do not cite as seizure evidence; recommend dropping. |
| "Adapted"/"reformulated" language for the gate | Unchanged — still required. |
| Abstract leads with harm, gate second | Unchanged. |

---

## 1. The direct parent — the gate we adapt

**[V] Choi, D.; Yip, C.; Choi, A.; Park, J. (2026).** *Fail closed trust gated synthetic augmentation
governs tail risk under subject shift in EEG.* bioRxiv, DOI `10.64898/2026.01.26.701638`, **v1,
posted 2026-01-28**, category *Bioinformatics*. **Preprint, not peer reviewed.**

Verified via the bioRxiv details API (authors, title, date, version, category, abstract).

- Mechanism: a real-data teacher scores each synthetic candidate for **label consistency + confidence**;
  only candidates above a **quantile gate** are admitted; a **fail-closed do-no-harm rule** then injects
  synthetic data only if validation **AUROC** improves by a pre-specified margin, else reverts to real-only.
- Failure modes named: **label-inconsistent** synthetic examples and **off-manifold** multichannel
  structure ("toxic training signals").
- Evidence: **chronic-pain resting-state EEG (189 subjects)** and **motor-imagery BCI** in a low-data regime.
- Headline number: harm reduced from **56% → 24% of cases**.
- Framing: explicit **tail-risk** argument — rare failures dominate the cost-benefit calculation in medicine.

**Implication for us.** The POSITIONING.md delta holds exactly as written: TGA is a balanced,
discriminative, window/AUROC-level result. Ours is rare-event seizure detection with an
**event-F1 + FP/24h** reformulation of both the admission rule and the fail-closed rule. Note the number
to contrast against: TGA governs the harm *rate* (56% → 24%); we additionally show that a good-enough
generator produces a **significant positive effect** (+0.083 event-F1, TCN), which TGA does not claim.

---

## 2. Seizure-specific generative EEG — the closest technical work

### 2.1 GP-EEG — the biggest competitive threat, and the biggest opportunity

**[V] Moutonnet, N.; Corneck, J.; Tobar, F.; Mandic, D.** *Synthesizing Epileptic Seizures: Gaussian
Processes for EEG Generation.* arXiv:2601.21752 (Imperial College London).

All points below verified against the arXiv full text (v1 HTML), not a snippet.

- **Datasets: CHB-MIT (22 pediatric subjects, 198 expert-annotated seizure events, 18 bipolar channels,
  256 Hz) + Siena (14 adults, 47 seizure events).** Essentially *our* preprocessing and *our* dataset
  pair — including the Siena second dataset we have scoped as next step 2.
- **They drop chb12 and chb15 for computational reasons, so the experiments actually use 20 patients,
  not 22.** Worth noting: our harness uses all 24 patients / 23 leakage-safe groups. That is a small,
  legitimate coverage advantage to state in one clause — not a claim to lean on.
- Pipeline: SVD compression → changepoint detection + Gaussian-process regression (quasi-periodic
  Matérn) → Conv-LSTM VAE domain adaptation → kernel-state discretisation → Poisson regime timing.
- Evaluation: **TSTR** and **augmentation**, both under an explicitly stated **leave-one-patient-out
  (LOPO)** scheme — "the test set consists exclusively of real EEG seizure and background segments from
  the held-out patient." Downstream classifier is **EEGNet4,2** (Lawhern et al., 2018).
- Generative baselines beaten: **TimeVAE, COSCI-GAN, ImagenTime**.
- Metrics: accuracy, precision, recall, F1 at **segment/window level**, plus feature-based fidelity
  (MDD, ACD, skewness/kurtosis differences), t-SNE and KDE.
  **Confirmed: no event-level metric, no false-alarm rate per hour or day, no harm or tail-risk
  analysis anywhere in the paper.** This is the crux of our delta and it now rests on a full-text read.

**Why this matters.** GP-EEG overlaps our setting more than anything else found: same corpus, same
montage, LOPO, EEGNet, augmentation utility. It is *not* a duplicate — it asks **"can we generate good
seizures?"**, not **"is it safe to train on them, and how do we govern that?"** No harm characterisation,
no tail risk, no gate, and (as far as the abstract and HTML show) no event-level FP/24h.

**Action:** (a) cite prominently and state the delta; (b) treat GP-EEG as the obvious *generator arm* —
it would test whether the gate's over-conservatism is generator-specific; (c) inherit its baseline set
(TimeVAE / COSCI-GAN / ImagenTime) so our generator comparison is not just cVAE vs WGAN-GP.

### 2.2 The "naive positive" claim we qualify

**[V] Abou-Abbas, L. et al. (2024).** *Generative AI with WGAN-GP for boosting seizure detection
accuracy.* Frontiers in Artificial Intelligence, 10.3389/frai.2024.1437315 (PMC11480023).

- **Patient-independent** via GroupKFold — so *not* a leakage strawman.
- WGAN-GP + BiLSTM: real-only **86% → 91.73% accuracy**; beats SMOTE (85.25%) and ADASYN (85.18%).
- **But: window/segment-level accuracy. No event-level metrics, no FP/24h, no tail risk, no per-fold
  harm reporting, no gate.**

This is the best foil in the literature: same generator family, same dataset, an honest split — and it
still reports a clean +5.7pp win because the *metric* hides the clinical failure mode. Our §7 and
§10–11 results are precisely the counter-argument. Lead the related work with this contrast.

### 2.3 Earlier seizure-GAN augmentation lineage

- **[S]** Pascual, D. et al. *Synthetic Epileptic Brain Activities Using GANs* / EpilepsyGAN, arXiv:1907.10518.
- **[S]** Truong, N. et al. (2019) — first DCGAN on spectrograms for seizure prediction, CHB-MIT +
  Epilepsy-Ecosystem.
- **[S]** Rasheed, K. et al. *A Generative Model to Synthesize EEG Data for Epileptic Seizure Prediction*,
  arXiv:2012.00430 (PMC8592500).
- **[S]** Xu, Y. et al. (2022) *Synthetic Epileptic Brain Activities with TripleGAN*, Comput Math Methods Med
  (PMC9440850) — time / frequency / time-frequency domains, CHB-MIT.
- **[S]** *Multichannel Synthetic Preictal EEG Signals to Enhance the Prediction of Epileptic Seizures*,
  arXiv:2205.03239.
- **[S]** *Pre-Ictal EEG Augmentation Based CDCGAN Model for Epileptic Seizure Prediction*, Technologies (2026),
  10.3390/technologies14020114.
- **[S]** *Epileptic Seizure Prediction by Synthesizing EEG Signals through GPT*, ICAIPR 2021,
  10.1145/3488933.3489016.

Almost all of these are **seizure prediction (pre-ictal), not detection**, and most report window-level
accuracy. Use them to establish "the literature reports it helps" without conceding they measured what
we measure.

### 2.4 Diffusion — the generator family we did not try (a reviewer will ask)

- **[S]** *EEGDiffuser: Label-guided EEG signals synthesis via diffusion model for BCI applications*,
  Neurocomputing (2026).
- **[S]** *Diffusion Model-Based Multi-Channel EEG Representation and Forecasting for Early Epileptic
  Seizure Warning*, Interdiscip Sci Comput Life Sci (2025), 10.1007/s12539-025-00750-2.
- **[S]** *Generative modeling and augmentation of EEG signals using improved diffusion probabilistic
  models*, PubMed 39693767.
- **[S]** *EEG Synthetic Data Generation Using Probabilistic Diffusion Models*, arXiv:2303.06068.

Pre-empt this in Limitations: we compared cVAE vs WGAN-GP; diffusion is the obvious third arm, and our
gate is **generator-agnostic** by construction.

---

## 3. Evaluation protocol, leakage, and event-level scoring — our methodological backbone

### 3.1 Adopt SzCORE. It is now the field standard.

- **[S]** *SzCORE: A Seizure Community Open-source Research Evaluation framework for the validation of
  EEG-based automated seizure detection algorithms*, arXiv:2402.13005 — event-based scoring by overlap;
  sensitivity, precision, **F1, FP/day**; BIDS-EEG + HED-SCORE file formats.
- **[V]** *Quantifying the Generalization Gap in Seizure Detection: A Large-Scale Empirical Benchmark via
  the SzCORE Challenge*, **arXiv:2505.18191** — note the **title changed at v2** (v1, May 2025, was
  *"SzCORE as a benchmark: report from the seizure detection challenge at the 2025 AI in Epilepsy and
  Neurological Disorders Conference"*; v2, May 2026, carries the current title). Cite the v2 title.
  Strictly held-out private dataset from the Filadelfia Danish Epilepsy Centre: **65 subjects, 398
  annotated seizures, 4,360 h**. 30 submissions from 19 teams, **28 successfully evaluated**.
- Site: epilepsybenchmarks.com

**Challenge results — the numbers that matter to us (verified against the v2 abstract and results):**

| | Sensitivity | Precision | F1 | FP/day |
|---|---|---|---|---|
| Winner (Sz Transformer) | 37% | 29% | **32%** | 1.34 |
| Top-5 submissions | 30–58% | 19–29% | — | 1.34–14 |
| Full field (28 algorithms) | up to 99% | — | — | **1–290** |
| STORM (max sensitivity) | 99% | — | — | 290 |

> **CORRECTION to the first pass of this search:** an earlier draft recorded the top F1 as *43%
> (sensitivity 37%, precision 45%)*. That was wrong — it came from a search snippet. The v2 abstract
> states **32% (sensitivity 37%, precision 29%)**. Use 32%.

**This is still the most actionable finding of the search, and the corrected numbers make the case
better, not worse.** Our headline metric (event-F1) and safety metric (FP/24h) are already SzCORE-shaped.
Declaring compliance costs little and buys external comparability plus a *sanity anchor*: the best
cross-subject event-F1 in an open challenge on unseen subjects was **0.32**, which places our real-only
baselines of **0.12–0.30** squarely in the honest cross-patient regime rather than looking like a broken
pipeline. Our best cell (LCT class-weighted, 0.295) is essentially at the challenge winner's level.

**But read §3.1a before using this anchor one-sidedly.**

### 3.1a The false-alarm gap — a real exposure, not a talking point

Our sensitivity is *better* than the challenge field: **0.569–0.720** against a top-5 range of 30–58%.
Our precision (**0.129–0.287**) sits right inside the top-5 range of 19–29%. But our **FP/24h of 34–65**
sits far above the top-5's **1.34–14/day**.

Two honest readings, and we should state both:

1. **Defensible:** the challenge itself demonstrates a steep sensitivity↔false-alarm trade-off — STORM
   bought 99% sensitivity at 290 FP/day. Our operating point (high sensitivity, high FP) is a normal
   position on that curve, and 34–65 FP/24h is comfortably inside the observed field range of 1–290.
   CHB-MIT is also ~2.2× denser in seizures than the challenge corpus (≈4.9 vs ≈2.2 seizures/day), which
   shifts FP/day upward at fixed precision.
2. **Exposed:** a SzCORE-literate reviewer will still note that we are 3–25× the top-5 false-alarm rate
   while claiming a safety-first framing. Pre-empt this explicitly, and report an operating point chosen
   at a comparable FP/day so at least one row is directly comparable to the challenge leaderboard.

**Internal check to run before submission.** The §6 numbers do not obviously reconcile with each other.
For LCT class-weighted (sens 0.577, prec 0.287, FP/24h 34.1): precision of 0.287 at 34.1 FP/day implies
≈13.7 true-positive events/day, which at 57.7% sensitivity implies ≈23.8 seizures/day in the test data —
but CHB-MIT averages ≈4.9 seizures/day. Likewise, the harmonic mean of the reported sensitivity and
precision is 0.383, not the reported event-F1 of 0.295. Both gaps are *plausibly* just artifacts of
averaging per-fold metrics rather than pooling events (mean-of-F1 ≠ F1-of-means), which is legitimate —
but a reviewer will do exactly this arithmetic. Confirm whether §6 reports means-of-folds or pooled
counts, and say so in the table caption. If it is means-of-folds, also report the pooled numbers.

### 3.2 Leakage / protocol critique — cite these to justify the harness

- **[V] Ali, E.; Angelova, M. N.; Karmakar, C. (2024).** *Epileptic seizure detection using CHB-MIT
  dataset: The overlooked perspectives.* R. Soc. Open Sci. **11**:230601, 10.1098/rsos.230601.
  Cross-subject **event** detection with post-processing: **72.63% sensitivity (subject-wise 5-fold)**,
  **75.34% (leave-one-out)**, false detection rate **5.32/h and 4.79/h**. Argues event detection is the
  realistic target; prior work "focussed only on random seizure segment detection".
  → **Note:** this paper makes *qualitative* claims only; it does **not** publish the "96% of studies leak"
  statistic. Do not attribute that number to it — the real source is Shafiezadeh et al., next bullet.
  → Useful anchor: their FDR of ~4.8–5.3/h ≈ **115–128 FP/24h**; our baselines run **34–65 FP/24h**.

- **[V] Shafiezadeh, S.; Duma, G. M.; Pozza, M.; Testolin, A. (2024).** *A systematic review of
  cross-patient approaches for EEG epileptic seizure prediction.* **J. Neural Eng. 21(6), 061004**,
  10.1088/1741-2552/ad9682, published 3 Dec 2024.
  **This is the citation to use for the leakage claim.** It reviewed **119 published studies** and found
  **over 96% evaluated using randomised or patient-specific splits** — i.e. only ~4% are genuinely
  leakage-free. Hard number, systematic review, and it is in **J Neural Eng**, one of our target venues.
  → **Caveat:** the review covers seizure *prediction*, not detection. Phrase our sentence to reflect
  that ("in the closely related seizure-prediction literature, a systematic review of 119 studies found…")
  rather than overclaiming it as a detection statistic.
  → **CORRECTION to the first pass of this search:** an earlier draft attributed the 96%/4% figure to
  CG-MambaNet. CG-MambaNet merely *cites* Shafiezadeh et al. for it. Cite the review directly.

- **[S]** *CG-MambaNet: a spatiotemporal framework for cross-patient epileptic seizure prediction using
  CNN-GCN-Mamba-BiLSTM with event-level clinical evaluation*, arXiv:2606.08226 (2026). LOPO × **5 random
  seeds**; event-level sensitivity + alarms/h; reports a **351× FPR reduction from window-level to
  event-level** reporting. Seizure *prediction*, so not a competitor, but a strong precedent for
  **LOPO × multi-seed + event-level**, i.e. exactly our next steps 1 and 4. Cite as convergent
  methodology — not as the source of the 96% figure.
- **[S]** *Generalization or mirage? Data leakage and reported performance in neonatal EEG seizure
  detection models: a systematic review.* BioData Mining (2025), 10.1186/s13040-025-00516-y.
  Leakage significantly predicts reported performance (**p = 0.005, R² = 0.202**); the **data-partitioning
  strategy** explains **59.8%** of leakage variance (p < 0.001). Quantitative ammunition for §2.3(a).
- **[S]** *Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods*,
  arXiv:2604.27033 / IOP (2026) — states plainly that same-subject segments in train and test are
  "fundamentally flawed for clinical EEG" and introduce "severe data leakage".

### 3.3 Controls and shortcut learning — supports our positive/negative control design

- **[S]** Zare, M. *What EEG Foundation Models Encode: Dataset Identity and a Negative-Control Suite for
  Clinical Benchmarks* (v2; v1 titled *Stress-Testing EEG Foundation Models for Clinical Decoding*),
  arXiv:2607.24519. Seven models (LaBraM, EEGMamba, CBraMod, REVE, LEAD, BENDR, BIOT), subject-disjoint
  LOSO / grouped 5-fold. **Dataset identity decodes at AUROC 1.000** from frozen embeddings while the
  clinical label decodes at 0.528. Negative-control suite: random init, random features, label
  permutation, scrambled-label fine-tuning.

Directly supports our outstanding **positive control** (does the gate admit *real* held-out ictal?) and
the **noise negative control**. Cite as precedent that control suites are now expected, not optional —
this raises the value of finishing `scripts/positive_control_gate.py`.

---

## 4. Does augmentation help at all? — the prior we are testing

- **[S]** Rommel, C. et al. (2022). *Data augmentation for learning predictive models on EEG: a systematic
  comparison.* J. Neural Eng., arXiv:2206.14483. 13 augmentations × 2 tasks × multiple datasets/models;
  **up to 45% accuracy gain in low-data regimes**, >10% for sleep staging and MI-BCI. This is the
  optimistic prior our paper pushes back on — note that it studies *signal transformations*, not
  generative synthesis, and not rare-event detection.
- **[S]** Wang, Z.; He, Z.; He, X.; Wang, H.; Jia, T.; Luo, J.; Li, S.; Chen, X.; Wu, D. (2026).
  *Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions.*
  arXiv:2603.12296 (submitted 2026-03-11, revised 2026-05-19). Taxonomy —
  signal-transformation / feature-based / model-based / translation-based — plus a benchmark across four
  BCI paradigms. **The survey to position against; it benchmarks utility but not harm.**
- **[S]** *A survey on data augmentation for EEG-based emotion recognition and cognitive workload decoding*,
  Front. Neurosci. (2026), 10.3389/fnins.2026.1789468. Explicitly warns that apparent gains "may partly
  reflect optimistic data partitioning rather than true improvements in cross-subject generalization" —
  a quotable independent statement of our thesis, from outside epilepsy.

---

## 5. Synthetic-data quality gating beyond EEG — where to take the over-conservative gate

Our #1 next step (tune the gate to capture the +0.083) has a ready-made principled answer here:

- **[S]** *Filtering with Confidence: When Data Augmentation Meets Conformal Prediction*, arXiv:2509.21479.
  **Conformal data augmentation** — filters poor-quality generations with **provable risk control**.
  → The natural upgrade path: replace the pre-registered fixed admission quantile `q` with a conformal
  admission set at a target risk level `α`. It converts "the gate is over-conservative and we hand-tuned
  it" into "the gate has a tunable, guaranteed harm budget" — a real methodological contribution, cheap
  to implement, and a direct answer to the reviewer question our current results invite.
- **[S]** Ye, J. et al. *Selective Synthetic Augmentation with Quality Assurance* — the closest precursor
  of the admit/reject-per-sample idea (histopathology).
- **[S]** *Ctrl-GenAug: Controllable Generative Augmentation for Medical Sequence Classification*,
  arXiv:2409.17091 — medical *sequence* data, includes a synthetic-sample filtering stage.
- **[S]** *Confidence-Guided Diffusion Augmentation…*, arXiv:2605.10916 — classifier-confidence filtering,
  the same primitive as our teacher-confidence admission.
- **[S]** Shumailov, I. et al. (2024). *AI models collapse when trained on recursively generated data.*
  Nature 631, 10.1038/s41586-024-07566-y — "tails of the original content distribution disappear".
  Use in the discussion: our over-smoothed cVAE (posterior collapse, lost spectrum) is the seizure-EEG
  instance of exactly this tail loss, and it is *why* naive admission harms.
- **[S]** *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic
  Data*, OpenReview `5B2K4LRgmz` — accumulating rather than replacing real data avoids collapse; supports
  our real+synthetic mixing design.

**[V] — RE THE POSITIONING.md CHECKLIST ITEM:** the "geometry-preserving / Riemannian VAE" citation
resolves to **Poļaka, V.; de Jong, I. P.; Sburlea, A. I.**, *Riemannian Geometry-Preserving Variational
Autoencoder for MI-BCI Data Augmentation* — originating as a **BSc thesis at the University of
Groningen** (fse.studenttheses.ub.rug.nl/36690) and **also posted as arXiv:2603.10564 (2026)**.

> **CORRECTION to the first pass of this search:** an earlier draft called this "a BSc thesis, not
> citable." That was too strong — the arXiv posting with faculty co-authors is a normal, citable
> preprint. The substantive advice is unchanged.

Its one relevant claim — that synthetic-EEG usefulness "depends on the paired classifier" — genuinely
matches our generator × detector finding. But it is **motor-imagery only**, and a bachelor's-project
preprint is thin support for a load-bearing sentence. Recommendation: either drop it in favour of
arXiv:2603.12296 (benchmark section) or arXiv:2604.27033, which make the same point with more weight, or
cite it only as a secondary corroborating reference — and never as seizure-detection evidence.

---

## 6. Fidelity ≠ utility — supports our "discriminator-AUC saturates" finding

- **[S]** TSTR / TRTS methodology and the **discriminative score** (a binary real-vs-synthetic classifier;
  ~0 = indistinguishable). Our §8–9 finding is that this score **saturates at 1.0** and therefore cannot
  grade a near-good generator — a concrete, citable critique of the standard metric.
- **[S]** *Challenges and Limitations in the Synthetic Generation of mHealth Sensor Data*, arXiv:2505.14206 —
  time-series generation "lacks universally accepted evaluation metrics".
- **[S]** *Synthetic Tabular Generators Fail to Preserve Behavioral Fraud Patterns*, arXiv:2604.13125 —
  the same fidelity-vs-utility dissociation in another domain; useful cross-domain corroboration.
- **[S]** *Enabling Granular Subgroup Level Model Evaluations by Generating Synthetic Medical Time Series*,
  arXiv:2510.19728 — discriminative fidelity + TSTR/TRTS on eICU/MIMIC.

---

## 7. Governance framing — cheap credibility for the harm-first pitch

- **[S]** *Governing synthetic data in medical research: the time is now.* Lancet Digital Health (2025),
  S2589-7500(25)00011-1.
- **[S]** *Protecting patient privacy in tabular synthetic health data: a regulatory perspective.*
  npj Digital Medicine (2025), 10.1038/s41746-025-02112-0.
- **[S]** FDA / NIST **AISAMD** (Artificial Intelligence Synthetic Data for Medical Devices) programme —
  a framework for using synthetic data to evaluate medical devices.

One intro paragraph citing these turns "we built a gate" into "we operationalise an emerging regulatory
expectation for a rare-event clinical detector". Low cost, high positioning value.

---

## 8. Bottom line for the paper

**Our niche is still open.** No seizure-specific trust-gated or harm-aware augmentation work exists.
But the space closed noticeably in 2026:

1. **GP-EEG (Jan 2026)** now occupies CHB-MIT + Siena + LOPO + EEGNet augmentation. Must-cite, and it
   makes "we tried two generators" look thin — adopting its baselines or its samples is the cheapest way
   to stay ahead.
2. **TGA (Jan 2026)** is public and unclaimed for seizures. Our event-level reformulation is defensible,
   but the window is finite.
3. **SzCORE compliance** is the highest-value / lowest-cost change available: external comparability plus
   the 0.43-F1 challenge anchor that makes our low baselines legible.
4. **Conformal admission (arXiv:2509.21479)** converts our known weakness (over-conservative gate,
   +0.034 delivered vs +0.083 available) into a contribution with a risk guarantee.
5. **Abou-Abbas 2024** is the perfect foil: patient-independent, WGAN-GP, +5.7pp — at window level.

**Revised venue read.** The harm characterisation + event-level gate + honest negative-space reporting
still points at J Neural Eng / IEEE JBHI / Clin Neurophysiol. Given SzCORE and the negative-control
literature, an ML4H / NeurIPS D&B framing is now also well-supported — but *only* with the second
dataset, because GP-EEG has already established CHB-MIT + Siena as the expected pairing.
