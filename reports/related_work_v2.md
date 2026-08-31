# Related Work & Positioning — v2 (source-verified)

**Date:** 2026-08-28 · **Supersedes:** `related_work.md` (first-pass search log) and the citation
guidance in `POSITIONING.md` (v5.3).

Every citation below was checked against an authoritative record — Crossref, Europe PMC, the arXiv API,
or publisher full text. **Thirteen errors from the first two passes were found and corrected**; §0.2
lists them, because several were the kind of error that survives casual re-reading.

**Reliability convention.** **[V]** = identifier and claim verified at source. **[V-id]** = identifier
verified, but the specific *number or quotation* attributed to it is not confirmed — treat the claim as
provisional. Nothing in this file is unmarked.

---

## 0.1 How these were verified

- **DOIs** resolved through the **Crossref REST API** (`api.crossref.org/works?filter=doi:…`), which
  returns registered title, journal and date. A DOI that does not appear in the result set does not
  exist as given.
- **arXiv IDs** resolved in bulk through the **arXiv API** (`export.arxiv.org/api/query?id_list=…`),
  which returns the current title for each ID. This catches both fabricated IDs and stale titles from
  superseded versions.
- **Publication status** checked through **Crossref plus Europe PMC**, never through the preprint server
  alone.

**Standing rule for anything added to this file later.** A search-result summary is not a source. Two of
the errors below were fabricated identifiers that came from search summaries and read as entirely
plausible. Resolve the identifier before writing it down.

## 0.2 Errors corrected in this pass

| # | Error | Correction |
|---|---|---|
| 1 | `arXiv:2603.10564` cited as the Riemannian VAE paper | **Fabricated attribution.** That ID is *Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents* — a telecoms paper. No arXiv version of the Riemannian work exists. |
| 2 | Riemannian work described as a preprint with faculty co-authors, 2026 | Single author **Poļaka, Agnese Viktorija**; de Jong and Sburlea are **supervisors, not co-authors**; **2025**; BSc thesis only. |
| 3 | Quote "fundamentally flawed for clinical EEG… severe data leakage" attributed to arXiv:2604.27033 | **Unsubstantiated.** Not present in the abstract; could not be located. **Removed.** |
| 4 | "our FP/24h is 3–25× the top-5" | Wrong arithmetic. Correct range is **≈2.4× to ≈49×**. |
| 5 | Neonatal leakage review dated 2025 | Published **8 January 2026**. |
| 6 | Lancet Digital Health cited as `S2589-7500(25)00011-1` | That is the **PII, not the DOI**. DOI is **10.1016/j.landig.2025.01.012**. |
| 7 | Zare paper cited by a superseded title with stale figures | Now at **v3**; retitled again; **five** encoders not seven; the "0.528" figure is not in the current version. |
| 8 | arXiv:2505.14206 title given as "…mHealth Sensor Data" | Actual title: *Challenges and Limitations of Generative AI in Synthesizing Wearable Sensor Data*. |
| 9 | CG-MambaNet "351× FPR reduction" stated as fact | **Not locatable in the abstract.** Demoted to [V-id]. |
| 10 | Ye et al. cited without identifier | **arXiv:1912.03837**, with a journal version (HistoGAN) in *Medical Image Analysis*. |
| 11 | "our best cell is essentially at the challenge winner's level" | **Cross-dataset leap.** Different corpora; cannot be compared like for like. Caveated. |
| 12 | Conformal augmentation presented as a drop-in fix | Its tasks are text, images and fraud — **no physiological time series**. Adopting it is an extension. |
| 13 | Wang et al. survey described as generic BCI | It **explicitly benchmarks epileptic seizure detection** as one of four paradigms. Closer neighbour than stated. |

---

## 1. The positioning claim

| | Asks | Measures | Governs synthetic data? |
|---|---|---|---|
| Seizure generative EEG | can we generate realistic ictal EEG? | segment accuracy / F1 | no |
| Protocol & scoring work | is the evaluation honest? | event-F1, FP/day | n/a — no synthetic data |
| Trust-gated augmentation | can augmentation be made safe? | window AUROC | yes — balanced, non-clinical tasks |
| **This benchmark** | **is synthetic ictal EEG safe to train on, and can that be governed?** | **event-F1 under an FP/24h constraint** | **yes** |

The contribution is not the gate and not the generator. It is:

1. **The harm object** — a seizure-specific characterisation of what ungated synthetic augmentation does
   at event level (false-alarm inflation, event-sensitivity loss on unseen patients), novel independent
   of any gating.
2. **The event-level reformulation** — admission and the fail-closed rule expressed in event-F1 under an
   explicit FP/24h constraint, which AUROC cannot express.
3. **Evidence it can pay** — with a band-limited WGAN-GP, admitted synthetic data significantly improved
   TCN event-F1 (+0.083, 95% CI +0.037 to +0.132, 8/9 cells, Wilcoxon p = 0.008).
4. **The gate-conservatism finding** — the pre-registered gate delivers only +0.034 of the available
   +0.083. Reporting and diagnosing that shortfall is a result, not a blemish.

**One-sentence claim.** *Naive synthetic ictal augmentation is a hazard at the event level; a fail-closed
trust gate removes the downside; and a spectrally faithful generator can clear the gate and deliver a
significant benefit — but only if the gate is loosened enough to admit it.*

---

## 2. Trust-gated augmentation — the method we adapt

**[V] Choi, D.; Yip, C.; Choi, A.; Park, J. (2026).** *Trust-gated synthetic EEG augmentation reduces
performance drops when generalizing to new patients.* **npj Digital Medicine 9(1)**,
`10.1038/s41746-026-02778-0`, published **2026-05-25**. Confirmed in Crossref **and** Europe PMC.

Earlier version: *Fail closed trust gated synthetic augmentation governs tail risk under subject shift in
EEG*, bioRxiv `10.64898/2026.01.26.701638` v1, 2026-01-28. Same four authors, University of Calgary.

> **Two passes of this search wrongly called this an unreviewed preprint**, because the bioRxiv record
> shows v1 with no "Now published in" banner. That linkage is often absent. Hence the standing rule
> in §0.1.

**[V-id] Method and results — from the bioRxiv abstract, not the version of record:**
a real-data teacher scores candidates for label consistency and confidence; those above a quantile gate
are admitted; a fail-closed rule injects them only if validation **AUROC** improves by a pre-specified
margin. Named failure modes: **label-inconsistent** samples and **off-manifold** structure. Evidence on
chronic-pain resting-state EEG (189 subjects) and low-data motor-imagery BCI. Ungated augmentation
"harmed 56% of paired runs"; TGA "reduced harm to 24%".

> **Open item — do not draft from the block above.** The title changed between the preprint and the npj
> article, so the numbers and framing may have changed too. Re-read the npj version and re-derive it.
> There are also known divergences between the published specification and our implementation.
> Those must be resolved before we claim to have reformulated *their* method.

**For the paper.** "Adapt" and "reformulate", never "propose" or "novel" — and this matters *more* now
that TGA is peer-reviewed in a high-visibility venue: the gate is unambiguously prior art and reviewers
are likelier to know it. State the priority plainly in the second paragraph of related work.

> **Coincidence to keep apart in the text.** TGA reports harm in 56% of paired runs; our gate reverts 56%
> of TCN cells. Unrelated quantities.

---

## 3. Seizure-specific generative EEG — the nearest neighbours

### 3.1 GP-EEG — highest overlap in the literature

**[V] Moutonnet, N.; Corneck, J.; Tobar, F.; Mandic, D.** *Synthesizing Epileptic Seizures: Gaussian
Processes for EEG Generation.* **arXiv:2601.21752**, Imperial College London. *(Everything below from the
full text.)*

- **CHB-MIT (22 pediatric subjects, 198 expert-annotated seizure events, 18 bipolar channels, 256 Hz)
  and Siena (14 adults, 47 events)** — our corpus, our montage, our planned second dataset.
- They exclude chb12 and chb15 for computational reasons: **experiments run on 20 patients, not 22.**
  Ours uses all 24 across 23 leakage-safe groups. One clause; not a claim to lean on.
- Pipeline: SVD compression → changepoint detection with GP regression (quasi-periodic Matérn) →
  Conv-LSTM VAE domain adaptation → kernel-state discretisation → Poisson regime timing.
- Evaluation: TSTR and augmentation, both under an explicitly stated **leave-one-patient-out** scheme,
  with **EEGNet4,2**. Beats **TimeVAE, COSCI-GAN, ImagenTime**.
- Metrics: accuracy, precision, recall, F1 at **segment level**; fidelity via MDD, ACD, skewness and
  kurtosis differences, t-SNE, KDE.
- **No event-level metric, no false-alarm rate, no harm or tail-risk analysis anywhere in the paper.**

**For the paper.** The citation that most threatens a novelty claim and most strengthens a scoping claim.
It asks whether good seizures can be *generated*; we ask whether they are *safe to train on*. That delta
rests on a full-text read. Practical consequence: two generators looks thin beside it, and it has
established CHB-MIT + Siena as the expected pairing.

### 3.2 Abou-Abbas et al. — the foil

**[V] Abou-Abbas, L. et al. (2024).** *Generative AI with WGAN-GP for boosting seizure detection
accuracy.* Frontiers in Artificial Intelligence, `10.3389/frai.2024.1437315`, published 2 Oct 2024
(PMC11480023).

- **GroupKFold, 10 folds**, stated so that all samples from a patient fall entirely in train or entirely
  in test. Patients genuinely disjoint — not a leakage strawman.
- WGAN-GP + BiLSTM: real-only **86% → 91.73%**; SMOTE 85.25%, ADASYN 85.18%.
- **No event-level metric and no false-alarm rate anywhere in the paper.**

**For the paper.** The cleanest demonstration that the *metric*, not the split, conceals the clinical
failure mode. Open the related work with this contrast; it does more argumentative work than any other
single citation.

### 3.3 Lineage and the missing arm

Overwhelmingly seizure **prediction from pre-ictal signal**, not detection, and overwhelmingly
window-level. All identifiers below verified via the arXiv API or Crossref:

**[V]** EpilepsyGAN, arXiv:1907.10518 · Rasheed et al., arXiv:2012.00430 · multichannel preictal
synthesis, arXiv:2205.03239 · CDCGAN, `10.3390/technologies14020114` (Technologies, 12 Feb 2026).
**[V-id]** TripleGAN, PMC9440850 · GPT-based synthesis, `10.1145/3488933.3489016` · Truong et al. 2019
DCGAN-on-spectrograms.

This lets us write "the literature reports that it helps" without conceding that anyone measured what we
measure.

**Diffusion is the arm we did not run** and the one a reviewer will name: **[V]** arXiv:2303.06068;
diffusion-based seizure warning `10.1007/s12539-025-00750-2` (11 Aug 2025); **[V-id]** EEGDiffuser
(Neurocomputing 2026). Pre-empt in Limitations; the gate is generator-agnostic by construction.

---

## 4. Protocol, leakage and event-level scoring

### 4.1 The leakage statistic

**[V] Shafiezadeh, S.; Duma, G. M.; Pozza, M.; Testolin, A. (2024).** *A systematic review of
cross-patient approaches for EEG epileptic seizure prediction.* **J. Neural Eng. 21(6), 061004**,
`10.1088/1741-2552/ad9682`, December 2024.

**119 studies reviewed; over 96% evaluated using randomised or patient-specific splits** — only ~4%
genuinely leakage-free. A hard number, from a systematic review, in one of our target journals.

- **Caveat:** it covers seizure *prediction*. Phrase as "in the closely related seizure-prediction
  literature, a systematic review of 119 studies found…"
- Attribute it to Shafiezadeh — **not** to Ali et al. and not to CG-MambaNet, which cites it.

**[V] Supporting:** *Generalization or mirage? Data leakage and reported performance in neonatal EEG
seizure detection models: a systematic review*, BioData Mining, `10.1186/s13040-025-00516-y`, published
**8 January 2026**. **[V-id]** its reported statistics (leakage predicts performance p = 0.005,
R² = 0.202; partitioning explains 59.8% of leakage variance) — confirm against the full text before use.

**[V] Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods**, arXiv:2604.27033
(Li, Yan, Dou, Song, Zhang). Relevant as a survey of the cross-subject problem.
> An earlier draft attributed a quotation to this survey calling same-subject splits "fundamentally
> flawed for clinical EEG". **That quotation could not be substantiated and has been removed.** Do not
> reintroduce it without reading the full text.

### 4.2 SzCORE — align to it

**[V]** Framework: **arXiv:2402.13005** — event-based scoring by overlap; sensitivity, precision, F1,
FP/day; BIDS-EEG and HED-SCORE conventions. `epilepsybenchmarks.com`.

**[V]** Challenge: **arXiv:2505.18191**, current title *"Quantifying the Generalization Gap in Seizure
Detection: A Large-Scale Empirical Benchmark via the SzCORE Challenge"* (v2, May 2026 — the v1 title was
different; cite v2). Strictly held-out private corpus, Filadelfia Danish Epilepsy Centre: **65 subjects,
398 annotated seizures, 4,360 h**; 30 submissions from 19 teams, 28 evaluated.

| SzCORE challenge | Sensitivity | Precision | F1 | FP/day |
|---|---|---|---|---|
| Winner (Sz Transformer) | 37% | 29% | **32%** | 1.34 |
| Top-5 submissions | 30–58% | 19–29% | — | 1.34–14 |
| Full field (28 algorithms) | up to 99% | — | — | **1–290** |
| STORM (max sensitivity) | 99% | — | — | 290 |
| **This benchmark, real-only** | **57–72%** | **13–29%** | **12–30%** | **34–65** |

Our event-F1 and FP/24h are already SzCORE-shaped, so declaring compliance is nearly free and buys
external comparability plus a sanity anchor: **the best cross-subject event-F1 achieved in an open
challenge on unseen subjects was 0.32**, which places our 0.12–0.30 in a plausible regime rather than a
broken one.

> **Do not write "our best cell matches the challenge winner."** Different corpora, different
> annotation protocols, different seizure density. The legitimate claim is about *order of magnitude* —
> that cross-patient event-F1 in the low tenths is the state of the field, not evidence of a broken
> pipeline. Anything stronger is a cross-dataset leap a reviewer will catch. **See §9.1.**

### 4.3 Convergent methodology and controls

**[V] Ali, E.; Angelova, M. N.; Karmakar, C. (2024).** *Epileptic seizure detection using CHB-MIT
dataset: the overlooked perspectives.* R. Soc. Open Sci. 11:230601, `10.1098/rsos.230601`, May 2024.
Cross-subject event detection: 72.63% sensitivity subject-wise 5-fold, 75.34% leave-one-out, at 5.32/h
and 4.79/h false detection — **≈115–128 FP/24h**, against our 34–65. Qualitative protocol claims only.

**[V] CG-MambaNet**, arXiv:2606.08226 (Chen, M. et al.). Verified from the abstract: strict
**leave-one-patient-out with five independent random seeds**; AUC-ROC **0.8152 ± 0.0176 on CHB-MIT
(n = 22)** and **0.7104 ± 0.0261 on SIENA (n = 6)**; an event-level persistence filter reducing false
predictions to **0.32 alarms/hour**. The abstract states plainly that "most studies use data splits that
permit patient-level information leakage."
**[V-id]** The "351× window-to-event FPR reduction" cited earlier **could not be located in the
abstract** — do not use it without confirming from the body.
Seizure prediction, so not a competitor, but a published precedent for exactly our next steps 1 and 4.

**[V] Zare, M.** *A Negative-Control Protocol for Clinical EEG Foundation-Model Benchmarks: Dataset
Identity and External-Cohort Stress Testing*, arXiv:2607.24519 — **now at v3 (13 Aug 2026); the title
has changed twice.** Cite the current title. Frozen encoders under subject-disjoint validation:
"All five encoders decoded dataset identity at 1.000 before and after in-fold PCA-50."
> An earlier draft said "seven models" and paired the 1.000 against a clinical-label AUROC of 0.528.
> **Neither matches v3**, which reports task-specific results instead. Re-read before citing numbers.
> **Note also — this paper now includes CHB-MIT ictal detection (REVE, 0.793 AUROC).** That makes it a
> nearer neighbour than previously recorded, and worth reading properly rather than citing in passing.

Either way the methodological point stands and is worth citing: **negative-control suites are now an
expectation**, which raises the value of finishing our positive control.

---

## 5. Does augmentation help at all — the prior we test

**[V] Rommel, C.; Paillard, J.; Moreau, T.; Gramfort, A. (2022).** *Data augmentation for learning
predictive models on EEG: a systematic comparison.* J. Neural Eng.; arXiv:2206.14483. Abstract states
verbatim: "employing the adequate data augmentations can bring up to 45% accuracy improvements in low
data regimes." Note it studies signal *transformations*, not generative synthesis, and not rare-event
detection — the distinction does real work in the introduction.

**[V] Wang, Z.; He, Z.; He, X.; Wang, H.; Jia, T.; Luo, J.; Li, S.; Chen, X.; Wu, D. (2026).** *Synthetic
Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions.*
arXiv:2603.12296. Four-way taxonomy (signal-transformation-, feature-, model-, translation-based)
confirmed verbatim, and it benchmarks four paradigms — **one of which is epileptic seizure detection.**
Its stated evaluation principles are signal realism, physiological plausibility, downstream utility and
privacy preservation: **utility and privacy, but not harm or safety.** That absence is the positioning
gift — but read the seizure-detection benchmark section before citing, because it is nearer to us than
the first pass recorded.

**[V] Zhu, Y.; Zhou, Y.; Wang, P.; Qiao, L. (2026).** *A survey on data augmentation for EEG-based
emotion recognition and cognitive workload decoding.* Front. Neurosci., `10.3389/fnins.2026.1789468`.
Quoted **verbatim, confirmed in the full text**:

> "These issues are particularly important when evaluating DA under subject-dependent settings, where
> apparent performance gains may partly reflect optimistic data partitioning rather than true
> improvements in cross-subject generalization."

Our thesis, stated independently, from outside epilepsy. This is the quotation to use — it is the one
that survived checking.

---

## 6. Synthetic-data gating beyond EEG — where the gate goes next

**[V] Wu, Z.; Jeong, S. W.; Liu, Y.; Jung, Y. J.; Donnat, C.** *Filtering with Confidence: When Data
Augmentation Meets Conformal Prediction.* arXiv:2509.21479. "Conformal data augmentation, a principled
data filtering framework that leverages the power of conformal prediction to produce diverse synthetic
data while filtering out poor-quality generations with **provable risk control**" — verbatim. Reports up
to 40 pp F1 gain over unaugmented baselines and 4 pp over other filtered-augmentation baselines.

Replacing our fixed admission quantile `q` with a conformal admission set at target risk α turns next
step 1 from threshold-tuning into *"the gate carries a tunable, guaranteed harm budget."* Strongest
methodological upgrade the search produced.

> **Caveat that must be stated if we adopt it.** Its demonstrated tasks are topic prediction, sentiment
> analysis, image classification and fraud detection — **no physiological time series, and nothing with
> our class imbalance.** Porting it to rare-event multichannel EEG is a genuine extension with its own
> exchangeability assumptions to check, not a drop-in.

**[V] Ye, J. et al.** *Selective Synthetic Augmentation with Quality Assurance*, arXiv:1912.03837
(journal version: *Selective synthetic augmentation with HistoGAN…*, Medical Image Analysis). Selects
synthetic images "based on the confidence of their assigned labels and their feature similarity to real
labeled images" — the closest precursor of per-sample admission.

**[V]** *Ctrl-GenAug: Controllable Generative Augmentation for Medical Sequence Classification*,
arXiv:2409.17091 — medical sequence data with a filtering stage.
**[V]** *Confidence-Guided Diffusion Augmentation*, arXiv:2605.10916 — same classifier-confidence
primitive; note the domain is Bangla character recognition, so cite it for the mechanism only.

**[V] Shumailov, I.; Shumaylov, Z.; Zhao, Y.; Papernot, N.; Anderson, R.; Gal, Y. (2024).** *AI models
collapse when trained on recursively generated data.* Nature **631**, `10.1038/s41586-024-07566-y`,
24 July 2024 — "tails of the original content distribution disappear". Our over-smoothed cVAE, with
posterior collapse and a flattened spectrum, is the seizure-EEG instance of that tail loss; it explains
*why* naive admission harms rather than merely reporting that it does.

**[V] Gerstgrasser, M. et al. (2024).** *Is Model Collapse Inevitable? Breaking the Curse of Recursion by
Accumulating Real and Synthetic Data.* arXiv:2404.01413; OpenReview `5B2K4LRgmz`. Accumulating rather
than replacing real data avoids collapse — supports our real+synthetic mixing design.

**[V] Poļaka, Agnese Viktorija (2025).** *Riemannian Geometry-Preserving Variational Autoencoder for
MI-BCI Data Augmentation.* **Bachelor's thesis, University of Groningen** (`fse.studenttheses.ub.rug.nl/36690`);
supervisors I. P. de Jong and A. I. Sburlea.

> **Corrected twice.** The first pass called it uncitable; the second "corrected" that by citing an arXiv
> version — **which does not exist**; that ID belongs to an unrelated telecoms paper. There is no journal
> or arXiv version. Single author; the supervisors are not co-authors.
> **Recommendation: drop it.** Carry the classifier-dependence point on arXiv:2603.12296 instead.

---

## 7. Fidelity is not utility

Our finding that discriminator AUC **saturates at 1.0** and cannot grade a near-good generator is a
sharper instance of a complaint appearing across domains. **[V]** arXiv:2505.14206, *Challenges and
Limitations of Generative AI in Synthesizing Wearable Sensor Data* · **[V]** arXiv:2604.13125,
*Synthetic Tabular Generators Fail to Preserve Behavioral Fraud Patterns* · **[V]** arXiv:2510.19728,
*Enabling Granular Subgroup Level Model Evaluations by Generating Synthetic Medical Time Series*.

**For the paper.** Frame saturation as a contribution: it is *why* the gate had to be built on a teacher
detector's seizure-confidence rather than on a realism score. That choice is currently justified
internally but not defended against the literature.

---

## 8. Governance framing

**[V] Boraschi, D.; van der Schaar, M.; Costa, A.; Milne, R. (2025).** *Governing synthetic data in
medical research: the time is now.* Lancet Digital Health **7**(4), e233–e234,
**`10.1016/j.landig.2025.01.012`** (PMID 39984419).
> Earlier drafts used the PII `S2589-7500(25)00011-1` as if it were a DOI. It is not.

**[V]** *Protecting patient privacy in tabular synthetic health data: a regulatory perspective*, npj
Digital Medicine, `10.1038/s41746-025-02112-0`, 28 Nov 2025.

**[V-id]** The **FDA/NIST AISAMD** programme (Artificial Intelligence Synthetic Data for Medical
Devices). Described in the synthetic-data-in-healthcare literature (e.g. npj Digital Medicine
`10.1038/s41746-023-00927-3`) rather than verified against a primary FDA source — **cite the secondary
source, or find the FDA page before asserting the programme directly.**

One intro paragraph reframes the gate from "we built a safeguard" to "we operationalise an emerging
regulatory expectation for a rare-event clinical detector."

---

## 9. Risk register

### 9.1 The false-alarm gap (highest exposure)

Our sensitivity **beats** the challenge field (57–72% vs a top-5 range of 30–58%) and our precision sits
inside the top-5 band (13–29% vs 19–29%). But our **34–65 FP/24h** compares against a top-5 of
**1.34–14/day** — a ratio of roughly **2.4× at best and ~49× at worst** (34.1/14 ≈ 2.4; 65/1.34 ≈ 49).

> An earlier draft wrote "3–25×". That was simply wrong arithmetic and would have been trivially
> checkable by a reviewer.

*Defence, and it is real:* the challenge itself shows a steep sensitivity↔false-alarm trade-off — STORM
bought 99% sensitivity at 290 FP/day, HySEIZa 60% at 13. Our operating point is a normal position on that
curve, inside the field range of 1–290. CHB-MIT is also **≈2.2× denser in seizures** than the challenge
corpus (198/974 h ≈ 4.9 seizures/day vs 398/4,360 h ≈ 2.2), which raises FP/day at fixed precision.
**Action:** state this, and report one operating point at a leaderboard-comparable FP/day so at least one
row is directly comparable. **Caveat the comparison as cross-dataset throughout.**

### 9.2 An internal arithmetic check

The §6 numbers do not obviously reconcile. For LCT class-weighted (sensitivity 0.577, precision 0.287,
FP/24h 34.1): precision 0.287 at 34.1 FP/day implies ≈13.7 true-positive events/day, which at 57.7%
sensitivity implies ≈**23.8 seizures/day** — but CHB-MIT averages ≈4.9. Separately, the harmonic mean of
the reported sensitivity and precision is 0.383, not the reported event-F1 of 0.295.

Both are plausibly artifacts of averaging per-fold metrics rather than pooling events
(mean-of-F1 ≠ F1-of-means), which is legitimate — **but a reviewer will run exactly this arithmetic.**
Confirm whether §6 reports means-of-folds or pooled counts, state it in the caption, and report pooled
numbers alongside if it is the former.

### 9.3 Structural exposures

- **Single dataset.** GP-EEG set CHB-MIT + Siena as the expected pairing; Siena is now a comparability
  requirement, not a convenience.
- **n = 9, one significant comparison.** p = 0.008 survives a ~6-comparison correction, but report with
  that caution and call for replication.
- **The significant effect is on the augmented model, not the deployed gate.** Do not let the abstract
  blur +0.083 (available) into +0.034 (delivered).
- **The method we adapt is peer-reviewed in a high-visibility venue.** TGA is *npj Digital Medicine*,
  May 2026 — unambiguously prior art. Resolve the known implementation divergences
  before claiming to have reformulated their method.
- **Timing.** GP-EEG appeared January 2026, TGA published May 2026; the niche between them is unclaimed
  but narrowing.

---

## 10. Paste-ready sentences

Each of these rests only on **[V]** claims.

> Recent work shows that ungated synthetic EEG augmentation can silently harm cross-subject
> generalisation, and proposes a fail-closed trust gate to control it [Choi et al., npj Digit. Med.
> 2026]. We ask whether the same harm appears in patient-independent seizure detection at the clinically
> relevant event level, and whether an event-level reformulation of fail-closed gating controls it
> without discarding genuine benefit.

> In the closely related seizure-prediction literature, a systematic review of 119 studies found that
> over 96% were evaluated using randomised or patient-specific splits [Shafiezadeh et al. 2024]. We
> therefore adopt a grouped, patient-disjoint protocol throughout and verify disjointness
> programmatically.

> Strong seizure-specific generators now exist: GP-EEG synthesises multivariate ictal EEG on CHB-MIT and
> Siena and improves downstream EEGNet performance under leave-one-patient-out validation [Moutonnet et
> al.]. Such results are reported at the segment level; whether the resulting training data is *safe*
> under event-level, false-alarm-constrained evaluation has not been asked.

> Reported gains from generative augmentation in seizure detection are typically measured as window-level
> accuracy — for example, WGAN-GP augmentation raises accuracy from 86% to 91.7% under patient-disjoint
> GroupKFold [Abou-Abbas et al. 2024]. We show that this class of metric can conceal event-level harm,
> including false-alarm inflation on held-out patients.

> Under SzCORE event-based scoring, the best of 28 algorithms evaluated on a held-out corpus of 65 unseen
> subjects achieved an event-F1 of 0.32 [SzCORE challenge]. Cross-patient seizure detection therefore
> remains unsolved, and our real-data baselines of 0.12–0.30 event-F1 fall within the range this
> literature reports for the patient-independent setting.

---

## 11. Actions, in order

1. **Declare SzCORE compliance, and confront the false-alarm gap in the same breath** (§9.1). Report one
   operating point at a leaderboard-comparable FP/day; caveat all cross-dataset comparison.
2. **Resolve the §6 arithmetic** and state means-vs-pooled in the caption (§9.2).
3. **Re-derive the implementation-divergence claims yourself**, then fix any real divergence from
   the npj specification before claiming to reformulate it.
4. **Cite GP-EEG prominently; decide race or absorb.** At minimum state the delta; better, run its
   baselines or add it as a third generator arm.
5. **Re-cast the gate fix as conformal admission** (arXiv:2509.21479), stating the physiological-time-series
   caveat explicitly.
6. **Finish the positive control** — fix the base-train prefetch in `scripts/positive_control_gate.py`.
7. **Second dataset — Siena**, now on comparability grounds.
8. **Read, don't skim, the two nearer-than-expected neighbours:** the seizure-detection benchmark section
   of arXiv:2603.12296, and the CHB-MIT ictal results in arXiv:2607.24519 v3.
9. **Drop the Riemannian thesis; fix the Lancet DOI; add the governance paragraph.**

**Venue.** The harm characterisation, event-level gate and honest negative-space reporting still point at
J Neural Eng, IEEE JBHI or Clinical Neurophysiology — and J Neural Eng hosts the leakage review we lean
on. With the second dataset an ML4H or NeurIPS D&B framing is well supported; without it, it is not.
