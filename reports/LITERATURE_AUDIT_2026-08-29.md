# Literature Audit — 2026-08-29

Independent literature check run after Decision Gate 1, at the point of deciding what Phase 2
should be. Scope: (a) verify the parent citation at the registries, (b) find the closest prior
work and locate this project's gap in it, (c) test whether Phase 1's findings are novel, already
known, or contradicted.

**Verification marks** follow `the verification record`:
**[S]** read from the primary source during this review · **[R]** recomputed from files in this
repo · **[U]** unverified.

> **Fetch caveat, stated once and applying throughout.** Full texts were read via HTML/derived
> renderings, not the publisher PDFs of record. Registry metadata (Crossref, Europe PMC) is
> authoritative as quoted. **Every table of numbers attributed to GP-EEG or to TGA below must be
> re-checked against the PDF before it enters a manuscript.** This repo's error history is
> entirely secondhand numbers; do not extend the streak.

---

## 1. The parent citation is correct — confirmed at two registries

`10.1038/s41746-026-02778-0` resolves to exactly one work [S]:

| field | value |
|---|---|
| title | *Trust-gated synthetic EEG augmentation reduces performance drops when generalizing to new patients* |
| authors | Daniel Choi, Cordelia Yip, Andrew Choi, Junho Park |
| venue | npj Digital Medicine **9(1)**, art. **634** |
| date | 2026-05-25 |
| PMID | **42185473** (Europe PMC, 1 hit, no PMCID, not open access) |

`the verification record` §2 is accurate and `LITERATURE.md` §10.1's denial was the error, as already
recorded. Note the DOI does **not** surface in general web search — only via direct registry
lookup. A failed search is not evidence of absence; that is the same trap `the verification record`
names in its verification standard.

**No independent replication, audit, or evaluation of TGA exists** [S]. Searches for one return
only the preprint and this repo's own framing. This project is the first — which is a stronger
claim than it has been making.

---

## 2. Three corrections to documents in this repo

### 2.1 · SzCORE **was** retitled at v2 — `the verification record` §6 is wrong

Fetched the v1 header directly [S]:

| version | date | title |
|---|---|---|
| **v1** | 19 May 2025 | *SzCORE as a benchmark: report from the seizure detection challenge at the 2025 AI in Epilepsy and Neurological Disorders Conference* |
| **v2** | 18 May 2026 | *Quantifying the Generalization Gap in Seizure Detection: A Large-Scale Empirical Benchmark via the SzCORE Challenge* |

`the verification record:375-376` asserts "**identical titles** — `related_work_v2`'s 'retitled at v2'
is wrong", and `:452` repeats it as a correction table row. **Both are wrong; `related_work_v2.md`
was right.** Corrected in place, this file cited as the source.

This is the third reversal on a citation detail in this repo (TGA existence, then TGA venue, now
SzCORE title). The pattern is consistent: each error came from asserting a *negative* from a
search rather than checking a registry or a version header. The challenge's substantive numbers
(winner *Sz Transformer*: F1 32%, sens 37%, prec 29%, 1.34 FP/day; 65 subjects, 4,360 h) are
confirmed correct [S].

### 2.2 · The parent's random-gating control mostly favours **random** — report its numbers

`the verification record` §2.2 quotes the parent's conclusion accurately but never reports the
underlying comparisons. From the full text (PainMunich, 25% scarcity; harm rate, lower better) [S]:

| comparison | trust gating | random gating | winner |
|---|---|---|---|
| ShallowConvNet q=0.99 | **0.20** | 0.32 | trust |
| ShallowConvNet q=0.95 | 0.48 | **0.24** | **random** |
| EEGNet q=0.99 | 0.48 | **0.36** | **random** |

Two of three favour random. From this the authors conclude that dose alone is insufficient and
that the gate's curation adds safety beyond downsampling. That reading is generous to their own
data, and the hedge they publish — *"reducing synthetic exposure can sometimes help, but it does
not provide a consistent safety guarantee and does not systematically reproduce trust gating"* —
is the defensible version.

**Consequence for this project.** A properly-powered matched-volume control does not merely
"resolve a control the parent left open" — it **corrects the parent's central mechanism claim**.
That raises the ceiling on Q2 considerably, and it is why Q2's retraction in
`DECISION_GATE_1.md` (CORRECTION 2) matters: the re-run has to be strong enough to carry a
correction, not just a null.

### 2.3 · The dose deviation is real but mis-framed — it is a *coverage hole*, not a ratio mismatch

`the verification record` §2.1 correctly records the parent's ladder
`r ∈ {0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30}` and this repo's hardcoded
`synthetic_ratio = 1.0`, and marks it **Critical**. It frames the defect as *conflating* dose with
admission quality. That framing is right but incomplete, and the incomplete part is what
threatens the headline.

Realized injection ratios in the `_v2` grid, `r = n_admitted / n_train_pos` [R]:

| arm | median admitted | realized **r** | vs parent's ladder |
|---|---|---|---|
| `gated` q=0.90 | 80 (min 0, max 580) | **0.032** | inside — near the parent's median injection of **58** windows |
| `random_gated` q=0.90 | 80 (matched) | **0.032** | inside |
| `gated` q=0.50 | 2,508 | **1.00** | **3.3× above the ceiling** |
| `ungated` | 2,508 (full) | **1.00** | **3.3× above the ceiling** |

So the grid's dose coverage is **bimodal — r ≈ 0.03 or r = 1.00 — with nothing in the 0.05–0.30
band where the parent's validation ladder actually selects.** The consequences are asymmetric and
both are bad:

- The arms that carry the **negative result against the simple baselines** (`ungated`, `gated`
  q=0.50) run at 3.3× the parent's maximum and ~43× its median injection. "Synthetic augmentation
  does not beat class weighting" is currently measured only at a dose the source method never
  used and the augmentation literature predicts is past the optimum.
- The arm that **is** in the parent's range (`gated` q=0.90) reverts in **81% of cells** [R], so
  it barely ships a model to test.

This is the single most exposed claim in the project, and a referee who knows the parent paper
will reach it on first read.

---

## 3. Closest prior work — GP-EEG, and the gap it leaves open

**Moutonnet, Corneck, Tobar & Mandic (2026).** *Synthesizing Epileptic Seizures: Gaussian
Processes for EEG Generation.* arXiv:2601.21752, 29 Jan 2026 [S].

This is the paper this project will be compared against. **CHB-MIT and Siena**, leave-one-patient-out,
EEGNet, four generators evaluated as augmentation. Reported Δ vs baseline [S]:

| method | CHB-MIT ΔF1 | ΔRecall | Siena ΔF1 |
|---|---|---|---|
| COSCI-GAN | **−10.52** | −17.23 | −0.54 |
| TimeVAE | **−14.23** | −22.31 | −12.70 |
| ImagenTime | −3.70 | −8.16 | −0.27 |
| GP-EEG | +2.75 | +2.26 | +5.11 |

Two findings that matter here:

1. **Harm from synthetic ictal augmentation on CHB-MIT is already published**, with this
   project's exact signature — precision up, recall down hard. The harm thesis is not novel in
   direction. What is novel is *characterising* it: pre-registered thresholds, tail-risk
   reporting, and an explicit false-alarm axis.
2. **The gap is precisely this project's contribution set** [S]:
   - **no non-generative baselines** — no class weighting, no oversampling, no classical augmentation;
   - **sample-level metrics only** (1024-sample segments); **no event-level scoring**;
   - **no FP/day**, so no operating-regime comparison and no tail-risk statement.

GP-EEG's own +2.75 F1 headline on CHB-MIT is therefore a sample-level number with no
class-weighting comparison — exactly the fragility this project quantified when it found that
baseline choice swings the apparent effect (±0.13) by more than the effect itself (0.01–0.06).

**Siena is now a comparability requirement, not reach** (`the implementation plan` §4.2 already
says so; this confirms it from the source). GP-EEG established CHB-MIT + Siena as the expected pairing.

---

## 4. Phase 1's findings against the literature

### 4.1 · Q2 (random ≈ teacher selection) — known phenomenon elsewhere, novel here, with a mechanism

The coreset / data-pruning literature reports that **random selection is a surprisingly strong
baseline**, and specifically that learned importance-based selection **degrades to or below random
at high pruning rates**, even where it wins at low ones [S].

This supplies Q2 with something it lacked: a mechanism and a prediction. At q=0.90 the gate keeps
~80 of a 15,048-window pool — a **99.5% prune**, exactly the regime where the literature predicts
learned selection collapses. That is a principled reason to expect the Phase 1 null *and* a
concrete prediction that selection should begin to matter as the admitted fraction grows.

The other session's fix (running `random_gated` at every q, not just q=0.90) tests this — but only
partly: q=0.50 still runs at `synthetic_ratio = 1.0`, so both control arms sit above the parent's
ceiling. **The discriminating axis is r, not q.**

### 4.2 · Dose — there is now a theory, and it is testable from cached checkpoints

**Shidani, Farghly, Sun, Ganjgahi & Deligiannidis (Apple / Oxford).** *Beyond Real Data: Synthetic
Data Through The Lens Of Regularization.* arXiv:2510.08095 [S].

Theorem 3.1 gives a stability-based generalization bound of the form

```
baseline risk  +  λ · ξ · W₂(p_x, p'_x)  +  (1 − λ) · stability
```

with λ the synthetic mixing ratio. The bound is **U-shaped in λ**: too little synthetic gives high
variance, too much lets distributional mismatch dominate, and *"for a fixed distributional
discrepancy W₂(p_x, p'_x), there exists an optimal mixing parameter λ"* [S]. Empirically validated
across datasets; broader surveys put the optimum around r ≈ 0.2–0.5 for most classification tasks [S].

Two consequences:

1. **The negative result needs the ladder to be safe.** A single uncontrolled point on a curve
   known to be U-shaped, chosen 3.3× above the source method's ceiling, cannot support "synthetic
   augmentation does not help."
2. **A cheap, novel, theory-linked contribution is available.** The optimum is a function of
   W₂(real, synthetic) — and this project trained **WGAN-GP**, whose critic *is* a W₂ estimator.
   Predicting the per-fold dose optimum from the cached critics needs no GPU training and ties the
   generator choice to current theory. Nothing in the seizure literature does this.

### 4.3 · Tail control — this is the parent's own thesis, so it cannot carry novelty

The parent's title is *"Fail closed trust gated synthetic augmentation **governs tail risk** under
subject shift in EEG"*, and it self-describes as *"an auditable, fail-closed control plane …
decoupled from generator architecture"* and *"a safety and governance method rather than a
clinically deployable … biomarker"* [S]. Its harm framing — *"with what probability does
augmentation harm subject-disjoint generalization by a clinically meaningful margin?"* — is the
same object as this project's Q4.

**Q4 is a replication, not a discovery.** It remains valuable and should be reported as what it
is: the **first independent replication of TGA**, the first at **event level**, and the first with
an explicit **FP/24h** axis on a rare-event clinical task. But `the project notes`'s "ACTUAL HEADLINE"
framing overstates it, and a referee holding the parent paper will say so.

Novelty has to be carried by: the seizure-specific event-level reformulation (the parent contains
no seizure content — confirmed [S]), the registered simple-baseline comparison, and Q2.

### 4.4 · The class-weighting result has direct literature support

*Stronger Baseline Models – A Key Requirement for Aligning Machine Learning Research with Clinical
Utility*, arXiv:2409.12116 [S], names **class balancing via inverse frequency weighting** as a
required clinical-ML baseline and documents cases where simple models match or beat complex ones
(logistic regression 0.80 vs transformer 0.77 AUROC; baselines ~0.83 vs autoencoder 0.70). Related:
*Meaningless comparisons lead to false optimism in medical machine learning* (arXiv:1707.06289),
and work quantifying how proposed-method win rates fall as the number of baselines rises [S].

The `class_weighted` finding is therefore **not** an idiosyncrasy of this benchmark — it is an
instance of a documented, citable pattern, which makes it considerably more defensible.

### 4.5 · Calibration — the parent's numbers confirmed verbatim

Ungated augmentation was the best-calibrated arm: **Brier 0.359 vs real-only 0.390**; **ECE 0.327
vs 0.384**; strict gating (q=0.99) returns to real-only levels (Brier 0.385, ECE 0.384) [S].
`the verification record` §2.3 is accurate. The parent's own framing — *"governance choices can trade
off different clinically relevant axes (tail risk vs calibration vs decision utility)"* — is a
better frame for `the implementation plan` §0.3 than "the parent's most uncomfortable finding".

---

## 5. What this changes

| item | before | after |
|---|---|---|
| **Ratio ladder** (`the implementation plan` §2.3) | third item of Phase 2, "restore for completeness" | **first priority; a validity requirement.** Closes a coverage hole at r ∈ [0.05, 0.30] that the headline negative result currently depends on |
| **Pool-relative q sweep** (§2.2) | a Phase 2 experiment | **drop the sweep.** Conditional on admitted count k, `admit_indices` returns the same top-k set for either reference [R] — it is a dose sweep. The parent's real knob is r |
| **Q2 / `random_gated`** | a control that "resolves against admission quality" | **a correction to the parent's mechanism claim.** Needs to be run at doses inside the parent's range to carry that weight |
| **Q4 / tail control** | "the ACTUAL HEADLINE" | **first independent replication**, first at event level with FP/24h. Real, but not novel in kind |
| **Siena** (§4.2) | reach | **comparability requirement** — GP-EEG set the pairing |
| **W₂ dose prediction** | not considered | new, cheap, theory-linked; runs off cached WGAN critics |
| **`class_weighted` result** | an awkward in-house finding | an instance of a documented clinical-ML pattern, with citations |

**Recommended Phase 2 shape:** one instrumented grid over the ratio ladder
`r ∈ {0, 0.02, 0.05, 0.10, 0.20, 0.30}` crossed with `selection ∈ {teacher, random}`, r selected on
validation as the parent does — which *is* implementing the published method, far more than
`reference="pool"` is. Everything else in the gate's policy space (comparison reference, fallback
target, selection statistic, `K_min`) is a re-selection among already-trained arms and belongs in
analysis, not in a grid.

---

## 6. Sources

| # | work | how verified |
|---|---|---|
| 1 | Choi, Yip, Choi & Park (2026), npj Digit. Med. 9(1) 634, `10.1038/s41746-026-02778-0` | Crossref API + Europe PMC REST, PMID 42185473 |
| 2 | TGA preprint full text, bioRxiv `10.64898/2026.01.26.701638v1` | full text |
| 3 | Moutonnet, Corneck, Tobar & Mandic (2026), arXiv:2601.21752 (GP-EEG) | abstract + full text |
| 4 | Shidani, Farghly, Sun, Ganjgahi & Deligiannidis, arXiv:2510.08095 | full text |
| 5 | *Stronger Baseline Models…*, arXiv:2409.12116 | full text |
| 6 | SzCORE challenge, arXiv:2505.18191 v1 and v2 | both version headers |
| 7 | *Meaningless comparisons…*, arXiv:1707.06289 | search-level only **[U]** |
| 8 | Coreset / data-pruning "random is strong" literature (e.g. arXiv:2210.15809) | search-level only **[U]** |

Items 7–8 are cited for a qualitative pattern only; check them before quoting any number.
