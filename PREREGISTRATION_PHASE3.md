# Pre-registration — Phase 3

**Written 2026-09-02, before any Phase 3 GPU cell has run.** This file fixes the design, the
single primary endpoint, the harm margin rule and the interpretation of the positive control
before the data exist. It amends `PREREGISTRATION.md` (Phases 1–2), which stays in force where
this file is silent. Deviations from either file are declared in the manuscript, never absorbed.

**External timestamp.** A file in a git repository is not a pre-registration a referee will treat
as binding, because history can be rewritten. Before the first Phase 3 job is submitted, the
author will (a) tag the commit containing this file (`git tag -s prereg-phase3`), and (b) deposit
the file's SHA-256 with a third party that timestamps it — an OSF registration
(https://osf.io/registries) or a Zenodo record — and paste the registration DOI here:

> Registration DOI / URL: **[to be added before the first run]**
> SHA-256 of this file at registration: **[to be added]**

Until that line is filled the Phase 3 results are exploratory by this file's own rule.

---

## 1. What Phase 3 exists to settle

Phases 1–2 established parity between synthetic ictal augmentation and simple baselines on
TCN at n = 9, and the free re-analysis of 2026-09-02 (`scripts/analyze_validation_selection.py`)
found that plain validation model selection, with no admission stage, matches the trust gate on
event-F1 and matches or beats it on the false-alarm tail in every Phase 2 grid. Both results are
underpowered: three folds give a fold-corrected minimum detectable effect near 0.20 event-F1.

Phase 3 asks two questions and registers one primary test for each.

**Q-A (ceiling).** Does injecting *real* held-out ictal from other training patients improve
patient-independent detection? If not, no generator that matches the real cross-patient
distribution can, and the parity result is about the task, not the WGAN.

**Q-B (mechanism).** Is the trust gate's deployed performance distinguishable from validation
model selection over already-trained arms, at a design powered to see 0.10 event-F1?

## 2. Runs, in priority order (all GPU; none has started)

| # | run | design | cells | purpose |
|---|---|---|---|---|
| 1 | **same-seed floor** | `ungated` r = 0.30, TCN, folds 0–2 × 3 seeds, `--synth-seed-offset 1000` | 9 | the noise floor that applies to a paired synthetic-vs-reference delta (same init, different synthetic draw). Sets the harm margin in §3. Runs FIRST. |
| 2 | **positive control** | `scripts/positive_control_gate.py`, folds 0–2 × 3 seeds × {EEGNet, LCT, TCN}, base scarcity 0.5, oracle pool = the other 50 % of training events' real ictal windows, r = 1.0, gate q ∈ {0.90, 0.50} with `real_ictal` reference, `--with-simple-baselines` | 27 blocks | Q-A |
| 3 | **seeded baselines** | `real_only / class_weighted / classical_aug`, EEGNet + LCT, folds 0–2 × 3 seeds, `--ratios` empty | 54 | makes Phase 1's cross-family rows init-controlled and unlocks the validation-selection analysis for two more detectors |
| 4 | **leave-one-group-out** | `splits_logo23_seed42.json`, TCN, 1 seed, r = 0.30, pool reference q = 0.95, arms real_only / class_weighted / classical_aug / ungated / gated / random_gated | 23 blocks | Q-B at a design whose fold-corrected MDE is 0.087 (Nadeau–Bengio ρ = 0.056) |

Runs 1–3 use the committed `splits_seed42.json` so their cells stay paired with Phases 1–2 on the
test side. Run 4 uses the leave-one-group-out file with rotating validation
(`scripts/make_phase3_splits.py`); validation never repeats the near-fixed panel of Phases 1–2
(7 of 23 groups). If budget forces a choice, run 4 is trimmed before runs 1–3 are.

Cost, order of magnitude, extrapolated from Phase 2's per-cell rate and excluding generator refits
(run 4 needs 23 of them): runs 1–3 ≈ $60–90 on Vertex Spot A100, run 4 ≈ $150–200. All four fit
inside a few weeks of Kaggle's free quota (`scripts/kaggle/README.md`).

## 3. Harm margin rule (fixed here, applied after run 1)

The Phase 1–2 harm thresholds (Δevent-F1 < −0.01 or ΔFP/24h > +0.25) sit far below every floor
measured so far: the different-initialisation floor flags 50 % of identical re-runs as harm at
those margins (`analysis_tierB/phase3_free/harm_curve_null.csv`). Phase 3 therefore reports harm
two ways and registers the margin rule before run 1 produces the number:

1. **Harm curve (always).** Harm rate as a function of the margin, one axis at a time, plotted
   against the null curve from run 1 (same-seed pairs) and from the 27 different-initialisation
   pairs. A harm rate is quoted only where it is separated from both null curves.
2. **Point harm rate at the registered margin.** Let σ₁ be the standard deviation of the 9
   same-seed paired Δevent-F1 from run 1 and τ₁ the same for ΔFP/24h. The registered margins are
   **m_F1 = max(0.05, 2σ₁)** and **m_FP = max(2.0, 2τ₁)**. These are chosen so that the null harm
   rate at the margin is at most ≈ 2.5 % per axis under a normal floor. The values of σ₁, τ₁,
   m_F1 and m_FP will be written into this section, with the run 1 CSV hash, before any run 2–4
   harm rate is computed.

The Phase 1–2 thresholds remain reported in the supplement for continuity, labelled as below the
floor.

## 4. Primary endpoints and tests

Exactly one primary comparison per question. Everything else is secondary and reported with the
comparison count.

**Q-A primary.** Deployed `ungated` (real oracle pool) minus `real_only`, Δevent-F1 on test,
pooled over the three detectors' 27 cells, Nadeau–Bengio corrected t-test (ρ from the run's
split geometry), two-sided α = 0.05. Interpretation is fixed:

| outcome | reading |
|---|---|
| Δ > 0, p_NB < 0.05 | real cross-patient ictal helps: augmentation has headroom on this task and generator fidelity is the binding constraint. The WGAN parity result is about the generator. |
| \|Δ\| small, p_NB ≥ 0.05, NB 95 % CI inside ±m_F1 | augmentation with distribution-matched positives does not move patient-independent event-F1 at this scarcity: the parity result is about the task. Generator work is not the next step. |
| Δ < 0, p_NB < 0.05 | adding real other-patient ictal *harms*; the peri-ictal / label-noise mechanism becomes the leading explanation and is the next experiment. |
| CI wider than ±m_F1 and p_NB ≥ 0.05 | inconclusive; report as such, do not read as equivalence. |

Secondary for Q-A: per-detector Δ; gate admission rate on real ictal (the "does the gate admit
real seizures" question the script was first written for); revert rate; ΔFP/24h.

**Q-B primary.** Deployed `valsel4_fpguard` minus deployed `gated q = 0.95`, Δevent-F1 on test,
run 4's 23 cells, Nadeau–Bengio corrected t-test with ρ = 0.056, two-sided α = 0.05, **and** a
two-one-sided-tests equivalence bound at ±0.10 event-F1. The gate is declared *not
distinguishable from validation selection* only if TOST establishes equivalence at ±0.10; a
non-significant difference alone is reported as a failure to detect, not as equivalence.

Secondary for Q-B: the same contrast on ΔFP/24h and on the tail count (> +20 FP/24h); the
`ungated_failclosed` and `gated → valsel3` policies; the harm curves of §3.

## 5. Reference and reporting rules carried forward

- The registered reference is the best of `real_only / class_weighted / classical_aug` per cell
  **chosen on validation event-F1**, scored on test. The test-selected form is reported once in the
  supplement, labelled as inflated by the measured +0.035.
- Realized quantities are reported alongside requested ones in every table: injected count per
  cell, admitted count, revert reason.
- Any policy that can select the reference arm coincides with it in some cells; harm rates of such
  policies against that reference are reported with the coincidence count and are not quoted as
  evidence of safety.
- Run environment (`run_environment<tag>.json`) is committed beside every CSV; cells from
  different torch builds or devices are never pooled in one paired comparison without a
  same-arm reproduction check.

## 6. What would make Phase 3 fail

Recorded so the failure modes are the record's, not the reviewer's.

- Run 1's σ₁ exceeds 0.10: the same-seed floor is as large as the different-init floor, every
  paired delta in the project is dominated by training noise, and the 3-fold results should be
  described as single draws. The manuscript would then lead with the floor itself.
- Run 2 lands in the inconclusive row: 27 cells at ρ = 0.32 have an MDE near 0.13; a real effect
  of the size the literature reports (0.03–0.05 event-F1) would not be seen. This is acknowledged
  in advance; the positive control is powered to detect a *large* ceiling, not a small one.
- Run 4 finishes with fewer than 18 folds (budget): the MDE rises above 0.10 and the TOST margin
  in §4 cannot be reached; report the CI and stop.
