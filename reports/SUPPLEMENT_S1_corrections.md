# Supplement S1 — Correction history and withdrawn analyses

Companion to `reports/PREPRINT_DRAFT.md` v0.5. The manuscript presents the final design and
results once; this supplement records how the analysis reached them, including three results the
project published internally and later withdrew. It exists because a pre-registered study owes the
reader its error record, and because two of the withdrawn results were the kind of error the
manuscript's §4.6 warns other studies against.

Sources: `reports/DECISION_GATE_1.md`, `reports/DECISION_GATE_2.md`,
`reports/AUDIT_2026-08-31.md`, `PROGRESS.md`. Every number here re-derives from committed CSVs.

## S1.1 Withdrawn: "+0.083 event-F1, Wilcoxon p = 0.008, significantly improves TCN"

The first multi-seed grid reported a gated-arm gain over real_only of +0.083 event-F1 in 8 of 9
cells. It was withdrawn because (a) the Nadeau–Bengio corrected p was 0.135 with a 95 % CI of
[−0.032, +0.198], the fold-level p was 0.088 and nothing met Bonferroni; and (b) it was measured
against real_only while the pre-registration named the best simple baseline as reference, a
substitution worth +0.163 event-F1 for TCN in the Tier B grid where it was first measured, twice
the claimed effect.

## S1.2 Withdrawn: "admission quality does not matter" (Phase 1 form)

The Phase 1 matched-volume comparison was scored *after* the fail-closed revert. In 18 of 27
cells both arms had reverted to the same real_only model and were bit-identical by construction,
driving the difference to 0.009 where the pre-revert difference was 2.6–4.1× larger. It was also
run only at q = 0.90 under the real-ictal reference, where the gate admitted 0.3–13 % of the
requested dose. The Phase 2b form (pool reference, real dose, scored on the augmented model) is
the one the manuscript reports.

## S1.3 Withdrawn: "−0.063 vs class_weighted" as a magnitude

Phase 1's pool-free baselines ran with unseeded weight initialisation. `WGANGPProvider.generate`
calls `torch.manual_seed(seed)` and samples on the GPU, leaving the CPU stream that `build_model`
draws from at exactly `manual_seed(seed)`, so pool-drawing arms were seeded by accident; the three
baselines were not. Re-running them for TCN under the fix moved class_weighted from 0.364 to
0.271 and the comparison from −0.063 to +0.030. The paired σ of the re-run is the
different-initialisation floor the manuscript uses (0.099 / 0.159 / 0.171).

## S1.4 Withdrawn: the equivalence reading of "equal deployments"

The first draft of the Phase 2b result read the −0.007 deployed difference between teacher and
random admission as equivalence. That is the inference the paper criticises elsewhere. The TOST
bound the data support is ±0.121, wider than the +0.072 model-level effect, so the result is a
failure to detect.

## S1.5 Corrected: the reference was chosen on test

`analyze_multiseed.py` chooses the best simple baseline per cell by *test* event-F1. Validation
and test agree on the arm in 5 of 9 cells; the test-selected reference scores 0.359 and the
validation-selected one 0.325, an inflation of +0.035. All "vs registered" numbers from
2026-09-02 on use validation selection (`analyze_validation_selection.py`); Phase 1's CSV lacks
validation columns and cannot be re-selected.

## S1.6 Corrected: the "→ best-baseline" fallback row

An earlier frontier table showed the gate reverting to the best simple baseline as dominant. That
row chose the fallback on test, and in the roughly 80 % of Phase 1 cells that revert its delta
against the reference is zero by construction. The manuscript reports `gated → valsel3` (fallback
chosen on validation) instead, and notes the coincidence caveat for every policy that can select
the reference arm.

## S1.7 Corrected: the gate could not inject a dose

`TrustGateConfig.reference` defaulted to `real_ictal` and the driver never set it. The admitted
count sat at 0.4–0.5 % of the pool regardless of the request (6 of 251, 23 of 752). The pool
reference (TGA's published rank cut) was in the code from the start and unreachable from the
driver until Phase 2b. Now the default.

## S1.8 Corrected: the analysis command in the preprint returned NaN

`analyze_multiseed.py --ratio` applied the rung filter to the ratio-less baseline rows, emptied
the reference and printed an all-NaN report. The published numbers had been derived
independently and were correct; the command was not. Fixed 2026-08-31 and guarded in CI.

## S1.9 Corrected: the scarcity axis had been run

The preprint declared scarcity 0.5 / 0.25 and the 5-fold design un-run. `tables/tierB_core.csv`
holds the full registered design at seed 42 with the cVAE; parity holds at every rung.

## S1.10 Documented, not yet fixed in the data: the narrow validation panel

The carve routine sorts the training pool by seizure count and takes the richest groups until a
duration target. Across the three folds run, 7 of 23 groups ever served as validation; chb01 and
chb13 in all three. Every gate decision and every validation-selection result in the manuscript
was made on that panel. `chbmit.splits.make_splits(val_strategy="rotate")` and
`scripts/make_phase3_splits.py` fix it for Phase 3.

## S1.11 Version log of the manuscript

| version | date | change |
|---|---|---|
| v0.1 | 2026-08-29 | first full draft |
| v0.2 | 2026-08-30 | equivalence claim restated with TOST; title no longer asserts the published method is unevaluable; tail control labelled a replication |
| v0.4 | 2026-08-31 | −0.063 withdrawn; deviations 3 and 6 corrected; +0.163 re-attributed; §8 command fixed; figures rendered |
| v0.5 | 2026-09-02 | rewritten as a single account; validation-selection competitors added (§4.5); harm as a curve (§4.6); reference chosen on validation; power design; references verified; correction history moved here |
