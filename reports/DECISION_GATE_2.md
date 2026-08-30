# Decision Gate 2 — Phase 2 results

**Date:** 2026-08-30 · **Grids:** `downstream_gated_p2.csv` (81 rows, ratio ladder) and
`downstream_gated_p3.csv` (72 rows, pool-relative admission) · **Run:** 6 Vertex AI Spot A100
jobs, ~11 h wall-clock across two grids, no preemptions · **Detector:** `tcn` only
(see `README.md` "Budget-constrained scope") · **Code:** `dbb143c`

Reproduce:

```
python3 scripts/analyze_multiseed.py --csv .../downstream_gated_p2.csv --conds-expected 9 --ratio 0.10
python3 scripts/analyze_multiseed.py --csv .../downstream_gated_p3.csv --conds-expected 8
```

---

## Headline

**Two questions closed, one of them against the hypothesis, and one long-standing null explained
as an artifact of this benchmark's own divergence from the published method.**

1. **The dose confound is dead.** Phase 1's negative result did not depend on injecting at
   r = 1.0. Sampling the parent method's own operating band leaves the curve monotone with no
   interior peak.
2. **The gate as this benchmark built it cannot inject a dose at all.** Under the real-ictal
   admission reference it admits 6–23 windows whatever is requested, and no ratio ladder moves it.
3. **With the published rank cut restored, admission quality produces better models but no better
   deployments** — and the effect does not survive correction for fold dependence, failing in
   exactly the way the retired `+0.083` headline failed.

---

## Q5 — Was the negative result an artifact of dose?

**No.** `the verification record` §2.1 and `LITERATURE_AUDIT_2026-08-29.md` §2.3 identified a coverage
hole: realized injection took only two values, r ≈ 0.032 and r = 1.00, with nothing in
[0.05, 0.30] — the band the parent's validation ladder selects from, ceiling 0.30. Since the
dose–performance curve is theoretically U-shaped, the negative result rested on a single point
3.3× above the source method's explored range.

The `ungated` arm receives the requested dose exactly, so the ladder samples it directly
(mean test event-F1, n = 9):

| arm | event-F1 | Δ vs `real_only` | cells better | p_wilcoxon |
|---|---|---|---|---|
| `real_only` (r = 0) | 0.283 | — | — | — |
| ungated r = 0.10 | 0.264 | −0.019 | 5/9 | 0.910 |
| ungated r = 0.30 | 0.217 | −0.065 | 4/9 | 0.203 |
| ungated r = 1.00 | 0.279 | −0.015 | 5/9 | 1.000 |

**Monotone through the parent's band, no interior peak, nothing significant.** At r = 0.10 the
median delta is +0.001 with 5/9 better, so even the small negative mean is outlier-driven. FP/24h
degrades too (worst cell +102.6 at r = 0.10).

The coverage hole is closed and the Phase 1 conclusion stands on better ground than before. This
is the cheapest result in the project and the one that most strengthens what was already there.

## Q6 — Why did the gate never inject anything?

**Because the admission threshold is calibrated on data the teacher has memorised.** Measured in
the `_p2` grid:

| target r | target windows | actually admitted | realized r |
|---|---|---|---|
| 0.10 | 251 | **6** | 0.0024 |
| 0.30 | 752 | **23** | 0.0092 |

Tripling the request moved the admitted count from 6 windows to 23. The admission rate sits at
0.4–0.5 % of the pool **regardless of target**, so the ratio ladder cannot move the gated arm at
all. `reference="real_ictal"` thresholds on the teacher's own training positives, which it
classifies near 1.0, and almost no generated window clears that bar.

Verified in closed form from the admission functions (no GPU, `synthetic/trust_gate.py`):

- `reference="real_ictal"` — admits 0–1 windows in simulation at any q; 6–23 in the real grid.
- `reference="pool"` — admits **exactly** `min(oversample·(1−q), 1) · n_synth`, so
  `q = 1 − r/oversample` hits any target r.

Confirmed live in `_p3`: q = 0.9917 → **126** admitted (predicted 125), q = 0.9500 → **755**
(predicted 752), matched cell-for-cell between teacher and random selection in 9 of 9 cells.

**This is a fidelity-to-published-method finding.** TGA publishes a pool rank cut; this benchmark
substituted a real-ictal quantile (`the verification record` §2.1 lists it as a known divergence). The
substitution did not merely change the operating point — it disabled the mechanism.

Compounding it: **TGA publishes `K_min = 200`; this benchmark used 1.** At both `_p2` rungs,
**0 of 9 cells** reach 200 admitted windows (median 6 and 23, max 13 and 38). Under the source
method's own safeguard, this gate would have admitted nothing in every cell of Phase 1.

## Q2 (reopened) — Does admission quality beat a random draw at a real dose?

`DECISION_GATE_1.md` CORRECTION 2 retired the Phase 1 answer as confounded and underpowered. With
pool-relative admission the comparison is finally well-posed: identical pool, identical admitted
count, init-controlled, at doses of 126 and 755 windows.

**On the augmented models, teacher selection is clearly better:**

| dose | metric | teacher − random | cells | p_wil | **p_NB** | p_fold |
|---|---|---|---|---|---|---|
| r ≈ 0.30 | event-F1 | **+0.072** | 8/9 | 0.020 | **0.140** | 0.103 |
| r ≈ 0.30 | FP/24h | −22.3 | 6/9 | 0.098 | 0.394 | 0.242 |
| r ≈ 0.05 | event-F1 | +0.043 | 6/9 | 0.164 | 0.379 | 0.160 |
| r ≈ 0.05 | FP/24h | −31.2 | 5/9 | 0.359 | 0.498 | 0.130 |

**It does not survive.** Bonferroni for this family (8 tests) is p < 0.0063; nothing meets it. More
tellingly, the headline row collapses under the fold-dependence correction in precisely the
pattern that killed the previous headline:

| claim | Wilcoxon | Nadeau–Bengio |
|---|---|---|
| retired `+0.083` (Phase 1, `the verification record` §3) | 0.008 | 0.135 |
| this `+0.072` | 0.020 | **0.140** |

Same magnitude, same collapse, same cause: nine cells sharing three splits are not nine
independent observations. Reporting +0.072 at p = 0.020 would reproduce the exact error two
corrections were written to remove. **Direction and count are reportable; significance is not.**

### The part that does hold, and is more interesting

**Admission quality improves models and not deployments.**

| arm | pre-revert event-F1 | deployed event-F1 | reverted |
|---|---|---|---|
| gated q = 0.95 | **0.277** | 0.300 | **2/9** |
| random q = 0.95 | **0.205** | 0.307 | **6/9** |
| gated q = 0.9917 | 0.263 | 0.297 | 6/9 |
| random q = 0.9917 | 0.220 | 0.306 | 6/9 |

Teacher admission produces better models (0.277 vs 0.205) and gets them past validation three
times as often (2/9 reverts vs 6/9). Yet **the deployed policies are indistinguishable — 0.300 vs
0.307, with random nominally ahead.** The fail-closed stage reverts random's bad models to
`real_only`, which lands where teacher selection arrives by actually working.

This does not depend on a p-value: it is a statement about what the two policies deliver. It is
also consistent with the fallback defect already on record — reverting to the best simple baseline
instead of `real_only` is worth +0.048 at r = 0.30 (`_p2`, 6/9 cells revert).

**A good fallback makes a good gate redundant.** That is the governance finding of this phase.

---

## Analysis-only results (no GPU, from the emitted validation columns)

These are the re-selections `val_event_f1` / `val_fp_per_24h` / `val_auprc` were emitted for. All
are post-hoc among already-trained arms and cost nothing.

1. **The selector statistic changes almost every decision.** The event-F1 rule would admit **9/9**
   cells on validation (the gate admits 6/9 and 3/9 because the FP safety constraint also fires);
   the parent's threshold-free AUPRC rule at margin 0.01 admits **1/9 and 0/9**. The two disagree
   on 8 of 9 and 9 of 9 cells. Which statistic the fail-closed rule compares is not a detail, and
   `the verification record` §4.2's "two selected maxima" critique is the right frame.

2. **Fallback target**, confirming CORRECTION 2 at the new rungs: reverting to the best simple
   baseline rather than `real_only` is worth **+0.004** at r = 0.10 and **+0.048** at r = 0.30.

3. **Reference selection bias, now measured rather than argued.** Best-of-3 chosen on **test**
   scores 0.359; the same choice made on **validation** and scored on test gives 0.325. The bias
   CORRECTION 1 identified is **+0.035 event-F1**, and validation and test pick the same arm in
   only **5 of 9** cells. Any "vs registered reference" number in this repo that selected on test
   is inflated by roughly this much.

4. **K_min.** Covered under Q6: 0 of 9 cells reach the published 200.

---

## Generator fidelity — a metric that does not saturate

`gen_w2_dose_prediction.py` (CPU, 6.5 min, ~$0.04, off cached generators):

| fold | W₂(real ictal, synthetic) | real-vs-real floor | ratio |
|---|---|---|---|
| 0 | 0.465 | 0.072 | 6.4× |
| 1 | 0.450 | 0.073 | 6.1× |
| 2 | 0.445 | 0.071 | 6.3× |

W₂/floor ≈ 6.4 is interpretable and varies across cells, where `discriminator_auc` pinned at
1.000 in all 11 sweep rows and was retired as useless. This is a usable generator-quality number
the project did not previously have, and it gives the pool-filtering result its mechanism: at
q = 0.95 the teacher keeps the top 5 % of a pool that sits 6.4× further from real ictal than real
ictal sits from itself, so it is discarding mostly-bad windows rather than selecting redundant
ones.

**Recorded as a negative:** the falsifiable prediction the script was built around — rank folds by
W₂, their optimal doses rank in reverse — is **not testable here.** W₂ spread across folds is
1.04×, implying an r\* spread of 1.09×, far below what n = 9 can resolve. What W₂ does settle is
that folds are near-identical, so one shared ladder is correct and per-fold dosing is not worth
buying.

**Correction to the audit's premise:** `LITERATURE_AUDIT_2026-08-29.md` §4.2 proposed reading W₂
off "the cached critics". `WGANGPProvider._save_state` persists only the generator, so **no cached
critic exists for any cell**; W₂ is estimated empirically by sliced Wasserstein instead.

---

## Corrections to the plan this phase forced

**`the implementation plan` §2.2 should not have been dropped.** The audit reasoned that
`admit_indices` returns the top-k by teacher score, so `reference` only changes k — "a dose knob,
not a selection knob" — and recommended dropping the pool-relative sweep. The reasoning is correct
and the conclusion was wrong: **dose is exactly what was broken.** Pool-relative admission is the
only mechanism that makes the gated arm carry a meaningful number of windows, and therefore the
only route to a well-posed Q2. It was reinstated and is what produced this phase's results.

**`admit_always q0.90/q0.50 pool` frontier policies do not match this grid.** `POLICIES` hardcodes
q = 0.90 / 0.50; `_p3` ran q = 0.9917 / 0.9500, so those rows were skipped and the `_p3` frontier
covers only `always_revert` and `ungated`. The deployable-policy comparison above was computed
directly. A q-agnostic frontier is a small analysis fix, not a re-run.

---

## What this phase costs and what it bought

~$100–130 across two grids plus $0.04 for the W₂ diagnostic. Two failed submissions of my own
making (a CRLF shebang, and a literal `\n` in a line continuation) burned a couple of dollars in
Spot startup before the argv validation now in `vertex_submit.sh` existed.

Bought: the dose confound closed, the no-op-injector mechanism identified and explained in closed
form, Q2 well-posed for the first time and answered in direction, four analysis-only findings, and
a non-saturating fidelity metric.

**Not bought: a positive headline.** The one candidate failed the same correction that killed the
previous one.

---

## Status of the questions

| | question | status |
|---|---|---|
| Q1 | Does synthetic augmentation beat the simple baselines? | **No** — and the dose confound is now closed |
| Q2 | Does admission quality beat a random draw? | **Better models, equal deployments.** Direction favours the teacher (+0.072, 8/9); fails Nadeau–Bengio and Bonferroni |
| Q3 | Capacity or architecture? | **Vacuous** — no heterogeneity to explain |
| Q4 | What does the gate control? | **False-alarm tail**, replicated, and survives the circularity objection (`DECISION_GATE_1.md` CORRECTION 2) |
| Q5 | Was the negative an artifact of dose? | **No** |
| Q6 | Why did the gate never inject? | **Real-ictal reference; a divergence from the published rank cut** |
