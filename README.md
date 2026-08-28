# CHB-MIT Synthetic Seizure Augmentation — Harm & Trust-Gate Benchmark

A leakage-safe, patient-independent benchmark whose **primary object of study is harm**:
can training-time synthetic ictal EEG **silently degrade** seizure detection on held-out
patients in ways that *mean window metrics hide*? We characterize that harm at the
clinically relevant **event level** (false-alarm inflation, event-sensitivity loss,
worst-fold degradation, tail risk), map where it depends on the **generator** and the
**detector**, and test whether a **fail-closed trust gate** — *adapted* from prior
subject-shift work (TGA: Choi et al., *npj Digital Medicine* 9(1) 634, 2026,
`10.1038/s41746-026-02778-0`), not proposed here — removes the harm and at what
cost to genuine gains. Implements the v5.3 reframe
(`chbmit_synthetic_augmentation_reframe_v5_3.md`).

The contribution is **not** a generator and **not** the gating concept. It is the
event-level, tail-risk **harm characterization** of synthetic ictal augmentation, plus the
seizure-specific event-level **reformulation** of the gate's admission/fail-closed criteria.
Every outcome — harm exists / harm is generator×detector-specific / synthetic doesn't beat
simple baselines / the gate fails to transfer — is a defensible, pre-registered result
(see [PREREGISTRATION.md](PREREGISTRATION.md), v5.3 §6).

## Research questions (v5.3 §1)

- **Q1 (harm):** does ungated synthetic augmentation inflate FP/24h and lose event
  sensitivity on held-out patients, and do mean window metrics conceal it?
- **Q2:** does the harm/benefit depend on the generator and detector?
- **Q3:** does a validation-selected, event-level fail-closed gate reduce harm **on test
  patients**, and how much genuine gain does it sacrifice?
- **Q4:** does ungated synthetic beat strong simple baselines (class weighting, classical
  augmentation) at the event level at all?
- **Q5:** do the patterns reproduce under a reduced Siena verification?

## What's implemented

| Axis | Components |
|---|---|
| **Data pipeline** | RECORDS/summary parser, channel-alias map + montage audit, EDF→zarr preprocessing, lazy windows, balanced patient-group splits, event-level scarcity |
| **Detectors** (`models/`) | EEGNet, LCT (lightweight conv-transformer), TCN — single-logit `(N,C,T)` contract |
| **Generators** (`synthetic/`) | patient-independent WGAN-GP, patient-independent conditional VAE, precomputed loader (provenance-enforced) |
| **Baselines** (`augmentation/`) | class weighting / focal, balanced sampler, classical augmentation |
| **Trust gate** (`synthetic/trust_gate.py`) | teacher = reused real-only model; q-quantile window admission calibrated on real ictal; **event-level fail-closed** selection (val event-F1 margin under an FP/24h safety constraint) |
| **Harm / tail-risk** (`evaluation/stats.py`) | harm rate at a pre-registered threshold, worst-fold delta, CVaR(0.10), paired Wilcoxon + bootstrap CIs |
| **Evaluation** (`evaluation/`) | SzCORE event metrics (via `timescoring`), window metrics (AUPRC-first), latency, sensitivity@FA-budgets, patient-level aggregation |
| **Quality** (`synthetic/quality_checks.py`) | PSD, cross-channel correlation, MMD, NN memorization, real-vs-synthetic discriminator AUC |
| **Orchestration** (`experiments/`) | single-cell runner, shared grid runner (real-only trained first and reused as teacher), Tier A/B/C/D scripts, harm/gate analysis + figures |

All leakage rules are enforced in code: patient-group split before windowing, train-only
generators/normalization, identical real subset across conditions, synthetic data
training-only, event metrics on continuous real timelines, full synthetic-window
provenance. The gate decides on **validation only** — it never sees test patients.

### Conditions (core grid, v5.3 §5.4)

`real_only` · `class_weighted` · `classical_aug` · `ungated_synthetic_aug` ·
`trust_gated_synthetic_aug`. Core generator **cVAE** (WGAN-GP is an appendix axis); the
gated condition reuses the `real_only` detector as its teacher, so the gate adds only cheap
inference plus one already-trained augmented model — no extra teacher training.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q          # 81 tests, runs entirely on synthetic fixtures (no data needed)
```

CPU-only by default (the bundled PyTorch is the CPU build). No GPU required to run the
tests; the full Tier B grid is intended for a GPU/disk-capable machine.

## Data

The pipeline never ships data. Place datasets under `data/`:

```
data/raw_chbmit/chbXX/chbXX_*.edf + chbXX-summary.txt   # CHB-MIT v1.0.0
data/raw_siena/PNxx/*.edf + Seizures-list-PNxx.txt       # Siena (Tier C)
```

CHB-MIT: <https://physionet.org/content/chbmit/1.0.0/> (~40 GB full).
Files listed in a summary but absent on disk are marked `edf_missing` and skipped,
so you can work with a subset. Prefer the SzCORE-formatted releases for standardized
scoring/montage.

## Running

```powershell
# Tier A — dev sanity (real-only + simple baselines, EEGNet+LCT)
python -m experiments.run_tierA_dev --folds 5 --epochs 40

# Tier B — core grid (real_only/class_weighted/classical_aug/ungated/gated x EEGNet+LCT+TCN
#          x cVAE x scarcity x seeds; real_only is reused as the gate's teacher)
python -m experiments.run_tierB_core            # full grid (GPU box)
python -m experiments.run_tierB_core --folds 0 --seeds 42 --scarcity 1.0 \
    --detectors eegnet --generators cvae --epochs 20   # quick subset

# Tier D — LOPO;  Tier C — reduced Siena verification (Q5)
python -m experiments.run_tierD_lopo --generator cvae
python -m experiments.run_tierC_siena --raw-root data/raw_siena

# Analysis: harm/tail-risk tables, gen x detector heatmap, gate-behavior summary
python -m experiments.analyze results_chbmit_synthetic/tables/tierB_core.csv
```

Every runner writes manifests, channel-audit, splits, scarcity event manifests,
per-cell metrics (CSV + raw JSON, including gate admission/revert provenance), quality
JSON, and figures under `results_chbmit_synthetic/`.

## Repository layout

`chbmit/` (data), `models/`, `augmentation/`, `synthetic/` (incl. `trust_gate.py`),
`evaluation/`, `experiments/`, `tests/`, `configs/`, `results_chbmit_synthetic/`.

## Claim discipline (v5.3 §4)

The gate is **adapted / reformulated**, never "proposed" or "novel" — the harm
characterization is the contribution. No clinical-deployment or "clinically realistic
synthetic EEG" claims; no "helps/does-not-help all detectors" claim unless shown;
generator-specific claims when a single provider is used. Pre-registered thresholds and
margins live in [PREREGISTRATION.md](PREREGISTRATION.md).
