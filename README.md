# CHB-MIT Synthetic Seizure Augmentation Benchmark

A leakage-safe, patient-independent benchmark that treats training-time synthetic
ictal EEG augmentation as a **controlled intervention** and asks whether its
benefit depends jointly on the **generator** that produced the data and the
**detector** that consumes it. Implements the v5.0 plan
(`chbmit_synthetic_augmentation_generator_x_detector_v5_0.md`).

The contribution is the **benchmark and its protocol**, not a new generator:
every outcome (no benefit / generator×detector interaction / broad benefit) is a
defensible result (see plan §19).

## What's implemented

| Axis | Components |
|---|---|
| **Data pipeline** | RECORDS/summary parser, channel-alias map + montage audit, EDF→zarr preprocessing, lazy windows, balanced patient-group splits, event-level scarcity |
| **Detectors** (`models/`) | EEGNet, LCT (lightweight conv-transformer), TCN — single-logit `(N,C,T)` contract |
| **Generators** (`synthetic/`) | patient-independent WGAN-GP, patient-independent conditional VAE, precomputed loader (provenance-enforced) |
| **Baselines** (`augmentation/`) | class weighting / focal, balanced sampler, classical augmentation |
| **Evaluation** (`evaluation/`) | SzCORE event metrics (via `timescoring`), window metrics (AUPRC-first), latency, sensitivity@FA-budgets, patient-level aggregation, paired Wilcoxon + bootstrap CIs |
| **Quality** (`synthetic/quality_checks.py`) | PSD, cross-channel correlation, MMD, NN memorization, real-vs-synthetic discriminator AUC |
| **Orchestration** (`experiments/`) | single-cell runner, shared grid runner, Tier A/B/C/D scripts, analysis + figures |

All leakage rules (plan §5) are enforced in code: patient-group split before
windowing, train-only generators/normalization, identical real subset across
conditions, synthetic data training-only, event metrics on continuous real
timelines, full synthetic-window provenance.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q          # 67 tests, runs entirely on synthetic fixtures (no data needed)
```

CPU-only by default (the bundled PyTorch is the CPU build). No GPU required to run
the tests; the full Tier B grid is intended for a GPU/disk-capable machine.

## Data

The pipeline never ships data. Place datasets under `data/`:

```
data/raw_chbmit/chbXX/chbXX_*.edf + chbXX-summary.txt   # CHB-MIT v1.0.0
data/raw_siena/PNxx/*.edf + Seizures-list-PNxx.txt       # Siena (Tier C)
```

CHB-MIT: <https://physionet.org/content/chbmit/1.0.0/> (~40 GB full).
Files listed in a summary but absent on disk are marked `edf_missing` and skipped,
so you can work with a subset. Prefer the SzCORE-formatted releases for
standardized scoring/montage.

## Running

```powershell
# Tier A — dev sanity (real-only + simple baselines, EEGNet+LCT)
python -m experiments.run_tierA_dev --folds 5 --epochs 40

# Tier B — core grid (EEGNet+LCT+TCN x 5 conditions x WGAN-GP/cVAE x scarcity x seeds)
python -m experiments.run_tierB_core            # full grid (GPU box)
python -m experiments.run_tierB_core --folds 0 --seeds 42 --scarcity 1.0 \
    --detectors eegnet --generators cvae --epochs 20   # quick subset

# Tier D — LOPO;  Tier C — Siena verification
python -m experiments.run_tierD_lopo --generator cvae
python -m experiments.run_tierC_siena --raw-root data/raw_siena

# Analysis: paired generator x detector deltas + Figure 8 heatmap
python -m experiments.analyze results_chbmit_synthetic/tables/tierB_core.csv
```

Every runner writes manifests, channel-audit, splits, scarcity event manifests,
per-cell metrics (CSV + raw JSON), quality JSON, and figures under
`results_chbmit_synthetic/`.

## Repository layout

Mirrors plan §21: `chbmit/` (data), `models/`, `augmentation/`, `synthetic/`,
`evaluation/`, `experiments/`, `tests/`, `configs/`, `results_chbmit_synthetic/`.

## Claim discipline

Generator-specific claims when only one provider is used; no clinical-deployment
or "clinically realistic" claims; no "helps/does-not-help all detectors" claim
unless shown (plan §4, §27).
