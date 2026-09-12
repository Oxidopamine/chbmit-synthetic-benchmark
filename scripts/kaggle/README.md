# Running the Phase 3 GPU cells on free quota (Kaggle / Colab)

The two experiments that change what the paper can claim (`PREREGISTRATION_PHASE3.md` Sec 2)
need roughly 60 to 120 A100-hours in total. Kaggle gives 30 GPU-hours a week (P100 or 2×T4) and
Colab's free tier gives intermittent T4 time, so the whole of Phase 3 fits inside a few weeks
of free quota. The only cash cost is one egress of the processed store out of GCS.

## 1. Mirror the processed store once (~$6 egress, ~52 GB)

The store is `gs://chbmit-bench-2486a474/processed_chbmit_real/` (`eeg.zarr` +
`processed_index.csv`) plus `real_validation/{windows,splits,generators}`. Kaggle datasets are
capped at 200 GB, so the store fits in one dataset. From a machine with `gcloud` and `kaggle`
CLIs:

```bash
bash scripts/kaggle/mirror_store_to_kaggle.sh <kaggle-user>/chbmit-processed
```

The script pulls the store to local disk with a parallel `gcloud storage rsync`, tars the zarr
(~60 000 small chunk objects; Kaggle's uploader is slow on many small files), and creates a
private dataset. Do this ONCE; every notebook then attaches the dataset read-only.

CHB-MIT is open data (PhysioNet, ODC-BY), so mirroring the processed derivative is permitted;
keep the dataset private anyway so nobody mistakes it for the PhysioNet record.

## 2. One cell per notebook session

Kaggle sessions are 12 hours max (9 h with GPU on some tiers) and the multiseed driver is
resumable per `(fold, seed, detector)` block, so run one shard per session and commit the CSV
as a notebook output between sessions:

```bash
# in a Kaggle notebook cell (GPU on, dataset attached at /kaggle/input/chbmit-processed)
!git clone --depth 1 https://github.com/Oxidopamine/chbmit-synthetic-benchmark.git
%cd chbmit-synthetic-benchmark
!pip install -q -r requirements.txt
!tar -xf /kaggle/input/chbmit-processed/eeg.zarr.tar -C /kaggle/working/
!bash scripts/kaggle/run_phase3_cell.sh baselines eegnet 0   # <job> <detector> <fold>
```

`run_phase3_cell.sh` accepts three jobs, in the priority order of `PREREGISTRATION_PHASE3.md`:

| job | what it runs | cells | approx. GPU-hours (T4) |
|---|---|---|---|
| `floor` | the same-seed floor: `ungated` at r = 0.30 with a re-drawn synthetic sample, 9 TCN cells | 9 | 2 |
| `baselines` | seeded `real_only / class_weighted / classical_aug` for EEGNet + LCT, folds 0-2 × 3 seeds | 54 | 8-10 |
| `positive_control` | real held-out ictal as the oracle pool, all three detectors, folds 0-2 × 3 seeds | 27 blocks (~135 cells) | 25-35 |
| `logo` | leave-one-group-out, TCN, one seed, the registered Phase 3 design (needs one WGAN fit per fold) | 23 blocks (~200 cells) | 60-80 |

Every job writes its CSV under `results_chbmit_synthetic/real_validation/analysis_tierB/` with
its own tag, plus `run_environment<tag>.json`. Save that directory as the notebook's output,
download it, and commit it; the next session resumes from the committed CSV.

## 3. Things that differ from Vertex

* **Device.** T4/P100 have 16 GB. The detectors are small; batch size 64 fits with room to spare.
  Speed is ~3-4× slower than an A100 per cell, which the table above already assumes.
* **Determinism.** `--num-workers 0` is the default in every driver and must stay so; with workers
  PyTorch draws worker seeds from the global RNG and the trained model changes.
* **Torch version.** Kaggle's image pins its own torch; `run_environment<tag>.json` records it.
  Cells from Kaggle and Vertex must never be pooled in one paired comparison without first
  checking that the same arm reproduces across the two environments (README, "Known issues").
