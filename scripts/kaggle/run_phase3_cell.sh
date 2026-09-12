#!/bin/bash
# Run one Phase 3 shard inside a Kaggle/Colab session. See scripts/kaggle/README.md.
#
#   bash scripts/kaggle/run_phase3_cell.sh <job> <detector> <fold> [extra driver args]
#
#   job ∈ floor | baselines | positive_control | logo     (PREREGISTRATION_PHASE3.md Sec 2 order)
#
# Expects the processed store at $CHBMIT_DATA (default /kaggle/working/eeg.zarr next to the
# attached dataset's processed_index.csv) and the repo checked out in $PWD. Everything is
# resumable: re-running the same command after a session timeout continues from the last
# complete (fold, seed, detector) block in the CSV.
set -euo pipefail

JOB="${1:?job}"; DET="${2:?detector}"; FOLD="${3:?fold}"; shift 3 || true
EXTRA="$*"
DATA_ROOT="${CHBMIT_DATA_ROOT:-/kaggle/input/chbmit-processed}"
export CHBMIT_STORE="${CHBMIT_STORE:-/kaggle/working/eeg.zarr}"
export CHBMIT_PROC="${CHBMIT_PROC:-$DATA_ROOT/processed_index.csv}"
export CHBMIT_RESULTS="${CHBMIT_RESULTS:-results_chbmit_synthetic/real_validation}"
SPL="$CHBMIT_RESULTS/splits"

# The attached dataset carries windows/splits/generators; make them visible where the drivers look.
for d in windows splits generators; do
  [ -d "$CHBMIT_RESULTS/$d" ] || { mkdir -p "$CHBMIT_RESULTS"; ln -s "$DATA_ROOT/real_validation/$d" "$CHBMIT_RESULTS/$d"; }
done

python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

case "$JOB" in
  floor)
    # Same seed, same init, DIFFERENT synthetic draw: the floor that applies to a paired delta.
    # Implemented as the ungated arm at r = 0.30 with the provider seed offset (see
    # run_multiseed_downstream.py --synth-seed-offset). TCN only; 9 cells.
    python scripts/run_multiseed_downstream.py --folds "$FOLD" --seeds 42 123 2024 --detectors tcn \
      --qs --random-qs --ratios 0.30 --synth-seed-offset 1000 --tag "_p4floor" --no-backup $EXTRA ;;
  baselines)
    # Empty --ratios => the three seeded baselines only, no generator loaded.
    python scripts/run_multiseed_downstream.py --folds "$FOLD" --seeds 42 123 2024 --detectors "$DET" \
      --qs --random-qs --ratios --tag "_p4base_${DET}" --no-backup $EXTRA ;;
  positive_control)
    python scripts/positive_control_gate.py --data-root "$DATA_ROOT" --device cuda \
      --folds "$FOLD" --seeds 42 123 2024 --detectors "$DET" --qs 0.90 0.50 \
      --with-simple-baselines --tag "_p4_${DET}" $EXTRA ;;
  logo)
    python scripts/run_multiseed_downstream.py --folds "$FOLD" --seeds 42 --detectors "$DET" \
      --qs 0.95 --random-qs 0.95 --ratios 0.30 --gate-reference pool \
      --splits "$SPL/splits_logo23_seed42.json" --tag "_p4logo_${DET}" --no-backup $EXTRA ;;
  *) echo "unknown job $JOB (floor | baselines | positive_control | logo)" >&2; exit 2 ;;
esac
echo "=== done: $JOB $DET fold $FOLD. Save $CHBMIT_RESULTS/analysis_tierB/ as the notebook output. ==="
