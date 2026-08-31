#!/bin/bash
# Vertex AI custom-job entrypoint for the Phase 1 grid.
#
# Runs inside the prebuilt Vertex container pytorch-gpu.2-4.py310. Everything it needs is staged
# in GCS by scripts/vertex_submit.sh, so there is no image to build or push.
#
# Spot-safety is the point of this script. Vertex Spot workers are reclaimed without warning and
# the Phase 1 grid runs far longer than a Spot lease. The driver is already resumable -- it skips
# (fold,seed,detector) blocks already complete in its output CSV -- but it resumes from a LOCAL
# file, and a preempted container's local disk is gone. So:
#   * on start, pull any existing CSV for this tag back from GCS before the driver reads it;
#   * while running, push the CSV to GCS every SYNC_SECS, bounding loss to one block of work;
#   * on exit (including preemption's SIGTERM), push once more.
# With restartJobOnWorkerRestart set, a reclaimed worker re-runs this script and continues from
# the next incomplete block.
#
# Required env: BUCKET, DETECTOR, TAG. Optional: FOLDS, SEEDS, QS, RATIOS, RANDOM_QS, EPOCHS,
# SYNC_SECS.
set -uo pipefail

BUCKET="${BUCKET:?BUCKET is required}"
DETECTOR="${DETECTOR:?DETECTOR is required}"
TAG="${TAG:?TAG is required}"
FOLDS="${FOLDS:-0 1 2}"
SEEDS="${SEEDS:-42 123 2024}"
QS="${QS:-0.90 0.50}"
# Phase 2 axes. Defaults reproduce the Phase 1 grid, so an unset RATIOS/RANDOM_QS changes
# nothing. RATIOS is the injection ladder (see DECISION_GATE_1 CORRECTION 2 for why r=1.0 alone
# was not enough); RANDOM_QS is the matched-volume control, one per gated arm.
RATIOS="${RATIOS:-1.0}"
RANDOM_QS="${RANDOM_QS:-$QS}"
# real_ictal (default, as Phase 1 ran) or pool. Under real_ictal the gate admits 6-23 windows
# whatever the target, so only "pool" gives the gated arm a real dose.
GATE_REFERENCE="${GATE_REFERENCE:-real_ictal}"
EPOCHS="${EPOCHS:-80}"
SYNC_SECS="${SYNC_SECS:-60}"
NUM_WORKERS="${NUM_WORKERS:-0}"

WORK=/workspace
RES="$WORK/results_chbmit_synthetic/real_validation"
CSV="$RES/analysis_tierB/downstream_gated${TAG}.csv"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Vertex mounts the project's buckets at /gcs via Cloud Storage FUSE. Prefer it -- it needs no
# CLI in the image. Fall back to whichever CLI is present if the mount is missing.
MNT="/gcs/$BUCKET"
if [ -d "$MNT" ]; then
  MODE=fuse
elif command -v gcloud >/dev/null 2>&1; then
  MODE=gcloud
elif command -v gsutil >/dev/null 2>&1; then
  MODE=gsutil
else
  echo "FATAL: no /gcs mount and no gcloud/gsutil in image"; exit 1
fi
log "GCS access mode: $MODE"

fetch_dir() {  # fetch_dir <bucket-relative-dir> <local-dir>
  mkdir -p "$2"
  case $MODE in
    fuse)   cp -r "$MNT/$1/." "$2/" ;;
    gcloud) gcloud storage rsync -r "gs://$BUCKET/$1" "$2" -q ;;
    gsutil) gsutil -q -m rsync -r "gs://$BUCKET/$1" "$2" ;;
  esac
}
fetch_bulk() {  # fetch_bulk <bucket-relative-dir> <local-dir> -- for the zarr store ONLY.
  # The store is ~52 GB spread over ~60,000 chunk objects. A FUSE "cp -r" walks those one at a
  # time and each file costs a round trip, so it takes hours. gcloud/gsutil copy in parallel and
  # do it in minutes. Prefer a CLI here even when the FUSE mount exists; fall back to FUSE only
  # if neither CLI is in the image.
  mkdir -p "$2"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync -r "gs://$BUCKET/$1" "$2"
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil -q -m rsync -r "gs://$BUCKET/$1" "$2"
  else
    log "WARNING: no CLI available, falling back to single-threaded FUSE copy - expect hours"
    cp -r "$MNT/$1/." "$2/"
  fi
}
fetch_file() {  # fetch_file <bucket-relative-file> <local-file>  (non-fatal)
  case $MODE in
    fuse)   [ -f "$MNT/$1" ] && cp "$MNT/$1" "$2" ;;
    gcloud) gcloud storage cp "gs://$BUCKET/$1" "$2" -q 2>/dev/null ;;
    gsutil) gsutil -q cp "gs://$BUCKET/$1" "$2" 2>/dev/null ;;
  esac
}
push_file() {  # push_file <local-file> <bucket-relative-file>
  [ -f "$1" ] || return 0
  case $MODE in
    fuse)   mkdir -p "$(dirname "$MNT/$2")" && cp "$1" "$MNT/$2" ;;
    gcloud) gcloud storage cp "$1" "gs://$BUCKET/$2" -q 2>/dev/null ;;
    gsutil) gsutil -q cp "$1" "gs://$BUCKET/$2" 2>/dev/null ;;
  esac
}

log "=== stage code ==="
mkdir -p "$WORK" && cd "$WORK"
fetch_file "code/repo.tar.gz" "$WORK/repo.tar.gz"
tar -xzf repo.tar.gz && rm repo.tar.gz

log "=== stage results tree (windows/events/splits + cached WGAN checkpoints) ==="
fetch_dir "real_validation" "$RES"

log "=== stage processed store to LOCAL disk (~52 GB / ~60k objects, parallel copy) ==="
fetch_bulk "processed_chbmit_real" "$WORK/data/processed_chbmit_real"
log "store staged: $(du -sh "$WORK/data/processed_chbmit_real" | cut -f1), $(find "$WORK/data/processed_chbmit_real" -type f | wc -l) files"

log "=== resume: pull any partial CSV for this tag ==="
mkdir -p "$RES/analysis_tierB"
fetch_file "runs/downstream_gated${TAG}.csv" "$CSV"
if [ -f "$CSV" ]; then log "resumed: $(wc -l < "$CSV") lines"; else log "no prior CSV - starting fresh"; fi

log "=== deps ==="
pip install -q "numpy>=1.24,<2.0" "zarr>=2.16,<3.0" numcodecs timescoring 2>&1 | tail -2
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# Persist it too -- printing to a log that is not committed is why the torch build behind the
# existing grids is unrecoverable (audit 2026-08-31).
python - <<'PYENV' > "${GCS_RESULTS:-.}/run_environment.json" 2>/dev/null || true
import json, platform, sys
try:
    import torch, numpy
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as e:                     # never fail the job over bookkeeping
    env = {"error": str(e)}
env.update({"python": sys.version, "platform": platform.platform()})
print(json.dumps(env, indent=2))
PYENV

sync_loop() {
  while true; do
    sleep "$SYNC_SECS"
    push_file "$CSV" "runs/downstream_gated${TAG}.csv"
    push_file "$WORK/run.log" "runs/log${TAG}.txt"
  done
}
sync_loop & SYNC_PID=$!
finish() {
  log "=== final sync ==="
  kill "$SYNC_PID" 2>/dev/null
  push_file "$CSV" "runs/downstream_gated${TAG}.csv"
  push_file "$WORK/run.log" "runs/log${TAG}.txt"
  log "synced runs/downstream_gated${TAG}.csv"
}
trap finish EXIT TERM INT

log "=== grid: detector=$DETECTOR folds=[$FOLDS] seeds=[$SEEDS] qs=[$QS] ratios=[$RATIOS] random_qs=[$RANDOM_QS] epochs=$EPOCHS ==="
export CHBMIT_STORE="$WORK/data/processed_chbmit_real/eeg.zarr"
export CHBMIT_PROC="$WORK/data/processed_chbmit_real/processed_index.csv"
export CHBMIT_RESULTS="$RES"
cd "$WORK"

# --no-backup: the driver's git commit/push backup is meaningless here (no remote credentials);
# the GCS sync above replaces it.
stdbuf -oL -eL python scripts/run_multiseed_downstream.py \
  --folds $FOLDS --seeds $SEEDS --detectors "$DETECTOR" \
  --qs $QS --ratios $RATIOS --random-qs $RANDOM_QS \
  --gate-reference "$GATE_REFERENCE" \
  --epochs "$EPOCHS" --num-workers "$NUM_WORKERS" --tag "$TAG" --no-backup 2>&1 | tee "$WORK/run.log"
rc=${PIPESTATUS[0]}
log "driver exit=$rc"
exit $rc
