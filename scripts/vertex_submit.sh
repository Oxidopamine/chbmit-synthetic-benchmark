#!/bin/bash
# Submit the Phase 1 grid to Vertex AI as one custom job per detector.
#
# WHY PER-DETECTOR. The three detectors are independent: each trains its own real_only teacher
# and its own augmented arms, and nothing crosses between them. Splitting by detector turns a
# ~20-35 h serial run into three parallel ~7-12 h jobs. the project notes' "never run two pipelines
# concurrently" was about a shared 62 GB RAM cgroup on one pod; on Vertex each job is its own VM
# with its own RAM, so it does not apply. Each job writes its own tagged CSV; merge afterwards
# with scripts/vertex_collect.sh.
#
# WHY SPOT. This project's Compute Engine GPU quota (GPUS_ALL_REGIONS) is 0, but Vertex custom
# training has a SEPARATE quota pool with 8 preemptible A100s in us-central1. Spot is the only
# GPU capacity available here. vertex_bootstrap.sh makes preemption survivable.
#
# QUOTA CEILING: "Custom model training preemptible CPUs" is 42 in us-central1 and a2-highgpu-1g
# is 12 vCPU, so at most 3 concurrent jobs fit. That is exactly the number of detectors.
#
# Usage: scripts/vertex_submit.sh [tag-suffix]      (default suffix: _v2)
set -euo pipefail

BUCKET="${BUCKET:-chbmit-bench-2486a474}"
REGION="${REGION:-us-central1}"
SUFFIX="${1:-_v2}"
IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
MACHINE="${MACHINE:-a2-highgpu-1g}"
ACCEL="${ACCEL:-NVIDIA_TESLA_A100}"
DETECTORS="${DETECTORS:-eegnet lct tcn}"
EPOCHS="${EPOCHS:-80}"
FOLDS="${FOLDS:-0 1 2}"
SEEDS="${SEEDS:-42 123 2024}"
QS="${QS:-0.90 0.50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
# Phase 2 axes. Defaults reproduce Phase 1 exactly, so an unset RATIOS changes nothing.
RATIOS="${RATIOS:-1.0}"
RANDOM_QS="${RANDOM_QS:-$QS}"
GATE_REFERENCE="${GATE_REFERENCE:-real_ictal}"
# How to shard across the 3-concurrent-job quota ceiling. "detector" is the Phase 1 pattern.
# "seed" is for reduced grids that run ONE detector: a single job would be ~14 h, long enough
# that Spot preemption is likely, so split into three ~5 h jobs instead. The run is resumable
# either way, but shorter jobs lose less when preempted.
SPLIT_BY="${SPLIT_BY:-detector}"
NAME="${NAME:-chbmit-phase1}"
UPLOAD_CODE="${UPLOAD_CODE:-0}"   # 1 = ship the committed tree + bootstrap before submitting
DRY_RUN="${DRY_RUN:-0}"
GC="${GC:-gcloud}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$UPLOAD_CODE" = "1" ]; then
  # Ship the COMMITTED tree so the run is identifiable by commit, plus the bootstrap the
  # container fetches. Without this, a job silently runs whatever code was uploaded last.
  D="$(mktemp -d)"
  git -C "$ROOT" archive --format=tar.gz -o "$D/repo.tar.gz" HEAD
  # Take the bootstrap from the git BLOB, not the working tree. .gitattributes normalises text to
  # LF in the repo, but a Windows checkout materialises it CRLF -- and a CRLF shebang makes the
  # kernel look for "/bin/bash\r", which exits 127 "command not found" a couple of minutes into
  # the job. Three jobs died that way before this line existed. `git archive` above is already
  # blob-sourced and so was never affected; only this direct upload was.
  git -C "$ROOT" show HEAD:scripts/vertex_bootstrap.sh > "$D/vertex_bootstrap.sh"
  # Check via od: Git-Bash's grep strips CR before matching, so `grep $'\r'` silently never
  # fires on the very platform that produces the problem. od renders it as a literal \r first.
  if od -c "$D/vertex_bootstrap.sh" | head -1 | grep -q '\\r'; then
    echo "bootstrap still has CRLF after blob extraction -- aborting rather than burning jobs" >&2
    exit 3
  fi
  echo "=== uploading code @ $(git -C "$ROOT" rev-parse --short HEAD) ($(du -h "$D/repo.tar.gz" | cut -f1)) ==="
  "$GC" storage cp "$D/repo.tar.gz" "gs://${BUCKET}/code/repo.tar.gz" -q
  "$GC" storage cp "$D/vertex_bootstrap.sh" "gs://${BUCKET}/code/vertex_bootstrap.sh" -q
fi

case "$SPLIT_BY" in
  seed)     SHARDS="$SEEDS" ;;
  detector) SHARDS="$DETECTORS" ;;
  *) echo "SPLIT_BY must be 'detector' or 'seed'" >&2; exit 2 ;;
esac

for SHARD in $SHARDS; do
  if [ "$SPLIT_BY" = "seed" ]; then
    DET="$DETECTORS"; JOB_SEEDS="$SHARD"; TAG="${SUFFIX}_s${SHARD}"
  else
    DET="$SHARD";     JOB_SEEDS="$SEEDS"; TAG="${SUFFIX}_${SHARD}"
  fi
  CFG="$(mktemp)"
  cat > "$CFG" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE}
      acceleratorType: ${ACCEL}
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 300
    containerSpec:
      imageUri: ${IMAGE}
      command:
        - /bin/bash
        - -c
        - |
          set -e
          B=/gcs/${BUCKET}/code/vertex_bootstrap.sh
          if [ -f "\$B" ]; then cp "\$B" /tmp/b.sh
          elif command -v gcloud >/dev/null; then gcloud storage cp gs://${BUCKET}/code/vertex_bootstrap.sh /tmp/b.sh
          else gsutil cp gs://${BUCKET}/code/vertex_bootstrap.sh /tmp/b.sh; fi
          chmod +x /tmp/b.sh
          exec /tmp/b.sh
      env:
        - name: BUCKET
          value: "${BUCKET}"
        - name: DETECTOR
          value: "${DET}"
        - name: TAG
          value: "${TAG}"
        - name: EPOCHS
          value: "${EPOCHS}"
        - name: FOLDS
          value: "${FOLDS}"
        - name: SEEDS
          value: "${JOB_SEEDS}"
        - name: QS
          value: "${QS}"
        - name: RATIOS
          value: "${RATIOS}"
        - name: RANDOM_QS
          value: "${RANDOM_QS}"
        - name: GATE_REFERENCE
          value: "${GATE_REFERENCE}"
        - name: NUM_WORKERS
          value: "${NUM_WORKERS}"
scheduling:
  strategy: SPOT
  restartJobOnWorkerRestart: true
YAML
  echo "=== submitting detector=${DET} seeds=[${JOB_SEEDS}] ratios=[${RATIOS}] tag=${TAG} ==="
  if [ "$DRY_RUN" = "1" ]; then
    echo "--- DRY RUN: not submitted ---"
    grep -E "machineType:|name: (DETECTOR|SEEDS|FOLDS|QS|RATIOS|RANDOM_QS|TAG)$" -A 1 "$CFG" \
      | grep -v '^--$'
  else
    "$GC" ai custom-jobs create \
      --region="${REGION}" \
      --display-name="${NAME}-${TAG#_}" \
      --config="$CFG" 2>&1 | tail -4
  fi
  rm -f "$CFG"
done

echo
echo "list:   $GC ai custom-jobs list --region=${REGION} --format='table(displayName,state,createTime)'"
echo "output: gs://${BUCKET}/runs/"
