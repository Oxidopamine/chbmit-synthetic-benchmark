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
GC="${GC:-gcloud}"

for DET in $DETECTORS; do
  TAG="${SUFFIX}_${DET}"
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
scheduling:
  strategy: SPOT
  restartJobOnWorkerRestart: true
YAML
  echo "=== submitting detector=${DET} tag=${TAG} ==="
  "$GC" ai custom-jobs create \
    --region="${REGION}" \
    --display-name="chbmit-phase1-${DET}" \
    --config="$CFG" 2>&1 | tail -4
  rm -f "$CFG"
done

echo
echo "list:   $GC ai custom-jobs list --region=${REGION} --format='table(displayName,state,createTime)'"
echo "output: gs://${BUCKET}/runs/"
