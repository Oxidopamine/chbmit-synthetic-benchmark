#!/bin/bash
# Submit the W2 dose-prediction diagnostic to Vertex AI as a CPU-only custom job.
#
# WHY A JOB AND NOT A VM. A custom job terminates when the script exits, so it cannot be left
# running and quietly billing -- which a GCE VM can. This project is unfunded (see README
# "Budget-constrained scope"), so the vehicle matters as much as the machine type.
#
# WHY CPU. gen_w2_dose_prediction.py does no training: it loads the 9 cached generators, draws
# samples, and computes a sliced Wasserstein distance against real ictal windows. No GPU, and
# the GPU quota here is Spot-only anyway.
#
# WHY NO 52 GB STAGING. vertex_bootstrap.sh copies the whole processed store to local disk
# because the Phase 1 grid reads it every epoch for hours. This job touches only the real ictal
# windows (~2.5k per cell x 9 cells), so it reads the zarr straight off the /gcs mount and needs
# neither a 300 GB boot disk nor a long staging step.
#
# Estimated cost: n1-standard-8 on-demand is ~$0.38/h; the run is I/O-bound on ~22k window reads
# over FUSE, expect well under 2 h => under ~$1.
#
# Usage: scripts/vertex_w2_submit.sh [tag-suffix]        (default suffix: _w2)
set -euo pipefail

BUCKET="${BUCKET:-chbmit-bench-2486a474}"
REGION="${REGION:-us-central1}"
SUFFIX="${1:-_w2}"
# Plain CPU Python 3.10 base; torch is pip-installed below. Two images were tried and rejected
# by the API first: the vertex-ai pytorch-gpu image REQUIRES an accelerator ("GPU Accelerator is
# required for image ..."), and no current pytorch-cpu image exists in either registry -- the
# only one is pytorch-cpu.1-4 (PyTorch 1.4, deprecated). Both rejections were free: the API
# validates the image at submit time, so a wrong name costs nothing.
IMAGE="${IMAGE:-us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cpu.py310:latest}"
MACHINE="${MACHINE:-n1-standard-8}"
# Hard ceiling on spend. The store is read over FUSE rather than staged, and zarr over FUSE can
# be far slower than local disk, so bound the downside instead of trusting the estimate:
# 4 h x ~$0.38/h ~= $1.50 worst case, and the job cannot outlive it.
TIMEOUT="${TIMEOUT:-14400s}"
GC="${GC:-gcloud}"
DRY_RUN="${DRY_RUN:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ship the COMMITTED tree, so what ran is identifiable by commit. The 9 generator checkpoints and
# splits_seed42.json are tracked, so they travel with it; only windows/ has to be fetched.
TAR="$(mktemp -d)/repo_w2.tar.gz"
git -C "$ROOT" archive --format=tar.gz -o "$TAR" HEAD
echo "repo archive: $(du -h "$TAR" | cut -f1)  @ $(git -C "$ROOT" rev-parse --short HEAD)"

CFG="$(mktemp)"
cat > "$CFG" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE}
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-standard
      bootDiskSizeGb: 100
    containerSpec:
      imageUri: ${IMAGE}
      command:
        - /bin/bash
        - -c
        - |
          set -eo pipefail
          G=/gcs/${BUCKET}
          echo "=== deps ==="
          pip install -q --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | tail -1
          pip install -q "numpy>=1.24,<2.0" "zarr>=2.16,<3.0" numcodecs pandas scipy 2>&1 | tail -1
          python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)"
          W=/tmp/w2run; mkdir -p "\$W"; cd "\$W"
          echo "=== code (via the /gcs mount -- no gcloud CLI dependency) ==="
          cp "\$G/code/repo${SUFFIX}.tar.gz" repo.tar.gz
          tar -xzf repo.tar.gz && rm repo.tar.gz
          RES="\$W/results_chbmit_synthetic/real_validation"
          echo "=== window/event tables (not in git) ==="
          mkdir -p "\$RES/windows"
          cp "\$G/real_validation/windows/windows.csv" "\$RES/windows/"
          cp "\$G/real_validation/windows/events.csv"  "\$RES/windows/"
          echo "generators: \$(ls "\$RES/generators" | wc -l)  splits: \$(ls "\$RES/splits")"
          echo "=== run (store read directly off /gcs, not staged) ==="
          export CHBMIT_STORE="\$G/processed_chbmit_real/eeg.zarr"
          export CHBMIT_PROC="\$G/processed_chbmit_real/processed_index.csv"
          export CHBMIT_RESULTS="\$RES"
          set +e
          stdbuf -oL -eL python scripts/gen_w2_dose_prediction.py --device cpu 2>&1 | tee w2.log
          rc=\${PIPESTATUS[0]}
          set -e
          echo "=== sync results (runs even on failure, so the log is recoverable) ==="
          mkdir -p "\$G/runs"
          cp w2.log "\$G/runs/w2${SUFFIX}.log" || true
          cp "\$RES/analysis_tierB/w2_dose_prediction.csv" \
             "\$G/runs/w2_dose_prediction${SUFFIX}.csv" || true
          echo "exit=\$rc"; exit \$rc
      env:
        - name: BUCKET
          value: "${BUCKET}"
scheduling:
  timeout: ${TIMEOUT}
YAML

echo "=== plan ==="
echo "  machine : ${MACHINE} (no accelerator), image ${IMAGE##*/}"
echo "  code    : gs://${BUCKET}/code/repo${SUFFIX}.tar.gz"
echo "  store   : /gcs/${BUCKET}/processed_chbmit_real/eeg.zarr (mounted, not copied)"
echo "  output  : gs://${BUCKET}/runs/w2_dose_prediction${SUFFIX}.csv and w2${SUFFIX}.log"

if [ "$DRY_RUN" = "1" ]; then
  echo; echo "=== DRY RUN: config below, nothing uploaded, nothing submitted ==="
  cat "$CFG"; rm -f "$CFG"; exit 0
fi

echo "=== upload code ==="
"$GC" storage cp "$TAR" "gs://${BUCKET}/code/repo${SUFFIX}.tar.gz" -q

echo "=== submit ==="
"$GC" ai custom-jobs create \
  --region="${REGION}" \
  --display-name="chbmit-w2-dose${SUFFIX}" \
  --config="$CFG" 2>&1 | tail -5
rm -f "$CFG"

echo
echo "watch:  $GC ai custom-jobs list --region=${REGION} --format='table(displayName,state,createTime)'"
echo "result: gs://${BUCKET}/runs/w2_dose_prediction${SUFFIX}.csv"
