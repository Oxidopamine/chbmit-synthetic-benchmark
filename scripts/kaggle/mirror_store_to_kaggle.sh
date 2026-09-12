#!/bin/bash
# Mirror the processed CHB-MIT store from GCS into a private Kaggle dataset, once.
#
#   bash scripts/kaggle/mirror_store_to_kaggle.sh <kaggle-user>/chbmit-processed
#
# Costs one GCS egress (~52 GB, ~$6 at us-central1 internet rates). Needs `gcloud` (authenticated
# to the project) and `kaggle` (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json). Run from
# a machine with ~120 GB free disk: the zarr is copied and then tarred (Kaggle's uploader is
# very slow on ~60 000 small chunk objects, and the tar is ~the same size).
set -euo pipefail

DATASET="${1:?usage: mirror_store_to_kaggle.sh <kaggle-user>/<dataset-slug>}"
BUCKET="${BUCKET:-chbmit-bench-2486a474}"
WORK="${WORK:-$PWD/kaggle_mirror}"

mkdir -p "$WORK/upload"
echo "=== pulling processed store from gs://$BUCKET (parallel rsync) ==="
gcloud storage rsync -r "gs://$BUCKET/processed_chbmit_real" "$WORK/processed_chbmit_real"
echo "=== pulling windows / splits / generators ==="
for d in windows splits generators; do
  gcloud storage rsync -r "gs://$BUCKET/real_validation/$d" "$WORK/upload/real_validation/$d"
done

echo "=== packing eeg.zarr (one object instead of ~60k) ==="
tar -cf "$WORK/upload/eeg.zarr.tar" -C "$WORK/processed_chbmit_real" eeg.zarr
cp "$WORK/processed_chbmit_real/processed_index.csv" "$WORK/upload/"
sha256sum "$WORK/upload/eeg.zarr.tar" "$WORK/upload/processed_index.csv" > "$WORK/upload/SHA256SUMS"

cat > "$WORK/upload/dataset-metadata.json" <<JSON
{
  "title": "CHB-MIT processed store (chbmit-synthetic-benchmark)",
  "id": "$DATASET",
  "licenses": [{"name": "ODC-BY-1.0"}]
}
JSON

echo "=== creating PRIVATE Kaggle dataset $DATASET ==="
kaggle datasets create -p "$WORK/upload" --dir-mode tar --private
echo "done. Attach '$DATASET' to a notebook; eeg.zarr.tar unpacks with:"
echo "  tar -xf /kaggle/input/$(basename "$DATASET")/eeg.zarr.tar -C /kaggle/working/"
