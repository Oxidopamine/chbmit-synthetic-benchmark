#!/bin/bash
# Pull the per-detector Phase 1 CSVs back from GCS and merge them into one file the analysis
# can read. Each Vertex job wrote gs://$BUCKET/runs/downstream_gated<suffix>_<det>.csv.
set -euo pipefail
BUCKET="${BUCKET:-chbmit-bench-2486a474}"
SUFFIX="${1:-_v2}"
OUT="results_chbmit_synthetic/real_validation/analysis_tierB"
GC="${GC:-gcloud}"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
"$GC" storage cp "gs://$BUCKET/runs/downstream_gated${SUFFIX}_*.csv" "$TMP/" -q
ls -la "$TMP"
python3 - "$TMP" "$OUT/downstream_gated${SUFFIX}.csv" <<'PY'
import glob, sys
import pandas as pd
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{src}/*.csv"))
if not files:
    raise SystemExit("no per-detector CSVs found")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out, index=False)
print(f"merged {len(files)} files -> {out}: {len(df)} rows")
print(df.groupby(['detector']).size().to_string())
PY
echo "now run: python3 scripts/analyze_multiseed.py --tag ${SUFFIX}"
