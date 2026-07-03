"""Preprocessing-only driver: raw EDFs -> processed zarr + windows + splits.

Runs manifest -> channel audit -> preprocess (zarr) -> window/event tables ->
balanced patient-group splits over the FULL downloaded CHB-MIT set, then STOPS
for review. No training cell is run here (that's the tier runners' job).

Designed to run unattended in the background. Safe to re-run: pass --reuse to
skip preprocessing if data/processed_chbmit_real/processed_index.csv already
exists (useful after a container reset, since installed packages are ephemeral
but the /workspace volume persists).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chbmit.make_manifest import build_manifest, summarize_manifest
from chbmit.channel_audit import audit_channels
from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig, summarize_windows
from experiments.prepare import prepare_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_chbmit"
PROC = ROOT / "data" / "processed_chbmit_real"
RES = ROOT / "results_chbmit_synthetic" / "real_validation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="skip preprocess if processed_index.csv already exists")
    args = ap.parse_args()

    t0 = time.time()
    print("== manifest ==", flush=True)
    man = build_manifest(RAW)
    print(json.dumps(summarize_manifest(man), indent=2)[:1200], flush=True)

    print("\n== channel audit ==", flush=True)
    audit = audit_channels(man[man["included"]], max_dropped_fraction=0.10)
    print(json.dumps({k: audit.summary()[k] for k in
                      ["accepted", "n_files_total", "n_files_kept",
                       "dropped_seizure_event_fraction", "final_channel_set"]},
                     indent=2), flush=True)

    print("\n== prepare (preprocess + windows + splits) ==", flush=True)
    n_groups = int(man[man["included"]]["group"].nunique())
    n_folds = max(3, min(5, n_groups))
    prepared = prepare_dataset(
        RAW, PROC, RES, n_folds=n_folds, seed=42,
        pre_cfg=PreprocessConfig(target_sampling_rate=256),
        win_cfg=WindowingConfig(sampling_rate=256),
        reuse_processed=args.reuse,
    )
    print("channels:", prepared.channels, flush=True)
    print("windows:", json.dumps(summarize_windows(prepared.windows_df), indent=2), flush=True)
    print("events:", len(prepared.events_df), flush=True)
    for s in prepared.splits:
        print(f"  fold {s.fold}: train={s.train_groups} val={s.val_groups} test={s.test_groups}",
              flush=True)

    print(f"\n== DONE preprocessing in {time.time() - t0:.0f}s; stopping for review ==",
          flush=True)
    print(f"store: {prepared.store}", flush=True)
    print(f"processed_index: {PROC / 'processed_index.csv'}", flush=True)


if __name__ == "__main__":
    main()
