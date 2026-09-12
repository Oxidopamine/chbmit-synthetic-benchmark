"""Write the Phase 3 split files: rotating validation on the committed 5-fold partition, and
leave-one-group-out over the 23 CHB-MIT groups.

Why two files:

  splits_rot5_seed42.json   -- the SAME five test folds as the committed ``splits_seed42.json``
                               (so Phase 3 cells on folds 0-2 remain paired with Phases 1-2 on the
                               test side), with validation = the next fold in cyclic order instead
                               of the seizure-richest groups. Fixes the narrow-panel defect
                               (7 of 23 groups ever validation; chb01 and chb13 in every run fold).
  splits_logo23_seed42.json -- 23 folds, one group each as test, five rotating groups as
                               validation. The fold count is what binds fold-corrected inference
                               (Nadeau-Bengio rho = n_test/n_train drops from 0.32 to 0.056), so
                               this is the registered Phase 3 design (PREREGISTRATION_PHASE3.md).

Both are derived from group NAMES read out of the committed split file, because the processed
index (which carries per-group durations) lives in GCS, not in the repository. Passing
``--index processed_index.csv`` weights the rotation by duration and checks seizure presence;
without it every group counts 1, and the file records ``"weighted": false``. Every CHB-MIT case
has at least one seizure, so a names-only validation panel of whole groups always contains one.

    python scripts/make_phase3_splits.py
    python scripts/make_phase3_splits.py --index /gcs/<bucket>/processed_chbmit_real/processed_index.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chbmit.splits import (compute_group_stats, make_logo_splits, splits_from_assignment,
                           validation_coverage)

SPLITS_DIR = Path("results_chbmit_synthetic/real_validation/splits")
COMMITTED = SPLITS_DIR / "splits_seed42.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="processed_index.csv (optional; weights by duration)")
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(SPLITS_DIR))
    args = ap.parse_args()

    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assign = {int(f["fold"]): list(f["test_groups"]) for f in committed["folds"]}
    groups = sorted({g for f in committed["folds"]
                     for g in f["train_groups"] + f["val_groups"] + f["test_groups"]})
    assert len(groups) == 23, f"expected 23 groups in the committed split file, found {len(groups)}"

    weights, stats = None, None
    if args.index:
        import pandas as pd
        stats = compute_group_stats(pd.read_csv(args.index))
        missing = set(groups) - set(stats)
        assert not missing, f"index lacks groups {sorted(missing)}"
        weights = {g: stats[g].total_duration_sec for g in groups}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rot5 = splits_from_assignment(assign, groups, args.val_fraction, weights)
    logo = make_logo_splits(groups, args.val_fraction, args.seed, weights)

    for name, splits, note in (
        ("splits_rot5_seed42.json", rot5,
         "committed 5-fold test partition of splits_seed42.json; validation = next fold, cyclic"),
        ("splits_logo23_seed42.json", logo,
         "leave-one-group-out over 23 groups; validation = next five groups in a seeded cyclic order"),
    ):
        if stats is not None:
            for s in splits:
                assert any(stats[g].has_seizure for g in s.val_groups), f"{name} fold {s.fold}: no seizure in val"
                assert any(stats[g].has_seizure for g in s.test_groups), f"{name} fold {s.fold}: no seizure in test"
        cov = validation_coverage(splits)
        payload = {
            "seed": args.seed, "n_folds": len(splits), "val_strategy": "rotate",
            "val_fraction": args.val_fraction, "weighted": weights is not None,
            "derived_from": str(COMMITTED).replace("\\", "/"), "note": note,
            "validation_coverage": cov,
            "folds": [s.as_dict() for s in splits],
        }
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n_val_groups = sorted({len(s.val_groups) for s in splits})
        print(f"wrote {path}: {len(splits)} folds; val panel sizes {n_val_groups}; "
              f"groups ever in validation {sum(v > 0 for v in cov.values())}/{len(cov)}")

    legacy_cov = validation_coverage(
        [type(rot5[0])(**{k: f[k] for k in ('fold', 'train_groups', 'val_groups', 'test_groups')})
         for f in committed["folds"]])
    print(f"for comparison, the committed carve file has {sum(v > 0 for v in legacy_cov.values())}/23 "
          f"groups ever in validation across all five folds; over folds 0-2 only: "
          f"{len({g for f in committed['folds'][:3] for g in f['val_groups']})}/23")


if __name__ == "__main__":
    main()
