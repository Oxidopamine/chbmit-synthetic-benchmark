"""Tests for leakage-safe balanced patient-group splits."""
import json

import pandas as pd

from chbmit.splits import (
    balanced_group_kfold,
    compute_group_stats,
    load_splits,
    make_splits,
    save_splits,
    split_to_file_ids,
)


def _index(n_groups=10, seed=0):
    """Fabricate a processed-index frame with varied seizure load per group."""
    import random

    rng = random.Random(seed)
    rows = []
    for g in range(n_groups):
        group = f"chb{g+1:02d}"
        n_files = rng.randint(1, 3)
        for fi in range(n_files):
            n_seiz = rng.choice([0, 0, 1, 1, 2, 3])
            seiz = []
            t = 50.0
            for _ in range(n_seiz):
                dur = rng.uniform(20, 60)
                seiz.append([t, t + dur])
                t += dur + 100
            rows.append({
                "file_id": f"{group}_{fi:02d}",
                "patient": group,
                "group": group,
                "duration_sec": rng.uniform(600, 3600),
                "n_seizures": n_seiz,
                "seizures": json.dumps(seiz),
            })
    return pd.DataFrame(rows)


def test_group_stats():
    df = _index(6, seed=1)
    stats = compute_group_stats(df)
    assert set(stats) == set(df["group"].unique())
    for g, s in stats.items():
        assert s.n_events >= 0
        assert s.total_duration_sec > 0


def test_kfold_partitions_groups_exactly_once():
    df = _index(10, seed=2)
    stats = compute_group_stats(df)
    assign = balanced_group_kfold(stats, n_folds=5, seed=42)
    all_assigned = [g for fold in assign.values() for g in fold]
    assert sorted(all_assigned) == sorted(stats)  # each group exactly once
    assert all(len(assign[f]) > 0 for f in assign)


def test_kfold_balances_seizure_events():
    df = _index(12, seed=3)
    stats = compute_group_stats(df)
    assign = balanced_group_kfold(stats, n_folds=4, seed=42)
    per_fold_events = [sum(stats[g].n_events for g in assign[f]) for f in assign]
    total = sum(per_fold_events)
    # Greedy LPT should keep folds within a reasonable band of the mean.
    mean = total / 4
    assert max(per_fold_events) - min(per_fold_events) <= max(3, 0.6 * mean)


def test_make_splits_leakage_safe_and_seizures_present():
    df = _index(12, seed=4)
    splits = make_splits(df, n_folds=4, val_fraction=0.25, seed=42)
    all_groups = set(df["group"].unique())
    seen_test = set()
    stats = compute_group_stats(df)
    for s in splits:
        tr, va, te = set(s.train_groups), set(s.val_groups), set(s.test_groups)
        # disjoint
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        # cover all groups
        assert tr | va | te == all_groups
        assert len(tr) > 0 and len(va) > 0 and len(te) > 0
        # seizures present in val and test
        assert any(stats[g].has_seizure for g in te)
        assert any(stats[g].has_seizure for g in va)
        seen_test |= te
    assert seen_test == all_groups  # every group tested exactly once


def test_determinism_and_roundtrip(tmp_path):
    df = _index(10, seed=5)
    s1 = make_splits(df, n_folds=5, seed=42)
    s2 = make_splits(df, n_folds=5, seed=42)
    assert [x.as_dict() for x in s1] == [x.as_dict() for x in s2]
    path = save_splits(s1, tmp_path, seed=42)
    loaded = load_splits(path)
    assert [x.as_dict() for x in loaded] == [x.as_dict() for x in s1]


def test_split_to_file_ids(tmp_path):
    df = _index(8, seed=6)
    splits = make_splits(df, n_folds=4, seed=42)
    ids = split_to_file_ids(df, splits[0])
    # No file id appears in two splits.
    assert set(ids["train"]).isdisjoint(ids["test"])
    assert set(ids["val"]).isdisjoint(ids["test"])
    total = len(ids["train"]) + len(ids["val"]) + len(ids["test"])
    assert total == len(df)
