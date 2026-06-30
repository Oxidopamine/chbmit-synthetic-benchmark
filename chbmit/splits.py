"""Leakage-safe balanced patient-group splits (plan Sections 5, 18, 22).

Splits are over *patient groups*, computed before any windowing (Rule 1), with
``chb01``/``chb21`` already merged into one group by the manifest (Rule 2).

``balanced_group_kfold`` assigns groups to folds with a greedy
longest-processing-time partition that balances seizure-event counts first
(rare class), then total duration. Each fold serves once as the test set;
validation is carved from that fold's training groups. We enforce that both the
validation and test sets contain at least one seizure-bearing group
(``require_seizure_in_val`` / ``require_seizure_in_test``).

All randomness is seeded and deterministic.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GroupStat:
    group: str
    n_events: int
    seizure_duration_sec: float
    total_duration_sec: float

    @property
    def has_seizure(self) -> bool:
        return self.n_events > 0


@dataclass
class Split:
    fold: int
    train_groups: List[str]
    val_groups: List[str]
    test_groups: List[str]

    def as_dict(self) -> dict:
        return asdict(self)


def compute_group_stats(index_df) -> Dict[str, GroupStat]:
    """Aggregate per-group seizure/duration stats from the processed index."""
    stats: Dict[str, GroupStat] = {}
    for group, sub in index_df.groupby("group"):
        stats[group] = GroupStat(
            group=group,
            n_events=int(sub["n_seizures"].sum()),
            seizure_duration_sec=float(
                sub["seizures"].apply(
                    lambda s: sum(e - st for st, e in json.loads(s))
                ).sum()
            ),
            total_duration_sec=float(sub["duration_sec"].sum()),
        )
    return stats


def balanced_group_kfold(
    stats: Dict[str, GroupStat], n_folds: int, seed: int = 42
) -> Dict[int, List[str]]:
    """Assign groups to ``n_folds`` test folds, balancing seizure events then duration."""
    groups = list(stats.keys())
    if n_folds > len(groups):
        raise ValueError(f"n_folds={n_folds} exceeds number of groups={len(groups)}")
    rng = random.Random(seed)
    rng.shuffle(groups)  # randomize tie order deterministically
    groups.sort(
        key=lambda g: (stats[g].n_events, stats[g].seizure_duration_sec, stats[g].total_duration_sec),
        reverse=True,
    )
    fold_events = [0] * n_folds
    fold_dur = [0.0] * n_folds
    assign: Dict[int, List[str]] = {f: [] for f in range(n_folds)}
    for g in groups:
        f = min(range(n_folds), key=lambda i: (fold_events[i], fold_dur[i]))
        assign[f].append(g)
        fold_events[f] += stats[g].n_events
        fold_dur[f] += stats[g].total_duration_sec
    _repair_seizure_coverage(assign, stats)
    return assign


def _repair_seizure_coverage(assign: Dict[int, List[str]], stats: Dict[str, GroupStat]) -> None:
    """Ensure every fold has >=1 seizure group by swapping where possible."""
    def fold_has_seizure(f):
        return any(stats[g].has_seizure for g in assign[f])

    for f in assign:
        if fold_has_seizure(f):
            continue
        donor = next(
            (d for d in assign if d != f and sum(stats[g].has_seizure for g in assign[d]) > 1),
            None,
        )
        if donor is None:
            continue
        sg = next(g for g in assign[donor] if stats[g].has_seizure)
        ng = next((g for g in assign[f] if not stats[g].has_seizure), None)
        assign[donor].remove(sg)
        assign[f].append(sg)
        if ng is not None:
            assign[f].remove(ng)
            assign[donor].append(ng)


def _carve_validation(
    train_groups: List[str],
    stats: Dict[str, GroupStat],
    val_fraction: float,
    seed: int,
    require_seizure: bool,
) -> tuple[List[str], List[str]]:
    if len(train_groups) < 2:
        raise ValueError("need >=2 training groups to carve a validation set")
    rng = random.Random(seed)
    ordered = sorted(train_groups, key=lambda g: stats[g].n_events, reverse=True)
    rng.shuffle(ordered)  # break ties among equal-seizure groups deterministically
    ordered.sort(key=lambda g: stats[g].n_events, reverse=True)

    total = sum(stats[g].total_duration_sec for g in train_groups)
    target = val_fraction * total
    val: List[str] = []
    acc = 0.0
    for g in ordered:
        if len(val) >= len(train_groups) - 1:
            break  # always leave >=1 group for training
        has_val_seizure = any(stats[v].has_seizure for v in val)
        if acc >= target and (not require_seizure or has_val_seizure):
            break
        val.append(g)
        acc += stats[g].total_duration_sec
    if require_seizure and not any(stats[v].has_seizure for v in val):
        seiz = next((g for g in ordered if stats[g].has_seizure and g not in val), None)
        if seiz is not None and len(val) < len(train_groups) - 1:
            val.append(seiz)
    train = [g for g in train_groups if g not in val]
    return train, val


def make_splits(
    index_df,
    n_folds: int = 5,
    val_fraction: float = 0.20,
    seed: int = 42,
    require_seizure_in_val: bool = True,
    require_seizure_in_test: bool = True,
) -> List[Split]:
    stats = compute_group_stats(index_df)
    assign = balanced_group_kfold(stats, n_folds, seed)
    splits: List[Split] = []
    for fold in range(n_folds):
        test_groups = sorted(assign[fold])
        if require_seizure_in_test and not any(stats[g].has_seizure for g in test_groups):
            raise ValueError(f"fold {fold} test set has no seizure group")
        train_pool = [g for g in stats if g not in test_groups]
        train_groups, val_groups = _carve_validation(
            train_pool, stats, val_fraction, seed + fold, require_seizure_in_val
        )
        splits.append(Split(
            fold=fold,
            train_groups=sorted(train_groups),
            val_groups=sorted(val_groups),
            test_groups=test_groups,
        ))
    return splits


def split_to_file_ids(index_df, split: Split) -> Dict[str, List[str]]:
    """Map a Split's groups to processed file_ids for train/val/test."""
    out = {}
    for name, groups in [("train", split.train_groups), ("val", split.val_groups),
                         ("test", split.test_groups)]:
        out[name] = index_df[index_df["group"].isin(groups)]["file_id"].tolist()
    return out


def save_splits(splits: List[Split], out_dir: str | Path, seed: int) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"splits_seed{seed}.json"
    path.write_text(
        json.dumps({"seed": seed, "n_folds": len(splits),
                    "folds": [s.as_dict() for s in splits]}, indent=2),
        encoding="utf-8",
    )
    return path


def load_splits(path: str | Path) -> List[Split]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Split(**f) for f in data["folds"]]
