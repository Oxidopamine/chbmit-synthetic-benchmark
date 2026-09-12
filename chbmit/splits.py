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
from typing import Dict, List, Optional, Sequence


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


def _rotate_validation(
    assign: Dict[int, List[str]],
    fold: int,
    weights: Optional[Dict[str, float]],
    val_fraction: float,
) -> List[str]:
    """Validation for ``fold`` = the next fold(s) in cyclic order, taken whole.

    Rotation is the fix for the narrow-panel defect of ``_carve_validation``: that routine sorts
    the training pool by seizure count and takes the richest groups first, so the SAME few
    seizure-rich patients serve as validation in almost every fold (7 of 23 groups across the
    three folds that were run; chb01 and chb13 in all three). Every gate decision in Phases 1-2
    was therefore made on a near-fixed panel. Taking whole folds in cyclic order instead gives
    every group a turn as validation, keeps validation disjoint from test by construction, and
    inherits the seizure balance of the fold partition.

    ``weights`` (typically total duration per group) decide how many folds to take: enough to
    reach ``val_fraction`` of the non-test material. Without weights every group counts 1, which
    is what the names-only Phase 3 split files use.
    """
    k = len(assign)
    non_test = [g for f in assign if f != fold for g in assign[f]]
    w = (lambda g: weights.get(g, 1.0)) if weights else (lambda g: 1.0)
    target = val_fraction * sum(w(g) for g in non_test)
    val: List[str] = []
    acc = 0.0
    for step in range(1, k):
        nxt = assign[(fold + step) % k]
        if acc >= target and val:
            break
        if len(val) + len(nxt) >= len(non_test):
            break  # always leave >= 1 group for training
        val.extend(nxt)
        acc += sum(w(g) for g in nxt)
    return val


def splits_from_assignment(
    assign: Dict[int, List[str]],
    all_groups: Sequence[str],
    val_fraction: float = 0.20,
    weights: Optional[Dict[str, float]] = None,
) -> List[Split]:
    """Build rotating-validation splits from an existing fold-to-test-groups assignment.

    Used to derive the Phase 3 split files from the COMMITTED test partition
    (``splits_seed42.json``), so the test folds stay identical to Phases 1-2 and only the
    validation panel changes.
    """
    splits: List[Split] = []
    for fold in sorted(assign):
        test_groups = sorted(assign[fold])
        val_groups = _rotate_validation(assign, fold, weights, val_fraction)
        train_groups = [g for g in all_groups if g not in test_groups and g not in val_groups]
        if not train_groups:
            raise ValueError(f"fold {fold}: no training groups left after rotation")
        splits.append(Split(fold=fold, train_groups=sorted(train_groups),
                            val_groups=sorted(val_groups), test_groups=test_groups))
    return splits


def make_splits(
    index_df,
    n_folds: int = 5,
    val_fraction: float = 0.20,
    seed: int = 42,
    require_seizure_in_val: bool = True,
    require_seizure_in_test: bool = True,
    val_strategy: str = "carve",
) -> List[Split]:
    """Balanced patient-group k-fold with either validation strategy.

    ``val_strategy="carve"`` reproduces the committed ``splits_seed42.json`` byte for byte and is
    kept as the default for that reason only. ``"rotate"`` is the Phase 3 strategy (see
    ``_rotate_validation``); ``PREREGISTRATION_PHASE3.md`` registers it.
    """
    if val_strategy not in ("carve", "rotate"):
        raise ValueError(f"val_strategy must be 'carve' or 'rotate', got {val_strategy!r}")
    stats = compute_group_stats(index_df)
    assign = balanced_group_kfold(stats, n_folds, seed)
    if val_strategy == "rotate":
        weights = {g: s.total_duration_sec for g, s in stats.items()}
        splits = splits_from_assignment(assign, list(stats), val_fraction, weights)
        for s in splits:
            if require_seizure_in_test and not any(stats[g].has_seizure for g in s.test_groups):
                raise ValueError(f"fold {s.fold} test set has no seizure group")
            if require_seizure_in_val and not any(stats[g].has_seizure for g in s.val_groups):
                raise ValueError(f"fold {s.fold} validation set has no seizure group")
        return splits
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


def make_logo_splits(
    groups: Sequence[str],
    val_fraction: float = 0.20,
    seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
) -> List[Split]:
    """Leave-one-group-out: every group is the test set once; validation rotates.

    Fold order is a seeded shuffle of the group names so the cyclic validation neighbours are
    not alphabetical (chb01..chb24 are recording-order, not clinically meaningful, but a fixed
    order would still make every validation panel a run of consecutive case numbers).
    """
    order = list(groups)
    random.Random(seed).shuffle(order)
    assign = {i: [g] for i, g in enumerate(order)}
    return splits_from_assignment(assign, sorted(groups), val_fraction, weights)


def validation_coverage(splits: Sequence[Split]) -> Dict[str, int]:
    """How many folds each group serves as validation in. The Phase 1-2 file scores 7 of 23
    groups > 0 across its three run folds; rotation scores every group at least once."""
    all_groups = sorted({g for s in splits for g in (s.train_groups + s.val_groups + s.test_groups)})
    return {g: sum(g in s.val_groups for s in splits) for g in all_groups}


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
