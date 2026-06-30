"""Seizure-event-level scarcity protocol (plan Section 14).

Scarcity is applied at the seizure-EVENT level, never by random ictal-window
sampling. For a given fraction we deterministically sample whole seizure events
(patient-aware: at least one event per training group where possible), then keep
only the ictal windows whose ``seizure_event_id`` is selected. Negatives are
sampled afterwards (Section 10) so the same subset feeds both the classifier and
the generator (Rule 4).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd


def select_seizure_events(
    events_df,
    fraction: float,
    seed: int,
    restrict_groups: Optional[Iterable[str]] = None,
) -> Set[int]:
    """Deterministically select a fraction of seizure events per group."""
    if fraction >= 1.0:
        sub = events_df
        if restrict_groups is not None:
            sub = sub[sub["group"].isin(set(restrict_groups))]
        return set(int(e) for e in sub["event_id"])

    restrict = set(restrict_groups) if restrict_groups is not None else None
    selected: List[int] = []
    for group, sub in events_df.groupby("group"):
        if restrict is not None and group not in restrict:
            continue
        ids = sorted(int(e) for e in sub["event_id"])
        n = len(ids)
        if n == 0:
            continue
        k = max(1, round(fraction * n))  # >=1 event per group where possible
        rng = random.Random((int(seed) * 1_000_003) ^ (hash(group) & 0x7FFFFFFF))
        chosen = sorted(rng.sample(ids, min(k, n)))
        selected.extend(chosen)
    return set(selected)


def apply_scarcity_to_windows(train_windows, selected_event_ids: Set[int]):
    """Keep all negatives; keep positives only from selected events."""
    pos = train_windows[
        (train_windows["label"] == 1)
        & (train_windows["seizure_event_id"].isin(selected_event_ids))
    ]
    neg = train_windows[train_windows["label"] == 0]
    return pd.concat([pos, neg]).reset_index(drop=True)


def save_selected_events(
    selected_event_ids: Set[int], events_df, out_dir: str | Path,
    fold: int, seed: int, fraction: float,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = events_df[events_df["event_id"].isin(selected_event_ids)].copy()
    path = out_dir / f"fold_{fold}_seed_{seed}_frac_{fraction}_selected_events.csv"
    sub.to_csv(path, index=False)
    return path


def scarcity_coverage_report(selected_event_ids: Set[int], events_df,
                             restrict_groups: Optional[Iterable[str]] = None) -> dict:
    restrict = set(restrict_groups) if restrict_groups is not None else None
    groups = sorted(set(events_df["group"]) if restrict is None else restrict)
    per_group = {}
    missing = []
    for g in groups:
        gids = set(int(e) for e in events_df[events_df["group"] == g]["event_id"])
        chosen = gids & selected_event_ids
        per_group[g] = {"available": len(gids), "selected": len(chosen)}
        if gids and not chosen:
            missing.append(g)
    return {"per_group": per_group, "groups_without_selected_event": missing,
            "n_selected": len(selected_event_ids)}
