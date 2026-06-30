"""Lazy window metadata (plan Sections 11, 23 step 5).

We never materialize windows here. For each preprocessed file we enumerate
windows (4 s / 1024 samples, stride 2 s / 512) and record metadata only:

    file_id, group, patient, window_index, start_sample, end_sample,
    center_sample, center_time_sec, label, seizure_overlap_fraction,
    is_boundary_window, seconds_from_nearest_seizure, seizure_event_id, excluded

Labels use the ``center_in_seizure`` rule. ``seizure_event_id`` is globally
unique so scarcity (Section 14) can sample at the seizure-event level. A
separate events table lists every seizure with its global id, used by the
scarcity sampler and event-level evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class WindowingConfig:
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    sampling_rate: int = 256
    label_rule: str = "center_in_seizure"
    exclude_boundary_seconds: float = 0.0

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sampling_rate))

    @property
    def stride_samples(self) -> int:
        return int(round(self.stride_seconds * self.sampling_rate))


def _intervals_to_samples(seizures, sr) -> List[Tuple[int, int]]:
    return [(int(round(s * sr)), int(round(e * sr))) for s, e in seizures]


def _overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def build_window_table(index_df, win: Optional[WindowingConfig] = None):
    """Return ``(windows_df, events_df)`` for the processed index.

    Reads only seizure intervals and ``n_samples`` (no signal data).
    """
    import numpy as np
    import pandas as pd

    win = win or WindowingConfig()
    sr = win.sampling_rate
    wlen = win.window_samples
    stride = win.stride_samples
    excl = int(round(win.exclude_boundary_seconds * sr))

    win_rows: List[dict] = []
    event_rows: List[dict] = []
    event_counter = 0

    for _, row in index_df.iterrows():
        n_samples = int(row["n_samples"])
        seizures = json.loads(row["seizures"])
        seiz_samp = _intervals_to_samples(seizures, sr)

        # Register events with global ids.
        local_event_ids: List[int] = []
        for (s0, s1) in seiz_samp:
            event_rows.append({
                "event_id": event_counter,
                "file_id": row["file_id"],
                "group": row["group"],
                "patient": row["patient"],
                "start_sample": s0,
                "end_sample": s1,
                "start_sec": s0 / sr,
                "end_sec": s1 / sr,
                "duration_sec": (s1 - s0) / sr,
            })
            local_event_ids.append(event_counter)
            event_counter += 1

        if n_samples < wlen:
            continue
        starts = range(0, n_samples - wlen + 1, stride)
        for wi, start in enumerate(starts):
            end = start + wlen
            center = start + wlen // 2
            center_t = center / sr

            label = 0
            event_id = -1
            for ev_local, (s0, s1) in enumerate(seiz_samp):
                if s0 <= center < s1:
                    label = 1
                    event_id = local_event_ids[ev_local]
                    break

            overlap_samples = sum(_overlap(start, end, s0, s1) for s0, s1 in seiz_samp)
            overlap_frac = overlap_samples / wlen

            if seiz_samp:
                # distance (seconds) from window center to nearest seizure interval
                dists = []
                for s0, s1 in seiz_samp:
                    if s0 <= center < s1:
                        dists.append(0.0)
                    else:
                        dists.append(min(abs(center - s0), abs(center - s1)) / sr)
                nearest = min(dists)
            else:
                nearest = float("inf")

            is_boundary = 0.0 < overlap_frac < 1.0
            excluded = False
            if excl > 0 and seiz_samp:
                for s0, s1 in seiz_samp:
                    if _overlap(start - excl, end + excl, s0, s1) > 0 and not (s0 <= center < s1):
                        excluded = True
                        break

            win_rows.append({
                "file_id": row["file_id"],
                "group": row["group"],
                "patient": row["patient"],
                "window_index": wi,
                "start_sample": start,
                "end_sample": end,
                "center_sample": center,
                "center_time_sec": center_t,
                "label": label,
                "seizure_overlap_fraction": float(overlap_frac),
                "is_boundary_window": bool(is_boundary),
                "seconds_from_nearest_seizure": float(nearest),
                "seizure_event_id": event_id,
                "excluded": bool(excluded),
            })

    windows_df = pd.DataFrame(win_rows)
    events_df = pd.DataFrame(event_rows)
    return windows_df, events_df


def summarize_windows(windows_df) -> dict:
    n = int(len(windows_df))
    n_pos = int((windows_df["label"] == 1).sum())
    return {
        "n_windows": n,
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "positive_fraction": (n_pos / n) if n else 0.0,
        "n_boundary": int(windows_df["is_boundary_window"].sum()),
        "n_excluded": int(windows_df["excluded"].sum()),
    }


def save_window_tables(windows_df, events_df, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = out_dir / "windows.parquet"
    try:
        windows_df.to_parquet(wpath)
        events_df.to_parquet(out_dir / "events.parquet")
    except Exception:  # pyarrow/fastparquet may be absent; fall back to CSV
        wpath = out_dir / "windows.csv"
        windows_df.to_csv(wpath, index=False)
        events_df.to_csv(out_dir / "events.csv", index=False)
    import json as _json

    (out_dir / "windows_summary.json").write_text(
        _json.dumps(summarize_windows(windows_df), indent=2), encoding="utf-8"
    )
    return wpath
