"""Siena Scalp EEG adapter for the cross-dataset verification tier (plan Section 18 Tier C).

Siena annotations live in ``Seizures-list-PNxx.txt`` files with clock times:

    Seizure n 1
    File name: PN00-1.edf
    Registration start time: 19.39.33
    Registration end time: 20.40.03
    Seizure start time: 19.58.36
    Seizure end time: 19.59.46

We convert clock times to file-relative seconds (handling midnight wrap) and
emit the SAME manifest schema as CHB-MIT, so the identical preprocess -> window
-> split -> train -> SzCORE-eval pipeline runs unchanged. Channels/sampling-rate
/duration are read from the EDF headers. Prefer the SzCORE-formatted (bipolar)
Siena release so the montage audit matches.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from chbmit.make_manifest import MANIFEST_COLUMNS, read_edf_header

_RE_SEIZ_NUM = re.compile(r"Seizure\s*n?\s*\d+", re.IGNORECASE)
_RE_FILE = re.compile(r"File\s*name:\s*(\S+)", re.IGNORECASE)
_RE_REG_START = re.compile(r"Registration\s*start\s*time:\s*([0-9.:]+)", re.IGNORECASE)
_RE_SEIZ_START = re.compile(r"Seizure\s*start\s*time:\s*([0-9.:]+)", re.IGNORECASE)
_RE_SEIZ_END = re.compile(r"Seizure\s*end\s*time:\s*([0-9.:]+)", re.IGNORECASE)


def _clock_to_sec(s: str) -> Optional[float]:
    parts = re.split(r"[.:]", s.strip())
    if len(parts) != 3:
        return None
    h, m, sec = (int(p) for p in parts)
    return h * 3600 + m * 60 + sec


def parse_siena_seizure_list(text: str) -> Dict[str, List[List[float]]]:
    """Parse a Seizures-list file -> ``{edf_name: [[start_sec, end_sec], ...]}``."""
    blocks = re.split(_RE_SEIZ_NUM, text)
    out: Dict[str, List[List[float]]] = {}
    for block in blocks:
        fm = _RE_FILE.search(block)
        rm = _RE_REG_START.search(block)
        sm = _RE_SEIZ_START.search(block)
        em = _RE_SEIZ_END.search(block)
        if not (fm and rm and sm and em):
            continue
        fname = fm.group(1)
        reg = _clock_to_sec(rm.group(1))
        s = _clock_to_sec(sm.group(1))
        e = _clock_to_sec(em.group(1))
        if None in (reg, s, e):
            continue
        # Convert to file-relative seconds, handling midnight wrap.
        s_rel = (s - reg) % 86400
        e_rel = (e - reg) % 86400
        if e_rel < s_rel:
            e_rel += 86400
        out.setdefault(fname, []).append([float(s_rel), float(e_rel)])
    return out


def find_siena_seizure_lists(raw_root: str | Path) -> Dict[str, Path]:
    root = Path(raw_root)
    out: Dict[str, Path] = {}
    for p in sorted(root.rglob("Seizures-list-*.txt")):
        subj = p.stem.replace("Seizures-list-", "")
        out[subj] = p
    return out


def build_siena_manifest(raw_root: str | Path, include_chb24: bool = True,
                         group_chb01_chb21: bool = True, read_headers: bool = True, **kwargs):
    """Build a CHB-MIT-schema manifest for a Siena dataset root."""
    import pandas as pd

    raw_root = Path(raw_root)
    lists = find_siena_seizure_lists(raw_root)
    rows: List[dict] = []
    for subject in sorted(lists):
        lpath = lists[subject]
        seizures_by_file = parse_siena_seizure_list(lpath.read_text(encoding="utf-8", errors="replace"))
        subj_dir = lpath.parent
        edfs = sorted(subj_dir.glob("*.edf")) + sorted(subj_dir.glob("*.EDF"))
        for edf in edfs:
            hdr = read_edf_header(edf) if read_headers else None
            seizures = seizures_by_file.get(edf.name, [])
            rows.append({
                "patient": subject,
                "group": subject,
                "edf_path": str(edf.relative_to(raw_root)).replace("\\", "/"),
                "edf_exists": True,
                "summary_present": True,
                "sampling_rate": (hdr.sampling_rate if hdr else None),
                "n_channels_hdr": (hdr.n_channels if hdr else None),
                "duration_sec": (hdr.duration_sec if hdr else None),
                "n_seizures": len(seizures),
                "seizure_duration_sec": float(sum(e - s for s, e in seizures)),
                "seizures": json.dumps(seizures),
                "channels_hdr": json.dumps(hdr.labels if hdr else []),
                "channels_summary": json.dumps(hdr.labels if hdr else []),
                "included": True,
                "exclude_reason": "",
            })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
