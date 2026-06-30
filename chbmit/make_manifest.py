"""Build the CHB-MIT manifest: one row per EDF file.

Sources of truth, reconciled (plan Sections 5, 6, 23):

* ``chbXX-summary.txt``  -> seizures, declared channels, clock times
* the EDF header         -> ground-truth sampling rate, channels, duration
* file existence on disk -> whether the row is usable

Patient grouping enforces Rule 2: ``chb01`` and ``chb21`` are the same subject
and share a group. ``chb24`` is handled explicitly via ``include_chb24`` and an
``exclude_reason`` column, never silently dropped.

The manifest is the spine the rest of the pipeline reads, so it is written as
both CSV (human-diffable) and Parquet/JSON where available.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chbmit.parse_annotations import find_summary_files, parse_summary

# chb21 is the same subject as chb01 (recorded ~1.5 years later).
SAME_SUBJECT: Dict[str, str] = {"chb21": "chb01"}

MANIFEST_COLUMNS = [
    "patient", "group", "edf_path", "edf_exists", "summary_present",
    "sampling_rate", "n_channels_hdr", "duration_sec",
    "n_seizures", "seizure_duration_sec", "seizures",
    "channels_hdr", "channels_summary",
    "included", "exclude_reason",
]


def assign_group(patient: str, group_chb01_chb21: bool = True) -> str:
    """Patient-group key used by all leakage-safe splits."""
    if group_chb01_chb21 and patient in SAME_SUBJECT:
        return SAME_SUBJECT[patient]
    return patient


@dataclass
class EdfHeader:
    sampling_rate: Optional[float]
    n_channels: int
    labels: List[str]
    duration_sec: Optional[float]


def read_edf_header(path: str | Path) -> Optional[EdfHeader]:
    """Read just the EDF header (no signal data) via pyedflib."""
    import pyedflib

    try:
        reader = pyedflib.EdfReader(str(path))
    except Exception:
        return None
    try:
        labels = list(reader.getSignalLabels())
        n = reader.signals_in_file
        # Per-signal sampling frequency; use the modal/first EEG rate.
        freqs = [reader.getSampleFrequency(i) for i in range(n)]
        sr = float(freqs[0]) if freqs else None
        duration = float(reader.file_duration)
        return EdfHeader(sampling_rate=sr, n_channels=n, labels=labels, duration_sec=duration)
    finally:
        reader.close()


def _decide_inclusion(
    patient: str, edf_exists: bool, summary_present: bool, include_chb24: bool
) -> Tuple[bool, str]:
    if patient == "chb24" and not include_chb24:
        return False, "chb24_excluded_by_config"
    if not edf_exists:
        return False, "edf_missing"
    if not summary_present:
        return False, "summary_missing"
    return True, ""


def build_manifest(
    raw_root: str | Path,
    include_chb24: bool = True,
    group_chb01_chb21: bool = True,
    read_headers: bool = True,
):
    """Return a pandas DataFrame manifest for a CHB-MIT raw root."""
    import pandas as pd

    raw_root = Path(raw_root)
    summaries = find_summary_files(raw_root)
    rows: List[dict] = []
    all_warnings: List[str] = []

    for patient in sorted(summaries):
        spath = summaries[patient]
        info = parse_summary(spath)
        all_warnings.extend(info.warnings)
        pdir = spath.parent
        for f in info.files:
            edf_path = pdir / f.file_name
            exists = edf_path.exists()
            hdr = read_edf_header(edf_path) if (exists and read_headers) else None
            included, reason = _decide_inclusion(patient, exists, True, include_chb24)
            rows.append({
                "patient": patient,
                "group": assign_group(patient, group_chb01_chb21),
                "edf_path": str(edf_path.relative_to(raw_root)).replace("\\", "/"),
                "edf_exists": exists,
                "summary_present": True,
                "sampling_rate": (hdr.sampling_rate if hdr else info.sampling_rate),
                "n_channels_hdr": (hdr.n_channels if hdr else None),
                "duration_sec": (hdr.duration_sec if hdr else None),
                "n_seizures": f.n_seizures,
                "seizure_duration_sec": f.seizure_duration_sec,
                "seizures": json.dumps([[s.start_sec, s.end_sec] for s in f.seizures]),
                "channels_hdr": json.dumps(hdr.labels if hdr else []),
                "channels_summary": json.dumps(f.channels),
                "included": included,
                "exclude_reason": reason,
            })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.attrs["warnings"] = all_warnings
    return df


def summarize_manifest(df) -> dict:
    """Compact stats for logging / sanity checks."""
    inc = df[df["included"]]
    return {
        "n_files": int(len(df)),
        "n_included": int(len(inc)),
        "n_patients": int(df["patient"].nunique()),
        "n_groups": int(df["group"].nunique()),
        "n_seizure_events": int(inc["n_seizures"].sum()),
        "total_seizure_duration_sec": float(inc["seizure_duration_sec"].sum()),
        "total_duration_sec": float(inc["duration_sec"].fillna(0).sum()),
        "patients": sorted(df["patient"].unique().tolist()),
        "groups": sorted(df["group"].unique().tolist()),
        "excluded": df[~df["included"]]["edf_path"].tolist(),
    }


def write_manifest(df, out_dir: str | Path) -> Path:
    """Write manifest CSV (+ summary JSON) to ``out_dir``; returns the CSV path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summarize_manifest(df), indent=2), encoding="utf-8"
    )
    warnings = df.attrs.get("warnings", [])
    (out_dir / "manifest_warnings.txt").write_text(
        "\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8"
    )
    return csv_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Build CHB-MIT manifest")
    ap.add_argument("--raw-root", default="data/raw_chbmit")
    ap.add_argument("--out-dir", default="results_chbmit_synthetic/manifests")
    ap.add_argument("--no-chb24", action="store_true")
    args = ap.parse_args()
    df = build_manifest(args.raw_root, include_chb24=not args.no_chb24)
    path = write_manifest(df, args.out_dir)
    print(f"wrote {path}")
    print(json.dumps(summarize_manifest(df), indent=2))
