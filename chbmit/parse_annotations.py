"""Parse CHB-MIT ``chbXX-summary.txt`` annotation files.

The CHB-MIT summaries are semi-structured text. This parser is deliberately
tolerant of the real-world variants found across the 24 patients:

* Seizure times appear either as ``Seizure Start Time: N seconds`` (single
  seizure) or ``Seizure 1 Start Time: N seconds`` (indexed, multiple seizures).
* The ``Channels in EDF Files:`` block can appear more than once in a single
  summary (the montage changes between recordings). We track the *current*
  channel list and attach it to every file that follows, until the next block.
* Some summaries (notably ``chb24``) omit ``File Start/End Time`` and may omit
  the channel block entirely. We record what is present and leave the rest to
  be filled from EDF headers downstream (see ``make_manifest``).

Pure stdlib so it is importable and testable before the heavy deps install.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "SeizureEvent",
    "FileRecord",
    "SummaryInfo",
    "parse_summary",
    "parse_summary_text",
    "find_summary_files",
]

# --- regexes ------------------------------------------------------------
_RE_RATE = re.compile(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", re.IGNORECASE)
_RE_CHAN_HEADER = re.compile(r"^Channels?\s+in\s+EDF\s+Files?", re.IGNORECASE)
_RE_CHAN_LINE = re.compile(r"^Channel\s+(\d+):\s*(.*?)\s*$", re.IGNORECASE)
_RE_FILE_NAME = re.compile(r"File\s+Name:\s*(\S+)", re.IGNORECASE)
_RE_FILE_START = re.compile(r"File\s+Start\s+Time:\s*([0-9:.]+)", re.IGNORECASE)
_RE_FILE_END = re.compile(r"File\s+End\s+Time:\s*([0-9:.]+)", re.IGNORECASE)
_RE_NSEIZ = re.compile(r"Number\s+of\s+Seizures\s+in\s+File:\s*(\d+)", re.IGNORECASE)
# Matches "Seizure Start Time: 2996 seconds" and "Seizure 1 Start Time: 327 seconds".
_RE_SEIZ_START = re.compile(
    r"Seizure\s*(\d*)\s*Start\s+Time:\s*([\d.]+)\s*seconds", re.IGNORECASE
)
_RE_SEIZ_END = re.compile(
    r"Seizure\s*(\d*)\s*End\s+Time:\s*([\d.]+)\s*seconds", re.IGNORECASE
)


@dataclass
class SeizureEvent:
    """One annotated seizure interval within a single EDF file (file-relative seconds)."""

    index: int  # 1-based ordinal within the file
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def as_dict(self) -> Dict[str, float]:
        return {"index": self.index, "start_sec": self.start_sec, "end_sec": self.end_sec}


@dataclass
class FileRecord:
    """Per-EDF record extracted from a summary file."""

    file_name: str
    start_clock: Optional[str] = None
    end_clock: Optional[str] = None
    n_seizures_declared: Optional[int] = None
    seizures: List[SeizureEvent] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)

    @property
    def n_seizures(self) -> int:
        return len(self.seizures)

    @property
    def seizure_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.seizures)


@dataclass
class SummaryInfo:
    """Parsed contents of one ``chbXX-summary.txt``."""

    patient: str
    sampling_rate: Optional[float]
    files: List[FileRecord] = field(default_factory=list)
    channel_blocks: List[List[str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_seizure_events(self) -> int:
        return sum(f.n_seizures for f in self.files)

    @property
    def files_with_seizures(self) -> List[FileRecord]:
        return [f for f in self.files if f.n_seizures > 0]


def _pair_seizures(
    starts: List[Tuple[str, float]],
    ends: List[Tuple[str, float]],
    declared: Optional[int],
    fname: str,
    warnings: List[str],
) -> List[SeizureEvent]:
    """Pair start/end times into ordered SeizureEvents, tolerating index gaps."""
    if len(starts) != len(ends):
        warnings.append(
            f"{fname}: {len(starts)} seizure starts but {len(ends)} ends; pairing by order"
        )
    events: List[SeizureEvent] = []
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        s_val, e_val = start[1], end[1]
        if e_val < s_val:
            warnings.append(f"{fname}: seizure {i} end {e_val} < start {s_val}; kept as-is")
        events.append(SeizureEvent(index=i, start_sec=s_val, end_sec=e_val))
    if declared is not None and declared != len(events):
        warnings.append(
            f"{fname}: declared {declared} seizures but parsed {len(events)}"
        )
    return events


def parse_summary_text(text: str, patient: str) -> SummaryInfo:
    """Parse the raw text of a summary file."""
    lines = text.splitlines()
    sampling_rate: Optional[float] = None
    channel_blocks: List[List[str]] = []
    current_channels: List[str] = []
    warnings: List[str] = []
    files: List[FileRecord] = []

    cur: Optional[FileRecord] = None
    cur_starts: List[Tuple[str, float]] = []
    cur_ends: List[Tuple[str, float]] = []

    in_channel_block = False
    pending_channels: List[str] = []

    def flush_file() -> None:
        nonlocal cur, cur_starts, cur_ends
        if cur is not None:
            cur.seizures = _pair_seizures(
                cur_starts, cur_ends, cur.n_seizures_declared, cur.file_name, warnings
            )
            files.append(cur)
        cur, cur_starts, cur_ends = None, [], []

    def close_channel_block() -> None:
        nonlocal in_channel_block, pending_channels, current_channels
        if in_channel_block:
            if pending_channels:
                current_channels = list(pending_channels)
                channel_blocks.append(list(pending_channels))
            in_channel_block = False
            pending_channels = []

    for raw_line in lines:
        line = raw_line.strip()

        m = _RE_RATE.search(line)
        if m:
            sampling_rate = float(m.group(1))
            continue

        if _RE_CHAN_HEADER.search(line):
            close_channel_block()
            in_channel_block = True
            pending_channels = []
            continue

        if in_channel_block:
            cm = _RE_CHAN_LINE.match(line)
            if cm:
                label = cm.group(2).strip()
                # CHB-MIT pads montages with empty/dummy slots ('-', '.', '').
                if label and label not in {"-", "."}:
                    pending_channels.append(label)
                continue
            # Stay in the block across blank lines and '****' decoration rules.
            if line == "" or set(line) <= {"*"}:
                continue
            # Any other token (e.g. 'File Name:') ends the block.
            close_channel_block()
            # fall through to handle this line (e.g. File Name)

        m = _RE_FILE_NAME.search(line)
        if m:
            flush_file()
            cur = FileRecord(file_name=m.group(1), channels=list(current_channels))
            continue

        if cur is not None:
            m = _RE_FILE_START.search(line)
            if m:
                cur.start_clock = m.group(1)
                continue
            m = _RE_FILE_END.search(line)
            if m:
                cur.end_clock = m.group(1)
                continue
            m = _RE_NSEIZ.search(line)
            if m:
                cur.n_seizures_declared = int(m.group(1))
                continue
            # Order matters: 'End Time' also contains 'Time', so test end first.
            m = _RE_SEIZ_END.search(line)
            if m:
                cur_ends.append((m.group(1), float(m.group(2))))
                continue
            m = _RE_SEIZ_START.search(line)
            if m:
                cur_starts.append((m.group(1), float(m.group(2))))
                continue

    close_channel_block()
    flush_file()

    if sampling_rate is None:
        warnings.append(f"{patient}: no 'Data Sampling Rate' found")
    if not channel_blocks:
        warnings.append(f"{patient}: no channel block found (will read channels from EDF)")

    return SummaryInfo(
        patient=patient,
        sampling_rate=sampling_rate,
        files=files,
        channel_blocks=channel_blocks,
        warnings=warnings,
    )


def parse_summary(summary_path: str | Path) -> SummaryInfo:
    """Parse a ``chbXX-summary.txt`` file from disk."""
    path = Path(summary_path)
    patient = path.stem.replace("-summary", "")
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_summary_text(text, patient=patient)


def find_summary_files(raw_root: str | Path) -> Dict[str, Path]:
    """Map patient id -> summary file path under a CHB-MIT raw root."""
    root = Path(raw_root)
    out: Dict[str, Path] = {}
    for p in sorted(root.rglob("*-summary.txt")):
        patient = p.stem.replace("-summary", "")
        out[patient] = p
    return out


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Parse a CHB-MIT summary file")
    ap.add_argument("summary", help="path to chbXX-summary.txt")
    args = ap.parse_args()
    info = parse_summary(args.summary)
    print(
        json.dumps(
            {
                "patient": info.patient,
                "sampling_rate": info.sampling_rate,
                "n_files": len(info.files),
                "n_seizure_events": info.n_seizure_events,
                "channel_blocks": info.channel_blocks,
                "files_with_seizures": [
                    {
                        "file": f.file_name,
                        "seizures": [s.as_dict() for s in f.seizures],
                    }
                    for f in info.files_with_seizures
                ],
                "warnings": info.warnings,
            },
            indent=2,
        )
    )
