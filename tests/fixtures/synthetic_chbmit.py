"""Generate a tiny synthetic CHB-MIT-style dataset for offline pipeline tests.

Produces, under a chosen root:

    chbXX/chbXX_NN.edf            (real EDF files, written via pyedflib)
    chbXX/chbXX-summary.txt       (annotation summary in CHB-MIT format)
    chbXX/RECORDS                 (per-patient record list)
    RECORDS                       (global record list)
    RECORDS-WITH-SEIZURES         (records containing seizures)

The signals are synthetic: background is band-limited noise; seizure intervals
add a higher-amplitude rhythmic component so detectors and generators have a
learnable ictal signature. This lets us validate manifest -> preprocess ->
splits -> windows -> metrics -> training end-to-end without the ~40GB real
corpus (which does not fit on this machine).

Summary-text generation (`summary_text`) is pure stdlib; dataset writing needs
numpy + pyedflib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chbmit.channel_map import CANONICAL_MONTAGE_18

__all__ = [
    "FileSpec",
    "PatientSpec",
    "summary_text",
    "default_spec",
    "write_dataset",
]


@dataclass
class FileSpec:
    name: str
    start_clock: str
    duration_sec: float
    seizures: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class PatientSpec:
    patient: str
    files: List[FileSpec]
    channels: Sequence[str] = tuple(CANONICAL_MONTAGE_18)
    sampling_rate: int = 256
    emit_clock_times: bool = True  # chb24-like summaries omit these


def _add_seconds(clock: str, seconds: float) -> str:
    h, m, s = (int(x) for x in clock.split(":"))
    total = h * 3600 + m * 60 + s + int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def summary_text(spec: PatientSpec) -> str:
    """Render a CHB-MIT-style ``chbXX-summary.txt`` body (pure stdlib)."""
    lines: List[str] = []
    lines.append(f"Data Sampling Rate: {spec.sampling_rate} Hz")
    lines.append("*************************")
    lines.append("")
    lines.append("Channels in EDF Files:")
    lines.append("**********************")
    for i, ch in enumerate(spec.channels, start=1):
        lines.append(f"Channel {i}: {ch}")
    lines.append("")
    for f in spec.files:
        lines.append(f"File Name: {f.name}")
        if spec.emit_clock_times:
            lines.append(f"File Start Time: {f.start_clock}")
            lines.append(f"File End Time: {_add_seconds(f.start_clock, f.duration_sec)}")
        lines.append(f"Number of Seizures in File: {len(f.seizures)}")
        if len(f.seizures) == 1:
            s, e = f.seizures[0]
            lines.append(f"Seizure Start Time: {int(s)} seconds")
            lines.append(f"Seizure End Time: {int(e)} seconds")
        else:
            for j, (s, e) in enumerate(f.seizures, start=1):
                lines.append(f"Seizure {j} Start Time: {int(s)} seconds")
                lines.append(f"Seizure {j} End Time: {int(e)} seconds")
        lines.append("")
    return "\n".join(lines) + "\n"


def default_spec(short: bool = True) -> List[PatientSpec]:
    """A compact multi-patient spec covering the cases the pipeline must handle.

    * chb01 & chb21  -> same subject (grouping test)
    * multiple seizures per file (indexed format)
    * a seizure-free patient's files and seizure-bearing files
    * chb24-like patient with no clock times in the summary
    """
    dur = 300.0 if short else 1200.0  # seconds per EDF
    montage = list(CANONICAL_MONTAGE_18)

    def f(name, start, seiz=None):
        return FileSpec(name=name, start_clock=start, duration_sec=dur, seizures=seiz or [])

    specs = [
        PatientSpec("chb01", [
            f("chb01_01.edf", "11:42:54"),
            f("chb01_03.edf", "13:43:04", [(100.0, 130.0)]),
            f("chb01_04.edf", "14:43:04", [(60.0, 90.0), (200.0, 235.0)]),
        ], channels=montage),
        PatientSpec("chb02", [
            f("chb02_01.edf", "09:00:00"),
            f("chb02_16.edf", "10:00:00", [(120.0, 160.0)]),
        ], channels=montage),
        PatientSpec("chb03", [
            f("chb03_01.edf", "08:00:00", [(80.0, 110.0)]),
            f("chb03_02.edf", "09:00:00", [(150.0, 195.0)]),
        ], channels=montage),
        PatientSpec("chb21", [  # same subject as chb01
            f("chb21_19.edf", "20:00:00", [(90.0, 120.0)]),
            f("chb21_20.edf", "21:00:00"),
        ], channels=montage),
        PatientSpec("chb24", [  # chb24-like: no clock times in summary
            f("chb24_01.edf", "00:00:00"),
            f("chb24_03.edf", "00:00:00", [(110.0, 145.0)]),
        ], channels=montage, emit_clock_times=False),
    ]
    return specs


def _synth_signal(n_channels, n_samples, sr, seizures, rng):
    import numpy as np

    t = np.arange(n_samples) / sr
    # Pink-ish background: sum of a few low-frequency sinusoids + noise, per channel.
    x = np.zeros((n_channels, n_samples), dtype="float64")
    for c in range(n_channels):
        for freq, amp in [(2.0, 12.0), (6.0, 8.0), (10.0, 5.0)]:
            phase = rng.uniform(0, 2 * np.pi)
            x[c] += amp * np.sin(2 * np.pi * freq * t + phase)
        x[c] += rng.normal(0, 6.0, size=n_samples)
    # Ictal segments: add a higher-frequency, higher-amplitude rhythm.
    for (s, e) in seizures:
        i0, i1 = int(s * sr), int(min(e, n_samples / sr) * sr)
        if i1 <= i0:
            continue
        seg_t = t[i0:i1]
        for c in range(n_channels):
            phase = rng.uniform(0, 2 * np.pi)
            x[c, i0:i1] += 45.0 * np.sin(2 * np.pi * 14.0 * seg_t + phase)
            x[c, i0:i1] += rng.normal(0, 10.0, size=i1 - i0)
    return x  # microvolt-like physical values


def write_dataset(
    root: str | Path,
    specs: Optional[List[PatientSpec]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Write a synthetic CHB-MIT dataset to ``root``. Returns a small summary dict."""
    import numpy as np
    import pyedflib

    if specs is None:
        specs = default_spec()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    global_records: List[str] = []
    seizure_records: List[str] = []

    for spec in specs:
        pdir = root / spec.patient
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{spec.patient}-summary.txt").write_text(
            summary_text(spec), encoding="utf-8"
        )
        per_patient_records: List[str] = []
        for fspec in spec.files:
            n_samples = int(fspec.duration_sec * spec.sampling_rate)
            sig = _synth_signal(
                len(spec.channels), n_samples, spec.sampling_rate, fspec.seizures, rng
            )
            edf_path = pdir / fspec.name
            writer = pyedflib.EdfWriter(
                str(edf_path), len(spec.channels), file_type=pyedflib.FILETYPE_EDFPLUS
            )
            headers = []
            for ch in spec.channels:
                headers.append({
                    "label": ch,
                    "dimension": "uV",
                    "sample_frequency": spec.sampling_rate,
                    "physical_max": 500.0,
                    "physical_min": -500.0,
                    "digital_max": 32767,
                    "digital_min": -32768,
                    "transducer": "",
                    "prefilter": "",
                })
            writer.setSignalHeaders(headers)
            writer.writeSamples([np.clip(sig[c], -500, 500) for c in range(len(spec.channels))])
            writer.close()

            rel = f"{spec.patient}/{fspec.name}"
            per_patient_records.append(rel)
            global_records.append(rel)
            if fspec.seizures:
                seizure_records.append(rel)
        (pdir / "RECORDS").write_text("\n".join(per_patient_records) + "\n", encoding="utf-8")

    (root / "RECORDS").write_text("\n".join(global_records) + "\n", encoding="utf-8")
    (root / "RECORDS-WITH-SEIZURES").write_text(
        "\n".join(seizure_records) + "\n", encoding="utf-8"
    )
    return {
        "root": str(root),
        "n_patients": len(specs),
        "n_files": len(global_records),
        "n_seizure_files": len(seizure_records),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Write a synthetic CHB-MIT dataset")
    ap.add_argument("root", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(write_dataset(args.root, seed=args.seed))
