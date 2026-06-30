"""Preprocess CHB-MIT EDFs into per-file ``(C, T)`` float32 arrays in zarr.

For each included file (Section 23 step 3):

1. select the final montage channels in a fixed canonical order (identical
   channel order across all files);
2. band-pass (0.5-40 Hz) and optional notch filter;
3. resample to the target rate (256 Hz);
4. store as ``signals/<file_id>`` in a single zarr store, with per-file
   attributes (patient, group, seizure intervals in seconds, n_samples).

We deliberately store *continuous per-file timelines* (not windows) so event
metrics can be computed on preserved real test timelines (Rules 5, 7). Windowing
is metadata-only and happens later.

Filtering uses zero-phase ``sosfiltfilt`` so seizure onset timing is preserved
(important for latency metrics).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chbmit.channel_map import match_montage


@dataclass
class PreprocessConfig:
    target_sampling_rate: int = 256
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    notch_hz: Optional[float] = None
    dtype: str = "float32"
    filter_order: int = 4


def _design_bandpass(sr: float, low: float, high: float, order: int):
    from scipy.signal import butter

    nyq = sr / 2.0
    high = min(high, nyq * 0.999)
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def preprocess_signal(x, sr_in: float, cfg: PreprocessConfig):
    """Filter + resample a ``(C, T)`` array. Returns ``(C, T')`` float32."""
    import numpy as np
    from scipy.signal import iirnotch, resample_poly, sosfiltfilt, tf2sos

    x = np.asarray(x, dtype="float64")
    sos = _design_bandpass(sr_in, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.filter_order)
    x = sosfiltfilt(sos, x, axis=-1)
    if cfg.notch_hz:
        b, a = iirnotch(cfg.notch_hz, Q=30.0, fs=sr_in)
        x = sosfiltfilt(tf2sos(b, a), x, axis=-1)
    # Resample to target rate.
    sr_in_int = int(round(sr_in))
    if sr_in_int != cfg.target_sampling_rate:
        x = resample_poly(x, cfg.target_sampling_rate, sr_in_int, axis=-1)
    return np.ascontiguousarray(x, dtype=cfg.dtype)


def _read_selected_signals(edf_path: Path, source_indices: List[int]):
    """Read selected signal channels from an EDF at their native rate."""
    import numpy as np
    import pyedflib

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        srs = {reader.getSampleFrequency(i) for i in source_indices}
        if len(srs) != 1:
            raise ValueError(f"{edf_path.name}: selected channels have mixed rates {srs}")
        sr = float(srs.pop())
        sigs = [reader.readSignal(i) for i in source_indices]
        n = min(len(s) for s in sigs)
        arr = np.stack([s[:n] for s in sigs], axis=0)
        return arr, sr
    finally:
        reader.close()


def file_id_from_path(edf_path: str) -> str:
    """Stable id for a file, e.g. ``chb01/chb01_03.edf`` -> ``chb01_03``."""
    return Path(edf_path).stem


def preprocess_dataset(
    manifest_df,
    raw_root: str | Path,
    out_dir: str | Path,
    final_channel_set: Sequence[str],
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, object]:
    """Preprocess every included+covered file into a zarr store. Returns stats."""
    import numpy as np
    import zarr

    cfg = cfg or PreprocessConfig()
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "eeg.zarr"
    root = zarr.open_group(str(store_path), mode="a")
    signals = root.require_group("signals")
    root.attrs["channels"] = list(final_channel_set)
    root.attrs["sampling_rate"] = cfg.target_sampling_rate
    root.attrs["n_channels"] = len(final_channel_set)

    inc = manifest_df[manifest_df["included"]].copy()
    index_rows: List[dict] = []
    n_written = 0
    n_skipped_missing = 0

    for _, row in inc.iterrows():
        raw_labels = json.loads(row["channels_hdr"]) or json.loads(row["channels_summary"])
        selected, missing = match_montage(raw_labels, final_channel_set)
        if missing:  # drop_file policy
            n_skipped_missing += 1
            continue
        # Order source indices to match final_channel_set order exactly.
        by_ch = {c: i for c, i in selected}
        source_indices = [by_ch[c] for c in final_channel_set]

        edf_path = raw_root / row["edf_path"]
        arr, sr_in = _read_selected_signals(edf_path, source_indices)
        proc = preprocess_signal(arr, sr_in, cfg)

        fid = file_id_from_path(row["edf_path"])
        seizures = json.loads(row["seizures"])
        z = signals.require_dataset(
            fid, shape=proc.shape, chunks=(proc.shape[0], min(proc.shape[1], 256 * 60)),
            dtype=cfg.dtype, overwrite=True,
        )
        z[:] = proc
        z.attrs.update({
            "file_id": fid,
            "patient": row["patient"],
            "group": row["group"],
            "edf_path": row["edf_path"],
            "sampling_rate": cfg.target_sampling_rate,
            "n_channels": proc.shape[0],
            "n_samples": int(proc.shape[1]),
            "duration_sec": float(proc.shape[1] / cfg.target_sampling_rate),
            "seizures": seizures,
        })
        index_rows.append({
            "file_id": fid,
            "patient": row["patient"],
            "group": row["group"],
            "edf_path": row["edf_path"],
            "n_samples": int(proc.shape[1]),
            "duration_sec": float(proc.shape[1] / cfg.target_sampling_rate),
            "n_seizures": len(seizures),
            "seizures": json.dumps(seizures),
        })
        n_written += 1

    import pandas as pd

    index_df = pd.DataFrame(index_rows)
    index_path = out_dir / "processed_index.csv"
    index_df.to_csv(index_path, index=False)
    stats = {
        "store": str(store_path),
        "n_written": n_written,
        "n_skipped_missing_channels": n_skipped_missing,
        "channels": list(final_channel_set),
        "sampling_rate": cfg.target_sampling_rate,
    }
    (out_dir / "preprocess_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from chbmit.channel_audit import audit_channels
    from chbmit.make_manifest import build_manifest

    ap = argparse.ArgumentParser(description="Preprocess CHB-MIT EDFs to zarr")
    ap.add_argument("--raw-root", default="data/raw_chbmit")
    ap.add_argument("--out-dir", default="data/processed_chbmit")
    args = ap.parse_args()
    df = build_manifest(args.raw_root)
    audit = audit_channels(df)
    stats = preprocess_dataset(df, args.raw_root, args.out_dir, audit.final_channel_set)
    print(json.dumps(stats, indent=2))
