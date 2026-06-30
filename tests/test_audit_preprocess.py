"""Tests for channel audit and EDF preprocessing on the synthetic fixture."""
import json

import numpy as np
import pytest

from chbmit.channel_audit import audit_channels, write_audit
from chbmit.channel_map import CANONICAL_MONTAGE_18
from chbmit.make_manifest import build_manifest
from chbmit.preprocess_edf import PreprocessConfig, preprocess_dataset, preprocess_signal
from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset


def _specs():
    def f(name, start, seiz=None, dur=24.0):
        return FileSpec(name=name, start_clock=start, duration_sec=dur, seizures=seiz or [])

    return [
        PatientSpec("chb01", [f("chb01_01.edf", "11:00:00"),
                               f("chb01_03.edf", "12:00:00", [(5.0, 9.0)])]),
        PatientSpec("chb02", [f("chb02_16.edf", "10:00:00", [(6.0, 12.0)])]),
    ]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    write_dataset(root, specs=_specs(), seed=2)
    return root


def test_audit_accepts_full_montage(dataset, tmp_path):
    df = build_manifest(dataset)
    res = audit_channels(df, max_dropped_fraction=0.10)
    assert res.accepted
    assert res.final_channel_set == list(CANONICAL_MONTAGE_18)
    assert res.dropped_seizure_event_fraction == 0.0
    write_audit(res, tmp_path)
    assert (tmp_path / "channel_coverage_report.csv").exists()
    assert (tmp_path / "final_channel_set.json").exists()


def test_audit_gate_fails_when_seizure_file_missing_channel():
    # Build a manifest-like frame where the only seizure file lacks a channel.
    import pandas as pd

    rows = [
        {"included": True, "patient": "chb01", "group": "chb01",
         "edf_path": "chb01/chb01_01.edf", "n_seizures": 0, "seizure_duration_sec": 0.0,
         "channels_hdr": json.dumps(list(CANONICAL_MONTAGE_18)),
         "channels_summary": json.dumps(list(CANONICAL_MONTAGE_18))},
        {"included": True, "patient": "chb01", "group": "chb01",
         "edf_path": "chb01/chb01_03.edf", "n_seizures": 1, "seizure_duration_sec": 30.0,
         "channels_hdr": json.dumps(list(CANONICAL_MONTAGE_18[:-1])),  # missing P8-O2
         "channels_summary": json.dumps(list(CANONICAL_MONTAGE_18[:-1]))},
    ]
    df = pd.DataFrame(rows)
    res = audit_channels(df, max_dropped_fraction=0.10)
    assert not res.accepted  # 100% of seizure events live in the dropped file
    assert "P8-O2" not in res.recommended_full_coverage_montage


def test_preprocess_signal_resamples_and_filters():
    sr_in = 512.0
    x = np.random.default_rng(0).normal(size=(18, sr_in.__int__() * 5))
    cfg = PreprocessConfig(target_sampling_rate=256)
    out = preprocess_signal(x, sr_in, cfg)
    assert out.dtype == np.float32
    assert out.shape[0] == 18
    assert abs(out.shape[1] - 256 * 5) <= 2  # resampled 512 -> 256 over 5 s


def test_preprocess_dataset_writes_zarr(dataset, tmp_path):
    import zarr

    df = build_manifest(dataset)
    res = audit_channels(df)
    stats = preprocess_dataset(df, dataset, tmp_path, res.final_channel_set)
    assert stats["n_written"] == 3
    root = zarr.open_group(str(tmp_path / "eeg.zarr"), mode="r")
    assert root.attrs["channels"] == list(CANONICAL_MONTAGE_18)
    assert root.attrs["sampling_rate"] == 256
    z = root["signals"]["chb01_03"]
    assert z.shape[0] == 18
    assert abs(z.shape[1] - 256 * 24) <= 2
    assert z.attrs["seizures"] == [[5.0, 9.0]]
    assert z.attrs["patient"] == "chb01"
