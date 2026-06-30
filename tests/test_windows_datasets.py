"""Integration test: manifest -> audit -> preprocess -> windows -> dataset."""
import numpy as np
import pytest

from chbmit.channel_audit import audit_channels
from chbmit.datasets import (
    WindowDataset,
    materialize_windows,
    negative_sample,
)
from chbmit.make_manifest import build_manifest
from chbmit.preprocess_edf import preprocess_dataset
from chbmit.window_metadata import WindowingConfig, build_window_table, summarize_windows
from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    # One 60 s file with a single clear seizure 20-40 s.
    specs = [PatientSpec("chb01", [
        FileSpec("chb01_03.edf", "12:00:00", duration_sec=60.0, seizures=[(20.0, 40.0)]),
        FileSpec("chb01_01.edf", "11:00:00", duration_sec=60.0, seizures=[]),
    ])]
    write_dataset(raw, specs=specs, seed=7)
    df = build_manifest(raw)
    res = audit_channels(df)
    proc = tmp_path_factory.mktemp("proc")
    preprocess_dataset(df, raw, proc, res.final_channel_set)
    import pandas as pd

    index_df = pd.read_csv(proc / "processed_index.csv")
    store = str(proc / "eeg.zarr")
    return index_df, store


def test_window_labels_match_known_seizure(prepared):
    index_df, store = prepared
    win = WindowingConfig(window_seconds=4, stride_seconds=2, sampling_rate=256)
    windows, events = build_window_table(index_df, win)
    seiz_file = windows[windows["file_id"] == "chb01_03"]
    # centers at (start+512)/256 s; positives where center in [20,40): 10 windows.
    assert int((seiz_file["label"] == 1).sum()) == 10
    # The seizure-free file has zero positives.
    assert int((windows[windows["file_id"] == "chb01_01"]["label"] == 1).sum()) == 0
    # Exactly one global event registered.
    assert len(events) == 1
    assert set(seiz_file[seiz_file["label"] == 1]["seizure_event_id"]) == {0}


def test_overlap_and_boundary_flags(prepared):
    index_df, store = prepared
    windows, _ = build_window_table(index_df, WindowingConfig())
    pos = windows[windows["label"] == 1]
    assert (pos["seizure_overlap_fraction"] > 0).all()
    # Fully-interior positive windows have overlap fraction 1.0.
    assert np.isclose(pos["seizure_overlap_fraction"].max(), 1.0)
    # Boundary windows exist around the seizure edges.
    assert int(windows["is_boundary_window"].sum()) >= 2


def test_dataset_returns_normalized_windows(prepared):
    index_df, store = prepared
    windows, _ = build_window_table(index_df, WindowingConfig())
    ds = WindowDataset(windows, store)
    x, y = ds[0]
    assert tuple(x.shape) == (18, 1024)
    # Per-window per-channel z-score: each channel ~ mean 0, std 1.
    xc = x.numpy()
    assert np.allclose(xc.mean(axis=-1), 0, atol=1e-4)
    assert np.allclose(xc.std(axis=-1), 1, atol=1e-2)
    assert y in (0, 1)


def test_negative_sampling_ratio_and_exclusion(prepared):
    index_df, store = prepared
    windows, _ = build_window_table(index_df, WindowingConfig())
    train = negative_sample(windows, ratio=5.0, exclude_seconds=10.0, seed=0)
    n_pos = int((train["label"] == 1).sum())
    n_neg = int((train["label"] == 0).sum())
    assert n_pos == 10
    assert n_neg <= 5 * n_pos
    # No sampled negative sits within the exclusion window of a seizure.
    assert (train[train["label"] == 0]["seconds_from_nearest_seizure"] >= 10.0).all()


def test_materialize_ictal_subset(prepared):
    index_df, store = prepared
    windows, _ = build_window_table(index_df, WindowingConfig())
    ictal = windows[windows["label"] == 1]
    X, y = materialize_windows(ictal, store)
    assert X.shape == (10, 18, 1024)
    assert (y == 1).all()
