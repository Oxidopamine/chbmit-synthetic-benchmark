"""End-to-end: prepare fixture -> run training cells -> metrics (real-only + baselines)."""
import math

import numpy as np
import pytest

from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig
from experiments.prepare import prepare_dataset
from experiments.training import CellSpec, TrainConfig, run_cell
from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset


def _specs():
    # 4 patient groups, ~120 s files, each with seizures -> supports 2 folds.
    def f(name, start, seiz, dur=120.0):
        return FileSpec(name=name, start_clock=start, duration_sec=dur, seizures=seiz)

    return [
        PatientSpec("chb01", [f("chb01_01.edf", "11:00:00", [(30, 55)]),
                              f("chb01_02.edf", "12:00:00", [(40, 70)])]),
        PatientSpec("chb02", [f("chb02_01.edf", "09:00:00", [(35, 60)]),
                              f("chb02_02.edf", "10:00:00", [(50, 80)])]),
        PatientSpec("chb03", [f("chb03_01.edf", "08:00:00", [(25, 50)]),
                              f("chb03_02.edf", "09:30:00", [(60, 95)])]),
        PatientSpec("chb04", [f("chb04_01.edf", "07:00:00", [(45, 75)]),
                              f("chb04_02.edf", "08:30:00", [(20, 48)])]),
    ]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    write_dataset(raw, specs=_specs(), seed=11)
    proc = tmp_path_factory.mktemp("proc")
    res = tmp_path_factory.mktemp("res")
    return prepare_dataset(raw, proc, res, n_folds=2, seed=42,
                           pre_cfg=PreprocessConfig(target_sampling_rate=256),
                           win_cfg=WindowingConfig(sampling_rate=256))


def _cfg():
    return TrainConfig(epochs=2, batch_size=32, early_stopping_patience=5,
                       monitor_max_neg_per_pos=10, device="cpu")


def test_prepare_artifacts(prepared):
    assert prepared.audit_accepted
    assert len(prepared.channels) == 18
    assert len(prepared.splits) == 2
    assert (prepared.windows_df["label"] == 1).sum() > 0
    assert len(prepared.events_df) == 8  # one seizure per file, 8 files


def test_real_only_cell_produces_metrics(prepared):
    spec = CellSpec(fold=0, seed=42, scarcity_fraction=1.0, detector="eegnet",
                    condition="real_only")
    res = run_cell(spec, prepared.index_df, prepared.windows_df, prepared.events_df,
                   prepared.store, prepared.splits[0], cfg=_cfg())
    assert 0.0 <= res["selected_threshold"] <= 1.0
    wm = res["window_metrics"]
    assert "auprc" in wm and "auroc" in wm
    em = res["event_metrics"]
    assert set(["event_sensitivity", "event_f1", "fp_per_24h"]).issubset(em)
    assert res["n_train_windows"] > 0
    assert res["n_params"] > 0
    assert isinstance(res["sensitivity_at_fa"], dict)


@pytest.mark.parametrize("condition", ["class_weighted", "balanced_sampler", "classical_aug"])
def test_baseline_conditions_run(prepared, condition):
    spec = CellSpec(fold=1, seed=42, scarcity_fraction=1.0, detector="tcn", condition=condition)
    res = run_cell(spec, prepared.index_df, prepared.windows_df, prepared.events_df,
                   prepared.store, prepared.splits[1], cfg=_cfg())
    em = res["event_metrics"]
    # Metrics must be finite numbers or NaN (not crashes); FP/24h non-negative when present.
    fp = em["fp_per_24h"]
    assert math.isnan(fp) or fp >= 0.0


def test_scarcity_reduces_training_positives(prepared):
    full = run_cell(CellSpec(0, 42, 1.0, "eegnet", "real_only"),
                    prepared.index_df, prepared.windows_df, prepared.events_df,
                    prepared.store, prepared.splits[0], cfg=_cfg())
    scarce = run_cell(CellSpec(0, 42, 0.5, "eegnet", "real_only"),
                      prepared.index_df, prepared.windows_df, prepared.events_df,
                      prepared.store, prepared.splits[0], cfg=_cfg())
    assert scarce["n_train_pos"] <= full["n_train_pos"]
