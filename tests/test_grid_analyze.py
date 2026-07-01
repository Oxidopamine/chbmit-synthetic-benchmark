"""Smoke test: run_grid (with generator + quality) -> analyze (deltas + Fig 8)."""
import pandas as pd
import pytest

from chbmit.preprocess_edf import PreprocessConfig
from chbmit.window_metadata import WindowingConfig
from experiments.aggregate_results import save_results
from experiments.analyze import generator_detector_deltas, run_analysis
from experiments.grid import GridSpec, run_grid
from experiments.prepare import prepare_dataset
from experiments.training import TrainConfig
from synthetic.cvae_provider import CVAEConfig
from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    def f(name, start, seiz, dur=120.0):
        return FileSpec(name=name, start_clock=start, duration_sec=dur, seizures=seiz)

    specs = [
        PatientSpec("chb01", [f("chb01_01.edf", "11:00:00", [(20, 65)]),
                              f("chb01_02.edf", "12:00:00", [(30, 80)])]),
        PatientSpec("chb02", [f("chb02_01.edf", "09:00:00", [(25, 70)]),
                              f("chb02_02.edf", "10:00:00", [(40, 95)])]),
        PatientSpec("chb03", [f("chb03_01.edf", "08:00:00", [(20, 70)]),
                              f("chb03_02.edf", "09:30:00", [(35, 90)])]),
        PatientSpec("chb04", [f("chb04_01.edf", "07:00:00", [(30, 85)]),
                              f("chb04_02.edf", "08:30:00", [(25, 75)])]),
    ]
    raw = tmp_path_factory.mktemp("raw")
    write_dataset(raw, specs=specs, seed=31)
    proc = tmp_path_factory.mktemp("proc")
    res = tmp_path_factory.mktemp("res")
    return prepare_dataset(raw, proc, res, n_folds=2, seed=42,
                           pre_cfg=PreprocessConfig(target_sampling_rate=256),
                           win_cfg=WindowingConfig(sampling_rate=256))


def test_grid_then_analyze(prepared, tmp_path):
    grid = GridSpec(
        detectors=["eegnet"],
        conditions=["real_only", "classical_aug", "ungated_synthetic_aug"],
        scarcity_fractions=[1.0],
        seeds=[42],
        generators=["cvae"],
        do_quality=True,
    )
    train_cfg = TrainConfig(epochs=1, batch_size=32, monitor_max_neg_per_pos=10)
    gen_cfgs = {"cvae": CVAEConfig(epochs=1, batch_size=32, min_ictal_windows=8, latent_dim=16)}

    out = run_grid(prepared, grid, train_cfg=train_cfg, gen_configs=gen_cfgs,
                   results_dir=tmp_path, log=lambda *a, **k: None)
    results = out["results"]
    # 2 folds x (real_only, classical_aug, ungated_synthetic_aug/cvae) = 6 cells
    assert len(results) == 6
    assert any(r["spec"]["condition"] == "ungated_synthetic_aug" for r in results)
    assert len(out["quality"]) >= 1  # quality ran per fitted generator

    csv = save_results(results, tmp_path / "tables", "grid")
    df = pd.read_csv(csv)
    deltas = generator_detector_deltas(df, metric="event_f1")
    assert not deltas.empty
    assert set(["detector", "generator", "mean_delta", "ci_low", "ci_high"]).issubset(deltas.columns)

    analysis = run_analysis(csv, tmp_path)
    assert analysis["figures"]["fig8_event_f1"] is not None
    from pathlib import Path
    assert Path(analysis["figures"]["fig8_event_f1"]).exists()
