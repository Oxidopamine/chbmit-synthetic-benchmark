"""Generator/provider tests: fit/generate/provenance/save-load + synthetic_aug cell."""
import numpy as np
import pytest

from synthetic.cvae_provider import CVAEConfig, CVAEProvider
from synthetic.precomputed_provider import PrecomputedProvider
from synthetic.wgan_gp_provider import WGANConfig, WGANGPProvider


def _ictal(n=40, C=18, T=256, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, C, T)).astype("float32")
    return x, np.ones(n, dtype="int64")


def _meta():
    return {
        "fold_id": 0,
        "source_train_patient_groups": ["chb01", "chb02"],
        "source_train_seizure_events": [0, 1, 2],
        "channels": [f"C{i}" for i in range(18)],
        "window_samples": 256,
        "normalization_protocol": "per_window_channel_zscore",
    }


@pytest.mark.parametrize("Provider,Config", [
    (WGANGPProvider, WGANConfig),
    (CVAEProvider, CVAEConfig),
])
def test_provider_fit_generate_shapes_and_provenance(Provider, Config):
    X, y = _ictal()
    cfg = Config(epochs=1, batch_size=16, min_ictal_windows=8)
    if hasattr(cfg, "latent_dim"):
        cfg.latent_dim = 16
    prov = Provider(cfg)
    prov.fit(X, y, _meta())
    assert prov.fitted and not getattr(prov, "skipped", False)
    out = prov.generate(5, seed=1)
    assert out.shape == (5, 18, 256)
    assert np.isfinite(out).all()
    # Provenance recorded (Rule 8).
    assert prov.metadata.fold_id == 0
    assert prov.metadata.source_train_seizure_events == [0, 1, 2]
    assert prov.metadata.n_train_ictal_windows == 40


def test_provider_skips_below_threshold():
    X, y = _ictal(n=3)
    prov = CVAEProvider(CVAEConfig(epochs=1, min_ictal_windows=8))
    prov.fit(X, y, _meta())
    assert prov.skipped and not prov.fitted
    with pytest.raises(RuntimeError):
        prov.generate(2)


def test_provider_save_load_roundtrip(tmp_path):
    X, y = _ictal()
    prov = CVAEProvider(CVAEConfig(epochs=1, batch_size=16, min_ictal_windows=8, latent_dim=16))
    prov.fit(X, y, _meta())
    prov.save(tmp_path / "cvae")
    loaded = CVAEProvider(CVAEConfig(latent_dim=16)).load(tmp_path / "cvae")
    out = loaded.generate(4, seed=2)
    assert out.shape == (4, 18, 256)
    assert loaded.metadata.fold_id == 0


def test_precomputed_provenance_enforced(tmp_path):
    windows = np.random.randn(10, 18, 256).astype("float32")
    good_meta = {
        "fold_id": 0, "source_train_patient_groups": ["chb01"],
        "source_train_seizure_events": [0], "provider_name": "ext",
        "paradigm": "patient_specific", "synthetic_ratio": 1.0, "generation_seed": 42,
        "channels": ["c"] * 18, "window_samples": 256,
        "normalization_protocol": "per_window_channel_zscore",
    }
    good = tmp_path / "good.npz"
    np.savez(good, windows=windows, metadata=good_meta)
    prov = PrecomputedProvider().load_npz(good)
    assert prov.generate(3).shape == (3, 18, 256)
    assert prov.paradigm == "patient_specific"

    bad_meta = {k: v for k, v in good_meta.items() if k != "fold_id"}
    bad = tmp_path / "bad.npz"
    np.savez(bad, windows=windows, metadata=bad_meta)
    with pytest.raises(ValueError):
        PrecomputedProvider().load_npz(bad)


# --- integration: synthetic_aug cell -----------------------------------
def test_synthetic_aug_cell(tmp_path_factory):
    from chbmit.preprocess_edf import PreprocessConfig
    from chbmit.window_metadata import WindowingConfig
    from experiments.prepare import prepare_dataset
    from experiments.training import CellSpec, TrainConfig, run_cell
    from synthetic.train_provider import fit_provider_for_cell
    from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset

    def f(name, start, seiz, dur=120.0):
        return FileSpec(name=name, start_clock=start, duration_sec=dur, seizures=seiz)

    specs = [
        PatientSpec("chb01", [f("chb01_01.edf", "11:00:00", [(20, 60)]),
                              f("chb01_02.edf", "12:00:00", [(30, 75)])]),
        PatientSpec("chb02", [f("chb02_01.edf", "09:00:00", [(25, 70)]),
                              f("chb02_02.edf", "10:00:00", [(40, 90)])]),
        PatientSpec("chb03", [f("chb03_01.edf", "08:00:00", [(20, 65)]),
                              f("chb03_02.edf", "09:30:00", [(35, 85)])]),
        PatientSpec("chb04", [f("chb04_01.edf", "07:00:00", [(30, 80)]),
                              f("chb04_02.edf", "08:30:00", [(25, 70)])]),
    ]
    raw = tmp_path_factory.mktemp("raw")
    write_dataset(raw, specs=specs, seed=21)
    proc = tmp_path_factory.mktemp("proc")
    res = tmp_path_factory.mktemp("res")
    prepared = prepare_dataset(raw, proc, res, n_folds=2, seed=42,
                               pre_cfg=PreprocessConfig(target_sampling_rate=256),
                               win_cfg=WindowingConfig(sampling_rate=256))

    prov = CVAEProvider(CVAEConfig(epochs=2, batch_size=32, min_ictal_windows=8, latent_dim=16))
    info = fit_provider_for_cell(prov, prepared.index_df, prepared.windows_df,
                                 prepared.events_df, prepared.store, prepared.splits[0],
                                 fold=0, seed=42, scarcity_fraction=1.0)
    assert info["fitted"] and not info["skipped"]

    spec = CellSpec(fold=0, seed=42, scarcity_fraction=1.0, detector="eegnet",
                    condition="synthetic_aug", generator="cvae", synthetic_ratio=1.0)
    out = run_cell(spec, prepared.index_df, prepared.windows_df, prepared.events_df,
                   prepared.store, prepared.splits[0],
                   cfg=TrainConfig(epochs=2, batch_size=32, monitor_max_neg_per_pos=10),
                   synthetic_provider=lambda n, seed: prov.generate(n, seed=seed))
    assert 0.0 <= out["selected_threshold"] <= 1.0
    assert out["spec"]["generator"] == "cvae"
    assert "event_f1" in out["event_metrics"]
