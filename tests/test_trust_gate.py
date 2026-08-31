"""Trust-gate unit + integration tests (v5.3 Sec 3.1, 5, 6).

Covers: teacher window scoring, q-quantile admission + monotonicity, the event-level
fail-closed decision, harm/tail-risk metrics, and end-to-end gated cells (admit + revert).
"""
import numpy as np
import pytest

from evaluation.stats import cvar, harm_rate, paired_delta, worst_delta
from models import build_model
from synthetic.trust_gate import (
    TrustGateConfig,
    admission_threshold,
    admit_indices,
    fail_closed_decision,
    run_admission,
    score_windows,
)


# --- admission stage ---------------------------------------------------
def test_score_windows_shape_and_range():
    model = build_model("eegnet", n_channels=4, n_samples=512)
    X = np.random.default_rng(0).standard_normal((7, 4, 512)).astype("float32")
    s = score_windows(model, X)
    assert s.shape == (7,)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert score_windows(model, np.empty((0, 4, 512), dtype="float32")).shape == (0,)


def test_admission_threshold_is_quantile():
    ref = np.linspace(0.0, 1.0, 101)
    assert abs(admission_threshold(ref, 0.5) - 0.5) < 1e-9
    assert admission_threshold(ref, 0.99) > admission_threshold(ref, 0.75)
    assert admission_threshold(np.array([]), 0.9) == 0.0  # empty ref -> admit all


def test_admission_stricter_with_higher_q():
    # Higher q -> higher threshold -> fewer admitted (monotone non-increasing).
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 1, 200)
    synth = rng.uniform(0, 1, 200)
    counts = []
    for q in (0.5, 0.75, 0.9, 0.99):
        thr = admission_threshold(ref, q)
        counts.append(len(admit_indices(synth, thr)))
    assert counts == sorted(counts, reverse=True)


def test_admit_indices_keeps_top_k():
    scores = np.array([0.1, 0.95, 0.6, 0.99, 0.5])
    idx = admit_indices(scores, threshold=0.4, max_keep=2)
    # highest two above 0.4 are indices 3 (0.99) and 1 (0.95)
    assert set(idx.tolist()) == {1, 3}


def test_run_admission_caps_at_target():
    model = build_model("eegnet", n_channels=4, n_samples=256)
    pool = np.random.default_rng(2).standard_normal((40, 4, 256)).astype("float32")
    real = np.random.default_rng(3).standard_normal((20, 4, 256)).astype("float32")
    adm = run_admission(model, pool, real, TrustGateConfig(q=0.5), target_count=5)
    assert adm.n_admitted <= 5
    assert adm.n_pool == 40
    assert 0.0 <= adm.admission_rate <= 1.0


# --- realized vs requested dose ----------------------------------------
# These encode DECISION_GATE_2.md Q6. The gate ran for an entire phase admitting 0.4% of what it
# was asked for, and nothing in the test suite could see it: the cap test above bounds admission
# from ABOVE only. The paper's transferable claim is that comparing realized against requested is
# the cheap diagnostic that catches this, so the diagnostic lives here as a test.

@pytest.mark.parametrize("q,target", [(0.95, 100), (0.9917, 60), (0.50, 40), (0.75, 30)])
def test_pool_reference_hits_the_requested_dose(q, target):
    """reference="pool" is a rank cut, so admitted == min(oversample*(1-q), 1) * target."""
    model = build_model("eegnet", n_channels=4, n_samples=256)
    cfg = TrustGateConfig(q=q, reference="pool")
    pool = np.random.default_rng(11).standard_normal(
        (cfg.oversample * target, 4, 256)).astype("float32")
    adm = run_admission(model, pool, None, cfg, target_count=target)
    predicted = int(min(cfg.oversample * (1.0 - q), 1.0) * target)
    # +/-1 for the quantile's interpolation on a finite pool.
    assert abs(adm.n_admitted - predicted) <= 1, (
        f"q={q} target={target}: admitted {adm.n_admitted}, closed form predicts {predicted}")


def test_real_ictal_reference_collapses_when_the_teacher_saturates():
    """The Phase 1/2 defect, pinned: a teacher that scores real ictal near 1.0 admits ~nothing.

    Regression guard -- if this ever starts admitting a real dose, the admission path changed and
    DECISION_GATE_2.md Q6 needs re-deriving.
    """
    class SaturatedTeacher:
        """Returns sample 0 of channel 0 as the logit, so the arrays below set the scores."""
        def eval(self):
            return self

        def __call__(self, xb):
            return xb[:, 0, 0]

    real = np.full((200, 4, 256), 0.0, dtype="float32")
    real[:, 0, 0] = 8.0                      # sigmoid -> ~1.0
    pool = np.full((600, 4, 256), 0.0, dtype="float32")
    pool[:, 0, 0] = np.random.default_rng(5).normal(-2.0, 1.0, 600)   # sigmoid -> mostly << 1
    cfg = TrustGateConfig(q=0.90, reference="real_ictal")
    adm = run_admission(SaturatedTeacher(), pool, real, cfg, target_count=100)
    assert adm.n_admitted < 0.05 * 100, (
        f"real_ictal reference admitted {adm.n_admitted} of a requested 100; the Q6 collapse "
        "is no longer reproduced")

    # Same teacher, same pool, published reference -> the requested dose arrives.
    adm_pool = run_admission(SaturatedTeacher(), pool, real,
                             TrustGateConfig(q=0.90, reference="pool"), target_count=100)
    assert adm_pool.n_admitted >= 55, adm_pool.n_admitted


def test_default_reference_is_the_published_rank_cut():
    """Guards the 2026-08-31 default flip: a fresh clone must not run the disabled gate."""
    assert TrustGateConfig().reference == "pool"


# --- fail-closed selection ---------------------------------------------
def test_fail_closed_admits_on_improvement():
    cfg = TrustGateConfig(admit_margin_event_f1=0.0, fp24h_safety_slack=0.25)
    d = fail_closed_decision(0.60, 5.0, 0.55, 5.0, cfg, n_admitted=10)
    assert d.admitted and not d.reverted


def test_fail_closed_reverts_when_f1_below_margin():
    cfg = TrustGateConfig(admit_margin_event_f1=0.0)
    d = fail_closed_decision(0.50, 5.0, 0.55, 5.0, cfg, n_admitted=10)
    assert d.reverted and d.reason == "val_event_f1_below_margin"


def test_fail_closed_reverts_when_fp_exceeds_slack():
    cfg = TrustGateConfig(admit_margin_event_f1=0.0, fp24h_safety_slack=0.25)
    d = fail_closed_decision(0.70, 6.0, 0.55, 5.0, cfg, n_admitted=10)
    assert d.reverted and d.reason == "val_fp24h_exceeds_safety_slack"


def test_fail_closed_reverts_when_nothing_admitted():
    d = fail_closed_decision(0.9, 1.0, 0.5, 1.0, TrustGateConfig(min_admitted=1), n_admitted=0)
    assert d.reverted and d.reason == "no_synthetic_admitted"


# --- harm / tail-risk metrics ------------------------------------------
def test_harm_metrics_known_values():
    deltas = np.array([-0.2, -0.1, 0.0, 0.05, 0.1])
    # event_f1 (higher better): harm if delta < -0.01 -> two of five
    assert harm_rate(deltas, -0.01, higher_is_better=True) == pytest.approx(0.4)
    assert worst_delta(deltas, higher_is_better=True) == pytest.approx(-0.2)
    # CVaR worst 40% (k=2) of the lowest deltas: mean(-0.2, -0.1)
    assert cvar(deltas, alpha=0.4, higher_is_better=True) == pytest.approx(-0.15)


def test_harm_metrics_fp_direction():
    deltas = np.array([-1.0, 0.0, 0.5, 1.0])  # FP/24h deltas: harm is positive inflation
    assert harm_rate(deltas, 0.25, higher_is_better=False) == pytest.approx(0.5)
    assert worst_delta(deltas, higher_is_better=False) == pytest.approx(1.0)


def test_paired_delta_reports_tail_risk():
    a = [0.5, 0.4, 0.6, 0.45]
    b = [0.55, 0.55, 0.55, 0.55]
    r = paired_delta(a, b, higher_is_better=True, harm_threshold=-0.01)
    assert r.worst_delta is not None and r.cvar is not None
    assert 0.0 <= r.harm_rate <= 1.0


# --- end-to-end gated cells --------------------------------------------
@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from chbmit.preprocess_edf import PreprocessConfig
    from chbmit.window_metadata import WindowingConfig
    from experiments.prepare import prepare_dataset
    from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset

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
    write_dataset(raw, specs=specs, seed=37)
    proc = tmp_path_factory.mktemp("proc")
    res = tmp_path_factory.mktemp("res")
    return prepare_dataset(raw, proc, res, n_folds=2, seed=42,
                           pre_cfg=PreprocessConfig(target_sampling_rate=256),
                           win_cfg=WindowingConfig(sampling_rate=256))


def _grid(prepared, gate_cfg, tmp_path):
    from experiments.grid import GridSpec, run_grid
    from experiments.training import TrainConfig
    from synthetic.cvae_provider import CVAEConfig

    grid = GridSpec(
        detectors=["eegnet"],
        conditions=["real_only", "ungated_synthetic_aug", "trust_gated_synthetic_aug"],
        scarcity_fractions=[1.0], seeds=[42], generators=["cvae"], do_quality=False,
    )
    train_cfg = TrainConfig(epochs=1, batch_size=32, monitor_max_neg_per_pos=10)
    gen_cfgs = {"cvae": CVAEConfig(epochs=1, batch_size=32, min_ictal_windows=8, latent_dim=16)}
    out = run_grid(prepared, grid, train_cfg=train_cfg, gen_configs=gen_cfgs,
                   results_dir=tmp_path, gate_cfg=gate_cfg, log=lambda *a, **k: None)
    return out["results"]


def test_gated_cell_admits(prepared, tmp_path):
    # Permissive gate (negative margin, huge slack) -> admit the augmented model.
    results = _grid(prepared, TrustGateConfig(q=0.5, admit_margin_event_f1=-1.0,
                                              fp24h_safety_slack=1e9), tmp_path)
    gated = [r for r in results if r["spec"]["condition"] == "trust_gated_synthetic_aug"]
    assert gated, "no gated cells produced"
    for r in gated:
        assert "gate" in r
        assert r["gate"]["n_pool"] >= r["gate"]["n_admitted"]
        # with this gate at least some cells should admit
    assert any(not r["reverted_to_real_only"] for r in gated)


def test_gated_cell_fails_closed(prepared, tmp_path):
    # Impossible margin -> always revert; reverted metrics must equal the real-only cell's.
    results = _grid(prepared, TrustGateConfig(q=0.9, admit_margin_event_f1=10.0), tmp_path)
    by = {}
    for r in results:
        by.setdefault((r["spec"]["fold"], r["spec"]["condition"]), r)
    for (fold, cond), r in by.items():
        if cond != "trust_gated_synthetic_aug":
            continue
        assert r["reverted_to_real_only"] is True
        assert r["gate"]["reverted"] is True
        ro = by[(fold, "real_only")]
        assert r["event_metrics"]["event_f1"] == ro["event_metrics"]["event_f1"]
