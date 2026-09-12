"""The free Phase 3 analysis must reproduce from the committed CSVs, and its pinned numbers must
stay pinned. These are regression guards in the same spirit as the CI check on
analyze_multiseed.py: a silent NaN or a silently re-selected reference is the failure mode this
project keeps meeting."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "results_chbmit_synthetic/real_validation/analysis_tierB"
NEEDED = [A / "downstream_gated_p2.csv", A / "downstream_gated_p3.csv", A / "downstream_gated_v2.csv"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                reason="committed Phase 1-2 CSVs not present")


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("phase3_free")
    r = subprocess.run([sys.executable, "scripts/analyze_validation_selection.py", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out, r.stdout


def test_reference_selection_inflation_is_the_measured_0_035(outputs):
    out, _ = outputs
    fr = pd.read_csv(out / "frontier_valref_p3_r100.csv")
    # Against the val-selected reference, the TEST-selected reference is exactly the inflation
    # DECISION_GATE_2.md measured (+0.035), and it is a policy no deployable system corresponds to.
    row = fr[(fr.reference == "registered (val-selected)") & (fr.comparison == "best_simple_TESTselected")].iloc[0]
    assert abs(row.mean_delta_event_f1 - 0.035) < 0.002
    chosen = pd.read_csv(out / "chosen_arms_p3_r100.csv")
    assert (chosen.ref_test == chosen.ref_val).sum() == 5     # same arm in 5 of 9 cells


def test_null_floor_flags_half_of_identical_reruns_at_the_registered_margins(outputs):
    out, _ = outputs
    nl = pd.read_csv(out / "harm_curve_null.csv")
    at_f1 = nl[(nl.axis == "event_f1") & np.isclose(nl.margin, 0.01)].harm_rate.iloc[0]
    at_fp = nl[(nl.axis == "fp24h") & np.isclose(nl.margin, 0.25)].harm_rate.iloc[0]
    assert 0.40 <= at_f1 <= 0.50 and 0.40 <= at_fp <= 0.50
    sig = pd.read_csv(out / "null_floor_sigma.csv").set_index("condition")
    # The audit's measured floor (AUDIT_2026-08-31.md Sec 2), re-derived here.
    assert abs(sig.loc["real_only", "sigma_f1"] - 0.099) < 0.002
    assert abs(sig.loc["class_weighted", "sigma_f1"] - 0.159) < 0.002
    assert abs(sig.loc["classical_aug", "sigma_f1"] - 0.171) < 0.002
    assert int(sig.n.sum()) == 27


def test_every_grid_has_the_competitor_policies_and_no_nan_means(outputs):
    out, _ = outputs
    for grid in ("p2_r010", "p2_r030", "p3_r100"):
        fr = pd.read_csv(out / f"frontier_valref_{grid}.csv")
        pols = set(fr.comparison)
        for p in ("valsel4", "valsel4_fpguard", "ungated_failclosed", "ungated", "class_weighted"):
            assert p in pols, (grid, p)
        assert fr.mean_event_f1.notna().all() and fr.mean_delta_event_f1.notna().all()
        h2h = pd.read_csv(out / f"head_to_head_{grid}.csv")
        assert len(h2h) >= 7 and h2h.p_nadeau_bengio.notna().all()


def test_power_table_says_folds_not_seeds(outputs):
    out, _ = outputs
    pw = pd.read_csv(out / "power_design.csv")
    ref = pw[pw.sd_source == "DG2 reference sd 0.094"].set_index("design")
    as_run = ref.loc["3 folds x 3 seeds (as run)"]
    logo1 = ref.loc["23 folds x 1 seed (leave-one-group-out)"]
    assert abs(as_run.mde_80pct - 0.196) < 0.005          # the preprint's own figure
    assert logo1.mde_80pct < 0.10                          # LOGO at ONE seed beats 3x3 by 2x
    # Infinite seeds cannot beat this floor at the as-run geometry.
    assert abs(as_run.se_floor_inf_seeds - 0.053) < 0.002


def test_valsel_fpguard_never_worse_than_gate_on_the_tail(outputs):
    """The gate is credited with controlling the false-alarm tail. Plain validation selection with
    the gate's own FP guard produces no tail event (> +20 FP/24h vs the val-selected reference) in
    any grid, where the gate as deployed produces some. Pinned as a count, not a p-value."""
    out, _ = outputs
    for grid, gate in (("p2_r010", "gated q0.9"), ("p2_r030", "gated q0.9"), ("p3_r100", "gated q0.95")):
        fr = pd.read_csv(out / f"frontier_valref_{grid}.csv")
        fr = fr[fr.reference == "registered (val-selected)"].set_index("comparison")
        assert fr.loc["valsel4_fpguard", "tail_fp24h_gt20"] == 0
        assert fr.loc["valsel4_fpguard", "tail_fp24h_gt20"] <= fr.loc[gate, "tail_fp24h_gt20"]
