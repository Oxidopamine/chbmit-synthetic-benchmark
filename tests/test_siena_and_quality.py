"""Tests for the Siena annotation parser and synthetic quality checks."""
import numpy as np

from chbmit.siena import parse_siena_seizure_list
from synthetic.quality_checks import (
    cross_channel_corr_frobenius,
    discriminator_auc,
    mmd_rbf,
    nearest_neighbor_memorization,
    psd_log_distance,
)

SIENA_SAMPLE = """
Seizure n 1
File name: PN00-1.edf
Registration start time: 19.39.33
Registration end time: 20.40.03
Seizure start time: 19.58.36
Seizure end time: 19.59.46

Seizure n 2
File name: PN00-1.edf
Registration start time: 19.39.33
Registration end time: 20.40.03
Seizure start time: 20.10.00
Seizure end time: 20.11.30
"""


def test_parse_siena_seizure_list():
    out = parse_siena_seizure_list(SIENA_SAMPLE)
    assert "PN00-1.edf" in out
    seizures = out["PN00-1.edf"]
    assert len(seizures) == 2
    # 19:58:36 - 19:39:33 = 1143 s; 19:59:46 - 19:39:33 = 1213 s
    s, e = seizures[0]
    assert abs(s - 1143) < 1 and abs(e - 1213) < 1
    assert seizures[1][0] < seizures[1][1]


def test_parse_siena_handles_midnight_wrap():
    txt = """
    Seizure n 1
    File name: PNx-1.edf
    Registration start time: 23.59.00
    Registration end time: 00.30.00
    Seizure start time: 00.01.00
    Seizure end time: 00.02.00
    """
    out = parse_siena_seizure_list(txt)
    s, e = out["PNx-1.edf"][0]
    # 00:01:00 next day - 23:59:00 = 120 s
    assert abs(s - 120) < 1 and e > s


def test_quality_metrics_distinguish_real_from_noise():
    rng = np.random.default_rng(0)
    # "real": low-frequency structured; "synth_good": similar; "synth_bad": white noise
    t = np.linspace(0, 1, 256)
    real = np.stack([np.stack([np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6)) for _ in range(8)])
                     for _ in range(40)]).astype("float32")
    synth_good = np.stack([np.stack([np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6)) for _ in range(8)])
                           for _ in range(40)]).astype("float32")
    synth_bad = rng.normal(size=(40, 8, 256)).astype("float32")

    auc_bad = discriminator_auc(real, synth_bad)
    auc_good = discriminator_auc(real, synth_good)
    assert auc_bad > auc_good  # white noise is easier to tell apart
    assert psd_log_distance(real, synth_bad) > psd_log_distance(real, synth_good)
    # numeric checks return finite values
    assert np.isfinite(cross_channel_corr_frobenius(real, synth_good))
    assert np.isfinite(mmd_rbf(real.reshape(40, -1), synth_good.reshape(40, -1)))
    nn = nearest_neighbor_memorization(synth_good, real)
    assert nn["nn_dist_mean"] >= nn["nn_dist_min"]
