"""End-to-end test of fixture writing + manifest building (uses real EDF IO)."""
import json

import pytest

from chbmit.make_manifest import assign_group, build_manifest, summarize_manifest
from tests.fixtures.synthetic_chbmit import FileSpec, PatientSpec, write_dataset


def _tiny_specs():
    """Short EDFs (20 s) so the test stays fast but exercises real EDF headers."""
    def f(name, start, seiz=None, dur=20.0):
        return FileSpec(name=name, start_clock=start, duration_sec=dur, seizures=seiz or [])

    return [
        PatientSpec("chb01", [
            f("chb01_01.edf", "11:00:00"),
            f("chb01_03.edf", "12:00:00", [(5.0, 9.0)]),
        ]),
        PatientSpec("chb21", [  # same subject as chb01
            f("chb21_19.edf", "20:00:00", [(4.0, 8.0)]),
        ]),
        PatientSpec("chb24", [
            f("chb24_03.edf", "00:00:00", [(6.0, 11.0)]),
        ], emit_clock_times=False),
    ]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw_chbmit")
    write_dataset(root, specs=_tiny_specs(), seed=1)
    return root


def test_assign_group_groups_chb01_chb21():
    assert assign_group("chb01") == "chb01"
    assert assign_group("chb21") == "chb01"
    assert assign_group("chb02") == "chb02"
    assert assign_group("chb21", group_chb01_chb21=False) == "chb21"


def test_manifest_rows_and_grouping(dataset):
    df = build_manifest(dataset, include_chb24=True)
    assert len(df) == 4  # 2 + 1 + 1 files
    groups = dict(zip(df["patient"], df["group"]))
    assert groups["chb01"] == "chb01" and groups["chb21"] == "chb01"
    assert df["group"].nunique() == 2  # {chb01(+chb21), chb24}


def test_manifest_headers_and_seizures(dataset):
    df = build_manifest(dataset, include_chb24=True)
    assert (df["sampling_rate"] == 256).all()
    assert (df["n_channels_hdr"] == 18).all()
    # Durations ~20s (allow EDF block rounding).
    assert df["duration_sec"].between(18, 22).all()
    assert int(df["n_seizures"].sum()) == 3
    seiz_row = df[df["edf_path"].str.endswith("chb01_03.edf")].iloc[0]
    assert json.loads(seiz_row["seizures"]) == [[5.0, 9.0]]


def test_chb24_inclusion_toggle(dataset):
    inc = build_manifest(dataset, include_chb24=True)
    exc = build_manifest(dataset, include_chb24=False)
    assert inc[inc["patient"] == "chb24"]["included"].all()
    chb24_exc = exc[exc["patient"] == "chb24"]
    assert not chb24_exc["included"].any()
    assert (chb24_exc["exclude_reason"] == "chb24_excluded_by_config").all()


def test_summary_stats(dataset):
    df = build_manifest(dataset, include_chb24=True)
    s = summarize_manifest(df)
    assert s["n_files"] == 4
    assert s["n_groups"] == 2
    assert s["n_seizure_events"] == 3
