"""Tests for channel-label normalization and montage matching."""
from chbmit.channel_map import (
    CANONICAL_MONTAGE_18,
    canonical_channel,
    match_montage,
)


def test_canonical_basic_and_case_space():
    assert canonical_channel("FP1-F7") == "FP1-F7"
    assert canonical_channel("Fp1-F7") == "FP1-F7"
    assert canonical_channel("  FP1 - F7 ") == "FP1-F7"
    assert canonical_channel("EEG FP1-F7") == "FP1-F7"
    assert canonical_channel("FP1-F7-REF") == "FP1-F7"


def test_old_to_new_nomenclature():
    # T3/T4/T5/T6 -> T7/T8/P7/P8
    assert canonical_channel("F7-T3") == "F7-T7"
    assert canonical_channel("T4-P4") == "T8-P4"
    assert canonical_channel("T5-O1") == "P7-O1"
    assert canonical_channel("T6-O2") == "P8-O2"


def test_dedup_digit_suffix():
    assert canonical_channel("T8-P8-0") == "T8-P8"
    assert canonical_channel("T8-P8-1") == "T8-P8"


def test_non_bipolar_returns_none():
    for bad in ["ECG", "VNS", ".", "-", "", "LOC", "EKG"]:
        assert canonical_channel(bad) is None


def test_match_full_montage():
    selected, missing = match_montage(list(CANONICAL_MONTAGE_18))
    assert missing == []
    assert [c for c, _ in selected] == list(CANONICAL_MONTAGE_18)
    assert [i for _, i in selected] == list(range(18))


def test_match_with_duplicates_and_extras():
    raw = ["EEG FP1-F7", "F7-T7", "T7-P7", "P7-O1", "ECG", "T8-P8-0", "T8-P8-1"]
    selected, missing = match_montage(raw, montage=["FP1-F7", "F7-T7", "T8-P8"])
    chosen = dict(selected)
    assert chosen["FP1-F7"] == 0
    assert chosen["T8-P8"] == 5  # first occurrence of the duplicate
    assert missing == []


def test_match_missing_reported():
    raw = ["FP1-F7", "F7-T7"]
    selected, missing = match_montage(raw)
    assert [c for c, _ in selected] == ["FP1-F7", "F7-T7"]
    assert "P8-O2" in missing and len(missing) == 16
