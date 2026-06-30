"""Tests for the CHB-MIT summary parser, including format variants."""
from chbmit.parse_annotations import parse_summary_text
from tests.fixtures.synthetic_chbmit import default_spec, summary_text


def test_basic_counts_and_seizures():
    spec = next(s for s in default_spec() if s.patient == "chb01")
    info = parse_summary_text(summary_text(spec), patient="chb01")
    assert info.sampling_rate == 256
    assert len(info.files) == 3
    assert info.n_seizure_events == 3  # 0 + 1 + 2
    by_name = {f.file_name: f for f in info.files}
    assert by_name["chb01_01.edf"].n_seizures == 0
    assert by_name["chb01_03.edf"].n_seizures == 1
    assert by_name["chb01_03.edf"].seizures[0].start_sec == 100.0
    assert by_name["chb01_03.edf"].seizures[0].end_sec == 130.0
    # Indexed multi-seizure format.
    multi = by_name["chb01_04.edf"]
    assert multi.n_seizures == 2
    assert [s.start_sec for s in multi.seizures] == [60.0, 200.0]
    assert [s.end_sec for s in multi.seizures] == [90.0, 235.0]


def test_channels_attached_to_every_file():
    spec = next(s for s in default_spec() if s.patient == "chb01")
    info = parse_summary_text(summary_text(spec), patient="chb01")
    assert len(info.channel_blocks) == 1
    for f in info.files:
        assert len(f.channels) == 18
        assert f.channels[0] == "FP1-F7"


def test_chb24_like_missing_clock_times():
    spec = next(s for s in default_spec() if s.patient == "chb24")
    info = parse_summary_text(summary_text(spec), patient="chb24")
    assert len(info.files) == 2
    for f in info.files:
        assert f.start_clock is None and f.end_clock is None
    assert info.files[1].n_seizures == 1


def test_end_before_time_not_confused_with_start():
    # 'End Time' contains 'Time'; ensure starts/ends are not mixed up.
    spec = next(s for s in default_spec() if s.patient == "chb03")
    info = parse_summary_text(summary_text(spec), patient="chb03")
    f0 = info.files[0]
    assert f0.seizures[0].start_sec < f0.seizures[0].end_sec


def test_seizure_duration_and_warnings_clean():
    spec = next(s for s in default_spec() if s.patient == "chb01")
    info = parse_summary_text(summary_text(spec), patient="chb01")
    multi = next(f for f in info.files if f.file_name == "chb01_04.edf")
    assert multi.seizure_duration_sec == (90 - 60) + (235 - 200)
    # No pairing/declared-count warnings expected for well-formed input.
    assert not any("declared" in w for w in info.warnings)
