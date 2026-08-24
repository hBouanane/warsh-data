"""Corrections are the layer humans edit, so the failure modes here are the ones
that quietly corrupt a reviewed corpus."""

from __future__ import annotations

import json

import pytest
from conftest import make_record

from warshdata.corrections import Correction, apply, read_corrections
from warshdata.corrections import _suffix


def ids(records):
    return [r["segment_id"] for r in records]


def test_untouched_records_pass_through():
    records = [make_record(i) for i in range(3)]
    out, report = apply(records, [])
    assert ids(out) == ids(records)
    assert report.applied == 0


def test_drop_does_not_renumber_the_rest():
    """The whole reason ids are ordinals assigned once, not recomputed."""
    records = [make_record(i) for i in range(5)]
    out, report = apply(records, [Correction(segment_id=records[2]["segment_id"], action="drop")])

    assert report.dropped == 1
    assert ids(out) == [
        "ibrahim-aldosari__087__0000",
        "ibrahim-aldosari__087__0001",
        "ibrahim-aldosari__087__0003",
        "ibrahim-aldosari__087__0004",
    ]


def test_adjust_retimes_samples_and_seconds_together():
    records = [make_record(0)]
    out, _ = apply(records, [Correction(segment_id=records[0]["segment_id"], action="adjust",
                                        start_seconds=0.5, end_seconds=7.5)])
    rec = out[0]
    assert rec["start_seconds"] == 0.5
    assert rec["start_sample"] == 8000
    assert rec["end_sample"] == 120000
    assert rec["duration_seconds"] == 7.0
    assert rec["corrected"] is True
    assert rec["segment_id"] == "ibrahim-aldosari__087__0000"  # id survives


def test_adjust_rejects_inverted_bounds():
    records = [make_record(0)]
    out, report = apply(records, [Correction(segment_id=records[0]["segment_id"], action="adjust",
                                             start_seconds=5.0, end_seconds=1.0)])
    assert report.invalid and "end" in report.invalid[0]
    assert out[0]["start_seconds"] == 0.0  # left alone rather than corrupted


def test_split_children_and_parent_link():
    records = [make_record(0)]
    out, report = apply(records, [Correction(segment_id=records[0]["segment_id"], action="split",
                                             at_seconds=[3.0, 5.0])])
    assert report.split_into == 3
    assert ids(out) == [
        "ibrahim-aldosari__087__0000_a",
        "ibrahim-aldosari__087__0000_b",
        "ibrahim-aldosari__087__0000_c",
    ]
    assert [(r["start_seconds"], r["end_seconds"]) for r in out] == [(0.0, 3.0), (3.0, 5.0), (5.0, 8.0)]
    assert all(r["parent_segment_id"] == "ibrahim-aldosari__087__0000" for r in out)
    # the clip on disk no longer matches these bounds
    assert all(r["audio_path"] is None for r in out)


def test_split_point_outside_segment_is_rejected():
    records = [make_record(0)]
    out, report = apply(records, [Correction(segment_id=records[0]["segment_id"], action="split",
                                             at_seconds=[99.0])])
    assert report.invalid
    assert ids(out) == ["ibrahim-aldosari__087__0000"]


def test_merge_keeps_the_first_id_and_spans_both():
    records = [make_record(i) for i in range(3)]
    out, report = apply(records, [Correction(segment_id=records[0]["segment_id"], action="merge",
                                             with_ids=[records[1]["segment_id"]])])
    assert report.merged_away == 1
    assert ids(out) == ["ibrahim-aldosari__087__0000", "ibrahim-aldosari__087__0002"]
    merged = out[0]
    assert (merged["start_seconds"], merged["end_seconds"]) == (0.0, 18.0)
    assert merged["merged_from"] == ["ibrahim-aldosari__087__0000", "ibrahim-aldosari__087__0001"]
    assert merged["audio_path"] is None


def test_merge_across_sources_is_refused():
    a = make_record(0)
    b = make_record(1, source_id="other/002")
    out, report = apply([a, b], [Correction(segment_id=a["segment_id"], action="merge",
                                            with_ids=[b["segment_id"]])])
    assert any("refusing to merge across sources" in m for m in report.invalid)
    assert a["segment_id"] in ids(out)


def test_merge_carries_is_last_of_source():
    a = make_record(0)
    b = make_record(1, is_last_of_source=True)
    out, _ = apply([a, b], [Correction(segment_id=a["segment_id"], action="merge",
                                       with_ids=[b["segment_id"]])])
    assert out[0]["is_last_of_source"] is True


def test_drift_is_reported_when_the_segment_moved():
    records = [make_record(0)]
    c = Correction(segment_id=records[0]["segment_id"], action="adjust", end_seconds=7.0,
                   orig_start_seconds=99.0)
    out, report = apply(records, [c])
    assert report.drifted == ["ibrahim-aldosari__087__0000"]
    assert out[0]["end_seconds"] == 7.0  # applied anyway by default


def test_strict_drift_skips_the_correction():
    records = [make_record(0)]
    c = Correction(segment_id=records[0]["segment_id"], action="adjust", end_seconds=7.0,
                   orig_start_seconds=99.0)
    out, report = apply(records, [c], strict_drift=True)
    assert report.drifted
    assert out[0]["end_seconds"] == 8.0  # untouched


def test_unmatched_correction_is_reported():
    out, report = apply([make_record(0)], [Correction(segment_id="does-not-exist", action="drop")])
    assert report.unmatched == ["does-not-exist"]
    assert len(out) == 1


def test_unknown_action_is_reported_not_applied():
    records = [make_record(0)]
    out, report = apply(records, [Correction(segment_id=records[0]["segment_id"], action="frobnicate")])
    assert report.invalid
    assert out[0]["end_seconds"] == 8.0


@pytest.mark.parametrize("n,expected", [(0, "a"), (25, "z"), (26, "aa"), (51, "az"), (52, "ba")])
def test_suffix_survives_more_than_26_children(n, expected):
    assert _suffix(n) == expected


def test_read_corrections_accepts_with_alias(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"segment_id": "a", "action": "merge", "with": ["b"]}) + "\n"
        + "# a comment line\n"
        + json.dumps({"segment_id": "c", "action": "drop"}) + "\n",
        encoding="utf-8",
    )
    cors = read_corrections(path)
    assert [c.segment_id for c in cors] == ["a", "c"]
    assert cors[0].with_ids == ["b"]
