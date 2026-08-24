"""The per-reciter view exists because the global one hides pace differences."""

from __future__ import annotations

import json

import pytest
from conftest import make_record

from warshdata.cli import main


def write_manifest(path, profiles, n=100):
    rows = []
    for slug, (lo, hi) in profiles.items():
        for i in range(n):
            d = lo + (hi - lo) * (i / max(1, n - 1))
            rows.append(
                make_record(
                    i,
                    segment_id=f"{slug}__087__{i:04d}",
                    reciter_slug=slug,
                    source_id=f"{slug}/087",
                    duration_seconds=round(d, 2),
                    start_seconds=0.0,
                    end_seconds=round(d, 2),
                )
            )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_stats_reports_each_reciter(tmp_path, capsys):
    m = tmp_path / "segments.jsonl"
    write_manifest(m, {"ibrahim-aldosari": (4.0, 9.0), "rachid-belalya": (3.0, 8.0)})

    assert main(["stats", str(m)]) == 0
    out = capsys.readouterr().out
    assert "ibrahim-aldosari" in out
    assert "rachid-belalya" in out
    assert "Segments      : 200" in out


def test_over_segmentation_is_flagged(tmp_path, capsys):
    m = tmp_path / "segments.jsonl"
    write_manifest(m, {"fast-reciter": (0.3, 2.0)})

    main(["stats", str(m)])
    out = capsys.readouterr().out
    assert "over-segmenting" in out
    assert "higher --min-silence-duration-ms" in out


def test_missed_waqf_is_flagged(tmp_path, capsys):
    m = tmp_path / "segments.jsonl"
    write_manifest(m, {"slow-mujawwad": (12.0, 30.0)})

    main(["stats", str(m)])
    out = capsys.readouterr().out
    assert "waqf being missed" in out
    assert "lower --min-silence-duration-ms" in out


def test_healthy_reciter_is_not_flagged(tmp_path, capsys):
    m = tmp_path / "segments.jsonl"
    write_manifest(m, {"ibrahim-aldosari": (4.0, 9.0)})

    main(["stats", str(m)])
    out = capsys.readouterr().out
    assert "Worth a listen" not in out


def test_the_global_view_alone_would_hide_both_problems(tmp_path, capsys):
    """A fast and a slow reciter average out to a healthy-looking median."""
    m = tmp_path / "segments.jsonl"
    write_manifest(m, {"fast-reciter": (0.3, 2.0), "slow-mujawwad": (12.0, 30.0)})

    main(["stats", str(m)])
    out = capsys.readouterr().out
    assert "over-segmenting" in out and "waqf being missed" in out


def test_stats_on_missing_manifest_is_an_error(tmp_path):
    assert main(["stats", str(tmp_path / "nope.jsonl")]) == 1
