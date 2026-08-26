"""Auditing source recordings.

Real wav files of known duration, so the probing path is genuinely exercised
rather than mocked out.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from warshdata.audit import Probe, find_outliers, probe_all, summary
from warshdata.cli import main
from warshdata.sources import discover


def write_audio(root, reciter, stem, seconds):
    path = root / reciter / f"{stem}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(16000 * seconds), dtype="float32"), 16000)
    return path


def build_tree(root, durations):
    """durations: {reciter: {stem: seconds}}"""
    for reciter, stems in durations.items():
        for stem, seconds in stems.items():
            write_audio(root, reciter, stem, seconds)
    return root


def test_probe_reads_durations(tmp_path):
    build_tree(tmp_path, {"r1": {"094": 2.0}, "r2": {"094": 3.0}})
    probes = probe_all(discover(tmp_path))
    by_id = {p.source_id: p for p in probes}

    assert pytest.approx(by_id["r1/094"].duration_seconds, abs=0.05) == 2.0
    assert pytest.approx(by_id["r2/094"].duration_seconds, abs=0.05) == 3.0
    assert all(p.error is None for p in probes)


def test_a_wildly_long_file_is_flagged(tmp_path):
    """The real case: one reciter's surah 94 was 19 minutes, not 38 seconds."""
    build_tree(tmp_path, {
        "r1": {"094": 2.0},
        "r2": {"094": 2.2},
        "r3": {"094": 1.9},
        "odd": {"094": 40.0},
    })
    findings = find_outliers(probe_all(discover(tmp_path)))

    assert [f.source_id for f in findings] == ["odd/094"]
    assert findings[0].kind == "too long"
    assert findings[0].ratio > 3


def test_a_wildly_short_file_is_flagged(tmp_path):
    build_tree(tmp_path, {
        "r1": {"002": 30.0},
        "r2": {"002": 32.0},
        "r3": {"002": 28.0},
        "truncated": {"002": 1.0},
    })
    findings = find_outliers(probe_all(discover(tmp_path)))

    assert [f.source_id for f in findings] == ["truncated/002"]
    assert findings[0].kind == "too short"


def test_normal_pace_variation_is_not_flagged(tmp_path):
    """Reciters differ in pace by up to roughly 2x; that is not an error."""
    build_tree(tmp_path, {
        "fast": {"094": 1.2},
        "mid": {"094": 2.0},
        "slow": {"094": 2.4},
    })
    assert find_outliers(probe_all(discover(tmp_path))) == []


def test_two_reciters_are_never_outliers_of_each_other(tmp_path):
    """With no majority, calling either one wrong would be a coin flip."""
    build_tree(tmp_path, {"r1": {"094": 1.0}, "r2": {"094": 30.0}})
    assert find_outliers(probe_all(discover(tmp_path))) == []


def test_empty_file_is_reported(tmp_path):
    build_tree(tmp_path, {"r1": {"094": 2.0}, "r2": {"094": 2.0}, "r3": {"094": 2.0}})
    broken = tmp_path / "r4" / "094.wav"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"")

    findings = find_outliers(probe_all(discover(tmp_path)))
    assert any(f.kind in {"unreadable", "empty"} and f.source_id == "r4/094" for f in findings)


def test_undecodable_file_is_reported(tmp_path):
    build_tree(tmp_path, {"r1": {"094": 2.0}})
    junk = tmp_path / "r2" / "094.wav"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"this is not audio at all")

    findings = find_outliers(probe_all(discover(tmp_path)))
    assert [f.kind for f in findings] == ["unreadable"]


def test_each_surah_is_compared_only_with_itself(tmp_path):
    """A long surah must not make a short surah look truncated."""
    build_tree(tmp_path, {
        "r1": {"002": 60.0, "114": 2.0},
        "r2": {"002": 62.0, "114": 2.1},
        "r3": {"002": 58.0, "114": 1.9},
    })
    assert find_outliers(probe_all(discover(tmp_path))) == []


def test_summary_counts(tmp_path):
    build_tree(tmp_path, {"r1": {"094": 3600.0}})
    info = summary(probe_all(discover(tmp_path)))
    assert info["files"] == 1
    assert info["readable"] == 1
    assert pytest.approx(info["hours"], abs=0.01) == 1.0


def test_cli_writes_suspect_ids(tmp_path, capsys):
    build_tree(tmp_path, {
        "r1": {"094": 2.0}, "r2": {"094": 2.0}, "r3": {"094": 2.0}, "odd": {"094": 40.0},
    })
    ids = tmp_path / "suspect.txt"

    assert main(["audit", str(tmp_path), "--write-ids", str(ids), "--bitrates"]) == 0
    assert ids.read_text(encoding="utf-8").split() == ["odd/094"]
    assert "suspect file" in capsys.readouterr().out


def test_cli_reports_a_clean_tree(tmp_path, capsys):
    build_tree(tmp_path, {"r1": {"094": 2.0}, "r2": {"094": 2.1}, "r3": {"094": 1.9}})
    assert main(["audit", str(tmp_path)]) == 0
    assert "Nothing suspicious" in capsys.readouterr().out


def test_cli_errors_on_empty_directory(tmp_path):
    assert main(["audit", str(tmp_path)]) == 1
