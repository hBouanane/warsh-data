"""Excerpting suspect recordings into a playable page."""

from __future__ import annotations

import base64
import re

import numpy as np
import pytest
import soundfile as sf

from warshdata.cli import main
from warshdata.preview import _positions, build_page, extract_excerpts


def write_audio(path, seconds, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), rate)
    return path


def test_positions_spread_across_the_recording():
    spots = _positions(duration=600.0, count=3, clip_seconds=15.0)
    labels = [label for label, _ in spots]
    starts = [start for _, start in spots]

    assert labels == ["start", "middle", "end"]
    assert starts == sorted(starts), "excerpts must be in chronological order"
    assert starts[0] < 60 and starts[-1] > 500


def test_short_recording_is_taken_whole():
    assert _positions(duration=10.0, count=3, clip_seconds=15.0) == [("whole", 0.0)]


def test_positions_never_run_past_the_end():
    duration, clip = 40.0, 15.0
    for _, start in _positions(duration, 3, clip):
        assert 0 <= start <= duration - clip


def test_extract_excerpts_writes_playable_clips(tmp_path):
    source = write_audio(tmp_path / "r" / "094.wav", 120.0)
    excerpts = extract_excerpts("r/094", source, 120.0, tmp_path / "clips",
                                count=3, clip_seconds=5.0)

    assert len(excerpts) == 3
    for excerpt in excerpts:
        assert excerpt.path.exists()
        assert excerpt.data
        data, rate = sf.read(excerpt.path, dtype="float32")
        # Roughly the requested length, and actually decodable.
        assert 4.0 < len(data) / rate <= 5.1


def test_excerpts_come_from_different_parts_of_the_file(tmp_path):
    """A wrong recitation usually sounds normal at the start; the middle tells."""
    rate = 16000
    path = tmp_path / "r" / "094.wav"
    path.parent.mkdir(parents=True)
    quiet = np.zeros(rate * 60, dtype="float32")
    loud = np.full(rate * 60, 0.5, dtype="float32")
    sf.write(path, np.concatenate([quiet, loud]), rate)

    excerpts = extract_excerpts("r/094", path, 120.0, tmp_path / "clips",
                                count=3, clip_seconds=5.0)
    peaks = [float(np.abs(sf.read(e.path, dtype="float32")[0]).max()) for e in excerpts]

    assert peaks[0] < 0.1, "first excerpt should come from the quiet half"
    assert peaks[-1] > 0.4, "last excerpt should come from the loud half"


def test_page_is_self_contained(tmp_path):
    source = write_audio(tmp_path / "r" / "094.wav", 60.0)
    excerpts = extract_excerpts("r/094", source, 60.0, tmp_path / "clips",
                                count=2, clip_seconds=3.0)
    page = build_page([("r/094", "too long: 29x the median", 60.0, excerpts)],
                      tmp_path / "index.html")

    text = page.read_text(encoding="utf-8")
    assert text.count("<audio") == len(excerpts)
    # Every player carries its audio inline: no external requests, nothing to
    # break when the page is downloaded off Colab.
    assert 'src="data:' in text
    assert "http://" not in text and "https://" not in text
    assert "r/094" in text and "29x the median" in text


def test_embedded_audio_decodes_back(tmp_path):
    source = write_audio(tmp_path / "r" / "094.wav", 30.0)
    excerpts = extract_excerpts("r/094", source, 30.0, tmp_path / "clips",
                                count=1, clip_seconds=3.0)
    page = build_page([("r/094", "note", 30.0, excerpts)], tmp_path / "index.html")

    encoded = re.search(r'src="data:audio/[^;]+;base64,([^"]+)"',
                        page.read_text(encoding="utf-8")).group(1)
    assert base64.b64decode(encoded) == excerpts[0].data


def test_source_ids_are_escaped(tmp_path):
    source = write_audio(tmp_path / "r" / "094.wav", 30.0)
    excerpts = extract_excerpts("r/094", source, 30.0, tmp_path / "clips",
                                count=1, clip_seconds=2.0)
    page = build_page([("<script>alert(1)</script>", "note", 30.0, excerpts)],
                      tmp_path / "index.html")
    assert "<script>alert" not in page.read_text(encoding="utf-8")


def test_cli_audits_then_excerpts_what_it_flagged(tmp_path):
    for reciter, seconds in (("r1", 20.0), ("r2", 21.0), ("r3", 19.0), ("odd", 200.0)):
        write_audio(tmp_path / "audio" / reciter / "094.wav", seconds)
    out = tmp_path / "listen"

    assert main(["listen", str(tmp_path / "audio"), "-o", str(out), "--seconds", "3"]) == 0
    assert (out / "index.html").exists()

    text = (out / "index.html").read_text(encoding="utf-8")
    assert "odd/094" in text
    assert "r1/094" not in text, "only flagged files should appear"


def test_cli_accepts_an_explicit_id_list(tmp_path):
    write_audio(tmp_path / "audio" / "r1" / "094.wav", 30.0)
    write_audio(tmp_path / "audio" / "r2" / "094.wav", 30.0)
    ids = tmp_path / "ids.txt"
    ids.write_text("r2/094\n", encoding="utf-8")
    out = tmp_path / "listen"

    assert main(["listen", str(tmp_path / "audio"), "-o", str(out),
                 "--ids", str(ids), "--seconds", "3"]) == 0
    text = (out / "index.html").read_text(encoding="utf-8")
    assert "r2/094" in text and "r1/094" not in text


def test_cli_reports_when_nothing_is_suspicious(tmp_path, capsys):
    for reciter in ("r1", "r2", "r3"):
        write_audio(tmp_path / "audio" / reciter / "094.wav", 20.0)

    assert main(["listen", str(tmp_path / "audio"), "-o", str(tmp_path / "out")]) == 0
    assert "Nothing suspicious" in capsys.readouterr().out
