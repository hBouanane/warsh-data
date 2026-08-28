"""Choosing segments to listen to, and putting them in one page."""

from __future__ import annotations

import pytest

from warshdata import review


def row(segment_id, duration, distance=0.01, **extra):
    base = {
        "segment_id": segment_id,
        "source_id": "rec/001",
        "duration_seconds": duration,
        "align_distance": distance,
        "asr": "قل هو الله احد",
        "label": "قُلْ هُوَ اَ۬للَّهُ أَحَدٌ",
        "align_ok": True,
    }
    base.update(extra)
    return base


@pytest.fixture
def rows():
    return [row(f"s{i}", float(i), 1.0 / (i + 1)) for i in range(1, 9)]


def test_longest_and_shortest_are_opposite_ends(rows):
    assert [r["duration_seconds"] for r in review.pick(rows, "duration", 3)] == [8, 7, 6]
    assert [r["duration_seconds"] for r in
            review.pick(rows, "duration", 3, largest=False)] == [1, 2, 3]


def test_unaligned_rows_are_not_the_worst_aligned(rows):
    """A segment with no distance was never aligned -- a category of its own.
    Sorted in as zero it would either top the worst list or fill the shortest
    one, and either way it displaces a row that actually needs hearing.
    """
    rows.append(row("never-aligned", 5.0, distance=None, asr=None, label=None))
    worst = review.pick(rows, "distance", 5)
    assert all(r["align_distance"] is not None for r in worst)
    assert "never-aligned" not in [r["segment_id"] for r in worst]

    # It still has a duration, so it remains eligible on that axis.
    by_duration = review.pick(rows, "duration", 9, largest=False)
    assert "never-aligned" in [r["segment_id"] for r in by_duration]


def test_asking_for_more_than_exists_is_not_an_error(rows):
    assert len(review.pick(rows, "duration", 500)) == len(rows)
    assert review.pick([], "duration", 10) == []


def test_page_is_self_contained_and_shows_both_texts(tmp_path):
    chosen = [row("s1", 31.0, 0.62, ayah_start=2255, ayah_end=2256)]
    page = review.build_page(
        [("Longest segments", "why these", chosen)],
        {"s1": b"fLaCpretend"},
        tmp_path / "review.html",
    )
    text = page.read_text(encoding="utf-8")

    # Embedded, not linked: the page is opened from a download, where a
    # relative path to a clip file is exactly what breaks.
    assert "data:audio/flac;base64," in text
    assert "<audio" in text and "src=\"clips/" not in text

    assert 'dir="rtl"' in text, "Arabic must render right to left"
    assert "قل هو الله احد" in text, "what the recogniser heard"
    assert "أَحَدٌ" in text, "the label, which is what training would see"
    assert "distance 0.620" in text
    assert "ayah 2255-2256" in text


def test_a_segment_without_audio_still_gets_a_block(tmp_path):
    """A shard that fails to read must not silently drop the row: the point is
    to see every extreme, and a missing clip is itself worth knowing about."""
    page = review.build_page(
        [("Worst aligned", "why", [row("s1", 3.0, 0.9)])], {},
        tmp_path / "review.html")
    text = page.read_text(encoding="utf-8")
    assert "audio unavailable" in text
    assert "s1" in text


def test_empty_sections_are_omitted(tmp_path):
    page = review.build_page(
        [("Longest segments", "why", []),
         ("Worst aligned", "why", [row("s1", 3.0, 0.9)])], {},
        tmp_path / "review.html")
    text = page.read_text(encoding="utf-8")
    assert "Longest segments" not in text
    assert "Worst aligned" in text


def test_missing_label_renders_as_none_not_a_crash(tmp_path):
    page = review.build_page(
        [("Longest", "why", [row("s1", 44.0, 0.5, asr=None, label=None)])], {},
        tmp_path / "review.html")
    assert ">none<" in page.read_text(encoding="utf-8")


def test_clips_are_fetched_one_shard_per_source(monkeypatch):
    """Ten segments of one recording is one shard read, not ten. Shards are
    ~16 MB because the audio is in them, so this is the difference between a
    review page costing 16 MB and costing 160 MB."""
    reads = []

    class FakeTable:
        def __init__(self, ids):
            self.ids = ids

        def column(self, name):
            if name == "segment_id":
                return FakeCol(self.ids)
            return FakeCol([{"bytes": f"audio-{i}".encode()} for i in self.ids])

    class FakeCol:
        def __init__(self, values):
            self.values = values

        def to_pylist(self):
            return self.values

    class FakeParquet:
        def __init__(self, ids):
            self.ids = ids

        def read(self, columns):
            return FakeTable(self.ids)

    def fake_open(fs, path):
        reads.append(path)
        return FakeParquet(["a1", "a2", "a3"] if "001" in path else ["b1"])

    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda **kw: object())
    monkeypatch.setattr("warshdata.hub._open_projected", fake_open)

    wanted = [
        {"segment_id": "a1", "source_id": "rec/001"},
        {"segment_id": "a2", "source_id": "rec/001"},
        {"segment_id": "a3", "source_id": "rec/001"},
        {"segment_id": "b1", "source_id": "rec/002"},
    ]
    clips = review.fetch_clips("some/repo", wanted)

    assert len(reads) == 2, f"one read per source, got {reads}"
    assert set(clips) == {"a1", "a2", "a3", "b1"}
    assert clips["a1"] == b"audio-a1"


def test_an_unreadable_shard_does_not_sink_the_run(monkeypatch, capsys):
    def boom(fs, path):
        raise OSError("gone")

    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda **kw: object())
    monkeypatch.setattr("warshdata.hub._open_projected", boom)

    clips = review.fetch_clips("some/repo", [{"segment_id": "a", "source_id": "r/1"}])
    assert clips == {}
    assert "could not read" in capsys.readouterr().out
