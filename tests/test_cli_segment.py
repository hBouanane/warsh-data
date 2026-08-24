"""End-to-end tests for ``warsh-data segment``.

``warshdata.segment`` imports torch and recitations_segmenter at module scope, so
a fake module is installed into ``sys.modules`` before the CLI imports it.  That
keeps these tests runnable anywhere, which matters because the bug they exist to
catch -- the end-of-run flush going missing -- is in the CLI's control flow, not
in the model.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from warshdata import manifest
from warshdata.manifest import SegmentRecord


class FakeWave:
    """Just enough of a torch tensor for the slicing the CLI does."""

    def __init__(self, samples: np.ndarray):
        self._s = samples

    def __getitem__(self, item):
        return FakeWave(self._s[item])

    def numpy(self):
        return self._s

    @property
    def shape(self):
        return self._s.shape


@dataclass
class FakeParams:
    min_silence_duration_ms: int = 200
    min_speech_duration_ms: int = 400
    pad_duration_ms: int = 40
    max_duration_ms: int = 19995
    batch_size: int = 8
    device: str = "cuda"
    dtype: str = "auto"


@pytest.fixture
def fake_segment_module(monkeypatch):
    """Install a stand-in for warshdata.segment; returns the call log."""
    calls: list[str] = []

    class FakeSegmenter:
        dtype = "torch.float16"

        def __init__(self, params, *a, **k):
            self.params = params

        def segment(self, source, clips_dir=None):
            calls.append(source.source_id)
            records = [
                SegmentRecord(
                    segment_id=f"{source.reciter_slug}__{source.path.stem}__{i:04d}",
                    reciter_slug=source.reciter_slug,
                    source_id=source.source_id,
                    source_path=str(source.path),
                    index=i,
                    start_sample=i * 16000,
                    end_sample=(i + 1) * 16000,
                    start_seconds=float(i),
                    end_seconds=float(i + 1),
                    duration_seconds=1.0,
                    sample_rate=16000,
                    is_last_of_source=(i == 2),
                )
                for i in range(3)
            ]
            return records, FakeWave(np.zeros(3 * 16000, dtype=np.float32))

    mod = types.ModuleType("warshdata.segment")
    mod.MODEL_ID = "obadx/recitation-segmenter-v2"
    mod.SegmentParams = FakeParams
    mod.Segmenter = FakeSegmenter
    mod.SAMPLE_RATE = 16000
    monkeypatch.setitem(sys.modules, "warshdata.segment", mod)
    return calls


class RecordingWriter:
    """Stands in for HubWriter and logs the order of operations."""

    instances: list["RecordingWriter"] = []

    def __init__(self, repo_id=None, work_dir=None, shard_bytes=None, private=False, upload_raw=True):
        self.repo_id = repo_id
        self.upload_raw = upload_raw
        self.added: list[str] = []
        self.flushed: list[str] = []
        self.queued: list[str] = []
        self.events: list[str] = []
        self.rows_written = 0
        self.shards_written = 0
        RecordingWriter.instances.append(self)

    def repo_files(self):
        return []

    def add(self, record, wave):
        self.added.append(record["segment_id"])
        self.rows_written += 1

    def queue_source(self, path, slug):
        self.queued.append(f"raw/{slug}/{Path(path).name}")

    #: rows buffered before a flush is reported; None means never flush early
    flush_after: int | None = None

    def flush(self):
        self.events.append("flush")
        self.shards_written += 1
        self.flushed = list(self.added)
        return "shard-00000.parquet"

    def maybe_flush(self):
        if self.flush_after is not None and len(self.added) - len(self.flushed) >= self.flush_after:
            return self.flush()
        return None

    def flush_sources(self):
        self.events.append("flush_sources")
        n = len(self.queued)
        self.queued = []
        return n

    def push_manifest(self, path):
        self.events.append("push_manifest")


@pytest.fixture
def fake_hub(monkeypatch):
    import warshdata.hub as hub

    RecordingWriter.instances = []
    monkeypatch.setattr(hub, "HubWriter", RecordingWriter)
    monkeypatch.setattr(hub, "done_sources_on_hub", lambda *a, **k: set())
    return RecordingWriter


def make_audio_tree(root: Path, reciters=("ibrahim-aldosari",), surahs=(1, 2, 3)):
    for r in reciters:
        for s in surahs:
            p = root / r / f"{s:03d}.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake mp3")
    return root


def run(argv):
    from warshdata.cli import main

    return main(argv)


def test_segment_writes_manifest_per_source(fake_segment_module, tmp_path):
    audio = make_audio_tree(tmp_path / "audio")
    out = tmp_path / "out"

    assert run(["segment", str(audio), "-o", str(out), "--no-clips"]) == 0

    records = list(manifest.read(out / "segments.jsonl"))
    assert len(records) == 9  # 3 sources x 3 segments
    assert {r["reciter_slug"] for r in records} == {"ibrahim-aldosari"}
    assert (out / "segment_params.json").exists()


def test_reciter_slug_comes_from_the_parent_directory(fake_segment_module, tmp_path):
    audio = tmp_path / "audio"
    (audio / "rachid-belalya").mkdir(parents=True)
    (audio / "rachid-belalya" / "087.mp3").write_bytes(b"x")
    (audio / "loose.mp3").write_bytes(b"x")
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips"])
    slugs = {r["reciter_slug"] for r in manifest.read(out / "segments.jsonl")}
    # A file at the root is attributed to `unknown` rather than dropped.
    assert slugs == {"rachid-belalya", "unknown"}


def test_resume_skips_sources_already_in_the_manifest(fake_segment_module, tmp_path):
    audio = make_audio_tree(tmp_path / "audio")
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--limit", "1"])
    first = len(list(manifest.read(out / "segments.jsonl")))

    calls_before = len(fake_segment_module)
    run(["segment", str(audio), "-o", str(out), "--no-clips", "--resume"])
    assert len(fake_segment_module) == calls_before + 2  # only the 2 remaining
    assert len(list(manifest.read(out / "segments.jsonl"))) == first + 6


def test_dry_run_writes_nothing(fake_segment_module, tmp_path):
    audio = make_audio_tree(tmp_path / "audio")
    out = tmp_path / "out"

    assert run(["segment", str(audio), "-o", str(out), "--dry-run"]) == 0
    assert not (out / "segments.jsonl").exists()
    assert fake_segment_module == []


def test_missing_input_is_an_error(tmp_path):
    assert run(["segment", str(tmp_path / "nope"), "-o", str(tmp_path / "out")]) == 1


def test_final_flush_happens(fake_segment_module, fake_hub, tmp_path):
    """Regression: the end-of-run flush was missing, so the last partial shard,
    the trailing raw files and the closing manifest push were all dropped."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips",
         "--push-to", "u/r", "--push-every", "10"])

    writer = fake_hub.instances[0]
    # 3 sources, push-every 10 -> nothing would be pushed without the final flush
    assert writer.events[-3:] == ["flush", "flush_sources", "push_manifest"]
    assert writer.rows_written == 9
    assert writer.queued == []


def test_push_every_throttles_intermediate_commits(fake_segment_module, fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=tuple(range(1, 7)))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips",
         "--push-to", "u/r", "--push-every", "2"])

    writer = fake_hub.instances[0]
    # No shard ever filled, so the manifest is pushed only by the final flush.
    assert writer.events.count("push_manifest") == 1
    assert writer.events.count("flush") == 1
    # Raw files still commit on the --push-every schedule.
    assert writer.events.count("flush_sources") >= 3


def test_every_segment_reaches_the_writer(fake_segment_module, fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])

    writer = fake_hub.instances[0]
    assert len(writer.added) == 6
    assert len(set(writer.added)) == 6, "segment ids must be unique across sources"


def test_resume_queues_raw_files_missing_from_the_hub(fake_segment_module, fake_hub, tmp_path):
    """A session that died between batch commits leaves sources in the manifest
    with no raw file; they are excluded from `pending`, so they must be queued
    explicitly or they never upload at all."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips"])  # local only
    run(["segment", str(audio), "-o", str(out), "--no-clips",
         "--push-to", "u/r", "--resume"])

    writer = fake_hub.instances[0]
    # All three were already segmented, so none are re-processed, but their raw
    # files are absent from the (empty) repo listing and must still be sent.
    assert writer.events.count("flush_sources") >= 1
    assert writer.rows_written == 0


def test_failure_on_one_source_does_not_end_the_run(fake_segment_module, tmp_path, monkeypatch):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    mod = sys.modules["warshdata.segment"]
    original = mod.Segmenter.segment

    def flaky(self, source, clips_dir=None):
        if source.path.stem == "002":
            raise RuntimeError("corrupt file")
        return original(self, source, clips_dir=clips_dir)

    monkeypatch.setattr(mod.Segmenter, "segment", flaky)

    assert run(["segment", str(audio), "-o", str(out), "--no-clips"]) == 0
    records = list(manifest.read(out / "segments.jsonl"))
    assert {r["source_id"].split("/")[-1] for r in records} == {"001", "003"}


def test_manifest_is_never_pushed_ahead_of_the_shards(fake_segment_module, fake_hub, tmp_path):
    """The invariant resumption rests on.

    ``done_sources_on_hub`` reads the uploaded manifest and skips those sources
    for good.  If the manifest is ever uploaded while segments are still sitting
    in the write buffer, those sources are marked done, skipped on the next run,
    and their audio is uploaded by nobody -- rows naming audio that exists
    nowhere.  So every manifest push must be immediately preceded by a flush.
    """
    audio = make_audio_tree(tmp_path / "audio", surahs=tuple(range(1, 9)))
    out = tmp_path / "out"

    fake_hub.flush_after = 6          # a shard fills every 2 sources (3 segs each)
    try:
        run(["segment", str(audio), "-o", str(out), "--no-clips",
             "--push-to", "u/r", "--push-every", "1"])
    finally:
        fake_hub.flush_after = None

    writer = fake_hub.instances[0]
    events = writer.events
    assert "push_manifest" in events
    for i, event in enumerate(events):
        if event == "push_manifest":
            assert "flush" in events[max(0, i - 2):i], (
                f"push_manifest at {i} without a preceding flush: {events}"
            )
    # And nothing is left buffered at the end.
    assert writer.flushed == writer.added


def test_raw_commits_do_not_drag_the_manifest_with_them(fake_segment_module, fake_hub, tmp_path):
    """--push-every governs raw provenance files only; it must not push the
    manifest, which is tied to shard flushes."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3, 4))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips",
         "--push-to", "u/r", "--push-every", "1"])

    writer = fake_hub.instances[0]
    # 4 sources -> 4 raw commits, but no shard ever filled, so the only manifest
    # push is the final one.
    assert writer.events.count("flush_sources") >= 4
    assert writer.events.count("push_manifest") == 1


def test_sources_file_overrides_resume(fake_segment_module, tmp_path):
    """Sources listed for repair are re-segmented even though the manifest
    already names them -- that is precisely why they need repairing."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips"])
    calls_before = len(fake_segment_module)

    listing = tmp_path / "missing.txt"
    listing.write_text("ibrahim-aldosari/002\n", encoding="utf-8")

    run(["segment", str(audio), "-o", str(out), "--no-clips",
         "--sources-file", str(listing), "--resume"])

    assert fake_segment_module[calls_before:] == ["ibrahim-aldosari/002"]


def test_sources_file_warns_about_unknown_ids(fake_segment_module, tmp_path, capsys):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1,))
    out = tmp_path / "out"
    listing = tmp_path / "missing.txt"
    listing.write_text("ibrahim-aldosari/001\nnobody/999\n", encoding="utf-8")

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--sources-file", str(listing)])
    err = capsys.readouterr().err
    assert "not found" in err and "nobody/999" in err
