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


class RecordingStore:
    """Stands in for HubStore and records every published source."""

    instances: list["RecordingStore"] = []
    fail_on: set[str] = set()

    def __init__(self, repo_id=None, work_dir=None, private=False, upload_raw=True):
        self.repo_id = repo_id
        self.upload_raw = upload_raw
        self.published: list[str] = []
        self.commits: list[list[str]] = []
        self.rows_written = 0
        self.sources_written = 0
        self.params = None
        self.existing: set[str] = set()
        RecordingStore.instances.append(self)

    def done_sources(self):
        return set(self.existing)

    def upload_params(self, params):
        self.params = params

    def write_source(self, reciter_slug, source_stem, records, waves, raw_path=None):
        source_id = f"{reciter_slug}/{source_stem}"
        if source_id in RecordingStore.fail_on:
            raise OSError("upload refused")
        assert len(records) == len(waves), "one waveform per record"
        paths = [f"data/{reciter_slug}/{source_stem}.parquet"]
        if self.upload_raw and raw_path is not None:
            paths.append(f"raw/{reciter_slug}/{Path(raw_path).name}")
        self.commits.append(paths)
        self.published.append(source_id)
        self.rows_written += len(records)
        self.sources_written += 1
        return paths[0]


@pytest.fixture
def fake_hub(monkeypatch):
    import warshdata.hub as hub

    RecordingStore.instances = []
    RecordingStore.fail_on = set()
    monkeypatch.setattr(hub, "HubStore", RecordingStore)
    return RecordingStore


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


def test_each_source_is_one_commit(fake_segment_module, fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])

    store = fake_hub.instances[0]
    assert store.published == [
        "ibrahim-aldosari/001", "ibrahim-aldosari/002", "ibrahim-aldosari/003",
    ]
    # Parquet and mp3 travel together, one commit per source.
    assert all(len(c) == 2 for c in store.commits)
    assert store.rows_written == 9


def test_resume_uses_the_published_files(fake_segment_module, fake_hub, tmp_path):
    """Resume asks the repo what exists; nothing else can contradict it."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    class Preloaded(RecordingStore):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.existing = {"ibrahim-aldosari/001", "ibrahim-aldosari/002"}

    import warshdata.hub as hub

    hub.HubStore = Preloaded
    try:
        run(["segment", str(audio), "-o", str(out), "--no-clips",
             "--push-to", "u/r", "--resume"])
    finally:
        hub.HubStore = RecordingStore

    store = RecordingStore.instances[-1]
    assert store.published == ["ibrahim-aldosari/003"]


def test_a_failed_upload_leaves_the_source_unpublished(fake_segment_module, fake_hub, tmp_path):
    """The failed source is simply absent, so a later resume retries it --
    there is no partial state to detect or clean up."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3))
    out = tmp_path / "out"

    fake_hub.fail_on = {"ibrahim-aldosari/002"}
    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])

    store = fake_hub.instances[0]
    assert store.published == ["ibrahim-aldosari/001", "ibrahim-aldosari/003"]
    # The manifest must not claim the source that never uploaded.
    ids = {r["source_id"] for r in manifest.read(out / "segments.jsonl")}
    assert ids == {"ibrahim-aldosari/001", "ibrahim-aldosari/003"}


def test_manifest_never_records_a_source_that_failed_to_publish(fake_segment_module,
                                                               fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1,))
    out = tmp_path / "out"
    fake_hub.fail_on = {"ibrahim-aldosari/001"}

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])
    assert list(manifest.read(out / "segments.jsonl")) == []


def test_rerunning_the_same_source_republishes_the_same_path(fake_segment_module,
                                                             fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1,))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])
    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])

    first, second = fake_hub.instances
    assert first.commits[0][0] == second.commits[0][0] == "data/ibrahim-aldosari/001.parquet"


def test_params_are_published(fake_segment_module, fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1,))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--push-to", "u/r"])
    store = fake_hub.instances[0]
    assert store.params["model_id"] == "obadx/recitation-segmenter-v2"
    assert store.params["resolved_dtype"] == "float16"


def test_dry_run_never_touches_the_hub(fake_segment_module, fake_hub, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--push-to", "u/r", "--dry-run"])
    assert fake_hub.instances == []


class FakeTranscriber:
    """Stands in for the NeMo model: returns the reference text of each clip."""

    made: list["FakeTranscriber"] = []
    texts: list[str] = []

    def __init__(self, model_id=None, checkpoint=None, device=None, batch_size=None):
        self.model_id = model_id
        self.calls = 0
        FakeTranscriber.made.append(self)

    def transcribe(self, waves):
        self.calls += 1
        return [FakeTranscriber.texts[i % len(FakeTranscriber.texts)]
                for i in range(len(waves))]


@pytest.fixture
def fake_asr(monkeypatch):
    import warshdata.asr as asr

    FakeTranscriber.made = []
    monkeypatch.setattr(asr, "Transcriber", FakeTranscriber)
    return FakeTranscriber


def test_asr_fills_the_transcript_column(fake_segment_module, fake_asr, tmp_path):
    fake_asr.texts = ["الحمد لله رب العلمين"]
    audio = make_audio_tree(tmp_path / "audio", surahs=(1,))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--asr"])

    records = list(manifest.read(out / "segments.jsonl"))
    assert records, "nothing written"
    assert all(r["asr"] == "الحمد لله رب العلمين" for r in records)
    assert all(r["surah_number"] == 1 for r in records)
    assert fake_asr.made[0].model_id == "mohammed/fastconformer-quran-ar"


def test_align_without_asr_is_refused(fake_segment_module, tmp_path):
    audio = make_audio_tree(tmp_path / "audio", surahs=(1,))
    assert run(["segment", str(audio), "-o", str(tmp_path / "out"),
                "--no-clips", "--align"]) == 1


def test_align_labels_from_the_reference_not_the_asr(fake_segment_module, fake_asr, tmp_path):
    """The point of the whole pipeline: a recognition error must not reach the
    label. The transcript here is misspelled; the label must not be."""
    pytest.importorskip("rapidfuzz")
    pytest.importorskip("warshlab")
    from warshdata import quran

    try:
        surah = quran.load()[112]
    except FileNotFoundError:
        pytest.skip("Warsh text not downloaded")

    # A mangled version of the opening words.
    fake_asr.texts = [surah.rasm(0, 4).replace("قل", "قد")]
    audio = make_audio_tree(tmp_path / "audio", surahs=(112,))
    out = tmp_path / "out"

    run(["segment", str(audio), "-o", str(out), "--no-clips", "--asr", "--align"])

    records = list(manifest.read(out / "segments.jsonl"))
    labelled = [r for r in records if r["label"]]
    assert labelled, "alignment produced no labels"
    for record in labelled:
        assert "قد" not in record["label"], "the ASR's error leaked into the label"
        assert record["ayah_start"] is not None
        assert record["align_distance"] is not None


@pytest.mark.parametrize("message,fatal", [
    ("CUDA error: an illegal memory access was encountered", True),
    ("CUDA error: device-side assert triggered", True),
    ("CUDA out of memory. Tried to allocate 2.00 GiB", False),
    ("some unrelated failure", False),
])
def test_only_context_killing_cuda_errors_stop_the_run(message, fatal):
    from warshdata.cli import _is_cuda_fatal

    assert _is_cuda_fatal(RuntimeError(message)) is fatal


def test_a_dead_cuda_context_stops_the_run(fake_segment_module, tmp_path, monkeypatch):
    """An illegal memory access leaves every later call failing too, so marching
    on burns the rest of the corpus reporting failures it cannot avoid."""
    audio = make_audio_tree(tmp_path / "audio", surahs=(1, 2, 3, 4, 5))
    out = tmp_path / "out"

    mod = sys.modules["warshdata.segment"]
    original = mod.Segmenter.segment
    calls = []

    def poisoned(self, source, clips_dir=None):
        calls.append(source.source_id)
        if source.path.stem == "002":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return original(self, source, clips_dir=clips_dir)

    monkeypatch.setattr(mod.Segmenter, "segment", poisoned)
    run(["segment", str(audio), "-o", str(out), "--no-clips"])

    assert calls == ["ibrahim-aldosari/001", "ibrahim-aldosari/002"], calls
