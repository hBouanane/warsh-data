"""HubWriter: sharding, resumption, and the invariant that local disk stays small."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import make_record, make_wave

from warshdata.hub import HubWriter, done_sources_on_hub, encode_flac


def test_encode_flac_round_trips():
    wave = make_wave(1.0)
    data, sr = sf.read(io.BytesIO(encode_flac(wave)), dtype="float32")
    assert sr == 16000
    assert data.shape == wave.shape
    # PCM_16 quantisation is the only loss.
    assert np.abs(data - wave).max() < 2e-4


def test_shard_flushes_when_full_and_deletes_local_copy(fake_api, tmp_path):
    writer = HubWriter(repo_id="u/r", work_dir=tmp_path / "_shards", shard_bytes=60_000)
    for i in range(12):
        writer.add(make_record(i), make_wave(1.0, seed=i))
    writer.flush()

    shards = [f for f in fake_api.uploads if f.startswith("data/")]
    assert len(shards) >= 2, "buffer should have flushed more than once"
    assert shards == sorted(shards), "shards must be uploaded in order"
    assert writer.rows_written == 12
    # The point of streaming: nothing accumulates on disk.
    assert list((tmp_path / "_shards").glob("*.parquet")) == []


def test_flush_is_a_noop_when_empty(fake_api, tmp_path):
    writer = HubWriter(repo_id="u/r", work_dir=tmp_path, shard_bytes=10**9)
    assert writer.flush() is None
    assert [f for f in fake_api.uploads if f.startswith("data/")] == []


def test_shard_numbering_continues_across_sessions(fake_api, tmp_path):
    first = HubWriter(repo_id="u/r", work_dir=tmp_path, shard_bytes=10**9)
    first.add(make_record(0), make_wave(0.5))
    first.flush()

    second = HubWriter(repo_id="u/r", work_dir=tmp_path, shard_bytes=10**9)
    # Restarting at 0 would overwrite the first session's shard.
    assert second.shard_index == first.shard_index


def test_readme_written_once(fake_api, tmp_path):
    HubWriter(repo_id="u/r", work_dir=tmp_path)
    assert fake_api.uploads.count("README.md") == 1
    HubWriter(repo_id="u/r", work_dir=tmp_path)
    assert fake_api.uploads.count("README.md") == 1


def test_sources_are_committed_in_one_batch(fake_api, tmp_path):
    writer = HubWriter(repo_id="u/r", work_dir=tmp_path)
    for i in range(5):
        path = tmp_path / f"{i:03d}.mp3"
        path.write_bytes(b"x")
        writer.queue_source(path, "ibrahim-aldosari")

    assert writer.flush_sources() == 5
    # One commit, not five: a full pass would otherwise be thousands.
    assert len(fake_api.commits) == 1
    assert fake_api.commits[0] == [f"raw/ibrahim-aldosari/{i:03d}.mp3" for i in range(5)]


def test_flush_sources_is_a_noop_when_nothing_queued(fake_api, tmp_path):
    writer = HubWriter(repo_id="u/r", work_dir=tmp_path)
    assert writer.flush_sources() == 0
    assert fake_api.commits == []


def test_no_raw_suppresses_source_upload(fake_api, tmp_path):
    writer = HubWriter(repo_id="u/r", work_dir=tmp_path, upload_raw=False)
    path = tmp_path / "001.mp3"
    path.write_bytes(b"x")
    writer.queue_source(path, "r")
    assert writer.flush_sources() == 0


def test_clip_filename_carries_timestamps_but_id_does_not(fake_api, tmp_path, keep_shards):
    import pyarrow.parquet as pq

    writer = HubWriter(repo_id="u/r", work_dir=tmp_path, shard_bytes=10**9)
    writer.add(
        make_record(3, start_sample=198400, end_sample=214400,
                    start_seconds=12.4, end_seconds=13.4),
        make_wave(1.0),
    )
    writer.flush()

    table = pq.read_table(next(tmp_path.glob("*.parquet")))
    row = table.to_pylist()[0]
    assert row["segment_id"] == "ibrahim-aldosari__087__0003"
    assert row["audio"]["path"] == "ibrahim-aldosari__087__0003__12400-13400ms.flac"

    data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    assert (sr, len(data)) == (16000, 16000)


def test_done_sources_on_hub_is_empty_when_repo_missing(monkeypatch):
    import huggingface_hub

    def boom(*args, **kwargs):
        raise OSError("no such repo")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert done_sources_on_hub("u/does-not-exist") == set()


def test_done_sources_on_hub_reads_manifest(monkeypatch, tmp_path):
    import json

    import huggingface_hub

    manifest = tmp_path / "segments.jsonl"
    manifest.write_text(
        json.dumps(make_record(0)) + "\n"
        + json.dumps(make_record(1)) + "\n"
        + json.dumps(make_record(2, source_id="other/002")) + "\n"
        + "{torn line\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(manifest))
    assert done_sources_on_hub("u/r") == {"ibrahim-aldosari/087", "other/002"}
