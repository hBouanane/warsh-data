"""Purging a source's rows from the shards.

Stale rows left behind by a crash cannot be de-duplicated by a streaming
consumer, so the corpus itself has to be corrected. These tests run against real
parquet files with real audio bytes, through a stubbed Hub filesystem.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import soundfile as sf
from conftest import make_record, make_wave

from warshdata.hub import HubWriter, purge_sources


@pytest.fixture
def shard_repo(fake_api, tmp_path, monkeypatch):
    """Write three real shards, then serve them through a fake HfFileSystem.

    Uploads are redirected into ``shards/`` so the writer's own cleanup of its
    work directory does not remove them -- the same reason the real thing can
    delete its local copy the moment the Hub has it.
    """
    shards = tmp_path / "shards"
    shards.mkdir()

    class FakeFs:
        def __init__(self, *a, **k):
            pass

        def glob(self, pattern):
            return [f"datasets/u/r/data/{p.name}" for p in sorted(shards.glob("*.parquet"))]

        def open(self, path, mode="rb"):
            return open(shards / Path(path).name, mode)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFs)

    def upload_file(path_or_fileobj=None, path_in_repo=None, **kwargs):
        if str(path_in_repo).endswith(".parquet"):
            (shards / Path(path_in_repo).name).write_bytes(Path(path_or_fileobj).read_bytes())
        fake_api.uploads.append(path_in_repo)
        if path_in_repo not in fake_api.files:
            fake_api.files.append(path_in_repo)

    def delete_file(path_in_repo=None, **kwargs):
        (shards / Path(path_in_repo).name).unlink()
        fake_api.deleted.append(path_in_repo)

    fake_api.deleted = []
    fake_api.upload_file = upload_file
    fake_api.delete_file = delete_file

    for shard_no, source in enumerate(["r/001", "r/002", "r/003"]):
        writer = HubWriter(repo_id="u/r", work_dir=tmp_path / "work", shard_bytes=10**9)
        writer.shard_index = shard_no
        for i in range(4):
            rec = make_record(i, segment_id=f"{source.replace('/', '__')}__{i:04d}",
                              source_id=source)
            writer.add(rec, make_wave(0.25, seed=i))
        writer.flush()

    assert len(list(shards.glob("*.parquet"))) == 3
    # Setup traffic is not what these tests are about.
    fake_api.uploads.clear()
    return shards


def sources_in(shards: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in sorted(shards.glob("*.parquet")):
        for src in pq.read_table(p).column("source_id").to_pylist():
            counts[src] = counts.get(src, 0) + 1
    return counts


def test_purge_removes_only_the_named_source(shard_repo, tmp_path):
    assert sources_in(shard_repo) == {"r/001": 4, "r/002": 4, "r/003": 4}

    report = purge_sources("u/r", {"r/002"}, work_dir=tmp_path / "work")

    assert report["rows_removed"] == 4
    assert sources_in(shard_repo) == {"r/001": 4, "r/003": 4}


def test_shard_left_empty_is_deleted(shard_repo, tmp_path, fake_api):
    purge_sources("u/r", {"r/002"}, work_dir=tmp_path / "work")
    # Each source occupied a whole shard here, so its shard should be gone.
    assert len(list(shard_repo.glob("*.parquet"))) == 2
    assert getattr(fake_api, "deleted", [])


def test_untouched_shards_are_not_rewritten(shard_repo, tmp_path, fake_api):
    report = purge_sources("u/r", {"r/002"}, work_dir=tmp_path / "work")
    assert report["scanned"] == 3
    # Only the shard holding r/002 is touched; the other two are never uploaded.
    assert len(fake_api.uploads) + len(getattr(fake_api, "deleted", [])) == 1


def test_dry_run_changes_nothing(shard_repo, tmp_path, fake_api):
    before = sources_in(shard_repo)
    report = purge_sources("u/r", {"r/002"}, work_dir=tmp_path / "work", dry_run=True)
    assert report["rows_removed"] == 4
    assert sources_in(shard_repo) == before
    assert fake_api.uploads == []


def test_purging_an_absent_source_is_a_noop(shard_repo, tmp_path):
    before = sources_in(shard_repo)
    report = purge_sources("u/r", {"nobody/999"}, work_dir=tmp_path / "work")
    assert report["rows_removed"] == 0
    assert sources_in(shard_repo) == before


def test_surviving_audio_still_decodes(shard_repo, tmp_path):
    """A rewrite must not re-encode or corrupt the audio it keeps."""
    mixed = shard_repo / "shard-00000.parquet"
    table = pq.read_table(mixed)
    # Put two sources in one shard so the rewrite has rows to keep.
    import pyarrow as pa

    ids = table.column("source_id").to_pylist()
    ids[:2] = ["r/999", "r/999"]
    table = table.set_column(table.schema.get_field_index("source_id"), "source_id", pa.array(ids))
    pq.write_table(table, mixed)

    purge_sources("u/r", {"r/999"}, work_dir=tmp_path / "work")

    kept = pq.read_table(mixed)
    assert kept.num_rows == 2
    row = kept.to_pylist()[0]
    data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    assert sr == 16000 and len(data) == 4000
