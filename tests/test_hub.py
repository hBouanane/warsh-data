"""HubStore: one source, one file, one commit.

The properties asserted here are exactly the ones the old buffered design kept
getting wrong, so they are checked directly rather than inferred.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from conftest import make_record, make_wave

from warshdata import hub
from warshdata.hub import (
    HubStore,
    build_parquet,
    encode_flac,
    shard_path_for,
    source_id_from_shard_path,
)


def records_and_waves(n=3, source_id="ibrahim-aldosari/087"):
    reciter, stem = source_id.split("/")
    records = [
        make_record(i, segment_id=f"{reciter}__{stem}__{i:04d}", source_id=source_id,
                    reciter_slug=reciter)
        for i in range(n)
    ]
    return records, [make_wave(0.25, seed=i) for i in range(n)]


def test_encode_flac_round_trips():
    wave = make_wave(1.0)
    data, sr = sf.read(io.BytesIO(encode_flac(wave)), dtype="float32")
    assert sr == 16000
    assert np.abs(data - wave).max() < 2e-4       # PCM_16 quantisation only


@pytest.mark.parametrize("reciter,stem,path", [
    ("ibrahim-aldosari", "087", "data/ibrahim-aldosari/087.parquet"),
    ("rachid-belalya", "002", "data/rachid-belalya/002.parquet"),
])
def test_shard_path_round_trips_to_the_source_id(reciter, stem, path):
    assert shard_path_for(reciter, stem) == path
    assert source_id_from_shard_path(path) == f"{reciter}/{stem}"


@pytest.mark.parametrize("path", [
    "README.md",
    "raw/ibrahim-aldosari/087.mp3",
    "segment_params.json",
    "data/loose.parquet",
    "data/a/b/c.parquet",
])
def test_non_shard_paths_are_not_mistaken_for_sources(path):
    assert source_id_from_shard_path(path) is None


def test_write_source_is_a_single_commit(fake_api, tmp_path):
    store = HubStore(repo_id="u/r", work_dir=tmp_path / "work")
    raw = tmp_path / "087.mp3"
    raw.write_bytes(b"fake mp3")
    records, waves = records_and_waves()

    store.write_source("ibrahim-aldosari", "087", records, waves, raw_path=raw)

    # Parquet and mp3 together in one commit: never one without the other.
    assert len(fake_api.commits) == 1
    assert fake_api.commits[0] == [
        "data/ibrahim-aldosari/087.parquet",
        "raw/ibrahim-aldosari/087.mp3",
    ]
    assert store.rows_written == 3
    assert store.sources_written == 1


def test_no_raw_omits_the_mp3(fake_api, tmp_path):
    store = HubStore(repo_id="u/r", work_dir=tmp_path / "work", upload_raw=False)
    raw = tmp_path / "087.mp3"
    raw.write_bytes(b"x")
    records, waves = records_and_waves()

    store.write_source("ibrahim-aldosari", "087", records, waves, raw_path=raw)
    assert fake_api.commits[0] == ["data/ibrahim-aldosari/087.parquet"]


def test_missing_raw_file_does_not_break_the_commit(fake_api, tmp_path):
    store = HubStore(repo_id="u/r", work_dir=tmp_path / "work")
    records, waves = records_and_waves()

    store.write_source("ibrahim-aldosari", "087", records, waves,
                       raw_path=tmp_path / "gone.mp3")
    assert fake_api.commits[0] == ["data/ibrahim-aldosari/087.parquet"]


def test_rewriting_a_source_targets_the_same_path(fake_api, tmp_path):
    """Why duplicates are impossible: the second write replaces the first."""
    store = HubStore(repo_id="u/r", work_dir=tmp_path / "work", upload_raw=False)
    records, waves = records_and_waves()

    first = store.write_source("ibrahim-aldosari", "087", records, waves)
    second = store.write_source("ibrahim-aldosari", "087", records, waves)
    assert first == second == "data/ibrahim-aldosari/087.parquet"


def test_local_staging_file_is_removed(fake_api, tmp_path):
    work = tmp_path / "work"
    store = HubStore(repo_id="u/r", work_dir=work, upload_raw=False)
    records, waves = records_and_waves()

    store.write_source("ibrahim-aldosari", "087", records, waves)
    assert list(work.glob("*.parquet")) == []


def test_staging_file_is_removed_even_when_the_commit_fails(fake_api, tmp_path):
    work = tmp_path / "work"
    store = HubStore(repo_id="u/r", work_dir=work, upload_raw=False)

    def boom(**kwargs):
        raise OSError("network down")

    store.api.create_commit = boom
    records, waves = records_and_waves()

    with pytest.raises(OSError):
        store.write_source("ibrahim-aldosari", "087", records, waves)
    assert list(work.glob("*.parquet")) == []
    assert store.sources_written == 0


def test_done_sources_reads_filenames_only(fake_api, tmp_path):
    fake_api.files = [
        "README.md",
        "segment_params.json",
        "data/ibrahim-aldosari/087.parquet",
        "data/ibrahim-aldosari/002.parquet",
        "data/rachid-belalya/114.parquet",
        "raw/ibrahim-aldosari/087.mp3",
    ]
    store = HubStore(repo_id="u/r", work_dir=tmp_path)
    assert store.done_sources() == {
        "ibrahim-aldosari/087",
        "ibrahim-aldosari/002",
        "rachid-belalya/114",
    }


def test_done_sources_is_empty_when_the_repo_cannot_be_listed(fake_api, tmp_path):
    store = HubStore(repo_id="u/r", work_dir=tmp_path)

    def boom(*args, **kwargs):
        raise OSError("offline")

    store.api.list_repo_files = boom
    assert store.done_sources() == set()


def test_readme_written_once(fake_api, tmp_path):
    HubStore(repo_id="u/r", work_dir=tmp_path)
    assert fake_api.uploads.count("README.md") == 1
    HubStore(repo_id="u/r", work_dir=tmp_path)
    assert fake_api.uploads.count("README.md") == 1


def test_parquet_contents_and_audio(tmp_path):
    records, waves = records_and_waves(n=2)
    records[0].update(start_sample=198400, end_sample=214400,
                      start_seconds=12.4, end_seconds=13.4)
    local = build_parquet(records, waves, tmp_path / "s.parquet")

    rows = pq.read_table(local).to_pylist()
    assert [r["segment_id"] for r in rows] == [
        "ibrahim-aldosari__087__0000", "ibrahim-aldosari__087__0001",
    ]
    # Timestamps live in the filename; the id stays put.
    assert rows[0]["audio"]["path"] == "ibrahim-aldosari__087__0000__12400-13400ms.flac"

    data, sr = sf.read(io.BytesIO(rows[0]["audio"]["bytes"]), dtype="float32")
    assert (sr, len(data)) == (16000, 4000)


def test_build_parquet_rejects_mismatched_inputs(tmp_path):
    records, waves = records_and_waves(n=3)
    with pytest.raises(ValueError):
        build_parquet(records, waves[:2], tmp_path / "s.parquet")
    with pytest.raises(ValueError):
        build_parquet([], [], tmp_path / "s.parquet")


def test_upload_params(fake_api, tmp_path):
    store = HubStore(repo_id="u/r", work_dir=tmp_path)
    store.upload_params({"model_id": "obadx/recitation-segmenter-v2", "batch_size": 16})
    assert "segment_params.json" in fake_api.uploads


def test_read_rows_keeps_shard_order_while_reading_in_parallel(monkeypatch):
    """Shards are read on a thread pool, so the order rows arrive in is a
    scheduling accident unless it is imposed. A manifest whose row order shifted
    run to run would make two manifests of the same corpus impossible to diff.
    """
    import random
    import time

    paths = [f"datasets/r/data/rec{i:02d}/001.parquet" for i in range(40)]
    random.Random(0).shuffle(paths)   # glob order is not sorted order

    class FakeFS:
        def __init__(self, *a, **k):
            pass

        def glob(self, pattern):
            return list(paths)

    def fake_read(fs, path):
        # Uneven latency: a serial reader cannot tell the difference, a
        # parallel one finishes out of order unless the results are reordered.
        time.sleep(random.Random(path).random() / 200)
        return [{"path": path, "i": i} for i in range(3)]

    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(hub, "_read_shard", fake_read)

    rows = list(hub.read_rows("r", workers=8))
    assert len(rows) == len(paths) * 3
    assert [r["path"] for r in rows[::3]] == sorted(paths)


def test_shards_are_opened_without_read_ahead_and_with_coalescing(monkeypatch):
    """Both settings are load-bearing and neither is a default, so a later
    tidy-up that drops one silently reintroduces the bug it fixed: without
    cache_type="none" fsspec's block cache pulls the audio column and a
    text-only manifest downloads the entire corpus; without pre_buffer the
    uncached reads go out one at a time and are slower than doing exactly that.
    """
    seen = {}

    class FakeFS:
        def open(self, path, mode, **kwargs):
            seen.update(kwargs)
            return io.BytesIO()

    def fake_parquet_file(handle, **kwargs):
        seen.update(kwargs)
        return "parquet"

    monkeypatch.setattr("pyarrow.parquet.ParquetFile", fake_parquet_file)
    assert hub._open_projected(FakeFS(), "some/shard.parquet") == "parquet"
    assert seen["cache_type"] == "none", "read-ahead would fetch the audio column"
    assert seen["pre_buffer"] is True, "uncached reads must be coalesced"


def test_read_rows_reports_progress_per_shard(monkeypatch):
    """A corpus-sized read is minutes of silence otherwise, which is
    indistinguishable from a hang."""
    paths = [f"datasets/r/data/rec{i:02d}/00{i}.parquet" for i in range(5)]

    class FakeFS:
        def __init__(self, *a, **k):
            pass

        def glob(self, pattern):
            return list(paths)

    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(hub, "_read_shard", lambda fs, p: [{"p": p}, {"p": p}])

    seen = []
    rows = list(hub.read_rows("r", on_progress=seen.append))

    assert len(rows) == 10
    assert [p.shards_done for p in seen] == [1, 2, 3, 4, 5]
    assert all(p.shards_total == 5 for p in seen)
    assert [p.rows for p in seen] == [2, 4, 6, 8, 10]
    # The source, not the full repo path, is what identifies progress usefully.
    assert seen[0].source == "rec00/000"
    assert seen[-1].fraction == 1.0
    assert seen[-1].eta_seconds == 0.0


def test_progress_line_survives_a_zero_length_read():
    """Division by shards_done and shards_total both have to be safe: an empty
    repo is a legitimate state, not an error."""
    empty = hub.ReadProgress(0, 0, 0, 0.0, "")
    assert empty.fraction == 1.0
    assert empty.eta_seconds == 0.0
    assert empty.line()
