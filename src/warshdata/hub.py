"""Streaming the segmented corpus into a Hugging Face dataset repo.

The layout is chosen so that training never downloads more than it reads:

``data/shard-NNNNN.parquet``
    Segment rows with the audio **embedded** as 16 kHz mono FLAC, in shards of a
    few hundred MB.  Parquet is columnar and range-readable over HTTP, so
    ``load_dataset(..., streaming=True)`` pulls only the shards it is currently
    iterating.  Embedding beats one file per clip by a wide margin: hundreds of
    thousands of small LFS objects are slow to list, slow to fetch, and awkward
    to shuffle.

``manifests/segments.jsonl``
    The same rows without audio.  Small, greppable, diffable -- this is what you
    read to plan a run, count segments, or resume one, without touching audio.

``raw/<reciter>/NNN.mp3``
    The source recordings, so the corpus can be rebuilt if the upstream site
    changes.  Never read during training.

FLAC rather than WAV because it is lossless and roughly half the size; lossless
rather than Opus because the source is already lossy mp3 and a second generation
of loss is not worth the saving at this corpus size.

Shards are flushed and uploaded as they fill, but only ever at a *source*
boundary, and the manifest is uploaded only immediately after a successful
shard flush.  Together those two rules give the invariant that resumption
depends on:

    every source named in the hub manifest has all of its audio in a shard

Without it, a manifest listing segments still sitting in the write buffer marks
those sources done, they are skipped on resume, and their audio is never
uploaded by anyone -- rows in the manifest pointing at audio that exists
nowhere.  A crash now costs re-segmenting up to one shard, never data.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import soundfile as sf

__all__ = ["HubWriter", "DEFAULT_SHARD_BYTES", "done_sources_on_hub", "encode_flac",
           "shard_sources", "manifest_sources"]

#: Target audio bytes per shard.  ~400 MB keeps the shard count sane for a few
#: hundred hours while staying small enough to stream comfortably.
DEFAULT_SHARD_BYTES = 400 * 1024 * 1024

_README = """---
license: cc-by-nc-4.0
task_categories:
  - automatic-speech-recognition
language:
  - ar
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/shard-*.parquet
---

# {repo_id}

Warsh (Rewayat Warsh A'n Nafi') Quran recitation, segmented at waqf with
[obadx/recitation-segmenter-v2](https://huggingface.co/obadx/recitation-segmenter-v2).

Built with [warsh-data](https://github.com/hBouanane/warsh-data).

## Layout

| path | what |
|---|---|
| `data/shard-*.parquet` | segment rows, audio embedded as 16 kHz mono FLAC |
| `manifests/segments.jsonl` | the same rows without audio |
| `raw/<reciter>/NNN.mp3` | source recordings from mp3quran.net |

## Load

    from datasets import load_dataset
    ds = load_dataset("{repo_id}", split="train", streaming=True)

Streaming reads only the shards it iterates, so training does not download the
whole corpus.

## Notes

Segment ids are `<reciter>__<surah>__<ordinal>` and do not encode timestamps, so
they survive hand-correction of boundaries.

Audio source: mp3quran.net. Check their terms before redistributing.
"""


def encode_flac(wave: np.ndarray, sample_rate: int = 16000) -> bytes:
    """16 kHz mono FLAC bytes, PCM_16."""
    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def done_sources_on_hub(repo_id: str, token: Optional[str] = None) -> Set[str]:
    """Source ids already in the hub manifest, so a run resumes across sessions.

    Resuming reads the *manifest*, not the shards -- a few MB instead of the
    whole corpus, which is the point of keeping the two separate.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="manifests/segments.jsonl",
            repo_type="dataset",
            token=token,
        )
    except Exception:
        return set()

    done: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["source_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def shard_sources(repo_id: str, token: Optional[str] = None) -> "collections.Counter[str]":
    """Count segments per source across the shards, reading only that column.

    Parquet is columnar, so this pulls the ``source_id`` column and nothing
    else -- the audio, which is all the size, is never transferred.
    """
    import collections

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    counts: "collections.Counter[str]" = collections.Counter()
    for path in sorted(fs.glob(f"datasets/{repo_id}/data/shard-*.parquet")):
        with fs.open(path, "rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["source_id"])
        counts.update(table.column("source_id").to_pylist())
    return counts


def manifest_sources(manifest_path: Path) -> "collections.Counter[str]":
    """Count segments per source in a manifest file."""
    import collections

    counts: "collections.Counter[str]" = collections.Counter()
    with Path(manifest_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                counts[json.loads(line)["source_id"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


@dataclass
class HubWriter:
    """Buffers segments, flushing a parquet shard whenever it fills."""

    repo_id: str
    work_dir: Path
    token: Optional[str] = None
    shard_bytes: int = DEFAULT_SHARD_BYTES
    private: bool = False
    upload_raw: bool = True

    def __post_init__(self) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi(token=self.token)
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.api.create_repo(
            self.repo_id, repo_type="dataset", private=self.private, exist_ok=True
        )
        self._ensure_readme()

        self._rows: List[Dict[str, Any]] = []
        self._buffered = 0
        self._pending_sources: List[tuple[Path, str]] = []
        self.shard_index = self._next_shard_index()
        self.shards_written = 0
        self.rows_written = 0

    def _ensure_readme(self) -> None:
        files = self._repo_files()
        if "README.md" in files:
            return
        self.api.upload_file(
            path_or_fileobj=_README.format(repo_id=self.repo_id).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    def repo_files(self) -> List[str]:
        """Files currently in the repo; empty if it cannot be listed."""
        return self._repo_files()

    def _repo_files(self) -> List[str]:
        try:
            return list(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        except Exception:
            return []

    def _next_shard_index(self) -> int:
        """Continue numbering after whatever the repo already holds.

        Restarting at zero would overwrite shards from an earlier session, which
        is exactly the failure a resumable pipeline exists to avoid.
        """
        used = [
            int(Path(f).stem.split("-")[-1])
            for f in self._repo_files()
            if f.startswith("data/shard-") and f.endswith(".parquet")
        ]
        return max(used) + 1 if used else 0

    def add(self, record: Dict[str, Any], wave: np.ndarray) -> None:
        """Buffer one segment.  Never flushes -- see :meth:`maybe_flush`."""
        row = dict(record)
        audio_bytes = encode_flac(wave)
        from .sources import clip_name

        name = clip_name(
            record["segment_id"], record["start_sample"], record["end_sample"], record["sample_rate"]
        )
        row["audio"] = {"bytes": audio_bytes, "path": f"{name}.flac"}
        self._rows.append(row)
        self._buffered += len(audio_bytes)

    def maybe_flush(self) -> Optional[str]:
        """Flush if the buffer is full.  Call only at a source boundary.

        Flushing mid-source would put some of a source's segments in a shard and
        leave the rest buffered; the manifest would then mark that source done
        while part of its audio had gone nowhere.  Shards may overshoot
        ``shard_bytes`` by up to one recording, which is the price of the
        invariant and is cheap.
        """
        if self._buffered >= self.shard_bytes:
            return self.flush()
        return None

    def flush(self) -> Optional[str]:
        """Write and upload the buffered rows as one shard."""
        if not self._rows:
            return None

        from datasets import Audio, Dataset

        shard_name = f"shard-{self.shard_index:05d}.parquet"
        local = self.work_dir / shard_name

        columns = {k: [r.get(k) for r in self._rows] for k in self._rows[0]}
        ds = Dataset.from_dict(columns).cast_column("audio", Audio(sampling_rate=16000))
        ds.to_parquet(str(local))

        self.api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"data/{shard_name}",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add {shard_name} ({len(self._rows)} segments)",
        )

        # Bounded local disk: the shard is durable on the Hub now.
        local.unlink(missing_ok=True)

        self.rows_written += len(self._rows)
        self.shards_written += 1
        self.shard_index += 1
        self._rows = []
        self._buffered = 0
        return shard_name

    def push_manifest(self, manifest_path: Path) -> None:
        """Re-upload the audio-free manifest so a later run can resume from it."""
        self.api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo="manifests/segments.jsonl",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Update manifest",
        )

    def queue_source(self, path: Path, reciter_slug: str) -> None:
        """Queue a source recording for the next batched commit."""
        if self.upload_raw:
            self._pending_sources.append((Path(path), reciter_slug))

    def flush_sources(self) -> int:
        """Commit every queued source file in a single commit.

        One commit per file would mean thousands of commits over a full pass --
        slow, and rate-limited.  ``create_commit`` takes the whole batch at once.
        """
        if not self._pending_sources:
            return 0

        from huggingface_hub import CommitOperationAdd

        ops = [
            CommitOperationAdd(
                path_in_repo=f"raw/{slug}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path, slug in self._pending_sources
        ]
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Add {len(ops)} source recording(s)",
        )
        n = len(ops)
        self._pending_sources = []
        return n
