"""Publishing the segmented corpus to a Hugging Face dataset repo.

One source recording produces exactly one parquet file, named after the source::

    data/<reciter-slug>/<surah>.parquet

That naming is the whole design, and it is what makes the pipeline safe to
interrupt:

*Resume is a file listing.*  The path encodes the source, so knowing what is
done costs one ``list_repo_files`` call -- nothing to download, and no record
that can disagree with the data.

*Re-running a source overwrites its own file.*  Segmenting the same recording
twice replaces its parquet rather than appending beside it, so duplicates cannot
occur and nothing ever needs purging.

*A source is committed whole or not at all.*  The parquet and the raw mp3 that
produced it go up in one atomic commit.  There is no window where one exists
without the other, and no buffer holding rows that something else already
believes are published.

The price is many small files -- roughly 1500 for a full Warsh corpus, averaging
~17 MB.  That streams fine; per-file overhead across an epoch is minutes against
hours of training.  ``warsh-data compact`` merges them into larger,
reciter-grouped shards afterwards, once the interruptible part is over.

Audio is embedded as 16 kHz mono FLAC: lossless, about half the size of WAV, and
preferred over Opus because the source is already lossy mp3.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

import numpy as np
import soundfile as sf

__all__ = [
    "HubStore",
    "shard_path_for",
    "source_id_from_shard_path",
    "encode_flac",
    "read_rows",
    "README",
]

SAMPLE_RATE = 16000

README = """---
license: cc-by-nc-4.0
task_categories:
  - automatic-speech-recognition
language:
  - ar
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/*/*.parquet
---

# {repo_id}

Warsh (Rewayat Warsh A'n Nafi') Quran recitation, segmented at waqf with
[obadx/recitation-segmenter-v2](https://huggingface.co/obadx/recitation-segmenter-v2).

Built with [warsh-data](https://github.com/hBouanane/warsh-data).

## Layout

| path | what |
|---|---|
| `data/<reciter>/<surah>.parquet` | one file per source recording, audio embedded as 16 kHz mono FLAC |
| `raw/<reciter>/<surah>.mp3` | the source recording it came from |
| `segment_params.json` | the settings this corpus was produced with |

One parquet per source recording, named after it, so re-running a recording
replaces its file and the corpus cannot accumulate duplicates.

## Load

    from datasets import load_dataset
    ds = load_dataset("{repo_id}", split="train", streaming=True)

## Notes

Segment ids are `<reciter>__<surah>__<ordinal>` and do not encode timestamps, so
they survive hand-correction of boundaries. Clip filenames do carry the
boundaries, because those are expected to change.

Audio source: mp3quran.net. Check their terms before redistributing.
"""


def encode_flac(wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """16 kHz mono FLAC bytes, PCM_16."""
    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def shard_path_for(reciter_slug: str, source_stem: str) -> str:
    """Repo path for a source's parquet.  Deterministic, hence overwritable."""
    return f"data/{reciter_slug}/{source_stem}.parquet"


def source_id_from_shard_path(path: str) -> Optional[str]:
    """Inverse of :func:`shard_path_for`; ``None`` for any other path."""
    parts = Path(path).parts
    if len(parts) != 3 or parts[0] != "data" or not parts[2].endswith(".parquet"):
        return None
    return f"{parts[1]}/{Path(parts[2]).stem}"


def build_parquet(
    records: Sequence[Dict[str, Any]],
    waves: Sequence[np.ndarray],
    local: Path,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Write one source's segments to a parquet with the audio embedded."""
    from datasets import Audio, Dataset

    from .sources import clip_name

    if len(records) != len(waves):
        raise ValueError(f"{len(records)} records but {len(waves)} waveforms")
    if not records:
        raise ValueError("nothing to write")

    rows = []
    for record, wave in zip(records, waves):
        row = dict(record)
        name = clip_name(
            record["segment_id"],
            record["start_sample"],
            record["end_sample"],
            record["sample_rate"],
        )
        row["audio"] = {"bytes": encode_flac(wave, sample_rate), "path": f"{name}.flac"}
        rows.append(row)

    columns = {key: [row.get(key) for row in rows] for key in rows[0]}
    dataset = Dataset.from_dict(columns).cast_column("audio", Audio(sampling_rate=sample_rate))
    local.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(local))
    return local


@dataclass
class HubStore:
    """Writes one source at a time to a dataset repo, atomically."""

    repo_id: str
    work_dir: Path
    token: Optional[str] = None
    private: bool = False
    upload_raw: bool = True

    def __post_init__(self) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi(token=self.token)
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.sources_written = 0
        self.rows_written = 0

        self.api.create_repo(
            self.repo_id, repo_type="dataset", private=self.private, exist_ok=True
        )
        self._ensure_readme()

    def repo_files(self) -> List[str]:
        try:
            return list(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        except Exception:
            return []

    def done_sources(self) -> Set[str]:
        """Sources already published, from shard filenames alone.

        If the file is there, that source was committed in full -- the commit
        that creates it contains the entire source.
        """
        out: Set[str] = set()
        for path in self.repo_files():
            source_id = source_id_from_shard_path(path)
            if source_id is not None:
                out.add(source_id)
        return out

    def _ensure_readme(self) -> None:
        if "README.md" in self.repo_files():
            return
        self.api.upload_file(
            path_or_fileobj=README.format(repo_id=self.repo_id).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    def write_source(
        self,
        reciter_slug: str,
        source_stem: str,
        records: Sequence[Dict[str, Any]],
        waves: Sequence[np.ndarray],
        raw_path: Optional[Path] = None,
    ) -> str:
        """Publish one source: its parquet and its mp3, in a single commit.

        Raises on failure rather than swallowing it -- a source that silently
        failed to upload would be absent from the repo and therefore retried,
        which is correct, but the caller should still hear about it.
        """
        from huggingface_hub import CommitOperationAdd

        path_in_repo = shard_path_for(reciter_slug, source_stem)
        local = self.work_dir / f"{reciter_slug}__{source_stem}.parquet"
        build_parquet(records, waves, local)

        operations = [CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local))]
        if self.upload_raw and raw_path is not None and Path(raw_path).exists():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"raw/{reciter_slug}/{Path(raw_path).name}",
                    path_or_fileobj=str(raw_path),
                )
            )

        try:
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"{reciter_slug}/{source_stem}: {len(records)} segments",
            )
        finally:
            # Scratch either way: a failed commit is retried from the source
            # audio, never from this file.
            local.unlink(missing_ok=True)

        self.sources_written += 1
        self.rows_written += len(records)
        return path_in_repo

    def upload_params(self, params: Dict[str, Any]) -> None:
        self.api.upload_file(
            path_or_fileobj=json.dumps(params, indent=2, ensure_ascii=False).encode("utf-8"),
            path_in_repo="segment_params.json",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Record segmentation parameters",
        )


def read_rows(repo_id: str, token: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield every segment row from a repo's shards, without the audio.

    Parquet is columnar, so the audio column is never transferred -- this reads
    a corpus-sized manifest for a few MB.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    for path in sorted(fs.glob(f"datasets/{repo_id}/data/*/*.parquet")):
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            columns = [name for name in parquet.schema_arrow.names if name != "audio"]
            table = parquet.read(columns=columns)
        for row in table.to_pylist():
            yield row
