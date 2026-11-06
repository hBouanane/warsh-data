"""Downloading a published corpus to work on locally.

Reviewing by ear is iterative -- listen, decide, adjust, listen again -- and
every pass over the hub costs the same seconds of round trips as the first.
Once a reciter is under suspicion, their shards are worth having on disk.

Downloaded per reciter rather than all at once.  A reciter is the unit of
repair here: segmentation thresholds that suit a fast voice cut a slow one
mid-phrase, so the thing being evaluated and re-run is one reciter's 114
recordings, about 1.4 GB, not the corpus's 21.

``hf_hub_download`` is used rather than a bulk snapshot so that an interrupted
pull resumes from its cache and already-downloaded shards are free.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

__all__ = ["Pulled", "shard_paths", "pull_shards", "local_rows"]


@dataclass
class Pulled:
    reciter: str
    surah: str
    path: Path
    bytes: int


def shard_paths(repo_id: str, reciters: Optional[Sequence[str]] = None,
                token: Optional[str] = None) -> List[str]:
    """Repo paths of the shards to fetch, in order.

    Listed from the repo rather than derived from a manifest, so a corpus can
    be pulled before it has ever been read.
    """
    from huggingface_hub import HfApi

    from .hub import source_id_from_shard_path

    wanted = set(reciters) if reciters else None
    out = []
    for path in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"):
        source_id = source_id_from_shard_path(path)
        if source_id is None:
            continue
        if wanted is not None and source_id.split("/")[0] not in wanted:
            continue
        out.append(path)
    return sorted(out)


def pull_shards(repo_id: str, dest: Path, reciters: Optional[Sequence[str]] = None,
                token: Optional[str] = None, workers: int = 8,
                on_each: Optional[Callable[[Pulled, int, int], None]] = None
                ) -> List[Pulled]:
    """Download shards into ``dest``, keeping the repo's own layout.

    The layout is preserved so a local tree and the repo are interchangeable:
    anything that reads ``data/<reciter>/<surah>.parquet`` works against either
    without being told which it has.
    """
    import concurrent.futures as futures

    from huggingface_hub import hf_hub_download

    paths = shard_paths(repo_id, reciters, token)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(path: str) -> Optional[Pulled]:
        try:
            local = hf_hub_download(repo_id, path, repo_type="dataset",
                                    local_dir=str(dest), token=token)
        except Exception as exc:
            print(f"warning: {path}: {exc}")
            return None
        parts = Path(path).parts
        got = Path(local)
        return Pulled(reciter=parts[1], surah=Path(parts[2]).stem,
                      path=got, bytes=got.stat().st_size)

    done: List[Pulled] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(fetch, paths):
            if result is None:
                continue
            done.append(result)
            if on_each is not None:
                on_each(result, len(done), len(paths))
    return done


def local_rows(root: Path, reciters: Optional[Sequence[str]] = None):
    """Yield manifest rows from a local tree, without the audio.

    The point of having the shards locally: this reads at disk speed, so
    filtering and re-filtering a reciter's segments is instant rather than a
    round trip each time.
    """
    import pyarrow.parquet as pq

    wanted = set(reciters) if reciters else None
    for shard in sorted(Path(root).glob("data/*/*.parquet")):
        if wanted is not None and shard.parent.name not in wanted:
            continue
        parquet = pq.ParquetFile(shard)
        columns = [n for n in parquet.schema_arrow.names if n != "audio"]
        for row in parquet.read(columns=columns).to_pylist():
            yield row


def local_clip(root: Path, source_id: str, segment_id: str) -> Optional[bytes]:
    """FLAC bytes for one segment from a local shard, or None if absent."""
    import pyarrow.parquet as pq

    shard = Path(root) / "data" / f"{source_id}.parquet"
    if not shard.exists():
        return None
    table = pq.ParquetFile(shard).read(columns=["segment_id", "audio"])
    for got, blob in zip(table.column("segment_id").to_pylist(),
                         table.column("audio").to_pylist()):
        if got == segment_id and isinstance(blob, dict):
            return blob.get("bytes")
    return None


def summarise(pulled: Sequence[Pulled]) -> str:
    by: Dict[str, List[Pulled]] = collections.defaultdict(list)
    for item in pulled:
        by[item.reciter].append(item)
    lines = []
    for reciter, items in sorted(by.items()):
        size = sum(i.bytes for i in items) / 1e9
        lines.append(f"  {reciter:<30}{len(items):>5} shards{size:>8.2f} GB")
    total = sum(i.bytes for i in pulled) / 1e9
    lines.append(f"  {'total':<30}{len(pulled):>5} shards{total:>8.2f} GB")
    return "\n".join(lines)
