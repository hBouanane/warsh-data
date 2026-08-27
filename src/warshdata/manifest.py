"""JSONL manifests: one record per audio segment.

JSONL rather than a single JSON array for one reason -- it appends cleanly, so a
run that dies at hour six leaves every completed record intact and readable.
Everything else in this module exists to support resuming from such a file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

__all__ = ["SegmentRecord", "append", "read", "done_sources", "write_params"]


@dataclass
class SegmentRecord:
    """One waqf-bounded segment, before any text label exists."""

    segment_id: str
    reciter_slug: str
    source_id: str
    source_path: str
    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    sample_rate: int
    audio_path: Optional[str] = None
    #: False when the segmenter ran out of audio mid-speech, i.e. the recording
    #: was cut off rather than ending at a stop.  The final segment of such a
    #: file is not waqf-bounded and should not be trusted as a training example.
    source_is_complete: bool = True
    is_last_of_source: bool = False

    #: Filled in by the transcribe + align pass.  ``asr`` is the recogniser's
    #: guess and is kept for diagnosis only; ``label`` is the reference text the
    #: aligner selected, which is what a model should be trained on.
    surah_number: Optional[int] = None
    asr: Optional[str] = None
    label: Optional[str] = None
    ref_start: Optional[int] = None
    ref_end: Optional[int] = None
    ayah_start: Optional[int] = None
    ayah_end: Optional[int] = None
    align_distance: Optional[float] = None
    align_ok: Optional[bool] = None
    is_formula: bool = False
    is_repeat: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append(path: Path, records: List[SegmentRecord]) -> None:
    """Append records and flush, so a crash costs at most the current file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
        fh.flush()


def read(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records, skipping a torn final line from an interrupted run."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def done_sources(path: Path) -> Set[str]:
    """Source ids already segmented, for resuming a partial run."""
    return {rec["source_id"] for rec in read(path) if "source_id" in rec}


def write_params(path: Path, params: Dict[str, Any]) -> None:
    """Record the settings a manifest was produced with, next to the manifest.

    Segment boundaries are only meaningful together with the thresholds that
    produced them; without this, a manifest cannot be reproduced or compared.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
