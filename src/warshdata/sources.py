"""Discovering input recordings and naming what comes out of them.

A *source* is one input recording.  Its ``reciter_slug`` comes from the parent
directory name, so the expected layout is::

    audio/
      ibrahim-al-dosary/
        002.mp3
      yassin-al-jazaery/
        002.mp3

Segment ids are derived from the source id and the sample offsets rather than
from a counter, so re-running with the same settings reproduces the same ids and
a partially rebuilt manifest stays consistent with an older one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

__all__ = ["Source", "AUDIO_SUFFIXES", "discover", "slugify", "segment_id"]

#: Formats the segmenter's reader accepts.
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Source:
    path: Path
    reciter_slug: str
    source_id: str


def slugify(name: str) -> str:
    return _SLUG_STRIP.sub("-", name.strip().lower()).strip("-")


def discover(root: Path) -> List[Source]:
    """Find every audio file under ``root``.

    A file directly inside ``root`` is attributed to the reciter ``unknown``
    rather than skipped -- losing recordings silently because of a layout
    mistake is worse than a slug that has to be corrected later.
    """
    root = Path(root)
    if root.is_file():
        files = [root] if root.suffix.lower() in AUDIO_SUFFIXES else []
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)

    sources: List[Source] = []
    for path in files:
        if path.parent == root or path.parent.name == "":
            reciter = "unknown"
        else:
            reciter = slugify(path.parent.name)
        sources.append(
            Source(
                path=path,
                reciter_slug=reciter,
                source_id=f"{reciter}/{path.stem}",
            )
        )
    return sources


def segment_id(source: Source, start_sample: int, end_sample: int) -> str:
    return f"{source.reciter_slug}__{slugify(source.path.stem)}__{start_sample}_{end_sample}"
