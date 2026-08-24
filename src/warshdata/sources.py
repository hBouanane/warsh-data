"""Discovering input recordings and naming what comes out of them.

A *source* is one input recording.  Its ``reciter_slug`` comes from the parent
directory name, so the expected layout is::

    audio/
      ibrahim-al-dosary/
        002.mp3
      yassin-al-jazaery/
        002.mp3

Segment ids are ``<reciter>__<source>__<ordinal>`` and deliberately do **not**
encode the boundaries.  Timestamps get corrected by hand; an id built from
start/end samples would change the moment a boundary moved, orphaning every
label, correction, and review note attached to it.  The ordinal is stable under
boundary edits, which is the operation that actually happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

__all__ = ["Source", "AUDIO_SUFFIXES", "discover", "slugify", "segment_id", "clip_name"]

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


def segment_id(source: Source, index: int) -> str:
    """Identity for a segment: stable under boundary correction.

    Not derived from start/end samples on purpose -- see the module docstring.
    Re-segmenting the same source with *different* thresholds does renumber
    these; that is a new derivation of the corpus, and corrections carry the
    boundaries they were written against so they can be re-matched by overlap
    rather than silently applied to the wrong audio.
    """
    return f"{source.reciter_slug}__{slugify(source.path.stem)}__{index:04d}"


def clip_name(segment_id: str, start_sample: int, end_sample: int, sample_rate: int = 16000) -> str:
    """Filename for a clip: the stable id plus the boundaries it currently has.

    The timestamps are here and not in the id on purpose.  A filename is a
    display of the segment's present state and is expected to change when a
    boundary is corrected; the id is what labels and review notes are keyed to,
    so it must not.
    """
    start_ms = int(round(start_sample * 1000 / sample_rate))
    end_ms = int(round(end_sample * 1000 / sample_rate))
    return f"{segment_id}__{start_ms}-{end_ms}ms"
