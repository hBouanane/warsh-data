"""Finding suspect source recordings before they are segmented.

The useful signal is *comparative*.  There is no absolute rule saying how long
Surah 94 should be, but fifteen reciters all recite the same surah, so a file
whose duration is wildly out of line with the other fourteen is wrong -- a
mislabelled file, a truncated download, or a different recording entirely.  The
median across reciters is the reference, and it needs no external data.

Two absolute checks come first, since they need no comparison: a file that will
not decode, and a file whose duration is zero.

Bitrate (bytes per second of audio) is reported rather than flagged.  A reciter
encoded at 32 kbps where others use 128 kbps is not an error, but it explains a
small file and is worth seeing before wondering about it.
"""

from __future__ import annotations

import concurrent.futures
import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .sources import Source

__all__ = ["Probe", "Finding", "probe_file", "probe_all", "find_outliers"]

#: A duration this many times off the median for the same surah is suspect.
#: Reciters genuinely differ in pace by up to roughly 2x, so 3x is comfortably
#: outside natural variation while still catching a swapped file.
DEFAULT_FACTOR = 3.0


@dataclass
class Probe:
    source_id: str
    reciter_slug: str
    stem: str
    path: Path
    size_bytes: int
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    @property
    def bytes_per_second(self) -> Optional[float]:
        if not self.duration_seconds:
            return None
        return self.size_bytes / self.duration_seconds


@dataclass
class Finding:
    source_id: str
    kind: str
    detail: str
    duration_seconds: Optional[float] = None
    expected_seconds: Optional[float] = None
    ratio: Optional[float] = None

    def line(self) -> str:
        if self.ratio is not None:
            return (f"{self.source_id:<34} {self.kind:<12} {self.detail}  "
                    f"({self.duration_seconds:.0f}s vs {self.expected_seconds:.0f}s "
                    f"median, {self.ratio:.1f}x)")
        return f"{self.source_id:<34} {self.kind:<12} {self.detail}"


def _ffprobe_duration(path: Path) -> Optional[float]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return None


def _soundfile_duration(path: Path) -> Optional[float]:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return float(info.frames) / info.samplerate
    except Exception:
        return None


def probe_file(source: Source) -> Probe:
    """Read one file's duration without decoding it fully."""
    path = Path(source.path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return Probe(source.source_id, source.reciter_slug, path.stem, path, 0,
                     error=f"cannot stat: {exc}")

    if size == 0:
        return Probe(source.source_id, source.reciter_slug, path.stem, path, 0,
                     error="empty file")

    duration = _ffprobe_duration(path) if shutil.which("ffmpeg") else None
    if duration is None:
        duration = _soundfile_duration(path)

    if duration is None:
        return Probe(source.source_id, source.reciter_slug, path.stem, path, size,
                     error="will not decode")
    return Probe(source.source_id, source.reciter_slug, path.stem, path, size, duration)


def probe_all(sources: Sequence[Source], workers: int = 8) -> List[Probe]:
    """Probe every source concurrently -- this is IO bound, not CPU bound."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(probe_file, sources))


def find_outliers(probes: Sequence[Probe], factor: float = DEFAULT_FACTOR) -> List[Finding]:
    """Flag unreadable files, and durations far off the median for that surah."""
    findings: List[Finding] = []

    for probe in probes:
        if probe.error:
            findings.append(Finding(probe.source_id, "unreadable", probe.error))
        elif not probe.duration_seconds:
            findings.append(Finding(probe.source_id, "empty", "zero duration"))

    by_stem: Dict[str, List[Probe]] = {}
    for probe in probes:
        if probe.duration_seconds:
            by_stem.setdefault(probe.stem, []).append(probe)

    for stem, group in sorted(by_stem.items()):
        if len(group) < 3:
            # With one or two reciters there is no majority to be an outlier
            # from; calling either of them wrong would be a coin flip.
            continue
        median = statistics.median(p.duration_seconds for p in group)
        if median <= 0:
            continue
        for probe in group:
            ratio = probe.duration_seconds / median
            if ratio >= factor:
                findings.append(Finding(
                    probe.source_id, "too long",
                    f"surah {stem} is much longer than other reciters'",
                    probe.duration_seconds, median, ratio))
            elif ratio <= 1 / factor:
                findings.append(Finding(
                    probe.source_id, "too short",
                    f"surah {stem} is much shorter than other reciters'",
                    probe.duration_seconds, median, ratio))

    order = {"unreadable": 0, "empty": 1, "too long": 2, "too short": 3}
    findings.sort(key=lambda f: (order.get(f.kind, 9), -(f.ratio or 0)))
    return findings


def summary(probes: Sequence[Probe]) -> Dict[str, object]:
    ok = [p for p in probes if p.duration_seconds]
    hours = sum(p.duration_seconds for p in ok) / 3600
    return {
        "files": len(probes),
        "readable": len(ok),
        "hours": round(hours, 2),
        "gigabytes": round(sum(p.size_bytes for p in probes) / 1e9, 2),
    }


def bitrate_table(probes: Sequence[Probe]) -> List[tuple]:
    """Median bytes-per-second per reciter, to explain small or large files."""
    by_reciter: Dict[str, List[float]] = {}
    for probe in probes:
        bps = probe.bytes_per_second
        if bps:
            by_reciter.setdefault(probe.reciter_slug, []).append(bps)
    rows = [
        (slug, statistics.median(values), len(values))
        for slug, values in sorted(by_reciter.items())
    ]
    return rows
