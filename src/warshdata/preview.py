"""Short listenable excerpts from suspect recordings.

A flagged file is a question, not an answer: ``rachid-belalya/094`` being 19
minutes where the median is 38 seconds says the file is wrong, but not *how*.
Only listening settles that, and nobody wants to scrub through 19 minutes to
find out.

So take a few short excerpts spread across the recording -- the beginning, the
middle, near the end -- and put them in one page.  Three 15-second clips are
enough to tell a mislabelled surah from a truncated download from a completely
different recording, and they are small enough to embed directly in the HTML, so
the page is one file that plays anywhere with no server and no missing assets.
"""

from __future__ import annotations

import base64
import html
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = ["Excerpt", "extract_excerpts", "build_page"]


@dataclass
class Excerpt:
    source_id: str
    label: str
    at_seconds: float
    path: Path
    data: bytes


def _positions(duration: float, count: int, clip_seconds: float) -> List[tuple]:
    """Where to cut, as (label, start_seconds).

    Spread across the recording rather than taken from the front: a file that is
    the wrong recitation entirely usually looks fine for the first few seconds
    (basmala), and only the middle gives it away.
    """
    if duration <= clip_seconds * 1.5:
        return [("whole", 0.0)]

    if count <= 1:
        return [("middle", max(0.0, duration / 2 - clip_seconds / 2))]

    labels = ["start", "middle", "end", "quarter", "three-quarters"]
    fractions = [0.02, 0.5, 0.92, 0.25, 0.75][:count]
    out = []
    for label, fraction in zip(labels, sorted(fractions)):
        start = max(0.0, min(duration - clip_seconds, duration * fraction))
        out.append((label, start))
    # Re-label by actual order so "start" really is the earliest.
    out.sort(key=lambda item: item[1])
    names = ["start", "middle", "end"] if len(out) == 3 else [f"@{s:.0f}s" for _, s in out]
    return [(name, start) for name, (_, start) in zip(names, out)]


def _cut_ffmpeg(path: Path, start: float, clip_seconds: float, dest: Path) -> Optional[bytes]:
    """Extract one excerpt as a small mono mp3."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{clip_seconds:.3f}",
        "-i", str(path),
        "-ac", "1", "-ar", "22050", "-b:a", "48k",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        return None
    return dest.read_bytes()


def _cut_soundfile(path: Path, start: float, clip_seconds: float, dest: Path) -> Optional[bytes]:
    """Fallback that reads only the frames it needs.

    Produces WAV rather than mp3 -- roughly ten times larger, which matters for
    the embedded page but not for the handful of files this is used on, and it
    keeps the tool usable on a machine without ffmpeg.
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))
        rate = info.samplerate
        begin = int(start * rate)
        frames = int(clip_seconds * rate)
        data, _ = sf.read(str(path), start=begin, frames=frames, dtype="float32",
                          always_2d=True)
        if data.size == 0:
            return None
        sf.write(dest, data.mean(axis=1), rate, subtype="PCM_16")
    except Exception:
        return None
    return dest.read_bytes()


def _cut(path: Path, start: float, clip_seconds: float, dest: Path) -> Optional[tuple]:
    """Return ``(bytes, path)`` for one excerpt, or None."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        data = _cut_ffmpeg(path, start, clip_seconds, dest.with_suffix(".mp3"))
        if data is not None:
            return data, dest.with_suffix(".mp3")
    data = _cut_soundfile(path, start, clip_seconds, dest.with_suffix(".wav"))
    if data is not None:
        return data, dest.with_suffix(".wav")
    return None


def extract_excerpts(
    source_id: str,
    path: Path,
    duration: float,
    out_dir: Path,
    count: int = 3,
    clip_seconds: float = 15.0,
) -> List[Excerpt]:
    """Cut ``count`` excerpts spread across one recording."""
    excerpts: List[Excerpt] = []
    slug = source_id.replace("/", "__")
    for label, start in _positions(duration, count, clip_seconds):
        dest = Path(out_dir) / f"{slug}__{label}"
        cut = _cut(Path(path), start, clip_seconds, dest)
        if cut is not None:
            data, written = cut
            excerpts.append(Excerpt(source_id, label, start, written, data))
    return excerpts


def _fmt(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def build_page(
    groups: Sequence[tuple],
    out_path: Path,
    title: str = "Suspect recordings",
) -> Path:
    """Write one self-contained HTML page with every excerpt embedded.

    ``groups`` is a sequence of ``(source_id, note, duration, [Excerpt, ...])``.
    Audio goes in as data URIs so the page is a single file: no server, no
    relative paths to break when it is downloaded off Colab.
    """
    blocks = []
    for source_id, note, duration, excerpts in groups:
        players = []
        for excerpt in excerpts:
            encoded = base64.b64encode(excerpt.data).decode("ascii")
            mime = "audio/wav" if excerpt.path.suffix == ".wav" else "audio/mpeg"
            players.append(
                f'<div class="clip">'
                f'<span class="at">{html.escape(excerpt.label)} '
                f'&middot; {_fmt(excerpt.at_seconds)}</span>'
                f'<audio controls preload="none" '
                f'src="data:{mime};base64,{encoded}"></audio>'
                f"</div>"
            )
        blocks.append(
            f'<section><h2>{html.escape(source_id)}</h2>'
            f'<p class="note">{html.escape(note)}</p>'
            f'<p class="meta">full recording {_fmt(duration)}</p>'
            f'<div class="clips">{"".join(players)}</div></section>'
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 2rem 1rem;
         max-width: 52rem; margin-inline: auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  .lede {{ opacity: .75; margin-top: 0; }}
  section {{ border-top: 1px solid rgba(128,128,128,.35); padding: 1.25rem 0; }}
  h2 {{ font-size: 1.05rem; font-family: ui-monospace, monospace; margin: 0 0 .35rem; }}
  .note {{ margin: .2rem 0; }}
  .meta {{ margin: .2rem 0 .75rem; opacity: .6; font-size: .85rem; }}
  .clips {{ display: grid; gap: .6rem; }}
  .clip {{ display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }}
  .at {{ min-width: 8rem; font-family: ui-monospace, monospace; font-size: .85rem;
         opacity: .8; }}
  audio {{ flex: 1 1 18rem; max-width: 100%; height: 34px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="lede">{len(groups)} recording(s). Excerpts are spread across each file &mdash;
the middle usually gives away a wrong recitation, since the opening sounds normal.</p>
{"".join(blocks)}
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
