"""Listening to specific segments, chosen from the manifest.

``listen`` answers "is this *recording* wrong?" by excerpting whole source
files.  This answers a different question -- "is this *segment* wrong?" -- and
the two extremes worth hearing are the ends of two distributions:

*The longest.*  A 76-second segment means the segmenter missed every stop
inside it.  Past 30 seconds nothing is transcribed or labelled at all, so these
are the rows that silently became holes in the corpus.

*The worst aligned.*  A high distance is either a genuinely misplaced span or
just a voice the bootstrap recogniser struggled with -- and those two need
opposite treatment.  Only listening while reading the label separates them.

Audio is pulled for the chosen rows alone.  Their shards come down whole, since
audio is most of a shard, but twenty segments is twenty shards rather than the
1494 a full download would be.
"""

from __future__ import annotations

import base64
import collections
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["pick", "fetch_clips", "build_page"]

FIELDS = {"duration": "duration_seconds", "distance": "align_distance"}


def pick(rows: Sequence[Dict[str, Any]], by: str, n: int,
         largest: bool = True) -> List[Dict[str, Any]]:
    """The ``n`` most extreme rows by ``duration`` or ``distance``.

    Rows missing the field are dropped rather than sorted as zero: a segment
    with no distance was never aligned at all, which is its own category and
    not the worst-aligned one.  Sorting those in as zeros would fill the
    shortest-first list with rows that have nothing to do with duration.
    """
    field = FIELDS[by]
    scored = [r for r in rows if r.get(field) is not None]
    scored.sort(key=lambda r: r[field], reverse=largest)
    return scored[:n]


def fetch_clips(repo_id: str, rows: Sequence[Dict[str, Any]],
                token: Optional[str] = None, workers: int = 8) -> Dict[str, bytes]:
    """FLAC bytes for each row's ``segment_id``, keyed by it.

    Rows are grouped by source first, so asking for several segments of one
    recording costs one shard read rather than one per segment.
    """
    import concurrent.futures as futures

    from huggingface_hub import HfFileSystem

    from .hub import _open_projected

    wanted: Dict[str, set] = collections.defaultdict(set)
    for row in rows:
        wanted[row["source_id"]].add(row["segment_id"])

    fs = HfFileSystem(token=token)

    def read_one(source_id: str) -> Dict[str, bytes]:
        path = f"datasets/{repo_id}/data/{source_id}.parquet"
        try:
            table = _open_projected(fs, path).read(columns=["segment_id", "audio"])
        except Exception as exc:
            print(f"warning: could not read {source_id}: {exc}")
            return {}
        found = {}
        for segment_id, blob in zip(table.column("segment_id").to_pylist(),
                                    table.column("audio").to_pylist()):
            if segment_id in wanted[source_id] and isinstance(blob, dict):
                found[segment_id] = blob.get("bytes") or b""
        return found

    clips: Dict[str, bytes] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(read_one, sorted(wanted)):
            clips.update(got)
    return clips


def _fmt(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}" if minutes else f"{seconds:.1f}s"


def _facts(row: Dict[str, Any]) -> str:
    out = [_fmt(row.get("duration_seconds") or 0)]
    if row.get("align_distance") is not None:
        out.append(f"distance {row['align_distance']:.3f}")
    if row.get("ayah_start") is not None:
        span = str(row["ayah_start"])
        if row.get("ayah_end") not in (None, row.get("ayah_start")):
            span += f"-{row['ayah_end']}"
        out.append(f"ayah {span}")
    if row.get("align_ok") is False:
        out.append("align_ok false")
    if row.get("is_repeat"):
        out.append("repeat")
    if row.get("is_formula"):
        out.append("formula")
    return "  ·  ".join(out)


def _arabic(kind: str, value) -> str:
    if not value:
        return (f'<div class="text"><span class="tag">{kind}</span>'
                f'<span class="none">none</span></div>')
    return (f'<div class="text"><span class="tag">{kind}</span>'
            f'<span dir="rtl" lang="ar">{html.escape(value)}</span></div>')


def _row_block(rank: int, row: Dict[str, Any], data: Optional[bytes]) -> str:
    if data:
        encoded = base64.b64encode(data).decode("ascii")
        player = (f'<audio controls preload="none" '
                  f'src="data:audio/flac;base64,{encoded}"></audio>')
    else:
        player = '<p class="missing">audio unavailable</p>'

    return (f'<section>'
            f'<h2><span class="rank">{rank}</span>'
            f'{html.escape(row["segment_id"])}</h2>'
            f'<p class="meta">{html.escape(_facts(row))}</p>'
            f'{player}'
            f'{_arabic("heard", row.get("asr"))}'
            f'{_arabic("label", row.get("label"))}'
            f'</section>')


_STYLE = """
  :root { color-scheme: light dark; }
  body { font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 2rem 1rem;
         max-width: 56rem; margin-inline: auto; }
  h1 { font-size: 1.4rem; margin-bottom: .25rem; }
  .lede { opacity: .75; margin-top: 0; }
  h3 { margin: 2.2rem 0 .5rem; font-size: 1.1rem; }
  h3 .why { font-weight: 400; opacity: .7; font-size: .85rem; display: block;
            margin-top: .15rem; }
  section { border-top: 1px solid rgba(128,128,128,.35); padding: 1rem 0; }
  h2 { font-size: .95rem; font-family: ui-monospace, monospace; margin: 0 0 .3rem;
       display: flex; align-items: baseline; gap: .6rem; }
  .rank { opacity: .5; font-variant-numeric: tabular-nums; }
  .meta { margin: .2rem 0 .7rem; opacity: .65; font-size: .85rem;
          font-variant-numeric: tabular-nums; }
  audio { width: 100%; height: 34px; margin-bottom: .6rem; }
  .text { display: flex; gap: .75rem; align-items: baseline; margin: .35rem 0; }
  .tag { flex: 0 0 3.4rem; font-size: .75rem; text-transform: uppercase;
         letter-spacing: .06em; opacity: .55; }
  .text span[dir=rtl] { font-size: 1.35rem; line-height: 2.1; flex: 1; }
  .none { opacity: .45; font-style: italic; }
  .missing { color: #b45309; margin: .3rem 0 .6rem; }
"""


def build_page(sections: Sequence[tuple], clips: Dict[str, bytes], out_path: Path,
               title: str = "Segments worth hearing") -> Path:
    """One self-contained page.  ``sections`` is ``(heading, why, [row, ...])``.

    Audio is embedded as data URIs rather than linked: the page gets opened
    from a Colab download or a file browser, where relative paths to clip files
    are exactly what breaks.
    """
    blocks = []
    for heading, why, rows in sections:
        if not rows:
            continue
        body = "".join(_row_block(i, row, clips.get(row["segment_id"]))
                       for i, row in enumerate(rows, 1))
        blocks.append(f'<h3>{html.escape(heading)}'
                      f'<span class="why">{html.escape(why)}</span></h3>{body}')

    page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{html.escape(title)}</title>\n<style>{_STYLE}</style>\n'
            f'</head>\n<body>\n<h1>{html.escape(title)}</h1>\n'
            f'<p class="lede">"heard" is the bootstrap recogniser and is kept for '
            f'diagnosis only. "label" is the reference text the aligner chose, and '
            f'is what training would see — so the label is what you are '
            f'judging.</p>\n{"".join(blocks)}\n</body>\n</html>\n')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
