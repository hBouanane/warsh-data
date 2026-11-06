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

__all__ = ["pick", "unlabelled", "by_reciter", "within", "reciters",
           "previous_ids", "fetch_clips", "build_page"]

FIELDS = {"duration": "duration_seconds", "distance": "align_distance"}


def pick(rows: Sequence[Dict[str, Any]], by: str, n: int,
         largest: bool = True) -> List[Dict[str, Any]]:
    """The ``n`` most extreme rows by ``duration`` or ``distance``.

    Rows missing the field are dropped rather than sorted as zero: a segment
    with no distance was never aligned at all, which is its own category and
    not the worst-aligned one.  Sorting those in as zeros would fill the
    shortest-first list with rows that have nothing to do with duration.

    By distance, rows lacking a transcript or a label are dropped too -- see
    the comment below for why they used to dominate the list while being
    indistinguishable from a perfect match on screen.
    """
    field = FIELDS[by]
    scored = [r for r in rows if r.get(field) is not None]
    if by == "distance":
        # A clip outside the recogniser's window is never transcribed, and the
        # aligner then scores an empty hypothesis against a real span: 1.0,
        # the top of the list. Those rows have no transcript and no label, so
        # the page printed "none" against "none" -- which reads as the two
        # being identical, with the worst possible score next to it. They are
        # an absent measurement, not a bad one, and belong in unlabelled().
        scored = [r for r in scored if r.get("asr") and r.get("label")]
    scored.sort(key=lambda r: r[field], reverse=largest)
    return scored[:n]


def by_reciter(rows: Sequence[Dict[str, Any]], reciter: Optional[str]
               ) -> List[Dict[str, Any]]:
    """Only ``reciter``'s rows, or all of them when it is None.

    Reciters are evaluated one at a time because the repair is per reciter:
    segmentation thresholds that suit a fast voice cut a slow one mid-phrase,
    and a corpus-wide list of the longest segments is just the slowest reciter
    thirty times over, which says nothing about anyone else.
    """
    if not reciter:
        return list(rows)
    return [r for r in rows if r.get("reciter_slug") == reciter]


def within(rows: Sequence[Dict[str, Any]], min_seconds: float = 0.0,
           max_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
    """Rows whose duration falls in the window, bounds included."""
    out = []
    for row in rows:
        seconds = row.get("duration_seconds")
        if seconds is None:
            continue
        if seconds < min_seconds:
            continue
        if max_seconds is not None and seconds > max_seconds:
            continue
        out.append(row)
    return out


def reciters(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({r.get("reciter_slug") for r in rows if r.get("reciter_slug")})


def previous_ids(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Each segment's immediate predecessor in its own recording.

    Needed for the commonest repair a short segment wants: a sub-second clip is
    usually the tail of the phrase before it, cut off at a breath the segmenter
    mistook for a stop, and it belongs merged back into that phrase.  A merge
    names the segment that survives, so the button has to know what came
    before -- and it must come from the whole manifest, not the handful of rows
    on the page, because the predecessor is almost never itself an extreme.
    """
    by_source: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row.get("source_id") and row.get("index") is not None:
            by_source[row["source_id"]].append(row)

    out: Dict[str, str] = {}
    for group in by_source.values():
        group.sort(key=lambda r: r["index"])
        for earlier, later in zip(group, group[1:]):
            out[later["segment_id"]] = earlier["segment_id"]
    return out


def unlabelled(rows, n: int):
    """``n`` rows carrying no label, longest first.

    These are not a tail of the distance distribution -- they have no distance
    worth reading, because nothing was transcribed to compare against. Longest
    first because duration is why most of them exist: past the recogniser's
    window a clip is skipped outright.
    """
    missing = [r for r in rows if not r.get("label")]
    missing.sort(key=lambda r: -(r.get("duration_seconds") or 0))
    return missing[:n]


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
        # Shown as a similarity: the stored field is a distance, where 0 means
        # identical, and that reads backwards to every human who sees it.
        out.append(f"match {1.0 - row['align_distance']:.3f}")
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


def _row_block(rank: int, row: Dict[str, Any], data: Optional[bytes],
               previous: Optional[str] = None) -> str:
    if data:
        encoded = base64.b64encode(data).decode("ascii")
        player = (f'<audio controls preload="none" '
                  f'src="data:audio/flac;base64,{encoded}"></audio>')
    else:
        player = '<p class="missing">audio unavailable</p>'

    segment_id = row["segment_id"]
    start = row.get("start_seconds") or 0.0
    end = row.get("end_seconds") or 0.0

    # "Split here" reads the player's position, so the cut is the point the
    # reviewer actually heard.  at_seconds is on the source timeline, hence the
    # clip's own start is added back on.
    merge = ""
    if previous:
        merge = (f'<button data-act="merge" title="absorb this into {previous}">'
                 f'merge into previous</button>')
    buttons = (f'<div class="acts" data-id="{html.escape(segment_id)}" '
               f'data-start="{start:.3f}" data-end="{end:.3f}" '
               f'data-prev="{html.escape(previous or "")}">'
               f'<button data-act="drop">drop</button>'
               f'{merge}'
               f'<button data-act="split">split here</button>'
               f'<span class="state"></span></div>')

    return (f'<section>'
            f'<h2><span class="rank">{rank}</span>'
            f'{html.escape(segment_id)}</h2>'
            f'<p class="meta">{html.escape(_facts(row))}</p>'
            f'{player}'
            f'{_arabic("heard", row.get("asr"))}'
            f'{_arabic("label", row.get("label"))}'
            f'{buttons}'
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
  .acts { display: flex; gap: .4rem; align-items: center; margin-top: .6rem;
          flex-wrap: wrap; }
  .acts button { font: inherit; font-size: .8rem; padding: .2rem .6rem;
                 border: 1px solid rgba(128,128,128,.5); border-radius: 5px;
                 background: transparent; color: inherit; cursor: pointer; }
  .acts button:hover { border-color: currentColor; }
  .acts button.on { background: #b45309; border-color: #b45309; color: #fff; }
  .state { font-size: .78rem; opacity: .7; font-family: ui-monospace, monospace; }
  #tray { position: fixed; right: 1rem; bottom: 1rem; z-index: 9;
          background: Canvas; color: CanvasText; max-width: 22rem;
          border: 1px solid rgba(128,128,128,.5); border-radius: 8px;
          padding: .7rem .9rem; font-size: .82rem;
          box-shadow: 0 6px 24px rgba(0,0,0,.18); }
  #tray[hidden] { display: none; }
  #tray h4 { margin: 0 0 .4rem; font-size: .85rem; }
  #tray ol { margin: 0 0 .6rem; padding-left: 1.2rem; max-height: 11rem;
             overflow: auto; font-family: ui-monospace, monospace;
             font-size: .74rem; }
  #tray button { font: inherit; padding: .3rem .7rem; border-radius: 5px;
                 border: 1px solid currentColor; background: transparent;
                 color: inherit; cursor: pointer; }
"""

#: Collects corrections in the page and writes them out as corrections.jsonl,
#: the format `warsh-data apply-corrections` already reads.  No server: the
#: page is opened from a download, so the file is built with a Blob.
#:
#: Each correction carries orig_start_seconds / orig_end_seconds -- the
#: boundaries actually listened to.  If segmentation is re-run with different
#: thresholds the segment moves, and apply reports the correction as drifted
#: rather than applying it to audio nobody reviewed.
_SCRIPT = """
const picked = new Map();

function draw() {
  const tray = document.getElementById('tray');
  const list = document.getElementById('tray-list');
  tray.hidden = picked.size === 0;
  list.innerHTML = '';
  for (const [id, c] of picked) {
    const li = document.createElement('li');
    li.textContent = c.action + ' ' + id.split('__').slice(-2).join('__');
    list.appendChild(li);
  }
  document.getElementById('tray-count').textContent = picked.size;
}

document.addEventListener('click', (ev) => {
  const button = ev.target.closest('.acts button');
  if (!button) return;
  const box = button.closest('.acts');
  const id = box.dataset.id;
  const act = button.dataset.act;
  const orig = {
    orig_start_seconds: Number(box.dataset.start),
    orig_end_seconds: Number(box.dataset.end),
  };

  // A second click on the active button clears the choice, so a misclick is
  // undone in place rather than by reloading and starting over.
  const already = picked.get(id);
  if (already && already.action === act) {
    picked.delete(id);
  } else if (act === 'drop') {
    picked.set(id, { segment_id: id, action: 'drop', note: 'reviewed by ear', ...orig });
  } else if (act === 'merge') {
    // The surviving record is the earlier one, which absorbs this segment --
    // so the correction is written against the predecessor, not against this.
    picked.set(id, { segment_id: box.dataset.prev, action: 'merge',
                     with: [id], note: 'tail of the previous phrase' });
  } else if (act === 'split') {
    const audio = box.closest('section').querySelector('audio');
    const at = Number(box.dataset.start) + (audio ? audio.currentTime : 0);
    if (!audio || audio.currentTime <= 0.05) {
      box.querySelector('.state').textContent =
        'play to the cut point first, then press split';
      return;
    }
    picked.set(id, { segment_id: id, action: 'split',
                     at_seconds: [Number(at.toFixed(3))],
                     note: 'waqf the segmenter missed', ...orig });
  }

  for (const other of box.querySelectorAll('button')) other.classList.remove('on');
  const chosen = picked.get(id);
  const chose_this = chosen && (chosen.action === act);
  if (chose_this) button.classList.add('on');
  box.querySelector('.state').textContent = chose_this
    ? (chosen.action === 'split'
        ? 'cut at ' + chosen.at_seconds[0].toFixed(2) + 's in the recording'
        : chosen.action === 'merge'
          ? 'into ' + chosen.segment_id.split('__').slice(-1)[0]
          : 'will be dropped')
    : '';
  draw();
});

function save() {
  const lines = [...picked.values()].map((c) => JSON.stringify(c)).join('
');
  const url = URL.createObjectURL(new Blob([lines + '
'],
                                           { type: 'application/x-ndjson' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'corrections.jsonl';
  a.click();
  URL.revokeObjectURL(url);
}
"""


def build_page(sections: Sequence[tuple], clips: Dict[str, bytes], out_path: Path,
               title: str = "Segments worth hearing",
               previous: Optional[Dict[str, str]] = None) -> Path:
    """One self-contained page.  ``sections`` is ``(heading, why, [row, ...])``.

    Audio is embedded as data URIs rather than linked: the page gets opened
    from a Colab download or a file browser, where relative paths to clip files
    are exactly what breaks.
    """
    blocks = []
    for heading, why, rows in sections:
        if not rows:
            continue
        body = "".join(
            _row_block(i, row, clips.get(row["segment_id"]),
                       (previous or {}).get(row["segment_id"]))
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
            f'judging. "match" is 1.00 when the two are identical.</p>\n'
            f'{"".join(blocks)}\n'
            f'<div id="tray" hidden><h4><span id="tray-count">0</span> '
            f'correction(s)</h4><ol id="tray-list"></ol>'
            f'<button onclick="save()">download corrections.jsonl</button></div>\n'
            f'<script>{_SCRIPT}</script>\n</body>\n</html>\n')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
