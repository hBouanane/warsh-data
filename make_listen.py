"""Build a listening page for hand-checking the aligner on real segments.

For each chosen segment: the audio, what the ASR heard, the reference text the
aligner assigned, and the surrounding Quran context with that span marked. The
point is to answer by ear the one question neither the ASR distance nor the old
labels can settle -- does the assigned text match what the reciter actually
said, and do the boundaries fall where he stopped.
"""

import base64
import html
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "../warsh-lab/src")

from warshdata import quran

CONTEXT = 12             # reference words shown either side of the span

# segment_index does not map to a dataset offset by any fixed shift -- it has
# gaps, and the drift grows through the surah -- so the offsets are read once
# and looked up by id.
OFFSETS = json.loads(Path("offsets.json").read_text(encoding="utf-8"))


def fetch_row(segment_id: str) -> dict:
    offset = OFFSETS[segment_id]
    url = ("https://datasets-server.huggingface.co/rows"
           "?dataset=Haitam03%2Fwarsh-segments&config=default&split=train"
           f"&offset={offset}&length=1")
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = json.load(response)
    return payload["rows"][0]["row"]


def download(url: str, dest: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "warsh-data/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    dest.write_bytes(data)
    return data


def main() -> int:
    picks = json.loads(Path("picks.json").read_text(encoding="utf-8"))
    aligned = {p["idx"]: p for p in
               json.loads(Path("aligned.json").read_text(encoding="utf-8"))}
    surah = quran.load()[2]

    out_dir = Path("listen_real")
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    blocks = []
    for entry in picks:
        index, note = entry["idx"], entry["note"]
        info = aligned[index]
        row = fetch_row(info["segment_id"])
        if row["segment_id"] != info["segment_id"]:
            print(f"  !! offset mismatch at {index}: {row['segment_id']}")
            continue

        sources = row.get("audio") or []
        if not sources:
            print(f"  !! no audio for {info['segment_id']}")
            continue
        clip = out_dir / "clips" / info["segment_id"]
        data = download(sources[0]["src"], clip)
        print(f"  {info['segment_id']}  {len(data)/1000:.0f} kB")

        start, end = info["mine"]
        before = surah.label(max(0, start - CONTEXT), start)
        span = surah.label(start, end)
        after = surah.label(end, min(len(surah.words), end + CONTEXT))
        encoded = base64.b64encode(data).decode("ascii")

        agree = ("matches your previous label" if info["agree"]
                 else f"differs -- you had {info['theirs']}")
        blocks.append(f"""
<section>
  <h2>{html.escape(info['segment_id'])}</h2>
  <p class="meta">segment {index} &middot; {info['dur']}s &middot;
     span {info['mine']} &middot; verses {info['verses']} &middot;
     distance {info['my_dist']} &middot; {html.escape(agree)}</p>
  <p class="why">{html.escape(note)}</p>
  <audio controls preload="none" src="data:audio/mpeg;base64,{encoded}"></audio>
  <p class="lbl">ASR heard</p><p class="ar asr">{html.escape(info['asr'] or '')}</p>
  <p class="lbl">aligner assigned</p>
  <p class="ar ctx"><span class="dim">{html.escape(before)}</span>
     <span class="hit">{html.escape(span)}</span>
     <span class="dim">{html.escape(after)}</span></p>
</section>""")

    page = f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aligner check</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.7 system-ui, sans-serif; max-width: 56rem;
        margin-inline: auto; padding: 2rem 1rem; }}
 h1 {{ font-size: 1.3rem; }}
 section {{ border-top: 1px solid rgba(128,128,128,.35); padding: 1.2rem 0; }}
 h2 {{ font-size: 1rem; font-family: ui-monospace, monospace; margin: 0 0 .3rem;
      direction: ltr; text-align: left; }}
 .meta, .why {{ direction: ltr; text-align: left; font-size: .85rem; }}
 .meta {{ opacity: .6; font-family: ui-monospace, monospace; margin: .2rem 0; }}
 .why {{ opacity: .85; margin: .2rem 0 .7rem; }}
 .lbl {{ direction: ltr; text-align: left; font-size: .75rem; letter-spacing: .05em;
        text-transform: uppercase; opacity: .55; margin: .8rem 0 .1rem; }}
 audio {{ width: 100%; height: 34px; }}
 .ar {{ font-size: 1.5rem; line-height: 2.1; margin: .2rem 0; }}
 .asr {{ opacity: .8; }}
 .dim {{ opacity: .35; }}
 .hit {{ background: rgba(120,180,255,.28); border-radius: .2rem;
        padding: .1rem .15rem; }}
</style></head><body>
<h1>Aligner check &mdash; Al-Baqarah, A-Benkirane</h1>
<p class="why" dir="ltr">Highlighted text is what the aligner assigned; greyed text
is the surrounding Quran context. Listen and judge whether the highlight is what
he recited, and whether it starts and ends where he did.</p>
{''.join(blocks)}
</body></html>"""

    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    print(f"\nwrote {target}  ({target.stat().st_size/1e6:.1f} MB, {len(blocks)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
