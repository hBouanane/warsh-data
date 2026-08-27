"""A listening page for waqf overlap: consecutive segments that share words.

A reciter who pauses often carries the last few words into the next segment, or
repeats them outright, so the same reference words genuinely belong to two
recordings.  Each block here is a *pair*: both clips, and the reference text with
each segment's span marked and the shared middle marked separately, so the
repetition can be heard in one clip and then again in the next.

These are also the cases the previous aligner handled least well, which is not a
coincidence -- an aligner that forces segments to partition the text has to clip
one side or the other exactly here.
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

CONTEXT = 8
OFFSETS = json.loads(Path("offsets.json").read_text(encoding="utf-8"))


def fetch_row(segment_id: str) -> dict:
    url = ("https://datasets-server.huggingface.co/rows"
           "?dataset=Haitam03%2Fwarsh-segments&config=default&split=train"
           f"&offset={OFFSETS[segment_id]}&length=1")
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)["rows"][0]["row"]


def audio_tag(segment_id: str, cache: dict) -> str:
    if segment_id not in cache:
        row = fetch_row(segment_id)
        source = (row.get("audio") or [{}])[0].get("src")
        if not source:
            return "<p>(no audio)</p>"
        request = urllib.request.Request(source, headers={"User-Agent": "warsh-data/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            cache[segment_id] = response.read()
        print(f"  {segment_id}  {len(cache[segment_id])/1000:.0f} kB", flush=True)
    encoded = base64.b64encode(cache[segment_id]).decode("ascii")
    return f'<audio controls preload="none" src="data:audio/mpeg;base64,{encoded}"></audio>'


def main() -> int:
    hard = json.loads(Path("hard.json").read_text(encoding="utf-8"))
    aligned = {p["idx"]: p for p in
               json.loads(Path("aligned.json").read_text(encoding="utf-8"))}
    surah = quran.load()[2]
    cache: dict = {}
    blocks = []

    for share, first_idx, second_idx in hard["overlaps"][:5]:
        a, b = aligned[first_idx], aligned[second_idx]
        a_start, a_end = a["mine"]
        b_start, b_end = b["mine"]

        before = surah.label(max(0, a_start - CONTEXT), a_start)
        only_a = surah.label(a_start, b_start)
        shared = surah.label(b_start, a_end)
        only_b = surah.label(a_end, b_end)
        after = surah.label(b_end, min(len(surah.words), b_end + CONTEXT))

        blocks.append(f"""
<section>
  <h2>{html.escape(a['segment_id'])} &nbsp;+&nbsp; {html.escape(b['segment_id'])}</h2>
  <p class="meta">{share} reference words recited in both clips &middot;
     spans {a['mine']} and {b['mine']}</p>
  <p class="lbl">clip 1 &mdash; listen for the ending</p>{audio_tag(a['segment_id'], cache)}
  <p class="lbl">clip 2 &mdash; the same words again at the start</p>{audio_tag(b['segment_id'], cache)}
  <p class="ar"><span class="dim">{html.escape(before)}</span>
    <span class="one">{html.escape(only_a)}</span>
    <span class="both">{html.escape(shared)}</span>
    <span class="two">{html.escape(only_b)}</span>
    <span class="dim">{html.escape(after)}</span></p>
  <p class="key" dir="ltr"><span class="one">clip 1 only</span>
     <span class="both">both clips</span> <span class="two">clip 2 only</span></p>
</section>""")

    for index in hard["hard"][:5]:
        info = aligned[index]
        start, end = info["mine"]
        blocks.append(f"""
<section>
  <h2>{html.escape(info['segment_id'])}</h2>
  <p class="meta">lowest confidence from the previous aligner &middot;
     span {info['mine']} &middot; my distance {info['my_dist']}</p>
  {audio_tag(info['segment_id'], cache)}
  <p class="lbl">ASR heard</p><p class="ar asr">{html.escape(info['asr'] or '')}</p>
  <p class="ar"><span class="dim">{html.escape(surah.label(max(0, start-CONTEXT), start))}</span>
    <span class="both">{html.escape(surah.label(start, end))}</span>
    <span class="dim">{html.escape(surah.label(end, min(len(surah.words), end+CONTEXT)))}</span></p>
</section>""")

    page = f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Waqf overlap</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.7 system-ui, sans-serif; max-width: 56rem;
        margin-inline: auto; padding: 2rem 1rem; }}
 h1 {{ font-size: 1.3rem; }}
 section {{ border-top: 1px solid rgba(128,128,128,.35); padding: 1.3rem 0; }}
 h2 {{ font-size: .95rem; font-family: ui-monospace, monospace; margin: 0 0 .3rem;
      direction: ltr; text-align: left; }}
 .meta, .lbl, .key, .intro {{ direction: ltr; text-align: left; }}
 .meta {{ opacity: .6; font-size: .85rem; font-family: ui-monospace, monospace; }}
 .lbl {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .05em;
        opacity: .55; margin: .8rem 0 .15rem; }}
 audio {{ width: 100%; height: 34px; }}
 .ar {{ font-size: 1.5rem; line-height: 2.2; margin: .9rem 0 .3rem; }}
 .asr {{ font-size: 1.25rem; opacity: .75; }}
 .dim {{ opacity: .3; }}
 .one  {{ background: rgba(120,180,255,.28); border-radius:.2rem; padding:.1rem .15rem; }}
 .both {{ background: rgba(255,190,90,.42); border-radius:.2rem; padding:.1rem .15rem; }}
 .two  {{ background: rgba(140,220,150,.32); border-radius:.2rem; padding:.1rem .15rem; }}
 .key span {{ font-size: .75rem; margin-right: .4rem; padding: .1rem .3rem; }}
</style></head><body>
<h1>Waqf overlap &mdash; Al-Baqarah, A-Benkirane</h1>
<p class="intro">The orange words are recited in <em>both</em> clips: the reciter
carries them across the pause. An aligner that makes segments partition the text
must clip one side or the other here, which is why these were the hardest cases.
Below the pairs are the five segments the previous aligner was least sure of.</p>
{''.join(blocks)}
</body></html>"""

    target = Path("listen_pairs.html")
    target.write_text(page, encoding="utf-8")
    print(f"\nwrote {target} ({target.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
