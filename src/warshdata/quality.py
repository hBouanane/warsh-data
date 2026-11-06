"""Per-reciter corpus health, as two separate questions.

*How well does the label match what was heard?*  The manifest stores
``align_distance``, a normalised Levenshtein between the skeleton of the ASR
transcript and the skeleton of the chosen reference span -- consonants only,
diacritics stripped on both sides.  Being a distance, 0 there means identical,
which reads backwards to everyone including its author.  So every figure here
is reported as ``match = 1 - distance``: 1.00 is a perfect match, 0.00 is
nothing in common.  The stored field keeps its own meaning, because 141k
published rows already carry it.

Reporting a similarity flips which tail matters.  The segments worth finding
are now the *low* ones, so the quantiles are p10 and p01 rather than p90 and
p99.  A
low match is one of two opposite things: a genuinely misplaced span, or a voice
the bootstrap recogniser found hard while the span is perfectly correct.  The
mean cannot tell them apart, but the shape can.  A heavy low tail with a normal
median is a handful of bad segments in good audio; a *depressed* median is the
whole reciter being off, which is the signature worth chasing.

The word-count column separates a third cause.  When the label carries
materially more words than the recogniser produced, the transcript is short --
the recogniser dropped speech, or the span over-claimed -- which inflates the
distance without any of the text being wrong.  That is a different repair from
either of the others.

*Is the segmenter finding this reciter's stops?*  Nothing past 30 s was
transcribed or labelled at all, so those rows are holes rather than data.  Slow,
deliberate reciters pause with less silence than the threshold expects, so their
segments run long and the holes cluster on them.  Duration and distance are
reported apart because they fail independently: a reciter can align beautifully
and still be badly segmented, which is exactly what the corpus shows.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["LabelFit", "WaqfFit", "label_fit", "waqf_fit",
           "format_label_table", "format_waqf_table"]

#: A segment this much longer in the label than in the transcript is treated as
#: the recogniser having dropped speech rather than as a wrong label.  Two words
#: rather than one: a single word's difference is ordinary at a segment edge,
#: where a reciter carries a word across the pause.
SHORT_BY_WORDS = 2

#: Past this the recogniser was not asked for a transcript, so the row has no
#: label at all.  Mirrors Transcriber.max_seconds.
UNLABELLED_OVER = 30.0


def _quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile.  No interpolation: these are reported to three
    decimals and an interpolated value between two real segments is not a
    segment anyone can go and listen to."""
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return sorted(values)[index]


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


@dataclass
class LabelFit:
    """How far a reciter's labels sit from what the recogniser heard."""

    reciter: str
    segments: int
    #: Segments with both a transcript and a distance, i.e. the ones these
    #: numbers actually describe.
    scored: int
    #: All as match = 1 - distance, so 1.00 is identical.
    mean: float
    p50: float
    #: The bad tail: a tenth and a hundredth of segments match worse than these.
    p10: float
    p01: float
    under_065: float
    under_050: float
    #: Share whose label holds >= SHORT_BY_WORDS more words than the transcript.
    transcript_short: float
    #: Share never transcribed -- clips outside the recogniser's duration
    #: window.  Excluded from every distance figure above; see label_fit.
    no_transcript: float
    no_label: float


def label_fit(rows: Sequence[Dict[str, Any]]) -> List[LabelFit]:
    """One row per reciter, worst mean distance first.

    Segments with no transcript are excluded from the distance figures rather
    than counted at their stored value.  A clip outside the recogniser's
    duration window is never sent to it, so the aligner compares an *empty*
    hypothesis against a real reference span and stores a distance of 1.0.
    That number is not a measurement of label fit -- it is the absence of one,
    and left in the distribution it masquerades as the corpus's worst
    misalignment.  Measured on the slow reciters, it was essentially the whole
    tail above 0.50: a segmentation problem wearing an alignment problem's
    clothes.  It is reported as its own column instead.
    """
    by: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[row.get("reciter_slug") or "?"].append(row)

    out = []
    for reciter, group in by.items():
        scored = [1.0 - r["align_distance"] for r in group
                  if r.get("align_distance") is not None and r.get("asr")]
        short = 0
        comparable = 0
        for row in group:
            asr, label = row.get("asr"), row.get("label")
            if not asr or not label:
                continue
            comparable += 1
            if len(label.split()) - len(asr.split()) >= SHORT_BY_WORDS:
                short += 1

        out.append(LabelFit(
            reciter=reciter,
            segments=len(group),
            scored=len(scored),
            mean=sum(scored) / len(scored) if scored else 0.0,
            p50=_quantile(scored, 0.50),
            p10=_quantile(scored, 0.10),
            p01=_quantile(scored, 0.01),
            under_065=_share(sum(1 for m in scored if m <= 0.65), len(scored)),
            under_050=_share(sum(1 for m in scored if m <= 0.50), len(scored)),
            transcript_short=_share(short, comparable),
            no_transcript=_share(sum(1 for r in group if not r.get("asr")),
                                 len(group)),
            no_label=_share(sum(1 for r in group if not r.get("label")), len(group)),
        ))
    return sorted(out, key=lambda f: f.mean)


@dataclass
class WaqfFit:
    """Whether the segmenter is finding a reciter's stops."""

    reciter: str
    segments: int
    p50: float
    p95: float
    p99: float
    longest: float
    over_20: float
    over_30: float
    #: Count, not share: these are the rows to go and re-segment.
    unlabelled_long: int
    hours: float


def waqf_fit(rows: Sequence[Dict[str, Any]]) -> List[WaqfFit]:
    """One row per reciter, longest-tailed first."""
    by: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[row.get("reciter_slug") or "?"].append(row)

    out = []
    for reciter, group in by.items():
        lengths = [r["duration_seconds"] for r in group
                   if r.get("duration_seconds") is not None]
        if not lengths:
            continue
        long_rows = [r for r in group
                     if (r.get("duration_seconds") or 0) > UNLABELLED_OVER]
        out.append(WaqfFit(
            reciter=reciter,
            segments=len(group),
            p50=_quantile(lengths, 0.50),
            p95=_quantile(lengths, 0.95),
            p99=_quantile(lengths, 0.99),
            longest=max(lengths),
            over_20=_share(sum(1 for d in lengths if d > 20.0), len(lengths)),
            over_30=_share(len(long_rows), len(lengths)),
            unlabelled_long=sum(1 for r in long_rows if not r.get("label")),
            hours=sum(lengths) / 3600.0,
        ))
    return sorted(out, key=lambda f: -f.over_20)


def format_label_table(fits: Sequence[LabelFit]) -> str:
    head = (f"{'reciter':<28}{'scored':>7}{'mean':>7}{'p50':>7}{'p10':>7}"
            f"{'p01':>7}{'<=.65':>7}{'<=.50':>7}{'short':>7}{'notx':>7}"
            f"{'nolab':>7}")
    lines = [head, "-" * len(head)]
    for f in fits:
        lines.append(
            f"{f.reciter:<28}{f.scored:>7}{f.mean:>7.3f}{f.p50:>7.3f}"
            f"{f.p10:>7.3f}{f.p01:>7.3f}{f.under_065:>6.1%}{f.under_050:>7.1%}"
            f"{f.transcript_short:>7.1%}{f.no_transcript:>7.1%}"
            f"{f.no_label:>7.1%}")
    return "\n".join(lines)


def format_waqf_table(fits: Sequence[WaqfFit]) -> str:
    head = (f"{'reciter':<28}{'segs':>7}{'hours':>7}{'p50':>7}{'p95':>7}"
            f"{'p99':>7}{'max':>7}{'>20s':>7}{'>30s':>7}{'holes':>7}")
    lines = [head, "-" * len(head)]
    for f in fits:
        lines.append(
            f"{f.reciter:<28}{f.segments:>7}{f.hours:>7.1f}{f.p50:>7.1f}"
            f"{f.p95:>7.1f}{f.p99:>7.1f}{f.longest:>7.1f}"
            f"{f.over_20:>6.1%}{f.over_30:>7.1%}{f.unlabelled_long:>7}")
    return "\n".join(lines)


def verdicts(labels: Sequence[LabelFit], waqf: Sequence[WaqfFit],
             median_mean: Optional[float] = None) -> List[str]:
    """Short notes on reciters that stand out, measured against the corpus.

    Thresholds are relative, not absolute: what counts as a bad mean depends on
    how hard the bootstrap recogniser finds this reading overall, and a fixed
    number would either flag everyone or no one on a different corpus.
    """
    notes = []
    means = sorted(f.mean for f in labels)
    middle = median_mean if median_mean is not None else _quantile(means, 0.50)
    middle_p50 = _quantile([g.p50 for g in labels], 0.50)

    for f in labels:
        # Measured as a shortfall from the corpus, since "bad" is now low.
        if middle and (1 - f.mean) >= 2.0 * (1 - middle) and                 (1 - f.p50) >= 1.5 * (1 - middle_p50):
            notes.append(
                f"{f.reciter}: mean match {f.mean:.3f} and median {f.p50:.3f} "
                f"both well below the corpus -- a depressed median is the whole "
                f"reciter being off, not a few bad segments; check the reading "
                f"is really Warsh")
        elif f.transcript_short >= 0.10:
            notes.append(
                f"{f.reciter}: {f.transcript_short:.0%} of labels hold "
                f"{SHORT_BY_WORDS}+ more words than the transcript -- the "
                f"recogniser is dropping speech, so the distance overstates how "
                f"wrong the labels are")

    for f in waqf:
        # Long segments only matter when they are actually costing data.  A
        # reciter can sit at 10% over 20 s with nothing past 30 s and align
        # better than anyone -- flagging that is noise, and the first version
        # of this did exactly that for the corpus's cleanest voice.
        if f.unlabelled_long and f.over_20 >= 0.08:
            notes.append(
                f"{f.reciter}: {f.over_20:.0%} of segments over 20 s and "
                f"{f.unlabelled_long} past 30 s with no label -- waqf is being "
                f"missed; re-segment with a lower --min-silence-duration-ms")
    return notes
