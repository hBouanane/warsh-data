"""Simulating ASR output, so the aligner can be measured before a model exists.

Take the real Warsh text, cut it at plausible waqf points, then damage each
piece the way a weak recogniser would -- substituting, dropping and inserting
characters at a chosen error rate.  Feed the damaged text to the aligner and
check whether each piece landed back where it came from.

Because the ground truth is known by construction, this gives an accuracy curve
against transcript quality without a single GPU-hour, and answers the question
that actually gates the pipeline: *how good does the bootstrap ASR have to be
before alignment works?*

The generator also reproduces the two things that break naive matching in real
recordings -- a reciter repeating a few words after a stop, and extra speech at
the start that is not in the text at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = ["Segmented", "segment_surah", "corrupt", "make_case", "score_case"]

#: Letters used when inventing a wrong character.  Drawn from the Arabic
#: alphabet so the damage resembles a recogniser's confusions rather than noise.
_ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


@dataclass
class Segmented:
    """A surah cut into pseudo-waqf segments, with the truth kept."""

    surah_number: int
    transcripts: List[str]
    #: Reference word range each transcript really came from.
    truth: List[Tuple[int, int]]
    clean: List[str] = field(default_factory=list)


def segment_surah(surah, rng: random.Random,
                  min_words: int = 3, max_words: int = 12) -> Segmented:
    """Cut a surah into segments of a few words, respecting verse ends.

    Real waqf segments mostly end at a verse boundary but not always, so cuts
    fall at verse ends when one is near and mid-verse otherwise.
    """
    verse_ends = {v.word_end for v in surah.verses}
    transcripts: List[str] = []
    truth: List[Tuple[int, int]] = []

    cursor = 0
    total = len(surah.words)
    while cursor < total:
        span = rng.randint(min_words, max_words)
        end = min(cursor + span, total)
        # Snap to a nearby verse end, which is where a reciter usually stops.
        for candidate in range(end, min(end + 4, total + 1)):
            if candidate in verse_ends:
                end = candidate
                break
        if end <= cursor:
            end = min(cursor + 1, total)
        transcripts.append(surah.rasm(cursor, end))
        truth.append((cursor, end))
        cursor = end

    return Segmented(surah_number=surah.number, transcripts=list(transcripts),
                     truth=truth, clean=list(transcripts))


def corrupt(text: str, rate: float, rng: random.Random) -> str:
    """Damage ``text`` to roughly ``rate`` character error rate."""
    if rate <= 0:
        return text

    out: List[str] = []
    for char in text:
        if char == " " or rng.random() >= rate:
            out.append(char)
            continue
        roll = rng.random()
        if roll < 0.5:                      # substitution
            out.append(rng.choice(_ALPHABET))
        elif roll < 0.8:                    # deletion
            continue
        else:                               # insertion
            out.append(char)
            out.append(rng.choice(_ALPHABET))
    return "".join(out).strip() or text


def make_case(
    surah,
    rate: float,
    seed: int = 0,
    repeat_segments: int = 0,
    intro_words: int = 0,
    opening: bool = False,
    closing: bool = False,
    drop_words: float = 0.0,
) -> Segmented:
    """Build one simulated recording of a surah.

    ``repeat_segments`` re-recites that many randomly chosen segments, the way a
    reciter does after a stop.  ``intro_words`` prepends speech that is not in
    the reference at all.
    """
    rng = random.Random(seed)
    base = segment_surah(surah, rng)

    transcripts = list(base.transcripts)
    truth = list(base.truth)

    if opening:
        # What reciters actually say before starting, and what the reference
        # text does not contain: isti'adha, then basmala.
        from .align import BASMALA, ISTIADHA

        for phrase in (ISTIADHA[0], BASMALA):
            transcripts.insert(0, phrase)
            truth.insert(0, (-1, -1))

    if closing:
        from .align import CLOSING

        transcripts.append(CLOSING[0])
        truth.append((-1, -1))

    for _ in range(repeat_segments):
        if len(transcripts) < 2:
            break
        at = rng.randrange(len(transcripts))
        # A repeat is extra audio, not extra text: it must align to the same
        # reference span the original did.
        transcripts.insert(at + 1, transcripts[at])
        truth.insert(at + 1, truth[at])

    if intro_words:
        filler = " ".join("".join(rng.choice(_ALPHABET) for _ in range(rng.randint(3, 6)))
                          for _ in range(intro_words))
        transcripts.insert(0, filler)
        truth.insert(0, (-1, -1))

    clean = list(transcripts)
    damaged = [corrupt(t, rate, rng) for t in transcripts]
    if drop_words:
        # Whole-word deletions, which a real recogniser does far more than a
        # uniform character process ever will.
        damaged = [
            " ".join(w for w in t.split() if rng.random() >= drop_words) or t
            for t in damaged
        ]
    return Segmented(surah_number=surah.number, transcripts=damaged,
                     truth=truth, clean=clean)


def score_case(case: Segmented, alignment, tolerance: int = 1) -> dict:
    """Compare an alignment against the known truth.

    ``tolerance`` allows a boundary to be off by that many words before it
    counts as wrong -- an off-by-one at a segment edge is not a misalignment in
    any sense that matters downstream.
    """
    exact = near = wrong = skipped = 0

    for aligned, (true_start, true_end) in zip(alignment.segments, case.truth):
        if true_start < 0:            # injected filler: correctly has no home
            skipped += 1
            continue
        if aligned.ref_start < 0:
            wrong += 1
            continue
        if aligned.ref_start == true_start and aligned.ref_end == true_end:
            exact += 1
        elif (abs(aligned.ref_start - true_start) <= tolerance
              and abs(aligned.ref_end - true_end) <= tolerance):
            near += 1
        else:
            wrong += 1

    scored = exact + near + wrong
    return {
        "segments": len(case.transcripts),
        "exact": exact,
        "near": near,
        "wrong": wrong,
        "filler": skipped,
        "accuracy": (exact + near) / scored if scored else 0.0,
        "coverage": alignment.coverage,
        "anchors": alignment.anchors,
    }
