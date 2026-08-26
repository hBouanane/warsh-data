"""Aligning a sequence of ASR transcripts against a surah's reference text.

The naive approach -- search the surah for each segment's transcript -- cannot
work.  Ar-Rahman repeats one verse 31 times, Al-Mursalat repeats another 10, and
plenty of surahs carry verses that differ by a single word.  Searched
independently, every copy is an equally good match.

So nothing is searched independently.  A reciter reads start to finish, so the
segments are *in order*, and the problem is a monotonic alignment between two
sequences: the surah's words, and the concatenated transcript words.  Under
monotonicity a repeated verse is unambiguous -- the seventeenth refrain is the
one that comes after the sixteenth, identified by its neighbours rather than its
content.

Three decisions carry most of the accuracy:

*Global, not greedy.*  A cursor walking forward commits to each decision with
only local information, and a wrong commit corrupts everything after it.  Global
dynamic programming evaluates all monotonic alignments at once, so a locally
tempting mistake loses on the whole-sequence score and is simply never chosen.
Errors do not propagate, because there is no "from here on".

*Substitution costs are graded, not binary.*  With a binary match/mismatch, a
mangled word and a genuinely different word look identical -- which is exactly
what makes near-duplicate verses dangerous.  Costing a substitution by the
character distance between the two words separates them: ASR noise is cheap,
a real difference is expensive, and the alignment prefers the copy whose
neighbours actually agree.

*Matching happens on the rasm.*  Short vowels are what an ASR gets wrong most,
and two verses differing only in vowelling should be told apart by context, not
by marks the model probably guessed.  Diacritics are scored afterwards instead.

Anchoring keeps it fast.  Word n-grams that occur exactly once in both sequences
pin the alignment; a longest-increasing-subsequence pass keeps only a mutually
consistent set, and full DP then runs on the small gaps between them.  In
Ar-Rahman the unique verses anchor either side of every refrain, so the hardest
case becomes one of the cheapest.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Aligned", "Alignment", "AlignConfig", "align_surah", "align_words",
           "ISTIADHA", "BASMALA", "CLOSING", "strip_formulas"]

#: Spoken before a recitation but not part of the text anywhere.  Several
#: wordings are current, so the common ones are all listed.
ISTIADHA = (
    "أعوذ بالله من "
    "الشيطان الرجيم",
    "أعوذ بالله السميع "
    "العليم من الشيطان "
    "الرجيم",
)

#: Spoken before every surah except At-Tawbah.  It is genuine reference text in
#: two places -- verse 1:1, and inside 27:30 -- so it is only ever stripped from
#: the *opening* run of a recitation, never searched for globally.
BASMALA = "بسم الله الرحمن الرحيم"

#: Sometimes spoken after finishing.
CLOSING = (
    "صدق الله العظيم",
)

#: Cost of dropping a word from either sequence.  Substitutions are scored on a
#: 0..1 scale, so a gap at 1.0 is never cheaper than even the worst substitution
#: -- which keeps the alignment from silently skipping material.
GAP_COST = 1.0


@dataclass
class AlignConfig:
    #: n-gram length for anchors.  Four words is long enough to be unique in
    #: most surahs and short enough to survive a couple of ASR errors.
    anchor_n: int = 4
    #: Maximum normalised distance for an anchor n-gram to count as matched.
    anchor_max_distance: float = 0.25
    #: Don't bother anchoring inside gaps already smaller than this.
    min_gap_to_split: int = 40
    #: A segment whose transcript is this far from its aligned reference text is
    #: reported as failed rather than trusted.
    max_segment_distance: float = 0.5
    #: How close a leading phrase must be to a known formula to be stripped.
    formula_max_distance: float = 0.35
    #: Strip isti'adha / basmala from the opening and sadaqallah from the close.
    strip_opening: bool = True


@dataclass
class Aligned:
    """Where one segment landed in the reference."""

    index: int
    ref_start: int
    ref_end: int
    label: str
    rasm: str
    verses: List[int]
    distance: float
    ok: bool
    hypothesis: str = ""
    #: True when this segment was an opening or closing formula, not recitation.
    formula: bool = False
    #: True when this segment re-recites the span a neighbour already covered.
    repeat: bool = False

    @property
    def word_count(self) -> int:
        return self.ref_end - self.ref_start


@dataclass
class Alignment:
    surah: int
    segments: List[Aligned]
    coverage: float
    distance: float
    anchors: int = 0
    unaligned: List[int] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.segments if s.ok)


def _distance(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.normalized_distance(a, b)


def _cost_matrix(ref: Sequence[str], hyp: Sequence[str]):
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    return cdist(ref, hyp, scorer=Levenshtein.normalized_distance, workers=-1)


def _dp_align(ref: Sequence[str], hyp: Sequence[str]) -> List[Optional[int]]:
    """Globally optimal monotonic alignment of ``hyp`` onto ``ref``.

    Returns, for each hypothesis word, the reference index it matched, or None
    where it was an insertion.
    """
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return [None] * m

    import numpy as np

    cost = _cost_matrix(ref, hyp)

    # dp[i][j] = best cost aligning ref[:i] with hyp[:j].  Kept as a full table
    # because the traceback needs it; the gaps between anchors are small.
    dp = np.empty((n + 1, m + 1), dtype=np.float32)
    dp[0, :] = np.arange(m + 1) * GAP_COST
    dp[:, 0] = np.arange(n + 1) * GAP_COST

    for i in range(1, n + 1):
        row_cost = cost[i - 1]
        prev, cur = dp[i - 1], dp[i]
        for j in range(1, m + 1):
            sub = prev[j - 1] + row_cost[j - 1]
            delete = prev[j] + GAP_COST
            insert = cur[j - 1] + GAP_COST
            cur[j] = sub if sub <= delete and sub <= insert else min(delete, insert)

    mapping: List[Optional[int]] = [None] * m
    i, j = n, m
    while i > 0 and j > 0:
        sub = dp[i - 1, j - 1] + cost[i - 1, j - 1]
        if abs(dp[i, j] - sub) < 1e-6:
            mapping[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif abs(dp[i, j] - (dp[i - 1, j] + GAP_COST)) < 1e-6:
            i -= 1
        else:
            j -= 1
    return mapping


def _ngram_positions(words: Sequence[str], n: int) -> Dict[Tuple[str, ...], List[int]]:
    positions: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i in range(len(words) - n + 1):
        positions[tuple(words[i:i + n])].append(i)
    return positions


def _find_anchors(ref: Sequence[str], hyp: Sequence[str], config: AlignConfig
                  ) -> List[Tuple[int, int]]:
    """Pairs ``(ref_index, hyp_index)`` that pin the alignment.

    Only n-grams unique on *both* sides qualify, so a repeated refrain can never
    become an anchor -- it is the unique verses around it that do the pinning.
    """
    n = config.anchor_n
    if len(ref) < n or len(hyp) < n:
        return []

    ref_positions = _ngram_positions(ref, n)
    hyp_positions = _ngram_positions(hyp, n)

    candidates: List[Tuple[int, int]] = []
    for gram, refs in ref_positions.items():
        if len(refs) != 1:
            continue
        hyps = hyp_positions.get(gram)
        if hyps and len(hyps) == 1:
            candidates.append((refs[0], hyps[0]))

    if not candidates:
        return []

    # Keep the largest monotonically increasing subset: any anchor that would
    # require going backwards is inconsistent with a reciter reading in order,
    # so it is dropped rather than trusted.
    candidates.sort()
    tails: List[int] = []
    prev: List[int] = []
    index_of: List[int] = []
    for position, (_, hyp_index) in enumerate(candidates):
        slot = bisect_left(tails, hyp_index)
        if slot == len(tails):
            tails.append(hyp_index)
            index_of.append(position)
        else:
            tails[slot] = hyp_index
            index_of[slot] = position
        prev.append(index_of[slot - 1] if slot else -1)

    chain: List[Tuple[int, int]] = []
    cursor = index_of[-1] if index_of else -1
    while cursor >= 0:
        chain.append(candidates[cursor])
        cursor = prev[cursor]
    chain.reverse()
    return chain


def strip_formulas(
    words: Sequence[str],
    formulas: Sequence[Sequence[str]],
    max_distance: float,
    from_end: bool = False,
) -> int:
    """How many words at one end are an opening/closing formula.

    Applied only to the leading (or trailing) run, never searched for globally:
    the basmala is genuine reference text inside 27:30, and a blanket strip
    would delete a real verse.  Formulas are matched repeatedly, since a reciter
    may say isti'adha *and* basmala before starting.
    """
    if not words or not formulas:
        return 0

    taken = 0
    remaining = list(words[::-1] if from_end else words)

    while remaining:
        matched = 0
        for formula in formulas:
            size = len(formula)
            if size == 0 or len(remaining) < size:
                continue
            head = remaining[:size]
            candidate = " ".join(head[::-1] if from_end else head)
            target = " ".join(formula)
            if _distance(candidate, target) <= max_distance:
                matched = size
                break
        if not matched:
            break
        taken += matched
        remaining = remaining[matched:]

    return taken


def align_words(ref: Sequence[str], hyp: Sequence[str],
                config: Optional[AlignConfig] = None) -> Tuple[List[Optional[int]], int]:
    """Align two word sequences.  Returns ``(mapping, anchor_count)``."""
    config = config or AlignConfig()
    anchors = _find_anchors(ref, hyp, config)

    if not anchors:
        return _dp_align(ref, hyp), 0

    mapping: List[Optional[int]] = [None] * len(hyp)
    ref_cursor = hyp_cursor = 0

    for ref_index, hyp_index in anchors + [(len(ref), len(hyp))]:
        if ref_index < ref_cursor or hyp_index < hyp_cursor:
            continue  # already covered by a previous anchor's span
        gap = _dp_align(ref[ref_cursor:ref_index], hyp[hyp_cursor:hyp_index])
        for offset, target in enumerate(gap):
            if target is not None:
                mapping[hyp_cursor + offset] = ref_cursor + target

        if ref_index < len(ref):
            for k in range(config.anchor_n):
                if hyp_index + k < len(hyp) and ref_index + k < len(ref):
                    mapping[hyp_index + k] = ref_index + k
            ref_cursor = ref_index + config.anchor_n
            hyp_cursor = hyp_index + config.anchor_n

    return mapping, len(anchors)


def align_surah(
    surah,
    transcripts: Sequence[str],
    config: Optional[AlignConfig] = None,
    normalizer=None,
) -> Alignment:
    """Align a sequence of segment transcripts against one surah.

    ``transcripts`` must be in recitation order -- that ordering is the
    information that makes repeated verses tractable.
    """
    config = config or AlignConfig()
    if normalizer is None:
        from warshlab import text as T

        def normalizer(value: str) -> List[str]:
            return [w for w in T.words(T.to_rasm(value)) if w]

    hyp_words: List[str] = []
    owner: List[int] = []
    for index, transcript in enumerate(transcripts):
        for word in normalizer(transcript):
            hyp_words.append(word)
            owner.append(index)

    formula_words: set = set()
    if config.strip_opening:
        openings = [normalizer(f) for f in ISTIADHA]
        # In Al-Fatiha the basmala *is* verse 1, so stripping it would delete
        # real reference text.  Everywhere else it precedes the text.
        if surah.number != 1:
            openings.append(normalizer(BASMALA))
        openings = [f for f in openings if f]

        lead = strip_formulas(hyp_words, openings, config.formula_max_distance)
        tail = strip_formulas(hyp_words, [normalizer(f) for f in CLOSING],
                              config.formula_max_distance, from_end=True)
        formula_words = set(range(lead)) | set(range(len(hyp_words) - tail, len(hyp_words)))

    kept = [i for i in range(len(hyp_words)) if i not in formula_words]
    mapping_kept, anchor_count = align_words(
        surah.words, [hyp_words[i] for i in kept], config)

    mapping: List[Optional[int]] = [None] * len(hyp_words)
    for position, hyp_index in enumerate(kept):
        mapping[hyp_index] = mapping_kept[position]

    formula_segments = {
        owner[i] for i in formula_words
    } - {owner[i] for i in kept}

    spans: Dict[int, List[int]] = defaultdict(list)
    for hyp_index, ref_index in enumerate(mapping):
        if ref_index is not None:
            spans[owner[hyp_index]].append(ref_index)

    results: List[Aligned] = []
    unaligned: List[int] = []
    covered = set()

    for index, transcript in enumerate(transcripts):
        hits = spans.get(index)
        if not hits:
            is_formula = index in formula_segments
            if not is_formula:
                unaligned.append(index)
            results.append(Aligned(index=index, ref_start=-1, ref_end=-1, label="",
                                   rasm="", verses=[], distance=0.0 if is_formula else 1.0,
                                   ok=is_formula, hypothesis=transcript,
                                   formula=is_formula))
            continue

        start, end = min(hits), max(hits) + 1
        covered.update(range(start, end))
        ref_rasm = surah.rasm(start, end)
        hyp_rasm = " ".join(normalizer(transcript))
        distance = _distance(hyp_rasm, ref_rasm)

        results.append(Aligned(
            index=index,
            ref_start=start,
            ref_end=end,
            label=surah.label(start, end),
            rasm=ref_rasm,
            verses=surah.verses_spanned(start, end),
            distance=distance,
            ok=distance <= config.max_segment_distance,
            hypothesis=transcript,
        ))

    # A reciter who stops and repeats produces two segments of the same audio.
    # Monotonic alignment can only give the reference span to one of them, so the
    # other comes back empty; it is not misaligned, it is a repeat, and it wants
    # the same label.  Recovered here rather than allowed in the DP, where going
    # backwards would open the door to the very errors monotonicity prevents.
    for position, result in enumerate(results):
        if result.ref_start >= 0 or result.formula:
            continue
        hyp_rasm = " ".join(normalizer(result.hypothesis))
        if not hyp_rasm:
            continue
        for neighbour in (position - 1, position + 1):
            if not (0 <= neighbour < len(results)):
                continue
            other = results[neighbour]
            if other.ref_start < 0:
                continue
            distance = _distance(hyp_rasm, other.rasm)
            if distance <= config.max_segment_distance:
                results[position] = Aligned(
                    index=result.index, ref_start=other.ref_start, ref_end=other.ref_end,
                    label=other.label, rasm=other.rasm, verses=list(other.verses),
                    distance=distance, ok=True, hypothesis=result.hypothesis,
                    repeat=True,
                )
                if result.index in unaligned:
                    unaligned.remove(result.index)
                break

    coverage = len(covered) / len(surah.words) if surah.words else 0.0
    scored = [r.distance for r in results if r.ref_start >= 0]
    return Alignment(
        surah=surah.number,
        segments=results,
        coverage=coverage,
        distance=sum(scored) / len(scored) if scored else 1.0,
        anchors=anchor_count,
        unaligned=unaligned,
    )
