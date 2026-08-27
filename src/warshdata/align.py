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

*Matching happens on the skeleton.*  Short vowels are what an ASR gets wrong
most, and two verses differing only in vowelling should be told apart by context
rather than by marks the model probably guessed.  Unifying letter shapes on top
of that absorbs the orthographic gap between a Hafs-trained recogniser and a
Warsh reference -- worth 35 points of word error on real output.  Diacritics are
scored afterwards instead, and labels still come from the fully pointed text.

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
    #: Word anchors are only added inside gaps at least this wide.  Swept
    #: empirically: at 40 they fire on text that already has plenty of n-gram
    #: anchors and cost accuracy there; at 120 they reach the refrain-heavy
    #: surahs that need them (+4.2 pts on Ar-Rahman) for 0.4 of a point
    #: elsewhere.
    min_gap_to_split: int = 120
    #: Also anchor on single words that occur exactly once in the surah.  Exact
    #: n-gram anchors vanish once the error rate is high enough that no four
    #: consecutive words survive intact; one distinctive long word often does.
    word_anchors: bool = True
    #: Shortest word eligible to be a word anchor.  Short words are common and
    #: easily confused, so they carry little evidence.
    word_anchor_min_len: int = 5
    #: Maximum distance for a word anchor to count as matched.
    word_anchor_max_distance: float = 0.34
    #: A word anchor is only trusted when the runner-up is this much worse,
    #: which is what stops a refrain word from anchoring to the wrong copy.
    word_anchor_margin: float = 0.15
    #: A segment whose transcript is this far from its aligned reference text is
    #: reported as failed rather than trusted.
    max_segment_distance: float = 0.5
    #: How close a leading phrase must be to a known formula to be stripped.
    formula_max_distance: float = 0.35
    #: Strip isti'adha / basmala from the opening and sadaqallah from the close.
    strip_opening: bool = True
    #: How similar two adjacent transcripts must be to count as one passage
    #: recited twice rather than two different passages that happen to adjoin.
    repeat_max_distance: float = 0.35


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


def _trim_outliers(hits: List[int], word_count: int, slack: int = 10) -> List[int]:
    """Drop mapped positions that cannot belong to the same segment.

    A segment of *n* transcript words covers about *n* reference words, so a hit
    hundreds of words away is a spurious match, not a boundary.  Taking the raw
    min and max lets one such hit swallow half a surah -- harmless in a short
    surah where there is nowhere far to go, ruinous in Al-Baqarah, and worst for
    a partial recitation, where most of the reference is unrelated text that a
    stray word can match.

    The median is the anchor because it survives outliers on either side.
    """
    if len(hits) < 2:
        return hits
    ordered = sorted(hits)
    median = ordered[len(ordered) // 2]
    reach = max(word_count * 3, word_count + slack)
    return [h for h in hits if abs(h - median) <= reach] or [median]


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


def _word_anchor_candidates(ref: Sequence[str], hyp: Sequence[str],
                            config: AlignConfig) -> List[Tuple[int, int]]:
    """Anchors from single words unique in the surah.

    Survives error rates that destroy every exact n-gram.  Two guards keep them
    honest: the word must be unique in the reference, and its best match in the
    hypothesis must beat the runner-up by a margin -- otherwise a word from a
    repeated refrain could pin the alignment to the wrong copy.
    """
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    counts = Counter(ref)
    rare = [
        (index, word) for index, word in enumerate(ref)
        if counts[word] == 1 and len(word) >= config.word_anchor_min_len
    ]
    if not rare or not hyp:
        return []

    import numpy as np

    matrix = cdist([word for _, word in rare], list(hyp),
                   scorer=Levenshtein.normalized_distance, workers=-1)

    out: List[Tuple[int, int]] = []
    for row, (ref_index, _) in enumerate(rare):
        distances = matrix[row]
        best = int(np.argmin(distances))
        if distances[best] > config.word_anchor_max_distance:
            continue
        if len(distances) > 1:
            runner_up = float(np.partition(distances, 1)[1])
            if runner_up - float(distances[best]) < config.word_anchor_margin:
                continue
        out.append((ref_index, best, 1))
    return out


def _lis_chain(candidates: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Largest subset that increases in both sequences.

    Must be a real longest-increasing-subsequence, not a greedy sweep: greedily
    keeping the first candidate that still increases lets one bad early anchor
    with a large hypothesis index block every good anchor after it.
    """
    if not candidates:
        return []

    ordered = sorted(candidates)
    tails: List[int] = []
    index_of: List[int] = []
    prev: List[int] = []

    for position, (_, hyp_index, _length) in enumerate(ordered):
        slot = bisect_left(tails, hyp_index)
        if slot == len(tails):
            tails.append(hyp_index)
            index_of.append(position)
        else:
            tails[slot] = hyp_index
            index_of[slot] = position
        prev.append(index_of[slot - 1] if slot else -1)

    chain: List[Tuple[int, int, int]] = []
    cursor = index_of[-1] if index_of else -1
    while cursor >= 0:
        chain.append(ordered[cursor])
        cursor = prev[cursor]
    chain.reverse()
    return chain


def _find_anchors(ref: Sequence[str], hyp: Sequence[str], config: AlignConfig
                  ) -> List[Tuple[int, int, int]]:
    """Pairs ``(ref_index, hyp_index)`` that pin the alignment.

    Only n-grams unique on *both* sides qualify, so a repeated refrain can never
    become an anchor -- it is the unique verses around it that do the pinning.
    """
    n = config.anchor_n
    if len(ref) < n or len(hyp) < n:
        return []

    ref_positions = _ngram_positions(ref, n)
    hyp_positions = _ngram_positions(hyp, n)

    candidates: List[Tuple[int, int, int]] = []
    for gram, refs in ref_positions.items():
        if len(refs) != 1:
            continue
        hyps = hyp_positions.get(gram)
        if hyps and len(hyps) == 1:
            candidates.append((refs[0], hyps[0], n))

    if not candidates and not config.word_anchors:
        return []

    # Deduplicate by reference position, preferring the earliest hypothesis hit,
    # so an n-gram anchor and a word anchor cannot contradict each other.
    # An n-gram anchor beats a word anchor at the same position: it rests on
    # more evidence.
    best_by_ref: Dict[int, Tuple[int, int]] = {}
    for ref_index, hyp_index, length in candidates:
        current = best_by_ref.get(ref_index)
        if current is None or length > current[1]:
            best_by_ref[ref_index] = (hyp_index, length)

    chain = _lis_chain([(r, h, l) for r, (h, l) in best_by_ref.items()])

    if not config.word_anchors:
        return chain

    # Word anchors are added only where n-gram anchors left a large gap.  Applied
    # everywhere they are net harmful: in text with plenty of unique phrasing
    # they add weak evidence beside strong, and the weak evidence wins ties it
    # should lose.  Confined to the gaps, they only ever replace nothing.
    filler: List[Tuple[int, int, int]] = []
    bounds = [(0, 0)] + [(r + l, h + l) for r, h, l in chain] + [(len(ref), len(hyp))]
    for (ref_from, hyp_from), (ref_to, hyp_to) in zip(bounds, bounds[1:]):
        if ref_to - ref_from <= config.min_gap_to_split:
            continue
        if hyp_to - hyp_from <= 0:
            continue
        found = _word_anchor_candidates(
            ref[ref_from:ref_to], hyp[hyp_from:hyp_to], config)
        filler.extend((ref_from + r, hyp_from + h, l) for r, h, l in found)

    if not filler:
        return chain

    return _lis_chain(chain + filler)


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

    # Bound the regions before the first and after the last anchor.  Deleting a
    # reference word costs the same wherever it happens, so when the hypothesis
    # is far shorter than the reference -- a partial recitation -- the DP is
    # nearly indifferent to position and will happily match a stray word
    # thousands of words away.  Anchors localise reliably, so the unanchored
    # head and tail are limited to what the remaining transcript could plausibly
    # cover.
    first_ref, first_hyp, _ = anchors[0]
    last_ref, last_hyp, last_len = anchors[-1]
    head_room = first_hyp * 3 + config.anchor_n + 10
    tail_room = (len(hyp) - last_hyp - last_len) * 3 + config.anchor_n + 10
    lo = max(0, first_ref - head_room)
    hi = min(len(ref), last_ref + last_len + tail_room)

    ref_cursor, hyp_cursor = lo, 0

    for ref_index, hyp_index, length in anchors + [(hi, len(hyp), 0)]:
        if ref_index < ref_cursor or hyp_index < hyp_cursor:
            continue  # already covered by a previous anchor's span
        gap = _dp_align(ref[ref_cursor:ref_index], hyp[hyp_cursor:hyp_index])
        for offset, target in enumerate(gap):
            if target is not None:
                mapping[hyp_cursor + offset] = ref_cursor + target

        if ref_index < len(ref):
            # Advance by the anchor's own length: a one-word anchor consumes one
            # word, not the n-gram width.
            for k in range(length):
                if hyp_index + k < len(hyp) and ref_index + k < len(ref):
                    mapping[hyp_index + k] = ref_index + k
            ref_cursor = ref_index + length
            hyp_cursor = hyp_index + length

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
            # Skeleton, not rasm: both sides must use the same form, and the
            # skeleton absorbs the orthographic difference between an ASR
            # trained on Hafs and a Warsh reference.
            return [w for w in T.words(T.to_skeleton(value)) if w]

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

        hits = _trim_outliers(hits, len(normalizer(transcript)))
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

    # A repeat does not always leave one segment empty: the DP will happily give
    # each copy half of the span, which looks aligned and is wrong.  Two adjacent
    # segments whose spans are contiguous *and* whose transcripts are near
    # duplicates of each other are one passage recited twice, so both get the
    # union rather than a half each.
    for position in range(len(results) - 1):
        first, second = results[position], results[position + 1]
        if first.ref_start < 0 or second.ref_start < 0:
            continue
        if first.formula or second.formula or first.ref_end != second.ref_start:
            continue
        left = " ".join(normalizer(first.hypothesis))
        right = " ".join(normalizer(second.hypothesis))
        if not left or not right:
            continue
        if _distance(left, right) > config.repeat_max_distance:
            continue

        start, end = first.ref_start, second.ref_end
        span_rasm = surah.rasm(start, end)

        # Resembling each other is not enough.  In a surah built on a refrain,
        # two *different* consecutive passages both containing it look alike and
        # sit side by side -- merging those spans loses a whole verse.  A real
        # repeat is two recitations of the same passage, so each transcript must
        # match the whole union, not half of it.
        if (_distance(left, span_rasm) > config.repeat_max_distance
                or _distance(right, span_rasm) > config.repeat_max_distance):
            continue

        span_label = surah.label(start, end)
        verses = surah.verses_spanned(start, end)
        for index, hypothesis in ((position, left), (position + 1, right)):
            results[index] = Aligned(
                index=results[index].index, ref_start=start, ref_end=end,
                label=span_label, rasm=span_rasm, verses=list(verses),
                distance=_distance(hypothesis, span_rasm), ok=True,
                hypothesis=results[index].hypothesis, repeat=True,
            )

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
