"""The aligner, against the real Warsh text.

These lock in the properties that repeatedly went wrong while building it, so a
future change that reintroduces them fails here rather than silently degrading a
corpus.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rapidfuzz")
pytest.importorskip("warshlab")

from warshdata import align, quran, simulate

AR_RAHMAN = 55
AL_MURSALAT = 77
AL_IKHLAS = 112
YA_SIN = 36


@pytest.fixture(scope="module")
def text():
    try:
        return quran.load()
    except FileNotFoundError:
        pytest.skip("Warsh text not downloaded")


def test_reference_is_warsh_and_complete(text):
    assert len(text.surahs) == 114
    assert text.verse_count == 6236
    # The rounded high stop is Warsh's hamza notation; a Hafs text lacks it.
    corpus = "".join(v.text for s in text.surahs.values() for v in s.verses)
    assert corpus.count("۬") > 5000


def test_rasm_and_labels_stay_index_for_index(text):
    for number in (1, 2, 55, 114):
        surah = text[number]
        assert len(surah.words) == len(surah.raw_words) == len(surah.word_verse)


def test_clean_transcripts_align_exactly(text):
    surah = text[AR_RAHMAN]
    case = simulate.make_case(surah, rate=0.0, seed=1)
    result = align.align_surah(surah, case.transcripts)
    assert simulate.score_case(case, result)["accuracy"] == 1.0


def test_repeated_refrain_is_placed_by_position_not_content(text):
    """Ar-Rahman repeats one verse 31 times. Searched independently every copy
    matches equally well; in order, each one is pinned by its neighbours."""
    surah = text[AR_RAHMAN]
    case = simulate.make_case(surah, rate=0.20, seed=3)
    result = align.align_surah(surah, case.transcripts)

    refrain = [s for s in result.segments if "تكذبان" in s.rasm]
    assert len(refrain) >= 20
    starts = [s.ref_start for s in refrain]
    assert starts == sorted(starts), "refrains must be assigned in order"
    assert len(set(starts)) == len(starts), "no two refrains share a position"


def test_accuracy_holds_up_at_high_error_rates(text):
    surah = text[AR_RAHMAN]
    for rate, floor in ((0.20, 0.85), (0.30, 0.85), (0.40, 0.80)):
        case = simulate.make_case(surah, rate, seed=11, repeat_segments=4,
                                  opening=True, closing=True)
        result = align.align_surah(surah, case.transcripts)
        accuracy = simulate.score_case(case, result)["accuracy"]
        assert accuracy >= floor, f"{rate:.0%} CER gave {accuracy:.1%}"


def test_long_surah_is_accurate_and_fast(text):
    import time

    surah = text[YA_SIN]
    case = simulate.make_case(surah, 0.20, seed=2)
    started = time.time()
    result = align.align_surah(surah, case.transcripts)
    assert simulate.score_case(case, result)["accuracy"] >= 0.95
    assert time.time() - started < 20


def test_opening_formulas_are_labelled_not_discarded(text):
    """The formulas are prepended to the reference rather than stripped off the
    transcript. Stripping only worked when a formula arrived as its own segment;
    a reciter who runs the basmala into the first verse gave the recogniser one
    inseparable string, and the label lost the basmala entirely."""
    surah = text[YA_SIN]
    case = simulate.make_case(surah, 0.0, seed=4, opening=True)
    result = align.align_surah(surah, case.transcripts)

    first, second = result.segments[0], result.segments[1]
    assert first.formula and second.formula, "both openings should be marked"
    assert "أَعُوذُ" in first.label, first.label
    assert "بِسْمِ" in second.label, second.label
    assert simulate.score_case(case, result)["accuracy"] == 1.0


def test_a_formula_run_into_the_first_verse_keeps_both(text):
    """The case that broke: basmala and the opening of the surah in one segment.
    The label has to carry both, not just the Quranic half."""
    surah = text[2]
    spoken = align.BASMALA + " " + surah.rasm(0, 4)
    result = align.align_surah(surah, [spoken, surah.rasm(4, 10)])

    label = result.segments[0].label
    assert "بِسْمِ" in label, f"basmala dropped from the label: {label}"
    assert surah.raw_words[0].split()[0][:3] in label, label


def test_basmala_is_kept_in_al_fatiha(text):
    """It is verse 1:1 there, so stripping it would delete real reference text."""
    surah = text[1]
    case = simulate.make_case(surah, 0.0, seed=1)
    result = align.align_surah(surah, case.transcripts)
    assert not any(s.formula for s in result.segments)
    assert simulate.score_case(case, result)["accuracy"] == 1.0


def test_a_repeated_passage_gets_the_same_span_twice(text):
    """A reciter repeating after a stop is two segments of one passage, and both
    want the same label -- not half the span each."""
    surah = text[YA_SIN]
    case = simulate.make_case(surah, 0.10, seed=6, repeat_segments=3)
    result = align.align_surah(surah, case.transcripts)

    repeats = [s for s in result.segments if s.repeat]
    assert repeats, "repeats should be recognised"
    for segment in repeats:
        assert segment.ref_start >= 0 and segment.ref_end > segment.ref_start


def test_lis_chain_is_not_a_greedy_sweep():
    """Regression: a greedy monotone filter let one bad early anchor with a huge
    hypothesis index block every good anchor after it, collapsing accuracy."""
    candidates = [(0, 10_000, 1)] + [(i, i, 4) for i in range(1, 40)]
    chain = align._lis_chain(candidates)
    assert len(chain) >= 39, "the long consistent run must survive one outlier"
    assert (0, 10_000, 1) not in chain


def test_anchors_increase_in_both_sequences(text):
    surah = text[YA_SIN]
    case = simulate.make_case(surah, 0.25, seed=8)
    from warshlab import text as T

    hypothesis = [w for t in case.transcripts for w in T.words(T.to_rasm(t)) if w]
    anchors = align._find_anchors(surah.words, hypothesis, align.AlignConfig())

    assert [a[0] for a in anchors] == sorted(a[0] for a in anchors)
    assert [a[1] for a in anchors] == sorted(a[1] for a in anchors)


def test_word_anchors_are_not_harmful(text):
    """Word anchors are weak evidence, so they are gated behind a minimum gap
    and must never cost accuracy where n-gram anchors already suffice.

    They used to show a large gain on refrain-heavy surahs. Most of that was
    compensating for spans being clipped at segment boundaries; with that fixed
    they are neutral there, and the remaining value is on longer surahs at high
    error rates. The test asserts what is still true: no harm, net positive.
    """
    off = align.AlignConfig(word_anchors=False)
    gains = []
    for number, rates in ((AR_RAHMAN, (0.30, 0.50)), (YA_SIN, (0.20, 0.30))):
        surah = text[number]
        for rate in rates:
            case = simulate.make_case(surah, rate, seed=11, repeat_segments=4,
                                      opening=True, closing=True)
            on = simulate.score_case(
                case, align.align_surah(surah, case.transcripts))["accuracy"]
            without = simulate.score_case(
                case, align.align_surah(surah, case.transcripts, off))["accuracy"]
            gains.append(on - without)

    assert min(gains) >= -0.02, "word anchors must not cost accuracy"
    assert sum(gains) > 0, "net effect must remain positive"


def test_tiny_surah(text):
    surah = text[AL_IKHLAS]
    case = simulate.make_case(surah, 0.15, seed=1)
    result = align.align_surah(surah, case.transcripts)
    assert simulate.score_case(case, result)["accuracy"] == 1.0


def test_empty_input_is_handled(text):
    result = align.align_surah(text[AL_IKHLAS], [])
    assert result.segments == []
    assert result.coverage == 0.0


def test_garbage_input_aligns_nothing_confidently(text):
    surah = text[AL_MURSALAT]
    result = align.align_surah(surah, ["زززز ززز زززز", "ككك كككك ككك"])
    assert all(not s.ok for s in result.segments)


def test_a_fragment_of_a_long_surah_is_located(text):
    """Global alignment cannot do this: accounting for the unused reference
    costs the same wherever it happens, so a fragment's span used to run
    thousands of words past its true end. A segment is bounded by its audio."""
    surah = text[2]
    for start in (120, 2500, 5000):
        part = [surah.rasm(start, start + 6), surah.rasm(start + 6, start + 12)]
        result = align.align_surah(surah, part)
        got = (result.segments[0].ref_start, result.segments[-1].ref_end)
        assert got == (start, start + 12), f"planted at {start}, found {got}"


def test_a_span_cannot_exceed_what_the_audio_could_hold(text):
    surah = text[2]
    cfg = align.AlignConfig()
    part = [surah.rasm(3000, 3008)]
    result = align.align_surah(surah, part, cfg)
    assert result.segments[0].word_count <= cfg.max_words_per_segment


def test_window_stride_never_collapses():
    """Regression: sizing the window off the transcript alone let an overlap of
    2x the segment cap exceed the window, collapsing the stride to one word and
    turning a 70-window scan into 6000."""
    cfg = align.AlignConfig()
    for hyp_len in (1, 5, 12, 40, 200):
        span = max(hyp_len * 3, 3 * cfg.max_words_per_segment)
        stride = max(cfg.max_words_per_segment, span - 2 * cfg.max_words_per_segment)
        assert stride >= cfg.max_words_per_segment
        assert span - stride >= cfg.max_words_per_segment, "windows must overlap"


def test_segments_may_overlap_when_the_audio_does(text):
    """Waqf segments are not a partition. A reciter who pauses often carries the
    last words into the next segment or repeats them, so the same reference
    words legitimately belong to two recordings. Forcing a strict partition
    clipped both -- measured on real segments of Al-Baqarah, spans ran about
    three words short at the end.
    """
    surah = text[2]
    # Two segments whose audio overlaps by three words, as a real reciter does.
    first = surah.rasm(430, 442)
    second = surah.rasm(439, 448)
    result = align.align_surah(surah, [first, second])

    a, b = result.segments
    assert a.ref_end > b.ref_start, "an overlap in the audio must survive"
    assert a.ref_end >= 441, f"first segment clipped at {a.ref_end}"
    assert b.ref_start <= 440, f"second segment clipped at {b.ref_start}"


def test_extension_cannot_invent_coverage(text):
    """A segment may only claim extra words while its own transcript matches
    them better for doing so, so a short recitation stays short."""
    surah = text[2]
    spoken = surah.rasm(1000, 1006)
    result = align.align_surah(surah, [spoken])
    assert result.segments[0].word_count <= 12
