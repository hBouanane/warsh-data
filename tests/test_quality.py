"""Per-reciter label fit and waqf fit."""

from __future__ import annotations

from warshdata import quality


def row(reciter, duration=8.0, distance=0.02, asr="قل هو الله احد",
        label="قُلْ هُوَ اَ۬للَّهُ أَحَدٌ"):
    return {
        "reciter_slug": reciter,
        "duration_seconds": duration,
        "align_distance": distance,
        "asr": asr,
        "label": label,
    }


def untranscribed(reciter, duration=34.0):
    """What a clip past the recogniser's window looks like in the manifest:
    no transcript, no label, and a stored distance of 1.0 from the aligner
    comparing an empty hypothesis against a real span."""
    return {
        "reciter_slug": reciter,
        "duration_seconds": duration,
        "align_distance": 1.0,
        "asr": None,
        "label": None,
    }


def test_untranscribed_clips_do_not_count_as_misalignment():
    """The defect this table was built to expose, and then had itself.

    A clip outside the duration window is never sent to the recogniser, so the
    aligner scores an empty hypothesis against a real span and stores 1.0.
    Counted as a distance it dominates the tail and makes a segmentation
    problem read as the corpus's worst alignment -- on real data it moved one
    reciter's mean from 0.032 to 0.065 and his p99 from 0.20 to 1.00.
    """
    rows = [row("slow", distance=0.02) for _ in range(90)]
    rows += [untranscribed("slow") for _ in range(10)]

    fit = quality.label_fit(rows)[0]
    assert fit.scored == 90, "untranscribed rows must not be scored"
    # Reported as match = 1 - distance, so a 0.02 distance is a 0.98 match.
    assert fit.mean == 0.98, f"mean polluted by empty transcripts: {fit.mean}"
    assert fit.p01 > 0.9, "the bad tail is the low end once this is a match"
    assert fit.under_050 == 0.0

    # Not hidden, though -- just counted as what it is.
    assert fit.no_transcript == 0.10
    assert fit.no_label == 0.10
    assert fit.segments == 100


def test_reciters_are_ordered_worst_match_first():
    rows = ([row("clean", distance=0.01)] * 10
            + [row("noisy", distance=0.30)] * 10
            + [row("middling", distance=0.08)] * 10)
    # Worst first, which with a match means lowest first.
    assert [f.reciter for f in quality.label_fit(rows)] == [
        "noisy", "middling", "clean"]


def test_a_short_transcript_is_reported_separately_from_distance():
    """The label holding more words than the transcript is a third cause: the
    recogniser dropped speech, so the distance overstates how wrong the label
    is. It needs a different repair from a misplaced span."""
    rows = [row("x", asr="قل هو", label="قُلْ هُوَ اَ۬للَّهُ أَحَدٌ") for _ in range(5)]
    rows += [row("x") for _ in range(5)]
    fit = quality.label_fit(rows)[0]
    assert fit.transcript_short == 0.5


def test_waqf_fit_counts_holes_not_just_long_segments():
    rows = [row("slow", duration=12.0) for _ in range(80)]
    rows += [row("slow", duration=24.0) for _ in range(15)]
    rows += [untranscribed("slow", duration=33.0) for _ in range(5)]

    fit = quality.waqf_fit(rows)[0]
    assert fit.over_20 == 0.20
    assert fit.over_30 == 0.05
    assert fit.unlabelled_long == 5, "the rows to go and re-segment"
    assert fit.longest == 33.0
    assert round(fit.hours, 3) == round((80 * 12 + 15 * 24 + 5 * 33) / 3600, 3)


def test_long_segments_are_only_flagged_when_they_cost_data():
    """Regression: the first version flagged the corpus's cleanest reciter for
    having 10% of segments over 20 s while losing nothing at all. A long
    segment the recogniser still handled is not a problem to act on."""
    fine = [row("fine", duration=22.0) for _ in range(20)]
    fine += [row("fine", duration=10.0) for _ in range(80)]

    costly = [row("costly", duration=22.0) for _ in range(20)]
    costly += [untranscribed("costly", duration=34.0) for _ in range(10)]
    costly += [row("costly", duration=10.0) for _ in range(70)]

    rows = fine + costly
    notes = quality.verdicts(quality.label_fit(rows), quality.waqf_fit(rows))
    assert any("costly" in n for n in notes)
    assert not any("fine" in n for n in notes)


def test_tables_render_for_an_empty_and_a_single_reciter():
    assert quality.format_label_table([])
    assert quality.format_waqf_table([])
    rows = [row("only")]
    assert "only" in quality.format_label_table(quality.label_fit(rows))
    assert "only" in quality.format_waqf_table(quality.waqf_fit(rows))


def test_rows_without_a_duration_do_not_crash_waqf_fit():
    rows = [{"reciter_slug": "x", "asr": "a", "label": "b", "align_distance": 0.1}]
    assert quality.waqf_fit(rows) == []
    assert quality.label_fit(rows)[0].segments == 1
