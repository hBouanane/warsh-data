"""Duration guards on the transcriber.

The published model was trained with min_duration 0.5 / max_duration 30. A clip
outside that range crashes the RNNT decoder with an illegal memory access, which
leaves the CUDA context dead for the rest of the run -- so the range is enforced
before anything reaches the GPU.

Out-of-range clips are skipped rather than trimmed to fit. A truncated
transcript would place a span shorter than the audio it was cut from, and the
segment would look labelled when it is not; left empty, it is easy to find and
redo later.
"""

from __future__ import annotations

import numpy as np
import pytest

from warshdata.asr import Transcriber


class Fake(Transcriber):
    def __init__(self, **kwargs):
        self.seen = []
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.min_seconds = kwargs.get("min_seconds", 0.5)
        self.max_seconds = kwargs.get("max_seconds", 30.0)
        self.batch_size = 4

    def _run(self, waves):
        self.seen = [len(w) for w in waves]
        return [f"t{i}" for i in range(len(waves))]


def clip(seconds):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def test_short_clips_are_skipped_not_sent():
    t = Fake()
    out = t.transcribe([clip(0.2), clip(3.0), clip(0.1)])
    assert t.seen == [48000], "only the usable clip should reach the model"
    assert out == ["", "t0", ""], out



def test_all_clips_unusable_returns_blanks_without_calling_the_model():
    t = Fake()
    assert t.transcribe([clip(0.1), clip(0.2)]) == ["", ""]
    assert t.seen == []


def test_order_is_preserved_across_skips():
    t = Fake()
    out = t.transcribe([clip(2.0), clip(0.1), clip(2.0)])
    assert out == ["t0", "", "t1"]


def test_empty_input():
    assert Fake().transcribe([]) == []


def test_over_long_clips_are_skipped_not_truncated():
    """Truncating would place a span shorter than the audio it came from, and
    the segment would look labelled when it is not. Better to leave it empty
    and findable."""
    t = Fake()
    out = t.transcribe([clip(45.0), clip(3.0)])
    assert t.seen == [48000], "only the in-range clip should reach the model"
    assert out == ["", "t0"]
