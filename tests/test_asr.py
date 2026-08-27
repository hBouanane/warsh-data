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


def test_memory_report_separates_fragmentation_from_use():
    """reserved minus allocated is memory the process owns but cannot use --
    the number that distinguishes a fragmenting allocator from a leak."""
    from warshdata.asr import MemoryReport

    report = MemoryReport(allocated=3.0, reserved=11.5, peak=12.0)
    assert report.fragmentation == pytest.approx(8.5)
    assert "frag  8.50 GB" in report.line()
    assert "alloc  3.00 GB" in report.line()


def test_memory_report_is_never_negative():
    from warshdata.asr import MemoryReport

    assert MemoryReport(allocated=5.0, reserved=4.0).fragmentation == 0.0


def test_memory_probe_survives_a_missing_cuda():
    """The probe must never be the thing that breaks a run."""
    class NoCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def memory_summary():
            raise RuntimeError("no device")

    class Torch:
        cuda = NoCuda

    probe = Fake()
    probe._torch = Torch
    probe.device = "cuda"
    assert probe.memory().allocated == 0.0
    assert "no memory summary" in probe.memory_summary()


def test_cuda_graphs_are_switched_off():
    """A CUDA graph replays kernel launches bound to fixed addresses. Anything
    that moves that memory turns the replay into an out-of-bounds read -- an
    illegal memory access on a nearly empty card, which is the failure this
    pipeline kept hitting a hundred sources into a run."""
    class Greedy:
        allow_cuda_graphs = True

    class Cfg:
        greedy = Greedy()

    class Decoding:
        cfg = Cfg()

        class decoding:
            allow_cuda_graphs = True

    class Model:
        decoding = Decoding()

    probe = Fake()
    probe.model = Model()
    Transcriber._disable_cuda_graphs(probe)

    assert Model.decoding.cfg.greedy.allow_cuda_graphs is False
    assert Model.decoding.decoding.allow_cuda_graphs is False


def test_disabling_graphs_survives_a_model_without_them():
    class Bare:
        pass

    probe = Fake()
    probe.model = Bare()
    Transcriber._disable_cuda_graphs(probe)      # must not raise


def test_the_allocator_is_left_alone_between_batches():
    """empty_cache() between batches was added for a fragmentation problem the
    allocator dump later disproved, and releasing memory mid-run is exactly what
    breaks a captured CUDA graph. It should be gone."""
    import inspect

    from warshdata import asr

    source = inspect.getsource(asr)
    assert "empty_cache" not in source
