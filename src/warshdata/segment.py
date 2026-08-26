"""Waqf segmentation with ``obadx/recitation-segmenter-v2``.

The one invariant worth stating: clips are cut from *the same* 16 kHz waveform
that was fed to the model, indexed by sample rather than by seconds.  Decoding
the source a second time, or round-tripping boundaries through floating point
seconds, is how a clip drifts a few tens of milliseconds away from the boundary
the model actually predicted -- which is exactly the kind of error that only
shows up much later, as a clipped first syllable in a training label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf
import torch

from recitations_segmenter import clean_speech_intervals, segment_recitations

from .audio import load_wave
from .manifest import SegmentRecord
from .sources import Source, clip_name, segment_id

__all__ = ["SegmentParams", "Segmenter", "MODEL_ID", "SAMPLE_RATE"]

MODEL_ID = "obadx/recitation-segmenter-v2"
SAMPLE_RATE = 16000

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _has_native_bf16() -> bool:
    """True only where bfloat16 has hardware support, i.e. Ampere (sm_80) or newer.

    ``torch.cuda.is_bf16_supported()`` alone is not enough: on pre-Ampere cards
    it reports True on the strength of *emulated* bf16, which runs but is far
    slower than fp16.  Newer PyTorch exposes ``including_emulation=False`` for
    exactly this; where that argument does not exist, fall back to the compute
    capability, which is the underlying fact either way.
    """
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8


@dataclass
class SegmentParams:
    """Thresholds handed to :func:`clean_speech_intervals`.

    The model card's defaults are 30/30/30 ms, which are the values for finding
    *any* speech boundary.  Waqf is a deliberate stop, so the silence floor is
    raised well above breath-length noise; the speech floor stays low enough to
    keep short waqf units, which a 900 ms floor would discard along with the
    short ayahs at the end of the mushaf.  Both are worth re-tuning against a
    hand-checked sample before a full run.
    """

    min_silence_duration_ms: int = 90
    min_speech_duration_ms: int = 400
    pad_duration_ms: int = 40
    max_duration_ms: int = 19995
    batch_size: int = 8
    device: str = "cuda"
    dtype: str = "auto"

    def torch_dtype(self) -> torch.dtype:
        """Resolve ``auto`` against the actual GPU.

        The model card uses bfloat16, which pre-Ampere cards (a Colab T4, for
        one) do not support -- asking for it there fails or falls back to
        something painfully slow.  ``auto`` picks bfloat16 where it is real,
        float16 on other CUDA devices, and float32 on CPU.
        """
        if self.dtype != "auto":
            return _DTYPES[self.dtype]
        if self.device == "cpu" or not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if _has_native_bf16() else torch.float16


class Segmenter:
    """Loads the model once and segments one recording at a time.

    One file per call, deliberately: a full-surah recording is already large in
    memory, and per-file granularity is what makes the run resumable.
    """

    def __init__(self, params: SegmentParams, model_id: str = MODEL_ID):
        self.params = params
        self.device = torch.device(params.device)
        self.dtype = params.torch_dtype()

        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioFrameClassification,
        )

        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    def segment(
        self,
        source: Source,
        clips_dir: Optional[Path] = None,
    ) -> Tuple[List[SegmentRecord], torch.Tensor]:
        """Segment one recording, optionally writing a clip per segment.

        Returns the records and the decoded waveform.  Raises whatever the
        segmenter raises -- the caller decides whether one bad file should stop
        a run; this layer does not swallow it.
        """
        p = self.params
        wave = load_wave(source.path, SAMPLE_RATE)

        outputs = segment_recitations(
            [wave],
            self.model,
            self.processor,
            device=self.device,
            dtype=self.dtype,
            batch_size=p.batch_size,
            max_duration_ms=p.max_duration_ms,
        )
        out = outputs[0]

        clean = clean_speech_intervals(
            out.speech_intervals,
            out.is_complete,
            min_silence_duration_ms=p.min_silence_duration_ms,
            min_speech_duration_ms=p.min_speech_duration_ms,
            pad_duration_ms=p.pad_duration_ms,
            sample_rate=SAMPLE_RATE,
            return_seconds=False,
        )

        intervals = clean.clean_speech_intervals.tolist()
        records: List[SegmentRecord] = []

        for i, (start, end) in enumerate(intervals):
            start, end = int(start), int(end)
            # Padding can push the last interval past the end of the waveform.
            start = max(0, start)
            end = min(end, wave.shape[-1])
            if end <= start:
                continue

            sid = segment_id(source, i)
            audio_path = None
            if clips_dir is not None:
                name = clip_name(sid, start, end, SAMPLE_RATE)
                clip_path = Path(clips_dir) / source.reciter_slug / "clips" / f"{name}.wav"
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(clip_path, wave[start:end].float().numpy(), SAMPLE_RATE)
                audio_path = str(clip_path.as_posix())

            records.append(
                SegmentRecord(
                    segment_id=sid,
                    reciter_slug=source.reciter_slug,
                    source_id=source.source_id,
                    source_path=str(source.path.as_posix()),
                    index=i,
                    start_sample=start,
                    end_sample=end,
                    start_seconds=round(start / SAMPLE_RATE, 3),
                    end_seconds=round(end / SAMPLE_RATE, 3),
                    duration_seconds=round((end - start) / SAMPLE_RATE, 3),
                    sample_rate=SAMPLE_RATE,
                    audio_path=audio_path,
                    source_is_complete=bool(clean.is_complete),
                    is_last_of_source=(i == len(intervals) - 1),
                )
            )

        return records, wave
