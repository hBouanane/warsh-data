"""Transcribing segments with a NeMo FastConformer model.

The bootstrap recogniser does not need to be right.  Its output is only ever
used to decide *where in the surah* a segment sits; the label that reaches the
dataset is the reference text the aligner selects, so recognition errors are
discarded rather than trained on.  That is why a Hafs-trained model is usable
against Warsh audio here: it has to be close enough to locate, not correct.

Default is ``mohammed/fastconformer-quran-ar``, a hybrid RNNT/CTC model.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["Transcriber", "DEFAULT_MODEL", "DEFAULT_CHECKPOINT"]

DEFAULT_MODEL = "mohammed/fastconformer-quran-ar"
DEFAULT_CHECKPOINT = "phase3_full/phase3_full_wer0.0014.nemo"
SAMPLE_RATE = 16000


def _as_text(result) -> str:
    """NeMo has returned bare strings and objects with ``.text`` across versions."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text is not None:
        return text
    if isinstance(result, (list, tuple)) and result:
        return _as_text(result[0])
    return str(result)


@dataclass
class Transcriber:
    """Loads the model once and transcribes batches of waveforms."""

    model_id: Optional[str] = None
    checkpoint: Optional[str] = None
    device: str = "cuda"
    #: Kept small on purpose: RNNT aside, decoding cost scales with the longest
    #: clip in the batch, and a batch of long segments is what pushes a 15 GB
    #: card over.
    batch_size: int = 8
    #: The published model was trained with min_duration 0.5 / max_duration 30.
    #: Handing the RNNT decoder a clip outside that range crashes it with an
    #: illegal memory access, which then poisons the CUDA context for good, so
    #: the range is enforced here rather than discovered at hour two.
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    #: Which head of the hybrid model decodes.  RNNT scores better; CTC is the
    #: fallback if RNNT keeps crashing, since it is one forward pass with no
    #: autoregressive loop to overflow.
    decoder: str = "rnnt"

    def __post_init__(self) -> None:
        self.model_id = self.model_id or DEFAULT_MODEL
        self.checkpoint = self.checkpoint or DEFAULT_CHECKPOINT

        import torch
        from huggingface_hub import hf_hub_download

        import nemo.collections.asr as nemo_asr

        path = (self.checkpoint if Path(self.checkpoint).exists()
                else hf_hub_download(repo_id=self.model_id, filename=self.checkpoint))
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(path)
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.decoder:
            try:
                self.model.change_decoding_strategy(decoder_type=self.decoder)
            except Exception as exc:      # not a hybrid model, or older NeMo
                print(f"could not select the {self.decoder} decoder ({exc}); "
                      f"using the model default")

        self._torch = torch

    def transcribe(self, waves: Sequence[np.ndarray]) -> List[str]:
        """One transcript per waveform, in order.

        Clips outside the model's training range come back empty rather than
        truncated or forced through: too long crashes the RNNT decoder, and a
        truncated transcript would place a span shorter than the audio it was
        cut from.  They keep their audio in the dataset with no transcript and
        no label, so they are easy to find and redo once there is a way to
        handle them.
        """
        if not waves:
            return []

        floor = int(self.min_seconds * SAMPLE_RATE)
        ceiling = int(self.max_seconds * SAMPLE_RATE)

        usable, positions = [], []
        for index, wave in enumerate(waves):
            array = np.asarray(wave, dtype=np.float32).reshape(-1)
            if array.size < floor or array.size > ceiling:
                continue
            usable.append(array)
            positions.append(index)

        if not usable:
            return [""] * len(waves)

        # Group clips of similar length together.  RNNT decodes greedily to the
        # longest clip in a batch, so mixing a one-second clip with a
        # twenty-nine-second one makes every clip in that batch pay the long
        # one's decode loop -- wasted work, and the peak allocation that takes
        # the card down mid-corpus.
        order = sorted(range(len(usable)), key=lambda i: usable[i].size)
        texts_by_slot = [""] * len(usable)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            for slot, text in zip(chunk, self._run([usable[i] for i in chunk])):
                texts_by_slot[slot] = text
            self._free()

        out = [""] * len(waves)
        for position, text in zip(positions, texts_by_slot):
            out[position] = text
        return out

    def _free(self) -> None:
        """Release cached blocks between batches.

        Two models sit on the card at once here -- the segmenter and this one --
        so the allocator has less room to work with than either expects.
        """
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass

    def _run(self, waves: Sequence[np.ndarray]) -> List[str]:
        try:
            with self._torch.no_grad():
                results = self.model.transcribe(
                    list(waves), batch_size=len(waves), verbose=False,
                )
            return [_as_text(r) for r in results]
        except (TypeError, ValueError, AttributeError):
            return self._transcribe_via_files(waves)

    def _transcribe_via_files(self, waves: Sequence[np.ndarray]) -> List[str]:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, wave in enumerate(waves):
                path = Path(tmp) / f"{index:06d}.wav"
                sf.write(path, np.asarray(wave, dtype=np.float32), SAMPLE_RATE)
                paths.append(str(path))
            with self._torch.no_grad():
                results = self.model.transcribe(
                    paths, batch_size=self.batch_size, verbose=False
                )
        return [_as_text(r) for r in results]
