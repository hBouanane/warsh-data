"""Decoding source recordings to a 16 kHz mono waveform.

``recitations_segmenter.read_audio`` routes through torchaudio, whose backend
API (``list_audio_backends``) was removed in recent versions -- on a current
Colab image it raises ``AttributeError`` before any audio is read.  Rather than
pin torchaudio against whatever torch the host ships, decode here.

ffmpeg is the primary path: it is already an install prerequisite, it handles
every format the segmenter accepts, and it does the resample and downmix in the
same pass, so the waveform is fully determined by one command.  soundfile is the
fallback for hosts without ffmpeg, and only where no resampling is needed --
guessing at a resampler would move segment boundaries.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

__all__ = ["load_wave", "SAMPLE_RATE"]

SAMPLE_RATE = 16000


def _ffmpeg_decode(path: Path, sample_rate: int) -> np.ndarray:
    """Decode to mono float32 PCM at ``sample_rate`` on stdout."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode('utf-8', 'replace').strip()}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg produced no audio for {path.name}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resample, preferring soxr and falling back to scipy.

    Both are polyphase/band-limited; naive decimation would alias, and aliasing
    in the input is exactly the kind of damage that shows up later as a
    segmenter boundary in the wrong place.
    """
    try:
        import soxr

        return soxr.resample(samples, src_rate, dst_rate).astype(np.float32)
    except ImportError:
        pass
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_rate, dst_rate)
        return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot resample {src_rate} Hz -> {dst_rate} Hz: install ffmpeg (preferred), "
            f"or pip install soxr, or pip install scipy."
        ) from exc


def _soundfile_decode(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        mono = _resample(mono, sr, sample_rate)
    return mono


def load_wave(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Return a 1-D float32 tensor: mono, ``sample_rate``, in [-1, 1].

    This is the waveform the model sees *and* the one clips are cut from, so
    every boundary the segmenter predicts indexes directly into it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if shutil.which("ffmpeg"):
        samples = _ffmpeg_decode(path, sample_rate)
    else:
        samples = _soundfile_decode(path, sample_rate)

    if samples.size == 0:
        raise RuntimeError(f"{path.name} decoded to an empty waveform")

    # np.frombuffer gives a read-only view; copy so torch owns writable memory.
    return torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32).copy())
