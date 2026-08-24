# warsh-data

Build a Warsh recitation dataset from raw recordings.

Stage 1 — **waqf segmentation**. Long recitation recordings are cut at stops using
[`obadx/recitation-segmenter-v2`](https://huggingface.co/obadx/recitation-segmenter-v2)
(MIT, 0.6B, Wav2Vec2-BERT frame classifier, ~3 GB VRAM), producing one manifest
record per segment. Labelling and Hugging Face export build on that manifest.

Segmenting at waqf rather than force-aligning a whole recording is what makes
repeated words tractable: a reciter who stops, backs up and repeats produces two
independent segments that each match text on their own, instead of breaking the
monotonic alignment of the whole file.

## Install

Needs `ffmpeg` and `libsndfile` first.

```bash
conda create -n warsh python=3.12 && conda activate warsh && conda install -c conda-forge ffmpeg libsndfile
```

```bash
pip install -e .
```

## Layout

The reciter slug is taken from the parent directory:

```
audio/
  ibrahim-al-dosary/002.mp3
  yassin-al-jazaery/002.mp3
```

## Use

```bash
warsh-data segment ./audio -o ./out --resume
```

```bash
warsh-data stats ./out/segments.jsonl
```

Outputs:

- `out/segments.jsonl` — one record per segment, appended per source file
- `out/segment_params.json` — the settings the manifest was produced with
- `out/segments/<reciter>/clips/<segment_id>.wav` — 16 kHz mono clips

## Notes

**Resume is real.** Records are appended and flushed after each source file, so an
interrupted run loses at most one recording. `--resume` skips sources already in
the manifest. Segment ids are derived from sample offsets, not a counter, so a
re-run with the same settings reproduces the same ids.

**Clips are cut from the waveform the model saw** — the same 16 kHz decode,
indexed by sample, never round-tripped through seconds. This is what keeps a clip
from drifting off the predicted boundary and clipping a first syllable.

**`source_is_complete: false`** means the segmenter ran out of audio mid-speech.
The final segment of such a recording is not waqf-bounded; `stats` counts these
separately and they should not be trusted as training examples.

**dtype is auto-detected.** bfloat16 where the GPU really supports it, float16 on
pre-Ampere cards (a Colab T4 has no bfloat16), float32 on CPU. Override with
`--dtype`. The resolved value is written to `segment_params.json`.

**On Colab, write the output somewhere that survives.** The runtime's local disk
is wiped when the session ends. Point `-o` at a mounted Drive folder, or push to
the Hub as soon as a pass finishes.

**The thresholds are a tuning knob, not a constant.** Defaults are 200 ms silence
floor / 400 ms speech floor / 40 ms padding — above breath noise, but low enough
to keep short waqf units. The model card's 30/30/30 finds any speech boundary,
not specifically a stop. Re-tune against a hand-checked sample before a full run.
