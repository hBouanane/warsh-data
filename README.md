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

## Fetch

Warsh recitations come from the [mp3quran.net](https://mp3quran.net) API
(`/api/v3/reciters`), which lists a `server` base URL per reciter; files are
`{server}{surah:03d}.mp3`. Paths are discovered, not hardcoded, so a reciter
added upstream is picked up for free.

```bash
warsh-data fetch --list
```

```bash
warsh-data fetch -o ./audio
```

15 Warsh reciters, ~1495 surah files, several GB. Downloads land in the
`audio/<reciter-slug>/NNN.mp3` layout `segment` expects. Files are written to
`.part` and renamed only when complete, so a re-run resumes safely and can never
mistake a truncated mp3 for a finished one.

One surah, one reciter -- the sane first run, and what fits a Colab session:

```bash
warsh-data fetch --reciter ibrahim-aldosary --surah 1 -o ./audio
```

**Tariq.** Warsh via Tariq Abi Baker al-Asbahani pronounces differently from the
standard Tariq al-Azraq. It is excluded by default and slugged
`--variant-tariq` when `--include-variant-tariq` is passed, so it cannot end up
in a training pool unnoticed. Reciters whose moshaf name spells out *Tariq
al-Azraq* are standard Warsh and are included normally.

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

## Checking the thresholds before a full pass

Reciters differ enormously in pace, and one silence floor does not fit all of
them: the same setting that over-segments a fast reciter makes a slow mujawwad
one swallow waqf. Probe a few surahs across every reciter first.

```bash
warsh-data fetch --surah 1 --surah 87 --surah 36 -o ./probe-audio
```

```bash
warsh-data segment ./probe-audio -o ./probe && warsh-data stats ./probe/segments.jsonl
```

`stats` breaks duration down per reciter and flags the two tails:

```
reciter                          segs   audio     p5    p50    p95    max    <1s   >20s
fast-reciter                      200   0.06h    0.4    1.1    2.0    2.0  40.5%   0.0%
ibrahim-aldosari                  200   0.37h    4.2    6.7    8.8    9.0   0.0%   0.0%
slow-mujawwad                     200   1.16h   12.6   20.5   29.1   29.9   0.0%  54.0%

Worth a listen:
  fast-reciter: 40% of segments under 1 s -- likely over-segmenting, ...
  slow-mujawwad: 54% of segments over 20 s -- waqf being missed, ...
```

Many sub-second segments mean the silence floor is too low. Many over 20 s mean
waqf is being missed *and* the segment exceeds the model's own 20 s window. The
global summary above this table would read a healthy p50 of 6.7 s and show
neither problem, which is why the per-reciter view exists.

Outputs:

- `out/segments.jsonl` — one record per segment, appended per source file
- `out/segment_params.json` — the settings the manifest was produced with
- `out/segments/<reciter>/clips/<segment_id>.wav` — 16 kHz mono clips

## Push to the Hub

```bash
warsh-data segment ./audio -o ./out --push-to <user>/warsh-segments-v2 --resume
```

Segments stream into parquet shards and upload as each fills, so local disk never
grows past one shard. `--resume` reads the *manifest* from the Hub -- a few MB,
not the corpus -- so a run continues across sessions.

The invariant that makes resume safe:

> every source named in the hub manifest has all of its audio in a shard

Two rules enforce it. Shards flush only at a **source boundary** (so a source's
segments are never split between a shard and the buffer), and the manifest is
uploaded only **immediately after a successful shard flush**. Without both, a
manifest can name sources whose audio is still buffered; resume then skips them
permanently and their audio is uploaded by nobody. A crash costs re-segmenting up
to one shard, never data.

`--push-every` governs the raw provenance files only, which commit on their own
schedule and never drag the manifest with them.

Repo layout:

| path | what | read during training |
|---|---|---|
| `data/shard-*.parquet` | rows + audio embedded as 16 kHz mono FLAC | yes, streamed |
| `manifests/segments.jsonl` | same rows, no audio | no |
| `raw/<reciter>/NNN.mp3` | source recordings | no |

```python
from datasets import load_dataset
ds = load_dataset("<user>/warsh-segments-v2", split="train", streaming=True)
```

Embedded audio in sharded parquet, rather than one file per clip: hundreds of
thousands of small LFS objects are slow to list, slow to fetch, and awkward to
shuffle. Streaming pulls only the shards being iterated.

## Correcting segments by hand

Reviewing by ear always turns up boundaries to nudge, junk to drop, and clips
that should have been two. Do **not** edit `segments.jsonl` -- the next
segmentation run regenerates it. Write a `corrections.jsonl` instead:

```json
{"segment_id": "ibrahim-aldosari__087__0003", "action": "adjust", "start_seconds": 12.4, "orig_start_seconds": 12.0, "note": "clipped alif"}
{"segment_id": "ibrahim-aldosari__087__0007", "action": "drop", "note": "station ident"}
{"segment_id": "ibrahim-aldosari__087__0011", "action": "split", "at_seconds": [31.8], "note": "waqf the model missed"}
{"segment_id": "ibrahim-aldosari__087__0014", "action": "merge", "with": ["ibrahim-aldosari__087__0015"], "note": "breath pause, not a waqf"}
```

`split` and `merge` are the two halves of a boundary error. A missed waqf splits
one segment into `__0011_a` / `__0011_b`, both recording `parent_segment_id`. An
invented boundary -- the model cutting at a breath pause mid-ayah -- merges the
extra segments into the first, which **keeps its own id**, so anything already
attached to it stays attached. Both clear `audio_path`, since the clip on disk no
longer matches the new boundaries and has to be re-cut.

Merging across different source recordings is refused rather than guessed at.

```bash
warsh-data apply-corrections ./out/segments.jsonl ./out/corrections.jsonl -o ./out/segments.final.jsonl
```

Segment ids are `<reciter>__<surah>__<ordinal>` and deliberately **do not encode
timestamps**, so moving a boundary does not change the id and does not orphan the
label, note, or review attached to it. Dropping a segment leaves every other id
alone -- ordinals are assigned once at segmentation and never recomputed, so a
drop leaves a gap rather than renumbering what follows.

Clip *filenames* do carry the boundaries --
`ibrahim-aldosari__087__0003__12400-18880ms.flac` -- because a filename shows the
segment's present state and is expected to change when it is corrected. The id is
what labels are keyed to, so it must not.

`orig_*_seconds` records what the reviewer actually listened to. If segmentation
is later re-run with different thresholds and a segment has moved, the correction
is reported as *drifted* rather than applied blind to audio nobody reviewed.

## Tests

```bash
pip install -e ".[dev]" && python -m pytest
```

62 tests, no network and no GPU: the segmenter and the Hub API are both stubbed,
so the suite runs anywhere. That is deliberate -- the bugs it exists to catch are
in the plumbing (does the final shard actually get pushed? does a drop renumber
its neighbours?), not in the model.

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

**dtype is auto-detected**, gated on compute capability >= 8.0 rather than on
`torch.cuda.is_bf16_supported()` alone -- that call reports True on pre-Ampere
cards (a T4) on the strength of emulated bfloat16, which runs but is slower than
fp16. So: bfloat16 on Ampere and newer, float16 on older CUDA cards, float32 on
CPU. Override with `--dtype`. The resolved value is written to
`segment_params.json`.

**On Colab, write the output somewhere that survives.** The runtime's local disk
is wiped when the session ends. Point `-o` at a mounted Drive folder, or push to
the Hub as soon as a pass finishes.

**Audio is decoded by ffmpeg, not torchaudio.** `recitations_segmenter.read_audio`
goes through torchaudio's backend API (`list_audio_backends`), which recent
torchaudio removed -- on a current Colab image it raises `AttributeError` before
reading a byte. `warshdata.audio.load_wave` shells out to ffmpeg instead, which
also does the downmix and the resample in one pass. Without ffmpeg it falls back
to soundfile plus soxr or scipy for the resample.

Source recordings are **22050 Hz**, so resampling to the model's 16 kHz always
happens -- it is not an edge case.

**The thresholds are a tuning knob, not a constant.** Defaults are 200 ms silence
floor / 400 ms speech floor / 40 ms padding — above breath noise, but low enough
to keep short waqf units. The model card's 30/30/30 finds any speech boundary,
not specifically a stop. Re-tune against a hand-checked sample before a full run.
