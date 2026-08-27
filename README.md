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

## Auditing the audio before you segment it

```bash
warsh-data audit ./audio --bitrates
```

Compares each recording against the median duration of the *same surah* across
the other reciters. There is no absolute rule for how long a surah should be, but
fifteen reciters reading the same one is all the reference needed -- a file far
out of line with the rest is a mislabelled file, a truncated download, or a
different recording altogether.

Real example from mp3quran:

```
rachid-belalya/094   too long   surah 094 is much longer than other reciters'  (1134s vs 38s median, 29.6x)
```

Ash-Sharh is 38 seconds. That file is 19 minutes, so it is not Ash-Sharh.

Files that will not decode and files of zero duration are reported first, since
those need no comparison. A surah held by fewer than three reciters is skipped:
with no majority, calling either one wrong is a coin flip. `--factor` sets how
far off the median counts as suspect (default 3x, comfortably outside the ~2x
pace variation between reciters).

`--bitrates` shows median kbps per reciter, which explains small files rather
than flagging them -- the corpus ranges from 32 to 326 kbps and all of it is
resampled to 16 kHz mono anyway.

```bash
warsh-data audit ./audio --write-ids suspect.txt
```

## Listening to what was flagged

A flag is a question, not an answer. `audit` says `rachid-belalya/094` is wrong;
only listening says *how*.

```bash
warsh-data listen ./audio -o ./listen
```

With no `--ids` it audits first and excerpts whatever it flagged, so this alone
is a complete workflow. Open `listen/index.html`.

Three 15-second excerpts per recording, taken from the **start, middle and end**
rather than the first 15 seconds: a file that is the wrong recitation entirely
usually sounds fine at the opening (basmala), and only the middle gives it away.

The page is one self-contained file -- audio embedded as data URIs, no server, no
external requests, nothing to break when it is downloaded off Colab. The
individual clips are also left in `listen/clips/`.

```bash
warsh-data listen ./audio --ids suspect.txt -o ./listen --clips 5 --seconds 20
```

Excerpts are mp3 where ffmpeg is available and WAV otherwise (about ten times
larger, which only matters for the embedded page).

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

## Transcribe and label in the same pass

```bash
warsh-data segment ./audio -o ./out --asr --align --push-to <user>/warsh-v3 --resume
```

`--asr` transcribes every segment with
[mohammed/fastconformer-quran-ar](https://huggingface.co/mohammed/fastconformer-quran-ar);
`--align` places those transcripts in the Warsh text and writes the reference
span as the label.

Done in one pass on purpose. Each source becomes one parquet named after it, so
adding text afterwards would mean rewriting all 1495 files -- exactly the
operation the layout exists to avoid. Segment, transcribe and align together and
each row is written once, complete.

**The transcript is not the label.** `asr` is kept for diagnosis; `label` is the
reference text the aligner selected. A recognition error changes where a segment
is placed, not what it is labelled with, so the errors are discarded rather than
trained on. That is also why a Hafs-trained recogniser is usable against Warsh
audio here -- it only has to be close enough to locate.

Columns added: `asr`, `label`, `ref_start`, `ref_end`, `ayah_start`, `ayah_end`,
`align_distance`, `align_ok`, `is_formula`, `is_repeat`.

`align_distance` is what to filter on afterwards: it is how far the transcript
sits from the text it was matched to, so the segments worth dropping sort to the
top.

## Push to the Hub

```bash
warsh-data segment ./audio -o ./out --push-to <user>/warsh-segments --resume
```

**One source recording becomes one parquet file, named after it:**

| path | what |
|---|---|
| `data/<reciter>/<surah>.parquet` | that recording's segments, audio embedded as 16 kHz mono FLAC |
| `raw/<reciter>/<surah>.mp3` | the recording it came from |
| `segment_params.json` | the settings used |

That naming carries the safety properties, and they are properties rather than
conventions -- there is no bookkeeping to get out of step:

- **Resume is a file listing.** The path encodes the source id, so `--resume`
  asks the repo what exists. Nothing is downloaded and no separate record can
  contradict the data.
- **Re-running a source overwrites its own file.** Duplicates cannot accumulate,
  so nothing ever needs purging or de-duplicating.
- **A source is committed whole or not at all.** Its parquet and its mp3 go up in
  a single atomic commit. A crash leaves the source simply absent, and the next
  `--resume` redoes it. There is no partial state to detect or repair.
- **Nothing is buffered.** A crash costs at most the recording in progress.

```python
from datasets import load_dataset
ds = load_dataset("<user>/warsh-segments", split="train", streaming=True)
```

Roughly 1500 files for a full Warsh corpus, averaging ~17 MB. Per-file overhead
across a training epoch is minutes against hours, and more files give the
streaming shuffler finer granularity.

To read the whole corpus as a manifest without downloading any audio -- parquet
is columnar, so the audio column is never fetched:

```bash
warsh-data manifest <user>/warsh-segments -o segments.jsonl
```

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
