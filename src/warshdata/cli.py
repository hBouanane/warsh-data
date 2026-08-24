"""``warsh-data`` command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import __version__, manifest
from .sources import discover

# ``segment`` pulls in torch and transformers.  It is imported inside the segment
# command so that ``stats``, ``--dry-run`` and ``--help`` work on a machine with
# only the manifest and no model stack installed.


def cmd_segment(args: argparse.Namespace) -> int:
    from .segment import MODEL_ID, SegmentParams, Segmenter

    out_dir = Path(args.output)
    manifest_path = out_dir / "segments.jsonl"
    params_path = out_dir / "segment_params.json"
    clips_dir = None if args.no_clips else out_dir / "segments"

    sources = discover(Path(args.input))
    if not sources:
        print(f"No audio found under {args.input}", file=sys.stderr)
        return 1

    already = manifest.done_sources(manifest_path) if args.resume else set()
    pending = [s for s in sources if s.source_id not in already]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Found {len(sources)} recording(s); {len(already)} already done; {len(pending)} to process.")

    params = SegmentParams(
        min_silence_duration_ms=args.min_silence_duration_ms,
        min_speech_duration_ms=args.min_speech_duration_ms,
        pad_duration_ms=args.pad_duration_ms,
        max_duration_ms=args.max_duration_ms,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )

    if args.dry_run:
        for s in pending:
            print(f"  {s.source_id}  <-  {s.path}")
        return 0

    if not pending:
        return 0

    segmenter = Segmenter(params)

    # Record the dtype that was actually used, not the literal "auto" -- the
    # point of this file is that a manifest can be reproduced from it.
    manifest.write_params(
        params_path,
        {
            "model_id": MODEL_ID,
            "warsh_data_version": __version__,
            **asdict(params),
            "resolved_dtype": str(segmenter.dtype).replace("torch.", ""),
        },
    )

    total, failures = 0, 0
    for n, source in enumerate(pending, start=1):
        try:
            records, _wave = segmenter.segment(source, clips_dir=clips_dir)
        except Exception as exc:  # one unreadable file must not end the run
            failures += 1
            print(f"[{n}/{len(pending)}] FAILED {source.source_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Written per source, so an interrupted run resumes at file granularity.
        manifest.append(manifest_path, records)
        total += len(records)
        flag = "" if (records and records[0].source_is_complete) else "  [incomplete: last segment is not waqf-bounded]"
        print(f"[{n}/{len(pending)}] {source.source_id}: {len(records)} segments{flag}")

    print(f"\nWrote {total} segments to {manifest_path}" + (f" ({failures} file(s) failed)" if failures else ""))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    records = list(manifest.read(Path(args.manifest)))
    if not records:
        print("Manifest is empty or missing.", file=sys.stderr)
        return 1

    per_reciter = Counter(r["reciter_slug"] for r in records)
    durations = sorted(r["duration_seconds"] for r in records)
    hours = sum(durations) / 3600
    incomplete = sum(1 for r in records if not r.get("source_is_complete", True) and r.get("is_last_of_source"))

    def pct(q: float) -> float:
        return durations[min(len(durations) - 1, int(q * len(durations)))]

    print(f"Segments      : {len(records)}")
    print(f"Sources       : {len({r['source_id'] for r in records})}")
    print(f"Audio         : {hours:.2f} h")
    print(f"Duration (s)  : min {durations[0]:.2f}  p50 {pct(0.5):.2f}  p95 {pct(0.95):.2f}  max {durations[-1]:.2f}")
    print(f"Unbounded end : {incomplete} segment(s) from cut-off recordings")
    print("\nPer reciter:")
    for slug, count in per_reciter.most_common():
        print(f"  {slug:<35} {count:>7}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warsh-data", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("segment", help="cut recordings at waqf into segments + manifest")
    p.add_argument("input", help="audio file, or directory laid out as <root>/<reciter-slug>/<file>")
    p.add_argument("-o", "--output", default="./out", help="output directory (default: ./out)")
    p.add_argument("--no-clips", action="store_true", help="write the manifest only, no clip files")
    p.add_argument("--resume", action="store_true", help="skip sources already in the manifest")
    p.add_argument("--limit", type=int, default=0, help="process at most N recordings")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed and exit")
    p.add_argument("--min-silence-duration-ms", type=int, default=200)
    p.add_argument("--min-speech-duration-ms", type=int, default=400)
    p.add_argument("--pad-duration-ms", type=int, default=40)
    p.add_argument("--max-duration-ms", type=int, default=19995)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto",
                   help="auto: bfloat16 where supported, float16 on older GPUs (e.g. T4), float32 on CPU")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("stats", help="summarise a segments manifest")
    p.add_argument("manifest", help="path to segments.jsonl")
    p.set_defaults(func=cmd_stats)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
