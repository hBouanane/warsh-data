"""``warsh-data`` command line entry point."""

from __future__ import annotations

import argparse
import json
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

    writer = None
    if args.push_to:
        from .hub import HubWriter, done_sources_on_hub

        writer = HubWriter(
            repo_id=args.push_to,
            work_dir=out_dir / "_shards",
            shard_bytes=int(args.shard_mb * 1024 * 1024),
            private=args.private,
            upload_raw=not args.no_raw,
        )

    already = manifest.done_sources(manifest_path) if args.resume else set()
    if writer is not None and args.resume:
        # Resuming across Colab sessions: the local manifest is gone but the
        # hub one is not, so ask the hub what has already been done.
        already |= done_sources_on_hub(args.push_to)
    if writer is not None and writer.upload_raw:
        # A source segmented in a session that died before its batch commit
        # is in the manifest but has no raw file. It is excluded from
        # `pending`, so queue it here or it is never uploaded at all.
        on_hub = set(writer.repo_files())
        for s in sources:
            if s.source_id in already and f"raw/{s.reciter_slug}/{s.path.name}" not in on_hub:
                writer.queue_source(s.path, s.reciter_slug)

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

        if writer is not None:
            for rec in records:
                writer.add(asdict(rec), _wave[rec.start_sample : rec.end_sample].numpy())
            writer.queue_source(source.path, source.reciter_slug)
            # Throttled: one manifest commit per source would be thousands of
            # commits over a full pass.
            if n % args.push_every == 0:
                writer.flush_sources()
                writer.push_manifest(manifest_path)
        flag = "" if (records and records[0].source_is_complete) else "  [incomplete: last segment is not waqf-bounded]"
        print(f"[{n}/{len(pending)}] {source.source_id}: {len(records)} segments{flag}")

    if writer is not None:
        # The last shard is almost never exactly full, and the last few
        # sources sit below the push-every threshold -- without this they
        # are simply never uploaded.
        writer.flush()
        writer.flush_sources()
        writer.push_manifest(manifest_path)
        print(f"Pushed {writer.rows_written} segments in {writer.shards_written} "
              f"shard(s) to {args.push_to}")

    print(f"\nWrote {total} segments to {manifest_path}" + (f" ({failures} file(s) failed)" if failures else ""))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .fetch import download_surah, list_moshafs, surah_url

    try:
        moshafs = list_moshafs(args.rewaya, include_variant_tariq=args.include_variant_tariq)
    except Exception as exc:
        print(f"Could not reach the mp3quran API: {exc}", file=sys.stderr)
        return 1

    if args.reciter:
        wanted = {r.lower() for r in args.reciter}
        moshafs = [m for m in moshafs if m.slug.lower() in wanted]
        if not moshafs:
            print("No reciter matched. Run --list to see the available slugs.", file=sys.stderr)
            return 1

    if args.list:
        print(f"{len(moshafs)} moshaf(s) matching rewaya={args.rewaya!r}:\n")
        for m in moshafs:
            flag = "  [variant tariq]" if m.variant_tariq else ""
            print(f"  {m.slug:<32} {m.n_surahs:>3} surahs  {m.moshaf_name}{flag}")
            print(f"  {'':<32} {m.server}")
        return 0

    out_dir = Path(args.output)
    jobs = []
    for m in moshafs:
        surahs = [s for s in m.surahs if not args.surah or s in set(args.surah)]
        jobs.extend((m, s) for s in surahs)

    if args.dry_run:
        for m, s in jobs:
            print(f"  {m.slug}/{s:03d}.mp3  <-  {surah_url(m, s)}")
        print(f"\n{len(jobs)} file(s) would be downloaded to {out_dir}")
        return 0

    print(f"Downloading {len(jobs)} file(s) from {len(moshafs)} reciter(s) to {out_dir}")
    counts = Counter()
    for n, (m, s) in enumerate(jobs, start=1):
        res = download_surah(m, s, out_dir, retries=args.retries)
        counts[res["status"]] += 1
        if res["status"] == "failed":
            print(f"[{n}/{len(jobs)}] FAILED {m.slug}/{s:03d}.mp3: {res['error']}", file=sys.stderr)
        elif res["status"] == "downloaded":
            print(f"[{n}/{len(jobs)}] {m.slug}/{s:03d}.mp3  {res['bytes'] / 1e6:.1f} MB")
        elif n % 25 == 0 or n == len(jobs):
            print(f"[{n}/{len(jobs)}] ...")

    print(f"\nDownloaded {counts['downloaded']}, skipped {counts['skipped']}, failed {counts['failed']}")
    return 1 if counts["failed"] and not counts["downloaded"] else 0


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


def cmd_apply_corrections(args: argparse.Namespace) -> int:
    from .corrections import apply, read_corrections

    records = list(manifest.read(Path(args.manifest)))
    if not records:
        print("Manifest is empty or missing.", file=sys.stderr)
        return 1

    corrections = read_corrections(Path(args.corrections))
    if not corrections:
        print(f"No corrections found in {args.corrections}", file=sys.stderr)
        return 1

    final, report = apply(records, corrections, strict_drift=args.strict_drift)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in final:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{len(records)} in -> {len(final)} out")
    print(f"  applied {report.applied}, dropped {report.dropped}, "
          f"split into {report.split_into}, merged away {report.merged_away}")
    for label, items in (("drifted", report.drifted), ("unmatched", report.unmatched), ("invalid", report.invalid)):
        if items:
            print(f"  {label}: {len(items)}")
            for item in items[:10]:
                print(f"    {item}")
    print(f"Wrote {out}")
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
    p.add_argument("--push-to", metavar="REPO_ID", help="stream shards to this HF dataset repo as they fill")
    p.add_argument("--shard-mb", type=int, default=400, help="audio MB per parquet shard (default: 400)")
    p.add_argument("--private", action="store_true", help="create the HF dataset repo private")
    p.add_argument("--no-raw", action="store_true", help="do not upload the source mp3s")
    p.add_argument("--push-every", type=int, default=10,
                   help="commit the manifest and queued sources every N recordings (default: 10)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto",
                   help="auto: bfloat16 where supported, float16 on older GPUs (e.g. T4), float32 on CPU")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("fetch", help="download recitations from mp3quran.net")
    p.add_argument("-o", "--output", default="./audio", help="output directory (default: ./audio)")
    p.add_argument("--rewaya", default="warsh", help="substring matched against the moshaf name (default: warsh)")
    p.add_argument("--reciter", action="append", help="restrict to this reciter slug; repeatable")
    p.add_argument("--surah", type=int, action="append", help="restrict to this surah number; repeatable")
    p.add_argument("--list", action="store_true", help="list matching reciters and exit")
    p.add_argument("--dry-run", action="store_true", help="list the files that would be downloaded and exit")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--include-variant-tariq", action="store_true",
                   help="also fetch Warsh via Tariq al-Asbahani, whose pronunciation differs")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("stats", help="summarise a segments manifest")
    p.add_argument("manifest", help="path to segments.jsonl")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("apply-corrections", help="apply hand corrections to a manifest")
    p.add_argument("manifest", help="path to segments.jsonl")
    p.add_argument("corrections", help="path to corrections.jsonl")
    p.add_argument("-o", "--output", default="./out/segments.final.jsonl")
    p.add_argument("--strict-drift", action="store_true",
                   help="skip corrections whose recorded boundaries no longer match")
    p.set_defaults(func=cmd_apply_corrections)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
