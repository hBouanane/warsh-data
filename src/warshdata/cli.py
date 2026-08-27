"""``warsh-data`` command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import __version__, manifest
from .align import align_surah
from .sources import discover


def _is_cuda_fatal(exc: BaseException) -> bool:
    """True for CUDA errors that leave the context dead rather than the call."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in text and any(
        marker in text
        for marker in ("illegal memory access", "device-side assert",
                       "unspecified launch failure", "uncorrectable")
    )


def _surah_of(source) -> Optional[int]:
    """Surah number from the filename, which is how mp3quran names its files."""
    digits = "".join(c for c in source.path.stem if c.isdigit())
    return int(digits) if digits else None

# ``segment`` pulls in torch and transformers.  It is imported inside the segment
# command so that ``stats``, ``--dry-run`` and ``--help`` work on a machine with
# only the manifest and no model stack installed.


def cmd_segment(args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    manifest_path = out_dir / "segments.jsonl"
    params_path = out_dir / "segment_params.json"
    clips_dir = None if args.no_clips else out_dir / "segments"

    sources = discover(Path(args.input))
    if not sources:
        print(f"No audio found under {args.input}", file=sys.stderr)
        return 1

    store = None
    if args.push_to and not args.dry_run:
        from .hub import HubStore

        store = HubStore(
            repo_id=args.push_to,
            work_dir=out_dir / "_staging",
            private=args.private,
            upload_raw=not args.no_raw,
        )

    if args.sources_file:
        wanted = {
            line.strip()
            for line in Path(args.sources_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        pending = [s for s in sources if s.source_id in wanted]
        unknown = wanted - {s.source_id for s in sources}
        if unknown:
            print(f"warning: {len(unknown)} source id(s) not found under {args.input}",
                  file=sys.stderr)
            for name in sorted(unknown)[:10]:
                print(f"  {name}", file=sys.stderr)
        already: set[str] = set()
    else:
        already = manifest.done_sources(manifest_path) if args.resume else set()
        if store is not None and args.resume:
            # The published files are the authority on what is done.  The path
            # encodes the source, so this is a listing, not a download.
            published = store.done_sources()
            if args.recheck_asr:
                # A source published before transcription existed looks finished
                # by filename alone.  This reads one column per published file to
                # find those, which costs a request each -- hence opt-in.
                print(f"Checking {len(published)} published source(s) for transcripts ...")
                stale = store.sources_missing_column("asr")
                if stale:
                    print(f"  {len(stale)} published without transcripts; redoing them")
                published -= stale
            already |= published
        pending = [s for s in sources if s.source_id not in already]

    if args.limit:
        pending = pending[: args.limit]

    print(f"Found {len(sources)} recording(s); {len(already)} already done; "
          f"{len(pending)} to process.")

    if args.dry_run:
        for source in pending:
            print(f"  {source.source_id}  <-  {source.path}")
        return 0

    if not pending:
        return 0

    # Imported here rather than at module scope: this is the only path needing
    # torch, so --help, --dry-run and a bad input stay usable without it.
    from .segment import MODEL_ID, SegmentParams, Segmenter

    params = SegmentParams(
        min_silence_duration_ms=args.min_silence_duration_ms,
        min_speech_duration_ms=args.min_speech_duration_ms,
        pad_duration_ms=args.pad_duration_ms,
        max_duration_ms=args.max_duration_ms,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )
    segmenter = Segmenter(params)

    transcriber = None
    if args.asr:
        from .asr import Transcriber

        transcriber = Transcriber(model_id=args.asr, checkpoint=args.asr_checkpoint,
                                  device=args.device, batch_size=args.asr_batch_size,
                                  decoder=args.asr_decoder)
    reference = None
    if args.align:
        if transcriber is None:
            print("--align needs --asr: there is nothing to align without transcripts",
                  file=sys.stderr)
            return 1
        from . import quran as quran_text

        reference = quran_text.load(args.quran_text)

    settings = {
        "model_id": MODEL_ID,
        "warsh_data_version": __version__,
        **asdict(params),
        "resolved_dtype": str(segmenter.dtype).replace("torch.", ""),
    }
    manifest.write_params(params_path, settings)
    if store is not None:
        store.upload_params(settings)

    total, failures = 0, 0
    for n, source in enumerate(pending, start=1):
        try:
            records, wave = segmenter.segment(source, clips_dir=clips_dir)
            if not records:
                raise RuntimeError("no speech intervals found")

            if transcriber is not None:
                surah_number = _surah_of(source)
                clips = [wave[r.start_sample : r.end_sample].numpy() for r in records]
                transcripts = transcriber.transcribe(clips)
                for record, transcript in zip(records, transcripts):
                    record.asr = transcript
                    record.surah_number = surah_number

                if reference is not None and surah_number in reference:
                    surah = reference[surah_number]
                    result = align_surah(surah, transcripts)
                    for record, spot in zip(records, result.segments):
                        # The label is the reference text, never the transcript:
                        # that is what keeps recognition errors out of the data.
                        record.label = spot.label or None
                        record.ref_start = spot.ref_start if spot.ref_start >= 0 else None
                        record.ref_end = spot.ref_end if spot.ref_end >= 0 else None
                        record.ayah_start = spot.verses[0] if spot.verses else None
                        record.ayah_end = spot.verses[-1] if spot.verses else None
                        record.align_distance = round(spot.distance, 4)
                        record.align_ok = spot.ok
                        record.is_formula = spot.formula
                        record.is_repeat = spot.repeat
                    good = sum(1 for r in records if r.align_ok)
                    print(f"      aligned {good}/{len(records)} "
                          f"(mean distance {result.distance:.3f})")
                elif reference is not None:
                    print(f"      no surah number in {source.source_id}; not aligned",
                          file=sys.stderr)

            if store is not None:
                # One source, one commit: the parquet and the mp3 that produced
                # it land together or not at all.
                store.write_source(
                    reciter_slug=source.reciter_slug,
                    source_stem=source.path.stem,
                    records=[asdict(r) for r in records],
                    waves=[wave[r.start_sample : r.end_sample].numpy() for r in records],
                    raw_path=source.path,
                )
        except Exception as exc:
            # This source is simply absent from the repo, so a later --resume
            # retries it.  Nothing partial is left behind to clean up.
            failures += 1
            print(f"[{n}/{len(pending)}] FAILED {source.source_id}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            if _is_cuda_fatal(exc):
                # An illegal memory access leaves the CUDA context unusable, so
                # every source after this one would fail too.  Stopping keeps
                # the failure to one file instead of the whole remaining run;
                # --resume picks up from here after a restart.
                print("\nCUDA context is unrecoverable -- stopping. Restart the "
                      "runtime and re-run with --resume.", file=sys.stderr)
                break
            continue

        manifest.append(manifest_path, records)
        total += len(records)
        flag = "" if records[0].source_is_complete else "  [incomplete: last segment not waqf-bounded]"
        print(f"[{n}/{len(pending)}] {source.source_id}: {len(records)} segments{flag}")

    if store is not None:
        print(f"Published {store.rows_written} segments from {store.sources_written} "
              f"source(s) to {args.push_to}")
    print(f"Wrote {total} segments to {manifest_path}"
          + (f" ({failures} source(s) failed)" if failures else ""))
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

    def pct(sorted_vals, q):
        return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]

    durations = sorted(r["duration_seconds"] for r in records)
    hours = sum(durations) / 3600
    incomplete = sum(
        1 for r in records if not r.get("source_is_complete", True) and r.get("is_last_of_source")
    )

    print(f"Segments      : {len(records)}")
    print(f"Sources       : {len({r['source_id'] for r in records})}")
    print(f"Audio         : {hours:.2f} h")
    print(f"Duration (s)  : min {durations[0]:.2f}  p50 {pct(durations, 0.5):.2f}  "
          f"p95 {pct(durations, 0.95):.2f}  max {durations[-1]:.2f}")
    print(f"Unbounded end : {incomplete} segment(s) from cut-off recordings")

    by_reciter = defaultdict(list)
    for r in records:
        by_reciter[r["reciter_slug"]].append(r)

    # Per reciter, because one set of thresholds does not fit every pace: a fast
    # reciter over-segments on the same silence floor that makes a slow one
    # under-segment.  The two tails are the tell.
    print()
    print(f"{'reciter':<30} {'segs':>6} {'audio':>7} {'p5':>6} {'p50':>6} {'p95':>6} "
          f"{'max':>6} {'<1s':>6} {'>20s':>6}")
    print("-" * 92)
    suspects = []
    for slug, rows in sorted(by_reciter.items()):
        d = sorted(x["duration_seconds"] for x in rows)
        short = sum(1 for x in d if x < 1.0) / len(d)
        long_ = sum(1 for x in d if x > 20.0) / len(d)
        print(f"{slug:<30} {len(d):>6} {sum(d) / 3600:>6.2f}h "
              f"{pct(d, 0.05):>6.1f} {pct(d, 0.5):>6.1f} {pct(d, 0.95):>6.1f} {d[-1]:>6.1f} "
              f"{short * 100:>5.1f}% {long_ * 100:>5.1f}%")
        if short > 0.10:
            suspects.append(f"{slug}: {short * 100:.0f}% of segments under 1 s -- "
                            f"likely over-segmenting, try a higher --min-silence-duration-ms")
        if long_ > 0.10:
            suspects.append(f"{slug}: {long_ * 100:.0f}% of segments over 20 s -- "
                            f"waqf being missed, and past the model's 20 s window; "
                            f"try a lower --min-silence-duration-ms")

    if suspects:
        print()
        print("Worth a listen:")
        for line in suspects:
            print(f"  {line}")
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


def cmd_manifest(args: argparse.Namespace) -> int:
    """Build a manifest from a repo's shards, reading no audio."""
    from .hub import read_rows

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in read_rows(args.repo):
            fh.write(json.dumps(row, ensure_ascii=False) + chr(10))
            count += 1
    print(f"Wrote {count} rows to {out}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Flag source recordings that look wrong before any GPU time is spent."""
    from .audit import bitrate_table, find_outliers, probe_all, summary

    sources = discover(Path(args.input))
    if not sources:
        print(f"No audio found under {args.input}", file=sys.stderr)
        return 1

    print(f"Probing {len(sources)} file(s) ...")
    probes = probe_all(sources, workers=args.workers)
    info = summary(probes)
    print(f"{info['files']} files, {info['readable']} readable, "
          f"{info['hours']} h, {info['gigabytes']} GB")

    if args.bitrates:
        print("")
        print(f"{'reciter':<30} {'kbps':>8} {'files':>6}")
        print("-" * 46)
        for slug, bps, count in bitrate_table(probes):
            print(f"{slug:<30} {bps * 8 / 1000:>8.0f} {count:>6}")

    findings = find_outliers(probes, factor=args.factor)
    print("")
    if not findings:
        print("Nothing suspicious.")
        return 0

    print(f"{len(findings)} suspect file(s):")
    print("")
    for finding in findings:
        print("  " + finding.line())

    print("")
    print("A 'too long' or 'too short' file is compared against the median for the")
    print("same surah across reciters, so it means this recording disagrees with the")
    print("others -- listen before trusting it.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([f.__dict__ for f in findings], indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    if args.write_ids:
        out = Path(args.write_ids)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(chr(10).join(f.source_id for f in findings) + chr(10), encoding="utf-8")
        print(f"Wrote {len(findings)} source id(s) to {out}")
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    """Cut short excerpts from suspect recordings into one playable page."""
    from .audit import find_outliers, probe_all
    from .preview import build_page, extract_excerpts

    sources = {s.source_id: s for s in discover(Path(args.input))}
    if not sources:
        print(f"No audio found under {args.input}", file=sys.stderr)
        return 1

    notes = {}
    if args.ids:
        wanted = [
            line.strip()
            for line in Path(args.ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        # No list given: audit first, so `listen` alone is a complete workflow.
        print(f"Auditing {len(sources)} file(s) ...")
        findings = find_outliers(probe_all(list(sources.values()), workers=args.workers),
                                 factor=args.factor)
        wanted = []
        for finding in findings:
            if finding.source_id not in notes:
                wanted.append(finding.source_id)
            notes[finding.source_id] = f"{finding.kind}: {finding.detail}"
        if not wanted:
            print("Nothing suspicious to listen to.")
            return 0

    missing = [sid for sid in wanted if sid not in sources]
    for sid in missing:
        print(f"warning: {sid} not found under {args.input}", file=sys.stderr)
    wanted = [sid for sid in wanted if sid in sources]
    if not wanted:
        print("No matching recordings.", file=sys.stderr)
        return 1

    probes = {p.source_id: p for p in probe_all([sources[sid] for sid in wanted])}

    out_dir = Path(args.output)
    groups = []
    for sid in wanted:
        probe = probes.get(sid)
        if probe is None or not probe.duration_seconds:
            print(f"warning: {sid} has no readable duration, skipping", file=sys.stderr)
            continue
        excerpts = extract_excerpts(
            sid, sources[sid].path, probe.duration_seconds,
            out_dir / "clips", count=args.clips, clip_seconds=args.seconds,
        )
        if not excerpts:
            print(f"warning: could not extract excerpts from {sid}", file=sys.stderr)
            continue
        groups.append((sid, notes.get(sid, "flagged for review"),
                       probe.duration_seconds, excerpts))
        print(f"  {sid}: {len(excerpts)} excerpt(s)")

    if not groups:
        print("Nothing to play.", file=sys.stderr)
        return 1

    page = build_page(groups, out_dir / "index.html")
    print("")
    print(f"Wrote {page}")
    print(f"Clips also in {out_dir / 'clips'}")
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
    p.add_argument("--sources-file", metavar="PATH",
                   help="restrict to the source ids listed in this file, one per line, "
                        "and process them even if --resume would skip them")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed and exit")
    p.add_argument("--min-silence-duration-ms", type=int, default=200)
    p.add_argument("--min-speech-duration-ms", type=int, default=400)
    p.add_argument("--pad-duration-ms", type=int, default=40)
    p.add_argument("--max-duration-ms", type=int, default=19995)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--asr", nargs="?", const="mohammed/fastconformer-quran-ar",
                   metavar="MODEL_ID",
                   help="transcribe each segment with this NeMo model; bare --asr "
                        "uses mohammed/fastconformer-quran-ar")
    p.add_argument("--asr-checkpoint", default=None,
                   help="checkpoint file inside the model repo")
    p.add_argument("--asr-batch-size", type=int, default=8)
    p.add_argument("--asr-decoder", choices=["ctc", "rnnt"], default="ctc",
                   help="which head of the hybrid model decodes (default: ctc, "
                        "which has no autoregressive loop to overflow)")
    p.add_argument("--align", action="store_true",
                   help="align the transcripts to the Warsh text and label each segment")
    p.add_argument("--recheck-asr", action="store_true",
                   help="with --resume, also redo sources published without transcripts "
                        "(reads one column per published file, so it is not free)")
    p.add_argument("--quran-text", default=None, help="path to the Warsh JSON")
    p.add_argument("--push-to", metavar="REPO_ID", help="stream shards to this HF dataset repo as they fill")
    p.add_argument("--private", action="store_true", help="create the HF dataset repo private")
    p.add_argument("--no-raw", action="store_true", help="do not upload the source mp3s")
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

    p = sub.add_parser("audit", help="flag suspect source recordings before segmenting")
    p.add_argument("input", help="audio directory laid out as <root>/<reciter-slug>/<file>")
    p.add_argument("--factor", type=float, default=3.0,
                   help="how far off the per-surah median counts as suspect (default: 3x)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bitrates", action="store_true", help="also show median kbps per reciter")
    p.add_argument("--json", metavar="PATH", help="write findings as JSON")
    p.add_argument("--write-ids", metavar="PATH", help="write suspect source ids, one per line")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("listen", help="excerpt suspect recordings into a playable page")
    p.add_argument("input", help="audio directory laid out as <root>/<reciter-slug>/<file>")
    p.add_argument("-o", "--output", default="./listen", help="output directory")
    p.add_argument("--ids", metavar="PATH",
                   help="file of source ids to excerpt; without it, audit and use its findings")
    p.add_argument("--clips", type=int, default=3, help="excerpts per recording (default: 3)")
    p.add_argument("--seconds", type=float, default=15.0, help="excerpt length (default: 15)")
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("manifest", help="build a manifest from a published repo")
    p.add_argument("repo", help="HF dataset repo id")
    p.add_argument("-o", "--output", default="./segments.jsonl")
    p.set_defaults(func=cmd_manifest)

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
