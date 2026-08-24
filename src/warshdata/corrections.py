"""Hand corrections, kept separate from the generated manifest.

Reviewing segments by ear always turns up boundaries to nudge, junk to drop, and
clips that should have been two.  Editing ``segments.jsonl`` in place loses all
of that the next time segmentation runs, so corrections live in their own file
and are *applied* to produce the final manifest:

    segments.jsonl  +  corrections.jsonl  ->  segments.final.jsonl

Each correction records the boundaries it was written against.  If the segment
has moved since -- because segmentation was re-run with different thresholds --
the correction is reported as drifted rather than applied blind to audio it was
never reviewed against.

Actions:

``adjust``  new ``start_seconds`` / ``end_seconds`` (either may be omitted)
``drop``    exclude from the final manifest, with the reason retained
``split``   a waqf the model missed: cut at ``at_seconds``, children suffixed
            ``_a``, ``_b``, ...
``merge``   the mirror case -- a boundary the model invented at a breath pause:
            absorb the segments in ``with`` into this one.  The surviving record
            keeps *this* segment's id, so whatever was already attached to it
            stays attached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["Correction", "read_corrections", "apply", "ApplyReport"]

_ACTIONS = {"adjust", "drop", "split", "merge"}


@dataclass
class Correction:
    segment_id: str
    action: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    at_seconds: List[float] = field(default_factory=list)
    #: For ``merge``: ids absorbed into this segment.
    with_ids: List[str] = field(default_factory=list)
    note: str = ""
    #: Boundaries the reviewer actually listened to, used to detect drift.
    orig_start_seconds: Optional[float] = None
    orig_end_seconds: Optional[float] = None


@dataclass
class ApplyReport:
    applied: int = 0
    dropped: int = 0
    split_into: int = 0
    merged_away: int = 0
    unmatched: List[str] = field(default_factory=list)
    drifted: List[str] = field(default_factory=list)
    invalid: List[str] = field(default_factory=list)


def read_corrections(path: Path) -> List[Correction]:
    out: List[Correction] = []
    if not Path(path).exists():
        return out
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            out.append(
                Correction(
                    segment_id=raw["segment_id"],
                    action=raw.get("action", "adjust"),
                    start_seconds=raw.get("start_seconds"),
                    end_seconds=raw.get("end_seconds"),
                    at_seconds=list(raw.get("at_seconds", [])),
                    with_ids=list(raw.get("with", raw.get("with_ids", []))),
                    note=raw.get("note", ""),
                    orig_start_seconds=raw.get("orig_start_seconds"),
                    orig_end_seconds=raw.get("orig_end_seconds"),
                )
            )
    return out


def _suffix(n: int) -> str:
    """a, b, ... z, aa, ab, ...  -- ``chr(ord('a') + n)`` breaks past 26."""
    out = ""
    n += 1
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def _drifted(rec: Dict[str, Any], c: Correction, tol: float = 0.05) -> bool:
    """True when the segment no longer sits where the reviewer heard it."""
    if c.orig_start_seconds is None and c.orig_end_seconds is None:
        return False
    if c.orig_start_seconds is not None and abs(rec["start_seconds"] - c.orig_start_seconds) > tol:
        return True
    if c.orig_end_seconds is not None and abs(rec["end_seconds"] - c.orig_end_seconds) > tol:
        return True
    return False


def _retime(rec: Dict[str, Any], start_s: float, end_s: float) -> Dict[str, Any]:
    sr = rec["sample_rate"]
    rec = dict(rec)
    rec["start_seconds"] = round(start_s, 3)
    rec["end_seconds"] = round(end_s, 3)
    rec["start_sample"] = int(round(start_s * sr))
    rec["end_sample"] = int(round(end_s * sr))
    rec["duration_seconds"] = round(end_s - start_s, 3)
    return rec


def apply(
    records: Iterable[Dict[str, Any]],
    corrections: Iterable[Correction],
    strict_drift: bool = False,
) -> Tuple[List[Dict[str, Any]], ApplyReport]:
    """Apply corrections to manifest records, returning the final records.

    Records with no correction pass through untouched, so a corrections file
    only ever needs to describe the segments a human actually changed.
    """
    by_id = {c.segment_id: c for c in corrections}
    report = ApplyReport()
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # Merges span several records, so the manifest has to be materialised and
    # indexed before anything is emitted.
    records = list(records)
    rec_by_id = {r.get("segment_id"): r for r in records}
    absorbed: Dict[str, str] = {}
    for c in by_id.values():
        if c.action != "merge":
            continue
        for other in c.with_ids:
            absorbed[other] = c.segment_id

    for rec in records:
        if rec.get("segment_id") in absorbed:
            # Emitted as part of the segment that absorbed it.
            report.merged_away += 1
            continue
        sid = rec.get("segment_id")
        c = by_id.get(sid)
        if c is None:
            out.append(rec)
            continue
        seen.add(sid)

        if c.action not in _ACTIONS:
            report.invalid.append(f"{sid}: unknown action {c.action!r}")
            out.append(rec)
            continue

        if _drifted(rec, c):
            report.drifted.append(sid)
            if strict_drift:
                out.append(rec)
                continue

        if c.action == "drop":
            report.dropped += 1
            continue

        if c.action == "merge":
            parts = [rec] + [rec_by_id[i] for i in c.with_ids if i in rec_by_id]
            missing = [i for i in c.with_ids if i not in rec_by_id]
            if missing:
                report.invalid.append(f"{sid}: merge target(s) not in manifest: {', '.join(missing)}")
            foreign = {p["source_id"] for p in parts} - {rec["source_id"]}
            if foreign:
                report.invalid.append(f"{sid}: refusing to merge across sources {sorted(foreign)}")
                out.append(rec)
                continue
            new = _retime(
                rec,
                min(p["start_seconds"] for p in parts),
                max(p["end_seconds"] for p in parts),
            )
            new["corrected"] = True
            new["correction_note"] = c.note
            new["merged_from"] = [p["segment_id"] for p in parts]
            # Spans audio the old clip never covered, so it must be re-cut.
            new["audio_path"] = None
            new["is_last_of_source"] = any(p.get("is_last_of_source") for p in parts)
            out.append(new)
            report.applied += 1
            continue

        if c.action == "adjust":
            start_s = c.start_seconds if c.start_seconds is not None else rec["start_seconds"]
            end_s = c.end_seconds if c.end_seconds is not None else rec["end_seconds"]
            if end_s <= start_s:
                report.invalid.append(f"{sid}: end {end_s} <= start {start_s}")
                out.append(rec)
                continue
            new = _retime(rec, start_s, end_s)
            new["corrected"] = True
            new["correction_note"] = c.note
            out.append(new)
            report.applied += 1
            continue

        # split
        cuts = sorted(t for t in c.at_seconds if rec["start_seconds"] < t < rec["end_seconds"])
        if not cuts:
            report.invalid.append(f"{sid}: no split point inside the segment")
            out.append(rec)
            continue
        edges = [rec["start_seconds"], *cuts, rec["end_seconds"]]
        for n, (a, b) in enumerate(zip(edges, edges[1:])):
            child = _retime(rec, a, b)
            child["segment_id"] = f"{sid}_{_suffix(n)}"
            child["parent_segment_id"] = sid
            child["corrected"] = True
            child["correction_note"] = c.note
            # The clip no longer matches the new boundaries; it must be re-cut.
            child["audio_path"] = None
            out.append(child)
        report.split_into += len(edges) - 1
        report.applied += 1

    report.unmatched = sorted(set(by_id) - seen - set(absorbed))
    return out, report
