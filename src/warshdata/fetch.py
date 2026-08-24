"""Downloading Warsh recitations from the mp3quran.net API.

The API (``/api/v3/reciters``) lists every reciter with one *moshaf* entry per
recitation, each carrying a ``server`` base URL and the surahs it covers.  Files
are then ``{server}{surah:03d}.mp3``.  Discovering paths this way rather than
hardcoding them means a reciter added upstream is picked up for free.

Downloads go to a ``.part`` file and are renamed only once complete, so an
interrupted run can never leave a truncated mp3 that a later ``--resume`` would
mistake for a finished one.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .sources import slugify

__all__ = ["Moshaf", "API_URL", "list_moshafs", "download_surah", "surah_url"]

API_URL = "https://mp3quran.net/api/v3/reciters?language=eng"

_USER_AGENT = "warsh-data/0.1 (+https://github.com/hBouanane/warsh-data)"


@dataclass
class Moshaf:
    """One reciter's recitation set."""

    reciter_id: int
    reciter_name: str
    slug: str
    moshaf_name: str
    server: str
    surahs: List[int]
    #: True for Warsh via Tariq Abi Baker al-Asbahani, whose pronunciation
    #: genuinely differs from the standard Tariq al-Azraq.  Excluded by default
    #: so it is never mixed into the same training pool unnoticed.
    variant_tariq: bool = False

    @property
    def n_surahs(self) -> int:
        return len(self.surahs)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_surah_list(raw: str, total: Optional[int]) -> List[int]:
    surahs = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if part.isdigit():
            surahs.append(int(part))
    if not surahs:
        surahs = list(range(1, (total or 114) + 1))
    return sorted(set(surahs))


def list_moshafs(rewaya: str = "warsh", include_variant_tariq: bool = False) -> List[Moshaf]:
    """Fetch the reciter index and return the moshafs matching ``rewaya``.

    Matching is on the moshaf name, which is where the API spells the riwayah
    out (``Rewayat Warsh A'n Nafi'``).
    """
    data = json.loads(_get(API_URL).decode("utf-8"))
    needle = rewaya.lower()

    out: List[Moshaf] = []
    for reciter in data.get("reciters", []):
        for moshaf in reciter.get("moshaf", []):
            name = moshaf.get("name", "") or ""
            if needle not in name.lower():
                continue

            # Tariq al-Azraq is the standard route for Warsh and is *not* a
            # variant -- the API just spells it out for some reciters.  Only
            # Alasbahani genuinely differs in pronunciation.
            squashed = name.lower().replace("-", "").replace(" ", "").replace("'", "")
            variant = "alasbahani" in squashed
            if variant and not include_variant_tariq:
                continue

            server = moshaf.get("server", "")
            if not server:
                continue
            if not server.endswith("/"):
                server += "/"

            slug = slugify(reciter.get("name", f"reciter-{reciter.get('id')}"))
            if variant:
                slug = f"{slug}--variant-tariq"

            out.append(
                Moshaf(
                    reciter_id=int(reciter.get("id", 0)),
                    reciter_name=reciter.get("name", ""),
                    slug=slug,
                    moshaf_name=name,
                    server=server,
                    surahs=_parse_surah_list(moshaf.get("surah_list"), moshaf.get("surah_total")),
                    variant_tariq=variant,
                )
            )
    return sorted(out, key=lambda m: m.slug)


def surah_url(moshaf: Moshaf, surah: int) -> str:
    return f"{moshaf.server}{surah:03d}.mp3"


def download_surah(
    moshaf: Moshaf,
    surah: int,
    dest_dir: Path,
    retries: int = 3,
    timeout: int = 300,
) -> Dict[str, object]:
    """Download one surah.  Returns a small result dict; never raises for a
    network failure, so one dead file does not end a long run."""
    dest_dir = Path(dest_dir) / moshaf.slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{surah:03d}.mp3"
    part = final.with_suffix(".mp3.part")

    if final.exists() and final.stat().st_size > 0:
        return {"status": "skipped", "path": final, "bytes": final.stat().st_size}

    url = surah_url(moshaf, surah)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                expected = resp.headers.get("Content-Length")
                with part.open("wb") as fh:
                    shutil.copyfileobj(resp, fh, length=1 << 20)

            size = part.stat().st_size
            if expected is not None and size != int(expected):
                raise IOError(f"short read: got {size} of {expected} bytes")
            if size == 0:
                raise IOError("empty response")

            # Rename only once the file is known-complete.
            part.replace(final)
            return {"status": "downloaded", "path": final, "bytes": size}

        except (urllib.error.URLError, urllib.error.HTTPError, IOError, TimeoutError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** attempt)

    return {"status": "failed", "path": final, "error": f"{type(last_error).__name__}: {last_error}"}
