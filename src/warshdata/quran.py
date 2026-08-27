"""The Warsh reference text, indexed for alignment.

A surah is held two ways at once, word for word:

``words``      the *skeleton* -- consonants with letter shapes unified, so
               alef-maqsura/yeh, the hamza forms and teh-marbuta/heh all collapse.
               What alignment matches on.  Measured on real ASR output against
               this Warsh text, matching on the skeleton rather than the plain
               rasm took word error from 44.4% to 9.7%: an ASR trained on Hafs
               orthography writes الذي where the Warsh mushaf has الذے, and every
               such word counts as wrong for one character's difference.

``raw_words``  the full diacritized text, index-for-index with ``words``.  What
               becomes the label once alignment has decided where a segment sits.

Keeping them parallel is the whole point: match on the robust form, label with
the exact one.  The invariant that they stay 1:1 is checked on load, because a
silent drift between them would mislabel everything downstream.

Text source: edition ``ara-quranwarsh`` from qurancomplex.gov.sa, 6236 verses
under Uthmani verse numbering.  Note that Warsh's own Madani numbering differs;
this edition was renumbered so that ayah ids line up with everything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Verse", "Surah", "QuranText", "load", "DEFAULT_PATH"]

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "quran-warsh.json"


def _text_module():
    """warsh-lab's normaliser, which owns the charset rules."""
    from warshlab import text as T

    return T


@dataclass(frozen=True)
class Verse:
    chapter: int
    verse: int
    text: str
    #: Index of this verse's first word in the surah-wide word stream.
    word_start: int
    word_count: int

    @property
    def word_end(self) -> int:
        return self.word_start + self.word_count


@dataclass
class Surah:
    number: int
    verses: List[Verse]
    #: Rasm words for the whole surah, in order.  Alignment works on these.
    words: List[str] = field(repr=False)
    #: Diacritized words, index-for-index with ``words``.  Labels come from these.
    raw_words: List[str] = field(repr=False)
    #: Verse number for each word index.
    word_verse: List[int] = field(repr=False)

    def __len__(self) -> int:
        return len(self.words)

    def verse_at(self, word_index: int) -> int:
        return self.word_verse[word_index]

    def verses_spanned(self, start: int, end: int) -> List[int]:
        """Verse numbers touched by the half-open word range ``[start, end)``."""
        if end <= start:
            return []
        seen = []
        for index in range(max(0, start), min(end, len(self.word_verse))):
            v = self.word_verse[index]
            if not seen or seen[-1] != v:
                seen.append(v)
        return seen

    def label(self, start: int, end: int) -> str:
        """Diacritized reference text for a word range -- the training label."""
        return " ".join(self.raw_words[max(0, start):min(end, len(self.raw_words))])

    def rasm(self, start: int, end: int) -> str:
        return " ".join(self.words[max(0, start):min(end, len(self.words))])


@dataclass
class QuranText:
    surahs: Dict[int, Surah]

    def __getitem__(self, number: int) -> Surah:
        return self.surahs[int(number)]

    def __contains__(self, number: int) -> bool:
        return int(number) in self.surahs

    @property
    def verse_count(self) -> int:
        return sum(len(s.verses) for s in self.surahs.values())

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.surahs.values())


def _build_surah(number: int, rows: Sequence[dict]) -> Surah:
    T = _text_module()

    verses: List[Verse] = []
    words: List[str] = []
    raw_words: List[str] = []
    word_verse: List[int] = []

    for row in rows:
        raw = T.collapse_whitespace(row["text"])
        raw_tokens = T.words(raw)
        # Normalise each token separately rather than the verse as a whole: that
        # is what guarantees the two lists stay index-for-index even if a rule
        # would otherwise merge or drop a token.
        rasm_tokens = [T.to_skeleton(token) for token in raw_tokens]

        keep = [(r, w) for r, w in zip(rasm_tokens, raw_tokens) if r]
        if len(keep) != len(raw_tokens):
            # A token that normalises to nothing (a lone ornament) is dropped
            # from both lists together, never from one.
            pass

        start = len(words)
        for rasm_token, raw_token in keep:
            words.append(rasm_token)
            raw_words.append(raw_token)
            word_verse.append(int(row["verse"]))

        verses.append(Verse(
            chapter=number,
            verse=int(row["verse"]),
            text=raw,
            word_start=start,
            word_count=len(keep),
        ))

    assert len(words) == len(raw_words) == len(word_verse)
    return Surah(number=number, verses=verses, words=words,
                 raw_words=raw_words, word_verse=word_verse)


@lru_cache(maxsize=4)
def load(path: Optional[str] = None) -> QuranText:
    """Load and index the Warsh text.  Cached: this is read many times."""
    source = Path(path) if path else DEFAULT_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Warsh text not found at {source}. Download it with:\n"
            f"  curl -sL https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/"
            f"editions/ara-quranwarsh.json -o {source}"
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["quran"] if isinstance(payload, dict) and "quran" in payload else payload

    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["chapter"]), []).append(row)

    surahs = {
        number: _build_surah(number, sorted(items, key=lambda r: int(r["verse"])))
        for number, items in sorted(grouped.items())
    }
    return QuranText(surahs=surahs)
