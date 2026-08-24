"""Discovery, manifest I/O, and the mp3quran API parsing -- all offline."""

from __future__ import annotations

import json

import pytest
from conftest import make_record

from warshdata import fetch, manifest
from warshdata.manifest import SegmentRecord
from warshdata.sources import clip_name, discover, segment_id, slugify


# -- sources ------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Ibrahim Al-Dosary", "ibrahim-al-dosary"),
    ("  Rachid  Belalya ", "rachid-belalya"),
    ("Abdelmoujib Benkirane", "abdelmoujib-benkirane"),
    ("!!!", ""),
])
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_discover_uses_parent_directory_as_reciter(tmp_path):
    (tmp_path / "Ibrahim Al-Dosary").mkdir()
    (tmp_path / "Ibrahim Al-Dosary" / "002.mp3").write_bytes(b"x")
    (tmp_path / "loose.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me")

    found = discover(tmp_path)
    assert [(s.reciter_slug, s.source_id) for s in found] == [
        ("ibrahim-al-dosary", "ibrahim-al-dosary/002"),
        ("unknown", "unknown/loose"),
    ]


def test_discover_accepts_a_single_file(tmp_path):
    path = tmp_path / "087.mp3"
    path.write_bytes(b"x")
    assert len(discover(path)) == 1


def test_discover_ignores_non_audio(tmp_path):
    (tmp_path / "readme.md").write_text("x")
    assert discover(tmp_path) == []


def test_segment_id_is_independent_of_boundaries(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "087.mp3").write_bytes(b"x")
    source = discover(tmp_path)[0]
    assert segment_id(source, 3) == "r__087__0003"


def test_clip_name_carries_boundaries():
    assert clip_name("r__087__0003", 198400, 214400) == "r__087__0003__12400-13400ms"
    # Nudging a boundary changes the filename, never the id.
    assert clip_name("r__087__0003", 200000, 214400) == "r__087__0003__12500-13400ms"


# -- manifest -----------------------------------------------------------------

def test_manifest_append_and_read(tmp_path):
    path = tmp_path / "segments.jsonl"
    manifest.append(path, [SegmentRecord(**make_record(i)) for i in range(3)])
    manifest.append(path, [SegmentRecord(**make_record(3))])
    assert len(list(manifest.read(path))) == 4


def test_manifest_read_skips_a_torn_final_line(tmp_path):
    path = tmp_path / "segments.jsonl"
    manifest.append(path, [SegmentRecord(**make_record(0))])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"segment_id": "hal')  # killed mid-write
    records = list(manifest.read(path))
    assert len(records) == 1


def test_manifest_read_missing_file_is_empty(tmp_path):
    assert list(manifest.read(tmp_path / "nope.jsonl")) == []


def test_done_sources(tmp_path):
    path = tmp_path / "segments.jsonl"
    manifest.append(path, [
        SegmentRecord(**make_record(0)),
        SegmentRecord(**make_record(1, source_id="other/002")),
    ])
    assert manifest.done_sources(path) == {"ibrahim-aldosari/087", "other/002"}


def test_arabic_text_survives_a_round_trip(tmp_path):
    path = tmp_path / "segments.jsonl"
    manifest.append(path, [SegmentRecord(**make_record(0, reciter_slug="ورش"))])
    assert list(manifest.read(path))[0]["reciter_slug"] == "ورش"


# -- fetch --------------------------------------------------------------------

API_SAMPLE = {
    "reciters": [
        {"id": 1, "name": "Ibrahim Aldosari", "moshaf": [
            {"name": "Rewayat Warsh A'n Nafi' - Murattal",
             "server": "https://server10.mp3quran.net/ibrahim_dosri/Rewayat-Warsh-A-n-Nafi",
             "surah_total": 114, "surah_list": ",".join(str(i) for i in range(1, 115))}]},
        {"id": 2, "name": "Ahmad Deban", "moshaf": [
            {"name": "Rewayat Warsh A'n Nafi' Men Tariq Alazraq - Murattal",
             "server": "https://server16.mp3quran.net/deban/x/",
             "surah_total": 114, "surah_list": ",".join(str(i) for i in range(1, 115))}]},
        {"id": 3, "name": "Mohammad Abdullkarem", "moshaf": [
            {"name": "Rewayat Warsh A'n Nafi' Men  Tariq Abi Baker Alasbahani - Murattal",
             "server": "https://server12.mp3quran.net/m_krm/y/",
             "surah_total": 114, "surah_list": "1,2,3"}]},
        {"id": 4, "name": "Some Hafs Reciter", "moshaf": [
            {"name": "Rewayat Hafs A'n Assem - Murattal",
             "server": "https://server1.mp3quran.net/hafs/", "surah_total": 114, "surah_list": "1"}]},
        {"id": 5, "name": "Younes Souilass", "moshaf": [
            {"name": "Rewayat Warsh A'n Nafi' - Murattal",
             "server": "https://server16.mp3quran.net/souilass/z/",
             "surah_total": 65, "surah_list": "1,2,5,9"}]},
    ]
}


@pytest.fixture
def stub_api(monkeypatch):
    monkeypatch.setattr(fetch, "_get", lambda url, timeout=60: json.dumps(API_SAMPLE).encode("utf-8"))


def test_only_warsh_reciters_are_returned(stub_api):
    slugs = [m.slug for m in fetch.list_moshafs("warsh")]
    assert "some-hafs-reciter" not in slugs


def test_tariq_alazraq_is_standard_warsh_and_kept(stub_api):
    """al-Azraq is the usual route for Warsh; only Alasbahani genuinely differs."""
    slugs = [m.slug for m in fetch.list_moshafs("warsh")]
    assert "ahmad-deban" in slugs
    assert not any("variant" in s for s in slugs)


def test_alasbahani_is_excluded_by_default_and_slugged_when_included(stub_api):
    default = {m.slug for m in fetch.list_moshafs("warsh")}
    assert not any("abdullkarem" in s for s in default)

    with_variant = fetch.list_moshafs("warsh", include_variant_tariq=True)
    variant = [m for m in with_variant if m.variant_tariq]
    assert [m.slug for m in variant] == ["mohammad-abdullkarem--variant-tariq"]


def test_server_url_is_normalised_and_surah_url_built(stub_api):
    m = [x for x in fetch.list_moshafs("warsh") if x.slug == "ibrahim-aldosari"][0]
    # The sample omits the trailing slash; it must be added, not doubled.
    assert m.server.endswith("Rewayat-Warsh-A-n-Nafi/")
    assert fetch.surah_url(m, 7).endswith("/007.mp3")
    assert fetch.surah_url(m, 114).endswith("/114.mp3")


def test_partial_surah_list_is_parsed(stub_api):
    m = [x for x in fetch.list_moshafs("warsh") if x.slug == "younes-souilass"][0]
    assert m.surahs == [1, 2, 5, 9]
    assert m.n_surahs == 4


def test_empty_surah_list_falls_back_to_the_total():
    assert fetch._parse_surah_list("", 5) == [1, 2, 3, 4, 5]
    assert fetch._parse_surah_list(None, None) == list(range(1, 115))


def test_download_skips_an_existing_file(tmp_path, stub_api):
    m = fetch.list_moshafs("warsh")[0]
    dest = tmp_path / m.slug
    dest.mkdir(parents=True)
    (dest / "007.mp3").write_bytes(b"already here")

    result = fetch.download_surah(m, 7, tmp_path)
    assert result["status"] == "skipped"


def test_download_failure_is_reported_not_raised(tmp_path, stub_api, monkeypatch):
    m = fetch.list_moshafs("warsh")[0]

    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", boom)
    result = fetch.download_surah(m, 7, tmp_path, retries=1)
    assert result["status"] == "failed"
    # A partial file must never be left behind looking complete.
    assert not (tmp_path / m.slug / "007.mp3").exists()
    assert not (tmp_path / m.slug / "007.mp3.part").exists()
