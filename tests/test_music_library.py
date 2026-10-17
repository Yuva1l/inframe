"""The songs that ship inside the package (src/inframe/music_library/): the committed library is complete and within its budget, and auto
music, `music.get`, the serve routes and `inframe music list` use it with the network blocked. A small synthetic library covers the
rank logic; the packaged one is only read (never changed) and checked by hash."""
import hashlib
import json
import re
import socket
import subprocess
import wave
from pathlib import Path

import pytest
from serve_helpers import call, start_server

from inframe import automusic, music, musiclib, net
from inframe.cli import main

PACKAGED = Path(musiclib.PACKAGED)
FAST = automusic.ReelShape([5.0] * 6, [0.6] * 6, 30.0)
SLOW = automusic.ReelShape([20.0] * 2, [0.6] * 2, 30.0)
EXPR = "(0.9*sin(2*PI*80*t)*exp(-mod(t-0.3,%s)*30)*gte(t,0.3)+0.3*sin(2*PI*1250*t))"
IDS = ["11111111-0000-0000-0000-000000000001", "22222222-0000-0000-0000-000000000002", "33333333-0000-0000-0000-000000000003"]
ONLINE = "99999999-0000-0000-0000-000000000009"


def _wav(path, period, seconds=40):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", f"aevalsrc='{EXPR % period}':s=22050:d={seconds}", str(path)], check=True)
    return path


def song(i, bpm, license="by", **kw):
    s = dict(id=IDS[i], title=f"Song {i}", artist=f"Artist {i}", artist_url=None, license=license, license_version="3.0" if license == "by" else None,
             license_url="https://creativecommons.org/licenses/by/3.0/" if license == "by" else "https://creativecommons.org/publicdomain/zero/1.0/",
             source_url=f"https://example.org/track/{i}", source="jamendo", duration_s=40.0, bpm=bpm, lufs=-14.0, moods=["calm"], tags=["instrumental"],
             genres=["electronic"], sha256="0" * 64, bytes=1, file=f"{IDS[i]}.wav", transcoded=False)
    s.update(kw)
    return s


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """Three 40 s songs: 120 BPM (CC BY), 83 BPM (CC BY), 120 BPM (CC0)."""
    d = tmp_path_factory.mktemp("lib")
    for i, period in enumerate(("0.5", "0.72", "0.5")):
        _wav(d / f"{IDS[i]}.wav", period)
    songs = [song(0, 120.0), song(1, 83.0), song(2, 120.0, license="cc0")]
    (d / "library.json").write_text(json.dumps(dict(schema=1, songs=songs)))
    return d


@pytest.fixture
def on(lib, monkeypatch, tmp_path):
    monkeypatch.setenv("INFRAME_MUSIC_LIBRARY", str(lib))
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(automusic.OFFLINE_ENV, raising=False)


@pytest.fixture
def blocked(monkeypatch):
    """No request can leave: the music and net seams fail the test, and so does opening a socket."""
    def boom(*a, **k):
        pytest.fail("the network was used")
    monkeypatch.setattr(music, "_get", boom)
    monkeypatch.setattr(net, "fetch", boom)
    real = socket.socket.connect

    def connect(self, address):  # the test client talks to the local server; any other address is the internet
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "localhost", "::1"):
            return real(self, address)
        pytest.fail(f"the network was used: {address}")
    monkeypatch.setattr(socket.socket, "connect", connect)


# ---------------------------------------------------------------- the committed library

def test_the_packaged_library_has_eight_songs_with_complete_records():
    doc = json.loads((PACKAGED / "library.json").read_text())
    assert doc["schema"] == 1 and len(doc["songs"]) == 8
    ids = [s["id"] for s in doc["songs"]]
    assert len(set(ids)) == 8
    for s in doc["songs"]:
        assert all(s.get(k) not in (None, "") for k in musiclib.REQUIRED), s["title"]
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
        assert s["license"] == "cc0" or (s["license"] == "by" and s["license_version"] in ("3.0", "4.0")), s["title"]
        assert "creativecommons.org" in s["license_url"] and s["source_url"].startswith("https://")
        assert 60 <= s["duration_s"] <= 240 and 70 <= s["bpm"] < 180
        assert music.ID_RE.match(s["id"])


def test_every_packaged_file_matches_its_hash_and_the_total_is_within_25_mb():
    doc = json.loads((PACKAGED / "library.json").read_text())
    total = 0
    for s in doc["songs"]:
        f = PACKAGED / s["file"]
        assert f.is_file() and f.suffix in (".mp3", ".m4a")
        assert hashlib.sha256(f.read_bytes()).hexdigest() == s["sha256"], s["file"]
        assert f.stat().st_size == s["bytes"]
        total += f.stat().st_size
    assert total == doc["total_bytes"] <= musiclib.TOTAL_BUDGET_BYTES == 25_000_000
    assert {p.name for p in PACKAGED.iterdir()} == {"library.json", "CREDITS.md"} | {s["file"] for s in doc["songs"]}


def test_the_songs_differ_in_tempo_and_mood():
    songs = json.loads((PACKAGED / "library.json").read_text())["songs"]
    assert len({round(s["bpm"] / 10) for s in songs}) >= 5
    assert len({m for s in songs for m in s["moods"]}) >= 5


def test_credits_have_one_attribution_line_per_cc_by_song_and_say_what_changed():
    doc = json.loads((PACKAGED / "library.json").read_text())
    credits = (PACKAGED / "CREDITS.md").read_text()
    for s in doc["songs"]:
        line = musiclib.tasl_line(s)
        assert credits.count(line) == 1, s["title"]
        assert s["title"] in line and s["artist"] in line and s["source_url"] in line and s["license_url"] in line
        if s["license"] == "by":
            assert f"CC BY {s['license_version']}" in line
    n_by = sum(s["license"] == "by" for s in doc["songs"])
    assert len([ln for ln in credits.splitlines() if ln.startswith("- ")]) == len(doc["songs"]) >= n_by
    transcoded = [s for s in doc["songs"] if s["transcoded"]]
    assert ("converted to AAC 160 kbit/s" in credits) == bool(transcoded)
    for s in transcoded:
        assert s["file"].endswith(".m4a") and "Changes were made" in musiclib.tasl_line(s)


def _build_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_music_library", Path(__file__).resolve().parent.parent / "tools" / "build_music_library.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_build_script_verifies_the_committed_library(capsys):
    mod = _build_module()
    assert mod.verify(PACKAGED) == 0
    assert "8 songs" in capsys.readouterr().out


def test_the_wheel_config_ships_the_library():
    import tomllib

    cfg = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "src/inframe/music_library/*" in cfg.get("artifacts", []) or "src/inframe" in cfg["packages"]


def test_the_packaged_library_is_found_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv(musiclib.LIBRARY_ENV)
    assert len(musiclib.songs()) == 8
    assert all(music.get(s["id"]).read_bytes()[:4] for s in musiclib.songs())  # no network: get() returns the file


def test_auto_music_with_the_network_blocked_picks_a_bundled_song_from_the_package(monkeypatch, blocked, tmp_path):
    monkeypatch.delenv(musiclib.LIBRARY_ENV)
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    ids = {s["id"] for s in musiclib.songs()}
    for rank in range(8):
        c = automusic.choose(FAST, "", False, rank, analyse=False)
        assert c.meta["id"] in ids and not c.builtin and c.meta["bundled"] is True
        assert c.audio.parent == PACKAGED and c.audio.is_file()
        assert c.rank == rank and c.n == 9 and not c.wrapped
    assert len({automusic.choose(FAST, "", False, r, analyse=False).meta["id"] for r in range(8)}) == 8


# ---------------------------------------------------------------- auto music over a small library

def test_ranks_before_the_end_of_the_library_are_bundled_and_make_no_request(on, blocked):
    r = automusic.rank_list(FAST, "", False, 1, analyse=False)
    assert [e.meta["id"] for e in r.entries] == [e.meta["id"] for e in r.entries if e.meta["bundled"]] and len(r.entries) == 3
    assert not r.complete and r.n == 4 and r.warnings == []
    # a fast reel suits 120 BPM better than 83 BPM
    assert r.entries[-1].meta["id"] == IDS[1]
    assert r.entries[0].meta["license"] == "cc0"  # equal fit: CC0 first


def test_the_choice_is_deterministic_with_and_without_the_beat_tracker(on, blocked):
    for analyse in (False, True):
        a = [automusic.choose(FAST, "", False, k, analyse=analyse).meta["id"] for k in range(3)]
        b = [automusic.choose(FAST, "", False, k, analyse=analyse).meta["id"] for k in range(3)]
        assert a == b and len(set(a)) == 3


def test_a_slow_reel_prefers_the_slow_song(on, blocked):
    assert automusic.choose(SLOW, "", False, 0, analyse=False).meta["id"] == IDS[1]


def test_cc0_only_leaves_only_the_cc0_song(on, blocked):
    r = automusic.rank_list(FAST, "", True, 0, analyse=False)
    assert [e.meta["id"] for e in r.entries] == [IDS[2]]
    assert automusic.choose(FAST, "", True, 0, analyse=False).meta["license"] == "cc0"


def test_past_the_last_bundled_song_the_online_songs_follow_and_the_built_in_beat_ends_the_list(on, monkeypatch, tmp_path):
    online = tmp_path / "o.wav"
    _wav(online, "0.5")
    row = dict(id=ONLINE, title="Online", creator="Someone", creator_url=None, license="by", license_version="4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
               url="https://example.org/o.wav", foreign_landing_url="https://example.org/o", duration=40000, attribution="x", source="jamendo", filetype="wav",
               tags=[dict(name="speed_high", accuracy=None)], genres=["rock"], mature=False)
    asked = []

    def fake(url, binary=False, limit=None):
        asked.append(url)
        if binary:
            return online.read_bytes()
        if "?" in url:
            return dict(page_count=1, results=[row] if "license=by" in url else [])
        return row

    monkeypatch.setattr(music, "_get", fake)
    first = automusic.rank_list(FAST, "", False, 2, analyse=False)
    assert asked == [] and len(first.entries) == 3  # rank 2 is still bundled
    r = automusic.rank_list(FAST, "", False, 3, analyse=False)
    assert asked and [e.meta["id"] for e in r.entries[:3]] == [e.meta["id"] for e in first.entries]
    assert r.entries[3].meta["id"] == ONLINE and r.entries[-1].builtin
    assert automusic.choose(FAST, "", False, 3, analyse=False).meta["id"] == ONLINE


def test_when_the_network_is_down_a_rank_past_the_library_still_gets_a_song(on, monkeypatch):
    def down(url, binary=False, limit=None):
        raise music.MusicError("could not reach api.openverse.org") from net.NetError("could not reach api.openverse.org")

    monkeypatch.setattr(music, "_get", down)
    c = automusic.choose(FAST, "", False, 3, analyse=False)
    assert c.builtin and any("could not be reached" in w for w in c.warnings)


def test_offline_switch_keeps_the_bundled_songs(on, monkeypatch, blocked):
    monkeypatch.setenv(automusic.OFFLINE_ENV, "1")
    r = automusic.rank_list(FAST, "", False, 3, analyse=False)
    assert [e.meta["bundled"] for e in r.entries[:3]] == [True] * 3 and r.entries[-1].builtin and r.warnings == []


# ---------------------------------------------------------------- music.get, records, credits

def test_get_returns_a_bundled_song_without_a_request_and_its_record_has_the_credit(on, blocked):
    p = music.get(IDS[0])
    assert p.parent == musiclib.library_dir() and p.name == f"{IDS[0]}.wav"
    meta = music.load_meta(p)
    assert meta["license"] == "by" and meta["bundled"] and meta["creator"] == "Artist 0"
    text = music.credits_text(meta)
    assert '"Song 0" by Artist 0 (https://example.org/track/0), licensed under CC BY 3.0' in text
    assert music.resolve(IDS[0])[0] == p and music.resolve(str(p))[1]["id"] == IDS[0]


def test_cc0_only_refuses_a_bundled_cc_by_song_and_accepts_the_cc0_one(on, blocked):
    with pytest.raises(music.LicenseError):
        music.get(IDS[0], cc0_only=True)
    assert music.get(IDS[2], cc0_only=True).name == f"{IDS[2]}.wav"


def test_a_library_with_a_bad_record_ignores_it(tmp_path, monkeypatch):
    d = tmp_path / "bad"
    d.mkdir()
    s1, s2, s3 = song(0, 120.0), song(1, 80.0, license="by-sa"), song(2, 100.0, title="")
    (d / "library.json").write_text(json.dumps(dict(schema=1, songs=[s1, s2, s3])))
    for s in (s1, s2, s3):
        (d / s["file"]).write_bytes(b"RIFF....WAVE")
    monkeypatch.setenv(musiclib.LIBRARY_ENV, str(d))
    assert [s["id"] for s in musiclib.songs()] == [IDS[0]]


def test_off_turns_the_library_off(monkeypatch):
    monkeypatch.setenv(musiclib.LIBRARY_ENV, "off")
    assert musiclib.songs() == [] and musiclib.find(IDS[0]) is None


# ---------------------------------------------------------------- serve and cli

def test_serve_lists_and_plays_the_bundled_songs_offline(on, blocked, tmp_path):
    base, app, stop = start_server(tmp_path)
    try:
        r = call(app, "GET", "/api/music/library")
        assert r.status == 200
        d = r.json()
        assert [t["id"] for t in d["tracks"]] == IDS and d["tracks"][0]["bundled"] and d["tracks"][0]["needs_credit"] and not d["tracks"][2]["needs_credit"]
        assert "offline" in d["notice"]
        assert [t["id"] for t in call(app, "GET", "/api/music/library?cc0_only=1").json()["tracks"]] == [IDS[2]]
        a = call(app, "GET", f"/api/music/{IDS[1]}/audio", key="media")
        assert a.status == 200 and a.headers["content-type"] == "audio/wav" and a.body[:4] == b"RIFF"
        g = call(app, "POST", "/api/music/get", {"id": IDS[0]})
        assert g.status == 200 and g.json()["license"] == "by" and g.json()["needs_credit"]
        assert call(app, "POST", "/api/music/get", {"id": IDS[0], "cc0_only": True}).status == 422
    finally:
        stop()


def test_a_library_inside_a_hidden_folder_is_still_served(lib, monkeypatch, blocked, tmp_path):
    """A wheel installed in .venv or ~/.local keeps the songs under a dot folder; the serve path rules would refuse that for a user's own file."""
    import shutil

    hidden = tmp_path / ".venv" / "site-packages" / "inframe" / "music_library"
    shutil.copytree(lib, hidden)
    monkeypatch.setenv(musiclib.LIBRARY_ENV, str(hidden))
    base, app, stop = start_server(tmp_path)
    try:
        a = call(app, "GET", f"/api/music/{IDS[0]}/audio", key="media")
        assert a.status == 200 and a.body[:4] == b"RIFF"
        assert call(app, "GET", f"/api/music/{ONLINE}/audio", key="media").status == 404
    finally:
        stop()


def test_cli_music_list_shows_the_bundled_songs(on, blocked, capsys):
    assert main(["music", "list"]) == 0
    out = capsys.readouterr().out
    assert all(i in out for i in IDS) and "CC BY 3.0" in out and "CC0" in out and "3 songs come with inframe" in out
    assert main(["music", "list", "--cc0-only", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert [s["id"] for s in d["songs"]] == [IDS[2]]


def test_cli_music_list_shows_the_packaged_songs(monkeypatch, capsys):
    monkeypatch.delenv(musiclib.LIBRARY_ENV)
    assert main(["music", "list"]) == 0
    assert "8 songs come with inframe" in capsys.readouterr().out


# ---------------------------------------------------------------- the build script on a rerun, covers, exclusions

def _measured(tid):
    t = music.Track(id=tid, title="T", creator="C", creator_url=None, license="by", license_version="3.0", license_url="https://creativecommons.org/licenses/by/3.0/",
                    url="https://example.org/f", landing_url="https://example.org/t", duration_s=100.0, attribution=None, source="jamendo", filetype="mp32")
    return dict(track=t.to_json(), mood="calm", duration_s=100.0, bpm=100.0, lufs=-14.0, tags=[], genres=[])


def test_a_rerun_keeps_the_downloaded_original_not_the_earlier_aac_transcode(tmp_path):
    """After one build the work folder holds {id}.mp3 and {id}.m4a; revive() used to take whichever glob listed first, so a rerun shipped the
    transcode as the original and the credits lost their 'converted to AAC' line."""
    mod = _build_module()
    tid = IDS[0]
    (tmp_path / f"{tid}.mp3").write_bytes(b"o" * 1000)
    (tmp_path / f"{tid}.m4a").write_bytes(b"t" * 400)  # what an older run left here
    (tmp_path / f"{tid}.json").write_text(json.dumps({"filetype": "mp32"}))
    for _ in range(2):  # first run, and the rerun that finds the leftovers
        c = mod.revive(tmp_path, _measured(tid))
        assert c["file"] == tmp_path / f"{tid}.mp3" and c["bytes"] == 1000

        def fake_transcode(src, dst):
            assert dst != src
            dst.write_bytes(b"t" * 400)
        mod.transcode = fake_transcode
        [c] = mod.fit_budget([c], [c], 1, 500, tmp_path)
        assert c["transcoded"] and c["final"] != c["file"] and c["file"] == tmp_path / f"{tid}.mp3"
        assert (tmp_path / f"{tid}.mp3").read_bytes() == b"o" * 1000  # the original is untouched
        e = mod.entry(c)
        assert e["transcoded"] and e["file"].endswith(".m4a") and e["original_bytes"] == 1000
        assert "were converted to AAC" in mod.credits_md([e], e["bytes"])


def test_an_original_that_is_already_m4a_is_found_by_its_recorded_type(tmp_path):
    mod = _build_module()
    tid = IDS[1]
    (tmp_path / f"{tid}.m4a").write_bytes(b"m" * 10)
    (tmp_path / f"{tid}.mp3").write_bytes(b"x" * 10)
    (tmp_path / f"{tid}.json").write_text(json.dumps({"filetype": "m4a"}))
    assert mod.original_file(tmp_path, tid) == tmp_path / f"{tid}.m4a"


def _mp3_with_cover(path):
    png = path.with_suffix(".png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", str(png)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "sine=f=440:d=2", "-i", str(png), "-map", "0:a", "-map", "1:v",
                    "-c:a", "libmp3lame", "-c:v", "copy", "-disposition:v", "attached_pic", "-metadata", "title=Song", "-id3v2_version", "3", str(path)], check=True)


def _audio_md5(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_the_build_strips_an_embedded_cover_and_keeps_the_audio_and_tags(tmp_path):
    mod = _build_module()
    src = tmp_path / "in.mp3"
    _mp3_with_cover(src)
    assert mod.has_picture(src)
    c = dict(file=src, bytes=src.stat().st_size)
    mod.clean_original(c, tmp_path)
    assert c["cover_removed"] and c["file"] != src and not mod.has_picture(c["file"])
    assert _audio_md5(c["file"]) == _audio_md5(src)
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title", "-of", "csv=p=0", str(c["file"])], capture_output=True, text=True).stdout
    assert "Song" in tags
    c2 = dict(file=c["file"], bytes=1)
    mod.clean_original(c2, tmp_path)  # a file without a picture is used as it is
    assert c2["cover_removed"] is False and c2["file"] == c["file"]


def test_no_packaged_file_carries_a_cover_picture():
    """The Openverse record licenses the audio only; an embedded album cover is a separate work with no license on record."""
    mod = _build_module()
    for f in sorted(PACKAGED.iterdir()):
        if f.suffix in (".mp3", ".m4a"):
            assert not mod.has_picture(f), f.name


def test_a_track_with_the_title_of_a_copyrighted_song_is_not_in_the_library_and_a_rebuild_refuses_it():
    mod = _build_module()
    doc = json.loads((PACKAGED / "library.json").read_text())
    assert not set(mod.EXCLUDED) & {s["id"] for s in doc["songs"]}
    assert not any(s["title"].lower().startswith("jingle bell") for s in doc["songs"])
    t = music.Track(**_measured("22a0cd91-ad66-4c36-ac70-c6d7a3b5c47d")["track"])
    assert "excluded" in mod.acceptable(t)


def test_a_bundled_song_has_one_bpm_with_and_without_the_beat_tracker(tmp_path, monkeypatch):
    """The audio beats at 120 BPM but the library says 100: the dialog, the reel's record and the tempo term must all use the library's number.
    (The order of several songs can still differ with the extra, since the analysis adds the other score terms; this checks one song's BPM.)"""
    pytest.importorskip("librosa")
    d = tmp_path / "lib"
    d.mkdir()
    _wav(d / f"{IDS[0]}.wav", "0.5")
    (d / "library.json").write_text(json.dumps(dict(schema=1, songs=[song(0, 100.0)])))
    monkeypatch.setenv("INFRAME_MUSIC_LIBRARY", str(d))
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(automusic.OFFLINE_ENV, raising=False)
    tb = round(automusic._tempo_term(100.0, FAST), 3)
    for analyse in (False, True):
        [e] = automusic.bundled_entries(FAST, "", False, analyse)
        assert e.bpm == 100.0 and e.meta["bpm"] == 100.0, analyse
        if analyse:
            assert e.score["Tb"] == tb
