"""inframe serve music routes (docs/T9A_DESIGN.md section 3.4, rows 49-52). No network: `music._get` is replaced (one test replaces
the `net` seams instead), the dialog is replaced, the download is a 3 s synthetic WAV."""
import email.message
import json
import math
import struct
import wave

import pytest
from serve_helpers import call, start_server

from inframe import music, net, serve_music

TRACK = "0f0f0f0f-1111-2222-3333-444444444444"
OTHER = "aaaaaaaa-1111-2222-3333-444444444444"


def make_wav(path, seconds=3.0, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i * 0.2))) for i in range(int(rate * seconds))))
    return path.read_bytes()


def record(**kw):
    r = dict(id=TRACK, title="Song", creator="Artist", creator_url=None, license="cc0", license_version="1.0", license_url=None,
             url="https://cdn.example.org/s.wav", foreign_landing_url=None, duration=3000, attribution="credit line", source="jamendo",
             filetype="wav")
    r.update(kw)
    return r


@pytest.fixture
def srv(tmp_path):
    base, app, stop = start_server(tmp_path)
    yield app
    stop()


@pytest.fixture
def wav(tmp_path):
    return make_wav(tmp_path / "s.wav")


@pytest.fixture
def openverse(monkeypatch, wav):
    """Fake Openverse: `ov.records` maps track id -> record; `ov.urls` lists every address asked for."""
    class OV:
        records = {TRACK: record(), OTHER: record(id=OTHER, license="by", license_url="https://creativecommons.org/licenses/by/4.0/")}
        urls: list = []
        fail: Exception | None = None

    def fake(url, binary=False, limit=None):
        OV.urls.append(url)
        if OV.fail:
            raise OV.fail
        if binary:
            return wav
        tid = url.rstrip("/").rsplit("/", 1)[-1]
        if "?" in url:
            return dict(page_count=1, results=[OV.records[TRACK], OV.records[OTHER]])
        return OV.records[tid]

    OV.urls = []
    OV.fail = None
    monkeypatch.setattr(music, "_get", fake)
    return OV


# ---------------------------------------------------------------- 49 search
def test_search_with_a_mood_returns_tracks_and_the_notice(srv, openverse):
    r = call(srv, "GET", "/api/music/search?mood=calm%20piano&min_s=2&limit=5")
    assert r.status == 200
    d = r.json()
    assert [t["id"] for t in d["tracks"]] == [TRACK, OTHER] and d["notice"] == music.OPENVERSE_NOTICE
    assert d["tracks"][0]["license"] == "cc0" and d["tracks"][0]["title"] == "Song"
    assert all("q=calm+piano" in u and "source=jamendo" in u for u in openverse.urls)
    assert any("license=cc0" in u for u in openverse.urls)
    assert call(srv, "GET", "/api/music/search?mood=calm").json()["tracks"][1]["license"] == "by"  # CC BY is on by default
    openverse.urls.clear()
    assert [t["license"] for t in call(srv, "GET", "/api/music/search?cc0_only=1").json()["tracks"]] == ["cc0"]  # unless "CC0 only" is ticked
    assert all("license=by&" not in u for u in openverse.urls)
    assert "q=upbeat" in openverse.urls[-1]  # no mood: the default query


def test_search_takes_only_a_short_plain_mood(srv, openverse):
    for bad in ("a" * 41, "calm;rm", "../../etc", "calm%0Apiano", "x/y", "a%26license%3Dby"):
        r = call(srv, "GET", "/api/music/search?mood=" + bad)
        assert r.status == 400, bad
    assert call(srv, "GET", "/api/music/search?mood=" + "a" * 40).status == 200
    assert call(srv, "GET", "/api/music/search?mood=lo-fi%20beats_2").status == 200
    for q in ("min_s=-1", "min_s=nan", "min_s=abc", "min_s=99999", "limit=0", "limit=51", "limit=2.5", "cc0_only=2", "cc0_only=yes"):
        assert call(srv, "GET", "/api/music/search?" + q).status == 400, q
    assert all(" " not in u and "etc" not in u for u in openverse.urls)


def test_search_failure_is_a_502_and_needs_the_session_token(srv, openverse):
    openverse.fail = music.MusicError("Openverse is rate-limiting anonymous requests (HTTP 429)")
    r = call(srv, "GET", "/api/music/search?mood=calm")
    assert r.status == 502 and "rate-limiting" in r.json()["error"]
    assert call(srv, "GET", "/api/music/search?mood=calm", key=None).status == 403
    assert call(srv, "GET", "/api/music/search?mood=calm", key="media").status == 403  # the media key opens audio files only


# ---------------------------------------------------------------- 50 get, 51 audio
def test_get_downloads_and_row_51_serves_it_with_range(srv, openverse, wav, tmp_path):
    r = call(srv, "POST", "/api/music/get", {"id": TRACK, "cc0_only": False})
    assert r.status == 200
    d = r.json()
    f = tmp_path / "cache" / "music" / f"{TRACK}.wav"
    assert d == {"id": TRACK, "file": str(f), "license": "cc0", "attribution": "credit line", "duration_s": 3.0, "needs_credit": False}
    assert f.read_bytes() == wav and f.with_suffix(".json").is_file()
    n = len(openverse.urls)
    call(srv, "POST", "/api/music/get", {"id": TRACK})
    assert len(openverse.urls) == n  # cached: no second download

    url = f"/api/music/{TRACK}/audio"
    full = call(srv, "GET", url, key="media")
    assert full.status == 200 and full.body == wav and full.headers["content-type"] == "audio/wav" and full.headers["accept-ranges"] == "bytes"
    part = call(srv, "GET", url, key="media", headers={"Range": "bytes=0-99"})
    assert part.status == 206 and part.body == wav[:100] and part.headers["content-range"] == f"bytes 0-99/{len(wav)}"
    tail = call(srv, "GET", url, key="media", headers={"Range": "bytes=-50"})
    assert tail.status == 206 and tail.body == wav[-50:]
    assert call(srv, "GET", url, key="media", headers={"Range": f"bytes={len(wav)}-"}).status == 416
    head = call(srv, "HEAD", url, key="media")
    assert head.status == 200 and head.headers["content-length"] == str(len(wav)) and head.body == b""
    assert call(srv, "GET", url, key="token").status == 200  # the session token works too
    assert call(srv, "GET", url, key=None).status == 403
    assert call(srv, "GET", url + "?m=wrong", key=None).status == 403


def test_bad_ids_are_400_and_unknown_ones_404(srv, openverse):
    for bad in ("../evil", "nothex-zzzzzzzz", "abc", "", "x" * 65, None, 12, ["a"]):
        assert call(srv, "POST", "/api/music/get", {"id": bad}).status == 400, bad
    assert call(srv, "POST", "/api/music/get", {}).status == 400
    for bad in ("evil", "..%2f..%2fetc", "a" * 70, "zzzzzzzz-zzzz"):
        assert call(srv, "GET", f"/api/music/{bad}/audio", key="media").status in (400, 403, 404), bad
    assert call(srv, "GET", "/api/music/evil/audio", key="media").status == 400
    assert call(srv, "GET", f"/api/music/{OTHER}/audio", key="media").status == 404
    assert openverse.urls == []


def test_cc_by_is_served_by_default_and_refused_with_cc0_only(srv, openverse, tmp_path):
    r = call(srv, "POST", "/api/music/get", {"id": OTHER, "cc0_only": True})
    assert r.status == 422 and "--cc0-only" in r.json()["error"]
    assert call(srv, "POST", "/api/music/get", {"id": OTHER, "cc0_only": "yes"}).status == 400
    assert call(srv, "POST", "/api/music/get", {"id": OTHER, "cc0_only": 1}).status == 400
    assert not list((tmp_path / "cache" / "music").glob("*"))  # nothing downloaded
    assert not any(u.endswith(".wav") for u in openverse.urls)
    r = call(srv, "POST", "/api/music/get", {"id": OTHER})
    assert r.status == 200 and r.json()["license"] == "by" and r.json()["needs_credit"] is True
    assert call(srv, "POST", "/api/music/get", {"id": OTHER, "cc0_only": True}).status == 422  # and a cached CC BY track is not handed out with CC0 only


def test_other_licenses_and_upstream_errors(srv, openverse):
    openverse.records[TRACK] = record(license="by-nc")
    r = call(srv, "POST", "/api/music/get", {"id": TRACK})
    assert r.status == 422 and "not CC0 or CC BY" in r.json()["error"]
    for lic in ("by-sa", "by-nd", "by-nc-nd"):  # re-read from the Openverse record each time it is downloaded
        openverse.records[TRACK] = record(license=lic, license_url="https://creativecommons.org/licenses/x/4.0/")
        assert call(srv, "POST", "/api/music/get", {"id": TRACK}).status == 422, lic
    openverse.fail = music.MusicError("could not reach cdn.example.org")
    assert call(srv, "POST", "/api/music/get", {"id": OTHER}).status == 502


def test_get_refuses_a_playlist_download(srv, monkeypatch, tmp_path):
    hls = b"#EXTM3U\n#EXTINF:10,\nfile:///etc/passwd\n"
    monkeypatch.setattr(music, "_get", lambda url, binary=False, limit=None: hls if binary else record())
    monkeypatch.setattr(music, "_check_audio", lambda p: pytest.fail("ffprobe must not run"))
    r = call(srv, "POST", "/api/music/get", {"id": TRACK})
    assert r.status == 502 and "not a playable audio file" in r.json()["error"]
    assert not list((tmp_path / "cache" / "music").glob("*"))


def test_only_regular_audio_files_of_the_cache_are_served(srv, wav, tmp_path):
    cache = tmp_path / "cache" / "music"
    cache.mkdir(parents=True)
    secret = tmp_path / "secret.wav"
    secret.write_bytes(wav)
    (cache / f"{TRACK}.wav").symlink_to(secret)  # a link out of the cache is not a track
    assert call(srv, "GET", f"/api/music/{TRACK}/audio", key="media").status == 404
    (cache / f"{OTHER}.json").write_text("{}")
    (cache / f"{OTHER}.wav.part").write_bytes(wav)
    (cache / f"{OTHER}.txt").write_bytes(wav)
    assert call(srv, "GET", f"/api/music/{OTHER}/audio", key="media").status == 404
    fake = "bbbbbbbb-1111"
    (cache / f"{fake}.mp3").write_bytes(b"#EXTM3U\nfile:///etc/passwd\n")  # right extension, wrong content
    assert call(srv, "GET", f"/api/music/{fake}/audio", key="media").status == 415


def test_search_and_get_go_only_through_net_fetch(srv, monkeypatch, wav):
    """With `net` doing the requests: search stays on api.openverse.org, the file comes from the record's host as audio."""
    asked = []

    def opener(url, kind):
        asked.append((url, kind))
        h = email.message.Message()
        if "cdn.example.org" in url:
            h["Content-Type"] = "audio/wav"
            body = wav
        else:
            h["Content-Type"] = "application/json"
            body = json.dumps(dict(page_count=1, results=[record()]) if "?" in url else record()).encode()
        h["Content-Length"] = str(len(body))

        class R:
            headers = h
            left = body

            def read(self, n=-1):
                out, self.left = self.left[:n], self.left[n:]
                return out

            def close(self):
                pass

        return R()

    monkeypatch.setattr(net, "_open", opener)
    monkeypatch.setattr(net, "_resolve", lambda host, port: ["93.184.216.34"])
    assert call(srv, "GET", "/api/music/search?mood=calm").json()["tracks"][0]["id"] == TRACK
    assert call(srv, "POST", "/api/music/get", {"id": TRACK}).status == 200
    assert {k for _, k in asked} == {"api", "audio"}
    assert all(u.startswith("https://api.openverse.org/") for u, k in asked if k == "api")
    assert ("https://cdn.example.org/s.wav", "audio") in asked
    monkeypatch.setattr(net, "_resolve", lambda host, port: ["10.0.0.5"] if host == "cdn.example.org" else ["93.184.216.34"])
    (srv.cache_root.parent / "music" / f"{TRACK}.wav").unlink()
    (srv.cache_root.parent / "music" / f"{TRACK}.json").unlink()
    r = call(srv, "POST", "/api/music/get", {"id": TRACK})
    assert r.status == 502 and "non-public" in r.json()["error"]


# ---------------------------------------------------------------- 52 choose-file
def test_choose_file_is_501_off_macos(srv, monkeypatch):
    monkeypatch.setattr(serve_music, "_is_mac", lambda: False)
    assert call(srv, "POST", "/api/music/choose-file", {}).status == 501


def test_choose_file_cancel_and_pick(srv, monkeypatch, wav, tmp_path):
    monkeypatch.setattr(serve_music, "_is_mac", lambda: True)
    monkeypatch.setattr(serve_music, "_choose_audio", lambda: None)
    assert call(srv, "POST", "/api/music/choose-file", {}).json() == {"cancelled": True}
    song = tmp_path / "mine" / "song.wav"
    song.parent.mkdir()
    song.write_bytes(wav)
    monkeypatch.setattr(serve_music, "_choose_audio", lambda: str(song))
    r = call(srv, "POST", "/api/music/choose-file", {})
    assert r.status == 200 and r.json() == {"file": str(song.resolve()), "license": None}
    song.with_suffix(".json").write_text(json.dumps({"license": "cc0"}))  # a cached track's metadata beside it
    assert call(srv, "POST", "/api/music/choose-file", {}).json()["license"] == "cc0"
    text = tmp_path / "mine" / "list.mp3"
    text.write_text("#EXTM3U\nfile:///etc/passwd\n")
    monkeypatch.setattr(serve_music, "_choose_audio", lambda: str(text))
    assert call(srv, "POST", "/api/music/choose-file", {}).status == 415
    other = tmp_path / "mine" / "clip.mp4"
    other.write_bytes(wav)
    monkeypatch.setattr(serve_music, "_choose_audio", lambda: str(other))
    assert call(srv, "POST", "/api/music/choose-file", {}).status == 415
    monkeypatch.setattr(serve_music, "_choose_audio", lambda: str(tmp_path / "mine" / "gone.wav"))
    assert call(srv, "POST", "/api/music/choose-file", {}).status == 404


def test_osascript_result_and_cancel(monkeypatch):
    seen = []

    class R:
        def __init__(self, code, out="", err=""):
            self.returncode, self.stdout, self.stderr = code, out, err

    answers = [R(0, "/Users/me/song.mp3\n"), R(1, "", "execution error: User canceled. (-128)"), R(1, "", "boom")]
    monkeypatch.setattr(serve_music.subprocess, "run", lambda cmd, **kw: (seen.append((cmd, kw)), answers.pop(0))[1])
    assert serve_music._choose_audio() == "/Users/me/song.mp3"
    cmd, kw = seen[0]
    assert cmd[0] == "osascript" and "public.audio" in cmd[2] and kw["timeout"] == 600
    assert serve_music._choose_audio() is None
    with pytest.raises(serve_music.HttpError) as e:
        serve_music._choose_audio()
    assert e.value.code == 500
