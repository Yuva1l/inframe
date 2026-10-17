"""Auto music (docs/AUTO_MUSIC.md): synthetic audio from ffmpeg only; Openverse is a stub, so no test reaches the network."""
import json
import re
import subprocess
import urllib.parse
import wave
from pathlib import Path

import pytest

from inframe import automusic, music as music_mod
from inframe.cli import main
from inframe.cut import cut_clip

FAST = automusic.ReelShape([5.0] * 6, [0.6] * 6, 30.0)  # T = 30 s, 12 clips a minute: fast
EXPR = "(0.9*sin(2*PI*80*t)*exp(-mod(t-0.3,%s)*30)*gte(t,0.3)+0.3*sin(2*PI*1250*t))"


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


def _wav(path, expr, seconds):
    _ff("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d={seconds}", str(path))
    return path


@pytest.fixture(scope="module")
def songs(tmp_path_factory):
    """A: 120 BPM. B: 83 BPM (period 0.72 s; at 90 BPM a multiple of the period, 4.67 s, falls inside the 0.6 s padding of a 5 s clip,
    so J would be 1). C: 120 BPM that is 10x quieter from 1.2 s to 12 s (the tracker drops quiet leading beats, so the quiet part must
    come after a loud first beat). 60 s each."""
    d = tmp_path_factory.mktemp("songs")
    return dict(A=_wav(d / "A.wav", EXPR % "0.5", 60), B=_wav(d / "B.wav", EXPR % "0.72", 60),
                C=_wav(d / "C.wav", EXPR % "0.5" + "*if(between(t,1.2,12),0.1,1)", 60),
                short=_wav(d / "short.wav", EXPR % "0.5", 20))


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(automusic.OFFLINE_ENV, raising=False)
    return tmp_path / "cache"


def row(tid, lic="by", dur=60000, tags=("speed_high", "instrumental"), genres=("electronic",), **kw):
    r = dict(id=tid, title=f"Song {tid[:2]}", creator="Artist", creator_url="https://example.org/a", license=lic,
             license_version="4.0" if lic == "by" else "1.0",
             license_url="https://creativecommons.org/licenses/by/4.0/" if lic == "by" else "https://creativecommons.org/publicdomain/zero/1.0/",
             url=f"https://example.org/{tid}.wav", foreign_landing_url="https://example.org/s", duration=dur, attribution="credit line",
             source="jamendo", filetype="wav", tags=[dict(name=t, accuracy=None) for t in tags], genres=list(genres), mature=False)
    r.update(kw)
    return r


IDS = dict(A="aaaaaaaa-0000-0000-0000-00000000000a", B="bbbbbbbb-0000-0000-0000-00000000000b", C="cccccccc-0000-0000-0000-00000000000c",
           S="dddddddd-0000-0000-0000-00000000000d", Z="eeeeeeee-0000-0000-0000-00000000000e")


class Stub:
    """Stands in for music._get: search pages, per-track records and file bytes. Counts and keeps every request."""

    def __init__(self, rows, files):
        self.rows, self.files, self.calls = rows, files, []

    def __call__(self, url, binary=False, limit=None):
        self.calls.append((url, binary))
        if binary:
            return self.files[url]
        u = urllib.parse.urlparse(url)
        if u.path == "/v1/audio/":
            q = urllib.parse.parse_qs(u.query)
            page = int(q["page"][0])
            return dict(page_count=1, results=[r for r in self.rows if r["license"] == q["license"][0]] if page == 1 else [])
        tid = u.path.strip("/").split("/")[-1]
        return next(r for r in self.rows if r["id"] == tid)

    @property
    def binaries(self):
        return [u for u, b in self.calls if b]


def install(monkeypatch, songs, rows, audio):
    """rows -> the stub; `audio` maps a row id to a wav path."""
    stub = Stub(rows, {r["url"]: Path(audio[r["id"]]).read_bytes() for r in rows if r["id"] in audio})
    monkeypatch.setattr(music_mod, "_get", stub)
    return stub


def standard(monkeypatch, songs):
    rows = [row(IDS["A"]), row(IDS["B"], tags=("speed_low", "instrumental")), row(IDS["C"]),
            row(IDS["S"], lic="by-sa"), row(IDS["Z"], lic="cc0", tags=("speed_high", "instrumental"))]
    return install(monkeypatch, songs, rows, {IDS["A"]: songs["A"], IDS["B"]: songs["B"], IDS["C"]: songs["C"], IDS["S"]: songs["A"],
                                              IDS["Z"]: songs["B"]})


# ---------------------------------------------------------------- selection fixes (review G1)

def _flaky(stub, monkeypatch, track_id, exc, binary_only=True):
    """The stub, except that the file of `track_id` fails with `exc` (a MusicError made from the net error) while `state["on"]`."""
    state = {"on": True, "tries": 0}

    def get(url, binary=False, limit=None):
        if state["on"] and (binary or not binary_only) and track_id in url:
            state["tries"] += 1
            raise music_mod.MusicError(str(exc)) from exc
        return stub(url, binary, limit)

    monkeypatch.setattr(music_mod, "_get", get)
    return state


def _ids(picks):
    return ["builtin" if p.builtin else p.meta["id"][:4] for p in picks]


def test_a_permanent_download_refusal_is_recorded_and_the_list_goes_on(monkeypatch, songs):
    pytest.importorskip("librosa")
    from inframe import net

    rows = [row(IDS["A"]), row(IDS["B"]), row(IDS["C"])]
    stub = install(monkeypatch, songs, rows, {IDS["A"]: songs["A"], IDS["C"]: songs["C"]})
    state = _flaky(stub, monkeypatch, IDS["B"], net.NetError("file is larger than 50 MB"))
    for _ in range(2):
        picks = [automusic.choose(FAST, "", False, k) for k in range(3)]
        assert _ids(picks) == ["aaaa", "cccc", "builtin"]  # B is dropped, C is ranked
        assert not any("list stops here" in w for p in picks for w in p.warnings)
    assert state["tries"] == 1 and automusic._known_bad(IDS["B"])  # asked once, then known bad


def test_only_a_network_failure_is_transient():
    import urllib.error
    from inframe import net

    def of(raise_in_open=None, ct="audio/mpeg", length="10", status=None):
        class Resp:
            headers = {"Content-Type": ct, "Content-Length": length}

            def read(self, n=-1):
                return b""

            def close(self):
                pass

        def fake_open(url, kind):
            if raise_in_open:
                raise raise_in_open
            return Resp()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(net, "_open", fake_open)
            mp.setattr(net, "check_url", lambda url, kind: ("example.org", 443))
            mp.setattr(music_mod, "net", net)
            try:
                net.fetch("https://example.org/x", kind="audio", max_bytes=1000)
            except net.NetError as e:
                err = e
        return automusic._transient(_wrapped(err))

    def _wrapped(e):
        try:
            raise music_mod.MusicError(str(e)) from e
        except music_mod.MusicError as m:
            return m

    assert of(urllib.error.URLError("connection refused")) is True  # could not reach the host
    assert of(ct="text/html") is False  # sent a web page, not audio
    assert of(length="999999") is False  # larger than the cap


def test_a_transient_error_mid_batch_never_moves_a_ranked_song(monkeypatch, songs):
    pytest.importorskip("librosa")
    from inframe import net

    rows = [row(IDS["A"]), row(IDS["B"], tags=("speed_low", "instrumental")), row(IDS["C"])]
    stub = install(monkeypatch, songs, rows, {IDS["A"]: songs["A"], IDS["B"]: songs["B"], IDS["C"]: songs["C"]})
    state = _flaky(stub, monkeypatch, IDS["C"], net.NetError("HTTP 503", 503))
    during = [automusic.choose(FAST, "", False, k) for k in range(4)]
    assert any("list stops here" in w for w in during[0].warnings)
    state["on"] = False
    after = [automusic.choose(FAST, "", False, k) for k in range(4)]
    for k, p in enumerate(during):  # a song shown at a rank is the song at that rank once the network is back
        assert p.builtin or p.meta["id"] == after[k].meta["id"], (k, _ids(during), _ids(after))


def test_a_local_failure_while_analysing_is_not_recorded_as_a_bad_song(monkeypatch, songs):
    pytest.importorskip("librosa")
    rows = [row(IDS["A"]), row(IDS["B"])]
    install(monkeypatch, songs, rows, {IDS["A"]: songs["A"], IDS["B"]: songs["B"]})
    real = automusic._duration

    def no_ffprobe(audio):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(automusic, "_duration", no_ffprobe)
    c = automusic.choose(FAST, "", False, 0)
    assert c.builtin and any("could not be analysed" in w for w in c.warnings)
    monkeypatch.setattr(automusic, "_duration", real)
    c = automusic.choose(FAST, "", False, 0)
    assert not c.builtin and c.n == 3
    assert not any("error" in (json.loads(f.read_text())) for f in (music_mod.cache_dir() / "auto" / "analysis").glob("*.json"))


def test_a_song_ffmpeg_cannot_read_is_recorded_as_bad_and_the_next_song_gets_its_turn(monkeypatch, songs):
    pytest.importorskip("librosa")
    rows = [row(IDS["Z"], lic="cc0"), row(IDS["A"])]  # the cc0 row ranks first on the tie
    install(monkeypatch, songs, rows, {IDS["A"]: songs["A"]})
    cd = music_mod.cache_dir()
    cd.mkdir(parents=True, exist_ok=True)
    (cd / f"{IDS['Z']}.mp3").write_bytes(b"ID3" + b"\x00" * 4000)  # starts like audio, but ffprobe and ffmpeg cannot read it
    (cd / f"{IDS['Z']}.json").write_text(json.dumps(dict(rows[0], duration_s=60)))
    for _ in range(2):
        picks = [automusic.choose(FAST, "", False, k) for k in range(2)]
        assert _ids(picks) == ["aaaa", "builtin"]
        assert not any("list stops here" in w for p in picks for w in p.warnings)
    assert automusic._known_bad(IDS["Z"])


def test_a_refusal_caused_by_the_network_setup_is_not_recorded_as_a_bad_song(monkeypatch, songs):
    from inframe import net

    rows = [row(IDS["A"]), row(IDS["B"])]
    stub = install(monkeypatch, songs, rows, {IDS["A"]: songs["A"], IDS["B"]: songs["B"]})
    e = net.NetError("refusing example.org: it resolves to a non-public address")
    state = _flaky(stub, monkeypatch, "", e)  # every song file fails: a DNS answer of 10.x from a captive portal
    for _ in range(3):
        c = automusic.choose(FAST, "", False, 0, analyse=False)
        assert c.builtin and any("could not be reached" in w for w in c.warnings)
    assert not automusic._known_bad(IDS["A"]) and not automusic._known_bad(IDS["B"])
    assert state["tries"] == 3  # asked again each time, never given up on


def test_a_failed_fallback_search_is_cached_for_a_short_time(monkeypatch, songs):
    stub = install(monkeypatch, songs, [row(IDS["A"])], {IDS["A"]: songs["A"]})
    monkeypatch.setattr(automusic, "can_analyse", lambda: False)
    down = {"on": True}

    def get(url, binary=False, limit=None):
        if down["on"] and not binary and "q=instrumental" in url:
            stub.calls.append((url, binary))
            raise music_mod.MusicError("Openverse is rate-limiting anonymous requests (HTTP 429)")
        return stub(url, binary, limit)

    monkeypatch.setattr(music_mod, "_get", get)

    def searches():
        return sum(1 for u, b in stub.calls if not b and "?" in u)

    automusic.choose(FAST, "", False, 0)
    first = searches()
    assert first > 0
    c = automusic.choose(FAST, "", False, 0)
    automusic.choose(FAST, "", False, 1)
    assert searches() == first and any("fallback search failed" in w for w in c.warnings)  # no new request within the short time
    f = next((music_mod.cache_dir() / "auto" / "search").glob("*.json"))
    d = json.loads(f.read_text())
    d["saved"] -= automusic.PARTIAL_TTL_S + 1
    f.write_text(json.dumps(d))
    down["on"] = False
    automusic.choose(FAST, "", False, 0)
    assert searches() > first  # asked again once the short time passed
    again = searches()
    automusic.choose(FAST, "", False, 0)
    assert searches() == again  # and now it is a full 7-day cache entry


# ---------------------------------------------------------------- 1, 2: scores and shape

def test_meta_score_prefers_a_matching_long_song_and_breaks_ties_by_license_then_id():
    good = dict(id="1", license="by", duration_s=60, tags=["speed_high", "instrumental"], genres=["electronic"])
    loop = dict(good, duration_s=20)
    slow = dict(good, tags=["speed_low", "instrumental"])
    s = automusic.meta_score
    assert s(good, FAST, "energetic") > s(loop, FAST, "energetic") and s(good, FAST, "energetic") > s(slow, FAST, "energetic")
    assert s(good, FAST, "electronic") > s(good, FAST, "jazz")  # the mood word is in the genres
    rows = [dict(good, id="b2"), dict(good, id="a2", license="cc0"), dict(good, id="a1")]
    assert [r["id"] for r, _ in automusic._order(rows, FAST, "energetic")] == ["a2", "a1", "b2"]  # CC0 first, then id


def test_parse_spec():
    assert automusic.parse_spec("auto:calm") == "calm" and automusic.parse_spec("auto") == ""
    assert automusic.parse_spec("x.mp3") is None and automusic.parse_spec(None) is None
    with pytest.raises(ValueError):
        automusic.parse_spec("auto:a/b")


def test_the_reel_shape_sets_speed_and_default_mood():
    assert (FAST.speed, FAST.default_mood) == ("fast", "energetic")
    slow = automusic.ReelShape([30.0, 30.0], [1.0, 1.0], 30.0)
    assert (slow.speed, slow.default_mood) == ("slow", "calm")
    assert automusic.ReelShape([10.0] * 6, [1.0] * 6, 30.0).speed == "medium"


# ---------------------------------------------------------------- 3: the audio score

def test_audio_score_ranks_a_before_c_before_b(songs):
    pytest.importorskip("librosa")
    an = {k: automusic.analyse_file(songs[k], k, FAST) for k in "ABC"}
    sc = {k: automusic.audio_score(an[k], FAST, 0.8) for k in "ABC"}
    assert sc["A"]["score"] > sc["C"]["score"] > sc["B"]["score"], sc
    assert sc["A"]["J"] == 1.0 and sc["B"]["J"] < 1.0 and sc["B"]["Tb"] == 0 and sc["C"]["Q"] < 0.5
    assert all(0 <= v <= 1 for s in sc.values() for v in s.values())
    again = automusic.analyse_file(songs["A"], "A", FAST)  # from the cache, equal
    assert again == an["A"] and an["A"]["analysed_s"] == pytest.approx(60, abs=0.5)


# ---------------------------------------------------------------- 4-8: choose with the stub

def test_choose_end_to_end_is_deterministic_and_never_fetches_other_licenses(monkeypatch, songs):
    pytest.importorskip("librosa")
    stub = standard(monkeypatch, songs)
    c0 = automusic.choose(FAST, "", False, 0)
    assert c0.meta["id"] == IDS["A"] and c0.rank == 0 and not c0.wrapped and not c0.builtin and c0.mood == "energetic"
    assert c0.n == 5 and c0.audio.is_file() and c0.meta["license"] == "by" and c0.score["score"] > 0.5 and c0.bpm == pytest.approx(120, abs=2)
    c1 = automusic.choose(FAST, "", False, 1)
    assert c1.meta["id"] != IDS["A"] and c1.rank == 1
    assert IDS["S"] not in "".join(u for u, _ in stub.calls)  # the by-sa song is dropped before any request about it
    n = len(stub.calls)
    again = [automusic.choose(FAST, "", False, k) for k in range(5)]
    assert len(stub.calls) == n  # the search, the files and the analyses all come from the caches
    assert [c.meta["id"] for c in again][:2] == [c0.meta["id"], c1.meta["id"]]
    assert again[4].builtin
    w = automusic.choose(FAST, "", False, 5)
    assert w.wrapped and w.meta["id"] == c0.meta["id"] and w.rank == 0
    assert c0.summary() == dict(rank=0, n=5, mood="energetic", speed="fast", score=c0.score, bpm=c0.bpm, builtin=False, wrapped=False)


def test_requests_carry_only_the_mood_word(monkeypatch, songs):
    pytest.importorskip("librosa")
    stub = standard(monkeypatch, songs)
    automusic.choose(automusic.ReelShape([4.123456] * 6, [0.61] * 6, 29.97), "calm", False, 1)
    seen = [u for u, b in stub.calls if not b]
    assert seen
    for u in seen:
        p = urllib.parse.urlparse(u)
        assert p.hostname == "api.openverse.org"
        q = urllib.parse.parse_qs(p.query)
        if p.path == "/v1/audio/":
            assert set(q) <= {"q", "category", "license", "source", "page_size", "page"} and q["q"][0] in ("calm", automusic.FALLBACK_QUERY)
        else:
            assert re.fullmatch(r"/v1/audio/[0-9a-f-]+/", p.path) and not p.query
        assert ".mov" not in u and "4.12" not in u and "29.97" not in u and not re.search(r"\d\.\d", u)


def test_at_most_eight_songs_are_downloaded_and_the_built_in_beat_is_last(monkeypatch, songs):
    pytest.importorskip("librosa")
    rows = [row(f"{i:08x}-0000-0000-0000-000000000000") for i in range(1, 31)]
    stub = install(monkeypatch, songs, rows, {r["id"]: songs["short"] for r in rows})
    picks = [automusic.choose(FAST, "", False, k) for k in range(10)]
    assert len(stub.binaries) <= 8
    assert [p.builtin for p in picks] == [False] * 8 + [True, False]
    assert picks[9].wrapped and picks[9].meta["id"] == picks[0].meta["id"] and picks[8].n == 9
    assert len({p.meta["id"] for p in picks[:8]}) == 8


def test_the_built_in_beat_is_the_fallback_when_nothing_is_found(monkeypatch, songs, cache, tmp_path):
    def broken(url, binary=False, limit=None):
        raise music_mod.MusicError("Openverse is rate-limiting anonymous requests (HTTP 429)")

    monkeypatch.setattr(music_mod, "_get", broken)
    c = automusic.choose(FAST, "", False, 0)
    assert c.builtin and c.n == 1 and any("Openverse could not be reached" in w for w in c.warnings)
    assert c.bpm in automusic.SPEEDS["fast"][3]
    with wave.open(str(c.audio)) as w:
        assert w.getnframes() / w.getframerate() == pytest.approx(120, abs=0.1)
    meta = json.loads(c.audio.with_suffix(".json").read_text())
    assert meta["license"] == "cc0" and meta["builtin"] and re.fullmatch(r"inframe built-in beat, \d{3} BPM", meta["title"])
    counted = Stub([], {})
    monkeypatch.setattr(music_mod, "_get", counted)
    assert music_mod.get(meta["id"]) == c.audio and not counted.calls  # music.get serves it from the cache
    monkeypatch.setenv(automusic.OFFLINE_ENV, "1")
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "other"))
    o = automusic.choose(FAST, "", False, 0)
    assert o.builtin and not counted.calls and o.n == 1 and o.rank == 0


def test_cc0_only_never_fetches_by_songs(monkeypatch, songs):
    pytest.importorskip("librosa")
    stub = standard(monkeypatch, songs)
    picks = [automusic.choose(FAST, "", True, k) for k in range(3)]
    assert [p.meta["id"] for p in picks[:1]] == [IDS["Z"]] and picks[1].builtin and picks[2].wrapped
    assert not any("license=by&" in u for u, _ in stub.calls)
    assert {u for u in stub.binaries} == {f"https://example.org/{IDS['Z']}.wav"}


def test_without_librosa_the_metadata_order_is_used_and_only_the_chosen_song_is_fetched(monkeypatch, songs):
    stub = standard(monkeypatch, songs)
    monkeypatch.setattr(automusic, "can_analyse", lambda: False)
    c = automusic.choose(FAST, "", False, 0)
    assert c.meta["id"] == IDS["Z"] and len(stub.binaries) == 1  # CC0 first among equal metadata scores
    assert "beat sync needs the music extra" in c.warnings and c.bpm is None
    assert not (music_mod.cache_dir() / "auto" / "analysis").exists()


def test_search_cache_is_used_for_seven_days_and_a_failed_search_falls_back_to_it(monkeypatch, songs):
    stub = standard(monkeypatch, songs)
    monkeypatch.setattr(automusic, "can_analyse", lambda: False)
    automusic.choose(FAST, "", False, 0)
    searches = sum(1 for u, b in stub.calls if not b and "?" in u)
    automusic.choose(FAST, "", False, 0)
    assert sum(1 for u, b in stub.calls if not b and "?" in u) == searches
    f = next((music_mod.cache_dir() / "auto" / "search").glob("*.json"))
    d = json.loads(f.read_text())
    d["saved"] -= 8 * 86400
    f.write_text(json.dumps(d))

    def broken(url, binary=False, limit=None):
        if binary or "?" not in url:
            return stub(url, binary, limit)
        raise music_mod.MusicError("HTTP 500")

    monkeypatch.setattr(music_mod, "_get", broken)
    c = automusic.choose(FAST, "", False, 0)
    assert c.meta["id"] == IDS["Z"] and any("using the songs found earlier" in w for w in c.warnings)


def test_auto_music_downloads_are_capped_and_evicted_by_last_use(monkeypatch, songs):
    monkeypatch.setattr(automusic, "KEEP_TRACKS", 3)
    pytest.importorskip("librosa")
    rows = [row(f"{i:08x}-0000-0000-0000-000000000000") for i in range(1, 9)]
    install(monkeypatch, songs, rows, {r["id"]: songs["short"] for r in rows})
    user = music_mod.get(rows[7]["id"])  # a track the user fetched with `music get`: never evicted
    c = automusic.choose(FAST, "", False, 7)
    ours = automusic._index()
    assert len(ours) <= 3 and rows[7]["id"] not in ours  # the user's download is not ours to track
    assert user.is_file() and c.audio.is_file()
    assert len(list(music_mod.cache_dir().glob("*.wav"))) <= 4  # three of ours and the user's


# ---------------------------------------------------------------- 9, 10: the CLI

@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("autosrc")
    plain, rot = d / "plain.mp4", d / "src.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=24", "-f", "lavfi", "-i", "sine=frequency=330:duration=24",
        "-c:v", "libx265", "-x265-params", "open-gop=1:keyint=30:min-keyint=30:bframes=4:scenecut=0:log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=full",
        "-tag:v", "hvc1", "-pix_fmt", "yuvj420p", "-c:a", "aac", "-shortest", str(plain))
    _ff("-display_rotation", "-90", "-i", str(plain), "-c", "copy", str(rot))
    return str(rot)


@pytest.fixture(scope="module")
def clips(src, tmp_path_factory):
    """Three padded clips (the last second of each is padding) and the doc that describes them."""
    d = tmp_path_factory.mktemp("autoclips")
    spans = [(1.0, 5.3), (8.0, 12.9), (15.2, 19.4)]
    paths = []
    for i, (s, e) in enumerate(spans):
        o = d / f"src_{i + 1:02d}.mov"
        cut_clip(src, o, s, e)
        paths.append(str(o))
    doc = dict(source=dict(path="/x/src.mov"), settings=dict(pad=1.0), presence_intervals=[[s + 1, e - 1] for s, e in spans],
               clips=[dict(index=i + 1, start=s, end=e, file=Path(p).name) for i, ((s, e), p) in enumerate(zip(spans, paths))])
    return dict(paths=paths, dir=d, doc=doc)


def make_run(clips, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for c in clips["paths"]:
        (run / Path(c).name).write_bytes(Path(c).read_bytes())
    (run / "segments.json").write_text(json.dumps(clips["doc"]))
    return run


def test_cli_export_with_auto_music_and_another_song(monkeypatch, songs, clips, tmp_path, capsys):
    pytest.importorskip("librosa")
    standard(monkeypatch, songs)
    run = make_run(clips, tmp_path)
    d0, d1 = tmp_path / "d0", tmp_path / "d1"
    assert main(["export", str(run), "--reel", "--music", "auto:calm", "--out", str(d0)]) == 0
    printed = capsys.readouterr().out
    assert re.search(r'music: auto pick 1 of \d+ \(".*" by Artist, CC BY 4\.0, 12\d BPM, fit [01]\.\d\d\); another song: --music-rank 1', printed), printed
    ex = json.loads((d0 / "export.json").read_text())
    mu = ex["reel"]["music"]
    assert ex["reel"]["ok"] and mu["auto"]["rank"] == 0 and mu["beat_sync"] is True and mu["title"] and mu["creator"] == "Artist"
    assert (d0 / "src_credits.txt").is_file() and mu["credits"] == "src_credits.txt"
    assert main(["export", str(run), "--reel", "--music", "auto:calm", "--music-rank", "1", "--no-beat-sync", "--out", str(d1)]) == 0
    mu1 = json.loads((d1 / "export.json").read_text())["reel"]["music"]
    assert mu1["file"] != mu["file"] and mu1["auto"]["rank"] == 1 and mu1["beat_sync"] is False
    capsys.readouterr()
    assert main(["music", "auto", str(run), "--mood", "calm", "--rank", "1", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [Path(mu["file"]).stem, Path(mu1["file"]).stem] == [s["id"] for s in listed["songs"][:2]]
    assert [s["chosen"] for s in listed["songs"]][:2] == [False, True] and listed["songs"][-1]["builtin"]
    assert main(["music", "auto", str(run), "--mood", "calm"]) == 0
    assert re.search(r"^\*  0 ", capsys.readouterr().out, re.M)


def test_cli_music_rank_needs_auto_and_a_good_mood(clips, tmp_path, capsys):
    run = make_run(clips, tmp_path)
    assert main(["export", str(run), "--reel", "--music", "x.mp3", "--music-rank", "1"]) == 2
    assert "--music-rank needs --music auto" in capsys.readouterr().err
    assert main(["export", str(run), "--reel", "--music-rank", "1"]) == 2
    assert main(["export", str(run), "--reel", "--music", "auto", "--music-rank", "64"]) == 2
    assert main(["export", str(run), "--reel", "--music", "auto:a/b"]) == 2
    assert main(["export", str(run), "--reel", "--beat-sync"]) == 2
    assert "--beat-sync needs --music" in capsys.readouterr().err


def test_cut_with_auto_music_offline_writes_the_reel_with_the_built_in_beat(static_box, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(automusic.OFFLINE_ENV, "1")
    monkeypatch.setattr(music_mod, "_get", lambda *a, **k: pytest.fail("a request was sent"))
    out = tmp_path / "o"
    assert main(["cut", static_box, "--mode", "motion", "--pad", "0.5", "--min-len", "0.5", "--music", "auto", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    ex = json.loads((out / "export.json").read_text())
    assert ex["reel"]["ok"] and ex["reel"]["music"]["auto"]["builtin"] and ex["reel"]["music"]["creator"] == "inframe"
    assert (out / ex["reel"]["file"]).is_file() and "music: auto pick 1 of 1" in printed
    assert not list(out.glob("*_credits.txt"))
    empty = tmp_path / "e"
    assert main(["cut", static_box, "--mode", "motion", "--min-len", "30", "--music", "auto", "--out", str(empty)]) == 0
    assert "reel not built: no clips" in capsys.readouterr().out and not (empty / "export.json").exists()
