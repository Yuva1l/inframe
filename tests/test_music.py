"""T4b: music for the reel. Synthetic clips and a synthetic click track only; the Openverse calls are replaced by a stub."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")

from inframe import music as music_mod  # noqa: E402
from inframe.beats import NoBeatError, beat_times, extend_beats, plan_beat_cuts  # noqa: E402
from inframe.cli import main  # noqa: E402
from inframe.cut import cut_clip  # noqa: E402
from inframe.export import export_run, tail_slack  # noqa: E402
from inframe.musicreel import audio_beat_check, build_music_reel, check_music_reel, pick_encoder, video_params  # noqa: E402
from inframe.probe import probe  # noqa: E402
from inframe.reel import shown_pictures  # noqa: E402
from inframe.soundtrack import measure_lufs, render_soundtrack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import synth_music  # noqa: E402


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


@pytest.fixture(scope="module")
def click(tmp_path_factory):
    """40 s click track at 120 BPM, first click at 0.37 s, on a steady tone (bare clicks cannot reach -14 LUFS under a -1 dBFS peak).
    Returns (path, true beat times)."""
    p = tmp_path_factory.mktemp("clicks") / "click.wav"
    y, beats = synth_music.render(120, 40, 0.37, "click", bed=0.5)
    synth_music.write_wav(str(p), y)
    return str(p), beats


@pytest.fixture(scope="module")
def short_click(tmp_path_factory):
    """9 s click track at 100 BPM from 0.2 s: shorter than the reel, so it loops."""
    p = tmp_path_factory.mktemp("short") / "short.wav"
    y, beats = synth_music.render(100, 9, 0.2, "click", bed=0.5)
    synth_music.write_wav(str(p), y)
    return str(p), beats


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """24 s, 30 fps, 320x180 HEVC with open GOPs, full-range BT.709 tags, display rotation -90, 330 Hz sine audio."""
    d = tmp_path_factory.mktemp("musicsrc")
    plain, rot = d / "plain.mp4", d / "src.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=24", "-f", "lavfi", "-i", "sine=frequency=330:duration=24",
        "-c:v", "libx265", "-x265-params", "open-gop=1:keyint=30:min-keyint=30:bframes=4:scenecut=0:log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=full",
        "-tag:v", "hvc1", "-pix_fmt", "yuvj420p", "-c:a", "aac", "-shortest", str(plain))
    _ff("-display_rotation", "-90", "-i", str(plain), "-c", "copy", str(rot))
    return str(rot)


@pytest.fixture(scope="module")
def clips(src, tmp_path_factory):
    """Three padded clips (the last second of each is padding) cut from the source, and the doc that describes them."""
    d = tmp_path_factory.mktemp("musicclips")
    spans = [(1.0, 5.3), (8.0, 12.9), (15.2, 19.4)]
    paths = []
    for i, (s, e) in enumerate(spans):
        o = d / f"src_{i + 1:02d}.mov"
        cut_clip(src, o, s, e)
        paths.append(str(o))
    doc = dict(source=dict(path="/x/src.mov"), settings=dict(pad=1.0),
               presence_intervals=[[s + 1, e - 1] for s, e in spans],
               clips=[dict(index=i + 1, start=s, end=e, file=Path(p).name) for i, ((s, e), p) in enumerate(zip(spans, paths))])
    return dict(paths=paths, dir=d, doc=doc)


# ---------------------------------------------------------------- beats

def test_beat_times_follow_a_click_track(tmp_path):
    path = str(tmp_path / "bare.wav")
    y, truth = synth_music.render(120, 40, 0.37, "click")
    synth_music.write_wav(path, y)
    g = beat_times(path)
    assert abs(g.tempo - 120) < 2
    err = [min(abs(b - t) for t in truth) for b in g.times]
    assert max(err) < 0.015, max(err)  # within 15 ms of a click
    assert len(g.times) >= 0.9 * len(truth)


def test_silence_has_no_beat(tmp_path):
    p = tmp_path / "silence.wav"
    synth_music.write_wav(str(p), np.zeros(synth_music.SR * 10, dtype=np.float32))
    with pytest.raises(NoBeatError):
        beat_times(str(p))


def test_plan_puts_every_join_within_half_a_picture_of_a_beat():
    beats = [0.4 + 0.5 * k for k in range(60)]
    p = plan_beat_cuts([140, 133, 121], [30, 30, 30], beats, 30.0)
    assert not p.warnings
    for j, avail in zip(p.joins, [140, 133, 121]):
        assert j["on_beat"] and abs(j["err_ms"]) <= 1000 / 30 / 2 + 1e-6
    for n, avail in zip(p.frames, [140, 133, 121]):
        assert avail - 30 <= n <= avail  # only the padding is trimmed
    assert p.music_start == pytest.approx(0.4)
    assert sum(p.frames) == p.joins[-1]["frame"]


def test_plan_loops_the_beat_grid_when_the_reel_is_longer_than_the_music():
    beats = [0.2 + 0.6 * k for k in range(10)]  # 5.4 s from the first to the last beat
    grid, start, end = extend_beats(beats, 20.0)
    assert start == pytest.approx(0.2) and end == pytest.approx(5.6)
    assert grid[9] == pytest.approx(5.4) and grid[10] == pytest.approx(6.0)  # the first beat of the second pass
    assert grid[-1] > 20.0
    p = plan_beat_cuts([300, 300], [30, 30], beats, 30.0)
    assert all(j["on_beat"] for j in p.joins) and p.loop_end == pytest.approx(5.6)


def test_plan_warns_when_no_beat_fits_the_padding():
    beats = [0.0 + 2.0 * k for k in range(30)]  # a beat only every 2 s
    p = plan_beat_cuts([100, 100], [10, 10], beats, 30.0)
    assert p.warnings and "into its content" in p.warnings[0]
    assert all(j["on_beat"] for j in p.joins)


def test_plan_joins_off_the_beat_when_a_clip_is_shorter_than_a_beat():
    p = plan_beat_cuts([5, 100], [2, 30], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 30.0)
    assert p.frames[0] == 5 and not p.joins[0]["on_beat"] and "off the beat" in p.warnings[0]


def test_tail_slack_is_the_padding_after_presence():
    doc = dict(settings=dict(pad=1.0), presence_intervals=[[10.0, 25.0], [60.0, 100.0]])
    assert tail_slack(doc, dict(start=9.0, end=26.0)) == pytest.approx(1.0)
    assert tail_slack(doc, dict(start=59.0, end=101.2)) == pytest.approx(1.2)
    assert tail_slack(dict(settings=dict(pad=0.7), presence_intervals=[]), dict(start=0, end=5)) == 0.7


# ---------------------------------------------------------------- soundtrack

def test_soundtrack_is_minus_14_lufs_trimmed_and_looped(click, short_click, tmp_path):
    out = tmp_path / "t.m4a"
    r = render_soundtrack(click[0], out, 12.0, start=0.37)
    assert not r["looped"] and abs(r["lufs"] + 14) <= 0.5
    assert abs(measure_lufs(out) + 14) <= 1.0  # measured on the encoded file
    assert abs(float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out)])) - 12.0) < 0.05
    out2 = tmp_path / "l.m4a"
    r2 = render_soundtrack(short_click[0], out2, 20.0, start=0.2, loop_end=8.6)
    assert r2["looped"] and abs(measure_lufs(out2) + 14) <= 1.0
    with pytest.raises(ValueError):
        render_soundtrack(short_click[0], tmp_path / "x.m4a", 5.0, start=50.0)


# ---------------------------------------------------------------- reels

def _decode_errors(path):
    r = subprocess.run(["ffmpeg", "-v", "warning", "-nostdin", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    return [ln for ln in r.stderr.splitlines() if "monotonically" not in ln]


def _dominant_hz(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    y = np.frombuffer(raw, dtype="<f4")
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, 1 / 16000)
    return float(freqs[1:][spec[1:].argmax()]), spec, freqs


def test_beat_synced_reel_joins_land_on_beats(clips, click, tmp_path):
    out = tmp_path / "beat_reel.mov"
    slack = [1.0, 1.0, 1.0]
    r = build_music_reel(clips["paths"], out, click[0], True, slack)
    ck = r["check"]
    assert r["ok"] and ck["ok"], ck["problems"]
    assert ck["streams"] == ["video", "audio"] and ck["audio_codec"] == "aac"
    assert len(ck["joins"]) == 3
    assert ck["max_join_err_frames"] <= 1.0, ck["joins"]
    assert abs(ck["lufs"] + 14) <= 1.0
    # picture: rotation, tag and colour kept; the length is the plan's
    info, p = probe(str(out)), video_params(str(out))
    assert info.rotation == -90 and info.codec == "hevc" and ck["video_tag"] == "hvc1"
    assert (p["pix_fmt"], p["color_range"], p["color_space"], p["color_transfer"], p["color_primaries"]) == ("yuvj420p", "pc", "bt709", "bt709", "bt709")
    plan = r["plan"]
    assert info.nb_frames == sum(plan["frames"]) or abs(info.duration - sum(plan["frames"]) / 30) < 0.05
    # each clip lost only padding
    avail = [shown_pictures(c) for c in clips["paths"]]
    assert all(a - 30 <= n <= a for n, a in zip(plan["frames"], avail))
    # the sound is the music, not the clips' 330 Hz tone
    hz, spec, freqs = _dominant_hz(str(out))
    assert abs(hz - 1250) < 100
    assert spec[(freqs > 300) & (freqs < 360)].max() < 0.3 * spec.max()
    assert _decode_errors(str(out)) == []
    # the beats found in the reel's own sound agree with the joins
    ab = ck["audio_beats"]
    assert ab["median_err_frames"] <= 1.0, ab
    assert abs(ck["audio_start_offset_ms"]) <= 40


def test_beat_synced_reel_loops_short_music(clips, short_click, tmp_path):
    r = build_music_reel(clips["paths"], tmp_path / "loop.mov", short_click[0], True, [1.0, 1.0, 1.0])
    assert r["ok"], r["check"]["problems"]
    assert r["music"]["looped"] and r["check"]["max_join_err_frames"] <= 1.0
    assert abs(r["check"]["lufs"] + 14) <= 1.0


def test_plain_music_reel_stays_lossless_and_drops_the_original_sound(clips, click, tmp_path):
    out = tmp_path / "plain_reel.mov"
    r = build_music_reel(clips["paths"], out, click[0], False, [1.0, 1.0, 1.0])
    assert r["ok"], r["check"]["problems"]
    assert r["music"]["beat_sync"] is False and abs(r["check"]["lufs"] + 14) <= 1.0
    assert r["check"]["streams"] == ["video", "audio"] and r["check"]["video_tag"] == "hvc1"
    assert probe(str(out)).rotation == -90
    hz, _, _ = _dominant_hz(str(out))
    assert abs(hz - 1250) < 100
    # video was copied: same pictures as the lossless reel (a re-encode would change the packet sizes)
    from inframe.reel import build_reel, packets

    lossless = tmp_path / "lossless.mov"
    build_reel(clips["paths"], lossless)
    assert [round(p, 4) for p, _ in packets(str(out))["v"]] == [round(p, 4) for p, _ in packets(str(lossless))["v"]]


def test_the_encoder_is_hevc(tmp_path):
    assert pick_encoder() in ("hevc_videotoolbox", "libx265")


# ---------------------------------------------------------------- export, credits, cli

def _meta(license_code="by"):
    return dict(id="0f0f0f0f-1111-2222-3333-444444444444", title="Test Song", creator="Some Artist", creator_url="https://example.org/artist",
                license=license_code, license_version="4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                landing_url="https://example.org/song", attribution='"Test Song" by Some Artist is licensed under CC BY 4.0.')


def test_export_writes_credits_for_cc_by_only(clips, click, tmp_path):
    out = tmp_path / "o"
    ex = export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), True)
    assert ex["ok"], ex
    text = (out / "src_credits.txt").read_text()
    assert "Test Song" in text and "Some Artist" in text and "CC BY 4.0" in text and "creativecommons.org/licenses/by/4.0" in text
    assert "Openverse" in text and ex["reel"]["music"]["credits"] == "src_credits.txt"
    out2 = tmp_path / "o2"
    ex2 = export_run(clips["doc"], clips["dir"], out2, True, [], (click[0], _meta("cc0")), False)
    assert ex2["ok"] and not (out2 / "src_credits.txt").exists()
    out3 = tmp_path / "o3"
    ex3 = export_run(clips["doc"], clips["dir"], out3, True, [], (click[0], None), False)
    assert ex3["ok"] and not (out3 / "src_credits.txt").exists() and any("no license record" in w for w in ex3["warnings"])


def test_export_reports_music_without_a_beat_instead_of_raising(clips, tmp_path):
    p = tmp_path / "silence.wav"
    synth_music.write_wav(str(p), np.zeros(synth_music.SR * 10, dtype=np.float32))
    ex = export_run(clips["doc"], clips["dir"], tmp_path / "o", True, [], (p, None), True)
    assert not ex["ok"] and ex["reel"]["file"] is None and any("reel failed" in w for w in ex["warnings"])


def test_cli_export_with_music_and_beat_sync(clips, click, tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    for c in clips["paths"]:
        (run / Path(c).name).write_bytes(Path(c).read_bytes())
    (run / "segments.json").write_text(json.dumps(clips["doc"]))
    audio = tmp_path / "song.wav"
    audio.write_bytes(Path(click[0]).read_bytes())
    audio.with_suffix(".json").write_text(json.dumps(_meta("by")))
    assert main(["export", str(run), "--reel", "--music", str(audio), "--beat-sync"]) == 0  # no flag: CC BY is the default
    printed = capsys.readouterr().out
    assert "music only" in printed and "beat sync" in printed and "LUFS" in printed
    assert (run / "src_credits.txt").is_file()
    ex = json.loads((run / "export.json").read_text())
    assert ex["reel"]["ok"] and ex["reel"]["music"]["file"] == "song.wav" and ex["reel"]["music"]["license"] == "by"


def test_cli_rejects_music_without_reel_and_beat_sync_without_music(clips, tmp_path, capsys):
    seg = tmp_path / "segments.json"
    seg.write_text(json.dumps(clips["doc"]))
    assert main(["export", str(seg), "--edit-list", "otio", "--music", "x.mp3"]) == 2
    assert "--music needs --reel" in capsys.readouterr().err
    assert main(["export", str(seg), "--reel", "--beat-sync"]) == 2
    assert "--beat-sync needs --music" in capsys.readouterr().err
    assert main(["export", str(seg), "--reel", "--music", "nothing-like-an-id"]) == 2


# ---------------------------------------------------------------- Openverse (stubbed)

def _result(**kw):
    r = dict(id="0f0f0f0f-1111-2222-3333-444444444444", title="Song", creator="Artist", creator_url="https://example.org/a", license="by",
             license_version="4.0", license_url="https://creativecommons.org/licenses/by/4.0/", url="https://example.org/s.mp3",
             foreign_landing_url="https://example.org/s", duration=95000, attribution="credit line", source="jamendo", filetype="mp3")
    r.update(kw)
    return r


def test_cc0_and_cc_by_tracks_are_kept_by_default_and_cc_by_not_with_cc0_only():
    assert music_mod.parse_track(_result()).license == "by"  # CC BY is the default (decided 2026-10-16)
    assert music_mod.parse_track(_result(), cc0_only=True) is None
    assert music_mod.parse_track(_result(license="cc0", license_url=None)).license == "cc0"
    assert music_mod.parse_track(_result(license="cc0", license_url=None), cc0_only=True).license == "cc0"
    assert music_mod.licenses_allowed() == ("cc0", "by") and music_mod.licenses_allowed(cc0_only=True) == ("cc0",)
    for only in (False, True):  # BY-SA, BY-ND and every NC license stay refused either way
        for lic in ("by-sa", "by-nc", "by-nd", "by-nc-sa", "by-nc-nd", "pdm", ""):
            assert music_mod.parse_track(_result(license=lic), only) is None
    assert music_mod.parse_track(_result(license_url=None)) is None  # CC BY without a license url
    assert music_mod.parse_track(_result(id="../../etc/passwd")) is None


def test_search_filters_and_lists(monkeypatch):
    seen = []

    def fake(url, binary=False, limit=None):
        seen.append(url)
        return dict(page_count=1, results=[_result(), _result(id="aaaaaaaa-1", license="by-sa"), _result(id="bbbbbbbb-2", duration=20000),
                                           _result(id="cccccccc-3", license="cc0", license_url=None, title="Free")])

    monkeypatch.setattr(music_mod, "_get", fake)
    got = music_mod.search("calm", min_seconds=60, limit=5)
    assert sorted(t.title for t in got) == ["Free", "Song"]  # CC0 and CC BY are in; by-sa and the 20 s track are out
    assert "q=calm" in seen[0] and "source=jamendo" in seen[0] and "category=music" in seen[0] and "license=cc0" in seen[0]
    assert "license=cc0" in seen[0] and any("license=by&" in u for u in seen)  # CC0 is asked for first, then CC BY
    assert {t.license for t in music_mod.search(None, 0, 10)} == {"by", "cc0"}
    seen.clear()
    got = music_mod.search("calm", min_seconds=60, limit=5, cc0_only=True)
    assert [t.title for t in got] == ["Free"]  # --cc0-only: CC BY, by-sa and the 20 s track are out
    assert all("license=by" not in u.replace("license=by-", "") for u in seen)  # CC BY is not even asked for
    assert {t.license for t in music_mod.search(None, 0, 10, cc0_only=True)} == {"cc0"}


def test_get_downloads_into_the_cache_with_license_json(monkeypatch, tmp_path, capsys):
    wav = tmp_path / "s.wav"
    y, _ = synth_music.render(120, 3, 0.1, "click")
    synth_music.write_wav(str(wav), y)
    calls = []

    def fake(url, binary=False, limit=None):
        calls.append(url)
        return wav.read_bytes() if binary else _result(filetype="wav")

    monkeypatch.setattr(music_mod, "_get", fake)
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    assert music_mod.cache_dir() == tmp_path / "cache" / "music"
    assert main(["music", "get", "0f0f0f0f-1111-2222-3333-444444444444", "--cc0-only"]) == 1  # --cc0-only refuses a CC BY track
    assert "--cc0-only" in capsys.readouterr().err and not calls[1:]  # refused before the file was fetched
    assert not list((tmp_path / "cache" / "music").glob("*"))
    assert main(["music", "get", "0f0f0f0f-1111-2222-3333-444444444444"]) == 0  # CC BY is the default
    printed = capsys.readouterr().out
    audio = tmp_path / "cache" / "music" / "0f0f0f0f-1111-2222-3333-444444444444.wav"
    assert audio.is_file() and str(audio) in printed
    meta = json.loads(audio.with_suffix(".json").read_text())
    assert meta["license"] == "by" and meta["attribution"] == "credit line" and "Openverse" in meta["openverse_notice"]
    n = len(calls)
    assert music_mod.get("0f0f0f0f-1111-2222-3333-444444444444") == audio and len(calls) == n  # cached: no second download
    with pytest.raises(music_mod.MusicError, match="--cc0-only"):  # ... and a cached CC BY track is not handed out with --cc0-only
        music_mod.get("0f0f0f0f-1111-2222-3333-444444444444", cc0_only=True)
    with pytest.raises(music_mod.MusicError, match="--cc0-only"):
        music_mod.resolve("0f0f0f0f-1111-2222-3333-444444444444", cc0_only=True)
    p, m = music_mod.resolve("0f0f0f0f-1111-2222-3333-444444444444")
    assert p == audio and m["creator"] == "Artist"
    with pytest.raises(music_mod.MusicError, match="--cc0-only"):  # the cached file's path is no way round --cc0-only
        music_mod.resolve(str(audio), cc0_only=True)
    assert music_mod.resolve(str(audio))[0] == audio


def test_jamendo_mp32_files_are_saved_as_mp3(monkeypatch, tmp_path):
    wav = tmp_path / "s.wav"
    y, _ = synth_music.render(120, 3, 0.1, "click")
    synth_music.write_wav(str(wav), y)
    monkeypatch.setattr(music_mod, "_get", lambda url, binary=False, limit=None: wav.read_bytes() if binary else _result(filetype="mp32"))
    assert music_mod.get("0f0f0f0f-1111-2222-3333-444444444444", tmp_path / "d").suffix == ".mp3"


def test_get_refuses_other_licenses_and_bad_ids_and_non_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(music_mod, "_get", lambda url, binary=False, limit=None: _result(license="by-nc"))
    with pytest.raises(music_mod.MusicError, match="not CC0 or CC BY"):
        music_mod.get("0f0f0f0f-1111-2222-3333-444444444444")
    with pytest.raises(music_mod.MusicError, match="not a track id"):
        music_mod.get("../evil")
    monkeypatch.setattr(music_mod, "_get", lambda url, binary=False, limit=None: b"<html>not audio</html>" if binary else _result(license="cc0"))
    with pytest.raises(music_mod.MusicError, match="not a playable audio file"):
        music_mod.get("0f0f0f0f-1111-2222-3333-444444444444")
    assert not list((tmp_path / "cache" / "music").glob("*"))  # nothing left behind
    with pytest.raises(music_mod.MusicError):
        music_mod.resolve("no-such-file.mp3")


def test_refuses_plain_http():
    with pytest.raises(music_mod.MusicError, match="non-https"):
        music_mod._get("http://example.org/x.mp3")


# ---------------------------------------------------------------- review fixes

def _tracker_vs_truth(path, truth):
    g = beat_times(path)
    return g, float(np.median([min(abs(b - t) for t in truth) for b in g.times]))


@pytest.mark.parametrize("bpm", [100, 124, 128])
def test_beats_land_on_the_kicks_not_the_off_beat_hats(tmp_path, bpm):
    path = str(tmp_path / "kickhat.wav")
    y, kicks = synth_music.render(bpm, 40, 0.8, "beat")
    synth_music.write_wav(path, y)
    g, med_kick = _tracker_vs_truth(path, kicks)
    off = kicks + 30.0 / bpm  # the hats
    med_hat = float(np.median([min(abs(b - t) for t in off) for b in g.times]))
    assert med_kick < 0.03 and med_kick < med_hat, (med_kick, med_hat)


def test_beat_synced_joins_are_on_the_true_beats_of_a_kick_and_hat_track(clips, tmp_path):
    path = str(tmp_path / "kickhat.wav")
    y, kicks = synth_music.render(124, 60, 0.8, "beat")
    synth_music.write_wav(path, y)
    r = build_music_reel(clips["paths"], tmp_path / "kh.mov", path, True, [1.0, 1.0, 1.0])
    # reel time 0 is the music's first beat; true beats in the reel are the kicks from that first beat on
    start = r["plan"]["music_start"]
    for j in r["check"]["joins"]:
        t = j["reel_s"] + start
        assert min(abs(t - k) for k in kicks) < 0.045, (j, start)


def _plain_src(tmp_path_factory):
    d = tmp_path_factory.mktemp("plainsrc")
    s = d / "src.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=24", "-f", "lavfi", "-i", "sine=frequency=330:duration=24",
        "-c:v", "libx265", "-x265-params", "open-gop=1:keyint=30:min-keyint=30:bframes=4:scenecut=0:log-level=error",
        "-tag:v", "hvc1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(s))
    paths = []
    for i, (a, b) in enumerate([(1.0, 5.3), (8.0, 12.9), (15.2, 19.4)]):
        o = d / f"c{i}.mov"
        cut_clip(str(s), o, a, b)
        paths.append(str(o))
    return paths


def test_plain_music_reel_of_clips_with_their_own_audio_lead_is_ok(tmp_path_factory, click, tmp_path):
    paths = _plain_src(tmp_path_factory)
    from inframe.reel import build_reel

    lossless = build_reel(paths, tmp_path / "l.mov")
    # these clips lead their audio by 22 ms on their own (past the 21 ms limit): that is the cut's, not the join's, so the join check passes
    assert lossless["ok"] and max(j["clip_av_offset_ms"] for j in lossless["check"]["joins"]) > 21, lossless["check"]
    r = build_music_reel(paths, tmp_path / "m.mov", click[0], False, [1.0] * 3)
    assert r["ok"] and r["check"]["ok"], r["check"]["problems"]
    assert abs(r["check"]["lufs"] + 14) <= 1.0


def test_a_real_video_join_problem_still_fails_the_plain_music_reel(clips, click, tmp_path, monkeypatch):
    from inframe import musicreel
    from inframe.reel import build_reel as real

    def broken(c, o):
        r = real(c, o)
        r["check"]["problems"].append("clip 2: a hole of 0.500 s in the pictures at the join")
        r["ok"] = False
        return r

    monkeypatch.setattr(musicreel, "build_reel", broken)
    r = build_music_reel(clips["paths"], tmp_path / "m.mov", click[0], False, [1.0] * 3)
    assert not r["ok"] and any("hole" in p for p in r["check"]["problems"])


def test_export_removes_a_stale_credit_and_keeps_one_per_video(clips, click, tmp_path):
    out = tmp_path / "o"
    ex = export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), False)
    assert ex["ok"] and (out / "src_credits.txt").is_file()
    ex2 = export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("cc0")), False)
    assert ex2["ok"] and ex2["reel"]["music"]["credits"] is None and not (out / "src_credits.txt").exists()
    export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), False)
    ex3 = export_run(clips["doc"], clips["dir"], out, True, [], None, False)
    assert ex3["ok"] and not (out / "src_credits.txt").exists()
    # another video exported into the same folder keeps its own credit
    other = dict(clips["doc"], source=dict(path="/x/other.mov"))
    export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), False)
    export_run(other, clips["dir"], out, True, [], (click[0], _meta("by")), False)
    assert (out / "src_credits.txt").is_file() and (out / "other_credits.txt").is_file()


# ---------------------------------------------------------------- lows: CC0 default, credit text, _get errors

def test_resolve_refuses_a_cached_cc_by_file_path_with_cc0_only(tmp_path):
    audio = tmp_path / "0f0f0f0f-1111-2222-3333-444444444444.mp3"
    audio.write_bytes(b"x")
    audio.with_suffix(".json").write_text(json.dumps(_meta("by")))
    with pytest.raises(music_mod.MusicError, match="CC0"):
        music_mod.resolve(str(audio), cc0_only=True)
    assert music_mod.resolve(str(audio))[0] == audio
    audio.with_suffix(".json").write_text(json.dumps(_meta("cc0")))
    assert music_mod.resolve(str(audio))[1]["license"] == "cc0"


def test_a_music_file_with_any_license_but_cc0_or_cc_by_is_refused(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    for lic in ("by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd", "pdm", "all-rights-reserved"):
        audio.with_suffix(".json").write_text(json.dumps(_meta(lic)))
        for only in (False, True):
            with pytest.raises(music_mod.LicenseError, match="CC0 and CC BY"):
                music_mod.resolve(str(audio), cc0_only=only)
    for lic in ("cc0", "by", "BY", " cc0 "):  # the two licenses inframe uses stay usable
        audio.with_suffix(".json").write_text(json.dumps(_meta(lic)))
        assert music_mod.resolve(str(audio))[0] == audio
    audio.with_suffix(".json").unlink()
    assert music_mod.resolve(str(audio)) == (audio, None)  # no record: used as before, export warns
    audio.with_suffix(".json").write_text(json.dumps(dict(_meta("by"), license="")))
    assert music_mod.resolve(str(audio))[0] == audio  # an empty license field: used as before, export warns


def test_the_credits_file_names_the_changes_the_source_and_the_artist(clips, click, tmp_path):
    out = tmp_path / "o"
    ex = export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), False)
    assert ex["ok"], ex
    text = (out / "src_credits.txt").read_text()
    assert music_mod.MODIFICATION_NOTICE in text
    assert "Source: https://example.org/song" in text and "Artist: https://example.org/artist" in text
    assert 'is licensed under CC BY 4.0.' in text and music_mod.OPENVERSE_NOTICE in text
    assert music_mod.MODIFICATION_NOTICE in ex["reel"]["music"]["credit_text"] and "Source: https://example.org/song" in ex["reel"]["music"]["credit_text"]


def test_credit_states_the_changes_and_is_in_the_reel_file(clips, click, tmp_path):
    assert "changes were made" in music_mod.credits_text(_meta("by")).lower()
    assert "changes were made" not in music_mod.credits_text(_meta("cc0")).lower()
    out = tmp_path / "o"
    ex = export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), False)
    assert ex["ok"], ex
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "default=nw=1:nk=1", str(out / ex["reel"]["file"])],
                       capture_output=True, text=True)
    assert "Test Song" in r.stdout and "CC BY 4.0" in r.stdout
    out2 = tmp_path / "o2"
    ex2 = export_run(clips["doc"], clips["dir"], out2, True, [], (click[0], _meta("cc0")), False)
    r2 = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "default=nw=1:nk=1", str(out2 / ex2["reel"]["file"])],
                        capture_output=True, text=True)
    assert r2.stdout.strip() == ""


def test_get_turns_bad_json_into_music_errors(monkeypatch):
    for body in (b"<html>not json</html>", b"\xff\xfe"):
        monkeypatch.setattr(music_mod.net, "fetch", lambda url, kind="api", max_bytes=None, body=body: body)
        with pytest.raises(music_mod.MusicError, match="JSON"):
            music_mod._get("https://example.org/x")
    monkeypatch.setattr(music_mod.net, "fetch", lambda url, kind="api", max_bytes=None: b"abc")
    assert music_mod._get("https://example.org/x", binary=True, limit=100) == b"abc"


def test_get_refuses_a_redirect_to_http(monkeypatch):
    def refuse(url, kind="api", max_bytes=None):
        raise music_mod.net.NetError("refusing a redirect to a non-https address: http://example.org/y")
    monkeypatch.setattr(music_mod.net, "fetch", refuse)
    with pytest.raises(music_mod.MusicError, match="non-https"):
        music_mod._get("https://example.org/x")


# ---------------------------------------------------------------- 2026-10-16: CC0 and CC BY by default, credit written automatically

def test_license_is_read_from_the_openverse_record_and_a_cached_record_alone_is_no_proof(monkeypatch, tmp_path):
    """A track that is not cached is licensed by the Openverse record fetched for it (never by a search row), and a cached record
    that names a license inframe does not use is refused."""
    tid = "0f0f0f0f-1111-2222-3333-444444444444"
    seen = []

    def fake(url, binary=False, limit=None):
        seen.append(url)
        return _result(license="by-sa")

    monkeypatch.setattr(music_mod, "_get", fake)
    for only in (False, True):
        with pytest.raises(music_mod.LicenseError, match="not CC0 or CC BY"):
            music_mod.get(tid, tmp_path / "d", cc0_only=only)
    assert seen and all(u.endswith(f"/{tid}/") for u in seen)  # the record of this track, not a binary download
    d = tmp_path / "cache"
    d.mkdir()
    (d / f"{tid}.mp3").write_bytes(b"x")
    (d / f"{tid}.json").write_text(json.dumps(_meta("by-nc")))
    with pytest.raises(music_mod.LicenseError, match="CC0 and CC BY"):
        music_mod.get(tid, d)
    (d / f"{tid}.json").write_text(json.dumps(_meta("by")))
    assert music_mod.get(tid, d) == d / f"{tid}.mp3"


def test_cli_cc0_only_flag_and_allow_by_no_op(monkeypatch, capsys):
    seen = []

    def fake(url, binary=False, limit=None):
        seen.append(url)
        return dict(page_count=1, results=[_result(), _result(id="cccccccc-3", license="cc0", license_url=None, title="Free")])

    monkeypatch.setattr(music_mod, "_get", fake)
    assert main(["music", "search"]) == 0
    out = capsys.readouterr().out
    assert "BY 4.0" in out and "Free" in out and "2 tracks (CC0 and CC BY)" in out
    assert main(["music", "search", "--allow-by"]) == 0  # still accepted: the same as no flag
    assert capsys.readouterr().out == out
    assert main(["music", "search", "--cc0-only"]) == 0
    out2 = capsys.readouterr().out
    assert "Free" in out2 and "BY 4.0" not in out2 and "CC0 only" in out2
    helps = [subprocess.run([sys.executable, "-m", "inframe.cli", *a, "--help"], capture_output=True, text=True).stdout for a in (["music", "search"], ["music", "get"], ["export"])]
    for h in helps:
        assert "--cc0-only" in h and "--allow-by" in h and "default" in h


def test_cli_export_with_a_cc_by_file_writes_credits_without_a_flag_and_cc0_only_refuses_it(clips, click, tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    for c in clips["paths"]:
        (run / Path(c).name).write_bytes(Path(c).read_bytes())
    (run / "segments.json").write_text(json.dumps(clips["doc"]))
    audio = tmp_path / "song.wav"
    audio.write_bytes(Path(click[0]).read_bytes())
    audio.with_suffix(".json").write_text(json.dumps(_meta("by")))
    assert main(["export", str(run), "--reel", "--music", str(audio), "--cc0-only"]) == 2  # refused before anything was built
    assert "CC0" in capsys.readouterr().err and not (run / "src_credits.txt").exists()
    assert main(["export", str(run), "--reel", "--music", str(audio)]) == 0
    printed = capsys.readouterr().out
    assert "credits in" in printed and (run / "src_credits.txt").is_file()
    ex = json.loads((run / "export.json").read_text())
    assert ex["reel"]["music"]["credits"] == "src_credits.txt"
    assert "Test Song" in ex["reel"]["music"]["credit_text"] and "CC BY 4.0" in ex["reel"]["music"]["credit_text"]


def test_export_credit_text_is_in_the_result_only_when_a_credit_is_needed(clips, click, tmp_path):
    ex = export_run(clips["doc"], clips["dir"], tmp_path / "o", True, [], (click[0], _meta("by")), False)
    assert ex["reel"]["music"]["credit_text"].startswith('Music: "Test Song" by Some Artist') and ex["reel"]["music"]["credits"] == "src_credits.txt"
    ex2 = export_run(clips["doc"], clips["dir"], tmp_path / "o2", True, [], (click[0], _meta("cc0")), False)
    assert ex2["reel"]["music"]["credit_text"] is None and ex2["reel"]["music"]["credits"] is None


# ---------------------------------------------------------------- review fixes: CC BY record completeness, failed re-export

def test_a_cc_by_record_without_licence_url_title_or_creator_is_refused_on_file_and_cache_paths(monkeypatch, tmp_path):
    tid = "0f0f0f0f-1111-2222-3333-444444444444"
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    d = tmp_path / "cache"
    d.mkdir()
    (d / f"{tid}.mp3").write_bytes(b"x")
    monkeypatch.setattr(music_mod, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    for drop in ("license_url", "title", "creator"):
        rec = {k: v for k, v in _meta("by").items() if k != drop}
        audio.with_suffix(".json").write_text(json.dumps(rec))
        with pytest.raises(music_mod.LicenseError, match=drop):
            music_mod.resolve(str(audio))
        (d / f"{tid}.json").write_text(json.dumps(rec))
        with pytest.raises(music_mod.LicenseError, match=drop):
            music_mod.get(tid, d)
    audio.with_suffix(".json").write_text(json.dumps({"license": "by"}))
    with pytest.raises(music_mod.LicenseError):
        music_mod.check_file_license({"license": "by"}, "song.mp3")
    audio.with_suffix(".json").write_text(json.dumps(_meta("by")))  # a complete record still passes
    assert music_mod.resolve(str(audio))[0] == audio
    (d / f"{tid}.json").write_text(json.dumps(_meta("by")))
    assert music_mod.get(tid, d) == d / f"{tid}.mp3"


def test_a_failed_re_export_keeps_the_earlier_reel_and_its_credits(clips, click, tmp_path):
    out = tmp_path / "o"
    ex = export_run(clips["doc"], clips["dir"], out, True, [], (click[0], _meta("by")), False)
    assert ex["ok"] and (out / "src_credits.txt").is_file() and (out / "src_reel.mov").is_file()
    before = (out / "src_reel.mov").read_bytes()
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio at all")
    ex2 = export_run(clips["doc"], clips["dir"], out, True, [], (str(broken), _meta("cc0")), False)
    assert not ex2["ok"] and any("reel failed" in w for w in ex2["warnings"])
    assert (out / "src_reel.mov").read_bytes() == before  # the earlier reel survived the failed build ...
    assert (out / "src_credits.txt").is_file()  # ... so its credit stays beside it


# ---------------------------------------------------------------- a source container that cannot hold H.264

def test_music_reel_of_a_vp9_webm_source_goes_into_an_mp4(click, tmp_path):
    """A .webm (or .ogv) source: the music reel is H.264/AAC, which that container cannot hold, so it is named <stem>_reel.mp4 and
    the old-export clearing, the credits and the served-name patterns all know it by that name."""
    from inframe.export import clear_old_exports, music_reel_name, reel_name
    from inframe.serve_export import name_patterns

    d = tmp_path / "clips"
    d.mkdir()
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=12", "-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime",
        "-cpu-used", "8", "-g", "30", str(tmp_path / "dog_walk.webm"))
    spans = [(1.0, 5.3), (6.0, 10.9)]
    for i, (s, e) in enumerate(spans):
        cut_clip(str(tmp_path / "dog_walk.webm"), d / f"dog_walk_{i + 1:02d}.webm", s, e)
    doc = dict(source=dict(path="/x/dog_walk.webm"), settings=dict(pad=1.0), presence_intervals=[[s + 1, e - 1] for s, e in spans],
               clips=[dict(index=i + 1, start=s, end=e, file=f"dog_walk_{i + 1:02d}.webm") for i, (s, e) in enumerate(spans)])
    out = tmp_path / "out"
    ex = export_run(doc, d, out, True, [], (click[0], _meta("by")), True)
    assert ex["reel"]["file"] == "dog_walk_reel.mp4", ex["warnings"]
    assert ex["ok"], ex["warnings"]
    assert (out / "dog_walk_reel.mp4").is_file() and not (out / "dog_walk_reel.webm").exists()
    assert [music_reel_name("v", x) for x in (".mov", ".MP4", ".m4v", ".ogv", ".ogg", ".webm", ".mkv", "")] == [
        "v_reel.mov", "v_reel.MP4", "v_reel.m4v", "v_reel.mp4", "v_reel.mp4", "v_reel.mp4", "v_reel.mp4", "v_reel.mp4"]
    assert reel_name("dog_walk", ".webm") == "dog_walk_reel.webm"  # the plain stream-copy reel keeps the source's container
    assert name_patterns("dog_walk")["reel"].match("dog_walk_reel.mp4")
    assert (out / "dog_walk_credits.txt").is_file()
    clear_old_exports(out, "dog_walk")
    assert not (out / "dog_walk_reel.mp4").exists() and not (out / "dog_walk_credits.txt").exists()


def test_music_reel_without_beat_sync_of_vp9_webm_clips(click, tmp_path):
    """No beat sync: the lossless reel of the VP9 clips is an intermediate file whose container must hold VP9 (not a .mov)."""
    src = tmp_path / "v.webm"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8", "-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime",
        "-cpu-used", "8", "-g", "15", str(src))
    clips = []
    for i, (s, e) in enumerate([(0.5, 3.0), (4.0, 7.0)]):
        o = tmp_path / f"c{i}.webm"
        cut_clip(str(src), o, s, e)
        clips.append(str(o))
    r = build_music_reel(clips, tmp_path / "v_reel.mp4", click[0], False, [1.0, 1.0])
    assert r["ok"], r["check"]["problems"]
    assert (tmp_path / "v_reel.mp4").is_file()


def test_music_reel_without_beat_sync_of_vp9_mp4_clips(click, tmp_path):
    """No beat sync, VP9+AAC clips in an .mp4 (an Android screen recording): the lossless join keeps the .mp4 suffix, so the
    vp09 tag survives to the check (an .mkv join lost it and the check failed with a '[0][0][0][0]' tag)."""
    src = tmp_path / "v.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8", "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime", "-cpu-used", "8", "-g", "15", "-c:a", "aac", "-shortest", str(src))
    clips = []
    for i, (s, e) in enumerate([(0.5, 3.0), (4.0, 7.0)]):
        o = tmp_path / f"c{i}.mp4"
        cut_clip(str(src), o, s, e)
        clips.append(str(o))
    r = build_music_reel(clips, tmp_path / "m_reel.mp4", click[0], False, [1.0, 1.0])
    assert r["ok"] and not r["fallback_used"], (r["check"]["problems"], r["fallback_used"])
    assert (tmp_path / "m_reel.mp4").is_file()
