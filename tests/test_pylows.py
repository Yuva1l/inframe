"""Python edge cases, part 1: each test failed before its fix."""
import json
import subprocess

import pytest


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-nostats", *args], check=True)


def _has_encoder(name):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return name in r.stdout


def _clip(path, vcodec, acodec, extra=()):
    _ff("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", vcodec, "-g", "15", *extra, "-c:a", acodec, "-shortest", str(path))
    return str(path)


def _vorbis():
    """(encoder, extra args) of whatever Vorbis encoder this ffmpeg has: libvorbis, or the native one (experimental, stereo only)."""
    if _has_encoder(" libvorbis "):
        return "libvorbis", ()
    if _has_encoder(" vorbis "):
        return "vorbis", ("-strict", "-2", "-ac", "2")
    return None


def test_av1_clips_make_a_reel(tmp_path):
    """work_format read the decoder name ('libdav1d'), so AV1 with AAC got a .mov copy and strip_outside raised UnknownCodecError."""
    if not _has_encoder("libsvtav1"):
        pytest.skip("no AV1 encoder")
    from inframe.reel import build_reel, work_format

    a = _clip(tmp_path / "a.mp4", "libsvtav1", "aac")
    assert work_format(a) == "mp4"
    r = build_reel([a, a], tmp_path / "reel.mp4")
    assert r["ok"], r["check"]["problems"]


@pytest.mark.parametrize("acodec,extra", [("libopus", ()), ("flac", ("-strict", "-2")), ("aac", ())])
def test_h264_with_b_frames_in_mkv_makes_a_reel_that_keeps_every_picture(tmp_path, acodec, extra):
    """Matroska gives no decode time to the first packets of a B-frame stream; strip_outside dropped them (the keyframe too)
    and the reel came out with fewer pictures than the clips show (ok=False)."""
    from inframe.reel import build_reel, shown_pictures

    a = _clip(tmp_path / "a.mkv", "libx264", acodec, extra)
    r = build_reel([a, a], tmp_path / "reel.mkv")
    assert r["ok"], r["check"]["problems"]
    assert shown_pictures(str(tmp_path / "reel.mkv")) == 2 * shown_pictures(a)


def test_vorbis_in_mkv_makes_a_reel_without_a_hole_at_the_join(tmp_path):
    """A Matroska file with Vorbis audio is longer than its last picture (the last audio block is whole); the concat demuxer placed
    the next clip by that length and the check found a hole at the join (ok=False)."""
    from inframe.reel import build_reel, shown_pictures

    enc = _vorbis()
    if enc is None:
        pytest.skip("no Vorbis encoder")
    a = _clip(tmp_path / "a.mkv", "libx264", *enc)
    r = build_reel([a, a], tmp_path / "reel.mkv")
    assert r["ok"], r["check"]["problems"]
    assert shown_pictures(str(tmp_path / "reel.mkv")) == 2 * shown_pictures(a)


def test_check_reel_compares_the_decode_times_a_matroska_reel_has(tmp_path, monkeypatch):
    """check_reel skipped the video decode-time check for every Matroska reel; it must skip only the packets with no decode time."""
    from inframe import reel as reelmod

    a = _clip(tmp_path / "a.mkv", "libx264", "libopus")
    real = reelmod.packets(a)
    assert real["matroska"]
    assert reelmod.check_reel(a, [a]).ok  # the file's own order passes: the stand-in times of the first packets (B-frames) are not compared

    def with_dts(i, value):
        v = list(real["v"])
        v[i] = (v[i][0], value)
        return dict(real, v=v)

    for bad_at, bad_dts in ((10, real["v"][9][1]), (3, real["v"][2][1] - 0.01)):  # a repeated time, a time that goes back
        monkeypatch.setattr(reelmod, "packets", lambda path, p=with_dts(bad_at, bad_dts): p)
        c = reelmod.check_reel(a, [a])
        assert not c.ok and any(f"video decode time does not increase at packet {bad_at}" in p for p in c.problems), c.problems


def test_music_reel_without_beat_sync_of_vp8_clips_says_so_before_any_work(tmp_path):
    """The final .mp4 cannot hold VP8: the run used to go through the join and fail in the mux with an ffmpeg message."""
    if not _has_encoder("libvpx "):
        pytest.skip("no VP8 encoder")
    from inframe.musicreel import build_music_reel

    a = _clip(tmp_path / "a.webm", "libvpx", "libopus")
    with pytest.raises(ValueError, match="VP8.*beat sync"):
        build_music_reel([a], tmp_path / "v_reel.mp4", str(tmp_path / "no_music.wav"), False, [0.0])
    assert [p.name for p in tmp_path.iterdir()] == ["a.webm"]  # no work folder left


def test_video_duration_without_any_duration_is_a_clear_error(monkeypatch):
    from inframe import musicreel

    monkeypatch.setattr(musicreel.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps({"streams": [{}]}), stderr=""))
    with pytest.raises(RuntimeError, match="neither a duration nor a DURATION tag"):
        musicreel.video_duration("x.mkv")


def test_video_duration_still_reads_the_matroska_tag(monkeypatch):
    from inframe import musicreel

    out = json.dumps({"streams": [{"tags": {"DURATION": "00:00:02.500000000"}}]})
    monkeypatch.setattr(musicreel.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    assert musicreel.video_duration("x.mkv") == 2.5


def test_a_part_file_of_a_killed_run_is_cleared_with_the_old_sidecars(tmp_path):
    from inframe.sidecar import clear_old_sidecars

    for n in ("v_01_0005-0008.inframe.json", "v_01_0005-0008.inframe.json.part", "v_02_0010-0020.inframe.json.part",
              "other_01_0005-0008.inframe.json.part", "v_01_0005-0008.mov"):
        (tmp_path / n).write_text("x")
    clear_old_sidecars(tmp_path, "v")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other_01_0005-0008.inframe.json.part", "v_01_0005-0008.mov"]
