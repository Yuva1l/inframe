"""Edge cases, part 6: each test failed before its fix."""
import subprocess

import pytest


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


def test_cut_intermediate_is_not_called_a_video(tmp_path, monkeypatch):
    from inframe import cut
    from inframe.probe import ProbeError

    def junk(cmd, what):
        open(cmd[-1], "wb").write(b"junk" * 100)

    monkeypatch.setattr(cut, "_run", junk)
    with pytest.raises(ProbeError) as e:
        cut.cut_clip("src.mp4", tmp_path / "c.mp4", 0.0, 1.0)
    assert "as a video" not in str(e.value) and "as a cut piece" in str(e.value)


def test_container_start_and_audio_frame_name_the_file(tmp_path):
    from inframe import probe

    bad = tmp_path / "x.part.mp4"
    bad.write_bytes(b"junk" * 100)
    for fn in (probe.container_start_time, probe.audio_frame_seconds):
        with pytest.raises(probe.ProbeError, match="as a video"):
            fn(str(bad))
        with pytest.raises(probe.ProbeError, match="as a cut piece"):
            fn(str(bad), what="a cut piece")


def test_nano_only_log_line_has_no_instead_of():
    from inframe.pipeline import model_note

    only_nano = model_note(16, "nano", "nano", "only the nano model is installed")
    assert "instead of" not in only_nano and "RF-DETR nano" in only_nano and "only the nano model" in only_nano
    assert "RF-DETR small instead of nano" in model_note(16, "small", "nano", "class targets do not run on nano")


def test_music_reel_precheck_looks_at_every_clip(tmp_path):
    from inframe.musicreel import build_music_reel

    _ff("-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "a.mp4"))
    try:
        _ff("-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1", "-c:v", "libvpx", str(tmp_path / "b.webm"))
    except subprocess.CalledProcessError:
        pytest.skip("no VP8 encoder")
    with pytest.raises(ValueError, match="VP8.*beat sync"):
        build_music_reel([str(tmp_path / "a.mp4"), str(tmp_path / "b.webm")], tmp_path / "r.mp4", str(tmp_path / "m.wav"), False, [0.0, 0.0])
