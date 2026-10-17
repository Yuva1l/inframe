"""The sidecar describes the file that was written, also when ffmpeg's mov demuxer starts a stream copy one GOP before the keyframe.

An iPhone HEVC file is open-GOP, its video track has a 'sync' sample group, and its DTS is shifted by the negative composition
offsets (`+negative_cts_offsets`). ffmpeg will not seek to a keyframe for a request in [keyframe, keyframe + 2 frames) in such a file
and goes back one GOP. The synthetic file here reproduces that with keyframes every 28 frames (0.9333 s): a sidecar that says the copy
starts at the keyframe while the file lasts one GOP longer would put every clip_time 0.933 s too small."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from inframe import pipeline
from inframe.cut import cut_clip
from inframe.detect import Detection
from inframe.pipeline import Settings, run_cut
from inframe.probe import keyframe_times, seek_landing
import mov_sync_group

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or "libx265" not in subprocess.run(
    ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout, reason="needs ffmpeg with libx265")

GOP = 28 / 30


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-nostats", "-loglevel", "error", "-nostdin", *args], check=True)


@pytest.fixture(scope="module")
def gop28(tmp_path_factory):
    """8 s, 30 fps, 320x180 HEVC, open GOP, a CRA keyframe every 28 frames, B-frames, negative cts offsets, sync sample group, AAC audio."""
    d = tmp_path_factory.mktemp("gop28")
    plain, out = d / "plain.mov", d / "gop28.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8", "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-c:v", "libx265", "-x265-params", "keyint=28:min-keyint=28:open-gop=1:bframes=3:scenecut=0:log-level=error",
        "-tag:v", "hvc1", "-pix_fmt", "yuv420p", "-movflags", "+negative_cts_offsets", "-c:a", "aac", "-ar", "44100", "-shortest", str(plain))
    mov_sync_group.main(str(plain), str(out))
    return str(out)


def _video_packets(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,size,flags", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["packets"]


def _video_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["streams"][0]["duration"])


class Present:
    """A person in the samples whose time is inside [a, b]."""
    backend, path = "fake", "fake.mlpackage"

    def __init__(self, a, b, fps):
        self.a, self.b, self.fps, self.n = a, b, fps, 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if self.a <= self.n / self.fps <= self.b else [])
            self.n += 1
        return out


@pytest.mark.parametrize("first_seen, start, want_copy_start", [(2.3, 1.9, GOP), (2.4, 2.0, 2 * GOP)])
def test_sidecar_matches_the_written_file(gop28, tmp_path, monkeypatch, first_seen, start, want_copy_start):
    # 1.9 is 0.033 s after the keyframe at 1.867: inside the window, the copy starts at 0.933. 2.0 is outside it: the copy starts at 1.867.
    assert any(abs(k - 2 * GOP) < 1e-3 for k in keyframe_times(gop28))
    monkeypatch.setattr(pipeline, "load_detector", lambda size: Present(first_seen, 6.0, 10.0))
    out = tmp_path / "o"
    doc = run_cut(gop28, str(out), Settings(pad=0.35, min_len=0.5, fps=10.0), None)
    c = doc["clips"][0]
    assert c["start"] == pytest.approx(start, abs=1e-6), c
    sc = json.loads((out / c["sidecar"]).read_text())
    clip = sc["clip"]
    assert clip["copy_start_s"] == pytest.approx(want_copy_start, abs=1e-3)
    # truth from the file: its first video packets are the source's packets from the keyframe at copy_start_s, in decode order
    src = _video_packets(gop28)
    cl = _video_packets(str(out / c["file"]))
    i0 = next(i for i, p in enumerate(src) if abs(float(p["pts_time"]) - clip["copy_start_s"]) < 2e-3 and "K" in p["flags"])
    assert [p["size"] for p in cl[:6]] == [p["size"] for p in src[i0:i0 + 6]]
    # the duration is the file's own, within one frame; clip_time is counted from the real first picture
    assert abs(clip["duration_s"] - _video_duration(str(out / c["file"]))) <= 1 / 30
    for s in sc["samples"]:
        assert s["clip_time"] == pytest.approx(s["source_time"] - clip["copy_start_s"], abs=1e-3)
    assert sc["samples"][0]["clip_time"] >= 0 and sc["presence_in_clip"]["clip_s"][0][0] >= 0


def test_cut_clip_returns_where_the_copy_started(gop28, tmp_path):
    assert cut_clip(gop28, tmp_path / "a.mov", 1.9, 4.0) == pytest.approx(GOP, abs=1e-3)
    assert cut_clip(gop28, tmp_path / "b.mov", 1.95, 4.0) == pytest.approx(2 * GOP, abs=1e-3)
    assert seek_landing(gop28, 1.9) == pytest.approx(GOP, abs=1e-3)
    assert seek_landing(gop28, 0.0) == pytest.approx(0.0, abs=1e-3)


def test_a_clip_that_lasts_far_from_the_sidecar_is_warned_about(tmp_path):
    from inframe.cut import duration_warning

    assert duration_warning("c.mov", 10.0, 10.0, 20.0) is None
    assert duration_warning("c.mov", None, 10.0, 20.0) is None
    w = duration_warning("/x/c.mov", 11.0, 10.0, 20.0)  # a made-up clip one second longer than the sidecar says
    assert w and "c.mov" in w and "11.000" in w and "10.000" in w


def test_sidecar_duration_is_the_measured_one_else_end_minus_copy_start():
    from inframe import sidecar

    info = dict(path="/x/v.mov", duration=20.0, fps=30.0, rotation=0, mirrored=False, width=1280, height=720, display_width=1280, display_height=720)
    clip = dict(index=1, start=5.0, end=8.0, file="v_01_0005-0008.mov", expected_copy_start=4.0)
    kw = dict(info=info, src_hash=dict(algorithm="sha256", value="0" * 64, size_bytes=1), n_clips=1, times=[4.5], scores=[0.9], boxes=[(0.1, 0.1, 0.2, 0.2)],
              track_ids=None, presence=[(5.0, 8.0)], settings=dict(fps=5.0), target="t", embedder=None, detector=dict(mode="detect"))
    assert sidecar.build_sidecar(clip=clip, **kw)["clip"]["duration_s"] == 4.0
    assert sidecar.build_sidecar(clip=dict(clip, file_duration_s=4.078), **kw)["clip"]["duration_s"] == 4.078


def test_a_sound_snapped_start_on_a_keyframe_lands_on_that_keyframe(gop28):
    """snap_clips puts the start on a keyframe; a request exactly there is inside ffmpeg's bad window and would go back one GOP."""
    from inframe import audio

    kfs = keyframe_times(gop28)
    (s, _e, _), = audio.snap_clips([(2.3, 6.0)], [(2.4, 5.0)], kfs, 0.0, 8.0)  # target 1.9 snaps to the keyframe at 1.867
    assert any(abs(k - GOP * 2) < 1e-3 for k in kfs) and s < 2.4 - 0.1
    assert seek_landing(gop28, s) == pytest.approx(2 * GOP, abs=1e-3)
    assert seek_landing(gop28, 2 * GOP) == pytest.approx(GOP, abs=1e-3)  # the bare keyframe time is what used to be requested


def test_a_snapped_start_never_falls_back_into_the_bad_window_when_the_move_cap_binds(gop28):
    """Keyframe 1.867 + 2 frames + eps is past s + 1 s = 1.88 + ... : the cap used to win and request 1.88, which lands one GOP early."""
    from inframe import audio

    kfs = keyframe_times(gop28)
    (s, _e, _), = audio.snap_clips([(0.88, 6.0)], [(2.37, 5.0)], kfs, 0.0, 8.0)
    assert s <= 0.88 + 1.0 + 1e-9  # the move cap still holds
    from inframe.cut import snap_to_keyframe

    land = seek_landing(gop28, s)
    assert land == pytest.approx(snap_to_keyframe(s, kfs), abs=1e-3), (s, land)  # the copy starts at the keyframe at or before the request, not one GOP earlier
