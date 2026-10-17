import subprocess

import numpy as np
import pytest

from inframe.cut import clip_name, cut_clip, mmss, snap_to_keyframe
from inframe.decode import choose_decoder, iter_samples, scaled_size, upright
from inframe.detect import postprocess
from inframe.probe import probe
from inframe.target import AnyPerson
from inframe.detect import Detection


def test_probe_synthetic(synth):
    i = probe(synth["rot"])
    assert (i.codec, i.width, i.height, i.fps) == ("h264", 320, 180, 30.0)
    assert i.rotation == -90 and i.display_size == (180, 320)
    assert i.duration == pytest.approx(3.0, abs=0.05)
    assert i.has_audio and i.audio_codec == "aac"
    assert i.keyframes == pytest.approx([0, 0.5, 1.0, 1.5, 2.0, 2.5], abs=1e-3)
    assert probe(synth["plain"]).rotation == 0


def test_choose_decoder(synth):
    assert choose_decoder(probe(synth["plain"], with_keyframes=False)) == "software"


def test_scaled_size_never_upscales_and_is_even():
    assert scaled_size(320, 180, 640) == (320, 180)
    assert scaled_size(1280, 720, 640) == (640, 360)
    assert scaled_size(3840, 2160, 640) == (640, 360)


def test_upright_all_rotations():
    a = np.arange(6).reshape(2, 3)
    assert upright(a, 0) is a
    assert upright(a, -90).shape == (3, 2) and upright(a, 90).shape == (3, 2)
    assert upright(a, 180).tolist() == [[5, 4, 3], [2, 1, 0]]
    assert upright(a, -90).tolist() == np.rot90(a, 3).tolist()  # quarter turn clockwise


def test_samples_by_timestamp(synth):
    p = synth["plain"]
    i = probe(p)
    ss = list(iter_samples(p, i, fps=5.0))
    ts = [s.t for s in ss]
    assert len(ss) == 15
    assert ts == pytest.approx([k * 0.2 for k in range(15)], abs=1e-3)
    assert ss[0].rgb.shape == (180, 320, 3) and ss[0].rgb.dtype == np.uint8


def test_samples_rate_and_keyframes(synth):
    p = synth["plain"]
    i = probe(p)
    assert len(list(iter_samples(p, i, fps=2.0))) == 6
    kf = [s.t for s in iter_samples(p, i, fps=None)]
    assert kf == pytest.approx(i.keyframes, abs=1e-3)


def test_rotation_applied_matches_ffmpeg_autorotate(synth):
    p = synth["rot"]
    i = probe(p)
    first = next(iter_samples(p, i, fps=5.0)).rgb
    assert first.shape == (320, 180, 3)  # portrait after rotation
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", p, "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    ref = np.frombuffer(raw, np.uint8).reshape(320, 180, 3)
    assert np.abs(ref.astype(int) - first.astype(int)).mean() < 3.0
    wrong = np.rot90(first, 2)
    assert np.abs(ref.astype(int) - wrong.astype(int)).mean() > 10.0  # the check can fail


def test_cut_keeps_audio_rotation_and_starts_on_keyframe(synth, tmp_path):
    out = tmp_path / "c.mp4"
    cut_clip(synth["rot"], out, 1.2, 2.2)
    i = probe(str(out))
    assert i.rotation == -90 and i.has_audio
    assert 1.0 <= i.duration <= 1.0 + 0.7 + 0.1  # request 1.0 s, copy starts at the keyframe 1.0
    assert snap_to_keyframe(1.2, [0, 0.5, 1.0, 1.5]) == 1.0
    assert snap_to_keyframe(0.0, [0, 0.5]) == 0.0


def test_names():
    assert mmss(83.9) == "0123" and mmss(3599) == "5959"
    assert clip_name("IMG_1", 3, 83.2, 125.0) == "IMG_1_03_0123-0205.mov"
    assert clip_name("a\nb\rc\x00d", 1, 0, 1) == "a_b_c_d_01_0000-0001.mov"  # no line break in a clip name


def test_postprocess_and_any_person():
    boxes = np.zeros((2, 3, 4), np.float32)
    boxes[0, 1] = [0.5, 0.5, 0.2, 0.4]
    logits = np.full((2, 3, 91), -10.0, np.float32)
    logits[0, 1, 1] = 2.0   # person, sigmoid ~0.88
    logits[0, 2, 41] = 5.0  # skateboard: ignored
    dets = postprocess(boxes, logits, n=2)
    assert len(dets[0]) == 1 and dets[1] == []
    assert dets[0][0].score == pytest.approx(0.8808, abs=1e-3)
    assert dets[0][0].box == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert AnyPerson().evaluate(0.0, None, dets[0]).score > 0.88
    assert AnyPerson(min_box=0.5).evaluate(0.0, None, dets[0]).score == 0.0  # box is 0.4 tall
    assert AnyPerson().evaluate(0.0, None, []).score == 0.0


def test_detect_postprocess_picks_boxes_and_logits_by_shape():
    from inframe.detect import _split_out

    boxes = np.zeros((8, 300, 4), np.float32)
    logits = np.zeros((8, 300, 91), np.float32)
    b, l = _split_out([logits, boxes])  # order does not matter
    assert b.shape[-1] == 4 and l.shape[-1] == 91
    # batch padded with a repeated frame: only the first n images are returned
    assert len(postprocess(boxes, logits - 10, n=3)) == 3


def test_display_size_applies_the_sample_aspect_ratio(tmp_path):
    """fix r1-g2 frontend-3: an anamorphic file (1440x1080, SAR 4:3) is shown 16:9 by a browser, so the display size the UI lays
    overlays out on is 1920x1080, and a turned one swaps after the stretch."""
    p, q = tmp_path / "sar.mp4", tmp_path / "sar_rot.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=1440x1080:rate=10:duration=1",
                    "-vf", "setsar=4/3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    i = probe(str(p), with_keyframes=False)
    assert (i.width, i.height) == (1440, 1080) and i.display_size == (1920, 1080)
    assert i.to_dict()["display_width"] == 1920
    from inframe.pick import grab_frame

    _, rgb = grab_frame(str(p), i, 0.0, long_side=640)  # the frame the tap page shows has the shape the overlay is laid out on
    assert abs(rgb.shape[1] / rgb.shape[0] - 16 / 9) < 0.02, rgb.shape
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-display_rotation", "-90", "-i", str(p), "-c", "copy", str(q)], check=True)
    assert probe(str(q), with_keyframes=False).display_size == (1080, 1920)
    plain = tmp_path / "sq.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)], check=True)
    assert probe(str(plain), with_keyframes=False).display_size == (320, 180)
