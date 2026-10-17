"""Photo mode: higher-resolution decode for crops, early embedding, photo and session thresholds apart."""
import subprocess

import numpy as np
import pytest

from inframe import identity, pipeline
from inframe.detect import Detection
from inframe.identity import MIN_CROP_H, PhotoTarget, SPECS, person_crop
from inframe.pipeline import Settings, run_cut
from inframe.tap import Tap, TapTarget
from inframe.target import Hit

from test_pipeline import FakeDetector
from test_t1c import RED, frame_with, photos_dir


# ---------------------------------------------------------------- crops from a higher-resolution frame
def test_a_small_person_is_embedded_from_the_1280_frame_not_the_640_one():
    box = (0.40, 0.30, 0.45, 0.30 + 30 / 360)  # 30 px high at 640x360, 60 px high at 1280x720
    low, high = np.zeros((360, 640, 3), np.uint8), np.zeros((720, 1280, 3), np.uint8)
    assert (box[3] - box[1]) * 360 < MIN_CROP_H <= (box[3] - box[1]) * 720
    assert person_crop(low, box) is None
    crop = person_crop(high, box)
    assert crop is not None and crop.shape[0] >= MIN_CROP_H


def _video_1280(tmp_path):
    p = tmp_path / "wide.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=10:duration=2",
                    "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(p)], check=True)
    return str(p)


class _Rec:
    """A target that only records the width of the frames it is given."""
    name, widths = "rec", None

    def __init__(self, long_side=None):
        self.widths = []
        if long_side:
            self.decode_long_side = long_side

    def evaluate(self, t, frame, cands):
        self.widths.append(frame.shape[1])
        return Hit(0.0, None)


class _SeeingDetector(FakeDetector):
    """The fake detector, recording the widths of the pictures it is given."""
    widths = None

    def detect(self, frames):
        self.widths = (self.widths or []) + [f.shape[1] for f in frames]
        return super().detect(frames)


def test_the_pipeline_decodes_at_the_long_side_the_target_asks_for_but_the_detector_still_sees_640(tmp_path, monkeypatch):
    v = _video_1280(tmp_path)
    seen = {}
    for name, target in (("plain", _Rec()), ("wide", _Rec(1280))):
        det = _SeeingDetector()
        monkeypatch.setattr(pipeline, "load_detector", lambda size, det=det: det)
        run_cut(v, str(tmp_path / name), Settings(), target)
        seen[name] = (set(target.widths), set(det.widths))
    assert seen["plain"] == ({640}, {640})
    assert seen["wide"] == ({1280}, {640})  # crops from 1280, detections as ever from the 640 picture (the approved baseline intervals ride on them)
    assert PhotoTarget.decode_long_side == 1280 and not hasattr(TapTarget([]), "decode_long_side")


# ---------------------------------------------------------------- embedding schedule
def test_a_short_track_gets_three_crops_and_then_one_per_second(tmp_path):
    d = photos_dir(tmp_path, RED)
    box = (0.05, 0.1, 0.30, 0.9)

    def crops_of(samples):
        t = PhotoTarget(str(d), embedder="color", rank_only=True)
        for i in range(samples):
            t.evaluate(round(i * 0.2, 1), frame_with([(RED, box)]), [Detection(0.9, box, 1)])
        t.finalize()
        return len(t.vecs)

    assert crops_of(3) == 3  # a 0.6 s track used to be embedded once and could not be ranked (MIN_EMBEDS = 2)
    assert crops_of(15) == 5  # samples 0, 1, 2, then 1.4 s and 2.4 s


# ---------------------------------------------------------------- photos and session gallery apart
class Axes:
    """An embedder whose vector for a crop is chosen by the colour of its centre pixel (colours in `table`)."""
    kind = "fake"

    def __init__(self, table):
        self.table = table

    def embed(self, crops):
        return np.stack([self.table[int(c[c.shape[0] // 2, c.shape[1] // 2, 0])] for c in crops]).astype(np.float32)

    def close(self):
        pass


def unit(spare, *parts):
    """A unit vector in 8-D: `parts` are the first components, the rest of the length goes to axis `spare` (own axis per vector, so that
    two of them are only as alike as their `parts` make them)."""
    v = np.zeros(8)
    v[:len(parts)] = parts
    v[spare] = np.sqrt(max(0.0, 1 - float(v @ v)))
    return v


def _frame(key, box, W=400, H=600):
    f = np.full((H, W, 3), 120, np.uint8)
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))
    f[y0:y1, x0:x1] = key
    return f


def test_a_stranger_who_matches_only_the_session_crops_in_between_is_refused():
    spec = SPECS["dinov2"]
    assert spec.session > spec.threshold
    mid = (spec.threshold + spec.session) / 2  # above the photo threshold, below the session one
    P, S = unit(7, 1.0), unit(7, 0.0, 1.0)
    table = {10: P, 20: S,
             30: unit(3, 0.0, mid),  # X: the stranger. Cosine to the photo 0, to the session crops `mid`
             40: unit(4, 0.0, spec.session + 0.08),  # Y: the target in other clothes: close to the session crops
             50: unit(5, spec.threshold + 0.05)}  # Z: close to the photo
    box = (0.05, 0.1, 0.30, 0.9)
    photo = np.full((200, 80, 3), 10, np.uint8)
    t = PhotoTarget("unused", [Tap(0.4, box=box)], embedder="dinov2", embedder_obj=Axes(table), photo_crops=[photo])
    plan = [(20, 1)] * 10 + [(30, 2)] * 10 + [(40, 3)] * 10 + [(50, 4)] * 10  # 40 samples, a new track every 2 s
    for i, (key, tid) in enumerate(plan):
        t.evaluate(round(i * 0.2, 1), _frame(key, box), [Detection(0.9, box, tid)])
    hits = t.finalize()
    got = [h.track_id for h in hits if h.box is not None]
    assert set(got) == {1, 3, 4}, t.info  # the stranger is not taken: with one gallery of photos + session crops he scored `mid`
    assert 2 not in got and t.info["tracks_accepted"] == 3
    # the old rule: best match over photos and session crops together, one threshold
    assert mid >= spec.threshold


def test_the_small_copy_is_what_a_640_decode_gives(tmp_path):
    from inframe.decode import iter_samples
    from inframe.probe import probe

    v = _video_1280(tmp_path)
    info = probe(v)
    both = list(iter_samples(v, info, fps=5.0, long_side=1280, small_side=640))
    plain = list(iter_samples(v, info, fps=5.0, long_side=640))
    assert len(both) == len(plain) > 0
    for a, b in zip(both, plain):
        assert a.rgb.shape[1] == 1280 and a.small.shape == b.rgb.shape
        assert np.array_equal(a.small, b.rgb)  # bit for bit: the detector's input does not change with the decode size
    assert all(s.small is None for s in plain)
