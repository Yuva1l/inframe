"""T6: zone geometry (also under display rotation), camera check, activity speed, and the pipeline with both."""
import json
import subprocess

import numpy as np
import pytest

from inframe import activity, camera, pipeline
from inframe.activity import ActivityTarget, smoothed_speeds, speed_at, step_speeds
from inframe.cli import build_parser, main as cli_main
from inframe.decode import upright
from inframe.detect import Detection
from inframe.motion import run_motion
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe
from inframe.target import AnyPerson, Hit
from inframe.zone import Zone, ZoneTarget, zone_from_target_files


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


# ---------------------------------------------------------------- Zone
def test_parse_and_validation():
    assert Zone.parse("0.1, 0.2,0.5,0.4").as_list() == [0.1, 0.2, 0.5, 0.4]
    for bad in ("1,2,3", "a,b,c,d", "0.5,0,0.6,1", "0,0,0,1", "-0.1,0,0.5,0.5", "0,0,1,1.2", "nan,0,1,1"):
        with pytest.raises(ValueError):
            Zone.parse(bad)


def test_centre_and_overlap_rules():
    z = Zone(0.4, 0.4, 0.2, 0.2)
    assert z.admits((0.45, 0.45, 0.55, 0.55))
    assert not z.admits((0.0, 0.0, 0.1, 0.1))
    box = (0.5, 0.5, 0.9, 0.9)  # centre (0.7, 0.7) is outside; a quarter of the area (0.1 x 0.1 of 0.4 x 0.4) is inside
    assert not z.admits(box) and z.overlap(box) == pytest.approx(0.0625)
    assert z.admits(box, 0.05) and not z.admits(box, 0.1)
    assert z.overlap((0.0, 0.0, 0.1, 0.1)) == 0.0 and z.overlap((0.45, 0.45, 0.5, 0.5)) == 1.0
    m = z.mask(90, 160)
    assert m.shape == (90, 160) and abs(m.mean() - 0.04) < 0.005 and m[45, 80] and not m[0, 0]
    assert Zone(0.0, 0.0, 0.001, 0.001).mask(90, 160).sum() >= 1


def _det(cx, cy, w=0.1, h=0.2, score=0.9, tid=None):
    return Detection(score, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), tid)


def test_zone_target_filters_candidates_for_a_stateless_target():
    zt = ZoneTarget(AnyPerson(), Zone(0.0, 0.0, 0.5, 1.0))
    assert not hasattr(zt, "finalize") and zt.wants_tracks is False and zt.name == "any-person"
    # the best person is outside the zone; the weaker one inside it must still count
    h = zt.evaluate(0.0, np.zeros((9, 16, 3), np.uint8), [_det(0.8, 0.5, score=0.95), _det(0.2, 0.5, score=0.6)])
    assert h.score == 0.6
    assert zt.evaluate(0.2, np.zeros((9, 16, 3), np.uint8), [_det(0.8, 0.5)]).score == 0.0


def test_zone_target_gates_an_identity_target_after_the_fact():
    class Ident:
        name, wants_tracks, wants_slices = "ident", True, True

        def evaluate(self, t, frame, cands):
            return Hit(cands[0].score, cands[0].box, cands[0].track_id)

        def finalize(self):
            return [Hit(0.9, _det(0.8, 0.5).box), Hit(0.9, _det(0.2, 0.5).box)]

    zt = ZoneTarget(Ident(), Zone(0.0, 0.0, 0.5, 1.0))
    assert zt.wants_slices and hasattr(zt, "finalize")
    assert zt.evaluate(0, None, [_det(0.8, 0.5)]).score == 0.0 and zt.evaluate(0, None, [_det(0.2, 0.5)]).score == 0.9
    assert [h.score for h in zt.finalize()] == [0.0, 0.9]


def test_zone_in_a_target_json(tmp_path):
    f = tmp_path / "target.json"
    f.write_text(json.dumps(dict(taps=[], zone=[0.1, 0.1, 0.5, 0.5])))
    assert zone_from_target_files([f"tap:{f}"]).as_list() == [0.1, 0.1, 0.5, 0.5]
    assert zone_from_target_files(["tap:3,0.5,0.5", "photos:/x"]) is None and zone_from_target_files(None) is None


# ---------------------------------------------------------------- geometry under rotation (motion mode, real decode)
STORED = (90, 160)  # stored height, width
BOX = (8, 38, 10, 40)  # y0, y1, x0, x1 of the white box in the STORED picture (top left)


@pytest.fixture(scope="module")
def rotated(tmp_path_factory):
    d = tmp_path_factory.mktemp("zone_rot")
    plain = d / "plain.mp4"
    y0, y1, x0, x1 = BOX
    _ff("-f", "lavfi", "-i", f"color=c=0x404040:s=160x90:r=10:d=8,drawgrid=w=20:h=20:t=1:c=white@0.4,noise=alls=4:allf=t,"
        f"drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}:color=white:t=fill:enable='between(t,2,4)'",
        "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(plain))
    out = {}
    for r in (0, 90, -90, 180):
        p = d / f"r{r}.mp4"
        _ff("-display_rotation", str(r), "-i", str(plain), "-c", "copy", str(p))
        out[r] = str(p)
    return out


def _upright_centre(rotation):
    arr = np.zeros(STORED, np.uint8)
    y0, y1, x0, x1 = BOX
    arr[y0:y1, x0:x1] = 1
    up = upright(arr, rotation)
    ys, xs = np.nonzero(up)
    return xs.mean() / up.shape[1], ys.mean() / up.shape[0]


@pytest.mark.parametrize("rotation", [0, 90, -90, 180])
def test_zone_geometry_under_rotation(rotated, rotation):
    info = probe(rotated[rotation])
    assert info.rotation % 360 == rotation % 360  # ffprobe may say -180
    cx, cy = _upright_centre(rotation)
    mk = lambda x, y: Zone(min(max(x - 0.15, 0), 0.7), min(max(y - 0.15, 0), 0.7), 0.3, 0.3)
    on, off = mk(cx, cy), mk(1 - cx, 1 - cy)  # the zone where the box shows up, and the mirrored one
    r_on = run_motion(rotated[rotation], info, 5.0, zone=on)
    r_off = run_motion(rotated[rotation], info, 5.0, zone=off)
    r_all = run_motion(rotated[rotation], info, 5.0)
    t = np.asarray(r_on.times)
    during = (t >= 2.2) & (t <= 3.8)
    assert np.max(np.asarray(r_on.scores)[during]) > 0.2, (rotation, cx, cy)
    assert np.max(np.asarray(r_off.scores)) < 0.02, (rotation, cx, cy)
    outside = (t < 1.8) | (t > 4.2)
    assert np.max(np.asarray(r_on.scores)[outside]) < 0.05
    assert max(r_all.scores) < 0.2  # without the zone the same box is a small part of the whole picture
    assert r_on.details["zone"] == on.as_list()


def test_corner_of_a_rotated_picture_is_where_the_player_shows_it(rotated):
    """Rotation -90 (a quarter turn clockwise) puts the stored top-left corner at the upright top-right."""
    cx, cy = _upright_centre(-90)
    assert cx > 0.7 and cy < 0.4
    cx, cy = _upright_centre(180)
    assert cx > 0.7 and cy > 0.5


# ---------------------------------------------------------------- camera check
def test_camera_check(static_box, shaky):
    ok = camera.check_static(static_box, probe(static_box))
    assert ok.static and ok.p95_px < 1.0, ok  # a large white box appears for 2 s: the only 'shift' there is
    bad = camera.check_static(shaky, probe(shaky))
    assert not bad.static and bad.p95_px > 1.5 and "limit" in bad.reason, bad


def test_zone_is_ignored_on_a_moving_camera(shaky, tmp_path):
    doc = run_cut(shaky, str(tmp_path / "o"), Settings(mode="motion", zone=(0.0, 0.0, 0.5, 0.5)))
    assert any("zone" in w and "static" in w for w in doc["warnings"]), doc["warnings"]
    assert doc["settings"]["zone_applied"] is None and doc["settings"]["camera_check"]["static"] is False


def test_zone_is_applied_on_a_static_camera(static_box, tmp_path):
    doc = run_cut(static_box, str(tmp_path / "o"), Settings(mode="motion", zone=(0.3, 0.2, 0.5, 0.7), pad=0.0, min_len=0.5))
    assert doc["settings"]["zone_applied"] == [0.3, 0.2, 0.5, 0.7] and not doc["warnings"]
    assert len(doc["presence_intervals"]) == 1 and 1.5 < doc["presence_intervals"][0][0] < 2.5
    away = run_cut(static_box, str(tmp_path / "p"), Settings(mode="motion", zone=(0.0, 0.0, 0.2, 0.2), min_len=0.5))
    assert away["presence_intervals"] == []


# ---------------------------------------------------------------- activity
def test_speed_units_aspect_and_slowmo():
    ts = np.arange(0, 3, 0.2)
    # a box 0.25 of the frame height tall moves 0.05 of the frame width per step (0.2 s); the frame is 16:9
    boxes = [(0.1 + 0.05 * i, 0.4, 0.2 + 0.05 * i, 0.65) for i in range(len(ts))]
    tm, v = step_speeds(ts, boxes, aspect=16 / 9)
    assert v == pytest.approx(0.05 * 16 / 9 / 0.25 / 0.2)  # 1.78 box heights per second
    assert step_speeds(ts, boxes, 16 / 9, slowmo_factor=8)[1][0] == pytest.approx(v[0] * 8)  # real seconds: the file plays 8 times slower
    assert np.allclose(step_speeds(ts, boxes, 1.0)[1], 0.05 / 0.25 / 0.2)


def test_smoothing_ignores_one_bad_box_and_a_gap():
    ts = np.arange(0, 4, 0.2)
    boxes = [(0.4, 0.4, 0.5, 0.65)] * len(ts)
    boxes = list(boxes)
    boxes[10] = (0.7, 0.4, 0.8, 0.65)  # one jumped box
    sp = smoothed_speeds(ts, boxes)
    assert max(sp[i] for i in range(len(ts))) < 0.5  # two big steps in a window of five do not move the median
    assert step_speeds([0.0, 2.0], [(0, 0, 1, 1)] * 2)[1].size == 0  # a gap is no step
    assert smoothed_speeds([0.0, 0.2], [(0, 0, 0.1, 0.2)] * 2) == [None, None]  # one step is not enough: no verdict


def _walker_and_rider(n, with_rider=(10, 40)):
    """Per sample at 5 fps: a still person on the left, and a rider who rolls to and fro on the right at 3 box heights per second."""
    box_h, step = 0.3, 3 * 0.3 * 0.2 / (16 / 9)  # x step per sample
    out = []
    for i in range(n):
        c = [Detection(0.9, (0.1, 0.5, 0.2, 0.5 + box_h), None)]
        if with_rider[0] <= i < with_rider[1]:
            k = (i - with_rider[0]) % 12
            x = 0.4 + step * (k if k <= 6 else 12 - k)
            c.append(Detection(0.85, (x, 0.3, x + 0.1, 0.3 + box_h), None))
        out.append(c)
    return out


class _FakeTracker:
    """Ids by horizontal band: the person on the left is track 1, anyone else track 2."""

    def __init__(self, *a, **k):
        pass

    def update(self, t, dets, frame):
        from dataclasses import replace
        return [replace(d, track_id=1 if d.box[0] < 0.3 else 2) for d in dets]

    def reset(self):
        pass

    def close(self):
        pass


class _FakeDetector:
    backend, path = "fake", "fake.mlpackage"

    def __init__(self, series):
        self.series, self.n = series, 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append(self.series[self.n] if self.n < len(self.series) else [])
            self.n += 1
        return out


@pytest.fixture(scope="module")
def long_static(tmp_path_factory):
    p = tmp_path_factory.mktemp("long") / "long.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x404040:s=320x180:r=10:d=10,drawgrid=w=40:h=40:t=2:c=white@0.4", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(p))
    return str(p)


def _run(video, out, monkeypatch, series, **kw):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _FakeDetector(series))
    monkeypatch.setattr(pipeline.tracks, "Tracker", _FakeTracker)
    return run_cut(video, str(out), Settings(pad=0.0, min_len=1.0, merge_gap=1.0, **kw))


def test_activity_riding_keeps_the_mover_only(long_static, tmp_path, monkeypatch):
    series = _walker_and_rider(50)
    everyone = _run(long_static, tmp_path / "a", monkeypatch, series)
    assert len(everyone["presence_intervals"]) == 1 and everyone["presence_intervals"][0][1] > 9.5  # the still person is there all the time
    riding = _run(long_static, tmp_path / "b", monkeypatch, _walker_and_rider(50), activity="riding")
    (s, e), = riding["presence_intervals"]
    assert 1.6 < s < 2.6 and 7.2 < e < 8.4, riding["presence_intervals"]  # the rider is in frame 2.0 s to 8.0 s
    assert riding["settings"]["activity_info"]["candidates_dropped"] > 40
    assert riding["settings"]["camera_check"]["static"]
    # a threshold above the rider's speed drops everyone
    none = _run(long_static, tmp_path / "c", monkeypatch, _walker_and_rider(50), activity="riding", activity_threshold=6.0)
    assert none["presence_intervals"] == []


def test_activity_with_a_zone_counts_the_rider_in_it_only(long_static, tmp_path, monkeypatch):
    # the rider moves between x 0.4 and 0.85; the zone is the left half: the rider's box centre is mostly outside it
    doc = _run(long_static, tmp_path / "d", monkeypatch, _walker_and_rider(50), activity="riding", zone=(0.0, 0.0, 0.55, 1.0))
    assert doc["settings"]["zone_applied"] == [0.0, 0.0, 0.55, 1.0]
    covered = sum(e - s for s, e in doc["presence_intervals"])
    full = _run(long_static, tmp_path / "e", monkeypatch, _walker_and_rider(50), activity="riding")
    assert covered < sum(e - s for s, e in full["presence_intervals"]) - 1.0
    # zone alone: the still person on the left counts all the time, the rider's extra time adds nothing
    zone_only = _run(long_static, tmp_path / "f", monkeypatch, _walker_and_rider(50), zone=(0.0, 0.0, 0.3, 1.0))
    assert zone_only["presence_intervals"][0][1] > 9.5


def test_activity_without_tracker_warns_and_keeps_everyone(long_static, tmp_path, monkeypatch):
    doc = _run(long_static, tmp_path / "g", monkeypatch, _walker_and_rider(10), activity="riding", fps=1.0)
    assert any("the activity setting was not applied" in w for w in doc["warnings"]), doc["warnings"]


def test_bad_settings(long_static, tmp_path):
    for kw, msg in ((dict(mode="motion", activity="riding"), "needs --mode detect"), (dict(activity="sprinting"), "must be one of"),
                    (dict(zone_overlap=0.5), "needs --zone"), (dict(zone=(0, 0, 1, 1), zone_overlap=1.5), "--zone-overlap"),
                    (dict(zone=(0.5, 0, 0.8, 1)), "--zone"), (dict(slowmo_factor=0), "slow-motion"), (dict(activity_threshold=-1), "--activity-threshold")):
        with pytest.raises(ValueError, match=msg):
            run_cut(long_static, str(tmp_path / "x"), Settings(**kw))


def test_cli_flags(long_static, tmp_path, capsys):
    a = build_parser().parse_args(["cut", "v.mp4", "--zone", "0,0,0.5,0.5", "--zone-overlap", "0.3", "--activity", "riding", "--slowmo-factor", "8"])
    assert (a.zone, a.zone_overlap, a.activity, a.slowmo_factor) == ("0,0,0.5,0.5", 0.3, "riding", 8.0)
    assert cli_main(["cut", long_static, "--zone", "0,0,2,2", "--out", str(tmp_path / "o")]) == 2
    assert cli_main(["cut", long_static, "--mode", "motion", "--activity", "riding", "--out", str(tmp_path / "o")]) == 2
    assert cli_main(["cut", long_static, "--zone-overlap", "0.4", "--out", str(tmp_path / "o")]) == 2
    assert "needs --zone" in capsys.readouterr().err


# ---------------------------------------------------------------- review findings
def _still_jitter_tracks(n_tracks=300, seconds=8, fps=5, slowmo=1.0):
    """Still boxes with detector jitter of about 2 % of the box height per axis, one sample every 1/fps file seconds."""
    rng = np.random.default_rng(7)
    ts = np.arange(0, seconds, 1 / fps)
    out = []
    for _ in range(n_tracks):
        h = 0.3
        boxes = [(0.4 + rng.normal(0, 0.02 * h) / 1.78, 0.4 + rng.normal(0, 0.02 * h), 0.5, 0.4 + h) for _ in ts]
        out.append(smoothed_speeds(ts, boxes, 16 / 9, slowmo))
    return [v for sp in out for v in sp if v is not None]


def test_slowmo_factor_does_not_multiply_detector_jitter():
    base = _still_jitter_tracks(slowmo=1.0)
    slow = _still_jitter_tracks(slowmo=8.0)
    assert np.mean(np.asarray(base) >= activity.THRESHOLD) == 0
    assert np.mean(np.asarray(slow) >= activity.THRESHOLD) < 0.01, np.mean(np.asarray(slow) >= activity.THRESHOLD)
    assert np.median(slow) < 2 * np.median(base)


def test_slowmo_factor_still_scales_a_real_mover_to_real_seconds():
    ts = np.arange(0, 20, 0.2)  # a file that plays 8 times slower: 3 box heights per real second = 0.375 per file second
    box_h, dx = 0.3, 3 * 0.3 / 8 * 0.2 / (16 / 9)
    boxes = [(0.1 + dx * i, 0.4, 0.2 + dx * i, 0.4 + box_h) for i in range(len(ts))]
    sp = [v for v in smoothed_speeds(ts, boxes, 16 / 9, 8.0) if v is not None]
    assert sp and np.allclose(sp, 3.0, rtol=0.05)


def test_activity_gates_a_low_score_bystander(long_static, tmp_path, monkeypatch):
    def series(score):
        out = []
        for i in range(50):
            c = [Detection(score, (0.1, 0.5, 0.2, 0.8), None)]  # stands still all 10 s
            if 10 <= i < 25:  # the rider is in frame 2 s to 5 s
                k = (i - 10) % 12
                x = 0.4 + 3 * 0.3 * 0.2 / (16 / 9) * (k if k <= 6 else 12 - k)
                c.append(Detection(0.85, (x, 0.3, x + 0.1, 0.6), None))
            out.append(sorted(c, key=lambda d: -d.score))  # the detector lists the best score first
        return out
    for score in (0.45, 0.35):
        doc = _run(long_static, tmp_path / f"s{score}", monkeypatch, series(score), activity="riding")
        iv = doc["presence_intervals"]
        assert len(iv) == 1 and iv[0][0] > 1.5 and iv[0][1] < 5.6, (score, iv)  # the rider only; the bystander does not hold the clip open


def test_camera_ignores_a_big_subject_that_locks_the_phase_correlation():
    rng = np.random.default_rng(1)  # seed 1: 2 of the 12 keyframes lock onto the subject (shifts of 80 and 64 px), the rest are under 1 px
    bg = np.full((180, 320), 64.0, np.float32)
    bg[::40, :] = bg[:, ::40] = 102.4
    bg = np.clip(bg + rng.normal(0, 20, bg.shape), 0, 255)
    subject = rng.random((150, 120)) * 255  # textured, 120 px wide, slides across a tripod shot
    grays = []
    for i in range(12):
        g = np.clip(bg + rng.normal(0, 10, bg.shape), 0, 255)
        x = 10 + int(i * 190 / 11)
        g[15:165, x:x + 120] = subject
        grays.append(camera.blur5(g.astype(np.uint8)))
    assert (camera.shifts_from_frames(grays) > 50).sum() == 2
    ck = camera.check_grays(grays)
    assert ck.static, ck


def test_a_drifting_camera_is_still_moving():
    rng = np.random.default_rng(4)
    bg = (rng.random((180, 480)) * 255).astype(np.float32)
    grays = [camera.blur5(bg[:, 6 * i:6 * i + 320]) for i in range(12)]  # slides 6 px per keyframe
    assert not camera.check_grays(grays).static


def test_pick_zone_drawn_to_the_edge_is_valid_after_rounding(tmp_path):
    from types import SimpleNamespace
    from inframe.pick import Picker

    pk = Picker.__new__(Picker)
    from inframe.classes import PERSON

    pk.taps, pk.zone, pk.out, pk.video, pk.photos, pk.cls = [], None, tmp_path / "t.json", "v.mp4", None, PERSON  # cls: T8a
    pk.model, pk.preset_name = "small", None  # T1e: target.json records the detector
    pk.info = SimpleNamespace(display_size=(1280, 720))
    for width in range(400, 1281):
        z = [1 / width, 0.2, 1 - 1 / width, 0.5]  # drag from 1 px in to past the right edge (clamped to 1)
        pk.set_zone(z)
        Zone.of(pk.zone)  # raises when the saved zone is not valid


def _reaim_grays(offsets):
    rng = np.random.default_rng(4)
    bg = (rng.random((180, 640)) * 255).astype(np.float32)  # a textured scene; each offset is where the tripod points
    return [camera.blur5(bg[:, o:o + 320]) for o in offsets]


@pytest.mark.parametrize("offsets", [[0] * 9 + [60] * 3, [200, 150, 100] + [0] * 13, [0] * 8 + [120] * 4])
def test_camera_step_change_held_by_a_run_of_keyframes_is_a_move(offsets):
    """A re-aimed tripod or a quick pan then tripod holds a large shift for consecutive keyframes: not subject lock."""
    ck = camera.check_grays(_reaim_grays(offsets))
    assert not ck.static, ck
    assert ck.reason


def test_activity_slowmo_short_or_flickering_tracks_still_get_a_speed():
    frame = np.zeros((720, 1280, 3), np.uint8)
    rng = np.random.default_rng(0)

    def run(f):
        act = ActivityTarget(AnyPerson(), slowmo_factor=f)
        for k in range(100):  # a still bystander at 5 samples/s whose detection is on 1.4 s and off 0.8 s
            t = k * 0.2
            on = (t % 2.2) < 1.4
            c = [Detection(0.8, (0.4 + rng.normal(0, .003), 0.4 + rng.normal(0, .006), 0.5, 0.7), None)] if on else []
            act.evaluate(t, frame, c)
        return sum(h.score > 0 for h in act.finalize()), act
    for f in (1.0, 8.0):
        kept, act = run(f)
        assert kept == 0, (f, kept, act.stats)  # a still person is never riding
    ts = np.arange(0, 1.6, 0.2)  # one continuous still track, shorter than a step at factor 8
    sp = smoothed_speeds(ts, [(0.4, 0.4, 0.5, 0.7)] * len(ts), 16 / 9, 8.0)
    assert all(s is not None and s < 0.5 for s in sp), sp


def test_camera_reaim_to_a_view_that_cannot_be_matched_is_a_move():
    """A tripod turned so far that the keyframes share nothing with the background gives unreadable shifts (nan), not small ones."""
    rng = np.random.default_rng(4)
    bg = (rng.random((180, 700)) * 255).astype(np.float32)
    grays = [camera.blur5(bg[:, o:o + 320]) for o in [0] * 9 + [330] * 3]
    assert np.isnan(camera.shifts_from_frames(grays)[-3:]).all()  # the premise: nothing to match against
    ck = camera.check_grays(grays)
    assert not ck.static, ck
    assert ck.reason


def test_camera_flat_scene_is_still_taken_as_static():
    """A flat picture (nothing to correlate) says nothing about the camera: unreadable keyframes alone are not a move."""
    grays = [np.full((180, 320), 120.0, np.float32) for _ in range(12)]
    assert camera.check_grays(grays).static


def test_camera_one_unreadable_keyframe_in_a_still_scene_is_static():
    rng = np.random.default_rng(4)
    bg = (rng.random((180, 700)) * 255).astype(np.float32)
    grays = [camera.blur5(bg[:, :320]) for _ in range(16)]
    grays[5] = np.full((180, 320), 120.0, np.float32)
    assert camera.check_grays(grays).static


def _bystander(f, on_s, dt=0.2, n=400, seed=0):
    """A still person detected in bursts of `on_s` file seconds, with gaps that end the track (over MAX_STEP_S real seconds)."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    rng = np.random.default_rng(seed)
    act = ActivityTarget(AnyPerson(), slowmo_factor=f)
    off = activity.MAX_STEP_S * f + 0.5
    for k in range(n):
        t = k * dt
        c = [Detection(0.8, (0.4 + rng.normal(0, .003), 0.4 + rng.normal(0, .006), 0.5, 0.7), None)] if (t % (on_s + off)) < on_s else []
        act.evaluate(t, frame, c)
    return sum(h.score > 0 for h in act.finalize())


@pytest.mark.parametrize("on_s", [0.6, 0.8, 1.0])
def test_activity_slowmo_short_separate_tracks_of_a_still_person_are_dropped(on_s):
    assert _bystander(8.0, on_s) == 0


def test_activity_slowmo_short_track_of_a_rider_is_kept():
    f, n = 8.0, 5
    ts = np.arange(n) * 0.2
    step = 3.0 * 0.3 * 0.2 / f / (16 / 9)  # 3 box heights per real second, in normalised x per sample (box height 0.3)
    boxes = [(0.2 + k * step, 0.4, 0.3 + k * step, 0.7) for k in range(n)]
    sp = smoothed_speeds(ts, boxes, 16 / 9, f)
    assert all(s is not None and s > activity.THRESHOLD for s in sp), sp
