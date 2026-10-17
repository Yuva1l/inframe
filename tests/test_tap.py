"""Tap mode: the gap-tolerant bridge, short-track stitching, and the deterministic (CPU_ONLY) detector."""
import os
from pathlib import Path

import numpy as np
import pytest

from inframe import detect, tap
from inframe.tap import Tap, TapTarget

from test_t1b import BLUE, BOX, GREEN, RED, frame_of, run

TAP = lambda t=0.4: TapTarget([Tap(t, point=(0.5, 0.5))])
on = lambda hits: [h.score > 0 for h in hits]


def test_bridge_takes_a_box_shrinking_at_the_frame_edge():
    """He walks out at the left edge: the track ends, the detector still boxes a shrinking sliver of him with no track.
    The slivers hardly overlap his last box (IoU < BRIDGE_IOU) but lie inside it."""
    edge = (0.0, 0.2, 0.2, 0.8)
    slivers = [(0.0, 0.2, 0.08, 0.8), (0.0, 0.2, 0.03, 0.8), (0.0, 0.2, 0.01, 0.8)]
    assert tap.iou(edge, slivers[-1]) < tap.BRIDGE_IOU
    shot = [[(edge, 1, RED, BLUE)]] * 6 + [[(b, None, RED, BLUE)] for b in slivers]
    hits = run(TapTarget([Tap(0.4, point=(0.1, 0.5))]), [shot])
    assert on(hits) == [True] * 9
    assert [h.box for h in hits[6:]] == slivers


def test_bridge_crosses_a_hole_of_three_samples():
    near = (0.45, 0.2, 0.65, 0.8)  # IoU 0.5 with BOX: a step to the side
    shot = [[(BOX, 1, RED, BLUE)]] * 5 + [[]] * 3 + [[(near, None, GREEN, GREEN)]] * 3
    assert tap.iou(BOX, near) < 0.6
    hits = run(TAP(), [shot])
    assert on(hits) == [True] * 5 + [False] * 3 + [True] * 3
    # and backwards: untracked boxes before the hole, his track after it
    shot = [[(near, None, GREEN, GREEN)]] * 3 + [[]] * 3 + [[(BOX, 1, RED, BLUE)]] * 5
    assert on(run(TAP(2.4), [shot])) == [True] * 3 + [False] * 3 + [True] * 5


def test_bridge_does_not_take_a_stranger_who_does_not_overlap():
    far = (0.7, 0.2, 0.9, 0.8)
    shot = [[(BOX, 1, RED, BLUE)]] * 5 + [[(far, None, GREEN, GREEN)]] * 4
    assert on(run(TAP(), [shot])) == [True] * 5 + [False] * 4
    # a stranger who overlaps a little but is not inside his box (IoU 0.05) is not him either
    graze = (0.58, 0.2, 0.78, 0.8)
    assert tap.iou(BOX, graze) < tap.BRIDGE_IOU and tap._inside(graze, BOX) < tap.BRIDGE_INSIDE
    shot = [[(BOX, 1, RED, BLUE)]] * 5 + [[(graze, None, GREEN, GREEN)]] * 4
    assert on(run(TAP(), [shot])) == [True] * 5 + [False] * 4


def test_bridge_chain_is_capped_from_the_last_matched_sample():
    """He leaves; an untracked stranger (IoU 0.14 with his box) stays. The chain must not follow the stranger for ever."""
    stranger = (0.55, 0.2, 0.75, 0.8)
    assert tap.iou(BOX, stranger) >= tap.BRIDGE_IOU
    shot = [[(BOX, 1, RED, BLUE)]] * 10 + [[(stranger, None, GREEN, GREEN)]] * 50
    hits = run(TAP(), [shot])
    gap = round(tap.LOST_BUFFER_S / 0.2)
    assert sum(on(hits)) <= 10 + gap
    assert on(hits)[10 + gap:] == [False] * (50 - gap)
    # the same walking away to the right edge
    walk = [(0.55 + 0.005 * i, 0.2, 0.75 + 0.005 * i, 0.8) for i in range(50)]
    shot = [[(BOX, 1, RED, BLUE)]] * 10 + [[(b, None, GREEN, GREEN)] for b in walk]
    hits = run(TAP(), [shot])
    assert on(hits)[10 + gap:] == [False] * (50 - gap)


def test_bridge_keeps_him_when_he_stays_untracked_and_looks_the_same():
    """Small or far away he gets no track id for many seconds (score under the tracker's threshold). The cap from the last matched
    sample must not drop him: a box that looks like the gallery is taken whatever its distance from the last match."""
    near = (0.41, 0.2, 0.61, 0.8)
    shot = [[(BOX, 1, RED, BLUE)]] * 10 + [[(near, None, RED, BLUE)]] * 40  # case A: tracked, then 8 s untracked
    assert on(run(TAP(), [shot])) == [True] * 50
    shot = [[(BOX, 1, RED, BLUE)]] * 10 + [[(BOX, None, RED, BLUE)]] * 30 + [[(BOX, 2, RED, BLUE)]] * 10  # case B: a hole of 6 s in one clip
    assert on(run(TAP(), [shot])) == [True] * 50
    # a stranger in the same place, other clothes, is still dropped once the cap is reached
    stranger = [[(BOX, 1, RED, BLUE)]] * 10 + [[(near, None, GREEN, GREEN)]] * 40
    assert sum(on(run(TAP(), [stranger]))) <= 10 + round(tap.LOST_BUFFER_S / 0.2)


def test_bridge_never_crosses_a_cut():
    near = (0.41, 0.2, 0.61, 0.8)
    a = [[(BOX, 1, RED, BLUE)]] * 5
    b = [[(near, None, GREEN, GREEN)]] * 4
    assert on(run(TAP(), [a, b])) == [True] * 5 + [False] * 4


def accepted_ids(t):
    return {a["id"] for a in t.info["accepted_tracks"]}


def test_short_track_is_stitched_after_an_accepted_one():
    """Track 1 (his) ends; a 2-sample track 2 starts two samples later where he was: him, whatever the clothes."""
    near = (0.42, 0.2, 0.62, 0.8)  # IoU 0.75 with BOX
    assert tap.iou(BOX, near) >= 0.7
    shot = [[(BOX, 1, RED, BLUE)]] * 6 + [[]] * 2 + [[(near, 2, GREEN, GREEN)]] * 2 + [[]] * 5
    t = TAP()
    hits = run(t, [shot])
    assert on(hits) == [True] * 6 + [False] * 2 + [True] * 2 + [False] * 5
    assert accepted_ids(t) == {1, 2}


def test_short_track_before_an_accepted_one_is_stitched():
    near = (0.42, 0.2, 0.62, 0.8)
    shot = [[]] * 4 + [[(near, 2, GREEN, GREEN)]] * 2 + [[]] * 2 + [[(BOX, 1, RED, BLUE)]] * 6
    t = TAP(2.2)
    hits = run(t, [shot])
    assert on(hits) == [False] * 4 + [True] * 2 + [False] * 2 + [True] * 6
    assert accepted_ids(t) == {1, 2}


def test_short_track_far_away_is_not_stitched():
    far = (0.7, 0.2, 0.9, 0.8)
    shot = [[(BOX, 1, RED, BLUE)]] * 6 + [[]] * 2 + [[(far, 3, GREEN, GREEN)]] * 2 + [[]] * 5
    t = TAP()
    assert on(run(t, [shot])) == [True] * 6 + [False] * 9
    assert accepted_ids(t) == {1}


def test_short_track_in_another_shot_or_too_late_is_not_stitched():
    near = (0.42, 0.2, 0.62, 0.8)
    t = TAP()
    assert on(run(t, [[[(BOX, 1, RED, BLUE)]] * 6, [[(near, 2, GREEN, GREEN)]] * 2])) == [True] * 6 + [False] * 2
    late = [[(BOX, 1, RED, BLUE)]] * 6 + [[]] * 15 + [[(near, 2, GREEN, GREEN)]] * 2
    t = TAP()
    assert on(run(t, [late])) == [True] * 6 + [False] * 17
    assert accepted_ids(t) == {1}


def test_a_long_rejected_track_is_not_stitched():
    near = (0.42, 0.2, 0.62, 0.8)
    shot = [[(BOX, 1, RED, BLUE)]] * 6 + [[(near, 2, GREEN, GREEN)]] * 8
    t = TAP()
    assert on(run(t, [shot])) == [True] * 6 + [False] * 8


# ---------------------------------------------------------------- deterministic detector
def test_units_env(monkeypatch):
    monkeypatch.delenv(detect.UNITS_ENV, raising=False)
    assert detect.compute_units_name() == "CPU_AND_GPU"  # the default (repeatable with the fp32-topk package)
    monkeypatch.setenv(detect.UNITS_ENV, "cpu_and_ne")
    assert detect.compute_units_name() == "CPU_AND_NE"  # the deterministic option that is not exact on the regression video
    monkeypatch.setenv(detect.UNITS_ENV, "cpu_only")
    assert detect.compute_units_name() == "CPU_ONLY"  # reproduces the reference run exactly
    monkeypatch.setenv(detect.UNITS_ENV, "bogus")
    assert detect.compute_units_name() == "CPU_AND_GPU"


def _have_models() -> bool:
    try:
        detect.find_package("small")
        return True
    except Exception:
        return False


def _synthetic(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    frames = []
    for k in range(n):
        f = rng.integers(60, 200, (360, 640, 3), dtype=np.uint8)
        x0 = 40 + 30 * k
        f[60:330, x0:x0 + 90] = (200, 40, 40)  # a person-sized block
        frames.append(f)
    return frames


needs_models = pytest.mark.skipif(not _have_models(), reason="needs the CoreML package (bench/detect/setup.sh)")


@needs_models
def test_default_detector_is_repeatable(monkeypatch):
    """The same 16 frames, detected twice: identical boxes and scores (raw outputs too, not only the persons kept). Needs the fp32-topk
    package (bench/detect/export_gpu.py): the plain fp16 one scored the same frame differently from one pass to the next on the GPU."""
    monkeypatch.delenv(detect.UNITS_ENV, raising=False)
    det = detect.load_detector("small")
    assert det.backend.endswith("cpu_and_gpu") and det.gpu_repeatable
    frames = _synthetic(16)
    once = lambda: sum((det.detect(frames[i:i + detect.BATCH]) for i in range(0, 16, detect.BATCH)), [])
    first, second = once(), once()
    assert [[(d.score, d.box) for d in f] for f in first] == [[(d.score, d.box) for d in f] for f in second]
    xs = [detect.preprocess(frames[i:i + detect.BATCH], det.res) for i in (0, detect.BATCH)]
    for x in xs:
        (b1, l1), (b2, l2) = det._run(x), det._run(x)
        assert np.array_equal(b1, b2) and np.array_equal(l1, l2)


@needs_models
@pytest.mark.parametrize("units", ["CPU_AND_NE", "CPU_ONLY"])
def test_deterministic_detect_is_batch_independent(monkeypatch, units):
    """The same frame gives the same scores whatever else is in its batch and at which position."""
    monkeypatch.setenv(detect.UNITS_ENV, units)
    det = detect.CoreMLDetector("small")
    assert det.backend.endswith(det.units.lower())
    frames = _synthetic(detect.BATCH + 3)
    a = det.detect(frames[:detect.BATCH])
    b = det.detect(frames[3:3 + detect.BATCH])  # batches shifted by three
    c = det.detect(frames[3:4])  # the same frame alone (padded to a full batch)
    for i in range(3, detect.BATCH):
        assert [(d.score, d.box) for d in a[i]] == [(d.score, d.box) for d in b[i - 3]]
    assert [(d.score, d.box) for d in a[3]] == [(d.score, d.box) for d in c[0]]
    xa, xb = detect.preprocess(frames[:detect.BATCH], det.res), detect.preprocess(frames[3:3 + detect.BATCH], det.res)
    (ba, la), (bb, lb) = det._run(xa), det._run(xb)  # raw outputs, not only the kept persons
    assert np.array_equal(la[3], lb[0]) and np.array_equal(ba[3], bb[0])


# ---------------------------------------------------------------- a detection that shows up out of an empty sample
def _run_scored(target, plan):
    """plan: one shot, a list of per-sample lists [(box, track_id, score)]. Colours do not matter here (same for all)."""
    from inframe.detect import Detection

    for k, cands in enumerate(plan):
        frame = np.full((360, 640, 3), 128, np.uint8)
        target.evaluate(round(0.2 * k, 1), frame, [Detection(sc, box, tid) for box, tid, sc in cands])
    return target.finalize()


def test_bridge_joins_a_weak_untracked_detection_after_a_hole():
    """The target leaves at the frame edge, nothing is detected for two samples, then an untracked 0.54 detection of the target's size
    shows up 0.2 of the frame away. Four samples after the last tracked hit it is the target (centre distance, no overlap)."""
    moved = (0.6, 0.2, 0.8, 0.8)  # IoU 0 with BOX, centres 0.2 apart
    assert tap.iou(BOX, moved) == 0 and tap._inside(moved, BOX) == 0
    plan = [[(BOX, 1, 0.9)]] * 6 + [[]] * 3 + [[(moved, None, 0.54)]] * 3
    hits = _run_scored(TAP(), plan)
    assert on(hits) == [True] * 6 + [False] * 3 + [True] * 3
    assert [h.box for h in hits[9:]] == [moved] * 3
    # and the same in front of a tracked stretch (the backward pass)
    plan = [[(moved, None, 0.54)]] * 3 + [[]] * 3 + [[(BOX, 1, 0.9)]] * 6
    assert on(_run_scored(TAP(2.2), plan)) == [True] * 3 + [False] * 3 + [True] * 6


def test_bridge_near_rule_has_limits():
    moved = (0.6, 0.2, 0.8, 0.8)
    far = (0.75, 0.2, 0.95, 0.8)  # centres 0.35 apart: further than a walker goes
    tiny = (0.65, 0.5, 0.67, 0.55)  # a fifth of his height
    for box, score, hole in ((far, 0.54, 3), (tiny, 0.54, 3), (moved, 0.25, 3), (moved, 0.54, 6), (moved, 0.54, 0)):
        plan = [[(BOX, 1, 0.9)]] * 6 + [[]] * hole + [[(box, None, score)]] * 3
        assert sum(on(_run_scored(TAP(), plan))[6:]) == 0, (box, score, hole)
    # across a cut it never joins
    from inframe.detect import Detection

    t = TAP()
    _run_scored(t, [[(BOX, 1, 0.9)]] * 6)
    t.on_cut()
    for k in range(3):
        t.evaluate(1.2 + 0.2 * k, np.full((360, 640, 3), 128, np.uint8), [Detection(0.54, moved, None)])
    assert on(t.finalize())[6:] == [False] * 3


@needs_models
def test_an_idle_cpu_only_detector_does_not_crash_the_process():
    """A live CPU_ONLY CoreML model that gets no predict for about a second kills the process with SIGSEGV (macOS 26, coremltools 9). The
    detector drops its model after IDLE_FREE_S and reloads it on the next call, so photo mode may pause for seconds (embedder start-up)."""
    import subprocess
    import sys

    code = ("import numpy as np, time; from inframe import detect; d = detect.CoreMLDetector('small'); x = [np.zeros((90, 160, 3), np.uint8)] * 2; "
            "a = d.detect(x); time.sleep(2.5); b = d.detect(x); assert [[x.score for x in f] for f in a] == [[x.score for x in f] for f in b]; print('alive')")
    env = dict(os.environ, **{detect.UNITS_ENV: "CPU_ONLY"})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "alive" in r.stdout, (r.returncode, r.stderr[-300:])




# ---------------------------------------------------------------- tools/splice.py: tile_at and the his-tile ceiling
sys_path_tools = str(Path(__file__).resolve().parents[1] / "tools")
if sys_path_tools not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_tools)
import splice  # noqa: E402


def synth_truth():
    left, right = [0, 0, 640, 720], [640, 0, 640, 720]
    return dict(name="t", width=1280, height=720, duration=10.0, target_intervals=[[0.0, 5.0]],
                segments=[dict(t0=0.0, t1=5.0, tiles=[dict(rect=left, role="target")]),
                          dict(t0=5.0, t1=10.0, tiles=[dict(rect=right, role="other")])])


def test_tile_at_shot_edge_belongs_to_the_next_shot():
    tr = synth_truth()
    assert splice.tile_at(tr, 4.8, 0.25, 0.5)[0] == 0
    si, tl = splice.tile_at(tr, 5.0, 0.75, 0.5)  # the first sample of the second shot, not the last of the first
    assert si == 1 and tl["role"] == "other"
    assert splice.tile_at(tr, 5.0, 0.25, 0.5) == (1, None)
    assert splice.tile_at(tr, 9.8, 0.75, 0.5)[0] == 1 and splice.tile_at(tr, 10.0, 0.75, 0.5)[0] == 1  # the very end still maps


class _S:
    def __init__(self, t, cands):
        self.t, self.cands = t, cands


def test_target_tile_ceiling_counts_only_detections_in_target_tile():
    from inframe.detect import Detection

    tr = synth_truth()
    his, stranger = (0.1, 0.2, 0.3, 0.8), (0.6, 0.2, 0.8, 0.8)
    samples = [_S(round(k * 0.2, 1), [Detection(0.9, his)] if (k < 25 and k % 5 != 4) else [Detection(0.9, stranger)]) for k in range(50)]
    spans = [[0.0, 5.0]]
    hc = splice.target_tile_ceiling(tr, samples, spans)
    assert hc["recall"] > 0.9 and hc["precision"] == 1.0
    none = splice.target_tile_ceiling(tr, [_S(s.t, [Detection(0.9, stranger)]) for s in samples], spans)
    assert none["recall"] == 0.0


# ---------------------------------------------------------------- sliced detection (small people)
class BlockDetector:
    """Fake detector: boxes the bright block in the frame, but only sees it when it is at least 30 % of the frame high
    (like a real one, which loses people that are a few pixels tall)."""
    backend = "fake"
    path = "fake.mlpackage"

    def detect(self, frames):
        out = []
        for f in frames:
            h, w = f.shape[:2]
            on = f[..., 0] > 240
            cols = np.nonzero(on.any(0))[0]
            dets = []
            for run in np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1) if len(cols) else []:  # blocks that do not share columns
                ys = np.nonzero(on[:, run[0]:run[-1] + 1].any(1))[0]
                box = (run[0] / w, ys.min() / h, (run[-1] + 1) / w, (ys.max() + 1) / h)
                if box[3] - box[1] >= 0.3:
                    dets.append(detect.Detection(0.9, box))
            out.append(dets)
        return out


def blocks(*boxes, h=360, w=640):
    f = np.zeros((h, w, 3), np.uint8)
    for x0, y0, x1, y1 in boxes:
        f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    return f


def test_slice_windows_cover_the_frame_with_overlap():
    wins = detect.slice_windows(360, 640, 2, 2, 0.2)
    assert len(wins) == 4
    assert min(w[0] for w in wins) == 0 and max(w[1] for w in wins) == 360 and min(w[2] for w in wins) == 0 and max(w[3] for w in wins) == 640
    assert wins[0][3] > 320 and wins[1][2] < 320  # the left and right slices share a strip
    assert detect.slice_windows(360, 640, 1, 1) == [(0, 360, 0, 640)]


def test_sliced_detector_finds_a_small_person_the_full_frame_misses():
    small = (0.30, 0.10, 0.34, 0.30)  # 20 % of the frame high: below the fake detector's 30 %, above it in a slice
    f = blocks(small)
    assert BlockDetector().detect([f]) == [[]]
    sd = detect.SlicedDetector(BlockDetector(), (2, 2))
    (dets,) = sd.detect([f])
    assert len(dets) == 1
    assert all(abs(a - b) < 0.01 for a, b in zip(dets[0].box, small))
    assert sd.backend == "fake+slices2x2" and sd.path == "fake.mlpackage"


def test_sliced_detector_does_not_double_count_or_split_a_big_person():
    big = (0.30, 0.15, 0.55, 0.95)  # seen whole in the full frame, and cut by a slice edge in some slices
    (dets,) = detect.SlicedDetector(BlockDetector(), (2, 2)).detect([blocks(big)])
    assert len(dets) == 1 and all(abs(a - b) < 0.01 for a, b in zip(dets[0].box, big))
    two = [(0.05, 0.1, 0.09, 0.30), (0.80, 0.55, 0.84, 0.75)]  # two small people in different slices
    (dets,) = detect.SlicedDetector(BlockDetector(), (2, 2)).detect([blocks(*two)])
    assert len(dets) == 2


def test_slice_grid_env(monkeypatch):
    monkeypatch.delenv(detect.SLICES_ENV, raising=False)
    assert detect.slice_grid() == (1, 1)  # off unless asked for: see README (precision cost on the composites)
    for v, g in (("off", (1, 1)), ("2x2", (2, 2)), ("3x2", (3, 2)), ("garbage", (1, 1)), ("9x9", (1, 1))):
        monkeypatch.setenv(detect.SLICES_ENV, v)
        assert detect.slice_grid() == g
    monkeypatch.setenv(detect.SLICES_ENV, "off")
    base = BlockDetector()
    assert detect.for_tap(base) is base
    monkeypatch.setenv(detect.SLICES_ENV, "2x2")
    assert isinstance(detect.for_tap(base), detect.SlicedDetector)
    assert TapTarget([]).wants_slices and not getattr(__import__("inframe.target", fromlist=["AnyPerson"]).AnyPerson(), "wants_slices", False)



# ---------------------------------------------------------------- the near rule needs a look that does not contradict him
def test_near_rule_does_not_take_a_stranger_in_other_clothes():
    """He (track 1, red over blue) leaves; one sample is empty; then an untracked stranger in green, 0.2 of the frame away with no
    overlap, stays for 15 samples. `_near` alone would take him and the chain would follow by overlap up to the cap."""
    moved = (0.6, 0.2, 0.8, 0.8)
    assert tap.iou(BOX, moved) == 0 and tap._near(BOX, moved)
    shot = [[(BOX, 1, RED, BLUE)]] * 6 + [[]] + [[(moved, None, GREEN, GREEN)]] * 15
    assert sum(on(run(TAP(), [shot]))) == 6
    # the same in front of his track (the backward pass)
    shot = [[(moved, None, GREEN, GREEN)]] * 15 + [[]] + [[(BOX, 1, RED, BLUE)]] * 6
    assert sum(on(run(TAP(3.4), [shot]))) == 6
    # the same box in his clothes is still bridged
    shot = [[(BOX, 1, RED, BLUE)]] * 6 + [[]] + [[(moved, None, RED, BLUE)]] * 3
    assert on(run(TAP(), [shot])) == [True] * 6 + [False] + [True] * 3


def _scaled(c, k):
    return tuple(int(v * k) for v in c)


def test_near_rule_look_survives_a_darker_frame():
    """He steps into shade during the one-sample hole: his clothes read 20 % darker. The colour histogram splits dark and light at
    V = 0.5, so x0.8 of a mid-bright red moved its whole mass into other bins (similarity 0.05) and the box that only the near
    rule can take was refused. The look gate must tolerate a change of light and still refuse other clothes."""
    red, blue = (150, 30, 30), (30, 30, 150)
    a = tap.crop_signature(frame_of(red, blue), BOX)
    dark = tap.crop_signature(frame_of(_scaled(red, 0.8), _scaled(blue, 0.8)), BOX)
    assert tap._not_contradicting(dark, [a])
    assert not tap._not_contradicting(tap.crop_signature(frame_of(GREEN, GREEN), BOX), [a])
    moved = (0.6, 0.2, 0.8, 0.8)
    assert tap.iou(BOX, moved) == 0 and tap._near(BOX, moved)
    shot = [[(BOX, 1, red, blue)]] * 6 + [[]] + [[(moved, None, _scaled(red, 0.8), _scaled(blue, 0.8))]] * 3
    assert on(run(TAP(), [shot])) == [True] * 6 + [False] + [True] * 3


def test_near_rule_compares_with_the_box_just_before_the_hole():
    """His track mean (the gallery) is five samples in white and one in the coat and trousers he is in at the end of the track; the
    box that comes back after the hole is in those. Against the mean it looks like a stranger, against the box he was in just before
    the hole it is him."""
    white, coat, trousers = (230, 230, 230), (30, 30, 200), (200, 30, 30)
    moved = (0.6, 0.2, 0.8, 0.8)
    shot = [[(BOX, 1, white, white)]] * 5 + [[(BOX, 1, coat, trousers)]] + [[]] + [[(moved, None, coat, trousers)]] * 3
    mean = tap.crop_signature(frame_of(white, white), BOX)
    assert tap._not_contradicting(tap.crop_signature(frame_of(coat, trousers, box=moved), moved), [mean]) is False
    assert on(run(TAP(), [shot])) == [True] * 6 + [False] + [True] * 3


# ---------------------------------------------------------------- the idle guard must not reload between close batches
class _FakeModel:
    def __init__(self, counter):
        counter.append(1)

    def get_spec(self):
        from types import SimpleNamespace
        return SimpleNamespace(description=SimpleNamespace(input=[SimpleNamespace(name="x")]))

    def predict(self, d):
        return {"boxes": np.zeros((detect.BATCH, 5, 4), np.float32), "logits": np.zeros((detect.BATCH, 5, 91), np.float32)}


def test_idle_guard_margin_against_a_gil_stall():
    """The free runs on a timer thread that needs the GIL; the crash starts at an idle of 1.0 s. A stall that holds the GIL (a long
    gc.collect(), a big sort) has to cover IDLE_FREE_S..1.0 s to crash, so IDLE_FREE_S = 0.75 (a 0.25 s stall: SIGSEGV, measured) is
    too late: the margin must be at least half a second."""
    assert 1.0 - detect.IDLE_FREE_S >= 0.5


def test_idle_guard_keeps_the_model_between_batches_close_together(monkeypatch):
    """Batches less than IDLE_FREE_S apart reload nothing; a pause longer than that (the crash starts at 1.0 s) frees the model."""
    import time

    loads = []
    monkeypatch.setenv(detect.UNITS_ENV, "CPU_ONLY")
    monkeypatch.setattr(detect, "find_package", lambda size: "fake.mlpackage")
    monkeypatch.setattr(detect.CoreMLDetector, "_load", lambda self: _FakeModel(loads))
    d = detect.CoreMLDetector("small")
    frames = [np.zeros((90, 160, 3), np.uint8)] * detect.BATCH
    d.detect(frames)
    for gap in (0.1, 0.2, detect.IDLE_FREE_S - 0.15):
        time.sleep(gap)
        d.detect(frames)
    assert len(loads) == 1, f"reloaded {len(loads) - 1} times"
    assert detect.IDLE_FREE_S < 0.9  # the crash comes at an idle of 1.0 s
    time.sleep(detect.IDLE_FREE_S + 0.3)
    assert d.model is None  # a longer pause frees it
    d.detect(frames)
    assert len(loads) == 2
    d._timer.cancel()
