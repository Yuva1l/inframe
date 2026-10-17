"""Golden vectors for the segment logic, the detector's pre- and postprocess, the identity-free targets and the tap gallery with the crop
geometry of the embedder, shared by pytest and vitest (docs/T9B_DESIGN.md sections 4 and 5).

`src/inframe/segments.py` is the source of truth. This script runs it on hand-made and seeded random inputs and writes the inputs
together with every intermediate result to `tests/golden/segments_v1.json`. The TypeScript port (`web/src/model/segments.ts`) must
reproduce each number; `tests/test_golden.py` fails when the committed file no longer matches `segments.py`.
The same pattern holds `tests/golden/detect_v1.json` (detect.preprocess / postprocess, TypeScript: web/src/browser/detect.ts) and
`tests/golden/targets_v1.json` (AnyPerson, AnyOfClass, ZoneTarget, the COCO table and class_id, TypeScript: web/src/browser/targets.ts) and
`tests/golden/gallery_v1.json` (identity.Gallery add/dedup/evict, identity.unit_score, identity.person_crop geometry, embed_worker.pad_square and
the DINOv2 preprocessing, TypeScript: web/src/browser/gallery.ts and embed.ts) and
`tests/golden/resample_v2.json` (PyAV's swscale YUV to RGB at the decode size, decode.upright and PIL's BILINEAR squash to the detector's square on
synthetic NV12 and I420 pictures, TypeScript: web/src/browser/resample.ts).

    uv run python tools/gen_golden.py            # (re)write the file
    uv run python tools/gen_golden.py --check    # exit 0 if the committed file is current, 1 if it drifted

Inputs come from the stdlib `random` with fixed seeds (numpy only rounds the logits to float32 and runs detect.py). Floats are written with
repr(), which round-trips exactly in JSON.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from PIL import Image

from inframe import classes
from inframe.detect import Detection, postprocess, preprocess
from inframe.embed_worker import pad_square
from inframe.identity import SPECS, Gallery, person_crop, unit_score
from inframe.segments import build_segments, drop_short, hysteresis, merge_gaps, pad, sample_edges
from inframe.target import AnyOfClass, AnyPerson, Hit
from inframe.zone import Zone, ZoneTarget

OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "segments_v1.json"
DETECT_OUT = OUT.with_name("detect_v1.json")
TARGETS_OUT = OUT.with_name("targets_v1.json")
GALLERY_OUT = OUT.with_name("gallery_v1.json")
RESAMPLE_OUT = OUT.with_name("resample_v2.json")
VERSION = 1
N_RANDOM = 45


def _pairs(xs):
    return [[float(a), float(b)] for a, b in xs]


def case(name, kind, times, scores, duration, enter=0.5, exit=0.3, merge_gap=1.5, min_len=1.0, pad_s=1.0, lo=0.0):
    """One input set with the result of every stage. A ValueError from hysteresis becomes an `error` instead of `expect`."""
    c = {
        "name": name, "kind": kind, "times": [float(t) for t in times], "scores": [float(s) for s in scores], "duration": float(duration),
        "enter": float(enter), "exit": float(exit), "merge_gap": float(merge_gap), "min_len": float(min_len), "pad": float(pad_s), "lo": float(lo),
    }
    try:
        hyst = hysteresis(c["times"], c["scores"], c["enter"], c["exit"], c["duration"], c["lo"])
    except ValueError as e:
        c["error"] = "length_mismatch" if "length" in str(e) else "exit_gt_enter"
        return c
    merged = merge_gaps(hyst, c["merge_gap"])
    kept = drop_short(merged, c["min_len"])
    presence, clips = build_segments(c["times"], c["scores"], c["duration"], c["enter"], c["exit"], c["merge_gap"], c["min_len"], c["pad"], c["lo"])
    assert presence == kept and clips == pad(kept, c["pad"], c["pad"], c["duration"], c["lo"])
    c["expect"] = {
        "edges": [float(x) for x in sample_edges(c["times"], c["duration"], c["lo"])],
        "hysteresis": _pairs(hyst), "merged": _pairs(merged), "kept": _pairs(kept), "presence": _pairs(presence), "clips": _pairs(clips),
    }
    return c


def _ramp(n, step=0.5, start=0.0):
    return [start + i * step for i in range(n)]


def hand_cases():
    """Numbers are multiples of 0.25 where an equality matters, so the doubles are exact."""
    H = "hand"
    t10 = _ramp(10, 1.0)
    out = [
        case("empty", H, [], [], 5.0),
        case("empty_with_lo", H, [], [], 5.0, lo=1.0),
        case("one_sample_above", H, [2.0], [0.9], 5.0),
        case("one_sample_below", H, [2.0], [0.1], 5.0),
        case("one_sample_clamped_to_lo", H, [0.25], [0.9], 5.0, lo=0.0, min_len=0.5),
        case("one_sample_clamped_to_duration", H, [4.75], [0.9], 5.0, min_len=0.5),
        case("two_samples", H, [0.0, 1.0], [0.9, 0.9], 1.5),
        case("all_above", H, t10, [0.9] * 10, 9.5),
        case("all_below", H, t10, [0.1] * 10, 9.5),
        case("exit_equals_enter_flip", H, t10, [0.9, 0.9, 0.2, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.9], 9.5, enter=0.5, exit=0.5, merge_gap=0.5, min_len=0.5),
        case("exit_equals_enter_score_on_threshold", H, t10, [0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5], 9.5, enter=0.5, exit=0.5, merge_gap=0.0, min_len=0.0, pad_s=0.0),
        case("hysteresis_holds_between_exit_and_enter", H, _ramp(6, 1.0), [0.0, 0.6, 0.4, 0.4, 0.6, 0.0], 5.5, min_len=0.0),
        case("hysteresis_splits_below_exit", H, _ramp(6, 1.0), [0.0, 0.6, 0.2, 0.4, 0.6, 0.0], 5.5, merge_gap=0.5, min_len=0.0),
        case("lo_positive", H, _ramp(8, 0.5, 2.0), [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9], 5.5, lo=2.0, merge_gap=0.5, min_len=0.5, pad_s=1.0),
        case("lo_positive_pad_clamps_to_lo", H, _ramp(8, 0.5, 2.0), [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.5, lo=2.0, merge_gap=0.5, min_len=0.5, pad_s=1.5),
        case("lo_inside_first_cell", H, [2.25, 2.75, 3.25], [0.9, 0.9, 0.0], 4.0, lo=2.0, merge_gap=0.0, min_len=0.0, pad_s=0.5),
        case("lo_above_first_edge", H, [0.0, 1.0, 2.0], [0.9, 0.9, 0.0], 3.0, lo=0.5, min_len=0.0, pad_s=0.0),
        case("clamp_at_duration", H, _ramp(5, 1.0), [0.0, 0.0, 0.9, 0.9, 0.9], 4.25, min_len=0.0, pad_s=2.0),
        case("duration_before_last_sample", H, _ramp(5, 1.0), [0.9, 0.0, 0.0, 0.9, 0.9], 3.5, min_len=0.0, pad_s=0.0),
        case("duplicate_times", H, [0.0, 1.0, 1.0, 2.0], [0.0, 0.9, 0.9, 0.0], 2.5, min_len=0.0, pad_s=0.0),
        case("gap_equals_merge_gap", H, _ramp(7, 0.5), [0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9], 3.25, merge_gap=1.5, min_len=0.0, pad_s=0.0),
        case("gap_just_over_merge_gap", H, _ramp(7, 0.5), [0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9], 3.25, merge_gap=1.25, min_len=0.0, pad_s=0.0),
        case("gap_just_under_merge_gap", H, _ramp(7, 0.5), [0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9], 3.25, merge_gap=1.75, min_len=0.0, pad_s=0.0),
        case("merge_gap_zero_keeps_separate", H, _ramp(9, 0.5), [0.9, 0.9, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9], 4.25, merge_gap=0.0, min_len=0.0, pad_s=0.0),
        case("merge_gap_swallows_all", H, t10, [0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0], 9.5, merge_gap=100.0, min_len=0.0, pad_s=0.0),
        case("min_len_exactly_met", H, _ramp(6, 1.0), [0.0, 0.9, 0.9, 0.0, 0.0, 0.0], 5.5, merge_gap=0.0, min_len=2.0, pad_s=0.0),
        case("min_len_just_missed", H, _ramp(6, 1.0), [0.0, 0.9, 0.9, 0.0, 0.0, 0.0], 5.5, merge_gap=0.0, min_len=2.25, pad_s=0.0),
        case("pad_merges_neighbours", H, _ramp(10, 1.0), [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], 9.5, merge_gap=0.0, min_len=0.5, pad_s=2.5),
        case("pad_touching_exactly_merges", H, _ramp(10, 1.0), [0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], 9.5, merge_gap=0.0, min_len=0.5, pad_s=1.5),
        case("pad_leaves_a_gap", H, _ramp(10, 1.0), [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], 9.5, merge_gap=0.0, min_len=0.5, pad_s=0.5),
        case("pad_zero", H, _ramp(6, 1.0), [0.0, 0.9, 0.9, 0.0, 0.9, 0.0], 5.5, merge_gap=0.0, min_len=0.5, pad_s=0.0),
        case("pad_covers_everything", H, _ramp(6, 1.0), [0.0, 0.0, 0.9, 0.9, 0.0, 0.0], 5.5, lo=0.0, merge_gap=0.0, min_len=0.5, pad_s=50.0),
        case("scores_outside_unit_range", H, _ramp(6, 1.0), [-1.0, 3.0, 2.0, -0.5, 7.0, -2.0], 5.5, merge_gap=0.5, min_len=0.0, pad_s=0.25),
        case("error_exit_above_enter", H, _ramp(4, 1.0), [0.9, 0.9, 0.9, 0.9], 4.0, enter=0.3, exit=0.5),
        case("error_length_mismatch", H, _ramp(4, 1.0), [0.9, 0.9, 0.9], 4.0),
    ]
    return out


def random_case(i):
    rng = random.Random(7000 + i)
    n = rng.choice([0, 1, 2, 5, 13, 20, 30, 40, 60, 80, 100, 150])
    rate = rng.choice([1, 2, 5, 10, 24, 30])
    lo = rng.choice([0.0, 0.0, 0.0, 0.04, 0.5, 2.0])
    jitter = rng.choice([0.0, 0.0, 0.2])
    times = [lo + k / rate + jitter * rng.random() / rate for k in range(n)]
    if n:
        tail = rng.choice([0.0, 0.1, 0.5, 1 / rate, 3.0])
        duration = times[-1] + tail
        if rng.random() < 0.15:
            duration = times[-1] - rng.random() * 2 / rate  # the file ends before the last sample's cell
    else:
        duration = rng.uniform(1, 10)
    scores, high, stay = [], rng.random() < 0.5, rng.choice([0.7, 0.85, 0.95])
    for _ in range(n):
        if rng.random() > stay:
            high = not high
        r = rng.random()
        scores.append(rng.uniform(0.5, 1.0) if high and r < 0.9 else rng.uniform(0.3, 0.5) if r < 0.95 else rng.uniform(0.0, 0.3))
    enter = rng.choice([0.4, 0.5, 0.6, 0.7])
    exit_ = min(enter, rng.choice([enter, 0.3, enter - 0.1, 0.2]))
    return case(f"random_{i:02d}", "random", times, scores, duration, enter, exit_, rng.choice([0.0, 0.5, 1.0, 1.5, 3.0]),
                rng.choice([0.0, 0.5, 1.0, 2.0]), rng.choice([0.0, 0.25, 1.0, 2.5]), lo)


def resegment_case(name, t, score, settings, params, offset, end):
    """The serve_runs.resegment fill-in rule: a missing or null field comes from the run's settings; none left is an error."""
    c = {"name": name, "t": [float(x) for x in t], "score": [float(x) for x in score], "settings": settings, "params": params,
         "offset": float(offset), "end": float(end)}
    vals = {}
    for k in ("enter", "exit", "merge_gap", "min_len", "pad"):
        v = params.get(k)
        v = settings.get(k) if v is None else v
        if v is None:
            c["error"] = "no_value"
            return c
        vals[k] = float(v)
    if vals["exit"] > vals["enter"]:
        c["error"] = "exit_gt_enter"
        return c
    presence, clips = build_segments(c["t"], c["score"], c["end"], vals["enter"], vals["exit"], vals["merge_gap"], vals["min_len"], vals["pad"], c["offset"])
    c["expect"] = {"presence": _pairs(presence), "clips": _pairs(clips)}
    return c


def resegment_cases():
    t = _ramp(16, 0.5, 1.0)
    s = [0.0, 0.9, 0.9, 0.2, 0.0, 0.0, 0.9, 0.45, 0.4, 0.9, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]
    full = {"enter": 0.5, "exit": 0.3, "merge_gap": 1.0, "min_len": 1.0, "pad": 0.25}
    return [
        resegment_case("settings_only", t, s, full, {}, 1.0, 8.75),
        resegment_case("all_null_params", t, s, full, {k: None for k in full}, 1.0, 8.75),
        resegment_case("one_override", t, s, full, {"min_len": 0.25}, 1.0, 8.75),
        resegment_case("override_enter_exit", t, s, full, {"enter": 0.35, "exit": 0.35}, 1.0, 8.75),
        resegment_case("zero_is_a_value", t, s, full, {"pad": 0.0, "merge_gap": 0.0}, 1.0, 8.75),
        resegment_case("run_settings_missing_pad", t, s, {k: v for k, v in full.items() if k != "pad"}, {"pad": 0.5}, 1.0, 8.75),
        resegment_case("empty_samples", [], [], full, {}, 0.0, 5.0),
        resegment_case("error_no_value", t, s, {k: v for k, v in full.items() if k != "pad"}, {}, 1.0, 8.75),
        resegment_case("error_null_in_settings", t, s, {**full, "exit": None}, {}, 1.0, 8.75),
        resegment_case("error_exit_above_enter", t, s, full, {"exit": 0.8}, 1.0, 8.75),
    ]


def document():
    cases = hand_cases() + [random_case(i) for i in range(N_RANDOM)]
    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names)
    return {"version": VERSION, "generator": "tools/gen_golden.py", "source": "src/inframe/segments.py", "cases": cases, "resegment": resegment_cases()}


def render(doc):
    """One case per line, so a drift diff stays readable."""
    def line(c):
        return "    " + json.dumps(c, separators=(",", ":"), allow_nan=False)
    head = {k: v for k, v in doc.items() if k not in ("cases", "resegment")}
    parts = [json.dumps(head, separators=(",", ":"))[:-1]]
    parts.append(',"cases":[\n' + ",\n".join(line(c) for c in doc["cases"]) + "\n  ]")
    parts.append(',"resegment":[\n' + ",\n".join(line(c) for c in doc["resegment"]) + "\n  ]}\n")
    return "".join(parts)

# ---------------------------------------------------------------- detect_v1: preprocess and postprocess
def _f32(xs):
    """Values rounded to float32, as Python floats (exact in JSON): what the model's float32 outputs hold."""
    return [float(v) for v in np.asarray(xs, np.float32).ravel()]


def preprocess_case(name, side, seed):
    rng = random.Random(seed)
    rgba = [rng.randrange(256) for _ in range(side * side * 4)]
    img = np.array(rgba, np.uint8).reshape(side, side, 4)
    out = preprocess([np.ascontiguousarray(img[:, :, :3])], side)  # same size: the bilinear resize is the identity
    return {"name": name, "side": side, "rgba": rgba, "tensor": _f32(out[0])}


def postprocess_case(name, seed, queries=12, nclass=91, cls=1, min_score=0.2, spread=3.0, ties=False, boundary=False):
    rng = random.Random(seed)
    logits = [[rng.gauss(-2.0, spread) for _ in range(nclass)] for _ in range(queries)]
    if ties:  # equal scores keep their query order
        logits[5] = list(logits[2])
        logits[7] = list(logits[2])
        for q in (2, 5, 7):
            logits[q][cls] = 1.25
    if boundary:  # a logit whose sigmoid is just above and just under the floor
        eps = 1e-3
        logits[1][cls] = float(np.log(min_score / (1 - min_score))) + eps
        logits[3][cls] = float(np.log(min_score / (1 - min_score))) - eps
    boxes = [[rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), rng.uniform(0.02, 0.6), rng.uniform(0.02, 0.8)] for _ in range(queries)]
    lg, bx = np.array(logits, np.float32)[None], np.array(boxes, np.float32)[None]
    dets = postprocess(bx, lg, 1, min_score=min_score, cls=cls)[0]
    return {"name": name, "queries": queries, "classes": nclass, "cls": cls, "min_score": min_score, "logits": _f32(lg), "boxes": _f32(bx),
            "expect": [[d.score, *d.box] for d in dets]}


def detect_document():
    pre = [preprocess_case("noise_2", 2, 1), preprocess_case("noise_4", 4, 2), preprocess_case("noise_8", 8, 3)]
    pre.append({"name": "extremes", "side": 2, "rgba": [0, 0, 0, 255, 255, 255, 255, 0, 128, 64, 32, 255, 1, 254, 127, 7],
                "tensor": _f32(preprocess([np.array([[[0, 0, 0], [255, 255, 255]], [[128, 64, 32], [1, 254, 127]]], np.uint8)], 2)[0])})
    post = [
        postprocess_case("person_default", 10),
        postprocess_case("person_high_floor", 11, min_score=0.5),
        postprocess_case("dog", 12, cls=18),
        postprocess_case("sports_ball_low_floor", 13, cls=37, min_score=0.05, spread=2.0),
        postprocess_case("ties_keep_query_order", 14, cls=1, ties=True),
        postprocess_case("boundary", 15, cls=1, boundary=True),
        postprocess_case("nothing_passes", 16, spread=0.5, min_score=0.99),
        postprocess_case("all_pass", 17, cls=2, spread=0.5, min_score=0.0),
        postprocess_case("last_class_slot", 18, cls=90),
        postprocess_case("few_queries", 19, queries=3),
    ]
    assert any(c["expect"] for c in post) and any(not c["expect"] for c in post)
    bad = postprocess_case("class_slot_zero", 20, cls=0)  # a valid slot (0 is unused in COCO but exists in the 91)
    post.append(bad)
    err = {"name": "error_class_outside_the_slots", "queries": 2, "classes": 91, "cls": 91, "min_score": 0.2, "logits": [0.0] * (2 * 91), "boxes": [0.5] * 8, "error": "class_outside"}
    try:
        postprocess(np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 91), np.float32), 1, cls=91)
        raise AssertionError("postprocess accepted class 91")
    except ValueError:
        pass
    post.append(err)
    return {"version": VERSION, "generator": "tools/gen_golden.py", "source": "src/inframe/detect.py", "preprocess": pre, "postprocess": post}


# ---------------------------------------------------------------- targets_v1: AnyPerson, AnyOfClass, ZoneTarget, the COCO table
class First:
    """A target that is not stateless (it answers with the first candidate, zone or not): ZoneTarget must gate its answer instead."""
    name = "first"
    cls = 1
    wants_tracks = False
    stateless = False

    def evaluate(self, t, frame, candidates):
        return Hit(candidates[0].score, candidates[0].box) if candidates else Hit(0.0, None)


def _hit(h):
    return {"score": h.score, "box": list(h.box) if h.box is not None else None}


def _cands(rng, n):
    out = []
    for _ in range(n):
        x0, y0 = rng.uniform(0, 0.9), rng.uniform(0, 0.9)
        out.append([round(rng.uniform(0.2, 0.99), 4), round(x0, 4), round(y0, 4), round(min(1.0, x0 + rng.uniform(0.02, 0.5)), 4), round(min(1.0, y0 + rng.uniform(0.02, 0.7)), 4)])
    return sorted(out, key=lambda c: -c[0])  # the detector hands them over best first


def target_case(name, kind, steps, cls=None, min_box=0.0, zone=None, zone_overlap=None, seed=0):
    rng = random.Random(seed)
    inner = {"any": lambda: AnyPerson(min_box), "class": lambda: AnyOfClass(cls, min_box), "first": First}[kind]()
    t = ZoneTarget(inner, Zone(*zone), zone_overlap) if zone else inner
    seq = [_cands(rng, n) for n in steps]
    res = [_hit(t.evaluate(float(i), None, [Detection(c[0], tuple(c[1:])) for c in cand])) for i, cand in enumerate(seq)]
    return {"name": name, "kind": kind, "cls": cls, "min_box": min_box, "zone": zone, "zone_overlap": zone_overlap, "steps": seq,
            "expect": res, "target_name": t.name}


def targets_document():
    cases = [
        target_case("any_empty_and_some", "any", [0, 1, 3, 6], seed=1),
        target_case("any_min_box", "any", [4, 4, 4, 4, 4], min_box=0.3, seed=2),
        target_case("any_min_box_none_tall_enough", "any", [3, 3], min_box=0.99, seed=3),
        target_case("class_dog", "class", [0, 2, 5, 3], cls=18, seed=4),
        target_case("class_sports_ball_min_box", "class", [4, 4, 4], cls=37, min_box=0.1, seed=5),
        target_case("zone_centre_rule", "any", [5, 5, 5, 5, 5, 5], zone=[0.25, 0.25, 0.5, 0.5], seed=6),
        target_case("zone_overlap_half", "any", [5, 5, 5, 5, 5, 5], zone=[0.25, 0.25, 0.5, 0.5], zone_overlap=0.5, seed=7),
        target_case("zone_overlap_strict", "any", [5, 5, 5, 5], zone=[0.1, 0.1, 0.3, 0.3], zone_overlap=0.9, seed=8),
        target_case("zone_overlap_wide", "any", [5, 5, 5, 5], zone=[0.1, 0.1, 0.3, 0.3], zone_overlap=0.05, seed=9),
        target_case("zone_whole_frame", "any", [4, 4], zone=[0.0, 0.0, 1.0, 1.0], seed=10),
        target_case("zone_edge_touching", "any", [6, 6, 6], zone=[0.0, 0.5, 0.5, 0.5], seed=11),
        target_case("zone_class_and_min_box", "class", [6, 6, 6, 6], cls=1, min_box=0.2, zone=[0.2, 0.0, 0.6, 1.0], seed=12),
        target_case("zone_around_a_target_that_is_not_stateless", "first", [5, 5, 5, 5, 5], zone=[0.25, 0.25, 0.5, 0.5], seed=13),
        target_case("zone_overlap_around_a_target_that_is_not_stateless", "first", [5, 5, 5, 5, 5], zone=[0.25, 0.25, 0.5, 0.5], zone_overlap=0.6, seed=14),
    ]
    assert len({c["name"] for c in cases}) == len(cases)
    assert any(h["box"] is not None for c in cases for h in c["expect"]) and any(h["box"] is None for c in cases for h in c["expect"])
    names = ["person", "People", " DOG ", "sports_ball", "sports-ball", "ball", "bike", "phone", "mobile  phone", "sofa", "hair dryer", "Hair-Drier",
             "puppy", "table", "tv", "toothbrush", "dogg", "xyz", "", "pers", "teddy"]
    ids = []
    for n in names:
        try:
            ids.append({"name": n, "id": classes.class_id(n)})
        except classes.UnknownClass as e:
            ids.append({"name": n, "error": str(e)})
    zones = [{"zone": z, "valid": ok} for z, ok in [([0, 0, 1, 1], True), ([0.1, 0.2, 0.3, 0.4], True), ([0.5, 0.5, 0.5, 0.5], True), ([0.6, 0.0, 0.5, 1.0], False),
                                                    ([-0.1, 0, 0.5, 0.5], False), ([0, 0, 0, 0.5], False), ([0, 0, 0.5, 0], False), ([0.5, 0.5, 0.5000001, 0.5], True)]]
    for z in zones:
        try:
            Zone(*z["zone"])
            assert z["valid"], z
        except ValueError:
            assert not z["valid"], z
    return {"version": VERSION, "generator": "tools/gen_golden.py", "source": "src/inframe/target.py, zone.py, classes.py",
            "coco_91": {str(k): v for k, v in classes.COCO_91.items()}, "aliases": classes.ALIASES, "class_ids": ids, "zones": zones, "cases": cases}



# ---------------------------------------------------------------- gallery_v1: Gallery, unit_score, person_crop, pad_square, the DINOv2 input
def _unit(rng, dim, centre=None, noise=1.0):
    """A unit vector (float32 rows, as the embedder returns them): `centre` plus gaussian noise, or pure noise."""
    v = np.array([rng.gauss(0.0, 1.0) * noise + (centre[i] if centre is not None else 0.0) for i in range(dim)], np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _hist(rng, bins=8):
    """A colour signature as tap.crop_signature makes one: two bands, each a histogram that sums to 1."""
    out = []
    for _ in range(2):
        h = [rng.random() ** 3 for _ in range(bins // 2)]
        t = sum(h)
        out.extend(x / t for x in h)
    return out


def gallery_case(name, seed, n_adds, dim=12, cap=6, spec="dinov2", n_centres=3, noise=(0.05, 0.4, 1.0), with_sig=False, sig_share=1.0):
    """Seeded vectors around a few centres (so some are near-duplicates and dedup fires, some are far), added one by one to a Gallery of
    `cap` members. Records, after every add, whether it joined and which adds (by index) are in the gallery."""
    rng = random.Random(seed)
    centres = [_unit(rng, dim) for _ in range(n_centres)]
    adds = []
    for i in range(n_adds):
        c = centres[rng.randrange(n_centres)]
        v = _unit(rng, dim, c * 6.0, noise[rng.randrange(len(noise))] * rng.uniform(0.5, 1.5))
        sig = _hist(rng) if with_sig and rng.random() < sig_share else None
        adds.append({"vec": _f32(v), "sig": sig})
    sp = SPECS[spec]
    steps = []

    def run(dedup_shift=0.0):
        g = Gallery(sp.__class__(**{**sp.__dict__, "dedup": sp.dedup + dedup_shift}), cap)
        out = []
        for i, a in enumerate(adds):
            v = np.array(a["vec"], np.float32)
            sg = np.array(a["sig"]) if a["sig"] is not None else None
            ok = g.add(v, sg, tag=i)
            out.append({"joined": bool(ok), "tags": list(g.tags)})
        return g, out

    g, steps = run()
    for shift in (1e-5, -1e-5):  # no dedup decision sits within 1e-5 of the threshold: float32 and double sums cannot disagree on it
        assert run(shift)[1] == steps, f"{name}: a dedup decision is within 1e-5 of the threshold"
    return {"name": name, "dim": dim, "cap": cap, "spec": spec, "dedup": sp.dedup, "adds": adds, "steps": steps,
            "final": {"tags": list(g.tags), "vectors": [_f32(r) for r in g.v]}}


def unit_score_case(name, seed, n_vecs, n_gal, dim=12, mismatch=False):
    rng = random.Random(seed)
    vecs = [_unit(rng, dim) for _ in range(n_vecs)]
    gal = [_unit(rng, dim + (3 if mismatch else 0)) for _ in range(n_gal)]
    v = np.array(vecs, np.float32).reshape(n_vecs, dim) if n_vecs else np.zeros((0, dim), np.float32)
    g = np.array(gal, np.float32).reshape(n_gal, dim + (3 if mismatch else 0)) if n_gal else np.zeros((0, dim), np.float32)
    return {"name": name, "vecs": [_f32(r) for r in vecs], "gallery": [_f32(r) for r in gal], "expect": unit_score(v, g)}


def _lcg_bytes(seed, n):
    """The shared pixel generator: s = s * 1664525 + 1013904223 mod 2^32, the byte is the top 8 bits (web/test/golden-gallery.test.ts has the same)."""
    out, s = bytearray(n), seed & 0xFFFFFFFF
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = s >> 24
    return bytes(out)


def _sha(b):
    import hashlib

    return hashlib.sha256(bytes(b)).hexdigest()


def _tensor_of(chw):
    flat = [float(x) for x in np.asarray(chw, np.float32).ravel()]
    rng = random.Random(len(flat))
    idx = sorted(rng.sample(range(len(flat)), 96)) + [0, len(flat) - 1]
    return {"n": len(flat), "mean": float(np.asarray(chw, np.float64).mean()), "samples": [[i, flat[i]] for i in idx]}


MEAN32, STD32 = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def model_input(square):
    """embed_worker's DINOv2 path after pad_square: the processor of facebook/dinov2-small (PIL bicubic resize of the short edge to 256, centre
    crop 224, rescale by 1/255, ImageNet mean and std), written out so no network and no transformers are needed. tests/test_golden.py checks
    it against the real processor when that is installed."""
    h, w = square.shape[:2]
    nw, nh = (256, int(256 * h / w)) if w < h else (int(256 * w / h), 256) if w > h else (256, 256)
    im = Image.fromarray(square).resize((nw, nh), Image.BICUBIC)
    a = np.asarray(im)
    top, left = (nh - 224) // 2, (nw - 224) // 2
    c = a[top:top + 224, left:left + 224]
    x = (c.astype(np.float64) * (1 / 255)).astype(np.float32)
    x = (x - MEAN32) / STD32
    return c, np.ascontiguousarray(x.transpose(2, 0, 1), np.float32)


def resize_case(name, w, h, ow, oh, seed):
    img = np.frombuffer(_lcg_bytes(seed, w * h * 3), np.uint8).reshape(h, w, 3)
    out = np.asarray(Image.fromarray(img).resize((ow, oh), Image.BICUBIC))
    return {"name": name, "w": w, "h": h, "ow": ow, "oh": oh, "seed": seed, "sha256": _sha(out.tobytes())}


def pipeline_case(name, fw, fh, box, seed):
    frame = np.frombuffer(_lcg_bytes(seed, fw * fh * 3), np.uint8).reshape(fh, fw, 3)
    crop = person_crop(frame, tuple(box))
    if crop is None:
        return {"name": name, "fw": fw, "fh": fh, "box": box, "seed": seed, "rect": None}
    # find the rect from the crop's position in the frame: person_crop is a view-copy of frame[b:d, a:c]
    h, w = crop.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * 0.04, (y1 - y0) * 0.04
    a, b = max(int((x0 - px) * fw), 0), max(int((y0 - py) * fh), 0)
    assert (frame[b:b + h, a:a + w] == crop).all()
    sq = pad_square(crop)
    c, t = model_input(sq)
    return {"name": name, "fw": fw, "fh": fh, "box": box, "seed": seed, "rect": [a, b, a + w, b + h], "side": int(sq.shape[0]),
            "square_sha256": _sha(sq.tobytes()), "crop224_sha256": _sha(c.tobytes()), "tensor": _tensor_of(t)}


def crop_rect_case(w, h, box):
    """person_crop's rectangle alone, read back from a frame whose pixels encode their own x and y."""
    ys, xs = np.mgrid[0:h, 0:w]
    frame = np.stack([xs & 255, ys & 255, (xs >> 8) | ((ys >> 8) << 3)], -1).astype(np.uint8)
    crop = person_crop(frame, tuple(box))
    if crop is None:
        return {"w": w, "h": h, "box": box, "rect": None}
    a = int(crop[0, 0, 0]) + 256 * (int(crop[0, 0, 2]) & 7)
    b = int(crop[0, 0, 1]) + 256 * (int(crop[0, 0, 2]) >> 3)
    return {"w": w, "h": h, "box": box, "rect": [a, b, a + crop.shape[1], b + crop.shape[0]]}


def pad_case(name, h, w, seed):
    img = np.frombuffer(_lcg_bytes(seed, h * w * 3), np.uint8).reshape(h, w, 3)
    out = pad_square(img)
    return {"name": name, "h": h, "w": w, "seed": seed, "side": int(out.shape[0]), "sha256": _sha(out.tobytes())}


def gallery_document():
    gal = [
        gallery_case("dedup_and_evict_dinov2", 101, 40, cap=6),
        gallery_case("tight_cap_4", 102, 30, cap=4),
        gallery_case("roomy_cap_24_default", 103, 60, cap=24, dim=16, n_centres=6),
        gallery_case("with_signatures", 104, 40, cap=6, with_sig=True),
        gallery_case("some_signatures_missing", 105, 40, cap=5, with_sig=True, sig_share=0.5),
        gallery_case("color_spec_dedup_098", 106, 40, cap=6, spec="color", with_sig=True),
        gallery_case("one_centre_many_duplicates", 107, 30, cap=5, n_centres=1, noise=(0.03, 0.1)),
        gallery_case("far_apart_nothing_dedups", 108, 20, cap=8, dim=24, n_centres=8, noise=(1.0,)),
        gallery_case("cap_one", 109, 12, cap=1),
        gallery_case("siglip_spec", 110, 30, cap=6, spec="siglip2"),
    ]
    assert any(not st["joined"] for c in gal for st in c["steps"]) and any(len(c["final"]["tags"]) < sum(st["joined"] for st in c["steps"]) for c in gal)  # dedup and eviction both happen
    us = [
        unit_score_case("odd_count", 201, 5, 7),
        unit_score_case("even_count", 202, 6, 7),
        unit_score_case("single_crop", 203, 1, 4),
        unit_score_case("single_member", 204, 9, 1),
        unit_score_case("many", 205, 25, 24, dim=16),
        unit_score_case("no_crops", 206, 0, 5),
        unit_score_case("empty_gallery", 207, 4, 0),
        unit_score_case("dimensions_differ", 208, 3, 3, mismatch=True),
    ]
    boxes = [
        [0.1, 0.1, 0.4, 0.9], [0.0, 0.0, 1.0, 1.0], [0.45, 0.2, 0.55, 0.8], [0.9, 0.05, 1.0, 0.95], [0.02, 0.3, 0.2, 0.7], [0.3, 0.3, 0.31, 0.31],
        [0.5, 0.5, 0.9, 0.52], [0.123456, 0.234567, 0.345678, 0.987654], [0.0, 0.0, 0.3, 0.3], [0.7, 0.7, 1.0, 1.0], [0.25, 0.1, 0.5, 0.1 + 48 / 360 * 1.0],
    ]
    rects = [crop_rect_case(w, h, b) for w, h in ((640, 360), (360, 640), (640, 640), (200, 120)) for b in boxes]
    rng = random.Random(300)
    for _ in range(60):
        w, h = rng.choice([320, 360, 480, 640]), rng.choice([240, 360, 480, 640])
        x0, y0 = rng.uniform(0, 0.95), rng.uniform(0, 0.95)
        rects.append(crop_rect_case(w, h, [x0, y0, min(1.0, x0 + rng.uniform(0.01, 0.8)), min(1.0, y0 + rng.uniform(0.02, 0.9))]))
    assert any(r["rect"] is None for r in rects) and sum(r["rect"] is not None for r in rects) > 50
    pads = [pad_case("tall", 12, 5, 401), pad_case("wide", 5, 12, 402), pad_case("square", 7, 7, 403), pad_case("one_wide", 1, 4, 404), pad_case("odd_difference", 9, 4, 405),
            pad_case("big_tall", 120, 41, 406)]
    resizes = [
        resize_case("up_square_100", 100, 100, 256, 256, 501), resize_case("down_square_400", 400, 400, 256, 256, 502), resize_case("down_square_300", 300, 300, 256, 256, 503),
        resize_case("same_size_is_identity", 64, 64, 64, 64, 504), resize_case("horizontal_only", 90, 40, 256, 40, 505), resize_case("vertical_only", 40, 90, 40, 256, 506),
        resize_case("up_tiny", 16, 16, 256, 256, 507), resize_case("down_hard_640", 640, 640, 256, 256, 508), resize_case("wide_to_short_256", 480, 200, 614, 256, 509),
        resize_case("tall_to_short_256", 150, 400, 256, 682, 510), resize_case("down_by_two", 512, 512, 256, 256, 511), resize_case("up_by_one_pixel", 255, 255, 256, 256, 512),
    ]
    pipes = [
        pipeline_case("tall_person", 640, 360, [0.30, 0.10, 0.45, 0.95], 601), pipeline_case("wide_box", 640, 360, [0.10, 0.45, 0.60, 0.80], 602),
        pipeline_case("small_upscaled", 320, 240, [0.40, 0.30, 0.52, 0.80], 603), pipeline_case("edge_clamped", 320, 240, [0.0, 0.0, 0.25, 0.7], 604),
        pipeline_case("right_edge", 360, 640, [0.80, 0.20, 1.0, 0.9], 605), pipeline_case("too_small_none", 320, 240, [0.4, 0.4, 0.45, 0.5], 606),
        pipeline_case("portrait_frame_tall_box", 360, 640, [0.2, 0.05, 0.5, 0.95], 607), pipeline_case("whole_frame", 240, 240, [0.0, 0.0, 1.0, 1.0], 608),
    ]
    assert sum(1 for p in pipes if p["rect"] is None) == 1
    return {"version": VERSION, "generator": "tools/gen_golden.py", "source": "src/inframe/identity.py (Gallery, unit_score, person_crop), embed_worker.py (pad_square)",
            "lcg": "s = s * 1664525 + 1013904223 mod 2^32; byte = s >> 24; pixels are R, G, B in row order", "gallery": gal, "unit_score": us, "crop_rect": rects, "pad_square": pads,
            "resize": resizes, "pipeline": pipes}


# ---------------------------------------------------------------- resample_v2 (the TypeScript side: web/src/browser/resample.ts)
MATRICES = {"bt709": "ITU709", "bt601": "ITU601"}


CHROMA = {"left": 1, "center": 2}  # av.video.reformatter ChromaLocation values (FFmpeg AVCHROMA_LOC_LEFT, AVCHROMA_LOC_CENTER)


def _frame(fmt, w, h, y, u, v, matrix, full, chroma="left"):
    """A PyAV frame of `fmt` (nv12 or yuv420p) from three planes (y: h x w, u and v: ceil(h/2) x ceil(w/2)), tagged like a decoder tags its pictures:
    FFmpeg's h264 and hevc decoders tag a 4:2:0 frame LEFT when the stream names no siting, and swscale reads the tag (a frame with none is centred)."""
    import av
    from av.video.reformatter import ColorRange, Colorspace

    fr = av.VideoFrame(w, h, fmt)
    cw, ch = (w + 1) // 2, (h + 1) // 2

    def put(plane, rows, width, get):
        ls = plane.line_size
        buf = bytearray(ls * rows)
        for r in range(rows):
            buf[r * ls:r * ls + width] = get(r)
        plane.update(bytes(buf))

    put(fr.planes[0], h, w, lambda r: y[r * w:(r + 1) * w])
    if fmt == "nv12":
        put(fr.planes[1], ch, 2 * cw, lambda r: bytes(b for i in range(cw) for b in (u[r * cw + i], v[r * cw + i])))
    else:
        put(fr.planes[1], ch, cw, lambda r: u[r * cw:(r + 1) * cw])
        put(fr.planes[2], ch, cw, lambda r: v[r * cw:(r + 1) * cw])
    fr.colorspace = getattr(Colorspace, MATRICES[matrix])
    fr.color_range = ColorRange.JPEG if full else ColorRange.MPEG
    fr.chroma_location = CHROMA[chroma]
    return fr


def _planes(seed, w, h, smooth=False):
    """The synthetic picture of a case: noise luma, chroma noise folded into 64..191 (the TypeScript test makes the same bytes). `smooth`: chroma is a
    ramp instead, as in a real picture (the pictures swscale does not scale convert some rows against the chroma row above, which noise exposes)."""
    cw, ch = (w + 1) // 2, (h + 1) // 2
    y = _lcg_bytes(seed, w * h)
    if smooth:
        u = bytes(96 + ((i % cw) * 2 + (i // cw) * 3) // 4 for i in range(cw * ch))
        v = bytes(160 - ((i % cw) * 3 + (i // cw) * 2) // 4 for i in range(cw * ch))
        return y, u, v
    u = bytes(64 + (b >> 1) for b in _lcg_bytes(seed + 1, cw * ch))
    v = bytes(64 + (b >> 1) for b in _lcg_bytes(seed + 2, cw * ch))
    return y, u, v


def _rgb(fr, ow, oh):
    return np.asarray(fr.reformat(width=ow, height=oh, format="rgb24").to_ndarray(), np.uint8)


def flat_case(matrix, full, seed, n=300):
    """Flat pictures (every pixel the same Y, U, V) through the scaling path: the colour matrix and range with nothing else in the way."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        y, u, v = rng.randrange(256), rng.randrange(256), rng.randrange(256)
        fr = _frame("nv12", 32, 32, bytes([y]) * 1024, bytes([u]) * 256, bytes([v]) * 256, matrix, full)
        r, g, b = (int(x) for x in _rgb(fr, 16, 16)[8, 8])
        rows.append([y, u, v, r, g, b])
    return {"matrix": matrix, "full": full, "seed": seed, "yuv_rgb": rows}


def scale_case(name, fmt, matrix, full, w, h, ow, oh, seed, tol, smooth=False, chroma="left"):
    y, u, v = _planes(seed, w, h, smooth)
    out = _rgb(_frame(fmt, w, h, y, u, v, matrix, full, chroma), ow, oh)
    return {"name": name, "fmt": fmt, "matrix": matrix, "full": full, "chroma": chroma, "w": w, "h": h, "ow": ow, "oh": oh, "seed": seed, "tol": tol, "smooth": smooth, "rgb": [int(x) for x in out.ravel()]}


def squash_case(name, w, h, side, seed):
    img = np.frombuffer(_lcg_bytes(seed, w * h * 3), np.uint8).reshape(h, w, 3)
    out = np.asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR))
    return {"name": name, "w": w, "h": h, "side": side, "seed": seed, "sha256": _sha(out.tobytes())}


def upright_case(rotation, mirrored, w=7, h=5, seed=77):
    from inframe.decode import upright

    img = np.frombuffer(_lcg_bytes(seed, w * h * 3), np.uint8).reshape(h, w, 3)
    out = upright(img, rotation, mirrored)
    return {"rotation": rotation, "mirrored": mirrored, "w": w, "h": h, "seed": seed, "ow": int(out.shape[1]), "oh": int(out.shape[0]), "sha256": _sha(out.tobytes())}


def decode_pipeline_case(name, fmt, matrix, full, w, h, long, rotation, mirrored, side, seed, tol, chroma="left"):
    """iter_samples + detect.preprocess up to the resize: reformat at scaled_size, upright, PIL BILINEAR squash to side x side."""
    from inframe.decode import scaled_size, upright

    sw, sh = scaled_size(w, h, long)
    y, u, v = _planes(seed, w, h)
    rgb = upright(_rgb(_frame(fmt, w, h, y, u, v, matrix, full, chroma), sw, sh), rotation, mirrored)
    out = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((side, side), Image.BILINEAR))
    return {"name": name, "fmt": fmt, "matrix": matrix, "full": full, "chroma": chroma, "w": w, "h": h, "long": long, "rotation": rotation, "mirrored": mirrored, "side": side, "seed": seed, "tol": tol,
            "sw": sw, "sh": sh, "rgb": [int(x) for x in out.ravel()]}


def resample_document():
    flats = [flat_case(m, f, 4000 + i) for i, (m, f) in enumerate([("bt709", True), ("bt709", False), ("bt601", True), ("bt601", False)])]
    # every scaled case is bit-exact in the TypeScript port except a few pixels (about 0.1 %) where a chroma sample lands one off in the rounding of the
    # 12-bit vertical weights and the blue or green channel moves 2 levels (tol 2); the full-range picture swscale does not scale (a stored long side of 640 or less) is held to 2 levels; the limited-range one is left out: swscale's
    # converter there gave different pixels on two runs of the same call (a threaded or vector path), so no golden file can hold it
    S = scale_case
    scales = [
        S("nv12_bt709_full_2x", "nv12", "bt709", True, 128, 72, 64, 36, 5001, 1), S("nv12_bt709_limited_2x", "nv12", "bt709", False, 128, 72, 64, 36, 5002, 1),
        S("nv12_bt601_full_2x", "nv12", "bt601", True, 128, 72, 64, 36, 5003, 1), S("nv12_bt601_limited_2x", "nv12", "bt601", False, 128, 72, 64, 36, 5004, 1),
        S("i420_bt709_limited_2x", "yuv420p", "bt709", False, 128, 72, 64, 36, 5005, 1), S("i420_bt709_full_2x", "yuv420p", "bt709", True, 128, 72, 64, 36, 5006, 1),
        S("nv12_bt709_limited_3x", "nv12", "bt709", False, 192, 108, 64, 36, 5007, 1), S("nv12_bt709_full_4x", "nv12", "bt709", True, 256, 144, 64, 36, 5008, 1),
        S("nv12_bt709_limited_2_4x", "nv12", "bt709", False, 96, 64, 40, 26, 5009, 1), S("nv12_bt601_full_1_6x", "nv12", "bt601", True, 64, 48, 40, 30, 5010, 2),
        S("nv12_bt709_limited_odd_size", "nv12", "bt709", False, 130, 74, 64, 36, 5011, 2), S("i420_bt601_limited_3_1x", "yuv420p", "bt601", False, 200, 120, 64, 38, 5012, 1),
        S("i420_bt709_full_not_scaled", "yuv420p", "bt709", True, 64, 48, 64, 48, 5014, 2, True),
        # the same siting as a frame that names none (PyAV's reformat then centres the chroma)
        # an odd stored size of 640 or less: scaled_size rounds 639 x 359 up to 640 x 360, so swscale enlarges it by a pixel (the last row and column are filter output, not black)
        S("nv12_bt709_limited_odd_enlarged", "nv12", "bt709", False, 639, 359, 640, 360, 5017, 2), S("i420_bt601_full_odd_enlarged", "yuv420p", "bt601", True, 639, 359, 640, 360, 5018, 2),
        # luma shrinks while chroma grows (a 960 x 540 or 854 x 480 video at 640 x 360): swscale floors only when both vertical filters have two taps
        S("nv12_bt709_limited_luma_down_chroma_up", "nv12", "bt709", False, 192, 108, 128, 72, 5019, 1), S("i420_bt601_full_luma_down_chroma_up", "yuv420p", "bt601", True, 192, 108, 128, 72, 5022, 1),
        S("nv12_bt709_limited_2x_centre", "nv12", "bt709", False, 128, 72, 64, 36, 5015, 1, chroma="center"),
        S("i420_bt601_full_3x_centre", "yuv420p", "bt601", True, 192, 108, 64, 36, 5016, 1, chroma="center"),
    ]
    squashes = [
        squash_case("landscape_640x360", 640, 360, 512, 5101), squash_case("portrait_360x640", 360, 640, 512, 5102), squash_case("landscape_to_256", 640, 360, 256, 5103),
        squash_case("square_up_480", 480, 480, 512, 5104), squash_case("small_up", 100, 60, 64, 5105), squash_case("odd_sizes", 37, 91, 50, 5106),
        squash_case("same_size", 64, 64, 64, 5107), squash_case("tall_thin", 90, 600, 512, 5108), squash_case("wide_flat", 640, 120, 512, 5109), squash_case("big_down", 1000, 300, 200, 5110),
    ]
    uprights = [upright_case(r, m) for r in (0, 90, 180, 270, -90, -180, -270) for m in (False, True)]
    pipes = [
        decode_pipeline_case("portrait_phone_2x", "nv12", "bt709", True, 192, 108, 96, -90, False, 48, 5201, 1),
        decode_pipeline_case("landscape_limited_3x", "nv12", "bt709", False, 288, 162, 96, 0, False, 48, 5202, 1),
        decode_pipeline_case("rotated_270_mirrored_i420", "yuv420p", "bt601", False, 192, 108, 96, 270, True, 40, 5203, 1),
        decode_pipeline_case("upside_down_4x", "nv12", "bt709", True, 384, 216, 96, 180, False, 48, 5204, 1),
    ]
    import av
    import PIL

    return {"version": VERSION, "generator": "tools/gen_golden.py", "source": f"PyAV {av.__version__} (libswscale {'.'.join(str(x) for x in av.library_versions['libswscale'])}) reformat to rgb24, inframe.decode (scaled_size, upright), Pillow {PIL.__version__} BILINEAR",
            "chroma": "left: chroma_location LEFT (1), as FFmpeg's h264 and hevc decoders tag a frame; center: CENTER (2), horizontal chroma half a sample further right. Vertical siting is centred in both",
            "planes": "Y = lcg_bytes(seed, w*h); U = 64 + (lcg_bytes(seed+1, cw*ch) >> 1); V = 64 + (lcg_bytes(seed+2, cw*ch) >> 1); cw, ch = ceil(w/2), ceil(h/2); NV12 interleaves U and V",
            "flat": flats, "scale": scales, "squash": squashes, "upright": uprights, "pipeline": pipes}


def render_lines(doc, keys):
    """One entry per line for the listed list-valued keys, so a drift diff stays readable."""
    head = {k: v for k, v in doc.items() if k not in keys}
    parts = [json.dumps(head, separators=(",", ":"), allow_nan=False)[:-1]]
    for k in keys:
        parts.append(f',"{k}":[\n' + ",\n".join("    " + json.dumps(c, separators=(",", ":"), allow_nan=False) for c in doc[k]) + "\n  ]")
    return "".join(parts) + "}\n"


FILES = {
    "segments": (lambda: OUT, lambda: render(document())),
    "detect": (lambda: DETECT_OUT, lambda: render_lines(detect_document(), ("preprocess", "postprocess"))),
    "targets": (lambda: TARGETS_OUT, lambda: render_lines(targets_document(), ("cases",))),
    "gallery": (lambda: GALLERY_OUT, lambda: render_lines(gallery_document(), ("gallery", "unit_score", "crop_rect", "pad_square", "resize", "pipeline"))),
    "resample": (lambda: RESAMPLE_OUT, lambda: render_lines(resample_document(), ("flat", "scale", "squash", "upright", "pipeline"))),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 when a committed file differs from what the Python code makes now")
    ap.add_argument("--out", type=Path, default=None, help="where the segments file goes (default tests/golden/segments_v1.json); the others stay beside it")
    a = ap.parse_args(argv)
    code = 0
    for name, (path, make) in FILES.items():
        out = path()
        if name == "segments" and a.out is not None:
            out = a.out
        elif a.out is not None:
            out = a.out.with_name(out.name)
        text = make()
        json.loads(text)  # must parse
        if a.check:
            have = out.read_text() if out.exists() else None
            if have == text:
                print(f"{out}: current")
            else:
                print(f"{out}: {'missing' if have is None else 'differs from the Python code'}; run: uv run python tools/gen_golden.py", file=sys.stderr)
                code = 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
            print(f"wrote {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
