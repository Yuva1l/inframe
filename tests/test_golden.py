"""The golden segment vectors (tests/golden/segments_v1.json) stay in step with segments.py and keep their coverage.

The same file is read by web/test/golden-segments.test.ts, so the TypeScript port is held to the numbers Python makes.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from inframe.segments import build_segments, drop_short, hysteresis, merge_gaps, sample_edges

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "segments_v1.json"
DOC = json.loads(GOLDEN.read_text())
CASES = DOC["cases"]
BY_NAME = {c["name"]: c for c in CASES}


def test_committed_file_matches_the_generator():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_golden.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr or r.stdout


def test_enough_cases_of_both_kinds():
    kinds = [c["kind"] for c in CASES]
    assert len(CASES) >= 65
    assert kinds.count("hand") >= 25 and kinds.count("random") >= 40
    assert len({c["name"] for c in CASES}) == len(CASES)
    assert sum(1 for c in CASES if "expect" in c and c["expect"]["clips"]) >= 30  # not a pile of empty answers
    assert {c["error"] for c in CASES if "error" in c} == {"exit_gt_enter", "length_mismatch"}


def _tuples(xs):
    return [tuple(x) for x in xs]


@pytest.mark.parametrize("c", [c for c in CASES if "expect" in c], ids=lambda c: c["name"])
def test_every_stage_replays_from_the_committed_inputs(c):
    e = c["expect"]
    args = (c["times"], c["scores"], c["enter"], c["exit"], c["duration"], c["lo"])
    assert sample_edges(c["times"], c["duration"], c["lo"]) == e["edges"]
    hyst = hysteresis(*args)
    assert hyst == _tuples(e["hysteresis"])
    merged = merge_gaps(hyst, c["merge_gap"])
    assert merged == _tuples(e["merged"])
    assert drop_short(merged, c["min_len"]) == _tuples(e["kept"])
    presence, clips = build_segments(c["times"], c["scores"], c["duration"], c["enter"], c["exit"], c["merge_gap"], c["min_len"], c["pad"], c["lo"])
    assert (presence, clips) == (_tuples(e["presence"]), _tuples(e["clips"]))


@pytest.mark.parametrize("c", [c for c in CASES if "error" in c], ids=lambda c: c["name"])
def test_error_cases_raise(c):
    with pytest.raises(ValueError):
        hysteresis(c["times"], c["scores"], c["enter"], c["exit"], c["duration"], c["lo"])


# Hand-written anchors, so a regenerated file cannot quietly bless a changed rule: these numbers are not read from the golden file.
def test_anchor_exit_equals_enter():
    c = BY_NAME["exit_equals_enter_flip"]
    assert c["enter"] == c["exit"] == 0.5
    assert c["expect"]["hysteresis"] == [[0.0, 1.5], [2.5, 4.5], [6.5, 9.5]]


def test_anchor_lo_above_zero_clamps_the_first_edge_and_padding():
    c = BY_NAME["lo_positive_pad_clamps_to_lo"]
    assert c["lo"] == 2.0
    assert c["expect"]["edges"][0] == 2.0
    assert c["expect"]["clips"] == [[2.0, 4.25]]


def test_anchor_gap_equal_to_merge_gap_merges_and_a_smaller_one_does_not():
    eq, over = BY_NAME["gap_equals_merge_gap"], BY_NAME["gap_just_over_merge_gap"]
    assert eq["expect"]["hysteresis"] == [[0.0, 0.75], [2.25, 3.25]]  # a gap of exactly 1.5 s
    assert eq["merge_gap"] == 1.5 and eq["expect"]["merged"] == [[0.0, 3.25]]
    assert over["merge_gap"] == 1.25 and over["expect"]["merged"] == [[0.0, 0.75], [2.25, 3.25]]


def test_anchor_padding_that_touches_merges():
    touch, apart = BY_NAME["pad_touching_exactly_merges"], BY_NAME["pad_leaves_a_gap"]
    assert touch["expect"]["presence"] == [[0.5, 1.5], [4.5, 5.5]] and touch["expect"]["clips"] == [[0.0, 7.0]]
    assert apart["expect"]["clips"] == [[0.0, 2.0], [5.0, 7.0]]


def test_resegment_cases_follow_the_fill_in_rule():
    rs = {c["name"]: c for c in DOC["resegment"]}
    assert len(rs) >= 8
    assert rs["settings_only"]["expect"] == rs["all_null_params"]["expect"]
    assert rs["zero_is_a_value"]["expect"]["clips"] == rs["zero_is_a_value"]["expect"]["presence"]  # pad 0 is not "missing"
    assert rs["error_no_value"]["error"] == "no_value" and rs["error_null_in_settings"]["error"] == "no_value"
    assert rs["error_exit_above_enter"]["error"] == "exit_gt_enter"
    for c in rs.values():
        if "expect" in c:
            st = c["settings"]
            v = {k: (c["params"].get(k) if c["params"].get(k) is not None else st[k]) for k in ("enter", "exit", "merge_gap", "min_len", "pad")}
            p, cl = build_segments(c["t"], c["score"], c["end"], v["enter"], v["exit"], v["merge_gap"], v["min_len"], v["pad"], c["offset"])
            assert (_tuples(c["expect"]["presence"]), _tuples(c["expect"]["clips"])) == (p, cl)


# ---------------------------------------------------------------- detect_v1 and targets_v1 (the TypeScript side: web/test/golden-detect.test.ts)
import numpy as np  # noqa: E402

from inframe import classes  # noqa: E402
from inframe.detect import Detection, postprocess, preprocess  # noqa: E402
from inframe.target import AnyOfClass, AnyPerson, Hit  # noqa: E402
from inframe.zone import Zone, ZoneTarget  # noqa: E402

DETECT = json.loads((ROOT / "tests" / "golden" / "detect_v1.json").read_text())
TARGETS = json.loads((ROOT / "tests" / "golden" / "targets_v1.json").read_text())


@pytest.mark.parametrize("c", DETECT["preprocess"], ids=lambda c: c["name"])
def test_preprocess_replays(c):
    s = c["side"]
    img = np.array(c["rgba"], np.uint8).reshape(s, s, 4)[:, :, :3]
    assert [float(v) for v in preprocess([np.ascontiguousarray(img)], s)[0].ravel()] == c["tensor"]


@pytest.mark.parametrize("c", [c for c in DETECT["postprocess"] if "expect" in c], ids=lambda c: c["name"])
def test_postprocess_replays(c):
    q, k = c["queries"], c["classes"]
    got = postprocess(np.array(c["boxes"], np.float32).reshape(1, q, 4), np.array(c["logits"], np.float32).reshape(1, q, k), 1, min_score=c["min_score"], cls=c["cls"])[0]
    assert [[d.score, *d.box] for d in got] == c["expect"]


def test_postprocess_error_case():
    c = next(c for c in DETECT["postprocess"] if "error" in c)
    with pytest.raises(ValueError):
        postprocess(np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 91), np.float32), 1, cls=c["cls"])


def test_postprocess_anchors():
    """Hand-written, not read from the file: sigmoid(0) = 0.5, the box comes from centre and size, best score first, the floor is inclusive."""
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.75, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]], np.float32)
    logits = np.full((1, 3, 91), -10.0, np.float32)
    logits[0, 0, 1] = 0.0
    logits[0, 1, 1] = 2.0
    logits[0, 2, 1] = float(np.log(0.2 / 0.8))
    got = postprocess(boxes, logits, 1)[0]
    assert [round(d.score, 4) for d in got] == [0.8808, 0.5, 0.2]
    assert [round(v, 4) for v in got[1].box] == [0.4, 0.3, 0.6, 0.7]
    assert [round(v, 4) for v in got[0].box] == [0.0, 0.5, 0.5, 1.0]
    assert postprocess(boxes, logits, 1, cls=2)[0] == []


def test_detect_coverage():
    names = {c["name"] for c in DETECT["postprocess"]}
    assert {"ties_keep_query_order", "boundary", "nothing_passes", "all_pass", "last_class_slot"} <= names
    assert any(c["cls"] == 18 for c in DETECT["postprocess"] if "expect" in c)
    ties = next(c for c in DETECT["postprocess"] if c["name"] == "ties_keep_query_order")
    assert len({e[0] for e in ties["expect"]}) < len(ties["expect"])  # real ties are in the answer


def test_class_table_and_ids_replay():
    assert {int(k): v for k, v in TARGETS["coco_91"].items()} == classes.COCO_91 and TARGETS["aliases"] == classes.ALIASES
    assert len(classes.COCO_91) == 80 and classes.COCO_91[1] == "person" and classes.COCO_91[90] == "toothbrush"
    for c in TARGETS["class_ids"]:
        if "id" in c:
            assert classes.class_id(c["name"]) == c["id"]
        else:
            with pytest.raises(classes.UnknownClass) as e:
                classes.class_id(c["name"])
            assert str(e.value) == c["error"]
    assert classes.class_id("Sports-Ball") == 37 and classes.class_id("sofa") == 63


def test_zone_validity_replays():
    for z in TARGETS["zones"]:
        if z["valid"]:
            Zone(*z["zone"])
        else:
            with pytest.raises(ValueError):
                Zone(*z["zone"])


class _First:
    name, cls, wants_tracks, stateless = "first", 1, False, False

    def evaluate(self, t, frame, cands):
        return Hit(cands[0].score, cands[0].box) if cands else Hit(0.0, None)


@pytest.mark.parametrize("c", TARGETS["cases"], ids=lambda c: c["name"])
def test_target_cases_replay(c):
    inner = {"any": lambda: AnyPerson(c["min_box"]), "class": lambda: AnyOfClass(c["cls"], c["min_box"]), "first": _First}[c["kind"]]()
    t = ZoneTarget(inner, Zone(*c["zone"]), c["zone_overlap"]) if c["zone"] else inner
    assert t.name == c["target_name"]
    got = [t.evaluate(float(i), None, [Detection(x[0], tuple(x[1:])) for x in step]) for i, step in enumerate(c["steps"])]
    assert [{"score": h.score, "box": list(h.box) if h.box else None} for h in got] == c["expect"]


def test_target_anchors():
    """Hand-written: the best candidate over the height floor wins; a zone keeps only people whose box centre is inside it."""
    tall, short = Detection(0.9, (0.0, 0.0, 0.1, 0.5)), Detection(0.8, (0.5, 0.5, 0.6, 0.55))
    assert AnyPerson(0.3).evaluate(0, None, [short, tall]).score == 0.9
    assert AnyPerson(0.3).evaluate(0, None, [short]).box is None
    z = ZoneTarget(AnyPerson(), Zone(0.4, 0.4, 0.4, 0.4))
    assert z.evaluate(0, None, [tall, short]).score == 0.8  # the better person is outside, the zone does not see them
    assert ZoneTarget(_First(), Zone(0.4, 0.4, 0.4, 0.4)).evaluate(0, None, [tall, short]).box is None  # a target with memory sees all and is gated
    assert AnyOfClass(18).name == "class:dog" and AnyOfClass(18).cls == 18


# ---------------------------------------------------------------- gallery_v1 (the TypeScript side: web/test/golden-gallery.test.ts)
from inframe.embed_worker import pad_square  # noqa: E402
from inframe.identity import SPECS, Gallery, person_crop, unit_score  # noqa: E402

GALLERY = json.loads((ROOT / "tests" / "golden" / "gallery_v1.json").read_text())


def _spec(name):
    return SPECS[name]


@pytest.mark.parametrize("c", GALLERY["gallery"], ids=lambda c: c["name"])
def test_gallery_replays_members_after_dedup_and_eviction(c):
    """identity.Gallery on the committed seeded vectors gives the same members after every add, and the same final vectors."""
    g = Gallery(_spec(c["spec"]), c["cap"])
    assert g.spec.dedup == c["dedup"]
    for i, a in enumerate(c["adds"]):
        joined = g.add(np.array(a["vec"], np.float32), np.array(a["sig"]) if a["sig"] is not None else None, tag=i)
        assert joined == c["steps"][i]["joined"] and g.tags == c["steps"][i]["tags"], f"add {i}"
    assert [[float(x) for x in r] for r in g.v] == c["final"]["vectors"]


@pytest.mark.parametrize("c", GALLERY["unit_score"], ids=lambda c: c["name"])
def test_unit_score_replays(c):
    d = len(c["vecs"][0]) if c["vecs"] else 12
    v = np.array(c["vecs"], np.float32).reshape(len(c["vecs"]), d)
    g = np.array(c["gallery"], np.float32).reshape(len(c["gallery"]), -1) if c["gallery"] else np.zeros((0, d), np.float32)
    assert unit_score(v, g) == c["expect"]


def test_gallery_anchors():
    """Hand-written, not read from the file: a near-duplicate is dropped, and at the cap the member most like another goes."""
    sp = SPECS["dinov2"]
    e = lambda *x: np.array(x, np.float32) / np.linalg.norm(x)  # noqa: E731
    g = Gallery(sp, cap=3)
    assert g.add(e(1, 0, 0), tag="a") and g.add(e(0, 1, 0), tag="b")
    assert not g.add(e(1, 0.1, 0), tag="dup")  # cosine 0.995 >= dedup 0.95
    assert g.add(e(0, 0, 1), tag="c") and g.add(e(0.6, 0.6, 0.5), tag="d")  # fourth member: over the cap
    assert len(g) == 3 and "d" in g.tags and len(set(g.tags)) == 3
    assert unit_score(np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], np.float32), np.array([[1, 0, 0]], np.float32)) == pytest.approx(0.6)
    assert unit_score(np.zeros((0, 3), np.float32), np.array([[1, 0, 0]], np.float32)) == 0.0


def test_crop_rect_replays_person_crop():
    for r in GALLERY["crop_rect"]:
        w, h = r["w"], r["h"]
        ys, xs = np.mgrid[0:h, 0:w]
        frame = np.stack([xs & 255, ys & 255, (xs >> 8) | ((ys >> 8) << 3)], -1).astype(np.uint8)
        crop = person_crop(frame, tuple(r["box"]))
        if r["rect"] is None:
            assert crop is None
        else:
            a, b, c, d = r["rect"]
            assert crop is not None and crop.shape[:2] == (d - b, c - a) and (crop == frame[b:d, a:c]).all()


def test_pad_square_anchor():
    """Hand-written: a 2 x 4 crop is centred on a 4 x 4 square of the ImageNet mean, one row above and one below."""
    crop = np.full((2, 4, 3), 7, np.uint8)
    out = pad_square(crop)
    assert out.shape == (4, 4, 3) and (out[1:3] == 7).all() and (out[0] == [124, 116, 104]).all() and (out[3] == [124, 116, 104]).all()
    assert pad_square(np.zeros((5, 5, 3), np.uint8)).shape == (5, 5, 3)


def test_reference_input_matches_the_dinov2_processor():
    """tools/gen_golden.py writes the DINOv2 preprocessing out by hand (PIL bicubic, centre crop, normalise). The real processor of
    facebook/dinov2-small (transformers 5.x) gives the same pixels to within two grey levels, so the golden vectors describe the production path."""
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    try:
        proc = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
    except OSError:
        pytest.skip("facebook/dinov2-small is not in the local Hugging Face cache")
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import gen_golden

    for seed, (h, w) in enumerate([(300, 300), (180, 90), (60, 140), (410, 205)]):
        frame = np.frombuffer(gen_golden._lcg_bytes(900 + seed, h * w * 3), np.uint8).reshape(h, w, 3)
        sq = pad_square(frame)
        _, mine = gen_golden.model_input(sq)
        theirs = proc(images=[Image.fromarray(sq)], return_tensors="np")["pixel_values"][0]
        assert mine.shape == theirs.shape == (3, 224, 224)
        assert float(np.abs(mine - theirs).max()) < 0.05  # one grey level is 0.017 after the std


# ---------------------------------------------------------------- resample_v2 (the TypeScript side: web/test/golden-resample.test.ts)
RESAMPLE = json.loads((ROOT / "tests" / "golden" / "resample_v2.json").read_text())


def _gen():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_golden

    return gen_golden


def test_resample_flat_colours_replay_through_pyav():
    """A sample of the committed flat-colour triples through PyAV now: swscale's matrix and range tables did not move."""
    g = _gen()
    for f in RESAMPLE["flat"]:
        for y, u, v, r, gg, b in f["yuv_rgb"][:25]:
            fr = g._frame("nv12", 32, 32, bytes([y]) * 1024, bytes([u]) * 256, bytes([v]) * 256, f["matrix"], f["full"])
            assert [int(x) for x in g._rgb(fr, 16, 16)[8, 8]] == [r, gg, b]


def test_resample_scale_cases_replay_through_pyav():
    g = _gen()
    for c in RESAMPLE["scale"][:6]:
        y, u, v = g._planes(c["seed"], c["w"], c["h"], c["smooth"])
        out = g._rgb(g._frame(c["fmt"], c["w"], c["h"], y, u, v, c["matrix"], c["full"], c["chroma"]), c["ow"], c["oh"])
        assert [int(x) for x in out.ravel()] == c["rgb"], c["name"]
    assert {c["chroma"] for c in RESAMPLE["scale"]} == {"left", "center"}


def test_resample_golden_frames_are_tagged_like_a_decoders():
    """An h264 yuv420p frame as FFmpeg decodes it (ffmpeg-encoded here, no stream siting) reformats to the same picture as _frame's planes with the
    default tag: the decoder tags a 4:2:0 frame LEFT, and a frame that names no siting is centred, which moved chroma a quarter sample (the decoder parity notes)."""
    import av

    g = _gen()
    w, h = 128, 72
    y, u, v = g._planes(7001, w, h, True)
    src = g._frame("yuv420p", w, h, y, u, v, "bt709", False, "left")
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=10)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"crf": "0", "preset": "ultrafast"}
        src.pts = 0
        for p in st.encode(src):
            c.mux(p)
        for p in st.encode(None):
            c.mux(p)
    buf.seek(0)
    with av.open(buf) as c:
        dec = next(c.decode(video=0))
    assert dec.format.name == "yuv420p" and int(dec.chroma_location) == 1
    # the decoder's planes, rebuilt as a golden frame (lossless x264 keeps all three); the colour tags the stream does not carry are set to the golden's
    from av.video.reformatter import ColorRange, Colorspace

    dec.colorspace, dec.color_range = Colorspace.ITU709, ColorRange.MPEG
    cw, ch = (w + 1) // 2, (h + 1) // 2
    cut = lambda i, rows, width: np.frombuffer(dec.planes[i], np.uint8).reshape(-1, dec.planes[i].line_size)[:rows, :width].tobytes()  # noqa: E731
    dy, du, dv = cut(0, h, w), cut(1, ch, cw), cut(2, ch, cw)
    assert (dy, du, dv) == (y, u, v)
    mine = g._frame("yuv420p", w, h, dy, du, dv, "bt709", False)
    a, b = g._rgb(dec, 64, 36), g._rgb(mine, 64, 36)
    assert np.array_equal(a, b)


def test_resample_squash_is_pils_bilinear():
    """The squash the detector's preprocess uses is PIL's Image.BILINEAR: the committed hashes replay from Pillow."""
    from PIL import Image

    g = _gen()
    for c in RESAMPLE["squash"]:
        img = np.frombuffer(g._lcg_bytes(c["seed"], c["w"] * c["h"] * 3), np.uint8).reshape(c["h"], c["w"], 3)
        assert g._sha(np.asarray(Image.fromarray(img).resize((c["side"], c["side"]), Image.BILINEAR)).tobytes()) == c["sha256"], c["name"]


def test_resample_upright_is_decode_upright():
    from inframe.decode import upright

    g = _gen()
    for c in RESAMPLE["upright"]:
        img = np.frombuffer(g._lcg_bytes(c["seed"], c["w"] * c["h"] * 3), np.uint8).reshape(c["h"], c["w"], 3)
        out = upright(img, c["rotation"], c["mirrored"])
        assert (out.shape[1], out.shape[0]) == (c["ow"], c["oh"]) and g._sha(out.tobytes()) == c["sha256"]


def test_resample_anchors():
    """Hand-written, not read from the file. A limited-range white (Y 235, grey chroma) is 255 when swscale does not scale and 253 when it does
    (libswscale 10.1: the scaled path lands about 1.3 levels lower on the limited ramp); a full-range grey is itself either way."""
    g = _gen()
    flat = lambda y, full, ow: [int(x) for x in g._rgb(g._frame("nv12", 32, 32, bytes([y]) * 1024, bytes([128]) * 256, bytes([128]) * 256, "bt709", full), ow, ow)[ow // 2, ow // 2]]  # noqa: E731
    assert flat(235, False, 32) == [255, 255, 255] and flat(235, False, 16) == [253, 253, 253]
    assert flat(100, True, 16) == [100, 100, 100] and flat(100, True, 32) == [100, 100, 100]
    # a decode.scaled_size the browser must reproduce, halves going to the even neighbour like Python's round()
    from inframe.decode import scaled_size

    assert scaled_size(1280, 720, 640) == (640, 360) and scaled_size(1081, 721, 640) == (640, 426) and scaled_size(4, 2, 640) == (4, 2)
