"""T8a: the COCO-91 class table, class id in postprocess, model size by object size, `--target class:NAME`."""
import numpy as np
import pytest

from inframe import classes, detect, identity
from inframe.tap import Tap, TapError, TapTarget, crop_signature
from inframe.target import AnyOfClass, AnyPerson

# the official COCO category ids (cocodataset.org, instances_val2017 categories), written out independently of classes.py
OFFICIAL = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5, "bus": 6, "train": 7, "truck": 8, "boat": 9,
    "traffic light": 10, "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15, "bird": 16, "cat": 17, "dog": 18,
    "horse": 19, "sheep": 20, "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27, "umbrella": 28,
    "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34, "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38,
    "baseball bat": 39, "baseball glove": 40, "skateboard": 41, "surfboard": 42, "tennis racket": 43, "bottle": 44,
    "wine glass": 46, "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51, "banana": 52, "apple": 53, "sandwich": 54,
    "orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61, "chair": 62, "couch": 63,
    "potted plant": 64, "bed": 65, "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73, "mouse": 74, "remote": 75,
    "keyboard": 76, "cell phone": 77, "microwave": 78, "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82, "book": 84,
    "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89, "toothbrush": 90,
}


def test_the_table_has_the_80_coco_classes():
    assert len(OFFICIAL) == 80 and len(classes.COCO_91) == 80
    assert max(classes.COCO_91) == 90 and classes.PERSON == 1


@pytest.mark.parametrize("name,cid", sorted(OFFICIAL.items(), key=lambda kv: kv[1]))
def test_class_id_per_class(name, cid):
    assert classes.class_id(name) == cid
    assert classes.class_name(cid) == name
    assert classes.class_id(name.upper().replace(" ", "_")) == cid  # case, underscores and dashes do not matter
    assert classes.parse_class_spec(name) == (cid, name)


def test_the_table_matches_the_rfdetr_package():
    ref = pytest.importorskip("rfdetr.assets.coco_classes").COCO_CLASSES
    assert dict(ref) == classes.COCO_91


def test_aliases_and_unknown_names():
    assert classes.class_id("bike") == 2 and classes.class_id("sofa") == 63 and classes.class_id(" Puppy ") == 18
    with pytest.raises(classes.UnknownClass, match="did you mean.*dog"):
        classes.class_id("dogg")
    with pytest.raises(classes.UnknownClass):
        classes.class_id("unicorn")
    with pytest.raises(classes.UnknownClass):
        classes.class_id("")


def _model_out(cls_hot: int, n_cls: int = 91):
    """Two queries: query 0 fires on class `cls_hot`, query 1 on the person slot."""
    boxes = np.zeros((1, 2, 4), np.float32)
    boxes[0, 0] = (0.5, 0.5, 0.2, 0.4)
    boxes[0, 1] = (0.2, 0.3, 0.1, 0.1)
    logits = np.full((1, 2, n_cls), -10.0, np.float32)
    logits[0, 0, cls_hot] = 4.0
    logits[0, 1, classes.PERSON] = 3.0
    return boxes, logits


@pytest.mark.parametrize("name", ["dog", "car", "cat", "skateboard", "sports ball"])
def test_postprocess_reads_the_requested_class(name):
    cid = classes.class_id(name)
    boxes, logits = _model_out(cid)
    got = detect.postprocess(boxes, logits, 1, cls=cid)[0]
    assert len(got) == 1 and got[0].score > 0.95 and got[0].box == pytest.approx((0.4, 0.3, 0.6, 0.7))
    person = detect.postprocess(boxes, logits, 1)[0]  # the default is still the person slot
    assert len(person) == 1 and person[0].box[0] == pytest.approx(0.15)


def test_postprocess_refuses_a_class_outside_the_model():
    boxes, logits = _model_out(18)
    with pytest.raises(ValueError, match="outside"):
        detect.postprocess(boxes, logits, 1, cls=91)


def test_model_for_person_keeps_the_preset():
    assert classes.model_for("nano", classes.PERSON, 0.02) == ("nano", None)


def test_model_for_a_class_never_runs_on_nano():
    m, why = classes.model_for("nano", 18)
    assert m == "small" and "nano" in why
    assert classes.model_for("small", 18) == ("small", None)


def test_model_for_small_objects_goes_up_when_a_bigger_model_exists():
    assert classes.model_for("small", 18, 0.30)[0] == "small"  # 0.30 * 512 = 154 px
    m, why = classes.model_for("small", 18, 0.10)  # 51 px in the small input
    assert m == "base" and "51" in why
    assert classes.model_for("small", 18, 0.10, have=("nano", "small")) == ("small", None)  # no base package: stays
    assert classes.model_for("nano", 3, 0.10)[0] == "base"


def test_the_boundary_is_80_px_of_the_models_input():
    assert classes.model_for("small", 18, 80 / 512 + 1e-6)[0] == "small"
    assert classes.model_for("small", 18, 80 / 512 - 1e-6)[0] == "base"


def test_any_of_class_scores_the_best_box_of_its_class():
    t = AnyOfClass(18, min_box=0.1)
    assert t.name == "class:dog" and t.cls == 18 and not t.wants_tracks
    d = [detect.Detection(0.9, (0, 0, 0.1, 0.05)), detect.Detection(0.6, (0.2, 0.2, 0.5, 0.6))]
    assert t.evaluate(0, np.zeros((10, 10, 3), np.uint8), d).score == 0.6  # the 0.05-high box is under min_box
    assert AnyPerson().cls == 1


def test_parse_targets_class_alone_is_every_instance():
    t = identity.parse_targets(["class:dog"])
    assert isinstance(t, AnyOfClass) and t.cls == 18


def test_parse_targets_class_with_a_tap_follows_one_object():
    t = identity.parse_targets(["class:car", "tap:3,0.5,0.5"])
    assert isinstance(t, TapTarget) and t.cls == 3 and t.taps == [Tap(3.0, point=(0.5, 0.5))]
    assert identity.parse_targets(["tap:3,0.5,0.5"]).cls == classes.PERSON


def test_parse_targets_reads_the_class_from_a_pick_file(tmp_path):
    f = tmp_path / "target.json"
    f.write_text('{"time": 2.0, "box": [0.1, 0.1, 0.4, 0.5], "class": "dog"}')
    t = identity.parse_targets([f"tap:{f}"])
    assert isinstance(t, TapTarget) and t.cls == 18


def test_parse_targets_errors():
    with pytest.raises(TapError, match="unknown class"):
        identity.parse_targets(["class:unicorn"])
    with pytest.raises(TapError, match="two classes"):
        identity.parse_targets(["class:dog", "class:cat"])


def test_photo_target_takes_the_class(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    t = identity.parse_targets(["class:dog", "photos:" + str(tmp_path)])
    assert isinstance(t, identity.PhotoTarget) and t.cls == 18


def test_tap_target_height_frac_is_the_smallest_tapped_box():
    t = TapTarget([Tap(1.0, box=(0, 0, 0.2, 0.3)), Tap(2.0, box=(0, 0, 0.2, 0.1)), Tap(3.0, point=(0.5, 0.5))], cls=18)
    assert t.height_frac() == pytest.approx(0.1)
    assert TapTarget([Tap(1.0, point=(0.5, 0.5))]).height_frac() is None


def test_signature_of_a_non_person_box_uses_the_centre_of_the_box():
    f = np.zeros((100, 100, 3), np.uint8)
    f[:, 25:75] = (200, 30, 30)
    a = crop_signature(f, (0.2, 0.2, 0.8, 0.8), person=False)
    assert a is not None and a.shape == crop_signature(f, (0.2, 0.2, 0.8, 0.8)).shape


# ---------------------------------------------------------------- the pipeline
class _ClassSpy:
    """A detector that sees a dog (class 18) in samples 5..9 and nothing of any other class; it records what it was asked."""
    backend, path = "fake", "fake.mlpackage"

    def __init__(self, size):
        self.size, self.n, self.asked = size, 0, []

    def detect(self, frames, cls=1):
        self.asked.append(cls)
        out = []
        for _ in frames:
            out.append([detect.Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if cls == 18 and 5 <= self.n <= 9 else [])
            self.n += 1
        return out


def test_a_class_target_runs_the_detector_on_that_class_and_never_on_nano(synth, tmp_path, monkeypatch):
    from inframe import pipeline

    made = []
    monkeypatch.setattr(pipeline, "load_detector", lambda size: made.append(_ClassSpy(size)) or made[-1])
    doc = pipeline.run_cut(synth["rot"], str(tmp_path / "o"), pipeline.Settings(preset="fast", fps=5, pad=0.5, min_len=0.5), identity.parse_targets(["class:dog"]))
    assert made[0].size == "small" and set(made[0].asked) == {18}  # fast = nano for people; a class gets small
    assert len(doc["presence_intervals"]) == 1 and doc["clips"][0]["peak_score"] == 0.9
    s = doc["settings"]
    assert s["target"] == "class:dog" and s["class"] == "dog" and s["class_id"] == 18 and s["model"] == "small" and "nano" in s["model_note"]


def test_a_person_run_is_unchanged(synth, tmp_path, monkeypatch):
    from inframe import pipeline

    seen = []

    class P:
        backend, path = "fake", "fake.mlpackage"

        def detect(self, frames):  # no cls parameter: a person run must not pass one
            return [[] for _ in frames]

    monkeypatch.setattr(pipeline, "load_detector", lambda size: seen.append(size) or P())
    doc = pipeline.run_cut(synth["rot"], str(tmp_path / "o"), pipeline.Settings(preset="fast", fps=5), None)
    assert seen == ["nano"] and "class" not in doc["settings"]


def test_a_tap_on_a_small_dog_moves_to_a_larger_model(synth, tmp_path, monkeypatch):
    import importlib.util

    if importlib.util.find_spec("trackers") is None:  # the tracking extra is not installed; find_spec does not import it (and its OpenCV)
        pytest.skip("needs the tracking extra")
    from inframe import pipeline

    made = []
    monkeypatch.setattr(pipeline, "load_detector", lambda size: made.append(_ClassSpy(size)) or made[-1])
    monkeypatch.setattr(pipeline, "available_sizes", lambda: ("nano", "small", "base"))
    tg = TapTarget([Tap(1.0, box=(0.1, 0.1, 0.2, 0.18))], cls=18)  # 8 % of the frame height = 41 px of the small input
    try:
        pipeline.run_cut(synth["rot"], str(tmp_path / "o"), pipeline.Settings(fps=5), tg)
    except TapError:
        pass  # the spy's dog is elsewhere; only the model choice matters here
    assert made[0].size == "base"


def test_pick_file_class_conflicting_with_class_target_errors(tmp_path):
    f = tmp_path / "target.json"
    f.write_text('{"class": "dog", "time": 0.0, "box": [0.4, 0.4, 0.6, 0.6]}')  # a valid pick file: parse_spec reads its tap first
    for order in (["class:cat", f"tap:{f}"], [f"tap:{f}", "class:cat"]):
        with pytest.raises(identity.TapError, match="two classes"):
            identity.parse_targets(order)


def test_pick_file_unknown_class_errors(tmp_path):
    f = tmp_path / "target.json"
    f.write_text('{"class": "unicorn", "time": 0.0, "box": [0.4, 0.4, 0.6, 0.6]}')
    with pytest.raises(identity.TapError, match="unknown class"):
        identity.parse_targets([f"tap:{f}"])
