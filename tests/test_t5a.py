"""T5a: the clip sidecar (<clip stem>.inframe.json): schema, time base, box frame under rotation and flips."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from inframe import pipeline, sidecar
from inframe.detect import Detection
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_sidecars  # noqa: E402

SCHEMA = json.loads((ROOT / "docs" / "sidecar.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


def errors(doc):
    return [e.message for e in VALIDATOR.iter_errors(doc)]


class FakeDetector:
    """A person (box 0.1,0.1,0.3,0.5) in samples 5..9 only."""
    backend, path = "fake", "fake.mlpackage"

    def __init__(self):
        self.n = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if 5 <= self.n <= 9 else [])
            self.n += 1
        return out


def test_schema_is_a_valid_schema():
    Draft202012Validator.check_schema(SCHEMA)


def test_sidecar_written_next_to_the_clip_and_validates(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out = tmp_path / "o"
    doc = run_cut(synth["rot"], str(out), Settings(pad=0.5, min_len=0.5), None)
    c = doc["clips"][0]
    assert c["sidecar"] == Path(c["file"]).stem + ".inframe.json"
    d = json.loads((out / c["sidecar"]).read_text())
    assert errors(d) == []
    assert validate_sidecars.main([str(out)]) == 0
    assert d["schema_version"] == "1.2" and d["try_number"] == 1 and d["tries_total"] == 1
    assert d["source"]["path"] == synth["rot"]
    assert d["source"]["hash"]["value"] == hashlib.sha256(Path(synth["rot"]).read_bytes()).hexdigest()
    assert d["video"]["rotation"] == -90 and not d["video"]["mirrored"]
    assert (d["video"]["display_width"], d["video"]["display_height"]) == (180, 320)  # stored 320x180, turned
    assert d["clip"]["file"] == c["file"] and d["clip"]["copy_start_s"] == c["expected_copy_start"] == 0.0
    assert d["clip"]["in_source_s"] == c["start"] and d["clip"]["out_source_s"] == c["end"]
    assert d["target"]["name"] == "any-person" and d["embedder"] is None
    assert d["detector"]["backend"] == "fake" and d["detector"]["tracker"] is None and d["track_ids"] == []
    s = d["samples"]
    assert s[0]["clip_time"] == pytest.approx(s[0]["source_time"] - d["clip"]["copy_start_s"], abs=1e-3)
    with_box = [x for x in s if x["box"]]
    assert with_box and all(x["box"] == [0.1, 0.1, 0.3, 0.5] and x["confidence"] == 0.9 for x in with_box)
    assert all(x["box"] is None and x["confidence"] == 0.0 for x in s if x not in with_box)
    # the presence stretch (about 0.9-1.9 s) is cut to the clip
    assert d["presence_in_clip"]["source_s"] == doc["presence_intervals"]
    assert d["presence_in_clip"]["clip_s"] == doc["presence_intervals"]  # copy_start is 0 here
    # the clip really starts where the sidecar says
    assert probe(str(out / c["file"])).rotation == -90


def test_clip_time_counts_from_the_keyframe_the_clip_starts_on(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out = tmp_path / "o"
    # padded start 1.0 - 0.2 = 0.8 snaps back to the keyframe at 0.5 (keyframe every 0.5 s)
    doc = run_cut(synth["plain"], str(out), Settings(pad=0.2, min_len=0.5), None)
    c = doc["clips"][0]
    d = json.loads((out / c["sidecar"]).read_text())
    assert errors(d) == [] and validate_sidecars.main([str(out)]) == 0
    assert d["clip"]["copy_start_s"] == pytest.approx(0.5) and c["start"] > 0.5
    assert d["clip"]["in_source_s"] == c["start"]
    assert d["samples"][0]["source_time"] >= 0.5 - 1e-6 and d["samples"][0]["clip_time"] >= 0
    first_box = next(x for x in d["samples"] if x["box"])
    assert first_box["clip_time"] == pytest.approx(first_box["source_time"] - 0.5, abs=1e-3)
    p = d["presence_in_clip"]
    assert p["clip_s"][0][0] == pytest.approx(p["source_s"][0][0] - 0.5, abs=1e-3)


def test_each_clip_gets_its_try_number_and_a_rerun_removes_stale_sidecars(synth, tmp_path, monkeypatch):
    class Twice:
        backend, path = "fake", "fake.mlpackage"
        n = 0

        def detect(self, frames):
            out = []
            for _ in frames:
                out.append([Detection(0.9, (0.2, 0.2, 0.4, 0.6))] if self.n in (1, 2, 3, 11, 12, 13) else [])
                self.n += 1
            return out

    monkeypatch.setattr(pipeline, "load_detector", lambda size: Twice())
    out = tmp_path / "o"
    doc = run_cut(synth["plain"], str(out), Settings(pad=0.0, min_len=0.4, merge_gap=0.5, fps=5.0), None)
    assert len(doc["clips"]) == 2
    docs = [json.loads((out / c["sidecar"]).read_text()) for c in doc["clips"]]
    assert [d["try_number"] for d in docs] == [1, 2] and all(d["tries_total"] == 2 for d in docs)
    assert all(errors(d) == [] for d in docs)
    (out / "plain_09_0000-0001.inframe.json").write_text("{}")  # left by an earlier run
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    doc2 = run_cut(synth["plain"], str(out), Settings(pad=0.5, min_len=0.5), None)
    assert sorted(p.name for p in out.glob("*.inframe.json")) == sorted(c["sidecar"] for c in doc2["clips"])


def test_track_ids_and_embedder_are_recorded():
    info = dict(path="/x/v.mov", duration=20.0, fps=30.0, rotation=0, mirrored=False, width=1280, height=720,
                display_width=1280, display_height=720)
    clip = dict(index=2, start=5.0, end=8.0, file="v_02_0005-0008.mov", expected_copy_start=4.5)
    times = [4.0, 4.5, 5.0, 5.5, 6.0, 8.5]
    boxes = [None, (0.1, 0.1, 0.2, 0.3), (0.1, 0.1, 0.2, 0.3), None, (0.5, 0.5, 0.9, 0.9), (0.1, 0.1, 0.2, 0.2)]
    boxes[4] = (0.9635, 0.4706, 1.0003, 0.5733)  # a person at the edge: the detector overshoots 1.0 by a thousandth
    boxes[2] = (-0.0004, 0.4, 0.2, 0.9)
    d = sidecar.build_sidecar(info=info, src_hash=dict(algorithm="sha256", value="0" * 64, size_bytes=1), clip=clip, n_clips=3,
                              times=times, scores=[0, .8, .9, 0, .7, .9], boxes=boxes, track_ids=[None, 4, 4, None, 7, 9],
                              presence=[(4.6, 6.2), (30.0, 31.0)], settings=dict(fps=5.0), target="saved:alice",
                              embedder="dinov2-small", detector=dict(mode="detect", backend="b", model="small", tracker="ocsort"))
    assert errors(d) == []
    assert [x["box"] for x in d["samples"] if x["source_time"] in (5.0, 6.0)] == [[0.0, 0.4, 0.2, 0.9], [0.9635, 0.4706, 1.0, 0.5733]]
    assert d["track_ids"] == [4, 7] and d["embedder"] == "dinov2-small" and d["try_number"] == 2
    assert [s["source_time"] for s in d["samples"]] == [4.5, 5.0, 5.5, 6.0]  # 4.0 is before the copy start, 8.5 after the end
    assert [s["clip_time"] for s in d["samples"]] == [0.0, 0.5, 1.0, 1.5]
    assert d["presence_in_clip"] == dict(source_s=[[4.6, 6.2]], clip_s=[[0.1, 1.7]])


def test_schema_rejects_bad_documents(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out = tmp_path / "o"
    doc = run_cut(synth["rot"], str(out), Settings(pad=0.5, min_len=0.5), None)
    good = json.loads((out / doc["clips"][0]["sidecar"]).read_text())
    assert errors(good) == []
    for mutate in (lambda d: d.update(schema_version="2.0"), lambda d: d.pop("samples"), lambda d: d.pop("embedder"),
                   lambda d: d["samples"][0].update(box=[0.1, 0.2]), lambda d: d["samples"][0].update(confidence=1.5),
                   lambda d: d["source"]["hash"].update(value="abc"), lambda d: d["coordinate_frame"].update(box_format="xywh")):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        assert errors(bad), mutate
    f = out / "broken.inframe.json"
    f.write_text(json.dumps(dict(good, try_number=5)))  # valid shape, but try 5 of 1
    assert validate_sidecars.main([str(f)]) == 1
    assert validate_sidecars.main([str(tmp_path / "none")]) == 1


def test_motion_mode_writes_sidecars_without_boxes(static_box, tmp_path):
    out = tmp_path / "m"
    doc = run_cut(static_box, str(out), Settings(mode="motion", pad=0.5, min_len=0.5), None)
    assert doc["clips"]
    for c in doc["clips"]:
        d = json.loads((out / c["sidecar"]).read_text())
        assert errors(d) == [] and d["target"]["name"] == "any-motion" and d["detector"]["mode"] == "motion"
        assert all(s["box"] is None for s in d["samples"])


# ---------------------------------------------------------------- boxes are in the upright display frame
class BrightBlobDetector:
    """Finds the white blob in the (upright) frame it is given: the box a real detector would report."""
    backend, path = "blob", "blob"

    def detect(self, frames):
        out = []
        for f in frames:
            m = f.mean(axis=2) > 200
            ys, xs = np.where(m)
            h, w = m.shape
            out.append([Detection(0.95, (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h))] if len(xs) else [])
        return out


@pytest.mark.parametrize("opts", [[], ["-display_rotation", "-90"], ["-display_rotation", "90"], ["-display_rotation", "180"],
                                  ["-display_hflip"], ["-display_rotation", "90", "-display_hflip"],
                                  ["-display_rotation", "-90", "-display_vflip"]])
def test_sidecar_boxes_follow_the_picture_ffmpeg_shows(tmp_path, monkeypatch, opts):
    base, v = tmp_path / "base.mp4", tmp_path / "v.mp4"
    _ff("-f", "lavfi", "-i", "color=c=black:s=192x128:r=5:d=4,drawbox=x=20:y=10:w=50:h=30:color=white:t=fill",
        "-c:v", "libx264", "-g", "5", "-pix_fmt", "yuv420p", str(base))
    _ff(*opts, "-i", str(base), "-c", "copy", str(v))
    info = probe(str(v))
    w, h = info.display_size
    raw = tmp_path / "ref.rgb"
    _ff("-i", str(v), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw))  # ffmpeg autorotates
    ref = np.frombuffer(raw.read_bytes(), np.uint8).reshape(h, w, 3)
    ys, xs = np.where(ref.mean(axis=2) > 200)
    want = np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h])
    monkeypatch.setattr(pipeline, "load_detector", lambda size: BrightBlobDetector())
    out = tmp_path / "o"
    doc = run_cut(str(v), str(out), Settings(pad=0.0, min_len=0.5, fps=5.0), None)
    d = json.loads((out / doc["clips"][0]["sidecar"]).read_text())
    assert errors(d) == [] and (d["video"]["display_width"], d["video"]["display_height"]) == (w, h)
    assert d["video"]["rotation"] == info.rotation and d["video"]["mirrored"] == info.mirrored
    boxes = np.array([s["box"] for s in d["samples"] if s["box"]])
    assert len(boxes) >= 15
    assert np.abs(boxes - want).max() < 0.03, (opts, boxes[0], want)


# ---------------------------------------------------------------- review findings
def test_a_sidecar_error_other_than_oserror_does_not_leave_a_stale_segments_json(synth, tmp_path, monkeypatch):
    from inframe.target import AnyPerson

    class BadEmbedderTarget(AnyPerson):
        embedder = object()  # not JSON serialisable: json.dumps raises TypeError inside write_sidecars

    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out = tmp_path / "o"
    run_cut(synth["plain"], str(out), Settings(pad=0.5, min_len=0.5), None)
    assert json.loads((out / "segments.json").read_text())["settings"]["pad"] == 0.5
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    doc = run_cut(synth["plain"], str(out), Settings(pad=0.2, min_len=0.5), BadEmbedderTarget())
    seg = json.loads((out / "segments.json").read_text())
    assert seg["settings"]["pad"] == 0.2  # rewritten for this run, not the earlier run's
    assert any("sidecars not written" in w for w in doc["warnings"])


def test_sidecar_records_what_the_confidence_means_and_the_thresholds(static_box, synth, tmp_path, monkeypatch):
    out = tmp_path / "m"
    doc = run_cut(static_box, str(out), Settings(mode="motion", pad=0.5, min_len=0.5), None)
    d = json.loads((out / doc["clips"][0]["sidecar"]).read_text())
    assert errors(d) == []
    assert d["score"]["kind"] == "motion_foreground_fraction"
    assert d["score"]["enter"] == pytest.approx(doc["settings"]["enter"]) and d["score"]["exit"] == pytest.approx(doc["settings"]["exit"])
    assert max(s["confidence"] for s in d["samples"]) >= d["score"]["enter"]
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out2 = tmp_path / "d"
    doc2 = run_cut(synth["plain"], str(out2), Settings(pad=0.5, min_len=0.5), None)
    d2 = json.loads((out2 / doc2["clips"][0]["sidecar"]).read_text())
    assert d2["score"]["kind"] == "detector_confidence" and errors(d2) == []
    assert d2["score"]["enter"] == pytest.approx(doc2["settings"]["enter"])
