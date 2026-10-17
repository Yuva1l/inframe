import json

from inframe import pipeline
from inframe.detect import Detection
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe


class FakeDetector:
    """A person in samples 5..9 only."""
    backend, path = "fake", "fake.mlpackage"

    def __init__(self):
        self.n = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if 5 <= self.n <= 9 else [])
            self.n += 1
        return out


def test_run_cut_end_to_end_with_fake_detector(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out = tmp_path / "o"
    # 15 samples at 5 fps; person at samples 5..9 = 1.0 s..2.0 s, so presence is about 0.9-1.9 s
    doc = run_cut(synth["rot"], str(out), Settings(pad=0.5, min_len=0.5), None)
    assert len(doc["presence_intervals"]) == 1
    s, e = doc["presence_intervals"][0]
    assert 0.8 < s < 1.1 and 1.8 < e < 2.1
    c = doc["clips"][0]
    assert c["file"].startswith("rot_01_") and (out / c["file"]).is_file()
    assert c["peak_score"] == 0.9 and c["expected_copy_start"] == 0.0  # padded start 0.4 snaps back to the keyframe at 0
    info = probe(str(out / c["file"]))
    assert info.rotation == -90 and info.has_audio
    assert json.loads((out / "segments.json").read_text())["clips"][0]["file"] == c["file"]
    assert (out / "rot_losslesscut.csv").read_text().startswith(f"{c['start']:.3f},{c['end']:.3f},{c['file']}")
    assert len((out / "samples.csv").read_text().splitlines()) == 16
