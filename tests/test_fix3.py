"""Review of T1a: track ids follow their detections, model look-up skips folders without the package,
a cancelled cut leaves no stale metadata, the GPL-built OpenCV is not in the default install."""
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from inframe import detect, models, pipeline, tracks
from inframe.detect import Detection
from inframe.pipeline import Settings, run_cut
from inframe.track_worker import align_ids

ROOT = Path(__file__).resolve().parents[1]
needs_trackers = pytest.mark.skipif(importlib.util.find_spec("trackers") is None,
                                    reason="the tracking extra is not installed (uv sync --extra tracking)")


# ---------------------------------------------------------------- track ids belong to their own detection
def test_align_ids_maps_library_rows_back_to_input_rows():
    assert list(align_ids(3, [1, 0], [5, -1])) == [-1, 5, -1]  # row 2 was not returned: -1
    assert list(align_ids(2, [], [])) == [-1, -1]
    assert list(align_ids(2, None, None)) == [-1, -1]


A = (0.10, 0.10, 0.20, 0.50)
B = (0.70, 0.10, 0.80, 0.50)
LOW = (0.40, 0.60, 0.45, 0.90)


@needs_trackers
@pytest.mark.parametrize("kind", ["ocsort", "botsort"])
def test_new_person_listed_first_does_not_take_over_the_existing_track(kind):
    frame = np.zeros((360, 640, 3), np.uint8)
    with tracks.Tracker(kind, 5.0) as tr:
        for k in range(6):
            alone = tr.update(k * 0.2, [Detection(0.9, A)], frame)
        id_a = alone[0].track_id
        assert id_a is not None
        for k in range(6, 14):
            # B has the higher score, so it is listed first; the low-confidence box is listed last
            out = tr.update(k * 0.2, [Detection(0.95, B), Detection(0.9, A), Detection(0.05, LOW)], frame)
            assert [d.box for d in out] == [B, A, LOW] and len(out) == 3  # same detections, same order, none dropped
            assert out[1].track_id == id_a  # A keeps its id the whole time
            assert out[0].track_id != id_a  # B is never given A's id
            assert out[2].track_id is None  # below the confidence floor: no id
        assert out[0].track_id is not None  # and B has its own id once confirmed


# ---------------------------------------------------------------- model look-up
def _package(root: Path, size="small") -> Path:
    p = root / f"{size}_coreml_b{detect.BATCH}" / f"rfdetr-{size}_fp16.mlpackage"
    p.mkdir(parents=True)
    return p


@pytest.mark.parametrize("where", ["cwd", "cache"])
def test_find_package_skips_folders_without_the_package(tmp_path, monkeypatch, where):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "rf-detr-small.pth").write_bytes(b"x")
    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "rf-detr-small.pth").write_bytes(b"x")
    work = tmp_path / "work"
    (work / "models").mkdir(parents=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv("INFRAME_MODELS", str(empty))
    monkeypatch.setattr(models, "cache_dir", lambda: cache)
    want = _package(work / "models" if where == "cwd" else cache)
    assert Path(detect.find_package("small")).resolve() == want.resolve()


def test_find_package_missing_lists_every_folder_searched(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INFRAME_MODELS", str(empty))
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path / "nocache")
    monkeypatch.setattr(detect, "BENCH_MODELS", tmp_path / "nobench")
    with pytest.raises(models.ModelsMissing) as e:
        detect.find_package("small")
    assert str(empty) in str(e.value) and "small CoreML package" in str(e.value)


# ---------------------------------------------------------------- a cancelled cut leaves a consistent folder
def test_cancelled_cut_does_not_leave_stale_metadata(static_box, tmp_path):
    out = tmp_path / "o"
    first = run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert first["clips"] and (out / "static_box_losslesscut.csv").is_file()

    class Stop(Exception):
        pass

    def progress(f, msg):
        if msg.startswith("cut clip"):
            raise Stop()

    with pytest.raises(Stop):
        run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=1.5, min_len=0.5), progress=progress)
    doc = json.loads((out / "segments.json").read_text())
    listed = [c["file"] for c in doc["clips"]]
    assert all((out / f).exists() for f in listed), "segments.json lists clips that are not on disk"
    assert sorted(p.name for p in out.glob("static_box_0*.mp4")) == sorted(listed), "clips on disk that segments.json does not list"
    assert doc.get("error") and doc["warnings"]
    csv = out / "static_box_losslesscut.csv"
    assert not csv.exists() or all(f in csv.read_text() for f in listed)


# ---------------------------------------------------------------- licenses
def test_default_install_has_no_opencv_pulling_dependency():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    names = lambda reqs: {r.split(">")[0].split("<")[0].split("=")[0].strip().lower() for r in reqs}  # noqa: E731
    assert "trackers" not in names(cfg["project"]["dependencies"])
    assert "trackers" not in names(cfg["dependency-groups"]["dev"]), "the dev group installs the GPL-built OpenCV wheel"
    assert "trackers" in names(cfg["project"]["optional-dependencies"]["tracking"])  # opt-in only
    readme = (ROOT / "README.md").read_text()
    assert "opencv-python (Apache-2.0; its own bundled FFmpeg is LGPL" not in readme
