"""A0: picker.pick(kind="images") passes --images to the helper; the default and kind="videos" do not. Fake helper via
INFRAME_PICKER_HELPER as in tests/test_picker.py; the real swiftc build and a real Photos pick are not run here."""
import json
import stat
import sys

import pytest

from inframe import picker

FAKE = '''#!{py}
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
out = Path(a[a.index("--out") + 1])
d = out / "01"; d.mkdir(parents=True, exist_ok=True)
(d / "p.jpg").write_text("x")
Path(os.environ["FAKE_ARGS"]).write_text(json.dumps(a))
print(json.dumps(dict(cancelled=False, files=[dict(path=str(d / "p.jpg"), name="p.jpg")], errors=[])))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    p = tmp_path / "fake-helper"
    p.write_text(FAKE.format(py=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(picker.HELPER_ENV, str(p))
    args = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_ARGS", str(args))
    return lambda: json.loads(args.read_text())


def test_images_kind_passes_images_flag(fake, tmp_path):
    res = picker.pick(tmp_path / "out", kind="images")
    assert fake() == ["--out", str(tmp_path / "out"), "--images"]
    assert [f.name for f in res.files] == ["p.jpg"]


def test_images_kind_keeps_limit(fake, tmp_path):
    picker.pick(tmp_path / "out", limit=5, kind="images")
    assert fake() == ["--out", str(tmp_path / "out"), "--limit", "5", "--images"]


@pytest.mark.parametrize("kw", [{}, {"kind": "videos"}])
def test_videos_never_get_images_flag(fake, tmp_path, kw):
    picker.pick(tmp_path / "out", **kw)
    assert "--images" not in fake()


def test_bad_kind_is_value_error_before_anything_runs(fake, tmp_path):
    with pytest.raises(ValueError):
        picker.pick(tmp_path / "out", kind="audio")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "args.json").exists()


def test_announce_names_the_kind(fake, tmp_path):
    msgs = []
    picker.pick(tmp_path / "o1", announce=msgs.append, kind="images")
    picker.pick(tmp_path / "o2", announce=msgs.append)
    assert "photos" in msgs[0] and "videos" in msgs[1]
