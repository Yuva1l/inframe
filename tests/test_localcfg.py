"""inframe.localcfg: per-machine paths for the developer tools (environment first, then ~/.config/inframe/local.json)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from inframe import localcfg  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for var in localcfg.KEYS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(localcfg.CONFIG_ENV, str(tmp_path / "local.json"))


def test_nothing_configured_gives_none(tmp_path):
    assert all(localcfg.get(k) is None for k in localcfg.KEYS)


def test_file_value_and_environment_precedence(monkeypatch, tmp_path):
    (tmp_path / "local.json").write_text(json.dumps({"regress_video": "/data/a.mov", "dev_samples": "/data/dev"}))
    assert localcfg.get("regress_video") == Path("/data/a.mov")
    assert localcfg.get("dev_samples") == Path("/data/dev")
    monkeypatch.setenv("INFRAME_REGRESS_VIDEO", "/other/b.mov")
    assert localcfg.get("regress_video") == Path("/other/b.mov")
    assert localcfg.get("regress_baseline") is None


def test_broken_or_foreign_config_file_is_ignored(tmp_path):
    (tmp_path / "local.json").write_text("{not json")
    assert localcfg.get("regress_video") is None
    (tmp_path / "local.json").write_text("[1, 2]")
    assert localcfg.get("regress_video") is None


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        localcfg.get("nope")


def test_regress_without_a_private_clip_prints_skip_and_exits_2(capsys):
    import regress

    assert regress.main([]) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith("SKIP regress: no private clip configured")


def test_regress_skip_does_not_depend_on_config_read_when_the_module_was_imported(capsys, monkeypatch):
    """regress is imported when pytest collects the tests; a clip configured then (environment or local.json) must not leak into this test."""
    import regress

    monkeypatch.setattr(regress, "VIDEO", Path("/nonexistent/synthetic.mov"), raising=False)
    monkeypatch.setattr(regress, "BASELINE", Path("/nonexistent/synthetic.json"), raising=False)
    assert regress.main([]) == 2
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("SKIP regress: no private clip configured")


def test_regress_and_browser_regress_read_the_config_when_run_not_when_imported(capsys, monkeypatch, tmp_path):
    import browser_regress
    import regress

    base = tmp_path / "b.json"
    base.write_text(json.dumps({"presence_intervals": [[1.0, 2.0]]}))
    monkeypatch.setenv("INFRAME_REGRESS_VIDEO", str(tmp_path / "absent.mov"))
    monkeypatch.setenv("INFRAME_REGRESS_BASELINE", str(base))
    assert regress.main([]) == 2
    assert "video not found" in capsys.readouterr().out
    assert browser_regress.main([]) == 2
    assert "SKIP" not in capsys.readouterr().out


def test_browser_regress_without_a_private_clip_prints_skip_and_exits_2(capsys):
    import browser_regress

    assert browser_regress.main([]) == 2
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("SKIP browser_regress: no private clip configured")


def test_regress_treats_a_missing_accepted_file_as_none(tmp_path):
    import regress

    assert regress.load_accepted(tmp_path / "absent.json") == []


def test_splice_finds_target_clips_by_index(tmp_path, monkeypatch):
    import splice

    d = tmp_path / "clips"
    d.mkdir()
    (d / "A_01_0000-0010.mov").write_bytes(b"")
    (d / "A_03_0020-0030.mov").write_bytes(b"")
    (d / "A_03_dup.mov").write_bytes(b"")
    monkeypatch.setattr(splice, "TARGET_DIR", d)
    assert splice.target_clip("01").name == "A_01_0000-0010.mov"
    with pytest.raises(SystemExit):
        splice.target_clip("03")  # two files carry the index
    with pytest.raises(SystemExit):
        splice.target_clip("05")  # none does
    monkeypatch.setattr(splice, "TARGET_DIR", None)
    with pytest.raises(SystemExit, match="no target clips configured"):
        splice.source_path("targetA")


def test_splice_target_windows_come_from_a_file_next_to_the_clips(tmp_path, monkeypatch):
    import splice

    monkeypatch.setattr(splice, "TARGET_WINDOW", None)
    monkeypatch.setattr(splice, "TARGET_DIR", None)
    with pytest.raises(SystemExit, match="no target clips configured"):
        splice.target_windows()
    monkeypatch.setattr(splice, "TARGET_DIR", tmp_path)
    with pytest.raises(SystemExit, match="target_windows.json is missing or malformed"):
        splice.target_windows()
    (tmp_path / "target_windows.json").write_text(json.dumps({"targetA": [10.0, 1.0, 9.0], "targetB": [12.5, 0.5, 12.0], "targetC": [8, 0, 8]}))
    assert splice.target_windows()["targetB"] == (12.5, 0.5, 12.0)
    iv, cuts = splice.present_in_shot("targetA", ss=0.0, secs=4.0)  # the window from the file: padding of 1 s in front
    assert iv == [[1.0, 4.0]] and cuts == []
