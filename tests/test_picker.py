"""T3p: `inframe photos pick` with a fake helper (the real picking is done by the person at the macOS picker). The Swift helper is compiled
by one test when swiftc exists (macOS); the picker window itself is never opened in tests."""
import json
import os
import shutil
import stat
import sys

import pytest

from inframe import cli, picker

FAKE = '''#!{py}
import json, shutil, sys, os
from pathlib import Path
a = sys.argv[1:]
out = Path(a[a.index("--out") + 1])
mode = os.environ.get("FAKE_MODE", "ok")
print("fake helper started", file=sys.stderr)
if mode == "cancel":
    print(json.dumps(dict(cancelled=True, files=[], errors=[]))); sys.exit(0)
if mode == "garbage":
    print("not json"); sys.exit(0)
if mode == "crash":
    sys.exit(7)
if mode == "exportcrash":
    (out / "01").mkdir(parents=True, exist_ok=True); (out / "01" / "IMG_0001.MOV").write_text("x"); sys.exit(7)
files = []
for i, src in enumerate(os.environ["FAKE_SRC"].split(os.pathsep), 1):
    d = out / f"{{i:02d}}"; d.mkdir(parents=True, exist_ok=True)
    dest = d / os.environ.get("FAKE_NAME", "IMG_0001.MOV")
    shutil.copy(src, dest)
    files.append(dict(path=str(dest), name=dest.name))
print(json.dumps(dict(cancelled=False, files=files, errors=["item 3: no file"] if mode == "partial" else [])))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    p = tmp_path / "fake-helper"
    p.write_text(FAKE.format(py=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(picker.HELPER_ENV, str(p))
    monkeypatch.chdir(tmp_path)
    return p


def stub_cut(monkeypatch, calls):
    def run_cut(video, out, settings, target=None):
        calls.append(dict(video=video, out=out, mode=settings.mode, reel=settings.reel, existed=os.path.isfile(video)))
        os.makedirs(out, exist_ok=True)
        return dict(clips=[dict(file="c1.mp4", start=1.0, end=2.0, peak_score=0.9)], presence_intervals=[[1.0, 2.0]])
    monkeypatch.setattr(cli, "run_cut", run_cut)


def test_pick_cuts_every_export_and_removes_the_temp_folder(fake, synth, monkeypatch, capfd, tmp_path):
    monkeypatch.setenv("FAKE_SRC", os.pathsep.join([synth["plain"], synth["rot"]]))
    calls = []
    stub_cut(monkeypatch, calls)
    code = cli.main(["photos", "pick", "--", "--mode", "motion", "--reel"])
    assert code == 0
    assert [c["existed"] for c in calls] == [True, True]  # the export was there while cutting
    assert all(c["mode"] == "motion" and c["reel"] for c in calls)
    assert [c["out"] for c in calls] == [str(tmp_path / f"IMG_0001_{k}_inframe" / "motion") for k in (1, 2)]  # cwd, never next to the temp export; equal names kept apart
    exported = os.path.dirname(os.path.dirname(calls[0]["video"]))
    assert not os.path.exists(exported)  # temp export removed
    o = capfd.readouterr()
    assert "fake helper started" in o.err and "[2/2] cutting IMG_0001.MOV" in o.out and "item 3" not in o.err


def test_keep_and_export_dir(fake, synth, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_SRC", synth["plain"])
    stub_cut(monkeypatch, [])
    ex = tmp_path / "ex"
    assert cli.main(["photos", "pick", "--export-dir", str(ex)]) == 0
    assert (ex / "01" / "IMG_0001.MOV").is_file()


def test_out_with_two_videos_gets_one_folder_each(fake, synth, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_SRC", os.pathsep.join([synth["plain"], synth["rot"]]))
    calls = []
    stub_cut(monkeypatch, calls)
    assert cli.main(["photos", "pick", "--", "--out", str(tmp_path / "o")]) == 0
    assert [c["out"] for c in calls] == [str(tmp_path / "o" / f"IMG_0001_{k}") for k in (1, 2)]


def test_no_cut_prints_the_paths(fake, synth, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SRC", synth["plain"])
    called = []
    stub_cut(monkeypatch, called)
    assert cli.main(["photos", "pick", "--no-cut"]) == 0
    out = capsys.readouterr().out
    d = json.loads(out[out.index("{"):out.rindex("}") + 1])
    assert len(d["files"]) == 1 and os.path.isfile(d["files"][0]) and not called


def test_cancel_exits_1_and_cuts_nothing(fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_MODE", "cancel")
    called = []
    stub_cut(monkeypatch, called)
    assert cli.main(["photos", "pick"]) == 1
    assert "nothing was picked" in capsys.readouterr().err and not called


@pytest.mark.parametrize("mode", ["garbage", "crash"])
def test_helper_failures_exit_2(fake, monkeypatch, capsys, mode):
    monkeypatch.setenv("FAKE_MODE", mode)
    assert cli.main(["photos", "pick"]) == 2
    assert "inframe:" in capsys.readouterr().err


def test_partial_errors_are_reported_but_picked_files_still_cut(fake, synth, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_MODE", "partial")
    monkeypatch.setenv("FAKE_SRC", synth["plain"])
    calls = []
    stub_cut(monkeypatch, calls)
    assert cli.main(["photos", "pick"]) == 0
    assert len(calls) == 1 and "item 3: no file" in capsys.readouterr().err


def test_bad_cut_options_fail_before_the_picker_opens(fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_MODE", "crash")  # would exit 2 with 'helper failed' if it were run
    assert cli.main(["photos", "pick", "--", "--preset", "nope"]) == 2
    assert cli.main(["photos", "pick", "--", "some.mov"]) == 2
    assert "photo picker helper" not in capsys.readouterr().err


@pytest.mark.parametrize("bad", [["--fps", "0"], ["--pad", "-1"], ["--edit-list", "bogus"], ["--target", "tap:/nonexistent/target.json"],
                                 ["--mode", "motion", "--target", "tap:1,0.5,0.5"]])
def test_number_edit_list_and_target_errors_fail_before_the_picker_opens(fake, monkeypatch, capsys, bad):
    ran = fake.parent / "helper-ran"
    fake.write_text(fake.read_text().replace('print("fake helper started"', f'open({str(ran)!r}, "w").write("x"); print("fake helper started"'))
    monkeypatch.setenv("FAKE_MODE", "crash")
    assert cli.main(["photos", "pick", "--", *bad]) == 2
    assert not ran.exists() and "photo picker helper" not in capsys.readouterr().err


def test_export_survives_a_failing_picker_step(fake, monkeypatch, tmp_path):
    exports = []
    real = picker.new_export_dir
    monkeypatch.setattr(picker, "new_export_dir", lambda: exports.append(real()) or exports[-1])
    monkeypatch.setenv("FAKE_MODE", "exportcrash")
    assert cli.main(["photos", "pick"]) == 2
    assert exports and not exports[0].exists()  # the exported original is removed although pick() raised


def test_ctrl_c_in_the_picker_removes_the_export(fake, monkeypatch):
    exports = []
    real = picker.new_export_dir
    monkeypatch.setattr(picker, "new_export_dir", lambda: exports.append(real()) or exports[-1])

    def boom(out, limit=0, announce=None):
        (out / "01").mkdir()
        (out / "01" / "IMG.MOV").write_text("x")
        raise KeyboardInterrupt

    monkeypatch.setattr(picker, "pick", boom)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["photos", "pick"])
    assert not exports[0].exists()


def test_real_cut_end_to_end_with_the_fake_helper(fake, static_box, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FAKE_SRC", static_box)
    assert cli.main(["photos", "pick", "--", "--mode", "motion"]) == 0
    assert "clips in" in capsys.readouterr().out
    assert (tmp_path / "IMG_0001_inframe" / "motion" / "segments.json").is_file()


def test_override_must_be_executable(monkeypatch, tmp_path):
    monkeypatch.setenv(picker.HELPER_ENV, str(tmp_path / "missing"))
    with pytest.raises(picker.PickerError, match="not an executable"):
        picker.ensure_helper()


def test_missing_swiftc_message(monkeypatch, tmp_path):
    monkeypatch.delenv(picker.HELPER_ENV, raising=False)
    monkeypatch.setenv(picker.BIN_ENV, str(tmp_path))
    monkeypatch.setattr(picker.shutil, "which", lambda n: None)
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    with pytest.raises(picker.PickerError, match="xcode-select"):
        picker.ensure_helper()


def test_build_is_cached_and_keyed_on_the_source(monkeypatch, tmp_path):
    monkeypatch.delenv(picker.HELPER_ENV, raising=False)
    monkeypatch.setenv(picker.BIN_ENV, str(tmp_path / "bin"))
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    n = []

    def fake_run(cmd, **kw):
        n.append(cmd)
        open(cmd[cmd.index("-o") + 1], "w").write("#!/bin/sh\n")
        os.chmod(cmd[cmd.index("-o") + 1], 0o755)
        return type("R", (), dict(returncode=0, stdout="", stderr=""))()

    monkeypatch.setattr(picker.shutil, "which", lambda n_: "/usr/bin/swiftc")
    monkeypatch.setattr(picker.subprocess, "run", fake_run)
    a = picker.ensure_helper()
    b = picker.ensure_helper()
    assert a == b and a.parent == tmp_path / "bin" and len(n) == 1 and a.name.startswith("inframe-photo-picker-")


def test_build_failure_shows_the_compiler_output(monkeypatch, tmp_path):
    monkeypatch.delenv(picker.HELPER_ENV, raising=False)
    monkeypatch.setenv(picker.BIN_ENV, str(tmp_path / "bin"))
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    monkeypatch.setattr(picker.shutil, "which", lambda n_: "/usr/bin/swiftc")
    monkeypatch.setattr(picker.subprocess, "run", lambda *a, **k: type("R", (), dict(returncode=1, stdout="", stderr="error: boom"))())
    with pytest.raises(picker.PickerError, match="boom"):
        picker.ensure_helper()


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("swiftc"), reason="needs macOS and swiftc")
def test_swift_helper_compiles_and_reports_its_version(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.delenv(picker.HELPER_ENV, raising=False)
    monkeypatch.setenv(picker.BIN_ENV, str(tmp_path))
    exe = picker.ensure_helper()
    r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout.startswith("inframe-photo-picker ")
    assert subprocess.run([str(exe)], capture_output=True, text=True, timeout=30).returncode == 2  # --out is required


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("swiftc"), reason="needs macOS and swiftc")
def test_swift_export_copy_can_reuse_an_export_dir(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.delenv(picker.HELPER_ENV, raising=False)
    monkeypatch.setenv(picker.BIN_ENV, str(tmp_path / "bin"))
    exe = picker.ensure_helper()
    src = tmp_path / "IMG_0001.MOV"
    src.write_text("video")
    ex = tmp_path / "ex"
    cmd = [str(exe), "--out", str(ex), "--selftest-copy", str(src)]
    assert subprocess.run(cmd, capture_output=True, text=True, timeout=30).returncode == 0
    (ex / "01" / "stale.MOV").write_text("old pick")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # the second run, same export dir
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (ex / "01").iterdir()) == ["IMG_0001.MOV"]
