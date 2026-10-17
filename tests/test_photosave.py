"""T3b: save exported clips and reels to the Photos library (add-only). A fake helper (INFRAME_PHOTOS_SAVE_HELPER) stands in for
the Swift build, so no real Photos library is touched; the real helper is compiled once and only asked for --version, --status's
absence of side effects is not exercised, and a missing-argument call (exit 2)."""
import asyncio
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from inframe import photosave

FAKE = '''#!{py}
import json, os, sys
a = sys.argv[1:]
open(os.environ["FAKE_ARGS"], "w").write(json.dumps(a))
mode = os.environ.get("FAKE_MODE", "saved")
if mode == "garbage":
    print("not json at all"); sys.exit(0)
if mode == "hang":
    import time
    time.sleep(30)
if mode in ("slow", "crash_after"):
    import time
    for i, p in enumerate(a):
        print(json.dumps(dict(row=dict(path=p, saved=True))), flush=True)
        if mode == "crash_after" and i == 1:
            sys.exit(9)
        time.sleep(1)
    print(json.dumps(dict(authorization="authorized", status="saved", files=[dict(path=p, saved=True) for p in a], message="m")))
    sys.exit(0)
if mode == "silent":
    sys.exit(7)
if mode in ("denied", "restricted"):
    auth = mode
    msg = "Photos access was denied, so nothing was saved."
    print(json.dumps(dict(authorization=auth, status="denied", message=msg,
                          files=[dict(path=p, saved=False, error="no permission to add to Photos") for p in a])))
    sys.exit(3)
rows = []
for i, p in enumerate(a):
    bad = mode == "partial" and i == 1
    rows.append(dict(path=p, saved=not bad, **(dict(error="The operation couldn't be completed. (PHPhotosErrorDomain error 3302.)") if bad else {{}})))
n = sum(r["saved"] for r in rows)
print(json.dumps(dict(authorization="authorized", status="saved" if n == len(rows) else "partial" if n else "failed", files=rows, message="m")))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    p = tmp_path / "fake-save"
    p.write_text(FAKE.format(py=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(photosave.HELPER_ENV, str(p))
    args = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_ARGS", str(args))

    class F:
        def mode(self, m):
            monkeypatch.setenv("FAKE_MODE", m)

        def args(self):
            return json.loads(args.read_text())
    monkeypatch.setenv("FAKE_MODE", "saved")
    return F()


def videos(tmp_path, n=3, suffix=".mp4"):
    d = tmp_path / "out"
    d.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        f = d / f"clip_{i + 1:02d}_0001-0002{suffix}"
        f.write_bytes(b"x")
        out.append(f)
    return out


def test_saved_passes_every_path_and_reports_each_one(fake, tmp_path):
    vs = videos(tmp_path)
    msgs = []
    res = photosave.save(vs, announce=msgs.append)
    assert fake.args() == [str(v) for v in vs]
    assert res.status == "saved" and res.ok and res.saved == vs and res.failed == []
    assert photosave.describe(res) == "Saved 3 video(s) to the Photos library."
    assert any("add-only" in m for m in msgs)
    assert res.to_json()["saved"] == [v.name for v in vs]


def test_denied_is_a_result_not_an_exception_and_says_where_to_allow_it(fake, tmp_path):
    fake.mode("denied")
    res = photosave.save(videos(tmp_path))
    assert res.status == "denied" and not res.ok and res.saved == [] and len(res.failed) == 3
    text = photosave.describe(res)
    assert "denied" in text and "System Settings > Privacy & Security > Photos" in text and "nothing was saved" in text


def test_restricted_has_its_own_sentence(fake, tmp_path):
    fake.mode("restricted")
    res = photosave.save(videos(tmp_path))
    assert res.status == "denied" and res.authorization == "restricted"
    assert "restricted" in photosave.describe(res)


def test_partial_names_the_file_that_failed(fake, tmp_path):
    fake.mode("partial")
    vs = videos(tmp_path)
    res = photosave.save(vs)
    assert res.status == "partial" and not res.ok
    assert res.saved == [vs[0], vs[2]] and [p for p, _ in res.failed] == [vs[1]]
    text = photosave.describe(res)
    assert text.startswith("Saved 2 of 3 videos") and vs[1].name in text and "3302" in text
    assert res.to_json()["failed"][0]["name"] == vs[1].name


def test_a_file_that_is_not_there_is_failed_and_never_sent_to_the_helper(fake, tmp_path):
    vs = videos(tmp_path, 2)
    ghost = tmp_path / "out" / "gone.mp4"
    res = photosave.save([vs[0], ghost, vs[1]])
    assert fake.args() == [str(vs[0]), str(vs[1])]
    assert res.status == "partial" and [p for p, _ in res.failed] == [ghost] and res.failed[0][1] == "no such file"


def test_nothing_to_save_does_not_start_the_helper(fake, tmp_path):
    res = photosave.save([])
    assert res.status == "skipped" and res.ok
    assert not (tmp_path / "args.json").exists()
    only_ghosts = photosave.save([tmp_path / "nope.mp4"])
    assert only_ghosts.status == "failed" and not (tmp_path / "args.json").exists()


def test_a_timeout_reports_the_videos_already_added_as_interrupted_not_as_nothing_saved(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(photosave, "RUN_TIMEOUT_S", 2.5)
    fake.mode("slow")
    vs = videos(tmp_path, 5)
    res = photosave.save(vs)
    assert res.status == "interrupted" and not res.ok
    assert 1 <= len(res.saved) < 5 and res.saved == vs[:len(res.saved)]
    assert all("may or may not" in e for p, e in res.failed) and [p for p, _ in res.failed] == vs[len(res.saved):]
    text = photosave.describe(res)
    assert "Nothing was saved" not in text and f"{len(res.saved)} of 5" in text and "Check Photos before exporting again" in text


def test_a_timeout_before_any_file_is_done_still_says_it_may_have_saved_one(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(photosave, "RUN_TIMEOUT_S", 0.5)
    fake.mode("hang")
    res = photosave.save(videos(tmp_path, 2))
    assert res.status == "interrupted" and res.saved == [] and "may or may not" in photosave.describe(res)


def test_a_crash_partway_keeps_the_rows_that_were_printed(fake, tmp_path):
    fake.mode("crash_after")
    vs = videos(tmp_path, 4)
    res = photosave.save(vs)
    assert res.status == "interrupted" and res.saved == vs[:2] and [p for p, _ in res.failed] == vs[2:]
    assert "Nothing was saved" not in photosave.describe(res)


def test_a_helper_that_prints_nothing_or_garbage_is_a_clear_error(fake, tmp_path):
    vs = videos(tmp_path, 1)
    fake.mode("silent")
    with pytest.raises(photosave.PhotosSaveError, match="exit 7"):
        photosave.save(vs)
    fake.mode("garbage")
    with pytest.raises(photosave.PhotosSaveError, match="not its JSON"):
        photosave.save(vs)


def test_a_file_the_helper_does_not_mention_counts_as_failed(tmp_path):
    vs = videos(tmp_path, 2)
    out = json.dumps({"authorization": "authorized", "status": "saved", "message": "", "files": [{"path": str(vs[0]), "saved": True}]})
    res = photosave.parse_output(out + "\n", vs)
    assert res.status == "partial" and res.saved == [vs[0]] and res.failed[0][0] == vs[1]


def test_helper_override_must_be_executable(tmp_path, monkeypatch):
    f = tmp_path / "plain"
    f.write_text("x")
    monkeypatch.setenv(photosave.HELPER_ENV, str(f))
    with pytest.raises(photosave.PhotosSaveError, match="not an executable"):
        photosave.ensure_helper()


def test_available_follows_platform_or_override(monkeypatch):
    monkeypatch.delenv(photosave.HELPER_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert not photosave.available()
    monkeypatch.setenv(photosave.HELPER_ENV, "/x")
    assert photosave.available()


def test_videos_of_export_takes_cut_clips_and_a_good_reel_only(tmp_path):
    d = tmp_path
    for n in ("a_01.mp4", "a_02.mov", "a_03.mp4", "a_reel.mp4", "a_01.inframe.json", "a_04.webm"):
        (d / n).write_bytes(b"x")
    doc = {"clips": [{"file": "a_01.mp4"}, {"file": "a_02.mov"}, {"file": "a_03.mp4", "error": "cut failed"}, {"file": "gone.mp4"},
                     {"file": "a_04.webm"}, {"file": "a_05.mp4", "kept": False}, {"file": "../a_01.mp4"}],
           "reel": {"file": "a_reel.mp4", "ok": True}}
    assert [p.name for p in photosave.videos_of_export(d, doc)] == ["a_01.mp4", "a_02.mov", "a_reel.mp4"]
    doc["reel"]["ok"] = False
    assert [p.name for p in photosave.videos_of_export(d, doc)] == ["a_01.mp4", "a_02.mov"]


# ---------------------------------------------------------------- `inframe export --to-photos`
def run_cli(argv, capsys):
    from inframe import cli
    code = cli.main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def seg_run(tmp_path, n=2):
    d = tmp_path / "run"
    d.mkdir()
    clips = []
    for i in range(n):
        name = f"v_{i + 1:02d}_0001-0002.mp4"
        (d / name).write_bytes(b"x")
        clips.append({"index": i + 1, "file": name})
    clips.append({"index": 9, "file": "v_09_0001-0002.mp4", "error": "cut failed"})
    (d / "segments.json").write_text(json.dumps({"source": {"path": str(tmp_path / "v.mp4")}, "clips": clips}))
    return d


def test_cli_to_photos_alone_saves_the_clips(fake, tmp_path, capsys):
    d = seg_run(tmp_path)
    code, out, err = run_cli(["export", str(d), "--to-photos"], capsys)
    assert code == 0, err
    assert fake.args() == [str(d / "v_01_0001-0002.mp4"), str(d / "v_02_0001-0002.mp4")]
    assert "Saved 2 video(s) to the Photos library." in out
    assert not (d / "export.json").exists()  # nothing else was asked for


def test_cli_to_photos_denied_exits_1_and_says_why(fake, tmp_path, capsys):
    fake.mode("denied")
    code, out, err = run_cli(["export", str(seg_run(tmp_path)), "--to-photos"], capsys)
    assert code == 1
    assert "Photos permission was denied" in err and "System Settings" in err


def test_cli_to_photos_partial_exits_1_and_lists_the_failed_file(fake, tmp_path, capsys):
    fake.mode("partial")
    code, out, err = run_cli(["export", str(seg_run(tmp_path, 3)), "--to-photos"], capsys)
    assert code == 1
    assert "Saved 2 of 3" in err and "not saved: v_02_0001-0002.mp4" in err


def test_cli_to_photos_with_the_reel_saves_clips_then_reel(fake, tmp_path, capsys, monkeypatch):
    d = seg_run(tmp_path)

    def fake_export_run(doc, clips_dir, out, reel, formats, track, beat):
        (out / "v_reel.mp4").write_bytes(b"x")
        return {"ok": True, "reel": {"file": "v_reel.mp4", "ok": True, "n_clips": 2, "check": {"duration_s": 2.0}}, "edit_lists": [], "warnings": []}
    monkeypatch.setattr("inframe.export.export_run", fake_export_run)
    code, out, err = run_cli(["export", str(d), "--reel", "--to-photos"], capsys)
    assert code == 0, err
    assert [Path(p).name for p in fake.args()] == ["v_01_0001-0002.mp4", "v_02_0001-0002.mp4", "v_reel.mp4"]


def test_cli_to_photos_missing_helper_is_one_line_exit_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(photosave.HELPER_ENV, str(tmp_path / "nope"))
    code, out, err = run_cli(["export", str(seg_run(tmp_path)), "--to-photos"], capsys)
    assert code == 1 and "--to-photos:" in err and "not an executable" in err


def test_cli_still_refuses_an_export_with_nothing_to_do(tmp_path, capsys):
    code, out, err = run_cli(["export", str(seg_run(tmp_path))], capsys)
    assert code == 2 and "--to-photos" in err


# ---------------------------------------------------------------- the MCP export_reel tool
def test_mcp_export_reel_to_photos_reports_saved_denied_and_partial(fake, tmp_path, monkeypatch):
    from inframe.mcp_server import export_reel
    d = seg_run(tmp_path, 3)

    def fake_export_run(doc, clips_dir, out, reel, formats, track, beat):
        (out / "v_reel.mp4").write_bytes(b"x")
        return {"ok": True, "reel": {"file": "v_reel.mp4", "ok": True, "n_clips": 2}, "edit_lists": [], "warnings": []}
    monkeypatch.setattr("inframe.export.export_run", fake_export_run)

    r = asyncio.run(export_reel(str(d), to_photos=True))
    assert r["photos"]["status"] == "saved" and len(r["photos"]["saved"]) == 4 and r["warnings"] == []
    assert fake.args()[-1].endswith("v_reel.mp4")

    fake.mode("denied")
    r = asyncio.run(export_reel(str(d), to_photos=True))
    assert r["ok"] is True and r["photos"]["status"] == "denied" and "System Settings" in r["photos"]["message"]
    assert r["photos"]["message"] in r["warnings"]

    fake.mode("partial")
    r = asyncio.run(export_reel(str(d), to_photos=True))
    assert r["photos"]["status"] == "partial" and len(r["photos"]["failed"]) == 1

    r = asyncio.run(export_reel(str(d)))
    assert "photos" not in r


def test_mcp_export_reel_to_photos_without_a_helper_is_reported_not_raised(tmp_path, monkeypatch):
    from inframe.mcp_server import export_reel
    d = seg_run(tmp_path)
    monkeypatch.setenv(photosave.HELPER_ENV, str(tmp_path / "nope"))
    monkeypatch.setattr("inframe.export.export_run", lambda *a, **k: {"ok": True, "reel": {"file": None, "ok": False, "problem": "x"}, "edit_lists": [], "warnings": []})
    r = asyncio.run(export_reel(str(d), to_photos=True))
    assert r["photos"]["status"] == "failed" and "Nothing was saved to Photos" in r["photos"]["message"]


# ---------------------------------------------------------------- the real Swift helper: compiled once, never run against Photos
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("swiftc"), reason="needs macOS and swiftc")
def test_the_real_helper_compiles_with_the_usage_description_and_refuses_bad_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv(photosave.BIN_ENV, str(tmp_path / "bin"))
    monkeypatch.delenv(photosave.HELPER_ENV, raising=False)
    exe = photosave.ensure_helper()
    assert exe.parent == tmp_path / "bin" and exe.name.startswith("inframe-photo-save-")
    assert subprocess.run([str(exe), "--version"], capture_output=True, text=True).stdout.strip() == "inframe-photo-save 2"
    r = subprocess.run([str(exe), "--nope"], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown argument" in r.stderr
    r = subprocess.run([str(exe)], capture_output=True, text=True)  # no files: refused before any Photos call
    assert r.returncode == 2 and "no files" in r.stderr
    plist = subprocess.run(["otool", "-s", "__TEXT", "__info_plist", str(exe)], capture_output=True, text=True).stdout
    assert "__info_plist" in plist and "0x" not in plist.splitlines()[0]  # the section exists (macOS refuses to ask without NSPhotoLibraryAddUsageDescription)
    assert photosave.ensure_helper() == exe  # cached: no second build


def test_every_surface_warns_that_icloud_photos_uploads_the_footage(capsys):
    from inframe import cli
    from inframe.mcp_server import export_reel
    with pytest.raises(SystemExit):
        cli.main(["export", "--help"])
    assert "iCloud Photos on, Apple uploads" in " ".join(capsys.readouterr().out.split())
    doc = " ".join((export_reel.__doc__ or "").split())
    assert "iCloud Photos on, Apple uploads" in doc and "leaves this Mac" in doc and "without any click" in doc
