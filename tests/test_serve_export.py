"""inframe serve, export (docs/T9A_DESIGN.md rows 42-45 and section 2.7): the edited clip list is cut into `<run>/edited/` with
sidecars, a reel and edit lists; the detector's segments.json is never touched; earlier exports are cleared by name pattern only.
Runs are real (motion mode on the synthetic `static_box` clip, no model); music is a generated wav or a monkeypatched `music.get`."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from serve_helpers import call, start_server

from inframe import music, serve_export

ROOT = Path(__file__).resolve().parents[1]
ADV = {"fps": 5.0, "pad": 0.5, "min_len": 1.0}
GOP_S = 1.0  # static_box: 10 fps, keyframe every 10 frames


def wait_job(app, jid, timeout=180.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = app.jobs.get(jid)
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job still running")


def duration(path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, static_box):
    vids = tmp_path / "vids"
    vids.mkdir()
    video = vids / "clip.mp4"
    shutil.copy(static_box, video)
    base, app, stop = start_server(tmp_path)
    media = app.open_media(str(video))
    r = call(app, "POST", "/api/runs", {"media_id": media.id, "target": {"kind": "motion"}, "advanced": ADV})
    assert r.status == 200, r.text
    rid = r.json()["run_id"]
    assert wait_job(app, r.json()["job"]["id"])["state"] == "done"
    run_dir = vids / "clip_inframe" / "motion-ui"
    seg = json.loads((run_dir / "segments.json").read_text())

    def split():
        """Edit the run's one clip into two halves and save them (revision 0 -> 1)."""
        c = seg["clips"][0]
        mid = round((c["start"] + c["end"]) / 2, 3)
        clips = [{"id": "d1", "start": c["start"], "end": mid, "origin": "edited"},
                 {"id": "u1", "start": mid, "end": c["end"], "origin": "split"}]
        p = call(app, "PUT", f"/api/runs/{rid}/edits", {"revision": 0, "clips": clips})
        assert p.status == 200, p.text
        return clips

    def export(body=None, expect=200):
        b = {"clips": True, "reel": False, "edit_lists": [], **(body or {})}
        r = call(app, "POST", f"/api/runs/{rid}/export", b)
        assert r.status == expect, r.text
        return r

    def done(body=None):
        r = export(body)
        j = wait_job(app, r.json()["job"]["id"])
        assert j["state"] == "done", j
        return j["result"]

    yield SimpleNamespace(app=app, media=media, rid=rid, run_dir=run_dir, seg=seg, split=split, export=export, done=done,
                          edited=run_dir / "edited", tmp=tmp_path, vids=vids)
    stop()


# ---------------------------------------------------------------- row 42: the whole export
def test_split_clip_exports_clips_reel_and_edit_lists(env):
    seg_bytes = (env.run_dir / "segments.json").read_bytes()
    env.split()
    res = env.done({"reel": True, "edit_lists": ["fcpxml", "otio"]})
    ed = env.edited
    clips = sorted(p for p in ed.glob("clip_0*.mp4"))
    assert [p.name for p in clips] == [c["file"] for c in res["clips"]] and len(clips) == 2
    for p, c in zip(clips, res["clips"]):  # stream copy starts at the keyframe before the clip: longer by less than a GOP
        want = c["end"] - c["start"]
        assert want - 0.35 <= duration(p) <= want + GOP_S + 0.35, (p.name, duration(p), want)
    assert res["reel"]["ok"] is True and res["reel"]["check"]["ok"] is True and (ed / "clip_reel.mp4").is_file()
    assert [(e["format"], e["ok"]) for e in res["edit_lists"]] == [("fcpxml", True), ("otio", True)]
    assert (ed / "clip.fcpxml").is_file() and (ed / "clip.otio").is_file() and (ed / "clip_losslesscut.csv").is_file()
    doc = json.loads((ed / "segments.json").read_text())
    assert len(doc["clips"]) == 2 and doc["edited"]["revision"] == 1 and doc["edited"]["from"] == "../segments.json"
    assert doc["edited"]["segments_sha256"] == sha(env.run_dir / "segments.json")
    assert [c["index"] for c in doc["clips"]] == [1, 2] and all(c["expected_copy_start"] <= c["start"] for c in doc["clips"])
    assert doc["presence_intervals"] == [[c["start"], c["end"]] for c in doc["clips"]]
    assert doc["reel"]["ok"] is True and len(doc["edit_lists"]) == 2
    assert all(c["sidecar"].endswith(".inframe.json") for c in doc["clips"])
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "validate_sidecars.py"), str(ed)], capture_output=True, text=True)
    assert r.stdout.strip().splitlines()[-1].startswith("PASS"), r.stdout + r.stderr
    assert (env.run_dir / "segments.json").read_bytes() == seg_bytes  # the detector's record is never written
    kinds = [o["kind"] for o in res["outputs"]]
    assert kinds == ["clip", "clip", "sidecar", "sidecar", "reel", "csv", "editlist", "editlist"]
    first = res["outputs"][0]
    assert first["url"] == f"api/runs/{env.rid}/files/{first['name']}" and first["size"] == clips[0].stat().st_size
    assert [o["url"] for o in res["outputs"] if o["kind"] in ("sidecar", "csv", "editlist")] == [""] * 5
    assert json.loads((ed / "export.json").read_text()) == res
    g = call(env.app, "GET", f"/api/runs/{env.rid}/export")
    assert g.status == 200 and g.json() == res
    assert call(env.app, "GET", f"/api/runs/{env.rid}").json()["export"] == res


def test_export_of_an_unedited_run_uses_the_detector_clips(env):
    res = env.done({"edit_lists": ["fcpxml"]})
    assert res["revision"] == 0 and [(c["start"], c["end"]) for c in res["clips"]] == [(env.seg["clips"][0]["start"], env.seg["clips"][0]["end"])]
    assert call(env.app, "GET", f"/api/runs/{env.rid}/export").status == 200


def test_clip_ids_export_only_those_clips(env):
    env.split()
    res = env.done({"clip_ids": ["u1"]})
    assert res["n_clips"] == 1 and len(list(env.edited.glob("clip_0*.mp4"))) == 1
    assert res["clips"][0]["file"].startswith("clip_01_")  # numbered from 1 in the export
    mid = json.loads((env.run_dir / "edits.json").read_text())["clips"][1]
    assert (res["clips"][0]["start"], res["clips"][0]["end"]) == (mid["start"], mid["end"])


def test_a_second_export_needs_replace(env):
    env.split()
    first = env.done()
    r = env.export(expect=409)
    assert r.json()["error"] == "exists" and r.json()["export"] == first
    assert env.done({"replace": True})["n_clips"] == 2


def test_replace_clears_by_pattern_and_keeps_everything_else(env):
    env.split()
    env.done({"reel": True})
    ed = env.edited
    (ed / "notes.txt").write_text("mine")
    (ed / "other_01_0001-0002.mp4").write_bytes(b"another video's clip")
    (ed / "clip_extra.mp4").write_bytes(b"not a clip name")
    old = ed / "clip_09_0000-0001.mov"  # a clip name of this video from some earlier cut
    old.write_bytes(b"old")
    res = env.done({"replace": True, "reel": False})
    assert (ed / "notes.txt").read_text() == "mine" and (ed / "other_01_0001-0002.mp4").is_file() and (ed / "clip_extra.mp4").is_file()
    assert not old.exists() and not (ed / "clip_reel.mp4").exists()  # this export has no reel, the old one went
    assert [o["kind"] for o in res["outputs"]].count("clip") == 2


def test_symlink_or_folder_with_an_output_name_blocks_and_nothing_is_removed(env, tmp_path):
    env.split()
    env.done()
    ed = env.edited
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    reel = ed / "clip_reel.mp4"
    reel.symlink_to(victim)
    before = sorted(os.listdir(ed))
    r = env.export({"replace": True}, expect=422)
    assert "clip_reel.mp4" in r.json()["error"] and sorted(os.listdir(ed)) == before and victim.read_text() == "keep"
    reel.unlink()
    (ed / "clip.otio").mkdir()
    assert "clip.otio" in env.export({"replace": True}, expect=422).json()["error"]


def test_a_cut_that_fails_is_reported_and_the_export_goes_on(env, monkeypatch):
    env.split()
    real = serve_export.cut.cut_clip
    calls = []

    def flaky(src, out, *a, **kw):
        calls.append(out)
        if len(calls) == 1:
            raise RuntimeError("ffmpeg cut failed (test)")
        return real(src, out, *a, **kw)

    monkeypatch.setattr(serve_export.cut, "cut_clip", flaky)
    res = env.done({"edit_lists": ["fcpxml"]})
    assert res["clips"][0]["error"] == "ffmpeg cut failed (test)" and "error" not in res["clips"][1]
    assert "1 of 2 clips could not be cut" in res["warnings"]
    names = [o["name"] for o in res["outputs"] if o["kind"] == "clip"]
    assert names == [res["clips"][1]["file"]] and len(list(env.edited.glob("*.inframe.json"))) == 1
    assert "1 of 2" not in (env.edited / "clip_losslesscut.csv").read_text() and len((env.edited / "clip_losslesscut.csv").read_text().splitlines()) == 1


def test_edit_lists_only_cuts_nothing(env):
    env.split()
    res = env.done({"clips": False, "edit_lists": ["fcpxml", "otio"]})
    assert not list(env.edited.glob("*.mp4")) and not list(env.edited.glob("*.inframe.json"))
    assert (env.edited / "clip.fcpxml").is_file() and sorted(o["kind"] for o in res["outputs"]) == ["editlist", "editlist"]


def test_reel_only_leaves_no_clip_files(env):
    env.split()
    res = env.done({"clips": False, "reel": True})
    assert res["reel"]["ok"] is True and (env.edited / "clip_reel.mp4").is_file()
    assert sorted(p.name for p in env.edited.glob("clip_0*")) == [] and sorted(env.edited.glob("*.inframe.json")) == []
    assert not (env.edited / "clip_losslesscut.csv").exists()
    assert [o["kind"] for o in res["outputs"]] == ["reel"]


def test_reel_only_segments_and_export_json_name_no_missing_clip_files(env):
    env.split()
    res = env.done({"clips": False, "reel": True})
    for doc in (res, json.loads((env.edited / "segments.json").read_text())):
        # the clip was cut and is a real clip of the run: only its file is gone, so nothing marks it as failed
        assert [c.get("file") for c in doc["clips"]] == [None, None] and all("error" not in c and c.get("kept") is False for c in doc["clips"]), doc["clips"]
    names = [c.get("file") for c in json.loads((env.edited / "export.json").read_text())["clips"]]
    assert names == [None, None] and sorted(p.name for p in env.edited.glob("clip_0*")) == []


def test_reel_only_folder_still_gives_edit_lists_and_an_honest_later_reel(env):
    from inframe import editlist, export as export_mod

    env.split()
    env.done({"clips": False, "reel": True})
    doc = json.loads((env.edited / "segments.json").read_text())
    cuts = editlist.cuts(doc)  # edit lists point at the source video, not at clip files
    assert len(cuts) == 2 and all(c["name"].startswith("clip_0") for c in cuts), cuts
    r = export_mod.export_run(doc, env.edited, env.tmp / "again", True, ["fcpxml"])
    assert [(e["format"], e["ok"]) for e in r["edit_lists"]] == [("fcpxml", True)], r
    assert r["reel"]["ok"] is False and r["reel"]["file"] is None
    assert not any("could not be cut" in w for w in r["warnings"]), r["warnings"]
    assert any("clip file(s) missing" in w for w in r["warnings"]), r["warnings"]


def test_reel_only_keeps_the_clips_when_the_reel_failed(env, monkeypatch):
    from inframe import export as export_mod

    env.split()
    monkeypatch.setattr(export_mod, "build_reel", lambda *a, **k: dict(file=None, ok=False, fallback_used=False, check=dict(problems=["boom"])))
    res = env.done({"clips": False, "reel": True})
    assert res["reel"]["ok"] is False
    kept = sorted(p.name for p in env.edited.glob("clip_0*.mp4"))
    assert len(kept) == 2, kept
    seg = json.loads((env.edited / "segments.json").read_text())
    assert [c["file"] for c in seg["clips"]] == kept and all("error" not in c for c in seg["clips"])
    assert any("kept" in w for w in res["warnings"])


def test_slow_motion_run_retimes_the_edited_clips(env):
    doc = env.seg
    doc["slowmo"] = {"requested": "2", "factor": 2.0, "capture_fps": 20.0}
    (env.run_dir / "segments.json").write_text(json.dumps(doc))
    env.split()
    res = env.done()
    sd = json.loads((env.edited / "segments.json").read_text())
    for c in sd["clips"]:
        assert c["slowmo"]["factor"] == 2.0 and c["slowmo"]["source_t0"] == c["expected_copy_start"]
        assert abs(duration(env.edited / c["file"]) - c["slowmo"]["duration_s"]) < 0.05
        assert c["slowmo"]["duration_s"] > 2 * (c["end"] - c["start"]) - 0.5
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "validate_sidecars.py"), str(env.edited)], capture_output=True, text=True)
    assert r.stdout.strip().splitlines()[-1].startswith("PASS"), r.stdout
    assert res["n_clips"] == 2


def test_custom_out_dir(env, tmp_path):
    env.split()
    dest = tmp_path / "chosen" / "deliver"
    res = env.done({"out_dir": str(dest), "edit_lists": ["fcpxml"]})
    assert res["dir"] == str(dest) and (dest / "export.json").is_file() and not env.edited.exists()
    doc = json.loads((dest / "segments.json").read_text())
    assert doc["edited"]["from"] == str(env.run_dir / "segments.json")
    assert call(env.app, "GET", f"/api/runs/{env.rid}/export").json() == res  # the last export of this run
    first = res["clips"][0]["file"]
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{first}", key="media").status == 200
    again = env.export(expect=200)  # the default folder is free, so no 'exists'
    wait_job(env.app, again.json()["job"]["id"])


def test_files_route_after_a_restart_serves_the_custom_folders_clip_not_the_detectors(env, tmp_path):
    """The edited clip has the same name as the detector's clip (names count whole seconds) but other boundaries; the folder it
    went to is not the default one, and a restart forgets the in-memory record of it."""
    c = env.seg["clips"][0]
    edited = [{"id": "u1", "start": round(c["start"] + 0.2, 3), "end": round(c["end"] - 0.2, 3), "origin": "edited"}]
    assert call(env.app, "PUT", f"/api/runs/{env.rid}/edits", {"revision": 0, "clips": edited}).status == 200
    dest = tmp_path / "chosen"
    res = env.done({"out_dir": str(dest)})
    name = res["clips"][0]["file"]
    assert name == c["file"] and (dest / name).read_bytes() != (env.run_dir / name).read_bytes()
    serve_export._dirs.pop(env.app, None)  # what a restart does
    r = call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}", key="media")
    assert r.status == 200 and r.body == (dest / name).read_bytes()
    assert call(env.app, "GET", f"/api/runs/{env.rid}/export").json() == res


def test_out_dir_rules(env, tmp_path):
    env.split()
    assert env.export({"out_dir": str(env.run_dir)}, expect=422).json()["error"].endswith("Choose another folder.")
    assert env.export({"out_dir": str(tmp_path / ".hidden" / "x")}, expect=403)
    assert env.export({"out_dir": "relative/dir"}, expect=400)
    assert env.export({"out_dir": 5}, expect=400)
    f = tmp_path / "afile"
    f.write_text("x")
    assert env.export({"out_dir": str(f)}, expect=403)
    real = tmp_path / "real"
    real.mkdir()
    (env.run_dir / "edited").symlink_to(real)
    assert "not a plain folder" in env.export(expect=422).json()["error"]
    assert not list(real.iterdir())


def test_bad_requests_are_refused_before_a_job_starts(env):
    env.split()
    for body, code in [({"clip_ids": []}, 422), ({"clip_ids": ["zz"]}, 422), ({"clip_ids": "d1"}, 400), ({"clip_ids": [1]}, 400),
                       ({"clips": False}, 422), ({"clips": "yes"}, 400), ({"edit_lists": ["pdf"]}, 422), ({"edit_lists": "edl"}, 400),
                       ({"music": {"track_id": "abc"}}, 422), ({"reel": True, "music": {}}, 422), ({"reel": True, "music": 5}, 400),
                       ({"reel": True, "music": {"track_id": "bad id!"}}, 422), ({"reel": True, "music": {"track_id": "0123abcd4567", "file": "/x.wav"}}, 422),
                       ({"reel": True, "music": {"file": "/no/such.wav"}}, 404), ({"beat_sync": True}, 422),
                       ({"reel": True, "music": {"track_id": "0123abcd4567", "cc0_only": "no"}}, 400)]:
        b = {"clips": True, "edit_lists": [], **body}
        r = call(env.app, "POST", f"/api/runs/{env.rid}/export", b)
        assert r.status == code, (body, r.status, r.text)
    assert env.app.jobs.current() is None and not env.edited.exists()
    assert call(env.app, "POST", "/api/runs/000000000000/export", {"clips": True}).status == 404


def test_a_failed_run_is_not_exported(env):
    doc = env.seg
    doc["error"] = "stopped during the cuts"
    (env.run_dir / "segments.json").write_text(json.dumps(doc))
    assert "nothing to export" in env.export(expect=422).json()["error"]


def test_export_while_a_job_runs_is_busy(env):
    import threading

    gate = threading.Event()
    job = env.app.jobs.start("run", lambda progress: gate.wait(20))
    try:
        r = env.export(expect=409)
        assert r.json()["error"] == "busy" and r.json()["job"]["id"] == job["id"]
    finally:
        gate.set()
        wait_job(env.app, job["id"])
    assert not env.edited.exists()


def test_cancel_during_the_cuts_ends_cancelled(env, monkeypatch):
    env.split()
    real = serve_export.cut.cut_clip

    def slow(*a, **kw):
        real(*a, **kw)
        time.sleep(0.5)

    monkeypatch.setattr(serve_export.cut, "cut_clip", slow)
    j = env.export().json()["job"]
    assert call(env.app, "POST", f"/api/jobs/{j['id']}/cancel", {}).status == 200
    assert wait_job(env.app, j["id"])["state"] == "cancelled"
    assert not (env.edited / "export.json").exists()


# ---------------------------------------------------------------- music
@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "tune.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "sine=frequency=440:duration=6", str(p)], check=True)
    return p


def test_reel_with_a_music_file(env, wav):
    env.split()
    res = env.done({"reel": True, "music": {"file": str(wav)}})
    assert res["reel"]["ok"] is True and res["reel"]["music"]["file"] == "tune.wav"
    assert (env.edited / "clip_reel.mp4").is_file()


def test_a_music_file_obeys_cc0_only_and_refuses_other_licenses(env, wav):
    env.split()
    meta = dict(id="0123abcd4567", title="Long Road", creator="B. Player", license="by", license_version="4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/")
    wav.with_suffix(".json").write_text(json.dumps(meta))
    r = env.export({"reel": True, "music": {"file": str(wav), "cc0_only": True}})  # the "CC0 only" box refuses a CC BY file you chose yourself
    j = wait_job(env.app, r.json()["job"]["id"])
    assert j["state"] == "failed" and "CC0" in j["error"], j
    assert not (env.edited / "clip_reel.mp4").exists()
    for lic in ("by-sa", "by-nd", "by-nc"):  # these are refused with or without the box
        wav.with_suffix(".json").write_text(json.dumps(dict(meta, license=lic)))
        for only in (False, True):
            r = env.export({"replace": True, "reel": True, "music": {"file": str(wav), "cc0_only": only}})
            j = wait_job(env.app, r.json()["job"]["id"])
            assert j["state"] == "failed" and "CC0 and CC BY" in j["error"], (lic, only, j)
    assert not (env.edited / "clip_reel.mp4").exists() and not list(env.edited.glob("*credits*"))
    wav.with_suffix(".json").write_text(json.dumps(meta))
    res = env.done({"replace": True, "reel": True, "music": {"file": str(wav)}})  # CC BY file without the box: credit written
    assert res["reel"]["ok"] is True and res["reel"]["music"]["credits"] == "clip_credits.txt"


def test_a_music_error_costs_nothing(env, monkeypatch):
    env.split()
    first = env.done({"reel": True})

    def refuse(track_id, cc0_only=False):
        raise music.MusicError("music: Openverse is rate-limiting (test)")

    monkeypatch.setattr(music, "get", refuse)
    r = env.export({"replace": True, "reel": True, "music": {"track_id": "0123abcd4567"}})
    j = wait_job(env.app, r.json()["job"]["id"])
    assert j["state"] == "failed" and "rate-limiting" in j["error"]
    assert json.loads((env.edited / "export.json").read_text()) == first and (env.edited / "clip_reel.mp4").is_file()


def test_a_music_track_id_goes_through_music_get(env, wav, monkeypatch):
    env.split()
    seen = []

    def fake_get(track_id, cc0_only=False):
        seen.append((track_id, cc0_only))
        return wav

    monkeypatch.setattr(music, "get", fake_get)
    res = env.done({"reel": True, "music": {"track_id": "0123abcd4567", "cc0_only": True}})
    assert seen == [("0123abcd4567", True)] and res["reel"]["ok"] is True
    env.done({"replace": True, "reel": True, "music": {"track_id": "0123abcd4567"}})
    assert seen[-1] == ("0123abcd4567", False)  # CC BY is allowed unless the request says cc0_only


def test_a_cc_by_track_writes_the_credits_file_and_returns_the_credit_without_any_flag(env, wav, monkeypatch):
    env.split()
    wav.with_suffix(".json").write_text(json.dumps(dict(id="0123abcd4567", title="Long Road", creator="B. Player", license="by", license_version="4.0",
                                                         license_url="https://creativecommons.org/licenses/by/4.0/", attribution='"Long Road" by B. Player')))
    monkeypatch.setattr(music, "get", lambda track_id, cc0_only=False: wav)
    res = env.done({"reel": True, "music": {"track_id": "0123abcd4567"}})
    m = res["reel"]["music"]
    assert res["reel"]["ok"] is True and m["license"] == "by" and m["credits"] == "clip_credits.txt"
    assert "Long Road" in m["credit_text"] and "CC BY 4.0" in m["credit_text"]
    assert (env.edited / "clip_credits.txt").read_text().strip() == m["credit_text"]
    assert [o["kind"] for o in res["outputs"]].count("credits") == 1
    wav.with_suffix(".json").write_text(json.dumps(dict(id="0123abcd4567", title="Free", creator="C", license="cc0")))
    res2 = env.done({"replace": True, "reel": True, "music": {"track_id": "0123abcd4567"}})
    assert res2["reel"]["music"]["credits"] is None and res2["reel"]["music"]["credit_text"] is None
    assert not (env.edited / "clip_credits.txt").exists() and not [o for o in res2["outputs"] if o["kind"] == "credits"]


# ---------------------------------------------------------------- row 44: files
def test_files_route_serves_ranges_and_only_listed_names(env):
    env.split()
    res = env.done({"reel": True, "edit_lists": ["fcpxml"]})
    name = res["clips"][0]["file"]
    path = env.edited / name
    size = path.stat().st_size
    full = call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}", key="media")
    assert full.status == 200 and full.body == path.read_bytes() and full.headers["content-type"] == "video/mp4"
    part = call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}", headers={"Range": "bytes=0-99"}, key="media")
    assert part.status == 206 and part.body == path.read_bytes()[:100] and part.headers["content-range"] == f"bytes 0-99/{size}"
    tail = call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}", headers={"Range": "bytes=-50"}, key="media")
    assert tail.status == 206 and tail.body == path.read_bytes()[-50:]
    head = call(env.app, "HEAD", f"/api/runs/{env.rid}/files/{name}", key="media")
    assert head.status == 200 and head.headers["content-length"] == str(size) and head.headers["accept-ranges"] == "bytes"
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/clip_reel.mp4", key="media").status == 200
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}").status == 200  # the session token works as well
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}", key=None).status == 403
    for n in ("nothing_01_0000-0001.mp4", "clip_99_0000-0001.mp4", "clip.fcpxml", "segments.json", "export.json", "clip_losslesscut.csv",
              res["outputs"][2]["name"]):  # not in the folder, or not a clip/reel/credits name
        assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{n}", key="media").status == 404, n
    for n in ("../segments.json", "..%2Fsegments.json", "-x.mp4", ".hidden", "a%5Cb.mp4", "%2e%2e", "x%00.mp4", "sub/clip.mp4", "a" * 300 + ".mp4"):
        assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{n}", key="media").status == 400, n
    assert call(env.app, "GET", "/api/runs/000000000000/files/x.mp4", key="media").status == 404


def test_files_route_does_not_follow_a_symlink_with_a_clip_name(env, tmp_path):
    env.split()
    res = env.done()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    name = "clip_07_0000-0001.mov"
    (env.edited / name).symlink_to(secret)
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{name}", key="media").status == 404
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{res['clips'][0]['file']}", key="media").status == 200


def test_files_route_also_serves_the_run_folders_own_clips(env):
    own = env.seg["clips"][0]["file"]
    assert (env.run_dir / own).is_file()
    assert call(env.app, "GET", f"/api/runs/{env.rid}/files/{own}", key="media").status == 200


def test_credits_file_is_text(env):
    env.split()
    env.done()
    (env.edited / "clip_credits.txt").write_text('Music: "x"\n')
    r = call(env.app, "GET", f"/api/runs/{env.rid}/files/clip_credits.txt", key="media")
    assert r.status == 200 and r.headers["content-type"].startswith("text/plain") and r.text == 'Music: "x"\n'


# ---------------------------------------------------------------- row 45: reveal
def test_reveal_passes_an_absolute_path_after_double_dash(env, monkeypatch):
    env.split()
    res = env.done({"edit_lists": ["fcpxml"]})
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(serve_export.subprocess, "run", fake_run)
    clip = res["clips"][0]["file"]
    monkeypatch.setattr(serve_export, "_platform", lambda: "darwin")
    assert call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {"name": clip}).json() == {"ok": True}
    assert cmds[-1] == ["open", "-R", "--", str((env.edited / clip).resolve())] and os.path.isabs(cmds[-1][-1])
    call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {"name": "clip.fcpxml"})  # every output can be shown, not only the served ones
    assert cmds[-1] == ["open", "-R", "--", str((env.edited / "clip.fcpxml").resolve())]
    call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {"name": None})
    assert cmds[-1] == ["open", "-R", "--", str(env.edited.resolve())]  # a folder is shown, never opened (a *.app would start)
    monkeypatch.setattr(serve_export, "_platform", lambda: "linux")
    call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {"name": clip})
    assert cmds[-1] == ["xdg-open", str(env.edited.resolve())]
    monkeypatch.setattr(serve_export, "_platform", lambda: "win32")
    call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {"name": clip})
    assert cmds[-1] == ["explorer", f"/select,{(env.edited / clip).resolve()}"]
    n = len(cmds)
    for bad in ("../segments.json", "-x.mp4", ".hidden", "nothing_01_0000-0001.mp4", "segments.json", "a/b"):
        r = call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {"name": bad})
        assert r.status in (400, 404), (bad, r.status)
    assert len(cmds) == n and call(env.app, "POST", "/api/runs/000000000000/reveal", {}).status == 404


def test_reveal_never_opens_a_folder_that_is_a_bundle(env, monkeypatch):
    """A shared `<stem>_inframe/Evil.app/segments.json` is listed as a run; `open <dir>` would launch it on macOS."""
    evil = env.run_dir.parent / "Evil.app"
    shutil.copytree(env.run_dir, evil)
    ids = {r["run_id"]: r["dir"] for r in call(env.app, "GET", f"/api/media/{env.media.id}/runs").json()["runs"]}
    rid = next(k for k, v in ids.items() if v.endswith("Evil.app"))
    cmds = []
    monkeypatch.setattr(serve_export.subprocess, "run", lambda cmd, **kw: cmds.append(cmd) or SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr(serve_export, "_platform", lambda: "darwin")
    assert call(env.app, "POST", f"/api/runs/{rid}/reveal", {"name": None}).status == 200
    assert cmds and cmds[-1][:3] == ["open", "-R", "--"], cmds


def test_reveal_failures(env, monkeypatch):
    env.split()
    env.done()
    monkeypatch.setattr(serve_export, "_platform", lambda: "linux")

    def missing(cmd, **kw):
        raise FileNotFoundError("xdg-open")

    monkeypatch.setattr(serve_export.subprocess, "run", missing)
    assert call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {}).status == 501
    monkeypatch.setattr(serve_export.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=3, stderr="boom", stdout=""))
    r = call(env.app, "POST", f"/api/runs/{env.rid}/reveal", {})
    assert r.status == 500 and "boom" in r.json()["error"]


# ---------------------------------------------------------------- GET export
def test_get_export_before_any_export_is_404(env):
    assert call(env.app, "GET", f"/api/runs/{env.rid}/export").status == 404


def test_name_rule_unit():
    for ok in ("clip_01_0001-0004.mp4", "a b.mp4", "ü.mov"):
        assert serve_export.check_name(ok) == ok
    for bad in ("", ".", "..", "../x", "a/b", "a\\b", "-x", ".x", "x\0", "é" * 200):
        with pytest.raises(Exception):
            serve_export.check_name(bad)


def test_name_patterns_unit():
    p = serve_export.name_patterns("IMG_0001")
    assert serve_export.classify("IMG_0001_01_0018-0036.mov", p) == "clip"
    assert serve_export.classify("IMG_0001_12_0018-0036.inframe.json", p) == "sidecar"
    assert serve_export.classify("IMG_0001_reel.mov", p) == "reel" and serve_export.classify("IMG_0001_credits.txt", p) == "credits"
    assert serve_export.classify("IMG_0001_losslesscut.csv", p) == "csv"
    assert [serve_export.classify(f"IMG_0001{e}", p) for e in (".fcpxml", "_fcp7.xml", ".kdenlive", ".edl", ".otio")] == ["editlist"] * 5
    for other in ("IMG_0002_01_0018-0036.mov", "IMG_0001.mov", "notes.txt", "IMG_0001_01_0018-0036.part.mov", "xIMG_0001_reel.mov", "segments.json"):
        assert serve_export.classify(other, p) is None, other


# ---------------------------------------------------------------- auto music (a job)
@pytest.fixture
def auto_stub(tmp_path, monkeypatch):
    """Openverse stub (rows A, B, a by-sa row and a cc0 row) with synthetic beat audio; nothing reaches the network."""
    import test_automusic as am

    d = tmp_path / "songs"
    d.mkdir()
    a, b = am._wav(d / "A.wav", am.EXPR % "0.5", 30), am._wav(d / "B.wav", am.EXPR % "0.72", 30)
    rows = [am.row(am.IDS["A"], dur=30000), am.row(am.IDS["B"], dur=30000, tags=("speed_low", "instrumental")),
            am.row(am.IDS["S"], lic="by-sa", dur=30000), am.row(am.IDS["Z"], lic="cc0", dur=30000)]
    return am.install(monkeypatch, None, rows, {am.IDS["A"]: a, am.IDS["B"]: b, am.IDS["S"]: a, am.IDS["Z"]: b})


def auto(env, expect=200, **body):
    r = call(env.app, "POST", f"/api/runs/{env.rid}/music/auto", {"mood": "", "cc0_only": False, "rank": 0, **body})
    assert r.status == expect, r.text
    return r


def picked(env, **body):
    j = wait_job(env.app, auto(env, **body).json()["job"]["id"])
    assert j["state"] == "done", j
    return j["result"]


def test_music_auto_is_a_job_that_returns_the_card_and_another_song_is_the_next_rank(env, auto_stub):
    env.split()
    r = auto(env).json()["job"]
    assert r["kind"] == "music-auto" and r["run_id"] == env.rid
    first = wait_job(env.app, r["id"])["result"]
    t = first["track"]
    assert first["rank"] == 0 and first["n"] >= 2 and first["notice"] == music.OPENVERSE_NOTICE and isinstance(first["warnings"], list)
    assert t["needs_credit"] is True and "License: CC BY 4.0" in t["credit_text"] and t["builtin"] is False and t["license"] == "by"
    assert music.ID_RE.match(t["id"])
    assert auto_stub.binaries and not any("0000-00000000000d" in u for u in auto_stub.binaries)  # the by-sa row is never fetched
    second = picked(env, rank=1)
    assert second["rank"] == 1 and second["track"]["id"] != t["id"]
    assert picked(env, rank=0)["track"]["id"] == t["id"]  # the same rank gives the same song
    assert picked(env, rank=first["n"])["wrapped"] is True
    only = picked(env, cc0_only=True)  # cc0 only: the cc0 row or the built-in beat, never a CC BY song
    assert only["track"]["license"] == "cc0" and only["track"]["needs_credit"] is False and only["track"]["credit_text"] is None


def test_music_auto_with_clip_ids_and_bad_bodies(env, auto_stub):
    env.split()
    assert picked(env, clip_ids=["d1"])["rank"] == 0
    for body, code in [({"mood": "a/b"}, 400), ({"mood": "x" * 41}, 400), ({"mood": 5}, 400), ({"rank": 64}, 400), ({"rank": -1}, 400),
                       ({"rank": 1.5}, 400), ({"rank": "1"}, 400), ({"rank": True}, 400), ({"cc0_only": "no"}, 400), ({"clip_ids": "d1"}, 400),
                       ({"clip_ids": ["zz"]}, 422), ({"clip_ids": []}, 422)]:
        assert auto(env, expect=code, **body).status == code, body
    assert call(env.app, "POST", "/api/runs/000000000000/music/auto", {}).status == 404
    assert env.app.jobs.current() is None


def test_music_auto_busy_cancel_and_failure(env, monkeypatch):
    import threading

    from inframe import automusic

    env.split()
    gate = threading.Event()
    monkeypatch.setattr(automusic, "choose", lambda *a, **kw: gate.wait(20))
    j = auto(env).json()["job"]
    try:
        assert auto(env, expect=409).json()["job"]["id"] == j["id"]  # a second request while one runs
        assert env.export(expect=409).json()["error"] == "busy"
        assert call(env.app, "POST", f"/api/jobs/{j['id']}/cancel", {}).status == 200
    finally:
        gate.set()
    assert wait_job(env.app, j["id"])["state"] == "cancelled"

    def refuse(*a, **kw):
        raise music.MusicError("music: Openverse is rate-limiting (test)")

    monkeypatch.setattr(automusic, "choose", refuse)
    f = wait_job(env.app, auto(env).json()["job"]["id"])
    assert f["state"] == "failed" and "rate-limiting" in f["error"]


def test_music_auto_cancel_takes_effect_while_choose_is_still_working(env, monkeypatch):
    import threading

    from inframe import automusic

    env.split()
    gate = threading.Event()
    monkeypatch.setattr(automusic, "choose", lambda *a, **kw: gate.wait(20))  # a long run of downloads that never reports progress
    j = auto(env).json()["job"]
    try:
        time.sleep(0.2)
        assert call(env.app, "POST", f"/api/jobs/{j['id']}/cancel", {}).status == 200
        end = time.monotonic() + 1.5
        while time.monotonic() < end and env.app.jobs.get(j["id"])["state"] == "running":
            time.sleep(0.05)
        assert env.app.jobs.get(j["id"])["state"] == "cancelled"  # not after choose() returns
    finally:
        gate.set()


def test_a_failed_run_gets_no_auto_music(env):
    env.seg["error"] = "stopped during the cuts"
    (env.run_dir / "segments.json").write_text(json.dumps(env.seg))
    assert "no reel" in auto(env, expect=422).json()["error"]


def test_export_with_auto_music_uses_the_song_the_preview_showed(env, auto_stub):
    pytest.importorskip("librosa")
    env.split()
    shown = picked(env, rank=1)
    res = env.done({"reel": True, "music": {"auto": {"mood": "", "rank": shown["rank"]}, "cc0_only": False}, "beat_sync": True})
    m = res["reel"]["music"]
    assert res["reel"]["ok"] is True and m["title"] == shown["track"]["title"] and m["creator"] == shown["track"]["creator"]
    assert m["auto"]["rank"] == 1 and m["auto"]["n"] == shown["n"] and m["beat_sync"] is True
    assert (m["credits"] is not None) is shown["track"]["needs_credit"]


def test_export_music_auto_is_validated_like_the_other_music_choices(env):
    env.split()
    ok = {"mood": "calm", "rank": 0}
    for music_, code in [({"auto": {"mood": "a/b", "rank": 0}}, 400), ({"auto": {"mood": "", "rank": 64}}, 400), ({"auto": {"mood": "", "rank": "0"}}, 400),
                         ({"auto": "calm"}, 400), ({"auto": ok, "track_id": "0123abcd4567"}, 422), ({"auto": ok, "file": "/x.wav"}, 422),
                         ({"auto": ok, "cc0_only": "no"}, 400)]:
        r = call(env.app, "POST", f"/api/runs/{env.rid}/export", {"clips": True, "reel": True, "music": music_})
        assert r.status == code, (music_, r.status, r.text)
    r = call(env.app, "POST", f"/api/runs/{env.rid}/export", {"clips": True, "reel": False, "music": {"auto": ok}})
    assert r.status == 422 and "reel" in r.json()["error"]
    assert env.app.jobs.current() is None


def _down(monkeypatch, files=True, search=False):
    """Openverse is unreachable: the song files (and, with `search`, the search too) fail as a lost connection does."""
    from inframe import net

    real = music._get

    def get(url, binary=False, limit=None):
        if (binary and files) or (not binary and search):
            e = net.NetError("could not reach example.org: offline (test)")
            raise music.MusicError(str(e)) from e
        return real(url, binary, limit)

    monkeypatch.setattr(music, "_get", get)


def test_export_does_not_silently_use_another_song_than_the_preview_showed(env, auto_stub, monkeypatch):
    pytest.importorskip("librosa")
    env.split()
    shown = picked(env, rank=0)
    body = {"reel": True, "music": {"auto": {"mood": "", "rank": 0, "track": shown["track"]["id"]}, "cc0_only": False}, "beat_sync": False}
    assert env.done(body)["reel"]["music"]["title"] == shown["track"]["title"]  # the same song: it goes through
    for f in music.cache_dir().glob(f"{shown['track']['id']}.*"):  # the shown song leaves the cache, and Openverse is down at export time
        f.unlink()
    _down(monkeypatch)
    j = wait_job(env.app, env.export({**body, "replace": True}).json()["job"]["id"])
    assert j["state"] == "failed" and "previewed" in j["error"] and shown["track"]["id"] in j["error"], j


def test_an_export_waiting_for_a_leftover_auto_download_shows_progress_and_can_be_cancelled(env, auto_stub):
    from inframe.serve_music import _download

    env.split()
    _download.acquire()  # a cancelled auto job's thread is still downloading
    try:
        j = env.export({"reel": True, "music": {"auto": {"mood": "", "rank": 0}, "cc0_only": False}, "beat_sync": False}).json()["job"]
        time.sleep(0.5)
        msg = env.app.jobs.get(j["id"])["message"]
        assert "waiting" in msg, msg
        assert call(env.app, "POST", f"/api/jobs/{j['id']}/cancel", {}).status == 200
        end = time.monotonic() + 2.0
        while time.monotonic() < end and env.app.jobs.get(j["id"])["state"] == "running":
            time.sleep(0.05)
        assert env.app.jobs.get(j["id"])["state"] == "cancelled"  # not after the leftover download ends
    finally:
        _download.release()
        wait_job(env.app, j["id"])


def test_export_with_auto_music_reports_the_pick_warnings(env, auto_stub, monkeypatch):
    env.split()
    _down(monkeypatch, search=True)  # no song can be found for a new mood: the built-in beat is used, and the export says why
    res = env.done({"reel": True, "music": {"auto": {"mood": "calm", "rank": 0}, "cc0_only": False}, "beat_sync": False})
    assert res["reel"]["music"]["auto"]["builtin"] is True
    assert any("Openverse could not be reached" in w for w in res["warnings"]), res["warnings"]


# ---------------------------------------------------------------- T3b: Save to Photos (a fake helper; no real Photos library)
PHOTOS_FAKE = '''#!{py}
import json, os, sys
a = sys.argv[1:]
open(os.environ["FAKE_ARGS"], "w").write(json.dumps(a))
mode = os.environ.get("FAKE_MODE", "saved")
if mode == "denied":
    print(json.dumps(dict(authorization="denied", status="denied", message="m", files=[dict(path=p, saved=False, error="no permission to add to Photos") for p in a])))
    sys.exit(3)
rows = [dict(path=p, saved=not (mode == "partial" and i == 0), **(dict(error="PHPhotosErrorDomain 3302") if mode == "partial" and i == 0 else {{}})) for i, p in enumerate(a)]
n = sum(r["saved"] for r in rows)
print(json.dumps(dict(authorization="authorized", status="saved" if n == len(rows) else "partial" if n else "failed", files=rows, message="m")))
'''


@pytest.fixture
def photos_helper(tmp_path, monkeypatch):
    from inframe import photosave
    p = tmp_path / "fake-save"
    p.write_text(PHOTOS_FAKE.format(py=sys.executable))
    p.chmod(0o755)
    monkeypatch.setenv(photosave.HELPER_ENV, str(p))
    args = tmp_path / "photos_args.json"
    monkeypatch.setenv("FAKE_ARGS", str(args))
    monkeypatch.setenv("FAKE_MODE", "saved")
    return SimpleNamespace(args=lambda: json.loads(args.read_text()), exists=args.exists, mode=lambda m: monkeypatch.setenv("FAKE_MODE", m))


def test_session_offers_save_photos_only_where_a_save_can_run(env, photos_helper, monkeypatch):
    from inframe import photosave
    assert call(env.app, "GET", "/api/session").json()["caps"]["savePhotos"] is True
    monkeypatch.delenv(photosave.HELPER_ENV)
    monkeypatch.setattr(sys, "platform", "linux")
    assert call(env.app, "GET", "/api/session").json()["caps"]["savePhotos"] is False


def test_export_to_photos_saves_clips_and_reel_and_records_it(env, photos_helper):
    env.split()
    res = env.done({"reel": True, "to_photos": True})
    names = [Path(p).name for p in photos_helper.args()]
    assert names == [c["file"] for c in res["clips"]] + ["clip_reel.mp4"]
    assert res["photos"]["status"] == "saved" and res["photos"]["saved"] == names and res["photos"]["failed"] == []
    assert res["warnings"] == [] or all("Photos" not in w for w in res["warnings"])
    assert json.loads((env.edited / "export.json").read_text())["photos"]["status"] == "saved"
    assert call(env.app, "GET", f"/api/runs/{env.rid}/export").json()["photos"]["status"] == "saved"
    assert all((env.edited / n).is_file() for n in names)  # the files stay in the folder


def test_export_to_photos_denied_keeps_the_export_and_says_so(env, photos_helper):
    photos_helper.mode("denied")
    res = env.done({"to_photos": True})
    assert res["photos"]["status"] == "denied" and "System Settings > Privacy & Security > Photos" in res["photos"]["message"]
    assert res["photos"]["message"] in res["warnings"]
    assert res["outputs"] and (env.edited / res["clips"][0]["file"]).is_file()


def test_export_to_photos_partial_lists_what_failed(env, photos_helper):
    env.split()
    photos_helper.mode("partial")
    res = env.done({"to_photos": True})
    assert res["photos"]["status"] == "partial" and len(res["photos"]["saved"]) == 1
    assert res["photos"]["failed"] == [{"name": res["clips"][0]["file"], "error": "PHPhotosErrorDomain 3302"}]


def test_export_to_photos_helper_failure_does_not_fail_the_job(env, photos_helper, monkeypatch, tmp_path):
    from inframe import photosave
    monkeypatch.setenv(photosave.HELPER_ENV, str(tmp_path / "gone"))  # planned with a helper, gone when the job runs
    plan_ok = photosave.available()
    assert plan_ok
    res = env.done({"to_photos": True})
    assert res["photos"]["status"] == "failed" and "Nothing was saved to Photos" in res["photos"]["message"]


def test_reel_only_to_photos_never_sends_the_clips_when_the_reel_failed(env, photos_helper, monkeypatch):
    from inframe import export as export_mod

    env.split()
    monkeypatch.setattr(export_mod, "build_reel", lambda *a, **k: dict(file=None, ok=False, fallback_used=False, check=dict(problems=["boom"])))
    res = env.done({"clips": False, "reel": True, "to_photos": True})
    assert res["reel"]["ok"] is False and len(list(env.edited.glob("clip_0*.mp4"))) == 2  # the fallback clips are on disk
    assert not photos_helper.exists(), "the helper must not receive clips the person did not choose"
    assert res["photos"]["status"] == "failed" and res["photos"]["saved"] == []


def test_export_without_to_photos_never_starts_the_helper(env, photos_helper):
    env.done({})
    assert not photos_helper.exists()


def test_to_photos_is_refused_before_a_job_starts_when_it_cannot_work(env, monkeypatch):
    from inframe import photosave
    monkeypatch.delenv(photosave.HELPER_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    r = env.export({"to_photos": True}, expect=422)
    assert "needs a Mac" in r.json()["error"]
    assert env.export({"to_photos": "yes"}, expect=400)
    assert not env.edited.exists()


def test_to_photos_needs_clips_or_a_reel(env, photos_helper):
    r = env.export({"clips": False, "edit_lists": ["fcpxml"], "to_photos": True}, expect=422)
    assert "clips or a reel" in r.json()["error"]
