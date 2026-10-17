"""inframe serve, runs (docs/T9A_DESIGN.md rows 34-41): the run job, the list of a video's runs and the rules for untrusted run
folders, samples, re-segment, and adding crops to a saved target. Motion mode needs no model, so the runs are real, on the
synthetic `static_box` clip (one clip around 2-4 s); the detector paths use a monkeypatched `run_cut`."""
import csv
import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from serve_helpers import call, start_server

from inframe import identity, serve_edits, serve_paths, serve_runs

ADV = {"fps": 5.0, "pad": 0.5, "min_len": 1.0}


def wait_job(app, jid, timeout=120.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = app.jobs.get(jid)
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job still running")


@pytest.fixture
def env(tmp_path, static_box):
    vids = tmp_path / "vids"
    vids.mkdir()
    video = vids / "clip.mp4"
    shutil.copy(static_box, video)
    base, app, stop = start_server(tmp_path)
    media = app.open_media(str(video))

    def start(target=None, **over):
        body = {"media_id": media.id, "target": target or {"kind": "motion"}, "advanced": ADV, **over}
        return call(app, "POST", "/api/runs", body)

    def run(**kw):
        r = start(**kw)
        assert r.status == 200, r.text
        j = wait_job(app, r.json()["job"]["id"])
        assert j["state"] == "done", j
        return r.json()["run_id"], j

    yield SimpleNamespace(app=app, media=media, video=video, start=start, run=run, tmp=tmp_path,
                          out=vids / "clip_inframe")
    stop()


# ---------------------------------------------------------------- row 35: the run job
def test_motion_run_goes_to_done_in_the_motion_folder(env):
    r = env.start()
    assert r.status == 200
    job, rid = r.json()["job"], r.json()["run_id"]
    assert job["kind"] == "run" and job["media_id"] == env.media.id and job["run_id"] == rid
    done = wait_job(env.app, job["id"])
    assert done["state"] == "done" and done["fraction"] == 1.0 and done["error"] is None
    assert done["result"]["run_id"] == rid and done["result"]["n_clips"] == 1
    folder = env.out / "motion-ui"
    assert folder.is_dir() and (folder.stat().st_mode & 0o777) == 0o700
    doc = json.loads((folder / "segments.json").read_text())
    assert doc["mode"] == "motion" and len(doc["clips"]) == 1
    assert 1.0 < doc["clips"][0]["start"] < doc["clips"][0]["end"] < 5.0
    assert (folder / "samples.csv").is_file() and (folder / doc["clips"][0]["file"]).is_file()
    assert rid == serve_edits.run_id_for(folder)


def test_get_run_clips_equal_segments_json(env):
    rid, _ = env.run()
    seg = json.loads((env.out / "motion-ui" / "segments.json").read_text())
    g = call(env.app, "GET", f"/api/runs/{rid}")
    assert g.status == 200
    b = g.json()
    assert b["run_id"] == rid and b["media_id"] == env.media.id and b["dir"] == str(env.out / "motion-ui")
    assert b["segments"]["clips"] == seg["clips"] and b["segments"]["presence_intervals"] == seg["presence_intervals"]
    assert b["segments"]["settings"] == seg["settings"] and b["segments"]["warnings"] == seg["warnings"]
    assert set(b["segments"]) == {"settings", "warnings", "scores", "timing_s", "presence_intervals", "clips", "slowmo", "error"}
    assert b["segments"]["slowmo"] is None and b["segments"]["error"] is None
    assert b["edits"] is None and b["edits_base_ok"] is True and b["export"] is None
    assert call(env.app, "GET", "/api/runs/000000000000").status == 404
    assert call(env.app, "GET", "/api/runs/zz").status == 404


def test_get_run_reads_the_export_json_of_the_edited_folder(env):
    rid, _ = env.run()
    (env.out / "motion-ui" / "edited").mkdir()
    (env.out / "motion-ui" / "edited" / "export.json").write_text(json.dumps({"reel": "x.mp4"}))
    assert call(env.app, "GET", f"/api/runs/{rid}").json()["export"] == {"reel": "x.mp4"}


def test_samples_arrays_equal_samples_csv(env):
    rid, _ = env.run()
    rows = list(csv.DictReader(open(env.out / "motion-ui" / "samples.csv", newline="")))
    r = call(env.app, "GET", f"/api/runs/{rid}/samples")
    assert r.status == 200
    s = r.json()
    assert s["t"] == [float(x["t"]) for x in rows] and s["score"] == [float(x["score"]) for x in rows]
    assert len(s["t"]) == 40 and s["box"] == [None] * 40 and s["track"] is None and s["mode"] == "motion"
    st = json.loads((env.out / "motion-ui" / "segments.json").read_text())["settings"]
    assert (s["enter"], s["exit"], s["merge_gap"], s["min_len"], s["pad"], s["sound"]) == (
        st["enter"], st["exit"], st["merge_gap"], st["min_len"], st["pad"], st["sound"])
    assert call(env.app, "GET", "/api/runs/000000000000/samples").status == 404


def test_run_request_is_checked(env):
    bad = [
        ({"target": {"kind": "nope"}}, 422), ({"target": "tap"}, 400), ({"advanced": {"enter": 2.0}}, 422),
        ({"advanced": {"enter": 0.3, "exit": 0.5}}, 422), ({"advanced": {"pad": -1}}, 422), ({"advanced": {"pad": "1"}}, 422),
        ({"advanced": {"bogus": 1}}, 422), ({"advanced": {"tracker": "x"}}, 422), ({"advanced": {"fps": float("nan")}}, 422),
        ({"advanced": 5}, 400), ({"preset": "turbo"}, 422), ({"slowmo": "fast"}, 422), ({"slowmo_audio": True}, 422),
        ({"zone": [0.5, 0.5, 0.9, 0.9]}, 422), ({"zone_overlap": 0.5}, 422), ({"zone_overlap": 3}, 422),
        ({"advanced": {**ADV, "activity": "riding"}}, 422),
    ]
    for over, code in bad:
        r = env.start(**{"advanced": ADV, **over}) if "advanced" not in over else env.start(**over)
        assert r.status == code, (over, r.status, r.text)
    assert call(env.app, "POST", "/api/runs", {"media_id": "000000000000"}).status == 404
    assert call(env.app, "POST", "/api/runs", {}).status == 404
    assert env.app.jobs.current() is None and not env.out.exists()  # nothing started, nothing written


def test_second_run_while_one_runs_is_busy(env, monkeypatch):
    release = threading.Event()

    def slow(video, out, settings, target=None, progress=None):
        progress(0.1, "working")
        release.wait(20)
        raise RuntimeError("stopped")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", slow)
    first = env.start()
    assert first.status == 200
    second = env.start()
    assert second.status == 409 and second.json()["error"] == "busy" and second.json()["job"]["id"] == first.json()["job"]["id"]
    release.set()
    assert wait_job(env.app, first.json()["job"]["id"])["state"] == "failed"


def test_ambiguous_photo_run_ends_needs_confirm(env, monkeypatch):
    def ambiguous(video, out, settings, target=None, progress=None):
        raise identity.Ambiguous("two people look equally like the photos: confirm with a tap")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", ambiguous)
    r = env.start()
    j = wait_job(env.app, r.json()["job"]["id"])
    assert j["state"] == "needs_confirm" and "two people" in j["error"]
    assert j["result"] == {"reason": "ambiguous", "run_id": r.json()["run_id"], "media_id": env.media.id}


def test_a_failed_run_reports_the_message_and_cancel_ends_cancelled(env, monkeypatch):
    def boom(video, out, settings, target=None, progress=None):
        raise ValueError("--exit 0.9 must not exceed the automatic enter threshold")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", boom)
    j = wait_job(env.app, env.start().json()["job"]["id"])
    assert j["state"] == "failed" and "--exit 0.9" in j["error"]
    monkeypatch.undo()

    def endless(video, out, settings, target=None, progress=None):
        for i in range(1000):
            progress(i / 1000, "step")
            time.sleep(0.01)

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", endless)
    jid = env.start().json()["job"]["id"]
    assert call(env.app, "POST", f"/api/jobs/{jid}/cancel", {}).status == 200
    assert wait_job(env.app, jid, 10)["state"] == "cancelled"


def test_settings_reach_run_cut(env, monkeypatch):
    seen = {}

    def fake(video, out, settings, target=None, progress=None):
        seen.update(video=video, out=out, settings=settings, target=target)
        raise RuntimeError("stop here")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", fake)
    body = {"target": {"kind": "any"}, "preset": "fast", "zone": [0.1, 0.2, 0.5, 0.6], "zone_overlap": 0.4, "slowmo": "auto",
            "slowmo_audio": True, "advanced": {"pad": 2, "min_len": 0.5, "merge_gap": 3, "enter": 0.6, "exit": 0.2, "min_box": 0.1,
                                               "tracker": "botsort", "activity": "riding", "fps": 2}}
    j = wait_job(env.app, env.start(**body).json()["job"]["id"])
    s = seen["settings"]
    assert j["state"] == "failed" and seen["video"] == str(env.media.path) and seen["out"] == str(env.out / "any-person")
    assert (s.mode, s.preset, s.zone, s.zone_overlap, s.slowmo, s.slowmo_audio) == ("detect", "fast", (0.1, 0.2, 0.5, 0.6), 0.4, "auto", True)
    assert (s.pad, s.min_len, s.merge_gap, s.enter, s.exit, s.min_box, s.tracker, s.activity, s.fps) == (
        2, 0.5, 3, 0.6, 0.2, 0.1, "botsort", "riding", 2)
    assert seen["target"] is None
    monkeypatch.undo()


def test_class_target_runs_in_its_own_folder(env, monkeypatch):
    seen = {}

    def fake(video, out, settings, target=None, progress=None):
        seen.update(out=out, target=target)
        raise RuntimeError("stop here")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", fake)
    wait_job(env.app, env.start({"kind": "class", "class": "Dog"}).json()["job"]["id"])
    assert seen["out"] == str(env.out / "class-dog") and seen["target"].name == "class:dog"
    assert env.start({"kind": "class", "class": "unicorn"}).status == 422
    assert env.start({"kind": "class"}).status == 422
    assert env.start({"kind": "saved", "name": "../x"}).status == 422
    assert env.start({"kind": "photos", "photos_id": "p9"}).status == 404


# ---------------------------------------------------------------- tap runs read the picker's file and re-check target.json
def put_pick_file(env, **extra):
    f = serve_paths.target_file(env.out, "person", "")
    f.parent.mkdir(parents=True)
    doc = {"video": str(env.media.path), "frame_size": [320, 180], "model": "small", "preset": "balanced", "time": 3.0,
           "box": [0.3, 0.3, 0.6, 0.9], "track_id": None, "taps": [{"time": 3.0, "box": [0.3, 0.3, 0.6, 0.9], "track_id": None, "index": 1}],
           **extra}
    f.write_text(json.dumps(doc))
    return f


def test_tap_run_copies_the_picker_file_and_follows_its_preset_and_zone(env, monkeypatch):
    put_pick_file(env, model="base", preset="accurate", zone=[0.1, 0.1, 0.5, 0.5])
    seen = {}

    def fake(video, out, settings, target=None, progress=None):
        seen.update(out=out, settings=settings, target=target)
        raise RuntimeError("stop here")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", fake)
    r = env.start({"kind": "tap"}, preset=None)
    assert r.status == 200, r.text
    wait_job(env.app, r.json()["job"]["id"])
    copy = env.out / "target.json"
    assert json.loads(copy.read_text())["taps"][0]["time"] == 3.0  # the inframe pick default location
    assert seen["out"] == str(env.out / "tap") and seen["settings"].preset == "accurate"
    assert seen["settings"].zone == (0.1, 0.1, 0.5, 0.5) and seen["target"].name == "tap"
    assert not list(env.out.glob("*.tmp"))
    # another preset than the one the tap was picked with is refused, as the CLI refuses it
    r = env.start({"kind": "tap"}, preset="fast")
    assert r.status == 422 and "picked with" in r.json()["error"]


def test_tap_run_needs_taps(env):
    r = env.start({"kind": "tap"})
    assert r.status == 422 and "No taps" in r.json()["error"]


def test_target_json_naming_photos_under_library_is_refused(env, monkeypatch):
    home = env.tmp / "home"
    (home / "Library" / "Photos").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    put_pick_file(env, photos=str(home / "Library" / "Photos"), embedder="siglip2")
    r = env.start({"kind": "tap"})
    assert r.status == 403, r.text
    assert not (env.out / "target.json").exists() and env.app.jobs.current() is None


def test_target_json_naming_a_hidden_folder_or_bad_saved_name_is_refused(env):
    hidden = env.tmp / ".hidden_photos"
    hidden.mkdir()
    f = put_pick_file(env, photos=str(hidden))
    assert env.start({"kind": "tap"}).status == 403
    doc = json.loads(f.read_text())
    doc["photos"] = "saved:../../etc"
    f.write_text(json.dumps(doc))
    assert env.start({"kind": "tap"}).status == 422
    doc.pop("photos")
    doc["video"] = str(env.tmp / ".secret.mp4")
    f.write_text(json.dumps(doc))
    assert env.start({"kind": "tap"}).status in (403, 404)


def test_symlinked_picker_file_is_refused(env):
    f = put_pick_file(env)
    real = f.with_name("real.json")
    shutil.move(f, real)
    f.symlink_to(real)
    assert env.start({"kind": "tap"}).status == 422


# ---------------------------------------------------------------- row 36: the list of runs and untrusted folders
def test_list_runs_newest_first_with_the_fields(env):
    rid, _ = env.run()
    time.sleep(0.05)
    rid2, _ = env.run(target={"kind": "motion"}, advanced={**ADV, "pad": 1.0})  # same folder: one run
    assert rid2 == rid
    r = call(env.app, "GET", f"/api/media/{env.media.id}/runs")
    assert r.status == 200
    runs = r.json()["runs"]
    assert len(runs) == 1
    x = runs[0]
    assert set(x) == {"run_id", "dir", "slug", "target", "preset", "created", "n_clips", "error", "has_edits", "edits_base_ok"}
    assert x["run_id"] == rid and x["slug"] == "motion-ui" and x["target"] == "any-motion" and x["preset"] == "balanced"
    assert x["n_clips"] == 1 and x["error"] is None and x["has_edits"] is False and x["edits_base_ok"] is True
    assert x["created"].endswith("Z") and x["dir"] == str(env.out / "motion-ui")
    assert call(env.app, "GET", "/api/media/000000000000/runs").status == 404


def test_runs_are_listed_newest_first_and_register_after_a_restart(env):
    env.run()
    first = env.out / "motion-ui"
    other = env.out / "copy-of-run"
    shutil.copytree(first, other)
    os.utime(first / "segments.json", (1_000_000, 1_000_000))
    runs = call(env.app, "GET", f"/api/media/{env.media.id}/runs").json()["runs"]
    assert [x["slug"] for x in runs] == ["copy-of-run", "motion-ui"]
    env.app.runs.clear()  # as after a server restart: GET needs the list first
    rid = serve_edits.run_id_for(other)
    assert call(env.app, "GET", f"/api/runs/{rid}").status == 404
    call(env.app, "GET", f"/api/media/{env.media.id}/runs")
    assert call(env.app, "GET", f"/api/runs/{rid}").status == 200


def test_untrusted_folders_are_not_listed(env, tmp_path):
    env.run()
    good = env.out / "motion-ui"
    # a symlinked segments.json
    a = env.out / "linked-segments"
    a.mkdir()
    shutil.copy(good / "segments.json", tmp_path / "elsewhere.json")
    (a / "segments.json").symlink_to(tmp_path / "elsewhere.json")
    # a run of another video
    b = env.out / "other-video"
    b.mkdir()
    doc = json.loads((good / "segments.json").read_text())
    doc["source"]["path"] = str(env.video.with_name("another.mp4"))
    (b / "segments.json").write_text(json.dumps(doc))
    # a symlinked folder
    shutil.copytree(good, tmp_path / "real_folder")
    (env.out / "linked-folder").symlink_to(tmp_path / "real_folder", target_is_directory=True)
    # junk
    c = env.out / "junk"
    c.mkdir()
    (c / "segments.json").write_text("not json")
    d = env.out / "not-an-object"
    d.mkdir()
    (d / "segments.json").write_text("[1]")
    h = env.out / ".hidden"
    shutil.copytree(good, h)
    (env.out / "plain-file.json").write_text("{}")
    runs = call(env.app, "GET", f"/api/media/{env.media.id}/runs").json()["runs"]
    assert [x["slug"] for x in runs] == ["motion-ui"]
    for ghost in ("linked-segments", "other-video", "linked-folder"):
        assert serve_edits.run_id_for(env.out / ghost) not in env.app.runs


def test_a_failed_run_is_listed_with_its_error(env):
    rid, _ = env.run()
    seg = env.out / "motion-ui" / "segments.json"
    doc = json.loads(seg.read_text())
    doc.update(error="scoring stopped", clips=[], presence_intervals=[])
    seg.write_text(json.dumps(doc))
    x = call(env.app, "GET", f"/api/media/{env.media.id}/runs").json()["runs"][0]
    assert x["error"] == "scoring stopped" and x["n_clips"] == 0


# ---------------------------------------------------------------- edits and a new run (2.6)
def test_a_new_run_moves_the_edits_aside(env):
    rid, _ = env.run()
    put = call(env.app, "PUT", f"/api/runs/{rid}/edits", {"revision": 0, "clips": [{"id": "u1", "start": 1.0, "end": 2.0, "origin": "added"}]})
    assert put.status == 200
    folder = env.out / "motion-ui"
    assert (folder / "edits.json").is_file()
    rid2, _ = env.run()
    assert rid2 == rid
    assert not (folder / "edits.json").exists()
    prev = [p for p in folder.iterdir() if p.name.startswith("edits.prev-")]
    assert len(prev) == 1 and json.loads(prev[0].read_text())["clips"][0]["id"] == "u1"
    g = call(env.app, "GET", f"/api/runs/{rid}").json()
    assert g["edits"] is None and g["segments"]["clips"]


def test_a_new_run_that_writes_the_same_bytes_still_moves_the_edits_aside(env, monkeypatch):
    rid, _ = env.run()
    call(env.app, "PUT", f"/api/runs/{rid}/edits", {"revision": 0, "clips": [{"id": "u1", "start": 1.0, "end": 2.0, "origin": "added"}]})
    seg = env.out / "motion-ui" / "segments.json"
    raw = seg.read_bytes()

    def same(video, out, settings, target=None, progress=None):  # a re-run whose rounded timings repeat writes identical bytes
        time.sleep(0.01)
        seg.write_bytes(raw)
        return json.loads(raw)

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", same)
    env.run()
    assert not (env.out / "motion-ui" / "edits.json").exists()
    assert [p for p in (env.out / "motion-ui").iterdir() if p.name.startswith("edits.prev-")]


def test_a_run_that_fails_before_writing_keeps_the_edits(env, monkeypatch):
    rid, _ = env.run()
    call(env.app, "PUT", f"/api/runs/{rid}/edits", {"revision": 0, "clips": [{"id": "u1", "start": 1.0, "end": 2.0, "origin": "added"}]})

    def boom(video, out, settings, target=None, progress=None):
        raise ValueError("bad setting")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", boom)
    assert wait_job(env.app, env.start().json()["job"]["id"])["state"] == "failed"
    assert (env.out / "motion-ui" / "edits.json").is_file()  # segments.json was not replaced, so the edits still match it


def test_a_run_that_replaces_segments_json_with_a_failed_record_moves_the_edits_aside(env, monkeypatch):
    rid, _ = env.run()
    call(env.app, "PUT", f"/api/runs/{rid}/edits", {"revision": 0, "clips": [{"id": "u1", "start": 1.0, "end": 2.0, "origin": "added"}]})

    def stopped(video, out, settings, target=None, progress=None):
        (Path(out) / "segments.json").write_text(json.dumps({"source": {"path": video, "start_time": 0, "duration": 8}, "error": "stopped",
                                                           "clips": []}))
        raise RuntimeError("stopped during the cuts")

    monkeypatch.setattr(serve_runs.pipeline, "run_cut", stopped)
    assert wait_job(env.app, env.start().json()["job"]["id"])["state"] == "failed"
    assert not (env.out / "motion-ui" / "edits.json").exists()
    assert len(list((env.out / "motion-ui").glob("edits.prev-*.json"))) == 1


# ---------------------------------------------------------------- row 41: re-segment
def test_resegment_with_no_fields_is_the_run_exactly(env):
    rid, _ = env.run()
    doc = json.loads((env.out / "motion-ui" / "segments.json").read_text())
    for body in ({}, {"enter": None}, {"enter": doc["settings"]["enter"], "exit": doc["settings"]["exit"], "merge_gap": 1.5, "min_len": 1.0, "pad": 0.5}):
        r = call(env.app, "POST", f"/api/runs/{rid}/resegment", body)
        assert r.status == 200, r.text
        b = r.json()
        assert b["same_as_run"] is True
        assert b["clips"] == [[c["start"], c["end"]] for c in doc["clips"]]
        assert b["presence"] == doc["presence_intervals"]


def test_resegment_same_as_run_matches_without_the_tolerance_on_this_clip(env):
    """The tolerance only covers the csv's rounding; here the numbers computed from the csv are the run's own."""
    from inframe.segments import build_segments

    rid, _ = env.run()
    d = env.out / "motion-ui"
    doc = json.loads((d / "segments.json").read_text())
    t, s, _, _ = serve_edits.load_samples_arrays(d)
    st = doc["settings"]
    presence, clips = build_segments(t, s, doc["source"]["duration"], st["enter"], st["exit"], st["merge_gap"], st["min_len"], st["pad"])
    assert [[round(a, 3), round(b, 3)] for a, b in clips] == [[c["start"], c["end"]] for c in doc["clips"]]


def test_resegment_with_other_values_previews_without_writing(env):
    rid, _ = env.run()
    before = (env.out / "motion-ui" / "segments.json").read_bytes()
    r = call(env.app, "POST", f"/api/runs/{rid}/resegment", {"pad": 0.0, "min_len": 0.2}).json()
    run_clip = json.loads(before)["clips"][0]
    assert r["same_as_run"] is False
    assert len(r["clips"]) == 1 and r["clips"][0][0] > run_clip["start"] and r["clips"][0][1] < run_clip["end"]
    assert r["presence"] == r["clips"]  # no padding: the presence is the clip
    r2 = call(env.app, "POST", f"/api/runs/{rid}/resegment", {"enter": 1.0, "exit": 1.0}).json()  # nothing reaches 1.0
    assert r2["clips"] == [] and r2["presence"] == [] and r2["same_as_run"] is False
    assert (env.out / "motion-ui" / "segments.json").read_bytes() == before
    assert not (env.out / "motion-ui" / "edits.json").exists()


def test_resegment_checks_its_numbers(env):
    rid, _ = env.run()
    for body in ({"enter": 0.1, "exit": 0.2}, {"pad": -1}, {"min_len": "1"}, {"merge_gap": float("nan")}, {"enter": True}):
        assert call(env.app, "POST", f"/api/runs/{rid}/resegment", body).status == 422, body
    assert call(env.app, "POST", "/api/runs/000000000000/resegment", {}).status == 404


# ---------------------------------------------------------------- row 34: add crops to a saved target
def test_target_add_from_a_run_is_a_job(env, monkeypatch):
    rid, _ = env.run()
    calls = {}

    def crops_from_run(seg):
        calls["seg"] = seg
        return ["crop-a", "crop-b"]

    def add_crops(name, crops, source, embedder, offline):
        calls.update(name=name, crops=crops, source=source, embedder=embedder, offline=offline)
        return {"name": name, "created": True, "added": 2, "offered": 2, "n_crops": 2}

    monkeypatch.setattr(serve_runs.targets, "crops_from_run", crops_from_run)
    monkeypatch.setattr(serve_runs.targets, "add_crops", add_crops)
    r = call(env.app, "POST", "/api/targets/alice/add", {"from": "run", "run_id": rid})
    assert r.status == 200 and r.json()["job"]["kind"] == "target-add"
    j = wait_job(env.app, r.json()["job"]["id"])
    assert j["state"] == "done" and j["result"] == {"name": "alice", "created": True, "added": 2, "offered": 2, "n_crops": 2}
    assert j["media_id"] == env.media.id and j["run_id"] == rid
    assert calls["seg"] == str(env.out / "motion-ui" / "segments.json") and calls["source"] == "run"
    assert calls["crops"] == ["crop-a", "crop-b"] and calls["offline"] is False


def test_target_add_from_photos_uses_the_chosen_folder(env, monkeypatch):
    folder = env.tmp / "photos"
    folder.mkdir()
    env.app.photos_dirs["p1"] = folder
    seen = {}
    monkeypatch.setattr(serve_runs.targets, "crops_from", lambda spec: (seen.setdefault("spec", spec) and ["c"], "photos:photos"))
    monkeypatch.setattr(serve_runs.targets, "add_crops", lambda name, crops, source, embedder, offline: {"name": name, "added": len(crops)})
    r = call(env.app, "POST", "/api/targets/me/add", {"from": "photos", "photos_id": "p1"})
    assert wait_job(env.app, r.json()["job"]["id"])["result"] == {"name": "me", "added": 1}
    assert seen["spec"] == f"photos:{folder.resolve()}"


def test_target_add_refusals(env):
    rid, _ = env.run()
    post = lambda name, body: call(env.app, "POST", f"/api/targets/{name}/add", body)
    assert post("bad%20name%21", {"from": "run", "run_id": rid}).status == 422
    assert post("..", {"from": "run", "run_id": rid}).status == 422
    assert post("ok", {"from": "run", "run_id": "000000000000"}).status == 404
    assert post("ok", {"from": "run"}).status == 404
    assert post("ok", {"from": "photos", "photos_id": "nope"}).status == 404
    assert post("ok", {"from": "disk"}).status == 400
    assert post("ok", {}).status == 400
    assert env.app.jobs.current() is None


def test_target_add_failure_is_reported_by_the_job(env):
    rid, _ = env.run()  # a motion run keeps no person: crops_from_run says so
    r = call(env.app, "POST", "/api/targets/alice/add", {"from": "run", "run_id": rid})
    j = wait_job(env.app, r.json()["job"]["id"])
    assert j["state"] == "failed" and "not one person" in j["error"]


def test_every_route_needs_the_session_token(env):
    rid, _ = env.run()
    for method, path in (("POST", "/api/runs"), ("GET", f"/api/media/{env.media.id}/runs"), ("GET", f"/api/runs/{rid}"),
                         ("GET", f"/api/runs/{rid}/samples"), ("PUT", f"/api/runs/{rid}/edits"), ("DELETE", f"/api/runs/{rid}/edits"),
                         ("POST", f"/api/runs/{rid}/resegment"), ("POST", "/api/targets/x/add")):
        body = {} if method in ("POST", "PUT", "DELETE") else None
        assert call(env.app, method, path, body, key=None).status == 403, path
        assert call(env.app, method, path, body, key="media").status == 403, path  # the media key opens no JSON route


# ---------------------------------------------------------------- UI motion runs and the CLI's motion folder
def test_a_ui_motion_run_does_not_touch_the_cli_motion_folder(env):
    cli = env.out / "motion"  # what `inframe cut --mode motion` writes by default: <stem>_inframe/<mode>
    cli.mkdir(parents=True)
    (cli / "segments.json").write_text('{"cli": true}')
    (cli / "clip_001.mp4").write_bytes(b"cli clip")
    rid, _ = env.run()
    assert (cli / "segments.json").read_text() == '{"cli": true}'
    assert (cli / "clip_001.mp4").read_bytes() == b"cli clip"
    assert Path(call(env.app, "GET", f"/api/runs/{rid}").json()["dir"]) != cli


def test_a_cancel_during_target_add_does_not_hide_the_added_crops(env, monkeypatch):
    rid, _ = env.run()
    seen = {}

    def add_crops(name, crops, source, embedder, offline):
        seen["added"] = True
        env.app.jobs.cancel(env.app.jobs.current()["id"])  # the user presses Cancel while the crops are being saved
        return {"name": name, "created": False, "added": 1, "offered": 1, "n_crops": 1}

    monkeypatch.setattr(serve_runs.targets, "crops_from_run", lambda seg: ["c"])
    monkeypatch.setattr(serve_runs.targets, "add_crops", add_crops)
    r = call(env.app, "POST", "/api/targets/alice/add", {"from": "run", "run_id": rid})
    j = wait_job(env.app, r.json()["job"]["id"])
    assert seen.get("added")
    assert j["state"] == "done" and j["result"]["added"] == 1, j
