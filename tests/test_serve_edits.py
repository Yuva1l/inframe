"""inframe serve, the edits store (docs/T9A_DESIGN.md 2.6, rows 39-40): clip rules, revision guard, atomic writes, revert, and the
untrusted-run-folder rules of `serve_edits`. Real motion runs on the synthetic `static_box` clip; no model, no network."""
import json
import math
import shutil
import time
from types import SimpleNamespace

import pytest
from serve_helpers import call, start_server

from inframe import pipeline, serve_edits
from inframe.serve_edits import ClipError, RunFolderError

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
    r = call(app, "POST", "/api/runs", {"media_id": media.id, "target": {"kind": "motion"}, "advanced": ADV})
    assert r.status == 200, r.text
    assert wait_job(app, r.json()["job"]["id"])["state"] == "done"
    rid = r.json()["run_id"]
    d = app.run_or_404(rid)
    yield SimpleNamespace(app=app, media=media, video=video, rid=rid, dir=d, doc=json.loads((d / "segments.json").read_text()),
                          put=lambda rev, clips, rid=rid: call(app, "PUT", f"/api/runs/{rid}/edits", {"revision": rev, "clips": clips}))
    stop()


def clip(i, s, e, origin="added"):
    return {"id": f"u{i}", "start": s, "end": e, "origin": origin}


# ---------------------------------------------------------------- the clip rules (pure)
OK = [clip(1, 1.0, 2.0), clip(2, 2.0, 3.0)]  # touching is not overlapping


def test_valid_list_passes_rounded_to_a_millisecond():
    out = serve_edits.validate_clips([clip(1, 1.00049, 2.0004), clip(2, 2.0004, 3)], 0.0, 8.0)
    assert out == [{"id": "u1", "start": 1.0, "end": 2.0, "origin": "added"}, {"id": "u2", "start": 2.0, "end": 3.0, "origin": "added"}]
    assert serve_edits.validate_clips([], 0.0, 8.0) == []
    assert serve_edits.validate_clips([clip(1, 1.0, 1.2)], 0.0, 8.0)  # exactly the 0.2 s minimum


def test_extra_keys_are_dropped():
    c = dict(clip(1, 1, 2), note="x", evil={"a": 1})
    assert serve_edits.validate_clips([c], 0.0, 8.0) == [clip(1, 1.0, 2.0)]


def test_one_ms_of_slack_at_the_bounds_clamps():
    out = serve_edits.validate_clips([clip(1, 3.0 - 0.0009, 3.5), clip(2, 7.0, 8.0009)], 3.0, 8.0)
    assert out[0]["start"] == 3.0 and out[1]["end"] == 8.0


@pytest.mark.parametrize("clips, why", [
    ([clip(1, 1, 3), clip(2, 2, 4)], "overlap"),
    ([clip(1, 5, 6), clip(2, 1, 2)], "sorted"),
    ([clip(1, 1.0, 1.19)], "shorter"),
    ([clip(1, 1.0, 1.0)], "shorter"),
    ([clip(1, -1.0, 2.0)], "outside"),
    ([clip(1, 7.0, 8.5)], "outside"),
    ([clip(1, 1, 2), clip(1, 3, 4)], "duplicate"),
    ([clip(1, math.nan, 2)], "finite"),
    ([clip(1, 1, math.inf)], "finite"),
    ([clip(1, True, 2)], "finite"),
    ([clip(1, "1", 2)], "finite"),
    ([dict(clip(1, 1, 2), id="A")], "id"),
    ([dict(clip(1, 1, 2), id="x" * 17)], "id"),
    ([dict(clip(1, 1, 2), id=5)], "id"),
    ([dict(clip(1, 1, 2), origin="hacked")], "origin"),
    (["x"], "object"),
    ({"a": 1}, "list"),
])
def test_rules_refuse(clips, why):
    with pytest.raises(ClipError) as e:
        serve_edits.validate_clips(clips, 0.0, 8.0)
    assert e.value.code == 422 and why in e.value.message


def test_more_than_2000_clips_refused_and_2000_allowed():
    many = [{"id": f"u{i}", "start": i * 0.5, "end": i * 0.5 + 0.25, "origin": "added"} for i in range(2001)]
    with pytest.raises(ClipError) as e:
        serve_edits.validate_clips(many, 0.0, 2000.0)
    assert "2000" in e.value.message
    assert len(serve_edits.validate_clips(many[:2000], 0.0, 2000.0)) == 2000


# ---------------------------------------------------------------- facts and samples (pure)
def test_edited_clip_facts():
    times, scores = [0.0, 1.0, 2.0, 3.0, 4.0], [0.1, 0.9, 0.4, 0.8, 0.2]
    kf = [0.0, 1.5, 3.2]
    assert serve_edits.edited_clip_facts(0.5, 2.5, times, scores, kf) == {"peak_score": 0.9, "expected_copy_start": 0.0}
    assert serve_edits.edited_clip_facts(1.6, 3.5, times, scores, kf) == {"peak_score": 0.8, "expected_copy_start": 1.5}
    assert serve_edits.edited_clip_facts(3.3, 3.9, [0.0, 1.0], [0.5, 0.5], kf) == {"peak_score": 0.0, "expected_copy_start": 3.2}
    assert serve_edits.edited_clip_facts(1.0, 2.0, [], [], []) == {"peak_score": 0.0, "expected_copy_start": 1.0}  # no keyframes: the start itself


def test_load_samples_arrays_is_the_inverse_of_write_samples(tmp_path):
    times, scores = [0.007, 0.207, 0.407], [0.0, 0.91234, 1.0]
    boxes = [None, (0.1, 0.2, 0.3, 0.4), (0.5, 0.5, 0.9, 0.95)]
    pipeline.write_samples(tmp_path, times, scores, boxes, [None, 7, 8])
    t, s, b, tr = serve_edits.load_samples_arrays(tmp_path)
    assert t == times and s == scores and b == [None, [0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.9, 0.95]] and tr == [None, 7, 8]
    pipeline.write_samples(tmp_path, times, scores, boxes)  # no tracker: no track column
    assert serve_edits.load_samples_arrays(tmp_path)[3] is None


def test_load_samples_skips_rows_that_are_not_numbers(tmp_path):
    (tmp_path / "samples.csv").write_text("t,score,x0,y0,x1,y1\n0.0,0.5,,,,\nnan,0.5,,,,\n1.0,inf,,,,\n2.0,x,,,,\n3.0,0.7,0.1,0.2,nan,0.4\n")
    t, s, b, tr = serve_edits.load_samples_arrays(tmp_path)
    assert t == [0.0, 3.0] and s == [0.5, 0.7] and b == [None, None] and tr is None


def test_samples_csv_that_is_a_symlink_or_has_no_header_is_refused(tmp_path):
    real = tmp_path / "real.csv"
    real.write_text("t,score\n0,1\n")
    (tmp_path / "samples.csv").symlink_to(real)
    with pytest.raises(RunFolderError) as e:
        serve_edits.load_samples_arrays(tmp_path)
    assert e.value.code == 404
    (tmp_path / "samples.csv").unlink()
    (tmp_path / "samples.csv").write_text("a,b\n1,2\n")
    with pytest.raises(RunFolderError) as e:
        serve_edits.load_samples_arrays(tmp_path)
    assert e.value.code == 422


# ---------------------------------------------------------------- PUT and the revision guard
def test_put_saves_a_revision_and_get_returns_it(env):
    clips = [clip(1, 1.0, 2.5), clip(2, 4.0, 6.0, "edited")]
    r = env.put(0, clips)
    assert r.status == 200, r.text
    doc = r.json()
    assert doc["format"] == 1 and doc["revision"] == 1 and doc["clips"] == clips and doc["next_id"] == 3
    assert doc["base"] == {"segments_sha256": serve_edits.segments_sha256(env.dir), "n_clips": len(env.doc["clips"])}
    assert json.loads((env.dir / "edits.json").read_text()) == doc
    g = call(env.app, "GET", f"/api/runs/{env.rid}").json()
    assert g["edits"] == doc and g["edits_base_ok"] is True
    assert call(env.app, "GET", f"/api/media/{env.media.id}/runs").json()["runs"][0]["has_edits"] is True


def test_no_tmp_file_is_left_behind(env):
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 200
    assert env.put(1, [clip(1, 1.0, 2.5)]).status == 200
    assert sorted(p.name for p in env.dir.iterdir() if p.name.startswith("edits")) == ["edits.json"]
    assert not list(env.dir.glob("*.tmp"))


def test_stale_revision_is_409_with_the_stored_document(env):
    first = env.put(0, [clip(1, 1.0, 2.0)]).json()
    second = env.put(1, [clip(1, 1.0, 2.0), clip(2, 3.0, 4.0)]).json()
    r = env.put(1, [clip(5, 5.0, 6.0)])  # edited revision 1, but 2 is stored
    assert r.status == 409
    assert r.json()["error"] == "conflict" and r.json()["edits"] == second and first["revision"] == 1
    assert json.loads((env.dir / "edits.json").read_text()) == second  # untouched


def test_conflict_without_a_stored_file_carries_the_detector_clips(env):
    r = env.put(4, [clip(1, 1.0, 2.0)])
    assert r.status == 409
    stored = r.json()["edits"]
    assert stored["revision"] == 0 and stored["clips"] == [
        {"id": f"d{i}", "start": c["start"], "end": c["end"], "origin": "detector"} for i, c in enumerate(env.doc["clips"], 1)]
    assert not (env.dir / "edits.json").exists()


def test_next_id_stays_above_every_user_id(env):
    r = env.put(0, [clip(31, 1.0, 2.0), clip(7, 3.0, 4.0)]).json()
    assert r["next_id"] == 32
    r = env.put(1, [clip(3, 1.0, 2.0)]).json()
    assert r["next_id"] == 32  # never goes back


def test_rules_are_enforced_over_http(env):
    end = env.doc["source"]["duration"]
    cases = [[clip(1, 1, 3), clip(2, 2, 4)], [clip(1, 5, 6), clip(2, 1, 2)], [clip(1, 1.0, 1.19)], [clip(1, 7.0, end + 1)],
             [clip(1, -2.0, 1.0)], [clip(1, 1, 2), clip(1, 3, 4)], [clip(1, math.nan, 2)],
             [{"id": f"u{i}", "start": i * 0.003, "end": i * 0.003 + 0.2, "origin": "added"} for i in range(2001)]]
    for bad in cases:
        r = env.put(0, bad)
        assert r.status == 422, (bad[:2], r.text)
    assert not (env.dir / "edits.json").exists()


def test_put_body_shape(env):
    for body in ({"clips": []}, {"revision": "0", "clips": []}, {"revision": True, "clips": []}, {"revision": -1, "clips": []},
                 {"revision": 0}, {"revision": 0, "clips": {}}):
        assert call(env.app, "PUT", f"/api/runs/{env.rid}/edits", body).status == 400, body
    assert call(env.app, "PUT", "/api/runs/000000000000/edits", {"revision": 0, "clips": []}).status == 404
    assert call(env.app, "PUT", f"/api/runs/{env.rid}/edits", {"revision": 0, "clips": []}).status == 200  # an empty list is a valid edit


def test_edits_made_on_another_run_are_flagged(env):
    env.put(0, [clip(1, 1.0, 2.0)])
    p = env.dir / "segments.json"
    p.write_text(p.read_text() + "\n")  # a different run of the same folder would change the bytes
    assert call(env.app, "GET", f"/api/runs/{env.rid}").json()["edits_base_ok"] is False
    assert call(env.app, "GET", f"/api/media/{env.media.id}/runs").json()["runs"][0]["edits_base_ok"] is False


# ---------------------------------------------------------------- DELETE (revert)
def test_delete_moves_the_edits_aside(env):
    env.put(0, [clip(1, 1.0, 2.0)])
    r = call(env.app, "DELETE", f"/api/runs/{env.rid}/edits")
    assert r.status == 200 and r.json() == {"ok": True}
    assert not (env.dir / "edits.json").exists()
    prev = [p.name for p in env.dir.iterdir() if p.name.startswith("edits.prev-")]
    assert len(prev) == 1 and prev[0].endswith("Z.json")
    assert json.loads((env.dir / prev[0]).read_text())["clips"] == [clip(1, 1.0, 2.0)]
    assert call(env.app, "GET", f"/api/runs/{env.rid}").json()["edits"] is None
    assert call(env.app, "DELETE", f"/api/runs/{env.rid}/edits").json() == {"ok": True}  # nothing left to move: still ok
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 200  # a fresh start at revision 0


def test_only_five_earlier_edit_files_are_kept(env):
    for i in range(7):
        assert env.put(0, [clip(1, 1.0, 2.0 + i * 0.1)]).status == 200
        assert call(env.app, "DELETE", f"/api/runs/{env.rid}/edits").status == 200
    prev = sorted(p.name for p in env.dir.iterdir() if p.name.startswith("edits.prev-"))
    assert len(prev) == 5
    kept = [json.loads((env.dir / n).read_text())["clips"][0]["end"] for n in prev]
    assert max(kept) == pytest.approx(2.6) and min(kept) == pytest.approx(2.2)  # the oldest two went


# ---------------------------------------------------------------- untrusted run folders
def test_a_symlinked_edits_file_counts_as_no_edits_and_is_never_written_through(env, tmp_path):
    secret = tmp_path / "outside.json"
    secret.write_text("keep me")
    (env.dir / "edits.json").symlink_to(secret)
    assert call(env.app, "GET", f"/api/runs/{env.rid}").json()["edits"] is None
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 200
    assert secret.read_text() == "keep me"
    assert not (env.dir / "edits.json").is_symlink()


def test_a_planted_tmp_symlink_is_not_followed(env, tmp_path):
    secret = tmp_path / "outside2.json"
    secret.write_text("keep me")
    (env.dir / "edits.json.tmp").symlink_to(secret)
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 200
    assert secret.read_text() == "keep me"
    assert not (env.dir / "edits.json.tmp").exists()


def test_a_symlinked_segments_json_is_not_served(env, tmp_path):
    real = tmp_path / "moved.json"
    shutil.move(env.dir / "segments.json", real)
    (env.dir / "segments.json").symlink_to(real)
    assert call(env.app, "GET", f"/api/runs/{env.rid}").status == 404
    assert call(env.app, "GET", f"/api/runs/{env.rid}/samples").status == 404
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 404
    assert call(env.app, "POST", f"/api/runs/{env.rid}/resegment", {}).status == 404


def test_a_run_folder_replaced_by_a_symlink_is_not_served(env, tmp_path):
    moved = tmp_path / "moved_run"
    shutil.move(str(env.dir), moved)
    env.dir.symlink_to(moved, target_is_directory=True)
    assert call(env.app, "GET", f"/api/runs/{env.rid}").status == 404


def test_a_run_whose_video_is_not_open_is_not_served(env):
    doc = json.loads((env.dir / "segments.json").read_text())
    doc["source"]["path"] = str(env.video.with_name("other.mp4"))
    (env.dir / "segments.json").write_text(json.dumps(doc))
    assert call(env.app, "GET", f"/api/runs/{env.rid}").status == 404
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 404


def test_a_segments_json_that_is_not_ours_is_refused(env):
    (env.dir / "segments.json").write_text("[1, 2, 3]")
    assert call(env.app, "GET", f"/api/runs/{env.rid}").status == 404
    (env.dir / "segments.json").write_text(json.dumps({"source": {"path": str(env.media.path), "duration": "x"}, "clips": []}))
    assert env.put(0, [clip(1, 1.0, 2.0)]).status == 422  # source times unusable


# ---------------------------------------------------------------- a video whose picture starts late (offset 2 s)
@pytest.fixture
def late_env(tmp_path):
    """6 s of picture that starts at 2.0 s in the file; a white box is in frame from 0.6 s to 3 s of picture time (2.6 to 5 s in the file)."""
    from conftest import _ff

    vids = tmp_path / "vids"
    vids.mkdir()
    plain, video = vids / "plain.mp4", vids / "late_box.mov"
    _ff("-f", "lavfi", "-i", "color=c=0x404040:s=320x180:r=10:d=6,drawgrid=w=40:h=40:t=2:c=white@0.4,noise=alls=4:allf=t,"
        "drawbox=x=120:y=60:w=50:h=70:color=white:t=fill:enable='between(t,0.6,3)'", "-f", "lavfi", "-i", "sine=frequency=330:duration=8", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", str(plain))
    _ff("-itsoffset", "2", "-i", str(plain), "-itsoffset", "0", "-i", str(plain), "-map", "0:v", "-map", "1:a", "-c", "copy", str(video))
    base, app, stop = start_server(tmp_path)
    media = app.open_media(str(video))
    assert media.info.offset == pytest.approx(2.0, abs=0.05)
    r = call(app, "POST", "/api/runs", {"media_id": media.id, "target": {"kind": "motion"}, "advanced": dict(ADV, pad=1.0)})
    assert r.status == 200, r.text
    assert wait_job(app, r.json()["job"]["id"])["state"] == "done"
    rid = r.json()["run_id"]
    yield SimpleNamespace(app=app, rid=rid, dir=app.run_or_404(rid), off=media.info.offset)
    stop()


def test_detector_clips_stay_inside_a_late_video_and_can_be_saved(late_env):
    e = late_env
    doc = json.loads((e.dir / "segments.json").read_text())
    assert doc["clips"], doc
    assert all(c["start"] >= e.off - 0.002 for c in doc["clips"]), (e.off, doc["clips"])
    d1 = serve_edits.detector_edits_doc(doc, serve_edits.segments_sha256(e.dir))["clips"]  # what the UI starts from
    assert d1 and all(c["origin"] == "detector" for c in d1)
    r = call(e.app, "PUT", f"/api/runs/{e.rid}/edits", {"revision": 0, "clips": d1})
    assert r.status == 200, r.text
    p = call(e.app, "POST", f"/api/runs/{e.rid}/resegment", {}).json()
    assert all(c[0] >= e.off - 0.002 for c in p["clips"]), p
    r = call(e.app, "PUT", f"/api/runs/{e.rid}/edits", {"revision": 1, "clips": [
        {"id": f"u{i}", "start": a, "end": b, "origin": "resegment"} for i, (a, b) in enumerate(p["clips"], 1)]})
    assert r.status == 200, r.text
