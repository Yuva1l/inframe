"""Part A3: the pick and targets routes of `inframe serve` (docs/T9A_DESIGN.md 2.8 and 3.3, routes 19-33).
A fake detector stands in for the model, `pipeline.score_detect` and `people.scan_people` are faked for the jobs; the videos are the
synthetic clips of conftest.py (copied into tmp_path so the `<stem>_inframe` folder is per test); no real footage, no network."""
import io
import json
import shutil
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from inframe import people, pick, picker, pipeline, serve_paths, serve_pick
from inframe.detect import Detection

from serve_helpers import call, start_server


class FakeDet:
    """Two boxes above the 0.4 line, one below it; records the class asked for."""

    def __init__(self):
        self.classes = []

    def detect(self, frames, cls=None):
        self.classes.append(cls)
        return [[Detection(0.9, (0.1, 0.1, 0.5, 0.9)), Detection(0.6, (0.5, 0.1, 0.9, 0.9)), Detection(0.3, (0.0, 0.0, 0.1, 0.1))]
                for _ in frames]


@pytest.fixture
def env(tmp_path, monkeypatch, synth):
    monkeypatch.setenv("INFRAME_TARGETS", str(tmp_path / "store"))
    monkeypatch.setattr(pick, "load_detector", lambda size, units=None: FakeDet())
    monkeypatch.setattr(pick.Picker, "local_track_id", lambda self, ft, box: 3)  # the tracking window costs seconds and is pick.py's own
    vid = tmp_path / "vid"
    vid.mkdir()
    video = vid / "clip.mp4"
    shutil.copy(synth["plain"], video)
    base, app, stop = start_server(tmp_path)
    media = app.open_media(str(video))
    yield SimpleNamespace(app=app, media=media, mid=media.id, tmp=tmp_path, video=video)
    stop()


def url(env, tail):
    return f"/api/media/{env.mid}/{tail}"


def out_root(env):
    return env.app.out_root(env.media)


def tfile(env, cls="person", key=""):
    return serve_paths.target_file(out_root(env), cls, key)


def tap(env, cls="person", t=1.0, box=(0.1, 0.1, 0.5, 0.9), index=1):
    return call(env.app, "POST", url(env, "pick/tap"), {"cls": cls, "t": t, "box": list(box), "index": index})


def wait_job(env, jid, states=("done", "failed", "cancelled", "needs_confirm"), timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        j = call(env.app, "GET", f"/api/jobs/{jid}").json()
        if j["state"] in states:
            return j
        time.sleep(0.02)
    raise AssertionError(f"job {jid} still {j['state']}")


def make_photos(folder, n=2):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 96), (i * 40, 90, 120)).save(folder / f"p{i}.jpg")
    return folder


def photos_id(env, folder):
    r = call(env.app, "POST", "/api/photos/choose", {"source": "path", "path": str(folder)})
    assert r.status == 200, r.text
    return r.json()["id"]


CANDS = [dict(rank=1, t=1.0, box=[0.1, 0.1, 0.5, 0.9], track_id=7, score=0.81, extra="x"),
         dict(rank=2, t=2.0, box=[0.5, 0.1, 0.9, 0.9], track_id=8, score=0.55)]


@pytest.fixture
def fake_rank(monkeypatch):
    """score_detect replaced: reports progress in steps, returns two ranked candidates; `calls` records what it was given."""
    calls = []

    def fake(video, info, preset, fps, target, progress, tracker_kind):
        calls.append(SimpleNamespace(target=target, progress=progress, preset=preset))
        for k in range(1, 5):
            if progress:
                progress(k / 4, f"step {k}")
        return SimpleNamespace(info={"target_info": {"ranking": [dict(c) for c in CANDS]}})

    monkeypatch.setattr(pipeline, "score_detect", fake)
    fake.calls = calls
    return fake


# ---------------------------------------------------------------- routes 19-20: boxes and the frame
def test_boxes_are_numbered_and_below_the_line_are_left_out(env):
    r = call(env.app, "GET", url(env, "boxes?t=1.0&cls=person"))
    assert r.status == 200
    d = r.json()
    assert [b["i"] for b in d["boxes"]] == [1, 2]
    assert d["boxes"][0]["box"] == [0.1, 0.1, 0.5, 0.9] and d["boxes"][0]["score"] == 0.9
    assert d["t"] == pytest.approx(1.0, abs=0.1) and d["t"] == round(d["t"], 3)


def test_boxes_for_another_class_ask_the_detector_for_that_class(env):
    assert call(env.app, "GET", url(env, "boxes?t=1.0&cls=dog")).status == 200
    pk = env.app.pickers[(env.mid, "dog", "")]
    assert pk.det.classes == [18]
    assert (env.mid, "person", "") not in env.app.pickers  # a dog request makes no person picker


def test_the_same_picker_is_reused_for_the_same_key(env):
    call(env.app, "GET", url(env, "boxes?t=1.0"))
    pk = env.app.pickers[(env.mid, "person", "")]
    call(env.app, "GET", url(env, "boxes?t=2.0"))
    call(env.app, "GET", url(env, "pick"))
    assert env.app.pickers[(env.mid, "person", "")] is pk and len(env.app.pickers) == 1


def test_a_superseded_request_is_409(env, monkeypatch):
    def gone(self, t):
        raise pick.Superseded()

    monkeypatch.setattr(serve_pick.ServePicker, "frame", gone)
    r = call(env.app, "GET", url(env, "boxes?t=1.0"))
    assert r.status == 409 and r.json() == {"error": "superseded"}


@pytest.mark.parametrize("q", ["", "t=abc", "t=nan", "t=inf", "t=1&cls=unicorn", "t=1&cls=" + "x" * 50])
def test_boxes_bad_input_is_400(env, q):
    r = call(env.app, "GET", url(env, "boxes?" + q))
    assert r.status == 400 and "error" in r.json()


def test_unknown_media_is_404(env):
    assert call(env.app, "GET", "/api/media/nope/boxes?t=1").status == 404


def test_a_missing_model_is_422_with_its_message(env, monkeypatch):
    from inframe import models

    def boom(size, units=None):
        raise models.ModelsMissing("no model here: run inframe fetch-models")

    monkeypatch.setattr(pick, "load_detector", boom)
    r = call(env.app, "GET", url(env, "boxes?t=1.0"))
    assert r.status == 422 and "fetch-models" in r.json()["error"]


def test_pickframe_is_the_frame_boxes_answered_for(env):
    t = call(env.app, "GET", url(env, "boxes?t=1.0")).json()["t"]
    r = call(env.app, "GET", url(env, f"pickframe.jpg?t={t}"), key="media")
    assert r.status == 200 and r.headers["content-type"] == "image/jpeg"
    got = Image.open(io.BytesIO(r.body))
    _, rgb = pick.grab_frame(str(env.video), env.media.info, t - 5e-4)
    assert got.size == (rgb.shape[1], rgb.shape[0])


def test_pickframe_takes_the_media_key_and_boxes_do_not(env):
    assert call(env.app, "GET", url(env, "boxes?t=1.0"), key="media").status == 403
    assert call(env.app, "GET", url(env, "pickframe.jpg?t=1.0"), key=None).status == 403
    assert call(env.app, "GET", url(env, "pickframe.jpg?t=1.0"), key="media").status == 200


# ---------------------------------------------------------------- routes 21-25: taps, zone, undo, reset
def test_a_tap_writes_the_target_file_of_that_class(env):
    r = tap(env)
    assert r.status == 200, r.text
    st = r.json()
    f = tfile(env)
    assert f.parent == out_root(env) / "targets" and f.name == "target-person.json"
    assert st["file"] == str(f) and st["target_key"] == "" and st["class"] == "person" and st["zone"] is None
    assert [(t["time"], t["box"], t["index"]) for t in st["taps"]] == [(1.0, [0.1, 0.1, 0.5, 0.9], 1)]
    doc = json.loads(f.read_text())
    assert doc["video"] == str(env.video.resolve()) and doc["preset"] == "balanced" and doc["model"] == "small"
    assert doc["taps"][0]["box"] == [0.1, 0.1, 0.5, 0.9] and "class" not in doc
    again = call(env.app, "GET", url(env, "pick?cls=person")).json()
    assert again == st


def test_taps_add_up_and_undo_removes_the_last(env):
    tap(env, t=1.0)
    tap(env, t=2.0)
    assert len(call(env.app, "GET", url(env, "pick")).json()["taps"]) == 2
    r = call(env.app, "POST", url(env, "pick/undo"), {"cls": "person"}).json()
    assert [t["time"] for t in r["taps"]] == [1.0] and r["file"] == str(tfile(env))


def test_undo_of_the_last_tap_removes_the_file(env):
    tap(env)
    assert tfile(env).is_file()
    r = call(env.app, "POST", url(env, "pick/undo"), {"cls": "person"}).json()
    assert r["taps"] == [] and r["file"] is None and not tfile(env).exists()
    assert call(env.app, "POST", url(env, "pick/undo"), {"cls": "person"}).status == 200  # nothing to undo is fine


def test_a_dog_tap_and_a_person_reset_leave_the_dog_file_intact(env):
    tap(env, "person")
    r = tap(env, "dog", box=(0.5, 0.1, 0.9, 0.9), index=2)
    assert r.status == 200 and r.json()["class"] == "dog"
    dog, person = tfile(env, "dog"), tfile(env, "person")
    assert dog.name == "target-dog.json" and dog.is_file() and person.is_file()
    before = dog.read_bytes()
    st = call(env.app, "POST", url(env, "pick/reset"), {"cls": "person"}).json()
    assert st["taps"] == [] and st["file"] is None
    assert not person.exists() and dog.read_bytes() == before
    assert json.loads(before)["class"] == "dog"
    assert len(call(env.app, "GET", url(env, "pick?cls=dog")).json()["taps"]) == 1


def test_the_zone_lands_in_the_file_and_null_clears_it(env):
    tap(env)
    r = call(env.app, "POST", url(env, "pick/zone"), {"cls": "person", "zone": [0.1, 0.2, 0.5, 0.6]})
    assert r.status == 200 and r.json()["zone"] == [0.1, 0.2, 0.5, 0.6]
    assert json.loads(tfile(env).read_text())["zone"] == [0.1, 0.2, 0.5, 0.6]
    r = call(env.app, "POST", url(env, "pick/zone"), {"cls": "person", "zone": None})
    assert r.json()["zone"] is None and "zone" not in json.loads(tfile(env).read_text())


@pytest.mark.parametrize("z", [[0.1, 0.2, 0.5], [0.5, 0.5, 0.9, 0.9], [0, 0, 0, 0.5], ["a", 0, 1, 1], [0.1, 0.1, 0.2, True], "0,0,1,1", [0.1, 0.1, 1e999, 1]])
def test_a_bad_zone_is_400_and_changes_nothing(env, z):
    tap(env)
    r = call(env.app, "POST", url(env, "pick/zone"), {"cls": "person", "zone": z})
    assert r.status == 400 and "error" in r.json()
    assert "zone" not in json.loads(tfile(env).read_text())


def test_zone_is_required(env):
    assert call(env.app, "POST", url(env, "pick/zone"), {"cls": "person"}).status == 400


def test_reset_clears_taps_and_zone(env):
    tap(env)
    call(env.app, "POST", url(env, "pick/zone"), {"cls": "person", "zone": [0, 0, 1, 1]})
    st = call(env.app, "POST", url(env, "pick/reset"), {"cls": "person"}).json()
    assert st["taps"] == [] and st["zone"] is None and not tfile(env).exists()


@pytest.mark.parametrize("body", [
    {"t": 1.0, "box": [0.1, 0.1, 0.5], "index": 1},
    {"t": 1.0, "box": [0.5, 0.1, 0.1, 0.9], "index": 1},
    {"t": 1.0, "box": [0.1, 0.1, 1.5, 0.9], "index": 1},
    {"t": 1.0, "box": [0.1, 0.1, "x", 0.9], "index": 1},
    {"t": "1", "box": [0.1, 0.1, 0.5, 0.9]},
    {"box": [0.1, 0.1, 0.5, 0.9]},
    {"t": 99.0, "box": [0.1, 0.1, 0.5, 0.9]},
    {"t": -5.0, "box": [0.1, 0.1, 0.5, 0.9]},
    {"t": 1.0, "box": [0.1, 0.1, 0.5, 0.9], "index": "1"},
    {"t": 1.0, "box": [0.1, 0.1, 0.5, 0.9], "cls": "unicorn"},
])
def test_a_bad_tap_is_400_and_writes_nothing(env, body):
    r = call(env.app, "POST", url(env, "pick/tap"), body)
    assert r.status == 400 and "error" in r.json()
    assert not tfile(env).exists()


def test_a_tap_without_index_is_fine(env):
    r = call(env.app, "POST", url(env, "pick/tap"), {"t": 1.0, "box": [0.1, 0.1, 0.5, 0.9]})
    assert r.status == 200 and r.json()["taps"][0]["index"] is None


def test_the_picker_state_routes_need_the_session_token(env):
    assert call(env.app, "GET", url(env, "pick"), key=None).status == 403
    assert call(env.app, "POST", url(env, "pick/tap"), {"t": 1, "box": [0, 0, 1, 1]}, key=None).status == 403
    assert call(env.app, "GET", url(env, "pick"), key="media").status == 403


# ---------------------------------------------------------------- routes 26-27: saved targets
def make_saved(store, name="alice", n=2, broken=False):
    d = store / name
    (d / "crops").mkdir(parents=True, exist_ok=True)
    crops = []
    for i in range(n):
        Image.new("RGB", (32, 48), (200, 20 * i, 20)).save(d / "crops" / f"c{i + 1:04d}.jpg")
        crops.append({"file": f"crops/c{i + 1:04d}.jpg", "source": "video:x"})
    doc = {"format": 1, "name": name, "created": "2026-10-16T00:00:00+00:00", "updated": "2026-10-16T00:00:00+00:00",
           "crops": crops, "embeddings": {}}
    (d / "target.json").write_text("not json" if broken else json.dumps(doc))
    return d


def test_targets_of_an_empty_store_is_an_empty_list(env):
    r = call(env.app, "GET", "/api/targets")
    assert r.status == 200 and r.json() == {"targets": []}


def test_targets_lists_summaries_and_unreadable_ones(env):
    store = env.tmp / "store"
    make_saved(store, "alice", 2)
    make_saved(store, "broken", 1, broken=True)
    got = {t["name"]: t for t in call(env.app, "GET", "/api/targets").json()["targets"]}
    assert got["alice"]["n_crops"] == 2 and got["alice"]["embeddings"] == {}
    assert set(got["broken"]) == {"name", "error"}


def test_a_crop_is_a_jpeg_served_with_the_media_key(env):
    make_saved(env.tmp / "store")
    r = call(env.app, "GET", "/api/targets/alice/crops/1.jpg", key="media")
    assert r.status == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.body)).size == (32, 48)
    assert call(env.app, "GET", "/api/targets/alice/crops/0.jpg", key="media").status == 200
    assert call(env.app, "GET", "/api/targets/alice/crops/2.jpg", key="media").status == 404
    assert call(env.app, "GET", "/api/targets/alice/crops/0.jpg", key=None).status == 403


@pytest.mark.parametrize("name", ["..", "a%20b", "-x", ".hidden", "x" * 70])
def test_a_bad_target_name_is_400(env, name):
    r = call(env.app, "GET", f"/api/targets/{name}/crops/0.jpg", key="media")
    assert r.status == 400 and "error" in r.json()


def test_an_unknown_target_name_is_404(env):
    assert call(env.app, "GET", "/api/targets/ghost/crops/0.jpg", key="media").status == 404


def test_a_crop_that_is_a_symlink_to_another_file_is_not_served(env):
    d = make_saved(env.tmp / "store")
    secret = env.tmp / "secret.txt"
    secret.write_text("secret")
    (d / "crops" / "c0001.jpg").unlink()
    (d / "crops" / "c0001.jpg").symlink_to(secret)
    r = call(env.app, "GET", "/api/targets/alice/crops/0.jpg", key="media")
    assert r.status == 404 and b"secret" not in r.body


# ---------------------------------------------------------------- route 28: photos
def test_photos_choose_by_path_counts_the_photos(env):
    folder = make_photos(env.tmp / "faces", 3)
    r = call(env.app, "POST", "/api/photos/choose", {"source": "path", "path": str(folder)})
    assert r.status == 200
    d = r.json()
    assert d == {"id": "p1", "dir": str(folder.resolve()), "n_photos": 3}
    assert env.app.photos_dirs["p1"] == folder.resolve()


def test_the_same_folder_keeps_its_id_and_another_gets_a_new_one(env):
    a, b = make_photos(env.tmp / "a"), make_photos(env.tmp / "b")
    assert photos_id(env, a) == "p1" and photos_id(env, b) == "p2" and photos_id(env, a) == "p1"


@pytest.mark.parametrize("make, status", [
    (lambda t: str(t / "nothing"), 404),
    (lambda t: str(t / "file.txt"), 404),
    (lambda t: str(t / ".hidden"), 403),
    (lambda t: "relative/dir", 400),
    (lambda t: str(t / "empty"), 422),
])
def test_photos_choose_by_path_refusals(env, make, status):
    (env.tmp / "file.txt").write_text("x")
    (env.tmp / ".hidden").mkdir()
    make_photos(env.tmp / ".hidden")
    (env.tmp / "empty").mkdir()
    r = call(env.app, "POST", "/api/photos/choose", {"source": "path", "path": make(env.tmp)})
    assert r.status == status and "error" in r.json()
    assert env.app.photos_dirs == {}


@pytest.mark.parametrize("body", [{}, {"source": "url"}, {"source": "path"}, {"source": "path", "path": 5}])
def test_photos_choose_bad_input_is_400(env, body):
    assert call(env.app, "POST", "/api/photos/choose", body).status == 400


def test_photos_dialog_returns_the_folder_or_cancelled(env, monkeypatch):
    folder = make_photos(env.tmp / "chosen")
    monkeypatch.setattr(serve_pick, "is_mac", lambda: True)
    monkeypatch.setattr(serve_pick, "run_osascript", lambda script: str(folder) + "/")
    r = call(env.app, "POST", "/api/photos/choose", {"source": "dialog"})
    assert r.status == 200 and r.json()["n_photos"] == 2
    monkeypatch.setattr(serve_pick, "run_osascript", lambda script: None)
    assert call(env.app, "POST", "/api/photos/choose", {"source": "dialog"}).json() == {"cancelled": True}


def test_photos_dialog_and_picker_are_501_off_macos(env, monkeypatch):
    monkeypatch.setattr(serve_pick, "is_mac", lambda: False)
    monkeypatch.delenv(picker.HELPER_ENV, raising=False)
    assert call(env.app, "POST", "/api/photos/choose", {"source": "dialog"}).status == 501
    assert call(env.app, "POST", "/api/photos/choose", {"source": "photos"}).status == 501


def test_photos_from_the_photos_app_flattens_the_helper_folders(env, monkeypatch):
    seen = {}

    def fake_pick(out_dir, limit=0, announce=lambda m: None, kind="videos"):
        seen["kind"] = kind
        files = []
        for i, name in enumerate(["IMG_1.jpg", "IMG_1.jpg"], 1):  # the helper numbers a folder per photo: equal names are fine
            d = out_dir / f"{i:02d}"
            d.mkdir()
            Image.new("RGB", (32, 32), (i * 50, 0, 0)).save(d / name)
            files.append(d / name)
        return picker.PickResult(files=files)

    monkeypatch.setattr(serve_pick, "is_mac", lambda: True)
    monkeypatch.setattr(picker, "pick", fake_pick)
    r = call(env.app, "POST", "/api/photos/choose", {"source": "photos"})
    assert r.status == 200, r.text
    d = r.json()
    assert seen["kind"] == "images" and d["n_photos"] == 2
    folder = env.app.photos_dirs[d["id"]]
    assert folder.name == "photos" and folder.parent.name.startswith(serve_paths.PICKED_PREFIX)
    assert sorted(p.name for p in folder.iterdir()) == ["01_IMG_1.jpg", "02_IMG_1.jpg"]


def test_photos_picker_cancel_and_error(env, monkeypatch):
    monkeypatch.setattr(serve_pick, "is_mac", lambda: True)
    monkeypatch.setattr(picker, "pick", lambda *a, **k: picker.PickResult(cancelled=True))
    assert call(env.app, "POST", "/api/photos/choose", {"source": "photos"}).json() == {"cancelled": True}

    def fail(*a, **k):
        raise picker.PickerError("swiftc not found: install the Xcode command line tools")

    monkeypatch.setattr(picker, "pick", fail)
    r = call(env.app, "POST", "/api/photos/choose", {"source": "photos"})
    assert r.status == 422 and "swiftc" in r.json()["error"]


# ---------------------------------------------------------------- routes 29-31: rank, candidates, confirm
def test_a_new_photos_id_makes_a_new_picker_and_the_same_id_reuses_it(env, fake_rank):
    p1, p2 = photos_id(env, make_photos(env.tmp / "a")), photos_id(env, make_photos(env.tmp / "b"))

    def rank(pid):
        r = call(env.app, "POST", url(env, "rank"), {"photos_id": pid})
        assert r.status == 200, r.text
        wait_job(env, r.json()["job"]["id"])
        return env.app.pickers[(env.mid, "person", f"photos:{pid}")]

    a, b = rank(p1), rank(p2)
    assert a is not b and a.photos != b.photos and a.out != b.out
    assert sorted(k for k in env.app.pickers if k[0] == env.mid) == [(env.mid, "person", "photos:p1"), (env.mid, "person", "photos:p2")]
    assert rank(p1) is a and rank(p2) is b  # identity, not equality
    assert len(env.app.pickers) == 2


def test_rank_reports_progress_and_returns_candidates(env, fake_rank):
    pid = photos_id(env, make_photos(env.tmp / "faces"))
    r = call(env.app, "POST", url(env, "rank"), {"photos_id": pid, "cls": "person"})
    assert r.status == 200
    job = r.json()["job"]
    assert job["kind"] == "rank" and job["media_id"] == env.mid
    done = wait_job(env, job["id"])
    assert done["state"] == "done" and done["fraction"] == 1.0
    assert done["message"] == "step 4"  # the progress callback reached score_detect
    assert done["result"] == {"candidates": [{k: c.get(k) for k in ("rank", "t", "box", "track_id", "score")} for c in CANDS]}
    call_ = fake_rank.calls[0]
    assert call_.target.rank_only and call_.target.photos == str(env.app.photos_dirs[pid]) and call_.target.cls == 1
    assert call_.preset.model == "small"


def test_rank_can_be_cancelled(env, monkeypatch):
    started = threading.Event()

    def forever(video, info, preset, fps, target, progress, tracker_kind):
        started.set()
        while True:
            progress(0.3, "scanning")
            time.sleep(0.01)

    monkeypatch.setattr(pipeline, "score_detect", forever)
    pid = photos_id(env, make_photos(env.tmp / "faces"))
    job = call(env.app, "POST", url(env, "rank"), {"photos_id": pid}).json()["job"]
    assert started.wait(10)
    assert call(env.app, "POST", f"/api/jobs/{job['id']}/cancel", {}).status == 200
    assert wait_job(env, job["id"], timeout=5)["state"] == "cancelled"
    assert env.app.pickers[(env.mid, "person", "photos:p1")].cands == []


def test_a_second_job_while_one_runs_is_409_busy(env, monkeypatch):
    stop = threading.Event()

    def slow(video, info, preset, fps, target, progress, tracker_kind):
        while not stop.is_set():
            progress(0.1, "x")
            time.sleep(0.01)
        return SimpleNamespace(info={})

    monkeypatch.setattr(pipeline, "score_detect", slow)
    pid = photos_id(env, make_photos(env.tmp / "faces"))
    job = call(env.app, "POST", url(env, "rank"), {"photos_id": pid}).json()["job"]
    r = call(env.app, "POST", url(env, "rank"), {"photos_id": pid})
    assert r.status == 409 and r.json()["error"] == "busy" and r.json()["job"]["id"] == job["id"]
    assert call(env.app, "POST", url(env, "people"), {}).status == 409
    stop.set()
    wait_job(env, job["id"])


@pytest.mark.parametrize("body, status", [
    ({}, 400),
    ({"photos_id": "p1", "saved": "alice"}, 400),
    ({"photos_id": 3}, 400),
    ({"photos_id": "p9"}, 404),
    ({"saved": "../x"}, 400),
    ({"saved": 7}, 400),
    ({"saved": "ghost"}, 404),
    ({"photos_id": "p1", "cls": "unicorn"}, 400),
])
def test_rank_refusals(env, fake_rank, body, status):
    photos_id(env, make_photos(env.tmp / "faces"))
    r = call(env.app, "POST", url(env, "rank"), body)
    assert r.status == status and "error" in r.json()
    assert fake_rank.calls == [] and call(env.app, "GET", "/api/jobs/current").json() == {"job": None}


def test_rank_of_a_saved_target_uses_its_own_picker_and_file(env, fake_rank):
    make_saved(env.tmp / "store")
    r = call(env.app, "POST", url(env, "rank"), {"saved": "alice"})
    assert r.status == 200, r.text
    assert wait_job(env, r.json()["job"]["id"])["state"] == "done"
    pk = env.app.pickers[(env.mid, "person", "saved:alice")]
    assert pk.photos == "saved:alice" and pk.target_key == "saved:alice"
    assert pk.out == tfile(env, "person", "saved:alice") and pk.out.name == "target-person-saved-alice.json"
    assert fake_rank.calls[0].target.photos == "saved:alice"


def test_rank_of_a_stale_saved_target_is_422(env, fake_rank):
    d = make_saved(env.tmp / "store")
    (d / "crops" / "c0001.jpg").unlink()
    r = call(env.app, "POST", url(env, "rank"), {"saved": "alice"})
    assert r.status == 422 and "thumbnail" in r.json()["error"]


def test_a_failed_rank_job_carries_the_message(env, monkeypatch):
    def boom(*a):
        raise RuntimeError("tracking failed (the tracking extra is missing)")

    monkeypatch.setattr(pipeline, "score_detect", boom)
    pid = photos_id(env, make_photos(env.tmp / "faces"))
    job = call(env.app, "POST", url(env, "rank"), {"photos_id": pid}).json()["job"]
    done = wait_job(env, job["id"])
    assert done["state"] == "failed" and "tracking extra" in done["error"]


def ranked(env, fake_rank):
    pid = photos_id(env, make_photos(env.tmp / "faces"))
    job = call(env.app, "POST", url(env, "rank"), {"photos_id": pid}).json()["job"]
    assert wait_job(env, job["id"])["state"] == "done"
    return pid


def test_a_candidate_crop_is_a_jpeg(env, fake_rank):
    ranked(env, fake_rank)
    r = call(env.app, "GET", url(env, "cands/0.jpg"), key="media")
    assert r.status == 200 and r.headers["content-type"] == "image/jpeg"
    im = Image.open(io.BytesIO(r.body))
    assert im.format == "JPEG" and max(im.size) <= 360
    assert call(env.app, "GET", url(env, "cands/1.jpg"), key="media").status == 200
    assert call(env.app, "GET", url(env, "cands/2.jpg"), key="media").status == 404
    assert call(env.app, "GET", url(env, "cands/0.jpg"), key=None).status == 403


def test_candidates_before_any_rank_are_404(env):
    assert call(env.app, "GET", url(env, "cands/0.jpg"), key="media").status == 404
    assert call(env.app, "POST", url(env, "pick/confirm"), {"cls": "person", "i": 0}).status == 404


def test_confirm_writes_the_tap_into_the_photos_target_file(env, fake_rank):
    ranked(env, fake_rank)
    r = call(env.app, "POST", url(env, "pick/confirm"), {"cls": "person", "i": 1})
    assert r.status == 200, r.text
    st = r.json()
    f = tfile(env, "person", "photos:p1")
    assert f.name == "target-person-photos-p1.json" and st["file"] == str(f) and st["target_key"] == "photos:p1"
    assert [(t["time"], t["box"], t["track_id"]) for t in st["taps"]] == [(2.0, [0.5, 0.1, 0.9, 0.9], 8)]
    assert [c["rank"] for c in st["candidates"]] == [1, 2]
    doc = json.loads(f.read_text())
    assert doc["photos"] == str((env.tmp / "faces").resolve()) and doc["embedder"] == "dinov2" and doc["taps"][0]["candidate"] == 2
    assert not tfile(env).exists()  # the plain tap file is not touched


def test_confirm_does_not_touch_the_plain_tap_picker(env, fake_rank):
    tap(env)
    plain = tfile(env).read_bytes()
    ranked(env, fake_rank)
    call(env.app, "POST", url(env, "pick/confirm"), {"cls": "person", "i": 0})
    assert tfile(env).read_bytes() == plain
    assert len(call(env.app, "GET", url(env, "pick")).json()["taps"]) == 1


@pytest.mark.parametrize("i", [-1, 2, 99, "0", None, True, 1.5])
def test_confirm_bad_index_is_400(env, fake_rank, i):
    ranked(env, fake_rank)
    r = call(env.app, "POST", url(env, "pick/confirm"), {"cls": "person", "i": i})
    assert r.status == 400 and "error" in r.json()
    assert not tfile(env, "person", "photos:p1").exists()


def test_confirm_for_a_class_that_was_not_ranked_is_404(env, fake_rank):
    ranked(env, fake_rank)
    assert call(env.app, "POST", url(env, "pick/confirm"), {"cls": "dog", "i": 0}).status == 404


def test_the_last_ranked_picker_answers_cands_and_confirm(env, fake_rank):
    make_saved(env.tmp / "store")
    ranked(env, fake_rank)
    time.sleep(0.01)
    assert wait_job(env, call(env.app, "POST", url(env, "rank"), {"saved": "alice"}).json()["job"]["id"])["state"] == "done"
    st = call(env.app, "POST", url(env, "pick/confirm"), {"cls": "person", "i": 0}).json()
    assert st["target_key"] == "saved:alice"


# ---------------------------------------------------------------- routes 32-33: people
def fake_scan(doc_people=1, scan_id="abcd1234"):
    def scan(video, out, progress=None, **kw):
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in range(1, doc_people + 1):
            Image.new("RGB", (40, 60), (30 * n, 50, 90)).save(out / f"person_{n:03d}_{scan_id}.jpg")
            rows.append(dict(person=n, track_id=n, first_s=0.0, last_s=2.0, seen_s=2.0, stretches=[[0.0, 2.0]],
                             thumbnail=str(out / f"person_{n:03d}_{scan_id}.jpg"), target=f"tap:{out / f'person_{n:03d}_{scan_id}.json'}"))
        progress(0.5, "halfway")
        doc = dict(scan_id=scan_id, video=video, n_people=len(rows), people=rows, warnings=[])
        (out / "people.json").write_text(json.dumps(doc))
        return doc

    return scan


def test_people_job_lists_people_with_crop_urls_and_serves_the_crops(env, monkeypatch):
    seen = {}
    scan = fake_scan(2)

    def spy(video, out, progress=None, **kw):
        seen.update(video=video, out=out)
        return scan(video, out, progress)

    monkeypatch.setattr(people, "scan_people", spy)
    r = call(env.app, "POST", url(env, "people"), {})
    assert r.status == 200
    job = r.json()["job"]
    assert job["kind"] == "people" and job["media_id"] == env.mid
    done = wait_job(env, job["id"])
    assert done["state"] == "done"
    assert seen["video"] == str(env.media.path) and seen["out"] == out_root(env) / "people"
    assert [p["crop_url"] for p in done["result"]["people"]] == [f"api/media/{env.mid}/people/1.jpg", f"api/media/{env.mid}/people/2.jpg"]
    assert done["result"]["people"][0]["target"].startswith("tap:")
    img = call(env.app, "GET", url(env, "people/2.jpg"), key="media")
    assert img.status == 200 and img.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(img.body)).size == (40, 60)
    assert call(env.app, "GET", url(env, "people/3.jpg"), key="media").status == 404
    assert call(env.app, "GET", url(env, "people/1.jpg"), key=None).status == 403


def test_people_crop_before_any_scan_is_404(env):
    assert call(env.app, "GET", url(env, "people/1.jpg"), key="media").status == 404


def test_people_crop_names_come_from_the_scan_id_only(env, monkeypatch):
    monkeypatch.setattr(people, "scan_people", fake_scan(1))
    wait_job(env, call(env.app, "POST", url(env, "people"), {}).json()["job"]["id"])
    d = out_root(env) / "people"
    secret = env.tmp / "secret.jpg"
    Image.new("RGB", (8, 8)).save(secret)
    doc = json.loads((d / "people.json").read_text())
    for bad in ("../../secret", "ABCD1234", "abcd123", "abcd1234/../x", 5, None):
        (d / "people.json").write_text(json.dumps(dict(doc, scan_id=bad)))
        assert call(env.app, "GET", url(env, "people/1.jpg"), key="media").status == 404
    # the thumbnail path written in people.json is never followed
    (d / "people.json").write_text(json.dumps(dict(doc, people=[dict(doc["people"][0], thumbnail=str(secret))])))
    r = call(env.app, "GET", url(env, "people/1.jpg"), key="media")
    assert r.status == 200 and Image.open(io.BytesIO(r.body)).size == (40, 60)
    # a symlinked crop is not served
    f = d / "person_001_abcd1234.jpg"
    f.unlink()
    f.symlink_to(secret)
    assert call(env.app, "GET", url(env, "people/1.jpg"), key="media").status == 404
    # a symlinked people.json is not read
    (d / "people.json").write_text(json.dumps(doc))
    f.unlink()
    Image.new("RGB", (8, 8)).save(f)
    real = d / "real.json"
    (d / "people.json").rename(real)
    (d / "people.json").symlink_to(real)
    assert call(env.app, "GET", url(env, "people/1.jpg"), key="media").status == 404


def test_a_failed_people_scan_is_a_failed_job(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("the tracking extra is missing")

    monkeypatch.setattr(people, "scan_people", boom)
    done = wait_job(env, call(env.app, "POST", url(env, "people"), {}).json()["job"]["id"])
    assert done["state"] == "failed" and "tracking extra" in done["error"]


def test_the_module_exports_routes_and_attach(env):
    assert len(serve_pick.ROUTES) == 15 and callable(serve_pick.attach)
    assert {r.pattern.pattern for r in serve_pick.ROUTES if r.media} >= {r"/api/media/(?P<mid>[A-Za-z0-9_-]{1,32})/pickframe\.jpg"}
    assert all(r.method == "GET" for r in serve_pick.ROUTES if r.media)
