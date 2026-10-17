"""inframe serve jobs (docs/T9A_DESIGN.md 2.6, routes 46-48): one at a time, progress, cancel, finished jobs stay readable."""
import threading
import time

import pytest
from serve_helpers import call, start_server

from inframe.serve_http import HttpError
from inframe.serve_jobs import Busy, JobCancelled, JobRunner, NeedsConfirm


def wait_state(get, jid, states, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = get(jid)
        if j["state"] in states:
            return j
        time.sleep(0.02)
    raise AssertionError(f"job {jid} still {get(jid)['state']}")


def slow_job(release: threading.Event, steps=1000):
    def fn(progress):
        for i in range(steps):
            progress(i / steps, f"step {i}")
            if release.wait(0.02):
                return {"steps": i}
        return {"steps": steps}
    return fn


@pytest.fixture
def runner():
    r = JobRunner()
    yield r
    r.shutdown()


def test_job_runs_to_done_with_result(runner):
    def fn(progress):
        progress(0.25, "a quarter")
        progress(0.5, "half")
        return {"clips": 3}

    j = runner.start("run", fn, media_id="m1", run_id="r1")
    assert j["state"] == "running" and j["kind"] == "run" and j["media_id"] == "m1" and j["run_id"] == "r1"
    done = wait_state(runner.get, j["id"], {"done"})
    assert done["result"] == {"clips": 3} and done["fraction"] == 1.0 and done["error"] is None
    assert set(done) == {"id", "kind", "state", "fraction", "message", "started", "elapsed_s", "media_id", "run_id", "result", "error"}
    assert runner.current() is None  # nothing is running any more
    assert runner.get(j["id"])["result"] == {"clips": 3}  # a finished job stays readable


def test_second_start_is_busy_and_names_the_running_job(runner):
    release = threading.Event()
    j = runner.start("export", slow_job(release))
    with pytest.raises(Busy) as e:
        runner.start("run", lambda p: None)
    assert e.value.code == 409 and e.value.message == "busy" and e.value.extra["job"]["id"] == j["id"]
    assert runner.current()["id"] == j["id"]
    release.set()
    wait_state(runner.get, j["id"], {"done"})
    runner.start("run", lambda p: "ok")  # free again


def test_progress_is_stored_and_cancel_ends_cancelled(runner):
    release = threading.Event()
    j = runner.start("run", slow_job(release))
    time.sleep(0.2)
    cur = runner.get(j["id"])
    assert cur["state"] == "running" and 0 < cur["fraction"] < 1 and cur["message"].startswith("step ") and cur["elapsed_s"] > 0
    t0 = time.monotonic()
    after = runner.cancel(j["id"])
    assert after["state"] == "running"  # still running until the next progress report
    done = wait_state(runner.get, j["id"], {"cancelled"}, timeout=2)
    assert time.monotonic() - t0 < 2 and done["error"] is None and done["result"] is None


def test_cancel_of_a_finished_or_unknown_job(runner):
    j = runner.start("rank", lambda p: 1)
    wait_state(runner.get, j["id"], {"done"})
    assert runner.cancel(j["id"])["state"] == "done"
    with pytest.raises(HttpError) as e:
        runner.cancel("nope")
    assert e.value.code == 404
    with pytest.raises(HttpError):
        runner.get("nope")


def test_failure_and_needs_confirm(runner):
    def boom(progress):
        raise FileNotFoundError("RF-DETR models not found. Run `inframe fetch-models`")

    j = runner.start("run", boom)
    f = wait_state(runner.get, j["id"], {"failed"})
    assert f["error"].startswith("RF-DETR models not found") and f["result"] is None

    def ambiguous(progress):
        raise NeedsConfirm("Cannot tell which person is you", {"candidates": [1, 2]})

    j = runner.start("run", ambiguous)
    n = wait_state(runner.get, j["id"], {"needs_confirm"})
    assert n["error"] == "Cannot tell which person is you" and n["result"] == {"candidates": [1, 2]}

    def silent(progress):
        raise RuntimeError()

    j = runner.start("run", silent)
    assert wait_state(runner.get, j["id"], {"failed"})["error"] == "RuntimeError"


def test_unknown_kind_is_refused(runner):
    with pytest.raises(ValueError):
        runner.start("dance", lambda p: None)


def test_finished_jobs_are_forgotten_after_ten_minutes():
    now = [1000.0]
    r = JobRunner(clock=lambda: now[0])
    j = r.start("run", lambda p: 1)
    wait_state(r.get, j["id"], {"done"})
    now[0] += 599
    assert r.get(j["id"])["state"] == "done"
    now[0] += 2
    with pytest.raises(HttpError):
        r.get(j["id"])


def test_progress_callback_raises_job_cancelled_and_clamps(runner):
    seen = []

    def fn(progress):
        progress(7, "over")
        seen.append(runner.current()["fraction"])
        progress(-3, "under")
        seen.append(runner.current()["fraction"])
        return 1

    j = runner.start("run", fn)
    wait_state(runner.get, j["id"], {"done"})
    assert seen == [1.0, 0.0]
    assert issubclass(JobCancelled, Exception)


def test_shutdown_cancels_the_running_job():
    r = JobRunner()
    j = r.start("run", slow_job(threading.Event()))
    r.shutdown()
    assert r.get(j["id"])["state"] == "cancelled"


# ---------------------------------------------------------------- the routes 46-48
@pytest.fixture
def srv(tmp_path):
    base, app, stop = start_server(tmp_path)
    yield app
    stop()


def test_job_routes(srv):
    assert call(srv, "GET", "/api/jobs/current").json() == {"job": None}
    release = threading.Event()
    j = srv.jobs.start("run", slow_job(release))
    cur = call(srv, "GET", "/api/jobs/current").json()["job"]
    assert cur["id"] == j["id"] and cur["state"] == "running"
    assert call(srv, "GET", f"/api/jobs/{j['id']}").json()["kind"] == "run"
    assert call(srv, "GET", "/api/session").json()["job"]["id"] == j["id"]
    with pytest.raises(Busy):
        srv.jobs.start("export", lambda p: None)
    t0 = time.monotonic()
    r = call(srv, "POST", f"/api/jobs/{j['id']}/cancel", {})
    assert r.status == 200 and r.json()["id"] == j["id"]
    done = wait_state(lambda i: call(srv, "GET", f"/api/jobs/{i}").json(), j["id"], {"cancelled"}, timeout=2)
    assert time.monotonic() - t0 < 2 and done["state"] == "cancelled"
    assert call(srv, "GET", "/api/jobs/current").json() == {"job": None}
    assert call(srv, "GET", f"/api/jobs/{j['id']}").json()["state"] == "cancelled"  # a finished job is readable


def test_job_routes_errors_and_secrets(srv):
    assert call(srv, "GET", "/api/jobs/doesnotexist").status == 404
    assert call(srv, "POST", "/api/jobs/doesnotexist/cancel", {}).status == 404
    assert call(srv, "GET", "/api/jobs/current", key=None).status == 403
    assert call(srv, "POST", "/api/jobs/x/cancel", {}, key=None).status == 403
    assert call(srv, "POST", "/api/jobs/x/cancel", {}, headers={"Origin": None}).status == 403
    assert call(srv, "GET", "/api/jobs/bad%20id").status == 404


def test_busy_error_body_shape():
    r = JobRunner()
    release = threading.Event()
    j = r.start("people", slow_job(release))
    try:
        r.start("run", lambda p: None)
    except Busy as e:
        assert (e.code, e.message, e.extra["job"]["id"], e.extra["job"]["kind"]) == (409, "busy", j["id"], "people")
    release.set()
    r.shutdown()
