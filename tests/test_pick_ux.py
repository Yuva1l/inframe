"""inframe pick: a slider drag must not queue one decode+detect per position (latest request wins), times are cached, the
GPU is used unless INFRAME_DETECT_UNITS says otherwise, a client that goes away is not an error, the browser is opened.
The detector is a fake that is slow on purpose; no real model, no real footage."""
import http.client
import json
import subprocess
import sys
import threading
import time

import pytest

from inframe import cli, pick
from inframe.detect import Detection


class SlowDet:
    def __init__(self, delay=0.4):
        self.delay, self.calls = delay, 0
        self.started = threading.Event()

    def detect(self, frames):
        self.calls += 1
        self.started.set()
        time.sleep(self.delay)
        return [[Detection(0.9, (0.1, 0.1, 0.5, 0.9)), Detection(0.6, (0.5, 0.1, 0.9, 0.9))] for _ in frames]


@pytest.fixture
def pk(synth, tmp_path):
    p = pick.Picker(synth["plain"], tmp_path / "t.json")
    p.det = SlowDet()
    yield p
    p.close()


def ask(p, t, out, name):
    def run():
        try:
            out[name] = p.frame(t)
        except pick.Superseded:
            out[name] = "superseded"
    th = threading.Thread(target=run)
    th.start()
    return th


def test_stale_requests_are_dropped_before_decode_and_detect(pk):
    out, ths = {}, []
    ths.append(ask(pk, 0.2, out, "a"))
    assert pk.det.started.wait(5)  # the first one is in the detector now
    for name, t in (("b", 0.6), ("c", 1.0), ("d", 1.6), ("e", 2.2)):  # the slider moves on while it runs
        ths.append(ask(pk, t, out, name))
        deadline = time.time() + 5
        while pk._latest != round(t, 2) and time.time() < deadline:
            time.sleep(0.01)
    for th in ths:
        th.join(10)
    assert out["b"] == out["c"] == out["d"] == "superseded"
    assert out["a"] != "superseded" and out["e"] != "superseded"
    assert pk.det.calls == 2  # the running one and the newest; never b, c, d
    assert pk.stats["dropped"] == 3
    assert pk.stats["decoded"] == 2


def test_a_request_replaced_while_decoding_skips_the_detector(pk, monkeypatch):
    gate, entered = threading.Event(), threading.Event()
    real = pick.grab_frame

    def slow_grab(*a, **k):
        entered.set()
        gate.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(pick, "grab_frame", slow_grab)
    out = {}
    first = ask(pk, 0.5, out, "a")
    assert entered.wait(5)
    second = ask(pk, 1.5, out, "b")
    while pk._latest != 1.5:
        time.sleep(0.01)
    gate.set()
    first.join(10), second.join(10)
    assert out["a"] == "superseded" and out["b"] != "superseded"
    assert pk.det.calls == 1


def test_the_same_time_is_computed_once(pk):
    pk.det.delay = 0.01
    a = pk.frame(1.0)
    b = pk.frame(1.0)
    assert a is b and pk.det.calls == 1 and pk.stats["hits"] == 1
    assert pk.frame(a[0]) is a  # asking by the time the frame really has (what the page does for /frame.jpg) is a hit too
    assert pk.det.calls == 1


def test_two_requests_for_one_time_while_it_runs_share_the_job(pk):
    out = {}
    ths = [ask(pk, 1.0, out, "a")]
    assert pk.det.started.wait(5)
    ths.append(ask(pk, 1.0, out, "b"))
    for th in ths:
        th.join(10)
    assert out["a"] is out["b"] and pk.det.calls == 1


def test_cache_is_bounded(pk, monkeypatch):
    monkeypatch.setattr(pick, "CACHE_MAX", 3)
    pk.det.delay = 0
    for t in (0.2, 0.6, 1.0, 1.4, 1.8):
        pk.frame(t)
    assert len(pk.cache) <= 3


def test_pick_uses_the_gpu_unless_the_env_says_otherwise(monkeypatch):
    monkeypatch.delenv("INFRAME_DETECT_UNITS", raising=False)
    assert pick.pick_units() == "CPU_AND_GPU"
    monkeypatch.setenv("INFRAME_DETECT_UNITS", "CPU_ONLY")
    assert pick.pick_units() == "CPU_ONLY"
    monkeypatch.setenv("INFRAME_DETECT_UNITS", "CPU_AND_NE")
    assert pick.pick_units() == "CPU_AND_NE"


def test_detector_gets_the_pick_units_and_the_pipeline_default_is_the_same(synth, tmp_path, monkeypatch):
    from inframe import detect

    monkeypatch.delenv("INFRAME_DETECT_UNITS", raising=False)
    seen = []
    monkeypatch.setattr(pick, "load_detector", lambda size, units=None: seen.append((size, units)) or SlowDet(0))
    p = pick.Picker(synth["plain"], tmp_path / "t.json")
    p.frame(1.0)
    p.close()
    assert seen == [("small", "CPU_AND_GPU")]
    assert detect.compute_units_name() == "CPU_AND_GPU"  # the pipeline default is the same units now


def serve_in_thread(p):
    from http.server import ThreadingHTTPServer

    stop, ref = threading.Event(), []
    httpd = pick.QuietServer(("127.0.0.1", 0), pick.make_handler(p, stop, ref))
    ref.append(httpd)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def test_http_superseded_is_409_and_a_cancelled_client_prints_nothing(pk, capfd):
    httpd, port = serve_in_thread(pk)
    try:
        def get(path):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            c.request("GET", f"{path}&k={pk.token}", headers={"Host": f"127.0.0.1:{port}"})
            r = c.getresponse()
            return r.status, json.loads(r.read())

        res = {}
        th = threading.Thread(target=lambda: res.update(first=get("/boxes?t=0.2")))
        th.start()
        assert pk.det.started.wait(5)
        t2 = threading.Thread(target=lambda: res.update(stale=get("/boxes?t=1.0")))
        t2.start()
        while pk._latest != 1.0:
            time.sleep(0.01)
        t3 = threading.Thread(target=lambda: res.update(last=get("/boxes?t=2.0")))
        t3.start()
        for x in (th, t2, t3):
            x.join(20)
        assert res["stale"][0] == 409 and res["stale"][1]["error"] == "superseded"
        assert res["first"][0] == 200 and res["last"][0] == 200 and res["last"][1]["boxes"]
        assert pk.det.calls == 2

        # a browser that cancels (closes the socket, with a reset) while the server is still busy
        import socket
        import struct

        pk.det.delay = 0.3
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(f"GET /boxes?t=2.7&k={pk.token} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        time.sleep(0.1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # close with RST
        s.close()
        time.sleep(0.8)
    finally:
        httpd.shutdown()
        httpd.server_close()
    err = capfd.readouterr().err
    assert "Traceback" not in err and "BrokenPipe" not in err and "ConnectionReset" not in err


def test_quiet_server_swallows_only_client_cancels(capsys):
    srv = pick.QuietServer.__new__(pick.QuietServer)
    for exc in (BrokenPipeError(), ConnectionResetError()):
        try:
            raise exc
        except OSError:
            srv.handle_error(None, ("127.0.0.1", 1))
    assert capsys.readouterr().err == ""
    try:
        raise RuntimeError("real bug")
    except RuntimeError:
        srv.handle_error(None, ("127.0.0.1", 1))
    assert "RuntimeError" in capsys.readouterr().err


def run_serve(synth, tmp_path, monkeypatch, open_browser):
    opened, said, loads = [], [], []
    monkeypatch.setattr(pick, "load_detector", lambda size, units=None: loads.append(units) or SlowDet(0))
    monkeypatch.setattr(pick.webbrowser, "open", lambda u: opened.append(u) or True)
    res = {}
    th = threading.Thread(target=lambda: res.update(code=pick.serve(synth["plain"], tmp_path / "t.json", announce=said.append, open_browser=open_browser)))
    th.start()
    deadline = time.time() + 10
    while not any("waiting for your tap" in m for m in said) and time.time() < deadline:
        time.sleep(0.02)
    url = next(m.split()[3] for m in said if m.startswith("inframe pick: open "))
    host = url.split("/")[2]
    c = http.client.HTTPConnection(host, timeout=10)
    c.request("POST", "/done?" + url.split("?")[1], headers={"Host": host, "Origin": f"http://{host}", "Content-Length": "0"})
    c.getresponse().read()
    th.join(10)
    res["loads"] = loads
    return opened, said, url, res


def test_serve_opens_the_browser_and_says_it_waits(synth, tmp_path, monkeypatch):
    opened, said, url, res = run_serve(synth, tmp_path, monkeypatch, True)
    assert opened == [url]
    assert any("waiting for your tap..." in m and "Ctrl-C" in m for m in said)
    assert res["code"] == 2  # no tap was made
    deadline = time.time() + 5
    while not res["loads"] and time.time() < deadline:
        time.sleep(0.02)
    assert res["loads"] == [pick.pick_units()]  # the detector is loaded at start, before the first slider move


def test_no_open_leaves_the_browser_alone(synth, tmp_path, monkeypatch):
    opened, said, url, res = run_serve(synth, tmp_path, monkeypatch, False)
    assert opened == [] and url.startswith("http://127.0.0.1:")


def test_cli_has_no_open_flag():
    a = cli.build_parser().parse_args(["pick", "v.mp4", "--no-open"])
    assert a.no_open is True
    assert cli.build_parser().parse_args(["pick", "v.mp4"]).no_open is False


def test_page_debounces_the_slider_and_ignores_old_answers():
    assert "setTimeout(()=>load(t),300)" in pick.PAGE and "oninput=e=>sched(" in pick.PAGE
    assert "AbortController" in pick.PAGE and "my!==seq" in pick.PAGE and "loading" in pick.PAGE


def test_importing_the_pick_page_silences_the_coremltools_notes():
    pytest.importorskip("coremltools")
    r = subprocess.run([sys.executable, "-c", "import inframe.pick, coremltools"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stderr.strip() == "", r.stderr


def fake_grab(frames):
    """A grab_frame over a 29.97 fps clip whose frame times are `frames` (first at or after t, as the real one)."""
    import numpy as np

    calls = []

    def grab(path, info, t, long_side=0):
        calls.append(t)
        ft = next(f for f in frames if f + 1e-4 >= t)
        return ft, np.full((4, 4, 3), int(ft * 10) % 255, dtype=np.uint8)

    return grab, calls


def test_frame_jpg_after_boxes_is_a_cache_hit_even_when_the_rounding_differs(pk, monkeypatch):
    frames = [2.581, 2.6146, 2.648]  # 29.97 fps: round(round(2.6146, 3), 2) = 2.62 but round(2.6146, 2) = 2.61
    assert round(round(2.6146, 3), 2) != round(2.6146, 2)
    grab, calls = fake_grab(frames)
    monkeypatch.setattr(pick, "grab_frame", grab)
    pk.det.delay = 0
    httpd, port = serve_in_thread(pk)
    try:
        def get(path):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            c.request("GET", f"{path}&k={pk.token}", headers={"Host": f"127.0.0.1:{port}"})
            r = c.getresponse()
            return r.status, r.read()

        status, body = get("/boxes?t=2.61")
        t = json.loads(body)["t"]
        assert status == 200 and t == 2.615
        n, d = len(calls), pk.det.calls
        status, body = get(f"/frame.jpg?t={t}")  # what the page asks
        assert status == 200 and body[:2] == b"\xff\xd8"
        assert len(calls) == n and pk.det.calls == d  # no second decode, no second detect, not the next frame (2.648)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_frame_jpg_miss_decodes_the_same_frame_and_never_runs_the_detector(pk, monkeypatch):
    frames = [4.571, 4.6046, 4.638]
    grab, calls = fake_grab(frames)
    monkeypatch.setattr(pick, "grab_frame", grab)
    rgb = pk.frame_image(4.605)  # evicted or never cached
    assert rgb[0, 0, 0] == int(4.6046 * 10) % 255 and pk.det.calls == 0


def test_frame_jpg_is_not_superseded_while_the_worker_is_busy(pk, monkeypatch):
    grab, calls = fake_grab([1.6, 2.4, 2.6146, 2.648])
    monkeypatch.setattr(pick, "grab_frame", grab)
    httpd, port = serve_in_thread(pk)
    try:
        pk.det.delay = 0.01
        ft, _, _ = pk.frame(2.61)  # the page was told t = 2.615, and then asks /frame.jpg?t=2.615
        pk.det.delay = 0.6
        pk.det.started.clear()
        out, res = {}, {}
        a = ask(pk, 1.6, out, "a")
        assert pk.det.started.wait(5)

        def get_jpg():
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            c.request("GET", f"/frame.jpg?t={round(ft, 3)}&k={pk.token}", headers={"Host": f"127.0.0.1:{port}"})
            r = c.getresponse()
            res["jpg"] = (r.status, r.read())

        g = threading.Thread(target=get_jpg)
        g.start()
        time.sleep(0.2)  # it is a hit now, or (before the fix) a queued job that the next request replaces
        b = ask(pk, 2.4, out, "b")
        for x in (g, a, b):
            x.join(10)
        assert res["jpg"][0] == 200 and res["jpg"][1][:2] == b"\xff\xd8"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_cache_hit_drops_the_pending_jobs_the_page_left(pk):
    pk.det.delay = 0.05
    pk.frame(1.0)
    pk.det.delay = 0.5
    pk.det.started.clear()
    out = {}
    a = ask(pk, 1.6, out, "a")
    assert pk.det.started.wait(5)
    b = ask(pk, 2.4, out, "b")
    while pk._latest != 2.4:
        time.sleep(0.01)
    pk.frame(1.0)  # back to a cached time: a hit
    assert pk._latest == 1.0
    a.join(10), b.join(10)
    assert out["b"] == "superseded"
    assert pk.det.calls == 2  # 1.0 and 1.6 (already running); never 2.4
    assert pk.stats["dropped"] >= 1 and pk.stats["decoded"] <= 2
