"""inframe serve makes no outbound connection (docs/T9A_DESIGN.md section 4, "Outbound network"): `socket.socket.connect`,
`connect_ex`, `socket.create_connection` and `socket.getaddrinfo` are patched to fail on any address that is not loopback, while a
video is opened, a motion run is made, edits are saved and an export (clips, reel, edit lists, no music) is written."""
import ipaddress
import json
import shutil
import socket
import time

import pytest
from serve_helpers import call, start_server

LOOPBACK_NAMES = ("localhost", "")


def is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host is None or host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture
def no_outbound(monkeypatch):
    """Record every non-loopback socket use and make it fail. Unix sockets (a str or bytes address) are local and pass."""
    attempts: list[tuple[str, object]] = []

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_create, real_getaddrinfo = socket.create_connection, socket.getaddrinfo

    def addr_host(address):
        return address[0] if isinstance(address, tuple) and address else None

    def connect(self, address):
        if isinstance(address, tuple) and not is_loopback(addr_host(address)):
            attempts.append(("connect", address))
            raise OSError(f"test: outbound connect to {address!r} refused")
        return real_connect(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and not is_loopback(addr_host(address)):
            attempts.append(("connect_ex", address))
            return 111
        return real_connect_ex(self, address)

    def create_connection(address, *a, **kw):
        if not is_loopback(addr_host(address)):
            attempts.append(("create_connection", address))
            raise OSError(f"test: outbound connection to {address!r} refused")
        return real_create(address, *a, **kw)

    def getaddrinfo(host, *a, **kw):
        if not is_loopback(host):
            attempts.append(("getaddrinfo", host))
            raise socket.gaierror(f"test: lookup of {host!r} refused")
        return real_getaddrinfo(host, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return attempts


def test_the_guard_itself_refuses_a_remote_address(no_outbound):
    with pytest.raises(OSError):
        socket.create_connection(("93.184.216.34", 80), timeout=1)
    with pytest.raises(OSError):
        socket.socket().connect(("10.0.0.5", 80))
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.invalid", 80)
    assert [k for k, _ in no_outbound] == ["create_connection", "connect", "getaddrinfo"]
    no_outbound.clear()


def wait_job(app, jid, timeout=180.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = app.jobs.get(jid)
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job still running")


def test_open_run_edit_and_export_stay_on_the_loopback(tmp_path, static_box, no_outbound):
    vids = tmp_path / "vids"
    vids.mkdir()
    video = vids / "clip.mp4"
    shutil.copy(static_box, video)
    base, app, stop = start_server(tmp_path)
    try:
        # open (the route, so the path rules, the probe and the recent list run too)
        r = call(app, "POST", "/api/media/open", {"path": str(video)})
        assert r.status == 200, r.text
        mid = r.json()["id"]
        assert call(app, "GET", "/api/session").status == 200
        # a motion run (no model, no download)
        r = call(app, "POST", "/api/runs", {"media_id": mid, "target": {"kind": "motion"}, "advanced": {"fps": 5.0, "pad": 0.5, "min_len": 1.0}})
        assert r.status == 200, r.text
        rid = r.json()["run_id"]
        assert wait_job(app, r.json()["job"]["id"])["state"] == "done"
        # edits: split the one clip
        run = call(app, "GET", f"/api/runs/{rid}").json()
        c = run["segments"]["clips"][0]
        mid_t = round((c["start"] + c["end"]) / 2, 3)
        clips = [{"id": "d1", "start": c["start"], "end": mid_t, "origin": "edited"}, {"id": "u1", "start": mid_t, "end": c["end"], "origin": "split"}]
        assert call(app, "PUT", f"/api/runs/{rid}/edits", {"revision": 0, "clips": clips}).status == 200
        assert call(app, "GET", f"/api/runs/{rid}/samples").status == 200
        assert call(app, "POST", f"/api/runs/{rid}/resegment", {}).status == 200
        # export: clips + reel + edit lists
        r = call(app, "POST", f"/api/runs/{rid}/export", {"clips": True, "reel": True, "edit_lists": ["fcpxml", "otio"]})
        assert r.status == 200, r.text
        j = wait_job(app, r.json()["job"]["id"])
        assert j["state"] == "done", j
        assert j["result"]["reel"]["ok"] is True and len(j["result"]["clips"]) == 2
        name = j["result"]["clips"][0]["file"]
        assert call(app, "GET", f"/api/runs/{rid}/files/{name}", headers={"Range": "bytes=0-9"}, key="media").status == 206
        assert call(app, "GET", f"/api/runs/{rid}/export").status == 200
        assert json.loads((vids / "clip_inframe" / "motion-ui" / "edited" / "export.json").read_text())["n_clips"] == 2
    finally:
        stop()
    assert no_outbound == [], f"outbound socket use: {no_outbound}"
