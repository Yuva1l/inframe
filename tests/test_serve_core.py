"""inframe serve, HTTP core (docs/T9A_DESIGN.md sections 2.3 and 4): secrets, Host / Origin / fetch metadata, headers, request
bodies, Range, static files and the fallback page, open_media, the session, quit and the route-module loader.
Synthetic clips only; no model, no network beyond 127.0.0.1."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest
from serve_helpers import call, start_server

from inframe import serve, serve_http
from inframe.serve_http import HttpError, Route, file_response, json_response


# ---------------------------------------------------------------- a server with a few test routes
FILES: dict[str, str] = {}
APPS: list = []


def echo(app, req):
    return json_response({"json": req.json, "query": req.query, "params": req.params, "n": len(req.body)})


def fileh(app, req):
    return file_response(FILES[req.params["name"]], "video/mp4", req)


def open_h(app, req):
    m = app.open_media(req.json["path"])
    return json_response({"id": m.id, "picked_copy": m.picked_copy})


TEST_ROUTES = [
    Route("POST", r"/api/test/echo", echo),
    Route("PUT", r"/api/test/echo", echo),
    Route("DELETE", r"/api/test/echo", echo),
    Route("GET", r"/api/test/echo/(?P<x>\w+)", echo),
    Route("GET", r"/api/test/file/(?P<name>\w+)", fileh, media=True),
    Route("POST", r"/api/test/open", open_h),
]


@pytest.fixture
def srv(tmp_path):
    base, app, stop = start_server(tmp_path, extra_routes=TEST_ROUTES)
    yield app
    stop()


def raw(app, data: bytes, wait: float = 5.0) -> bytes:
    s = socket.create_connection(("127.0.0.1", app.port), timeout=wait)
    try:
        s.sendall(data)
        out = b""
        while True:
            try:
                chunk = s.recv(65536)
            except (TimeoutError, ConnectionResetError):
                break
            if not chunk:
                break
            out += chunk
        return out
    finally:
        s.close()


# ---------------------------------------------------------------- secrets
def test_token_missing_or_wrong_is_403(srv):
    assert call(srv, "GET", "/api/session", key=None).status == 403
    assert call(srv, "GET", "/api/session", headers={"X-Inframe-Token": "nope"}).status == 403
    assert call(srv, "GET", "/api/session", headers={"X-Inframe-Token": ""}).status == 403
    assert call(srv, "GET", "/api/session").status == 200
    r = call(srv, "GET", "/api/session", key=None)
    assert "error" in r.json()


def test_old_k_parameter_is_not_accepted(srv):
    assert call(srv, "GET", f"/api/session?k={srv.token}", key=None).status == 403
    assert call(srv, "GET", f"/api/test/file/plain?k={srv.token}", key=None).status == 403


def test_media_key_only_on_media_get_routes(srv, synth):
    FILES["plain"] = synth["plain"]
    assert call(srv, "GET", "/api/test/file/plain", key="media").status == 200
    assert call(srv, "HEAD", "/api/test/file/plain", key="media").status == 200
    assert call(srv, "GET", "/api/session", key="media").status == 403
    assert call(srv, "GET", "/api/jobs/current", key="media").status == 403
    assert call(srv, "POST", "/api/test/echo", {}, key="media").status == 403
    assert call(srv, "GET", "/api/test/file/plain?m=wrong", key=None).status == 403
    assert call(srv, "GET", "/api/test/file/plain", key=None).status == 403
    assert call(srv, "GET", "/api/test/file/plain").status == 200  # the session token works there too


def test_unknown_route_and_method(srv):
    assert call(srv, "GET", "/api/nothing").status == 404
    assert call(srv, "GET", "/api/nothing", key=None).status == 403  # a stranger learns nothing about which routes exist
    assert call(srv, "GET", "/api/quit").status == 405
    assert call(srv, "GET", "/nothing-here").status == 404


def test_nonce_claim_once(srv):
    n = srv.new_nonce()
    r = call(srv, "POST", "/api/session/claim", {"nonce": n}, key=None)
    assert r.status == 200 and r.json() == {"token": srv.token, "media_key": srv.media_key}
    assert call(srv, "POST", "/api/session/claim", {"nonce": n}, key=None).status == 403
    assert call(srv, "POST", "/api/session/claim", {"nonce": "unknown"}, key=None).status == 403
    assert call(srv, "POST", "/api/session/claim", {"nonce": 5}, key=None).status == 403
    assert call(srv, "POST", "/api/session/claim", {}, key=None).status == 403
    # a claim is still a POST: it needs the page's Origin
    n2 = srv.new_nonce()
    assert call(srv, "POST", "/api/session/claim", {"nonce": n2}, key=None, headers={"Origin": None}).status == 403
    assert call(srv, "POST", "/api/session/claim", {"nonce": n2}, key=None).status == 200


def test_nonce_expires_after_30_seconds(srv, monkeypatch):
    t0 = time.monotonic()
    n = srv.new_nonce()
    monkeypatch.setattr(serve, "_clock", lambda: t0 + 31)
    assert call(srv, "POST", "/api/session/claim", {"nonce": n}, key=None).status == 403
    monkeypatch.setattr(serve, "_clock", lambda: t0 + 5)
    n2 = srv.new_nonce()
    monkeypatch.setattr(serve, "_clock", lambda: t0 + 29)
    assert call(srv, "POST", "/api/session/claim", {"nonce": n2}, key=None).status == 200


def test_token_and_nonce_are_128_bit_and_different(srv):
    assert len(srv.token) >= 22 and len(srv.media_key) >= 22 and srv.token != srv.media_key
    assert srv.new_nonce() != srv.new_nonce()


# ---------------------------------------------------------------- Host, Origin, fetch metadata, verbs
@pytest.mark.parametrize("host", ["localhost:{port}", "evil.example:{port}", "127.0.0.1", "127.0.0.1:1", "127.0.0.1.evil.example:{port}", ""])
def test_wrong_host_is_403(srv, host):
    h = {"Host": host.format(port=srv.port)}
    assert call(srv, "GET", "/api/session", headers=h).status == 403
    assert call(srv, "GET", "/", headers=h, key=None).status == 403  # static files too


def test_missing_host_is_403(srv):
    assert call(srv, "GET", "/api/session", headers={"Host": None}).status == 403


def test_post_origin_rules(srv):
    assert call(srv, "POST", "/api/test/echo", {"a": 1}, headers={"Origin": None}).status == 403
    assert call(srv, "POST", "/api/test/echo", {"a": 1}, headers={"Origin": "http://evil.example"}).status == 403
    assert call(srv, "POST", "/api/test/echo", {"a": 1}, headers={"Origin": f"http://localhost:{srv.port}"}).status == 403
    assert call(srv, "POST", "/api/test/echo", {"a": 1}, headers={"Origin": f"https://127.0.0.1:{srv.port}"}).status == 403
    assert call(srv, "PUT", "/api/test/echo", {"a": 1}, headers={"Origin": None}).status == 403
    assert call(srv, "DELETE", "/api/test/echo", headers={"Origin": None}).status == 403
    assert call(srv, "POST", "/api/test/echo", {"a": 1}).status == 200
    # a GET may omit Origin, but one that is present must be ours
    assert call(srv, "GET", "/api/session", headers={"Origin": "http://evil.example"}).status == 403
    assert call(srv, "GET", "/api/session", headers={"Origin": f"http://127.0.0.1:{srv.port}"}).status == 200


def test_sec_fetch_site(srv):
    assert call(srv, "GET", "/api/session", headers={"Sec-Fetch-Site": "cross-site"}).status == 403
    assert call(srv, "GET", "/api/session", headers={"Sec-Fetch-Site": "same-site"}).status == 403
    assert call(srv, "GET", "/", headers={"Sec-Fetch-Site": "cross-site"}, key=None).status == 403
    assert call(srv, "GET", "/api/session", headers={"Sec-Fetch-Site": "same-origin"}).status == 200
    assert call(srv, "GET", "/api/session", headers={"Sec-Fetch-Site": "none"}).status == 200
    assert call(srv, "GET", "/api/session").status == 200  # absent: nothing is relaxed, nothing is demanded


def test_options_is_403_and_nothing_sends_cors_headers(srv):
    r = call(srv, "OPTIONS", "/api/session", headers={"Access-Control-Request-Method": "GET"})
    assert r.status == 403
    r = call(srv, "OPTIONS", "/", key=None)
    assert r.status == 403
    for resp in (r, call(srv, "GET", "/api/session"), call(srv, "GET", "/", key=None), call(srv, "GET", "/api/session", key=None)):
        assert not [k for k in resp.headers if k.startswith("access-control")]


def test_unknown_verb_is_refused_in_json(srv):
    r = call(srv, "TRACE", "/api/session")
    assert r.status == 501 and "error" in r.json()


def test_response_headers(srv):
    for r in (call(srv, "GET", "/api/session"), call(srv, "GET", "/api/session", key=None), call(srv, "GET", "/", key=None),
              call(srv, "GET", "/api/nothing")):
        assert r.headers["cross-origin-resource-policy"] == "same-origin"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["x-frame-options"] == "DENY"
    assert call(srv, "GET", "/api/session").headers["cache-control"] == "no-store"
    assert call(srv, "GET", "/api/session", key=None).headers["cache-control"] == "no-store"
    page = call(srv, "GET", "/", key=None)
    csp = page.headers["content-security-policy"]
    for part in ("default-src 'self'", "script-src 'self'", "connect-src 'self'", "frame-ancestors 'none'", "base-uri 'none'",
                 "form-action 'none'", "media-src 'self' blob:"):
        assert part in csp
    assert "content-security-policy" not in call(srv, "GET", "/api/session").headers  # JSON needs no CSP
    assert page.headers["content-length"] == str(len(page.body))


# ---------------------------------------------------------------- request bodies
@pytest.mark.parametrize("length", ["-1", "2147483648", "1e3", "", " ", "0x10", "+5", "12345678901"])
def test_bad_content_length_is_400(srv, length):
    r = call(srv, "POST", "/api/test/echo", b"{}", headers={"Content-Length": length})
    assert r.status == 400


def test_body_over_one_megabyte_is_413_without_reading(srv):
    # Content-Length says 1 MB + 1, one byte is sent: the answer comes at once
    t0 = time.monotonic()
    r = call(srv, "POST", "/api/test/echo", b"{", headers={"Content-Length": str(1024 * 1024 + 1)})
    assert r.status == 413 and time.monotonic() - t0 < 3


def test_body_of_exactly_one_megabyte_is_read(srv):
    pad = "x" * (1024 * 1024 - len('{"a":""}'))
    body = ('{"a":"%s"}' % pad).encode()
    assert len(body) == 1024 * 1024
    r = call(srv, "POST", "/api/test/echo", body)
    assert r.status == 200 and r.json()["n"] == len(body)


def test_chunked_body_is_400_and_missing_length_is_411(srv):
    assert call(srv, "POST", "/api/test/echo", b"2\r\n{}\r\n0\r\n\r\n",
                headers={"Transfer-Encoding": "chunked", "Content-Length": None}).status == 400
    assert call(srv, "POST", "/api/test/echo", None, headers={"Content-Length": None}).status == 411
    assert call(srv, "PUT", "/api/test/echo", None, headers={"Content-Length": None}).status == 411
    assert call(srv, "DELETE", "/api/test/echo", None, headers={"Content-Length": None}).status == 200


def test_json_body_rules(srv):
    r = call(srv, "POST", "/api/test/echo", {"a": [1, 2]})
    assert r.json()["json"] == {"a": [1, 2]}
    assert call(srv, "POST", "/api/test/echo", b"").json()["json"] == {}
    assert call(srv, "POST", "/api/test/echo", b"{not json").status == 400
    assert call(srv, "POST", "/api/test/echo", b"[1]").status == 400
    assert call(srv, "POST", "/api/test/echo", b"\xff\xfe").status == 400
    r = call(srv, "GET", "/api/test/echo/abc?x=1&y=&x=2")
    assert r.json()["query"] == {"x": "1", "y": ""} and r.json()["params"] == {"x": "abc"}


def test_a_body_that_stops_half_way_ends_with_408(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_http, "SOCKET_TIMEOUT", 1.0)
    base, app, stop = start_server(tmp_path, extra_routes=TEST_ROUTES)
    try:
        head = (f"POST /api/test/echo HTTP/1.1\r\nHost: 127.0.0.1:{app.port}\r\nOrigin: http://127.0.0.1:{app.port}\r\n"
                f"X-Inframe-Token: {app.token}\r\nContent-Length: 100\r\n\r\n").encode()
        t0 = time.monotonic()
        out = raw(app, head + b"{", wait=16)
        assert out.startswith(b"HTTP/1.1 408") and time.monotonic() - t0 < 16
    finally:
        stop()


def test_error_leaves_no_unread_body_for_the_next_request(srv):
    # a 403 on a request with a body closes the connection instead of reading the body as the next request
    head = (f"POST /api/test/echo HTTP/1.1\r\nHost: evil.example\r\nContent-Length: 40\r\n\r\n"
            "GET /api/session HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").encode()
    out = raw(srv, head)
    assert out.count(b"HTTP/1.1 ") == 1 and out.startswith(b"HTTP/1.1 403") and b"Connection: close" in out


def test_handler_thread_cap_answers_503(tmp_path):
    base, app, stop = start_server(tmp_path, max_threads=1, extra_routes=TEST_ROUTES)
    hold = socket.create_connection(("127.0.0.1", app.port))
    try:
        time.sleep(0.3)  # the held connection owns the only slot
        r = call(app, "GET", "/api/session")
        assert r.status == 503
    finally:
        hold.close()
    time.sleep(0.3)
    try:
        assert call(app, "GET", "/api/session").status == 200  # the slot came back
    finally:
        stop()


# ---------------------------------------------------------------- Range
def test_range_requests(srv, synth):
    FILES["plain"] = synth["plain"]
    data = Path(synth["plain"]).read_bytes()
    size = len(data)
    assert size > 1000
    full = call(srv, "GET", "/api/test/file/plain", key="media")
    assert full.status == 200 and full.body == data
    assert full.headers["accept-ranges"] == "bytes" and full.headers["content-type"] == "video/mp4"
    assert full.headers["content-length"] == str(size)

    r = call(srv, "GET", "/api/test/file/plain", headers={"Range": "bytes=0-99"}, key="media")
    assert r.status == 206 and len(r.body) == 100 and r.body == data[:100]
    assert r.headers["content-range"] == f"bytes 0-99/{size}" and r.headers["content-length"] == "100"

    r = call(srv, "GET", "/api/test/file/plain", headers={"Range": "bytes=-50"}, key="media")
    assert r.status == 206 and r.body == data[-50:]
    assert r.headers["content-range"] == f"bytes {size - 50}-{size - 1}/{size}"

    r = call(srv, "GET", "/api/test/file/plain", headers={"Range": "bytes=100-"}, key="media")
    assert r.status == 206 and r.body == data[100:] and r.headers["content-range"] == f"bytes 100-{size - 1}/{size}"

    r = call(srv, "GET", "/api/test/file/plain", headers={"Range": f"bytes={size - 10}-{size + 500}"}, key="media")
    assert r.status == 206 and r.body == data[-10:]  # an end past the file is cut at the file

    r = call(srv, "GET", "/api/test/file/plain", headers={"Range": f"bytes=-{size * 3}"}, key="media")
    assert r.status == 206 and r.body == data  # a suffix longer than the file is the whole file

    for bad in (f"bytes={size}-", f"bytes={size + 10}-{size + 20}", "bytes=-0"):
        r = call(srv, "GET", "/api/test/file/plain", headers={"Range": bad}, key="media")
        assert r.status == 416 and r.headers["content-range"] == f"bytes */{size}", bad

    # not a single satisfiable bytes range: the whole file
    for odd in ("bytes=0-9,20-29", "items=0-9", "bytes=9-3", "garbage", "bytes=-"):
        r = call(srv, "GET", "/api/test/file/plain", headers={"Range": odd}, key="media")
        assert r.status == 200 and r.body == data, odd


def test_head_has_headers_and_no_body(srv, synth):
    FILES["plain"] = synth["plain"]
    size = os.path.getsize(synth["plain"])
    r = call(srv, "HEAD", "/api/test/file/plain", key="media")
    assert r.status == 200 and r.body == b"" and r.headers["content-length"] == str(size)
    r = call(srv, "HEAD", "/api/test/file/plain", headers={"Range": "bytes=0-9"}, key="media")
    assert r.status == 206 and r.headers["content-length"] == "10" and r.headers["content-range"] == f"bytes 0-9/{size}"
    assert call(srv, "HEAD", "/api/test/file/plain", key=None).status == 403


def test_file_that_vanishes_is_404_json(srv, tmp_path):
    p = tmp_path / "gone.mp4"
    p.write_bytes(b"x" * 10)
    FILES["gone"] = str(p)
    p.unlink()
    r = call(srv, "GET", "/api/test/file/gone", key="media")
    assert r.status == 404 and "error" in r.json()


def test_client_that_leaves_mid_download_is_quiet(srv, tmp_path, capfd):
    p = tmp_path / "big.mp4"
    p.write_bytes(os.urandom(8 * 1024 * 1024))
    FILES["big"] = str(p)
    s = socket.create_connection(("127.0.0.1", srv.port))
    s.sendall(f"GET /api/test/file/big?m={srv.media_key} HTTP/1.1\r\nHost: 127.0.0.1:{srv.port}\r\n\r\n".encode())
    s.recv(1000)
    s.close()
    time.sleep(0.5)
    assert call(srv, "GET", "/api/session").status == 200
    assert "Traceback" not in capfd.readouterr().err


# ---------------------------------------------------------------- static files
def make_dist(root: Path) -> Path:
    d = root / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>ui</title><script src=\"./assets/app.js\"></script>")
    (d / "assets" / "app.js").write_text("console.log(1)")
    (d / "assets" / "app.css").write_text("body{}")
    (d / "assets" / "_private.js").write_text("secret")
    (d / "favicon.svg").write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"/>")
    (d / "_headers").write_text("/*\n  X: y")
    (d / ".hidden").write_text("secret")
    (root / "outside.txt").write_text("outside")
    os.symlink(root / "outside.txt", d / "assets" / "link.txt")
    return d


def test_static_files_from_dist(tmp_path):
    d = make_dist(tmp_path)
    base, app, stop = start_server(tmp_path, dist_dir=d)
    try:
        r = call(app, "GET", "/", key=None)
        assert r.status == 200 and "<title>ui</title>" in r.text and r.headers["content-type"].startswith("text/html")
        assert "content-security-policy" in r.headers
        assert call(app, "GET", "/index.html", key=None).status == 200
        js = call(app, "GET", "/assets/app.js", key=None)
        assert js.status == 200 and js.headers["content-type"].startswith("text/javascript") and js.text == "console.log(1)"
        assert call(app, "GET", "/assets/app.css", key=None).headers["content-type"].startswith("text/css")
        assert call(app, "GET", "/favicon.svg", key=None).headers["content-type"] == "image/svg+xml"
        assert call(app, "HEAD", "/assets/app.js", key=None).status == 200
        # names starting with _ or . are 404, so are traversal, symlinks and missing files
        for bad in ("/_headers", "/.hidden", "/assets/_private.js", "/assets/%2e%2e/outside.txt", "/assets/../outside.txt",
                    "/assets/link.txt", "/assets/", "/assets", "/nope.js", "//etc/passwd", "/assets/..%2f..%2foutside.txt", "/assets/a%00b"):
            assert call(app, "GET", bad, key=None).status == 404, bad
        assert call(app, "POST", "/", {}, key=None).status == 405
    finally:
        stop()


def test_fallback_page_when_dist_is_absent(srv):
    for path in ("/", "/index.html"):
        r = call(srv, "GET", path, key=None)
        assert r.status == 200 and "The web UI is not built" in r.text and "npm --prefix web ci" in r.text
        assert r.headers["content-type"].startswith("text/html") and "content-security-policy" in r.headers
    assert call(srv, "GET", "/assets/app.js", key=None).status == 404
    assert call(srv, "GET", "/favicon.svg", key=None).status == 404


def test_default_dist_folder_is_the_package_web_dist():
    assert serve._default_dist().parts[-3:] == ("inframe", "web", "dist")


# ---------------------------------------------------------------- session, quit
def test_session(srv, synth):
    s = call(srv, "GET", "/api/session").json()
    assert s["engine"] == "local" and s["version"] == "0.1.0" and s["platform"] == sys.platform
    assert s["presets"] == ["fast", "balanced", "accurate"] and s["default_preset"] == "balanced"
    assert "person" in s["classes"] and "dog" in s["classes"] and s["classes"] == sorted(s["classes"])
    assert s["defaults"]["preset"] == "balanced" and s["defaults"]["pad"] == 1.0 and s["defaults"]["min_len"] == 1.0
    assert s["job"] is None and s["media"] == [] and s["start_media"] is None
    caps = s["caps"]
    assert caps["output"] == "server-folder" and caps["fileBrowser"] is True and caps["streamCopyCut"] is True
    assert set(caps) >= {"openDialog", "photosPicker", "locate", "reveal", "music", "beatSync", "savedTargets", "people", "proxy",
                         "chooseOutputFolder"}
    pf = s["preflight"]
    assert set(pf) >= {"ffmpeg", "models", "identity", "tracking", "music"} and set(pf["models"]) == {"fast", "balanced", "accurate"}
    assert pf["ffmpeg"] is True and all(isinstance(v, bool) for v in pf["models"].values())
    assert caps["beatSync"] == pf["music"]
    m = srv.open_media(synth["plain"])
    s = call(srv, "GET", "/api/session").json()
    assert [x["id"] for x in s["media"]] == [m.id]
    assert s["media"][0]["display_w"] == 320 and s["media"][0]["duration"] > 2.9 and s["media"][0]["keyframes"][:2] == [0.0, 0.5]


def test_quit_stops_the_server(tmp_path):
    base, app, stop = start_server(tmp_path)
    try:
        assert call(app, "POST", "/api/quit", {}, key=None).status == 403
        assert call(app, "POST", "/api/quit", {}).json() == {"ok": True}
        assert app.stopped.wait(5)  # test_run_prints_a_one_use_link_and_stops_on_quit checks that the serve loop returns
    finally:
        stop()


# ---------------------------------------------------------------- open_media
def copy_to(tmp_path, src, name="clip.mp4"):
    dst = tmp_path / name
    shutil.copy(src, dst)
    return dst


def test_open_media_registers_and_id_is_stable(srv, synth, tmp_path):
    p = copy_to(tmp_path, synth["plain"])
    m = srv.open_media(str(p))
    assert len(m.id) == 12 and m.path == p.resolve() and m.size == p.stat().st_size and m.picked_copy is False
    assert m.info.codec == "h264" and m.capture_fps == pytest.approx(30, abs=0.5)
    assert m.cache_dir == srv.cache_root / m.id and srv.media_or_404(m.id) is m
    assert srv.open_media(p) is m and srv.open_media(str(p)).id == m.id
    j = m.info_json()
    assert j["name"] == "clip.mp4" and j["slowmo_suggested"] is False and j["has_audio"] is True and j["picked_copy"] is False
    json.dumps(j)
    # same path, new mtime -> a new id
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    m2 = srv.open_media(str(p))
    assert m2.id != m.id and m2.path == m.path
    # the rotated clip: display size swaps
    r = srv.open_media(synth["rot"])
    assert (r.info_json()["display_w"], r.info_json()["display_h"]) == (180, 320)


def test_open_media_refusals(srv, synth, tmp_path):
    def code(path):
        with pytest.raises(HttpError) as e:
            srv.open_media(path)
        return e.value.code, e.value.message

    assert code(tmp_path / "missing.mp4")[0] == 404
    assert "No such file" in code(tmp_path / "missing.mp4")[1]
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert code(txt)[0] == 415
    fake = tmp_path / "fake.mp4"
    fake.write_text("hello, not a video")
    assert code(fake)[0] == 415
    hls = tmp_path / "list.mp4"
    hls.write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n")
    assert code(hls)[0] == 415
    concat = tmp_path / "concat.mov"
    concat.write_text("ffconcat version 1.0\nfile '/etc/hosts'\n")
    assert code(concat)[0] == 415
    (tmp_path / ".hidden").mkdir()
    shutil.copy(synth["plain"], tmp_path / ".hidden" / "clip.mp4")
    assert code(tmp_path / ".hidden" / "clip.mp4")[0] == 403
    assert code(tmp_path)[0] == 404  # a folder
    assert code("relative/clip.mp4")[0] == 400
    audio_only = tmp_path / "audio.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(audio_only)], check=True)
    assert code(audio_only) == (415, "This file has no video stream ffprobe can read.")
    assert srv.media == {}


def test_open_media_errors_become_json(srv, synth, tmp_path):
    r = call(srv, "POST", "/api/test/open", {"path": str(tmp_path / "missing.mp4")})
    assert r.status == 404 and "No such file" in r.json()["error"]
    f = tmp_path / "fake.mp4"
    f.write_text("nope")
    assert call(srv, "POST", "/api/test/open", {"path": str(f)}).status == 415
    p = copy_to(tmp_path, synth["plain"])
    assert call(srv, "POST", "/api/test/open", {"path": str(p)}).status == 200
    assert call(srv, "POST", "/api/test/open", {}).status == 500  # a handler bug is a JSON 500, not a dropped connection


def test_media_or_404_and_run_or_404(srv):
    for fn, arg in ((srv.media_or_404, "abcdef123456"), (srv.run_or_404, "abcdef123456")):
        with pytest.raises(HttpError) as e:
            fn(arg)
        assert e.value.code == 404
    srv.runs["r1"] = Path("/x")
    assert srv.run_or_404("r1") == Path("/x")


def test_picked_copy_in_numbered_subfolder_is_detected(srv, synth, tmp_path, monkeypatch):
    # the Swift helper exports to <tmp>/inframe-picked-X/01/IMG.mp4
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    d = Path(tempfile.mkdtemp(prefix="inframe-picked-"))
    try:
        (d / "01").mkdir()
        shutil.copy(synth["plain"], d / "01" / "IMG_2.mp4")
        m = srv.open_media(d / "01" / "IMG_2.mp4")
        assert m.picked_copy is True
        out = srv.out_root(m)
        assert d.resolve() not in out.parents and home in out.parents
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_picked_copy_is_detected_and_out_root(srv, synth, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    d = Path(tempfile.mkdtemp(prefix="inframe-picked-"))
    try:
        shutil.copy(synth["plain"], d / "IMG_1.mp4")
        m = srv.open_media(d / "IMG_1.mp4")
        assert m.picked_copy is True
        out = srv.out_root(m)
        assert out.name == "IMG_1_inframe" and out.parent.parent.name in ("Movies", home.name) and home in out.parents
    finally:
        shutil.rmtree(d, ignore_errors=True)
    p = copy_to(tmp_path, synth["plain"], "beside.mp4")
    m = srv.open_media(p)
    assert srv.out_root(m) == p.resolve().parent / "beside_inframe"
    ro = tmp_path / "ro"
    ro.mkdir()
    shutil.copy(synth["plain"], ro / "locked.mp4")
    m = srv.open_media(ro / "locked.mp4")
    ro.chmod(0o555)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("running as a user that ignores directory permissions")
        assert srv.out_root(m).parent.parent == home or home in srv.out_root(m).parents
    finally:
        ro.chmod(0o755)


# ---------------------------------------------------------------- route modules
def test_route_modules_are_loaded_in_order_and_absent_ones_skipped(tmp_path, monkeypatch):
    attached = []
    fake = types.ModuleType("inframe.serve_fake")
    fake.ROUTES = [Route("GET", r"/api/fake/ping", lambda app, req: json_response({"pong": True}))]
    fake.attach = lambda app: attached.append(app)
    real_import = serve.importlib.import_module

    def fake_import(name, *a, **kw):
        if name == "inframe.serve_fake":
            return fake
        if name == "inframe.serve_absent":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(serve.importlib, "import_module", fake_import)
    monkeypatch.setattr(serve, "ROUTE_MODULES", ("serve_absent", "serve_fake"))
    base, app, stop = start_server(tmp_path)
    try:
        assert attached == [app]
        assert call(app, "GET", "/api/fake/ping").json() == {"pong": True}
    finally:
        stop()


def test_import_error_inside_a_route_module_is_not_swallowed(tmp_path, monkeypatch):
    def broken(name, *a, **kw):
        raise ModuleNotFoundError("No module named 'something_else'", name="something_else")

    monkeypatch.setattr(serve.importlib, "import_module", broken)
    monkeypatch.setattr(serve, "ROUTE_MODULES", ("serve_media",))
    with pytest.raises(ModuleNotFoundError):
        start_server(tmp_path)[2]()


def test_core_modules_do_not_import_serve():
    # route modules import serve_http, serve_paths, serve_jobs, never serve.py (no import cycle)
    src = Path(serve.__file__).parent
    for name in ("serve_http", "serve_paths", "serve_jobs"):
        text = (src / f"{name}.py").read_text()
        assert "from .serve import" not in text and "from . import serve\n" not in text and "import inframe.serve\n" not in text


def test_run_prints_a_one_use_link_and_stops_on_quit(synth, tmp_path, monkeypatch):
    monkeypatch.setenv("INFRAME_SERVE_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    lines: list[str] = []
    monkeypatch.setattr(serve.webbrowser, "open", lambda url: lines.append("OPENED " + url) or True)
    clip = copy_to(tmp_path, synth["plain"])
    result = []
    t = threading.Thread(target=lambda: result.append(serve.run(str(clip), 0, True, announce=lines.append)), daemon=True)
    t.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not any(x.startswith("OPENED") for x in lines):
        time.sleep(0.05)
    link = next(x for x in lines if x.startswith("OPENED")).split(" ", 1)[1]
    assert link.startswith("http://127.0.0.1:") and "/#n=" in link and "?" not in link
    port, nonce = int(link.split(":")[2].split("/")[0]), link.split("#n=")[1]

    class A:  # just enough of an app for call()
        pass

    a = A()
    a.port, a.token, a.media_key = port, "", ""
    r = call(a, "POST", "/api/session/claim", {"nonce": nonce}, key=None)
    assert r.status == 200
    a.token = r.json()["token"]
    s = call(a, "GET", "/api/session").json()
    assert len(s["media"]) == 1 and s["start_media"] == s["media"][0]["id"]
    assert call(a, "POST", "/api/quit", {}).status == 200
    t.join(10)
    assert not t.is_alive() and result == [0]


def test_run_with_a_bad_video_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("INFRAME_SERVE_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    lines: list[str] = []
    assert serve.run(str(tmp_path / "missing.mp4"), 0, False, announce=lines.append) == 2
    assert any("No such file" in x for x in lines)
