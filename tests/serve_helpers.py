"""Shared helpers for the `inframe serve` tests (every part imports this file).

    base, app, stop = start_server(tmp_path, **app_kw)     # port 0, INFRAME_SERVE_STATE and INFRAME_CACHE inside tmp_path
    r = call(app, "POST", "/api/quit", {})                  # r.status, r.headers, r.body, r.json()
    stop()

`call(app, method, path, body=None, headers=None, key="token")`: `key="token"` sends the session token in X-Inframe-Token,
`key="media"` appends `m=<media key>` to the query, `key=None` sends no secret. A dict `body` is sent as JSON; bytes as they
are. POST, PUT and DELETE get the page's own Origin unless `headers` sets one; a header whose value is None is left out
(`{"Origin": None}`, `{"Host": None}`). `start_server` takes `extra_routes=[Route(...)]` and `max_threads`, and any `App` argument
(`dist_dir=...`).
"""
from __future__ import annotations

import http.client
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from inframe import serve

ENV = ("INFRAME_SERVE_STATE", "INFRAME_CACHE")


@dataclass
class Reply:
    status: int
    headers: dict[str, str]  # lower-cased names
    body: bytes

    def json(self):
        return json.loads(self.body.decode())

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")


def start_server(tmp_path: Path, **app_kw):
    """Start `inframe serve` on 127.0.0.1:0 in a thread. Returns (base_url, app, stop)."""
    saved = {k: os.environ.get(k) for k in ENV}
    os.environ["INFRAME_SERVE_STATE"] = str(Path(tmp_path) / "serve-state")
    os.environ["INFRAME_CACHE"] = str(Path(tmp_path) / "cache")
    app_kw.setdefault("dist_dir", Path(tmp_path) / "no-dist")
    httpd, app = serve.create(0, **app_kw)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()

    def stop():
        try:
            httpd.shutdown()
            httpd.server_close()
            thread.join(5)
            app.close()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return f"http://127.0.0.1:{app.port}", app, stop


def call(app, method: str, path: str, body=None, headers: dict | None = None, key: str | None = "token", timeout: float = 30,
         host: str = "127.0.0.1") -> Reply:
    port = app.port
    h: dict[str, str | None] = {"Host": f"127.0.0.1:{port}"}
    if method in ("POST", "PUT", "DELETE"):
        h["Origin"] = f"http://127.0.0.1:{port}"
    if key == "token":
        h["X-Inframe-Token"] = app.token
    elif key == "media":
        path += ("&" if "?" in path else "?") + "m=" + app.media_key
    data = None
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    elif isinstance(body, (bytes, bytearray)):
        data = bytes(body)
    elif isinstance(body, str):
        data = body.encode()
    if data is not None:
        h["Content-Length"] = str(len(data))
    h.update(headers or {})
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in h.items():
            if v is not None:
                conn.putheader(k, v)
        conn.endheaders(data)
        r = conn.getresponse()
        return Reply(r.status, {k.lower(): v for k, v in r.getheaders()}, r.read() if method != "HEAD" else b"")
    finally:
        conn.close()
