"""GET /models/<name> (docs/T9B_DESIGN.md section 3): the browser-mode models and the onnxruntime-web .wasm, read-only from
$INFRAME_MODELS/onnx. Only the names in manifest.json (and manifest.json itself) are served, Range works, and the page's CSP
allows WebAssembly (and nothing else new)."""
import json
import os
from pathlib import Path

import pytest

from inframe import serve
from serve_helpers import call, start_server

ROOT = Path(__file__).resolve().parents[1]
MODEL = bytes(range(256)) * 40  # 10240 bytes


@pytest.fixture
def srv(tmp_path, monkeypatch):
    models = tmp_path / "models"
    onnx = models / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "m_v1.onnx").write_bytes(MODEL)
    (onnx / "rt.wasm").write_bytes(b"\0asm\1\0\0\0")
    (onnx / "stray.onnx").write_bytes(b"not listed in the manifest")
    (onnx / ".hidden").write_bytes(b"x")
    (onnx / "manifest.json").write_text(json.dumps({"format": 1, "files": {"m_v1.onnx": {"bytes": len(MODEL)}, "rt.wasm": {"bytes": 8}, "../x": {}}}))
    (models / "secret.txt").write_text("outside onnx/")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>x</title>")
    (dist / "assets").mkdir()
    (dist / "assets" / "worker-abc.js").write_text("self.onmessage=()=>{}")
    (dist / "assets" / "chunk.mjs").write_text("export {}")
    (dist / "assets" / "style.css").write_text("body{}")
    monkeypatch.setenv("INFRAME_MODELS", str(models))
    base, app, stop = start_server(tmp_path, dist_dir=dist)
    yield app
    stop()


def get(app, path, headers=None, method="GET"):
    return call(app, method, path, headers=headers, key=None)


def test_listed_names_are_served_with_their_types(srv):
    r = get(srv, "/models/m_v1.onnx")
    assert r.status == 200 and r.body == MODEL
    assert r.headers["content-type"] == "application/octet-stream" and r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["accept-ranges"] == "bytes" and r.headers["cache-control"] == "no-cache"
    assert get(srv, "/models/rt.wasm").headers["content-type"] == "application/wasm"
    assert get(srv, "/models/manifest.json").json()["format"] == 1
    h = get(srv, "/models/m_v1.onnx", method="HEAD")
    assert h.status == 200 and h.headers["content-length"] == str(len(MODEL)) and h.body == b""


@pytest.mark.parametrize("path", [
    "/models/stray.onnx",            # in the folder, not in the manifest
    "/models/.hidden", "/models/nope.onnx", "/models/", "/models",
    "/models/../secret.txt", "/models/%2e%2e/secret.txt", "/models/..%2Fsecret.txt", "/models/m_v1.onnx/", "/models/sub/m_v1.onnx",
    "/models/%2e%2e%2fmodels%2fonnx%2fstray.onnx", "/models/m_v1.onnx%00", "/models/..\\secret.txt",
])
def test_other_paths_are_404(srv, path):
    assert get(srv, path).status == 404


def test_only_get_and_head(srv):
    for m in ("POST", "PUT", "DELETE"):
        assert call(srv, m, "/models/m_v1.onnx", body={}, key=None).status == 405
    assert call(srv, "OPTIONS", "/models/m_v1.onnx", key=None).status == 403


def test_range_requests(srv):
    r = get(srv, "/models/m_v1.onnx", {"Range": "bytes=10-19"})
    assert r.status == 206 and r.body == MODEL[10:20] and r.headers["content-range"] == f"bytes 10-19/{len(MODEL)}"
    r = get(srv, "/models/m_v1.onnx", {"Range": "bytes=-5"})
    assert r.status == 206 and r.body == MODEL[-5:]
    r = get(srv, "/models/m_v1.onnx", {"Range": f"bytes={len(MODEL) - 3}-"})
    assert r.status == 206 and r.body == MODEL[-3:]
    r = get(srv, "/models/m_v1.onnx", {"Range": f"bytes={len(MODEL)}-"})
    assert r.status == 416 and r.headers["content-range"] == f"bytes */{len(MODEL)}"


def test_symlink_in_the_folder_is_not_followed(srv, tmp_path):
    onnx = Path(os.environ["INFRAME_MODELS"]) / "onnx"
    (onnx / "link.onnx").symlink_to(tmp_path / "models" / "secret.txt")
    man = json.loads((onnx / "manifest.json").read_text())
    man["files"]["link.onnx"] = {}
    (onnx / "manifest.json").write_text(json.dumps(man))
    assert get(srv, "/models/link.onnx").status == 404


def test_no_models_folder_or_manifest_is_404(tmp_path, monkeypatch):
    monkeypatch.delenv("INFRAME_MODELS", raising=False)
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(serve, "BENCH_MODELS", tmp_path / "bench")
    base, app, stop = start_server(tmp_path)
    try:
        assert get(app, "/models/manifest.json").status == 404
        monkeypatch.setenv("INFRAME_MODELS", str(tmp_path / "empty"))
        (tmp_path / "empty" / "onnx").mkdir(parents=True)
        (tmp_path / "empty" / "onnx" / "a.onnx").write_bytes(b"x")  # no manifest: nothing is listed
        assert get(app, "/models/a.onnx").status == 404
    finally:
        stop()


def test_csp_allows_wasm_and_the_models_stay_on_the_origin():
    assert serve.CSP == ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
                         "media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; font-src 'self'; frame-ancestors 'none'; "
                         "base-uri 'none'; form-action 'none'")
    assert "'unsafe-eval'" not in serve.CSP


def test_page_carries_the_csp(srv):
    assert get(srv, "/").headers["content-security-policy"] == serve.CSP


def test_module_worker_script_carries_the_csp_too(srv):
    # The Worker that holds the user's File is a script, not a page: a policy sent only with text/html leaves it with none.
    for path in ("/assets/worker-abc.js", "/assets/chunk.mjs"):
        r = get(srv, path)
        assert r.status == 200 and r.headers["content-type"].startswith("text/javascript"), path
        assert r.headers["content-security-policy"] == serve.CSP, path
    assert "content-security-policy" not in get(srv, "/assets/style.css").headers
    assert "content-security-policy" not in get(srv, "/api/session").headers


def test_pages_headers_file_has_the_same_csp():
    text = (ROOT / "web" / "public" / "_headers").read_text()
    assert f"Content-Security-Policy: {serve.CSP}" in text


def _onnx(folder: Path, payload: bytes) -> None:
    (folder / "onnx").mkdir(parents=True)
    (folder / "onnx" / "m_v1.onnx").write_bytes(payload)
    (folder / "onnx" / "manifest.json").write_text(json.dumps({"format": 1, "files": {"m_v1.onnx": {"bytes": len(payload)}}}))


def test_without_inframe_models_the_cache_then_the_repo_models_are_served(tmp_path, monkeypatch):
    """An installed `inframe serve` has no INFRAME_MODELS; it must find the models where the CLI finds its CoreML packages."""
    monkeypatch.delenv("INFRAME_MODELS", raising=False)
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(serve, "BENCH_MODELS", tmp_path / "bench")
    _onnx(tmp_path / "bench", b"bench")
    (tmp_path / "cache" / "models" / "onnx").mkdir(parents=True)  # a cache onnx/ without a manifest is skipped
    base, app, stop = start_server(tmp_path)
    try:
        r = get(app, "/models/m_v1.onnx")
        assert r.status == 200 and r.body == b"bench"
        (tmp_path / "cache" / "models" / "onnx").rmdir()
        _onnx(tmp_path / "cache" / "models", b"cache")
        assert get(app, "/models/m_v1.onnx").body == b"cache"
    finally:
        stop()
