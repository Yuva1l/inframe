"""The Cloudflare Pages build (docs/T9C_DESIGN.md section 2): `vite build --mode pages` into a temp folder with a 127.0.0.1 model host. Checks the
generated _headers, the pinned manifest, that the placeholder host is refused, and the Pages limits. Nothing is deployed or uploaded.
Skipped while web/node_modules is absent."""
import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

from test_serve_dist import PAGES_FILE_LIMIT, check_pages_limits, scan_dist

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
VITE = WEB / "node_modules" / ".bin" / "vite"
ORIGIN = "http://127.0.0.1:8123"
BASE = f"{ORIGIN}/resolve/{'0' * 40}/"
PLACEHOLDER = "https://huggingface.co/OWNER/inframe-models/resolve/COMMIT/"

pytestmark = pytest.mark.skipif(not VITE.is_file(), reason="web/node_modules is not installed")


def vite_build(out: Path, **env: str) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("VITE_", "INFRAME_PAGES_"))}
    e.update(env)
    return subprocess.run([str(VITE), "build", "--mode", "pages", "--outDir", str(out)], cwd=WEB, env=e, capture_output=True, text=True, timeout=300)


@pytest.fixture(scope="module")
def pages(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("pages") / "pages-dist"
    r = vite_build(out, VITE_MODEL_BASE=BASE, INFRAME_PAGES_CONNECT=ORIGIN, INFRAME_PAGES_ALLOW_LOCAL="1")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out


def csp_directives(headers: str) -> dict[str, str]:
    line = next(l for l in headers.splitlines() if l.strip().startswith("Content-Security-Policy:"))
    parts = [p.strip() for p in line.split(":", 1)[1].split(";") if p.strip()]
    return {p.split(" ", 1)[0]: p.split(" ", 1)[1] if " " in p else "" for p in parts}


def test_headers_differ_from_public_only_in_connect_src(pages):
    mine = (pages / "_headers").read_text()
    public = (WEB / "public" / "_headers").read_text()
    a, b = csp_directives(mine), csp_directives(public)
    assert a["connect-src"] == f"'self' {ORIGIN}"
    assert b["connect-src"] == "'self'"
    assert {k: v for k, v in a.items() if k != "connect-src"} == {k: v for k, v in b.items() if k != "connect-src"}
    assert list(a) == list(b)  # same directives in the same order
    # the other headers are public/_headers's, plus the two additions
    others = lambda t: [l.strip() for l in t.splitlines() if l.startswith("  ") and "Content-Security-Policy" not in l]  # noqa: E731
    assert others(mine) == others(public) + ["Permissions-Policy: camera=(), microphone=(), geolocation=()", "Cache-Control: public, max-age=31536000, immutable"]
    assert re.search(r"^/assets/\*\n  Cache-Control: public, max-age=31536000, immutable$", mine, re.M)
    assert "unsafe-eval" not in mine.replace("wasm-unsafe-eval", "")


def test_the_pinned_manifest_is_copied_byte_for_byte(pages):
    assert (pages / "models" / "manifest.json").read_bytes() == (WEB / "pages" / "manifest.json").read_bytes()
    assert [p.name for p in (pages / "models").iterdir()] == ["manifest.json"]  # no model file is shipped


def test_the_page_points_at_the_model_host_and_the_local_manifest(pages):
    js = "".join(p.read_text(errors="replace") for p in (pages / "assets").glob("*.js"))
    assert BASE in js and "./models/manifest.json" in js


def test_no_asset_reaches_the_local_server_api(pages):
    for p in pages.rglob("*"):
        if p.is_file():
            assert "api/session" not in p.read_text(errors="replace"), p.name


def test_pages_limits_hold(pages):
    assert check_pages_limits(pages) == []
    assert max(p.stat().st_size for p in pages.rglob("*") if p.is_file()) < PAGES_FILE_LIMIT


def test_scan_is_clean_with_the_model_host_allowed(pages):
    assert scan_dist(pages, {"127.0.0.1"}) == []
    assert any("127.0.0.1" in p for p in scan_dist(pages))  # without allow_hosts the base is an off-origin address


def test_the_placeholder_base_fails_the_build(tmp_path):
    for env in ({"VITE_MODEL_BASE": PLACEHOLDER, "INFRAME_PAGES_CONNECT": "https://huggingface.co/OWNER/inframe-models/resolve/COMMIT/"},
                {"VITE_MODEL_BASE": PLACEHOLDER, "INFRAME_PAGES_CONNECT": "https://huggingface.co/OWNER/inframe-models/resolve/COMMIT/", "INFRAME_PAGES_ALLOW_LOCAL": "1"}):
        r = vite_build(tmp_path / "out", **env)
        assert r.returncode != 0 and "placeholder" in r.stdout + r.stderr
        assert not (tmp_path / "out" / "index.html").exists()


def test_a_base_without_a_commit_or_over_http_fails_the_build(tmp_path):
    r = vite_build(tmp_path / "a", VITE_MODEL_BASE="https://huggingface.co/me/m/resolve/main/", INFRAME_PAGES_CONNECT="https://huggingface.co/me/m/resolve/main/")
    assert r.returncode != 0 and "40-hex" in r.stdout + r.stderr
    r = vite_build(tmp_path / "b", VITE_MODEL_BASE=BASE, INFRAME_PAGES_CONNECT=ORIGIN)  # local base without INFRAME_PAGES_ALLOW_LOCAL
    assert r.returncode != 0 and "https" in r.stdout + r.stderr


def test_committed_manifest_matches_the_exported_one():
    models = os.environ.get("INFRAME_MODELS")
    src = Path(models) / "onnx" / "manifest.json" if models else None
    if not src or not src.is_file():
        pytest.skip("INFRAME_MODELS has no onnx/manifest.json")
    mine = WEB / "pages" / "manifest.json"
    assert mine.stat().st_size == src.stat().st_size
    assert hashlib.sha256(mine.read_bytes()).hexdigest() == hashlib.sha256(src.read_bytes()).hexdigest()


def test_the_committed_host_file_is_the_placeholder_or_a_pinned_https_repo():
    import json

    j = json.loads((WEB / "pages" / "model-host.json").read_text())
    pinned = re.fullmatch(r"https://[a-z0-9.-]+/[^?#]*/resolve/[0-9a-f]{40}/", j["base"])
    assert j["base"] == PLACEHOLDER or pinned
    # a first-hop path (not the whole of huggingface.co) plus Hugging Face's regional CDN hosts as the one wildcard https://*.cdn.hf.co (never all of hf.co, never the Xet upload host): no other wildcard, no bare Hugging Face origin
    assert all(re.fullmatch(r"https://(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)+(/[A-Za-z0-9._~-]+)*/?", c) for c in j["connect"])
    assert [c for c in j["connect"] if "*" in c] in ([], ["https://*.cdn.hf.co"])
    assert not any(re.fullmatch(r"https://(\*\.)?(cas-server\.)?(xethub\.)?hf\.co/?", c) or "cas-server" in c for c in j["connect"])
    assert not any(c.rstrip("/") in ("https://huggingface.co", "https://hf.co", "https://*.huggingface.co") for c in j["connect"])
    if "https://*.cdn.hf.co" in j["connect"]:
        assert re.match(r"https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/", j["base"])
    assert j["base"] == PLACEHOLDER or any(j["base"].startswith(c) and "/resolve/" in c for c in j["connect"])  # the first hop is pinned to the repo and commit
