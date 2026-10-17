"""tools/pages_preflight.py (docs/T9C_DEPLOY.md): the local checks run on tmp folders with tiny fake files; --remote runs only against a stub server on
127.0.0.1 (a guard makes any other host fail the test, so Hugging Face is never contacted)."""
import hashlib
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pages_preflight as pf  # noqa: E402

COMMIT = "a" * 40
BASE = f"https://huggingface.co/me/inframe-models/resolve/{COMMIT}/"
CONNECT = [BASE, "https://cas-bridge.xethub.hf.co"]
DATA = {"a.onnx": b"A" * 40, "b.wasm": b"B" * 25}


@pytest.fixture(autouse=True)
def only_loopback(monkeypatch):
    """No test may reach another host: name lookups other than loopback fail."""
    real = socket.getaddrinfo

    def guard(host, *a, **k):
        if host not in ("127.0.0.1", "localhost", None):
            raise AssertionError(f"a test tried to resolve {host!r}")
        return real(host, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", guard)


def write_manifest(path: Path, data=DATA) -> Path:
    files = {n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in data.items()}
    path.write_text(json.dumps({"format": 1, "files": files}))
    return path


def make_site(tmp: Path, base=BASE, connect=CONNECT, headers_connect=None):
    """A fake pages-dist, model-host.json and manifest."""
    dist = tmp / "pages-dist"
    tmp.mkdir(parents=True, exist_ok=True)
    (dist / "assets").mkdir(parents=True)
    (dist / "models").mkdir()
    (dist / "index.html").write_text("<html></html>")
    (dist / "assets" / "app.js").write_text(f'const base="{base}";')
    manifest = write_manifest(tmp / "manifest.json")
    (dist / "models" / "manifest.json").write_bytes(manifest.read_bytes())
    tokens = ["'self'", *(connect if headers_connect is None else headers_connect)]
    (dist / "_headers").write_text(f"/*\n  Content-Security-Policy: default-src 'self'; connect-src {' '.join(tokens)}; frame-ancestors 'none'\n")
    host = tmp / "model-host.json"
    host.write_text(json.dumps({"base": base, "connect": connect}))
    return dist, host, manifest


def local(tmp, **kw):
    allow = kw.pop("allow_local", False)
    dist, host, manifest = make_site(tmp, **kw)
    return pf.check_local(dist, host, manifest, allow)


def test_a_good_site_has_no_problems(tmp_path):
    assert local(tmp_path) == []


def test_an_oversize_file_is_reported(tmp_path):
    dist, host, manifest = make_site(tmp_path)
    with open(dist / "assets" / "big.bin", "wb") as f:
        f.truncate(pf.PAGES_FILE_LIMIT + 1)
    problems = pf.check_local(dist, host, manifest)
    assert len(problems) == 1 and "assets/big.bin" in problems[0] and "25 MiB" in problems[0]
    (dist / "assets" / "big.bin").write_bytes(b"")
    with open(dist / "assets" / "ok.bin", "wb") as f:
        f.truncate(pf.PAGES_FILE_LIMIT)  # exactly 25 MiB is allowed
    assert pf.check_local(dist, host, manifest) == []


def test_too_many_files_are_reported(tmp_path, monkeypatch):
    dist, host, manifest = make_site(tmp_path)
    monkeypatch.setattr(pf, "PAGES_FILES_MAX", 5)
    for i in range(5):
        (dist / "assets" / f"f{i}.txt").write_text("x")
    assert any("files; static assets allow fewer than 5" in p for p in pf.check_local(dist, host, manifest))


def test_a_connect_src_that_does_not_match_the_model_host_is_reported(tmp_path):
    problems = local(tmp_path, headers_connect=[BASE])  # cas-bridge missing
    assert len(problems) == 1 and "does not match model-host.json" in problems[0] and "cas-bridge.xethub.hf.co" in problems[0]
    problems = local(tmp_path / "x", headers_connect=[*CONNECT, "https://evil.example"])
    assert len(problems) == 1 and "extra ['https://evil.example']" in problems[0]


def test_a_placeholder_base_is_reported(tmp_path):
    ph = "https://huggingface.co/OWNER/inframe-models/resolve/COMMIT/"
    problems = local(tmp_path, base=ph, connect=[ph, "https://cas-bridge.xethub.hf.co"])
    assert len(problems) == 1 and "placeholder" in problems[0]


@pytest.mark.parametrize("base, what", [
    ("https://huggingface.co/me/inframe-models/resolve/main/", "40-hex commit"),
    (f"https://huggingface.co/me/inframe-models/resolve/{'a' * 39}/", "40-hex commit"),
    (f"http://huggingface.co/me/inframe-models/resolve/{COMMIT}/", "not https"),
    (f"https://huggingface.co/me/inframe-models/resolve/{COMMIT}", "slash"),
])
def test_a_base_that_is_not_a_pinned_https_commit_is_reported(tmp_path, base, what):
    problems = local(tmp_path, base=base, connect=[base, "https://cas-bridge.xethub.hf.co"])
    assert any(what in p for p in problems), problems


def test_a_base_not_covered_by_connect_and_a_whole_huggingface_entry_are_reported(tmp_path):
    problems = local(tmp_path, connect=["https://huggingface.co/other/repo/resolve/" + COMMIT + "/"])
    assert any("not covered by connect" in p for p in problems)
    problems = local(tmp_path / "y", connect=["https://huggingface.co", "https://hf.co"])
    assert sum("opens all of Hugging Face" in p for p in problems) == 2


def test_the_hf_cdn_wildcard_is_accepted_only_for_a_pinned_huggingface_base(tmp_path):
    """Hugging Face redirects to a regional CDN host that changes over time: `https://*.cdn.hf.co` is the one wildcard allowed, and only with a pinned huggingface.co resolve base."""
    ok = [BASE, "https://*.cdn.hf.co", "https://cas-bridge.xethub.hf.co"]
    assert local(tmp_path, connect=ok) == []
    assert pf.check_host({"base": BASE, "connect": ok}) == []
    # the headers file must carry the same entry
    assert local(tmp_path / "h", connect=ok, headers_connect=[BASE]) != []
    # another base host (a pinned URL elsewhere) does not get the wildcard
    other = f"https://example.org/me/inframe-models/resolve/{COMMIT}/"
    problems = local(tmp_path / "o", base=other, connect=[other, "https://*.cdn.hf.co"])
    assert any("accepted only when the model base is a pinned https://huggingface.co" in p for p in problems), problems
    # a huggingface.co base that is not pinned to a commit does not get it either
    unpinned = "https://huggingface.co/me/inframe-models/resolve/main/"
    problems = local(tmp_path / "u", base=unpinned, connect=[unpinned, "https://*.cdn.hf.co"])
    assert any("accepted only when" in p for p in problems), problems


@pytest.mark.parametrize("entry", ["https://cas-server.xethub.hf.co", "https://cas-server.xethub.hf.co/v1/"])
def test_the_hugging_face_upload_host_is_never_allowed(tmp_path, entry):
    """The Xet CAS server takes uploads: connect-src is the backstop that footage stays in the browser, so no entry may cover it."""
    problems = local(tmp_path, connect=[*CONNECT, entry])
    assert any("upload host" in p for p in problems), problems
    assert not any("upload host" in p for p in local(tmp_path / "ok", connect=[BASE, "https://*.cdn.hf.co", "https://cas-bridge.xethub.hf.co"]))
    assert not pf.connect_covers("https://*.cdn.hf.co", "https://cas-server.xethub.hf.co/v1/xorbs/default/0", False)


@pytest.mark.parametrize("entry", ["https://*.huggingface.co", "https://*.hf.co", "https://*.xethub.hf.co", "https://*.*.hf.co", "https://*.example.co.uk", "https://*.example.com", "https://*.cdn.hf.co:8443", "https://*.cdn.hf.co/", "https://*.cdn.hf.co/x/"])
def test_every_other_wildcard_stays_refused_even_for_a_pinned_huggingface_base(tmp_path, entry):
    problems = local(tmp_path, connect=[BASE, entry])
    assert any(entry in p and "connect entry" in p for p in problems), problems


@pytest.mark.parametrize("entry", ["*", "https:", "data:", "blob:", "wss:", "https://*", "https://*.com", "https://*.com.", "https://*.co.uk", "https://*.co.uk.", "https://*.github.io", "https://*.pages.dev", "https://example.com:*", "'self'", "'unsafe-inline'"])
def test_a_connect_entry_that_opens_the_page_to_anywhere_is_reported(tmp_path, entry):
    problems = local(tmp_path, connect=[*CONNECT, entry])
    assert any(entry in p and "connect entry" in p for p in problems), problems


def test_a_connect_path_without_a_trailing_slash_matches_exactly(tmp_path):
    exact = "https://huggingface.co/me/inframe-models/resolve/" + COMMIT
    assert not pf.connect_covers(exact, exact + "/a.onnx", True) and not pf.connect_covers(exact, exact + "/", True)
    assert pf.connect_covers(exact, exact, True) and pf.connect_covers(exact, exact + "/a.onnx", False)
    problems = local(tmp_path, base=BASE, connect=[BASE.rstrip("/"), "https://cas-bridge.xethub.hf.co"])
    assert any("not covered by connect" in p for p in problems)


def test_a_missing_dist_and_a_stale_build_are_reported(tmp_path):
    host = tmp_path / "h.json"
    host.write_text(json.dumps({"base": BASE, "connect": CONNECT}))
    manifest = write_manifest(tmp_path / "m.json")
    assert "build:pages" in pf.check_local(tmp_path / "nope", host, manifest)[0]
    dist, host, manifest = make_site(tmp_path / "s")
    (dist / "assets" / "app.js").write_text("const base='https://huggingface.co/someone/else/resolve/" + COMMIT + "/';")
    (dist / "models" / "manifest.json").write_text("{}")
    problems = pf.check_local(dist, host, manifest)
    assert any("differs from web/pages/manifest.json" in p for p in problems) and any("another host" in p for p in problems)
    (dist / "_headers").unlink()
    assert any("_headers is missing" in p for p in pf.check_local(dist, host, manifest))


def test_connect_covers_follows_csp_path_rules():
    repo = f"https://huggingface.co/me/inframe-models/resolve/{COMMIT}/"
    assert pf.connect_covers(repo, repo + "a.onnx", True)
    assert not pf.connect_covers(repo, "https://huggingface.co/me/other/resolve/x/a.onnx", True)
    assert pf.connect_covers(repo, "https://huggingface.co/cdn/whatever?Signature=1", False)  # after a redirect the origin alone is compared
    assert pf.connect_covers("https://*.cdn.hf.co", "https://us.aws.cdn.hf.co/x", False)
    assert not pf.connect_covers("https://*.cdn.hf.co", "https://hf.co/x", False) and not pf.connect_covers("https://*.cdn.hf.co", "https://evilcdn.hf.co/x", False)
    assert not pf.connect_covers("https://a.example", "http://a.example/x", False) and not pf.connect_covers("https://a.example", "https://a.example:8443/x", False)


def test_the_committed_files_are_checked_without_a_dist():
    host, errs = pf.load_host(pf.WEB / "pages" / "model-host.json")
    assert errs == [] and host
    problems = pf.check_host(host)
    assert all("placeholder" in p for p in problems) or problems == []  # the committed file is the placeholder until the repo is pinned


# --- remote mode, against a stub server on 127.0.0.1 ------------------------------------------------------------------------------------------

class Stub(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, data=DATA, acao="*", cdn_acao="*", range_ok=True, cdn_host=None, content_length_lie=0, get_status=None):
        super().__init__(("127.0.0.1", 0), Handler)
        self.data, self.acao, self.cdn_acao, self.range_ok, self.cdn_host, self.lie = data, acao, cdn_acao, range_ok, cdn_host, content_length_lie
        self.get_status = get_status  # a status every GET on /cdn/ answers with (HEAD still gets the file), e.g. 403 for signed URLs that refuse GET
        self.log: list[tuple[str, str, str | None]] = []

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.serve(False)

    def do_GET(self):
        self.serve(True)

    def serve(self, body: bool):
        s: Stub = self.server  # type: ignore[assignment]
        s.log.append((self.command, self.path, self.headers.get("Origin")))
        path = self.path.split("?")[0]
        if path.startswith("/resolve/"):
            name = path.rsplit("/", 1)[1]
            self.send_response(302)
            self.send_header("Location", f"{s.cdn_host or ''}/cdn/{name}?Expires=1&Signature=x")
            if s.acao:
                self.send_header("Access-Control-Allow-Origin", s.acao)
            self.send_header("Content-Length", "0")
            return self.end_headers()
        data = s.data.get(path.rsplit("/", 1)[1])
        if path.startswith("/cdn/") and data is not None and body and s.get_status:
            self.send_response(s.get_status)
            if s.cdn_acao:
                self.send_header("Access-Control-Allow-Origin", s.cdn_acao)
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if path.startswith("/cdn/") and data is not None:
            rng = self.headers.get("Range")
            status, out, extra = 200, data, {}
            if rng == "bytes=0-0" and s.range_ok:
                status, out, extra = 206, data[:1], {"Content-Range": f"bytes 0-0/{len(data)}"}
            self.send_response(status)
            if s.cdn_acao:
                self.send_header("Access-Control-Allow-Origin", s.cdn_acao)
            self.send_header("Accept-Ranges", "bytes")
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(out) + (s.lie if not extra else 0)))
            self.end_headers()
            if body:
                self.wfile.write(out)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def stub():
    servers = []

    def make(**kw):
        s = Stub(**kw)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.shutdown()
        s.server_close()


def host_for(s: Stub, connect=None):
    return {"base": f"{s.origin}/resolve/{COMMIT}/", "connect": [s.origin] if connect is None else connect}


def remote(tmp, s, connect=None, digest=False, origin="https://inframe.pages.dev"):
    m = write_manifest(tmp / "m.json")
    return pf.check_remote(host_for(s, connect), m, origin, timeout=5, digest=digest, allow_local=True)


def test_remote_ok_reports_acao_sizes_and_the_redirect_host(tmp_path, stub):
    s = stub()
    rep = remote(tmp_path, s, digest=True)
    assert rep.problems == []
    text = "\n".join(rep.lines)
    assert "ACAO *" in text and "a.onnx (40 bytes)" in text and "HEAD content-length 40, manifest 40" in text
    assert "range: 206, total 40" in text and "digest: 40 bytes" in text and "matches the manifest" in text
    assert list(rep.hosts) == [s.origin.removeprefix("http://")] and all(rep.hosts.values())
    assert {m for m, _, _ in s.log} == {"HEAD", "GET"}
    assert {o for _, _, o in s.log} == {"https://inframe.pages.dev"}  # the Origin header reached the server


def test_remote_reports_a_missing_acao_on_the_redirect_and_a_wrong_origin(tmp_path, stub):
    rep = remote(tmp_path, stub(cdn_acao=None))
    assert any("hop 2" in p and "ACAO (none)" in p for p in rep.problems)
    rep = remote(tmp_path, stub(acao="https://other.example", cdn_acao="https://other.example"), origin="https://inframe.pages.dev")
    assert any("https://other.example" in p and "blocked" in p for p in rep.problems)
    rep = remote(tmp_path, stub(acao="https://inframe.pages.dev", cdn_acao="https://inframe.pages.dev"))
    assert rep.problems == []


def test_remote_reports_a_size_that_differs_from_the_manifest(tmp_path, stub):
    rep = remote(tmp_path, stub(data={"a.onnx": b"A" * 41, "b.wasm": DATA["b.wasm"]}))
    assert any("a.onnx" in p and "content-length 41" in p for p in rep.problems)
    assert any("a.onnx" in p and "Content-Range total 41" in p for p in rep.problems)
    assert not any("b.wasm" in p for p in rep.problems)


def test_remote_reports_a_redirect_host_that_connect_src_does_not_cover(tmp_path, stub):
    s = stub()
    other = stub(data=DATA)  # the CDN: same server software on another port, reached as localhost so the host name differs
    s.cdn_host = f"http://localhost:{other.server_address[1]}"
    rep = remote(tmp_path, s)
    assert any("not covered by connect-src" in p and "localhost" in p for p in rep.problems)
    assert any(not c for c in rep.hosts.values()) and "NOT covered" in "\n".join(rep.lines)
    covered_rep = remote(tmp_path, s, connect=[s.origin, s.cdn_host])
    assert covered_rep.problems == [], covered_rep.problems


def test_remote_notes_a_host_that_ignores_range_but_does_not_fail_it(tmp_path, stub):
    rep = remote(tmp_path, stub(range_ok=False))
    assert rep.problems == [] and "Range answered 200" in "\n".join(rep.lines)


def test_remote_reports_a_host_whose_range_get_is_refused(tmp_path, stub):
    """HEAD answers 200 but every GET (the request the page makes) is refused: that is a problem, not a note."""
    for status in (403, 404, 500, 302):
        rep = remote(tmp_path, stub(get_status=status))
        assert any("GET" in p and f"status {status}" in p for p in rep.problems), (status, rep.problems)
    rep = remote(tmp_path, stub(get_status=403))
    assert not any("not a problem for the page" in line for line in rep.lines)


def test_remote_a_cross_origin_redirect_makes_later_hops_send_origin_null(tmp_path, stub):
    """Fetch spec: after a redirect to another origin the tainted-origin flag is set and the next request carries `Origin: null`, so an ACAO equal to
    the page origin is blocked on that hop; only `*` or `null` passes."""
    page = "https://inframe.pages.dev"
    s, other = stub(acao=page), stub(cdn_acao=page)
    s.cdn_host = f"http://localhost:{other.server_address[1]}"  # a different origin: 127.0.0.1 -> localhost:other-port
    rep = remote(tmp_path, s, connect=[s.origin, s.cdn_host], origin=page)
    assert any("hop 2" in p and "blocked" in p and page in p for p in rep.problems), rep.problems
    assert {o for _, path, o in other.log if path.startswith("/cdn/")} == {"null"}  # what the CDN really saw
    assert {o for _, path, o in s.log} == {page}  # the first hop still carries the page origin
    for allowed in ("*", "null"):
        s2, other2 = stub(acao=page), stub(cdn_acao=allowed)
        s2.cdn_host = f"http://localhost:{other2.server_address[1]}"
        assert remote(tmp_path, s2, connect=[s2.origin, s2.cdn_host], origin=page).problems == []


def test_remote_digest_catches_changed_bytes(tmp_path, stub):
    rep = remote(tmp_path, stub(data={"a.onnx": b"Z" * 40, "b.wasm": DATA["b.wasm"]}), digest=True)
    assert [p for p in rep.problems if "differs from the manifest" in p and "a.onnx" in p]
    assert not any("b.wasm" in p for p in rep.problems)


def test_remote_reports_a_missing_file_and_a_refused_connection(tmp_path, stub):
    s = stub(data={"a.onnx": DATA["a.onnx"]})
    rep = remote(tmp_path, s)
    assert any("b.wasm" in p and "status 404" in p for p in rep.problems)
    s.shutdown()
    s.server_close()
    rep = remote(tmp_path, s)
    assert any("request failed" in p for p in rep.problems)


def test_main_remote_refuses_the_placeholder_and_never_resolves_a_host(tmp_path, capsys):
    host = tmp_path / "h.json"
    ph = "https://huggingface.co/OWNER/inframe-models/resolve/COMMIT/"
    host.write_text(json.dumps({"base": ph, "connect": [ph]}))
    rc = pf.main(["--remote", "--dist", str(tmp_path / "none"), "--host-file", str(host), "--manifest", str(write_manifest(tmp_path / "m.json"))])
    out = capsys.readouterr().out
    assert rc == 1 and "remote: not run" in out and "placeholder" in out


def test_main_runs_both_modes_against_the_stub(tmp_path, stub, capsys):
    s = stub()
    dist, host, manifest = make_site(tmp_path, base=f"{s.origin}/resolve/{COMMIT}/", connect=[s.origin])
    argv = ["--dist", str(dist), "--host-file", str(host), "--manifest", str(manifest), "--allow-local"]
    assert pf.main(argv) == 0
    assert pf.main([*argv, "--remote", "--origin", "https://inframe.pages.dev", "--timeout", "5"]) == 0
    out = capsys.readouterr().out
    assert "local: OK" in out and "remote: OK" in out
    s.cdn_acao = None
    assert pf.main([*argv, "--remote", "--timeout", "5"]) == 1
    assert "PROBLEM" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pf.main([*argv, "--digest"])  # --digest needs --remote


def test_no_dist_checks_only_the_host_file(tmp_path, capsys):
    host = tmp_path / "h.json"
    host.write_text(json.dumps({"base": BASE, "connect": CONNECT}))
    m = write_manifest(tmp_path / "m.json")
    assert pf.check_local(None, host, m) == []
    assert pf.main(["--no-dist", "--host-file", str(host), "--manifest", str(m), "--dist", str(tmp_path / "none")]) == 0
    assert pf.main(["--host-file", str(host), "--manifest", str(m), "--dist", str(tmp_path / "none")]) == 1
    assert "build:pages" in capsys.readouterr().out


def test_a_wildcard_over_a_registrable_name_is_still_allowed():
    assert pf.broad_connect("https://*.cdn.hf.co") is None and pf.broad_connect("https://*.example.co.uk") is None
