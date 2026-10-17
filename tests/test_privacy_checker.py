"""The pure privacy checker of the Pages test (tests/browser/privacy.py) on fake logs: a clean log passes, and a POST, a third origin, the canary in a query
string (and in other forms and places), a body, a wrong path and a long URL are each reported. No browser, no network."""
from __future__ import annotations

import base64
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "browser"))

import pytest  # noqa: E402
from privacy import Seen, check, variants  # noqa: E402
from static_hosts import headers_for, parse_headers_file, pattern_matches  # noqa: E402

A, B = "http://127.0.0.1:5001", "http://127.0.0.1:5002"
COMMIT = "0" * 40
NAMES = ["rfdetr_small_v1.onnx", "dinov2_s_v1.onnx", "ort-wasm-simd-threaded.asyncify.wasm"]
MODEL_PATHS = {p for n in NAMES for p in (f"/resolve/{COMMIT}/{n}", f"/cdn/{n}")}
PAGE_FILES = {"/", "/index.html", "/assets/index-abc.js", "/models/manifest.json"}
CANARY = b"3f9a6c0d1b7e4a5c8d2e9f0a1b3c4d5e"
SECRETS = {"canary": CANARY, "file name": b"PRIVATE-9d8c-session.mp4", "clip run": bytes(range(7, 71))}
CLEAN = [
    Seen("GET", f"{A}/", {"accept": "text/html"}),
    Seen("GET", f"{A}/assets/index-abc.js"),
    Seen("GET", f"{A}/models/manifest.json", {"cache-control": "no-cache"}),
    Seen("GET", f"{B}/resolve/{COMMIT}/rfdetr_small_v1.onnx", {"origin": A}),
    Seen("GET", f"{B}/cdn/rfdetr_small_v1.onnx?Expires=1&Signature=abcd", {"origin": A, "range": "bytes=0-99"}),
    Seen("HEAD", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=abcd"),
    Seen("GET", "blob:http://127.0.0.1:5001/1234-abcd"),
    Seen("GET", "data:application/octet-stream;base64,AAAA"),
]


ISSUED = {s.url for s in CLEAN if "/cdn/" in s.url}  # the signed URLs the (fake) model host put in its Location headers
SIZES = {"rfdetr_small_v1.onnx": 123_000_000, "dinov2_s_v1.onnx": 88_000_000, "ort-wasm-simd-threaded.asyncify.wasm": 26_000_000}


def run(seen, **kw):
    args = dict(page_origin=A, model_origin=B, model_paths=MODEL_PATHS, page_files=PAGE_FILES, secrets=SECRETS, issued=ISSUED, sizes=SIZES)
    args.update(kw)
    return check(seen, **args)


def test_a_clean_log_passes():
    assert run(CLEAN) == []


def test_a_post_is_reported_with_its_body():
    out = run(CLEAN + [Seen("POST", f"{A}/assets/index-abc.js", {}, b"x" * 10)])
    assert any("method POST" in p for p in out) and any("request body of 10 bytes" in p for p in out)


def test_options_put_and_a_head_to_the_page_host_are_reported():
    for method, url in (("OPTIONS", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=a"), ("PUT", f"{B}/resolve/{COMMIT}/dinov2_s_v1.onnx"), ("HEAD", f"{A}/")):
        assert any(f"method {method}" in p for p in run(CLEAN + [Seen(method, url)])), method


def test_a_third_origin_is_reported():
    out = run(CLEAN + [Seen("GET", "https://analytics.example.com/collect")])
    assert [p for p in out if "third origin https://analytics.example.com" in p]
    # also a second port on the same host, and the same host over https
    assert run(CLEAN + [Seen("GET", "http://127.0.0.1:5003/x")])
    assert run(CLEAN + [Seen("GET", f"{A.replace('http', 'https')}/")])


def test_the_canary_in_a_query_string_is_reported():
    out = run(CLEAN + [Seen("GET", f"{A}/assets/index-abc.js?id={CANARY.decode()}")])
    assert any("secret 'canary' found in the URL" in p for p in out)
    assert any("query string on the page host" in p for p in out)


@pytest.mark.parametrize("make", [
    lambda s: Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature={s}"),  # in a model query
    lambda s: Seen("GET", f"{A}/assets/index-abc.js", {"x-trace": s}),  # in a header value
    lambda s: Seen("GET", f"{A}/assets/index-abc.js", {s: "1"}),  # in a header name
    lambda s: Seen("GET", f"{A}/assets/index-abc.js", {}, b"hello " + s.encode()),  # in a body
    lambda s: Seen("GET", f"{A}/assets/index-abc.js#" + s),  # in the fragment
])
def test_the_canary_is_found_wherever_it_hides(make):
    out = run(CLEAN + [make(CANARY.decode())])
    assert any("secret 'canary'" in p for p in out), out


@pytest.mark.parametrize("label", ["canary", "file name", "clip run"])
def test_every_secret_form_is_found(label):
    secret = SECRETS[label]
    for form in variants(secret):
        text = form.decode("latin-1")
        out = run(CLEAN + [Seen("GET", f"{A}/assets/index-abc.js", {"x-a": text})])
        assert any(f"secret '{label}'" in p for p in out), (label, form)
    # base64 inside a URL, hex inside a URL, and the file name percent-encoded
    b64 = base64.urlsafe_b64encode(secret).rstrip(b"=").decode()
    assert any(f"secret '{label}'" in p for p in run(CLEAN + [Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature={b64}")]))
    assert any(f"secret '{label}'" in p for p in run(CLEAN + [Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature={secret.hex()}")]))


@pytest.mark.parametrize("label", ["canary", "file name", "clip run"])
@pytest.mark.parametrize("encode", [base64.b64encode, base64.urlsafe_b64encode], ids=["standard", "urlsafe"])
def test_a_secret_inside_a_longer_base64_text_is_found_at_every_offset(label, encode):
    """base64(prefix + secret + suffix) does not hold base64(secret) unless the prefix is a multiple of 3 bytes: all three alignments must be found."""
    secret = SECRETS[label]
    missed = []
    for pre in (b"", b"x", b"xy", b"xyz", b"{\"a\":\""):
        for post in (b"", b"}", b"\"}", b"abc", b"abcd"):
            blob = encode(pre + secret + post).decode()
            for place in (Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature={blob}"), Seen("GET", f"{A}/assets/index-abc.js", {"x-t": blob.rstrip("=")})):
                if not any(f"secret '{label}'" in p for p in run(CLEAN + [place])):
                    missed.append((pre, post))
    assert missed == []


def test_clean_base64_that_holds_no_secret_is_not_reported():
    blob = base64.b64encode(bytes(range(200, 256)) * 3).decode()
    assert not [p for p in run(CLEAN + [Seen("GET", f"{A}/assets/index-abc.js", {"x-t": blob})]) if "secret" in p]


def test_derived_clip_data_in_headers_range_and_signature_is_reported():
    import hashlib

    digest = hashlib.sha256(b"PRIVATE-9d8c-session.mp4").hexdigest()
    cdn = f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature="
    bad = {
        "custom header with the clip's facts": Seen("GET", f"{B}/resolve/{COMMIT}/dinov2_s_v1.onnx", {"x-clip": "dur=10.0;size=1234567;w=640;h=360"}),
        "header named after a hash": Seen("GET", cdn + "abcd", {"x-" + digest[:16]: "1"}),
        "range that is a number from a hash": Seen("GET", cdn + "abcd", {"range": "bytes=0-" + str(int(digest, 16))}),
        "range outside the file": Seen("GET", cdn + "abcd", {"range": "bytes=0-1234567890"}),
        "range that is not a plain byte range": Seen("GET", cdn + "abcd", {"range": "bytes=0-9, 20-29"}),
        "range with a unit": Seen("GET", cdn + "abcd", {"range": "dur=10.0"}),
        "signature that is a hash of the name": Seen("GET", cdn + digest),
        "expires that is a number from the clip": Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1234567&Signature=abcd"),
    }
    for what, s in bad.items():
        assert run(CLEAN + [s]), what
    # the unit-test fixtures are not weakened: with nothing in `issued` and `sizes` the old rules still pass what they passed
    assert run(CLEAN, issued=None, sizes=None) == []


def test_a_cdn_on_a_third_host_follows_the_redirect_rules():
    c = "http://127.0.0.1:5003"
    cdn = f"{c}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=abcd"
    seen = [s for s in CLEAN if "/cdn/" not in s.url and not s.url.startswith(B + "/cdn")] + [Seen("GET", cdn, {"range": "bytes=0-99"}), Seen("HEAD", cdn)]
    kw = dict(redirect_origin=c, issued={cdn})
    assert run(seen, **kw) == []
    assert any("not the host the model host redirects to" in p for p in run(seen + [Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=a")], **kw))
    assert any("not the model host" in p for p in run(seen + [Seen("GET", f"{c}/resolve/{COMMIT}/dinov2_s_v1.onnx")], **kw))
    assert any("not one the model host issued" in p for p in run(seen + [Seen("GET", f"{c}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=other")], **kw))
    assert any(f"the cdn origin {c} was never contacted" in p for p in run([s for s in seen if not s.url.startswith(c)], **kw))
    assert any("third origin" in p for p in run(seen + [Seen("GET", "http://127.0.0.1:5009/cdn/dinov2_s_v1.onnx?Expires=1&Signature=a")], **kw))


def test_the_file_name_is_found_in_a_different_case():
    assert any("secret 'file name'" in p for p in run(CLEAN + [Seen("GET", f"{A}/assets/index-abc.js", {"x-n": "private-9D8C-Session.MP4"})]))


def test_model_host_rules():
    bad = [
        Seen("GET", f"{B}/resolve/{COMMIT}/manifest.json"),  # the manifest is the page's own
        Seen("GET", f"{B}/resolve/{COMMIT}/rfdetr_small_v1.onnx?x=1"),  # a query on the first hop
        Seen("GET", f"{B}/cdn/rfdetr_small_v1.onnx?Expires=1&Signature=a&user=me"),  # an extra key
        Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=a", {"cookie": "a=b"}),
        Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature=a", {"referer": f"{A}/"}),
    ]
    for s in bad:
        assert run(CLEAN + [s]), s


def test_page_host_rules():
    assert any("not in the built folder" in p for p in run(CLEAN + [Seen("GET", f"{A}/api/session")]))
    assert any("query string" in p for p in run(CLEAN + [Seen("GET", f"{A}/?x=1")]))
    assert any("query string" in p for p in run(CLEAN + [Seen("GET", f"{A}/assets/index-abc.js?")]))


def test_a_long_url_is_reported():
    out = run(CLEAN + [Seen("GET", f"{B}/cdn/dinov2_s_v1.onnx?Expires=1&Signature={'a' * 600}")])
    assert any("characters (limit 512)" in p for p in out)
    assert run([replace(CLEAN[0])] + CLEAN[1:], max_url=10)


def test_both_origins_must_be_contacted():
    out = run([s for s in CLEAN if not s.url.startswith(B)])
    assert any(f"the model origin {B} was never contacted" in p for p in out)
    assert run([s for s in CLEAN if not s.url.startswith(B)], require_both=False) == []


def test_all_three_faults_at_once_are_all_reported():
    seen = CLEAN + [
        Seen("POST", f"{A}/assets/index-abc.js"),
        Seen("GET", "https://tracker.example.net/p.gif"),
        Seen("GET", f"{A}/assets/index-abc.js?c={CANARY.decode()}"),
    ]
    out = run(seen)
    assert any("method POST" in p for p in out)
    assert any("third origin https://tracker.example.net" in p for p in out)
    assert any("secret 'canary' found in the URL" in p for p in out)


# ---- the _headers parser the page host uses
HEADERS = "# c\n/*\n  A: 1\n  CSP: x; y\n\n/assets/*\n  Cache-Control: immutable\n  A: 2\n"


def test_headers_file_rules():
    rules = parse_headers_file(HEADERS)
    assert rules == [("/*", {"A": "1", "CSP": "x; y"}), ("/assets/*", {"Cache-Control": "immutable", "A": "2"})]
    assert headers_for(rules, "/") == {"A": "1", "CSP": "x; y"}
    assert headers_for(rules, "/assets/x.js") == {"A": "1, 2", "CSP": "x; y", "Cache-Control": "immutable"}
    assert pattern_matches("/assets/*", "/assets/a/b.js") and not pattern_matches("/assets/*", "/assetsx") and pattern_matches("/*", "/")


# ---- the two local hosts of the Pages test, on tiny files
def _req(origin, method, path, headers=None, body=None):
    import http.client
    from urllib.parse import urlsplit

    c = http.client.HTTPConnection(urlsplit(origin).netloc, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, data)
    c.close()
    return out


def test_the_model_host_behaves_like_huggingface_and_logs_everything(tmp_path):
    """The first hop (/resolve/) and the signed files (/cdn/) are two different origins, as at Hugging Face; the Location is absolute and logged."""
    from static_hosts import start_model_host

    (tmp_path / "m.onnx").write_bytes(bytes(range(256)) * 4)
    h = start_model_host(tmp_path, COMMIT)
    c = h.cdn
    try:
        assert c is not None and c.origin != h.origin
        st, hd, _ = _req(h.origin, "GET", f"/resolve/{COMMIT}/m.onnx")
        assert st == 302 and hd["access-control-allow-origin"] == "*" and hd["cache-control"] == "no-store"
        loc = hd["location"]
        assert loc.startswith(c.origin + "/cdn/m.onnx?Expires=") and "Signature=" in loc
        path = loc[len(c.origin):]
        st, hd, body = _req(c.origin, "GET", path)
        assert st == 200 and len(body) == 1024 and hd["accept-ranges"] == "bytes" and hd["access-control-allow-origin"] == "*"
        st, hd, body = _req(c.origin, "GET", path, {"Range": "bytes=10-19"})
        assert st == 206 and body == bytes(range(10, 20)) and hd["content-range"] == "bytes 10-19/1024"
        st, hd, body = _req(c.origin, "HEAD", path)
        assert st == 200 and body == b"" and hd["content-length"] == "1024"
        assert _req(c.origin, "GET", "/cdn/m.onnx")[0] == 403  # unsigned
        assert _req(h.origin, "GET", path)[0] == 404 and _req(c.origin, "GET", f"/resolve/{COMMIT}/m.onnx")[0] == 404  # each hop only on its own origin
        assert _req(h.origin, "GET", f"/resolve/{'1' * 40}/m.onnx")[0] == 404  # another commit
        assert _req(h.origin, "GET", f"/resolve/{COMMIT}/../x")[0] == 404
        st, _, _ = _req(c.origin, "POST", "/cdn/m.onnx?Expires=1&Signature=a", body=b"secret")
        assert st == 405
        assert _req(c.origin, "OPTIONS", "/cdn/m.onnx")[0] == 405
        first, signed = h.log.snapshot(), c.log.snapshot()
        assert [e.method for e in first] == ["GET", "GET", "GET", "GET"] and first[0].status == 302 and first[0].location == loc
        assert [e.method for e in signed] == ["GET", "GET", "HEAD", "GET", "GET", "POST", "OPTIONS"]
        assert signed[1].headers["range"] == "bytes=10-19" and signed[5].body == b"secret" and {e.host for e in first} == {"model"} and {e.host for e in signed} == {"cdn"}
    finally:
        h.stop()


def test_the_page_host_applies_the_headers_file_and_serves_only_files(tmp_path):
    from static_hosts import start_page_host

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<p>hi</p>")
    (tmp_path / "assets" / "a.js").write_text("1")
    (tmp_path / "_headers").write_text(HEADERS)
    h = start_page_host(tmp_path)
    try:
        st, hd, body = _req(h.origin, "GET", "/")
        assert st == 200 and body == b"<p>hi</p>" and hd["a"] == "1" and hd["csp"] == "x; y" and "cache-control" not in hd
        st, hd, _ = _req(h.origin, "GET", "/assets/a.js")
        assert st == 200 and hd["a"] == "1, 2" and hd["cache-control"] == "immutable"
        assert _req(h.origin, "GET", "/nope")[0] == 404 and _req(h.origin, "GET", "/../etc/passwd")[0] == 404
        assert _req(h.origin, "POST", "/", body=b"x")[0] == 405
        assert [(e.method, e.path, e.status) for e in h.log.snapshot()] == [("GET", "/", 200), ("GET", "/assets/a.js", 200), ("GET", "/nope", 404), ("GET", "/../etc/passwd", 404), ("POST", "/", 405)]
    finally:
        h.stop()
