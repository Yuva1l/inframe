"""The built UI that ships inside the package (docs/T9A_DESIGN.md 2.2): src/inframe/web/dist holds index.html, assets and the
licence notices, refers only to files that are there, and mentions no host outside a short allowlist (the page may talk to
nothing but the loopback server). Skipped while dist is absent unless INFRAME_REQUIRE_DIST=1 (parts C1-C3 set it).
The scan itself is tested on small folders that are built here, so it is checked even while dist is absent."""
import hashlib
import os
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "src" / "inframe" / "web" / "dist"
PAGES = ROOT / "web" / "pages-dist"
SVG_NS = "www.w3.org/2000/svg"
URL = re.compile(r"https?://[^\s\"'<>)\];]+")  # `;` ends a URL, so `connect-src https://*.hf.co;` yields the host `*.hf.co`
# The one address the page links to on purpose: the "Source code" link on the open screen (web/src/model/about.ts). It is a link the visitor follows, not a
# request the page makes (the CSP's connect-src does not list it). Any other URL of the same host is still reported. Allowed in the built scripts only.
SOURCE_LINK = "https://github.com/Yuva1l/inframe"
REF = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""")
MEDIA = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".webm", ".m4v", ".mkv", ".gif", ".mp3", ".wav"}


PAGES_FILE_LIMIT = 25 * 1024 * 1024  # Cloudflare static assets: bytes per file
PAGES_FILES_MAX = 20_000  # Cloudflare static assets: files per Worker version (free plan)


def host_allowed(host: str | None, allow_hosts) -> bool:
    """`host` is one of allow_hosts, or under an entry written `*.example.com` (at least one label in front)."""
    if not host:
        return False
    for a in allow_hosts:
        if host == a or (a.startswith("*.") and host.endswith(a[1:]) and host != a[2:]):
            return True
    return False


def manifest_source_hosts(dist: Path) -> set[str]:
    """Hosts named in the `source` fields of models/manifest.json (provenance text, like the licence notices)."""
    f = dist / "models" / "manifest.json"
    if not f.is_file():
        return set()
    hosts: set[str] = set()
    for entry in json.loads(f.read_text()).get("files", {}).values():
        for url in URL.findall(str(entry.get("source", ""))):
            hosts.add(urlsplit(url).hostname or "")
    return hosts - {""}


def connect_hosts(dist: Path) -> set[str]:
    """Hosts in the connect-src of a built _headers file (a wildcard stays `*.host`)."""
    f = dist / "_headers"
    if not f.is_file():
        return set()
    m = re.search(r"connect-src ([^;\n]*)", f.read_text())
    return {urlsplit(t).hostname or "" for t in (m.group(1).split() if m else []) if "://" in t} - {""}


def check_pages_limits(dist: Path) -> list[str]:
    """Cloudflare static-assets limits: every file under 25 MiB, fewer than 20,000 files."""
    files = [p for p in dist.rglob("*") if p.is_file()]
    problems = [f"{p.relative_to(dist)}: {p.stat().st_size} bytes is over the 25 MiB static-assets limit" for p in files if p.stat().st_size > PAGES_FILE_LIMIT]
    if len(files) >= PAGES_FILES_MAX:
        problems.append(f"{len(files)} files; static assets allow fewer than {PAGES_FILES_MAX}")
    return problems


def scan_dist(dist: Path, allow_hosts=()) -> list[str]:
    """Problems found in a built UI folder (empty list = fine). `allow_hosts` (hosts, or `*.host`) are the model host of the Pages build: addresses
    on them are not problems. The hosts in the `source` fields of models/manifest.json are provenance text and are let through in that one file only."""
    problems: list[str] = []
    allow = set(allow_hosts)
    sources = manifest_source_hosts(dist)
    index = dist / "index.html"
    if not index.is_file():
        return [f"{dist}: index.html is missing"]
    if not (dist / "THIRD_PARTY_LICENSES.txt").is_file():
        problems.append("THIRD_PARTY_LICENSES.txt is missing")
    for ref in REF.findall(index.read_text(errors="replace")):
        if re.match(r"^[a-z][a-z0-9+.-]*:|^//", ref):
            problems.append(f"index.html refers to an outside address: {ref}")
            continue
        target = (dist / ref.split("?")[0].split("#")[0]).resolve()
        if not target.is_file() or not target.is_relative_to(dist.resolve()):
            problems.append(f"index.html refers to a file that is not in dist: {ref}")
        elif target.parent.name != "assets" and target.name != "favicon.svg":
            problems.append(f"index.html refers to a file outside assets/: {ref}")
    for f in sorted(p for p in dist.rglob("*") if p.is_file()):
        rel = f.relative_to(dist)
        if f.suffix.lower() in MEDIA:
            problems.append(f"{rel}: media file in dist")
        text = f.read_text(errors="replace")
        for url in URL.findall(text):
            if SVG_NS in url:
                continue
            if f.name == "THIRD_PARTY_LICENSES.txt":
                continue  # licence notices carry their project and licence URLs
            if host_allowed(urlsplit(url).hostname, allow):
                continue
            if url == SOURCE_LINK and rel.parent.as_posix() == "assets" and f.suffix == ".js":
                continue
            if rel.as_posix() == "models/manifest.json" and urlsplit(url).hostname in sources:
                continue
            problems.append(f"{rel}: off-origin URL {url[:80]}")
    return problems


def test_dist_is_complete_and_makes_no_off_origin_reference():
    if not (DIST / "index.html").is_file():
        if os.environ.get("INFRAME_REQUIRE_DIST") == "1":
            pytest.fail(f"INFRAME_REQUIRE_DIST=1 but {DIST} has no index.html")
        pytest.skip("src/inframe/web/dist is not built (part C1 commits it)")
    assert scan_dist(DIST) == []
    assert (DIST / "assets").is_dir() and any((DIST / "assets").glob("*.js"))


def test_pages_dist_makes_no_off_origin_reference():
    if not (PAGES / "index.html").is_file():
        pytest.skip("web/pages-dist is not built (T9c)")
    # the model host of a Pages build is whatever its _headers connect-src names; nothing else may appear
    assert scan_dist(PAGES, connect_hosts(PAGES)) == []
    assert check_pages_limits(PAGES) == []


def build(root: Path, index: str, files: dict[str, str]) -> Path:
    d = root / "dist"
    d.mkdir()
    (d / "index.html").write_text(index)
    for name, text in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    return d


GOOD_INDEX = '<!doctype html><link rel="icon" href="./favicon.svg"><link rel="stylesheet" href="./assets/a.css"><script type="module" src="./assets/a.js"></script>'


def test_scan_accepts_a_clean_dist(tmp_path):
    d = build(tmp_path, GOOD_INDEX, {"assets/a.js": "var x=1", "assets/a.css": "body{}", "favicon.svg": '<svg xmlns="http://www.w3.org/2000/svg"/>',
                                     "THIRD_PARTY_LICENSES.txt": "preact MIT https://github.com/preactjs/preact"})
    assert scan_dist(d) == []


def test_scan_finds_problems(tmp_path):
    d = build(tmp_path, GOOD_INDEX + '<script src="https://cdn.example/x.js"></script><script src="./assets/missing.js"></script><img src="./logo.svg">',
              {"assets/a.js": 'fetch("https://api.example.com/x")', "assets/a.css": "body{}", "favicon.svg": "<svg/>",
               "logo.svg": "<svg/>", "assets/pic.png": "x"})
    found = " | ".join(scan_dist(d))
    assert "THIRD_PARTY_LICENSES.txt is missing" in found
    assert "outside address: https://cdn.example/x.js" in found
    assert "not in dist: ./assets/missing.js" in found
    assert "outside assets/: ./logo.svg" in found
    assert "a.js: off-origin URL https://api.example.com/x" in found
    assert "pic.png: media file in dist" in found
    assert scan_dist(tmp_path / "nowhere")[0].endswith("index.html is missing")


WEB = ROOT / "web"


def source_hash(web: Path) -> str:
    """Fingerprint of everything the UI build reads; the same algorithm as sourceHash() in web/vite.config.ts."""
    files = [str(p.relative_to(web).as_posix()) for d in ("src", "public") for p in (web / d).rglob("*") if p.is_file() and p.name != ".DS_Store"]
    files += ["index.html", "package-lock.json", "vite.config.ts", "tsconfig.json"]
    lines = "".join(f"{f}\0{hashlib.sha256((web / f).read_bytes()).hexdigest()}\n" for f in sorted(files, key=lambda f: f.encode()))
    return hashlib.sha256(lines.encode()).hexdigest()


def test_dist_is_built_from_the_current_web_sources():
    """The committed bundle must match web/src: a source change without `cd web && npm run build` fails here."""
    if not (DIST / "index.html").is_file() or not (WEB / "src").is_dir():
        pytest.skip("dist or web sources are absent")
    stamp = DIST / "SOURCE_HASH.txt"
    assert stamp.is_file(), "dist has no SOURCE_HASH.txt: run `cd web && npm run build` and commit src/inframe/web/dist"
    assert stamp.read_text().strip() == source_hash(WEB), "dist is older than web/: run `cd web && npm run build` and commit src/inframe/web/dist"


def test_shipped_notices_cover_the_pillow_port_and_the_inlined_onnxruntime_code():
    """embed.ts is a line-by-line port of Pillow's Resample.c (HPND / MIT-CMU: the notice must travel with it), and onnxruntime-common is compiled
    into the onnxruntime-web bundle. Both notices are in the shipped file; the packages ort only uses in other builds are not claimed."""
    text = (DIST / "THIRD_PARTY_LICENSES.txt").read_text()
    assert "Name: Pillow" in text and "Secret Labs AB" in text and "Fredrik Lundh" in text
    assert "provided that the above copyright notice appears in all copies" in text
    assert "Name: onnxruntime-common" in text and "Copyright (c) Microsoft Corporation" in text
    for shipped in ("preact", "mp4box", "client-zip", "onnxruntime-web"):
        assert f"Name: {shipped}" in text
    head = (ROOT / "web" / "src" / "browser" / "embed.ts").read_text().split("export const CROP_PAD")[0]
    assert "Pillow" in head and "HPND" in head and "Secret Labs AB" in head and "THIRD_PARTY_LICENSES.txt" in head
    # the wasm runtime the page downloads statically contains abseil, onnx (Apache-2.0), protobuf (BSD-3-Clause) and Eigen (MPL-2.0): their notices ship too
    assert "Copyright 2008 Google Inc." in text and "Redistributions in binary form must reproduce the above" in text
    assert "Copyright (c) Facebook, Inc. and Microsoft Corporation" in text and "abseil-cpp" in text and "Mozilla Public License Version 2.0" in text
    bundle_notices = (ROOT / "tools" / "hf_files" / "THIRD_PARTY_NOTICES").read_text().split("_____", 1)[1].strip()
    assert bundle_notices in text, "the model repo's THIRD_PARTY_NOTICES and the page's THIRD_PARTY_LICENSES.txt must carry the same library texts"
    readme = (ROOT / "README.md").read_text()
    assert "onnxruntime-common" in readme and "Resample.c" in readme and "protobufjs" in readme
    # what the README says is not shipped really is not in the built bundles
    for js in (DIST / "assets").glob("*.js"):
        code = js.read_text(errors="replace")
        for package, markers in NOT_SHIPPED.items():
            for marker in markers:
                assert marker not in code, f"{js.name} holds {marker} (from {package}): list the package in THIRD_PARTY_LICENSES.txt"


# Strings a minifier keeps (property names and message text, not exported const names), one set per package the notices say is not shipped.
NOT_SHIPPED = {
    "flatbuffers": ("forceDefaults", "getBufferIdentifier", "addFieldInt32"),
    "guid-typescript": ("isGuid", "00000000-0000-0000-0000-000000000000"),
    "long": ("__isLong__",),
    "platform": ("Opera Mini", "Maxthon"),
    "protobufjs": ("illegal buffer", "invalid encoding"),
}
NOT_SHIPPED_ENTRY = {
    "flatbuffers": 'import {Builder} from "flatbuffers"; console.log(Builder);',
    "guid-typescript": 'import {Guid} from "guid-typescript"; console.log(Guid.create());',
    "long": 'import L from "long"; console.log(L.fromNumber(1));',
    "platform": 'import p from "platform"; console.log(p.os);',
    "protobufjs": 'import p from "protobufjs/minimal.js"; console.log(p.Reader);',
}


@pytest.mark.parametrize("package", sorted(NOT_SHIPPED))
def test_not_shipped_markers_survive_a_minified_bundle_of_their_package(package, tmp_path):
    """The guard above only works if each marker is still in the package's code after bundling and minifying: bundle the package alone and look."""
    import subprocess

    esbuild = ROOT / "web" / "node_modules" / ".bin" / "esbuild"
    if not esbuild.is_file() or not (ROOT / "web" / "node_modules" / package).is_dir():
        pytest.skip("web/node_modules is not installed")
    entry = tmp_path / "entry.js"
    entry.write_text(NOT_SHIPPED_ENTRY[package])
    out = tmp_path / "out.js"
    subprocess.run(
        [str(esbuild), str(entry), "--bundle", "--minify", "--format=esm", f"--outfile={out}", "--log-level=error"],
        check=True, env={**os.environ, "NODE_PATH": str(ROOT / "web" / "node_modules")}, capture_output=True,
    )
    code = out.read_text()
    missing = [m for m in NOT_SHIPPED[package] if m not in code]
    assert not missing, f"{package}: markers {missing} do not survive bundling"


def test_scan_allows_only_the_given_hosts_and_the_manifest_sources(tmp_path):
    files = {"assets/a.js": 'var b="https://models.example/o/m/resolve/abc/";var c="http://127.0.0.1:8123/x";var d="https://us.cdn.hf.co/f";var e=fetch("https://cas-server.xethub.hf.co/v1/xorbs/default/0",{method:"POST",mode:"no-cors",body:b})', "assets/a.css": "body{}",
             "favicon.svg": "<svg/>", "THIRD_PARTY_LICENSES.txt": "x",
             "models/manifest.json": json.dumps({"format": 1, "files": {"m.onnx": {"source": "https://huggingface.co/facebook/dinov2-small (opset 17)"}}}),
             "_headers": "/*\n  Content-Security-Policy: connect-src 'self' https://models.example https://*.cdn.hf.co\n"}
    d = build(tmp_path, GOOD_INDEX, files)
    found = " | ".join(scan_dist(d))
    assert "off-origin URL https://models.example/o/m/resolve/abc/" in found and "http://127.0.0.1:8123/x" in found and "us.cdn.hf.co" in found and "cas-server.xethub.hf.co" in found
    assert "huggingface.co/facebook" not in found  # a manifest source
    assert connect_hosts(d) == {"models.example", "*.cdn.hf.co"}
    allowed = {"models.example", "127.0.0.1", "*.cdn.hf.co"}
    # the Xet upload host is not under the CDN wildcard, so a hard-coded upload to it is still reported
    assert [p for p in scan_dist(d, allowed) if "cas-server.xethub.hf.co" in p] and not [p for p in scan_dist(d, allowed) if "us.cdn.hf.co" in p]
    assert any("off-origin" in p for p in scan_dist(d, {"models.example", "127.0.0.1"}))  # the wildcard is not implied
    assert not host_allowed("cdn.hf.co", {"*.cdn.hf.co"}) and not host_allowed("evilcdn.hf.co", {"*.cdn.hf.co"}) and host_allowed("a.b.cdn.hf.co", {"*.cdn.hf.co"})
    assert not host_allowed("cas-server.xethub.hf.co", {"*.cdn.hf.co"})


def test_pages_limits(tmp_path, monkeypatch):
    d = build(tmp_path, GOOD_INDEX, {"assets/a.js": "x"})
    assert check_pages_limits(d) == []
    big = d / "assets" / "big.bin"
    with big.open("wb") as f:
        f.truncate(PAGES_FILE_LIMIT + 1)  # sparse
    assert "big.bin" in " ".join(check_pages_limits(d))
    big.unlink()
    monkeypatch.setattr(sys.modules[__name__], "PAGES_FILES_MAX", 2)
    assert "files; static assets allow fewer than 2" in " ".join(check_pages_limits(d))


def test_manifest_source_hosts_are_let_through_in_the_manifest_only(tmp_path):
    manifest = json.dumps({"format": 1, "files": {"m.onnx": {"source": "https://huggingface.co/a/b https://github.com/x/y https://www.npmjs.com/package/z"}}})
    js = 'fetch("https://github.com/x/upload",{method:"POST"});fetch("https://www.npmjs.com/y");fetch("https://huggingface.co/api/upload")'
    d = build(tmp_path, GOOD_INDEX, {"assets/a.js": js, "assets/a.css": "body{}", "favicon.svg": "<svg/>", "THIRD_PARTY_LICENSES.txt": "x", "models/manifest.json": manifest})
    found = scan_dist(d)
    assert len([p for p in found if p.startswith("assets/a.js: off-origin")]) == 3, found
    assert not any(p.startswith("models/manifest.json") for p in found), found
    # only an allowed host (the connect-src of the build) lets an address through in the bundle
    assert len(scan_dist(d, {"huggingface.co"})) == 2


def test_the_source_code_link_is_the_one_github_address_a_built_script_may_carry(tmp_path):
    js = f'a.href="{SOURCE_LINK}";'
    files = {"assets/a.js": js, "assets/a.css": "body{}", "favicon.svg": "<svg/>", "THIRD_PARTY_LICENSES.txt": "x"}
    def fresh(name: str) -> Path:
        (tmp_path / name).mkdir()
        return tmp_path / name

    assert scan_dist(build(fresh("ok"), GOOD_INDEX, files)) == []
    # another address on the same host, a longer path, or the link in another kind of file is still reported
    for other in (SOURCE_LINK + "/issues", SOURCE_LINK.replace("inframe", "other"), "https://github.com/x/upload"):
        d = build(fresh(other.rsplit("/", 1)[-1] + str(len(other))), GOOD_INDEX, {**files, "assets/a.js": f'fetch("{other}")'})
        assert any("off-origin URL" in p for p in scan_dist(d)), other
    d = build(fresh("css"), GOOD_INDEX, {**files, "assets/a.css": f"/* {SOURCE_LINK} */"})
    assert any("a.css: off-origin" in p for p in scan_dist(d))
