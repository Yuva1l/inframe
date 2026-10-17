"""Browser tests (docs/T9B_DESIGN.md section 8). They need Chrome, the exported models and `--group browser`, so they are skipped unless

    INFRAME_BROWSER_TESTS=1 INFRAME_MODELS=<folder with onnx/> uv run --group browser pytest -q tests/browser

Only synthetic input is used (LCG noise, ffmpeg testsrc2); no footage is read and no picture is looked at. The page under test is
web/selftest (the real BrowserEngine, Worker and model code with a thin window.__t), served by the real `inframe serve` server so the
real /models route, Range handling and CSP are what the browser meets. Every request the browser makes and the server answers is logged.
Safari (INFRAME_SAFARI=1) opens the same page in the real Safari, which POSTs its answer back to the server.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from inframe import serve
from inframe.serve_http import Response, Route, bytes_response, error, json_response
from serve_helpers import start_server

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
CHROME_ARGS = ["--enable-unsafe-webgpu", "--enable-features=WebGPU,Vulkan", "--ignore-gpu-blocklist"]


def pytest_collection_modifyitems(items):
    if os.environ.get("INFRAME_BROWSER_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="browser tests: set INFRAME_BROWSER_TESTS=1 (and INFRAME_MODELS, uv run --group browser)")
    for it in items:
        if "tests/browser" in str(it.fspath).replace(os.sep, "/"):
            it.add_marker(skip)


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-nostats", "-loglevel", "error", *args], check=True)


class Site:
    """The running page server, its request log and the results Safari posts back."""

    def __init__(self, app, base, data_dir, dist):
        self.app, self.base, self.data_dir, self.dist = app, base, data_dir, dist
        self.log: list[tuple[str, str]] = []  # (method, path) of every request the server answered
        self.posted: list[dict] = []
        self.posted_event = threading.Event()

    @property
    def url(self) -> str:
        return f"{self.base}/#token={self.app.token}"

    def safari_url(self) -> str:
        return f"{self.base}/#token={self.app.token}&auto=parity"


@pytest.fixture(scope="session")
def models_dir():
    base = os.environ.get("INFRAME_MODELS")
    d = Path(base).expanduser() / "onnx" if base else None
    if d is None or not (d / "manifest.json").is_file():
        pytest.skip("INFRAME_MODELS/onnx/manifest.json is missing: run tools/export_onnx.py in its bench venv")
    return d


@pytest.fixture(scope="session")
def site(tmp_path_factory, models_dir):
    if not shutil.which("npm"):
        pytest.skip("npm is not installed")
    root = tmp_path_factory.mktemp("selftest")
    out = root / "dist"
    env = dict(os.environ, SELFTEST_OUT=str(out))
    r = subprocess.run(["npm", "exec", "--", "vite", "build", "--config", "selftest/vite.config.ts", "--logLevel", "error"], cwd=WEB, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    data = root / "data"
    data.mkdir()
    holder: dict = {}

    def get_data(app, req):
        f = (data / req.params["name"]).resolve()
        if data.resolve() not in f.parents or not f.is_file():
            return error(404, "no such data")
        return bytes_response(f.read_bytes(), "application/octet-stream")

    def post_result(app, req):
        s: Site = holder["site"]
        s.posted.append(req.json)
        s.posted_event.set()
        return json_response({"ok": True})

    os.environ["INFRAME_MODELS"] = str(models_dir.parent)
    base, app, stop = start_server(root, dist_dir=out, extra_routes=[Route("GET", r"/api/selftest/data/(?P<name>[A-Za-z0-9_.-]+)", get_data), Route("POST", r"/api/selftest/result", post_result)])
    s = Site(app, base, data, out)
    holder["site"] = s
    handler_cls = app.httpd.RequestHandlerClass
    handler_cls.log_request = lambda self, code="-", size="-": s.log.append((self.command, self.path))  # one line per answered request
    yield s
    stop()


@pytest.fixture(scope="session")
def chrome_browser():
    """Google Chrome through Playwright, shared by every fixture that needs a browser (one Playwright per process)."""
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=os.environ.get("HEADED") != "1", args=CHROME_ARGS)
        except Exception as e:  # Chrome is not installed
            pytest.skip(f"Google Chrome could not be started: {e}")
        yield b
        b.close()


@pytest.fixture(scope="session")
def chrome(site, chrome_browser):
    ctx = chrome_browser.new_context()
    ctx.requests = []  # every request any page or worker of the context makes
    ctx.on("request", lambda r: ctx.requests.append((r.method, r.url)))
    ctx.problems = []
    yield ctx
    ctx.close()


def open_page(ctx, site: Site):
    page = ctx.new_page()
    page.on("pageerror", lambda e: ctx.problems.append(f"pageerror: {e}"))
    # onnxruntime-web prints its warnings through console.error; they are logs, not errors
    page.on("console", lambda m: ctx.problems.append(f"console.{m.type}: {m.text[:300]}") if m.type == "error" and not re.search(r"\[[VIWEF]:onnxruntime", m.text) else None)
    page.goto(site.url)
    page.wait_for_function("window.__t && window.__t.ready", timeout=30000)
    return page


@pytest.fixture(scope="session")
def models_cached(chrome, site):
    """The models downloaded into this Chrome context's Cache Storage (once per run of the tests)."""
    page = open_page(chrome, site)
    res = page.evaluate("window.__t.download()", )
    page.close()
    assert res["allReady"], res
    return res


# ---------------------------------------------------------------- inputs shared by the tests
def lcg(n: int) -> np.ndarray:
    """The page's generator (web/selftest/main.ts): s = s * 1664525 + 1013904223 mod 2^32, scaled to -1..1."""
    s, a = 12345, np.empty(n, np.float32)
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        a[i] = s / 4294967296 * 2 - 1
    return a


@pytest.fixture(scope="session")
def testsrc2_frame(tmp_path_factory):
    """One ffmpeg testsrc2 frame (synthetic), as an RGB array."""
    from PIL import Image

    d = tmp_path_factory.mktemp("frame")
    png = d / "f.png"
    _ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=1:duration=1", "-frames:v", "1", str(png))
    return np.asarray(Image.open(png).convert("RGB"))


@pytest.fixture(scope="session")
def hevc20(tmp_path_factory):
    """20 s, 30 fps, 1280x720 testsrc2 as HEVC (hvc1) in an mp4, a keyframe every second."""
    d = tmp_path_factory.mktemp("hevc")
    p = d / "synthetic_hevc_20s.mp4"
    base = ["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-g", "30", "-movflags", "+faststart"]
    try:
        _ff(*base, "-c:v", "hevc_videotoolbox", "-b:v", "4M", str(p))
    except subprocess.CalledProcessError:
        _ff(*base, "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "keyint=30:min-keyint=30:log-level=error", str(p))
    return p


@pytest.fixture(scope="session")
def open_gop6(tmp_path_factory):
    """(mp4 path, the pictures as (pts, is_keyframe) in decode order): 6 s, 30 fps, 640x360 testsrc2 as HEVC (hvc1) with libx265 open GOPs (keyint 30, 4 B-frames),
    so the pictures shown just before each later keyframe are leading pictures that a decoder started at that keyframe drops. Synthetic; nothing is looked at."""
    d = tmp_path_factory.mktemp("opengop")
    p = d / "open_gop_6s.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-c:v", "libx265", "-preset", "ultrafast",
        "-x265-params", "open-gop=1:keyint=30:min-keyint=30:bframes=4:scenecut=0:log-level=error", "-movflags", "+faststart", str(p))
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(p)], check=True, capture_output=True, text=True).stdout
    pics = [(float(a), "K" in b) for a, b in (line.split(",")[:2] for line in out.splitlines() if line)]
    return p, pics


# ---------------------------------------------------------------- an openly licensed composite with known person-free gaps
# Shots of splice.py (tools/splice.py): people from Pexels clips, gaps from two Wikimedia Commons dog clips (docs/TESTDATA.md). No personal footage is used.
# Every tile that holds a person has role "target", so the truth (splice.truth_for) is "a person is in frame" second by second.
GAPS_RECIPE = [
    ("full", 6.4, [("lone1", "target", 3.5)]),  # a person from 2.9 s to the end of the clip (measured by the detector, not looked at)
    ("full", 8, [("dogs2", "other", 0)]),
    ("full", 12, [("lone2", "target", 0)]),
    ("full", 8, [("dogs1", "other", 20)]),
    ("full", 8, [("pair1", "target", 0)]),
    ("side", 8, [("dogs2", "other", 12), ("pair2", "target", 0)]),
    ("full", 6, [("dogs1", "other", 30)]),
    ("full", 8, [("pair2", "target", 0)]),
]


@pytest.fixture(scope="session")
def gaps_composite(tmp_path_factory):
    """(mp4 path, truth dict): the shots above joined by ffmpeg, written to a temp folder; skipped when a source clip is not on this Mac."""
    sys_path = str(ROOT / "tools")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import splice

    recipe = [(layout, secs, [splice.T(src, role, ss) for src, role, ss in tiles]) for layout, secs, tiles in GAPS_RECIPE]
    missing = [t["src"] for _, _, tiles in recipe for t in tiles if not splice.source_path(t["src"]).is_file()]
    if missing:
        pytest.skip(f"source clips missing (docs/TESTDATA.md): {sorted(set(missing))}")
    d = tmp_path_factory.mktemp("gaps")
    parts = []
    for i, (layout, secs, tiles) in enumerate(recipe):
        seg = d / f"seg{i}.mp4"
        cmd = splice.segment_command(layout, secs, tiles, seg)
        subprocess.run([cmd[0], "-nostats", *cmd[1:]], check=True)
        parts.append(seg)
    lst = d / "concat.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    mp4 = d / "open_gaps.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-nostats", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(mp4)], check=True)
    truth = splice.truth_for("open_gaps", recipe)
    truth["measured_duration"] = round(splice.ffprobe_duration(mp4), 3)
    return mp4, truth


# ---------------------------------------------------------------- the Pages build on two local origins (docs/T9C_DESIGN.md section 3)
PAGES_COMMIT = "c0ffee" * 6 + "c0ff"  # the fake 40-hex commit of the model host's /resolve/ path


@dataclass
class PagesSite:
    dist: Path
    page: object  # static_hosts.Host serving pages-dist with its _headers
    model: object  # static_hosts.Host answering /resolve/ like Hugging Face (302 to the cdn host); model.cdn is the third origin that serves /cdn/
    commit: str
    names: list[str]  # the files of the page's own manifest
    page_files: set[str]  # the paths that exist in pages-dist ("/" for the index)

    @property
    def cdn(self):
        return self.model.cdn

    def model_requests(self) -> int:
        """Requests that reached either model-side host (the first hop and the signed one)."""
        return len(self.model.log) + len(self.cdn.log)

    @property
    def connect(self) -> list[str]:
        """The connect-src entries the page was built with: the pinned path on the first-hop host and the bare redirect origin (the shape of web/pages/model-host.json)."""
        return [f"{self.model.origin}/resolve/{self.commit}/", self.cdn.origin]

    @property
    def model_paths(self) -> set[str]:
        return {p for n in self.names for p in (f"/resolve/{self.commit}/{n}", f"/cdn/{n}")}


@pytest.fixture(scope="session")
def pages_site(tmp_path_factory, models_dir):
    """`vite build --mode pages` into a temp folder with the model base on a second local origin, then both hosts running. Nothing is deployed or uploaded."""
    from static_hosts import start_model_host, start_page_host

    vite = WEB / "node_modules" / ".bin" / "vite"
    if not vite.is_file():
        pytest.skip("web/node_modules is not installed (npm ci in web/)")
    model = start_model_host(models_dir, PAGES_COMMIT)
    out = tmp_path_factory.mktemp("pages") / "pages-dist"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VITE_", "INFRAME_PAGES_"))}
    env.update(VITE_MODEL_BASE=f"{model.origin}/resolve/{PAGES_COMMIT}/", INFRAME_PAGES_CONNECT=",".join([f"{model.origin}/resolve/{PAGES_COMMIT}/", model.cdn.origin]), INFRAME_PAGES_ALLOW_LOCAL="1")
    r = subprocess.run([str(vite), "build", "--mode", "pages", "--outDir", str(out), "--logLevel", "error"], cwd=WEB, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        model.stop()
        pytest.fail("pages build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    page = start_page_host(out)
    names = sorted(json.loads((out / "models" / "manifest.json").read_text())["files"])
    files = {"/" + p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} | {"/"}
    yield PagesSite(out, page, model, PAGES_COMMIT, names, files)
    page.stop()
    model.stop()


@dataclass
class Canary:
    path: Path
    name: str  # the file name (also a secret)
    token: str  # 32 hex, random per run, in the title metadata and in a free box
    secrets: dict[str, bytes]  # label -> bytes that must never appear in a request
    tap_time: float = 0.0  # a moment at which the detector (python onnxruntime, same ONNX file) finds a person in this very file
    tap_box: tuple[float, float, float, float] = (0, 0, 0, 0)  # that person's box, x0 y0 x1 y1 normalised: the page's tap lands on it and DINOv2 embeds the crop


def _person_box(clip: Path, models_dir: Path) -> tuple[float, tuple[float, float, float, float]]:
    """(time, box) of the best person the RF-DETR small ONNX file finds in `clip` at 3..7 s, by python onnxruntime on the CPU: numbers only, no picture is looked at."""
    ort = pytest.importorskip("onnxruntime")
    from inframe import detect

    sess = ort.InferenceSession(str(models_dir / "rfdetr_small_v1.onnx"), providers=["CPUExecutionProvider"])
    best = (0.0, 0.0, (0.0, 0.0, 0.0, 0.0))
    for t in (3, 4, 5, 6, 7):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-nostats", "-ss", str(t), "-i", str(clip), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        frame = np.frombuffer(raw, np.uint8).reshape(360, 640, 3)
        boxes, logits = detect._split_out(sess.run(None, {sess.get_inputs()[0].name: detect.preprocess([frame], 512)}))
        for d in detect.postprocess(boxes, logits, 1)[0]:
            area = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
            if d.score >= 0.7 and area >= 0.03 and d.score > best[1]:
                best = (float(t), d.score, tuple(round(v, 4) for v in d.box))
    if best[1] == 0.0:
        pytest.skip("the detector finds no person with score >= 0.7 in the person clip: the tap run could not embed one")
    return best[0], best[2]


@pytest.fixture(scope="session")
def canary_clip(tmp_path_factory, models_dir) -> Canary:
    """A 10 s clip (HEVC, 640x360) that carries a canary: a random 32-hex token in its title metadata and in an appended `free` box, with a random file
    name. The page must never send the token, the name or a run of the clip's bytes anywhere. The pictures are an openly licensed Pexels shot with one
    person in it (splice.py's "lone2", docs/TESTDATA.md), so the detector finds a person and the tap run crops and embeds them (DINOv2); skipped when that
    source clip is not on this Mac. Nothing is looked at: the person's box comes from the detector's numbers."""
    import secrets as _secrets
    import struct

    sys_path = str(ROOT / "tools")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import splice

    src = splice.source_path("lone2")
    if not src.is_file():
        pytest.skip(f"source clip missing (docs/TESTDATA.md): {src}")
    token = _secrets.token_hex(16)
    name = f"PRIVATE-{_secrets.token_hex(4)}-session.mp4"
    d = tmp_path_factory.mktemp("canary")
    p = d / name
    base = ["-i", str(src), "-t", "10", "-an", "-vf", "scale=640:360,fps=30", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-g", "30", "-metadata", f"title={token}", "-movflags", "+faststart"]
    try:
        _ff(*base, "-c:v", "hevc_videotoolbox", "-b:v", "2M", str(p))
    except subprocess.CalledProcessError:
        _ff(*base, "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "keyint=30:min-keyint=30:log-level=error", str(p))
    tap_time, tap_box = _person_box(p, models_dir)
    payload = b"canary:" + token.encode() + b":" + _secrets.token_bytes(128)
    with p.open("ab") as f:  # a free box after the last box: a valid mp4 box the demuxer skips
        f.write(struct.pack(">I4s", 8 + len(payload), b"free") + payload)
    data = p.read_bytes()
    assert token.encode() in data[: 1 << 16]  # the title is in the moov box (faststart puts it first)
    runs = {f"clip bytes at {off}": data[off : off + 64] for off in (len(data) // 4, len(data) // 2, 3 * len(data) // 4)}
    return Canary(p, name, token, {"canary": token.encode(), "free box": payload, "file name": name.encode(), **runs}, tap_time, tap_box)
