#!/usr/bin/env python3
"""Measure how long a video the browser engine can take (docs/T9B_DESIGN.md section 7, docs/BENCHMARK.md section 23).

    INFRAME_MODELS=<folder with onnx/> uv run --group browser python tools/browser_limits.py [--minutes 5 20 60]

For each length it builds a synthetic 1080p30 HEVC (testsrc2 and a sine, AAC; one minute encoded at a phone's bitrate, then repeated with
stream copy), opens it in the real page (web/selftest: BrowserEngine, Worker, WebCodecs, onnxruntime-web WebGPU) in the installed Google Chrome
through Playwright, runs balanced any-person (or `--preset fast`, keyframes only), and prints one JSON line per length: wall seconds, seconds per video minute, and the peak resident
memory of the whole Chrome process tree (browser, renderer, GPU and utility processes; shared pages are counted once per process, so it is an
upper bound). The source file is deleted after its run. Nothing of the video is looked at: only times, byte counts and process sizes are read.

The machine's load average is printed with every row (record the load with every timing). Each run takes the regress lock
(~/.cache/inframe/regress.lock) so two heavy jobs never overlap. Safari is not driven by this tool; its
numbers would come from the open-and-post-back page of tests/browser.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
LOCK = Path.home() / ".cache/inframe/regress.lock"
OUT = Path.home() / ".cache/inframe/browser-limits"
CHROME_ARGS = ["--enable-unsafe-webgpu", "--enable-features=WebGPU,Vulkan", "--ignore-gpu-blocklist"]
BITRATE = "8M"  # 1080p30 HEVC from an iPhone is about 6 to 10 Mbit/s


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-nostats", *args], check=True)


def make_minute(dst: Path) -> None:
    """One minute of 1080p30 testsrc2 + sine as HEVC (hvc1) at a phone's bitrate, a keyframe every second."""
    base = ["-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=60", "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
            "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-g", "30", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    try:
        ff(*base, "-c:v", "hevc_videotoolbox", "-b:v", BITRATE, str(dst))
    except subprocess.CalledProcessError:
        ff(*base, "-c:v", "libx265", "-preset", "ultrafast", "-b:v", BITRATE, "-x265-params", "keyint=30:min-keyint=30:log-level=error", str(dst))


def make_source(minutes: int, work: Path, minute: Path) -> Path:
    """`minutes` of video: the one-minute file repeated by stream copy (the moov ends up at the end, as a phone writes it)."""
    out = work / f"synthetic_1080p_hevc_{minutes}min.mp4"
    ff("-stream_loop", str(minutes - 1), "-i", str(minute), "-c", "copy", str(out))
    return out


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_frames:format=duration,size",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    return dict(codec=d["streams"][0]["codec_name"], w=d["streams"][0]["width"], h=d["streams"][0]["height"], duration=float(d["format"]["duration"]), size=int(d["format"]["size"]))


def processes() -> list[tuple[int, int, int, str]]:
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)", line)
        if m:
            rows.append((int(m[1]), int(m[2]), int(m[3]), m[4]))
    return rows


def chrome_root(profile_marker: str) -> int | None:
    """The browser process Playwright started: the one with our user-data-dir and no --type= (renderers, GPU and utility processes have one)."""
    for pid, _ppid, _rss, cmd in processes():
        if profile_marker in cmd and "--type=" not in cmd:
            return pid
    return None


def tree_rss_mb(root: int) -> float:
    rows = processes()
    kids: dict[int, list[int]] = {}
    rss = {}
    for pid, ppid, r, _ in rows:
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    total, stack, seen = 0, [root], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        total += rss.get(p, 0)
        stack += kids.get(p, [])
    return total / 1024


class Peak(threading.Thread):
    """Samples the Chrome process tree's resident size every half second and keeps the maximum."""

    def __init__(self, root: int):
        super().__init__(daemon=True)
        self.root, self.peak, self.last, self.stop_ev = root, 0.0, 0.0, threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            self.last = tree_rss_mb(self.root)
            self.peak = max(self.peak, self.last)
            self.stop_ev.wait(0.5)


def build_page(out: Path) -> None:
    env = dict(os.environ, SELFTEST_OUT=str(out))
    r = subprocess.run(["npm", "exec", "--", "vite", "build", "--config", "selftest/vite.config.ts", "--logLevel", "error"], cwd=WEB, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"selftest page did not build:\n{r.stdout}{r.stderr}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=int, nargs="+", default=[5, 20, 60])
    ap.add_argument("--preset", default="balanced", choices=["balanced", "fast"], help="balanced = 5 samples per second, fast = keyframes only")
    ap.add_argument("--keep", action="store_true", help="keep the synthetic files")
    a = ap.parse_args(argv)
    models = Path(os.environ.get("INFRAME_MODELS", ROOT / "bench/detect/out/models")).expanduser()
    if not (models / "onnx/manifest.json").is_file():
        sys.exit("INFRAME_MODELS/onnx/manifest.json is missing: run tools/export_onnx.py in its bench venv, or point INFRAME_MODELS at the folder that holds onnx/")
    os.environ["INFRAME_MODELS"] = str(models)
    sys.path.insert(0, str(ROOT / "src"))
    from playwright.sync_api import sync_playwright

    from inframe import serve

    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="inframe-limits-", dir=OUT))
    page_dir = work / "page"
    build_page(page_dir)
    os.environ["INFRAME_SERVE_STATE"] = str(work / "state")
    os.environ["INFRAME_CACHE"] = str(work / "cache")
    httpd, app = serve.create(0, dist_dir=page_dir)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    minute = work / "minute.mp4"
    make_minute(minute)
    results = OUT / "results.jsonl"
    rc = 0
    try:
        with sync_playwright() as p:
            profile = work / "profile"
            ctx = p.chromium.launch_persistent_context(str(profile), channel="chrome", headless=True, args=CHROME_ARGS)
            try:
                root = None
                for _ in range(40):
                    root = chrome_root(str(profile))
                    if root:
                        break
                    time.sleep(0.25)
                if not root:
                    sys.exit("could not find the Chrome process Playwright started")
                page = ctx.new_page()
                page.goto(f"http://127.0.0.1:{app.port}/#token={app.token}")
                page.wait_for_function("window.__t && window.__t.ready", timeout=30000)
                dl = page.evaluate("window.__t.download()")
                print(f"[limits] models in the page cache: {dl['cachedBytes'] / 1e6:.0f} MB, all ready: {dl['allReady']}", flush=True)
                page.close()
                for minutes in a.minutes:
                    src = make_source(minutes, work, minute)
                    facts = probe(src)
                    LOCK.parent.mkdir(parents=True, exist_ok=True)
                    with open(LOCK, "w") as lf:
                        fcntl.flock(lf, fcntl.LOCK_EX)
                        load = os.getloadavg()
                        pg = ctx.new_page()
                        pg.goto(f"http://127.0.0.1:{app.port}/#token={app.token}")
                        pg.wait_for_function("window.__t && window.__t.ready", timeout=30000)
                        pg.set_input_files("#file", str(src))
                        idle = tree_rss_mb(root)
                        peak = Peak(root)
                        peak.start()
                        t0 = time.perf_counter()
                        try:
                            res = pg.evaluate("(o) => window.__t.run(o)", {"preset": a.preset})
                            err = None
                        except Exception as e:  # the tab crashed or the evaluate was cut off
                            res, err = {}, f"{type(e).__name__}: {str(e)[:200]}"
                        wall = time.perf_counter() - t0
                        peak.stop_ev.set()
                        peak.join(2)
                        try:
                            pg.close()
                        except Exception:
                            pass
                    row = dict(
                        minutes=minutes, preset=a.preset, duration_s=round(facts["duration"], 1), size_mb=round(facts["size"] / 1e6), codec=facts["codec"], w=facts["w"], h=facts["h"],
                        state=res.get("state", "crashed"), error=err or res.get("error"), wall_s=round(wall, 1), sec_per_video_min=round(wall / (facts["duration"] / 60), 2),
                        n_samples=res.get("nSamples"), n_clips=res.get("nClips"), timing=res.get("timing"), detector=(res.get("settings") or {}).get("detector"),
                        idle_rss_mb=round(idle), peak_rss_mb=round(peak.peak), load_avg=[round(x, 2) for x in load], browser="Google Chrome (Playwright, headless)",
                        when=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    print(json.dumps(row), flush=True)
                    with open(results, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if not a.keep:
                        src.unlink(missing_ok=True)
                    if row["state"] != "done":
                        rc = 1
                        break  # a longer file will not do better
            finally:
                ctx.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        shutil.rmtree(work, ignore_errors=True)
    print(f"[limits] rows are also in {results}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
