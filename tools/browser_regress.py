#!/usr/bin/env python3
"""Parity for browser mode: the browser engine's balanced any-person run on the configured regression video, compared with
the baseline run exactly as tools/regress.py compares the Mac's.

    INFRAME_MODELS=<folder with onnx/> uv run --group browser python tools/browser_regress.py [--video PATH] [--tol 0.3]

It takes the regress lock (~/.cache/inframe/regress.lock, so no second heavy run overlaps), builds the test page (web/selftest: the real
BrowserEngine, Worker, WebCodecs and onnxruntime-web), starts `inframe serve` for it, opens Google Chrome through Playwright and gives the file to the
page in place with setInputFiles (nothing is copied or uploaded; the video is never opened as a picture). The run's segments.json goes to
~/.cache/inframe/regress-browser/, then `tools/regress.py --from-run` compares its presence intervals with the baseline. Only interval times and
PASS or FAIL are printed. Run it yourself (about 6 to 20 minutes on a quiet machine).
Exit code: 0 PASS, 1 FAIL, 2 the video, baseline or models are missing (the check only runs where a reference clip is configured, see inframe.localcfg; with none configured it prints SKIP and exits 2).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_limits import CHROME_ARGS, build_page  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from inframe import localcfg  # noqa: E402

LOCK = Path.home() / ".cache/inframe/regress.lock"
OUT = Path.home() / ".cache/inframe/regress-browser"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    video, baseline = localcfg.get("regress_video"), localcfg.get("regress_baseline")  # read now, not when the module was imported
    ap.add_argument("--video", default=str(video) if video else None)
    ap.add_argument("--baseline", default=str(baseline) if baseline else None)
    ap.add_argument("--tol", type=float, default=0.3, help="seconds each interval boundary may move (default 0.3)")
    a = ap.parse_args(argv)
    if not a.video or not a.baseline:
        print("SKIP browser_regress: no private clip configured (INFRAME_REGRESS_VIDEO and INFRAME_REGRESS_BASELINE, or ~/.config/inframe/local.json)")
        return 2
    models = Path(os.environ.get("INFRAME_MODELS", ROOT / "bench/detect/out/models")).expanduser()
    for what, p in (("video", Path(a.video)), ("baseline", Path(a.baseline)), ("models", models / "onnx/manifest.json")):
        if not p.is_file():
            print(f"FAIL browser_regress: {what} not found: {p} (this check only runs where the private clip is configured)")
            return 2
    os.environ["INFRAME_MODELS"] = str(models)
    sys.path.insert(0, str(ROOT / "src"))
    from playwright.sync_api import sync_playwright

    from inframe import serve

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[browser_regress] another regress run holds the lock; waiting", file=sys.stderr, flush=True)
            fcntl.flock(lf, fcntl.LOCK_EX)
        work = Path(tempfile.mkdtemp(prefix="inframe-bregress-"))
        try:
            build_page(work / "page")
            os.environ["INFRAME_SERVE_STATE"] = str(work / "state")
            os.environ["INFRAME_CACHE"] = str(work / "cache")
            httpd, app = serve.create(0, dist_dir=work / "page")
            threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
            try:
                with sync_playwright() as p:
                    ctx = p.chromium.launch_persistent_context(str(work / "profile"), channel="chrome", headless=True, args=CHROME_ARGS)
                    try:
                        page = ctx.new_page()
                        page.goto(f"http://127.0.0.1:{app.port}/#token={app.token}")
                        page.wait_for_function("window.__t && window.__t.ready", timeout=30000)
                        print(f"[browser_regress] models ready: {page.evaluate('window.__t.download()')['allReady']}", file=sys.stderr, flush=True)
                        page.set_input_files("#file", a.video)
                        t0 = time.perf_counter()
                        res = page.evaluate("window.__t.run({})")
                        wall = time.perf_counter() - t0
                    finally:
                        ctx.close()
            finally:
                httpd.shutdown()
                httpd.server_close()
                app.close()
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if res.get("state") != "done":
        print(f"FAIL browser_regress: the run ended {res.get('state')}: {res.get('error')}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    seg = res["segments"]
    (OUT / "segments.json").write_text(json.dumps(seg, indent=1))
    mine = [(round(s, 1), round(e, 1)) for s, e in seg["presence_intervals"]]
    print(f"[browser_regress] {len(mine)} intervals in {wall:.0f} s ({res.get('nSamples')} samples, detector {res.get('settings', {}).get('detector')}): "
          + ", ".join(f"{s}-{e}" for s, e in mine))
    r = subprocess.run([sys.executable, str(ROOT / "tools/regress.py"), "--from-run", str(OUT), "--tol", str(a.tol)], capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or ["FAIL browser_regress: regress.py printed nothing"])[-1])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
