"""Parallel ranged, resumable, md5-checked download of the 3 Apache-2.0 RF-DETR weights into ~/.roboflow/models."""
import os, sys, hashlib, threading, urllib.request
G = "https://storage.googleapis.com/rfdetr"
FILES = {
 "rf-detr-nano.pth": (G + "/nano_coco/checkpoint_best_regular.pth", "fb6504cce7fbdc783f7a46991f07639f"),
 "rf-detr-small.pth": (G + "/small_coco/checkpoint_best_regular.pth", "fb37061c1af7bace359c91b723a8d5c1"),
 "rf-detr-base.pth": (G + "/rf-detr-base-coco.pth", "b4d3ce46099eaed50626ede388caf979"),
}
W = os.path.expanduser("~/.roboflow/models"); os.makedirs(W, exist_ok=True)
NCH = 12
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def size(url):
    r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30)
    return int(r.headers["Content-Length"])
def chunk(url, path, lo, hi):
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if lo + have > hi: return
        try:
            rq = urllib.request.Request(url, headers={"Range": f"bytes={lo+have}-{hi}"})
            with urllib.request.urlopen(rq, timeout=60) as r, open(path, "ab") as f:
                while True:
                    b = r.read(1 << 18)
                    if not b: break
                    f.write(b)
        except Exception as e:
            print("retry", path, e, flush=True)
def get(name):
    url, want = FILES[name]; dst = os.path.join(W, name)
    if os.path.exists(dst) and md5(dst) == want: print("ok", name, flush=True); return
    n = size(url); step = -(-n // NCH); parts = []; ts = []
    for i in range(NCH):
        lo, hi = i * step, min(n, (i + 1) * step) - 1
        p = f"{dst}.r{i:02d}"; parts.append(p)
        t = threading.Thread(target=chunk, args=(url, p, lo, hi)); t.start(); ts.append(t)
    for t in ts: t.join()
    with open(dst + ".tmp", "wb") as o:
        for p in parts:
            with open(p, "rb") as f: o.write(f.read())
    if md5(dst + ".tmp") != want: sys.exit("md5 mismatch " + name)
    os.replace(dst + ".tmp", dst)
    for p in parts: os.remove(p)
    print("downloaded", name, n, flush=True)
order = sys.argv[1:] or list(FILES)
for nm in order: get(nm)
