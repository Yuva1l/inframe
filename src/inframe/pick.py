"""`inframe pick VIDEO`: a local page (stdlib http.server, plain HTML/JS) where you pick the person by clicking a numbered
box in a frame of the video. A click writes target.json (time, box, track id, and the detector preset and model the box came from); `inframe cut --target tap:target.json` uses it.

The server listens on 127.0.0.1 only, on a free port, and shows nothing but frames of the video you named. A web page in
another tab could still reach it (a request to 127.0.0.1, or DNS rebinding), so every request must carry the Host of
127.0.0.1:PORT or localhost:PORT and the per-run token that is part of the URL printed by the command, and a POST must
come from the page itself (its Origin).
The track id in target.json is the id the tracker gave the box in a short window before the tap (informational: ids differ
between runs, `cut` finds the person again by the box). It is null when the tracking extra is not installed.
"""
from __future__ import annotations

import hmac
import io
import json
import logging
import os
import secrets
import threading
import warnings
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# coremltools prints version warnings at its first import (scikit-learn and torch newer than it was tested with, an invalid
# escape sequence in its own docstrings); none of them matters for picking a person, so they are kept off the terminal.
logging.getLogger("coremltools").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*(scikit-learn|[Tt]orch) version.*")
warnings.filterwarnings("ignore", message=r"invalid escape sequence", category=SyntaxWarning)

import av  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from . import tracks  # noqa: E402
from .classes import PERSON, class_name  # noqa: E402
from .decode import scaled_size, upright  # noqa: E402
from .detect import UNITS_ENV, Detection, compute_units_name, for_tap, load_detector  # noqa: E402
from .probe import VideoInfo, probe  # noqa: E402
from .tap import iou  # noqa: E402

STEP = 0.2  # seconds between the frames the slider offers (the 5 fps of the balanced preset)
LONG_SIDE = 960
CACHE_MAX = 48  # frames (960 px long side, about 1.5 MB each) kept with their boxes
PICK_UNITS = "CPU_AND_GPU"  # the GPU is several times faster than CPU_ONLY; also the pipeline's default now (detect.DEFAULT_UNITS)


def pick_units() -> str:
    """INFRAME_DETECT_UNITS when it is set, else CPU_AND_GPU (the same as the pipeline's default)."""
    return compute_units_name() if os.environ.get(UNITS_ENV, "").strip() else PICK_UNITS


class Superseded(Exception):
    """The page asked for another time before this one was started: nobody is waiting for it any more."""


class _Job:
    def __init__(self, key: float, t: float):
        self.key, self.t = key, t
        self.event = threading.Event()
        self.result = None
        self.error: BaseException | None = None


def grab_frame(path: str, info: VideoInfo, t: float, long_side: int = LONG_SIDE) -> tuple[float, np.ndarray]:
    """(time of the frame, upright RGB) of the first frame at or after t (file time), found by seeking."""
    c = av.open(path)
    try:
        st = c.streams.video[0]
        tb = float(st.time_base)
        c.seek(int((t + info.format_start_time) / tb), stream=st, backward=True)
        w, h = scaled_size(round(info.width * info.sar), info.height, long_side)  # the shape a player shows (anamorphic: wider than the stored pixels)
        last = None
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            ft = float(fr.pts) * tb - info.format_start_time
            last = (ft, fr)
            if ft + 1e-4 >= t:
                break
        if last is None:
            raise ValueError(f"no frame at {t:.2f} s")
        ft, fr = last
        return ft, upright(fr.reformat(width=w, height=h, format="rgb24").to_ndarray(), info.rotation, info.mirrored)
    finally:
        c.close()


class Picker:
    def __init__(self, video: str, out: Path, preset_model: str = "small", photos: str | None = None, embedder: str = "dinov2",
                 offline: bool = False, cls: int = PERSON, preset_name: str | None = None):
        self.cls = cls
        self.video, self.out = video, out
        self.preset_name = preset_name
        self.photos, self.embedder, self.offline = photos, embedder, offline
        self.cands: list[dict] = []  # "Is this you?": the top tracks against the photos
        self.info = probe(video)
        self.model = preset_model
        self.det = None
        self.lock = threading.Lock()  # the detector: one predict at a time (the worker and the tap's tracking window share it)
        self.token = secrets.token_urlsafe(16)
        self.taps: list[dict] = []
        self.zone: list[float] | None = None  # x, y, w, h (fractions of the upright frame) drawn on the page; cut --target tap:FILE uses it
        self.units = pick_units()
        self.cache: OrderedDict[float | tuple, tuple[float, np.ndarray, list[Detection]]] = OrderedDict()
        self.stats = dict(decoded=0, detected=0, hits=0, dropped=0)
        self._cv = threading.Condition()
        self._pending: dict[float, _Job] = {}
        self._running: _Job | None = None
        self._latest: float | None = None
        self._worker: threading.Thread | None = None
        self._closed = False

    def detector(self):
        with self.lock:
            if self.det is None:
                self.det = for_tap(load_detector(self.model, self.units))
            return self.det

    def warm(self) -> None:
        """Load the detector now (the first load takes seconds, 20 s on a busy Mac) while the browser opens; errors show on the first request."""
        try:
            self.detector()
        except Exception:
            pass

    def frame(self, t: float):
        """(time of the frame, RGB, boxes best first) at t. Cached per time. The decode and the detector run on one worker thread
        and only the newest request is served: a request that another one replaced before the worker reached it raises Superseded."""
        t = min(max(t, self.info.offset), max(self.info.end_time - 0.05, self.info.offset))
        key = round(t, 2)
        with self._cv:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                self._latest = key  # the page is here now: pending jobs for times it has left are stale
                self._drop_pending()
                return hit
            self._latest = key
            job = self._pending.get(key)
            if job is None and self._running is not None and self._running.key == key:
                job = self._running
            if job is None:
                job = self._pending[key] = _Job(key, t)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="inframe-pick-worker", daemon=True)
                self._worker.start()
            self._cv.notify_all()
        job.event.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def _drop_pending(self) -> None:
        """Called with self._cv held: fail every pending job that is not the newest time."""
        for k in [k for k in self._pending if k != self._latest]:
            j = self._pending.pop(k)
            j.error = Superseded()
            j.event.set()
            self.stats["dropped"] += 1

    def frame_image(self, t: float) -> np.ndarray:
        """The RGB of the frame whose time the page was told (/boxes sends round(frame time, 3)). Looked up by that time, so a
        frame is never confused with the next one; on a miss (evicted) it is decoded directly: no detector, no latest-wins."""
        with self._cv:
            hit = self.cache.get(("ft", round(t, 3)))
            if hit is not None:
                self.cache.move_to_end(("ft", round(t, 3)))
                return hit[1]
        return grab_frame(self.video, self.info, t - 5e-4)[1]  # the page's t is rounded to 1 ms: take the frame that rounds to it

    def close(self) -> None:
        with self._cv:
            self._closed = True
            for j in self._pending.values():
                j.error = Superseded()
                j.event.set()
            self._pending.clear()
            self._cv.notify_all()

    def _work(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                self._drop_pending()  # the page has moved on from these
                job = self._pending.pop(self._latest, None)
                if job is None:
                    continue
                self._running = job
            try:
                job.result = self._compute(job)
            except BaseException as e:  # reported to the waiting requests, never kills the worker
                job.error = e
            with self._cv:
                self._running = None
            job.event.set()

    def _compute(self, job: _Job):
        ft, rgb = grab_frame(self.video, self.info, job.t)
        self.stats["decoded"] += 1
        with self._cv:
            if job.key != self._latest:  # replaced while decoding: skip the slow part
                self.stats["dropped"] += 1
                raise Superseded()
        det = self.detector()
        with self.lock:
            dets = sorted(det.detect([rgb], **({} if self.cls == PERSON else {"cls": self.cls}))[0], key=lambda d: -d.score)
        self.stats["detected"] += 1
        entry = (ft, rgb, dets)
        with self._cv:
            for k in {job.key, round(ft, 2), ("ft", round(ft, 3))}:  # the page asks for the frame again by the time the frame really has
                self.cache[k] = entry
                self.cache.move_to_end(k)
            while len(self.cache) > CACHE_MAX:
                self.cache.popitem(last=False)
        return entry

    def rank(self, announce=print) -> list[dict]:
        """Run the whole video once (detect, track, embed) and keep the top tracks against the photos."""
        from . import identity
        from .pipeline import PRESETS, Settings, effective_fps, score_detect

        preset = next((p for p in PRESETS.values() if p.model == self.model and p.detect_fps), PRESETS["balanced"])  # the page's detector
        fps = effective_fps(Settings(), preset)
        target = identity.PhotoTarget(self.photos, embedder=self.embedder, offline=self.offline, rank_only=True, detect=identity.lazy_detect(self.model),
                                      cls=self.cls)
        announce(f"inframe pick: ranking the people in the video against the photos ({self.embedder}); this reads the whole video once")
        sc = score_detect(self.video, self.info, preset, fps, target, None, "ocsort")
        self.cands = list(sc.info.get("target_info", {}).get("ranking", []))
        announce(f"inframe pick: {len(self.cands)} candidates" + ("" if self.cands else "; nobody was tracked, use the frames below"))
        return self.cands

    def thumb(self, i: int) -> bytes:
        c = self.cands[i]
        _, rgb = grab_frame(self.video, self.info, c["t"])
        x0, y0, x1, y1 = c["box"]
        h, w = rgb.shape[:2]
        crop = rgb[max(int(y0 * h) - 4, 0):int(y1 * h) + 4, max(int(x0 * w) - 4, 0):int(x1 * w) + 4]
        im = Image.fromarray(crop)
        im.thumbnail((240, 360))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def confirm(self, i: int) -> dict:
        """The person confirmed candidate i: one tap on that track at the moment that looked most like the photos."""
        c = self.cands[i]
        tap = dict(time=c["t"], box=c["box"], track_id=c["track_id"], index=None, candidate=c["rank"])
        self.taps = [tap]
        self.save()
        return tap

    def local_track_id(self, ft: float, box) -> int | None:
        """The id a tracker gives `box` after following the persons of the 2 s before it (None without the tracking extra)."""
        try:
            n = int(2.0 / STEP)
            times = [max(self.info.offset, ft - STEP * k) for k in range(n, -1, -1)]
            det = self.detector()
            with tracks.Tracker("ocsort", 1 / STEP) as tr:
                ids = None
                for tt in times:
                    _, rgb = grab_frame(self.video, self.info, tt)
                    with self.lock:
                        dets = det.detect([rgb], **({} if self.cls == PERSON else {"cls": self.cls}))[0]
                    ids = tr.update(tt, dets, rgb)
            best = max(ids or [], key=lambda d: iou(d.box, box), default=None)
            return best.track_id if best is not None and iou(best.box, box) >= 0.5 else None
        except Exception:
            return None

    def set_zone(self, zone) -> None:
        """Remember the zone drawn on the page (None clears it); it is written with the taps."""
        from .zone import Zone

        self.zone = _round_zone(Zone.of(zone)) if zone else None
        self.save()

    def add_tap(self, ft: float, box: list[float], index: int | None) -> dict:
        tap = dict(time=round(ft, 3), box=[round(float(v), 4) for v in box], track_id=self.local_track_id(ft, box), index=index)
        self.taps.append(tap)
        self.save()
        return tap

    def save(self) -> None:
        first = self.taps[0] if self.taps else {}
        from .pipeline import PRESETS

        name = self.preset_name or next((n for n, p in PRESETS.items() if p.model == self.model and p.detect_fps), None)
        doc = dict(video=str(Path(self.video).resolve()), frame_size=[self.info.display_size[0], self.info.display_size[1]],
                   model=self.model, **({"preset": name} if name else {}),  # `cut` finds the box again with this detector (tap.resolve_preset)
                   time=first.get("time"), box=first.get("box"), track_id=first.get("track_id"), taps=self.taps,
                   **({"zone": self.zone} if self.zone else {}),
                   **({"photos": self.photos if self.photos.startswith("saved:") else str(Path(self.photos).expanduser().resolve()),
                      "embedder": self.embedder} if self.photos else {}),
                   **({"class": class_name(self.cls)} if self.cls != PERSON else {}))
        self.out.parent.mkdir(parents=True, exist_ok=True)
        if self.taps:
            self.out.write_text(json.dumps(doc, indent=1))
        else:
            self.out.unlink(missing_ok=True)


def _round_zone(z) -> list[float]:
    """x, y, w, h to 4 decimals without the rounded zone leaving the frame: round x and y, then derive w and h from the rounded far edge."""
    x, y = round(z.x, 4), round(z.y, 4)
    return [x, y, round(min(round(z.x + z.w, 4), 1.0) - x, 4), round(min(round(z.y + z.h, 4), 1.0) - y, 4)]


PAGE = """<!doctype html><meta charset=utf-8><title>inframe pick</title>
<style>body{font:15px system-ui;margin:12px;background:#111;color:#eee}canvas{max-width:100%;cursor:crosshair;display:block}
button{font:inherit;padding:4px 10px;margin:2px}#msg{min-height:1.4em;color:#9d9}input[type=range]{width:min(100%,900px)}</style>
<div><button id=prev>-1 s</button><button id=next>+1 s</button> <input id=sl type=range min=0 max=1 step=0.2 value=0> <span id=tl></span></div>
<div id=cands style="display:none"><b>Is this you?</b> <span id=cmsg></span><div id=cl style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0"></div></div>
<canvas id=cv></canvas>
<div id=msg>Click the person you want to follow (numbered boxes). Click more than one frame if they leave and come back.</div>
<div><button id=undo>Undo last tap</button><button id=zbtn>Draw zone</button><button id=zclr>Clear zone</button><button id=done>Done</button> <span id=taps></span> <span id=zinfo></span></div>
<script>
const $=id=>document.getElementById(id),cv=$('cv'),cx=cv.getContext('2d');let cur=null,info=null,img=new Image();
const K=new URLSearchParams(location.search).get('k')||'',say=m=>{$('msg').textContent=m};
async function api(path,opt){const r=await fetch(path+(path.includes('?')?'&':'?')+'k='+encodeURIComponent(K),opt);let d={};
 try{d=await r.json()}catch(e){}if(!r.ok||d.error)throw new Error(d.error||('request failed ('+r.status+')'));return d}
let seq=0,ctl=null,timer=null;
const sched=t=>{clearTimeout(timer);$('tl').textContent=Number(t).toFixed(1)+' s ...';timer=setTimeout(()=>load(t),300)};
img.onerror=()=>say('The frame could not be loaded (see the terminal where inframe pick runs).');
async function load(t){clearTimeout(timer);const my=++seq;if(ctl)ctl.abort();ctl=new AbortController();$('tl').textContent=Number(t).toFixed(1)+' s, loading...';
 let d;try{d=await api('/boxes?t='+t,{signal:ctl.signal})}catch(e){if(my!==seq||e.name==='AbortError'||e.message==='superseded')return;say('Error: '+e.message);return}
 if(my!==seq)return;cur=d;img.onload=draw;img.src='/frame.jpg?t='+cur.t+'&k='+encodeURIComponent(K);$('tl').textContent=cur.t.toFixed(1)+' s of '+info.duration.toFixed(1)}
function draw(){cv.width=img.naturalWidth;cv.height=img.naturalHeight;cx.drawImage(img,0,0);cx.lineWidth=3;cx.font='bold 22px system-ui';
 drawZone(zone);cur.boxes.forEach(b=>{const[x0,y0,x1,y1]=b.box;cx.strokeStyle='#4ade80';cx.strokeRect(x0*cv.width,y0*cv.height,(x1-x0)*cv.width,(y1-y0)*cv.height);
 cx.fillStyle='#4ade80';cx.fillRect(x0*cv.width,y0*cv.height-26,34,26);cx.fillStyle='#000';cx.fillText(b.i,x0*cv.width+6,y0*cv.height-5)})}
cv.onclick=async e=>{if(zoneMode)return;const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
 const hit=cur.boxes.filter(b=>b.box[0]<=x&&x<=b.box[2]&&b.box[1]<=y&&y<=b.box[3]).sort((a,b)=>area(a)-area(b))[0];
 if(!hit){$('msg').textContent='No box there: click inside a numbered box.';return}
 let d;try{d=await api('/tap',{method:'POST',body:JSON.stringify({t:cur.t,box:hit.box,index:hit.i})})}catch(e){say('Error: '+e.message);return}
 $('msg').textContent='Saved '+d.file+' ('+d.taps+' tap'+(d.taps>1?'s':'')+'). Press Done when finished.';$('taps').textContent=d.taps+' tap(s)'};
let zoneMode=false,zone=null,drag=null;
const pt=e=>{const r=cv.getBoundingClientRect();return[Math.min(Math.max((e.clientX-r.left)/r.width,0),1),Math.min(Math.max((e.clientY-r.top)/r.height,0),1)]};
const zinfo=()=>{$('zinfo').textContent=zone?'zone '+zone.map(v=>v.toFixed(2)).join(','):''};
function drawZone(z){if(!z)return;cx.save();cx.setLineDash([10,6]);cx.strokeStyle='#facc15';cx.lineWidth=3;cx.strokeRect(z[0]*cv.width,z[1]*cv.height,z[2]*cv.width,z[3]*cv.height);cx.restore()}
cv.onmousedown=e=>{if(!zoneMode)return;e.preventDefault();drag=pt(e)};
cv.onmousemove=e=>{if(!zoneMode||!drag)return;const p=pt(e);draw();drawZone([Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])])};
cv.onmouseup=async e=>{if(!zoneMode||!drag)return;const p=pt(e),z=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])];drag=null;
 if(z[2]<0.02||z[3]<0.02){draw();drawZone(zone);return}
 try{await api('/zone',{method:'POST',body:JSON.stringify({zone:z})});zone=z;zinfo();say('Zone saved: only people inside it will count (static camera only). Press Done when finished.')}catch(err){say('Error: '+err.message)}
 zoneMode=false;$('zbtn').textContent='Draw zone';cv.style.cursor='crosshair';draw();drawZone(zone)};
$('zbtn').onclick=()=>{zoneMode=!zoneMode;$('zbtn').textContent=zoneMode?'Drawing: drag on the picture':'Draw zone';cv.style.cursor=zoneMode?'cell':'crosshair'};
$('zclr').onclick=async()=>{try{await api('/zone',{method:'POST',body:JSON.stringify({zone:null})});zone=null;zinfo();draw()}catch(e){say('Error: '+e.message)}};
const area=b=>(b.box[2]-b.box[0])*(b.box[3]-b.box[1]);
$('sl').oninput=e=>sched(e.target.value);$('prev').onclick=()=>{$('sl').value=Math.max(0,cur.t-1);load($('sl').value)};
$('next').onclick=()=>{$('sl').value=Math.min(info.duration,cur.t+1);load($('sl').value)};
$('undo').onclick=async()=>{try{const d=await api('/undo',{method:'POST'});$('taps').textContent=d.taps+' tap(s)'}catch(e){say('Error: '+e.message)}};
$('done').onclick=async()=>{try{await api('/done',{method:'POST'});say('Done. You can close this tab.')}catch(e){say('Error: '+e.message)}};
async function cands(){let d;try{d=await api('/cands')}catch(e){return}if(!d.cands.length)return;$('cands').style.display='block';
 d.cands.forEach((c,i)=>{const b=document.createElement('button');b.style.cssText='background:#222;color:#eee;border:1px solid #444;padding:4px';
  b.innerHTML='<img src="/cand.jpg?i='+i+'&k='+encodeURIComponent(K)+'" style="height:160px;display:block"> #'+c.rank+' at '+c.t.toFixed(0)+' s, '+Math.round(c.score*100)+'%';
  b.onclick=async()=>{try{await api('/confirm',{method:'POST',body:JSON.stringify({i})});$('cmsg').textContent='Saved. Press Done.';$('taps').textContent='1 confirm'}catch(e){say('Error: '+e.message)}};$('cl').appendChild(b)})}
cands();
api('/info').then(i=>{info=i;zone=i.zone;zinfo();$('sl').max=i.duration;$('sl').min=i.offset;$('sl').value=i.offset+i.duration/2;load($('sl').value)}).catch(e=>say('Error: '+e.message));
</script>"""


CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class QuietServer(ThreadingHTTPServer):
    """A client that went away (a cancelled fetch, a closed tab) is a normal event here: no traceback on the terminal."""

    def handle_error(self, request, client_address):
        import sys

        if not isinstance(sys.exc_info()[1], CLIENT_GONE):
            super().handle_error(request, client_address)


def make_handler(pk: Picker, stop: threading.Event, httpd_ref: list):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal quiet
            pass

        def handle(self):  # the page cancels a request by closing it (a newer slider position, a closed tab): that is not an error
            try:
                super().handle()
            except CLIENT_GONE:
                self.close_connection = True

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except CLIENT_GONE:
                self.close_connection = True

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _allowed(self, q: dict, post: bool) -> bool:
            """Host is this server, the token is this run's, and a POST comes from the page itself."""
            port = httpd_ref[0].server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                return False
            if post and self.headers.get("Origin") not in {f"http://{h}" for h in hosts}:
                return False
            return hmac.compare_digest(q.get("k", [""])[0].encode(), pk.token.encode())

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._allowed(q, False):
                return self._json(dict(error="forbidden"), 403)
            try:
                if u.path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif u.path == "/info":
                    self._json(dict(duration=pk.info.duration, offset=pk.info.offset, width=pk.info.display_size[0],
                                    height=pk.info.display_size[1], zone=pk.zone))
                elif u.path == "/boxes":
                    ft, _, dets = pk.frame(float(q["t"][0]))
                    self._json(dict(t=round(ft, 3), boxes=[dict(i=i + 1, box=[round(v, 4) for v in d.box], score=round(d.score, 3))
                                                           for i, d in enumerate(d for d in dets if d.score >= 0.4)]))
                elif u.path == "/cands":
                    self._json(dict(cands=pk.cands))
                elif u.path == "/cand.jpg":
                    self._send(200, pk.thumb(int(q["i"][0])), "image/jpeg")
                elif u.path == "/frame.jpg":
                    rgb = pk.frame_image(float(q["t"][0]))
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, "JPEG", quality=85)
                    self._send(200, buf.getvalue(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            except Superseded:  # a newer time was asked for before this one started
                self._json(dict(error="superseded"), 409)
            except CLIENT_GONE:
                self.close_connection = True
            except (KeyError, ValueError) as e:
                self._json(dict(error=str(e)), 400)
            except Exception as e:  # a missing model, a corrupt seek: say so on the page instead of dropping the connection
                self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

        def do_POST(self):
            u = urlparse(self.path)
            if not self._allowed(parse_qs(u.query), True):
                return self._json(dict(error="forbidden"), 403)
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            try:
                if u.path == "/tap":
                    d = json.loads(body)
                    box = [float(v) for v in d["box"]]
                    pk.add_tap(float(d["t"]), box, d.get("index"))
                    self._json(dict(taps=len(pk.taps), file=str(pk.out)))
                elif u.path == "/zone":
                    pk.set_zone(json.loads(body).get("zone"))
                    self._json(dict(zone=pk.zone, file=str(pk.out)))
                elif u.path == "/confirm":
                    pk.confirm(int(json.loads(body)["i"]))
                    self._json(dict(taps=1, file=str(pk.out)))
                elif u.path == "/undo":
                    if pk.taps:
                        pk.taps.pop()
                        pk.save()
                    self._json(dict(taps=len(pk.taps)))
                elif u.path == "/done":
                    self._json(dict(taps=len(pk.taps)))
                    stop.set()
                    threading.Thread(target=httpd_ref[0].shutdown, daemon=True).start()
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError, TypeError) as e:
                self._json(dict(error=str(e)), 400)
            except Exception as e:
                self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

    return H


def serve(video: str, out: Path, port: int = 0, model: str = "small", announce=print, photos: str | None = None,
          embedder: str = "dinov2", offline: bool = False, cls: int = PERSON, open_browser: bool = True, preset_name: str | None = None) -> int:
    """Run the pick page until Done is pressed (or Ctrl-C). Returns 0 when target.json was written, 2 when no tap was made.
    The page is opened in the default browser unless open_browser is False."""
    pk = Picker(video, out, model, photos, embedder, offline, cls, preset_name)
    if photos:
        from .identity import photo_files

        if photos.startswith("saved:"):
            from .targets import check_usable

            check_usable(photos[len("saved:"):])
        else:
            photo_files(photos)  # fail before the long ranking run when the folder is no good
        pk.rank(announce)
    stop = threading.Event()
    ref: list = []
    httpd = QuietServer(("127.0.0.1", port), make_handler(pk, stop, ref))
    ref.append(httpd)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/?k={pk.token}"
    announce(f"inframe pick: open {url}  (click the person; press Done when finished; Ctrl-C also stops)")
    if open_browser:
        try:
            if not webbrowser.open(url):
                announce("inframe pick: could not open a browser; open the address above yourself")
        except Exception:
            announce("inframe pick: could not open a browser; open the address above yourself")
    threading.Thread(target=pk.warm, name="inframe-pick-warm", daemon=True).start()
    announce("inframe pick: waiting for your tap... (press Done on the page, or Ctrl-C here, to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        pk.close()
    return 0 if pk.taps else 2
