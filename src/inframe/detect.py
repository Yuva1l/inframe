"""RF-DETR person detection. CoreML fp16, CPU_AND_GPU by default (the fastest, and deterministic with the package whose topk is fp32,
bench/detect/export_gpu.py; CPU_ONLY and CPU_AND_NE stay selectable with INFRAME_DETECT_UNITS), fixed batch 8.

The CoreML packages are exported by bench/detect (see bench/detect/setup.sh). Look-up order for
the models folder: $INFRAME_MODELS, ./models, the cache, <repo>/bench/detect/out/models; the first one that holds
the package wins (the cache usually holds only the .pth checkpoints).
Torch MPS is only a fallback when CoreML fails, and it says so.
"""
from __future__ import annotations

import gc
import glob
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import models
from .classes import PERSON  # COCO category id 1; RF-DETR COCO checkpoints use 91 class slots (classes.py has the table)
BATCH = 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
# size name -> (input resolution, rfdetr class for the torch fallback)
MODELS = {"nano": (384, "RFDETRNano"), "small": (512, "RFDETRSmall"), "base": (560, "RFDETRBase")}
MIN_PERSON_SCORE = 0.2  # detections of the wanted class below this are not kept at all (thresholds are applied later)


@dataclass
class Detection:
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 normalised to the frame
    track_id: int | None = None  # set by tracks.Tracker (detect_fps >= 5); None = not tracked yet or no tracker ran


BENCH_MODELS = Path(__file__).resolve().parents[2] / "bench/detect/out/models"


def model_dirs() -> list[Path]:
    """Existing folders that may hold the CoreML packages, in look-up order."""
    cands = [os.environ.get("INFRAME_MODELS"), "models", str(models.cache_dir()), str(BENCH_MODELS)]
    return [Path(c) for c in cands if c and Path(c).is_dir()]


def models_dir(size: str | None = None) -> Path:
    """The first existing folder (with `size`: the first that holds that size's CoreML package).
    A folder that exists but only holds other files (the cache holds just the .pth checkpoints) is skipped."""
    dirs = model_dirs()
    if size is not None:
        dirs = [d for d in dirs if _packages(d, size)]
    if not dirs:
        raise models.ModelsMissing(models.missing_message(
            f"the {size} CoreML package (batch {BATCH} or 1)" if size else "RF-DETR CoreML models", searched=_searched() if size else None))
    return dirs[0]


def _searched() -> str:
    return ", ".join(str(d) for d in model_dirs()) or "no existing models folder"


def _packages(d: Path, size: str) -> list[str]:
    """The CoreML packages of `size` in `d`, the batch-8 export first (the fixed batch of nano and small), then batch 1 (base ships as
    base_coreml_b1: the batch-8 base export was never made), then any other batch."""
    found = glob.glob(str(d / f"{size}_coreml_b*" / "**" / "*.mlpackage"), recursive=True)
    return sorted(found, key=lambda p: (package_batch(p) != BATCH, package_batch(p) != 1, p))


def package_batch(path: str) -> int:
    """The fixed batch of an exported package, from its folder name (<size>_coreml_b<N>); 0 when the name does not say."""
    m = re.search(r"_coreml_b(\d+)(?:/|$)", path)
    return int(m.group(1)) if m else 0


def spec_batch(spec) -> int:
    """The fixed batch of a package, from its first input's shape (N, 3, H, W); 0 when the spec does not say."""
    try:
        shape = list(spec.description.input[0].type.multiArrayType.shape)
    except Exception:
        return 0
    return int(shape[0]) if len(shape) == 4 and shape[0] > 0 else 0


def available_sizes() -> tuple[str, ...]:
    """Model sizes (smallest first) that have a CoreML package in some models folder."""
    return tuple(s for s in MODELS if any(_packages(d, s) for d in model_dirs()))


def find_package(size: str) -> str:
    return _packages(models_dir(size), size)[0]


def preprocess(frames: list[np.ndarray], res: int) -> np.ndarray:
    """List of (H, W, 3) uint8 -> (N, 3, res, res) float32, squashed to res x res and normalised.
    Squashing keeps normalised box coordinates valid for the original frame."""
    a = np.stack([
        np.asarray(Image.fromarray(f).resize((res, res), Image.BILINEAR), np.float32) / 255.0 for f in frames
    ]).transpose(0, 3, 1, 2)
    return np.ascontiguousarray((a - MEAN) / STD)


def postprocess(boxes: np.ndarray, logits: np.ndarray, n: int, min_score: float = MIN_PERSON_SCORE, cls: int = PERSON) -> list[list[Detection]]:
    """boxes (B,Q,4) cxcywh normalised, logits (B,Q,C) -> detections of class id `cls` (COCO-91, default person) per image (first n of the batch)."""
    if not 0 <= cls < logits.shape[-1]:
        raise ValueError(f"class id {cls} is outside the model's {logits.shape[-1]} class slots")
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits[:, :, cls], np.float32)))
    out = []
    for b in range(n):
        dets = []
        for q in np.nonzero(p[b] >= min_score)[0]:
            cx, cy, w, h = (float(v) for v in boxes[b, q])
            dets.append(Detection(float(p[b, q]), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        out.append(sorted(dets, key=lambda d: -d.score))
    return out


def _split_out(outs: list) -> tuple[np.ndarray, np.ndarray]:
    outs = [np.asarray(o) for o in outs]
    boxes = next(o for o in outs if o.shape[-1] == 4)
    logits = next(o for o in outs if o is not boxes and o.ndim == 3)
    return boxes, logits


UNITS_ENV = "INFRAME_DETECT_UNITS"  # CPU_AND_GPU (the default; deterministic only with an fp32-topk package), CPU_ONLY or CPU_AND_NE
UNITS = ("CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU")
DEFAULT_UNITS = "CPU_AND_GPU"


def compute_units_name() -> str:
    """CPU_AND_GPU (the default), CPU_ONLY or CPU_AND_NE. With the plain fp16 export, the GPU path is not deterministic: its fp16 topk
    (RF-DETR's query selection, top 300 of 1024 proposals) gives a different result on every call of the same batch when the batch is
    larger than 1, so a frame's person score moved by up to 0.92 between two identical passes. A package
    exported with the topk in fp32 (bench/detect/export_gpu.py) is bit-identical on the GPU call to call, process to process and whatever
    the batch, at the same speed (small: 11.0 ms per frame against 34.7 CPU_ONLY); CoreMLDetector warns when CPU_AND_GPU gets a package
    that is not. CPU_ONLY and CPU_AND_NE give identical scores for the same frame, twice and whatever the batch. CPU_AND_GPU is the default
    (decided 2026-10-16): about 3x faster than CPU_ONLY and repeatable with that package; on the regression clip it moves one interval boundary by 0.4 s,
    an accepted change (tools/regress_accepted.json, if you keep one). CPU_ONLY reproduces the reference run exactly (max deviation
    0.20 s); CPU_AND_NE is about 1.4x faster than CPU_ONLY but moves some boundaries by 0.4-1.0 s (regress FAIL)."""
    v = os.environ.get(UNITS_ENV, "").strip().upper()
    return v if v in UNITS else DEFAULT_UNITS


def topk_precision(spec) -> str | None:
    """"fp16" or "fp32": the dtype of the input of the package's topk op (RF-DETR's two-stage query selection); None when the
    program has no topk or the spec cannot be read."""
    try:
        from coremltools.proto import MIL_pb2

        f = spec.mlProgram.functions["main"]
        ops = f.block_specializations[f.opset].operations
    except Exception:
        return None
    dtypes = {o.name: o.type.tensorType.dataType for op in ops for o in op.outputs}
    names = {MIL_pb2.DataType.Value("FLOAT16"): "fp16", MIL_pb2.DataType.Value("FLOAT32"): "fp32"}
    for op in ops:
        if op.type == "topk" and "x" in op.inputs and op.inputs["x"].arguments:
            return names.get(dtypes.get(op.inputs["x"].arguments[0].name))
    return None


def stale_package_warning(path, size: str, batch: int, units: str, spec) -> str | None:
    """The warning for a package that is not repeatable on the units in use (an fp16 topk at batch > 1 on CPU_AND_GPU), with the command
    that re-exports it; None when the package is fine."""
    if units != "CPU_AND_GPU" or gpu_repeatable(spec, batch):
        return None
    return (f"[detect] WARNING: {path} is stale: it has an fp16 topk at batch {batch}, so on CPU_AND_GPU (the default) its scores change from "
            f"run to run. Re-export it with: bench/detect/.venv/bin/python bench/detect/export_gpu.py {size}, "
            f"or set {UNITS_ENV}=CPU_ONLY")


def gpu_repeatable(spec, batch: int) -> bool:
    """Whether CPU_AND_GPU gives the same scores twice with this package: batch 1, or a topk that is not fp16."""
    return batch == 1 or topk_precision(spec) != "fp16"


IDLE_FREE_S = 0.5  # CPU_ONLY only: the model is dropped after this long without a predict (see CoreMLDetector). The crash comes at an idle of 1.0 s (3 of 3 runs; 0.9 s: 0 of 3) and the timer needs the GIL to run, so a C call that holds the GIL while the timer is due delays the free: the shorter this is, the longer such a stall has to be to crash (measured with a 1.15 s sorted() stall and a gc.collect() of a 12M-object heap, both starting 0.6 s after a predict: 0.75 exited 139, 0.5 exited 0; a stall must cover IDLE_FREE_S to 1.0 s, so the margin is 1.0 - IDLE_FREE_S). The price is a reload (0.4-0.7 s) after every pause longer than this, e.g. between photo-mode batches


class CoreMLDetector:
    """The RF-DETR CoreML package. A CPU_ONLY model that stays idle for about a second crashes the whole process with SIGSEGV (found on
    macOS 26.5 with coremltools 9.0: a live model, no predict for 1 s, whatever the main thread does; the process is fine when the model is
    freed first). Photo mode idles the detector for seconds (the embedder starts up, the decision at the end), so a CPU_ONLY model is dropped
    after IDLE_FREE_S without a predict and loaded again (0.4 s) by the next one. CPU_AND_NE and CPU_AND_GPU are not affected."""

    def __init__(self, size: str, units: str | None = None):
        self.size, (self.res, _) = size, MODELS[size]
        self.path = find_package(size)
        self.batch = package_batch(self.path) or BATCH  # the package takes exactly this many frames per call
        self.units = units if units in UNITS else compute_units_name()  # `units` is for callers that do not need repeatable scores (inframe pick)
        self._lock, self._timer, self._last = threading.Lock(), None, time.monotonic()
        try:
            self.model = self._load()
        except Exception as e:
            if self.units == "CPU_ONLY":
                raise
            where = {"CPU_AND_NE": "on the Neural Engine", "CPU_AND_GPU": "on the GPU"}[self.units]
            print(f"[detect] WARNING: CoreML would not load {where} ({type(e).__name__}: {e}); using CPU_ONLY", file=sys.stderr)
            self.units = "CPU_ONLY"
            self.model = self._load()
        self.backend = "coreml-fp16-" + self.units.lower()
        spec = self.model.get_spec()
        self.input = spec.description.input[0].name
        self.batch = spec_batch(spec) or self.batch  # the input shape is the truth; the folder name only a guess
        self.gpu_repeatable = gpu_repeatable(spec, self.batch)
        warning = stale_package_warning(self.path, size, self.batch, self.units, spec)
        if warning:
            print(warning, file=sys.stderr)
        self._arm()

    def _load(self):
        import coremltools as ct

        return ct.models.MLModel(self.path, compute_units=getattr(ct.ComputeUnit, self.units))

    def _arm(self) -> None:
        self._last = time.monotonic()
        if self.units == "CPU_ONLY":
            self._timer = threading.Timer(IDLE_FREE_S, self._free)
            self._timer.daemon = True
            self._timer.start()

    def _free(self) -> None:
        with self._lock:
            if time.monotonic() - self._last >= IDLE_FREE_S - 0.01:
                self.model = None
                gc.collect()

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            if self.model is None:
                self.model = self._load()
            try:
                return _split_out(list(self.model.predict({self.input: x}).values()))
            finally:
                self._arm()

    def detect(self, frames: list[np.ndarray], cls: int = PERSON) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for i in range(0, len(frames), self.batch):  # the package has a fixed batch (8, or 1 for base): chunk, and pad the last chunk
            chunk = frames[i:i + self.batch]
            n = len(chunk)
            x = preprocess(chunk, self.res)
            if n < self.batch:  # pad by repeating the last frame
                x = np.concatenate([x, np.repeat(x[-1:], self.batch - n, axis=0)])
            boxes, logits = self._run(x)
            out += postprocess(boxes, logits, n, cls=cls)
        return out


class TorchDetector:
    """Fallback only: torch MPS fp16 through the rfdetr package (pip install inframe[torch])."""

    backend = "torch-mps-fp16 (FALLBACK)"

    def __init__(self, size: str, checkpoint: Path | str):
        import rfdetr
        import torch

        self.torch, self.size, (self.res, cls) = torch, size, MODELS[size]
        # an explicit path: rfdetr's default is relative to the current directory and is downloaded when missing
        self.model = getattr(rfdetr, cls)(pretrain_weights=str(checkpoint)).model.model.eval().to("mps").half()
        self.path = str(checkpoint)

    def detect(self, frames: list[np.ndarray], cls: int = PERSON) -> list[list[Detection]]:
        x = preprocess(frames, self.res)
        with self.torch.no_grad():
            o = self.model(self.torch.from_numpy(x).to("mps", self.torch.float16))
            if isinstance(o, dict):
                o = (o["pred_boxes"], o["pred_logits"])
            boxes, logits = (t.float().cpu().numpy() for t in (o[0], o[1]))
        return postprocess(boxes, logits, len(frames), cls=cls)


def load_fallback_detector(size: str):
    """The torch MPS detector; raises when torch / rfdetr are not installed (uv sync --extra torch).
    The checkpoint comes from the inframe cache (checked, downloaded there when missing); with --offline it must
    already be there, and rfdetr is given that exact path so it never looks (or downloads) anywhere else."""
    from . import models

    return TorchDetector(size, models.torch_checkpoint(size))


def load_detector(size: str, units: str | None = None):
    """CoreML first; torch MPS only if CoreML fails (and a warning says so). A CoreML predict() that
    fails later, mid-run, is handled by the pipeline with load_fallback_detector. `units` overrides the compute units
    (default: INFRAME_DETECT_UNITS, else CPU_AND_GPU)."""
    from . import models

    try:
        return CoreMLDetector(size, units)
    except Exception as e:
        print(f"[detect] WARNING: CoreML failed ({type(e).__name__}: {e}); FALLING BACK to torch MPS", file=sys.stderr)
        try:
            return load_fallback_detector(size)
        except Exception as e2:
            if isinstance(e, models.ModelsMissing):  # the first failure is the useful one: the models are not there
                raise models.ModelsMissing(str(e)) from e2
            raise


# ---------------------------------------------------------------- sliced detection (tap mode)
SLICES_ENV = "INFRAME_SLICES"  # "2x2", "3x2", ...: tap mode runs the detector on slices too. Default off: see README
SLICE_GRID = (1, 1)  # columns, rows when INFRAME_SLICES is unset or not a grid
SLICE_OVERLAP = 0.2  # fraction of a slice shared with its neighbour: a person cut by one slice edge is whole in the next
MERGE_IOU = 0.5  # boxes this much alike are one person; the higher score stays
MERGE_IOS = 0.8  # a box this much inside a better box that touches a slice edge is that person cut by the edge


def slice_grid() -> tuple[int, int]:
    v = os.environ.get(SLICES_ENV, "").strip().lower()
    try:
        c, r = (int(x) for x in v.split("x"))
        if 1 <= c <= 6 and 1 <= r <= 6:
            return (c, r)
    except ValueError:
        pass
    return SLICE_GRID


def slice_windows(h: int, w: int, cols: int, rows: int, overlap: float = SLICE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """(y0, y1, x0, x1) pixel windows that cover the frame, each cols x rows cell grown by `overlap` of its size on every inner side."""
    out = []
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            x0, x1 = max(0.0, c * cw - overlap * cw), min(float(w), (c + 1) * cw + overlap * cw)
            y0, y1 = max(0.0, r * ch - overlap * ch), min(float(h), (r + 1) * ch + overlap * ch)
            out.append((int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))))
    return out


def _iou_ios(a, b) -> tuple[float, float]:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return (inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0), (inter / min(aa, ab) if min(aa, ab) > 0 else 0.0)


def merge_detections(dets: list[Detection], cut_edges: list[bool] | None = None) -> list[Detection]:
    """Greedy merge of the detections of the full frame and of its slices (the postprocess of SAHI, github.com/obss/sahi, MIT):
    best score first; a box is dropped when it is IoU >= MERGE_IOU with a kept one, or lies IOS >= MERGE_IOS inside a kept
    one while it touches an inner slice edge (`cut_edges[i]`, the person cut in half by a slice)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    kept: list[int] = []
    for i in order:
        drop = False
        for j in kept:
            o, ios = _iou_ios(dets[i].box, dets[j].box)
            if o >= MERGE_IOU or (ios >= MERGE_IOS and cut_edges is not None and cut_edges[i]):
                drop = True
                break
        if not drop:
            kept.append(i)
    return [dets[i] for i in kept]


class SlicedDetector:
    """Runs `base` on the whole frame and on overlapping slices of it (SAHI's sliced inference, implemented here: no dependency),
    so a small person gets more detector pixels. Cost: (1 + cols x rows) forward passes per frame. Boxes stay normalised to the frame."""

    def __init__(self, base, grid: tuple[int, int] | None = None, overlap: float = SLICE_OVERLAP):
        self.base, self.grid, self.overlap = base, grid or slice_grid(), overlap
        self.backend = f"{getattr(base, 'backend', 'detector')}+slices{self.grid[0]}x{self.grid[1]}"

    def __getattr__(self, name):  # path, size, res, model: whatever the pipeline reads from the base detector
        if name == "base" or (name.startswith("__") and name.endswith("__")):  # copy / pickle probe a bare instance: no base yet
            raise AttributeError(name)
        return getattr(self.base, name)

    def _run(self, frames: list[np.ndarray], cls: int = PERSON) -> list[list[Detection]]:
        out = []
        for i in range(0, len(frames), BATCH):
            out += self.base.detect(frames[i:i + BATCH], **({} if cls == PERSON else {"cls": cls}))
        return out

    def detect(self, frames: list[np.ndarray], cls: int = PERSON) -> list[list[Detection]]:
        full = self.base.detect(frames, **({} if cls == PERSON else {"cls": cls}))
        cols, rows = self.grid
        if cols * rows <= 1 or not frames:
            return full
        h, w = frames[0].shape[:2]
        wins = slice_windows(h, w, cols, rows, self.overlap)
        crops = [np.ascontiguousarray(f[y0:y1, x0:x1]) for f in frames for (y0, y1, x0, x1) in wins]
        got = self._run(crops, cls)
        eps = 0.02  # a box within 2 % of the crop edge, where the crop is not the frame edge, is probably cut by it
        res = []
        for k in range(len(frames)):
            dets, cut = list(full[k]), [False] * len(full[k])
            for m, (y0, y1, x0, x1) in enumerate(wins):
                for d in got[k * len(wins) + m]:
                    b = d.box
                    box = ((x0 + b[0] * (x1 - x0)) / w, (y0 + b[1] * (y1 - y0)) / h, (x0 + b[2] * (x1 - x0)) / w, (y0 + b[3] * (y1 - y0)) / h)
                    touches = (x0 > 0 and b[0] <= eps) or (x1 < w and b[2] >= 1 - eps) or (y0 > 0 and b[1] <= eps) or (y1 < h and b[3] >= 1 - eps)
                    dets.append(Detection(d.score, box))
                    cut.append(touches)
            res.append(merge_detections(dets, cut))
        return res


def for_tap(det):
    """The detector as tap mode wants it: sliced (SlicedDetector) when INFRAME_SLICES names a grid, else as it is."""
    return det if slice_grid() == (1, 1) else SlicedDetector(det)
