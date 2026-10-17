"""Zone: only the part of the picture that counts (a spot where the tries happen), for a static camera.

The zone is a rectangle (x, y, w, h), every number a fraction 0..1 of the UPRIGHT display frame: the picture as a player
shows it, after the display matrix (rotation and flip) is applied. Frames, detector boxes and the motion mask are all upright
already (decode.py applies the matrix), so the zone needs no conversion for any rotation; it is a rectangle of what you see.

Presence: a person counts when the CENTRE of the box is inside the zone (default), or, with `min_overlap`, when at least
that fraction of the box area is inside (a person who is half out of the zone still counts at 0.5; a rectangle zone and a
box whose centre is outside overlap by at most 0.5, so values below 0.5 widen the rule and values above 0.5 tighten it).
Motion mode counts the foreground pixels inside the zone.

The zone makes sense for a static camera only: with a moving camera the same spot of the world is not the same place in the
picture. The pipeline runs camera.check_static first and, if the camera moves, warns and drops the zone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detect import Detection
from .target import Hit, Target

EPS = 1e-9


@dataclass(frozen=True)
class Zone:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self):
        vals = (self.x, self.y, self.w, self.h)
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in vals):
            raise ValueError(f"--zone needs four numbers x,y,w,h (got {vals})")
        if self.x < 0 or self.y < 0 or self.w <= 0 or self.h <= 0 or self.x + self.w > 1 + 1e-6 or self.y + self.h > 1 + 1e-6:
            raise ValueError("--zone x,y,w,h are fractions of the upright frame: x and y from 0, w and h above 0, "
                             f"x+w and y+h at most 1 (got {self.x:g},{self.y:g},{self.w:g},{self.h:g})")

    @classmethod
    def parse(cls, text: str) -> "Zone":
        try:
            vals = [float(p) for p in str(text).replace(" ", "").split(",")]
        except ValueError:
            raise ValueError(f"--zone must be four numbers x,y,w,h separated by commas (got {text!r})") from None
        if len(vals) != 4:
            raise ValueError(f"--zone must be four numbers x,y,w,h separated by commas (got {len(vals)} in {text!r})")
        return cls(*vals)

    @classmethod
    def of(cls, v) -> "Zone":
        """A Zone from a Zone, a 'x,y,w,h' string or a 4-sequence."""
        if isinstance(v, Zone):
            return v
        return cls.parse(v) if isinstance(v, str) else cls(*[float(p) for p in v])

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    def contains_point(self, px: float, py: float) -> bool:
        return self.x - EPS <= px <= self.x + self.w + EPS and self.y - EPS <= py <= self.y + self.h + EPS

    def overlap(self, box) -> float:
        """Fraction (0..1) of the box (x0, y0, x1, y1) that lies inside the zone."""
        x0, y0, x1, y1 = box
        area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        if area <= 0:
            return 0.0
        iw = min(x1, self.x + self.w) - max(x0, self.x)
        ih = min(y1, self.y + self.h) - max(y0, self.y)
        return max(iw, 0.0) * max(ih, 0.0) / area

    def admits(self, box, min_overlap: float | None = None) -> bool:
        """Centre rule, or the overlap rule when `min_overlap` is given."""
        if min_overlap is None:
            return self.contains_point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        return self.overlap(box) >= min_overlap - EPS

    def mask(self, height: int, width: int) -> np.ndarray:
        """Boolean (height, width) mask of the pixels inside the zone; never empty (at least one pixel)."""
        x0, x1 = int(round(self.x * width)), int(round((self.x + self.w) * width))
        y0, y1 = int(round(self.y * height)), int(round((self.y + self.h) * height))
        m = np.zeros((height, width), bool)
        m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True
        return m


def check_overlap(v: float | None) -> float | None:
    if v is not None and not (isinstance(v, (int, float)) and 0.0 < v <= 1.0):
        raise ValueError(f"--zone-overlap must be a fraction above 0 and at most 1 (got {v})")
    return v


def zone_from_target_files(specs: list[str] | None) -> Zone | None:
    """The zone drawn on the pick page, stored in target.json: read from the first `tap:<file>.json` spec that holds one."""
    for s in specs or []:
        if s.startswith("tap:") and s.endswith(".json"):
            try:
                z = json.loads(Path(s[4:]).expanduser().read_text()).get("zone")
            except (OSError, ValueError, AttributeError):
                continue
            if z:
                return Zone.of(z)
    return None


class ZoneTarget:
    """Wraps a Target so only people in the zone count.

    A target that does not look at pixels or remember anything (`stateless`, like AnyPerson) gets only the candidates the zone
    admits, so a person outside it cannot hide one inside. An identity target (tap, photos) sees every candidate, because who
    is who needs the others for contrast, and its answer is dropped when its box is outside the zone."""

    def __init__(self, inner: Target, zone: Zone, min_overlap: float | None = None):
        self.inner, self.zone, self.min_overlap = inner, zone, min_overlap
        self.stateless = bool(getattr(inner, "stateless", False))
        if hasattr(inner, "finalize"):  # the pipeline calls finalize() only on targets that decide at the end (tap mode)
            self.finalize = lambda: [self._gate(h) for h in inner.finalize()]

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def wants_tracks(self) -> bool:
        return self.inner.wants_tracks

    def __getattr__(self, item):  # wants_slices, decode_long_side, embedder, describe, on_cut ...: the inner target's
        if item in ("inner", "zone", "min_overlap", "stateless", "finalize") or item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.inner, item)

    def _gate(self, h: Hit) -> Hit:
        if h.box is not None and not self.zone.admits(h.box, self.min_overlap):
            return Hit(0.0, None)
        return h

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        if self.stateless:
            return self.inner.evaluate(t, frame, [d for d in candidates if self.zone.admits(d.box, self.min_overlap)])
        return self._gate(self.inner.evaluate(t, frame, candidates))
