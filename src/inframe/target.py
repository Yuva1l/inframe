"""What counts as 'the target'. v0 has AnyPerson; the embedding gallery (DINOv2-S etc.) plugs in later
by implementing the same interface and using the frame pixels plus the detector's candidates.

A Target is stateful: the pipeline calls evaluate() once per sample, in time order, so an implementation may remember
what it saw (which track is the target, a running gallery). When `wants_tracks` is true and the sample rate is at
least tracks.MIN_TRACK_FPS, each candidate arrives with a `track_id` from the tracker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .classes import PERSON
from .detect import Detection


@dataclass
class Hit:
    score: float  # 0 when the target is not in the frame
    box: tuple[float, float, float, float] | None = None
    track_id: int | None = None  # the tracker's id of the box, when there is one


class Target(Protocol):
    name: str
    cls: int  # COCO id of the class the detector looks for (classes.py); a target without it looks for persons
    wants_tracks: bool  # True: run the tracker (when the sample rate allows) and fill in Detection.track_id

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        """Score how sure we are that the target is in `frame` (H, W, 3 uint8, upright), sampled at `t` seconds,
        given the detector's person candidates (best first). Called in time order."""
        ...


class AnyPerson:
    """Every person is the target. Score = best person score; persons whose box is shorter than
    `min_box` (fraction of frame height) are ignored."""

    name = "any-person"
    cls = PERSON
    wants_tracks = False  # identity does not matter: every person counts
    stateless = True  # looks at neither the pixels nor the past: a wrapper may filter the candidates before it and call it late

    def __init__(self, min_box: float = 0.0):
        self.min_box = min_box

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        for d in candidates:  # best first
            if d.box[3] - d.box[1] >= self.min_box:
                return Hit(d.score, d.box, d.track_id)
        return Hit(0.0, None)


class AnyOfClass(AnyPerson):
    """Every instance of one class is the target (`--target class:dog`): the same rule as AnyPerson for another COCO class.
    Score = best score of the class; boxes shorter than `min_box` (fraction of frame height) are ignored."""

    wants_tracks = False

    def __init__(self, cls: int, min_box: float = 0.0):
        from .classes import class_name

        super().__init__(min_box)
        self.cls = cls
        self.name = f"class:{class_name(cls)}"
