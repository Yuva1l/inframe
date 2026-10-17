"""Child process for tracks.Tracker: runs roboflow/trackers and talks pickle over stdin/stdout.

It is a separate process on purpose: the `trackers` wheel pulls in OpenCV, whose bundled FFmpeg libraries clash with
PyAV's inside one process (macOS prints "Class AVFFrameReceiver is implemented in both", and warns of crashes).
The parent never imports cv2.
"""
from __future__ import annotations

import pickle
import sys


def align_ids(n: int, out_idx, out_ids):
    """Ids in the order of the n input detections. The library returns its own order (matched first, then new,
    then low-confidence; BoT-SORT also drops detections), so every output row carries the index of the input row it
    came from (`data["idx"]`); an input row the library did not return gets -1."""
    import numpy as np

    ids = np.full(n, -1, np.int64)
    if out_ids is not None and out_idx is not None and len(out_idx):
        ids[np.asarray(out_idx, np.int64)] = np.asarray(out_ids, np.int64)
    return ids


def main() -> int:
    proto_in, proto_out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # anything the libraries print must not corrupt the protocol

    def send(obj) -> None:
        pickle.dump(obj, proto_out, protocol=4)
        proto_out.flush()

    tracker = None
    while True:
        try:
            msg = pickle.load(proto_in)
        except EOFError:
            return 0
        op = msg[0]
        try:
            if op in ("init", "reset"):
                if op == "init":
                    _, kind, kwargs = msg
                    import trackers

                    cls = {"ocsort": trackers.OCSORTTracker, "botsort": trackers.BoTSORTTracker}[kind]
                tracker = cls(**kwargs)  # reset: a fresh tracker with the same settings (a new shot)
                send(("ready", None))
            elif op == "update":
                import numpy as np
                import supervision as sv

                _, t, xyxy, conf, frame = msg
                xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
                det = sv.Detections(xyxy=xyxy, confidence=np.asarray(conf, np.float64), data={"idx": np.arange(len(xyxy))})
                out = tracker.update(det, frame=frame, timestamp=t)
                idx = out.data.get("idx") if out.data else None
                if idx is None and len(out):
                    raise RuntimeError("the tracker library dropped the detection index; cannot map ids back")
                send(("ids", align_ids(len(xyxy), idx, out.tracker_id)))
            elif op == "close":
                send(("bye", None))
                return 0
        except Exception as e:  # report and keep serving; the parent decides what to do
            send(("error", f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    raise SystemExit(main())
