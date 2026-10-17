"""One job at a time for `inframe serve` (docs/T9A_DESIGN.md section 2.6, routes 46-48).

A job (`run`, `rank`, `people`, `export`, `target-add`, `music-auto`) runs in a daemon thread. Its `progress(fraction, message)` callback
stores both and raises `JobCancelled` once cancel was asked for (the MCP `_run_exclusive` pattern), so a cancel takes effect at
the next progress report. A second start while one runs is `Busy` (409 `{"error": "busy", "job": <running job>}`). Finished jobs
stay readable for 10 minutes.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .serve_http import HttpError

KINDS = ("run", "rank", "people", "export", "target-add", "music-auto")
KEEP_FINISHED_S = 600.0


class JobCancelled(Exception):
    """Raised by the progress callback after `JobRunner.cancel`."""


class NeedsConfirm(Exception):
    """Raised by a job function that cannot decide on its own (photo mode, `identity.Ambiguous`): the job ends in state
    `needs_confirm`, `error` holds the message and `result` what the UI needs to ask."""

    def __init__(self, message: str, result=None):
        super().__init__(message)
        self.result = result


class Busy(HttpError):
    def __init__(self, job: dict):
        super().__init__(409, "busy", job=job)


@dataclass
class Job:
    id: str
    kind: str
    media_id: str | None = None
    run_id: str | None = None
    state: str = "running"  # running | done | failed | cancelled | needs_confirm
    fraction: float = 0.0
    message: str = ""
    started: float = field(default_factory=time.time)  # wall clock (epoch seconds)
    result: object = None
    error: str | None = None
    _t0: float = 0.0  # monotonic
    _end: float | None = None  # monotonic, when it finished
    _progress_at: float = 0.0  # monotonic time of the last progress report
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_json(self, now: float) -> dict:
        end = self._end if self._end is not None else now
        return {"id": self.id, "kind": self.kind, "state": self.state, "fraction": round(self.fraction, 4), "message": self.message,
                "started": round(self.started, 3), "elapsed_s": round(max(0.0, end - self._t0), 3), "media_id": self.media_id,
                "run_id": self.run_id, "result": self.result, "error": self.error}


class JobRunner:
    def __init__(self, clock: Callable[[], float] = time.monotonic, keep_s: float = KEEP_FINISHED_S):
        self._clock, self._keep = clock, keep_s
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------ reading
    def _purge(self) -> None:
        now = self._clock()
        for jid in [j.id for j in self._jobs.values() if j._end is not None and now - j._end > self._keep]:
            del self._jobs[jid]
            self._threads.pop(jid, None)

    def _running(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.state == "running"), None)

    def current(self) -> dict | None:
        with self._lock:
            self._purge()
            j = self._running()
            return j.to_json(self._clock()) if j else None

    def get(self, jid: str) -> dict:
        with self._lock:
            self._purge()
            j = self._jobs.get(jid)
            if j is None:
                raise HttpError(404, f"no such job: {jid}")
            return j.to_json(self._clock())

    # ------------------------------------------------------------ writing
    def start(self, kind: str, fn: Callable[[Callable[[float, str], None]], object], media_id: str | None = None,
              run_id: str | None = None) -> dict:
        """Start `fn(progress)` in a thread and return the job (state `running`). `fn`'s return value becomes `result`.
        Raises `Busy` when a job is running."""
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        with self._lock:
            self._purge()
            cur = self._running()
            if cur is not None:
                raise Busy(cur.to_json(self._clock()))
            job = Job(id=secrets.token_hex(6), kind=kind, media_id=media_id, run_id=run_id)
            job._t0 = job._progress_at = self._clock()
            self._jobs[job.id] = job
            t = threading.Thread(target=self._run, args=(job, fn), name=f"inframe-job-{job.id}", daemon=True)
            self._threads[job.id] = t
            snap = job.to_json(self._clock())
            t.start()
        return snap

    def _run(self, job: Job, fn) -> None:
        def progress(fraction: float, message: str = "") -> None:
            with self._lock:
                job.fraction = min(1.0, max(0.0, float(fraction)))
                job.message = str(message)
                job._progress_at = self._clock()
            if job._cancel.is_set():
                raise JobCancelled()

        state, result, err = "done", None, None
        try:
            result = fn(progress)
        except JobCancelled:
            state = "cancelled"
        except NeedsConfirm as e:
            state, result, err = "needs_confirm", e.result, str(e)
        except BaseException as e:  # a job must always end in a final state
            state, err = "failed", str(e) or type(e).__name__
        with self._lock:
            job.result, job.error = result, err
            if state == "done":
                job.fraction = 1.0
            job.state = state
            job._end = self._clock()

    def cancel(self, jid: str) -> dict:
        """Ask a running job to stop; it ends `cancelled` at its next progress report. A finished job is returned unchanged."""
        with self._lock:
            j = self._jobs.get(jid)
            if j is None:
                raise HttpError(404, f"no such job: {jid}")
            if j.state == "running":
                j._cancel.set()
            return j.to_json(self._clock())

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel the running job and wait for its thread (at stop)."""
        with self._lock:
            j = self._running()
            t = self._threads.get(j.id) if j else None
            if j:
                j._cancel.set()
        if t is not None:
            t.join(timeout)
