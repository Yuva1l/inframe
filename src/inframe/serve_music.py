"""Music routes of `inframe serve` (docs/T9A_DESIGN.md section 3.4, rows 49-52): search, download into the music cache, play from
the cache, and a native choose-file dialog. Every outbound request goes through `music` -> `net.fetch`. The songs bundled in the package (musiclib) are listed and played
without any request.

The search query is the mood field and nothing else (at most 40 characters of letters, digits, `_`, space, `-`); the UI never
fills it from file, folder or target names.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from . import music, musiclib
from .serve_http import HttpError, Request, Route, file_response, json_response
from .serve_paths import AUDIO_EXTS, check_audio

MOOD_RE = re.compile(r"[\w -]{0,40}")
MAX_MIN_S = 36000.0  # ten hours
MAX_LIMIT = 50
DIALOG_TIMEOUT_S = 600
AUDIO_TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".wav": "audio/wav", ".flac": "audio/flac",
               ".ogg": "audio/ogg", ".opus": "audio/ogg"}
_download = threading.Lock()  # two downloads of one track would share a .part file


def _is_mac() -> bool:
    return sys.platform == "darwin"


def _flag(raw, name: str) -> bool:
    if raw in (None, "", "0", "false", False):
        return False
    if raw in ("1", "true", True):
        return True
    raise HttpError(400, f"{name} must be 0 or 1")


def _number(raw, name: str, lo: float, hi: float, default: float, integer: bool = False) -> float:
    if raw in (None, ""):
        return default
    try:
        x = float(raw)
    except (TypeError, ValueError):
        raise HttpError(400, f"{name} is not a number") from None
    if not math.isfinite(x) or x < lo or x > hi or (integer and x != int(x)):
        raise HttpError(400, f"{name} must be {'a whole number ' if integer else ''}from {lo:g} to {hi:g}")
    return x


def _track_id(raw) -> str:
    if not isinstance(raw, str) or not music.ID_RE.match(raw):
        raise HttpError(400, "not a track id")
    return raw


# ---------------------------------------------------------------- 48
LIBRARY_NOTICE = "These songs come with inframe and work offline. Each keeps its own license; a CC BY song's credit line is written next to the reel."


def library(app, req: Request):
    """The bundled songs (CC0 only with `cc0_only=1`), no network: id, title, creator, license, length, BPM, tags."""
    cc0_only = _flag(req.query.get("cc0_only"), "cc0_only")
    return json_response({"tracks": musiclib.listing(cc0_only), "notice": LIBRARY_NOTICE})


# ---------------------------------------------------------------- 49
def search(app, req: Request):
    mood = req.query.get("mood", "")
    if not MOOD_RE.fullmatch(mood):
        raise HttpError(400, "mood: at most 40 letters, digits, spaces, _ or -")
    min_s = _number(req.query.get("min_s"), "min_s", 0, MAX_MIN_S, 0.0)
    limit = int(_number(req.query.get("limit"), "limit", 1, MAX_LIMIT, 10, integer=True))
    cc0_only = _flag(req.query.get("cc0_only"), "cc0_only")
    try:
        tracks = music.search(mood.strip() or None, min_s, limit, cc0_only=cc0_only)
    except music.MusicError as e:
        raise HttpError(502, str(e)) from e
    return json_response({"tracks": [t.to_json() for t in tracks], "notice": music.OPENVERSE_NOTICE})


# ---------------------------------------------------------------- 50
def _file_json(track_id: str, audio: Path) -> dict:
    meta = music.load_meta(audio) or {}
    lic = meta.get("license")
    return {"id": track_id, "file": str(audio), "license": lic, "attribution": meta.get("attribution"),
            "duration_s": meta.get("duration_s"), "needs_credit": music.needs_credit(lic)}


def get(app, req: Request):
    body = req.json or {}
    track_id = _track_id(body.get("id"))
    cc0_only = body.get("cc0_only", False)
    if not isinstance(cc0_only, bool):
        raise HttpError(400, "cc0_only must be true or false")
    try:
        with _download:
            audio = music.get(track_id, cc0_only=cc0_only)
    except music.LicenseError as e:
        raise HttpError(422, str(e)) from e
    except music.MusicError as e:
        raise HttpError(502, str(e)) from e
    return json_response(_file_json(track_id, audio))


# ---------------------------------------------------------------- 51
def cached_audio(track_id: str) -> Path:
    """`<music cache>/<id>.<ext>` as a regular file (no symlink) of an audio type, found by a directory scan; 404 when absent."""
    folder = music.cache_dir()
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name)
    except OSError:
        raise HttpError(404, "no such track in the music cache") from None
    for e in entries:
        stem, dot, ext = e.name.rpartition(".")
        if stem == track_id and dot and "." + ext.lower() in AUDIO_EXTS and e.is_file(follow_symlinks=False):
            return Path(e.path)
    raise HttpError(404, "no such track in the music cache")


def audio(app, req: Request):
    track_id = _track_id(req.params["id"])
    song = musiclib.find(track_id)
    if song is not None:  # package data, not a path the caller named: it may sit in a hidden folder (.venv, ~/.local) that check_read refuses
        path = musiclib.audio_path(song)
        if path.is_symlink() or not path.is_file():
            raise HttpError(404, "no such track in the library")
        return file_response(path, AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream"), req)
    path = check_audio(cached_audio(track_id))
    return file_response(path, AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream"), req)


# ---------------------------------------------------------------- 52
def _choose_audio() -> str | None:
    """The macOS open panel for one audio file: its POSIX path, or None when the user cancels."""
    script = 'POSIX path of (choose file with prompt "Choose a music file" of type {"public.audio"})'
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=DIALOG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    except OSError as e:
        raise HttpError(501, f"the file dialog is not available: {e}") from e
    if r.returncode != 0:
        if "cancel" in r.stderr.lower():  # "User canceled." (-128)
            return None
        raise HttpError(500, f"the file dialog failed: {r.stderr.strip()[:200]}")
    return r.stdout.strip() or None


def choose_file(app, req: Request):
    if not _is_mac():
        raise HttpError(501, "the file dialog needs macOS: use the file browser")
    chosen = _choose_audio()
    if chosen is None:
        return json_response({"cancelled": True})
    path = check_audio(chosen)
    meta = music.load_meta(path)
    return json_response({"file": str(path), "license": (meta or {}).get("license")})


ROUTES = [
    Route("GET", r"/api/music/library", library),
    Route("GET", r"/api/music/search", search),
    Route("POST", r"/api/music/get", get),
    Route("GET", r"/api/music/(?P<id>[^/]+)/audio", audio, media=True),
    Route("POST", r"/api/music/choose-file", choose_file),
]


def attach(app) -> None:
    """Nothing to set up: the music cache is a folder, not state."""
