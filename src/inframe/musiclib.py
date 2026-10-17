"""The songs that ship with inframe: `src/inframe/music_library/` holds a few CC BY (or CC0) songs, `library.json` describes them and
`CREDITS.md` has their credit lines. They work offline; online search (music.py) is the "more songs" part.

Built by tools/build_music_library.py (it measures tempo, length and loudness by code). A bundled song's id is its Openverse id, so
`music.get(id)`, `--music ID` and the serve routes find it without a download. Its record has the shape of the `<id>.json` the
music cache writes (title, creator, license, license_url, ...), so credits and license checks treat it like any other track.

`INFRAME_MUSIC_LIBRARY=/folder` reads another library folder; `off` (or `0`, or empty) means no bundled songs (the tests use it)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

LIBRARY_ENV = "INFRAME_MUSIC_LIBRARY"
PACKAGED = Path(__file__).resolve().parent / "music_library"
SCHEMA = 1
REQUIRED = ("id", "title", "artist", "license", "license_url", "source_url", "duration_s", "bpm", "sha256", "file")
TOTAL_BUDGET_BYTES = 25 * 10**6  # 25 MB, the size the library stays within


def library_dir() -> Path | None:
    raw = os.environ.get(LIBRARY_ENV)
    if raw is None:
        return PACKAGED
    if raw.strip().lower() in ("", "0", "off", "none"):
        return None
    return Path(raw).expanduser()


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> tuple[dict, ...]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ()
    out = []
    for s in doc.get("songs") or []:
        if isinstance(s, dict) and all(s.get(k) not in (None, "") for k in REQUIRED) and s["license"] in ("cc0", "by") \
                and "/" not in str(s["file"]) and "\\" not in str(s["file"]) and not str(s["file"]).startswith("."):
            out.append(s)
    return tuple(out)


def songs(cc0_only: bool = False) -> list[dict]:
    """The library.json entries whose audio file is there, in file order; only CC0 ones with `cc0_only`."""
    d = library_dir()
    if d is None:
        return []
    f = d / "library.json"
    try:
        mtime = f.stat().st_mtime
    except OSError:
        return []
    return [s for s in _load(str(f), mtime) if (d / s["file"]).is_file() and (s["license"] == "cc0" or not cc0_only)]


def find(track_id: str, cc0_only: bool = False) -> dict | None:
    for s in songs(cc0_only):
        if s["id"] == track_id:
            return s
    return None


def audio_path(song: dict) -> Path:
    d = library_dir()
    assert d is not None
    return d / song["file"]


def record(song: dict) -> dict:
    """The song as the `<id>.json` record of the music cache (what credits_text and the license checks read)."""
    cc0 = song["license"] == "cc0"
    tasl = tasl_line(song)
    return dict(id=song["id"], title=song["title"], creator=song["artist"], creator_url=song.get("artist_url"), license=song["license"],
                license_version=song.get("license_version"), license_url=song["license_url"], url=song.get("download_url") or song["source_url"],
                landing_url=song["source_url"], duration_s=song["duration_s"], attribution=None if cc0 else tasl, source=song.get("source") or "jamendo",
                filetype=Path(song["file"]).suffix.lstrip("."), bundled=True, bpm=song["bpm"], tags=list(song.get("tags") or []),
                genres=list(song.get("genres") or []), transcoded=bool(song.get("transcoded")))


def tasl_line(song: dict) -> str:
    """Title, Author, Source, License: the credit line a CC BY song needs."""
    ver = f" {song['license_version']}" if song.get("license_version") else ""
    line = (f'"{song["title"]}" by {song["artist"]} ({song["source_url"]}), licensed under '
            f'{"CC0 1.0" if song["license"] == "cc0" else "CC BY" + ver} ({song["license_url"]}).')
    if song.get("transcoded"):
        line += " Changes were made: converted to AAC 160 kbit/s."
    return line


def song_for_path(path) -> dict | None:
    """The library entry whose audio file is `path` (a bundled song's file), else None."""
    d = library_dir()
    if d is None:
        return None
    try:
        p = Path(path).resolve()
    except OSError:
        return None
    if p.parent != d.resolve():
        return None
    for s in songs():
        if s["file"] == p.name:
            return s
    return None


def listing(cc0_only: bool = False) -> list[dict]:
    """Songs for the UI and the CLI: the public fields, no paths."""
    out = []
    for s in songs(cc0_only):
        out.append(dict(id=s["id"], title=s["title"], creator=s["artist"], license=s["license"], license_version=s.get("license_version"),
                        license_url=s["license_url"], landing_url=s["source_url"], duration_s=s["duration_s"], bpm=s["bpm"],
                        tags=list(s.get("tags") or []), moods=list(s.get("moods") or []), bytes=s.get("bytes"), bundled=True,
                        needs_credit=s["license"] != "cc0"))
    return out
