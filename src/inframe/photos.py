"""Read videos from the macOS Photos library (T3a): list them, resolve `photos:<uuid>` to the original file.

Only five fields of osxphotos' PhotoInfo are read: uuid, date, slow_mo, ismissing and path. Nothing else (no location,
people, albums, keywords or names) is touched, copied or printed. The original is read in place, never copied.

Reading a Photos library needs Full Disk Access for the app that runs inframe (Terminal, an editor, ...); only the
person at the Mac can grant it. osxphotos is imported lazily so the rest of inframe works without it.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

PREFIX = "photos:"
PLAYBACK_FPS = 30.0  # Photos plays a slow-motion clip back at 30 fps; the capture rate over this is the slow-down factor

FDA_HELP = (
    "inframe cannot read the Photos library: macOS blocks it until the app that runs inframe has Full Disk Access. "
    "Open System Settings > Privacy & Security > Full Disk Access, switch on Terminal (or the app you run inframe from), "
    "quit and reopen that app, then run the command again."
)
ICLOUD_HELP = (
    "the original of this video is not on this Mac (it is in iCloud only). Open Photos, select the video and choose "
    "File > Download Original (or turn on Settings > iCloud > Download Originals to this Mac), wait until it finishes, then run again."
)


class PhotosError(RuntimeError):
    """A problem the person can fix; the message says how."""


class FullDiskAccessError(PhotosError):
    pass


class ICloudOnlyError(PhotosError):
    pass


@dataclass(frozen=True)
class PhotoRecord:
    uuid: str
    date: str  # ISO 8601 with the offset Photos stores
    slow_mo: bool
    ismissing: bool  # the original is not on disk (iCloud only)
    path: str | None  # the original in the library, or None when missing

    def to_dict(self) -> dict:
        return asdict(self)


def is_photos_ref(video: str) -> bool:
    return str(video).startswith(PREFIX)


def parse_ref(video: str) -> str:
    """'photos:<uuid>' -> '<uuid>' (upper-cased, as Photos stores it)."""
    u = str(video)[len(PREFIX):].strip()
    if not u or any(c in u for c in "/\\ ") or len(u) > 64:
        raise PhotosError(f"not a Photos reference: {video!r}; expected photos:<uuid> (see `inframe photos list`)")
    return u.upper()


def parse_when(text: str) -> dt.datetime:
    """YYYY-MM-DD or a full ISO date-time; naive values are local time (osxphotos' rule)."""
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        raise PhotosError(f"not a date: {text!r}; use YYYY-MM-DD or YYYY-MM-DDTHH:MM") from None


def _import_osxphotos():
    try:
        import osxphotos
    except ImportError as e:
        raise PhotosError("reading Photos needs osxphotos (macOS only): run `uv sync --extra photos`") from e
    return osxphotos


def _no_access(e: BaseException) -> bool:
    if isinstance(e, PermissionError):
        return True
    if isinstance(e, sqlite3.Error):
        m = str(e).lower()
        return "unable to open" in m or "authorization" in m or "not permitted" in m
    return isinstance(e, OSError) and e.errno in (1, 13)  # EPERM, EACCES


def open_db(library: str | None = None):
    """A PhotosDB for `library` (a .photoslibrary folder) or, when None, the last library Photos opened."""
    osx = _import_osxphotos()
    if library is None:
        from osxphotos.utils import get_last_library_path, get_system_library_path

        library = get_last_library_path() or get_system_library_path()
        if not library:
            # With no Full Disk Access the preference file that names the library cannot be read either.
            raise FullDiskAccessError(f"inframe could not find the Photos library. {FDA_HELP} "
                                      "If you use a library in another place, pass --library <path to the .photoslibrary>.")
    lib = Path(library).expanduser()
    if not lib.exists():
        raise PhotosError(f"no such Photos library: {lib}")
    try:
        list(os.scandir(lib / "database" if (lib / "database").is_dir() else lib))
        return osx.PhotosDB(dbfile=str(lib))
    except Exception as e:  # noqa: BLE001 - sorted below into "no access" and "other"
        if _no_access(e):
            raise FullDiskAccessError(FDA_HELP) from e
        if type(e).__name__ == "PhotosDBReadError" or isinstance(e, FileNotFoundError):
            raise PhotosError(f"{lib} is not a Photos library osxphotos can read: {e}. "
                              "Pass the .photoslibrary folder to --library.") from e
        raise


def record_of(p) -> PhotoRecord:
    """The five fields, nothing else."""
    d = p.date
    return PhotoRecord(uuid=str(p.uuid), date=d.isoformat() if hasattr(d, "isoformat") else str(d),
                       slow_mo=bool(p.slow_mo), ismissing=bool(p.ismissing), path=p.path)


def _when(date: str) -> float:
    """Sort key for an ISO date string: the instant it names, so '10:00+02:00' sorts before '09:00+00:00' (a plain string compare puts
    the later instant first when the UTC offsets differ). A naive date counts as local time; an unparsable one sorts last."""
    try:
        return dt.datetime.fromisoformat(date).timestamp()
    except (ValueError, OverflowError, OSError):
        return float("inf")


def list_videos(db, since: dt.datetime | None = None, until: dt.datetime | None = None,
                slomo_only: bool = False) -> list[PhotoRecord]:
    """Videos in the library, oldest first. `since` is inclusive, `until` exclusive."""
    out = [record_of(p) for p in db.photos(images=False, movies=True, from_date=since, to_date=until)]
    if slomo_only:
        out = [r for r in out if r.slow_mo]
    return sorted(out, key=lambda r: (_when(r.date), r.uuid))


def resolve(db, uuid: str) -> PhotoRecord:
    """The record of one video whose original is on disk; clear errors otherwise."""
    found = db.photos(uuid=[uuid], images=False, movies=True)
    if not found:
        raise PhotosError(f"no video with uuid {uuid} in this Photos library (is it a photo, or in Recently Deleted?)")
    r = record_of(found[0])
    if r.ismissing or not r.path:
        raise ICloudOnlyError(f"{r.uuid}: {ICLOUD_HELP}")
    if not Path(r.path).is_file():
        raise ICloudOnlyError(f"{r.uuid}: Photos lists the original but the file is not on disk. {ICLOUD_HELP}")
    return r


def slowmo_factor(fps: float, playback_fps: float = PLAYBACK_FPS) -> float:
    """Constant slow-down factor of a slow-motion original: capture rate over playback rate (240 fps -> 8, 120 fps -> 4).
    The ramp inside Photos' slow-motion edit cannot be read from the file, so one
    constant factor stands for the whole clip. 1.0 when the file is not faster than playback."""
    if not fps or fps <= playback_fps * 1.5:
        return 1.0
    return float(round(fps / playback_fps))


def time_map(slow_mo: bool, fps: float) -> dict:
    """`fps` is the capture rate: pass r_frame_rate (probe.capture_fps), not the average, which a 30 fps pre-roll lowers."""
    f = slowmo_factor(fps) if slow_mo else 1.0
    return dict(factor=f, source_fps=round(fps, 3), playback_fps=PLAYBACK_FPS,
                rule="playback_s = source_s * factor (constant factor; the Photos ramp is not read)")


def annotate(doc: dict, rec: PhotoRecord, capture_fps: float | None = None) -> dict:
    """Add the Photos facts and the source-to-playback time map to a finished run's segments.json document.
    Times in `clips` and `presence_intervals` stay in source (file) time: that is what the clips were cut at."""
    fps = float(doc["source"]["fps"])
    if capture_fps and capture_fps >= fps:  # r_frame_rate; the average in the document is lower after a 30 fps pre-roll
        fps = float(capture_fps)
    tm = time_map(rec.slow_mo, fps)
    f = tm["factor"]
    doc["photos"] = dict(uuid=rec.uuid, date=rec.date, slow_mo=rec.slow_mo, time_map=tm)
    for c in doc["clips"]:
        c["playback_start"] = round(c["start"] * f, 3)
        c["playback_end"] = round(c["end"] * f, 3)
    doc["presence_playback"] = [[round(s * f, 3), round(e * f, 3)] for s, e in doc["presence_intervals"]]
    return doc


def playback_time(s: float, factor: float, head_s: float = 0.0, tail_from_s: float | None = None,
                  offset_s: float = 0.0) -> float:
    """Where source time `s` lands on the export's timeline. Photos' slow-motion edit plays the head of the clip at 1x
    (the first `head_s` source seconds), the middle at `factor`, and the tail from `tail_from_s` on at 1x again; a trim
    that cuts `offset_s` playback seconds off the start of the export shifts everything earlier by that much."""
    head = min(s, max(head_s, 0.0))
    tail = float("inf") if tail_from_s is None else max(tail_from_s, head)
    return head + max(0.0, min(s, tail) - head) * factor + max(0.0, s - tail) - offset_s


def compare_tries(original: dict, export: dict, factor: float | None = None, tol: float = 0.3,
                  head_s: float = 0.0, tail_from_s: float | None = None, offset_s: float = 0.0) -> dict:
    """Do an original's tries (presence intervals) match the same session's 30 fps export?

    Count and order must be equal. Each boundary of the original is mapped onto the export's timeline (see
    playback_time: constant `factor` by default; `head_s` and `tail_from_s` are the source seconds Photos plays at 1x
    before and after the slow part, `offset_s` the playback seconds a trim cut off the export's start) and must lie
    within tol * (1 + factor) playback seconds of the export's boundary: tol on the export's own timeline plus tol of
    real time on the original's (the regression contract is 0.3 s), which the factor stretches. Returns a report with `ok`."""
    if factor is None:
        factor = ((original.get("photos") or {}).get("time_map") or {}).get("factor")
        if factor is None:
            factor = slowmo_factor(float(original["source"]["fps"]))
    a = [tuple(x) for x in original["presence_intervals"]]
    b = [tuple(x) for x in export["presence_intervals"]]
    allow = tol * (1.0 + factor)
    rows = []
    err = 0.0

    def to_export(x: float) -> float:
        return playback_time(x, factor, head_s, tail_from_s, offset_s)

    for i, ((s1, e1), (s2, e2)) in enumerate(zip(a, b), 1):
        m1, m2 = to_export(s1), to_export(e1)
        ds, de = abs(m1 - s2), abs(m2 - e2)
        err = max(err, ds, de)
        rows.append(dict(try_=i, original_playback=[round(m1, 3), round(m2, 3)],
                         export=[round(s2, 3), round(e2, 3)], start_err_s=round(ds, 3), end_err_s=round(de, 3)))
    same_count = len(a) == len(b)
    ok = same_count and bool(a) and err <= allow  # two files with no tries match on nothing: not a pass
    return dict(ok=ok, n_original=len(a), n_export=len(b), same_count=same_count, factor=factor,
                head_s=head_s, tail_from_s=tail_from_s, offset_s=offset_s,
                tolerance_playback_s=round(allow, 3), max_err_playback_s=round(err, 3), tries=rows)
