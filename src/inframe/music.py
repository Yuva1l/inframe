"""Free music for the reel: search and download from Openverse (api.openverse.org). CC0 and CC BY tracks by default (decided
2026-10-16: Openverse has no CC0 tracks in its music category); `cc0_only` (the CLI flag --cc0-only) restores CC0 alone. A CC BY
track needs a credit line, which is written to `<video>_credits.txt` next to the reel without any flag.

- Anonymous use needs no key. Openverse's terms: credit CC works, no scraping of the catalogue, no spreading requests over
  several machines to get around rate limits, and a product built on the API says it uses Openverse and is not endorsed by it
  (README, Licenses). Openverse indexes metadata and does not check licenses, so every track found is kept only when its
  license field is `cc0` (or `by` unless cc0_only, and then the record must name its license URL); `by-sa`, `nc` and `nd` are dropped.
- Source `jamendo` gives full songs. Jamendo tracks may be claimed by Content ID systems [unverified]; the credit is
  written to `<video>_credits.txt` next to the reel when the license asks for one.
- The songs bundled in the package (musiclib, `src/inframe/music_library/`) are found by `get` and `load_meta` without any request; an id that is
  not bundled goes to Openverse as before.
- Downloads go to ~/Library/Caches/inframe/music (~/.cache/inframe/music elsewhere; $INFRAME_CACHE replaces the base):
  `<id>.<ext>` and `<id>.json` with the license and the attribution."""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

from . import musiclib, net

API = "https://api.openverse.org/v1/audio/"
ALLOWED_LICENSES = ("cc0", "by")  # what inframe can ever use; the default is both (licenses_allowed)
MAX_BYTES = net.MAX_AUDIO_BYTES  # 50 MB
ID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")
MODIFICATION_NOTICE = "Changes were made: the music was trimmed or looped, faded in and out, and set to -14 LUFS for this reel."
OPENVERSE_NOTICE = "Found through Openverse (openverse.org); inframe is not endorsed by Openverse."


class MusicError(RuntimeError):
    pass


class LicenseError(MusicError):
    """The track's license is not one inframe uses (CC0, or CC BY unless cc0_only)."""


@dataclass
class Track:
    id: str
    title: str
    creator: str
    creator_url: str | None
    license: str  # cc0 | by
    license_version: str | None
    license_url: str | None
    url: str  # direct file
    landing_url: str | None
    duration_s: float
    attribution: str | None
    source: str | None
    filetype: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def needs_credit(self) -> bool:
        return needs_credit(self.license)


def licenses_allowed(cc0_only: bool = False) -> tuple[str, ...]:
    """CC0 and CC BY (decided 2026-10-16), or CC0 alone when the caller asked for `cc0_only`."""
    return ("cc0",) if cc0_only else ALLOWED_LICENSES


def needs_credit(license_code: str | None) -> bool:
    return (license_code or "").lower() not in ("cc0", "pdm")


def cache_dir() -> Path:
    base = os.environ.get("INFRAME_CACHE")
    if base:
        return Path(base).expanduser() / "music"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "inframe" / "music"
    return Path.home() / ".cache" / "inframe" / "music"


def _get(url: str, binary: bool = False, limit: int | None = None):
    """The Openverse JSON at `url` (`binary=False`), or the track file's bytes (`binary=True`). Every request goes through
    `net.fetch` (https only, global addresses, same-host redirects, size cap, audio content type for files)."""
    try:
        data = net.fetch(url, kind="audio" if binary else "api", max_bytes=limit)
    except net.NetError as e:
        if e.status == 429:
            raise MusicError("Openverse is rate-limiting anonymous requests (HTTP 429); wait a minute, or register an app for a token") from e
        if e.status and not binary:
            raise MusicError(f"Openverse answered HTTP {e.status} for {url.split('?')[0]}") from e
        raise MusicError(str(e)) from e
    if binary:
        return data
    try:
        return json.loads(data)
    except ValueError as e:
        raise MusicError(f"Openverse sent something that is not JSON for {url.split('?')[0]}") from e


def parse_track(r: dict, cc0_only: bool = False) -> Track | None:
    """A Track from one Openverse result, or None when the license is not CC0 or CC BY (CC0 alone with `cc0_only`) or the record lacks what we need."""
    lic = str(r.get("license") or "").lower()
    if lic not in licenses_allowed(cc0_only) or not r.get("id") or not r.get("url"):
        return None
    if lic == "by" and not r.get("license_url"):
        return None
    if not ID_RE.match(str(r["id"])):
        return None
    ms = r.get("duration")
    return Track(id=str(r["id"]), title=str(r.get("title") or "untitled"), creator=str(r.get("creator") or "unknown"),
                 creator_url=r.get("creator_url"), license=lic, license_version=r.get("license_version"), license_url=r.get("license_url"),
                 url=r["url"], landing_url=r.get("foreign_landing_url"), duration_s=round((ms or 0) / 1000, 1),
                 attribution=r.get("attribution"), source=r.get("source") or r.get("provider"), filetype=r.get("filetype"))


def search(mood: str | None = None, min_seconds: float = 0.0, limit: int = 10, source: str = "jamendo", max_pages: int = 3,
           cc0_only: bool = False) -> list[Track]:
    """Tracks whose license is CC0 or CC BY (CC0 alone with `cc0_only`; CC0 comes first), at least `min_seconds` long. `mood` is the free-text
    query (default: upbeat). One query per license: `license=by,cc0` as a list returns nothing, and `license_type=commercial` also returns by-sa and by-nd
    (still dropped here). Jamendo files are reported as filetype `mp32` (MP3 at the second quality level)."""
    out: list[Track] = []
    for lic in licenses_allowed(cc0_only):
        for page in range(1, max_pages + 1):
            q = dict(q=mood or "upbeat", category="music", license=lic, source=source, page_size=20, page=page)
            d = _get(API + "?" + urllib.parse.urlencode(q))
            for r in d.get("results", []):
                t = parse_track(r, cc0_only)
                if t and t.duration_s >= min_seconds and all(t.id != o.id for o in out):
                    out.append(t)
            if len(out) >= limit or page >= int(d.get("page_count") or 1):
                break
        if len(out) >= limit:
            break
    return out[:limit]


def _meta_path(audio: Path) -> Path:
    return audio.with_suffix(".json")


def _refuse_by(license_code: str | None, track_id: str) -> None:
    raise LicenseError(f"track {track_id} is CC {str(license_code or '?').upper()}, not CC0, and you asked for CC0 only (--cc0-only); "
                       "drop that option to use a CC BY track (inframe writes its credit line next to the reel)")


def _check_by_record(meta: dict | None, name: str) -> None:
    """A CC BY record must carry what its credit line needs: the licence URL, the title and the creator (parse_track refuses an Openverse
    record without a licence URL; a record read from a file or the cache gets the same check)."""
    missing = [k for k in ("license_url", "title", "creator") if not str((meta or {}).get(k) or "").strip()]
    if missing:
        raise LicenseError(f"{name}: its CC BY record lacks {', '.join(missing)}, so a complete credit cannot be written; add it to the record")


def get(track_id: str, dest: Path | None = None, cc0_only: bool = False) -> Path:
    """Download track `track_id` into the music cache (or `dest`), with `<id>.json` beside it. Returns the audio path.
    A bundled song is returned from the package (`dest=None`). A track already there is not downloaded again. A CC BY track is refused with `cc0_only` (also when it is already in the cache). The license is read from the cached record or from Openverse, never assumed."""
    if not ID_RE.match(track_id):
        raise MusicError(f"not a track id: {track_id!r}")
    if dest is None:  # a bundled song needs no download and no network
        song = musiclib.find(track_id)
        if song is not None:
            if song["license"] == "by" and cc0_only:
                _refuse_by("by", track_id)
            return musiclib.audio_path(song)
    dest = Path(dest) if dest else cache_dir()
    for f in dest.glob(f"{track_id}.*") if dest.is_dir() else []:
        if f.suffix != ".json" and ".part" not in f.name and _meta_path(f).is_file():
            lic = (load_meta(f) or {}).get("license")
            if str(lic).lower() == "by" and cc0_only:
                _refuse_by(lic, track_id)
            if str(lic).lower() not in ALLOWED_LICENSES:  # a cached record is no proof: only CC0 and CC BY are ever handed out
                raise LicenseError(f"the cached record of track {track_id} says license {lic!r}; inframe uses CC0 and CC BY tracks only")
            if str(lic).lower() == "by":
                _check_by_record(load_meta(f), f"cached track {track_id}")
            return f
    d = _get(API + track_id + "/")
    t = parse_track(d, cc0_only)
    if t is None:
        if str(d.get("license") or "").lower() == "by" and cc0_only:
            _refuse_by("by", track_id)
        raise LicenseError(f"track {track_id} is not CC0 or CC BY (license: {d.get('license')!r}); inframe uses CC0 and CC BY tracks only")
    kind = re.sub(r"[^a-z0-9]", "", (t.filetype or "mp3").lower())[:5] or "mp3"
    ext = "." + ("mp3" if kind.startswith("mp3") else kind)
    dest.mkdir(parents=True, exist_ok=True)
    audio = dest / f"{t.id}{ext}"
    part = audio.with_name(audio.name + ".part")
    data = _get(t.url, binary=True, limit=MAX_BYTES)
    if len(data) > MAX_BYTES:
        raise MusicError(f"file is larger than {MAX_BYTES // 2**20} MB")
    _sniff_audio(data)
    part.write_bytes(data)
    try:
        _check_audio(part)
    except MusicError:
        part.unlink(missing_ok=True)
        raise
    part.replace(audio)
    _meta_path(audio).write_text(json.dumps(dict(t.to_json(), openverse_notice=OPENVERSE_NOTICE), indent=1))
    return audio


def _sniff_audio(data: bytes) -> None:
    """The first bytes must look like audio (ID3 or MPEG sync, ftyp, RIFF WAVE, fLaC, OggS, ADTS), so a text file (an HLS or
    ffconcat playlist, say) never reaches ffprobe. The file type named by Openverse is not trusted for this."""
    from .serve_paths import AUDIO_EXTS, audio_magic_ok  # lazy: serve_paths pulls in the picker and PyAV

    head = data[:64]
    if not any(audio_magic_ok(ext, head) for ext in AUDIO_EXTS):
        raise MusicError("the download is not a playable audio file (it does not start like audio)")


def _check_audio(path: Path) -> None:
    """The download must be something ffprobe reads as audio of more than a second (it is not decoded or run). ffprobe may open
    local files and pipes only, so a playlist that slipped through cannot fetch anything."""
    import subprocess

    r = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "a:0", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(Path(path).resolve())],
                       capture_output=True, text=True)
    try:
        ok = float(r.stdout.strip()) > 1.0
    except ValueError:
        ok = False
    if not ok:
        raise MusicError("the download is not a playable audio file")


def load_meta(audio: Path) -> dict | None:
    song = musiclib.song_for_path(audio)
    if song is not None:  # a bundled song: its record comes from library.json
        return musiclib.record(song)
    m = _meta_path(audio)
    try:
        d = json.loads(m.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def check_file_license(meta: dict | None, name: str, cc0_only: bool = False) -> None:
    """A music file you choose yourself: its `<name>.json` may say only `cc0` or `by` (and `by` not with `cc0_only`); by-sa, by-nd, by-nc* and
    anything else is refused. A file without a record, or with an empty license field, passes (the export warns that no credits file was written)."""
    lic = str((meta or {}).get("license") or "").strip().lower()
    if not lic:
        return
    if lic not in ALLOWED_LICENSES:
        raise LicenseError(f"{name}: its record says license {lic!r}; inframe uses CC0 and CC BY tracks only")
    if lic == "by" and cc0_only:
        _refuse_by(lic, str((meta or {}).get("id") or name))
    if lic == "by":
        _check_by_record(meta, name)


def resolve(arg: str, cc0_only: bool = False) -> tuple[Path, dict | None]:
    """`--music` value -> (audio file, license metadata or None). An existing file is used as it is, with `<name>.json` beside it
    as its metadata when that exists (the cache writes one). Anything else must be a track id: from the cache, or downloaded."""
    p = Path(arg).expanduser()
    if p.is_file():
        meta = load_meta(p)
        check_file_license(meta, p.name, cc0_only)
        return p, meta
    if ID_RE.match(arg):
        audio = get(arg, cc0_only=cc0_only)
        return audio, load_meta(audio)
    raise MusicError(f"--music: {arg!r} is neither a file nor a track id")


def credits_text(meta: dict) -> str:
    lines = [f'Music: "{meta.get("title", "untitled")}" by {meta.get("creator", "unknown")}']
    if meta.get("attribution"):
        lines.append(str(meta["attribution"]))
    lic = str(meta.get("license", "")).upper()
    ver = meta.get("license_version")
    lines.append(f"License: CC {lic}" + (f" {ver}" if ver else "") + (f" ({meta['license_url']})" if meta.get("license_url") else ""))
    for k, label in (("creator_url", "Artist"), ("landing_url", "Source")):
        if meta.get(k):
            lines.append(f"{label}: {meta[k]}")
    if needs_credit(meta.get("license")):
        lines.append(MODIFICATION_NOTICE)
    lines.append(OPENVERSE_NOTICE)
    return "\n".join(lines) + "\n"
