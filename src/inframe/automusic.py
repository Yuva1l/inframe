"""Auto music: pick a free song that suits the reel (docs/AUTO_MUSIC.md). No LLM, no new dependency.

The songs bundled in the package (`musiclib`, src/inframe/music_library/) come first and need no network: they are scored from their
metadata, then, with librosa, from their audio, and ranks 0 to B-1 are those songs. Only "another song" past the last bundled song
("more songs") searches Openverse; then the only things that leave the machine are the Openverse search text (a mood word) and the
track downloads; the reel's shape (clip lengths, count) never does. Online songs come through `music` (CC0 and CC BY, CC0 alone with
`cc0_only`) and are scored the same way. The ranked list is fixed; "another song" is the next rank. The last entry of every complete
list is inframe's built-in beat, so it is never empty."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import beats as beats_mod
from . import music, musiclib, net, synthbeat
from .beats import NoBeatError, plan_beat_cuts

MOOD_RE = re.compile(r"[\w -]{0,40}")  # the same rule as the serve route
OFFLINE_ENV = "INFRAME_MUSIC_OFFLINE"
SEARCH_V = 1
SEARCH_TTL_S = 7 * 86400
PARTIAL_TTL_S = 600  # a search whose fallback query failed is kept this long, so a 429 is not fed with the same requests again
FALLBACK_QUERY = "instrumental"
MIN_ROWS = 5  # fewer rows than this after the filters: also search the fallback query
MAX_ROWS_PER_LICENSE = 30
PAGES = (1, 2)
MIN_SONG_S, MAX_SONG_S = 30.0, 600.0
BATCHES = ((0, 4), (4, 6), (6, 8))  # positions in the metadata order; at most 8 downloads per (shape, mood, cc0_only)
MAX_DOWNLOADS = BATCHES[-1][1]
EXTRA_ANALYSIS_S = 40.0
KEEP_BYTES = 500 * 2**20
KEEP_TRACKS = 60
BUILTIN_SECONDS = 120
BUILTIN_ID = "00000000-0000-0000-0000-000000000%03d"
WARN_BELOW = 0.50
SPEED_VALUES = {"verylow": 0, "low": 1, "medium": 2, "high": 3, "veryhigh": 4}
# speed -> (default mood, wanted Jamendo speed tag, tempo band of the folded BPM, built-in beat BPM choices)
SPEEDS = {"slow": ("calm", 1, (70, 100), (80, 90, 100)),
          "medium": ("upbeat", 2, (95, 130), (100, 112, 120, 128)),
          "fast": ("energetic", 3, (120, 160), (120, 128, 140, 150))}


# ---------------------------------------------------------------- the reel's shape

@dataclass
class ReelShape:
    durations: list[float]  # seconds each clip shows, in reel order
    slack: list[float]  # seconds of padding at the end of each clip
    fps: float

    @property
    def T(self) -> float:
        return sum(self.durations)

    @property
    def n(self) -> int:
        return len(self.durations)

    @property
    def rate(self) -> float:
        return 60.0 * self.n / self.T if self.T > 0 else 0.0

    @property
    def speed(self) -> str:
        return "slow" if self.rate < 4 else "medium" if self.rate < 10 else "fast"

    @property
    def default_mood(self) -> str:
        return SPEEDS[self.speed][0]


def shape_from_files(doc: dict, clips_dir: str | Path) -> ReelShape:
    """The shape from the cut clip files (their real length) and the document (their padding)."""
    from .export import tail_slack
    from .musicreel import video_params
    from .reel import shown_pictures

    good = [c for c in doc["clips"] if "error" not in c]
    if not good:
        raise ValueError("no clips")
    paths = [str(Path(clips_dir) / c["file"]) for c in good]
    fps = video_params(paths[0])["fps"]
    return ReelShape([shown_pictures(p) / fps for p in paths], [tail_slack(doc, c) for c in good], fps)


def shape_from_doc(doc: dict, clips: list[dict], fps: float, factor: float = 1.0) -> ReelShape:
    """The shape from the document alone, before the clips are cut (an estimate for retimed clips)."""
    from .export import tail_slack

    if not clips:
        raise ValueError("no clips")
    return ReelShape([(c["end"] - (c.get("expected_copy_start") or c["start"])) * factor for c in clips],
                     [tail_slack(doc, c) for c in clips], fps)


def parse_spec(arg: str | None) -> str | None:
    """`--music` value -> the mood: "" for `auto`, "calm" for `auto:calm`, None for anything else. A bad mood raises ValueError."""
    if arg == "auto":
        return ""
    if isinstance(arg, str) and arg.startswith("auto:"):
        mood = arg[5:]
        if not MOOD_RE.fullmatch(mood):
            raise ValueError(f"the mood must be up to 40 letters, digits, spaces or dashes (got {mood!r})")
        return mood
    return None


def can_analyse() -> bool:
    import importlib.util

    return importlib.util.find_spec("librosa") is not None


def _offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip() not in ("", "0")


# ---------------------------------------------------------------- scores

def _length_term(dur: float, need: float) -> float:
    if dur < need:  # the song loops
        return 0.2 + 0.3 * dur / need
    return 1.0 if dur <= 3 * need else max(0.6, 1 - 0.1 * (dur / need - 3))


def _speed_tag(tags: list[str]) -> int | None:
    for t in tags:
        m = re.fullmatch(r"speed_(verylow|low|medium|high|veryhigh)", t)
        if m:
            return SPEED_VALUES[m.group(1)]
    return None


def meta_score(row: dict, shape: ReelShape, mood: str) -> float:
    """Score from the search row alone (no download), between 0 and 1."""
    tags = [str(t).lower() for t in row.get("tags") or []]
    genres = [str(g).lower() for g in row.get("genres") or []]
    L = _length_term(float(row.get("duration_s") or 0), shape.T + 3)
    tag = _speed_tag(tags)
    S = 0.5 if tag is None else max(0.0, 1 - 0.35 * abs(tag - SPEEDS[shape.speed][1]))
    words = re.findall(r"\w+", (mood or shape.default_mood).lower())
    M = 1.0 if any(w in tags or w in genres for w in words) else 0.6
    I = 1.0 if "instrumental" in tags else 0.6 if "vocal" in tags else 0.8
    return 0.45 * L + 0.30 * S + 0.15 * M + 0.10 * I


def _fold(bpm: float) -> float:
    while 0 < bpm < 70:
        bpm *= 2
    while bpm >= 180:
        bpm /= 2
    return bpm


def _plan_fit(shape: ReelShape, beats: list[float]) -> float:
    """J: the share of clips whose end can land on a beat inside its padding (what beat sync needs to trim only padding)."""
    fps = shape.fps
    avail = [max(1, round(d * fps)) for d in shape.durations]
    slack = [max(0, min(a - 1, round(s * fps))) for a, s in zip(avail, shape.slack)]
    try:
        plan = plan_beat_cuts(avail, slack, beats, fps)
    except (NoBeatError, ValueError):
        return 0.0
    warned = {int(w.split(":")[0].split()[1]) for w in plan.warnings if w.startswith("clip ")}
    return sum(1 for j in plan.joins if j["on_beat"] and j["clip"] not in warned) / shape.n


def _tempo_term(bpm: float, shape: ReelShape) -> float:
    lo, hi = SPEEDS[shape.speed][2]
    bpm = _fold(bpm)
    return 1.0 if lo <= bpm <= hi else max(0.0, 1 - (lo - bpm if bpm < lo else bpm - hi) / 25)


def audio_score(analysis: dict, shape: ReelShape, meta: float) -> dict:
    """The final score from an audio analysis: the terms and `score`, all between 0 and 1."""
    bt = analysis["beats"]
    J = _plan_fit(shape, bt)
    Lp = _length_term(float(analysis["duration_s"]), shape.T + bt[0])
    Tb = _tempo_term(float(analysis["bpm"]), shape)
    P = min(1.0, max(0.0, (float(analysis["pulse"]) - 1) / 1.5))
    Q = min(1.0, max(0.0, float(analysis["intro_ratio"]) / 0.6))
    score = 0.35 * J + 0.20 * Lp + 0.15 * Tb + 0.15 * P + 0.10 * Q + 0.05 * meta
    return dict(J=round(J, 3), Lp=round(Lp, 3), Tb=round(Tb, 3), P=round(P, 3), Q=round(Q, 3), meta=round(meta, 3), score=round(score, 3))


# ---------------------------------------------------------------- the cache under music.cache_dir()/auto

def _auto_dir() -> Path:
    return music.cache_dir() / "auto"


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _write_json(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(d))
    part.replace(path)


def _analysis_path(track_id: str) -> Path:
    return _auto_dir() / "analysis" / f"{track_id}.json"


def _known_bad(track_id: str) -> bool:
    return "error" in (_read_json(_analysis_path(track_id)) or {})


def _record_bad(track_id: str, e: Exception) -> None:
    _write_json(_analysis_path(track_id), dict(v=1, error=f"{type(e).__name__}: {str(e)[:200]}"))


# net.NetError without a status is a network failure only for these; a size cap, a non-audio answer or a refused redirect is final
NET_DOWN = ("could not reach", "could not read from", "could not resolve")
NON_PUBLIC = "it resolves to a non-public address"  # the network setup (DNS, a captive portal) refused the host, not the song


def _transient(e: music.MusicError) -> bool:
    """A network failure, a 429 or 5xx, or a name that resolves to a private address (a captive portal): not recorded, because asking again may work."""
    c = e.__cause__
    if not isinstance(c, net.NetError):
        return False
    if c.status is None:
        return str(c).startswith(NET_DOWN) or str(c).endswith(NON_PUBLIC)
    return c.status == 429 or c.status >= 500


def _cached_audio(track_id: str) -> Path | None:
    d = music.cache_dir()
    for f in d.glob(f"{track_id}.*") if d.is_dir() else []:
        if f.suffix != ".json" and ".part" not in f.name and f.with_suffix(".json").is_file():
            return f
    return None


def _index() -> dict:
    return (_read_json(_auto_dir() / "index.json") or {}).get("ids") or {}


def _save_index(ids: dict) -> None:
    _write_json(_auto_dir() / "index.json", dict(v=1, ids=ids))


def _touch(track_id: str, new: bool = False) -> None:
    ids = _index()
    if track_id in ids or new:  # a track the user downloaded with `music get` is not ours to track or evict
        ids[track_id] = dict(last_used=time.time())
        _save_index(ids)


def _evict(protect: set[str]) -> None:
    """Keep the files auto music downloaded within 500 MB and 60 tracks: remove the least recently used (audio, record, analysis)."""
    ids = _index()
    before = dict(ids)

    def size(i: str) -> int:
        return sum(f.stat().st_size for f in music.cache_dir().glob(f"{i}.*") if f.suffix != ".json" and f.is_file())

    for i in [i for i in ids if _cached_audio(i) is None]:
        del ids[i]
    sizes = {i: size(i) for i in ids}
    while ids and (sum(sizes.values()) > KEEP_BYTES or len(ids) > KEEP_TRACKS):
        victims = [i for i in ids if i not in protect]
        if not victims:
            break
        v = min(victims, key=lambda i: ids[i].get("last_used", 0))
        for f in music.cache_dir().glob(f"{v}.*"):
            f.unlink(missing_ok=True)
        _analysis_path(v).unlink(missing_ok=True)
        del ids[v], sizes[v]
    if ids != before:
        _save_index(ids)


# ---------------------------------------------------------------- search

def _tag_names(values) -> list[str]:
    out = []
    for v in values or []:
        name = v.get("name") if isinstance(v, dict) else v
        if name:
            out.append(str(name).lower())
    return out


def _search_rows(query: str, cc0_only: bool, seen: set[str]) -> list[dict]:
    """The rows that pass the filters for one query: at most 2 pages for each license (`cc0` first), so at most 4 requests."""
    import urllib.parse

    out: list[dict] = []
    for lic in music.licenses_allowed(cc0_only):
        kept = 0
        for page in PAGES:
            q = dict(q=query, category="music", source="jamendo", license=lic, page_size=20, page=page)
            d = music._get(music.API + "?" + urllib.parse.urlencode(q))
            for r in d.get("results", []):
                t = music.parse_track(r, cc0_only)
                if t is None or not MIN_SONG_S <= t.duration_s <= MAX_SONG_S or r.get("mature") is True or t.id in seen:
                    continue
                seen.add(t.id)
                kept += 1
                out.append(dict(t.to_json(), tags=_tag_names(r.get("tags")), genres=_tag_names(r.get("genres")), query=query, page=page))
            if page >= int(d.get("page_count") or 1) or kept >= MAX_ROWS_PER_LICENSE:
                break
    return out


def _candidates(query: str, cc0_only: bool, warnings: list[str]) -> tuple[list[dict], str | None]:
    """(rows, why none): the kept search rows for `query`, from the 7-day search cache or from Openverse."""
    key = hashlib.sha256(json.dumps([SEARCH_V, query, cc0_only]).encode()).hexdigest()[:16]
    path = _auto_dir() / "search" / f"{key}.json"
    cached = _read_json(path)
    if cached and cached.get("v") != SEARCH_V:
        cached = None
    ttl = PARTIAL_TTL_S if cached and cached.get("partial") else SEARCH_TTL_S
    if cached and (_offline() or time.time() - float(cached.get("saved", 0)) < ttl):
        if cached.get("partial") and not _offline():
            warnings.append(f"the fallback search failed a moment ago; asking again in {PARTIAL_TTL_S // 60} minutes")
        return cached["rows"], None
    if _offline():
        return [], "offline"
    seen: set[str] = set()
    try:
        rows = _search_rows(query, cc0_only, seen)
    except music.MusicError as e:
        if cached:
            warnings.append(f"Openverse could not be reached ({str(e)[:100]}); using the songs found earlier")
            return cached["rows"], None
        warnings.append(f"Openverse could not be reached ({str(e)[:100]})")
        return [], "unreachable"
    complete = True
    if len(rows) < MIN_ROWS and query != FALLBACK_QUERY:
        try:
            rows += _search_rows(FALLBACK_QUERY, cc0_only, seen)
        except music.MusicError as e:
            complete = False
            warnings.append(f"the fallback search failed ({str(e)[:100]})")
    _write_json(path, dict(v=SEARCH_V, saved=time.time(), query=query, cc0_only=cc0_only, rows=rows, **({} if complete else dict(partial=True))))
    return rows, None


# ---------------------------------------------------------------- audio analysis

def _duration(audio: Path) -> float:
    import subprocess

    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(audio)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise NoBeatError("ffprobe could not read the song's length") from None  # ffprobe ran and the song is unreadable: final


def analyse_file(audio: Path, track_id: str, shape: ReelShape) -> dict:
    """Beats and feel of a song, cached in auto/analysis/<id>.json; analysed again only when it covers less than the reel needs now."""
    path = _analysis_path(track_id)
    old = _read_json(path)
    if old and "error" not in old and old.get("v") == 1:
        need = min(float(old["duration_s"]), shape.T + EXTRA_ANALYSIS_S, MAX_SONG_S)
        if float(old["analysed_s"]) + 0.5 >= need:
            return old
    import numpy as np

    librosa = beats_mod._librosa()
    full = _duration(audio)
    try:
        y = beats_mod._load_mono(str(audio), min(shape.T + EXTRA_ANALYSIS_S, MAX_SONG_S))
    except NoBeatError:
        raise
    except RuntimeError as e:  # ffmpeg ran and could not decode this song: final (a missing ffmpeg is an OSError, the machine)
        raise NoBeatError(str(e)) from e
    grid, oenv = beats_mod.analyse_signal(y)
    sr = beats_mod.ANALYSIS_RATE
    frames = [min(len(oenv) - 1, round(t * sr / beats_mod.HOP)) for t in grid.times]
    pulse = float(np.mean(oenv[frames]) / max(1e-9, np.mean(oenv)))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    t = np.arange(len(rms)) * 512 / sr
    first = grid.times[0]
    window = rms[(t >= first) & (t < first + 10)]
    intro_ratio = float(window.mean() / max(1e-9, np.median(rms))) if window.size else 0.0
    p90 = float(np.percentile(rms, 90))
    loud = np.nonzero(20 * np.log10(np.maximum(rms, 1e-9) / max(p90, 1e-9)) >= -20)[0]
    d = dict(v=1, bpm=round(grid.tempo, 2), beats=[round(b, 3) for b in grid.times], duration_s=round(full, 2),
             analysed_s=round(grid.duration, 2), pulse=round(pulse, 3), intro_ratio=round(intro_ratio, 3),
             intro_s=round(float(t[loud[0]]), 2) if loud.size else None)
    _write_json(path, d)
    return d


# ---------------------------------------------------------------- the built-in beat

def _builtin_pick(shape: ReelShape) -> tuple[int, float]:
    """(BPM, J): the built-in beat tempo of the reel's speed with the best plan fit, the lower BPM on a tie. Needs no librosa."""
    best = None
    for bpm in SPEEDS[shape.speed][3]:
        j = _plan_fit(shape, [float(b) for b in synthbeat.beat_grid(bpm, BUILTIN_SECONDS)])
        if best is None or j > best[1]:
            best = (bpm, j)
    return best


def _builtin_meta(bpm: int) -> dict:
    return dict(id=BUILTIN_ID % bpm, title=f"inframe built-in beat, {bpm:03d} BPM", creator="inframe", license="cc0", license_version="1.0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/", attribution=None, source="inframe",
                duration_s=BUILTIN_SECONDS, builtin=True)


def builtin_beat(shape: ReelShape) -> tuple[Path, dict, int, float]:
    """(audio, its record, BPM, J): inframe's own beat in the music cache, rendered on first use. Allowed with cc0_only."""
    bpm, j = _builtin_pick(shape)
    meta = _builtin_meta(bpm)
    d = music.cache_dir()
    audio = d / f"{meta['id']}.wav"
    if not (audio.is_file() and audio.with_suffix(".json").is_file()):
        d.mkdir(parents=True, exist_ok=True)
        y, _ = synthbeat.render(bpm, BUILTIN_SECONDS, 0.0, "beat")
        part = d / f"{meta['id']}.wav.part"
        synthbeat.write_wav(str(part), y)
        part.replace(audio)
        audio.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return audio, meta, bpm, j


# ---------------------------------------------------------------- the bundled songs

def _bundled_row(song: dict) -> dict:
    row = musiclib.record(song)
    row["tags"] = sorted({str(t).lower() for t in [*(song.get("tags") or []), *(song.get("moods") or [])]})
    row["genres"] = [str(g).lower() for g in song.get("genres") or []]
    return row


def bundled_entries(shape: ReelShape, mood: str, cc0_only: bool, analyse: bool) -> list[Entry]:
    """The library's songs, best fit first. No network: the audio is in the package. With librosa the score is the audio score of any
    other song; without it, 60% the metadata score and 40% how well the measured BPM suits the reel (BPM was measured when the library was built).
    A bundled song has one BPM everywhere, the library's: the dialog, `music list`, the reel's record and the tempo term of the score all use it.
    The order can still differ with the `music` extra: the analysis supplies the beats and the other score terms (it replaces the 60/40 blend with
    the audio score), but its own tempo estimate is not used."""
    query = mood or shape.default_mood
    out: list[Entry] = []
    for song in musiclib.songs(cc0_only):
        row = _bundled_row(song)
        audio = musiclib.audio_path(song)
        m = meta_score(row, shape, query)
        score = dict(meta=round(m, 3), score=round(0.6 * m + 0.4 * _tempo_term(float(song["bpm"]), shape), 3))
        bpm = float(song["bpm"])
        if analyse:
            try:
                ana = analyse_file(audio, song["id"], shape)
                score = audio_score(dict(ana, bpm=bpm), shape, m)
            except (NoBeatError, RuntimeError, OSError):  # no steady beat here, or the machine cannot analyse: the metadata score stands
                pass
        out.append(Entry(meta=row, score=score, bpm=bpm, audio=audio))
    return sorted(out, key=lambda e: (-round(e.score["score"], 6), 0 if e.meta["license"] == "cc0" else 1, e.meta["id"]))


# ---------------------------------------------------------------- the ranked list

@dataclass
class Entry:
    meta: dict  # the track record (the search row until the song is fetched)
    score: dict
    bpm: float | None = None
    audio: Path | None = None  # None until fetched (a built-in beat is rendered, and a song fetched without analysis, when chosen)
    builtin: bool = False


@dataclass
class Ranked:
    entries: list[Entry]
    n: int  # the length of the list; before every batch is fetched, the most it can have
    complete: bool
    mood: str
    speed: str
    warnings: list[str] = field(default_factory=list)


class _Transient(Exception):
    pass


def _order(rows: list[dict], shape: ReelShape, query: str) -> list[tuple[dict, float]]:
    scored = [(r, meta_score(r, shape, query)) for r in rows]
    return sorted(scored, key=lambda x: (-round(x[1], 6), 0 if x[0]["license"] == "cc0" else 1, x[0]["id"]))


def _entry(row: dict, m: float, shape: ReelShape, cc0_only: bool) -> Entry | None:
    """Fetch one candidate and score its audio; None when it is dropped (and the reason recorded unless it was a network failure)."""
    tid = row["id"]
    existed = _cached_audio(tid) is not None
    if _offline() and not existed:
        return None
    try:
        audio = music.get(tid, cc0_only=cc0_only)
    except music.MusicError as e:
        if not isinstance(e, music.LicenseError) and _transient(e):
            raise _Transient(f"Openverse could not be reached ({str(e)[:100]})") from e
        _record_bad(tid, e)
        return None
    if not existed:
        _touch(tid, new=True)
    meta = music.load_meta(audio) or row
    try:
        ana = analyse_file(audio, tid, shape)
    except NoBeatError as e:  # this song has no steady beat: final
        _record_bad(tid, e)
        return None
    except (RuntimeError, OSError) as e:  # ffprobe or ffmpeg missing or failing, librosa missing, a full disk: the machine, not the song
        raise _Transient(f"the song could not be analysed here ({type(e).__name__}: {str(e)[:100]})") from e
    _touch(tid)
    return Entry(meta=meta, score=audio_score(ana, shape, m), bpm=ana["bpm"], audio=audio)


def rank_list(shape: ReelShape, mood: str, cc0_only: bool, upto: int, analyse: bool | None = None) -> Ranked:
    """The ranked list, fetched in fixed batches as far as rank `upto` needs (see docs/AUTO_MUSIC.md 1.4); the built-in beat is last."""
    analyse = can_analyse() if analyse is None else analyse
    query = mood or shape.default_mood
    warnings: list[str] = []
    bundled = bundled_entries(shape, mood, cc0_only, analyse)
    if upto < len(bundled):  # the rank asked for is a bundled song: no search, no network
        return Ranked(bundled, len(bundled) + 1, False, query, shape.speed, warnings)
    rows, why = _candidates(query, cc0_only, warnings)
    have = {str(e.meta["id"]) for e in bundled}
    ordered = _order([r for r in rows if r["id"] not in have], shape, query)[:MAX_DOWNLOADS]
    if not ordered and not bundled and why != "unreachable":  # an unreachable Openverse has been warned about already
        warnings.append(f"music search is off ({OFFLINE_ENV}) and no song is cached for mood {query}" if why == "offline"
                        else f"no free song was found for mood {query}")
    entries: list[Entry] = list(bundled)
    complete = True
    pos = 0
    if analyse:
        for lo, hi in BATCHES:
            if len(entries) > upto and lo < len(ordered):
                complete = False
                break
            batch: list[tuple[Entry, int]] = []
            stopped = False
            try:
                for k, (row, m) in enumerate(ordered[lo:hi], lo):
                    if not _known_bad(row["id"]) and (e := _entry(row, m, shape, cc0_only)):
                        batch.append((e, k))
            except _Transient as t:  # the whole batch is left out, so no song's rank moves when it is fetched in full later
                stopped = True
                batch.clear()
                warnings.append(f"{t}; the list stops here")
            pos = min(hi, len(ordered))
            entries += [e for e, _ in sorted(batch, key=lambda x: (-x[0].score["score"], x[1]))]
            if stopped:
                break
    else:
        for row, m in ordered:
            if not _known_bad(row["id"]):
                entries.append(Entry(meta=row, score=dict(meta=round(m, 3), score=round(m, 3))))
        pos = len(ordered)
    n = len(entries) + 1 + (len(ordered) - pos if not complete else 0)
    if complete:
        bpm, j = _builtin_pick(shape)
        entries.append(Entry(meta=_builtin_meta(bpm), score=dict(J=round(j, 3)), bpm=bpm, builtin=True))
    return Ranked(entries, n, complete, query, shape.speed, warnings)


@dataclass
class Choice:
    audio: Path
    meta: dict  # the track's <id>.json
    rank: int
    n: int
    wrapped: bool
    mood: str  # the search text used
    speed: str
    score: dict
    bpm: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def builtin(self) -> bool:
        return bool(self.meta.get("builtin"))

    def summary(self) -> dict:
        """What export.json keeps as reel.music.auto."""
        return dict(rank=self.rank, n=self.n, mood=self.mood, speed=self.speed, score=self.score, bpm=self.bpm,
                    builtin=self.builtin, wrapped=self.wrapped)


def choose(shape: ReelShape, mood: str, cc0_only: bool, rank: int, analyse: bool | None = None) -> Choice:
    """Entry `rank % len` of the ranked list. The same inputs and caches always give the same song; "another song" is `rank + 1`."""
    if shape.n < 1:
        raise ValueError("no clips")
    analyse = can_analyse() if analyse is None else analyse
    for _ in range(MAX_DOWNLOADS + 2):  # each lap drops one song that could not be fetched
        r = rank_list(shape, mood, cc0_only, rank, analyse)
        wrapped = rank >= len(r.entries)
        k = rank % len(r.entries)
        e = r.entries[k]
        warnings = list(r.warnings)
        if e.builtin:
            audio, meta, bpm, _ = builtin_beat(shape)
        elif e.audio is not None:
            audio, meta, bpm = e.audio, e.meta, e.bpm
            _evict({str(meta["id"])})
        else:  # no analysis: only the chosen song is fetched
            tid = e.meta["id"]
            existed = _cached_audio(tid) is not None
            try:
                if _offline() and not existed:
                    raise music.MusicError("music search is off and the song is not cached")
                audio = music.get(tid, cc0_only=cc0_only)
            except music.MusicError as ex:
                if not isinstance(ex, music.LicenseError) and (_transient(ex) or _offline()):
                    k = len(r.entries) - 1
                    audio, meta, bpm, _ = builtin_beat(shape)
                    warnings.append(f"Openverse could not be reached ({str(ex)[:100]}); using the built-in beat")
                    e = r.entries[k]
                else:
                    _record_bad(tid, ex)
                    continue
            else:
                if not existed:
                    _touch(tid, new=True)
                _touch(tid)
                _evict({tid})
                meta, bpm = music.load_meta(audio) or e.meta, None
        if not can_analyse():
            warnings.append("beat sync needs the music extra")
        if analyse and k == 0 and not e.builtin and e.score["score"] < WARN_BELOW:
            warnings.append(f"no song fits this reel well (best fit {e.score['score']:.2f}); try another mood")
        return Choice(audio, meta, k, r.n, wrapped, r.mood, r.speed, e.score, bpm, warnings)
    raise music.MusicError("no song could be fetched")
