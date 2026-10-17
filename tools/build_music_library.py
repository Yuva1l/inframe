#!/usr/bin/env python3
"""Build the songs that ship inside inframe: src/inframe/music_library/ (audio files, library.json, CREDITS.md).

    uv run python tools/build_music_library.py [--out DIR] [--work DIR] [--count 8] [--budget-mb 25]
    uv run python tools/build_music_library.py --verify          # re-check the committed library, no network

What it does (no LLM; every number is measured by code):
  1. music.search (Openverse, Jamendo) for several mood words; a candidate stays when it is CC0, or CC BY 3.0 / 4.0 (never BY-SA, BY-ND, NC),
     60 to 240 s long, and its record names the title, the artist and the license URL.
  2. music.get downloads each candidate into a work folder; ffprobe gives the duration, librosa (inframe.beats) the tempo, ffmpeg
     (ebur128) the integrated loudness.
  3. It picks `--count` songs that differ in tempo and in mood, within `--budget-mb` megabytes in total. The original file is kept when the
     budget allows; otherwise the biggest files are transcoded to AAC 160 kbit/s (m4a) until the library fits, and the credits say so.
  4. It writes library.json (title, artist, license, license_url, source_url, duration_s, bpm, sha256, file, ...) and CREDITS.md.

Openverse asks anonymous users to stay gentle: about 40 requests and 24 downloads per run, one at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inframe import beats, music, musiclib  # noqa: E402

MOODS = ("calm", "upbeat", "energetic", "cinematic", "funk", "acoustic guitar", "electronic", "rock", "chill")
PER_MOOD = 2  # candidates downloaded per mood word
MIN_S, MAX_S = 60.0, 240.0
OK_BY_VERSIONS = ("3.0", "4.0")
LUFS_RANGE = (-30.0, -6.0)
AAC_KBPS = 160
PAUSE_S = 1.0
# Tracks a rebuild must never pick again, with the reason. 22a0cd91 carries the title of "Jingle Bell Rock" (Beal/Boothe, 1957, in copyright);
# if it is a cover, a CC BY label cannot license the composition. Nobody checked whether it is an original [unverified], so it is out.
EXCLUDED = {"22a0cd91-ad66-4c36-ac70-c6d7a3b5c47d": "title of a copyrighted 1957 song, possibly a cover"}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- measuring

def fold(bpm: float) -> float:
    while 0 < bpm < 70:
        bpm *= 2
    while bpm >= 180:
        bpm /= 2
    return bpm


def probe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


def measure_lufs(path: Path) -> float | None:
    """Integrated loudness (EBU R128) by ffmpeg's ebur128 filter; the last value it prints."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-i", str(path), "-vn", "-af",
                        "ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.I:file=-", "-f", "null", "-"], capture_output=True, text=True)
    vals = re.findall(r"lavfi\.r128\.I=(-?[0-9.]+)", r.stdout)
    return float(vals[-1]) if vals else None


TEMPO_V = 2  # 2: librosa.feature.tempo with a flat prior (v1 used the beat tracker's, which pulls every song towards 120-135 BPM)


def measure_tempo(path: Path) -> float | None:
    """BPM of the first 120 s: the strongest periodicity of the onset envelope (librosa `feature.tempo`, flat prior over 60..200 BPM,
    median over the song), folded into 70..180. The beat tracker of inframe.beats is not used for this number because its start_bpm
    prior pulls every estimate towards 120-135 BPM."""
    import numpy as np
    import scipy.stats

    try:
        y = beats._load_mono(str(path), 120.0)
        librosa = beats._librosa()
        oenv = librosa.onset.onset_strength(y=y, sr=beats.ANALYSIS_RATE, hop_length=beats.HOP)
        t = librosa.feature.tempo(onset_envelope=oenv, sr=beats.ANALYSIS_RATE, hop_length=beats.HOP, aggregate=np.median, prior=scipy.stats.uniform(60, 140))
    except (beats.NoBeatError, RuntimeError, ValueError):
        return None
    bpm = float(np.atleast_1d(t)[0])
    return round(fold(bpm), 1) if bpm > 0 else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- candidates

def acceptable(t: music.Track) -> str | None:
    """Why this track is not allowed, or None."""
    if t.id in EXCLUDED:
        return f"excluded: {EXCLUDED[t.id]}"
    if t.license == "by" and t.license_version not in OK_BY_VERSIONS:
        return f"CC BY {t.license_version}"
    if t.license not in ("cc0", "by"):
        return f"license {t.license}"
    if not MIN_S <= t.duration_s <= MAX_S:
        return f"length {t.duration_s:.0f} s"
    if not (t.title.strip() and t.creator.strip() and t.creator != "unknown" and t.license_url):
        return "record lacks title, artist or license URL"
    if not (t.landing_url or t.url):
        return "no source URL"
    return None


def tag_names(values) -> list[str]:
    out = []
    for v in values or []:
        name = v.get("name") if isinstance(v, dict) else v
        if name:
            out.append(str(name).lower())
    return out


def load_measured(work: Path) -> dict[str, dict]:
    try:
        return json.loads((work / "measured.json").read_text())
    except (OSError, ValueError):
        return {}


def gather(work: Path, resume: bool = False) -> list[dict]:
    """Search, filter, download and measure. One dict per usable candidate. Every measurement is kept in work/measured.json, so a second
    run measures nothing again; with `resume` and enough measured candidates it does not even search."""
    measured = load_measured(work)
    for tid, m in measured.items():  # tempo measured by an older method: measure it again from the downloaded file
        if m.get("tempo_v") != TEMPO_V and m.get("ok") is not None:
            f = original_file(work, tid)
            m["bpm"] = measure_tempo(f) if f else None
            m["tempo_v"] = TEMPO_V
            m["ok"] = bool(f) and m["bpm"] is not None and MIN_S <= m["duration_s"] <= MAX_S and m["lufs"] is not None and LUFS_RANGE[0] <= m["lufs"] <= LUFS_RANGE[1]
            log(f"  tempo {tid[:8]}: {m['bpm']} BPM")
    (work / "measured.json").write_text(json.dumps(measured))
    if resume and len(measured) >= 12:
        log(f"resume: {len(measured)} candidates measured earlier, no search")
        return [revive(work, m) for tid, m in measured.items() if m.get("ok") and tid not in EXCLUDED]
    seen: dict[str, dict] = {}
    for mood in MOODS:
        try:
            found = music.search(mood, MIN_S, 20, max_pages=1)
        except music.MusicError as e:
            log(f"search {mood!r}: {e}")
            continue
        time.sleep(PAUSE_S)
        kept = [t for t in found if acceptable(t) is None and t.id not in seen]
        log(f"search {mood!r}: {len(found)} found, {len(kept)} usable, taking {min(PER_MOOD, len(kept))}")
        for t in kept[:PER_MOOD]:
            seen[t.id] = dict(track=t, mood=mood)
    out = []
    for tid, c in seen.items():
        t: music.Track = c["track"]
        if tid in measured:
            m = measured[tid]
            if m.get("ok") and tid not in EXCLUDED:
                out.append(revive(work, m))
            continue
        try:
            raw = music._get(music.API + tid + "/")
            time.sleep(PAUSE_S)
            audio = music.get(tid, dest=work)
            time.sleep(PAUSE_S)
        except music.MusicError as e:
            log(f"  skip {tid}: {e}")
            continue
        dur = probe_duration(audio)
        lufs = measure_lufs(audio)
        bpm = measure_tempo(audio)
        m = dict(track=t.to_json(), mood=c["mood"], tempo_v=TEMPO_V, duration_s=round(dur, 1), bpm=bpm, lufs=None if lufs is None else round(lufs, 1),
                 tags=tag_names(raw.get("tags")), genres=tag_names(raw.get("genres")))
        m["ok"] = MIN_S <= dur <= MAX_S and bpm is not None and lufs is not None and LUFS_RANGE[0] <= lufs <= LUFS_RANGE[1]
        measured[tid] = m
        (work / "measured.json").write_text(json.dumps(measured))
        if not m["ok"]:
            log(f"  skip {tid} {t.title!r}: duration {dur:.0f} s, bpm {bpm}, loudness {lufs}")
            continue
        out.append(revive(work, m))
        log(f"  ok   {tid} {t.title[:40]!r} by {t.creator[:24]!r}: {dur:.0f} s, {bpm} BPM, {lufs:.1f} LUFS, {audio.stat().st_size / 1e6:.1f} MB, {c['mood']}")
    return out


def original_file(work: Path, tid: str) -> Path | None:
    """The file music.get downloaded for `tid`, never one of our own outputs. Transcodes live in work/aac/ (see `transcode_path`), so work/
    holds only originals; a work folder from an older run may still hold a `{id}.m4a` transcode beside the `{id}.mp3` original, so the
    extension music.get would have used (the sidecar's filetype, mp3 for Jamendo) wins, then mp3 before m4a."""
    found = {f.suffix: f for f in work.glob(f"{tid}.*") if f.suffix in (".mp3", ".m4a") and ".part" not in f.name}
    try:
        kind = re.sub(r"[^a-z0-9]", "", str(json.loads((work / f"{tid}.json").read_text()).get("filetype") or "mp3").lower())[:5] or "mp3"
    except (OSError, ValueError, AttributeError):
        kind = "mp3"
    want = "." + ("mp3" if kind.startswith("mp3") else kind)
    return found.get(want) or found.get(".mp3") or found.get(".m4a")


def transcode_path(work: Path, tid: str) -> Path:
    d = work / "aac"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tid}.m4a"


def revive(work: Path, m: dict) -> dict:
    t = music.Track(**m["track"])
    audio = original_file(work, t.id)
    if audio is None:
        raise SystemExit(f"{t.id}: the downloaded original is missing from {work}")
    return dict(track=t, mood=m["mood"], file=audio, bytes=audio.stat().st_size, duration_s=m["duration_s"], bpm=m["bpm"], lufs=m["lufs"],
                tags=m["tags"], genres=m["genres"])


# ---------------------------------------------------------------- picking

def aac_bytes(c: dict) -> int:
    """About what the song weighs as AAC 160 kbit/s (2% for the container)."""
    return int(c["duration_s"] * AAC_KBPS * 1000 / 8 * 1.02)


def pick(cands: list[dict], count: int, budget: int) -> list[dict]:
    """Greedy: each round takes the candidate that adds the most difference in tempo (distance in octaves to the nearest chosen song)
    and in mood (a mood word not used yet counts one octave and a half), with a small penalty for size; ties by smaller file, then id.
    A candidate is allowed only while the chosen songs, as AAC 160k, still leave room for the remaining slots at the smallest size
    in the pool, so the library always fits. Deterministic."""
    chosen: list[dict] = []
    pool = sorted(cands, key=lambda c: c["track"].id)
    while pool and len(chosen) < count:
        used = sum(aac_bytes(o) for o in chosen)
        rest = count - len(chosen) - 1
        floor = sorted(aac_bytes(c) for c in pool)

        def room(c):  # c itself, then the `rest` smallest others
            others = sorted(aac_bytes(x) for x in pool if x is not c)[:rest]
            return used + aac_bytes(c) + sum(others) <= budget and len(others) == rest

        ok = [c for c in pool if room(c)]
        if not ok:
            raise SystemExit(f"cannot fit {count} songs in {budget} bytes from {len(cands)} candidates (smallest as AAC: {floor[:count]})")

        def gain(c):
            tempo = min((abs(math.log2(c["bpm"] / o["bpm"])) for o in chosen), default=1.0)
            fresh = 1.5 if all(c["mood"] != o["mood"] for o in chosen) else 0.0
            small = 0.5 * (1 - aac_bytes(c) / (budget / count * 1.5))  # a song at the average size costs nothing
            by = 0.0 if chosen else (0.1 if c["track"].license == "cc0" else 0.0)
            return (round(tempo + fresh + small + by, 6), -c["bytes"])
        best = max(ok, key=gain)
        chosen.append(best)
        pool.remove(best)
    return chosen


def transcode(src: Path, dst: Path) -> None:
    r = subprocess.run(["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map", "0:a:0", "-map_metadata", "-1",
                        "-c:a", "aac", "-b:a", f"{AAC_KBPS}k", "-movflags", "+faststart", str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg transcode failed: {r.stderr[-300:]}")


def has_picture(path: Path) -> bool:
    """True when the file carries a picture stream (an embedded album cover), which is a separate work with its own license."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def strip_pictures(src: Path, dst: Path) -> None:
    """Copy the audio stream bit for bit and the tags, but no picture: the Openverse record licenses the audio track only."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:a", "-c", "copy", "-map_metadata", "0",
                        "-id3v2_version", "3", str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg picture strip failed: {r.stderr[-300:]}")


def clean_original(c: dict, work: Path) -> None:
    """Replace c['file'] by a copy without the embedded cover picture (once), so size, hash and budget are those of what ships."""
    if c.get("cover_removed") is not None:
        return
    c["cover_removed"] = has_picture(c["file"])
    if c["cover_removed"]:
        d = work / "clean"
        d.mkdir(parents=True, exist_ok=True)
        dst = d / c["file"].name
        strip_pictures(c["file"], dst)
        c["file"], c["bytes"] = dst, dst.stat().st_size


def fit_budget(chosen: list[dict], cands: list[dict], count: int, budget: int, work: Path) -> list[dict]:
    """Original files while they fit; else transcode the biggest to AAC 160k, one at a time, until the total is within budget. If even
    that does not fit, drop the biggest song and pick a replacement."""
    banned: set[str] = set()
    while True:
        for c in chosen:
            clean_original(c, work)
        for c in chosen:
            c["final"], c["transcoded"] = c["file"], False
        for c in sorted(chosen, key=lambda c: -c["bytes"]):
            if sum(x["final"].stat().st_size for x in chosen) <= budget:
                break
            dst = transcode_path(work, c["track"].id)
            assert dst != c["file"], "a transcode must never overwrite its source"
            transcode(c["file"], dst)
            if dst.stat().st_size < c["file"].stat().st_size:
                c["final"], c["transcoded"] = dst, True
        total = sum(x["final"].stat().st_size for x in chosen)
        if total <= budget:
            return chosen
        worst = max(chosen, key=lambda c: c["final"].stat().st_size)
        banned.add(worst["track"].id)
        log(f"still {total / 1e6:.1f} MB over {budget / 1e6:.0f} MB with AAC 160k: dropping {worst['track'].title!r}")
        chosen = pick([c for c in cands if c["track"].id not in banned], count, budget)
        if len(chosen) < count:
            raise SystemExit("not enough songs fit the budget")


# ---------------------------------------------------------------- writing

def entry(c: dict) -> dict:
    t: music.Track = c["track"]
    final: Path = c["final"]
    ver = t.license_version
    return dict(id=t.id, title=t.title.strip(), artist=t.creator.strip(), artist_url=t.creator_url, license=t.license, license_version=ver,
                license_url=t.license_url, source_url=t.landing_url or t.url, source=t.source or "jamendo", duration_s=c["duration_s"],
                bpm=c["bpm"], lufs=c["lufs"], moods=[c["mood"]], tags=sorted(set(c["tags"])), genres=sorted(set(c["genres"])),
                sha256=sha256(final), bytes=final.stat().st_size, file=f"{t.id}{final.suffix}", transcoded=c["transcoded"], cover_removed=bool(c.get("cover_removed")) and not c["transcoded"],
                original_bytes=c["file"].stat().st_size if c["transcoded"] else None)


CREDITS_HEAD = """# Credits for the bundled songs

These songs ship inside inframe (`src/inframe/music_library/`) so that auto music works without a download. They are not part of
inframe's Apache-2.0 code: each song keeps its own license, named below. The songs were found through Openverse (openverse.org) and come from
Jamendo; inframe is not endorsed by Openverse or Jamendo. `library.json` has the same facts in machine-readable form, with the SHA-256 of every file.

Music inframe adds to a reel is trimmed or looped, faded in and out, and set to -14 LUFS; a reel made with a CC BY song gets its credit
line in `<video>_credits.txt` next to the reel.

"""


def credits_md(entries: list[dict], total: int) -> str:
    lines = [CREDITS_HEAD]
    by = [e for e in entries if e["license"] == "by"]
    cc0 = [e for e in entries if e["license"] == "cc0"]
    if by:
        lines.append("## CC BY (title, author, source, license)\n")
        lines += [f"- {musiclib.tasl_line(e)}" for e in by]
        lines.append("")
    if cc0:
        lines.append("## CC0\n")
        lines += [f"- {musiclib.tasl_line(e)}" for e in cc0]
        lines.append("")
    tr = [e for e in entries if e["transcoded"]]
    lines.append("## Files\n")
    if tr:
        lines.append(f"The original file was kept for {len(entries) - len(tr)} song(s). To stay within {musiclib.TOTAL_BUDGET_BYTES // 10**6} MB, "
                     f"{len(tr)} song(s) were converted to AAC {AAC_KBPS} kbit/s with ffmpeg (a change to the work, as CC BY asks us to say): "
                     + ", ".join(f'"{e["title"]}"' for e in tr) + ".")
    else:
        lines.append("Every file is the original, unchanged.")
    nc = sum(bool(e.get("cover_removed")) for e in entries)
    if nc:
        lines.append(f"\nThe album cover pictures embedded in {nc} of the original files were removed (they are separate works; the Openverse record licenses the audio "
                     "only). The audio of those files is copied unchanged.")
    lines.append(f"\n{len(entries)} songs, {total} bytes ({total / 1e6:.1f} MB) in total.")
    return "\n".join(lines) + "\n"


def write_library(chosen: list[dict], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file() and f.suffix in (".mp3", ".m4a", ".json", ".md"):
            f.unlink()
    entries = sorted((entry(c) for c in chosen), key=lambda e: (e["bpm"], e["id"]))
    for c in chosen:
        shutil.copyfile(c["final"], out / f"{c['track'].id}{c['final'].suffix}")
    total = sum(e["bytes"] for e in entries)
    (out / "library.json").write_text(json.dumps(dict(schema=musiclib.SCHEMA, total_bytes=total, songs=entries), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "CREDITS.md").write_text(credits_md(entries, total), encoding="utf-8")
    return entries


def verify(out: Path) -> int:
    """Offline: library.json complete, every file there with the recorded size and SHA-256, licenses allowed, total within budget."""
    doc = json.loads((out / "library.json").read_text(encoding="utf-8"))
    credits = (out / "CREDITS.md").read_text(encoding="utf-8")
    bad = 0
    total = 0
    for s in doc["songs"]:
        miss = [k for k in musiclib.REQUIRED if s.get(k) in (None, "")]
        f = out / str(s["file"])
        problems = [f"missing {', '.join(miss)}"] if miss else []
        if not f.is_file():
            problems.append("file missing")
        else:
            total += f.stat().st_size
            if sha256(f) != s["sha256"]:
                problems.append("sha256 differs")
        if s["license"] == "by" and (s.get("license_version") not in OK_BY_VERSIONS or s["title"] not in credits):
            problems.append("CC BY version or credit line")
        if s["license"] not in ("cc0", "by"):
            problems.append("license not allowed")
        if not MIN_S <= s["duration_s"] <= MAX_S + 1:
            problems.append("duration out of range")
        log(f"{'FAIL' if problems else 'ok  '} {s['id']} {s['title']}: {'; '.join(problems)}")
        bad += bool(problems)
    log(f"{len(doc['songs'])} songs, {total} bytes ({total / 1e6:.2f} MB)")
    if total > musiclib.TOTAL_BUDGET_BYTES:
        log("over budget")
        bad += 1
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "src" / "inframe" / "music_library")
    ap.add_argument("--work", type=Path, default=ROOT / ".logs" / "music_work", help="download folder (not committed)")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--budget-mb", type=float, default=25.0, help="decimal megabytes for all files together")
    ap.add_argument("--resume", action="store_true", help="use the candidates measured by an earlier run in --work and do not search again")
    ap.add_argument("--verify", action="store_true", help="check the committed library offline and exit")
    a = ap.parse_args()
    if a.verify:
        return verify(a.out)
    budget = int(a.budget_mb * 10**6)
    a.work.mkdir(parents=True, exist_ok=True)
    cands = gather(a.work, a.resume)
    log(f"{len(cands)} candidates measured")
    if len(cands) < a.count:
        log("not enough candidates")
        return 1
    chosen = fit_budget(pick(cands, a.count, budget), cands, a.count, budget, a.work)
    entries = write_library(chosen, a.out)
    for e in entries:
        log(f"{e['bpm']:6.1f} BPM  {e['duration_s']:5.0f} s  {e['lufs']:6.1f} LUFS  {e['bytes'] / 1e6:5.2f} MB  CC {e['license'].upper()} {e['license_version'] or ''}  "
            f"{'AAC ' if e['transcoded'] else ''}{e['moods'][0]:<16} {e['title']} - {e['artist']}")
    total = sum(e["bytes"] for e in entries)
    log(f"{len(entries)} songs, {total} bytes ({total / 1e6:.2f} MB) in {a.out}, budget {budget} bytes, "
        f"{sum(e['transcoded'] for e in entries)} transcoded to AAC {AAC_KBPS}k")
    return verify(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
