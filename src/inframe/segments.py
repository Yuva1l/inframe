"""Pure segment logic: hysteresis, gap merge, minimum length, padding, clamping. No I/O."""
from __future__ import annotations

Interval = tuple[float, float]


def sample_edges(times: list[float], duration: float, lo: float = 0.0) -> list[float]:
    """Boundaries of each sample's cell: midpoints between neighbours; the first and last cell extend
    half a gap outward, clamped to [lo, duration]. Returns len(times) + 1 edges."""
    n = len(times)
    if n == 0:
        return []
    if n == 1:
        return [max(lo, times[0] - 0.5), min(duration, times[0] + 0.5)]
    mids = [(a + b) / 2 for a, b in zip(times, times[1:])]
    first = max(lo, times[0] - (mids[0] - times[0]))
    last = min(duration, times[-1] + (times[-1] - mids[-1]))
    return [first, *mids, last]


def hysteresis(times: list[float], scores: list[float], enter: float, exit: float, duration: float, lo: float = 0.0) -> list[Interval]:
    """Presence intervals from per-sample scores. Enter when score >= enter, leave when score < exit
    (exit <= enter keeps one weak sample from splitting a stretch). Boundaries sit halfway between
    the last sample on one side and the first on the other."""
    if len(times) != len(scores):
        raise ValueError("times and scores differ in length")
    if exit > enter:
        raise ValueError("exit threshold must not exceed enter threshold")
    edges = sample_edges(times, duration, lo)
    out: list[Interval] = []
    start: float | None = None
    for i, s in enumerate(scores):
        if start is None:
            if s >= enter:
                start = edges[i]
        elif s < exit:
            out.append((start, edges[i]))
            start = None
    if start is not None:
        out.append((start, edges[-1]))
    return out


def merge_gaps(intervals: list[Interval], gap: float) -> list[Interval]:
    """Join neighbours separated by at most `gap` seconds."""
    out: list[Interval] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def drop_short(intervals: list[Interval], min_len: float) -> list[Interval]:
    return [(s, e) for s, e in intervals if e - s >= min_len]


def pad(intervals: list[Interval], before: float, after: float, duration: float, lo: float = 0.0) -> list[Interval]:
    """Grow each interval, clamp to [lo, duration] (lo is the video's first picture in file time, 0 when it starts at 0), then merge any that now touch or overlap."""
    grown = [(max(lo, s - before), min(duration, e + after)) for s, e in intervals]
    return merge_gaps(grown, 0.0)


def build_segments(
    times: list[float],
    scores: list[float],
    duration: float,
    enter: float = 0.5,
    exit: float = 0.3,
    merge_gap: float = 1.5,
    min_len: float = 1.0,
    pad_s: float = 1.0,
    lo: float = 0.0,
) -> tuple[list[Interval], list[Interval]]:
    """(presence intervals unpadded, clip intervals padded). Order: hysteresis, merge, min length, pad."""
    presence = drop_short(merge_gaps(hysteresis(times, scores, enter, exit, duration, lo), merge_gap), min_len)
    return presence, pad(presence, pad_s, pad_s, duration, lo)
