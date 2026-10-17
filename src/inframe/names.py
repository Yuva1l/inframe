"""The one place a source's file stem becomes the stem of the names this tool writes."""
from __future__ import annotations


def safe_stem(stem: str) -> str:
    """`stem` with each control character (a line break, a NUL, DEL) replaced by `_`, so it cannot split a concat list or a log line.
    Every name this tool writes (clips, sidecars, reel, credits, edit lists, LosslessCut CSV) and every pattern that finds them
    again must start from this, or a re-run or re-export would not recognise the files it wrote before."""
    return "".join("_" if ord(ch) < 32 or ord(ch) == 127 else ch for ch in stem)
