"""Add an ISO 14496-12 'sync' sample group (sgpd + sbgp) to the video track of an ffmpeg-written .mov, the way an iPhone
HEVC file carries it: every sync sample after the first is mapped to a group entry whose NAL unit type is CRA (21).
moov must come after mdat (ffmpeg's default), so growing moov does not move any chunk offset."""
import struct
import sys

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def boxes(buf, start, end):
    i = start
    while i < end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        yield i, size, typ
        i += size


def find_path(buf, start, end, path):
    """Offsets of each box along `path` (first match; for trak, the one whose hdlr is vide)."""
    for off, size, typ in boxes(buf, start, end):
        if typ != path[0]:
            continue
        if typ == b"trak" and b"vide" not in buf[off:off + size][:400]:
            continue
        if len(path) == 1:
            return [off]
        rest = find_path(buf, off + 8, off + size, path[1:])
        if rest:
            return [off] + rest
    return None


def main(src, dst):
    buf = bytearray(open(src, "rb").read())
    chain = find_path(buf, 0, len(buf), [b"moov", b"trak", b"mdia", b"minf", b"stbl"])
    stbl = chain[-1]
    stbl_size = struct.unpack(">I", buf[stbl:stbl + 4])[0]
    stss = find_path(buf, stbl + 8, stbl + stbl_size, [b"stss"])[0]
    n = struct.unpack(">I", buf[stss + 12:stss + 16])[0]
    sync = [struct.unpack(">I", buf[stss + 16 + 4 * k:stss + 20 + 4 * k])[0] for k in range(n)]
    stsz = find_path(buf, stbl + 8, stbl + stbl_size, [b"stsz"])[0]
    total = struct.unpack(">I", buf[stsz + 16:stsz + 20])[0]
    # runs of (sample_count, group_description_index): sample 1 (the IDR) gets 0, every later sync sample gets 1 (CRA)
    runs, prev = [], 0
    for s in sync[1:]:
        if s - 1 - prev > 0:
            runs.append((s - 1 - prev, 0))
        runs.append((1, 1))
        prev = s
    if total - prev > 0:
        runs.append((total - prev, 0))
    sgpd_body = struct.pack(">B3s4sII", 1, b"\0\0\0", b"sync", 1, 1) + bytes([21])  # v1, default_length 1, 1 entry: NAL 21
    sgpd = struct.pack(">I4s", 8 + len(sgpd_body), b"sgpd") + sgpd_body
    sbgp_body = struct.pack(">B3s4sI", 0, b"\0\0\0", b"sync", len(runs)) + b"".join(struct.pack(">II", c, g) for c, g in runs)
    sbgp = struct.pack(">I4s", 8 + len(sbgp_body), b"sbgp") + sbgp_body
    extra = sgpd + sbgp
    mdat = find_path(buf, 0, len(buf), [b"mdat"])[0]
    assert mdat < chain[0], "moov must follow mdat"
    end = stbl + stbl_size
    buf[end:end] = extra
    for off in chain:
        size = struct.unpack(">I", buf[off:off + 4])[0]
        buf[off:off + 4] = struct.pack(">I", size + len(extra))
    open(dst, "wb").write(buf)
    print(f"sync samples {len(sync)}, total samples {total}, sbgp runs {len(runs)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
