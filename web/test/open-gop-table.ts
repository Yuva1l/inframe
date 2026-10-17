// A synthetic open-GOP sample table shared by the plan tests (nothing is decoded).
/** An open-GOP stream: a keyframe every `gop` pictures, shown `lead` pictures after the first picture of its group, so `lead` pictures decoded after it are shown before it
 *  (decode order: key, the leading pictures, the rest). Group k covers shown slots k * gop .. (k + 1) * gop - 1. */
export function openGopTable(fps: number, groups: number, gop: number, lead: number) {
  const n = groups * gop;
  const pts = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = Math.floor(i / gop) * gop;
    const k = i - g;
    pts[i] = (g + (k === 0 ? lead : k <= lead ? k - 1 : k)) / fps;
  }
  const order = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => pts[a] - pts[b]);
  const sync = Uint8Array.from({ length: n }, (_, i) => (i % gop === 0 ? 1 : 0));
  return { n, pts, dts: Float64Array.from({ length: n }, (_, i) => i / fps), offset: new Float64Array(n), size: new Uint32Array(n).fill(1), sync, order, tsUs: Float64Array.from(pts, (p) => Math.round(p * 1e6)) };
}
