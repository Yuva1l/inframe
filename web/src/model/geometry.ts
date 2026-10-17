// Placing overlays over a letterboxed video (docs/T9A_DESIGN.md 1.1, 2.4): object-fit: contain math on the server's display size
// (rotation applied), never on videoWidth/videoHeight.

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Where `contentW x contentH` lands inside a `boxW x boxH` box with object-fit: contain. */
export function containRect(boxW: number, boxH: number, contentW: number, contentH: number): Rect {
  if (!(boxW > 0 && boxH > 0 && contentW > 0 && contentH > 0)) return { x: 0, y: 0, w: Math.max(0, boxW), h: Math.max(0, boxH) };
  const s = Math.min(boxW / contentW, boxH / contentH);
  const w = contentW * s;
  const h = contentH * s;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

/** A box [x0, y0, x1, y1] of frame fractions as pixels inside `frame`. */
export function boxToRect(frame: Rect, box: readonly [number, number, number, number]): Rect {
  return { x: frame.x + box[0] * frame.w, y: frame.y + box[1] * frame.h, w: (box[2] - box[0]) * frame.w, h: (box[3] - box[1]) * frame.h };
}

/** A zone [x, y, w, h] of frame fractions as pixels inside `frame`. */
export function zoneToRect(frame: Rect, zone: readonly [number, number, number, number]): Rect {
  return { x: frame.x + zone[0] * frame.w, y: frame.y + zone[1] * frame.h, w: zone[2] * frame.w, h: zone[3] * frame.h };
}

/** A pixel position (relative to the same box as `frame`) as frame fractions, or null outside the frame. */
export function pointToFrame(frame: Rect, px: number, py: number): [number, number] | null {
  if (!(frame.w > 0 && frame.h > 0)) return null;
  const fx = (px - frame.x) / frame.w;
  const fy = (py - frame.y) / frame.h;
  return fx < 0 || fx > 1 || fy < 0 || fy > 1 ? null : [fx, fy];
}

/** The index of the smallest box containing the point (fractions), or -1. */
export function hitBox(boxes: readonly (readonly [number, number, number, number])[], fx: number, fy: number): number {
  let best = -1;
  let bestArea = Infinity;
  boxes.forEach((b, i) => {
    if (fx >= b[0] && fx <= b[2] && fy >= b[1] && fy <= b[3]) {
      const a = (b[2] - b[0]) * (b[3] - b[1]);
      if (a < bestArea) {
        best = i;
        bestArea = a;
      }
    }
  });
  return best;
}
