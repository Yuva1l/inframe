// The zone rectangle (docs/T9A_DESIGN.md 1.4): [x, y, w, h] as fractions of the upright frame, each side at least 0.02, never outside
// the frame, rounded as pick._round_zone does (x and y to 4 decimals, then w and h from the rounded far edge).
import type { Zone } from '../engine/types';

export const MIN_SIDE = 0.02;
export type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';
export const HANDLES: Handle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);
const r4 = (v: number) => Math.round(v * 10000) / 10000;

export function roundZone(z: Zone): Zone {
  const x = r4(z[0]);
  const y = r4(z[1]);
  return [x, y, r4(Math.min(r4(z[0] + z[2]), 1) - x), r4(Math.min(r4(z[1] + z[3]), 1) - y)];
}

/** The rectangle between two points (frame fractions), or null when either side is under the minimum. */
export function fromDrag(a: readonly [number, number], b: readonly [number, number]): Zone | null {
  const x0 = clamp01(Math.min(a[0], b[0]));
  const x1 = clamp01(Math.max(a[0], b[0]));
  const y0 = clamp01(Math.min(a[1], b[1]));
  const y1 = clamp01(Math.max(a[1], b[1]));
  if (x1 - x0 < MIN_SIDE || y1 - y0 < MIN_SIDE) return null;
  return roundZone([x0, y0, x1 - x0, y1 - y0]);
}

/** Move one edge or corner by (dx, dy) fractions; the opposite sides stay put, and the minimum side holds. */
export function resize(z: Zone, handle: Handle, dx: number, dy: number): Zone {
  let [x0, y0, x1, y1] = [z[0], z[1], z[0] + z[2], z[1] + z[3]];
  if (handle.includes('w')) x0 = Math.min(clamp01(x0 + dx), x1 - MIN_SIDE);
  if (handle.includes('e')) x1 = Math.max(clamp01(x1 + dx), x0 + MIN_SIDE);
  if (handle.includes('n')) y0 = Math.min(clamp01(y0 + dy), y1 - MIN_SIDE);
  if (handle.includes('s')) y1 = Math.max(clamp01(y1 + dy), y0 + MIN_SIDE);
  return roundZone([x0, y0, x1 - x0, y1 - y0]);
}

/** Move the whole rectangle, keeping it inside the frame. */
export function move(z: Zone, dx: number, dy: number): Zone {
  const x = Math.min(Math.max(z[0] + dx, 0), 1 - z[2]);
  const y = Math.min(Math.max(z[1] + dy, 0), 1 - z[3]);
  return roundZone([x, y, z[2], z[3]]);
}

/** "Zone 20 % x 30 %, 10 % from the left" is too long for a card: the size as a share of the frame. */
export function zoneText(z: Zone): string {
  return `${Math.round(z[2] * 100)} % x ${Math.round(z[3] * 100)} % of the frame`;
}
