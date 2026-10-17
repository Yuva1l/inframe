import { describe, expect, it } from 'vitest';
import { clockOffset, toFileTime, toMediaTime } from '../src/model/clock';

describe('clock mapping', () => {
  it('maps the "late" clip (first frame at file time 2.0) without a browser', () => {
    const m0 = 0; // the fake rVFC mediaTime of the first frame
    const offset = clockOffset(2.0, m0);
    expect(offset).toBe(2);
    expect(toFileTime(5.2, offset)).toBeCloseTo(7.2, 9);
    expect(toMediaTime(7.2, offset)).toBeCloseTo(5.2, 9);
    // the white flash is at file time 7.0-7.4: white at 7.2, black at 6.8 and 7.6
    const white = (file: number) => file >= 7.0 && file <= 7.4;
    expect(white(toFileTime(5.2, offset))).toBe(true);
    expect(white(toFileTime(4.8, offset))).toBe(false);
    expect(white(toFileTime(5.6, offset))).toBe(false);
  });
  it('allows a player whose first frame is not at 0', () => {
    const offset = clockOffset(0.007, 0.033);
    expect(toFileTime(0.033, offset)).toBeCloseTo(0.007, 9);
    expect(toMediaTime(toFileTime(12.345, offset), offset)).toBeCloseTo(12.345, 9);
  });
  it('is the identity when both clocks start together', () => {
    expect(clockOffset(0, 0)).toBe(0);
    expect(toFileTime(3.3, 0)).toBe(3.3);
  });
});
