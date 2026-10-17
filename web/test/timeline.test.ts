import { describe, expect, it } from 'vitest';
import { fit, formatTime, snap, tickStep, timeToX, viewEnd, xToTime, zoomAt } from '../src/model/timeline';

const span = { offset: 0, end: 600 };

describe('view maths', () => {
  it('maps time to x and back', () => {
    const v = { t0: 10, pps: 20, width: 1000 };
    expect(timeToX(v, 12.5)).toBe(50);
    expect(xToTime(v, 50)).toBe(12.5);
    expect(viewEnd(v)).toBe(60);
  });
  it('fits the whole video, from its first frame', () => {
    expect(fit(600, 1200, 0)).toEqual({ t0: 0, pps: 2, width: 1200 });
    expect(fit(600, 1200, 2).t0).toBe(2);
  });
});

describe('zoomAt', () => {
  it('keeps the anchor where it is', () => {
    const v = { t0: 100, pps: 4, width: 1000 };
    const z = zoomAt(v, 150, 2, span);
    expect(z.pps).toBe(8);
    expect(timeToX(z, 150)).toBeCloseTo(timeToX(v, 150), 9);
  });
  it('never zooms out past the whole video or in past 400 px/s', () => {
    const v = fit(600, 1200, 0);
    expect(zoomAt(v, 300, 0.1, span)).toEqual(v);
    expect(zoomAt({ t0: 0, pps: 300, width: 1200 }, 5, 10, span).pps).toBe(400);
  });
  it('stays inside the video', () => {
    const z = zoomAt({ t0: 0, pps: 4, width: 1000 }, 0, 0.5, span);
    expect(z.t0).toBe(0);
    const w = zoomAt({ t0: 350, pps: 4, width: 1000 }, 599, 2, span);
    expect(viewEnd(w)).toBeLessThanOrEqual(600 + 1e-9);
  });
});

describe('tickStep', () => {
  it('picks the first step with at least 80 px between ticks', () => {
    expect(tickStep(400)).toBe(0.2);
    expect(tickStep(100)).toBe(1);
    expect(tickStep(2)).toBe(60);
    expect(tickStep(0.01)).toBe(1800);
    expect(tickStep(10000)).toBeCloseTo(1 / 30, 9);
  });
});

describe('formatTime', () => {
  it('writes m:ss and h:mm:ss', () => {
    expect(formatTime(65, 1)).toBe('1:05');
    expect(formatTime(0, 5)).toBe('0:00');
    expect(formatTime(3725, 1)).toBe('1:02:05');
    expect(formatTime(59.9996, 1)).toBe('1:00');
  });
  it('adds tenths below one second steps', () => {
    expect(formatTime(65.34, 0.5)).toBe('1:05.3');
    expect(formatTime(0, 0.1)).toBe('0:00.0');
    expect(formatTime(59.97, 0.2)).toBe('1:00.0');
    expect(formatTime(3600.5, 0.5)).toBe('1:00:00.5');
  });
  it('does not go negative', () => expect(formatTime(-3, 1)).toBe('0:00'));
});

describe('snap', () => {
  it('takes the nearest candidate within 8 px', () => {
    expect(snap(5.5, [5.2, 9, 5.45], 10)).toBe(5.45);
    expect(snap(5.5, [7], 10)).toBe(5.5);
    expect(snap(5.5, [6.2], 10)).toBe(6.2); // 0.7 s is 7 px at 10 px/s
  });
  it('otherwise lands on the frame grid', () => {
    expect(snap(5.51, [100], 10, 8, 30)).toBeCloseTo(5.5, 9);
    expect(snap(5.514, [], 10, 8, 30)).toBeCloseTo(5.5, 9);
    expect(snap(5.514, [], 10)).toBe(5.514);
  });
  it('follows the first frame of the file when it is not at zero', () => {
    expect(snap(2.51, [], 10, 8, 10, 2.03)).toBeCloseTo(2.53, 9);
  });
});
