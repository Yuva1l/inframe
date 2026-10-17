import { describe, expect, it } from 'vitest';
import { formatLength } from '../src/model/format';

describe('formatLength', () => {
  it('prints short and long lengths', () => {
    expect(formatLength(7.5)).toBe('7.5 s');
    expect(formatLength(72)).toBe('1 min 12 s');
  });
  it('never prints 60 s next to the minutes', () => {
    expect(formatLength(119.6)).toBe('2 min 0 s');
    expect(formatLength(59.96)).toBe('1 min 0 s');
    expect(formatLength(179.5)).toBe('3 min 0 s');
    expect(formatLength(60)).toBe('1 min 0 s');
  });
});
