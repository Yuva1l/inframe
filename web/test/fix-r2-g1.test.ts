// UI fixes: the client names the motion run folder like the server (serve_runs.py: "motion-ui").
import { describe, expect, it } from 'vitest';
import { slugOf } from '../src/workflow';

describe('rr-1: motion run folder', () => {
  it('slugOf(motion) is motion-ui, so checkRerun finds the old run and the edits ask shows', () => {
    expect(slugOf({ kind: 'motion' })).toBe('motion-ui');
  });
  it('the other kinds keep their server names', () => {
    expect(slugOf({ kind: 'tap' })).toBe('tap');
    expect(slugOf({ kind: 'photos' })).toBe('photos');
    expect(slugOf({ kind: 'any' })).toBe('any-person');
  });
});
