import { describe, expect, it } from 'vitest';
import type { EditClip } from '../src/engine/types';
import {
  ADD_LEN, HISTORY_CAP, History, MAX_CLIPS, add, applyResegment, diffSummary, fromDetector, fromDoc, mergeIds, mergeNext, remove, revert,
  split, trim, validate, type Bounds, type EditState,
} from '../src/model/edits';

const B: Bounds = { offset: 0, end: 60 };
const mk = (...iv: [number, number][]): EditState => fromDetector(iv.map(([start, end]) => ({ start, end })));
const spans = (s: EditState) => s.clips.map((c) => [c.start, c.end]);
const clip = (id: string, start: number, end: number): EditClip => ({ id, start, end, origin: 'added' });

describe('fromDetector / fromDoc', () => {
  it('names clips d<index> and starts new ids after them', () => {
    const s = fromDetector([{ index: 1, start: 1, end: 2 }, { index: 2, start: 3, end: 4 }]);
    expect(s.clips.map((c) => c.id)).toEqual(['d1', 'd2']);
    expect(s.clips.every((c) => c.origin === 'detector')).toBe(true);
    expect(s.nextId).toBe(3);
  });
  it('keeps next_id of a stored document but never reuses an id in it', () => {
    expect(fromDoc({ clips: [clip('u27', 1, 2)], next_id: 5 }).nextId).toBe(28);
    expect(fromDoc({ clips: [clip('u2', 1, 2)], next_id: 9 }).nextId).toBe(9);
  });
});

describe('trim', () => {
  const s = mk([5, 12], [20, 28], [40, 47]);
  it('moves an edge and marks a detector clip edited', () => {
    const r = trim(s, 'd2', 'start', 18.5, B);
    expect(r.changed).toBe(true);
    expect(r.state.clips[1]).toMatchObject({ start: 18.5, end: 28, origin: 'edited' });
    expect(s.clips[1].start).toBe(20); // the old state is untouched
  });
  it('clamps at the neighbours', () => {
    expect(trim(s, 'd2', 'start', 1, B).state.clips[1].start).toBe(12);
    expect(trim(s, 'd2', 'end', 99, B).state.clips[1].end).toBe(40);
  });
  it('clamps at the video bounds', () => {
    expect(trim(s, 'd1', 'start', -4, B).state.clips[0].start).toBe(0);
    expect(trim(s, 'd3', 'end', 99, B).state.clips[2].end).toBe(60);
    expect(trim(s, 'd1', 'start', 3, { offset: 4.5, end: 60 }).state.clips[0].start).toBe(4.5);
  });
  it('keeps the 0.2 s minimum', () => {
    expect(trim(s, 'd1', 'start', 99, B).state.clips[0].start).toBeCloseTo(11.8, 6);
    expect(trim(s, 'd1', 'end', 0, B).state.clips[0].end).toBeCloseTo(5.2, 6);
  });
  it('rounds to a millisecond and says why when nothing moves', () => {
    expect(trim(s, 'd1', 'end', 13.00049, B).state.clips[0].end).toBe(13);
    const same = trim(s, 'd1', 'start', 5, B);
    expect(same.changed).toBe(false);
    expect(same.reason).toBeTruthy();
    expect(trim(s, 'nope', 'start', 5, B).changed).toBe(false);
  });
  it('keeps the origin of a clip the user made', () => {
    const a = add(mk(), 10, B).state;
    expect(trim(a, a.clips[0].id, 'end', 10.5, B).state.clips[0].origin).toBe('added');
  });
});

describe('split', () => {
  const s = mk([5, 12], [20, 28]);
  it('splits the clip under the playhead; the first part keeps its id', () => {
    const r = split(s, 8, undefined);
    expect(r.changed).toBe(true);
    expect(spans(r.state)).toEqual([[5, 8], [8, 12], [20, 28]]);
    expect(r.state.clips.map((c) => c.id)).toEqual(['d1', 'u3', 'd2']);
    expect(r.state.clips[0].origin).toBe('split');
    expect(r.state.clips[1].origin).toBe('split');
    expect(r.state.nextId).toBe(4);
  });
  it('refuses outside any clip, in the wrong clip, or when a part would be under 0.2 s', () => {
    expect(split(s, 15).reason).toMatch(/not inside a clip/);
    expect(split(s, 8, 'd2').reason).toMatch(/not inside that clip/);
    expect(split(s, 5.1).reason).toBe('Too short to split here');
    expect(split(s, 11.9).reason).toBe('Too short to split here');
    expect(split(s, 5.2).changed).toBe(true);
  });
});

describe('merge', () => {
  const s = mk([5, 12], [20, 28], [40, 47]);
  it('merges a clip with the next one, gap included', () => {
    const r = mergeNext(s, 'd1');
    expect(spans(r.state)).toEqual([[5, 28], [40, 47]]);
    expect(r.state.clips[0]).toMatchObject({ id: 'd1', origin: 'merge' });
  });
  it('has nothing to merge with after the last clip', () => {
    expect(mergeNext(s, 'd3').changed).toBe(false);
  });
  it('merges three selected clips into one', () => {
    const r = mergeIds(s, ['d1', 'd2', 'd3']);
    expect(spans(r.state)).toEqual([[5, 47]]);
    expect(r.state.clips[0]).toMatchObject({ id: 'd1', origin: 'merge' });
  });
  it('takes the clips between two selected ones along, and needs two', () => {
    expect(spans(mergeIds(s, ['d1', 'd3']).state)).toEqual([[5, 47]]);
    expect(mergeIds(s, ['d2']).changed).toBe(false);
  });
});

describe('remove', () => {
  it('deletes the given clips', () => {
    const s = mk([5, 12], [20, 28], [40, 47]);
    expect(spans(remove(s, ['d1', 'd3']).state)).toEqual([[20, 28]]);
    expect(remove(s, ['zzz']).changed).toBe(false);
  });
});

describe('add', () => {
  it('makes a 2 s clip centred on the playhead, in order', () => {
    const s = mk([5, 12], [40, 47]);
    const r = add(s, 20, B);
    expect(spans(r.state)).toEqual([[5, 12], [19, 21], [40, 47]]);
    expect(r.state.clips[1]).toMatchObject({ id: 'u3', origin: 'added' });
    expect(ADD_LEN).toBe(2);
  });
  it('shrinks into a narrow gap', () => {
    const s = mk([5, 12], [12.8, 20]);
    expect(spans(add(s, 12.4, B).state)).toEqual([[5, 12], [12, 12.8], [12.8, 20]]);
  });
  it('shrinks at the ends of the video', () => {
    expect(spans(add(mk(), 0.5, B).state)).toEqual([[0, 1.5]]);
    expect(spans(add(mk(), 59.5, B).state)).toEqual([[58.5, 60]]);
  });
  it('refuses inside a clip, or in a gap under 0.2 s', () => {
    const s = mk([5, 12], [12.1, 20]);
    expect(add(s, 8, B).reason).toMatch(/inside a clip/);
    expect(add(s, 12.05, B).reason).toBe('No room for a clip here');
  });
});

describe('applyResegment', () => {
  it('replaces the list with fresh ids and origin resegment', () => {
    const s = mk([5, 12]);
    const r = applyResegment(s, [[1, 3], [6, 9]], B);
    expect(r.changed).toBe(true);
    expect(r.state.clips.map((c) => [c.id, c.origin])).toEqual([['u2', 'resegment'], ['u3', 'resegment']]);
    expect(r.state.nextId).toBe(4);
  });
  it('refuses a list that breaks the rules, or changes nothing', () => {
    const s = mk([5, 12]);
    expect(applyResegment(s, [[1, 1.1]], B).changed).toBe(false);
    expect(applyResegment(s, [[5, 12]], B).changed).toBe(false);
  });
});

describe('history', () => {
  it('undoes and redoes whole states; a new change drops the redo steps', () => {
    let s = mk([5, 12]);
    const h = new History();
    const step = (r: ReturnType<typeof split>) => { h.push(s); s = r.state; };
    step(split(s, 8));
    step(remove(s, ['d1']));
    expect(s.clips).toHaveLength(1);
    s = h.undo(s)!;
    expect(s.clips).toHaveLength(2);
    s = h.undo(s)!;
    expect(s.clips).toHaveLength(1);
    expect(h.undo(s)).toBeNull();
    s = h.redo(s)!;
    expect(s.clips).toHaveLength(2);
    step(remove(s, ['d1']));
    expect(h.canRedo).toBe(false);
  });
  it('keeps ids unique across an undo', () => {
    let s = mk([5, 12]);
    const h = new History();
    h.push(s);
    s = split(s, 8).state;
    expect(h.undo(s)!.nextId).toBe(3);
  });
  it('keeps at most 200 steps', () => {
    const h = new History();
    let s = mk([5, 12]);
    for (let i = 0; i < 250; i++) { h.push(s); s = { ...s, nextId: s.nextId + 1 }; }
    expect(h.past).toHaveLength(HISTORY_CAP);
    let n = 0;
    while ((s = h.undo(s) ?? s) && h.canUndo) n++;
    expect(n + 1).toBe(200);
  });
  it('treats Revert to detector as one undoable step', () => {
    const detector = mk([5, 12], [20, 28]);
    let s = split(trim(detector, 'd2', 'end', 30, B).state, 8).state;
    const h = new History();
    h.push(s);
    const r = revert(s, detector.clips);
    expect(r.changed).toBe(true);
    s = r.state;
    expect(spans(s)).toEqual([[5, 12], [20, 28]]);
    expect(revert(s, detector.clips).changed).toBe(false);
    s = h.undo(s)!;
    expect(spans(s)).toEqual([[5, 8], [8, 12], [20, 30]]);
  });
});

describe('validate (the server’s table)', () => {
  const ok = [clip('a1', 1, 3), clip('a2', 3, 5)];
  it('accepts a clean list, touching clips, and exactly 0.2 s', () => {
    expect(validate(ok, B)).toEqual([]);
    expect(validate([clip('a1', 1, 1.2)], B)).toEqual([]);
    expect(validate([clip('a1', 0, 60)], B)).toEqual([]);
    expect(validate([], B)).toEqual([]);
  });
  it('refuses overlap', () => expect(validate([clip('a1', 1, 4), clip('a2', 3, 5)], B).join()).toMatch(/overlap/));
  it('refuses an unsorted list', () => expect(validate([clip('a1', 10, 12), clip('a2', 3, 5)], B).join()).toMatch(/sorted/));
  it('refuses 0.19 s', () => expect(validate([clip('a1', 1, 1.19)], B).join()).toMatch(/shorter/));
  it('refuses clips outside the video, with 1 ms slack', () => {
    expect(validate([clip('a1', -0.5, 2)], B).join()).toMatch(/outside/);
    expect(validate([clip('a1', 59, 60.5)], B).join()).toMatch(/outside/);
    expect(validate([clip('a1', 59, 60.0005)], B)).toEqual([]);
    expect(validate([clip('a1', 1, 3)], { offset: 2, end: 60 }).join()).toMatch(/outside/);
  });
  it('refuses 2001 clips', () => {
    const many = Array.from({ length: MAX_CLIPS + 1 }, (_, i) => clip(`c${i}`, i * 0.25, i * 0.25 + 0.2));
    expect(validate(many, { offset: 0, end: 1000 }).join()).toMatch(/too many/);
    expect(validate(many.slice(0, MAX_CLIPS), { offset: 0, end: 1000 })).toEqual([]);
  });
  it('refuses duplicate ids and bad ids', () => {
    expect(validate([clip('a1', 1, 3), clip('a1', 4, 6)], B).join()).toMatch(/duplicate/);
    expect(validate([clip('A_1', 1, 3)], B).join()).toMatch(/bad id/);
  });
  it('refuses NaN and infinity', () => {
    expect(validate([clip('a1', NaN, 3)], B).join()).toMatch(/finite/);
    expect(validate([clip('a1', 1, Infinity)], B).join()).toMatch(/finite/);
  });
});

describe('diffSummary', () => {
  it('counts changed, deleted and added clips against the detector', () => {
    const det = mk([5, 12], [20, 28], [40, 47]);
    let s = trim(det, 'd1', 'end', 13, B).state;
    s = remove(s, ['d2']).state;
    s = add(s, 55, B).state;
    const d = diffSummary(s.clips, det.clips);
    expect(d).toMatchObject({ changed: 1, deleted: 1, added: 1, text: 'edited: 1 changed, 1 deleted, 1 added' });
    expect(diffSummary(det.clips, det.clips).text).toBe('');
  });
  it('counts a split as one changed and one added, a merge as a change and a deletion', () => {
    const det = mk([5, 12], [20, 28]);
    expect(diffSummary(split(det, 8).state.clips, det.clips).text).toBe('edited: 1 changed, 1 added');
    expect(diffSummary(mergeNext(det, 'd1').state.clips, det.clips).text).toBe('edited: 1 changed, 1 deleted');
  });
});
