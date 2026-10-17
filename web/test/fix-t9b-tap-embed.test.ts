// BrowserTap.resolve and embedDetections in crowded pictures: the tap lands on a detection that has a look, and the tapped person is embedded
// whatever their rank (docs/T9B_DESIGN.md section 5).
import { describe, expect, it } from 'vitest';
import type { Detection } from '../src/browser/detect';
import { embedDetections, MAX_EMBED, type EmbedStats } from '../src/browser/embed';
import { BrowserTap, TapError } from '../src/browser/targets';

type B = [number, number, number, number];
const look = (n = 0) => { const v = new Float32Array(8); v[n] = 1; return v; };
const pic = { rgba: new Uint8ClampedArray(640 * 640 * 4), w: 640, h: 640 };
const fakeDino = { embed: async () => look(0) };

describe('BrowserTap.resolve skips detections that were never embedded', () => {
  it('a faint duplicate that overlaps the tap more than the real person does not make the run fail', () => {
    const t = new BrowserTap([{ t: 1, box: [0.3, 0.1, 0.5, 0.9] }]);
    t.evaluate([{ score: 0.9, box: [0.33, 0.1, 0.53, 0.9], embedding: look(0) }, { score: 0.25, box: [0.3, 0.12, 0.49, 0.9], embedding: null }] as Detection[], 1);
    const out = t.finalize();
    expect(out[0].score).toBe(0.9);
  });

  it('falls to another picture within 1 s when the nearest one has nothing embedded under the tap', () => {
    const t = new BrowserTap([{ t: 1, box: [0.3, 0.1, 0.5, 0.9] }]);
    t.evaluate([{ score: 0.25, box: [0.3, 0.1, 0.5, 0.9], embedding: null }] as Detection[], 1);
    t.evaluate([{ score: 0.9, box: [0.3, 0.1, 0.5, 0.9], embedding: look(0) }] as Detection[], 1.4);
    expect(t.finalize().map((h) => h.score)).toEqual([0, 0.9]);
  });

  it('still says too small when every overlapping detection lacks a look, and not detected when nothing overlaps', () => {
    const small = new BrowserTap([{ t: 1, box: [0.3, 0.1, 0.5, 0.9] }]);
    small.evaluate([{ score: 0.9, box: [0.3, 0.1, 0.5, 0.9], embedding: null }] as Detection[], 1);
    expect(() => small.finalize()).toThrow(/too small to recognise/);
    const none = new BrowserTap([{ t: 1, box: [0.3, 0.1, 0.5, 0.9] }]);
    none.evaluate([{ score: 0.9, box: [0.7, 0.1, 0.9, 0.9], embedding: look(0) }] as Detection[], 1);
    expect(() => none.finalize()).toThrow(TapError);
    expect(() => none.finalize()).toThrow(/nothing at that box/);
  });
});

describe('embedDetections in a crowded picture', () => {
  const crowd = (): Detection[] =>
    Array.from({ length: 10 }, (_, i) => ({ score: 0.95 - i * 0.05, box: [0.02 + i * 0.095, 0.1, 0.09 + i * 0.095, 0.9] as B, embedding: null })) as Detection[];

  it('the 10th person of ten, tapped, is embedded and the tap resolves on them', async () => {
    const dets = crowd();
    const tapBox = dets[9].box;
    await embedDetections(fakeDino, pic, dets, { wanted: [tapBox] });
    expect(dets[9].embedding).not.toBeNull();
    const t = new BrowserTap([{ t: 2, box: tapBox }]);
    t.evaluate(dets, 2);
    expect(t.finalize()[0].box).toEqual(tapBox);
  });

  it('without a tap, only MAX_EMBED are embedded and the rest are counted in stats.capped', async () => {
    const dets = crowd();
    const stats: EmbedStats = { capped: 0 };
    const n = await embedDetections(fakeDino, pic, dets, { stats });
    expect(n).toBe(MAX_EMBED);
    expect(stats.capped).toBe(2);
    expect(dets.slice(MAX_EMBED).every((d) => d.embedding === null)).toBe(true);
  });
});
