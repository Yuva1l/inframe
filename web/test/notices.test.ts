// The licence decisions the release depends on are written down where the page and the repo carry them,
// and what is written about the origin of the YUV to RGB stage is what actually happened.
import { describe, expect, it } from 'vitest';
// @ts-ignore
import { readFileSync, existsSync } from 'node:fs';

const read = (rel: string): string => readFileSync((globalThis as any).process.cwd() + '/' + rel, 'utf8');
const entryOf = (text: string): string => text.split('\n---\n').find((e: string) => /^\s*Name: The YUV to RGB stage/.test(e)) ?? '';
// Claims that the author of yuv.ts had not seen the earlier implementation, or that the spec came from the maths alone, are false (see the notice).
const readmeLine = (): string => read('../README.md').split('\n').find((l: string) => /yuv\.ts/.test(l) && /Added in the browser parity fix, part 2/.test(l)) ?? '';
const FALSE_CLAIMS = [/had not read/i, /written from the maths and from/i, /from that specification alone, without reading/i, /is a clean-room implementation/i];

describe('third-party notices', () => {
  it('web/src/notices.txt carries a provenance note for the YUV to RGB stage', () => {
    const notices = read('src/notices.txt');
    const entry = entryOf(notices);
    expect(entry).toMatch(/License: Apache-2\.0/);
    expect(entry).toContain('web/src/browser/yuv.ts');
    expect(entry).toContain('docs/t9b/YUV_SPEC.md');
  });
  it('web/src/notices.txt names FFmpeg libswscale, its LGPL licence and the file that follows it', () => {
    const entry = read('src/notices.txt').split('\n---\n').find((e: string) => /^\s*Name: FFmpeg libswscale/.test(e)) ?? '';
    expect(entry).toMatch(/License: LGPL-2\.1-or-later/);
    expect(entry).toContain('web/src/browser/resample.ts');
    expect(entry).toContain('web/src/browser/yuv.ts');
    expect(entry).toMatch(/Decision \(recorded \d{4}-\d{2}-\d{2}/);
    expect(entry).toContain('GNU LESSER GENERAL PUBLIC LICENSE');
  });
  it('the notice says what happened: the spec was written from the swscale-following code, which stayed reachable, so it is not a verified clean-room', () => {
    const entry = entryOf(read('src/notices.txt'));
    for (const re of FALSE_CLAIMS) expect(entry).not.toMatch(re);
    expect(entry).toMatch(/not a verified clean-room/);
    expect(entry).toMatch(/LGPL/);
    expect(entry).toMatch(/git history/);
    expect(entry).toMatch(/tableCache|colour tables/);
    expect(entry).toMatch(/no history/);
  });
  it('the notice names the filter construction that ships and treats it as derived, not as scaffolding', () => {
    const entry = entryOf(read('src/notices.txt'));
    const line = readmeLine();
    // the identifiers of yuv.ts that are the earlier code's own: all of them are named, the filter first
    for (const id of ['makeFilter', 'filterCache', 'tableCache', 'tablesOf']) expect(entry, id).toContain(id);
    expect(entry).toMatch(/treated as derived/);
    expect(entry).toMatch(/ships with the page/);
    expect(entry).not.toMatch(/not swscale's filter/);
    expect(entry).not.toMatch(/nothing from FFmpeg ships/i);
    expect(line).toContain('makeFilter');
    expect(line).toMatch(/treated as derived/);
    expect(line).not.toMatch(/not swscale's filter/);
    // the code does carry them, so the list cannot go stale unnoticed
    const code = read('src/browser/yuv.ts');
    for (const id of ['makeFilter', 'filterCache']) expect(code, id).toContain(id);
  });
  it('the notice and the README do not point at git history or a private record file', () => {
    const entry = entryOf(read('src/notices.txt'));
    const line = readmeLine();
    for (const [name, text] of [['notice', entry], ['readme', line], ['spec', read('../docs/t9b/YUV_SPEC.md')]] as const) {
      expect(text, name).not.toMatch(/\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b/);
      expect(text, name).not.toMatch(/commit [0-9a-f]{7}/);
    }
  });
  it('the shipped THIRD_PARTY_LICENSES.txt carries the same entry as the source notice', () => {
    expect(entryOf(read('../src/inframe/web/dist/THIRD_PARTY_LICENSES.txt')).trim()).toBe(entryOf(read('src/notices.txt')).trim());
  });
  it('the README states the same provenance next to the Pillow notice, with no false clean-room claim', () => {
    const readme = read('../README.md');
    const line = readmeLine();
    expect(line).toMatch(/not a verified clean-room/);
    for (const re of FALSE_CLAIMS) expect(line).not.toMatch(re);
    expect(readme).not.toMatch(/Licence decision[^.]*initFilter/);
  });
  it('the spec and the code headers do not call the stage clean-room', () => {
    const spec = read('../docs/t9b/YUV_SPEC.md');
    expect(spec).not.toMatch(/This is the clean-room specification/);
    expect(spec).toMatch(/not independent of the earlier code/);
    for (const f of ['src/browser/yuv.ts', 'src/browser/resample.ts']) {
      const head = read(f).split('\n').slice(0, 14).join('\n');
      expect(head, f).not.toMatch(/written clean-room/i);
    }
  });
  it('the build appends notices.txt to THIRD_PARTY_LICENSES.txt, so the entry ships', () => {
    expect(read('vite.config.ts')).toContain("'notices.txt'");
  });
});

describe('the licence of the repository', () => {
  it('has a LICENSE file with the Apache-2.0 text; the web package says Apache-2.0 AND LGPL-2.1-or-later (yuv.ts and resample.ts), the Python package adds CC-BY-3.0 (the bundled songs)', () => {
    expect(existsSync((globalThis as any).process.cwd() + '/../LICENSE')).toBe(true);
    const lic = read('../LICENSE');
    expect(lic).toMatch(/Apache License\s+Version 2\.0, January 2004/);
    expect(lic).toBe(read('../tools/hf_files/LICENSE-APACHE'));
    expect(JSON.parse(read('package.json')).license).toBe('Apache-2.0 AND LGPL-2.1-or-later');
    expect(JSON.parse(read('package-lock.json')).packages[''].license).toBe('Apache-2.0 AND LGPL-2.1-or-later');
    expect(read('../pyproject.toml')).toMatch(/^license = "Apache-2\.0 AND LGPL-2\.1-or-later AND CC-BY-3\.0"/m);
    expect(read('../pyproject.toml')).toMatch(/^license-files = \[[^\]]*"src\/inframe\/music_library\/CREDITS\.md"/m);
  });
});
