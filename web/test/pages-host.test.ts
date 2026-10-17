// @vitest-environment node
// T9c: the Pages build's host settings (pagesHost, pagesHeaders, connectCovers in vite.config.ts). Node environment: vite does not load in jsdom.
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { connectCovers, pagesHeaders, pagesHost } from '../vite.config';

describe('Pages build settings (vite.config.ts)', () => {
  const tmp = () => {
    const d = mkdtempSync(path.join(tmpdir(), 'pages-'));
    mkdirSync(path.join(d, 'pages'));
    return d;
  };
  const withHost = (j: object) => { const d = tmp(); writeFileSync(path.join(d, 'pages', 'model-host.json'), JSON.stringify(j)); return d; };
  const REAL = 'https://huggingface.co/me/inframe-models/resolve/' + 'c'.repeat(40) + '/';
  const CONNECT = ['https://huggingface.co/me/inframe-models/resolve/' + 'c'.repeat(40) + '/', 'https://cas-bridge.xethub.hf.co'];

  it('takes the file as it is and lets the environment override it', () => {
    expect(pagesHost(withHost({ base: REAL, connect: CONNECT }), {})).toEqual({ base: REAL, connect: CONNECT });
    const d = withHost({ base: REAL, connect: CONNECT });
    expect(pagesHost(d, { VITE_MODEL_BASE: REAL.replace('me', 'you').replace(/\/$/, ''), INFRAME_PAGES_CONNECT: 'https://huggingface.co/you/inframe-models/resolve/' + 'c'.repeat(40) + '/, https://x.hf.co' }))
      .toEqual({ base: REAL.replace('me', 'you'), connect: ['https://huggingface.co/you/inframe-models/resolve/' + 'c'.repeat(40) + '/', 'https://x.hf.co'] });
  });

  it('refuses the placeholder, a base without a commit, http, a loose connect entry and an uncovered base', () => {
    const ph = withHost({ base: 'https://huggingface.co/OWNER/inframe-models/resolve/COMMIT/', connect: CONNECT });
    expect(() => pagesHost(ph, {})).toThrow(/placeholder/);
    expect(() => pagesHost(ph, { INFRAME_PAGES_ALLOW_LOCAL: '1' })).toThrow(/placeholder/);
    const d = withHost({ base: REAL, connect: CONNECT });
    expect(() => pagesHost(d, { VITE_MODEL_BASE: 'https://huggingface.co/me/m/resolve/main/' })).toThrow(/40-hex/);
    expect(() => pagesHost(d, { VITE_MODEL_BASE: REAL.replace('https', 'http') })).toThrow(/https/);
    expect(() => pagesHost(d, { INFRAME_PAGES_CONNECT: "https://huggingface.co; script-src *" })).toThrow(/plain origin/);
    expect(() => pagesHost(d, { INFRAME_PAGES_CONNECT: 'https://*' })).toThrow(/plain origin/);
    expect(() => pagesHost(d, { INFRAME_PAGES_CONNECT: 'https://other.example' })).toThrow(/not covered by connect/);
  });

  it('refuses a connect entry that opens a whole public suffix or all of Hugging Face', () => {
    const d = withHost({ base: REAL, connect: CONNECT });
    for (const bad of ['https://*.co', 'https://*.com', 'https://co', 'https://*', 'https://com/', 'https://*.huggingface.co', 'https://*.hf.co', 'https://*.xethub.hf.co', 'https://cas-server.xethub.hf.co', 'https://*.example.com', 'https://*.hf.co/', 'https://huggingface.co', 'https://huggingface.co/', 'https://hf.co', 'https://huggingface.co/a/../b/']) {
      expect(() => pagesHost(d, { INFRAME_PAGES_CONNECT: [bad, ...CONNECT].join(',') }), bad).toThrow();
    }
    // a path-scoped entry on the upload host is refused too: CSP matches its path on the first request
    for (const bad of ['https://cas-server.xethub.hf.co/v1/', 'https://cas-server.xethub.hf.co/', 'https://cas-server.xethub.hf.co/a/b/']) {
      expect(() => pagesHost(d, { INFRAME_PAGES_CONNECT: [...CONNECT, 'https://*.cdn.hf.co', bad].join(',') }), bad).toThrow(/upload host/);
    }
    // the one wildcard accepted is https://*.cdn.hf.co (Hugging Face's regional CDN hosts; never all of hf.co, which includes the Xet upload host), and only for a pinned huggingface.co base
    expect(pagesHost(d, { INFRAME_PAGES_CONNECT: [CONNECT[0], 'https://*.cdn.hf.co'].join(',') }).connect).toEqual([CONNECT[0], 'https://*.cdn.hf.co']);
    const elsewhere = { VITE_MODEL_BASE: 'https://example.org/me/m/resolve/' + 'c'.repeat(40) + '/', INFRAME_PAGES_CONNECT: 'https://example.org/me/m/resolve/' + 'c'.repeat(40) + '/,https://*.cdn.hf.co' };
    expect(() => pagesHost(d, elsewhere)).toThrow(/only when the model base is a pinned/);
    // the repo path alone, with the redirect host, is what the committed file names
    expect(pagesHost(d, { INFRAME_PAGES_CONNECT: CONNECT.join(',') }).connect).toEqual(CONNECT);
    // a path entry covers only the base under it
    expect(() => pagesHost(d, { INFRAME_PAGES_CONNECT: 'https://huggingface.co/other/inframe-models/resolve/' + 'c'.repeat(40) + '/' })).toThrow(/not covered by connect/);
  });

  it('allows an http://127.0.0.1 base only with INFRAME_PAGES_ALLOW_LOCAL=1', () => {
    const d = withHost({ base: REAL, connect: CONNECT });
    const env = { VITE_MODEL_BASE: 'http://127.0.0.1:8123/resolve/x', INFRAME_PAGES_CONNECT: 'http://127.0.0.1:8123' };
    expect(() => pagesHost(d, env)).toThrow();
    expect(pagesHost(d, { ...env, INFRAME_PAGES_ALLOW_LOCAL: '1' })).toEqual({ base: 'http://127.0.0.1:8123/resolve/x/', connect: ['http://127.0.0.1:8123'] });
    expect(() => pagesHost(d, { ...env, VITE_MODEL_BASE: 'http://localhost:8123/x/', INFRAME_PAGES_ALLOW_LOCAL: '1' })).toThrow();
  });

  it('allows a path entry for a local origin, the shape of the shipped connect list, and still refuses a local path with dots or no slash', () => {
    const d = withHost({ base: REAL, connect: CONNECT });
    const base = 'http://127.0.0.1:8123/resolve/' + 'c'.repeat(40) + '/';
    const env = { VITE_MODEL_BASE: base, INFRAME_PAGES_ALLOW_LOCAL: '1' };
    expect(pagesHost(d, { ...env, INFRAME_PAGES_CONNECT: `${base},http://127.0.0.1:8124` }).connect).toEqual([base, 'http://127.0.0.1:8124']);
    expect(() => pagesHost(d, { ...env, INFRAME_PAGES_CONNECT: 'http://127.0.0.1:8123/resolve/other/' })).toThrow(/not covered by connect/);
    for (const bad of [base.slice(0, -1), 'http://127.0.0.1:8123/a/../b/', 'http://127.0.0.1:8123//', 'http://localhost:8123/x/']) {
      expect(() => pagesHost(d, { ...env, INFRAME_PAGES_CONNECT: bad }), bad).toThrow();
    }
    expect(() => pagesHost(d, { ...env, INFRAME_PAGES_CONNECT: `${base},http://127.0.0.1:8124`, INFRAME_PAGES_ALLOW_LOCAL: '' })).toThrow();
  });

  it('covers a wildcard origin only with a label in front', () => {
    expect(connectCovers('https://*.cdn.hf.co', 'https://us.aws.cdn.hf.co')).toBe(true);
    expect(connectCovers('https://*.cdn.hf.co', 'https://cdn.hf.co')).toBe(false);
    expect(connectCovers('https://*.cdn.hf.co', 'https://evilcdn.hf.co')).toBe(false);
    expect(connectCovers('https://*.cdn.hf.co', 'https://cas-server.xethub.hf.co/v1/xorbs/default/0')).toBe(false);
    expect(connectCovers('https://huggingface.co', 'https://huggingface.co')).toBe(true);
    expect(connectCovers('https://huggingface.co/me/m/resolve/abc/', 'https://huggingface.co/me/m/resolve/abc/model.onnx')).toBe(true);
    expect(connectCovers('https://huggingface.co/me/m/resolve/abc/', 'https://huggingface.co/api/upload')).toBe(false);
    expect(connectCovers('https://huggingface.co/me/m/resolve/abc/', 'https://huggingface.co')).toBe(false);
  });

  it('widens only connect-src and adds the asset cache and a Permissions-Policy', () => {
    const pub = "# c\n/*\n  Content-Security-Policy: default-src 'self'; connect-src 'self'; worker-src 'self'\n  Referrer-Policy: no-referrer\n";
    const out = pagesHeaders(pub, CONNECT);
    expect(out).toContain(`Content-Security-Policy: default-src 'self'; connect-src 'self' ${CONNECT.join(' ')}; worker-src 'self'`);
    expect(out).toContain('/assets/*\n  Cache-Control: public, max-age=31536000, immutable');
    expect(out).toContain('Permissions-Policy: camera=(), microphone=(), geolocation=()');
    expect(() => pagesHeaders(pub.replace("connect-src 'self';", ''), CONNECT)).toThrow(/exactly once/);
  });
});
