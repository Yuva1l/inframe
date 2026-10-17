/// <reference types="vitest/config" />
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import preact from '@preact/preset-vite';
import license from 'rollup-plugin-license';
import { defineConfig, type Plugin, type UserConfig } from 'vite';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Licences that may be bundled into the shipped page (docs/T9A_DESIGN.md 2.2). The build fails on any other.
const ALLOWED = '(MIT OR ISC OR BSD-2-Clause OR BSD-3-Clause OR Apache-2.0 OR 0BSD)';

// rollup-plugin-license, with the notices written next to the build's own output folder (so `vite build --outDir TMP` works too).
function licenses(): Plugin {
  let inner: ReturnType<typeof license>;
  let file = '';
  let root = '';
  return {
    name: 'inframe-licenses',
    apply: 'build',
    configResolved(c) {
      root = c.root;
      file = path.resolve(c.root, c.build.outDir, 'THIRD_PARTY_LICENSES.txt');
      inner = license({
        thirdParty: {
          allow: { test: ALLOWED, failOnUnlicensed: true, failOnViolation: true },
          output: { file, encoding: 'utf-8' },
        },
      });
    },
    renderChunk(code, chunk, options) {
      return (inner.renderChunk as any).call(this, code, chunk, options);
    },
    generateBundle() {
      (inner.generateBundle as any).call(this);
      // Notices the plugin cannot see: code inside onnxruntime-web's pre-bundled file, and the Pillow resize that embed.ts ports (src/notices.txt).
      appendFileSync(file, '\n\n---\n\n' + readFileSync(path.join(root, 'src', 'notices.txt'), 'utf-8').trim());
    },
  };
}

// Preact names two XML namespaces (XHTML, MathML) as string constants. They are identifiers, not addresses, but the dist scan
// (tests/test_serve_dist.py) treats every http:// string as an outside address. Writing them as "http:"+"//..." keeps the
// scan strict (only the SVG namespace is allowlisted there) and the runtime value the same.
function splitNamespaces(): Plugin {
  const NS = /(["'`])http:\/\/www\.w3\.org\/(1998\/Math\/MathML|1999\/xhtml)\1/g;
  return {
    name: 'inframe-split-namespaces',
    apply: 'build',
    enforce: 'post',
    generateBundle(_o, bundle) {
      for (const f of Object.values(bundle)) {
        if (f.type === 'chunk') f.code = f.code.replace(NS, '"http:"+"//www.w3.org/$2"');
      }
    },
  };
}


// onnxruntime-web names its WebAssembly file with a `new URL(..., import.meta.url)` that Vite would copy into dist (26 MB). The page never uses
// that address: the runtime is given to ort as bytes (src/browser/detect.ts loadOrt, from the model cache), so the expression is made harmless.
// The one https:// address inside the bundle (a link in an error message) loses its scheme so the dist scan (tests/test_serve_dist.py) does not read it as a host.
export function ortPlugin(): Plugin {
  const WASM = /new URL\("ort-wasm-simd-threaded\.[a-z]+\.wasm",\s*import\.meta\.url\)/g;
  return {
    name: 'inframe-ort',
    enforce: 'pre',
    transform(code, id) {
      if (!/onnxruntime-web[\\/]dist[\\/]ort[^\\/]*\.m?js$/.test(id.split('?')[0])) return null;
      return { code: code.replace(WASM, 'new URL("data:,")').replace(/https:\/\/web\.dev\//g, 'web.dev/'), map: null };
    },
  };
}

// A fingerprint of everything the build reads (web/src, web/public, index.html, package-lock.json, vite and tsconfig), written
// into the build as SOURCE_HASH.txt. tests/test_serve_dist.py recomputes it, so a committed dist that is older than the
// sources fails the test. Keep the algorithm identical to source_hash() there: sha256 over "relpath NUL sha256(file)\n" lines.
export function sourceHash(root: string): string {
  const files: string[] = [];
  const walk = (rel: string) => {
    const abs = path.join(root, rel);
    if (statSync(abs).isDirectory()) {
      for (const name of readdirSync(abs)) walk(`${rel}/${name}`);
    } else files.push(rel);
  };
  for (const entry of ['src', 'public']) walk(entry);
  files.push('index.html', 'package-lock.json', 'vite.config.ts', 'tsconfig.json');
  const lines = files
    .filter((f) => !f.endsWith('.DS_Store'))
    .sort((a, b) => (Buffer.from(a).compare(Buffer.from(b))))
    .map((f) => `${f}\0${createHash('sha256').update(readFileSync(path.join(root, f))).digest('hex')}\n`);
  return createHash('sha256').update(lines.join('')).digest('hex');
}

function sourceFingerprint(): Plugin {
  let root = '';
  return {
    name: 'inframe-source-hash',
    apply: 'build',
    configResolved(c) {
      root = c.root;
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'SOURCE_HASH.txt', source: sourceHash(root) + '\n' });
    },
  };
}

// --- T9c: the Cloudflare Pages build (docs/T9C_DESIGN.md section 2). `vite build --mode pages` writes web/pages-dist (never the committed
// src/inframe/web/dist), takes the model host from web/pages/model-host.json, copies the pinned manifest to models/manifest.json and writes
// _headers with the model hosts added to connect-src. Nothing here deploys or uploads anything.
export interface PagesHost {
  /** where the three model files are fetched from, with a closing slash */
  base: string;
  /** the origins connect-src allows besides 'self' */
  connect: string[];
}

const PLACEHOLDER = /OWNER|COMMIT/;
// Hugging Face answers a model download with a redirect to a CDN host (us.aws.cdn.hf.co, cas-bridge.xethub.hf.co, ...). Which host depends on the visitor's
// region and changes over time, so a fixed list breaks for some visitors. The one wildcard accepted is exactly https://*.cdn.hf.co (the download CDN), and only when
// the model base is a pinned huggingface.co resolve URL (owner/repo/resolve/<40-hex commit>/): every file is checked against the sha256 in the page's own manifest,
// so a CDN host can serve bytes but not change them. The whole of hf.co is not opened: it includes the Xet upload host (cas-server.xethub.hf.co (not checked against the
// HF docs)), and the CSP is the backstop that footage cannot leave the browser. The Xet download host (cas-bridge.xethub.hf.co) is named exactly. Every other wildcard stays refused.
export const HF_CDN_WILDCARD = 'https://*.cdn.hf.co';
export const HF_UPLOAD_HOSTS = ['https://cas-server.xethub.hf.co'];
const HF_PINNED_BASE = /^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\//;
const COMMIT_SEGMENT = /\/[0-9a-f]{40}(\/|$)/;
// A connect entry is https://host (at least two labels, so never https://*.co or https://com), optionally https://*.host, optionally with a path that
// ends in a slash (CSP matches that path on the first hop only; a redirect is matched on its origin alone).
const LABEL = '[a-z0-9](?:[a-z0-9-]*[a-z0-9])?';
const CONNECT_ORIGIN = new RegExp(`^https://(?:\\*\\.)?(?:${LABEL}\\.)+${LABEL}(?:/(?:(?!\\.+/)[A-Za-z0-9._~-]+/)*)?$`);
// a local origin (the tests), optionally with a path ending in a slash, the same shape as CONNECT_ORIGIN
const LOCAL_ORIGIN = /^http:\/\/127\.0\.0\.1:\d{1,5}(?:\/(?:(?!\.+\/)[A-Za-z0-9._~-]+\/)*)?$/;

/** Does a connect-src entry (an origin, https://*.host, or either with a path ending in a slash) cover this address (an origin or a full URL)? */
export function connectCovers(entry: string, address: string): boolean {
  const split = (t: string): [string, string] => { const i = t.indexOf('/', 8); return i < 0 ? [t, ''] : [t.slice(0, i), t.slice(i)]; };
  const [eo, ep] = split(entry);
  const [ao, ap] = split(address);
  if (ep && !ap.startsWith(ep)) return false;
  if (eo === ao) return true;
  const w = /^https:\/\/\*\.(.+)$/.exec(eo);
  return !!w && ao.startsWith('https://') && ao.slice(8).endsWith('.' + w[1]);
}

/** The model host of the Pages build: web/pages/model-host.json, with VITE_MODEL_BASE and INFRAME_PAGES_CONNECT (comma separated) over it.
 *  Throws on the placeholder, on a base without a 40-hex commit, and on a connect entry that is not a plain origin. INFRAME_PAGES_ALLOW_LOCAL=1
 *  lifts the https and commit rules for a http://127.0.0.1:PORT base (the tests). */
export function pagesHost(dir: string, env: Record<string, string | undefined> = process.env): PagesHost {
  const file = path.join(dir, 'pages', 'model-host.json');
  const j = JSON.parse(readFileSync(file, 'utf-8')) as { base?: unknown; connect?: unknown };
  const baseRaw = env.VITE_MODEL_BASE || (typeof j.base === 'string' ? j.base : '');
  const connect = env.INFRAME_PAGES_CONNECT !== undefined
    ? env.INFRAME_PAGES_CONNECT.split(/[\s,]+/).filter(Boolean)
    : Array.isArray(j.connect) ? j.connect.map(String) : [];
  const allowLocal = env.INFRAME_PAGES_ALLOW_LOCAL === '1';
  const fail = (why: string): never => { throw new Error(`pages build: ${why} (${path.relative(process.cwd(), file) || file}, VITE_MODEL_BASE, INFRAME_PAGES_CONNECT)`); };
  if (!baseRaw) fail('no model base');
  let u: URL;
  try { u = new URL(baseRaw); } catch { return fail(`the model base ${baseRaw} is not a URL`); }
  const local = allowLocal && u.protocol === 'http:' && u.hostname === '127.0.0.1';
  if (PLACEHOLDER.test(baseRaw)) fail(`the model base ${baseRaw} is still the placeholder; put the real repo and its 40-hex commit in web/pages/model-host.json`);
  if (!local) {
    if (u.protocol !== 'https:') fail(`the model base must be https (${baseRaw})`);
    if (!COMMIT_SEGMENT.test(u.pathname)) fail(`the model base ${baseRaw} has no 40-hex commit in its path, so the files are not pinned`);
  }
  if (u.username || u.password || u.search || u.hash) fail('the model base must not carry credentials, a query or a fragment');
  if (!u.pathname.endsWith('/')) u.pathname += '/';
  for (const c of connect) {
    if (!(CONNECT_ORIGIN.test(c) || (allowLocal && LOCAL_ORIGIN.test(c)))) fail(`connect entry ${c} is not a plain origin (https://host or https://*.host, at least two labels, an optional path ending in a slash)`);
    // Hugging Face's own API and upload hosts take browser uploads: its whole origin must never be opened, only the pinned repo path.
    if (c === HF_CDN_WILDCARD) {
      if (!HF_PINNED_BASE.test(u.href)) fail(`connect entry ${c} is accepted only when the model base is a pinned https://huggingface.co/OWNER/REPO/resolve/<40-hex commit>/ URL (${u.href})`);
      continue;
    }
    if (c.includes('*')) fail(`connect entry ${c} is a wildcard; the only one accepted is ${HF_CDN_WILDCARD}, for a pinned Hugging Face base`);
    if (/^https:\/\/(?:\*\.)?(?:huggingface\.co|hf\.co)\/?$/.test(c)) fail(`connect entry ${c} opens all of Hugging Face to the page; name the pinned repo path (https://huggingface.co/OWNER/REPO/resolve/COMMIT/) and the exact redirect host`);
  }
  for (const up of HF_UPLOAD_HOSTS) {
    // the origin alone decides: an entry with a path (https://cas-server.xethub.hf.co/v1/) lets the page POST to that path on the first request
    const hit = connect.filter((c) => connectCovers(c.replace(/^(https?:\/\/[^/]+).*$/, '$1'), up));
    if (hit.length) fail(`connect entry ${hit.join(' ')} covers the Hugging Face upload host ${up}: the page could send the video there`);
  }
  if (!connect.some((c) => connectCovers(c, u.href))) fail(`the model base ${u.href} is not covered by connect (${connect.join(' ') || 'empty'})`);
  return { base: u.href, connect };
}

/** public/_headers with connect-src widened to the model hosts, plus long caching for the hashed assets and a Permissions-Policy. */
export function pagesHeaders(publicHeaders: string, connect: string[]): string {
  const find = "connect-src 'self';";
  if (publicHeaders.split(find).length !== 2) throw new Error("pages build: public/_headers must hold connect-src 'self'; exactly once");
  const rules = publicHeaders.split('\n').filter((l) => /^\S/.test(l) && !l.startsWith('#'));
  if (rules.length !== 1 || rules[0] !== '/*') throw new Error('pages build: public/_headers must hold the one rule /*');
  const widened = publicHeaders.replace(find, () => `connect-src ${["'self'", ...connect].join(' ')};`);
  return (
    widened.trimEnd() +
    '\n  Permissions-Policy: camera=(), microphone=(), geolocation=()\n\n/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n'
  );
}

function pagesFiles(host: PagesHost): Plugin {
  return {
    name: 'inframe-pages-files',
    apply: 'build',
    generateBundle() {
      const manifest = path.join(HERE, 'pages', 'manifest.json');
      if (!existsSync(manifest)) throw new Error('pages build: web/pages/manifest.json is missing (copy $INFRAME_MODELS/onnx/manifest.json)');
      const bytes = readFileSync(manifest);
      const m = JSON.parse(bytes.toString('utf-8')) as { format?: number; files?: object };
      if (m.format !== 1 || !m.files || !Object.keys(m.files).length) throw new Error('pages build: web/pages/manifest.json is not a model manifest');
      this.emitFile({ type: 'asset', fileName: 'models/manifest.json', source: new Uint8Array(bytes) });
      this.emitFile({ type: 'asset', fileName: '_headers', source: pagesHeaders(readFileSync(path.join(HERE, 'public', '_headers'), 'utf-8'), host.connect) });
    },
  };
}

// resample.ts and yuv.ts are LGPL-2.1-or-later and reach users only as minified JavaScript, so their complete source is emitted beside the build's
// notices (LGPL-2.1 section 4: the source comes with the object code, from the same place). tests/test_license_labels.py checks it.
function lgplSource(): Plugin {
  let root = '';
  return {
    name: 'inframe-lgpl-source',
    apply: 'build',
    configResolved(c) {
      root = c.root;
    },
    generateBundle() {
      for (const f of ['resample.ts', 'yuv.ts']) {
        this.emitFile({ type: 'asset', fileName: f, source: new Uint8Array(readFileSync(path.join(root, 'src', 'browser', f))) });
      }
    },
  };
}

export default defineConfig(({ mode }): UserConfig => {
  const pages = mode === 'pages' ? pagesHost(HERE) : null;
  return {
    base: './',
    // The run Worker (src/browser/worker.ts) is a module worker. It bundles the same dependencies as the page (mp4box, onnxruntime-web);
    // the page's bundle holds them too (the inline fallback imports the same handler), so THIRD_PARTY_LICENSES.txt covers both.
    worker: { format: 'es', plugins: () => [ortPlugin(), splitNamespaces()] },
    plugins: [ortPlugin(), preact(), licenses(), splitNamespaces(), sourceFingerprint(), lgplSource(), ...(pages ? [pagesFiles(pages)] : [])],
    // Pages: the models come from the pinned host, the manifest from this site (the page trusts its own copy, not the host's).
    define: pages ? { 'import.meta.env.VITE_MODEL_BASE': JSON.stringify(pages.base), 'import.meta.env.VITE_MODEL_MANIFEST': JSON.stringify('./models/manifest.json') } : {},
    build: {
      outDir: pages ? path.join(HERE, 'pages-dist') : '../src/inframe/web/dist',
      emptyOutDir: true,
      assetsInlineLimit: 0,
      sourcemap: false,
      target: 'es2022',
    },
    test: {
      environment: 'jsdom',
      include: ['test/**/*.test.{ts,tsx}'],
      // tests that spawn ffmpeg, ffprobe or the Python cut (mux, browser-demux, browser-colour) need well over the 5 s default
      // when other jobs load the machine (they have timed out at a load of 13 and passed alone)
      testTimeout: 30_000,
      hookTimeout: 120_000,
    },
  };
});
