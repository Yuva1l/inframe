// Which engines the page can use (docs/T9A_DESIGN.md 2.10, docs/T9B_DESIGN.md section 1): `inframe serve` answers /api/session, a
// static host (Cloudflare Pages) does not. Under `inframe serve` both engines exist; on a static host only the browser engine does.
import { BrowserEngine } from './browser';
import { LocalEngine, OLDER_SERVE } from './local';
import { EngineError, type Engine } from './types';

export interface SelectOptions {
  onDown?: (down: boolean) => void;
  fetch?: typeof fetch;
}

export interface Engines {
  /** null on a static host */
  local: Engine | null;
  browser: Engine;
  /** no `inframe serve` behind this page: "My Mac" cannot be used */
  staticHost: boolean;
}

export async function availableEngines(o: SelectOptions = {}): Promise<Engines> {
  // literal comparisons, so the build drops the mock engine from the shipped page
  if (import.meta.env.MODE === 'mock') return { local: new (await import('./mock')).MockEngine(), browser: new BrowserEngine(), staticHost: false };
  if (import.meta.env.MODE === 'pages') return { local: null, browser: new BrowserEngine(), staticHost: true };
  const local = await LocalEngine.connect({ onDown: o.onDown, fetch: o.fetch });
  if (local) return { local, browser: new BrowserEngine(), staticHost: false };
  // No nonce and nothing stored: a local server still answers 403 with JSON; a static host has no /api.
  const f = o.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  try {
    const r = await f('./api/session');
    if (r.status === 403 && (r.headers.get('content-type') ?? '').includes('json')) throw new EngineError(403, OLDER_SERVE);
  } catch (e) {
    if (e instanceof EngineError) throw e;
  }
  return { local: null, browser: new BrowserEngine(), staticHost: true };
}

/** The engine the page opens with when nothing else is said: My Mac when it exists, else this browser. */
export async function selectEngine(o: SelectOptions = {}): Promise<Engine> {
  const e = await availableEngines(o);
  return e.local ?? e.browser;
}
