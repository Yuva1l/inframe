// `inframe-blob:` keys and the blob URLs made from them (docs/T9B_DESIGN.md section 2). The engine hands the UI a key where the Mac engine
// hands it a real URL; the UI turns a key into a blob URL on demand (Engine.resolveUrl, useEngineUrl). The cache keeps the newest
// blob URLs and revokes the rest, so a long session does not hold every frame it ever drew. Pure: no DOM beyond URL.createObjectURL.
export const BLOB_PREFIX = 'inframe-blob:';

export interface BlobKey {
  kind: string;
  parts: string[];
}

export const makeKey = (kind: string, ...parts: (string | number)[]): string => `${BLOB_PREFIX}${kind}/${parts.map((p) => encodeURIComponent(String(p))).join('/')}`;

/** null when `url` is not one of our keys (a real blob: URL, an http URL, ...). */
export function parseKey(url: string): BlobKey | null {
  if (!url.startsWith(BLOB_PREFIX)) return null;
  const [kind, ...parts] = url.slice(BLOB_PREFIX.length).split('/');
  if (!kind) return null;
  try {
    return { kind, parts: parts.map(decodeURIComponent) };
  } catch {
    return null;
  }
}

export interface UrlOps {
  create: (b: Blob) => string;
  revoke: (u: string) => void;
}

const browserOps = (): UrlOps => ({ create: (b) => URL.createObjectURL(b), revoke: (u) => URL.revokeObjectURL(u) });

interface Pending {
  p: Promise<string>;
  ctl: AbortController;
  /** requesters still waiting; one that passed no signal never leaves */
  holders: number;
}

const abortError = (): Error => new DOMException('The request was abandoned.', 'AbortError');

/** key -> blob URL, least recently used out first. Two callers asking for the same key share one render; the render is told to stop
 *  (its AbortSignal fires) once every caller that passed a signal has abandoned it and none is left. A caller that passes a signal also
 *  holds the URL it got until that signal aborts (the hooks abort on unmount or when the key changes): a held URL is never revoked, so the
 *  cache may stand over `max` while more than `max` pictures are on screen, and is trimmed again as they are let go. */
export class BlobUrlCache {
  private urls = new Map<string, string>();
  private pending = new Map<string, Pending>();
  private gen = 0;
  private holds = new Map<string, number>();

  constructor(readonly max = 160, private ops: UrlOps = browserOps()) {}

  get size(): number {
    return this.urls.size;
  }

  has(key: string): boolean {
    return this.urls.has(key);
  }

  async get(key: string, make: (signal: AbortSignal) => Promise<Blob>, signal?: AbortSignal): Promise<string> {
    const hit = this.urls.get(key);
    if (hit !== undefined) {
      this.urls.delete(key); // re-insert: newest
      this.urls.set(key, hit);
      this.hold(key, signal);
      return hit;
    }
    if (signal?.aborted) throw abortError();
    let e = this.pending.get(key);
    if (!e) {
      const gen = this.gen;
      const ctl = new AbortController();
      const entry: Pending = { ctl, holders: 0, p: undefined as unknown as Promise<string> };
      entry.p = make(ctl.signal).then((blob) => {
        if (gen !== this.gen) throw new Error('cleared'); // clear() ran while this was rendering
        if (ctl.signal.aborted) throw abortError(); // everyone left: a late result is not kept
        const u = this.ops.create(blob);
        this.urls.set(key, u);
        this.trim(key);
        return u;
      }).finally(() => {
        if (this.pending.get(key) === entry) this.pending.delete(key);
      });
      entry.p.catch(() => {}); // an abandoned render may fail with nobody listening
      this.pending.set(key, entry);
      e = entry;
    }
    const entry = e;
    entry.holders++;
    if (!signal) return entry.p;
    return new Promise<string>((resolve, reject) => {
      let over = false;
      const leave = () => {
        if (over) return;
        over = true;
        signal.removeEventListener('abort', leave);
        if (--entry.holders === 0) {
          entry.ctl.abort();
          if (this.pending.get(key) === entry) this.pending.delete(key); // the next asker starts afresh
        }
        reject(abortError());
      };
      signal.addEventListener('abort', leave, { once: true });
      entry.p.then(
        (u) => {
          if (over) return;
          over = true;
          signal.removeEventListener('abort', leave);
          this.hold(key, signal);
          resolve(u);
        },
        (err) => {
          if (over) return;
          over = true;
          signal.removeEventListener('abort', leave);
          reject(err);
        },
      );
    });
  }

  /** The URL under `key` stays alive until `signal` aborts. */
  private hold(key: string, signal?: AbortSignal): void {
    if (!signal || signal.aborted) return;
    const gen = this.gen;
    this.holds.set(key, (this.holds.get(key) ?? 0) + 1);
    signal.addEventListener('abort', () => {
      if (gen !== this.gen) return; // clear() already let everything go
      const n = (this.holds.get(key) ?? 1) - 1;
      if (n > 0) this.holds.set(key, n);
      else this.holds.delete(key);
      this.trim();
    }, { once: true });
  }

  /** Revoke the oldest URLs nobody holds until the cache is within `max`; `keep` (a URL just made, not yet handed out) stays. */
  private trim(keep?: string): void {
    for (const [k, u] of this.urls) {
      if (this.urls.size <= this.max) break;
      if (k === keep || this.holds.has(k)) continue;
      this.urls.delete(k);
      this.ops.revoke(u);
    }
  }

  /** Revoke every URL (a closed video). */
  clear(): void {
    this.gen++;
    this.holds.clear();
    for (const u of this.urls.values()) this.ops.revoke(u);
    this.urls.clear();
    for (const e of this.pending.values()) e.ctl.abort();
    this.pending.clear();
  }
}
