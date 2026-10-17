// An <img> or CSS url() cannot load an `inframe-blob:` key, so a URL from the engine goes through Engine.resolveUrl first
// (docs/T9B_DESIGN.md section 2). An engine without resolveUrl (My Mac) hands out real URLs: those, and any URL that is not one of our
// keys, come back unchanged on the first render, with no state and no flicker.
import { useEffect, useState } from 'preact/hooks';
import type { Engine } from '../engine/types';

/** The prefix of the keys BrowserEngine hands out (browser/urls.ts BLOB_PREFIX). */
const KEY = 'inframe-blob:';

const needsResolve = (engine: Engine, url: string | null | undefined): url is string => !!engine.resolveUrl && !!url && url.startsWith(KEY);

/** `url` ready for an <img src>; undefined while a key is being rendered (or when it failed). */
export function useEngineUrl(engine: Engine, url: string | null | undefined): string | undefined {
  const [got, setGot] = useState<{ engine: Engine; key: string; out: string } | null>(null);
  const resolve = needsResolve(engine, url);
  useEffect(() => {
    if (!resolve) return;
    let live = true;
    const ctl = new AbortController();
    engine.resolveUrl!(url!, ctl.signal).then((out) => live && setGot({ engine, key: url!, out }), () => live && setGot({ engine, key: url!, out: '' }));
    return () => {
      live = false;
      ctl.abort(); // a render still waiting for the decoder is dropped
    };
  }, [engine, url, resolve]);
  if (!resolve) return url ?? undefined;
  return got && got.engine === engine && got.key === url && got.out ? got.out : undefined;
}

/** The same for a list: returns a function from a raw URL to its ready URL (undefined while pending). Hooks cannot run in a loop, so
 *  a component with many images (the filmstrip's sheets) asks for them all here. */
export function useEngineUrls(engine: Engine, urls: readonly string[]): (url: string) => string | undefined {
  const [got, setGot] = useState<{ engine: Engine; map: Record<string, string> }>({ engine, map: {} });
  const want = urls.filter((u) => needsResolve(engine, u));
  const wantKey = want.join('\n');
  useEffect(() => {
    let live = true;
    const ctl = new AbortController();
    for (const u of want) {
      // asked again even when it is already shown: the signal is the engine's lease on the URL, and the last effect's ended with it
      engine.resolveUrl!(u, ctl.signal).then((out) => live && setGot((g) => (g.engine === engine && g.map[u] === out ? g : { engine, map: { ...(g.engine === engine ? g.map : {}), [u]: out } })), () => {});
    }
    return () => {
      live = false;
      ctl.abort();
    };
  }, [engine, wantKey]);
  return (u: string) => (needsResolve(engine, u) ? (got.engine === engine ? got.map[u] : undefined) : u);
}
