// The checklist of what a run needs (docs/T9A_DESIGN.md 1.5). A missing item is red and names the command that fixes it. In the browser
// engine the checklist is WebGPU, video decoding and the downloaded models, with the buttons that fetch and remove the models (T9B_DESIGN.md 3).
import { useEffect, useState } from 'preact/hooks';
import type { ModelControl } from '../engine/types';
import { span } from '../model/estimate';
import { preflightRows } from '../model/preflight';
import type { Store } from '../state';

const NUMBER_WORDS: Record<number, string> = { 1: 'one', 2: 'two', 3: 'three' };
const mb = (b: number) => `${Math.round(b / 1e6)} MB`;

function Models({ store, models }: { store: Store; models: ModelControl }) {
  const [busy, setBusy] = useState<{ done: number; total: number; name: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ac, setAc] = useState<AbortController | null>(null);
  const [st, setSt] = useState<Awaited<ReturnType<ModelControl['status']>> | null>(null);
  const refresh = async () => {
    try {
      setSt(await models.status());
      store.session.value = await store.engine.session();
    } catch { /* the checklist keeps what it had */ }
  };
  useEffect(() => { void models.status().then(setSt, () => undefined); }, [models]);
  const go = async () => {
    const c = new AbortController();
    setAc(c);
    setErr(null);
    try {
      await models.download({ signal: c.signal, onProgress: (p) => setBusy({ done: p.done, total: p.total, name: p.name }) });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setAc(null);
      await refresh();
    }
  };
  const remove = async () => {
    setErr(null);
    try {
      await models.remove();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
    await refresh();
  };
  if (!st) return null;
  const left = st.totalBytes - st.cachedBytes;
  const todo = st.filesLeft ?? 0;
  return (
    <div class="models" data-testid="models">
      {busy ? (
        <>
          <progress max={busy.total} value={busy.done} aria-label="Model download progress" />
          <p class="muted small">Downloading {busy.name}: {mb(busy.done)} of {mb(busy.total)}</p>
          <button onClick={() => ac?.abort()}>Cancel</button>
        </>
      ) : (
        <>
          {left > 0 && st.available && <button class="primary" onClick={() => void go()}>Download models (about {mb(left)}, once)</button>}
          {left > 0 && st.available && st.host && (
            <p class="muted small" data-testid="models-privacy">
              Your video never leaves this browser. Only the {NUMBER_WORDS[todo] ?? todo} model {todo === 1 ? 'file' : 'files'} (about {mb(left)}) are downloaded, once, from {st.host}{st.host === 'this site' ? '' : ', which hands the files on to its own download servers'}.
            </p>
          )}
          {st.cachedBytes > 0 && <button onClick={() => void remove()}>Remove downloaded models</button>}
          {st.cachedBytes > 0 && <p class="muted small">{mb(st.cachedBytes)} kept in this browser{st.persisted === false ? ' (the browser may clear it when space runs low)' : ''}.</p>}
        </>
      )}
      {err && <p class="bad small" role="alert">{err}</p>}
    </div>
  );
}

export function Preflight({ store, all }: { store: Store; /** the first-use screen: every check, whatever the target */ all?: boolean }) {
  const p = store.session.value?.preflight;
  if (!p) return null;
  const rows = all ? preflightRows(p, { preset: null, identity: null, tracking: null, follow: null }) : store.rows.value;
  const models = p.browser ? store.engine.models : undefined;
  return (
    <>
      <ul class="preflight" aria-label="What this machine has">
        {rows.map((r) => (
          <li key={r.key} class={r.ok ? 'ok' : 'bad'}>
            {r.text}
            {!r.ok && r.fix && <span class="muted"> - {r.fix}</span>}
          </li>
        ))}
      </ul>
      {models && <Models store={store} models={models} />}
    </>
  );
}

/** "Use My Mac" when the browser engine cannot or should not take this video (docs/T9B_DESIGN.md section 7): no video decoding or HEVC here, no WebGPU, an
 *  estimate past 10 minutes, or a video longer than the longest one measured. Draws nothing on My Mac or when the browser is fine. */
export function UseMyMac({ store }: { store: Store }) {
  const s = store.localSuggestion.value;
  if (!s || !s.suggest) return null;
  return (
    <div class="warn" role="status" data-testid="use-my-mac">
      <p>{s.text}</p>
    </div>
  );
}

/** "about 19 min (estimate), about 2 min on My Mac" for the browser engine; the Mac's own line elsewhere. */
export function macComparison(store: Store): string {
  const be = store.browserEst.value;
  return be ? `about ${span(be.macSeconds)} on My Mac` : '';
}
