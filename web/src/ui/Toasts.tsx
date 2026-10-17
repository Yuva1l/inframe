import type { Store } from '../state';

export function Toasts({ store }: { store: Store }) {
  return (
    <div class="toasts" role="status" aria-live="polite">
      {store.toasts.value.map((t) => (
        <button key={t.id} class={`toast ${t.kind}`} onClick={() => store.dismissToast(t.id)} title="Dismiss">
          {t.text}
        </button>
      ))}
    </div>
  );
}
