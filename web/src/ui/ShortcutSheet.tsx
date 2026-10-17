import type { ComponentChildren } from 'preact';
import { useEffect, useRef } from 'preact/hooks';
import { SHORTCUTS } from '../model/keys';
import type { Store } from '../state';
import { Icon } from './Icons';

/** A modal dialog: a backdrop, focus moved in, a close button. Esc is handled by the hotkey handler (it closes the top dialog). */
export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ComponentChildren; wide?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    box.current?.focus();
    return () => prev?.focus?.();
  }, []);
  return (
    <div class="backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div class={`modal${wide ? ' wide' : ''}`} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={box} onKeyDown={(e) => {
        if (e.key !== 'Escape') return;
        e.stopPropagation(); // one Esc closes one dialog: the page-wide handler must not also clear the selection
        onClose();
      }}>
        <div class="modal-head">
          <h2>{title}</h2>
          <button class="icon ghost" onClick={onClose} aria-label="Close"><Icon name="close" /></button>
        </div>
        {children}
      </div>
    </div>
  );
}

const GROUPS = ['Playback', 'Clips', 'Timeline', 'App'] as const;

export function ShortcutSheet({ store }: { store: Store }) {
  return (
    <Modal title="Keyboard shortcuts" onClose={() => (store.shortcutsOpen.value = false)}>
      <div class="shortcuts">
        {GROUPS.map((g) => (
          <section key={g}>
            <h3>{g}</h3>
            <dl>
              {SHORTCUTS.filter((s) => s.group === g).map((s) => (
                <div class="srow" key={s.action}>
                  <dt><kbd>{s.keys}</kbd></dt>
                  <dd>{s.label}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </Modal>
  );
}
