// Hotkeys (docs/T9A_DESIGN.md 1.6). One table drives both the key handler and the "?" sheet.

export type Action =
  | 'playPause' | 'back' | 'forward' | 'stepBack' | 'stepFwd' | 'stepBack1s' | 'stepFwd1s' | 'prevClip' | 'nextClip'
  | 'setStart' | 'setEnd' | 'split' | 'merge' | 'delete' | 'add' | 'undo' | 'redo'
  | 'zoomIn' | 'zoomOut' | 'zoomFit' | 'clipsOnly' | 'showBoxes' | 'export' | 'shortcuts' | 'escape';

export interface KeyEventLike {
  key: string;
  shiftKey?: boolean;
  metaKey?: boolean;
  ctrlKey?: boolean;
  altKey?: boolean;
  target?: EventTarget | null;
}

export interface Shortcut {
  action: Action;
  /** how the keys read on the sheet */
  keys: string;
  label: string;
  group: 'Playback' | 'Clips' | 'Timeline' | 'App';
}

export const SHORTCUTS: Shortcut[] = [
  { action: 'playPause', keys: 'Space / K', label: 'Play or pause', group: 'Playback' },
  { action: 'back', keys: 'J', label: 'Play backwards (1x, 2x, 4x on repeat)', group: 'Playback' },
  { action: 'forward', keys: 'L', label: 'Play forwards (1x, 2x, 4x on repeat)', group: 'Playback' },
  { action: 'stepBack', keys: 'Left', label: 'One frame back', group: 'Playback' },
  { action: 'stepFwd', keys: 'Right', label: 'One frame forward', group: 'Playback' },
  { action: 'stepBack1s', keys: 'Shift+Left', label: 'One second back', group: 'Playback' },
  { action: 'stepFwd1s', keys: 'Shift+Right', label: 'One second forward', group: 'Playback' },
  { action: 'clipsOnly', keys: 'P', label: 'Play clips only (jump over gaps)', group: 'Playback' },
  { action: 'showBoxes', keys: 'B', label: 'Show the target box', group: 'Playback' },
  { action: 'prevClip', keys: 'Up', label: 'Previous clip', group: 'Clips' },
  { action: 'nextClip', keys: 'Down', label: 'Next clip', group: 'Clips' },
  { action: 'setStart', keys: 'I', label: 'Set the selected clip’s start to the playhead', group: 'Clips' },
  { action: 'setEnd', keys: 'O', label: 'Set the selected clip’s end to the playhead', group: 'Clips' },
  { action: 'split', keys: 'S', label: 'Split at the playhead', group: 'Clips' },
  { action: 'merge', keys: 'M', label: 'Merge with the next clip (or the selected clips)', group: 'Clips' },
  { action: 'delete', keys: 'Delete / Backspace', label: 'Delete the selected clips', group: 'Clips' },
  { action: 'add', keys: 'N', label: 'Add a 2 s clip at the playhead', group: 'Clips' },
  { action: 'undo', keys: 'Cmd+Z', label: 'Undo (Ctrl+Z on other systems)', group: 'Clips' },
  { action: 'redo', keys: 'Shift+Cmd+Z', label: 'Redo', group: 'Clips' },
  { action: 'zoomIn', keys: '+', label: 'Zoom in', group: 'Timeline' },
  { action: 'zoomOut', keys: '-', label: 'Zoom out', group: 'Timeline' },
  { action: 'zoomFit', keys: '0', label: 'Fit the whole video', group: 'Timeline' },
  { action: 'export', keys: 'Cmd+E', label: 'Export', group: 'App' },
  { action: 'shortcuts', keys: '?', label: 'This sheet', group: 'App' },
  { action: 'escape', keys: 'Esc', label: 'Close a dialog, or clear the selection', group: 'App' },
];

const PLAIN: Record<string, Action> = {
  ' ': 'playPause', k: 'playPause', j: 'back', l: 'forward', arrowup: 'prevClip', arrowdown: 'nextClip',
  i: 'setStart', o: 'setEnd', s: 'split', m: 'merge', delete: 'delete', backspace: 'delete', n: 'add',
  '+': 'zoomIn', '=': 'zoomIn', '-': 'zoomOut', '0': 'zoomFit', p: 'clipsOnly', b: 'showBoxes', '?': 'shortcuts', escape: 'escape',
};

/** Focus is somewhere text goes: hotkeys must not fire. */
export function isTypingTarget(t: EventTarget | null | undefined): boolean {
  const el = t as (Element & { isContentEditable?: boolean }) | null | undefined;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName.toLowerCase();
  if (tag === 'input') return true;
  if (tag === 'textarea' || tag === 'select') return true;
  return el.isContentEditable === true || el.getAttribute?.('contenteditable') === '' || el.getAttribute?.('contenteditable') === 'true';
}

/** A focused <input type=range>: it owns arrows, Home/End, PageUp/PageDown and Delete/Backspace. */
function isSliderTarget(t: EventTarget | null | undefined): boolean {
  const el = t as (Element & { type?: string }) | null | undefined;
  return !!el && typeof el.tagName === 'string' && el.tagName.toLowerCase() === 'input' && el.type === 'range';
}

/** Keys that stay live on a focused slider: they mean nothing to a slider. */
const SLIDER_LIVE = new Set([' ', 'k', 'j', 'l']);

/** The action a key press means, or null when the page (or the browser) should keep it. */
export function actionForKey(e: KeyEventLike, dialogOpen = false): Action | null {
  const key = e.key.toLowerCase();
  if (isSliderTarget(e.target)) {
    if (dialogOpen || e.metaKey || e.ctrlKey || e.altKey || !SLIDER_LIVE.has(key)) return null;
    return PLAIN[key] ?? null;
  }
  if (isTypingTarget(e.target)) return null;
  if (key === 'escape') return 'escape';
  if (dialogOpen) return null;
  const mod = !!(e.metaKey || e.ctrlKey);
  if (mod) {
    if (e.altKey) return null;
    if (key === 'z') return e.shiftKey ? 'redo' : 'undo';
    if (key === 'e' && !e.shiftKey) return 'export';
    return null; // every other Cmd/Ctrl combination belongs to the browser
  }
  if (e.altKey) return null;
  if (key === 'arrowleft') return e.shiftKey ? 'stepBack1s' : 'stepBack';
  if (key === 'arrowright') return e.shiftKey ? 'stepFwd1s' : 'stepFwd';
  if (key === '?') return 'shortcuts';
  if (e.shiftKey && key !== '+') return null;
  return PLAIN[key] ?? null;
}
