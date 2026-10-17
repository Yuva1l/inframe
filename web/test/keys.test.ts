import { describe, expect, it } from 'vitest';
import { SHORTCUTS, actionForKey, isTypingTarget, type KeyEventLike } from '../src/model/keys';

const div = { tagName: 'DIV' } as unknown as EventTarget;
const k = (key: string, extra: Partial<KeyEventLike> = {}): KeyEventLike => ({ key, target: div, ...extra });

describe('actionForKey', () => {
  it('maps the table of section 1.6', () => {
    const cases: [KeyEventLike, string][] = [
      [k(' '), 'playPause'], [k('k'), 'playPause'], [k('j'), 'back'], [k('l'), 'forward'],
      [k('ArrowLeft'), 'stepBack'], [k('ArrowRight'), 'stepFwd'], [k('ArrowLeft', { shiftKey: true }), 'stepBack1s'],
      [k('ArrowRight', { shiftKey: true }), 'stepFwd1s'], [k('ArrowUp'), 'prevClip'], [k('ArrowDown'), 'nextClip'],
      [k('i'), 'setStart'], [k('o'), 'setEnd'], [k('s'), 'split'], [k('m'), 'merge'], [k('Delete'), 'delete'], [k('Backspace'), 'delete'],
      [k('n'), 'add'], [k('p'), 'clipsOnly'], [k('b'), 'showBoxes'], [k('+', { shiftKey: true }), 'zoomIn'], [k('-'), 'zoomOut'], [k('0'), 'zoomFit'],
      [k('z', { metaKey: true }), 'undo'], [k('z', { ctrlKey: true }), 'undo'], [k('z', { metaKey: true, shiftKey: true }), 'redo'],
      [k('e', { metaKey: true }), 'export'], [k('?', { shiftKey: true }), 'shortcuts'], [k('Escape'), 'escape'], [k('S'), 'split'],
    ];
    for (const [e, want] of cases) expect(actionForKey(e), JSON.stringify(e.key)).toBe(want);
  });
  it('leaves other Cmd or Ctrl combinations to the browser', () => {
    for (const key of ['c', 'v', 'a', 'r', 'l', 's', 'w']) expect(actionForKey(k(key, { metaKey: true }))).toBeNull();
    expect(actionForKey(k('t', { ctrlKey: true }))).toBeNull();
    expect(actionForKey(k('s', { altKey: true }))).toBeNull();
  });
  it('ignores keys without a meaning', () => {
    expect(actionForKey(k('q'))).toBeNull();
    expect(actionForKey(k('Tab'))).toBeNull();
  });
  it('ignores hotkeys while focus is in an input, a textarea, a select or a contenteditable', () => {
    for (const t of [{ tagName: 'INPUT' }, { tagName: 'TEXTAREA' }, { tagName: 'SELECT' }, { tagName: 'DIV', isContentEditable: true }]) {
      const target = t as unknown as EventTarget;
      expect(isTypingTarget(target)).toBe(true);
      expect(actionForKey({ key: 's', target })).toBeNull();
      expect(actionForKey({ key: 'z', metaKey: true, target })).toBeNull();
      expect(actionForKey({ key: '?', target })).toBeNull();
    }
    expect(isTypingTarget(div)).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
  it('keeps Space, J, K, L live on a range slider but leaves its own keys to the slider', () => {
    const slider = { tagName: 'INPUT', type: 'range' } as unknown as EventTarget;
    expect(actionForKey({ key: ' ', target: slider })).toBe('playPause');
    expect(actionForKey({ key: 'k', target: slider })).toBe('playPause');
    expect(actionForKey({ key: 'j', target: slider })).toBe('back');
    expect(actionForKey({ key: 'l', target: slider })).toBe('forward');
    for (const key of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown', 'Delete', 'Backspace', '0', 's', 'Escape']) {
      expect(actionForKey({ key, target: slider }), key).toBeNull();
    }
    expect(actionForKey({ key: 'ArrowRight', shiftKey: true, target: slider })).toBeNull();
    expect(actionForKey({ key: 'z', metaKey: true, target: slider })).toBeNull();
    expect(actionForKey({ key: ' ', target: slider }, true)).toBeNull();
    expect(isTypingTarget({ tagName: 'INPUT', type: 'text' } as unknown as EventTarget)).toBe(true);
    expect(isTypingTarget({ tagName: 'INPUT' } as unknown as EventTarget)).toBe(true);
  });
  it('ignores hotkeys while a dialog is open, except Esc', () => {
    expect(actionForKey(k('s'), true)).toBeNull();
    expect(actionForKey(k(' '), true)).toBeNull();
    expect(actionForKey(k('z', { metaKey: true }), true)).toBeNull();
    expect(actionForKey(k('Escape'), true)).toBe('escape');
  });
  it('opens the sheet with ?', () => {
    expect(actionForKey(k('?'))).toBe('shortcuts');
    expect(actionForKey(k('?', { shiftKey: true }))).toBe('shortcuts');
  });
});

describe('SHORTCUTS (the sheet’s source)', () => {
  it('lists each action once, with keys and a label', () => {
    const actions = SHORTCUTS.map((s) => s.action);
    expect(new Set(actions).size).toBe(actions.length);
    expect(SHORTCUTS.every((s) => s.keys && s.label)).toBe(true);
    expect(actions).toEqual(expect.arrayContaining(['split', 'merge', 'undo', 'redo', 'shortcuts', 'export']));
  });
});
