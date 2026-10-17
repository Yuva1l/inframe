// The page's start (main.tsx calls it): build the engines and the shell, draw it. A start that fails still draws the page with the engine switcher,
// so the user can fall back to This browser.
import { render } from 'preact';
import { BrowserEngine } from './engine/browser';
import { availableEngines, type Engines, type SelectOptions } from './engine';
import { createShell, type Shell } from './shell';

export async function start(root: HTMLElement, avail: (o?: SelectOptions) => Promise<Engines> = availableEngines): Promise<Shell> {
  let shell: Shell | undefined;
  try {
    const engines = await avail({
      // a blip of the My Mac server is only news while My Mac is the engine in use; the browser store has no server to lose
      onDown: (down) => {
        const s = shell?.current.value;
        if (s && s.engine.kind === 'local') s.serverDown.value = down;
      },
    });
    shell = createShell(engines);
    await shell.start();
  } catch (e) {
    shell = createShell({ local: null, browser: new BrowserEngine() }, { failed: e instanceof Error ? e.message : String(e) });
    await shell.start();
  }
  render(<shell.Root />, root);
  return shell;
}
