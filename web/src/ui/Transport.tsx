import { formatClock } from '../model/format';
import type { Store } from '../state';
import { Icon } from './Icons';

const RATES = [0.25, 0.5, 1, 1.5, 2];

export function Transport({ store }: { store: Store }) {
  const m = store.media.value;
  if (!m) return null;
  const playing = store.playing.value;
  const t = store.time.value;
  return (
    <div class="transport">
      <button class="icon" aria-label="Previous clip" title="Previous clip (Up)" onClick={() => store.dispatch('prevClip')}><Icon name="skipPrev" /></button>
      <button class="icon" aria-label="Back one frame" title="Back one frame (Left)" onClick={() => store.dispatch('stepBack')}><Icon name="stepBack" /></button>
      <button class="play" aria-label={playing ? 'Pause' : 'Play'} title="Play or pause (Space)" onClick={() => store.playPause()}>
        <Icon name={playing ? 'pause' : 'play'} size={18} />
      </button>
      <button class="icon" aria-label="Forward one frame" title="Forward one frame (Right)" onClick={() => store.dispatch('stepFwd')}><Icon name="stepFwd" /></button>
      <button class="icon" aria-label="Next clip" title="Next clip (Down)" onClick={() => store.dispatch('nextClip')}><Icon name="skipNext" /></button>
      <span class="mono readout" data-testid="readout">{formatClock(t)} / {formatClock(m.end_time)}</span>
      <span class="sr-only" aria-live="polite">{playing ? '' : `Playhead at ${formatClock(Math.floor(t))}`}</span>
      <span class="spacer" />
      <label class="muted">
        Speed{' '}
        <select value={store.rate.value} aria-label="Playback speed" onChange={(e) => store.setRate(Number((e.target as HTMLSelectElement).value))}>
          {RATES.map((r) => <option key={r} value={r}>{r}x</option>)}
          {!RATES.includes(store.rate.value) && <option value={store.rate.value}>{store.rate.value}x</option>}
        </select>
      </label>
      <button class={`tog${store.clipsOnly.value ? ' on' : ''}`} aria-pressed={store.clipsOnly.value} title="Play clips only (P)" onClick={() => store.dispatch('clipsOnly')}>
        clips only
      </button>
      <button class={`tog${store.showBoxes.value ? ' on' : ''}`} aria-pressed={store.showBoxes.value} title="Show the target box (B)" onClick={() => store.dispatch('showBoxes')}>
        boxes
      </button>
      {m.has_audio && (
        <label class="vol" title="Volume">
          <Icon name="volume" />
          <input type="range" min={0} max={1} step={0.05} defaultValue={1} aria-label="Volume" onInput={(e) => store.setVolume(Number((e.target as HTMLInputElement).value))} />
        </label>
      )}
    </div>
  );
}
