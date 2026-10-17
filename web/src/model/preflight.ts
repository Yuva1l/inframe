// The checklist of what a run needs (docs/T9A_DESIGN.md 1.5): a missing item names the command that fixes it.
import type { Preflight } from '../engine/types';

export interface PreflightNeeds {
  /** only this preset's detector model is checked; null checks every preset (the first-use screen) */
  preset: string | null;
  /** motion mode needs no model */
  motion?: boolean;
  /** saved and photo targets need the embedder; null = show it (the first-use screen) */
  identity: boolean | null;
  /** targets that follow one identity (tap, saved, photos) and activity riding need the tracker (`wants_tracks`); null = show it */
  tracking: boolean | null;
  /** browser: the target follows one person by how they look (a tap), so the recognition model is needed; null = show it (the first-use screen). Left out, `tracking` decides. */
  follow?: boolean | null;
}

export interface PreflightRow {
  key: string;
  ok: boolean;
  text: string;
  fix?: string;
}

export const FIX_MODELS = 'Run `inframe fetch-models`';
export const FIX_IDENTITY = 'Run `uv sync --extra identity`';
export const FIX_TRACKING = 'Run `uv sync --extra tracking`';
export const FIX_FFMPEG = 'Install ffmpeg (brew install ffmpeg)';

export const FIX_BROWSER = 'Use Chrome or Safari 26, or My Mac';
export const FIX_DOWNLOAD = 'Download the models below';

export function preflightRows(p: Preflight, n: PreflightNeeds): PreflightRow[] {
  if (p.browser) {
    const b = p.browser;
    const rows: PreflightRow[] = [
      // no WebGPU is a warning, not a stop: the models still run on the CPU (about 19 times slower)
      { key: 'webgpu', ok: true, text: b.webgpu ? 'WebGPU' : 'No WebGPU here: the models run on the CPU, about 19 times slower' },
      { key: 'webcodecs', ok: b.webcodecs, text: 'Video decoding (WebCodecs)', fix: FIX_BROWSER },
    ];
    if (b.hevc === false) rows.push({ key: 'hevc', ok: true, text: 'This browser cannot decode HEVC video (iPhone video); H.264 works' });
    if (!n.motion) rows.push({ key: 'models', ok: b.models.detectorReady, text: 'Detector model downloaded', fix: b.models.available ? FIX_DOWNLOAD : (b.models.error ?? 'No models are published here') });
    // a tap follows one person by how they look: the DINOv2 file is needed too
    // (activity riding needs no such model in the browser, so `tracking` alone does not ask for it)
    const follow = n.follow === undefined ? n.tracking : n.follow;
    if (!n.motion && follow !== false) rows.push({ key: 'embedder', ok: b.models.embedderReady === true, text: 'Recognition model downloaded', fix: b.models.available ? FIX_DOWNLOAD : (b.models.error ?? 'No models are published here') });
    return rows;
  }
  const rows: PreflightRow[] = [{ key: 'ffmpeg', ok: p.ffmpeg, text: 'ffmpeg and ffprobe', fix: FIX_FFMPEG }];
  if (!n.motion) {
    const names = n.preset === null ? Object.keys(p.models) : [n.preset];
    for (const name of names) rows.push({ key: `model-${name}`, ok: p.models[name] === true, text: `detector model: ${name}`, fix: FIX_MODELS });
  }
  if (n.identity !== false) rows.push({ key: 'identity', ok: p.identity, text: 'photo and saved targets (embedder)', fix: FIX_IDENTITY });
  if (n.tracking !== false) rows.push({ key: 'tracking', ok: p.tracking, text: 'tracking', fix: FIX_TRACKING });
  return rows;
}

/** The rows that stop a run before it starts. */
export const blocking = (rows: readonly PreflightRow[]): PreflightRow[] => rows.filter((r) => !r.ok);
