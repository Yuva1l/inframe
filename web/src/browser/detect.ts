// Detections and the box-to-score rule (docs/T9B_DESIGN.md sections 3 to 5): the TypeScript port of src/inframe/detect.py (preprocess,
// postprocess) and the onnxruntime-web sessions the Worker runs. preprocess and postprocess are pure and checked against the golden
// vectors of tools/gen_golden.py (tests/golden/detect_v1.json); the sessions need a browser with WebGPU [unverified in Node].
import { EngineError } from '../engine/types';

/** COCO category id 1; RF-DETR COCO checkpoints use 91 class slots (detect.py PERSON). */
export const PERSON = 1;
export const MIN_PERSON_SCORE = 0.2;
export const MEAN = [0.485, 0.456, 0.406] as const;
export const STD = [0.229, 0.224, 0.225] as const;
/** detect.MODELS: input side of each RF-DETR size (the browser ships small only). */
export const INPUT_SIDE = { small: 512 } as const;
/** DINOv2-S input side (the embedder, part 4 uses it; part 3 checks its parity). */
export const DINO_SIDE = 224;

export type Box4 = [number, number, number, number];
export interface Detection {
  score: number;
  /** x0, y0, x1, y1 normalised to the frame */
  box: Box4;
  trackId?: number | null;
  /** the DINOv2 look of the box (L2-normalised), filled for tap targets by embed.ts; null = not embedded (too small or too faint) */
  embedding?: Float32Array | null;
}

/** An RGBA picture of the detector's input size (width x height, already squashed) as the NCHW float tensor: v / 255, then (v - mean) / std.
 *  detect.preprocess does the same in float32; the squash itself (a bilinear resize) is the canvas's job in the browser. */
export function preprocess(rgba: ArrayLike<number>, width: number, height: number): Float32Array {
  const n = width * height;
  if (rgba.length < n * 4) throw new Error(`preprocess: ${rgba.length} values are too few for ${width} x ${height} RGBA`);
  const out = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    const mean = Math.fround(MEAN[c]);
    const std = Math.fround(STD[c]);
    const o = c * n;
    for (let i = 0; i < n; i++) out[o + i] = (Math.fround(rgba[i * 4 + c] / 255) - mean) / std;
  }
  return out;
}

/** detect.postprocess for one image: boxes (Q x 4, cxcywh normalised), logits (Q x C) -> detections of class id `cls` with sigmoid score
 *  of at least `minScore`, best first (ties keep their query order). */
export function postprocess(boxes: ArrayLike<number>, logits: ArrayLike<number>, queries: number, classes: number, cls = PERSON, minScore = MIN_PERSON_SCORE): Detection[] {
  if (!(cls >= 0 && cls < classes)) throw new Error(`class id ${cls} is outside the model's ${classes} class slots`);
  const floor = Math.fround(minScore);
  const out: Detection[] = [];
  for (let q = 0; q < queries; q++) {
    const p = Math.fround(1 / (1 + Math.exp(-logits[q * classes + cls])));
    if (!(p >= floor)) continue;
    const cx = boxes[q * 4];
    const cy = boxes[q * 4 + 1];
    const w = boxes[q * 4 + 2];
    const h = boxes[q * 4 + 3];
    out.push({ score: p, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
  }
  return out.sort((a, b) => b.score - a.score);
}

// ---------------------------------------------------------------- onnxruntime-web sessions
/** The parts of an ort session this code uses (so a test can stand in for it). */
export interface Session {
  inputName: string;
  outputNames: string[];
  /** which execution provider the session runs on */
  provider: 'webgpu' | 'wasm';
  run(input: Float32Array, dims: readonly number[]): Promise<Record<string, { data: Float32Array; dims: readonly number[] }>>;
  release(): Promise<void>;
}

export interface SessionOptions {
  /** ort log severity: 0 verbose, 1 info, 2 warning (default), 3 error */
  logSeverity?: number;
  /** execution providers in order of preference; default webgpu, then wasm with a warning */
  providers?: ('webgpu' | 'wasm')[];
  onWarn?: (text: string) => void;
}

let ortReady: Promise<typeof import('onnxruntime-web/webgpu')> | null = null;

/** onnxruntime-web (WebGPU build, loaded when first needed). Its WebAssembly runtime comes from bytes we already hold, so ort fetches nothing. */
export function loadOrt(wasm: Uint8Array) {
  return (ortReady ??= import('onnxruntime-web/webgpu').then((ort) => {
    ort.env.wasm.wasmBinary = wasm;
    ort.env.wasm.numThreads = 1; // no cross-origin isolation, so no threads
    ort.env.wasm.proxy = false;
    return ort;
  }));
}

export async function createSession(model: Uint8Array, wasm: Uint8Array, opts: SessionOptions = {}): Promise<Session> {
  const ort = await loadOrt(wasm);
  const order = opts.providers ?? ['webgpu', 'wasm'];
  let lastError: unknown = null;
  for (const ep of order) {
    try {
      const s = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: 'all', logSeverityLevel: (opts.logSeverity ?? 2) as 0 | 1 | 2 | 3 | 4 });
      if (ep === 'wasm' && order[0] !== 'wasm') opts.onWarn?.('WebGPU was not available: the model runs on the CPU in this tab, about 19 times slower.');
      return {
        inputName: s.inputNames[0],
        outputNames: [...s.outputNames],
        provider: ep,
        async run(input, dims) {
          const t = new ort.Tensor('float32', input, [...dims]);
          const r = await s.run({ [s.inputNames[0]]: t });
          const out: Record<string, { data: Float32Array; dims: readonly number[] }> = {};
          for (const k of Object.keys(r)) out[k] = { data: (await r[k].getData()) as Float32Array, dims: r[k].dims };
          for (const k of Object.keys(r)) r[k].dispose();
          return out;
        },
        release: () => s.release(),
      };
    } catch (e) {
      lastError = e;
    }
  }
  throw new EngineError(500, `The model could not start (${order.join(', ')}): ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

/** RF-DETR small on a session. `detect` takes the NCHW tensor of one squashed picture. */
export class RfDetr {
  readonly res = INPUT_SIDE.small;
  constructor(readonly session: Session) {}

  async detect(tensor: Float32Array, cls = PERSON, minScore = MIN_PERSON_SCORE): Promise<Detection[]> {
    const out = await this.session.run(tensor, [1, 3, this.res, this.res]);
    const names = Object.keys(out);
    const boxes = names.find((k) => out[k].dims[out[k].dims.length - 1] === 4);
    const logits = names.find((k) => k !== boxes && out[k].dims.length === 3);
    if (!boxes || !logits) throw new EngineError(500, 'The detector gave outputs this page does not know.');
    const [, q, c] = out[logits].dims;
    return postprocess(out[boxes].data, out[logits].data, q, c, cls, minScore);
  }

  release = () => this.session.release();
}
