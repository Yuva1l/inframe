// WorkerRunner: a cancelled job that does not stop in time costs only itself; the jobs nobody cancelled go to a fresh Worker.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WorkerRunner } from '../src/engine/browser';
import { NO_WEBGPU_HERE } from '../src/browser/worker';

const inlineSeen: any[] = [];
vi.mock('../src/browser/worker', async (orig) => ({
  ...(await orig<typeof import('../src/browser/worker')>()),
  createHandler: (post: (r: any) => void) => (m: any) => { inlineSeen.push(m); post({ type: 'done', id: m.id, result: `inline-${m.id}` }); },
}));

class FakeWorker {
  sent: any[] = [];
  terminated = false;
  onmessage: ((e: { data: any }) => void) | null = null;
  onerror: ((e: { message: string }) => void) | null = null;
  postMessage(m: any) { this.sent.push(m); }
  terminate() { this.terminated = true; }
  reply(data: any) { this.onmessage?.({ data }); }
}

describe('WorkerRunner cancel grace', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('rejects the cancelled job only; a live job in the same Worker keeps its Worker and is not sent again from zero', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    const progress: number[] = [];
    const b = r.send<string>({ op: 'run' } as any, (f) => progress.push(f));
    const aSettled = a.then(() => 'ok', (e) => e);
    let bState = 'pending';
    b.then(() => (bState = 'ok'), () => (bState = 'rejected'));
    workers[0].reply({ type: 'progress', id: 2, fraction: 0.5, message: 'half' });
    ac.abort();
    expect(workers[0].sent.at(-1)).toMatchObject({ op: 'cancel', job: 1 });
    await vi.advanceTimersByTimeAsync(5001);
    expect(await aSettled).toMatchObject({ status: 499 });
    expect(workers[0].terminated).toBe(false);
    expect(workers).toHaveLength(1);
    expect(workers[0].sent.filter((m) => m.id === 2)).toHaveLength(1); // the run was posted once
    expect(bState).toBe('pending');
    workers[0].reply({ type: 'done', id: 1, result: 'late' }); // the stopped job's late answer is ignored
    workers[0].reply({ type: 'progress', id: 2, fraction: 0.75, message: '' });
    workers[0].reply({ type: 'done', id: 2, result: 'fine' });
    await expect(b).resolves.toBe('fine');
    expect(progress).toEqual([0.5, 0.75]);
  });

  it('a live job that has shown no progress is queued behind the stuck one: the Worker is ended and the job goes to a fresh one', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    const b = r.send<string>({ op: 'boxes' } as any); // a tap picture, waiting for the turn A holds
    const aSettled = a.then(() => 'ok', (e) => e);
    ac.abort();
    await vi.advanceTimersByTimeAsync(5001);
    expect(await aSettled).toMatchObject({ status: 499 });
    expect(workers).toHaveLength(2);
    expect(workers[0].terminated).toBe(true);
    expect(workers[1].sent.filter((m) => m.id === 2)).toHaveLength(1);
    workers[0].reply({ type: 'done', id: 2, result: 'from the old one' }); // an ended Worker is not heard
    workers[1].reply({ type: 'done', id: 2, result: 'fresh' });
    await expect(b).resolves.toBe('fresh');
  });

  it('a running live job keeps the Worker while the stuck one is given up on, and the Worker is ended when that job settles', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    const b = r.send<string>({ op: 'run' } as any);
    const aSettled = a.then(() => 'ok', (e) => e);
    workers[0].reply({ type: 'progress', id: 2, fraction: 0.1, message: '' });
    ac.abort();
    await vi.advanceTimersByTimeAsync(5001);
    expect(await aSettled).toMatchObject({ status: 499 });
    expect(workers[0].terminated).toBe(false);
    // a picture asked for meanwhile has not started either; B is running, so the Worker stays
    const c = r.send<string>({ op: 'boxes' } as any);
    workers[0].reply({ type: 'done', id: 2, result: 'run done' });
    await expect(b).resolves.toBe('run done');
    // B settled: only C is live and it has shown nothing, so it may be stuck behind A: a fresh Worker takes it
    expect(workers[0].terminated).toBe(true);
    expect(workers).toHaveLength(2);
    workers[1].reply({ type: 'done', id: 3, result: 'c' });
    await expect(c).resolves.toBe('c');
  });

  it('the Worker is ended when the last live job settles after a stuck one was given up on', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    const b = r.send<string>({ op: 'run' } as any);
    a.catch(() => {});
    workers[0].reply({ type: 'progress', id: 2, fraction: 0.1, message: '' });
    ac.abort();
    await vi.advanceTimersByTimeAsync(5001);
    expect(workers[0].terminated).toBe(false);
    workers[0].reply({ type: 'done', id: 2, result: 'fine' });
    await expect(b).resolves.toBe('fine');
    expect(workers[0].terminated).toBe(true);
  });

  it('a stopped job that ends by itself after the grace leaves its Worker alone', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    const b = r.send<string>({ op: 'run' } as any);
    a.catch(() => {});
    workers[0].reply({ type: 'progress', id: 2, fraction: 0.1, message: '' });
    ac.abort();
    await vi.advanceTimersByTimeAsync(5001);
    workers[0].reply({ type: 'error', id: 1, status: 499, message: 'Cancelled.' }); // late, but it did stop
    workers[0].reply({ type: 'done', id: 2, result: 'fine' });
    await expect(b).resolves.toBe('fine');
    expect(workers[0].terminated).toBe(false);
  });

  it('a stuck cancelled job alone gets its Worker ended', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    const settled = a.then(() => 'ok', (e) => e);
    ac.abort();
    await vi.advanceTimersByTimeAsync(5001);
    expect(await settled).toMatchObject({ status: 499 });
    expect(workers[0].terminated).toBe(true);
  });

  it('a Worker that stops within the grace is kept', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const ac = new AbortController();
    const a = r.send<string>({ op: 'boxes' } as any, undefined, ac.signal);
    ac.abort();
    workers[0].reply({ type: 'error', id: 1, status: 499, message: 'Cancelled.' });
    await expect(a).rejects.toMatchObject({ status: 499 });
    await vi.advanceTimersByTimeAsync(6000);
    expect(workers[0].terminated).toBe(false);
  });
});

describe('WorkerRunner without WebGPU in Workers', () => {
  const noGpu = (id: number) => ({ type: 'error', id, status: 501, message: 'no webgpu', code: NO_WEBGPU_HERE });
  beforeEach(() => { inlineSeen.length = 0; });

  it('two jobs failing together both run on this thread', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const a = r.send<string>({ op: 'boxes' } as any);
    const b = r.send<string>({ op: 'run' } as any);
    workers[0].reply(noGpu(1));
    workers[0].reply(noGpu(2));
    await expect(Promise.all([a, b])).resolves.toHaveLength(2);
    expect(workers[0].terminated).toBe(true);
    expect(inlineSeen.map((m) => m.op).sort()).toEqual(['boxes', 'run']);
  });

  it('a job whose failure never arrives (the Worker was ended after the first) is sent to this thread too', async () => {
    const workers: FakeWorker[] = [];
    const r = new WorkerRunner((() => { const w = new FakeWorker(); workers.push(w); return w; }) as any);
    const a = r.send<string>({ op: 'boxes' } as any);
    const b = r.send<string>({ op: 'run' } as any);
    workers[0].reply(noGpu(1));
    await expect(a).resolves.toBeTypeOf('string');
    workers[0].reply(noGpu(2)); // an ended Worker is not heard
    await expect(b).resolves.toBeTypeOf('string');
    expect(inlineSeen.map((m) => m.op).sort()).toEqual(['boxes', 'run']);
  });
});
