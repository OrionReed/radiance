// GPU profiler for the WebGPU pages: GPU time per pass from timestamp queries, CPU time per frame, bytes per buffer, and
// bandwidth where a page declares the bytes a pass moves. docs/profiling.md says what the browser can measure.
//
// Use: make one Profiler right after requestDevice (with Profiler.features(adapter) among the required features), then
// per frame:
//   profiler.beginFrame();
//   const pass = profiler.computePass(enc, 'merge');   // or timestampWrites: profiler.timestampWrites('blit')
//   profiler.name(pass, 'merge 3');                     // per-dispatch mode: names the dispatches that follow
//   ...
//   profiler.endFrame(enc);
//   device.queue.submit([enc.finish()]);
//   profiler.collect();
// While off (the default) every call is one branch; GPU resources are made when it is first switched on.
//
// On Metal (checked on an M2 Pro) timestamps are taken at pass boundaries only, so a pass with several dispatches gives one time.
// Per-dispatch mode (perDispatch) runs each dispatch in its own timed pass, at about 12-15 µs of wall time per dispatch.
// A pass's time runs from the later of its begin and the previous pass's end: a render pass after compute passes has its
// begin stamped when the frame starts, so raw times would count the frame twice.

export interface Traffic {
  read: number; // bytes read per occurrence of the pass
  write: number; // bytes written per occurrence
}

interface PassTime {
  label: string;
  begin: number; // ns from the frame's first timestamp, no earlier than the previous pass's end
  end: number;
}

interface Frame {
  passes: PassTime[];
  quantised: boolean;
}

interface CpuFrame {
  encode: number; // ms from beginFrame to endFrame
  other: number; // ms of the rest of the frame (beginFrame to the next beginFrame)
}

export interface PassStats {
  label: string;
  ms: number; // mean GPU time per frame, all occurrences summed
  min: number; // fastest frame
  count: number; // occurrences per frame
  traffic?: Traffic;
}

export interface FrameStats {
  frames: number;
  cpuMs: number; // encode time per frame
  gpuMs: number; // sum of pass times per frame
  spanMs: number; // first pass begin to last pass end
  quantised: boolean;
  passes: PassStats[];
}

const CAPACITY = 1024; // timestamps per frame (two per pass)
const HISTORY = 120; // frames kept (drawn)
const AVERAGE = 60; // frames averaged in stats()
const QUANTUM = 65536; // Chrome rounds timestamps down to 2^16 ns without the developer-features flag

export class Profiler {
  static features(adapter: GPUAdapter): GPUFeatureName[] {
    return adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  }

  readonly supported: boolean;
  perDispatch = false;
  /** a line the page shows in the panel, e.g. its dispatches and memory per frame */
  info = '';
  /** frames that can be in flight to the CPU; set before first enabling (more for frames submitted back to back) */
  readbacks = 6;
  /** bytes a pass moves, by label (per occurrence); shown as bandwidth */
  readonly traffic = new Map<string, Traffic>();
  /** live buffers made on the device since the profiler was created */
  readonly buffers = new Map<GPUBuffer, { label: string; size: number }>();

  #device: GPUDevice;
  #enabled = false;
  #querySet: GPUQuerySet | null = null;
  #resolve: GPUBuffer | null = null;
  #free: GPUBuffer[] = [];
  #labels: string[] = [];
  #cpuStart = 0;
  #encodeMs = 0;
  #pending: { buffer: GPUBuffer; labels: string[]; generation: number }[] = [];
  #frames: Frame[] = [];
  #cpu: CpuFrame[] = [];
  #generation = 0; // frames encoded before the last reset are dropped
  #panel: Panel | null = null;

  constructor(device: GPUDevice) {
    this.#device = device;
    this.supported = device.features.has('timestamp-query');
    const create = device.createBuffer.bind(device);
    device.createBuffer = (descriptor: GPUBufferDescriptor) => {
      const buffer = create(descriptor);
      this.buffers.set(buffer, { label: descriptor.label || '(unlabelled)', size: descriptor.size });
      const destroy = buffer.destroy.bind(buffer);
      buffer.destroy = () => (this.buffers.delete(buffer), destroy());
      return buffer;
    };
  }

  get enabled() {
    return this.#enabled;
  }
  set enabled(on: boolean) {
    if (on === this.#enabled) return;
    this.#enabled = on;
    this.reset();
    if (on && this.supported && !this.#querySet) {
      const usage = GPUBufferUsage;
      this.#querySet = this.#device.createQuerySet({ type: 'timestamp', count: CAPACITY });
      this.#resolve = this.#device.createBuffer({ label: 'profiler', size: CAPACITY * 8, usage: usage.QUERY_RESOLVE | usage.COPY_SRC });
      for (let i = 0; i < this.readbacks; i++)
        this.#free.push(this.#device.createBuffer({ label: 'profiler', size: CAPACITY * 8, usage: usage.MAP_READ | usage.COPY_DST }));
      for (const b of [this.#resolve, ...this.#free]) this.buffers.delete(b);
    }
    this.#panel?.show(on);
  }

  /** forget the frames so far (e.g. after a change of settings) */
  reset() {
    this.#generation++;
    this.#frames = [];
    this.#cpu = [];
    this.#cpuStart = 0;
  }

  beginFrame() {
    if (!this.#enabled) return;
    const now = performance.now();
    if (this.#cpuStart) push(this.#cpu, { encode: this.#encodeMs, other: Math.max(0, now - this.#cpuStart - this.#encodeMs) });
    this.#labels = [];
    this.#cpuStart = now;
  }

  /** timestamp writes for one pass, or undefined when off */
  timestampWrites(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.#enabled || !this.#querySet || 2 * this.#labels.length + 2 > CAPACITY) return undefined;
    const i = 2 * this.#labels.length;
    this.#labels.push(label);
    return { querySet: this.#querySet, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 };
  }

  /** a compute pass timed as label; in per-dispatch mode each dispatch is its own timed pass */
  computePass(enc: GPUCommandEncoder, label: string): GPUComputePassEncoder {
    if (!this.#enabled) return enc.beginComputePass();
    if (this.perDispatch && this.#querySet) return new SplitPass(this, enc, label) as unknown as GPUComputePassEncoder;
    return enc.beginComputePass({ timestampWrites: this.timestampWrites(label) });
  }

  /** names the following dispatches of a pass from computePass (per-dispatch mode only) */
  name(pass: GPUComputePassEncoder, label: string) {
    if (pass instanceof SplitPass) pass.next = label;
  }

  endFrame(enc: GPUCommandEncoder) {
    if (!this.#enabled) return;
    this.#encodeMs = performance.now() - this.#cpuStart;
    const buffer = this.#labels.length ? this.#free.pop() : undefined;
    if (!buffer) return; // no timed passes, or the CPU is ahead of the readbacks: this frame is not read
    const n = 2 * this.#labels.length;
    enc.resolveQuerySet(this.#querySet!, 0, n, this.#resolve!, 0);
    enc.copyBufferToBuffer(this.#resolve!, 0, buffer, 0, n * 8);
    this.#pending.push({ buffer, labels: this.#labels, generation: this.#generation });
    this.#labels = [];
  }

  /** after submit: read back the frames resolved so far */
  collect() {
    for (const { buffer, labels, generation } of this.#pending.splice(0)) {
      buffer.mapAsync(GPUMapMode.READ).then(
        () => {
          const t = new BigInt64Array(buffer.getMappedRange(0, labels.length * 16));
          const t0 = t.reduce((a, b) => (b < a ? b : a));
          const quantised = t.every((x) => x % BigInt(QUANTUM) === 0n);
          let previous = 0;
          const passes = labels.map((label, i) => {
            const begin = Number(t[2 * i] - t0), end = Number(t[2 * i + 1] - t0);
            const p = { label, begin: Math.min(end, Math.max(begin, previous)), end };
            previous = Math.max(previous, end);
            return p;
          });
          buffer.unmap();
          this.#free.push(buffer);
          if (this.#enabled && generation === this.#generation) push(this.#frames, { passes, quantised });
        },
        () => this.#free.push(buffer),
      );
    }
  }

  /** GPU frames read back, oldest first */
  get frames(): readonly Frame[] {
    return this.#frames;
  }

  /** CPU frames, oldest first */
  get cpuFrames(): readonly CpuFrame[] {
    return this.#cpu;
  }

  /** means over the last frames read back */
  stats(): FrameStats {
    const F = this.#frames.slice(-AVERAGE), C = this.#cpu.slice(-AVERAGE);
    const per = new Map<string, number[]>(), counts = new Map<string, number>();
    let gpu = 0, span = 0;
    for (const f of F) {
      const sums = new Map<string, number>();
      for (const p of f.passes) sums.set(p.label, (sums.get(p.label) ?? 0) + (p.end - p.begin) / 1e6);
      for (const [label, ms] of sums) (per.get(label) ?? per.set(label, []).get(label)!).push(ms);
      gpu += [...sums.values()].reduce((a, b) => a + b, 0);
      span += f.passes.length ? Math.max(...f.passes.map((p) => p.end)) / 1e6 : 0;
    }
    for (const p of F.at(-1)?.passes ?? []) counts.set(p.label, (counts.get(p.label) ?? 0) + 1);
    const n = Math.max(1, F.length);
    const passes = [...per].map(([label, ms]) => ({
      label,
      ms: ms.reduce((a, b) => a + b, 0) / n,
      min: Math.min(...ms),
      count: counts.get(label) ?? 0,
      traffic: this.traffic.get(label),
    }));
    const cpuMs = C.reduce((a, c) => a + c.encode, 0) / Math.max(1, C.length);
    return { frames: F.length, cpuMs, gpuMs: gpu / n, spanMs: span / n, quantised: F.length > 0 && F.every((f) => f.quantised), passes };
  }

  /** the panel (made on first call; shown while enabled) */
  panel(): HTMLElement {
    this.#panel ??= new Panel(this);
    this.#panel.show(this.#enabled);
    return this.#panel.root;
  }
}

function push<T>(list: T[], item: T) {
  list.push(item);
  if (list.length > HISTORY) list.shift();
}

// A stand-in for a compute pass that runs each dispatch in a pass of its own, timed under the name set last (or the
// pass's label). Covers what the pages use: setPipeline, setBindGroup, dispatchWorkgroups, end.
class SplitPass {
  next: string;
  #profiler: Profiler;
  #enc: GPUCommandEncoder;
  #pipeline: GPUComputePipeline | null = null;
  #groups = new Map<number, [GPUBindGroup | null, Iterable<number> | undefined]>();

  constructor(profiler: Profiler, enc: GPUCommandEncoder, label: string) {
    this.#profiler = profiler;
    this.#enc = enc;
    this.next = label;
  }
  setPipeline(pipeline: GPUComputePipeline) {
    this.#pipeline = pipeline;
  }
  setBindGroup(index: number, group: GPUBindGroup | null, offsets?: Iterable<number>) {
    this.#groups.set(index, [group, offsets && [...offsets]]);
  }
  dispatchWorkgroups(x: number, y = 1, z = 1) {
    const pass = this.#enc.beginComputePass({ timestampWrites: this.#profiler.timestampWrites(this.next) });
    if (this.#pipeline) pass.setPipeline(this.#pipeline);
    for (const [i, [g, o]] of this.#groups) if (o) pass.setBindGroup(i, g, o);
    else pass.setBindGroup(i, g);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
  }
  end() {}
}

// ── panel: GPU time per pass, frame by frame as stacked bars, with a legend beside each pass's band ──

const STYLE = `
.gpu-profiler { font: 12px/1.3 system-ui, -apple-system, sans-serif; color: #aaa; background: rgba(16, 17, 20, 0.94);
  border: 1px solid #2c2e33; border-radius: 8px; padding: 8px 10px 10px; user-select: none; }
.gpu-profiler[hidden] { display: none; }
.gpu-profiler header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px; }
.gpu-profiler header b { color: #fff; font-weight: 600; }
.gpu-profiler .note { color: rgba(255, 255, 255, 0.55); margin: 2px 0 6px; }
.gpu-profiler .num { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
.gpu-profiler .hot { color: #f5a623; }
.gpu-profiler header label { margin-left: auto; cursor: pointer; display: flex; align-items: center; gap: 4px; }
.gpu-profiler canvas { display: block; }
`;

const HUES = ['#4e9bd6', '#ef8b2c', '#5bb26a', '#e1575a', '#9a7fd1', '#e3c348', '#4bbbb0', '#d77fb3', '#a08a6c', '#9aa3ad'];
const BAR = 3; // px per frame
const ROW = 15; // legend row height
const GRAPH = HISTORY * BAR;
const GAP = 28; // graph to legend
const LEGEND = 230;
const SCALE_FRAMES = 30; // the scale follows the latest frames: their 90th percentile (taller ones are cut off)
// the CPU figure is marked when encode (mean of the last 10 frames) is above the budget, or above twice its usual
// (median) plus a margin: performance.now() has 0.1 ms steps on pages that are not cross-origin isolated
const CPU_BUDGET = 2;
const CPU_MARGIN = 0.5;
const MONO = '11px ui-monospace, Menlo, monospace';
const SANS = '12px system-ui, -apple-system, sans-serif';

const fmtMs = (ms: number) => ms.toFixed(3);
const fmtBytes = (b: number) => (b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(2)} GiB` : `${(b / (1 << 20)).toFixed(1)} MiB`);
const median = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1] ?? 0;

// a stage's colour by its first word; levels of it ("merge 3") alternate lighter and darker
function colourOf(label: string, stages: Map<string, string>) {
  const [stage, level] = label.split(' ');
  if (!stages.has(stage)) stages.set(stage, HUES[stages.size % HUES.length]);
  const base = stages.get(stage)!, n = Number(level);
  if (!Number.isFinite(n)) return base;
  const k = ((n % 4) - 1.5) * 0.14; // -0.21 .. 0.21
  const c = [1, 3, 5].map((i) => parseInt(base.slice(i, i + 2), 16)).map((v) => Math.round(k > 0 ? v + (255 - v) * k : v * (1 + k)));
  return `rgb(${c.join(',')})`;
}

interface Series {
  label: string;
  colour: string;
  ms: number; // mean over the recent frames
  extra?: string;
}

class Panel {
  root: HTMLElement;
  #profiler: Profiler;
  #gpu: HTMLElement;
  #cpu: HTMLElement;
  #info: HTMLElement;
  #page: HTMLElement;
  #split: HTMLInputElement;
  #canvas: HTMLCanvasElement;
  #stages = new Map<string, string>();
  #timer = 0;

  constructor(profiler: Profiler) {
    this.#profiler = profiler;
    if (!document.getElementById('gpu-profiler-style'))
      document.head.append(Object.assign(document.createElement('style'), { id: 'gpu-profiler-style', textContent: STYLE }));
    const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: object = {}): HTMLElementTagNameMap[K] => Object.assign(document.createElement(tag), props);
    this.root = el('div', { className: 'gpu-profiler', hidden: true });
    this.#gpu = el('b');
    this.#cpu = el('span');
    this.#info = el('span');
    this.#split = el('input', { type: 'checkbox' });
    this.#split.addEventListener('change', () => ((profiler.perDispatch = this.#split.checked), profiler.reset()));
    const toggle = el('label');
    toggle.append(this.#split, 'per dispatch');
    const header = el('header');
    header.append(this.#gpu, this.#cpu, this.#info, toggle);
    this.#canvas = el('canvas');
    this.#page = el('div', { className: 'note' });
    this.root.append(header, this.#page, this.#canvas);
  }

  show(on: boolean) {
    this.root.hidden = !on;
    clearInterval(this.#timer);
    if (on) this.#timer = window.setInterval(() => this.render(), 100);
  }

  render() {
    const p = this.#profiler, s = p.stats();
    this.#split.checked = p.perDispatch;

    // header: GPU, CPU encode (marked when well above its usual value or the budget), memory
    this.#gpu.innerHTML = `GPU <span class="num">${fmtMs(s.gpuMs)}</span> ms`;
    this.#gpu.title = p.perDispatch ? 'a timed pass per dispatch: per-level times, about 12–15 µs of extra wall time per dispatch' : 'timestamps around each pass';
    this.#page.textContent = p.info;
    this.#page.hidden = !p.info;
    const cpu = p.cpuFrames, recent = cpu.slice(-10).map((c) => c.encode), now = recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length);
    const base = median(cpu.map((c) => c.encode)), frame = median(cpu.map((c) => c.encode + c.other));
    const hot = now > CPU_BUDGET || (cpu.length > 30 && now > 2 * base + CPU_MARGIN);
    this.#cpu.innerHTML = `CPU <span class="num">${fmtMs(now)}</span> ms`;
    this.#cpu.className = hot ? 'hot' : '';
    this.#cpu.title = `encode; usual ${fmtMs(base)} ms; frame ${fmtMs(frame)} ms`;
    const buffers = [...p.buffers.values()].sort((a, b) => b.size - a.size);
    this.#info.textContent = fmtBytes(buffers.reduce((a, b) => a + b.size, 0)) + (!p.supported ? ' · no timestamps' : s.quantised ? ' · 65 µs steps' : '');
    this.#info.title = buffers.slice(0, 12).map((b) => `${b.label}  ${fmtBytes(b.size)}`).join('\n');

    const frames = p.frames.map((f) => {
      const m = new Map<string, number>();
      for (const q of f.passes) m.set(q.label, (m.get(q.label) ?? 0) + (q.end - q.begin) / 1e6);
      return m;
    });
    const series: Series[] = s.passes.map((q) => {
      const t = q.traffic, gbps = t && q.ms > 0 ? ((t.read + t.write) * q.count) / (q.ms * 1e6) : 0;
      return { label: q.label, colour: colourOf(q.label, this.#stages), ms: q.ms, extra: gbps ? `${gbps.toFixed(0)} GB/s` : undefined };
    });

    const dpr = window.devicePixelRatio || 1;
    const H = Math.max(140, series.length * ROW + 8), W = GRAPH + GAP + LEGEND, c = this.#canvas;
    if (c.width !== W * dpr || c.height !== H * dpr) Object.assign(c, { width: W * dpr, height: H * dpr }), Object.assign(c.style, { width: W + 'px', height: H + 'px' });
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    graph(ctx, H, frames, series);
  }
}

// Stacked bars of the frames (newest on the right), scaled to fit most of the latest frames. Each legend row sits at
// the centre of its pass's band in the mean stack, rows pushed apart only as far as they must be to not overlap.
function graph(ctx: CanvasRenderingContext2D, h: number, frames: Map<string, number>[], series: Series[]) {
  ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.fillRect(0, 0, GRAPH, h);
  const totals = frames.map((f) => [...f.values()].reduce((a, b) => a + b, 0));
  const mean = series.reduce((a, s) => a + s.ms, 0);
  const recent = totals.slice(-SCALE_FRAMES).sort((a, b) => a - b);
  const scale = niceMax(Math.max(1e-3, mean, recent[Math.floor(0.9 * (recent.length - 1))] ?? 0));
  const y = (ms: number) => Math.max(0, h - (ms / scale) * h);
  const x0 = GRAPH - frames.length * BAR;
  frames.forEach((f, i) => {
    let acc = 0;
    for (const s of series) {
      const v = f.get(s.label) ?? 0, a = y(acc), b = y(acc + v);
      acc += v;
      if (a - b > 0) (ctx.fillStyle = s.colour), ctx.fillRect(x0 + i * BAR, b, BAR - 1, a - b);
    }
  });
  ctx.fillStyle = '#666';
  ctx.font = MONO;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText(`${+scale.toPrecision(3)} ms`, 3, 3);

  // bands of the mean stack, and rows placed at their centres
  let acc = 0;
  const bands = series.map((s) => {
    const a = y(acc), b = y(acc + s.ms);
    acc += s.ms;
    return [b, a];
  });
  const rows = place(bands.map(([b, a]) => (a + b) / 2 - ROW / 2), ROW, 0, h);
  ctx.font = MONO;
  const numW = Math.max(0, ...series.map((s) => ctx.measureText(fmtMs(s.ms)).width));
  const lx = GRAPH + GAP;
  series.forEach((s, k) => {
    const [b, a] = bands[k], r = rows[k];
    ctx.fillStyle = s.colour;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(GRAPH, b);
    ctx.lineTo(lx - 4, r + 2);
    ctx.lineTo(lx - 4, r + ROW - 2);
    ctx.lineTo(GRAPH, Math.max(a, b + 0.5));
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.textBaseline = 'middle';
    ctx.font = MONO;
    ctx.fillStyle = '#eee';
    ctx.fillText(fmtMs(s.ms), lx, r + ROW / 2);
    ctx.font = SANS;
    ctx.fillStyle = s.colour;
    ctx.fillText(s.label, lx + numW + 8, r + ROW / 2);
    if (s.extra) {
      const w = ctx.measureText(s.label).width;
      ctx.fillStyle = '#777';
      ctx.font = '10px system-ui, -apple-system, sans-serif';
      ctx.fillText(s.extra, lx + numW + 14 + w, r + ROW / 2);
    }
  });
}

// Tops of rows of height `size` as close as possible (least squares) to `wanted`, without overlap, within [lo, hi]:
// isotonic regression on wanted[i] - i * size, in order of wanted, then clamped to the bounds.
function place(wanted: number[], size: number, lo: number, hi: number): number[] {
  const order = wanted.map((_, i) => i).sort((i, j) => wanted[i] - wanted[j]);
  const blocks: { sum: number; n: number }[] = [];
  order.forEach((i, k) => {
    blocks.push({ sum: wanted[i] - k * size, n: 1 });
    while (blocks.length > 1 && blocks.at(-2)!.sum / blocks.at(-2)!.n > blocks.at(-1)!.sum / blocks.at(-1)!.n) {
      const top = blocks.pop()!;
      blocks.at(-1)!.sum += top.sum;
      blocks.at(-1)!.n += top.n;
    }
  });
  const q = blocks.flatMap((b) => Array(b.n).fill(b.sum / b.n) as number[]);
  const pos = q.map((v, k) => v + k * size);
  for (let k = 0; k < pos.length; k++) pos[k] = Math.max(pos[k], lo + k * size, k ? pos[k - 1] + size : -Infinity);
  for (let k = pos.length - 1; k >= 0; k--) pos[k] = Math.min(pos[k], hi - (pos.length - k) * size, k < pos.length - 1 ? pos[k + 1] - size : Infinity);
  const out: number[] = [];
  order.forEach((i, k) => (out[i] = pos[k]));
  return out;
}

function niceMax(v: number) {
  const e = 10 ** Math.floor(Math.log10(v));
  return ([1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((m) => m * e >= v * 1.02) ?? 10) * e;
}
