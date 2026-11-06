// References for the scheme pages: a progressive path tracer and an exact angular quadrature, the HRC testbed's two
// references (site/hrc/hrc.ts) with bounces off and no sky, on the scheme pages' scene buffers.
//
// Input, per pixel, [x * N + y] with y up: a radiance S and a transmittance A across the pixel. A <= 0 is solid and shows
// S; 0 < A < 1 is a volume of source radiance S (a chord of length l adds S (1 - A^l)); A = 1 with S > 0 emits S per
// unit length. Rays leave the pixel centre and walk the pixels they cross (DDA), with the exact chord through each.
// Output, in the same layout: the mean radiance over angle at each pixel centre, L = fluence / 2π.
// The path tracer adds samples every frame; the exact reference integrates M directions per pixel by the trapezoid rule,
// locating each jump between pixels by bisection, in chunks run in the background, refining in place.

const WORKGROUP = 8;

const shaderSource = /* wgsl */ `
struct Params { n: u32, frame: u32, samples: u32, m: u32, chunkLen: u32, chunk: u32, rowStart: u32, rowCount: u32, depth: u32, scale: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> L: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

fn index(c: vec2i) -> u32 { return u32(c.x) * p.n + u32(c.y); }

struct Hit { radiance: f32, id: i32 };   // id: the pixel the ray ended in, -1 outside

fn trace(px: i32, py: i32, theta: f32) -> Hit {
  let dir = vec2f(cos(theta), sin(theta));
  let n = i32(p.n);
  // the pixel the ray starts in: a solid shows its radiance, a see-through light adds its half chord
  let i0 = index(vec2i(px, py)); let opacity0 = 1.0 - max(A[i0], 0.0);
  var radiance = S[i0] * opacity0;
  if (opacity0 < 1e-6) { radiance += S[i0] * 0.5 / max(abs(dir.x), abs(dir.y)); }
  var through = 1.0 - opacity0;
  var id = i32(i0);
  if (through < 1e-6) { return Hit(radiance, id); }
  let o = vec2f(f32(px) + 0.5, f32(py) + 0.5);
  var cell = vec2i(px, py);
  let ax = abs(dir.x); let ay = abs(dir.y);
  let sx = select(-1, 1, dir.x >= 0.0); let sy = select(-1, 1, dir.y >= 0.0);
  let tdx = select(1e20, 1.0 / ax, ax > 1e-12); let tdy = select(1e20, 1.0 / ay, ay > 1e-12);
  var tmx = select(1e20, (select(f32(cell.x), f32(cell.x + 1), dir.x >= 0.0) - o.x) / dir.x, ax > 1e-12);
  var tmy = select(1e20, (select(f32(cell.y), f32(cell.y + 1), dir.y >= 0.0) - o.y) / dir.y, ay > 1e-12);
  var tPrev = min(tmx, tmy);
  if (tmx < tmy) { cell.x += sx; tmx += tdx; } else { cell.y += sy; tmy += tdy; }
  for (var step = 0u; step < 16384u; step++) {
    if (cell.x < 0 || cell.y < 0 || cell.x >= n || cell.y >= n) { id = -1; break; }
    let tExit = min(tmx, tmy); let len = tExit - tPrev;
    let i = index(cell); let opacity = 1.0 - max(A[i], 0.0);
    if (opacity < 1e-6 || len <= 0.0) {
      radiance += through * S[i] * len;
    } else {
      let t = pow(1.0 - opacity, len);
      radiance += through * S[i] * (1.0 - t);
      through *= t;
      if (through < 1e-6) { id = i32(i); break; }
    }
    tPrev = tExit;
    if (tmx < tmy) { cell.x += sx; tmx += tdx; } else { cell.y += sy; tmy += tdy; }
  }
  return Hit(radiance, id);
}

fn hash(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }

// path tracer: stratified random directions, blended into L
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn pathTrace(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  var seed = (gid.x + gid.y * p.n) * 1099u + p.frame * 6971u;
  var sum = 0.0;
  for (var s = 0u; s < p.samples; s++) {
    seed = hash(seed);
    let theta = (f32(s) + f32(seed) / 4294967295.0) * 6.2831853 / f32(p.samples);
    sum += trace(i32(gid.x), i32(gid.y), theta).radiance;
  }
  let i = gid.x * p.n + gid.y;
  L[i] = mix(L[i], sum / f32(p.samples), 1.0 / f32(p.frame + 1u));
}

fn jumps(a: Hit, b: Hit) -> bool {
  if (a.id == b.id) { return false; }
  return abs(a.radiance - b.radiance) > max(abs(a.radiance), abs(b.radiance)) * 0.01 + 1e-6;
}
// exact reference: one chunk of chunkLen direction intervals for a band of rows
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn exact(@builtin(global_invocation_id) gid: vec3u) {
  let px = i32(gid.x); let py = i32(p.rowStart + gid.y);
  if (gid.x >= p.n || gid.y >= p.rowCount || u32(py) >= p.n) { return; }
  let dth = 6.28318530 / f32(p.m);
  let k0 = p.chunk * p.chunkLen;
  var th0 = (f32(k0) + 0.5) * dth;
  var h0 = trace(px, py, th0);
  var sum = 0.0;
  for (var k = 0u; k < p.chunkLen; k++) {
    let th1 = (f32(k0 + k + 1u) + 0.5) * dth;
    let h1 = trace(px, py, th1);
    if (jumps(h0, h1)) {
      var a = th0; var b = th1; var ha = h0; var hb = h1;
      for (var d = 0u; d < p.depth; d++) {
        let mid = 0.5 * (a + b); let hm = trace(px, py, mid);
        if (hm.id == ha.id) { a = mid; ha = hm; } else { b = mid; hb = hm; }
      }
      let ts = 0.5 * (a + b);
      sum += (a - th0) * 0.5 * (h0.radiance + ha.radiance) + (ts - a) * ha.radiance + (b - ts) * hb.radiance + (th1 - b) * 0.5 * (hb.radiance + h1.radiance);
    } else {
      sum += dth * 0.5 * (h0.radiance + h1.radiance);
    }
    th0 = th1; h0 = h1;
  }
  let i = u32(px) * p.n + u32(py);
  acc[i] += sum;
  L[i] = acc[i] * p.scale;
}
`;

export interface ExactProgress {
  m: number;
  fraction: number;
  done: boolean;
  elapsedMs: number;
  etaMs: number;
}

const same = (a: Float32Array, b: Float32Array) => a.length === b.length && a.every((v, i) => v === b[i]);

interface ExactRun {
  m: number;
  depth: number;
  chunkLen: number;
  nChunks: number;
  order: Uint32Array;
  nextChunk: number;
  nextRow: number;
  rows: number;
  workDone: number;
  elapsedMs: number;
  running: boolean;
}

export class Reference {
  readonly N: number;
  readonly device: GPUDevice;
  readonly pathTraceOutput: GPUBuffer;
  readonly exactOutput: GPUBuffer;
  pathTraceFrames = 0;
  version = 0; // bumps whenever an output changes
  #S: GPUBuffer;
  #A: GPUBuffer;
  #acc: GPUBuffer;
  #params: GPUBuffer;
  #pipelines: { pathTrace: GPUComputePipeline; exact: GPUComputePipeline };
  #layout: GPUBindGroupLayout;
  #exact: ExactRun | null = null;
  #generation = 0; // a new scene stops the exact run of the old one
  #last: { S: Float32Array; A: Float32Array } | null = null;

  constructor(device: GPUDevice, N: number) {
    this.device = device;
    this.N = N;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const buffer = (size: number, usage = storage) => device.createBuffer({ size: Math.max(size, 16), usage });
    this.#S = buffer(N * N * 4);
    this.#A = buffer(N * N * 4);
    this.pathTraceOutput = buffer(N * N * 4, storage | GPUBufferUsage.COPY_SRC);   // COPY_SRC: read back for checks
    this.exactOutput = buffer(N * N * 4, storage | GPUBufferUsage.COPY_SRC);
    this.#acc = buffer(N * N * 4);
    this.#params = buffer(256 * 2, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const module = device.createShaderModule({ code: shaderSource });
    const kinds: GPUBufferBindingLayout[] = [{ type: 'read-only-storage' }, { type: 'read-only-storage' }, { type: 'storage' }, { type: 'storage' }, { type: 'uniform' }];
    this.#layout = device.createBindGroupLayout({ entries: kinds.map((b, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: b })) });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.#layout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout, compute: { module, entryPoint } });
    this.#pipelines = { pathTrace: pipeline('pathTrace'), exact: pipeline('exact') };
  }

  /** A new scene: both references start again. The same scene again changes nothing. */
  upload(S: Float32Array, A: Float32Array) {
    if (this.#last && same(this.#last.S, S) && same(this.#last.A, A)) return;
    this.#last = { S: S.slice(), A: A.slice() };
    const q = this.device.queue;
    q.writeBuffer(this.#S, 0, S as Float32Array<ArrayBuffer>);
    q.writeBuffer(this.#A, 0, A as Float32Array<ArrayBuffer>);
    const zeros = new Float32Array(this.N * this.N);
    for (const b of [this.pathTraceOutput, this.exactOutput, this.#acc]) q.writeBuffer(b, 0, zeros);
    this.pathTraceFrames = 0;
    this.#generation++;
    this.#exact = null;
    this.version++;
  }

  #bindGroup(output: GPUBuffer, slot: number) {
    return this.device.createBindGroup({
      layout: this.#layout,
      entries: [this.#S, this.#A, output, this.#acc, this.#params].map((buffer, binding) => ({ binding, resource: binding === 4 ? { buffer, offset: slot * 256, size: 48 } : { buffer } })),
    });
  }
  #writeParams(slot: number, v: Partial<Record<'frame' | 'samples' | 'm' | 'chunkLen' | 'chunk' | 'rowStart' | 'rowCount' | 'depth' | 'scale', number>>) {
    const data = new ArrayBuffer(48), u = new Uint32Array(data), f = new Float32Array(data);
    u.set([this.N, v.frame ?? 0, v.samples ?? 0, v.m ?? 0, v.chunkLen ?? 0, v.chunk ?? 0, v.rowStart ?? 0, v.rowCount ?? 0, v.depth ?? 0]);
    f[9] = v.scale ?? 0;
    this.device.queue.writeBuffer(this.#params, slot * 256, data);
  }

  /** One more frame of path-traced samples (16 per pixel). */
  encodePathTrace(enc: GPUCommandEncoder) {
    this.#writeParams(0, { frame: this.pathTraceFrames, samples: 16 });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.#pipelines.pathTrace);
    pass.setBindGroup(0, this.#bindGroup(this.pathTraceOutput, 0));
    pass.dispatchWorkgroups(Math.ceil(this.N / WORKGROUP), Math.ceil(this.N / WORKGROUP));
    pass.end();
    this.pathTraceFrames++;
    this.version++;
  }

  /** Start (or resume) the exact reference with m directions per pixel. */
  startExact(m: number, depth = 20) {
    const e = this.#exact;
    if (e && e.m === m) {
      if (!e.running && e.nextChunk < e.nChunks) this.#runExact(e, this.#generation);
      return;
    }
    if (e) this.device.queue.writeBuffer(this.#acc, 0, new Float32Array(this.N * this.N));
    const chunkLen = 16, nChunks = Math.ceil(m / chunkLen), order = new Uint32Array(nChunks);
    // chunks in a scrambled order, so the partial result covers all directions evenly
    let step = Math.round(nChunks * 0.618034) | 1;
    const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
    while (gcd(step, nChunks) !== 1) step += 2;
    for (let i = 0; i < nChunks; i++) order[i] = (i * step) % nChunks;
    this.#exact = { m, depth, chunkLen, nChunks, order, nextChunk: 0, nextRow: 0, rows: Math.max(1, Math.round(this.N / 16)), workDone: 0, elapsedMs: 0, running: false };
    this.#runExact(this.#exact, this.#generation);
  }
  pauseExact() {
    if (this.#exact) this.#exact.running = false;
  }
  get exactProgress(): ExactProgress | null {
    const e = this.#exact;
    if (!e) return null;
    const total = this.N * this.N * e.m, rate = e.elapsedMs > 0 ? e.workDone / e.elapsedMs : 0;
    return { m: e.m, fraction: e.workDone / total, done: e.nextChunk >= e.nChunks, elapsedMs: e.elapsedMs, etaMs: rate > 0 ? (total - e.workDone) / rate : NaN };
  }

  async #runExact(e: ExactRun, generation: number) {
    const TARGET_MS = 40;
    e.running = true;
    while (e.running && e.nextChunk < e.nChunks && generation === this.#generation && this.#exact === e) {
      const rows = Math.min(e.rows, this.N - e.nextRow), t0 = performance.now();
      this.#writeParams(1, { m: e.m, chunkLen: e.chunkLen, chunk: e.order[e.nextChunk], rowStart: e.nextRow, rowCount: rows, depth: e.depth, scale: e.nChunks / (e.nextChunk + 1) / (2 * Math.PI) });
      const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(this.#pipelines.exact);
      pass.setBindGroup(0, this.#bindGroup(this.exactOutput, 1));
      pass.dispatchWorkgroups(Math.ceil(this.N / WORKGROUP), Math.ceil(rows / WORKGROUP));
      pass.end();
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const dt = performance.now() - t0;
      e.elapsedMs += dt;
      e.workDone += rows * this.N * e.chunkLen;
      e.nextRow += rows;
      if (e.nextRow >= this.N) ((e.nextRow = 0), e.nextChunk++);
      // keep each submit near the time budget
      e.rows = Math.max(1, Math.min(this.N, Math.round(e.rows * Math.min(2, Math.max(0.5, TARGET_MS / Math.max(1, dt))))));
      this.version++;
    }
    e.running = false;
  }

  destroy() {
    this.#generation++;
    for (const b of [this.#S, this.#A, this.pathTraceOutput, this.exactOutput, this.#acc, this.#params]) b.destroy();
  }
}
