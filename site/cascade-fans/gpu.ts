// The fan's solve on the GPU: ./fans.ts's solveCone, one cone after another, each a frustum-like run of passes in the
// cone's own frame. A frame point (X, Y) is the pixel (X − off + xmin) u + (Y − padLo + ymin) w, so the shader reads
// the scene through that integer map and never stores it. Per cone:
//   inject:   level 1, a thread per ray (X, Y, d): T = f(S) f(E), R = e(E) + f(E) e(S);
//   up:       level L from L / 2, a thread per ray (b, y, D): the sum over its pairs of the join;
//   down:     M of level L from M of 2L, a thread per ray, gathering the merge's scatter (the top's M is 0);
//   readout:  a thread per pixel, the transpose of the injection, added to the fan's sum.
// Then one pass masks the sum by the free field and adds the luminance. The pairs, order of the sums and the cone
// weights (computed here in f64, as the CPU has them) are the CPU's, so the two agree to float precision.
// One change of layout: level L stores only the start rows whose rays can touch the image, rows [padLo − hi L,
// padLo + H − lo L) for an image H rows high, not the CPU's Yn = H + (hi − lo) Xn. A ray that touches no pixel has
// T = 0 and R = 0, and its M reaches only rays that touch none either, so dropping them changes no value.
// A record is (R rgb, T); M is (rgb, 0). Arrays are [x][y] flattened as x N + y, y up.

import { coneWeight, frameGeometry, type Cone } from './fans';
import { rgba, staging } from '../_shared/upload';

const WORKGROUP = 256;
const SLOT = 256; // uniform slot per dispatch

const shader = (N: number) => /* wgsl */ `
const N: i32 = ${N};
struct Params {
  a: vec4i,   // u, w
  b: vec4i,   // lo, hi, edge, Xn
  c: vec4i,   // Yn, off, xmin, ymin
  d: vec4i,   // padLo, L (the level written; the readout's 1), count, top (down: the parent is the top, M = 0)
  e: vec4i,   // the cone weights of the parent level start here (down); H, the image's rows
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> scene: array<vec4f>;    // (emission rgb, 1 solid or lit / 0 free)
@group(0) @binding(2) var<storage, read> src: array<vec4f>;      // up: the halves; down: the level's records; readout: M of level 1; final: the sum
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(4) var<storage, read> parent: array<vec4f>;   // down: M of level 2L
@group(0) @binding(5) var<storage, read> weights: array<f32>;    // cone weights, per level P, D in [lo P, hi P]

fn lo() -> i32 { return p.b.x; }
fn nd(L: i32) -> i32 { return (p.b.y - p.b.x) * L + 1; }
// level L's start rows: from padLo − hi L, H + (hi − lo) L of them
fn first(L: i32) -> i32 { return p.d.x - p.b.y * L; }
fn rows(L: i32) -> i32 { return p.e.y + (p.b.y - p.b.x) * L; }
fn inside(L: i32, y: i32) -> bool { return y >= first(L) && y < first(L) + rows(L); }
fn at(L: i32, b: i32, y: i32, D: i32) -> i32 { return (b * rows(L) + y - first(L)) * nd(L) + D - lo() * L; }
// the pixel at frame point (X, Y), or −1
fn pixel(X: i32, Y: i32) -> i32 {
  if (Y < 0 || Y >= p.c.x || X < 0 || X > p.b.w) { return -1; }
  let x = X - p.c.y + p.c.z; let y = Y - p.d.x + p.c.w;
  let i = x * p.a.x + y * p.a.z; let j = x * p.a.y + y * p.a.w;
  if (i < 0 || j < 0 || i >= N || j >= N) { return -1; }
  return i * N + j;
}
fn freeAt(k: i32) -> f32 {
  if (k < 0) { return 0.0; }
  return select(1.0, 0.0, scene[k].a > 0.5);
}
fn lightAt(k: i32) -> vec3f {
  if (k < 0) { return vec3f(0.0); }
  return scene[k].rgb;
}

fn thread(id: vec3u, g: vec3u) -> i32 { return i32(id.x + id.y * g.x * ${WORKGROUP}u); }

@compute @workgroup_size(${WORKGROUP}) fn inject(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let i = thread(id, g);
  if (i >= p.d.z) { return; }
  let n = nd(1); let d = i % n + lo(); let Y = (i / n) % rows(1) + first(1); let X = i / (n * rows(1));
  let S = pixel(X, Y); let E = pixel(X + 1 - p.b.z, Y + d);
  let fS = freeAt(S); let fE = freeAt(E);
  dst[i] = vec4f(lightAt(E) + fE * lightAt(S), fS * fE);
}

// a half off its level's stored rows touches no pixel: T = R = 0
fn half(h: i32, b: i32, y: i32, D: i32) -> vec4f {
  if (!inside(h, y)) { return vec4f(0.0); }
  return src[at(h, b, y, D)];
}
@compute @workgroup_size(${WORKGROUP}) fn up(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let i = thread(id, g);
  if (i >= p.d.z) { return; }
  let L = p.d.y; let h = L / 2;
  let n = nd(L); let D = i % n + lo() * L; let y = (i / n) % rows(L) + first(L); let b = i / (n * rows(L));
  let a0 = D >> 1u;
  let odd = (D & 1) != 0;
  var out = vec4f(0.0);
  for (var k = 0; k < select(1, 2, odd); k++) {
    let a = select(a0, a0 + select(0, 1, k == 1), odd); let bb = select(a0, a0 + select(1, 0, k == 1), odd);
    let wt = select(1.0, 0.5, odd);
    let A = half(h, 2 * b, y, a); let B = half(h, 2 * b + 1, y + a, bb);
    out = vec4f(out.rgb + wt * (B.rgb + B.w * A.rgb), out.w + wt * A.w * B.w);
  }
  dst[i] = out;
}

// the merge's scatter, gathered at child (c, y, d) of level L: a first half takes its parents' M; a second half the M
// let through its sibling plus its sibling's light at the parent's cone weight. Parents in the CPU's order, D rising.
@compute @workgroup_size(${WORKGROUP}) fn down(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let i = thread(id, g);
  if (i >= p.d.z) { return; }
  let L = p.d.y; let P = 2 * L; let top = p.d.w != 0;
  let n = nd(L); let d = i % n + lo() * L; let y = (i / n) % rows(L) + first(L); let c = i / (n * rows(L)); let b = c >> 1u;
  let Dlo = max(2 * d - 1, p.b.x * P); let Dhi = min(2 * d + 1, p.b.y * P);
  var m = vec3f(0.0);
  for (var D = Dlo; D <= Dhi; D++) {
    let wt = select(1.0, 0.5, (D & 1) != 0);
    if ((c & 1) == 0) {
      if (!top && inside(P, y)) { m += wt * parent[at(P, b, y, D)].rgb; }
      continue;
    }
    // a sibling or parent off the stored rows touches no pixel, nor then does this ray
    let a = D - d; let ys = y - a;
    if (!inside(L, ys)) { continue; }
    let A = src[at(L, 2 * b, ys, a)];
    var F = vec3f(0.0);
    if (!top && inside(P, ys)) { F = parent[at(P, b, ys, D)].rgb; }
    m += wt * (A.w * F + weights[p.e.x + D - p.b.x * P] * A.rgb);
  }
  dst[i] = vec4f(m, 0.0);
}

// the transpose of the injection: half of M of the level-1 rays starting at the pixel, and half f(S) M of those ending
@compute @workgroup_size(${WORKGROUP}) fn readout(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let q = thread(id, g);
  if (q >= N * N) { return; }
  let i = q / N; let j = q % N; let Xn = p.b.w;
  let X = (p.a.w * i - p.a.z * j) - p.c.z + p.c.y; let Y = (-p.a.y * i + p.a.x * j) - p.c.w + p.d.x;
  var sum = vec3f(0.0);
  for (var d = p.b.x; d <= p.b.y; d++) {
    if (X < Xn) { sum += 0.5 * src[at(1, X, Y, d)].rgb; }
    let XS = X - 1 + p.b.z; let YS = Y - d;
    if (XS >= 0 && XS < Xn && inside(1, YS)) { sum += 0.5 * freeAt(pixel(XS, YS)) * src[at(1, XS, YS, d)].rgb; }
  }
  dst[q] = vec4f(dst[q].rgb + sum, 0.0);
}

@compute @workgroup_size(${WORKGROUP}) fn finish(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let q = thread(id, g);
  if (q >= N * N) { return; }
  let v = src[q].rgb * freeAt(q);
  dst[q] = vec4f(v, dot(v, vec3f(0.2126, 0.7152, 0.0722)));
}`;

interface Dispatch { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; count: number }

export interface FanScene {
  emission: Float32Array;
  solid: ArrayLike<number>;
}

export class FanEngine {
  readonly device: GPUDevice;
  readonly N: number;
  readonly fan: string;
  readonly sceneBuffer: GPUBuffer;
  readonly outputBuffer: GPUBuffer;
  #sum: GPUBuffer;
  #buffers: GPUBuffer[] = [];
  #program: Dispatch[] = [];

  /** fan: its id, for the caller's bookkeeping; cones: its cones */
  constructor(device: GPUDevice, N: number, fan: string, cones: Cone[]) {
    this.device = device;
    this.N = N;
    this.fan = fan;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (size: number, use = usage) => {
      const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage: use });
      this.#buffers.push(b);
      return b;
    };
    this.sceneBuffer = buffer(N * N * 16);
    this.outputBuffer = buffer(N * N * 16);
    this.#sum = buffer(N * N * 16);
    const none = buffer(16);

    // per cone: its frame, and the sizes of its levels L = 1 … Xn / 2; the level buffers are shared, each as large as
    // the largest cone needs
    const frames = cones.map((c) => ({ c, ...frameGeometry(N, c) }));
    const rowsOf = (f: (typeof frames)[number], L: number) => f.Yn - (f.c.hi - f.c.lo) * (f.Xn - L);
    const size = (f: (typeof frames)[number], L: number) => (f.Xn / L) * rowsOf(f, L) * ((f.c.hi - f.c.lo) * L + 1);
    const levels: number[] = [];
    let largest = 0;
    for (const f of frames)
      for (let k = 0; 2 ** k <= f.Xn / 2; k++) {
        levels[k] = Math.max(levels[k] ?? 0, size(f, 2 ** k));
        largest = Math.max(largest, size(f, 2 ** k));
      }
    const level = levels.map((n) => buffer(n * 16));
    const M = [buffer(largest * 16), buffer(largest * 16)];

    // cone weights, per cone per parent level P = 2 … Xn
    const weights: number[] = [], weightAt = frames.map((f) => {
      const at = new Map<number, number>();
      for (let P = 2; P <= f.Xn; P *= 2) {
        at.set(P, weights.length);
        for (let D = f.c.lo * P; D <= f.c.hi * P; D++) weights.push(coneWeight(f.c, P, D));
      }
      return at;
    });
    const weightBuffer = buffer(weights.length * 4);
    device.queue.writeBuffer(weightBuffer, 0, new Float32Array(weights));

    const module = device.createShaderModule({ code: shader(N) });
    const entries = [0, 1, 2, 3, 4, 5].map((binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: (binding === 0 ? 'uniform' : binding === 3 ? 'storage' : 'read-only-storage') as GPUBufferBindingType },
    }));
    const bindGroupLayout = device.createBindGroupLayout({ entries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const [inject, up, down, readout, finish] = ['inject', 'up', 'down', 'readout', 'finish'].map((entryPoint) => device.createComputePipeline({ layout, compute: { module, entryPoint } }));

    const steps: { pipeline: GPUComputePipeline; params: Int32Array; src: GPUBuffer; dst: GPUBuffer; parent: GPUBuffer }[] = [];
    for (const [ci, f] of frames.entries()) {
      const { c, Xn, Yn, off, padLo, xmin, ymin } = f;
      const params = (L: number, count: number, top = 0, weight = 0) => new Int32Array([
        c.u[0], c.u[1], c.w[0], c.w[1],
        c.lo, c.hi, c.edge ? 1 : 0, Xn,
        Yn, off, xmin, ymin,
        padLo, L, count, top,
        weight, Yn - (c.hi - c.lo) * Xn, 0, 0,
      ]);
      steps.push({ pipeline: inject, params: params(1, size(f, 1)), src: none, dst: level[0], parent: none });
      for (let k = 1; 2 ** k <= Xn / 2; k++) steps.push({ pipeline: up, params: params(2 ** k, size(f, 2 ** k)), src: level[k - 1], dst: level[k], parent: none });
      let from = none, flip = 0;
      for (let k = Math.log2(Xn) - 1; k >= 0; k--) {
        const L = 2 ** k, to = M[flip];
        steps.push({ pipeline: down, params: params(L, size(f, L), from === none ? 1 : 0, weightAt[ci].get(2 * L)!), src: level[k], dst: to, parent: from });
        from = to;
        flip ^= 1;
      }
      steps.push({ pipeline: readout, params: params(1, N * N), src: from, dst: this.#sum, parent: none });
    }
    steps.push({ pipeline: finish, params: new Int32Array(20), src: this.#sum, dst: this.outputBuffer, parent: none });

    const uniforms = buffer(steps.length * SLOT, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    for (const [k, s] of steps.entries()) {
      device.queue.writeBuffer(uniforms, k * SLOT, s.params);
      const resources = [{ buffer: uniforms, offset: k * SLOT, size: 80 }, { buffer: this.sceneBuffer }, { buffer: s.src }, { buffer: s.dst }, { buffer: s.parent }, { buffer: weightBuffer }];
      const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: resources.map((resource, binding) => ({ binding, resource })) });
      this.#program.push({ pipeline: s.pipeline, bindGroup, count: s.params[14] || N * N });
    }
  }

  get dispatches() {
    return this.#program.length;
  }

  get bytes() {
    return this.#buffers.reduce((s, b) => s + b.size, 0);
  }

  #staging = staging();
  upload({ emission, solid }: FanScene) {
    this.device.queue.writeBuffer(this.sceneBuffer, 0, rgba(this.#staging(this.N * this.N * 4), { emission, solid }, 1, 0));
  }

  encodeAll(encoder: GPUCommandEncoder) {
    encoder.clearBuffer(this.#sum);
    const pass = encoder.beginComputePass();
    for (const d of this.#program) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      const groups = Math.ceil(d.count / WORKGROUP);
      pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
    }
    pass.end();
  }

  /** the fluence (rgb, luminance) per pixel, [x N + y], as ./fans.ts's solveFan */
  async solve(scene: FanScene): Promise<Float32Array> {
    this.upload(scene);
    const bytes = this.N * this.N * 16;
    const out = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    this.encodeAll(encoder);
    encoder.copyBufferToBuffer(this.outputBuffer, 0, out, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await out.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(out.getMappedRange().slice(0));
    out.destroy();
    return data;
  }

  destroy() {
    for (const b of this.#buffers) b.destroy();
  }
}
