// Reciprocal HRC in the dual column form. One octant at a time (4 frustums, each with steps 0 / +1 and its mirror
// image across rows): scene image → up passes (butterflies) → down passes (their transpose) → fluence.
//
// Atom p is a dual pixel: the right half of column p - 1, then the left half of column p. The tree spans 2N atoms; the
// scene sits at its start (frustums 0 and 2) or its end (1 and 3), and the rest is free and dark: the passes run only
// over the blocks that touch the scene's atoms, and read a block wholly outside them as free. Level L joins L atoms
// with rises s in [0, L), stored as one image per level: column block L + s, by end row (rays) or start row (merged).
// A parent (2L, s) is its first half (L, a), a step of j rows, its second half (L, b), a + j + b = s, mixed by the
// linear rule; the down pass reads the same terms backwards. Rays are f16 (rgb, transmittance); merged light RGB9E5.

import type { Profiler } from '../_shared/profiler';
import { rgba, staging } from '../_shared/upload';

/** World arrays [x * N + y], y up: emission rgb (3 per pixel) and solid (1 opaque). A pixel with emission is a light. */
export interface DualScene {
  emission: Float32Array;
  solid: Float32Array | Uint8Array;
}

const WORKGROUP = 256;
const MAX_GROUPS_X = 32768;
const PARAM_STRIDE = 256;

/** The line measure between unit windows g apart at rise s (rise 0 halved: both octants hold it). Per child level L
 * (g = 2L - 1), rises 0 … 2L - 1, at offset 2L - 2. In double precision: it is a second difference. */
function croftonTable(N: number) {
  const G = (t: number) => {
    const a = Math.min(1, Math.abs(t));
    return a * Math.atan(a) - 0.5 * Math.log(1 + a * a) + (Math.PI / 4) * (Math.abs(t) - a);
  };
  const out: number[] = [];
  for (let L = 1; L <= N; L *= 2)
    for (let s = 0; s < 2 * L; s++) {
      const g = 2 * L - 1, w = g * (G((s + 1) / g) - 2 * G(s / g) + G((s - 1) / g));
      out.push(s === 0 ? w / 2 : w);
    }
  return new Float32Array(out);
}

const shader = (N: number, wu: boolean) => /* wgsl */ `
const N: i32 = ${N};
const WU = ${wu};
struct Params { L: i32, frustum: i32, flip: i32, off: i32, top: i32, count: u32, first: i32 };   // first: the pass's first column
@group(0) @binding(0) var<storage, read> scene: array<vec4f>;       // world [x * N + y]: (radiance rgb, transmittance)
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> crofton: array<f32>;
@group(0) @binding(3) var<storage, read_write> rays: array<vec2u>;  // levels 2 … N, each 2N columns × N end rows
@group(0) @binding(4) var<storage, read> gIn: array<u32>;           // merged light, 2N columns × (N + 1) start rows
@group(0) @binding(5) var<storage, read_write> gOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> fluence: array<vec4f>;  // world [x * N + y]: (rgb, luminance)

struct Ray { radiance: vec3f, transmittance: f32 };
const OPAQUE = Ray(vec3f(0.0), 0.0);
fn chain(first: Ray, next: Ray) -> Ray {
  return Ray(next.radiance + next.transmittance * first.radiance, first.transmittance * next.transmittance);
}
fn packRay(r: Ray) -> vec2u { return vec2u(pack2x16float(r.radiance.rg), pack2x16float(vec2f(r.radiance.b, r.transmittance))); }
fn unpackRay(p: vec2u) -> Ray { let bt = unpack2x16float(p.y); return Ray(vec3f(unpack2x16float(p.x), bt.x), bt.y); }
fn packG(m: vec3f) -> u32 {
  let e = clamp(i32((bitcast<u32>(max(m.r, max(m.g, m.b))) >> 23u) & 255u) - 111, 0, 31);
  let q = vec3u(min(max(m, vec3f(0.0)) * bitcast<f32>(u32(151 - e) << 23u) + 0.5, vec3f(511.0)));
  return q.r | (q.g << 9u) | (q.b << 18u) | (u32(e) << 27u);
}
fn unpackG(p: u32) -> vec3f { return vec3f(vec3u(p, p >> 9u, p >> 18u) & vec3u(511u)) * bitcast<f32>(((p >> 27u) + 103u) << 23u); }

// The octant's frame: frustum column c, row r → world pixel. Frustums travel +x, -x, +y, -y; flip mirrors the rows.
fn world(c: i32, row: i32) -> i32 {
  let r = select(row, N - 1 - row, params.flip == 1);
  var frame = array<vec2i, 4>(vec2i(c, r), vec2i(N - 1 - c, r), vec2i(r, c), vec2i(r, N - 1 - c));
  let p = frame[params.frustum];
  return p.x * N + p.y;
}
// Tree column q holds frustum column q - 1 - off: off the scene's columns free, off its rows opaque.
fn pixel(q: i32, row: i32) -> Ray {
  let c = q - 1 - params.off;
  if (c < 0 || c >= N) { return Ray(vec3f(0.0), 1.0); }
  if (row < 0 || row >= N) { return OPAQUE; }
  let p = scene[world(c, row)];
  return Ray(p.rgb, p.w);
}
// Level L's ray at column block L + s from a start row. Level 1 is the atom; a block outside the scene's atoms
// off … off + N is free; stored rays end inside the rows.
fn ray(L: i32, block: i32, s: i32, start: i32) -> Ray {
  if (L == 1) { return chain(pixel(block, start), pixel(block + 1, start)); }
  if (block * L > params.off + N || (block + 1) * L <= params.off) { return Ray(vec3f(0.0), 1.0); }
  let end = start + s;
  if (end < 0 || end >= N) { return OPAQUE; }
  return unpackRay(rays[(firstTrailingBit(u32(L)) - 1u) * u32(2 * N * N) + u32((block * L + s) * N + end)]);
}
fn merged(column: i32, start: i32) -> vec3f {
  if (params.top == 1 || start < 0 || start > N) { return vec3f(0.0); }
  return unpackG(gIn[column * (N + 1) + start]);
}

// The linear rule: parent (2L, s) as three terms (a, j, b) whose mean junction row is the centre line's. Terms that
// leave [0, L) are dropped and the rest renormalised.
struct Term { a: i32, j: i32, b: i32, w: f32 };
fn split(L: i32, s: i32) -> array<Term, 3> {
  let k = s / 2; let g = f32(2 * L - 1);
  var t: array<Term, 3>;
  if (s % 2 == 1) {
    let e = (f32(k) + 0.5) / g;
    t = array<Term, 3>(Term(k, 1, k, 2.0 * e), Term(k + 1, 0, k, 0.5 - e), Term(k, 0, k + 1, 0.5 - e));
  } else {
    let d = f32(k) / g;
    t = array<Term, 3>(Term(k, 0, k, 1.0 - 2.0 * d), Term(k - 1, 1, k, d), Term(k, 1, k - 1, d));
  }
  var total = 0.0;
  for (var i = 0; i < 3; i++) {
    if (t[i].a < 0 || t[i].a >= L || t[i].b < 0 || t[i].b >= L || t[i].w <= 0.0) { t[i].w = 0.0; }
    total += t[i].w;
  }
  for (var i = 0; i < 3; i++) { t[i].w /= total; }
  return t;
}

fn threadIndex(id: vec3u, groups: vec3u) -> u32 { return id.x + id.y * groups.x * ${WORKGROUP}u; }

// Up: level 2L from level L, one thread per (column, end row).
@compute @workgroup_size(${WORKGROUP})
fn up(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let L = params.L; let column = params.first + i32(i) / N; let end = i32(i) % N;
  let block = column / (2 * L); let s = column % (2 * L); let y = end - s;
  let t = split(L, s);
  var sum = Ray(vec3f(0.0), 0.0);
  for (var k = 0; k < 3; k++) {
    if (t[k].w == 0.0) { continue; }
    let p = chain(ray(L, 2 * block, t[k].a, y), ray(L, 2 * block + 1, t[k].b, y + t[k].a + t[k].j));
    sum = Ray(sum.radiance + t[k].w * p.radiance, sum.transmittance + t[k].w * p.transmittance);
  }
  rays[(firstTrailingBit(u32(2 * L)) - 1u) * u32(2 * N * N) + u32(column * N + end)] = packRay(sum);
}

// Down: level L's merged light from level 2L's, one thread per (column, start row). A first half takes a term's share
// of its parent's light; a second half takes it through the first half, plus the first half's own light times the
// term's share of the parent's line measure. Wu: with three terms, the parent's light leans by u = ¼ r toward the term
// whose junction row is higher (and away from the lower), r = (G[y+1] - G[y-1]) / (G[y-1] + G[y] + G[y+1]), clipped
// so no weight goes negative.
@compute @workgroup_size(${WORKGROUP})
fn down(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let L = params.L; let column = params.first + i32(i) / (N + 1); let y = i32(i) % (N + 1);
  let block = column / L; let c = column % L; let second = block % 2 == 1;
  var sum = vec3f(0.0);
  for (var s = max(0, 2 * c - 1); s <= min(2 * L - 1, 2 * c + 2); s++) {
    let t = split(L, s); let k = s / 2; let odd = s % 2 == 1;
    for (var n = 0; n < 3; n++) {
      if (t[n].w == 0.0 || select(t[n].a, t[n].b, second) != c) { continue; }
      let start = select(y, y - t[n].a - t[n].j, second);
      let parent = (block / 2) * 2 * L + s;
      let G = merged(parent, start);
      var w = vec3f(t[n].w);
      if (WU && t[0].w > 0.0 && t[1].w > 0.0 && t[2].w > 0.0) {
        let below = merged(parent, start - 1); let above = merged(parent, start + 1); let total = below + G + above;
        // the higher term is (k + 1, 0, k) (odd s) or (k, 1, k - 1) (even s); the lower (k, 0, k + 1) or (k - 1, 1, k)
        let room = min(t[1].w, t[2].w);
        let u = clamp(0.25 * select(vec3f(0.0), (above - below) / total, total > vec3f(0.0)), vec3f(-room), vec3f(room));
        let hi = select(t[n].a + t[n].j > k, t[n].a == k + 1, odd);
        let lo = select(t[n].a < k, t[n].j == 0 && t[n].b == k + 1, odd);
        w += select(vec3f(0.0), u, hi) - select(vec3f(0.0), u, lo);
      }
      if (second) {
        let A = ray(L, block - 1, t[n].a, start);
        sum += A.transmittance * w * G + t[n].w * crofton[2 * L - 2 + s] * A.radiance;
      } else {
        sum += w * G;
      }
    }
  }
  gOut[column * (N + 1) + y] = packG(sum);
}

// A pixel's light per octant: the mean of its left edge (atom p's start light, through the right half of the column
// before, plus that half's own light over the octant) and its centre (atom p + 1's start light). Octants add up.
@compute @workgroup_size(${WORKGROUP})
fn deposit(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let c = i32(i) / N; let row = i32(i) % N; let p = params.off + c;
  let before = pixel(p, row);
  let right = select(vec3f(0.0), merged(p + 1, row), p + 1 < 2 * N);
  let value = 0.5 * (before.transmittance * merged(p, row) + 0.78539816 * before.radiance + right);
  let at = world(c, row);
  fluence[at] += vec4f(value, dot(value, vec3f(0.2126, 0.7152, 0.0722)));
}

// Solid pixels: lights show 2π E, walls 0.
@compute @workgroup_size(${WORKGROUP})
fn solids(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let p = scene[i];
  if (p.w == 0.0) { fluence[i] = vec4f(6.2831853 * p.rgb, dot(6.2831853 * p.rgb, vec3f(0.2126, 0.7152, 0.0722))); }
}
`;

type Stage = 'up' | 'down' | 'deposit' | 'solids';
interface Dispatch { stage: Stage; label: string; offset: number; count: number; group: 0 | 1 }

export class DualHRC {
  readonly device: GPUDevice;
  readonly N: number;
  readonly wu: boolean;
  profiler: Profiler | null = null;
  /** world [x * N + y]: (radiance rgb, transmittance) */
  readonly sceneBuffer: GPUBuffer;
  /** world [x * N + y]: (fluence rgb, luminance) */
  readonly outputBuffer: GPUBuffer;
  #buffers: GPUBuffer[];
  #pipelines: Record<Stage, GPUComputePipeline>;
  #groups: [GPUBindGroup, GPUBindGroup];
  #program: Dispatch[] = [];

  constructor(device: GPUDevice, N: number, { wu = true }: { wu?: boolean } = {}) {
    if (N < 4 || (N & (N - 1)) !== 0) throw new Error('DualHRC: N must be a power of 2, at least 4');
    this.device = device;
    this.N = N;
    this.wu = wu;
    const levels = Math.log2(N);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, size: number, use = usage) => device.createBuffer({ label: `dual HRC ${label}`, size, usage: use });
    this.sceneBuffer = buffer('scene', N * N * 16);
    this.outputBuffer = buffer('fluence', N * N * 16);
    const crofton = croftonTable(N), cones = buffer('crofton', crofton.byteLength);
    device.queue.writeBuffer(cones, 0, crofton);
    const rays = buffer('rays', levels * 2 * N * N * 8);
    const g = [buffer('merged a', 2 * N * (N + 1) * 4), buffer('merged b', 2 * N * (N + 1) * 4)];

    // the program: per octant, up L = 1 … N / 2 (building 2L), down L = N … 1 (ping-pong), deposit; then the solids
    const params: number[][] = [];
    const add = (stage: Stage, label: string, count: number, p: number[], group: 0 | 1 = 0, first = 0) => {
      this.#program.push({ stage, label, count, group, offset: params.length * PARAM_STRIDE });
      params.push([...p, count, first]);
    };
    // the columns of the blocks of `size` atoms that touch atoms off … off + N
    const span = (off: number, size: number) => {
      const first = Math.floor(off / size) * size, last = Math.min(2 * N, (Math.floor((off + N) / size) + 1) * size);
      return [first, last - first];
    };
    for (let frustum = 0; frustum < 4; frustum++)
      for (const flip of [0, 1]) {
        const off = frustum % 2 ? N - 1 : 0;
        for (let L = 1; L < N; L *= 2) {
          const [first, columns] = span(off, 2 * L);
          add('up', `up ${2 * L}`, columns * N, [L, frustum, flip, off, 0], 0, first);
        }
        let group: 0 | 1 = 0;   // group 0 reads a and writes b; 1 the reverse
        for (let L = N; L >= 1; L /= 2, group = (1 - group) as 0 | 1) {
          const [first, columns] = span(off, L);
          add('down', `down ${L}`, columns * (N + 1), [L, frustum, flip, off, L === N ? 1 : 0], group, first);
        }
        add('deposit', 'deposit', N * N, [1, frustum, flip, off, 0], group);
      }
    add('solids', 'solids', N * N, [0, 0, 0, 0, 0]);
    const data = new ArrayBuffer(params.length * PARAM_STRIDE);
    params.forEach((p, k) => new Int32Array(data, k * PARAM_STRIDE, 7).set(p));
    const uniforms = buffer('params', data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uniforms, 0, data);
    this.#buffers = [this.sceneBuffer, this.outputBuffer, cones, rays, ...g, uniforms];

    const module = device.createShaderModule({ label: 'dual HRC', code: shader(N, wu) });
    const kinds: GPUBufferBindingType[] = ['read-only-storage', 'uniform', 'read-only-storage', 'storage', 'read-only-storage', 'storage', 'storage'];
    const layout = device.createBindGroupLayout({
      entries: kinds.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, hasDynamicOffset: type === 'uniform' } })),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint: Stage) => device.createComputePipeline({ label: `dual HRC ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } });
    this.#pipelines = { up: pipeline('up'), down: pipeline('down'), deposit: pipeline('deposit'), solids: pipeline('solids') };
    const group = (gIn: GPUBuffer, gOut: GPUBuffer) => device.createBindGroup({
      layout,
      entries: [this.sceneBuffer, uniforms, cones, rays, gIn, gOut, this.outputBuffer].map((b, binding) => ({ binding, resource: binding === 1 ? { buffer: b, size: 32 } : { buffer: b } })),
    });
    this.#groups = [group(g[0], g[1]), group(g[1], g[0])];
  }

  get dispatches() {
    return this.#program.length;
  }

  get bytes() {
    return this.#buffers.reduce((s, b) => s + b.size, 0);
  }

  destroy() {
    for (const b of this.#buffers) b.destroy();
  }

  #staging = staging();
  upload({ emission, solid }: DualScene) {
    this.device.queue.writeBuffer(this.sceneBuffer, 0, rgba(this.#staging(this.N * this.N * 4), { emission, solid }, 0, 1, true));
  }

  /** consecutive dispatches of one stage share a compute pass */
  encodeAll(encoder: GPUCommandEncoder) {
    encoder.clearBuffer(this.outputBuffer);
    let pass: GPUComputePassEncoder | null = null, stage = '';
    for (const d of this.#program) {
      if (d.stage !== stage) {
        pass?.end();
        stage = d.stage;
        pass = this.profiler ? this.profiler.computePass(encoder, stage) : encoder.beginComputePass();
      }
      this.profiler?.name(pass!, d.label);
      pass!.setPipeline(this.#pipelines[d.stage]);
      pass!.setBindGroup(0, this.#groups[d.group], [d.offset]);
      const groups = Math.ceil(d.count / WORKGROUP);
      pass!.dispatchWorkgroups(Math.min(groups, MAX_GROUPS_X), Math.ceil(groups / MAX_GROUPS_X));
    }
    pass?.end();
  }

  /** upload, solve once and read the fluence back: world [x * N + y], (rgb, luminance) */
  async solve(scene: DualScene): Promise<Float32Array> {
    this.upload(scene);
    const bytes = this.N * this.N * 16;
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    this.encodeAll(encoder);
    encoder.copyBufferToBuffer(this.outputBuffer, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }
}
