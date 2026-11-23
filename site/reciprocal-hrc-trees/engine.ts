// Reciprocal HRC on any column tree (./trees.ts). Per frustum: extensions shortest first, merges longest first (each
// ray gathers from the parents it is part of), then the pixels pass. On the dyadic tree this is the Reciprocal HRC
// page's engine without its options and memory reuse.
//
// A node of length L ≥ 2 owns (2L + 1) N slots in both buffers: rays (f16 rgb and transmittance, by end row) and merged
// light (RGB9E5, by start row). The four frustums reuse the buffers in turn.

import type { Profiler, Traffic } from '../_shared/profiler';
import { NODE_FIELDS, frustumLayouts, type Layout, type Tree } from './trees';
import { rgba, staging, transpose } from '../_shared/upload';

/** World arrays [x * N + y], y up: emission rgb (3 per pixel) and solid. A pixel with emission is a light. */
export interface TreeScene {
  emission: Float32Array;
  solid: Float32Array | Uint8Array;
}

const WORKGROUP = 256;
const MAX_GROUPS_X = 32768;
const PARAM_STRIDE = 256;
const RAY_BYTES = 8;
const MERGED_BYTES = 4;
const PIXEL_BYTES = 16;

const shader = (N: number, root: number, wu: boolean) => /* wgsl */ `
const N: i32 = ${N};
const ROOT: i32 = ${root};  // frame width; the scene is at its start (frustums 0, 2) or end (1, 3)
const WU = ${wu};

struct Params {
  length: i32, cut: i32,
  nodeStart: i32, count: u32,
  frustum: i32, base: i32,            // base: the level's first slot
  sceneStart: i32, leafStart: i32,    // the scene's first column, and its 1-px node
};
struct Node {
  column: i32, length: i32, base: i32,
  first: i32, second: i32,            // -1: off the scene
  parent: i32, parentLength: i32,     // -1: the root
  cut: i32, piece: i32, sibling: i32, // the parent's cut, which piece this is, the other piece
};
@group(0) @binding(0) var<storage, read> scene: array<vec4f>;       // [axis][column][row]: (radiance rgb, transmittance)
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read_write> rays: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> merged: array<u32>;
@group(0) @binding(5) var<storage, read_write> axisFluence: array<vec4f>;  // [axis][column][row]
@group(0) @binding(6) var<storage, read_write> fluence: array<vec4f>;      // world [x * N + y]: (rgb, luminance)

fn threadIndex(id: vec3u, groups: vec3u) -> u32 { return id.x + id.y * groups.x * ${WORKGROUP}u; }
fn floorDiv(a: i32, b: i32) -> i32 { return select(a / b, (a - b + 1) / b, a < 0); }   // b > 0

// radiance arriving at the ray's end, and its transmittance
struct Ray { radiance: vec3f, transmittance: f32 };
const OPAQUE = Ray(vec3f(0.0), 0.0);
fn extend(first: Ray, next: Ray) -> Ray {
  return Ray(next.radiance + next.transmittance * first.radiance, first.transmittance * next.transmittance);
}
// off the scene: opaque and dark
fn halfPixel(column: i32, row: i32) -> Ray {
  let x = select(column, ROOT - 1 - column, (params.frustum & 1) == 1);
  if (row < 0 || row >= N || x < 0 || x >= N) { return OPAQUE; }
  let p = scene[((params.frustum >> 1) * N + x) * N + row];
  return Ray(p.rgb, p.w);
}

fn packRay(ray: Ray) -> vec2u {
  return vec2u(pack2x16float(ray.radiance.rg), pack2x16float(vec2f(ray.radiance.b, ray.transmittance)));
}
fn unpackRay(packed: vec2u) -> Ray {
  let bt = unpack2x16float(packed.y);
  return Ray(vec3f(unpack2x16float(packed.x), bt.x), bt.y);
}
fn packMerged(m: vec3f) -> u32 {
  let exponent = clamp(i32((bitcast<u32>(max(m.r, max(m.g, m.b))) >> 23u) & 255u) - 111, 0, 31);
  let mantissa = vec3u(min(max(m, vec3f(0.0)) * bitcast<f32>(u32(151 - exponent) << 23u) + 0.5, vec3f(511.0)));
  return mantissa.r | (mantissa.g << 9u) | (mantissa.b << 18u) | (u32(exponent) << 27u);
}
fn unpackMerged(packed: u32) -> vec3f {
  return vec3f(vec3u(packed, packed >> 9u, packed >> 18u) & vec3u(511u)) * bitcast<f32>(((packed >> 27u) + 103u) << 23u);
}

// 1-px rays are two half pixels; a stored ray that ends off the rows is opaque
fn ray(id: i32, rise: i32, startRow: i32) -> Ray {
  if (id < 0) { return OPAQUE; }
  let length = nodes[id].length;
  if (length == 1) {
    let column = nodes[id].column;
    return extend(halfPixel(column, startRow), halfPixel(column, startRow + rise));
  }
  let endRow = startRow + rise;
  if (endRow < 0 || endRow >= N) { return OPAQUE; }
  return unpackRay(rays[nodes[id].base + (rise + length) * N + endRow]);
}
// light arriving at the ray's start; none behind the root or off the rows
fn mergedAt(id: i32, length: i32, rise: i32, startRow: i32) -> vec3f {
  if (id < 0 || startRow < 0 || startRow >= N) { return vec3f(0.0); }
  return unpackMerged(merged[nodes[id].base + (rise + length) * N + startRow]);
}

// the angle of slopes (rise ± ½) / length within the frustum; 1-px rays: π/2 - ln 2 and ½ ln 2
fn coneWeight(length: i32, rise: i32) -> f32 {
  if (length == 1) { return select(0.34657359, 0.87764915, rise == 0); }
  let l = f32(length);
  return atan(min(1.0, (f32(rise) + 0.5) / l)) - atan(max(-1.0, (f32(rise) - 0.5) / l));
}

// The linear rule: the chains through rows q0 (weight 1 - f) and q0 + 1 (weight f) at the cut. A chain through row q is
// the first piece at rise q, then the second at rise d - q.
struct Split { q0: i32, f: f32 };
fn split(length: i32, cut: i32, rise: i32) -> Split {
  let q0 = floorDiv(cut * rise, length);
  return Split(q0, f32(cut * rise - q0 * length) / f32(length));
}
fn chainWeight(s: Split, q: i32) -> f32 { return select(0.0, 1.0 - s.f, q == s.q0) + select(0.0, s.f, q == s.q0 + 1); }

struct Slot { node: i32, rise: i32, row: i32 };
fn slotOf(i: u32) -> Slot {
  let L = params.length; let nodeAndRise = i32(i / u32(N));
  return Slot(params.nodeStart + nodeAndRise / (2 * L + 1), nodeAndRise % (2 * L + 1) - L, i32(i % u32(N)));
}

// extensions: one thread per (node, rise, end row)
@compute @workgroup_size(${WORKGROUP})
fn extensions(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let t = slotOf(i);
  let startRow = t.row - t.rise;
  let first = nodes[t.node].first; let second = nodes[t.node].second;
  let s = split(params.length, params.cut, t.rise);
  var sum = Ray(vec3f(0.0), 0.0);
  for (var q = s.q0; q <= s.q0 + 1; q++) {
    let w = chainWeight(s, q);
    if (w == 0.0) { continue; }
    let chain = extend(ray(first, q, startRow), ray(second, t.rise - q, startRow + q));
    sum = Ray(sum.radiance + w * chain.radiance, sum.transmittance + w * chain.transmittance);
  }
  rays[u32(params.base) + i] = packRay(sum);
}

// The merge, the transpose of the extensions: a ray takes each chain weight of the merged light of the parents it is a
// piece of. A second piece takes it through the first piece, plus the first piece's light times the parent's cone.
// Wu moves a two-chain parent's light by its row slope s: M (1 + (1 - f) s) up, M (1 - f s) down.
fn gather(id: i32, rise: i32, row: i32) -> vec3f {
  let parent = nodes[id].parent; let L = nodes[id].parentLength; let cut = nodes[id].cut;
  let second = nodes[id].piece == 1; let sibling = nodes[id].sibling;
  // the parent rises whose crossing is within a row of this rise
  let piece = select(cut, L - cut, second);
  let lo = max(-L, floorDiv((rise - 1) * L, piece) + 1);
  let hi = min(L, -floorDiv(-(rise + 1) * L, piece) - 1);
  var sum = vec3f(0.0);
  for (var d = lo; d <= hi; d++) {
    let s = split(L, cut, d);
    let q = select(rise, d - rise, second);   // the chain's row at the cut
    let w = chainWeight(s, q);
    if (w == 0.0) { continue; }
    let start = select(row, row - q, second); // the parent's start row
    var m = mergedAt(parent, L, d, start);
    if (WU && s.f > 0.0) {
      let below = mergedAt(parent, L, d, start - 1); let above = mergedAt(parent, L, d, start + 1);
      let total = below + m + above;
      m *= 1.0 + (f32(q - s.q0) - s.f) * select(vec3f(0.0), (above - below) / total, total > vec3f(0.0));
    }
    if (second) {
      let before = ray(sibling, q, start);
      m = before.transmittance * m + coneWeight(L, d) * before.radiance;
    }
    sum += w * m;
  }
  return sum;
}

// merge: one thread per (node, rise, start row)
@compute @workgroup_size(${WORKGROUP})
fn merge(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let t = slotOf(i);
  merged[u32(params.base) + i] = packMerged(gather(t.node, t.rise, t.row));
}

// The merge into the 1-px rays, and the deposit: a pixel is the mean of the three 1-px rays from its row (left half)
// and the three that end in it (right half, through the left half pixel). Solid pixels hold nothing.
@compute @workgroup_size(${WORKGROUP})
fn pixels(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let x = i32(i) / N; let row = i32(i) % N;
  let mirrored = (params.frustum & 1) == 1;
  let column = select(x, ROOT - 1 - x, mirrored);
  let leaf = params.leafStart + column - params.sceneStart;
  var value = vec3f(0.0);
  for (var rise = -1; rise <= 1; rise++) {
    let left = gather(leaf, rise, row);
    var toRow = left;
    if (rise != 0) { toRow = gather(leaf, rise, row - rise); }
    let before = halfPixel(column, row - rise);
    value += 0.5 * (left + before.transmittance * toRow + coneWeight(1, rise) * before.radiance);
  }
  value *= select(0.0, 1.0, halfPixel(column, row).transmittance > 0.0);
  let slot = ((params.frustum >> 1) * N + x) * N + row;
  if (mirrored) { value += axisFluence[slot].rgb; }
  axisFluence[slot] = vec4f(value, 0.0);
}

// fluence: the two axes' sum
@compute @workgroup_size(${WORKGROUP})
fn output(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let x = i32(i) / N; let y = i32(i) % N;
  let value = axisFluence[x * N + y].rgb + axisFluence[(N + y) * N + x].rgb;
  fluence[i] = vec4f(value, dot(value, vec3f(0.2126, 0.7152, 0.0722)));
}
`;

interface Dispatch {
  label: string;
  stage: 'extensions' | 'merge' | 'pixels' | 'output';
  offset: number;
  count: number;
}

export class TreeHRC {
  readonly device: GPUDevice;
  readonly N: number;
  readonly tree: Tree;
  readonly wu: boolean;
  /** frustums 0 and 2, then 1 and 3 */
  readonly layouts: [Layout, Layout];
  profiler: Profiler | null = null;
  /** [x * N + y] then [y * N + x]: (radiance rgb, transmittance) */
  readonly sceneBuffer: GPUBuffer;
  /** world [x * N + y]: (fluence rgb, luminance) */
  readonly outputBuffer: GPUBuffer;
  #nodes: GPUBuffer;
  #rays: GPUBuffer;
  #merged: GPUBuffer;
  #axisFluence: GPUBuffer;
  #params: GPUBuffer;
  #pipelines: Record<Dispatch['stage'], GPUComputePipeline>;
  #bindGroup: GPUBindGroup;
  #program: Dispatch[] = [];

  constructor(device: GPUDevice, N: number, tree: Tree, { wu = true }: { wu?: boolean } = {}) {
    this.device = device;
    this.N = N;
    this.tree = tree;
    this.wu = wu;
    this.layouts = frustumLayouts(tree, N);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, size: number, use = usage) => device.createBuffer({ label: `tree HRC ${label}`, size: Math.max(16, size), usage: use });
    this.sceneBuffer = buffer('scene', 2 * N * N * PIXEL_BYTES);
    this.outputBuffer = buffer('fluence', N * N * PIXEL_BYTES);
    this.#axisFluence = buffer('fluence per axis', 2 * N * N * PIXEL_BYTES);
    const slots = Math.max(...this.layouts.map((l) => l.slots));
    this.#rays = buffer('rays', slots * RAY_BYTES);
    this.#merged = buffer('merged', slots * MERGED_BYTES);

    // both layouts in one node table
    const [a, b] = this.layouts, offsets = [0, a.nodes.length];
    const table = new Int32Array((a.nodes.length + b.nodes.length) * NODE_FIELDS);
    this.layouts.forEach((lay, k) => lay.nodes.forEach((n, i) => {
      const shift = (id: number) => (id < 0 ? -1 : id + offsets[k]);
      table.set([n.column, n.length, n.base, shift(n.first), shift(n.second), shift(n.parent), n.parentLength, n.cut, n.piece, shift(n.sibling)], (offsets[k] + i) * NODE_FIELDS);
    }));
    this.#nodes = buffer('nodes', table.byteLength);
    device.queue.writeBuffer(this.#nodes, 0, table);

    const module = device.createShaderModule({ label: 'tree HRC', code: shader(N, tree.root, wu) });
    const kinds: GPUBufferBindingType[] = ['read-only-storage', 'uniform', 'read-only-storage', 'storage', 'storage', 'storage', 'storage'];
    const bindLayout = device.createBindGroupLayout({
      entries: kinds.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, hasDynamicOffset: type === 'uniform' } })),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `tree HRC ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } });
    this.#pipelines = { extensions: pipeline('extensions'), merge: pipeline('merge'), pixels: pipeline('pixels'), output: pipeline('output') };

    const params: number[][] = [];
    const add = (stage: Dispatch['stage'], label: string, count: number, p: number[]) => {
      this.#program.push({ stage, label, count, offset: params.length * PARAM_STRIDE });
      params.push(p);
    };
    for (let frustum = 0; frustum < 4; frustum++) {
      const lay = this.layouts[frustum & 1], offset = offsets[frustum & 1];
      const stored = lay.levels.filter((l) => l.length > 1);
      const level = (l: (typeof stored)[number]) => [l.length, l.cut, offset + l.nodeStart, l.nodeCount * (2 * l.length + 1) * N, frustum, l.base, 0, 0];
      for (const l of stored) add('extensions', `extensions ${l.length}`, l.nodeCount * (2 * l.length + 1) * N, level(l));
      for (const l of [...stored].reverse()) add('merge', `merge ${l.length}`, l.nodeCount * (2 * l.length + 1) * N, level(l));
      add('pixels', 'merge 1 and pixels', N * N, [1, 0, 0, N * N, frustum, 0, lay.sceneStart, offset + lay.levels[0].nodeStart]);
    }
    add('output', 'output', N * N, [0, 0, 0, N * N, 0, 0, 0, 0]);
    const data = new ArrayBuffer(params.length * PARAM_STRIDE);
    params.forEach((p, k) => new Int32Array(data, k * PARAM_STRIDE, 8).set(p));
    this.#params = buffer('params', data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.#params, 0, data);
    const bound = [this.sceneBuffer, this.#params, this.#nodes, this.#rays, this.#merged, this.#axisFluence, this.outputBuffer];
    this.#bindGroup = device.createBindGroup({
      layout: bindLayout,
      entries: bound.map((b, binding) => ({ binding, resource: binding === 1 ? { buffer: b, size: 32 } : { buffer: b } })),
    });
  }

  get dispatches() {
    return this.#program.length;
  }

  get bytes() {
    return [this.sceneBuffer, this.outputBuffer, this.#nodes, this.#rays, this.#merged, this.#axisFluence, this.#params].reduce((s, b) => s + b.size, 0);
  }

  destroy() {
    for (const b of [this.sceneBuffer, this.outputBuffer, this.#nodes, this.#rays, this.#merged, this.#axisFluence, this.#params]) b.destroy();
  }

  #staging = staging();
  upload({ emission, solid }: TreeScene) {
    const { N } = this, data = rgba(this.#staging(2 * N * N * 4), { emission, solid }, 0, 1, true);
    transpose(data, N, 4, 0, N * N);
    this.device.queue.writeBuffer(this.sceneBuffer, 0, data);
  }

  /** consecutive dispatches of one stage share a compute pass */
  encodeAll(encoder: GPUCommandEncoder) {
    let pass: GPUComputePassEncoder | null = null, stage = '';
    for (const d of this.#program) {
      if (d.stage !== stage) {
        pass?.end();
        stage = d.stage;
        pass = this.profiler ? this.profiler.computePass(encoder, stage) : encoder.beginComputePass();
      }
      this.profiler?.name(pass!, d.label);
      pass!.setPipeline(this.#pipelines[d.stage]);
      pass!.setBindGroup(0, this.#bindGroup, [d.offset]);
      const groups = Math.ceil(d.count / WORKGROUP);
      pass!.dispatchWorkgroups(Math.min(groups, MAX_GROUPS_X), Math.ceil(groups / MAX_GROUPS_X));
    }
    pass?.end();
  }

  /** solve once and read the fluence back: [x * N + y], (rgb, luminance) */
  async solve(scene: TreeScene): Promise<Float32Array> {
    this.upload(scene);
    const bytes = this.N * this.N * PIXEL_BYTES;
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

  /** bytes each pass moves in frustum 0, each element once */
  traffic(): Record<string, Traffic> {
    const { N } = this, lay = this.layouts[0], out: Record<string, Traffic> = {};
    const levelSlots = (length: number) => {
      const l = lay.levels.find((x) => x.length === length);
      return l ? l.nodeCount * (2 * length + 1) * N : 0;
    };
    const pieceBytes = (length: number) => (length === 1 ? 2 * N * N * PIXEL_BYTES : levelSlots(length) * RAY_BYTES);
    const parentLengths = (length: number) => new Set(lay.nodes.filter((n) => n.length === length && n.parent >= 0).map((n) => n.parentLength));
    for (const l of lay.levels) {
      if (l.length === 1) continue;
      const slots = levelSlots(l.length), [p0, p1] = this.tree.pieces.get(l.length)!;
      out[`extensions ${l.length}`] = { read: pieceBytes(p0) + (p1 !== p0 ? pieceBytes(p1) : 0), write: slots * RAY_BYTES };
      const parents = [...parentLengths(l.length)].reduce((s, P) => s + levelSlots(P) * MERGED_BYTES, 0);
      out[`merge ${l.length}`] = { read: parents + slots * RAY_BYTES, write: slots * MERGED_BYTES };
    }
    const leafParents = [...parentLengths(1)].reduce((s, P) => s + levelSlots(P) * MERGED_BYTES, 0);
    out['merge 1 and pixels'] = { read: leafParents + 2 * N * N * PIXEL_BYTES, write: 2 * N * N * PIXEL_BYTES };
    out.output = { read: 2 * N * N * PIXEL_BYTES, write: N * N * PIXEL_BYTES };
    return out;
  }
}
