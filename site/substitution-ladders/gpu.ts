// The ladder's solve (./ladder.ts, frustum and solveChannels) on the GPU. The ladder is irregular, so its tables are
// built on the CPU and uploaded once per rule, depth and law: the tile classes, the paths per rise with their junction
// rows and weights, each tile's children, each child tile's parent, and the merge's inverse table (per parent class,
// child position q and child rise d, the paths whose q-th child has rise d). The shader only indexes them.
//
// A record is (R rgb, T): the light a ray of a tile carries out of its far end and its transmittance, for every tile,
// start row and rise, at every level. One up step per level builds a parent's records from its children's along its
// paths. The merge is a gather: each child record sums, over the inverse table, what the paths through it hand it
// (the same terms as the CPU's scatter), so the transport is the CPU's, in f32. The outside light M is kept for two
// levels only (ping-pong). The four frustum views run one after another into the same buffers; each view's readout
// adds into the output, and the last writes the luminance.

import type { Ladder } from './ladder';

const WORKGROUP = 256;
const CLASS = 10;   // u32 per class in the table

/** the u32 table the shader reads: a header of section offsets, then the sections */
function tables(lad: Ladder) {
  const cls = lad.classes, N = lad.N, rows = 3 * N, depth = lad.depth;
  const entry: number[] = [], size: number[] = [];
  let e = 0;
  for (const c of cls) { entry.push(e); size.push(c.xs.length * rows * (2 * c.L + 1)); e += size.at(-1)!; }
  const levels: number[] = [];
  for (let l = 0; l <= depth; l++) {
    const ids = cls.map((c, i) => [c, i] as const).filter(([c]) => c.level === l).map(([, i]) => i);
    if (!ids.length || ids.some((v, i) => v !== ids[0] + i)) throw new Error(`level ${l} has no contiguous classes`);
    levels.push(ids[0], ids.length, entry[ids[0]], ids.reduce((s, i) => s + size[i], 0));
  }

  // paths: per class, the range for each rise (with a sentinel), then (w, j offset, D) per path, j in jtab
  const pathStart: number[] = [], paths: number[] = [], jtab: number[] = [], pathBase: number[] = [];
  const pathIds: number[][][] = [];   // [class][D + L] → global path ids
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  const bits = (w: number) => ((f32[0] = w), u32[0]);
  for (const c of cls) {
    pathBase.push(pathStart.length);
    const ids: number[][] = [];
    for (let q = 0; q < c.paths.length; q++) {
      pathStart.push(paths.length / 3);
      ids.push(c.paths[q].map((p) => {
        const id = paths.length / 3;
        paths.push(bits(p.w), jtab.length, (q - c.L) >>> 0);
        for (const j of p.j) jtab.push(j >>> 0);
        return id;
      }));
    }
    pathStart.push(paths.length / 3);
    pathIds.push(ids);
  }

  // children: per class and position q, (child class, its inverse-table base); per tile, its children's tiles
  const kids: number[] = [], kidBase: number[] = [], kidTile: number[] = [], kidTileBase: number[] = [];
  const invStart: number[] = [], invList: number[] = [];
  cls.forEach((c, ci) => {
    kidBase.push(kids.length / 2);
    kidTileBase.push(kidTile.length);
    for (const t of c.kidIndex) kidTile.push(t);
    c.kids.forEach((kc, q) => {
      const Lq = cls[kc].L, slots: number[][] = Array.from({ length: 2 * Lq + 1 }, () => []);
      c.paths.forEach((ps, D) => ps.forEach((p, n) => slots[p.j[q + 1] - p.j[q] + Lq].push(pathIds[ci][D][n])));
      kids.push(kc, invStart.length);
      for (const s of slots) { invStart.push(invList.length); invList.push(...s); }
      invStart.push(invList.length);
    });
  });
  // parents: per child tile, (parent class, parent tile, q)
  const tileBase: number[] = [];
  let tiles = 0;
  for (const c of cls) { tileBase.push(tiles); tiles += c.xs.length; }
  const parent = new Array<number>(3 * tiles).fill(0);
  cls.forEach((c, ci) => {
    const k = c.kids.length;
    for (let ti = 0; ti < c.xs.length; ti++)
      for (let q = 0; q < k; q++) {
        const at = 3 * (tileBase[c.kids[q]] + c.kidIndex[ti * k + q]);
        parent[at] = ci; parent[at + 1] = ti; parent[at + 2] = q;
      }
  });

  const classes: number[] = [];
  cls.forEach((c, i) => classes.push(c.L, c.kids.length, c.xs.length, entry[i], size[i], pathBase[i], kidBase[i], kidTileBase[i], tileBase[i], c.level));
  const sections = [classes, levels, pathStart, paths, jtab, kids, kidTile, parent, invStart, invList];
  const header = sections.length, out: number[] = [];
  let at = header;
  for (const s of sections) { out.push(at); at += s.length; }
  for (const s of sections) for (const v of s) out.push(v);
  return { table: Uint32Array.from(out), entries: e, levels, largestLevel: Math.max(...levels.filter((_, i) => i % 4 === 3)) };
}

const shader = /* wgsl */ `
struct Params { level: u32, view: u32, N: u32, rows: u32, top: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scene: array<vec4f>;            // world [x * N + y]: (emission rgb, 1 opaque)
@group(0) @binding(2) var<storage, read_write> records: array<vec4f>;    // (R rgb, T) per class, tile, row, rise
@group(0) @binding(3) var<storage, read> above: array<vec4f>;            // M one level up, by entry within its level
@group(0) @binding(4) var<storage, read_write> here: array<vec4f>;       // M at this level
@group(0) @binding(5) var<storage, read> tab: array<u32>;
@group(0) @binding(6) var<storage, read_write> fluence: array<vec4f>;    // world [x * N + y]: (rgb, luminance)

const CLASS = ${CLASS}u;
fn sec(s: u32) -> u32 { return tab[s]; }
fn cl(c: u32, f: u32) -> u32 { return tab[sec(0u) + c * CLASS + f]; }
fn lev(l: u32, f: u32) -> u32 { return tab[sec(1u) + l * 4u + f]; }

// the world pixel of frustum pixel (x, y) in view v
fn world(v: u32, x: u32, y: u32) -> u32 {
  let N = params.N;
  switch v {
    case 0u: { return x * N + y; }
    case 1u: { return (N - 1u - x) * N + y; }
    case 2u: { return y * N + x; }
    default: { return y * N + (N - 1u - x); }
  }
}
// free and emission at frustum column x, padded row y (rows off the scene are opaque and dark)
fn freeAt(x: u32, y: i32) -> f32 {
  let N = i32(params.N);
  if (y < N || y >= 2 * N) { return 0.0; }
  return 1.0 - scene[world(params.view, x, u32(y - N))].w;
}
fn emissionAt(x: u32, y: i32) -> vec3f {
  let N = i32(params.N);
  if (y < N || y >= 2 * N) { return vec3f(0.0); }
  return scene[world(params.view, x, u32(y - N))].rgb;
}

fn record(c: u32, tile: u32, row: i32, d: i32) -> u32 {
  let L = i32(cl(c, 0u));
  return cl(c, 3u) + (tile * params.rows + u32(row)) * u32(2 * L + 1) + u32(d + L);
}

struct At { c: u32, tile: u32, row: i32, d: i32 }
fn locate(level: u32, i: u32) -> At {
  let e = lev(level, 2u) + i;
  var c = lev(level, 0u);
  loop { if (e < cl(c, 3u) + cl(c, 4u)) { break; } c += 1u; }
  let L = cl(c, 0u); let local = e - cl(c, 3u); let slots = 2u * L + 1u; let rest = local / slots;
  return At(c, rest / params.rows, i32(rest % params.rows), i32(local % slots) - i32(L));
}

fn cone(L: f32, d: f32) -> f32 {
  return atan(clamp((d + 0.5) / L, -1.0, 1.0)) - atan(clamp((d - 0.5) / L, -1.0, 1.0));
}

fn index(id: vec3u, groups: vec3u) -> u32 { return id.x + id.y * groups.x * ${WORKGROUP}u; }

// level 0: the pixel's rays of rise −1, 0, 1
@compute @workgroup_size(${WORKGROUP}) fn pixels(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = index(id, groups);
  if (i >= lev(0u, 3u)) { return; }
  let at = locate(0u, i);
  let x = at.tile; let y = at.row; let d = at.d;
  let f0 = freeAt(x, y); let f1 = freeAt(x, y + d);
  records[lev(0u, 2u) + i] = vec4f(emissionAt(x, y + d) + f1 * emissionAt(x, y), f0 * f1);
}

// a level's records from its children's, along its paths
@compute @workgroup_size(${WORKGROUP}) fn up(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let level = params.level;
  let i = index(id, groups);
  if (i >= lev(level, 3u)) { return; }
  let at = locate(level, i);
  let c = at.c; let L = i32(cl(c, 0u)); let k = cl(c, 1u); let rows = i32(params.rows);
  let ps = sec(2u) + cl(c, 5u) + u32(at.d + L);
  var T = 0.0; var R = vec3f(0.0);
  for (var p = tab[ps]; p < tab[ps + 1u]; p++) {
    let w = bitcast<f32>(tab[sec(3u) + 3u * p]); let j = sec(4u) + tab[sec(3u) + 3u * p + 1u];
    var Tw = w; var Ra = vec3f(0.0); var ok = true;
    for (var q = 0u; q < k; q++) {
      let j0 = bitcast<i32>(tab[j + q]); let j1 = bitcast<i32>(tab[j + q + 1u]);
      let yc = at.row + j0;
      if (yc < 0 || yc >= rows) { ok = false; break; }
      let kc = tab[sec(5u) + 2u * (cl(c, 6u) + q)];
      let kt = tab[sec(6u) + cl(c, 7u) + at.tile * k + q];
      let r = records[record(kc, kt, yc, j1 - j0)];
      Ra = r.rgb + r.w * Ra;
      Tw *= r.w;
    }
    if (ok) { T += Tw; R += w * Ra; }
  }
  records[lev(level, 2u) + i] = vec4f(R, T);
}

// a level's outside light: what the paths of its tile's parent through each record hand it
@compute @workgroup_size(${WORKGROUP}) fn down(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let level = params.level;
  let i = index(id, groups);
  if (i >= lev(level, 3u)) { return; }
  let at = locate(level, i);
  let rows = i32(params.rows);
  let pt = 3u * (cl(at.c, 8u) + at.tile);
  let pc = tab[sec(7u) + pt]; let ptile = tab[sec(7u) + pt + 1u]; let q = tab[sec(7u) + pt + 2u];
  let P = i32(cl(pc, 0u)); let k = cl(pc, 1u);
  let Lq = i32(cl(at.c, 0u));
  let aboveStart = lev(level + 1u, 2u);
  let hasAbove = level + 1u < params.top;
  let kb = cl(pc, 6u);
  let slot = sec(8u) + tab[sec(5u) + 2u * (kb + q) + 1u] + u32(at.d + Lq);
  var M = vec3f(0.0);
  for (var n = tab[slot]; n < tab[slot + 1u]; n++) {
    let p = tab[sec(9u) + n];
    let w = bitcast<f32>(tab[sec(3u) + 3u * p]); let j = sec(4u) + tab[sec(3u) + 3u * p + 1u];
    let D = bitcast<i32>(tab[sec(3u) + 3u * p + 2u]);
    let y = at.row - bitcast<i32>(tab[j + q]);
    if (y < 0 || y >= rows) { continue; }
    var Tw = w; var Ra = vec3f(0.0); var ok = true;
    for (var r = 0u; r < q; r++) {
      let j0 = bitcast<i32>(tab[j + r]); let j1 = bitcast<i32>(tab[j + r + 1u]);
      let yr = y + j0;
      if (yr < 0 || yr >= rows) { ok = false; break; }
      let kc = tab[sec(5u) + 2u * (kb + r)];
      let kt = tab[sec(6u) + cl(pc, 7u) + ptile * k + r];
      let rec = records[record(kc, kt, yr, j1 - j0)];
      Ra = rec.rgb + rec.w * Ra;
      Tw *= rec.w;
    }
    if (!ok) { continue; }
    var far = vec3f(0.0);
    if (hasAbove) { far = above[record(pc, ptile, y, D) - aboveStart].rgb; }
    if (q == 0u) { M += w * far; } else { M += Tw * far + w * cone(f32(P), f32(D)) * Ra; }
  }
  here[i] = vec4f(M, 0.0);
}

// per pixel and rise, the two length-1 rays whose middles bracket the pixel centre; M at level 0 is in 'above'
@compute @workgroup_size(${WORKGROUP}) fn readout(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let N = params.N; let i = index(id, groups);
  if (i >= N * N) { return; }
  let x = i / N; let y = i32(i % N + N); let rows = i32(params.rows);
  var sum = vec3f(0.0);
  for (var d = -1; d <= 1; d++) {
    sum += 0.5 * above[record(0u, x, y, d)].rgb;
    let ys = y - d;
    if (ys >= 0 && ys < rows) { sum += 0.5 * freeAt(x, ys) * above[record(0u, x, ys, d)].rgb; }
  }
  sum *= freeAt(x, y);
  let o = world(params.view, x, u32(y) - N);
  var total = sum;
  if (params.view > 0u) { total += fluence[o].rgb; }
  var lum = 0.0;
  if (params.view == 3u) { lum = dot(total, vec3f(0.2126, 0.7152, 0.0722)); }
  fluence[o] = vec4f(total, lum);
}
`;

const modules = new WeakMap<GPUDevice, { layout: GPUBindGroupLayout; pipelines: Record<'pixels' | 'up' | 'down' | 'readout', GPUComputePipeline> }>();
function programFor(device: GPUDevice) {
  let m = modules.get(device);
  if (m) return m;
  const module = device.createShaderModule({ code: shader });
  const types: GPUBufferBindingType[] = ['uniform', 'read-only-storage', 'storage', 'read-only-storage', 'storage', 'read-only-storage', 'storage'];
  const layout = device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const make = (entryPoint: string) => device.createComputePipeline({ layout: pl, compute: { module, entryPoint } });
  m = { layout, pipelines: { pixels: make('pixels'), up: make('up'), down: make('down'), readout: make('readout') } };
  modules.set(device, m);
  return m;
}

/** the solve of one ladder: scene (rgb, opaque) per world pixel in, (fluence rgb, luminance) per world pixel out */
export class LadderSolver {
  readonly N: number;
  readonly bytes: number;
  readonly dispatches: number;
  #buffers: GPUBuffer[] = [];
  #program: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; count: number }[] = [];

  constructor(readonly device: GPUDevice, lad: Ladder, scene: GPUBuffer, output: GPUBuffer) {
    const N = (this.N = lad.N), rows = 3 * N, depth = lad.depth, { table, entries, levels, largestLevel } = tables(lad);
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (entries * 16 > limit) throw new Error(`N = ${N}: the records need ${(entries * 16 / 2 ** 20).toFixed(0)} MiB, over this GPU's ${(limit / 2 ** 20).toFixed(0)} MiB`);
    const buffer = (size: number, usage = GPUBufferUsage.STORAGE) => {
      const b = device.createBuffer({ size: Math.max(16, size), usage });
      this.#buffers.push(b);
      return b;
    };
    const tab = buffer(table.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(tab, 0, table);
    const records = buffer(entries * 16), M = [buffer(largestLevel * 16), buffer(largestLevel * 16)];
    this.bytes = this.#buffers.reduce((s, b) => s + b.size, 0);
    const { layout, pipelines } = programFor(device);
    const add = (pipeline: GPUComputePipeline, view: number, level: number, count: number, above: GPUBuffer, here: GPUBuffer) => {
      const params = buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0, new Uint32Array([level, view, N, rows, depth, 0, 0, 0]));
      const resources = [params, scene, records, above, here, tab, output];
      this.#program.push({ pipeline, count, bindGroup: device.createBindGroup({ layout, entries: resources.map((b, binding) => ({ binding, resource: { buffer: b } })) }) });
    };
    const count = (l: number) => levels[4 * l + 3];
    for (let view = 0; view < 4; view++) {
      add(pipelines.pixels, view, 0, count(0), M[1], M[0]);
      for (let l = 1; l < depth; l++) add(pipelines.up, view, l, count(l), M[1], M[0]);
      // level l writes M[l % 2] and reads level l + 1's from M[(l + 1) % 2]; nothing lies outside the top
      for (let l = depth - 1; l >= 0; l--) add(pipelines.down, view, l, count(l), M[(l + 1) % 2], M[l % 2]);
      add(pipelines.readout, view, 0, N * N, M[0], M[1]);
    }
    this.dispatches = this.#program.length;
  }

  encode(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();
    for (const d of this.#program) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      const groups = Math.ceil(d.count / WORKGROUP);
      pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
    }
    pass.end();
  }

  destroy() {
    for (const b of this.#buffers) b.destroy();
  }
}
