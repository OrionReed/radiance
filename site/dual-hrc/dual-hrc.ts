// HRC on the GPU and its dual, on the same scene. Mono, one probe per pixel, four frustums, no sky.
//
// Primal: HRC as hrc.py runs it. Seed levels 0 and 1, extensions, merge from the top down; a pixel reads probe p + 1 of
// each frustum; the cross blur on the output.
//
// Dual: the primal's transpose in gather form: every output reads a fixed list of inputs, nothing is scattered. Light
// leaves the lights and is carried out through the cascade by the dual extensions; dual gathering halves the rays back
// down to c0, and the transposed seed splats them into the pixels. A source group is a set of sources with one beam
// (the solid lights' faces, or one spot beam). Cone values are kept pattern-free, R*_l(q, c) = λ(l − 1, c) R~_l(q, c) with
// λ the beam's mean over the sub-cone, so the carry needs no split ratio; λ is in the splat tables. Solid lights enter
// from their faces, with each face's near field from a table; a spot is a transparent point source. The dual's cross
// blur mixes E + W across y and N + S across x.
//
// Layout: frustum cell (p, s) is world pixel [x * N + y] (y up) by direction d: E (x = p, y = s), N (s, p), W (N − p, s),
// S (s, N − p); W and S have p = 1 .. N. Ray level l: probe q, ray k, row s at TOFF[l] + ((q − lo) (2^l + 1) + k) N + s;
// the primal keeps (rad, tr) interleaved, the dual its splats at DBASE + that index (transmittance in the shared table
// below DBASE). Cone level l: (((q − lo) << l) + cone) N + s; the primal ping-pongs two such planes, the dual keeps one per
// source group and level l ≥ 1 (level l, source group c at (l − 1) P.rIn + c N²).

import type { Profiler, Traffic } from '../_shared/profiler';

export type Form = 'primal' | 'dual';
export interface DualOptions {
  form: Form;
  /** dual: keep one merge-part sum per ray instead of every source group's cone values (the same image) */
  raySums?: boolean;
}
/** World arrays [x * N + y], y up. spotHalf < 0 (or absent): not a spot. Angles in radians, world, 0 along +x. */
export interface DualScene {
  E: Float32Array;
  A: Float32Array;
  spotDir?: Float32Array;
  spotHalf?: Float32Array;
}

const WG = 64;
export const MAX_LOBES = 8;
const PARAM_STRIDE = 512;
const MAX_CHANNELS = 2 + MAX_LOBES;
const PARAM_BYTES = 32 + 16 * MAX_CHANNELS;
// scene planes: emission, the primal's and the dual's transmittance, spot direction, spot half width, spot beam index
const PLANES = { E: 0, AP: 1, AD: 2, DIR: 3, HALF: 4, LOBE: 5 } as const;
// The dual's source planes: one per spot beam, then the near faces pass's eight (facing lo, hi, side lo, hi for E and N,
// then for W and S). Each is kept twice: [x * N + y], and after the first P.srcPlanes [y * N + x], so that every frustum
// reads its sources along its rows (the faces' planes are written in their frustum's layout only).
const FLAG = { TOP: 1, SPOT: 2 } as const;

function ceilLog2(n: number) {
  return n > 1 ? Math.ceil(Math.log2(n)) : 0;
}

// Exact DDA chords (dx, dy, length) of the segment a→b through unit cells, relative to floor(a).
function chordList(ax: number, ay: number, bx: number, by: number): [number, number, number][] {
  const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
  let ix = Math.floor(ax), iy = Math.floor(ay);
  const sx = ux > 0 ? 1 : -1, sy = uy > 0 ? 1 : -1;
  let tmx = ux ? ((ux > 0 ? ix + 1 : ix) - ax) / ux : 1e30;
  let tmy = uy ? ((uy > 0 ? iy + 1 : iy) - ay) / uy : 1e30;
  const tdx = ux ? Math.abs(1 / ux) : 1e30, tdy = uy ? Math.abs(1 / uy) : 1e30;
  const out: [number, number, number][] = [];
  let t0 = 0;
  while (t0 < L) {
    const t1 = Math.min(tmx, tmy, L);
    if (t1 - t0 > 1e-9) out.push([ix - Math.floor(ax), iy - Math.floor(ay), t1 - t0]);
    if (t1 >= L) break;
    if (tmx < tmy) ((ix += sx), (tmx += tdx));
    else ((iy += sy), (tmy += tdy));
    t0 = t1;
  }
  return out;
}

/** The seed rays of level L: per ray k, its two sub-lines (at ±¼ across), each a list of chords. */
function seedRays(L: number) {
  const n = 1 << L;
  return Array.from({ length: n + 1 }, (_, k) => [-0.25, 0.25].map((u) => chordList(0.5, 0.5 + u, 0.5 + n, 0.5 + u + 2 * k - n)));
}

// ── scheme tables, at wbase(l) + sub per level: levels 0 .. l − 1 hold 2, 4, .. 2^l entries ──
const wbase = (l: number) => (2 << l) - 2;
/** the primal's merge weight: the arc of sub-cone `sub` of level l */
const coneArc = (l: number, sub: number) => {
  const n = 2 << l;
  return Math.atan2(2 * sub - n + 2, n) - Math.atan2(2 * sub - n, n);
};
/** the dual's splat weight in length along the frustum axis: ∫ dσ / √(1 + σ²) over the sub-cone */
const asinhW = (l: number, sub: number) => {
  const m = 1 << l;
  return Math.asinh((sub + 1 - m) / m) - Math.asinh((sub - m) / m);
};

// ── beams over the frustum-local angle w ∈ [−π/4, π/4] ──
// In length along the frustum axis a sub-cone weighs its ray by ∫ sec w dw, so a beam is given as Gx(w) = ∫ beam(w) sec w dw
// and its mean over a sub-cone is taken in that measure.
type Beam = (w: number) => number;
const xlenCos: Beam = (w) => w;
const xlenAbsSin: Beam = (w) => -Math.sign(w) * Math.log(Math.cos(w));
const wrap = (a: number) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
/** a beam of world direction theta0 and half width `half`, value 1 inside, in frustum d (null if it misses the frustum) */
function xlenBeam(theta0: number, half: number, d: number): Beam | null {
  const wc = wrap([theta0, Math.PI / 2 - theta0, Math.PI - theta0, theta0 + Math.PI / 2][d]);
  const lo = wc - half, hi = wc + half, q = Math.PI / 4, shifts = [-2 * Math.PI, 0, 2 * Math.PI];
  if (!shifts.some((k) => Math.max(lo + k, -q) < Math.min(hi + k, q))) return null;
  const F = (w: number) => Math.asinh(Math.tan(Math.min(Math.max(w, -q), q)));
  return (w) => shifts.reduce((a, k) => a + F(Math.min(Math.max(w, lo + k), hi + k)) - F(lo + k), 0);
}
/** a source group's splat table: asinhW(l, sub) × λ(l, sub), λ the beam's mean over the sub-cone */
function depositTable(G: Beam, levels: number) {
  const out: number[] = [];
  for (let l = 0; l < levels; l++)
    for (let sub = 0; sub < 2 << l; sub++) {
      const n = 2 << l, t0 = (2 * sub - n) / n, t1 = (2 * sub - n + 2) / n;
      out.push(G(Math.atan(t1)) - G(Math.atan(t0)));   // = asinhW × λ
    }
  return out;
}

interface Channel {
  label: string;
  dirs: ({ lo: number; hi: number; table: number[] } | null)[]; // per frustum: source planes of the lower and upper level-0 sub-cone
}

interface Dispatch {
  pipeline: GPUComputePipeline;
  slot: number;
  x: number; // threads along rows (or y)
  y: number; // rows
  label: string;
  tile?: boolean; // 2-D workgroups of TILE × TILE pixels
}

// ── WGSL ──
const wgslCommon = (N: number, levels: number, toff: number[], dbase: number) => /* wgsl */ `
// a source group of the dual: source planes of the lower and upper level-0 sub-cone, offset of its splat table
struct Chan { lo: u32, hi: u32, w: u32, p3: u32 };
struct Params {
  dir: u32, level: u32, flags: u32, plane: u32,
  rIn: u32, rOut: u32, chanCount: u32, srcPlanes: u32,
  chans: array<Chan, ${MAX_CHANNELS}>,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> scene: array<f32>;
@group(0) @binding(2) var<storage, read_write> T: array<f32>;
@group(0) @binding(3) var<storage, read_write> R: array<f32>;
@group(0) @binding(4) var<storage, read> tables: array<f32>;
@group(0) @binding(5) var<storage, read_write> src: array<f32>;
@group(0) @binding(6) var<storage, read_write> outF: array<f32>;

const N: i32 = ${N};
const NN: u32 = ${N * N}u;
const DBASE: u32 = ${dbase}u; // the dual's ray splats: T[DBASE + tIndex]
var<private> TOFF: array<u32, ${levels + 1}> = array<u32, ${levels + 1}>(${toff.map((v) => `${v}u`).join(', ')});
const PE: u32 = ${PLANES.E}u; const PAP: u32 = ${PLANES.AP}u; const PAD: u32 = ${PLANES.AD}u;
const PDIR: u32 = ${PLANES.DIR}u; const PHALF: u32 = ${PLANES.HALF}u; const PLOBE: u32 = ${PLANES.LOBE}u;
const TOP: u32 = ${FLAG.TOP}u; const SPOT: u32 = ${FLAG.SPOT}u;

fn has(f: u32) -> bool { return (P.flags & f) != 0u; }
// W and S use E's and N's probe lattice (a W ray reversed is an E ray): their cell p is world N − p, p = 1 .. N, and their
// probes are q = 1 .. N >> l at each level; E and N have p = 0 .. N − 1, q = 0 .. (N >> l) − 1.
fn lo() -> i32 { return select(0, 1, P.dir >= 2u); }
// world pixel of frustum cell (p, s)
fn world(p: i32, s: i32) -> u32 {
  switch P.dir {
    case 0u: { return u32(p * N + s); }
    case 1u: { return u32(s * N + p); }
    case 2u: { return u32((N - p) * N + s); }
    default: { return u32(s * N + N - p); }
  }
}
fn inside(p: i32, s: i32) -> bool { return p >= lo() && p < lo() + N && s >= 0 && s < N; }
// the dual's source of pixel p, from the copy laid out along this frustum's rows (rows for E and W, columns for N and S)
fn srcIndex(p: i32, s: i32) -> u32 {
  return select(u32((N - p) * N + s), u32(p * N + s), P.dir < 2u) + select(0u, P.srcPlanes * NN, (P.dir & 1u) == 1u);
}
fn sceneAt(plane: u32, p: i32, s: i32, outside: f32) -> f32 {
  if (!inside(p, s)) { return outside; }
  return scene[plane * NN + world(p, s)];
}
// (1 − e^−x) / x, accurate for small x
fn expRatio(x: f32) -> f32 {
  if (x < 0.5) { return 1.0 - x / 2.0 * (1.0 - x / 3.0 * (1.0 - x / 4.0 * (1.0 - x / 5.0 * (1.0 - x / 6.0 * (1.0 - x / 7.0))))); }
  return (1.0 - exp(-x)) / x;
}
fn powA(a: f32, l: f32) -> f32 {
  if (a <= 0.0) { return 0.0; }
  if (a >= 1.0) { return 1.0; }
  return exp(l * log(a));
}
fn rOk(l: u32, q: i32, s: i32) -> bool { return q >= lo() && q < lo() + (N >> l) && s >= 0 && s < N; }
fn tOk(l: u32, q: i32, k: i32, s: i32) -> bool { return rOk(l, q, s) && k >= 0 && k <= (1 << l); }
fn tIndex(l: u32, q: i32, k: i32, s: i32) -> u32 { return TOFF[l] + u32(((q - lo()) * ((1 << l) + 1) + k) * N + s); }
fn rIndex(l: u32, q: i32, c: i32, s: i32) -> u32 { return u32((((q - lo()) << l) + c) * N + s); }
fn wbase(l: u32) -> u32 { return (2u << l) - 2u; }
`;

// ── the primal ──
// (rad, tr) of T_l, (0, 1) outside
const wgslPrimalT = /* wgsl */ `
fn loadP(l: u32, q: i32, k: i32, s: i32) -> vec2f {
  if (!tOk(l, q, k, s)) { return vec2f(0.0, 1.0); }
  let i = tIndex(l, q, k, s);
  return vec2f(T[2u * i], T[2u * i + 1u]);
}
fn storeP(l: u32, q: i32, k: i32, s: i32, v: vec2f) {
  let i = tIndex(l, q, k, s);
  T[2u * i] = v.x;
  T[2u * i + 1u] = v.y;
}
`;

const name = (a: number, b: number) => `c${a < 0 ? 'm' : ''}${Math.abs(a)}_${b < 0 ? 'm' : ''}${Math.abs(b)}`;
const f32 = (v: number) => {
  const s = Math.fround(v).toPrecision(9);
  return s.includes('.') || s.includes('e') ? s : s + '.0';
};

// Primal seed level L (one invocation per probe and row): (rad, tr) of each ray. A chord through a cell of transmittance a
// adds e (1 − a^l), times what is left in front of it.
function seedShader(common: string, L: number) {
  const n = 1 << L, rays = seedRays(L);
  const cells = new Map<string, [number, number]>();
  for (const subs of rays) for (const c of subs) for (const [a, b] of c) cells.set(`${a},${b}`, [a, b]);
  const fetch = [...cells.values()].map(([a, b]) => `  let ${name(a, b)} = cell(x0 + ${a}, s + ${b});`);
  const body = rays.map((subs, k) => {
    const sigma = (2 * k - n) / n;
    // the world direction light leaves a cell in to travel along this ray, per frustum
    const th = [[1, sigma], [sigma, 1], [-1, sigma], [sigma, -1]].map(([x, y]) => f32(Math.atan2(-y, -x)));
    const lines = [`  {`, `    var ths = array<f32, 4>(${th.join(', ')}); let th = ths[P.dir];`, `    var rad = 0.0; var tr = 0.0;`];
    for (const c of subs) {
      lines.push(`    { var r = 0.0; var t = 1.0;`);
      for (const [a, b, len] of c) {
        const v = name(a, b), l = f32(len);
        lines.push(`      r += t * emit(${v}, th) * absorbW(${v}.y, ${l});`);
        lines.push(`      t *= powA(${v}.y, ${l});`);
      }
      lines.push(`      rad += r; tr += t; }`);
    }
    lines.push(`    storeP(${L}u, q, ${k}, s, vec2f(0.5 * rad, 0.5 * tr));`);
    lines.push('  }');
    return lines.join('\n');
  });
  return /* wgsl */ `${common}
${wgslPrimalT}
// (emission, transmittance, spot direction, spot half width); vacuum outside
fn cell(p: i32, s: i32) -> vec4f {
  if (!inside(p, s)) { return vec4f(0.0, 1.0, 0.0, -1.0); }
  let i = world(p, s);
  if (!has(SPOT)) { return vec4f(scene[PE * NN + i], scene[PAP * NN + i], 0.0, -1.0); }
  return vec4f(scene[PE * NN + i], scene[PAP * NN + i], scene[PDIR * NN + i], scene[PHALF * NN + i]);
}
// the share of a chord of length l that a cell of transmittance a absorbs, 1 − a^l
fn absorbW(a: f32, l: f32) -> f32 {
  if (a <= 0.0) { return 1.0; }
  if (a >= 1.0) { return 0.0; }
  let x = -l * log(a);
  return x * expRatio(x);
}
fn emit(c: vec4f, th: f32) -> f32 {
  if (c.w < 0.0) { return c.x; }
  let d = th - c.z;
  let wrapped = d - 6.2831853 * floor((d + 3.14159265) / 6.2831853);
  return select(0.0, c.x, abs(wrapped) <= c.w);
}

@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = i32(gid.x); let q = i32(gid.y) + lo();
  if (!rOk(${L}u, q, s)) { return; }
  let x0 = q * ${n};
${fetch.join('\n')}
${body.join('\n')}
}
`;
}

// Primal extension of level L = P.level from L − 1 (one invocation per ray pair 2h, 2h + 1)
const extendShader = (common: string) => /* wgsl */ `${common}
${wgslPrimalT}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let L = P.level;
  let half = 1 << (L - 1u);
  let pairs = half + 1;
  let s = i32(gid.x); let x = i32(gid.y);
  let row = x / pairs; let h = x - row * pairs; let Q = row + lo();
  if (!rOk(L, Q, s)) { return; }
  let p0 = 2 * Q;
  let offA = 2 * h - half;
  let a = loadP(L - 1u, p0, h, s);
  let b = loadP(L - 1u, p0 + 1, h, s + offA);
  storeP(L, Q, 2 * h, s, vec2f(a.x + b.x * a.y, a.y * b.y));
  if (h < half) {
    let c = loadP(L - 1u, p0 + 1, h + 1, s + offA);
    let d = loadP(L - 1u, p0, h + 1, s);
    let b2 = loadP(L - 1u, p0 + 1, h, s + 2 * (h + 1) - half);
    let cA = vec2f(a.x + c.x * a.y, a.y * c.y);
    let cB = vec2f(d.x + b2.x * d.y, d.y * b2.y);
    storeP(L, Q, 2 * h + 1, s, 0.5 * (cA + cB));
  }
}
`;

// Primal merge of level l = P.level (one invocation per probe, cone and row). Level 0 adds the probe's fluence to the
// pixel before it.
const mergeShader = (common: string) => /* wgsl */ `${common}
${wgslPrimalT}
fn loadR(l: u32, q: i32, c: i32, s: i32) -> f32 {
  if (has(TOP) || !rOk(l, q, s)) { return 0.0; }
  return R[P.rIn + rIndex(l, q, c, s)];
}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let l = P.level;
  let nd = 1 << l;
  let s = i32(gid.x); let x = i32(gid.y);
  let q = (x >> l) + lo(); let cone = x & (nd - 1);
  if (!rOk(l, q, s)) { return; }
  let even = (q % 2) == 0;
  let fs = select(1, 2, even);
  var r = 0.0;
  for (var j = 0; j < 2; j++) {
    let sub = 2 * cone + j;
    let ray = (sub + 1) >> 1;
    let off = 2 * ray - nd;
    let rr = loadP(l, q, ray, s);
    let w = tables[wbase(l) + u32(sub)];
    let far = loadR(l + 1u, (q + fs) >> 1, sub, s + off * fs);
    if (even) {
      let e = loadP(l, q + 1, ray, s + off);
      let coarse = loadR(l + 1u, q >> 1, sub, s);
      r += 0.5 * ((rr.x + e.x * rr.y) * w + far * rr.y * e.y + coarse);
    } else {
      r += rr.x * w + far * rr.y;
    }
  }
  if (l == 0u) {
    if (inside(q - 1, s)) { outF[world(q - 1, s)] += r; }
    return;
  }
  R[P.rOut + rIndex(l, q, cone, s)] = r;
}
`;

// World pixels (one invocation per pixel): the primal's image, the summed fluence with the cross blur (centre ×4, each
// 4-neighbour of like opacity ×1, clamped at the border).
const blurShader = (common: string) => /* wgsl */ `${common}
fn opacityAt(x: i32, y: i32) -> f32 { return 1.0 - scene[PAP * NN + u32(x * N + y)]; }
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let y = i32(gid.x); let x = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let i = u32(x * N + y);
  let o = opacityAt(x, y);
  var sum = 4.0 * outF[i]; var wt = 4.0;
  var nb = array<vec2i, 4>(vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
  for (var k = 0; k < 4; k++) {
    let c = clamp(vec2i(x, y) + nb[k], vec2i(0), vec2i(N - 1));
    if (abs(opacityAt(c.x, c.y) - o) < 0.5) { sum += outF[u32(c.x * N + c.y)]; wt += 1.0; }
  }
  outF[NN + i] = sum / wt;
}
`;

// ── the dual ──
// World pixels: the dual's image, from the cascade's splats, E + W (slot 0) mixed across y and N + S (slot 2) across x,
// each (2 own + the 4-neighbours on that axis of like opacity) / (2 + their count), plus the near field (slot 3), unmixed.
const endMixShader = (common: string) => /* wgsl */ `${common}
fn tAt(x: i32, y: i32) -> f32 { return scene[PAD * NN + u32(x * N + y)]; }
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let y = i32(gid.x); let x = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let i = u32(x * N + y);
  let t = tAt(x, y);
  var ew = 2.0 * outF[i]; var wy = 2.0;
  var ns = 2.0 * outF[2u * NN + i]; var wx = 2.0;
  if (y >= 1 && abs(tAt(x, y - 1) - t) < 0.5) { ew += outF[i - 1u]; wy += 1.0; }
  if (y + 1 < N && abs(tAt(x, y + 1) - t) < 0.5) { ew += outF[i + 1u]; wy += 1.0; }
  if (x >= 1 && abs(tAt(x - 1, y) - t) < 0.5) { ns += outF[2u * NN + i - u32(N)]; wx += 1.0; }
  if (x + 1 < N && abs(tAt(x + 1, y) - t) < 0.5) { ns += outF[2u * NN + i + u32(N)]; wx += 1.0; }
  outF[NN + i] = outF[3u * NN + i] + ew / wy + ns / wx;
}
`;

// World pixels: the spot source image of beam P.plane, in both layouts
const spotsShader = (common: string) => /* wgsl */ `${common}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let y = i32(gid.x); let x = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let j = u32(x * N + y);
  let e = scene[PE * NN + j];
  let u = select(0.0, e, e > 0.0 && scene[PHALF * NN + j] >= 0.0 && scene[PLOBE * NN + j] == f32(P.plane));
  src[P.plane * NN + j] = u;
  src[(P.srcPlanes + P.plane) * NN + u32(y * N + x)] = u;
}
`;

// ── the near field of solid faces ──
// A face of a light pixel, in its own frame: the pixel is [0, 1]², the face is a = 1, b ∈ [0, 1], with a along the outward
// normal and b along the facing frustum's +s. The face is a source at two source positions of ½ in each frustum it reaches,
// and the cascade picks a source position's light up at the line through the centre of the pixel after it (the hand-off line):
//   FN  facing frustum, at the light pixel    hand-off a = 1.5   directions within 45° of +a, weight cos, halves lo / hi
//   FF  facing frustum, at the pixel across   hand-off a = 2.5
//   SN  side frustum of axis +b, at (1, 0)     hand-off b = 1.5   directions within 45° of +b leaning to +a, weight sin
//   SF  side frustum of axis +b, at (1, 1)     hand-off b = 2.5
// and SN, SF mirrored in b for the side frustum of axis −b. The near field splats the face's own lines between the face and
// the hand-off line (track length per cell), each line stopping at the first opaque cell; the cascade is handed the share
// of the lines that reach the hand-off line. Both depend only on which cells of the piece's stencil (the cells its lines can
// enter) are opaque, so each is a table over that key: a splat per stencil cell, a hand-off transmittance per half.
// The integrals are exact to rounding: one line per interval of start v between corner crossings, and 12-point
// Gauss–Legendre per interval of angle between corner alignments.
interface NearPiece {
  axis: number; // 0: facing, 1: side
  cells: [number, number][]; // the stencil, face-local; the cell across the face (never opaque, not in the key) first
  halves: number;
  dep: number[]; // [key × cells + cell], × ½ for the source position
  T: number[]; // [key × halves + half]
}
const NEAR_PIECES: [string, number, number][] = [['FN', 0, 1.5], ['FF', 0, 2.5], ['SN', 1, 1.5], ['SF', 1, 2.5]];

function gaussLegendre(n: number): [number[], number[]] {
  const x: number[] = [], w: number[] = [];
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5)), dp = 1;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) [p0, p1] = [p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k];
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x.push(z);
    w.push(2 / ((1 - z * z) * dp * dp));
  }
  return [x, w];
}

/** the line from (1, v) along (ua, ub) to the hand-off line (a = h for a facing piece, b = h for a side piece): its cells
 * (a, b) and chords */
function nearLine(v: number, ua: number, ub: number, axis: number, h: number): [number, number, number][] {
  const tend = axis === 0 ? (h - 1) / ua : (h - v) / ub;
  let ia = 1, ib = Math.floor(v);
  const sb = ub > 0 ? 1 : -1;
  let tma = ua > 0 ? 1 / ua : 1e30, tmb = ub ? ((ub > 0 ? ib + 1 : ib) - v) / ub : 1e30;
  const tda = ua > 0 ? 1 / ua : 1e30, tdb = ub ? Math.abs(1 / ub) : 1e30;
  const out: [number, number, number][] = [];
  let t0 = 0;
  while (t0 < tend) {
    const t1 = Math.min(tma, tmb, tend);
    if (t1 > t0) out.push([ia, ib, t1 - t0]);
    if (t1 >= tend) break;
    if (tma < tmb) ((ia += 1), (tma += tda));
    else ((ib += sb), (tmb += tdb));
    t0 = t1;
  }
  return out;
}

function nearPiece(axis: number, h: number): NearPiece {
  const q = Math.PI / 4, lo = axis === 0 ? -q : 0, halves = axis === 0 ? 2 : 1;
  // the corners whose lines break v: cell corners, and where the hand-off line crosses the grid
  const pts: [number, number][] = [];
  for (let a = 1; a <= 4; a++) for (let b = -3; b <= 4; b++) pts.push([a, b]);
  if (axis === 0) for (let b = -3; b <= 4; b++) pts.push([h, b]);
  else for (let a = 1; a <= 4; a++) pts.push([a, h]);
  const ends: [number, number][] = [...pts, [1, 0], [1, 1]];
  const crit = new Set([lo, q, 0]);
  for (let i = 0; i < ends.length; i++)
    for (let j = i + 1; j < ends.length; j++) {
      let da = ends[j][0] - ends[i][0], db = ends[j][1] - ends[i][1];
      if (da < 0 || (da === 0 && db < 0)) ((da = -da), (db = -db));
      if (da === 0 && db === 0) continue;
      const ang = axis === 0 ? Math.atan2(db, da) : Math.atan2(da, db);
      if (ang > lo && ang < q) crit.add(ang);
    }
  const angles = [...crit].sort((x, y) => x - y);
  const [gx, gw] = gaussLegendre(12);
  const lines: { cells: [number, number, number][]; w: number; half: number }[] = [];
  for (let i = 0; i + 1 < angles.length; i++) {
    const a0 = angles[i], a1 = angles[i + 1];
    if (a1 - a0 < 1e-15) continue;
    for (let g = 0; g < gx.length; g++) {
      const ang = 0.5 * (a0 + a1) + 0.5 * (a1 - a0) * gx[g];
      const ua = axis === 0 ? Math.cos(ang) : Math.sin(ang), ub = axis === 0 ? Math.sin(ang) : Math.cos(ang);
      const w = 0.5 * (a1 - a0) * gw[g] * (axis === 0 ? Math.cos(ang) : Math.sin(ang));
      const vs = [0, 1];
      for (const [pa, pb] of pts) {
        const v = pb - ((pa - 1) * ub) / ua;
        if (v > 0 && v < 1) vs.push(v);
      }
      vs.sort((x, y) => x - y);
      for (let k = 0; k + 1 < vs.length; k++)
        if (vs[k + 1] > vs[k])
          lines.push({ cells: nearLine(0.5 * (vs[k] + vs[k + 1]), ua, ub, axis, h), w: w * (vs[k + 1] - vs[k]), half: axis === 0 && ang > 0 ? 1 : 0 });
    }
  }
  const seen = new Set(lines.flatMap((l) => l.cells.map(([a, b]) => `${a},${b}`)));
  seen.delete('1,0');
  const cells: [number, number][] = [[1, 0], ...[...seen].map((c) => c.split(',').map(Number) as [number, number]).sort((x, y) => x[0] - y[0] || x[1] - y[1])];
  const index = new Map(cells.map(([a, b], k) => [`${a},${b}`, k]));
  const nc = cells.length, keys = 1 << (nc - 1);
  const dep = new Array<number>(keys * nc).fill(0), T = new Array<number>(keys * halves).fill(0), wsum = [0, 0];
  for (const l of lines) {
    const idx = l.cells.map(([a, b]) => index.get(`${a},${b}`)!);
    wsum[l.half] += l.w;
    for (let key = 0; key < keys; key++) {
      let alive = 1;
      for (let c = 0; c < idx.length; c++) {
        if (idx[c] > 0 && (key >> (idx[c] - 1)) & 1) {
          alive = 0;
          break;
        }
        dep[key * nc + idx[c]] += l.w * l.cells[c][2];
      }
      T[key * halves + l.half] += l.w * alive;
    }
  }
  return { axis, cells, halves, dep: dep.map((v) => 0.5 * v), T: T.map((v, i) => v / wsum[i % halves]) };
}

let nearTables: Record<string, NearPiece> | null = null;
/** the four tables, and each one's offset (splats, then transmittances) in the table buffer (they start it) */
export function nearTable() {
  nearTables ??= Object.fromEntries(NEAR_PIECES.map(([n, axis, h]) => [n, nearPiece(axis, h)]));
  let at = 0;
  const offsets: Record<string, { dep: number; T: number }> = {};
  const values: number[] = [];
  for (const [n] of NEAR_PIECES) {
    const t = nearTables[n];
    offsets[n] = { dep: at, T: at + t.dep.length };
    values.push(...t.dep, ...t.T);
    at = values.length;
  }
  return { pieces: nearTables, offsets, values };
}

const TILE = 16, HALO = 6, SPAN = TILE + 2 * HALO; // a row of the tile's flags fits in 32 bits
const AXES: [number, number][] = [[1, 0], [0, 1], [-1, 0], [0, -1]]; // frustum axes E, N, W, S = face normals
const SPLUS: [number, number][] = [[0, 1], [1, 0], [0, 1], [1, 0]]; // each frustum's +s

// World pixels, TILE × TILE per workgroup: the solid lights' sources and near field. Per pixel and frustum d, four source
// planes (facing lo, hi; side lo, hi) at P.plane + 4 (d >> 1) + j, written in d's own layout only; and the near field, in
// slot 3 of the fluence. Every face that can reach the pixel lies within 3 px of it and its keys within 6, so the tile holds
// opacity, 'is a light' and 'is free' as bits per row over a 6-px halo, and the emission.
function nearFacesShader(common: string) {
  const { pieces, offsets } = nearTable();
  const used = { o: new Set<number>(), l: new Set<number>(), f: new Set<number>() };
  const nm = (v: number) => (v < 0 ? `m${-v}` : `${v}`);
  const bit = (k: 'o' | 'l' | 'f', dx: number, dy: number) => (used[k].add(dx), `((${k}${nm(dx)} >> ${dy + HALO}u) & 1u)`);
  const at = (F: number[], f: number, a: number, b: number) => [F[0] + a * AXES[f][0] + b * SPLUS[f][0], F[1] + a * AXES[f][1] + b * SPLUS[f][1]];
  // the face of normal f at offset F: exposed (a light, the pixel across free), its emission, a piece's key
  const exposed = (F: number[], f: number) => `${bit('l', F[0], F[1])} == 1u && ${bit('f', F[0] + AXES[f][0], F[1] + AXES[f][1])} == 1u`;
  const emission = (F: number[]) => `em[${(HALO + F[0]) * SPAN + HALO + F[1]} + (lx * ${SPAN} + ly)]`;
  const key = (piece: string, sg: number, F: number[], f: number) =>
    pieces[piece].cells.slice(1).map(([a, b], i) => `(${bit('o', ...(at(F, f, a, sg * b) as [number, number]))} << ${i}u)`).join(' | ');
  const T = (piece: string, k: string, half: number) => `tables[${offsets[piece].T}u + (${k}) * ${pieces[piece].halves}u + ${half}u]`;
  const eq = (u: number[], v: number[]) => u[0] === v[0] && u[1] === v[1];

  // sources: plane j of frustum d
  const planes: string[] = [];
  for (let d = 0; d < 4; d++)
    for (let j = 0; j < 4; j++) {
      const terms: [string, number[], number, number, number][] = []; // piece, face offset, normal, mirror, half
      if (j < 2) {
        terms.push(['FN', [0, 0], d, 1, j], ['FF', [-AXES[d][0], -AXES[d][1]], d, 1, j]);
      } else {
        const m = j === 2 ? [-SPLUS[d][0], -SPLUS[d][1]] : SPLUS[d], f = AXES.findIndex((n) => eq(n, m));
        const sg = eq(AXES[d], SPLUS[f]) ? 1 : -1;
        terms.push(['SN', [-m[0], -m[1]], f, sg, 0], ['SF', [-m[0] - AXES[d][0], -m[1] - AXES[d][1]], f, sg, 0]);
      }
      const body = terms.map(([piece, F, f, sg, half]) => `    if (${exposed(F, f)}) { u += 0.5 * ${emission(F)} * ${T(piece, key(piece, sg, F, f), half)}; }`);
      const index = d % 2 === 0 ? `(P.plane + ${4 * (d >> 1) + j}u) * NN + u32(x * N + y)` : `(P.srcPlanes + P.plane + ${4 * (d >> 1) + j}u) * NN + u32(y * N + x)`;
      planes.push(`  { var u = 0.0;\n${body.join('\n')}\n    src[${index}] = u; }`);
    }

  // the near field: every (face, piece, stencil cell) that lands on this pixel, grouped by face
  const near: string[] = [];
  for (let f = 0; f < 4; f++) {
    const n = AXES[f], s = SPLUS[f];
    const inst: [string, number][] = [['FN', 1], ['FF', 1], ['SN', 1], ['SF', 1], ['SN', -1], ['SF', -1]];
    const byFace = new Map<string, { F: number[]; terms: string[] }>();
    for (const [piece, sg] of inst)
      pieces[piece].cells.forEach(([a, b], k) => {
        const F = [-(a * n[0] + sg * b * s[0]), -(a * n[1] + sg * b * s[1])];
        const id = F.join(',');
        if (!byFace.has(id)) byFace.set(id, { F, terms: [] });
        byFace.get(id)!.terms.push(`tables[${offsets[piece].dep}u + (${key(piece, sg, F, f)}) * ${pieces[piece].cells.length}u + ${k}u]`);
      });
    for (const { F, terms } of byFace.values()) near.push(`  if (${exposed(F, f)}) { nd += ${emission(F)} * (${terms.join(' + ')}); }`);
  }
  // the rows of the 13 × 13 neighbourhood that the pixel reads, as bits dy + HALO
  const rows = (k: 'o' | 'l' | 'f', arr: string) =>
    [...used[k]].sort((a, b) => a - b).map((dx) => `  let ${k}${nm(dx)} = (atomicLoad(&${arr}[u32(lx + ${HALO + dx})]) >> u32(ly)) & ${(1 << (2 * HALO + 1)) - 1}u;`);
  return /* wgsl */ `${common}
var<workgroup> rowOpaque: array<atomic<u32>, ${SPAN}>;
var<workgroup> rowLight: array<atomic<u32>, ${SPAN}>;
var<workgroup> rowFree: array<atomic<u32>, ${SPAN}>;
var<workgroup> em: array<f32, ${SPAN * SPAN}>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let x0 = i32(wid.y) * ${TILE} - ${HALO}; let y0 = i32(wid.x) * ${TILE} - ${HALO};
  // the tile: row r is world x = x0 + r, bit c of a row is y = y0 + c; outside the world nothing is set
  for (var t = li; t < ${SPAN * SPAN}u; t += ${TILE * TILE}u) {
    let r = t / ${SPAN}u; let c = t % ${SPAN}u;
    let x = x0 + i32(r); let y = y0 + i32(c);
    var e = 0.0;
    if (x >= 0 && x < N && y >= 0 && y < N) {
      let j = u32(x * N + y);
      e = scene[PE * NN + j];
      let a = scene[PAD * NN + j];
      if (a <= 0.0) { atomicOr(&rowOpaque[r], 1u << c); }
      if (e > 0.0 && scene[PHALF * NN + j] < 0.0) { atomicOr(&rowLight[r], 1u << c); }
      if (a > 0.0 && e == 0.0) { atomicOr(&rowFree[r], 1u << c); }
    }
    em[t] = e;
  }
  workgroupBarrier();
  let lx = i32(lid.y); let ly = i32(lid.x);
  let x = x0 + ${HALO} + lx; let y = y0 + ${HALO} + ly;
  if (x >= N || y >= N) { return; }
${[...rows('o', 'rowOpaque'), ...rows('l', 'rowLight'), ...rows('f', 'rowFree')].join('\n')}
${planes.join('\n')}
  var nd = 0.0;
${near.join('\n')}
  outF[3u * NN + u32(x * N + y)] = nd;
}
`;
}

// The transmittance table shared by a frustum and its opposite (E with W, N with S). It holds E's (N's) segments: level l,
// m = −1 .. (N >> l) − 1 (the segment from cell m·2^l to cell (m + 1)·2^l), ray k, and every start row s from which the
// segment has an end inside the row range, at s + max(0, 2k − 2^l) in a row of N + 2^l. Each frustum reads only the rays
// that start inside its own row range; every path is monotone in row, so those values are the segments' true
// transmittances, the same as a table of its own.
const wgslShared = (N: number, soff: number[]) => /* wgsl */ `
var<private> SOFF: array<u32, ${soff.length}> = array<u32, ${soff.length}>(${soff.map((v) => `${v}u`).join(', ')});
fn sOk(l: u32, m: i32, k: i32, s: i32) -> bool {
  let n = 1 << l;
  let e = s + 2 * k - n;
  return m >= -1 && m < (N >> l) && k >= 0 && k <= n && ((s >= 0 && s < N) || (e >= 0 && e < N));
}
fn sIndex(l: u32, m: i32, k: i32, s: i32) -> u32 {
  let n = 1 << l;
  return SOFF[l] + u32(((m + 1) * (n + 1) + k) * (N + n) + s + max(0, 2 * k - n));
}
fn trS(l: u32, m: i32, k: i32, s: i32) -> f32 {
  if (!sOk(l, m, k, s)) { return 1.0; }
  return T[sIndex(l, m, k, s)];
}
// this frustum's ray (q, k, s): 1 if it starts outside the frustum's row range; W and S read the reversed segment
fn trD(l: u32, q: i32, k: i32, s: i32) -> f32 {
  if (!tOk(l, q, k, s)) { return 1.0; }
  if (P.dir < 2u) { return trS(l, q, k, s); }
  let n = 1 << l;
  return trS(l, (N >> l) - 1 - q, n - k, s + 2 * k - n);
}
// a segment's transmittance from level l − 1 (the primal's extension on transmittance alone)
fn trBuild(m: i32, k: i32, s: i32) -> f32 {
  let l = P.level - 1u;
  let half = 1 << l;
  let h = k >> 1;
  let offA = 2 * h - half;
  let a = trS(l, 2 * m, h, s);
  if ((k & 1) == 0) { return a * trS(l, 2 * m + 1, h, s + offA); }
  return 0.5 * (a * trS(l, 2 * m + 1, h + 1, s + offA) + trS(l, 2 * m, h + 1, s) * trS(l, 2 * m + 1, h, s + 2 * (h + 1) - half));
}`;

// The table's seed segments of level L (E and N only; one invocation per segment m and start row s, all its rays k from
// one load of the cells)
function tableSeedShader(common: string, shared: string, L: number) {
  const n = 1 << L, rays = seedRays(L);
  const cells = new Map<string, [number, number]>();
  for (const subs of rays) for (const c of subs) for (const [a, b] of c) cells.set(`${a},${b}`, [a, b]);
  const fetch = [...cells.values()].map(([a, b]) => `  let ${name(a, b)} = sceneAt(PAD, x0 + ${a}, s + ${b}, 1.0);`);
  const body = rays.map((subs, k) => {
    const lines = subs.map((c, i) => `  var t${k}_${i} = 1.0;\n` + c.map(([a, b, len]) => `  t${k}_${i} *= powA(${name(a, b)}, ${f32(len)});`).join('\n'));
    return `  if (sOk(${L}u, m, ${k}, s)) {\n${lines.join('\n')}\n  T[sIndex(${L}u, m, ${k}, s)] = 0.5 * (t${k}_0 + t${k}_1);\n  }`;
  });
  return /* wgsl */ `${common}
${shared}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = i32(gid.x) - ${n}; let m = i32(gid.y) - 1;
  if (s >= N + ${n} || m >= (N >> ${L}u)) { return; }
  let x0 = m * ${n};
${fetch.join('\n')}
${body.join('\n')}
}
`;
}

// The pattern-free cone value R~ of source group c at level l for sub-cone j (cone j >> 1), 0 outside. Level 0 reads the source of
// pixel q − 1 (lower or upper sub-cone's plane). With ray sums (cbase given) the cone values of level l ≥ 1 are kept only
// until level l + 1: two levels, by parity, from R[cbase].
const wgslCone = (level0: boolean, cbase: number | null = null) => /* wgsl */ `
fn cone(ch: Chan, c: u32, l: u32, q: i32, j: i32, s: i32) -> f32 {
  if (!rOk(l, q, s)) { return 0.0; }
  ${
    level0
      ? `if (!inside(q - 1, s)) { return 0.0; }
  return src[select(ch.hi, ch.lo, j == 0) * NN + srcIndex(q - 1, s)];`
      : cbase === null
        ? `return R[(l - 1u) * P.rIn + c * NN + rIndex(l, q, j >> 1, s)];`
        : `return R[${cbase}u + (l & 1u) * P.rIn + c * NN + rIndex(l, q, j >> 1, s)];`
  }
}`;

// The splat D_l(p, h, s) of a ray is its merge part, from the kept cone values (sub-cones 2h − 1, 2h of probe p, × ½ on
// even plane; on odd plane also those of the even plane p − 1 through this ray, × its transmittance), plus, when ext, extension
// transposed from D_{l+1}. Odd p's far-half shares and the merge part through p − 1 use the same transmittances.
const wgslDeposit = () => /* wgsl */ `
fn rd(l: u32, q: i32, k: i32, s: i32) -> f32 {
  if (!tOk(l, q, k, s)) { return 0.0; }
  return T[DBASE + tIndex(l, q, k, s)];
}
fn deposit(l: u32, p: i32, h: i32, s: i32, ext: bool) -> f32 {
  let half = 1 << l;
  let even = (p & 1) == 0;
  let ss = s - (2 * h - half);
  var own = 0.0; var prev = 0.0;
  // A loop bounded by a uniform (P.chanCount) is compiled with a loop guard and not unrolled, and this pass is bound by its
  // instruction count, so the bound is fixed.
  for (var c = 0u; c < ${MAX_CHANNELS}u; c++) {
    if (c >= P.chanCount) { break; }
    let ch = P.chans[c];
    let wt = ch.w + wbase(l);
    // sub-cones 2h − 1 (none for h = 0) and 2h (none for h = 2^l)
    if (h >= 1) {
      let w = tables[wt + u32(2 * h - 1)];
      own += w * cone(ch, c, l, p, 2 * h - 1, s);
      if (!even) { prev += w * cone(ch, c, l, p - 1, 2 * h - 1, ss); }
    }
    if (h < half) {
      let w = tables[wt + u32(2 * h)];
      own += w * cone(ch, c, l, p, 2 * h, s);
      if (!even) { prev += w * cone(ch, c, l, p - 1, 2 * h, ss); }
    }
  }
  return depositRest(l, p, h, s, ext, own, prev);
}
${wgslDepositRest}`;

// With ray sums: the merge parts are the kept sums S_l(p, h, s) and S_l(p − 1, h, ss), at R[tIndex].
const wgslDepositSums = () => /* wgsl */ `
fn rd(l: u32, q: i32, k: i32, s: i32) -> f32 {
  if (!tOk(l, q, k, s)) { return 0.0; }
  return T[DBASE + tIndex(l, q, k, s)];
}
fn sumAt(l: u32, q: i32, k: i32, s: i32) -> f32 {
  if (!tOk(l, q, k, s)) { return 0.0; }
  return R[tIndex(l, q, k, s)];
}
fn deposit(l: u32, p: i32, h: i32, s: i32, ext: bool) -> f32 {
  let ss = s - (2 * h - (1 << l));
  return depositRest(l, p, h, s, ext, sumAt(l, p, h, s), select(0.0, sumAt(l, p - 1, h, ss), (p & 1) == 1));
}
${wgslDepositRest}`;

// The rest of the splat from the merge parts own (probe p) and prev (probe p − 1, odd plane only)
const wgslDepositRest = /* wgsl */ `
fn depositRest(l: u32, p: i32, h: i32, s: i32, ext: bool, own: f32, prev: f32) -> f32 {
  let half = 1 << l;
  let even = (p & 1) == 0;
  let ss = s - (2 * h - half);
  let L = l + 1u;
  let Q = p >> 1;
  if (even) {
    var v = 0.5 * own;
    if (ext) {
      v += rd(L, Q, 2 * h, s);
      if (h < half) { v += 0.5 * rd(L, Q, 2 * h + 1, s); }
      if (h >= 1) { v += 0.5 * rd(L, Q, 2 * h - 1, s); }
    }
    return v;
  }
  var near = 0.5 * prev;
  if (ext) { near += rd(L, Q, 2 * h, ss); }
  var v = own + trD(l, p - 1, h, ss) * near;
  if (ext) {
    if (h >= 1) { let s1 = s - (2 * (h - 1) - half); v += 0.5 * trD(l, p - 1, h - 1, s1) * rd(L, Q, 2 * h - 1, s1); }
    if (h < half) { let s2 = s - (2 * (h + 1) - half); v += 0.5 * trD(l, p - 1, h + 1, s2) * rd(L, Q, 2 * h + 1, s2); }
  }
  return v;
}`;

// The dual's extensions at level l = P.level ≥ 1: every source group's cone value R~_l(q, k, s), carried from level l − 1 (the far reads of even
// probe 2q − 2 and odd 2q − 1, the near-plane read of even plane 2q), and kept. Variants:
//   'level1'  one invocation per level-0 ray (p, h) and row: also makes level 0's splats, which need nothing from
//             above (level 0 is a seed: nothing is built from it); rows below N carry cone (x >> 1, x & 1).
//   'build'   E and N, l ≥ 2: one invocation per table entry (segment m, ray k, start row), which it builds from level
//             l − 1; entries that are this frustum's own rays (m ≥ 0, start inside, k < 2^l) also carry their cone.
//   'carry'   W and S, l ≥ 2: one invocation per ray; the transmittance comes from the opposite frustum's table.
const dualExtensionsShader = (common: string, shared: string, variant: 'level1' | 'build' | 'carry') => /* wgsl */ `${common}
${shared}
${wgslCone(variant === 'level1')}
${variant === 'level1' ? wgslDeposit() : ''}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let l = P.level;
  let nd = 1 << l;
  let x = i32(gid.y);
  ${
    {
      level1: `let s = i32(gid.x);
  if (s >= N || x >= 2 * N) { return; }
  let p = (x >> 1) + lo();
  T[DBASE + tIndex(0u, p, x & 1, s)] = deposit(0u, p, x & 1, s, false);
  if (x >= N) { return; }
  let q = p; let k = x & 1;`,
      build: `let m = x / (nd + 1) - 1; let k = x - (m + 1) * (nd + 1);
  let s = i32(gid.x) - max(0, 2 * k - nd);
  if (m >= (N >> l) || i32(gid.x) >= N + nd) { return; }
  if (sOk(l, m, k, s)) { T[sIndex(l, m, k, s)] = trBuild(m, k, s); }
  let q = m;
  if (!rOk(l, q, s) || k >= nd) { return; }`,
      carry: `let s = i32(gid.x);
  let a = x / (nd + 1); let k = x - a * (nd + 1); let q = a + lo();
  if (!rOk(l, q, s) || k >= nd) { return; }`,
    }[variant]
  }
  let lb = l - 1u;
  let ray = (k + 1) >> 1;
  let off = 2 * ray - (nd >> 1);
  let trOdd = trD(lb, 2 * q - 1, ray, s - off);
  let trEven = trD(lb, 2 * q - 2, ray, s - 2 * off);
  for (var c = 0u; c < ${MAX_CHANNELS}u; c++) {
    if (c >= P.chanCount) { break; }
    let ch = P.chans[c];
    let v = 0.5 * trEven * trOdd * cone(ch, c, lb, 2 * q - 2, k, s - 2 * off) + trOdd * cone(ch, c, lb, 2 * q - 1, k, s - off) + 0.5 * cone(ch, c, lb, 2 * q, k, s);
    R[lb * P.rIn + c * NN + rIndex(l, q, k, s)] = v;
  }
}
`;

// The dual's extensions with ray sums, level l = P.level ≥ 1: one invocation per probe q, run of SUMS_RUN cones and row s. It walks its cones in
// order (with the one before the run), makes every source group's cone value as the dual's extensions does and writes it for level l + 1,
// and writes the merge-part sum of each ray it owns: ray k takes sub-cone 2k − 1 of cone k − 1 and sub-cone 2k of cone k,
// S_l(q, k, s) = Σ_c w_c(2k − 1) R~_c(q, k − 1, s) + w_c(2k) R~_c(q, k, s); the run that ends the probe also writes ray 2^l.
// With build (E and N, l ≥ 2) the invocations reach from segment −1 and to row N + 2^l, and also build the shared table's
// level l: entry (q, k, s − max(0, 2k − 2^l)) for each k of the run (and k = 2^l in the last run).
const SUMS_RUN = 4;
const dualExtensionsSumsShader = (common: string, shared: string, cbase: number, level0: boolean, build: boolean) => /* wgsl */ `${common}
${shared}
${wgslCone(level0, cbase)}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let l = P.level;
  let nd = 1 << l;
  let run = min(${SUMS_RUN}, nd);
  let runs = nd / run;
  let s = i32(gid.x); let x = i32(gid.y);
  ${
    build
      ? `let q = x / runs - 1; let m = x - (q + 1) * runs;
  if (q >= (N >> l) || s >= N + nd) { return; }
  let k0 = m * run; let k1 = k0 + run;
  for (var k = k0; k <= select(k1 - 1, nd, k1 == nd); k++) {
    let st = s - max(0, 2 * k - nd);
    if (sOk(l, q, k, st)) { T[sIndex(l, q, k, st)] = trBuild(q, k, st); }
  }`
      : `let q = x / runs + lo(); let m = x - (q - lo()) * runs;
  let k0 = m * run; let k1 = k0 + run;`
  }
  if (!rOk(l, q, s)) { return; }
  let lb = l - 1u;
  let wb = wbase(l);
  let out = ${cbase}u + (l & 1u) * P.rIn;
  var prev: array<f32, ${MAX_CHANNELS}>; // cone k − 1, per source group
  for (var k = max(k0 - 1, 0); k < k1; k++) {
    let ray = (k + 1) >> 1;
    let off = 2 * ray - (nd >> 1);
    let trOdd = trD(lb, 2 * q - 1, ray, s - off);
    let trEven = trD(lb, 2 * q - 2, ray, s - 2 * off);
    let own = k >= k0;
    var sum = 0.0;
    for (var c = 0u; c < ${MAX_CHANNELS}u; c++) {
      if (c >= P.chanCount) { break; }
      let ch = P.chans[c];
      let v = 0.5 * trEven * trOdd * cone(ch, c, lb, 2 * q - 2, k, s - 2 * off) + trOdd * cone(ch, c, lb, 2 * q - 1, k, s - off) + 0.5 * cone(ch, c, lb, 2 * q, k, s);
      if (own) {
        R[out + c * NN + rIndex(l, q, k, s)] = v;
        sum += tables[ch.w + wb + u32(2 * k)] * v;
        if (k >= 1) { sum += tables[ch.w + wb + u32(2 * k - 1)] * prev[c]; }
      }
      prev[c] = v;
    }
    if (own) { R[tIndex(l, q, k, s)] = sum; }
  }
  if (k1 == nd) {
    var sum = 0.0;
    for (var c = 0u; c < ${MAX_CHANNELS}u; c++) {
      if (c >= P.chanCount) { break; }
      sum += tables[P.chans[c].w + wb + u32(2 * nd - 1)] * prev[c];
    }
    R[tIndex(l, q, nd, s)] = sum;
  }
}
`;

// Level 0's splats with ray sums (one invocation per level-0 ray); without ray sums the level-1 dual extensions make them
const deposit0Shader = (common: string, shared: string) => /* wgsl */ `${common}
${shared}
${wgslCone(true)}
${wgslDeposit()}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = i32(gid.x); let x = i32(gid.y);
  if (s >= N || x >= 2 * N) { return; }
  let p = (x >> 1) + lo();
  T[DBASE + tIndex(0u, p, x & 1, s)] = deposit(0u, p, x & 1, s, false);
}
`;

// Dual gathering at level l = P.level ≥ 1 (one invocation per probe p, ray h and row s): the ray's splat, with extension
// transposed from the level above except at the top. Its merge part comes from the kept cone values, or with ray sums
// from the kept sums.
const dualGatheringShader = (common: string, shared: string, sums: boolean) => /* wgsl */ `${common}
${shared}
${sums ? wgslDepositSums() : wgslCone(false) + wgslDeposit()}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let l = P.level;
  let s = i32(gid.x); let x = i32(gid.y);
  let a = x / ((1 << l) + 1); let h = x - a * ((1 << l) + 1); let p = a + lo();
  if (!rOk(l, p, s)) { return; }
  T[DBASE + tIndex(l, p, h, s)] = deposit(l, p, h, s, !has(TOP));
}
`;

// Seed transposed (one invocation per frustum cell): the cell sums, over the seed rays of levels 0 and 1 that cross it,
// ½ D(probe, ray) × cos θ_k × the sum over the ray's chords in the cell of the transmittance in front × the weighted track
// length, and adds it to its world pixel: E and W into slot 0, N and S into slot 2. Each ray's splat is read once, for both
// its sub-lines.
function seedDepositShader(common: string) {
  const offsets = new Map<string, [number, number]>();
  const cellName = (dx: number, dy: number) => (offsets.set(`${dx},${dy}`, [dx, dy]), name(dx, dy));
  type Chord = { before: [number, number, number][]; len: number };
  // the share of ray k of level L that cell (p, s) takes, its chords starting (a, b) from the ray's start
  const share = (L: number, a: number, b: number, k: number, chords: Chord[]) => {
    const n = 1 << L, cos = n / Math.hypot(n, 2 * k - n);
    const sum = chords.map((c) => [...c.before.map(([oa, ob, lj]) => `powA(${cellName(oa, ob)}, ${f32(lj)})`), `trackW(a0, ${f32(c.len)})`].join(' * '));
    return `    { let q = (p - ${a}) / ${n}; let ss = s - ${b};
      if (rOk(${L}u, q, ss)) { acc += T[DBASE + tIndex(${L}u, q, ${k}, ss)] * (${sum.join(' + ')}) * ${f32(cos)}; } }`;
  };
  const blocks: string[] = [];
  for (const L of [0, 1]) {
    const n = 1 << L, rays = seedRays(L);
    const crossing = new Map<string, { a: number; b: number; k: number; chords: Chord[] }>(); // by the cell's offset from the ray's start
    rays.forEach((subs, k) => {
      for (const chords of subs)
        chords.forEach(([a, b, len], i) => {
          const key = `${a},${b},${k}`;
          if (!crossing.has(key)) crossing.set(key, { a, b, k, chords: [] });
          crossing.get(key)!.chords.push({ before: chords.slice(0, i).map(([aj, bj, lj]) => [aj - a, bj - b, lj]), len });
        });
    });
    const byParity: string[][] = [[], []];
    for (const { a, b, k, chords } of crossing.values()) byParity[((a % n) + n) % n].push(share(L, a, b, k, chords));
    if (n === 1) blocks.push(byParity[0].join('\n'));
    else blocks.push(`  if ((p & 1) == 0) {\n${byParity[0].join('\n')}\n  } else {\n${byParity[1].join('\n')}\n  }`);
  }
  offsets.set('0,0', [0, 0]);
  const loads = [...offsets.values()].map(([dx, dy]) => `  let ${name(dx, dy)} = sceneAt(PAD, p + ${dx}, s + ${dy}, 1.0);`);
  return /* wgsl */ `${common}
// the track length of a chord of length l through a cell of transmittance a, weighted by what is left in front of each point
fn trackW(a: f32, l: f32) -> f32 {
  if (a >= 1.0) { return l; }
  if (a <= 0.0) { return 0.0; }
  return l * expRatio(-l * log(a));
}
@compute @workgroup_size(${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = i32(gid.x); let p = i32(gid.y) + lo();
  if (!inside(p, s)) { return; }
${loads.join('\n')}
  let a0 = ${name(0, 0)};
  var acc = 0.0;
${blocks.join('\n')}
  outF[select(0u, 2u * NN, (P.dir & 1u) == 1u) + world(p, s)] += 0.5 * acc;
}
`;
}

// ── engine ──
export class DualHRC {
  readonly device: GPUDevice;
  readonly N: number;
  readonly levels: number;
  readonly options: DualOptions;
  profiler: Profiler | null = null;
  /** final fluence, world [x * N + y], at outputOffset */
  readonly outBuffer: GPUBuffer;
  readonly outputOffset: number;
  /** scene planes (emission, primal and dual transmittance, spot direction, half width, beam), world [plane * N² + x * N + y] */
  readonly sceneBuffer: GPUBuffer;
  /** beams the last scene's spots use (distinct direction and half width), and spots dropped beyond MAX_LOBES */
  lobes: [number, number][] = [];
  droppedSpots = 0;

  readonly #dual: boolean;
  readonly #toff: number[];
  /** the dual's shared transmittance table: offset of each level, and the whole size (floats) */
  readonly #soff: number[];
  readonly #layout: GPUBindGroupLayout;
  #buffers: Record<'params' | 'T' | 'R' | 'tables' | 'src', GPUBuffer>;
  #pipelines: Record<string, GPUComputePipeline>;
  #bindGroup: GPUBindGroup;
  #program: Dispatch[][] = []; // compute passes, each a list of dispatches
  #passLabels: string[] = [];
  #channels: Channel[] = [];

  constructor(device: GPUDevice, N: number, options: DualOptions) {
    this.device = device;
    this.N = N;
    this.options = { ...options };
    this.levels = ceilLog2(N);
    const nl = this.levels, dual = options.form === 'dual';
    this.#dual = dual;
    this.#toff = [0];
    for (let l = 0; l < nl; l++) this.#toff.push(this.#toff[l] + (N >> l) * ((1 << l) + 1) * N);
    this.#soff = [0];
    for (let l = 0; l < nl; l++) this.#soff.push(this.#soff[l] + ((N >> l) + 1) * ((1 << l) + 1) * (N + (1 << l)));
    const tsize = this.#toff[nl], ssize = this.#soff[nl];

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, bytes: number, usage = storage) => device.createBuffer({ label, size: Math.max(16, bytes), usage });
    this.sceneBuffer = buffer('dual-hrc scene', 6 * N * N * 4);
    // the summed fluence and the final image; the dual also keeps N + S's splats and the near field
    this.outBuffer = buffer('dual-hrc fluence', (dual ? 4 : 2) * N * N * 4);
    this.outputOffset = N * N * 4;
    this.#buffers = {
      params: buffer('dual-hrc params', PARAM_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      // primal: (rad, tr) per ray; dual: the shared transmittance table, then a splat per ray
      T: buffer('dual-hrc rays', (dual ? ssize + tsize : 2 * tsize) * 4),
      R: buffer('dual-hrc cones', 2 * N * N * 4),
      tables: buffer('dual-hrc tables', 16),
      src: buffer('dual-hrc sources', 16),
    };

    const common = wgslCommon(N, nl, this.#toff, ssize);
    const shared = wgslShared(N, this.#soff.slice(0, nl));
    const layout = device.createBindGroupLayout({
      label: 'dual-hrc',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAM_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.#layout = layout;
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (label: string, code: string) =>
      device.createComputePipeline({ label: `dual-hrc ${label}`, layout: pipelineLayout, compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' } });
    this.#pipelines = dual
      ? {
          spots: pipeline('spots', spotsShader(common)),
          nearFaces: pipeline('near faces', nearFacesShader(common)),
          table0: pipeline('table 0', tableSeedShader(common, shared, 0)),
          table1: pipeline('table 1', tableSeedShader(common, shared, 1)),
          extensions1: pipeline('extensions 1', dualExtensionsShader(common, shared, 'level1')),
          extensionsBuild: pipeline('extensions, building the table', dualExtensionsShader(common, shared, 'build')),
          extensionsCarry: pipeline('extensions', dualExtensionsShader(common, shared, 'carry')),
          gathering: pipeline('gathering', dualGatheringShader(common, shared, false)),
          ...(options.raySums
            ? {
                deposit0: pipeline('deposit 0', deposit0Shader(common, shared)),
                extensionsSums1: pipeline('extensions 1, ray sums', dualExtensionsSumsShader(common, shared, tsize, true, false)),
                extensionsSumsBuild: pipeline('extensions, ray sums, building the table', dualExtensionsSumsShader(common, shared, tsize, false, true)),
                extensionsSums: pipeline('extensions, ray sums', dualExtensionsSumsShader(common, shared, tsize, false, false)),
                gatheringSums: pipeline('gathering, ray sums', dualGatheringShader(common, shared, true)),
              }
            : {}),
          depositSeed: pipeline('deposit seed', seedDepositShader(common)),
          endMix: pipeline('end mix', endMixShader(common)),
        }
      : {
          seed0: pipeline('seed 0', seedShader(common, 0)),
          seed1: pipeline('seed 1', seedShader(common, 1)),
          extend: pipeline('extend', extendShader(common)),
          merge: pipeline('merge', mergeShader(common)),
          blur: pipeline('blur', blurShader(common)),
        };
    this.#bindGroup = this.#makeBindGroup();
  }

  #makeBindGroup() {
    const b = this.#buffers;
    return this.device.createBindGroup({
      layout: this.#layout,
      entries: [
        { binding: 0, resource: { buffer: b.params, size: PARAM_BYTES } },
        { binding: 1, resource: { buffer: this.sceneBuffer } },
        { binding: 2, resource: { buffer: b.T } },
        { binding: 3, resource: { buffer: b.R } },
        { binding: 4, resource: { buffer: b.tables } },
        { binding: 5, resource: { buffer: b.src } },
        { binding: 6, resource: { buffer: this.outBuffer } },
      ],
    });
  }

  destroy() {
    for (const b of [this.sceneBuffer, this.outBuffer, ...Object.values(this.#buffers)]) b.destroy();
  }

  /** The scene planes (on the CPU), the source groups and their tables, and the dispatch list. */
  upload(scene: DualScene) {
    const NN = this.N * this.N;
    const planes = new Float32Array(6 * NN);
    const E = planes.subarray(0, NN), AP = planes.subarray(NN, 2 * NN), AD = planes.subarray(2 * NN, 3 * NN);
    const DIR = planes.subarray(3 * NN, 4 * NN), HALF = planes.subarray(4 * NN, 5 * NN), LOBE = planes.subarray(5 * NN, 6 * NN);
    const sdir = scene.spotDir ?? new Float32Array(NN), shalf = scene.spotHalf ?? new Float32Array(NN).fill(-1);
    const beamIds = new Map<string, number>();
    this.lobes = [];
    this.droppedSpots = 0;
    HALF.fill(-1);
    LOBE.fill(-1);
    for (let i = 0; i < NN; i++) {
      const e = scene.E[i], a = scene.A[i];
      if (!(e > 0)) {
        AP[i] = AD[i] = a;
        continue;
      }
      // a light: opaque in both forms, unless it is a spot (a transparent point source in the dual)
      let spot = shalf[i] >= 0;
      if (spot) {
        const key = `${sdir[i]},${shalf[i]}`;
        if (!beamIds.has(key) && beamIds.size < MAX_LOBES) (beamIds.set(key, beamIds.size), this.lobes.push([sdir[i], shalf[i]]));
        const id = beamIds.get(key);
        if (id === undefined) {
          this.droppedSpots++;
          AP[i] = AD[i] = a;
          continue;
        }
        LOBE[i] = id;
        DIR[i] = sdir[i];
        HALF[i] = shalf[i];
      }
      E[i] = e;
      AP[i] = 0;
      AD[i] = spot ? 1 : 0;
    }
    this.device.queue.writeBuffer(this.sceneBuffer, 0, planes);
    this.#channels = this.#dual ? this.#makeChannels() : [];
    this.#build(this.lobes.length > 0);
  }

  // the dual's source groups: the faces (cos and |sin| beams), from the near faces pass's planes, then one per spot beam
  #makeChannels(): Channel[] {
    const nl = this.levels, L = this.lobes.length;
    const plane = (d: number, j: number) => L + 4 * (d >> 1) + j;
    const cos = depositTable(xlenCos, nl), sin = depositTable(xlenAbsSin, nl);
    const out: Channel[] = [
      { label: 'facing faces', dirs: [0, 1, 2, 3].map((d) => ({ lo: plane(d, 0), hi: plane(d, 1), table: cos })) },
      { label: 'side faces', dirs: [0, 1, 2, 3].map((d) => ({ lo: plane(d, 2), hi: plane(d, 3), table: sin })) },
    ];
    this.lobes.forEach(([dir, half], i) =>
      out.push({
        label: `spot ${i}`,
        dirs: [0, 1, 2, 3].map((d) => {
          const G = xlenBeam(dir, half, d);
          return G && { lo: i, hi: i, table: depositTable(G, nl) };
        }),
      }),
    );
    return out;
  }

  // the tables, the parameter slots and the dispatch list
  #build(spots: boolean) {
    const { N, levels: nl, device } = this, NN = N * N, dual = this.#dual;
    // primal: the merge weights; dual: the near table, then each frustum's source group tables
    const tab: number[] = [];
    if (!dual) for (let l = 0; l < nl; l++) for (let sub = 0; sub < 2 << l; sub++) tab.push(coneArc(l, sub));
    else tab.push(...nearTable().values);
    const chans: [number, number, number][][] = [0, 1, 2, 3].map((d) =>
      this.#channels.flatMap((c) => {
        const t = c.dirs[d];
        if (!t) return [];
        const at = tab.length;
        tab.push(...t.table);
        return [[t.lo, t.hi, at]];
      }),
    );
    const tables = new Float32Array(tab);
    const C = Math.max(1, ...chans.map((c) => c.length));
    const srcPlanes = this.lobes.length + 8;
    let rebind = false;
    const grow = (key: 'tables' | 'src' | 'R' | 'params', bytes: number, usage: number) => {
      if (this.#buffers[key].size >= bytes) return;
      this.#buffers[key].destroy();
      this.#buffers[key] = device.createBuffer({ label: `dual-hrc ${key}`, size: bytes, usage });
      rebind = true;
    };
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    grow('tables', tables.byteLength, storage);
    grow('src', dual ? 2 * srcPlanes * NN * 4 : 16, storage);
    const sums = dual && !!this.options.raySums;
    // primal: two cone levels; dual: every level ≥ 1 kept, or with ray sums a sum per ray and two levels of cone values
    grow('R', (sums ? this.#toff[nl] + 2 * C * NN : dual ? C * Math.max(1, nl - 1) * NN : 2 * NN) * 4, storage);
    device.queue.writeBuffer(this.#buffers.tables, 0, tables);

    // parameter slots
    const slots: ArrayBuffer[] = [];
    const slot = (v: Partial<Record<'dir' | 'level' | 'flags' | 'plane' | 'rIn' | 'rOut', number>> & { chans?: number[][] }) => {
      const data = new ArrayBuffer(PARAM_BYTES), u = new Uint32Array(data);
      u.set([v.dir ?? 0, v.level ?? 0, v.flags ?? 0, v.plane ?? 0, v.rIn ?? 0, v.rOut ?? 0, v.chans?.length ?? 0, srcPlanes]);
      v.chans?.forEach((c, i) => u.set(c, 8 + 4 * i));
      slots.push(data);
      return slots.length - 1;
    };
    const P = this.#pipelines;
    const passes: Dispatch[][] = [], labels: string[] = [];
    const pass = (label: string, list: Dispatch[]) => (passes.push(list), labels.push(label));
    const d0 = (pipeline: GPUComputePipeline, s: number, x: number, y: number, label: string): Dispatch => ({ pipeline, slot: s, x, y, label });
    const flags = spots ? FLAG.SPOT : 0;

    if (dual) {
      const emit: Dispatch[] = [{ ...d0(P.nearFaces, slot({ plane: this.lobes.length }), N, N, 'near faces'), tile: true }];
      for (let i = 0; i < this.lobes.length; i++) emit.push(d0(P.spots, slot({ plane: i }), N, N, 'spots'));
      pass('sources', emit);
      // E then W, N then S: the table E (N) builds serves W (S) straight after
      for (const d of [0, 2, 1, 3]) {
        const list: Dispatch[] = [];
        const rows = (l: number) => (N >> l) * ((1 << l) + 1); // one invocation per ray
        const lvl = (level: number, top = false) => slot({ dir: d, level, flags: top ? FLAG.TOP : 0, rIn: C * NN, chans: chans[d] });
        if (d < 2) {
          list.push(d0(P.table0, lvl(0), N + 2, N + 1, 'transmittance 0'));
          list.push(d0(P.table1, lvl(1), N + 4, (N >> 1) + 1, 'transmittance 1'));
        }
        if (sums) {
          const runs = (l: number) => Math.max(1, (1 << l) / SUMS_RUN); // runs of cones per probe
          list.push(d0(P.deposit0, lvl(0), N, 2 * N, 'gathering 0'));
          list.push(d0(P.extensionsSums1, lvl(1), N, (N >> 1) * runs(1), 'extensions 1'));
          for (let l = 2; l < nl; l++)
            list.push(d < 2 ? d0(P.extensionsSumsBuild, lvl(l), N + (1 << l), ((N >> l) + 1) * runs(l), `extensions ${l}`) : d0(P.extensionsSums, lvl(l), N, (N >> l) * runs(l), `extensions ${l}`));
          for (let l = nl - 1; l >= 1; l--) list.push(d0(P.gatheringSums, lvl(l, l === nl - 1), N, rows(l), `gathering ${l}`));
        } else {
          list.push(d0(P.extensions1, lvl(1), N, 2 * N, 'extensions 1'));
          for (let l = 2; l < nl; l++)
            list.push(d < 2 ? d0(P.extensionsBuild, lvl(l), N + (1 << l), ((N >> l) + 1) * ((1 << l) + 1), `extensions ${l}`) : d0(P.extensionsCarry, lvl(l), N, rows(l), `extensions ${l}`));
          for (let l = nl - 1; l >= 1; l--) list.push(d0(P.gathering, lvl(l, l === nl - 1), N, rows(l), `gathering ${l}`));
        }
        list.push(d0(P.depositSeed, lvl(0), N, N, 'deposit seed'));
        pass(`direction ${d}`, list);
      }
      pass('finish', [d0(P.endMix, slot({}), N, N, 'end mix')]);
    } else {
      const half = NN; // the two halves of the cone buffer
      for (const d of [0, 1, 2, 3]) {
        const list: Dispatch[] = [];
        const lvl = (level: number, extra: object = {}) => slot({ dir: d, level, flags, ...extra });
        list.push(d0(P.seed0, lvl(0), N, N, 'seed'));
        list.push(d0(P.seed1, lvl(1), N, N >> 1, 'seed'));
        for (let L = 2; L < nl; L++) list.push(d0(P.extend, lvl(L), N, (N >> L) * ((1 << (L - 1)) + 1), `extend ${L}`));
        let cur = 0;
        for (let l = nl - 1; l >= 0; l--) {
          list.push(d0(P.merge, lvl(l, { flags: flags | (l === nl - 1 ? FLAG.TOP : 0), rIn: cur, rOut: half - cur }), N, (N >> l) << l, `merge ${l}`));
          cur = half - cur;
        }
        pass(`direction ${d}`, list);
      }
      pass('finish', [d0(P.blur, slot({}), N, N, 'blur')]);
    }

    const bytes = slots.length * PARAM_STRIDE;
    grow('params', bytes, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    if (rebind) this.#bindGroup = this.#makeBindGroup();
    const all = new Uint8Array(bytes);
    slots.forEach((s, i) => all.set(new Uint8Array(s), i * PARAM_STRIDE));
    device.queue.writeBuffer(this.#buffers.params, 0, all);
    this.#program = passes;
    this.#passLabels = labels;
  }

  /** the whole solve: clears the summed fluence, then the passes */
  encodeAll(enc: GPUCommandEncoder) {
    const NN4 = this.N * this.N * 4;
    enc.clearBuffer(this.outBuffer, 0, NN4);
    if (this.#dual) enc.clearBuffer(this.outBuffer, 2 * NN4, NN4);
    this.#program.forEach((list, i) => {
      const pass = this.profiler ? this.profiler.computePass(enc, this.#passLabels[i]) : enc.beginComputePass();
      for (const d of list) {
        this.profiler?.name(pass, d.label);
        pass.setPipeline(d.pipeline);
        pass.setBindGroup(0, this.#bindGroup, [d.slot * PARAM_STRIDE]);
        if (d.tile) pass.dispatchWorkgroups(Math.ceil(d.x / TILE), Math.ceil(d.y / TILE));
        else pass.dispatchWorkgroups(Math.ceil(d.x / WG), d.y);
      }
      pass.end();
    });
  }

  /** number of dispatches per solve */
  get dispatches() {
    return this.#program.reduce((a, l) => a + l.length, 0);
  }

  /** the dual's source groups for the last scene, and in how many frustums each runs */
  get channels(): { label: string; directions: number }[] {
    return this.#channels.map((c) => ({ label: c.label, directions: c.dirs.filter(Boolean).length }));
  }

  async read(buffer: GPUBuffer, offset: number, bytes: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** upload, solve once and read the final fluence back (world [x * N + y]) */
  async solve(scene: DualScene): Promise<Float32Array> {
    this.upload(scene);
    const enc = this.device.createCommandEncoder();
    this.encodeAll(enc);
    this.device.queue.submit([enc.finish()]);
    return this.read(this.outBuffer, this.outputOffset, this.N * this.N * 4);
  }

  /** Bytes each dispatch label reads and writes per occurrence: every array element it touches, counted once (a lower
   * bound on memory traffic). Ray levels are t_l = (N >> l)(2^l + 1) N elements. */
  traffic(): Record<string, Traffic> {
    const { N, levels: nl } = this, NN = N * N, B = 4;
    const t = (l: number) => (N >> l) * ((1 << l) + 1) * N;
    const out: Record<string, Traffic> = {};
    if (!this.#dual) {
      out.seed = { read: NN * (this.lobes.length ? 4 : 2) * B, write: ((t(0) + t(1)) / 2) * 2 * B };
      for (let L = 2; L < nl; L++) out[`extend ${L}`] = { read: t(L - 1) * 2 * B, write: t(L) * 2 * B };
      for (let l = 0; l < nl; l++) out[`merge ${l}`] = { read: t(l) * 2 * B + (l < nl - 1 ? NN * B : 0) + (l === 0 ? NN * B : 0), write: NN * B };
      out.blur = { read: 2 * NN * B, write: NN * B };
      return out;
    }
    // source groups per frustum, mean over the four
    const C = [0, 1, 2, 3].reduce((a, d) => a + this.#channels.filter((c) => c.dirs[d]).length, 0) / 4;
    // the shared table's entries of level l (segments −1 .. (N >> l) − 1, start rows with an end inside), built by E and N
    const table = (l: number) => {
      const n = 1 << l;
      let rows = 0;
      for (let k = 0; k <= n; k++) rows += N + Math.abs(2 * k - n);
      return ((N >> l) + 1) * rows;
    };
    // the scene's emission, transmittance and spot planes; eight source planes, each in one layout, and the near field
    out['near faces'] = { read: 3 * NN * B, write: 9 * NN * B };
    out.spots = { read: 3 * NN * B, write: 2 * NN * B };
    for (let l = 0; l < 2; l++) out[`transmittance ${l}`] = { read: NN * B, write: table(l) * B };
    // dual extensions: the carry reads tr_{l−1} and the cone values of level l − 1 (level 1: the source images) and writes
    // R~_l (level 1 also D_0). Above level 1, E and N also read the table's level l − 1 whole and write its level l; W and S
    // read only their rays' tr_{l−1}. Each label's numbers are the mean over the four frustums.
    if (this.options.raySums) {
      // ray sums: level 0's splats on their own; the dual extensions also write a sum per ray, which dual gathering reads
      out['gathering 0'] = { read: t(0) * B + 2 * C * NN * B, write: t(0) * B };
      out['extensions 1'] = { read: t(0) * B + 2 * C * NN * B, write: (C * NN + t(1)) * B };
      for (let l = 2; l < nl; l++) out[`extensions ${l}`] = { read: (C * NN + (table(l - 1) + t(l - 1)) / 2) * B, write: (C * NN + t(l) + table(l) / 2) * B };
      for (let l = 1; l < nl; l++) out[`gathering ${l}`] = { read: t(l) * B + (t(l) / 2) * B + (l < nl - 1 ? t(l + 1) * B : 0), write: t(l) * B };
    } else {
      out['extensions 1'] = { read: t(0) * B + 2 * C * NN * B, write: (t(0) + C * NN) * B };
      for (let l = 2; l < nl; l++) out[`extensions ${l}`] = { read: (C * NN + (table(l - 1) + t(l - 1)) / 2) * B, write: (C * NN + table(l) / 2) * B };
      // dual gathering: the kept cone values, even probes' tr, D_{l+1} (below the top)
      for (let l = 1; l < nl; l++) out[`gathering ${l}`] = { read: C * NN * B + (t(l) / 2) * B + (l < nl - 1 ? t(l + 1) * B : 0), write: t(l) * B };
    }
    out['deposit seed'] = { read: (t(0) + t(1)) * B + 2 * NN * B, write: NN * B };
    // E + W and N + S splats, the near field, the dual's transmittance
    out['end mix'] = { read: 4 * NN * B, write: NN * B };
    return out;
  }

  /** bytes of the stores the solve keeps (rays, cones, sources, scene, output) */
  get bytes() {
    const b = this.#buffers;
    return b.T.size + b.R.size + b.src.size + this.sceneBuffer.size + this.outBuffer.size;
  }
}
