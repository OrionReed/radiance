// Reciprocal HRC with dithered junctions. The one-product engine of site/reciprocal-napkin with one change to its
// pair table: where a cell of length L has an even rise D with 2 <= |D| <= L - 2, it is made of its straight halves
// (D/2, D/2) at 3/4 and of the two bent ones (D/2 - 1, D/2 + 1) and (D/2 + 1, D/2 - 1) at 1/8 each. Everywhere else it
// is plain: an odd rise takes floor and ceil at 1/2, and the rises 0 and ±L stay straight.
//
// Read as rounding, plain puts a cell's junction row at round(D/2 + u) with u uniform on a row (box dither); the table
// above is round(D/2 + u - u') (triangular dither). The second moment of the bend is then 1 for every rise, so the
// error no longer depends on the binary digits of D (the spokes), at twice plain's blur. Keeping a bent path's slopes
// on one side of 0 and inside ±45° (hence the excluded rises) keeps every pixel's line count exact for solid lights
// of any size, and keeps every path inside the frustum, so no rays beyond 45° are needed.
//
// Two fixes for the lines the straight rises leave (one row per pixel only; both keep the transport reciprocal):
//   diagonal: D = ±L also dithers, from L = 4. Its bent pairs reach one rise past 45°, so every level from 1 up keeps
//             a guard band of rises ±(L + 1): cone weight 0, plain floor/ceil halves. Level 1's guard rays (rise ±2,
//             one pixel long) are not stored: a level-2 cell reads one from its three pixels (first hit), and the
//             pixel readout gives it its four half-pixel crossings at half a half pixel each, which is the ray's
//             share of each pixel's area (1/4, 1/2, 1/4). The top level has no guard rays.
//   axis:     D = ±2 goes plain (dither for 4 <= |D| <= L - 2 and, with the diagonal fix, |D| = L), and D = ±1 of a
//             cell of L >= 16 takes 1/2 (0, E) + 1/2 (L, 0): its rise-1 half steps on the side nearer the junction.
//             E and L are two more rays of rise ±1 per level from length 8: E = 2/3 (E, 0) + 1/3 (0, 1), L its mirror,
//             with the plain rise-1 ray below length 8. A rise-1 cell's cone weight is shared among its three kinds
//             per block, by the measure a uniform far field hands each (computed here, once per engine).
//
// A record (F, V, T) is a stretch of line: light leaving its right end, light leaving its left end, light let through;
// join(first, second) is the stretch made of both. One step serves every pass, for a cell in either role:
//   as the cell (up):   the weighted sum over its pairs of join(left half, right half);
//   as a half (down):   its outside, the light arriving at it: its sibling joined onto the cell's outside, the
//                       sibling's own light at the cell's angle; the side away from the sibling is the cell's.
// Both passes read the same table, and one record carries both directions, so the transport is exactly reciprocal.
// Fluence: the outsides of each pixel's two half pixels (level 0, read from the scene), both axes. Axis 0 runs along
// x, axis 1 along y. Rows off the scene are opaque and dark; a level keeps the start rows whose lines can reach the
// scene.

import { rgba, staging } from '../_shared/upload';

export interface DitherScene {
  /** world [x * N + y], rgb */
  emission: Float32Array;
  /** world [x * N + y]: 1 opaque; a pixel with emission is opaque */
  solid: Float32Array | Uint8Array;
}

const WORKGROUP = 256;
/** the largest rise of level k (length 2^k / 2 pixels) at s rows per pixel: 45° */
const bandOf = (s: number) => (k: number) => (k === 0 ? 0 : Math.ceil(s * 2 ** (k - 1) - 1e-9));
/** start rows of a frustum at s rows per pixel */
const rowsOf = (N: number, s: number) => Math.ceil(N * s - 1e-9);

/** one way to make a cell from two halves: first half rise a and kind ka, second half rise b and kind kb starting off
 * rows past the first, and its weight. Kinds: 0 the plain ray of a rise, 1 E (rise ±1, its row step nearer its left
 * end), 2 L (nearer its right end) */
export interface Pair { a: number; b: number; off: number; ka: number; kb: number; w: number }

/** the pair table and the stored layout, as the shader has them (the shader's own functions mirror these) */
export function layout(N: number, { s = 1, dither = true, diag = false, axis = false } = {}) {
  const top = Math.log2(N) + 1, band = bandOf(s), R = rowsOf(N, s);
  /** guard rays exist (level 1's are virtual) / are stored; E and L kinds are stored */
  const guards = (k: number) => (diag && k >= 1 && k < top ? 1 : 0);
  const stored = (k: number) => (diag && k >= 2 && k < top ? 1 : 0);
  const kinds = (k: number) => axis && k >= 4 && k < top;
  const reach = (k: number) => band(k) + stored(k);
  const slots = (k: number) => 2 * reach(k) + 1 + (kinds(k) ? 4 : 0);
  const slot = (k: number, z: number, kind: number) => (kind === 0 ? z + reach(k) : 2 * reach(k) + 1 + 2 * (kind - 1) + (z > 0 ? 1 : 0));
  const blocks = (k: number) => (2 * N) >> k;
  const cells = (k: number) => 2 * blocks(k) * slots(k) * (R + 2 * reach(k));
  const dithers = (P: number, m: number, D: number) => {
    if (!dither || m === 0 || D % 2 !== 0) return false;
    const A = Math.abs(D);
    if (A >= (axis ? 4 : 2) && A <= P - 2) return true;
    return diag && A === P && P >= 4;
  };
  const typed = (k: number, D: number) => axis && Math.abs(D) === 1 && kinds(k - 1);
  /** pair `way` (0..3) of a cell of level k, rise D, kind; weight 0 where it has no such pair */
  const pair = (k: number, D: number, kind: number, way: number): Pair => {
    const m = band(k - 1), P = band(k), h = m + guards(k - 1), none = { a: 0, b: 0, off: 0, ka: 0, kb: 0, w: 0 };
    if (kind !== 0) {
      const inner = kinds(k - 1) ? kind : 0, near = kind === 1 ? 2 / 3 : 1 / 3;
      if (way === 0) return { a: D, b: 0, off: D, ka: kind === 1 ? inner : 0, kb: 0, w: near };
      if (way === 1) return { a: 0, b: D, off: 0, ka: 0, kb: kind === 2 ? inner : 0, w: 1 - near };
      return none;
    }
    if (typed(k, D)) {
      if (way === 0) return { a: 0, b: D, off: 0, ka: 0, kb: 1, w: 0.5 };
      if (way === 1) return { a: D, b: 0, off: D, ka: 2, kb: 0, w: 0.5 };
      return none;
    }
    const bent = dithers(P, m, D);
    let a: number, b: number, off: number;
    if (way >= 2) ((a = (D >> 1) + (way === 3 ? 1 : -1)), (b = D - a), (off = a));
    else {
      const f = way === 1 ? -(-D >> 1) : D >> 1;
      b = Math.max(-h, Math.min(h, D - f));
      a = Math.max(-h, Math.min(h, f));
      off = D - b;
    }
    const w = bent ? (way >= 2 ? 0.125 : 0.375) : way >= 2 ? 0 : 0.5;
    return { a, b, off, ka: 0, kb: 0, w };
  };
  /** the parent slots of level k: (rise, kind) */
  const parents = (k: number) => {
    const P = band(k), G = stored(k), out: [number, number][] = [];
    for (let D = -P - G; D <= P + G; D++) out.push([D, 0]);
    if (kinds(k)) for (const kind of [1, 2]) for (const D of [-1, 1]) out.push([D, kind]);
    return out;
  };
  return { top, band, R, guards, stored, kinds, reach, slots, slot, blocks, cells, pair, parents };
}

/** a rise-1 cell's cone weight shared among its kinds, per block: the measure a uniform far field (1 on every rise at
 * the top) hands each kind, descending the table. [level][3 * block + kind], for the levels that store kinds */
export function sharesOf(N: number, t: ReturnType<typeof layout>) {
  const at = new Int32Array(t.top + 1), out: number[] = [];
  let m = new Float64Array(t.slots(t.top) * t.blocks(t.top));
  for (const [D] of t.parents(t.top)) if (Math.abs(D) <= t.band(t.top)) m[t.slot(t.top, D, 0)] = 1;
  for (let K = t.top; K >= 5; K--) {
    const k = K - 1, child = new Float64Array(t.slots(k) * t.blocks(k)), nK = t.slots(K), nk = t.slots(k);
    for (let B = 0; B < t.blocks(K); B++)
      for (const [D, kind] of t.parents(K)) {
        if (K === t.top && Math.abs(D) > t.band(K)) continue;
        const v = m[B * nK + t.slot(K, D, kind)];
        if (!v) continue;
        for (let way = 0; way < 4; way++) {
          const q = t.pair(K, D, kind, way);
          if (!q.w) continue;
          child[2 * B * nk + t.slot(k, q.a, q.ka)] += q.w * v;
          child[(2 * B + 1) * nk + t.slot(k, q.b, q.kb)] += q.w * v;
        }
      }
    if (t.kinds(k)) {
      at[k] = out.length;
      for (let b = 0; b < t.blocks(k); b++) {
        const v = [0, 1, 2].map((kind) => child[b * nk + t.slot(k, 1, kind)]), sum = v[0] + v[1] + v[2];
        out.push(...(sum > 0 ? v.map((x) => x / sum) : [1, 0, 0]));
      }
    }
    m = child;
  }
  return { at, shares: new Float32Array(out.length ? out : [1]) };
}

/** child reads per parent ray, as lab 3 counts them (rules.ts readsPerRay): two per distinct pair, averaged over the
 * in-frustum rises |D| <= L of levels 2 .. top - 1, weighted by blocks; `all` also counts the guard and E, L slots */
export function readsPerRay(N: number, options: Parameters<typeof layout>[1] = {}) {
  const t = layout(N, options);
  let reads = 0, rays = 0, readsAll = 0, raysAll = 0;
  for (let k = 2; k < t.top; k++)
    for (const [D, kind] of t.parents(k)) {
      const seen = new Set<string>();
      for (let way = 0; way < 4; way++) {
        const q = t.pair(k, D, kind, way);
        if (q.w) seen.add(`${q.a} ${q.ka} ${q.b} ${q.kb} ${q.off}`);
      }
      const r = 2 * seen.size * t.blocks(k);
      readsAll += r; raysAll += t.blocks(k);
      if (kind === 0 && Math.abs(D) <= t.band(k)) ((reads += r), (rays += t.blocks(k)));
    }
  return { inFrustum: reads / rays, all: readsAll / raysAll };
}

const shader = (N: number, s: number, top: number, f16: boolean, dither: boolean, diag: boolean, axis: boolean, shareAt: Int32Array) => /* wgsl */ `
const N: i32 = ${N};
const S: f32 = ${s.toFixed(8)};   // rows per pixel, across each axis's rays
const R: i32 = ${rowsOf(N, s)};   // start rows
const BANDS = array<i32, ${top + 1}>(${Array.from({ length: top + 1 }, (_, k) => bandOf(s)(k)).join(', ')});
const DITHER: bool = ${dither};
const DIAG: bool = ${diag};       // the diagonal fix: D = ±L dithers, guard rises ±(L + 1)
const AXIS: bool = ${axis};       // the axis fix: D = ±2 plain, D = ±1 with typed halves E and L
const SHARE_AT = array<i32, ${top + 1}>(${Array.from(shareAt).join(', ')});
const TOP: i32 = ${top};
struct Rec { F: vec3f, V: vec3f, T: f32 };
struct Params { k: i32, count: u32 };
// a pair: first half rise a, kind ka; second half rise b, kind kb, starting off rows past the first; weight w
struct Pair { a: i32, b: i32, off: i32, ka: i32, kb: i32, w: f32 };
struct Cell { c: vec4i, kind: i32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scene: array<vec4f>;      // world [x * N + y]: (rgb, 1 if opaque)
@group(0) @binding(2) var<storage, read> inside: array<${f16 ? 'vec4u' : 'Rec'}>;       // the halves' level
@group(0) @binding(3) var<storage, read> outside: array<${f16 ? 'vec4u' : 'Rec'}>;      // the cells' level (down)
@group(0) @binding(4) var<storage, read_write> written: array<${f16 ? 'vec4u' : 'Rec'}>;
@group(0) @binding(5) var<storage, read_write> fluence: array<vec4f>;  // world [x * N + y]: (rgb, luminance)
@group(0) @binding(6) var<storage, read> shares: array<f32>;      // the axis fix: rise-1 kinds' cone shares
@group(0) @binding(7) var<storage, read> aux: array<${f16 ? 'vec4u' : 'Rec'}>;          // pixels, diagonal fix: level 2's outsides

${f16 ? `fn unpack(p: vec4u) -> Rec {
  let a = unpack2x16float(p.x); let b = unpack2x16float(p.y); let c = unpack2x16float(p.z); let d = unpack2x16float(p.w);
  return Rec(vec3f(a, b.x), vec3f(b.y, c), d.x);
}
fn pack(r: Rec) -> vec4u {
  return vec4u(pack2x16float(r.F.xy), pack2x16float(vec2f(r.F.z, r.V.x)), pack2x16float(r.V.yz), pack2x16float(vec2f(r.T, 0.0)));
}` : 'fn unpack(r: Rec) -> Rec { return r; }\nfn pack(r: Rec) -> Rec { return r; }'}
const NONE = Rec(vec3f(0.0), vec3f(0.0), 1.0);
const DARK = Rec(vec3f(0.0), vec3f(0.0), 0.0);
fn join(a: Rec, b: Rec) -> Rec { return Rec(b.F + b.T * a.F, a.V + a.T * b.V, a.T * b.T); }
fn plus(acc: Rec, a: Rec, w: f32) -> Rec { return Rec(acc.F + w * a.F, acc.V + w * a.V, acc.T + w * a.T); }
fn lit(a: Rec, w: f32) -> Rec { return Rec(w * a.F, w * a.V, a.T); }

fn band(k: i32) -> i32 { return BANDS[k]; }
fn guards(k: i32) -> i32 { return select(0, 1, DIAG && k >= 1 && k < TOP); }   // guard rays exist; level 1's are virtual
fn stored(k: i32) -> i32 { return select(0, 1, DIAG && k >= 2 && k < TOP); }   // and are stored
fn kinds(k: i32) -> bool { return AXIS && k >= 4 && k < TOP; }                  // E and L are stored (length >= 8)
fn reach(k: i32) -> i32 { return band(k) + stored(k); }
// pair \`way\` of a cell of level k with rise D: its halves have largest rise m (h with the guard band), the cell P (its
// 45° rise). Ways 0 and 1 round D/2 down and up; ways 2 and 3 bend by a row each way, where the cell dithers: even D,
// not straight along the axis (with the axis fix not ±2 either), not bending past 45° (with the diagonal fix: or
// D = ±P from P = 4, bending into the guard band), and halves that can bend (m > 0).
fn dithers(P: i32, m: i32, D: i32) -> bool {
  if (!DITHER || m == 0 || (D & 1) != 0) { return false; }
  if (abs(D) >= select(2, 4, AXIS) && abs(D) <= P - 2) { return true; }
  return DIAG && abs(D) == P && P >= 4;
}
fn split(D: i32, m: i32, way: i32) -> vec3i {
  if (way >= 2) { let a = (D >> 1u) + select(-1, 1, way == 3); return vec3i(a, D - a, a); }
  let a = select(D >> 1u, -((-D) >> 1u), way == 1);
  let b = clamp(D - a, -m, m);
  return vec3i(clamp(a, -m, m), b, D - b);
}
fn share(P: i32, m: i32, D: i32, way: i32) -> f32 {
  if (!dithers(P, m, D)) { return select(0.5, 0.0, way >= 2); }
  return select(0.375, 0.125, way >= 2);
}
// the axis fix: rise ±1 of a cell whose halves have kinds takes 1/2 (0, E) + 1/2 (L, 0); kind E (1) of rise ±1 is
// 2/3 (E, 0) + 1/3 (0, 1), L (2) is 1/3 (1, 0) + 2/3 (0, L), with plain halves below the levels with kinds
fn typed(k: i32, D: i32) -> bool { return AXIS && abs(D) == 1 && kinds(k - 1); }
fn pairOf(k: i32, D: i32, kind: i32, way: i32) -> Pair {
  let m = band(k - 1); let P = band(k);
  if (kind != 0) {
    let inner = select(0, kind, kinds(k - 1)); let near = select(1.0 / 3.0, 2.0 / 3.0, kind == 1);
    if (way == 0) { return Pair(D, 0, D, select(0, inner, kind == 1), 0, near); }
    if (way == 1) { return Pair(0, D, 0, 0, select(0, inner, kind == 2), 1.0 - near); }
    return Pair(0, 0, 0, 0, 0, 0.0);
  }
  if (typed(k, D)) {
    if (way == 0) { return Pair(0, D, 0, 0, 1, 0.5); }
    if (way == 1) { return Pair(D, 0, D, 2, 0, 0.5); }
    return Pair(0, 0, 0, 0, 0, 0.0);
  }
  let s = split(D, m + guards(k - 1), way);
  return Pair(s.x, s.y, s.z, 0, 0, share(P, m, D, way));
}
fn angle(L: i32, D: i32) -> f32 {               // the angle of a cell L pixels long; at L = 1, S = 1 the exact one-pixel values
  if (L == 1 && S == 1.0) { return select(0.34657359, 0.87764914, D == 0); }
  let l = S * f32(L);
  return atan(clamp((f32(D) + 0.5) / l, -1.0, 1.0)) - atan(clamp((f32(D) - 0.5) / l, -1.0, 1.0));
}
fn index(k: i32, c: vec4i, kind: i32) -> i32 {  // c = (axis, block, rise, start row); slots: rises, then E-, E+, L-, L+
  let g = reach(k);
  var slot = c.z + g;
  if (kind != 0) { slot = 2 * g + 1 + 2 * (kind - 1) + select(0, 1, c.z > 0); }
  return ((c.x * ((2 * N) >> u32(k)) + c.y) * (2 * g + 1 + select(0, 4, kinds(k))) + slot) * (R + 2 * g) + c.w + g;
}
fn location(k: i32, i: i32) -> Cell {
  let g = reach(k); let rows = R + 2 * g; let slots = 2 * g + 1 + select(0, 4, kinds(k)); let blocks = (2 * N) >> u32(k);
  let slot = (i / rows) % slots;
  var z = slot - g; var kind = 0;
  if (slot > 2 * g) { let e = slot - 2 * g - 1; kind = 1 + e / 2; z = select(-1, 1, (e & 1) == 1); }
  return Cell(vec4i(i / (rows * slots * blocks), (i / (rows * slots)) % blocks, z, i % rows - g), kind);
}
fn half(c: vec4i) -> Rec {                      // level 0: a half pixel from the scene; off the rows: opaque and dark
  if (c.w < 0 || c.w >= R) { return DARK; }
  // the pixels this row spans, by the share of the row each covers: a solid pixel's share blocks and emits
  let lo = f32(c.w) / S; let hi = f32(c.w + 1) / S;
  var light = vec3f(0.0); var open = 1.0;
  for (var p = i32(floor(lo)); f32(p) < hi; p++) {
    let share = (min(hi, f32(p + 1)) - max(lo, f32(p))) * S;
    if (share <= 0.0) { continue; }
    if (p >= N) { open -= share; continue; }   // off the scene: opaque and dark
    let s = scene[select(p * N + c.y / 2, (c.y / 2) * N + p, c.x == 0)];
    if (s.a > 0.5) { light += share * s.rgb; open -= share; }
  }
  if (open >= 1.0) { return NONE; }
  return Rec(light, light, max(open, 0.0));
}
// level 1's guard ray (rise ±2, one pixel): its rows r, r ± 1, r ± 2 front to back, solids first hit (one row per pixel,
// so the middle pixel's two half-pixel crossings are that pixel once)
fn guard1(c: vec4i) -> Rec {
  let s = c.z / 2;
  return join(join(half(vec4i(c.x, 2 * c.y, 0, c.w)), half(vec4i(c.x, 2 * c.y, 0, c.w + s))), half(vec4i(c.x, 2 * c.y + 1, 0, c.w + 2 * s)));
}
fn record(k: i32, c: vec4i, kind: i32) -> Rec { // inside; half pixels from the scene; off the rows: opaque and dark
  if (DIAG && k == 1 && abs(c.z) == 2) { return guard1(c); }
  if (k == 0) { return half(c); }
  if (c.w < -reach(k) || c.w >= R + reach(k)) { return DARK; }
  return unpack(inside[index(k, c, kind)]);
}

// for cell c (of kind \`kind\`) of level k, as the cell (up) or as a half (down)
fn visit(k: i32, c: vec4i, kind: i32, up: bool) -> Rec {
  var out = Rec(vec3f(0.0), vec3f(0.0), 0.0);
  let second = (c.y & 1) == 1;
  // K: the cell's level; m: the halves' largest rise; P: the cell's; L: the cell's length in pixels
  let K = select(k + 1, k, up);
  let m = band(K - 1); let P = band(K); let G = stored(K);
  let L = 1 << u32(K - 1);
  // down: a half of rise r is a half of the cells with rises 2r - 1 ... 2r + 1, and of 2r ± 2 where those bend to it; a
  // half pixel (rise 0) of every 1-px cell
  let reach2 = select(1, 2, DITHER && m > 0);
  let lo = select(select(2 * c.z - reach2, -P, m == 0), c.z, up); let hi = select(select(2 * c.z + reach2, P, m == 0), c.z, up);
  for (var D = max(lo, -P - G); D <= min(hi, P + G); D++) {
    let nk = select(1, 3, !up && abs(D) == 1 && kinds(K));
    for (var kp = 0; kp < nk; kp++) {
      let kindP = select(kp, kind, up);
      for (var way = 0; way < 4; way++) {
        let q = pairOf(K, D, kindP, way);
        let p = q.w;
        if (p == 0.0) { continue; }
        if (up) {
          let first = vec4i(c.x, 2 * c.y, q.a, c.w);
          out = plus(out, join(record(k - 1, first, q.ka), record(k - 1, vec4i(c.x, 2 * c.y + 1, q.b, c.w + q.off), q.kb)), p);
          continue;
        }
        if (select(q.a, q.b, second) != c.z || select(q.ka, q.kb, second) != kind) { continue; }
        let row = select(c.w, c.w - q.off, second);
        var cell = NONE;
        if (K < TOP && row >= -reach(K) && row < R + reach(K)) { cell = unpack(outside[index(K, vec4i(c.x, c.y >> 1u, D, row), kindP)]); }
        var w = angle(L, D) / 2.0;
        if (kinds(K) && abs(D) == 1) { w *= shares[SHARE_AT[K] + 3 * (c.y >> 1u) + kindP]; }
        if (second) {
          let sibling = lit(record(k, vec4i(c.x, c.y - 1, q.a, row), q.ka), w);
          out = plus(out, Rec(join(Rec(cell.F, vec3f(0.0), 1.0), sibling).F, cell.V, 0.0), p);
        } else {
          let sibling = lit(record(k, vec4i(c.x, c.y + 1, q.b, row + q.off), q.kb), w);
          out = plus(out, Rec(cell.F, join(sibling, Rec(vec3f(0.0), cell.V, 1.0)).V, 0.0), p);
        }
      }
    }
  }
  return out;
}

// the diagonal fix: the outside of level 1's guard ray c = (axis, pixel column, ±2, start row), a half of level 2's
// guard cells (cone weight 0: its sibling only lets light through), from level 2's outsides (aux)
fn guardOutside(c: vec4i) -> Rec {
  var out = Rec(vec3f(0.0), vec3f(0.0), 0.0);
  if (TOP <= 2) { return out; }
  let second = (c.y & 1) == 1; let D = 3 * (c.z / 2);
  for (var way = 0; way < 2; way++) {
    let q = pairOf(2, D, 0, way);
    if (select(q.a, q.b, second) != c.z) { continue; }
    let row = select(c.w, c.w - q.off, second);
    var cell = NONE;
    if (row >= -reach(2) && row < R + reach(2)) { cell = unpack(aux[index(2, vec4i(c.x, c.y >> 1u, D, row), 0)]); }
    if (second) { out = plus(out, Rec(record(1, vec4i(c.x, c.y - 1, q.a, row), 0).T * cell.F, cell.V, 0.0), q.w); }
    else { out = plus(out, Rec(cell.F, record(1, vec4i(c.x, c.y + 1, q.b, row + q.off), 0).T * cell.V, 0.0), q.w); }
  }
  return out;
}

fn thread(id: vec3u, groups: vec3u) -> u32 { return id.x + id.y * groups.x * ${WORKGROUP}u; }
@compute @workgroup_size(${WORKGROUP}) fn up(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let i = thread(id, g);
  if (i < params.count) { let at = location(params.k, i32(i)); written[i] = pack(visit(params.k, at.c, at.kind, true)); }
}
@compute @workgroup_size(${WORKGROUP}) fn down(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let i = thread(id, g);
  if (i < params.count) { let at = location(params.k, i32(i)); written[i] = pack(visit(params.k, at.c, at.kind, false)); }
}
@compute @workgroup_size(${WORKGROUP}) fn pixels(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) g: vec3u) {
  let i = i32(thread(id, g));
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N; let s = scene[i];
  var sum = vec3f(0.0);
  for (var h = 0; h < 4; h++) {
    let axis = h >> 1u; let across = select(y, x, axis == 1);
    // the rows across this pixel, each weighted by the part of it inside the pixel
    var part = vec3f(0.0); var total = 0.0;
    for (var row = i32(floor(f32(across) * S)); f32(row) < f32(across + 1) * S; row++) {
      let w = min(f32(row + 1), f32(across + 1) * S) - max(f32(row), f32(across) * S);
      if (w <= 0.0) { continue; }
      let r = visit(0, vec4i(axis, 2 * select(x, y, axis == 1) + (h & 1), 0, row), 0, false);
      part += w * (r.F + r.V); total += w;
    }
    sum += part / total;
  }
  if (DIAG) {
    // level 1's guard rays cross a pixel's rows in four quarters of their length (r, r ± 1 twice, r ± 2): each is read
    // at half a half pixel, the light arriving at it through the crossings before it (this pixel is free, T = 1)
    for (var axis = 0; axis < 2; axis++) {
      let along = select(x, y, axis == 1); let p = select(y, x, axis == 1);
      for (var t = -1; t <= 1; t += 2) {
        let after1 = half(vec4i(axis, 2 * along, 0, p + t)).T; let after2 = half(vec4i(axis, 2 * along, 0, p + 2 * t)).T;
        let before1 = half(vec4i(axis, 2 * along, 0, p - t)).T; let before2 = half(vec4i(axis, 2 * along, 0, p - 2 * t)).T;
        let a = guardOutside(vec4i(axis, along, 2 * t, p));           // this pixel is the ray's start
        let b = guardOutside(vec4i(axis, along, 2 * t, p - t));       // its middle (two crossings)
        let e = guardOutside(vec4i(axis, along, 2 * t, p - 2 * t));   // its end
        sum += 0.5 * (a.F + after1 * after2 * a.V) + (before1 * b.F + after1 * b.V) + 0.5 * (before2 * before1 * e.F + e.V);
      }
    }
  }
  let v = select(sum, 6.2831853 * s.rgb, s.a > 0.5);
  fluence[i] = vec4f(v, dot(v, vec3f(0.2126, 0.7152, 0.0722)));
}`;

interface Dispatch { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; count: number }

export class Dither {
  readonly device: GPUDevice;
  readonly N: number;
  /** the dithered pair table (false: plain) */
  readonly dither: boolean;
  /** the diagonal fix (only with the dither, at one row per pixel) */
  readonly diag: boolean;
  /** the axis fix (one row per pixel only) */
  readonly axis: boolean;
  readonly sceneBuffer: GPUBuffer;
  readonly outputBuffer: GPUBuffer;
  #buffers: GPUBuffer[] = [];
  #program: Dispatch[] = [];

  /** rows per pixel across each axis's rays (1: the pixel rows) */
  readonly s: number;

  /** storage: 'f32' records (32 B) or 'f16' (16 B); dither: the dithered pair table (off: plain, as site/reciprocal-napkin);
   * s: rows per pixel. The dither's blur is 1/12 row² per octave against plain's 1/24 at one row per pixel, so s = √2
   * matches plain's blur in pixels, at about twice the stores. diag, axis: the two fixes (s = 1 only; diag needs the
   * dither); off by default */
  constructor(device: GPUDevice, N: number, { storage = 'f32', dither = true, s = 1, diag = false, axis = false }: { storage?: 'f32' | 'f16'; dither?: boolean; s?: number; diag?: boolean; axis?: boolean } = {}) {
    this.device = device;
    this.N = N;
    this.dither = dither;
    this.s = s;
    this.diag = diag && dither && s === 1;
    this.axis = axis && s === 1;
    const t = layout(N, { s, dither, diag: this.diag, axis: this.axis });
    const { top } = t, RECORD = storage === 'f16' ? 16 : 32;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (size: number, use = usage) => {
      const b = device.createBuffer({ size: Math.max(32, size), usage: use });
      this.#buffers.push(b);
      return b;
    };
    this.sceneBuffer = buffer(N * N * 16);
    this.outputBuffer = buffer(N * N * 16);
    const none = buffer(32), sink = buffer(32);
    const { at, shares } = this.axis ? sharesOf(N, t) : { at: new Int32Array(top + 1), shares: new Float32Array([1]) };
    const shareBuffer = buffer(shares.byteLength);
    device.queue.writeBuffer(shareBuffer, 0, shares);
    const inside = [none];
    for (let k = 1; k < top; k++) inside.push(buffer(t.cells(k) * RECORD));
    let largest = 0;
    for (let k = 1; k < top; k++) largest = Math.max(largest, t.cells(k));
    const outside = [buffer(largest * RECORD), buffer(largest * RECORD)];
    const module = device.createShaderModule({ code: shader(N, s, top, storage === 'f16', dither, this.diag, this.axis, at) });
    const entries = [0, 1, 2, 3, 4, 5, 6, 7].map((binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: (binding === 0 ? 'uniform' : binding === 4 || binding === 5 ? 'storage' : 'read-only-storage') as GPUBufferBindingType },
    }));
    const bindGroupLayout = device.createBindGroupLayout({ entries });
    const layout_ = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const [up, down, pixels] = ['up', 'down', 'pixels'].map((entryPoint) => device.createComputePipeline({ layout: layout_, compute: { module, entryPoint } }));
    const add = (pipeline: GPUComputePipeline, k: number, count: number, read: GPUBuffer, cell: GPUBuffer, written: GPUBuffer, aux = none) => {
      const params = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0, new Int32Array([k, count]));
      const resources = [params, this.sceneBuffer, read, cell, written, pipeline === pixels ? this.outputBuffer : sink, shareBuffer, aux];
      const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: resources.map((b, binding) => ({ binding, resource: { buffer: b } })) });
      this.#program.push({ pipeline, bindGroup, count });
    };
    // the top cell (length N) is never stored: nothing lies outside it, and the step at level top - 1 knows that
    for (let k = 1; k < top; k++) add(up, k, t.cells(k), inside[k - 1], none, inside[k]);
    let from = none;
    for (let k = top - 1; k >= 1; k--) {
      const to = outside[k % 2];
      add(down, k, t.cells(k), inside[k], from, to);
      from = to;
    }
    // the readout reads level 1's records and, with the diagonal fix, level 2's outsides (still in outside[0])
    add(pixels, 0, N * N, inside[Math.min(1, top - 1)], from, sink, this.diag && top > 2 ? outside[0] : none);
    this.readsPerRay = readsPerRay(N, { s, dither, diag: this.diag, axis: this.axis });
  }

  /** child reads per parent ray (lab 3's count, see readsPerRay) */
  readonly readsPerRay: { inFrustum: number; all: number };

  get dispatches() {
    return this.#program.length;
  }

  get bytes() {
    return this.#buffers.reduce((s, b) => s + b.size, 0);
  }

  #staging = staging();
  upload({ emission, solid }: DitherScene) {
    this.device.queue.writeBuffer(this.sceneBuffer, 0, rgba(this.#staging(this.N * this.N * 4), { emission, solid }, 1, 0));
  }

  encodeAll(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();
    for (const d of this.#program) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      const groups = Math.ceil(d.count / WORKGROUP);
      pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
    }
    pass.end();
  }

  async solve(scene: DitherScene): Promise<Float32Array> {
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

  destroy() {
    for (const b of this.#buffers) b.destroy();
  }
}
