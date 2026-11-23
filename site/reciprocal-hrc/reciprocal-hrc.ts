// Reciprocal HRC: HRC's frustums, rays and extensions, with a merge that reads each parent ray at its own start, through
// its own first half. The merge is then the transpose of the extensions: light from p to q equals light from q to p.
// Media: a pixel holds absorption σ per unit path and source radiance S. In a 1-px ray of rise d each of its two half
// pixels has t = exp(-w_d σ/2) and r = S (1 - t), with w_d from MEDIUM_LENGTH.
// Storage, per frustum: column block j of a length L holds the rays that start at column j L, as f16 (r, t) by end row
// and merged radiance (RGB9E5) by start row. A length keeps its even blocks; its odd blocks wait in length 4L's buffer.

import type { Profiler, Traffic } from '../_shared/profiler';

/** World arrays [x * N + y], y up: emission rgb (3 per pixel) and solid (1 opaque, 0 free). A pixel with emission is a
 * light and is solid whatever `solid` says. */
export interface ReciprocalScene {
  emission: Float32Array;
  solid: Float32Array | Uint8Array;
  /** absorption per pixel of path length (∞ opaque; none: all 0). A pixel with absorption and emission is an emitting
   * medium; with emission and no absorption it is a light */
  absorption?: Float32Array;
  /** with the normals option: unit normals (3 per pixel), z out of the plane (none: all (0, 0, 1)) */
  normal?: Float32Array;
}

export interface Options {
  /** the Wu split: the merge gives an odd parent's far light to its two bent orders by its slope across rows, not ½ / ½
   * (off by default) */
  wuSplit?: boolean;
  /** the coupling read: the light a first half hands to a bent parent's second half follows a half-limited slope of the
   * first half's light (off by default): 'fan', across its fan at a fixed start (strength 0.53), or 'rows', across its
   * start rows (0.1) */
  couplingRead?: false | 'fan' | 'rows';
  /** offscreen light from a producer on the GPU: outside the scene is free space, and the light entering it comes from
   * the producer (off: dark and solid). Its WGSL defines `fn outsidePrepare(length: i32, row: i32, local: u32)`, run by
   * each thread of a fill workgroup (FILL_THREADS) for one start row before a barrier, and
   * `fn outsideValue(length: i32, block: i32, rise: i32, startRow: i32) -> vec3f`, the light entering the scene over the
   * lines of the ray of that length and rise that starts at column 0 (block 0) and row startRow. It may read
   * `params.frustum`, `N` and its buffers, bound at 20, 21, ... (each must be used). It runs every frame, in the solve,
   * before each length's merge reads it. */
  producer?: OutsideProducer;
  /** shade each pixel's 12 directional bins with the normal map (upload's normal) and write the irradiance vector to
   * vectorBuffer (off by default) */
  normals?: boolean;
  /** how rays and merged radiance are stored: 'packed' (the default: rays f16 (r, t), merged radiance RGB9E5), 'f16'
   * (both f16) or 'f32' (both f32) */
  storage?: Storage;
}

export type Storage = 'packed' | 'f16' | 'f32';

export interface OutsideProducer {
  wgsl: string;
  buffers: GPUBuffer[];
}

/** a half pixel's path-length factor in a 1-px ray of rise 0 and of rise ±1: the minimax line in |s| to √(1 + s²) */
export const MEDIUM_LENGTH = [0.95509, 1.3693] as const;

const WORKGROUP = 256;
/** threads in a fill workgroup (a producer's outsidePrepare strides by it) */
export const FILL_THREADS = WORKGROUP;
const MERGE_THREADS = 65536; // a merge thread takes a run of rises, as long as the dispatch keeps about this many threads
const MAX_GROUPS_X = 32768;
const PARAM_STRIDE = 256; // uniform offsets must be multiples of 256
/** bytes per stored ray and per merged radiance */
const SIZES: Record<Storage, { ray: number; merged: number }> = { packed: { ray: 8, merged: 4 }, f16: { ray: 8, merged: 8 }, f32: { ray: 16, merged: 16 } };
const RISE_CLASSES = 2; // half pixels per pixel and axis: in a 1-px ray of rise 0, and of rise ±1

const riseCount = (length: number) => 2 * length + 1;
/** rays of one length kept in one frustum: N/length - 1 column blocks, every row */
const storedRays = (N: number, length: number) => (N / length - 1) * riseCount(length) * N;
const evenRays = (N: number, length: number) => (N / length / 2) * riseCount(length) * N;
const oddRays = (N: number, length: number) => (N / length / 2 - 1) * riseCount(length) * N;

/** the angle of the slopes (rise ± ½) / length, clipped to [-1, 1]. 1-px rays: ½ ln 2 between row neighbours, and
 * π/2 - ln 2 from a pixel's left half pixel to its right one */
export function coneWeight(length: number, rise: number) {
  if (length === 1) return rise === 0 ? Math.PI / 2 - Math.log(2) : Math.log(2) / 2;
  return Math.atan(Math.min(1, (rise + 0.5) / length)) - Math.atan(Math.max(-1, (rise - 0.5) / length));
}

/** every level's cone weights in one table, level 0 first */
function coneTable(N: number) {
  const offsets: number[] = [], weights: number[] = [];
  for (let length = 1; length <= N; length *= 2) {
    offsets.push(weights.length);
    for (let rise = -length; rise <= length; rise++) weights.push(coneWeight(length, rise));
  }
  return { offsets, weights: new Float32Array(weights) };
}

/** With a producer, block 0 of every length holds the outside's light (a fill writes it, before the merge reads it) and
 * blocks ≥ 1 the merge's, at the scene's rows and at the pad rows the merge reads. Per level (length 2^level) and rise,
 * the start rows read, traced up from the pixels pass: an even child reads its parents at its own row, an odd child
 * rise + side rows on (parent rise 2 rise + side), the Wu split one row more each way. */
export function outsideLayout(N: number) {
  type Range = [number, number];
  const widen = (r: Range, lo: number, hi: number) => ((r[0] = Math.min(r[0], lo)), (r[1] = Math.max(r[1], hi)));
  const outside: Range[][] = [[[-1, N], [-1, N], [-1, N]]];
  const merged: Range[][] = [[[-1, N], [-1, N], [-1, N]]]; // length 1: the pixels pass's 1-px rays
  for (let length = 1; length < N; length *= 2) {
    const parent = 2 * length, zero: Range[] = [], others: Range[] = [];
    for (let i = 0; i < riseCount(parent); i++) (zero.push([Infinity, -Infinity]), others.push([Infinity, -Infinity]));
    merged[merged.length - 1].forEach(([lo, hi], index) => {
      const rise = index - length;
      for (let side = -1; side <= 1; side++) {
        const d = 2 * rise + side, shift = rise + side;
        if (Math.abs(d) > parent) continue;
        widen(others[d + parent], lo - 1, hi + 1);
        widen(zero[d + parent], lo - shift - 1, hi - shift + 1);
        widen(others[d + parent], lo - shift - 1, hi - shift + 1);
      }
    });
    outside.push(zero.map(([lo, hi]) => (lo > hi ? [0, -1] : [lo, hi])));
    merged.push(others.map(([lo, hi]) => [Math.min(0, lo), Math.max(N - 1, hi)]));
  }
  const pad = (ranges: Range[]) => Math.max(0, ...ranges.filter(([lo, hi]) => lo <= hi).flatMap(([lo, hi]) => [-lo, hi - N + 1]));
  return outside.map((zero, level) => {
    const length = 2 ** level, pad0 = pad(zero), pad1 = level === 0 ? 0 : pad(merged[level]);
    // stored: block 0 at rows -pad0 ... N + pad0 - 1; blocks 1 ... (from length 2 to N / 2) at rows -pad1 ... N + pad1 - 1
    const others = length > 1 && length < N ? (N / length - 1) * riseCount(length) * (N + 2 * pad1) : 0;
    return { length, zero, merged: merged[level], pad0, pad1, cells: riseCount(length) * (N + 2 * pad0) + others };
  });
}
const common = (N: number, free: boolean, storage: Storage) => /* wgsl */ `
const N: i32 = ${N};${formats(storage)}
const CONE_OFFSETS = array<i32, ${Math.log2(N) + 1}>(${coneTable(N).offsets.join(', ')});
// riseClass: where the scene's half pixels of rise-±1 1-px rays start
struct Params { length: i32, count: u32, frustum: i32, riseClass: u32 };
@group(0) @binding(0) var<storage, read> scene: array<vec4f>;   // [rise class][axis][column][row]: a half pixel (r rgb, t)
@group(0) @binding(1) var<uniform> params: Params;

fn threadIndex(id: vec3u, groups: vec3u) -> u32 { return id.x + id.y * groups.x * ${WORKGROUP}u; }

// frustum: axis frustum / 2, mirrored along the axis if odd. Off the rows: ${free ? 'free' : 'solid and dark'}.
fn pixelAt(column: i32, row: i32, rise: i32) -> vec4f {
  if (row < 0 || row >= N) { return vec4f(${free ? '0.0, 0.0, 0.0, 1.0' : '0.0'}); }
  let axis = params.frustum >> 1; let mirrored = (params.frustum & 1) == 1;
  return scene[select(0u, params.riseClass, rise != 0) + u32((axis * N + select(column, N - 1 - column, mirrored)) * N + row)];
}

struct Ray { transmittance: f32, radiance: vec3f };
const NOTHING = Ray(1.0, vec3f(0.0));
fn extend(first: Ray, next: Ray) -> Ray {
  return Ray(first.transmittance * next.transmittance, next.radiance + next.transmittance * first.radiance);
}
fn halfPixel(column: i32, row: i32, rise: i32) -> Ray { let p = pixelAt(column, row, rise); return Ray(p.w, p.rgb); }

// A length's rises run -length ... length.
fn riseCount(length: i32) -> i32 { return 2 * length + 1; }
fn riseIndex(length: i32, rise: i32) -> i32 { return rise + length; }
fn riseAt(length: i32, index: i32) -> i32 { return index - length; }
fn hasRise(length: i32, rise: i32) -> bool { return abs(rise) <= length; }

// A ray is its two halves, or, when they can't have equal rises, the mean of both orders of the two nearest rises.
struct Pair { leftRise: i32, rightRise: i32, weight: f32 };
fn pairOf(length: i32, rise: i32, which: i32) -> Pair {
  if (rise % 2 == 0) {
    if (which == 0) { return Pair(rise / 2, rise / 2, 1.0); }
    return Pair(0, 0, 0.0);
  }
  let lower = (rise - 1) / 2; let upper = (rise + 1) / 2;
  if (which == 0) { return Pair(lower, upper, 0.5); }
  return Pair(upper, lower, 0.5);
}
`;

// the stored forms of a ray (RayWord) and of merged radiance (MergedWord), and their conversions
const formats = (storage: Storage) => {
  const f16Ray = /* wgsl */ `
alias RayWord = vec2u;
fn unpackRay(packed: vec2u) -> Ray {
  let blueAndTransmittance = unpack2x16float(packed.y);
  return Ray(blueAndTransmittance.y, vec3f(unpack2x16float(packed.x), blueAndTransmittance.x));
}
fn packRay(ray: Ray) -> vec2u {
  return vec2u(pack2x16float(ray.radiance.rg), pack2x16float(vec2f(ray.radiance.b, ray.transmittance)));
}`;
  if (storage === 'f32')
    return /* wgsl */ `
alias RayWord = vec4f;
fn unpackRay(packed: vec4f) -> Ray { return Ray(packed.w, packed.rgb); }
fn packRay(ray: Ray) -> vec4f { return vec4f(ray.radiance, ray.transmittance); }
alias MergedWord = vec4f;
fn unpackMerged(packed: vec4f) -> vec3f { return packed.rgb; }
fn packMerged(merged: vec3f) -> vec4f { return vec4f(merged, 0.0); }`;
  if (storage === 'f16')
    return /* wgsl */ `${f16Ray}
alias MergedWord = vec2u;
fn unpackMerged(packed: vec2u) -> vec3f { return vec3f(unpack2x16float(packed.x), unpack2x16float(packed.y).x); }
fn packMerged(merged: vec3f) -> vec2u { return vec2u(pack2x16float(merged.rg), pack2x16float(vec2f(merged.b, 0.0))); }`;
  return /* wgsl */ `${f16Ray}
alias MergedWord = u32;   // RGB9E5
fn unpackMerged(packed: u32) -> vec3f {
  return vec3f(vec3u(packed, packed >> 9u, packed >> 18u) & vec3u(511u)) * bitcast<f32>(((packed >> 27u) + 103u) << 23u);
}
fn packMerged(merged: vec3f) -> u32 {
  let exponent = clamp(i32((bitcast<u32>(max(merged.r, max(merged.g, merged.b))) >> 23u) & 255u) - 111, 0, 31);
  let mantissa = vec3u(min(max(merged, vec3f(0.0)) * bitcast<f32>(u32(151 - exponent) << 23u) + 0.5, vec3f(511.0)));
  return mantissa.r | (mantissa.g << 9u) | (mantissa.b << 18u) | (u32(exponent) << 27u);
}`;
};
const loadRayFn = /* wgsl */ `
fn loadRay(slot: i32) -> Ray { return unpackRay(raysIn[slot]); }`;
const storeRayFn = /* wgsl */ `
fn storeRay(slot: u32, ray: Ray) { raysOut[slot] = packRay(ray); }`;
const loadMergedFn = /* wgsl */ `
fn loadMerged(slot: i32) -> vec3f { return unpackMerged(mergedIn[slot]); }`;
const storeMergedFn = /* wgsl */ `
fn storeMerged(slot: u32, merged: vec3f) { mergedOut[slot] = packMerged(merged); }`;

// A 1-px ray is built where it is read, from its two half pixels (the leaves, read from the scene). A ray whose end is
// off the rows is not stored: it carries nothing, and blocks unless the outside is free (it then lies outside). Reads
// load first and choose after, with no branch around the load (a branch keeps a thread from issuing its loads
// together). A slot off the rows may lie outside the buffer; WebGPU then reads some value inside it, which is discarded.
const rayFn = (oddBlocks: boolean, free: boolean) => /* wgsl */ `
fn ray(length: i32, block: i32, rise: i32, startRow: i32) -> Ray {
  if (length == 1) { return extend(halfPixel(block, startRow, rise), halfPixel(block, startRow + rise, rise)); }
  let endRow = startRow + rise; let stored = endRow >= 0 && endRow < N;
  let slot = ((block >> 1) * riseCount(length) + riseIndex(length, rise)) * N + endRow;${
    oddBlocks
      ? `
  var packed: RayWord;
  if ((block & 1) == 1) { packed = raysInOdd[slot]; } else { packed = raysIn[slot]; }
  let loaded = unpackRay(packed);`
      : `
  let loaded = loadRay(slot);`
  }
  return Ray(select(${free ? '1.0' : '0.0'}, loaded.transmittance, stored), select(vec3f(0.0), loaded.radiance, stored));
}
`;

// splitOut: write even blocks to raysOut and odd blocks to raysOutOdd
const extensionsShader = (N: number, splitOut: boolean, free: boolean, storage: Storage) => /* wgsl */ `
${common(N, free, storage)}
@group(0) @binding(2) var<storage, read> raysIn: array<RayWord>;          // length / 2: even blocks
@group(0) @binding(9) var<storage, read> raysInOdd: array<RayWord>;       // and odd blocks
@group(0) @binding(3) var<storage, read_write> raysOut: array<RayWord>;${
  splitOut
    ? `
@group(0) @binding(10) var<storage, read_write> raysOutOdd: array<RayWord>;`
    : ''
}
${loadRayFn}
${storeRayFn}
${rayFn(true, free)}
@compute @workgroup_size(${WORKGROUP})
fn extensions(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let length = params.length; let half = length / 2;
  let endRow = i32(i % u32(N)); let blockAndRise = i32(i / u32(N));
  let rise = riseAt(length, blockAndRise % riseCount(length)); let block = blockAndRise / riseCount(length);
  let startRow = endRow - rise;
  var sum = Ray(0.0, vec3f(0.0));
  for (var which = 0; which < 2; which++) {
    let pair = pairOf(length, rise, which);
    if (pair.weight == 0.0) { continue; }
    let left = ray(half, 2 * block, pair.leftRise, startRow);
    let right = ray(half, 2 * block + 1, pair.rightRise, startRow + pair.leftRise);
    let joined = extend(left, right);
    sum = Ray(sum.transmittance + pair.weight * joined.transmittance, sum.radiance + pair.weight * joined.radiance);
  }${
    splitOut
      ? `
  let slot = u32(((block >> 1) * riseCount(length) + blockAndRise % riseCount(length)) * N + endRow);
  if ((block & 1) == 0) { storeRay(slot, sum); } else { raysOutOdd[slot] = packRay(sum); }`
      : `
  storeRay(i, sum);`
  }
}
`;

// a parent's merged radiance as a ray reads it: with the Wu split, an odd parent's shifted between its two bent orders
// (sign +1 for the order whose left half has the higher rise)
const parentRead = (wu: boolean, args: string, sign: string) => (wu ? `wuShare(rowsAt(${args}), ${sign})` : `mergedAt(${args})`);

// with a producer: a stored slot (see outsideLayout): block 0's rows first, then the other blocks'
const paddedSlotFn = (N: number) => {
  const layout = outsideLayout(N);
  return /* wgsl */ `
const PAD0 = array<i32, ${layout.length}>(${layout.map((l) => l.pad0).join(', ')});
const PAD1 = array<i32, ${layout.length}>(${layout.map((l) => l.pad1).join(', ')});
fn mergedSlot(length: i32, block: i32, index: i32, startRow: i32) -> i32 {
  let level = firstTrailingBit(u32(length)); let pad0 = PAD0[level]; let pad1 = PAD1[level];
  let others = riseCount(length) * (N + 2 * pad0) + ((block - 1) * riseCount(length) + index) * (N + 2 * pad1) + startRow + pad1;
  return select(others, index * (N + 2 * pad0) + startRow + pad0, block == 0);
}
fn storedRow(length: i32, block: i32, startRow: i32) -> bool {
  let level = firstTrailingBit(u32(length)); let pad = select(PAD1[level], PAD0[level], block == 0);
  return startRow >= -pad && startRow < N + pad;
}`;
};

const mergeFns = (wu: boolean, padded: boolean, N: number) => /* wgsl */ `
@group(0) @binding(8) var<storage, read> coneWeights: array<f32>;
fn coneWeight(length: i32, rise: i32) -> f32 {
  return coneWeights[CONE_OFFSETS[firstTrailingBit(u32(length))] + riseIndex(length, rise)];
}${
  padded
    ? `${paddedSlotFn(N)}
// one load: block 0 holds the outside's light, the other blocks the merge's, the pad rows included
fn mergedAt(length: i32, block: i32, rise: i32, startRow: i32) -> vec3f {
  let merged = loadMerged(mergedSlot(length, block, riseIndex(length, rise), startRow));
  return select(vec3f(0.0), merged, hasRise(length, rise) && storedRow(length, block, startRow));
}
`
    : `
// none at length N, in column block 0 or off the rows (loaded, then discarded)
fn mergedAt(length: i32, block: i32, rise: i32, startRow: i32) -> vec3f {
  let merged = loadMerged(((block - 1) * riseCount(length) + riseIndex(length, rise)) * N + startRow);
  let outside = length == N || block < 1 || startRow < 0 || startRow >= N;
  return select(vec3f(0.0), select(merged, vec3f(0.0), outside), hasRise(length, rise));
}
`
}// a parent's merged radiance carried through the part of it before the ray
fn carry(merged: vec3f, before: Ray, cone: f32) -> vec3f {
  return before.transmittance * merged + cone * before.radiance;
}${
  wu
    ? `
// The Wu split, κ = ¼: an odd parent's far light goes ½ ± κ r to its two bent orders, with
// r = (M(y0 + 1) - M(y0 - 1)) / (M(y0 - 1) + M(y0) + M(y0 + 1)) per channel over the parent's start rows.
struct Rows { below: vec3f, at: vec3f, above: vec3f };
fn rowsAt(length: i32, block: i32, rise: i32, startRow: i32) -> Rows {
  return Rows(mergedAt(length, block, rise, startRow - 1), mergedAt(length, block, rise, startRow), mergedAt(length, block, rise, startRow + 1));
}
fn wuShare(m: Rows, sign: f32) -> vec3f {
  let sum = m.below + m.at + m.above;
  let r = select(vec3f(0.0), (m.above - m.below) / sum, sum > vec3f(0.0));
  return m.at * (1.0 + sign * 0.5 * r);   // 2κ: gather and merge take ½ of it
}`
    : ''
}
// The merged radiance of a ray of rise 'rise' from its three parents, weighted as pairOf builds them: the bent parent
// below (½), the parent of twice its rise (1) and the bent parent above (½). Before a first half there is nothing;
// before a second half, the parent's first half.
fn gather(parentLength: i32, rise: i32, lower: vec3f, lowerBefore: Ray, same: vec3f, sameBefore: Ray, upper: vec3f, upperBefore: Ray) -> vec3f {
  var merged = vec3f(0.0);
  if (hasRise(parentLength, 2 * rise - 1)) { merged += 0.5 * carry(lower, lowerBefore, coneWeight(parentLength, 2 * rise - 1)); }
  merged += carry(same, sameBefore, coneWeight(parentLength, 2 * rise));
  if (hasRise(parentLength, 2 * rise + 1)) { merged += 0.5 * carry(upper, upperBefore, coneWeight(parentLength, 2 * rise + 1)); }
  return merged;
}
fn leftHalf(length: i32, block: i32, rise: i32, endRow: i32) -> Ray {
  let half = ray(length, block, rise, endRow - rise); let exists = hasRise(length, rise);
  return Ray(select(1.0, half.transmittance, exists), select(vec3f(0.0), half.radiance, exists));
}
// gather's sum, loading the parents itself; an odd block is a second half. Kept as a loop: written out like gather,
// the pixels pass rounds differently.
fn merge(length: i32, block: i32, rise: i32, row: i32) -> vec3f {
  let parentLength = 2 * length; let isRight = (block & 1) == 1;
  var merged = vec3f(0.0);
  for (var side = -1; side <= 1; side++) {
    let parentRise = 2 * rise + side;
    if (!hasRise(parentLength, parentRise)) { continue; }
    let beforeRise = select(0, parentRise - rise, isRight);
    var before = NOTHING;
    if (isRight) { before = ray(length, block - 1, beforeRise, row - beforeRise); }
    merged += select(1.0, 0.5, side != 0) * carry(${parentRead(wu, 'parentLength, block / 2, parentRise, row - beforeRise', 'f32(select(-side, side, isRight))')}, before, coneWeight(parentLength, parentRise));
  }
  return merged;
}
`;

// One thread per (block, row) and a run of RUN consecutive rises, so that the parents and first halves one rise shares
// with the next stay in registers.
const mergeShader = (N: number, run: number, wu: boolean, read: false | 'fan' | 'rows', padded: boolean, storage: Storage) => /* wgsl */ `
${common(N, padded, storage)}
@group(0) @binding(2) var<storage, read> raysIn: array<RayWord>;         // length
@group(0) @binding(4) var<storage, read> mergedIn: array<MergedWord>;    // parent length
@group(0) @binding(5) var<storage, read_write> mergedOut: array<MergedWord>;  // length
${loadRayFn}
${loadMergedFn}
${storeMergedFn}
${rayFn(false, padded)}
${mergeFns(wu, padded, N)}${
  read
    ? `
// The coupling read: a first half's light enters a bent parent's second half weighted 1 ± g s, + where the first half
// has the lower rise. With below, own and above three of the first half's neighbours' light, up = above - own and
// down = own - below, s is the mean of the slope (up + down) / sum and its limited form 2 minmod(up, down) / sum,
// sum = below + own + above (minmod: the smaller of two differences of one sign, else 0), per channel. ${
      read === 'fan'
        ? `The neighbours
// are its fan at a fixed start, rises ∓ 1 (a rise past the fan's edge reads as its own), and g = 0.53.`
        : `The neighbours
// are its start rows ∓ 1, and g = 0.4 (strength 0.1).`
    }
fn couplingSlope(length: i32, block: i32, rise: i32, endRow: i32, own: vec3f) -> vec3f {${
      read === 'fan'
        ? `
  let below = select(own, leftHalf(length, block, rise - 1, endRow - 1).radiance, hasRise(length, rise - 1));
  let above = select(own, leftHalf(length, block, rise + 1, endRow + 1).radiance, hasRise(length, rise + 1));`
        : `
  let below = leftHalf(length, block, rise, endRow - 1).radiance; let above = leftHalf(length, block, rise, endRow + 1).radiance;`
    }
  let up = above - own; let down = own - below; let sum = below + own + above;
  let limited = select(vec3f(0.0), sign(up) * min(abs(up), abs(down)), up * down > vec3f(0.0));
  return select(vec3f(0.0), (limited + 0.5 * (up + down)) / sum, sum > vec3f(0.0));
}
fn steered(before: Ray, slope: vec3f, sign: f32) -> Ray {
  return Ray(before.transmittance, before.radiance * (1.0 + sign * ${read === 'fan' ? '0.53' : '0.4'} * slope));
}`
    : ''
}
const RUN: i32 = ${run};${
  padded
    ? `
// per level and run of rises, the start rows the merge computes (the rows of the scene and the pads read): a block's
// threads are its runs' rows, one after another
const RUN_OFFSETS = array<i32, ${Math.log2(N) + 1}>(${runRows(N, run).offsets.join(', ')});
const RUN_THREADS = array<i32, ${Math.log2(N) + 1}>(${runRows(N, run).totals.join(', ')});
@group(0) @binding(19) var<storage, read> runRows: array<vec4i>;   // first row, last row, first thread, 0`
    : ''
}
@compute @workgroup_size(${WORKGROUP})
fn mergeLevel(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let parentLength = params.length; let length = parentLength / 2;
  let count = riseCount(length); let runs = (count + RUN - 1) / RUN;${
    padded
      ? `
  let level = firstTrailingBit(u32(length)); let perBlock = RUN_THREADS[level]; let base = RUN_OFFSETS[level];
  let j = i32(i) % perBlock;
  var low = 0; var high = runs - 1;   // the run whose threads hold j
  while (low < high) { let middle = (low + high + 1) / 2; if (runRows[base + middle].z <= j) { low = middle; } else { high = middle - 1; } }
  let rows = runRows[base + low];
  let row = rows.x + j - rows.z; let blockAndRun = (i32(i) / perBlock) * runs + low;`
      : `
  let row = i32(i % u32(N)); let blockAndRun = i32(i / u32(N));`
  }
  let first = (blockAndRun % runs) * RUN; let last = min(first + RUN, count);
  let block = blockAndRun / runs + 1; let parentBlock = block / 2;
  if ((block & 1) == 0) {
    var lower = ${wu ? 'rowsAt' : 'mergedAt'}(parentLength, parentBlock, 2 * riseAt(length, first) - 1, row);
    for (var q = first; q < last; q++) {
      let rise = riseAt(length, q);
      let upper = ${wu ? 'rowsAt' : 'mergedAt'}(parentLength, parentBlock, 2 * rise + 1, row);
      let merged = gather(parentLength, rise, ${wu ? 'wuShare(lower, 1.0)' : 'lower'}, NOTHING, mergedAt(parentLength, parentBlock, 2 * rise, row), NOTHING, ${wu ? 'wuShare(upper, -1.0)' : 'upper'}, NOTHING);
      storeMerged(u32(${padded ? 'mergedSlot(length, block, q, row)' : '((block - 1) * count + q) * N + row'}), merged);
      lower = upper;
    }
  } else {
    var lowerBefore = leftHalf(length, block - 1, riseAt(length, first) - 1, row);
    var sameBefore = leftHalf(length, block - 1, riseAt(length, first), row);${
      read
        ? `
    var lowerSlope = couplingSlope(length, block - 1, riseAt(length, first) - 1, row, lowerBefore.radiance);
    var sameSlope = couplingSlope(length, block - 1, riseAt(length, first), row, sameBefore.radiance);`
        : ''
    }${
      wu
        ? `
    var lower = rowsAt(parentLength, parentBlock, 2 * riseAt(length, first) - 1, row - (riseAt(length, first) - 1));`
        : ''
    }
    for (var q = first; q < last; q++) {
      let rise = riseAt(length, q);
      let upperBefore = leftHalf(length, block - 1, rise + 1, row);${
        read
          ? `
      let upperSlope = couplingSlope(length, block - 1, rise + 1, row, upperBefore.radiance);`
          : ''
      }${
        wu
          ? `
      let upper = rowsAt(parentLength, parentBlock, 2 * rise + 1, row - (rise + 1));`
          : ''
      }
      let merged = gather(parentLength, rise,
        ${wu ? 'wuShare(lower, -1.0)' : 'mergedAt(parentLength, parentBlock, 2 * rise - 1, row - (rise - 1))'}, ${read ? 'steered(lowerBefore, lowerSlope, 1.0)' : 'lowerBefore'},
        mergedAt(parentLength, parentBlock, 2 * rise, row - rise), sameBefore,
        ${wu ? 'wuShare(upper, 1.0)' : 'mergedAt(parentLength, parentBlock, 2 * rise + 1, row - (rise + 1))'}, ${read ? 'steered(upperBefore, upperSlope, -1.0)' : 'upperBefore'});
      storeMerged(u32(${padded ? 'mergedSlot(length, block, q, row)' : '((block - 1) * count + q) * N + row'}), merged);
      lowerBefore = sameBefore; sameBefore = upperBefore;${read ? ' lowerSlope = sameSlope; sameSlope = upperSlope;' : ''}${
        wu
          ? `
      // the next rise's lower parent is this upper one, started a row higher
      lower = Rows(upper.at, upper.above, mergedAt(parentLength, parentBlock, 2 * rise + 1, row - rise + 1));`
          : ''
      }
    }
  }
}
`;

// The bottom of the tree. Half pixels are its leaves: the 1-px rays are built from them (the extensions at length 1),
// and here the merge runs to length 1 and then into them. A pixel is the left half pixel of the three 1-px rays from
// its row and the right half pixel of the three that end in its row: rise 0 from its own row, 1 from the row below, -1
// from the row above.
const pixelsShader = (N: number, wu: boolean, padded: boolean, normals: boolean, storage: Storage) => /* wgsl */ `
${common(N, padded, storage)}
// (1 - m, (1 - m) / (1 - t)), m = (1 - t) / (-ln t) the mean transmittance over the half pixel
fn slab(t: f32) -> vec2f {
  let a = select(-log(max(t, 1e-30)), 0.0, t >= 1.0);   // log(1) need not be 0 on the GPU
  if (a < 1e-3) { return vec2f(a / 2.0 - a * a / 6.0, 0.5 + a / 12.0); }
  let lost = 1.0 - (1.0 - t) / a;
  return vec2f(lost, lost / (1.0 - t));
}
@group(0) @binding(2) var<storage, read> raysIn: array<RayWord>;        // bound, never read: 1-px rays come from the scene
@group(0) @binding(4) var<storage, read> mergedIn: array<MergedWord>;   // length 2
@group(0) @binding(6) var<storage, read_write> axisFluence: array<vec4f>;  // [axis][world column][row]${
  normals
    ? `
@group(0) @binding(12) var<storage, read> normalMap: array<vec4f>;      // world [x * N + y]: (normal, 0), z out of the plane
@group(0) @binding(13) var<storage, read_write> axisVector: array<vec4f>;  // [axis][world column][row]: (irradiance vector, 0, 0)`
    : ''
}
${loadRayFn}
${loadMergedFn}
${rayFn(false, padded)}
${mergeFns(wu, padded, N)}${
  padded
    ? `
@group(0) @binding(18) var<storage, read> outsideOne: array<MergedWord>;   // length 1, padded: the outside cells only
// a 1-px ray's merged radiance; in column 0, the outside's (elsewhere the merge's, off the rows from the pads)
fn merged1(column: i32, rise: i32, startRow: i32) -> vec3f {
  let outside = unpackMerged(outsideOne[mergedSlot(1, 0, riseIndex(1, rise), startRow)]);
  return select(merge(1, column, rise, startRow), outside, column < 1);
}`
    : ''
}
@compute @workgroup_size(${WORKGROUP})
fn pixels(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let column = i32(i) / N; let row = i32(i) % N;
  var value = vec3f(0.0);${
    normals
      ? `
  // Each of the 12 bins (3 rises, 4 frustums) holds the light from one direction u in the plane, against the bin's travel.
  // A bin lights the pixel as light from u at 45° above the plane: a flat normal takes it whole.
  let axis = params.frustum >> 1; let along = select(1.0, -1.0, (params.frustum & 1) == 1);
  let worldColumn = select(column, N - 1 - column, (params.frustum & 1) == 1);
  let normal = normalMap[select(worldColumn * N + row, row * N + worldColumn, axis == 1)].xyz;
  var vector = vec2f(0.0);`
      : ''
  }
  for (var rise = -1; rise <= 1; rise++) {
    // The merge at length 1/2, with the 1-px cone weights (exact: π/2 - ln 2 within a row, ½ ln 2 between rows). A left
    // half takes its 1-px ray's merged radiance through nothing before it; a right half through the left half pixel.
    // (At rise 0 both halves are the same 1-px ray's.)
    let cone = coneWeight(1, rise);
    let fromRow = ${padded ? 'merged1' : 'merge'}(${padded ? '' : '1, '}column, rise, row);
    var toRow = fromRow;
    if (rise != 0) { toRow = ${padded ? 'merged1' : 'merge'}(${padded ? '' : '1, '}column, rise, row - rise); }
    let left = carry(fromRow, NOTHING, cone);
    let right = carry(toRow, halfPixel(column, row - rise, rise), cone);
    // Both halves deposit: their light over a half pixel's length (½), less what the medium absorbs over it, plus its
    // own light along the ray.
    let own = pixelAt(column, row, rise); let s = slab(own.w);
${
      normals
        ? `    let bin = 0.5 * ((1.0 - s.x) * (left + right) + 2.0 * cone * s.y * own.rgb);
    let u = -normalize(select(vec2f(along, f32(rise)), vec2f(f32(rise), along), axis == 1));
    value += max(0.0, normal.z + dot(normal.xy, u)) * bin;
    vector += dot(bin, vec3f(0.2126, 0.7152, 0.0722)) * u;`
        : `    value += 0.5 * ((1.0 - s.x) * (left + right) + 2.0 * cone * s.y * own.rgb);`
    }
  }
  value *= select(0.0, 1.0, pixelAt(column, row, 0).w > 0.0);   // a solid pixel holds no fluence
  let mirrored = (params.frustum & 1) == 1;
  let slot = ((params.frustum >> 1) * N + select(column, N - 1 - column, mirrored)) * N + row;
  if (mirrored) { value += axisFluence[slot].rgb; }   // the mirror-image frustum adds to the forward one
  axisFluence[slot] = vec4f(value, 0.0);${
    normals
      ? `
  vector *= select(0.0, 1.0, pixelAt(column, row, 0).w > 0.0);
  if (mirrored) { vector += axisVector[slot].xy; }
  axisVector[slot] = vec4f(vector, 0.0, 0.0);`
      : ''
  }
}
`;

// The outside's light, written into block 0 of a length's merged radiance before the merge reads it: one workgroup per
// start row (rows -pad0 ... N + pad0 - 1), its threads over the rises, each rise at the rows the merge reads only. The
// source prepares a row first (outsidePrepare, in uniform control flow), then gives each cell (outsideValue).
const fillShader = (N: number, source: string, storage: Storage) => /* wgsl */ `
${common(N, true, storage)}
@group(0) @binding(5) var<storage, read_write> mergedOut: array<MergedWord>;
@group(0) @binding(19) var<storage, read> outsideRows: array<vec2i>;   // per level (as the cone weights) and rise
${storeMergedFn}
${paddedSlotFn(N)}
${source}
@compute @workgroup_size(${WORKGROUP})
fn fill(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32) {
  let length = params.length;
  let row = i32(group.x + group.y * groups.x) - PAD0[firstTrailingBit(u32(length))];
  outsidePrepare(length, row, local);
  workgroupBarrier();
  for (var index = i32(local); index < riseCount(length); index += ${WORKGROUP}) {
    let rows = outsideRows[CONE_OFFSETS[firstTrailingBit(u32(length))] + index];
    if (row < rows.x || row > rows.y) { continue; }
    storeMerged(u32(mergedSlot(length, 0, index, row)), outsideValue(length, 0, riseAt(length, index), row));
  }
}
`;

/** the start rows the merge computes, per level and run of rises (blocks ≥ 1): (first row, last row, the run's first
 * thread in its block, 0); where each level's runs start, and each level's threads per block */
function runRows(N: number, run: number) {
  const offsets: number[] = [], totals: number[] = [], rows: number[] = [];
  for (const level of outsideLayout(N)) {
    offsets.push(rows.length / 4);
    let threads = 0;
    for (let first = 0; first < riseCount(level.length); first += run) {
      const ranges = level.merged.slice(first, first + run), lo = Math.min(...ranges.map((r) => r[0])), hi = Math.max(...ranges.map((r) => r[1]));
      rows.push(lo, hi, threads, 0);
      threads += hi - lo + 1;
    }
    totals.push(threads);
  }
  return { offsets, totals, rows: new Int32Array(rows) };
}

// the fluence: the two axes' sum
const outputShader = (N: number, normals: boolean, storage: Storage) => /* wgsl */ `
${common(N, false, storage)}
@group(0) @binding(6) var<storage, read> axisFluence: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> result: array<vec4f>;  // world [x * N + y]: (fluence rgb, its luminance)${
  normals
    ? `
@group(0) @binding(13) var<storage, read> axisVector: array<vec4f>;
@group(0) @binding(14) var<storage, read_write> vectors: array<vec4f>;  // world [x * N + y]: see vectorBuffer`
    : ''
}
@compute @workgroup_size(${WORKGROUP})
fn output(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let x = i32(i) / N; let y = i32(i) % N;
  let value = axisFluence[x * N + y].rgb + axisFluence[(N + y) * N + x].rgb;   // [axis][column][row]
  result[i] = vec4f(value, dot(value, vec3f(0.2126, 0.7152, 0.0722)));${
    normals
      ? `
  let vector = axisVector[x * N + y].xy + axisVector[(N + y) * N + x].xy;
  let size = length(vector);
  vectors[i] = vec4f(size * (0.5 + 0.5 * cos(atan2(vector.y, vector.x) - vec3f(0.0, 2.0943951, 4.1887902))), size);`
      : ''
  }
}
`;

interface Dispatch {
  label: string;
  stage: string;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  count: number;
}

export class ReciprocalHRC {
  readonly device: GPUDevice;
  readonly N: number;
  readonly wuSplit: boolean;
  readonly couplingRead: false | 'fan' | 'rows';
  readonly normals: boolean;
  readonly storage: Storage;
  /** consecutive rises per merge thread */
  readonly mergeRun: number;
  profiler: Profiler | null = null;
  /** the scene's half pixels, [rise class][axis][column][row]: (r rgb, t) */
  readonly sceneBuffer: GPUBuffer;
  /** fluence, world [x * N + y]: (rgb, luminance) */
  readonly outputBuffer: GPUBuffer;
  /** with the normals option, world [x * N + y]: the irradiance vector E (the bins' luminance along their directions) as
   * (|E| (½ + ½ cos(φ - k 2π/3)) for k = 0, 1, 2, |E|), φ its angle */
  readonly vectorBuffer: GPUBuffer | null = null;
  #normalMap: GPUBuffer | null = null;
  #uploadData: Float32Array | null = null;
  #axisVector: GPUBuffer | null = null;
  #producer: OutsideProducer | null;
  #topOutside: GPUBuffer | null = null; // with a producer: merged radiance of length N (the outside's only)
  #outsideOne: GPUBuffer | null = null; // and of length 1
  #rowTables: GPUBuffer | null = null;
  #runRowsAt = 0;
  #rays = new Map<number, GPUBuffer>(); // even blocks by length; later also the merged radiance of length / 2
  #oddRays = new Map<number, GPUBuffer>();
  #topMerged: GPUBuffer; // merged radiance of length N / 2
  #axisFluence: GPUBuffer;
  #coneWeights: GPUBuffer;
  #params: GPUBuffer;
  #program: Dispatch[] = [];

  constructor(device: GPUDevice, N: number, { wuSplit = false, couplingRead = false, producer, normals = false, storage = 'packed' }: Options = {}) {
    if (N < 4 || (N & (N - 1)) !== 0) throw new Error('ReciprocalHRC: N must be a power of 2, at least 4');
    this.device = device;
    this.N = N;
    this.wuSplit = wuSplit;
    this.couplingRead = couplingRead;
    this.#producer = producer ?? null;
    this.normals = normals;
    this.storage = storage;
    const { ray: RAY_BYTES, merged: MERGED_BYTES } = SIZES[storage];
    // a merge dispatch has about 2 N² rises
    this.mergeRun = Math.min(16, 2 ** Math.max(0, Math.floor(Math.log2((2 * N * N) / MERGE_THREADS))));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, size: number, use = usage) => device.createBuffer({ label, size: Math.max(16, size), usage: use });
    this.sceneBuffer = buffer('reciprocal HRC scene', RISE_CLASSES * 2 * N * N * 16);
    this.outputBuffer = buffer('reciprocal HRC fluence', N * N * 16);
    // each buffer holds its even blocks, then the merged radiance of length / 2, and the odd blocks of length / 4
    const oddBytes = (length: number) => (length >= 8 ? oddRays(N, length / 4) * RAY_BYTES : 0);
    const padded = this.#producer !== null;
    const layout = padded ? outsideLayout(N) : null;
    const mergedBytes = (length: number) => (layout ? layout[Math.log2(length)].cells : storedRays(N, length)) * MERGED_BYTES;
    for (const length of this.#stored()) {
      const merged = length > 2 ? mergedBytes(length / 2) : 0;
      this.#rays.set(length, buffer(`reciprocal HRC rays ${length}`, Math.max(evenRays(N, length) * RAY_BYTES, merged, oddBytes(length))));
    }
    this.#topMerged = buffer(`reciprocal HRC merged ${N / 2}`, Math.max(mergedBytes(N / 2), oddBytes(N)));
    if (padded) {
      this.#topOutside = buffer(`reciprocal HRC merged ${N}`, mergedBytes(N));
      this.#outsideOne = buffer('reciprocal HRC merged 1', mergedBytes(1));
      // the rows each fill writes (per level and rise) and each merge computes (per level and run), in one buffer
      const fills = layout!.flatMap((l) => l.zero.flat()), runs = runRows(N, this.mergeRun).rows;
      this.#runRowsAt = Math.ceil((4 * fills.length) / 256) * 256; // a binding's offset is a multiple of 256 (and of 16)
      const rows = new Int32Array(this.#runRowsAt / 4 + runs.length);
      rows.set(fills);
      rows.set(runs, this.#runRowsAt / 4);
      this.#rowTables = buffer('reciprocal HRC outside rows', rows.byteLength);
      device.queue.writeBuffer(this.#rowTables, 0, rows);
    }
    for (const length of this.#stored())
      if (oddRays(N, length) > 0) this.#oddRays.set(length, 4 * length === N ? this.#topMerged : this.#rays.get(4 * length)!);
    this.#axisFluence = buffer('reciprocal HRC fluence per axis', 2 * N * N * 16);
    const cones = coneTable(N).weights;
    this.#coneWeights = buffer('reciprocal HRC cone weights', cones.byteLength);
    device.queue.writeBuffer(this.#coneWeights, 0, cones);
    if (normals) {
      this.#normalMap = buffer('reciprocal HRC normals', N * N * 16);
      this.#axisVector = buffer('reciprocal HRC irradiance vector per axis', 2 * N * N * 16);
      this.vectorBuffer = buffer('reciprocal HRC irradiance vector', N * N * 16);
    }
    this.#params = buffer('reciprocal HRC params', this.dispatches * PARAM_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.#build();
  }

  /** the lengths whose rays are stored, 2 ... N / 2 */
  #stored() {
    const lengths = [];
    for (let length = 2; length < this.N; length *= 2) lengths.push(length);
    return lengths;
  }

  get dispatches() {
    const fills = this.#producer ? this.#stored().length + 2 : 0;
    return 4 * (2 * this.#stored().length + 1 + fills) + 1;
  }

  #build() {
    const { device, N } = this;
    const pipeline = (code: string, entryPoint: string) =>
      device.createComputePipeline({
        label: `reciprocal HRC ${entryPoint}`,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: `reciprocal HRC ${entryPoint}`, code }), entryPoint },
      });
    const producer = this.#producer, padded = producer !== null;
    const pipelines = {
      extensions: pipeline(extensionsShader(N, false, padded, this.storage), 'extensions'),
      extensionsSplit: pipeline(extensionsShader(N, true, padded, this.storage), 'extensions'),
      merge: pipeline(mergeShader(N, this.mergeRun, this.wuSplit, this.couplingRead, padded, this.storage), 'mergeLevel'),
      pixels: pipeline(pixelsShader(N, this.wuSplit, padded, this.normals, this.storage), 'pixels'),
      output: pipeline(outputShader(N, this.normals, this.storage), 'output'),
      ...(producer ? { fill: pipeline(fillShader(N, producer.wgsl, this.storage), 'fill') } : {}),
    };
    const params = new ArrayBuffer(this.dispatches * PARAM_STRIDE);
    let slot = 0;
    type Variant = 'extensions' | 'extensionsSplit' | 'merge' | 'pixels' | 'output' | 'fill';
    const add = (stage: 'extensions' | 'merge' | 'pixels' | 'output' | 'outside', label: string, frustum: number, length: number, count: number, buffers: Record<number, GPUBuffer>, variant: Variant = stage as Variant, ranges: Record<number, GPUBufferBinding> = {}) => {
      const offset = slot++ * PARAM_STRIDE;
      new Int32Array(params, offset, 4).set([length, 0, frustum]);
      const unsigned = new Uint32Array(params, offset, 4);
      unsigned[1] = count;
      unsigned[3] = (RISE_CLASSES - 1) * 2 * N * N; // riseClass
      const entries: GPUBindGroupEntry[] = [
        { binding: 1, resource: { buffer: this.#params, offset, size: 16 } },
        ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
        ...Object.entries(ranges).map(([binding, resource]) => ({ binding: Number(binding), resource })),
      ];
      const p = (pipelines as Partial<Record<Variant, GPUComputePipeline>>)[variant]!;
      this.#program.push({ label, stage, pipeline: p, bindGroup: device.createBindGroup({ layout: p.getBindGroupLayout(0), entries }), count });
    };
    const scene = this.sceneBuffer;
    // what the producer's fill reads
    const source: Record<number, GPUBuffer> = Object.fromEntries((producer?.buffers ?? []).map((b, i) => [20 + i, b]));
    const layout = padded ? outsideLayout(N) : [], runThreads = runRows(N, this.mergeRun).totals;
    const rowTable = (offset: number, size?: number) => ({ buffer: this.#rowTables!, offset, size });
    // a workgroup per row of block 0
    const fill = (frustum: number, length: number, target: GPUBuffer) =>
      add('merge', `outside ${length}`, frustum, length, (N + 2 * layout[Math.log2(length)].pad0) * WORKGROUP, { 5: target, ...source }, 'fill', { 19: rowTable(0, this.#runRowsAt) });
    const normals: Record<number, GPUBuffer> = this.#normalMap ? { 12: this.#normalMap, 13: this.#axisVector! } : {};
    const vectors: Record<number, GPUBuffer> = this.vectorBuffer ? { 13: this.#axisVector!, 14: this.vectorBuffer } : {};
    // length 1 comes from the scene: these bindings are never read
    const rays = (length: number) => (length === 1 ? scene : this.#rays.get(length)!);
    const odd = (length: number) => (length === 1 ? scene : this.#oddRays.get(length)!);
    // written over the rays of twice the length, which the merge has read by then
    const merged = (length: number) => (length === N ? (this.#topOutside ?? scene) : 2 * length === N ? this.#topMerged : rays(2 * length));
    for (let frustum = 0; frustum < 4; frustum++) {
      for (const length of this.#stored()) {
        const buffers = { 0: scene, 2: rays(length / 2), 9: odd(length / 2), 3: rays(length) };
        if (this.#oddRays.has(length)) add('extensions', `extensions ${length}`, frustum, length, storedRays(N, length), { ...buffers, 10: odd(length) }, 'extensionsSplit');
        else add('extensions', `extensions ${length}`, frustum, length, storedRays(N, length), buffers);
      }
      // with a producer, a length's outside cells are filled just before the merge that reads them, once the rays their
      // buffer held are read
      for (const length of this.#stored().reverse()) {
        if (padded) fill(frustum, 2 * length, merged(2 * length));
        const count = (N / length - 1) * (padded ? runThreads[Math.log2(length)] : Math.ceil(riseCount(length) / this.mergeRun) * N);
        add('merge', `merge ${2 * length}`, frustum, 2 * length, count, { 0: scene, 2: rays(length), 4: merged(2 * length), 5: merged(length), 8: this.#coneWeights }, 'merge', padded ? { 19: rowTable(this.#runRowsAt) } : {});
      }
      if (padded) (fill(frustum, 2, merged(2)), fill(frustum, 1, this.#outsideOne!));
      add('pixels', 'merge 2, 1 and pixels', frustum, 2, N * N, { 0: scene, 2: rays(2), 4: merged(2), 6: this.#axisFluence, 8: this.#coneWeights, ...(padded ? { 18: this.#outsideOne! } : ({} as Record<number, GPUBuffer>)), ...normals });
    }
    add('output', 'output', 0, 1, N * N, { 6: this.#axisFluence, 7: this.outputBuffer, ...vectors });
    device.queue.writeBuffer(this.#params, 0, params);
  }

  #buffers() {
    const optional = [this.vectorBuffer, this.#normalMap, this.#axisVector, this.#topOutside, this.#outsideOne, this.#rowTables].filter((b) => b !== null);
    return new Set([this.sceneBuffer, this.outputBuffer, ...this.#rays.values(), ...this.#oddRays.values(), this.#topMerged, this.#axisFluence, this.#coneWeights, ...optional]);
  }

  destroy() {
    for (const b of this.#buffers()) b.destroy();
    this.#params.destroy();
  }

  upload({ emission, solid, absorption, normal }: ReciprocalScene) {
    // per rise class, both axes column by column, so that every frustum reads along its rows: pixel (x, y) at
    // [class][0][x][y] and [class][1][y][x]. Written element by element: this runs on every edit.
    const { N } = this, data = (this.#uploadData ??= new Float32Array(RISE_CLASSES * 2 * N * N * 4));   // every element is rewritten
    for (let x = 0; x < N; x++)
      for (let y = 0; y < N; y++) {
        const i = x * N + y, red = emission[3 * i], green = emission[3 * i + 1], blue = emission[3 * i + 2];
        const sigma = absorption ? absorption[i] : 0;
        const opaque = solid[i] || ((red > 0 || green > 0 || blue > 0) && !(sigma > 0));
        for (let c = 0; c < RISE_CLASSES; c++) {
          const t = opaque ? 0 : sigma > 0 ? Math.exp((-MEDIUM_LENGTH[c] * sigma) / 2) : 1;
          const along = 4 * (c * 2 * N * N + i), across = 4 * (c * 2 * N * N + N * N + y * N + x);
          data[along] = data[across] = red * (1 - t);
          data[along + 1] = data[across + 1] = green * (1 - t);
          data[along + 2] = data[across + 2] = blue * (1 - t);
          data[along + 3] = data[across + 3] = t;
        }
      }
    this.device.queue.writeBuffer(this.sceneBuffer, 0, data);
    if (this.#normalMap) {
      const normals = new Float32Array(N * N * 4);
      for (let i = 0; i < N * N; i++) {
        normals[4 * i] = normal ? normal[3 * i] : 0;
        normals[4 * i + 1] = normal ? normal[3 * i + 1] : 0;
        normals[4 * i + 2] = normal ? normal[3 * i + 2] : 1;
      }
      this.device.queue.writeBuffer(this.#normalMap, 0, normals);
    }
  }

  /** the dispatches' labels, in order */
  get labels() {
    return this.#program.map((d) => d.label);
  }

  /** consecutive dispatches of one stage share a compute pass; every dispatch, or those whose label `only` accepts */
  encodeAll(encoder: GPUCommandEncoder, only?: (label: string) => boolean) {
    let pass: GPUComputePassEncoder | null = null, stage = '';
    for (const d of this.#program) {
      if (only && !only(d.label)) continue;
      if (d.stage !== stage) {
        pass?.end();
        stage = d.stage;
        pass = this.profiler ? this.profiler.computePass(encoder, stage) : encoder.beginComputePass();
      }
      this.profiler?.name(pass!, d.label);
      pass!.setPipeline(d.pipeline);
      pass!.setBindGroup(0, d.bindGroup);
      const groups = Math.ceil(d.count / WORKGROUP);
      pass!.dispatchWorkgroups(Math.min(groups, MAX_GROUPS_X), Math.ceil(groups / MAX_GROUPS_X));
    }
    pass?.end();
  }

  /** upload, solve once and read the fluence back: world [x * N + y], (rgb, luminance) */
  async solve(scene: ReciprocalScene): Promise<Float32Array> {
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

  /** bytes each dispatch label reads and writes in one frustum, each element counted once (a lower bound) */
  traffic(): Record<string, Traffic> {
    const { N } = this, pixels = N * N, pixelBytes = 16;
    const { ray: RAY_BYTES, merged: MERGED_BYTES } = SIZES[this.storage];
    const rays = (length: number) => storedRays(N, length);
    const out: Record<string, Traffic> = {};
    for (const length of this.#stored()) {
      const half = length / 2;
      out[`extensions ${length}`] = { read: half === 1 ? RISE_CLASSES * pixels * pixelBytes : rays(half) * RAY_BYTES, write: rays(length) * RAY_BYTES };
    }
    for (const length of this.#stored().reverse()) {
      const parentLength = 2 * length;
      out[`merge ${parentLength}`] = { read: (parentLength < N ? rays(parentLength) * MERGED_BYTES : 0) + rays(length) * RAY_BYTES, write: rays(length) * MERGED_BYTES };
    }
    out['merge 2, 1 and pixels'] = { read: rays(2) * MERGED_BYTES + (1 + RISE_CLASSES) * pixels * pixelBytes, write: pixels * pixelBytes };
    out.output = { read: 3 * pixels * pixelBytes, write: pixels * pixelBytes };
    return out;
  }

  get bytes() {
    return [...this.#buffers()].reduce((sum, b) => sum + b.size, 0);
  }
}
