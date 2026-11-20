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
}

export interface Options {
  /** the Wu split: the merge gives an odd parent's far light to its two bent orders by its slope across rows, not ½ / ½
   * (off by default) */
  wuSplit?: boolean;
  /** the coupling read: the light a first half hands to a bent parent's second half follows the first half's
   * half-limited slope across rows (off by default) */
  couplingRead?: boolean;
}

/** a half pixel's path-length factor in a 1-px ray of rise 0 and of rise ±1: the minimax line in |s| to √(1 + s²) */
export const MEDIUM_LENGTH = [0.95509, 1.3693] as const;

const WORKGROUP = 256;
const MERGE_THREADS = 65536; // a merge thread takes a run of rises, as long as the dispatch keeps about this many threads
const MAX_GROUPS_X = 32768;
const PARAM_STRIDE = 256; // uniform offsets must be multiples of 256
const RAY_BYTES = 8;
const MERGED_BYTES = 4;
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

const common = (N: number) => /* wgsl */ `
const N: i32 = ${N};
const CONE_OFFSETS = array<i32, ${Math.log2(N) + 1}>(${coneTable(N).offsets.join(', ')});
// riseClass: where the scene's half pixels of rise-±1 1-px rays start
struct Params { length: i32, count: u32, frustum: i32, riseClass: u32 };
@group(0) @binding(0) var<storage, read> scene: array<vec4f>;   // [rise class][axis][column][row]: a half pixel (r rgb, t)
@group(0) @binding(1) var<uniform> params: Params;

fn threadIndex(id: vec3u, groups: vec3u) -> u32 { return id.x + id.y * groups.x * ${WORKGROUP}u; }

// frustum: axis frustum / 2, mirrored along the axis if odd. Off the rows: solid and dark.
fn pixelAt(column: i32, row: i32, rise: i32) -> vec4f {
  if (row < 0 || row >= N) { return vec4f(0.0); }
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

const loadRayFn = /* wgsl */ `
fn unpackRay(packed: vec2u) -> Ray {
  let blueAndTransmittance = unpack2x16float(packed.y);
  return Ray(blueAndTransmittance.y, vec3f(unpack2x16float(packed.x), blueAndTransmittance.x));
}
fn loadRay(slot: i32) -> Ray { return unpackRay(raysIn[slot]); }`;
const storeRayFn = /* wgsl */ `
fn packRay(ray: Ray) -> vec2u {
  return vec2u(pack2x16float(ray.radiance.rg), pack2x16float(vec2f(ray.radiance.b, ray.transmittance)));
}
fn storeRay(slot: u32, ray: Ray) { raysOut[slot] = packRay(ray); }`;
const loadMergedFn = /* wgsl */ `
fn loadMerged(slot: i32) -> vec3f {
  let packed = mergedIn[slot];
  return vec3f(vec3u(packed, packed >> 9u, packed >> 18u) & vec3u(511u)) * bitcast<f32>(((packed >> 27u) + 103u) << 23u);
}`;
const storeMergedFn = /* wgsl */ `
fn storeMerged(slot: u32, merged: vec3f) {
  let exponent = clamp(i32((bitcast<u32>(max(merged.r, max(merged.g, merged.b))) >> 23u) & 255u) - 111, 0, 31);
  let mantissa = vec3u(min(max(merged, vec3f(0.0)) * bitcast<f32>(u32(151 - exponent) << 23u) + 0.5, vec3f(511.0)));
  mergedOut[slot] = mantissa.r | (mantissa.g << 9u) | (mantissa.b << 18u) | (u32(exponent) << 27u);
}`;

// A 1-px ray is built where it is read, from its two half pixels (the leaves, read from the scene). A ray whose end is
// off the rows is not stored: it carries nothing and blocks. Reads load first and choose after, with no branch around
// the load (a branch keeps a thread from issuing its loads together). A slot off the rows may lie outside the buffer;
// WebGPU then reads some value inside it, which is discarded.
const rayFn = (oddBlocks: boolean) => /* wgsl */ `
fn ray(length: i32, block: i32, rise: i32, startRow: i32) -> Ray {
  if (length == 1) { return extend(halfPixel(block, startRow, rise), halfPixel(block, startRow + rise, rise)); }
  let endRow = startRow + rise; let stored = endRow >= 0 && endRow < N;
  let slot = ((block >> 1) * riseCount(length) + riseIndex(length, rise)) * N + endRow;${
    oddBlocks
      ? `
  var packed: vec2u;
  if ((block & 1) == 1) { packed = raysInOdd[slot]; } else { packed = raysIn[slot]; }
  let loaded = unpackRay(packed);`
      : `
  let loaded = loadRay(slot);`
  }
  return Ray(select(0.0, loaded.transmittance, stored), select(vec3f(0.0), loaded.radiance, stored));
}
`;

// splitOut: write even blocks to raysOut and odd blocks to raysOutOdd
const extensionsShader = (N: number, splitOut: boolean) => /* wgsl */ `
${common(N)}
@group(0) @binding(2) var<storage, read> raysIn: array<vec2u>;          // length / 2: even blocks
@group(0) @binding(9) var<storage, read> raysInOdd: array<vec2u>;       // and odd blocks
@group(0) @binding(3) var<storage, read_write> raysOut: array<vec2u>;${
  splitOut
    ? `
@group(0) @binding(10) var<storage, read_write> raysOutOdd: array<vec2u>;`
    : ''
}
${loadRayFn}
${storeRayFn}
${rayFn(true)}
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

const mergeFns = (wu: boolean) => /* wgsl */ `
@group(0) @binding(8) var<storage, read> coneWeights: array<f32>;
fn coneWeight(length: i32, rise: i32) -> f32 {
  return coneWeights[CONE_OFFSETS[firstTrailingBit(u32(length))] + riseIndex(length, rise)];
}
// none at length N, in column block 0 or off the rows (loaded, then discarded)
fn mergedAt(length: i32, block: i32, rise: i32, startRow: i32) -> vec3f {
  let merged = loadMerged(((block - 1) * riseCount(length) + riseIndex(length, rise)) * N + startRow);
  return select(vec3f(0.0), merged, hasRise(length, rise) && length != N && block >= 1 && startRow >= 0 && startRow < N);
}
// a parent's merged radiance carried through the part of it before the ray
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
const mergeShader = (N: number, run: number, wu: boolean, read: boolean) => /* wgsl */ `
${common(N)}
@group(0) @binding(2) var<storage, read> raysIn: array<vec2u>;         // length
@group(0) @binding(4) var<storage, read> mergedIn: array<u32>;         // parent length
@group(0) @binding(5) var<storage, read_write> mergedOut: array<u32>;  // length
${loadRayFn}
${loadMergedFn}
${storeMergedFn}
${rayFn(false)}
${mergeFns(wu)}${
  read
    ? `
// The coupling read, strength 0.1: a first half's light enters a bent parent's second half weighted 1 ± 0.4 s. Over the
// first half's start rows, with up = R(y0 + 1) - R(y0) and down = R(y0) - R(y0 - 1), s is the mean of the slope
// (up + down) / sum and its limited form 2 minmod(up, down) / sum, sum = R(y0 - 1) + R(y0) + R(y0 + 1) (minmod: the
// smaller of two differences of one sign, else 0), per channel. + where the first half has the lower rise.
fn rowSlope(length: i32, block: i32, rise: i32, endRow: i32, own: vec3f) -> vec3f {
  let below = leftHalf(length, block, rise, endRow - 1).radiance; let above = leftHalf(length, block, rise, endRow + 1).radiance;
  let up = above - own; let down = own - below; let sum = below + own + above;
  let limited = select(vec3f(0.0), sign(up) * min(abs(up), abs(down)), up * down > vec3f(0.0));
  return select(vec3f(0.0), (limited + 0.5 * (up + down)) / sum, sum > vec3f(0.0));
}
fn steered(before: Ray, slope: vec3f, sign: f32) -> Ray {
  return Ray(before.transmittance, before.radiance * (1.0 + sign * 0.4 * slope));
}`
    : ''
}
const RUN: i32 = ${run};
@compute @workgroup_size(${WORKGROUP})
fn mergeLevel(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let parentLength = params.length; let length = parentLength / 2;
  let count = riseCount(length); let runs = (count + RUN - 1) / RUN;
  let row = i32(i % u32(N)); let blockAndRun = i32(i / u32(N));
  let first = (blockAndRun % runs) * RUN; let last = min(first + RUN, count);
  let block = blockAndRun / runs + 1; let parentBlock = block / 2;
  if ((block & 1) == 0) {
    var lower = ${wu ? 'rowsAt' : 'mergedAt'}(parentLength, parentBlock, 2 * riseAt(length, first) - 1, row);
    for (var q = first; q < last; q++) {
      let rise = riseAt(length, q);
      let upper = ${wu ? 'rowsAt' : 'mergedAt'}(parentLength, parentBlock, 2 * rise + 1, row);
      let merged = gather(parentLength, rise, ${wu ? 'wuShare(lower, 1.0)' : 'lower'}, NOTHING, mergedAt(parentLength, parentBlock, 2 * rise, row), NOTHING, ${wu ? 'wuShare(upper, -1.0)' : 'upper'}, NOTHING);
      storeMerged(u32(((block - 1) * count + q) * N + row), merged);
      lower = upper;
    }
  } else {
    var lowerBefore = leftHalf(length, block - 1, riseAt(length, first) - 1, row);
    var sameBefore = leftHalf(length, block - 1, riseAt(length, first), row);${
      read
        ? `
    var lowerSlope = rowSlope(length, block - 1, riseAt(length, first) - 1, row, lowerBefore.radiance);
    var sameSlope = rowSlope(length, block - 1, riseAt(length, first), row, sameBefore.radiance);`
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
      let upperSlope = rowSlope(length, block - 1, rise + 1, row, upperBefore.radiance);`
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
      storeMerged(u32(((block - 1) * count + q) * N + row), merged);
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
const pixelsShader = (N: number, wu: boolean) => /* wgsl */ `
${common(N)}
// (1 - m, (1 - m) / (1 - t)), m = (1 - t) / (-ln t) the mean transmittance over the half pixel
fn slab(t: f32) -> vec2f {
  let a = select(-log(max(t, 1e-30)), 0.0, t >= 1.0);   // log(1) need not be 0 on the GPU
  if (a < 1e-3) { return vec2f(a / 2.0 - a * a / 6.0, 0.5 + a / 12.0); }
  let lost = 1.0 - (1.0 - t) / a;
  return vec2f(lost, lost / (1.0 - t));
}
@group(0) @binding(2) var<storage, read> raysIn: array<vec2u>;          // bound, never read: 1-px rays come from the scene
@group(0) @binding(4) var<storage, read> mergedIn: array<u32>;          // length 2
@group(0) @binding(6) var<storage, read_write> axisFluence: array<vec4f>;  // [axis][world column][row]
${loadRayFn}
${loadMergedFn}
${rayFn(false)}
${mergeFns(wu)}
@compute @workgroup_size(${WORKGROUP})
fn pixels(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let column = i32(i) / N; let row = i32(i) % N;
  var value = vec3f(0.0);
  for (var rise = -1; rise <= 1; rise++) {
    // The merge at length 1/2, with the 1-px cone weights (exact: π/2 - ln 2 within a row, ½ ln 2 between rows). A left
    // half takes its 1-px ray's merged radiance through nothing before it; a right half through the left half pixel.
    // (At rise 0 both halves are the same 1-px ray's.)
    let cone = coneWeight(1, rise);
    let fromRow = merge(1, column, rise, row);
    var toRow = fromRow;
    if (rise != 0) { toRow = merge(1, column, rise, row - rise); }
    let left = carry(fromRow, NOTHING, cone);
    let right = carry(toRow, halfPixel(column, row - rise, rise), cone);
    // Both halves deposit: their light over a half pixel's length (½), less what the medium absorbs over it, plus its
    // own light along the ray.
    let own = pixelAt(column, row, rise); let s = slab(own.w);
    value += 0.5 * ((1.0 - s.x) * (left + right) + 2.0 * cone * s.y * own.rgb);
  }
  value *= select(0.0, 1.0, pixelAt(column, row, 0).w > 0.0);   // a solid pixel holds no fluence
  let mirrored = (params.frustum & 1) == 1;
  let slot = ((params.frustum >> 1) * N + select(column, N - 1 - column, mirrored)) * N + row;
  if (mirrored) { value += axisFluence[slot].rgb; }   // the mirror-image frustum adds to the forward one
  axisFluence[slot] = vec4f(value, 0.0);
}
`;

// the fluence: the two axes' sum
const outputShader = (N: number) => /* wgsl */ `
${common(N)}
@group(0) @binding(6) var<storage, read> axisFluence: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> result: array<vec4f>;  // world [x * N + y]: (fluence rgb, its luminance)
@compute @workgroup_size(${WORKGROUP})
fn output(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let x = i32(i) / N; let y = i32(i) % N;
  let value = axisFluence[x * N + y].rgb + axisFluence[(N + y) * N + x].rgb;   // [axis][column][row]
  result[i] = vec4f(value, dot(value, vec3f(0.2126, 0.7152, 0.0722)));
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
  readonly couplingRead: boolean;
  /** consecutive rises per merge thread */
  readonly mergeRun: number;
  profiler: Profiler | null = null;
  /** the scene's half pixels, [rise class][axis][column][row]: (r rgb, t) */
  readonly sceneBuffer: GPUBuffer;
  /** fluence, world [x * N + y]: (rgb, luminance) */
  readonly outputBuffer: GPUBuffer;
  #rays = new Map<number, GPUBuffer>(); // even blocks by length; later also the merged radiance of length / 2
  #oddRays = new Map<number, GPUBuffer>();
  #topMerged: GPUBuffer; // merged radiance of length N / 2
  #axisFluence: GPUBuffer;
  #coneWeights: GPUBuffer;
  #params: GPUBuffer;
  #program: Dispatch[] = [];

  constructor(device: GPUDevice, N: number, { wuSplit = false, couplingRead = false }: Options = {}) {
    if (N < 4 || (N & (N - 1)) !== 0) throw new Error('ReciprocalHRC: N must be a power of 2, at least 4');
    this.device = device;
    this.N = N;
    this.wuSplit = wuSplit;
    this.couplingRead = couplingRead;
    // a merge dispatch has about 2 N² rises
    this.mergeRun = Math.min(16, 2 ** Math.max(0, Math.floor(Math.log2((2 * N * N) / MERGE_THREADS))));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, size: number, use = usage) => device.createBuffer({ label, size: Math.max(16, size), usage: use });
    this.sceneBuffer = buffer('reciprocal HRC scene', RISE_CLASSES * 2 * N * N * 16);
    this.outputBuffer = buffer('reciprocal HRC fluence', N * N * 16);
    // each buffer holds its even blocks, then the merged radiance of length / 2, and the odd blocks of length / 4
    const oddBytes = (length: number) => (length >= 8 ? oddRays(N, length / 4) * RAY_BYTES : 0);
    for (const length of this.#stored()) {
      const merged = length > 2 ? storedRays(N, length / 2) * MERGED_BYTES : 0;
      this.#rays.set(length, buffer(`reciprocal HRC rays ${length}`, Math.max(evenRays(N, length) * RAY_BYTES, merged, oddBytes(length))));
    }
    this.#topMerged = buffer(`reciprocal HRC merged ${N / 2}`, Math.max(storedRays(N, N / 2) * MERGED_BYTES, oddBytes(N)));
    for (const length of this.#stored())
      if (oddRays(N, length) > 0) this.#oddRays.set(length, 4 * length === N ? this.#topMerged : this.#rays.get(4 * length)!);
    this.#axisFluence = buffer('reciprocal HRC fluence per axis', 2 * N * N * 16);
    const cones = coneTable(N).weights;
    this.#coneWeights = buffer('reciprocal HRC cone weights', cones.byteLength);
    device.queue.writeBuffer(this.#coneWeights, 0, cones);
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
    return 4 * (2 * this.#stored().length + 1) + 1;
  }

  #build() {
    const { device, N } = this;
    const pipeline = (code: string, entryPoint: string) =>
      device.createComputePipeline({
        label: `reciprocal HRC ${entryPoint}`,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: `reciprocal HRC ${entryPoint}`, code }), entryPoint },
      });
    const pipelines = {
      extensions: pipeline(extensionsShader(N, false), 'extensions'),
      extensionsSplit: pipeline(extensionsShader(N, true), 'extensions'),
      merge: pipeline(mergeShader(N, this.mergeRun, this.wuSplit, this.couplingRead), 'mergeLevel'),
      pixels: pipeline(pixelsShader(N, this.wuSplit), 'pixels'),
      output: pipeline(outputShader(N), 'output'),
    };
    const params = new ArrayBuffer(this.dispatches * PARAM_STRIDE);
    let slot = 0;
    const add = (stage: 'extensions' | 'merge' | 'pixels' | 'output', label: string, frustum: number, length: number, count: number, buffers: Record<number, GPUBuffer>, variant: keyof typeof pipelines = stage) => {
      const offset = slot++ * PARAM_STRIDE;
      new Int32Array(params, offset, 4).set([length, 0, frustum]);
      const unsigned = new Uint32Array(params, offset, 4);
      unsigned[1] = count;
      unsigned[3] = (RISE_CLASSES - 1) * 2 * N * N; // riseClass
      const entries: GPUBindGroupEntry[] = [
        { binding: 1, resource: { buffer: this.#params, offset, size: 16 } },
        ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
      ];
      const p = pipelines[variant];
      this.#program.push({ label, stage, pipeline: p, bindGroup: device.createBindGroup({ layout: p.getBindGroupLayout(0), entries }), count });
    };
    const scene = this.sceneBuffer;
    // length 1 comes from the scene: these bindings are never read
    const rays = (length: number) => (length === 1 ? scene : this.#rays.get(length)!);
    const odd = (length: number) => (length === 1 ? scene : this.#oddRays.get(length)!);
    // written over the rays of twice the length, which the merge has read by then
    const merged = (length: number) => (length === N ? scene : 2 * length === N ? this.#topMerged : rays(2 * length));
    for (let frustum = 0; frustum < 4; frustum++) {
      for (const length of this.#stored()) {
        const buffers = { 0: scene, 2: rays(length / 2), 9: odd(length / 2), 3: rays(length) };
        if (this.#oddRays.has(length)) add('extensions', `extensions ${length}`, frustum, length, storedRays(N, length), { ...buffers, 10: odd(length) }, 'extensionsSplit');
        else add('extensions', `extensions ${length}`, frustum, length, storedRays(N, length), buffers);
      }
      for (const length of this.#stored().reverse()) {
        const count = (N / length - 1) * Math.ceil(riseCount(length) / this.mergeRun) * N;
        add('merge', `merge ${2 * length}`, frustum, 2 * length, count, { 0: scene, 2: rays(length), 4: merged(2 * length), 5: merged(length), 8: this.#coneWeights });
      }
      add('pixels', 'merge 2, 1 and pixels', frustum, 2, N * N, { 0: scene, 2: rays(2), 4: merged(2), 6: this.#axisFluence, 8: this.#coneWeights });
    }
    add('output', 'output', 0, 1, N * N, { 6: this.#axisFluence, 7: this.outputBuffer });
    device.queue.writeBuffer(this.#params, 0, params);
  }

  #buffers() {
    return new Set([this.sceneBuffer, this.outputBuffer, ...this.#rays.values(), ...this.#oddRays.values(), this.#topMerged, this.#axisFluence, this.#coneWeights]);
  }

  destroy() {
    for (const b of this.#buffers()) b.destroy();
    this.#params.destroy();
  }

  upload({ emission, solid, absorption }: ReciprocalScene) {
    // per rise class, both axes column by column, so that every frustum reads along its rows
    const { N } = this, data = new Float32Array(RISE_CLASSES * 2 * N * N * 4);
    for (let i = 0; i < N * N; i++) {
      const red = emission[3 * i], green = emission[3 * i + 1], blue = emission[3 * i + 2];
      const sigma = absorption ? absorption[i] : 0;
      const opaque = solid[i] || ((red > 0 || green > 0 || blue > 0) && !(sigma > 0));
      for (let c = 0; c < RISE_CLASSES; c++) {
        const t = opaque ? 0 : Math.exp((-MEDIUM_LENGTH[c] * sigma) / 2);
        const pixel = [red * (1 - t), green * (1 - t), blue * (1 - t), t];
        data.set(pixel, 4 * (c * 2 * N * N + i));
        data.set(pixel, 4 * (c * 2 * N * N + N * N + (i % N) * N + Math.floor(i / N)));
      }
    }
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
