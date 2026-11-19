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
  /** 2 (the default): 2L + 1 rays per probe; 1: HRC's L + 1 */
  baseRayCount?: 1 | 2;
  /** blur each axis's fluence [1 2 1] across its rows over free pixels before the sum (on by default) */
  rowBlur?: boolean;
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

const stepAt = (length: number, riseStep: number) => (length === 1 ? 1 : riseStep);
const riseCount = (length: number, riseStep: number) => (2 * length) / stepAt(length, riseStep) + 1;
const riseAt = (length: number, index: number, riseStep: number) => index * stepAt(length, riseStep) - length;
/** rays of one length kept in one frustum: N/length - 1 column blocks, every row */
const storedRays = (N: number, length: number, riseStep: number) => (N / length - 1) * riseCount(length, riseStep) * N;
const evenRays = (N: number, length: number, riseStep: number) => (N / length / 2) * riseCount(length, riseStep) * N;
const oddRays = (N: number, length: number, riseStep: number) => (N / length / 2 - 1) * riseCount(length, riseStep) * N;

/** the angle of the slopes (rise ± step/2) / length, clipped to [-1, 1]. 1-px rays: ½ ln 2 between row neighbours, and
 * π/2 - ln 2 from a pixel's left half pixel to its right one */
export function coneWeight(length: number, rise: number, step: number) {
  if (length === 1) return rise === 0 ? Math.PI / 2 - Math.log(2) : Math.log(2) / 2;
  return Math.atan(Math.min(1, (rise + step / 2) / length)) - Math.atan(Math.max(-1, (rise - step / 2) / length));
}

/** every level's cone weights in one table, level 0 first */
function coneTable(N: number, riseStep: number) {
  const offsets: number[] = [], weights: number[] = [];
  for (let length = 1; length <= N; length *= 2) {
    offsets.push(weights.length);
    for (let index = 0; index < riseCount(length, riseStep); index++) weights.push(coneWeight(length, riseAt(length, index, riseStep), stepAt(length, riseStep)));
  }
  return { offsets, weights: new Float32Array(weights) };
}

const common = (N: number, riseStep: number) => /* wgsl */ `
const N: i32 = ${N};
const CONE_OFFSETS = array<i32, ${Math.log2(N) + 1}>(${coneTable(N, riseStep).offsets.join(', ')});
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

// Rises run -length ... length in steps of 1 for the 1-px rays and ${riseStep} above, kept as a shift: a GPU divides by
// a variable slowly.
fn stepLog(length: i32) -> u32 { return select(${Math.log2(riseStep)}u, 0u, length == 1); }
fn stepAt(length: i32) -> i32 { return 1 << stepLog(length); }
fn riseCount(length: i32) -> i32 { return ((2 * length) >> stepLog(length)) + 1; }
fn riseIndex(length: i32, rise: i32) -> i32 { return (rise + length) >> stepLog(length); }
fn riseAt(length: i32, index: i32) -> i32 { return (index << stepLog(length)) - length; }
fn hasRise(length: i32, rise: i32) -> bool { return abs(rise) <= length && ((rise + length) & (stepAt(length) - 1)) == 0; }

// A ray is its two halves, or, when they can't have equal rises, the mean of both orders of the two nearest rises.
struct Pair { leftRise: i32, rightRise: i32, weight: f32 };
fn pairOf(length: i32, rise: i32, which: i32) -> Pair {
  let half = length / 2;
  if (rise % 2 == 0 && hasRise(half, rise / 2)) {
    if (which == 0) { return Pair(rise / 2, rise / 2, 1.0); }
    return Pair(0, 0, 0.0);
  }
  let spacing = stepAt(half);
  let lower = (rise - spacing) / 2; let upper = (rise + spacing) / 2;
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

// A ray whose end is off the rows is not stored: it carries nothing and blocks.
const rayFn = (oddBlocks: boolean) => /* wgsl */ `
fn ray(length: i32, block: i32, rise: i32, startRow: i32) -> Ray {
  if (length == 1) { return extend(halfPixel(block, startRow, rise), halfPixel(block, startRow + rise, rise)); }
  let endRow = startRow + rise;
  if (endRow < 0 || endRow >= N) { return Ray(0.0, vec3f(0.0)); }
  let slot = ((block >> 1) * riseCount(length) + riseIndex(length, rise)) * N + endRow;${
    oddBlocks
      ? `
  if ((block & 1) == 1) { return unpackRay(raysInOdd[slot]); }`
      : ''
  }
  return loadRay(slot);
}
`;

// splitOut: write even blocks to raysOut and odd blocks to raysOutOdd
const extensionsShader = (N: number, riseStep: number, splitOut: boolean) => /* wgsl */ `
${common(N, riseStep)}
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

const mergeFns = /* wgsl */ `
@group(0) @binding(8) var<storage, read> coneWeights: array<f32>;
fn coneWeight(length: i32, rise: i32) -> f32 {
  return coneWeights[CONE_OFFSETS[firstTrailingBit(u32(length))] + riseIndex(length, rise)];
}
// none at length N, in column block 0 or off the rows
fn mergedAt(length: i32, block: i32, rise: i32, startRow: i32) -> vec3f {
  if (!hasRise(length, rise) || length == N || block < 1 || startRow < 0 || startRow >= N) { return vec3f(0.0); }
  return loadMerged(((block - 1) * riseCount(length) + riseIndex(length, rise)) * N + startRow);
}
// a parent's merged radiance carried through the part of it before the ray
fn carry(merged: vec3f, before: Ray, cone: f32) -> vec3f {
  return before.transmittance * merged + cone * before.radiance;
}
// The merged radiance of a ray of rise 'rise' from its three parents, weighted as pairOf builds them: the bent parent
// below (½), the parent of twice its rise (1) and the bent parent above (½). Before a first half there is nothing;
// before a second half, the parent's first half.
fn gather(parentLength: i32, rise: i32, spacing: i32, lower: vec3f, lowerBefore: Ray, same: vec3f, sameBefore: Ray, upper: vec3f, upperBefore: Ray) -> vec3f {
  var merged = vec3f(0.0);
  if (hasRise(parentLength, 2 * rise - spacing)) { merged += 0.5 * carry(lower, lowerBefore, coneWeight(parentLength, 2 * rise - spacing)); }
  merged += carry(same, sameBefore, coneWeight(parentLength, 2 * rise));
  if (hasRise(parentLength, 2 * rise + spacing)) { merged += 0.5 * carry(upper, upperBefore, coneWeight(parentLength, 2 * rise + spacing)); }
  return merged;
}
fn leftHalf(length: i32, block: i32, rise: i32, endRow: i32) -> Ray {
  if (!hasRise(length, rise)) { return NOTHING; }
  return ray(length, block, rise, endRow - rise);
}
// gather's sum, loading the parents itself; an odd block is a second half. Kept as a loop: written out like gather,
// the pixels pass rounds differently.
fn merge(length: i32, block: i32, rise: i32, row: i32) -> vec3f {
  let parentLength = 2 * length; let spacing = stepAt(length); let isRight = (block & 1) == 1;
  var merged = vec3f(0.0);
  for (var side = -1; side <= 1; side++) {
    let parentRise = 2 * rise + side * spacing;
    if (!hasRise(parentLength, parentRise)) { continue; }
    let beforeRise = select(0, parentRise - rise, isRight);
    var before = NOTHING;
    if (isRight) { before = ray(length, block - 1, beforeRise, row - beforeRise); }
    merged += select(1.0, 0.5, side != 0) * carry(mergedAt(parentLength, block / 2, parentRise, row - beforeRise), before, coneWeight(parentLength, parentRise));
  }
  return merged;
}
`;

// One thread per (block, row) and a run of RUN consecutive rises, so that the parents and first halves one rise shares
// with the next stay in registers.
const mergeShader = (N: number, riseStep: number, run: number) => /* wgsl */ `
${common(N, riseStep)}
@group(0) @binding(2) var<storage, read> raysIn: array<vec2u>;         // length
@group(0) @binding(4) var<storage, read> mergedIn: array<u32>;         // parent length
@group(0) @binding(5) var<storage, read_write> mergedOut: array<u32>;  // length
${loadRayFn}
${loadMergedFn}
${storeMergedFn}
${rayFn(false)}
${mergeFns}
const RUN: i32 = ${run};
@compute @workgroup_size(${WORKGROUP})
fn mergeLevel(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let parentLength = params.length; let length = parentLength / 2; let spacing = stepAt(length);
  let count = riseCount(length); let runs = (count + RUN - 1) / RUN;
  let row = i32(i % u32(N)); let blockAndRun = i32(i / u32(N));
  let first = (blockAndRun % runs) * RUN; let last = min(first + RUN, count);
  let block = blockAndRun / runs + 1; let parentBlock = block / 2;
  if ((block & 1) == 0) {
    var lower = mergedAt(parentLength, parentBlock, 2 * riseAt(length, first) - spacing, row);
    for (var q = first; q < last; q++) {
      let rise = riseAt(length, q);
      let upper = mergedAt(parentLength, parentBlock, 2 * rise + spacing, row);
      let merged = gather(parentLength, rise, spacing, lower, NOTHING, mergedAt(parentLength, parentBlock, 2 * rise, row), NOTHING, upper, NOTHING);
      storeMerged(u32(((block - 1) * count + q) * N + row), merged);
      lower = upper;
    }
  } else {
    var lowerBefore = leftHalf(length, block - 1, riseAt(length, first) - spacing, row);
    var sameBefore = leftHalf(length, block - 1, riseAt(length, first), row);
    for (var q = first; q < last; q++) {
      let rise = riseAt(length, q);
      let upperBefore = leftHalf(length, block - 1, rise + spacing, row);
      let merged = gather(parentLength, rise, spacing,
        mergedAt(parentLength, parentBlock, 2 * rise - spacing, row - (rise - spacing)), lowerBefore,
        mergedAt(parentLength, parentBlock, 2 * rise, row - rise), sameBefore,
        mergedAt(parentLength, parentBlock, 2 * rise + spacing, row - (rise + spacing)), upperBefore);
      storeMerged(u32(((block - 1) * count + q) * N + row), merged);
      lowerBefore = sameBefore; sameBefore = upperBefore;
    }
  }
}
`;

// Level 1 into the 1-px rays, then the 1-px rays into the pixels. A pixel is the left half pixel of its own three 1-px
// rays and the right half pixel of three more: its own rise 0, rise 1 from the row below, rise -1 from the row above.
const pixelsShader = (N: number, riseStep: number) => /* wgsl */ `
${common(N, riseStep)}
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
${mergeFns}
@compute @workgroup_size(${WORKGROUP})
fn pixels(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let column = i32(i) / N; let row = i32(i) % N;
  let level = merge(1, column, 0, row);
  let below = merge(1, column, -1, row); let above = merge(1, column, 1, row);
  let fromOwn = carry(level, halfPixel(column, row, 0), coneWeight(1, 0));
  let fromBelow = carry(merge(1, column, 1, row - 1), halfPixel(column, row - 1, 1), coneWeight(1, 1));
  let fromAbove = carry(merge(1, column, -1, row + 1), halfPixel(column, row + 1, -1), coneWeight(1, -1));
  var merged = below + level + above;
  merged += fromOwn;
  merged += fromBelow;
  merged += fromAbove;
  // each half pixel's mean over its length, with the t of its own 1-px ray, plus its own light
  let own0 = pixelAt(column, row, 0); let own1 = pixelAt(column, row, 1);
  let s0 = slab(own0.w); let s1 = slab(own1.w);
  let lost = s0.x * (level + fromOwn) + s1.x * (below + above + fromBelow + fromAbove);
  let ownLight = ${(Math.PI / 2 - Math.log(2)).toPrecision(9)} * s0.y * own0.rgb + ${Math.log(2).toPrecision(9)} * s1.y * own1.rgb;
  var value = select(0.0, 1.0, own0.w > 0.0) * (0.5 * merged - 0.5 * lost + ownLight);
  let mirrored = (params.frustum & 1) == 1;
  let slot = ((params.frustum >> 1) * N + select(column, N - 1 - column, mirrored)) * N + row;
  if (mirrored) { value += axisFluence[slot].rgb; }   // the mirror-image frustum adds to the forward one
  axisFluence[slot] = vec4f(value, 0.0);
}
`;

const outputShader = (N: number, riseStep: number, rowBlur: boolean) => /* wgsl */ `
${common(N, riseStep)}
@group(0) @binding(6) var<storage, read> axisFluence: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> result: array<vec4f>;  // world [x * N + y]: (fluence rgb, its luminance)
fn free(axis: i32, column: i32, row: i32) -> f32 {
  if (row < 0 || row >= N) { return 0.0; }
  return select(0.0, 1.0, scene[(axis * N + column) * N + row].w > 0.0);
}
fn axisValue(axis: i32, column: i32, row: i32) -> vec3f {
  let base = (axis * N + column) * N;
  let value = axisFluence[base + row].rgb;${
    rowBlur
      ? `
  let below = free(axis, column, row - 1) * (axisFluence[base + max(row - 1, 0)].rgb - value);
  let above = free(axis, column, row + 1) * (axisFluence[base + min(row + 1, N - 1)].rgb - value);
  return value + 0.25 * free(axis, column, row) * (below + above);`
      : `
  return value;`
  }
}
@compute @workgroup_size(${WORKGROUP})
fn output(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= params.count) { return; }
  let x = i32(i) / N; let y = i32(i) % N;
  let value = axisValue(0, x, y) + axisValue(1, y, x);
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
  readonly baseRayCount: 1 | 2;
  /** the spacing of rises: 1 at a 2× base ray count, 2 at 1× */
  readonly riseStep: number;
  readonly rowBlur: boolean;
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

  constructor(device: GPUDevice, N: number, { baseRayCount = 2, rowBlur = true }: Options = {}) {
    if (N < 4 || (N & (N - 1)) !== 0) throw new Error('ReciprocalHRC: N must be a power of 2, at least 4');
    this.device = device;
    this.N = N;
    this.baseRayCount = baseRayCount;
    const riseStep = (this.riseStep = baseRayCount === 2 ? 1 : 2);
    this.rowBlur = rowBlur;
    // a merge dispatch has about N² (2 / riseStep) rises
    this.mergeRun = Math.min(16, 2 ** Math.max(0, Math.floor(Math.log2((N * N * (2 / riseStep)) / MERGE_THREADS))));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, size: number, use = usage) => device.createBuffer({ label, size: Math.max(16, size), usage: use });
    this.sceneBuffer = buffer('reciprocal HRC scene', RISE_CLASSES * 2 * N * N * 16);
    this.outputBuffer = buffer('reciprocal HRC fluence', N * N * 16);
    // each buffer holds its even blocks, then the merged radiance of length / 2, and the odd blocks of length / 4
    const oddBytes = (length: number) => (length >= 8 ? oddRays(N, length / 4, riseStep) * RAY_BYTES : 0);
    for (const length of this.#stored()) {
      const merged = length > 2 ? storedRays(N, length / 2, riseStep) * MERGED_BYTES : 0;
      this.#rays.set(length, buffer(`reciprocal HRC rays ${length}`, Math.max(evenRays(N, length, riseStep) * RAY_BYTES, merged, oddBytes(length))));
    }
    this.#topMerged = buffer(`reciprocal HRC merged ${N / 2}`, Math.max(storedRays(N, N / 2, riseStep) * MERGED_BYTES, oddBytes(N)));
    for (const length of this.#stored())
      if (oddRays(N, length, riseStep) > 0) this.#oddRays.set(length, 4 * length === N ? this.#topMerged : this.#rays.get(4 * length)!);
    this.#axisFluence = buffer('reciprocal HRC fluence per axis', 2 * N * N * 16);
    const cones = coneTable(N, riseStep).weights;
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
    const { device, N, riseStep } = this;
    const pipeline = (code: string, entryPoint: string) =>
      device.createComputePipeline({
        label: `reciprocal HRC ${entryPoint}`,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: `reciprocal HRC ${entryPoint}`, code }), entryPoint },
      });
    const pipelines = {
      extensions: pipeline(extensionsShader(N, riseStep, false), 'extensions'),
      extensionsSplit: pipeline(extensionsShader(N, riseStep, true), 'extensions'),
      merge: pipeline(mergeShader(N, riseStep, this.mergeRun), 'mergeLevel'),
      pixels: pipeline(pixelsShader(N, riseStep), 'pixels'),
      output: pipeline(outputShader(N, riseStep, this.rowBlur), 'output'),
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
        if (this.#oddRays.has(length)) add('extensions', `extensions ${length}`, frustum, length, storedRays(N, length, riseStep), { ...buffers, 10: odd(length) }, 'extensionsSplit');
        else add('extensions', `extensions ${length}`, frustum, length, storedRays(N, length, riseStep), buffers);
      }
      for (const length of this.#stored().reverse()) {
        const count = (N / length - 1) * Math.ceil(riseCount(length, riseStep) / this.mergeRun) * N;
        add('merge', `merge ${2 * length}`, frustum, 2 * length, count, { 0: scene, 2: rays(length), 4: merged(2 * length), 5: merged(length), 8: this.#coneWeights });
      }
      add('pixels', 'merge 2, 1 and pixels', frustum, 2, N * N, { 0: scene, 2: rays(2), 4: merged(2), 6: this.#axisFluence, 8: this.#coneWeights });
    }
    const outputBuffers: Record<number, GPUBuffer> = { 6: this.#axisFluence, 7: this.outputBuffer };
    if (this.rowBlur) outputBuffers[0] = scene; // the blur reads which pixels are free
    add('output', 'output', 0, 1, N * N, outputBuffers);
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
    const { N, riseStep } = this, pixels = N * N, pixelBytes = 16;
    const rays = (length: number) => storedRays(N, length, riseStep);
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
