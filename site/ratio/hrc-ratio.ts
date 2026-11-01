import { reactive, type Changes } from '../_shared/reactive';
import { makeShaderDataDefinitions, makeStructuredView, type StructuredView } from 'webgpu-utils';
import type { Shape } from '../_shared/shape-editor';

type Line = [
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  r: number,
  g: number,
  b: number,
  thickness: number,
  opacity: number,
  albedo: number,
];

function nextPow2(n: number): number {
  let v = 1;
  while (v < n) v *= 2;
  return v;
}

function ceilLog2(n: number): number {
  let levels = 0;
  let v = 1;
  while (v < n) {
    v *= 2;
    levels++;
  }
  return levels;
}

function ceilDiv(n: number, d: number): number {
  return Math.ceil(n / d);
}

// Closed-form fluence kernel of a unit-square volumetric emitter (NOTES §47.1), float64 for the near-field taps.
function kernelClosed(x: number, y: number): number {
  const G = (u: number, v: number) => {
    const r = Math.hypot(u, v);
    return (Math.abs(u) > 1e-12 ? u * Math.log(Math.max(v + r, 1e-300)) : 0) + (Math.abs(v) > 1e-12 ? v * Math.log(Math.max(u + r, 1e-300)) : 0);
  };
  return G(x + 0.5, y + 0.5) - G(x - 0.5, y + 0.5) - G(x + 0.5, y - 0.5) + G(x - 0.5, y - 0.5);
}
// Near-field tap table (E77b). Far field: 2× tent splat (per axis pixel 2c → c:¾, c−1:¼; 2c+1 → c:¾, c+1:¼),
// coarse kernel Kc(dc) = (K_far − σ²/2·∇²K)(2dc)·w_dir(2dc)·cut(|2dc|), bilinear gather at (p+½)/2−½. For receiver
// phase φ = p mod 2 and offset d (|d|∞ ≤ W) the near weight per direction is K(d)·w_dir(d) minus the coarse path's
// contribution for the emitter p + d. Entries (dx, dy, weight, dir) grouped by phase; returns offsets and counts.
const NEAR_R = 6;
const NEAR_RHO = 0.5;
const NEAR_SIG2 = 1.5;
const NEAR_TOL = 1e-2; // relative to K(d); E77b
function quadrantWeightsJS(dx: number, dy: number): number[] {
  if (dx === 0 && dy === 0) return [0.25, 0.25, 0.25, 0.25];
  const ax = Math.abs(dx), ay = Math.abs(dy);
  const diag = ax === ay ? 0.5 : 1;
  const w = [0, 0, 0, 0];
  if (dx >= 1 && ay <= dx) w[0] = diag;
  if (dy >= 1 && ax <= dy) w[1] = diag;
  if (dx <= -1 && ay <= -dx) w[2] = diag;
  if (dy <= -1 && ax <= -dy) w[3] = diag;
  return w;
}
function nearTaps(): { data: Float32Array; offsets: number[][]; ends: number[][] } {
  const cut = (r: number) => {
    const t = Math.min(1, Math.max(0, (r - NEAR_R * (1 - NEAR_RHO)) / (2 * NEAR_RHO * NEAR_R)));
    return t * t * (3 - 2 * t);
  };
  const kc = (dcx: number, dcy: number, dir: number) => {
    const fx = 2 * dcx, fy = 2 * dcy;
    const r = Math.hypot(fx, fy);
    if (r === 0) return 0;
    const k = 1 / r + 1 / (24 * r ** 3) - (NEAR_SIG2 / 2) / r ** 3;
    return k * quadrantWeightsJS(fx, fy)[dir] * cut(r);
  };
  const splat = (q: number) => {
    const c = Math.floor(q / 2);
    return [[c, 0.75], [q - 2 * c === 0 ? c - 1 : c + 1, 0.25]] as [number, number][];
  };
  const gather = (p: number) => {
    const cc = (p + 0.5) / 2 - 0.5;
    const c0 = Math.floor(cc);
    const f = cc - c0;
    return [[c0, 1 - f], [c0 + 1, f]] as [number, number][];
  };
  const out: number[] = [];
  const offsets: number[][] = [];
  const ends: number[][] = [];
  // prune relative to the kernel value at the offset: far residues (phase-dependent interpolation error of the
  // coarse path) below NEAR_TOL·K(d) are dropped, the ramp region is kept
  for (let phase = 0; phase < 4; phase++) {
    const px = 100 + (phase & 1), py = 100 + (phase >> 1);
    const gx = gather(px), gy = gather(py);
    const off: number[] = [], end: number[] = [];
    for (let dir = 0; dir < 4; dir++) {
      off.push(out.length / 4);
      for (let dy = -NEAR_W; dy <= NEAR_W; dy++) {
        for (let dx = -NEAR_W; dx <= NEAR_W; dx++) {
          const sx = splat(px + dx), sy = splat(py + dy);
          const exact = kernelClosed(dx, dy) * quadrantWeightsJS(dx, dy)[dir];
          let far = 0;
          for (const [csy, wsy] of sy)
            for (const [csx, wsx] of sx)
              for (const [cgy, wgy] of gy)
                for (const [cgx, wgx] of gx) far += wsy * wsx * wgy * wgx * kc(csx - cgx, csy - cgy, dir); // emitter − receiver
          const w = exact - far;
          if (Math.abs(w) > NEAR_TOL * kernelClosed(dx, dy)) out.push(dx, dy, w, dir);
        }
      }
      end.push(out.length / 4);
    }
    offsets.push(off);
    ends.push(end);
  }
  for (let phase = 0; phase < 4; phase++) {
    if (ends[phase][3] - offsets[phase][0] > NEAR_MAX_TAPS) throw new Error('hrc-ratio: near tap table exceeds NEAR_MAX_TAPS');
  }
  return { data: new Float32Array(out), offsets, ends };
}

const f32v = new Float32Array(1);
const u32v = new Uint32Array(f32v.buffer);
type ExactEntry = {
  key: string;
  w: number;
  h: number;
  m: number;
  chunkLen: number;
  depth: number;
  nChunks: number;
  order: Uint32Array;
  nextChunk: number; // index into order of the chunk in progress
  nextRow: number; // first row not yet done for that chunk
  rows: number; // band height (adapted to the time budget)
  acc: GPUBuffer;
  out: GPUTexture;
  outView: GPUTextureView;
  world: GPUTexture | null; // snapshot (freed when done)
  sky: GPUTexture | null;
  params: GPUBuffer;
  started: number;
  elapsedMs: number;
  workDone: number; // pixel-intervals done
  running: boolean;
};

function halfToFloat(h: number): number {
  const e = (h >> 10) & 31, m = h & 1023, sgn = h >> 15 ? -1 : 1;
  if (e === 0) return sgn * m * 2 ** -24;
  if (e === 31) return m ? NaN : sgn * Infinity;
  return sgn * (1 + m / 1024) * 2 ** (e - 15);
}

function floatToHalf(x: number): number {
  f32v[0] = x;
  const b = u32v[0];
  const sign = (b >>> 16) & 0x8000;
  let exp = (b >>> 23) & 0xff;
  const mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - e);
    return sign | ((m + 0x1000) >>> 13);
  }
  return sign | (e << 10) | ((mant + 0x1000) >>> 13);
}

const TEX_RENDER = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
const TEX_STORAGE = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
const SKY_CIRCLE_SIZE = 256;

const WG_BOUNCE = [16, 16] as const;

function uboView(shader: string, name: string): StructuredView {
  return makeStructuredView(makeShaderDataDefinitions(shader).uniforms[name]);
}

function tex(device: GPUDevice, label: string, w: number, h: number, format: GPUTextureFormat, usage: number) {
  const t = device.createTexture({ label, size: { width: w, height: h }, format, usage });
  return [t, t.createView()] as const;
}

function bg(device: GPUDevice, layout: GPUBindGroupLayout, ...resources: GPUBindingResource[]) {
  return device.createBindGroup({
    layout,
    entries: resources.map((resource, binding) => ({ binding, resource })),
  });
}

function computePipeline(device: GPUDevice, label: string, code: string) {
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
}

function uploadVertexData(
  device: GPUDevice,
  existing: GPUBuffer | undefined,
  data: Float32Array<ArrayBuffer>,
  count?: number,
): GPUBuffer {
  const bytes = (count ?? data.length) * 4;
  if (!existing || existing.size < bytes) {
    existing?.destroy();
    existing = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }
  device.queue.writeBuffer(existing, 0, data, 0, count);
  return existing;
}

// ── Shared WGSL functions ──

const wgslPackF16 = /*wgsl*/ `
fn packF16(v: vec4f) -> vec2u { return vec2u(pack2x16float(v.xy), pack2x16float(v.zw)); }
fn unpackF16(p: vec2u) -> vec4f { return vec4f(unpack2x16float(p.x), unpack2x16float(p.y)); }
`;

const wgslSrgb = /*wgsl*/ `
// exact sRGB transfer (IEC 61966-2-1), not the 2.2 power approximation
fn srgbToLinear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }
`;

const wgslSliceOffset = /*wgsl*/ `
fn dirToSliceOffset(dirIdx: i32, intervalSize: i32) -> i32 {
  return dirIdx * 2 - intervalSize;
}
`;

const wgslRGB9E5 = /*wgsl*/ `
fn packRGB9E5(c: vec3f) -> u32 {
  let maxC = max(c.r, max(c.g, c.b));
  var exp_shared: i32;
  var scale: f32;
  if (maxC < 6.10352e-5) {
    exp_shared = 0;
    scale = 0.0;
  } else {
    let e = clamp(i32(ceil(log2(maxC))) + 15, 0, 31);
    exp_shared = e;
    scale = exp2(f32(-e + 15 + 9));
  }
  let r = u32(clamp(c.r * scale, 0.0, 511.0));
  let g = u32(clamp(c.g * scale, 0.0, 511.0));
  let b = u32(clamp(c.b * scale, 0.0, 511.0));
  return r | (g << 9u) | (b << 18u) | (u32(exp_shared) << 27u);
}

fn unpackRGB9E5(p: u32) -> vec3f {
  let r = f32(p & 0x1FFu);
  let g = f32((p >> 9u) & 0x1FFu);
  let b = f32((p >> 18u) & 0x1FFu);
  let e = f32((p >> 27u) & 0x1Fu) - 15.0 - 9.0;
  return vec3f(r, g, b) * exp2(e);
}
`;

// ── Cascade channel modes ──
// The cascade shaders are written against an abstract radiance type `Rad` so
// the same transport code runs in two modes:
//   rgb  — Rad = vec3f. Rays: 4×f16 (8 B). Merge/fluence: RGB9E5 (4 B).
//   mono — Rad = f32 (luminance). Rays: 2×f16 (4 B). Merge/fluence: f32 (4 B).
// Transport is linear per channel with scalar transmittance, so a mono cascade
// computes exactly dot(LUMA, rgb cascade) up to quantization.

type ChannelMode = 'rgb' | 'mono';

const wgslLuma = /*wgsl*/ `
const LUMA = vec3f(0.2126, 0.7152, 0.0722);
`;

const wgslChannel: Record<ChannelMode, string> = {
  rgb:
    wgslPackF16 +
    wgslRGB9E5 +
    /*wgsl*/ `
alias Rad = vec3f;
alias RayPacked = vec2u;
alias MergePacked = u32;
struct RayData { rad: Rad, trans: f32 }
fn toRad(c: vec3f) -> Rad { return c; }
fn packRay(r: RayData) -> RayPacked { return packF16(vec4f(r.rad, r.trans)); }
fn unpackRay(p: RayPacked) -> RayData { let v = unpackF16(p); return RayData(v.rgb, v.a); }
fn packMerge(r: Rad) -> MergePacked { return packRGB9E5(r); }
fn unpackMerge(p: MergePacked) -> Rad { return unpackRGB9E5(p); }
`,
  mono:
    wgslLuma +
    /*wgsl*/ `
alias Rad = f32;
alias RayPacked = u32;
alias MergePacked = f32;
struct RayData { rad: Rad, trans: f32 }
fn toRad(c: vec3f) -> Rad { return dot(c, LUMA); }
fn packRay(r: RayData) -> RayPacked { return pack2x16float(vec2f(r.rad, r.trans)); }
fn unpackRay(p: RayPacked) -> RayData { let v = unpack2x16float(p); return RayData(v.x, v.y); }
fn packMerge(r: Rad) -> MergePacked { return r; }
fn unpackMerge(p: MergePacked) -> Rad { return p; }
`,
};

const RAY_BYTES: Record<ChannelMode, number> = { rgb: 8, mono: 4 };

// ── World-render shaders (shapes, lines, mouse light → world + material) ──
//
// Two render targets (MRT):
//   location(0) = world    — rgb: emitted radiance, a: opacity (0–1)
//   location(1) = material — r: albedo (0–1)
//
// Opacity is the per-pixel extinction: fraction of light absorbed or scattered
// per unit traversal. opacity=1 is a fully opaque solid, opacity=0 is vacuum.
//
// Albedo is the single-scattering albedo (ω₀): the fraction of interacted
// light that is re-emitted rather than absorbed. It unifies surface
// reflectance and volume scattering into one parameter:
//   - Opaque surface (opacity=1, albedo=0.8): 80% of light reflects diffusely
//   - Volume (opacity=0.03, albedo=0.5): each step scatters 1.5% of light
//   - Absorber (albedo=0): all interacted light is destroyed

const worldRenderShader =
  wgslSrgb +
  /*wgsl*/ `
struct VertexInput {
  @location(0) position: vec2f,
  @location(1) color: vec3f,
  @location(2) opacity: f32,
  @location(3) albedo: f32,
}
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) opacity: f32,
  @location(2) albedo: f32,
}
@vertex fn vertex_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(input.position, 0.0, 1.0);
  out.color = input.color;
  out.opacity = input.opacity;
  out.albedo = input.albedo;
  return out;
}
struct FragOut { @location(0) world: vec4f, @location(1) material: vec4f }
@fragment fn fragment_main(in: VertexOutput) -> FragOut {
  var out: FragOut;
  out.world = vec4f(srgbToLinear(in.color), in.opacity);
  out.material = vec4f(in.albedo, 0.0, 0.0, 0.0);
  return out;
}
`;

const lineRenderShader =
  wgslSrgb +
  /*wgsl*/ `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) p1: vec2f,
  @location(2) p2: vec2f,
  @location(3) radius: f32,
  @location(4) opacity: f32,
  @location(5) albedo: f32,
}
struct Canvas { width: f32, height: f32 }
@group(0) @binding(0) var<uniform> canvas: Canvas;
@vertex fn vertex_main(
  @builtin(vertex_index) vid: u32,
  @location(0) p1: vec2f, @location(1) p2: vec2f,
  @location(2) color: vec3f, @location(3) thickness: f32,
  @location(4) opacity: f32, @location(5) albedo: f32,
) -> VertexOutput {
  let r = thickness * 0.5;
  let minP = min(p1, p2) - vec2f(r);
  let maxP = max(p1, p2) + vec2f(r);
  var corners = array<vec2f, 6>(
    vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(1,0), vec2f(1,1), vec2f(0,1),
  );
  let c = corners[vid];
  let pixel = minP + (maxP - minP) * c;
  let clip = vec2f(pixel.x / canvas.width * 2.0 - 1.0, 1.0 - pixel.y / canvas.height * 2.0);
  var out: VertexOutput;
  out.position = vec4f(clip, 0.0, 1.0);
  out.color = color; out.p1 = p1; out.p2 = p2; out.radius = r;
  out.opacity = opacity; out.albedo = albedo;
  return out;
}
struct FragOut { @location(0) world: vec4f, @location(1) material: vec4f }
@fragment fn fragment_main(in: VertexOutput) -> FragOut {
  let pos = in.position.xy;
  let ab = in.p2 - in.p1; let ap = pos - in.p1;
  let lenSq = dot(ab, ab);
  let t = select(clamp(dot(ap, ab) / lenSq, 0.0, 1.0), 0.0, lenSq < 0.001);
  let nearest = in.p1 + ab * t;
  let d = length(pos - nearest) - in.radius;
  if (d > 0.0) { discard; }
  var out: FragOut;
  out.world = vec4f(srgbToLinear(in.color), in.opacity);
  out.material = vec4f(in.albedo, 0.0, 0.0, 0.0);
  return out;
}
`;

// ── HRC Phase A: Ray Seed — direct tracing of T_0 and T_1 (paper §4.2, Alg. 1) ──
// Samples world + bounce textures at each probe position. Computes per-cell
// radiance and transmittance using discrete Beer-Lambert: T = (1−α)^spacing.
// Emission is multiplied by (1 - T) because emission and absorption are coupled
// through the extinction coefficient: a cell that absorbs fraction (1-T) of
// passing light also converts fraction (1-T) of its own emission into the ray.
// This is the source function integral over the cell thickness (paper Eq. 4).
// The bounce texture (at probe resolution) provides the diffuse re-emission
// from the previous frame's fluence, enabling multi-bounce GI over time.
//
// Both levels are traced as unit-width strips (mean of two sub-lines at ±¼ cell): rays from pixel
// centre to pixel centre at ±45° pass exactly through cell corners, where a thin ray under-counts
// opaque cells (the diagonal "X") and slips through diagonal walls. The seed rays have fixed geometry,
// so their exact cell chords are generated here; one invocation per probe fetches its footprint
// cells once and composes all 2^n+1 rays from registers.

// Cascade buffers are slice-fastest: ray (probe, ray, slice) at (probe·numRays + ray)·sliceCount + slice,
// merge (x, slice) at x·sliceCount + slice, and every pass walks slices with 64-wide workgroups.

// Exact DDA chords (dx, dy, length) of the segment a→b through unit cells, relative to floor(a).
function chordList(ax: number, ay: number, bx: number, by: number): [number, number, number][] {
  const dx = bx - ax,
    dy = by - ay,
    L = Math.hypot(dx, dy),
    ux = dx / L,
    uy = dy / L;
  let ix = Math.floor(ax),
    iy = Math.floor(ay);
  const sx = ux > 0 ? 1 : -1,
    sy = uy > 0 ? 1 : -1;
  let tmx = ux ? ((ux > 0 ? ix + 1 : ix) - ax) / ux : 1e30;
  let tmy = uy ? ((uy > 0 ? iy + 1 : iy) - ay) / uy : 1e30;
  const tdx = ux ? Math.abs(1 / ux) : 1e30,
    tdy = uy ? Math.abs(1 / uy) : 1e30;
  const out: [number, number, number][] = [];
  let t0 = 0;
  while (t0 < L) {
    const t1 = Math.min(tmx, tmy, L);
    if (t1 - t0 > 1e-9) out.push([ix - Math.floor(ax), iy - Math.floor(ay), t1 - t0]);
    if (t1 >= L) break;
    if (tmx < tmy) {
      ix += sx;
      tmx += tdx;
    } else {
      iy += sy;
      tmy += tdy;
    }
    t0 = t1;
  }
  return out;
}

const raySeedShader = (mode: ChannelMode, level: number) => {
  const n = 1 << level;
  const rays = Array.from({ length: n + 1 }, (_, k) =>
    [-0.25, 0.25].map((u) => chordList(0.5, 0.5 + u, 0.5 + n, 0.5 + u + 2 * k - n)),
  );
  const cells = new Map<string, [number, number]>();
  for (const subs of rays) for (const c of subs) for (const [a, b] of c) cells.set(`${a},${b}`, [a, b]);
  const name = (a: number, b: number) => `c${a < 0 ? 'm' : ''}${Math.abs(a)}_${b < 0 ? 'm' : ''}${Math.abs(b)}`;
  const fetches = [...cells.values()].map(([a, b]) => `  let ${name(a, b)} = cellEB(vec2i(x0 + ${a}, y + ${b}));`);
  const body = rays.map((subs, k) => {
    const lines = ['  {', '    var sum = vec4f(0.0);'];
    for (const c of subs) {
      lines.push('    { var acc = vec4f(0.0, 0.0, 0.0, 1.0); var t = 0.0;');
      for (const [a, b, len] of c) {
        const v = name(a, b);
        // Volumetric emitters (opacity 0): emission ∝ chord length (NOTES §44/§47); otherwise first-hit emission E·(1 − t).
        lines.push(`      { let L = ${len.toFixed(9)} * params.probeSpacing; t = pow(${v}.a, L); acc = vec4f(acc.rgb + acc.a * ${v}.rgb * select(1.0 - t, L, ${v}.a > 0.99999), acc.a * t); }`);
      }
      lines.push('      sum += acc; }');
    }
    lines.push(`    let r = 0.5 * sum;`);
    lines.push(`    rayOut[(gid.y * ${n + 1}u + ${k}u) * sc + gid.x] = packRay(RayData(toRad(r.rgb), r.a));`);
    lines.push('  }');
    return lines.join('\n');
  });
  return (
    wgslChannel[mode] +
    /*wgsl*/ `
struct SeedParams {
  probeCount: u32,
  sliceCount: u32,
  screenW: f32,
  screenH: f32,
  probeSpacing: f32,
  freeWorld: u32,
  transformX: vec4f,
  transformY: vec4f,
};

@group(0) @binding(0) var worldTex: texture_2d<f32>;
@group(0) @binding(1) var bounceTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> rayOut: array<RayPacked>;
@group(0) @binding(3) var<uniform> params: SeedParams;

// (emission + bounce, 1 − opacity) of frustum cell c; vacuum outside the screen
fn cellEB(c: vec2i) -> vec4f {
  let p = vec3f(f32(c.x) + 0.5, f32(c.y) + 0.5, 1.0);
  let wp = vec2f(dot(params.transformX.xyz, p), dot(params.transformY.xyz, p));
  let px = vec2i(floor(wp));
  if (px.x < 0 || px.y < 0 || px.x >= i32(params.screenW) || px.y >= i32(params.screenH)) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let world = textureLoad(worldTex, px, 0);
  let bdim = vec2i(textureDimensions(bounceTex, 0));
  let bounce = textureLoad(bounceTex, clamp(vec2i(wp / vec2f(params.screenW, params.screenH) * vec2f(bdim)), vec2i(0), bdim - 1), 0).rgb;
  // freeWorld: the same emitters with every cell transparent (the ratio's free channel, NOTES §47); emission per
  // unit length E (volumetric light) or E·α (medium), matching the scene cascade's E·(1 − t) to first order
  if (params.freeWorld == 1u) {
    return vec4f(world.rgb * select(world.a, 1.0, world.a < 1e-6) + bounce, 1.0);
  }
  return vec4f(world.rgb + bounce, 1.0 - world.a);
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let levelProbes = (params.probeCount + ${n - 1}u) >> ${level}u;
  if (gid.y >= levelProbes || gid.x >= params.sliceCount) { return; }
  let x0 = i32(gid.y) * ${n};
  let y = i32(gid.x);
  let sc = params.sliceCount;
${fetches.join('\n')}
${body.join('\n')}
}
`
  );
};

// ── HRC Phase B: Ray Extension — "Merge Up", building T_1..T_N (paper §4.1) ──
// Composes two shorter rays from the previous level into one longer ray.
// Even k: exact composition via Eq. 18: T_{n+1}(p,2k) = Merge(T_n(p,k), T_n(p+v_n(k),k))
// Odd k: averaged cross-composition via Eq. 19-20, interpolating between the
// two nearest finer-level directions.
// Transmittance is scalar, packed in the ray texture alpha channel.
// One invocation per ray pair (2h, 2h+1): the pair shares loads (5 instead of 8).

const rayExtendShader = (mode: ChannelMode) =>
  wgslChannel[mode] +
  wgslSliceOffset +
  /*wgsl*/ `
struct ExtendParams {
  probeCount: u32,
  level: u32,
  sliceCount: u32,
  // equal footprints (research/tree/notes/merge_read_mechanism.md §7, §10): lateral tap [t, 1-2t, t] on the emission of interior
  // parent rays, one for even and one for odd parents (0 = production; negative = sharpen)
  equalTap: f32,
  equalTapOdd: f32,
  equalTapAxis: f32, // the axis ray (rise 0, a ray at every level) has its own tap
};

@group(0) @binding(0) var<storage, read> prevRay: array<RayPacked>;
@group(0) @binding(1) var<storage, read_write> currRay: array<RayPacked>;
@group(0) @binding(2) var<uniform> params: ExtendParams;

fn loadPrev(probeIdx: i32, rayIdx: i32, sliceIdx: i32) -> RayData {
  let prevLevel = params.level - 1u;
  let prevNumProbes = i32((params.probeCount + (1u << prevLevel) - 1u) >> prevLevel);
  let prevNumRays = i32(1u << prevLevel) + 1;
  if (probeIdx < 0 || probeIdx >= prevNumProbes ||
      rayIdx < 0 || rayIdx >= prevNumRays ||
      sliceIdx < 0 || sliceIdx >= i32(params.sliceCount)) {
    return RayData(Rad(0.0), 1.0);
  }
  let idx = (probeIdx * prevNumRays + rayIdx) * i32(params.sliceCount) + sliceIdx;
  return unpackRay(prevRay[idx]);
}

fn compositeRay(near: RayData, far: RayData) -> RayData {
  return RayData(near.rad + far.rad * near.trans, near.trans * far.trans);
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let interval = i32(1u << params.level);
  let half = interval / 2;
  let pairs = half + 1;                       // ray pairs (2h, 2h+1); the last has only ray 2h = interval
  let levelProbes = i32((params.probeCount + (1u << params.level) - 1u) >> params.level);
  let x = i32(gid.y);
  let sliceIdx = i32(gid.x);
  let probeIdx = i32(floor((f32(x) + 0.5) / f32(pairs)));
  let h = x - probeIdx * pairs;
  if (probeIdx >= levelProbes || sliceIdx >= i32(params.sliceCount)) { return; }

  let p0 = probeIdx * 2;
  let a = loadPrev(p0, h, sliceIdx);
  let offA = dirToSliceOffset(h, half);
  let farA = loadPrev(p0 + 1, h, sliceIdx + offA);
  var even = compositeRay(a, farA);
  if (params.equalTap > 0.0 && h > 0 && h < half) {
    // blur the children's radiance across slices (transmittance from the own slice), as research/tree/lab/backport_gain.extension
    let t = select(params.equalTap, params.equalTapAxis, 2 * h == half);
    let nr = (1.0 - 2.0 * t) * a.rad + t * (loadPrev(p0, h, sliceIdx - 1).rad + loadPrev(p0, h, sliceIdx + 1).rad);
    let fr = (1.0 - 2.0 * t) * farA.rad + t * (loadPrev(p0 + 1, h, sliceIdx + offA - 1).rad + loadPrev(p0 + 1, h, sliceIdx + offA + 1).rad);
    even = RayData(nr + fr * a.trans, even.trans);
  }
  let sc = i32(params.sliceCount);
  let base = (probeIdx * (interval + 1) + 2 * h) * sc + sliceIdx;
  currRay[base] = packRay(even);
  if (h < half) {
    let offB = dirToSliceOffset(h + 1, half);
    let b = loadPrev(p0, h + 1, sliceIdx);
    let farA1 = loadPrev(p0 + 1, h + 1, sliceIdx + offA);
    let farB = loadPrev(p0 + 1, h, sliceIdx + offB);
    let crossA = compositeRay(a, farA1);
    let crossB = compositeRay(b, farB);
    var rad = (crossA.rad + crossB.rad) * 0.5;
    if (params.equalTapOdd != 0.0) {
      let t = params.equalTapOdd;
      let ra = (1.0 - 2.0 * t) * a.rad + t * (loadPrev(p0, h, sliceIdx - 1).rad + loadPrev(p0, h, sliceIdx + 1).rad);
      let fa = (1.0 - 2.0 * t) * farA1.rad + t * (loadPrev(p0 + 1, h + 1, sliceIdx + offA - 1).rad + loadPrev(p0 + 1, h + 1, sliceIdx + offA + 1).rad);
      let rb = (1.0 - 2.0 * t) * b.rad + t * (loadPrev(p0, h + 1, sliceIdx - 1).rad + loadPrev(p0, h + 1, sliceIdx + 1).rad);
      let fb = (1.0 - 2.0 * t) * farB.rad + t * (loadPrev(p0 + 1, h, sliceIdx + offB - 1).rad + loadPrev(p0 + 1, h, sliceIdx + offB + 1).rad);
      rad = ((ra + fa * a.trans) + (rb + fb * b.trans)) * 0.5;
    }
    currRay[base + sc] = packRay(RayData(rad, (crossA.trans + crossB.trans) * 0.5));
  }
}
`;

// ── HRC Phase C: Cascade Merge — "Merge Down", computing R_{N-1}..R_0 (paper §4.2) ──
// Reads pre-computed ray data and merges with far-field fluence from higher levels.
// Parity-dependent cascade connection:
//   Odd probes (Eq. 14): Merge_r(A_{n+1}(j) × Trace(p,q), R_{n+1}(q,j))
//   Even probes (Eq. 15-17): Richardson-average of local composite with
//     coarser level's result to cancel first-order spatial interpolation error.

// One invocation per cone pair (2w, 2w+1) at levels ≥ 1 — the pair's rays are 2w, 2w+1, 2w+2 and the
// middle one (and, for even probes, its extension) is shared — and one per probe at level 0.
// Merge-side read (research/tree/notes/merge_footprints.md): a sub-bin's near radiance is a stencil over the rays of the same probe
// from slices s-1..s+1 (rows, nd = 3) and neighbouring ray indices (slopes: noo offsets for odd probes at level n, noe for even
// probes' own chain at level n+1), each ray with its own gated radiance; the far term keeps the centre ray's transmittance.
// Weights per (level, branch, sub-bin) as 3 slices x 5 ray offsets. One workgroup = 64 slices of one (probe, cone pair): the
// window of rays is loaded once into workgroup memory (own slice + one halo slice each side). Even probes read their own chain
// from the stored level-(n+1) rays (the top level composes it on the fly, even offsets only).
interface MergeReadShape {
  nd: number;
  noo: number;
  noe: number;
  param?: boolean; // rows read with weights computed from per-level constants (any depth; no table loads)
  constW?: boolean; // timing only: constant weights instead of the table (isolates the ray loads)
  sg?: boolean; // slice neighbours by subgroup shuffles (needs the 'subgroups' feature); edge lanes load directly
  parity?: 'wg' | 'direct' | 'sg' | 'halo' | 'none' | 'sghalo'; // the derived parity read (neighbours via workgroup memory, direct loads or subgroup shuffles) (research/tree/notes/focus_and_extension.md §13–14): weights from kappa, no table
}
const MERGE_READ_SHAPES: Record<number, MergeReadShape> = {
  1: { nd: 3, noo: 1, noe: 1 },
  2: { nd: 1, noo: 3, noe: 5 },
  3: { nd: 3, noo: 3, noe: 5 },
  11: { nd: 3, noo: 1, noe: 1, constW: true },
  12: { nd: 1, noo: 3, noe: 5, constW: true },
  13: { nd: 3, noo: 3, noe: 5, constW: true },
  4: { nd: 3, noo: 1, noe: 1, param: true },
  // parity read variants (same output; merge pass at 1024^2, M2 Pro, production 5.4-5.7 ms): 5 workgroup memory + edge loads 11.0,
  // 6 direct neighbour loads 9.0, 7 subgroup shuffles + edge loads 10.0, 8 workgroup halo 8.1, 9 no neighbours (baseline) 6.3,
  // 10 subgroup halo 7.1-7.4 (parityRead uses 10, else 8)
  5: { nd: 3, noo: 1, noe: 1, parity: 'wg' },
  6: { nd: 3, noo: 1, noe: 1, parity: 'direct' },
  7: { nd: 3, noo: 1, noe: 1, parity: 'sg', sg: true },
  8: { nd: 3, noo: 1, noe: 1, parity: 'halo' },
  9: { nd: 3, noo: 1, noe: 1, parity: 'none' },
  10: { nd: 3, noo: 1, noe: 1, parity: 'sghalo', sg: true }, // subgroup shuffles, 30 slices per subgroup of 32 (timing experiment) // timing only: the same merge without the neighbours (d = 0)
  21: { nd: 3, noo: 1, noe: 1, sg: true },
  23: { nd: 3, noo: 3, noe: 5, sg: true },
};
// fitted on the research training scenes (research/tree/lab/read_param.py); entries [aEdge, aInt, dEdge, dInt, R]
const MERGE_READ_PARAMS: Record<string, { o: number[]; e: number[] }> = {};
const mergeReadDecls = (r: MergeReadShape) => {
  const HD = (r.nd - 1) / 2,
    HO = (r.noo - 1) / 2,
    HE = (r.noe - 1) / 2;
  const RW = Math.max(3 + 2 * HO, 5 + 2 * HE);
  return /*wgsl*/ `
@group(0) @binding(7) var<storage, read> rayNext: array<RayPacked>;
@group(0) @binding(8) var<storage, read> readW: array<f32>;
const HD: i32 = ${HD};
const HO: i32 = ${HO};
const HE: i32 = ${HE};
const RW: i32 = ${RW};
fn loadRayNext(probeIdx: i32, rayIdx: i32, sliceIdx: i32) -> RayData {
  let nr = (2 << params.level) + 1;
  if (probeIdx < 0 || probeIdx >= i32(params.nextProbes) || rayIdx < 0 || rayIdx >= nr ||
      sliceIdx < 0 || sliceIdx >= i32(params.sliceCount)) {
    return RayData(Rad(0.0), 1.0);
  }
  return unpackRay(rayNext[(probeIdx * nr + rayIdx) * i32(params.sliceCount) + sliceIdx]);
}
// window ray idx (level n for odd probes, level n+1 for even probes) at slice s
fn winRay(isEven: bool, probeIdx: i32, idx: i32, s: i32) -> RayData {
  if (!isEven) { return loadRay(probeIdx, idx, s); }
  if (params.isTopCascade == 0u) { return loadRayNext(probeIdx / 2, idx, s); }
  if ((idx & 1) != 0) { return RayData(Rad(0.0), 1.0); }
  let k = idx / 2;
  let a = loadRay(probeIdx, k, s);
  let b = loadRay(probeIdx + 1, k, s + dirToSliceOffset(k, i32(1u << params.level)));
  return RayData(a.rad + b.rad * a.trans, a.trans * b.trans);
}
fn readWeight(subBin: u32, isEven: bool, a: i32, b: i32) -> f32 {
  var base = params.readBase + subBin * 15u;
  if (isEven) { base += (2u << params.level) * 15u; }
  return readW[base + u32(a * 5 + b)];
}
fn contribRead(nearRad: Rad, nearT: f32, isEven: bool, farStep: i32, probeIdx: i32, sliceIdx: i32, subBin: u32, rayIdx: i32) -> Rad {
  let level = params.level;
  let weight = getConeArc(subBin);
  let sliceOff = dirToSliceOffset(rayIdx, i32(1u << level));
  let farX = ((probeIdx + farStep) << level) + i32(subBin);
  let farSlice = sliceIdx + sliceOff * farStep;
  var farFluence: Rad;
  if (params.isTopCascade == 1u ||
      farX < 0 || farX >= i32(params.mergeInWidth) ||
      farSlice < 0 || farSlice >= i32(params.sliceCount)) {
    farFluence = loadSkyFluence(subBin);
  } else {
    farFluence = unpackMerge(mergeInBuf[farX * i32(params.sliceCount) + farSlice]);
  }
  let merged = nearRad * weight + farFluence * nearT;
  if (isEven) { return mix(merged, loadMerge((probeIdx << level) + i32(subBin), sliceIdx), params.evenWeight); }
  return merged;
}`;
};
const mergeReadMain = (r: MergeReadShape) => {
  // Fully unrolled: own-slice rays in registers, the slice neighbours (rows) through workgroup memory, compile-time centres.
  const HD = (r.nd - 1) / 2,
    HO = (r.noo - 1) / 2,
    HE = (r.noe - 1) / 2;
  const NWo = 3 + 2 * HO,
    NWe = 5 + 2 * HE;
  const RW = Math.max(NWo, NWe);
  const block = (even: boolean) => {
    const NW = even ? NWe : NWo,
      H = even ? HE : HO,
      L: string[] = [];
    const lo = even ? `2 * b0 - ${HE}` : `b0 - ${HO}`;
    const load = (ri: number, slice: string) =>
      even
        ? `winEven(probeIdx, ${lo} + ${ri}, ${slice})`
        : `loadRay(probeIdx, ${lo} + ${ri}, ${slice})`;
    for (let ri = 0; ri < NW; ri++) {
      if (even && (ri - HE) % 2 !== 0 && false) continue;
      L.push(`    let w${ri} = ${load(ri, 'sliceIdx')};`);
      if (HD > 0 && r.sg) {
        // slice neighbours from the neighbouring lanes; the subgroup's edge lanes load theirs
        L.push(`    var m${ri} = subgroupShuffleUp(w${ri}.rad, 1u);`);
        L.push(`    var p${ri} = subgroupShuffleDown(w${ri}.rad, 1u);`);
        L.push(`    if (sgLane == 0u) { m${ri} = ${load(ri, 'sliceIdx - 1')}.rad; }`);
        L.push(`    if (sgLane == sgSize - 1u) { p${ri} = ${load(ri, 'sliceIdx + 1')}.rad; }`);
      } else if (HD > 0) {
        // slice neighbours: direct loads (the neighbouring threads fetch the same cache lines)
        L.push(`    let m${ri} = ${load(ri, 'sliceIdx - 1')}.rad;`);
        L.push(`    let p${ri} = ${load(ri, 'sliceIdx + 1')}.rad;`);
      }
    }
    L.push('    if (sliceIdx < sc) {');
    for (let q = 0; q < 4; q++) {
      const kOff = (q + 1) >> 1; // kj = b0 + kOff
      const c = even ? 2 * kOff + HE : kOff + HO;
      const terms: string[] = [];
      for (let da = -HD; da <= HD; da++)
        for (let o = -H; o <= H; o++) {
          const src = da === 0 ? `w${c + o}.rad` : `${da < 0 ? 'm' : 'p'}${c + o}`;
          terms.push(
            r.param
              ? `pw${q}[${da + 1}] * ${src}`
              : r.constW
                ? `${(0.1 + 0.01 * ((da + 1) * 5 + o + 2)).toFixed(3)} * ${src}`
                : `readW[wb${q} + ${(da + 1) * 5 + (o + 2)}u] * ${src}`,
          );
        }
      const guard = q >= 2 ? '      if (level > 0u) {\n' : '      {\n';
      L.push(guard + `        let wb${q} = params.readBase + u32(2 * b0 + ${q}) * 15u + ${even ? '(2u << level) * 15u' : '0u'};`);
      if (r.param) L.push(`        let pw${q} = paramRowWeights(u32(2 * b0 + ${q}), ${even});`);
      L.push(`        let near = ${terms.join(' + ')};`);
      L.push(`        let v = contribRead(near, w${c}.trans, ${even}, farStep, probeIdx, sliceIdx, u32(2 * b0 + ${q}), b0 + ${kOff});`);
      L.push(`        ${q < 2 ? 'res0' : 'res1'} += v;\n      }`);
    }
    L.push('    }');
    return L.join('\n');
  };
  return /*wgsl*/ `
// compact rows read: (a - d, 1 - 2a, a + d) over slices s-1, s, s+1 for sub-bins of the lower half (i = j + 1/2 from the quadrant
// edge), mirrored for the upper half; a, d relax from their quadrant-edge to their interior values as exp(-i / R)
fn paramRowWeights(subBin: u32, isEven: bool) -> vec3f {
  let L = 1u << params.level;
  var p = vec4f(params.rpO0, params.rpO1, params.rpO2, params.rpO3);
  var invR = params.rpO4;
  if (isEven) { p = vec4f(params.rpE0, params.rpE1, params.rpE2, params.rpE3); invR = params.rpE4; }
  let upper = subBin >= L;
  let i = f32(select(subBin, 2u * L - 1u - subBin, upper)) + 0.5;
  let e = exp(-i * invR);
  let a = p.y + (p.x - p.y) * e;
  let d = p.w + (p.z - p.w) * e;
  return select(vec3f(a - d, 1.0 - 2.0 * a, a + d), vec3f(a + d, 1.0 - 2.0 * a, a - d), upper);
}
fn winEven(probeIdx: i32, idx: i32, s: i32) -> RayData {
  if (params.isTopCascade == 0u) { return loadRayNext(probeIdx / 2, idx, s); }
  if ((idx & 1) != 0) { return RayData(Rad(0.0), 1.0); }
  let k = idx / 2;
  let a = loadRay(probeIdx, k, s);
  let b = loadRay(probeIdx + 1, k, s + dirToSliceOffset(k, i32(1u << params.level)));
  return RayData(a.rad + b.rad * a.trans, a.trans * b.trans);
}
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lidv: vec3u, @builtin(workgroup_id) wid: vec3u${r.sg ? ', @builtin(subgroup_invocation_id) sgLane: u32, @builtin(subgroup_size) sgSize: u32' : ''}) {
  let level = params.level;
  let numDirections = i32(1u << level);
  let sliceIdx = i32(gid.x);
  let x = i32(wid.y);   // workgroups are 64 x 1: one x per workgroup (uniform)
  let sc = i32(params.sliceCount);
  var probeIdx = x;
  var b0 = 0;
  if (level > 0u) {
    probeIdx = x >> (level - 1u);
    b0 = 2 * (x & (numDirections / 2 - 1));
  }
  if (probeIdx >= i32(params.levelProbes)) { return; }
  let isEven = (probeIdx % 2 == 0);
  let farStep = select(1, 2, isEven);
  let l = i32(lidv.x);
  _ = &readW;
  _ = &rayNext;
  var res0 = Rad(0.0);
  var res1 = Rad(0.0);
  if (isEven) {
${block(true)}
  } else {
${block(false)}
  }
  if (sliceIdx >= sc) { return; }
  if (level == 0u) {
    var fc: vec2i;
    switch (params.direction) {
      case 0u: { fc = vec2i(probeIdx - 1, sliceIdx); }
      case 1u: { fc = vec2i(sliceIdx, probeIdx - 1); }
      case 2u: { fc = vec2i(i32(params.fluenceW) - probeIdx, sliceIdx); }
      case 3u: { fc = vec2i(sliceIdx, i32(params.fluenceH) - probeIdx); }
      default: { fc = vec2i(probeIdx - 1, sliceIdx); }
    }
    if (fc.x >= 0 && fc.x < i32(params.fluenceW) && fc.y >= 0 && fc.y < i32(params.fluenceH)) {
      var fi = fc.y * i32(params.fluenceStride) + fc.x;
      if (params.fluenceDirs == 4u) { fi = fi * 4 + i32(params.direction); }
      fluenceBuf[fi] = packMerge(unpackMerge(fluenceBuf[fi]) + res0);
    }
    return;
  }
  let outX = (probeIdx << level) + b0;
  mergeOutBuf[outX * sc + sliceIdx] = packMerge(res0);
  mergeOutBuf[(outX + 1) * sc + sliceIdx] = packMerge(res1);
}
`;
};

// Parity read (merge-read 5): sub-bin j's near radiance read as r(s) + (mu_j / 2) (r(s + 1) - r(s - 1)), mu_j = +-kappa/2 by the
// parity of j / 2 (toward the bin's odd edge), the sub-bin next to the quadrant edge mu = edgeShift, mirrored in the upper half.
// It compensates the extension's bent chains (odd rays assembled from segments along the even rays' slopes). The neighbours are
// the neighbouring threads' own near radiance (workgroup memory): no extra rays, no table.
// 'halo': each workgroup of 64 threads writes 62 slices; threads 0 and 63 only load the neighbours (dispatch ceil(sc / 62)).
const mergeParityMain = (pm: 'wg' | 'direct' | 'sg' | 'halo' | 'none' | 'sghalo') => /*wgsl*/ `
var<workgroup> nearWG: array<array<Rad, 3>, 66>;
fn parityMu(subBin: u32) -> f32 {
  let L = 1u << params.level;
  let upper = subBin >= L;
  let j = select(subBin, 2u * L - 1u - subBin, upper);
  var mu = select(-0.5, 0.5, ((j >> 1u) & 1u) == 0u) * params.kappa;
  if (L > 1u && j == 1u) { mu = params.edgeShift; }
  return select(mu, -mu, upper);
}
fn nearOf(probeIdx: i32, k: i32, s: i32, isEven: bool, numDirections: i32) -> Rad {
  let r = loadRay(probeIdx, k, s);
  if (!isEven) { return r.rad; }
  let e = loadRay(probeIdx + 1, k, s + dirToSliceOffset(k, numDirections));
  return r.rad + e.rad * r.trans;
}
fn contribNear(nearRad: Rad, nearT: f32, isEven: bool, farStep: i32, probeIdx: i32, sliceIdx: i32, subBin: u32, rayIdx: i32) -> Rad {
  let level = params.level;
  let sliceOff = dirToSliceOffset(rayIdx, i32(1u << level));
  let farX = ((probeIdx + farStep) << level) + i32(subBin);
  let farSlice = sliceIdx + sliceOff * farStep;
  var farFluence: Rad;
  if (params.isTopCascade == 1u || farX < 0 || farX >= i32(params.mergeInWidth) || farSlice < 0 || farSlice >= i32(params.sliceCount)) {
    farFluence = loadSkyFluence(subBin);
  } else {
    farFluence = unpackMerge(mergeInBuf[farX * i32(params.sliceCount) + farSlice]);
  }
  let merged = nearRad * getConeArc(subBin) + farFluence * nearT;
  if (isEven) { return mix(merged, loadMerge((probeIdx << level) + i32(subBin), sliceIdx), params.evenWeight); }
  return merged;
}
fn parityLevel0(probeIdx: i32, sliceIdx: i32) {
  let level = params.level;
    let isEven = (probeIdx % 2 == 0);
  let farStep = select(1, 2, isEven);
  let ray0 = loadRay(probeIdx, 0, sliceIdx);
  let ray1 = loadRay(probeIdx, 1, sliceIdx);
  let ext0 = extOf(probeIdx, 0, sliceIdx, isEven, 1);
  let ext1 = extOf(probeIdx, 1, sliceIdx, isEven, 1);
  let result = contribNear(nearOf(probeIdx, 0, sliceIdx, isEven, 1), ray0.trans * ext0.trans, isEven, farStep, probeIdx, sliceIdx, 0u, 0)
             + contribNear(nearOf(probeIdx, 1, sliceIdx, isEven, 1), ray1.trans * ext1.trans, isEven, farStep, probeIdx, sliceIdx, 1u, 1);
  var fc: vec2i;
  switch (params.direction) {
    case 0u: { fc = vec2i(probeIdx - 1, sliceIdx); }
    case 1u: { fc = vec2i(sliceIdx, probeIdx - 1); }
    case 2u: { fc = vec2i(i32(params.fluenceW) - probeIdx, sliceIdx); }
    case 3u: { fc = vec2i(sliceIdx, i32(params.fluenceH) - probeIdx); }
    default: { fc = vec2i(probeIdx - 1, sliceIdx); }
  }
  if (fc.x >= 0 && fc.x < i32(params.fluenceW) && fc.y >= 0 && fc.y < i32(params.fluenceH)) {
    var fi = fc.y * i32(params.fluenceStride) + fc.x;
    if (params.fluenceDirs == 4u) { fi = fi * 4 + i32(params.direction); }
    fluenceBuf[fi] = packMerge(unpackMerge(fluenceBuf[fi]) + result);
  }
}
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lidv: vec3u, @builtin(workgroup_id) wid: vec3u${pm === 'sg' || pm === 'sghalo' ? ', @builtin(subgroup_invocation_id) sgLane: u32, @builtin(subgroup_size) sgSize: u32' : ''}) {
  _ = &readW;
  _ = &rayNext;
  let level = params.level;
  let numDirections = i32(1u << level);
  let l = i32(lidv.x);
  let sliceIdx = ${pm === 'halo' ? 'i32(wid.x) * 62 + l - 1' : pm === 'sghalo' ? 'i32(wid.x) * 60 + (l >> 5u) * 30 + (l & 31) - 1' : 'i32(gid.x)'};
  let x = i32(wid.y);
  let sc = i32(params.sliceCount);
  if (level == 0u) {
    // level 0: no shift (mu = 0), production merge
    let probeIdx = x;
    if (probeIdx < i32(params.levelProbes) && sliceIdx >= 0 && sliceIdx < sc${pm === 'halo' ? ' && l > 0 && l < 63' : pm === 'sghalo' ? ' && (l & 31) > 0 && (l & 31) < 31' : ''}) { parityLevel0(probeIdx, sliceIdx); }
    return;
  }
  let probeIdx = x >> (level - 1u);
  let b0 = 2 * (x & (numDirections / 2 - 1));
  if (probeIdx >= i32(params.levelProbes)) { return; }   // uniform per workgroup
  let isEven = (probeIdx % 2 == 0);
  let farStep = select(1, 2, isEven);
${pm === 'wg' || pm === 'halo' ? `  var tr: array<f32, 3>;
  for (var i = 0; i < 3; i++) {
    let k = b0 + i;
    let r = loadRay(probeIdx, k, sliceIdx);
    var nr = r.rad;
    var t = r.trans;
    if (isEven) {
      let e = loadRay(probeIdx + 1, k, sliceIdx + dirToSliceOffset(k, numDirections));
      nr = r.rad + e.rad * r.trans;
      t = r.trans * e.trans;
    }
    tr[i] = t;
    nearWG[l + 1][i] = nr;
${pm === 'wg' ? `    if (l == 0) { nearWG[0][i] = nearOf(probeIdx, k, sliceIdx - 1, isEven, numDirections); }
    if (l == 63) { nearWG[65][i] = nearOf(probeIdx, k, sliceIdx + 1, isEven, numDirections); }` : ''}
  }
  workgroupBarrier();
  if (sliceIdx >= sc${pm === 'halo' ? ' || l == 0 || l == 63' : ''}) { return; }` : ''}
${[0, 1, 2].map((i) => {
  const c = `    let r${i} = loadRay(probeIdx, b0 + ${i}, sliceIdx);
    var n${i} = r${i}.rad;
    var t${i} = r${i}.trans;
    if (isEven) {
      let e = loadRay(probeIdx + 1, b0 + ${i}, sliceIdx + dirToSliceOffset(b0 + ${i}, numDirections));
      n${i} = r${i}.rad + e.rad * r${i}.trans;
      t${i} = r${i}.trans * e.trans;
    }`;
  if (pm === 'wg' || pm === 'halo') return `  let d${i} = nearWG[l + 2][${i}] - nearWG[l][${i}];
  let n${i} = nearWG[l + 1][${i}];
  let t${i} = tr[${i}];`;
  if (pm === 'sghalo') return `  ${c.trim()}
  let d${i} = subgroupShuffleDown(n${i}, 1u) - subgroupShuffleUp(n${i}, 1u);`;
  if (pm === 'none') return `  ${c.trim()}
  let d${i} = Rad(0.0);`;
  if (pm === 'direct') return `  ${c.trim()}
  let d${i} = nearOf(probeIdx, b0 + ${i}, sliceIdx + 1, isEven, numDirections) - nearOf(probeIdx, b0 + ${i}, sliceIdx - 1, isEven, numDirections);`;
  return `  ${c.trim()}
  var p${i} = subgroupShuffleDown(n${i}, 1u);
  var m${i} = subgroupShuffleUp(n${i}, 1u);
  if (sgLane == sgSize - 1u) { p${i} = nearOf(probeIdx, b0 + ${i}, sliceIdx + 1, isEven, numDirections); }
  if (sgLane == 0u) { m${i} = nearOf(probeIdx, b0 + ${i}, sliceIdx - 1, isEven, numDirections); }
  let d${i} = p${i} - m${i};`;
}).join('\n')}
  let sb = u32(2 * b0);
  let res0 = contribNear(n0 + (0.5 * parityMu(sb)) * d0, t0, isEven, farStep, probeIdx, sliceIdx, sb, b0)
           + contribNear(n1 + (0.5 * parityMu(sb + 1u)) * d1, t1, isEven, farStep, probeIdx, sliceIdx, sb + 1u, b0 + 1);
  let res1 = contribNear(n1 + (0.5 * parityMu(sb + 2u)) * d1, t1, isEven, farStep, probeIdx, sliceIdx, sb + 2u, b0 + 1)
           + contribNear(n2 + (0.5 * parityMu(sb + 3u)) * d2, t2, isEven, farStep, probeIdx, sliceIdx, sb + 3u, b0 + 2);
  if (sliceIdx >= sc${pm === 'sghalo' ? ' || sliceIdx < 0 || (l & 31) == 0 || (l & 31) == 31' : ''}) { return; }
  let outX = (probeIdx << level) + b0;
  mergeOutBuf[outX * sc + sliceIdx] = packMerge(res0);
  mergeOutBuf[(outX + 1) * sc + sliceIdx] = packMerge(res1);
}
`;

const cascadeMergeShader = (mode: ChannelMode, read: MergeReadShape | null = null) =>
  (read?.sg ? 'enable subgroups;\n' : '') +
  wgslChannel[mode] +
  wgslSliceOffset +
  /*wgsl*/ `
struct MergeParams {
  probeCount: u32,
  levelProbes: u32,
  numRays: u32,
  isTopCascade: u32,
  fluenceW: u32,
  fluenceStride: u32,
  skyRow: u32,
  skyShift: u32,
  level: u32,
  coneArcBase: u32,
  sliceCount: u32,
  mergeInWidth: u32,
  direction: u32,
  fluenceH: u32,
  fluenceDirs: u32,
  readBase: u32,
  nextProbes: u32,
  evenWeight: f32,
  // compact rows read (merge-read 4): per branch (odd, even) a_edge, a_int, d_edge, d_int, 1/R
  rpO0: f32, rpO1: f32, rpO2: f32, rpO3: f32, rpO4: f32,
  rpE0: f32, rpE1: f32, rpE2: f32, rpE3: f32, rpE4: f32,
  // parity read (merge-read 5): kappa of this level, shift of the sub-bin next to the quadrant edge
  kappa: f32,
  edgeShift: f32,
};

@group(0) @binding(0) var<storage, read> rayBuf: array<RayPacked>;
@group(0) @binding(1) var<storage, read> mergeInBuf: array<MergePacked>;
@group(0) @binding(2) var<storage, read_write> mergeOutBuf: array<MergePacked>;

@group(0) @binding(3) var<uniform> params: MergeParams;
@group(0) @binding(4) var<storage, read_write> fluenceBuf: array<MergePacked>;
@group(0) @binding(5) var skyPrefixTex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> coneArcs: array<f32>;
${read ? mergeReadDecls(read) : ''}

fn loadRay(probeIdx: i32, rayIdx: i32, sliceIdx: i32) -> RayData {
  let effProbes = i32(params.levelProbes);
  if (probeIdx < 0 || probeIdx >= effProbes ||
      rayIdx < 0 || rayIdx >= i32(params.numRays) ||
      sliceIdx < 0 || sliceIdx >= i32(params.sliceCount)) {
    return RayData(Rad(0.0), 1.0);
  }
  return unpackRay(rayBuf[(probeIdx * i32(params.numRays) + rayIdx) * i32(params.sliceCount) + sliceIdx]);
}

fn loadMerge(texX: i32, sliceIdx: i32) -> Rad {
  if (params.isTopCascade == 1u ||
      texX < 0 || texX >= i32(params.mergeInWidth) ||
      sliceIdx < 0 || sliceIdx >= i32(params.sliceCount)) {
    return Rad(0.0);
  }
  return unpackMerge(mergeInBuf[texX * i32(params.sliceCount) + sliceIdx]);
}

// Paper Eq. 13: A_n(i) = angle(v_n(i+½)) - angle(v_n(i-½))
fn getConeArc(subBin: u32) -> f32 {
  return coneArcs[params.coneArcBase + subBin];
}

fn loadSkyFluence(subBin: u32) -> Rad {
  let base = subBin << params.skyShift;
  let end = base + (1u << params.skyShift);
  let row = i32(params.skyRow);
  return toRad(textureLoad(skyPrefixTex, vec2i(i32(end), row), 0).rgb
             - textureLoad(skyPrefixTex, vec2i(i32(base), row), 0).rgb);
}
fn contrib(ray: RayData, ext: RayData, isEven: bool, farStep: i32, probeIdx: i32, sliceIdx: i32, subBin: u32, rayIdx: i32) -> Rad {
  let level = params.level;
  let weight = getConeArc(subBin);
  let sliceOff = dirToSliceOffset(rayIdx, i32(1u << level));
  let farX = ((probeIdx + farStep) << level) + i32(subBin);
  let farSlice = sliceIdx + sliceOff * farStep;
  var farFluence: Rad;
  if (params.isTopCascade == 1u ||
      farX < 0 || farX >= i32(params.mergeInWidth) ||
      farSlice < 0 || farSlice >= i32(params.sliceCount)) {
    farFluence = loadSkyFluence(subBin);
  } else {
    farFluence = unpackMerge(mergeInBuf[farX * i32(params.sliceCount) + farSlice]);
  }
  if (isEven) {
    let cRad = ray.rad + ext.rad * ray.trans;
    let cTrans = ray.trans * ext.trans;
    let merged = cRad * weight + farFluence * cTrans;
    let coarseFluence = loadMerge((probeIdx << level) + i32(subBin), sliceIdx);
    return mix(merged, coarseFluence, params.evenWeight);
  }
  return ray.rad * weight + farFluence * ray.trans;
}

fn extOf(probeIdx: i32, rayIdx: i32, sliceIdx: i32, isEven: bool, numDirections: i32) -> RayData {
  if (!isEven) { return RayData(Rad(0.0), 1.0); }
  return loadRay(probeIdx + 1, rayIdx, sliceIdx + dirToSliceOffset(rayIdx, numDirections));
}

${read ? (read.parity ? mergeParityMain(read.parity) : mergeReadMain(read)) : ''}
@compute @workgroup_size(64, 1)
fn ${read ? 'mainProduction' : 'main'}(@builtin(global_invocation_id) gid: vec3u) {
  let level = params.level;
  let numDirections = i32(1u << level);
  let sliceIdx = i32(gid.x);
  let x = i32(gid.y);
  let sc = i32(params.sliceCount);
  if (sliceIdx >= sc) { return; }

  if (level == 0u) {
    let probeIdx = x;
    if (probeIdx >= i32(params.levelProbes)) { return; }
    let isEven = (probeIdx % 2 == 0);
    let farStep = select(1, 2, isEven);
    let ray0 = loadRay(probeIdx, 0, sliceIdx);
    let ray1 = loadRay(probeIdx, 1, sliceIdx);
    let result = contrib(ray0, extOf(probeIdx, 0, sliceIdx, isEven, 1), isEven, farStep, probeIdx, sliceIdx, 0u, 0)
               + contrib(ray1, extOf(probeIdx, 1, sliceIdx, isEven, 1), isEven, farStep, probeIdx, sliceIdx, 1u, 1);
    var fc: vec2i;
    switch (params.direction) {
      case 0u: { fc = vec2i(probeIdx - 1, sliceIdx); }
      case 1u: { fc = vec2i(sliceIdx, probeIdx - 1); }
      case 2u: { fc = vec2i(i32(params.fluenceW) - probeIdx, sliceIdx); }
      case 3u: { fc = vec2i(sliceIdx, i32(params.fluenceH) - probeIdx); }
      default: { fc = vec2i(probeIdx - 1, sliceIdx); }
    }
    if (fc.x >= 0 && fc.x < i32(params.fluenceW) && fc.y >= 0 && fc.y < i32(params.fluenceH)) {
      var fi = fc.y * i32(params.fluenceStride) + fc.x;
      if (params.fluenceDirs == 4u) { fi = fi * 4 + i32(params.direction); }   // ratio mode keeps the 4 directions apart
      fluenceBuf[fi] = packMerge(unpackMerge(fluenceBuf[fi]) + result);
    }
    return;
  }

  let pairsPerProbe = numDirections / 2;
  let probeIdx = x >> (level - 1u);
  let w = x & (pairsPerProbe - 1);
  if (probeIdx >= i32(params.levelProbes)) { return; }
  let isEven = (probeIdx % 2 == 0);
  let farStep = select(1, 2, isEven);
  let b0 = 2 * w;
  let ray0 = loadRay(probeIdx, b0, sliceIdx);
  let ray1 = loadRay(probeIdx, b0 + 1, sliceIdx);
  let ray2 = loadRay(probeIdx, b0 + 2, sliceIdx);
  let ext0 = extOf(probeIdx, b0, sliceIdx, isEven, numDirections);
  let ext1 = extOf(probeIdx, b0 + 1, sliceIdx, isEven, numDirections);
  let ext2 = extOf(probeIdx, b0 + 2, sliceIdx, isEven, numDirections);
  let s0 = u32(2 * b0);
  // Quadrant-edge drain (research/tree/notes/merge_read_mechanism.md §7): the sub-bin next to each quadrant edge (1 and 2L-2) reads
  // its ray edgeShift rows inward. Only the first and last pair's workgroups take the branch (uniform per workgroup).
  var ray1a = ray1;
  var ext1a = ext1;
  var ray1b = ray1;
  var ext1b = ext1;
  let lastPair = b0 == numDirections - 2;
  if (params.edgeShift != 0.0 && (b0 == 0 || lastPair)) {
    let h = 0.5 * params.edgeShift;   // weights (-mu/2, 1, mu/2), mu = +edgeShift at sub-bin 1, -edgeShift at 2L-2 (level 1: both here)
    let dr = loadRay(probeIdx, b0 + 1, sliceIdx + 1).rad - loadRay(probeIdx, b0 + 1, sliceIdx - 1).rad;
    var de = Rad(0.0);
    if (isEven) {
      let off = dirToSliceOffset(b0 + 1, numDirections);
      de = loadRay(probeIdx + 1, b0 + 1, sliceIdx + 1 + off).rad - loadRay(probeIdx + 1, b0 + 1, sliceIdx - 1 + off).rad;
    }
    // the even branch composes ray + ext (ext weighted by the centre transmittance in contrib): shift both radiances
    if (b0 == 0) { ray1a = RayData(ray1.rad + h * dr, ray1.trans); ext1a = RayData(ext1.rad + h * de, ext1.trans); }
    if (lastPair) { ray1b = RayData(ray1.rad - h * dr, ray1.trans); ext1b = RayData(ext1.rad - h * de, ext1.trans); }
  }
  let r0 = contrib(ray0, ext0, isEven, farStep, probeIdx, sliceIdx, s0, b0)
         + contrib(ray1a, ext1a, isEven, farStep, probeIdx, sliceIdx, s0 + 1u, b0 + 1);
  let r1 = contrib(ray1b, ext1b, isEven, farStep, probeIdx, sliceIdx, s0 + 2u, b0 + 1)
         + contrib(ray2, ext2, isEven, farStep, probeIdx, sliceIdx, s0 + 3u, b0 + 2);
  let outX = (probeIdx << level) + b0;
  mergeOutBuf[outX * sc + sliceIdx] = packMerge(r0);
  mergeOutBuf[(outX + 1) * sc + sliceIdx] = packMerge(r1);
}
`;

// ── Final blit shader ──
// Bilinearly upscales fluence from probe resolution to screen resolution.
// Uses world texture alpha (opacity) to mask indirect light at surfaces.

const blitShader = /*wgsl*/ `
struct BlitParams { exposure: f32, screenW: f32, screenH: f32, debugMode: f32, falseColor: f32, banded: f32, pad1: f32, pad2: f32 };
const TWO_PI = 6.2831853;

fn acesTonemap(x: vec3f) -> vec3f {
  return clamp(
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
    vec3f(0.0), vec3f(1.0),
  );
}
fn linearToSrgb(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308)); }

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn triangularDither(fragCoord: vec2u) -> vec3f {
  let seed = fragCoord.x + fragCoord.y * 8192u;
  let r0 = f32(pcg(seed)) / 4294967295.0;
  let r1 = f32(pcg(seed + 1u)) / 4294967295.0;
  return vec3f((r0 + r1 - 1.0) / 255.0);
}

fn tonemapAndDither(hdr: vec3f, fragCoord: vec2u) -> vec4f {
  return vec4f(linearToSrgb(acesTonemap(hdr)) + triangularDither(fragCoord), 1.0);
}

// viridis (matplotlib), 17 samples; sRGB values, written straight to the (non-sRGB) canvas
const VIRIDIS = array<vec3f, 17>(
  vec3f(0.2670, 0.0049, 0.3294), vec3f(0.2823, 0.0950, 0.4173), vec3f(0.2788, 0.1755, 0.4834), vec3f(0.2590, 0.2515, 0.5247),
  vec3f(0.2297, 0.3224, 0.5457), vec3f(0.1994, 0.3876, 0.5546), vec3f(0.1727, 0.4488, 0.5579), vec3f(0.1490, 0.5081, 0.5573),
  vec3f(0.1276, 0.5669, 0.5506), vec3f(0.1206, 0.6258, 0.5335), vec3f(0.1579, 0.6838, 0.5017), vec3f(0.2461, 0.7389, 0.4520),
  vec3f(0.3692, 0.7889, 0.3829), vec3f(0.5160, 0.8312, 0.2943), vec3f(0.6785, 0.8637, 0.1895), vec3f(0.8456, 0.8873, 0.0997),
  vec3f(0.9932, 0.9062, 0.1439),
);

fn viridis(t: f32) -> vec3f {
  let s = clamp(t, 0.0, 1.0) * 16.0;
  let i = min(u32(s), 15u);
  return mix(VIRIDIS[i], VIRIDIS[i + 1u], s - f32(i));
}

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let pos = array(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, 1));
  return vec4f(pos[i], 0, 1);
}

@group(0) @binding(0) var fluenceTex: texture_2d<f32>;
@group(0) @binding(1) var worldTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: BlitParams;
@group(0) @binding(3) var linearSamp: sampler;
@group(0) @binding(4) var ptAccumTex: texture_2d<f32>;

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / vec2f(params.screenW, params.screenH);
  let fluence = textureSampleLevel(fluenceTex, linearSamp, uv, 0.0).rgb;
  let world = textureLoad(worldTex, vec2u(pos.xy), 0);
  let emissive = world.rgb * world.a;
  let indirect = fluence / TWO_PI * (1.0 - world.a);
  let dm = i32(params.debugMode);
  let fc = params.falseColor > 0.5;

  var hdr: vec3f;
  var fcSource: vec3f;
  var forceFc = false;

  if (dm == 1) {
    let ptRad = textureLoad(ptAccumTex, vec2i(pos.xy), 0).rgb;
    hdr = ptRad;
    fcSource = ptRad;
  } else if (dm == 2) {
    let ptRad = textureLoad(ptAccumTex, vec2i(pos.xy), 0).rgb;
    hdr = abs((emissive + indirect) - ptRad);
    fcSource = hdr;
    forceFc = true;
  } else {
    hdr = emissive + indirect;
    fcSource = fluence;
  }

  // banded: log10(L · exposure) in 22 bands of 0.1 decade over [-2.2, 0] (research/tree/lab/parity_figs.py banded);
  // opaque cells black, emitters red. In the diff view the bands are of |HRC − PT|.
  if (params.banded > 0.5) {
    if (world.a > 0.5 && dm != 2) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    if (dot(world.rgb, vec3f(1.0)) > 0.0 && dm != 2) { return vec4f(1.0, 0.15, 0.1, 1.0); }
    let src = select(fluence / TWO_PI, hdr, dm != 0);
    let lum = dot(src, vec3f(0.2126, 0.7152, 0.0722)) * params.exposure;
    let band = clamp(floor((log2(max(lum, 1e-12)) * 0.30103 + 2.2) * 10.0), 0.0, 21.0);
    return vec4f(viridis(band / 21.0), 1.0);
  }
  if (fc || forceFc) {
    let mag = dot(fcSource, vec3f(0.2126, 0.7152, 0.0722));
    let t = clamp(log2(mag * 10000.0 + 1.0) / log2(10001.0), 0.0, 1.0);
    return vec4f(viridis(t), 1.0);
  }
  return tonemapAndDither(hdr * params.exposure, vec2u(pos.xy));
}
`;

// ── Fluence cross-blur compute shader (HRC paper Eq. 21) ──
// Opacity-aware 1px cross blur on fluence at probe resolution.
// Kernel [0,1,0; 1,4,1; 0,1,0] / 8, skipping neighbors whose
// opacity differs significantly from the center probe's opacity.
// This cancels the period-2 checkerboard from even/odd merge
// parity without leaking light across surface boundaries.
//
// In mono mode this pass also reconstructs color: the blurred full-res
// luminance Y is combined with the (already blurred) low-res RGB fluence by
// keeping the coarse color's chromaticity and rescaling it to match Y.

const WG_BLUR = [16, 16] as const;

const fluenceBlurShader = (mode: ChannelMode) =>
  wgslChannel[mode] +
  /*wgsl*/ `
struct BlurParams { w: u32, h: u32, stride: u32, screenW: u32, screenH: u32, colorRatio: f32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<storage, read> src: array<MergePacked>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;
@group(0) @binding(3) var worldTex: texture_2d<f32>;

fn load(x: i32, y: i32) -> Rad {
  return unpackMerge(src[y * i32(params.stride) + x]);
}
${
  mode === 'mono'
    ? /*wgsl*/ `
@group(0) @binding(4) var colorTex: texture_2d<f32>;
@group(0) @binding(5) var linearSamp: sampler;

// colorRatio = colorProbes / monoProbes maps a mono probe center onto the
// color probe grid (both grids share the origin and cover the same square).
fn toRGB(Y: f32, x: i32, y: i32) -> vec3f {
  let uv = (vec2f(f32(x), f32(y)) + 0.5) * params.colorRatio / vec2f(textureDimensions(colorTex));
  let rgb = textureSampleLevel(colorTex, linearSamp, uv, 0.0).rgb;
  let coarseY = dot(rgb, LUMA);
  return select(vec3f(Y), rgb * (Y / coarseY), coarseY > 1e-8);
}`
    : /*wgsl*/ `
fn toRGB(c: vec3f, x: i32, y: i32) -> vec3f { return c; }`
}

fn probeOpacity(px: i32, py: i32, spacing: f32) -> f32 {
  let wp = vec2i(vec2f(f32(px) + 0.5, f32(py) + 0.5) * spacing);
  return textureLoad(worldTex, wp, 0).a;
}

@compute @workgroup_size(${WG_BLUR[0]}, ${WG_BLUR[1]})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(params.w);
  let h = i32(params.h);
  if (x >= w || y >= h) { return; }

  let spacing = f32(params.screenW) / f32(w);
  let centerOpacity = probeOpacity(x, y, spacing);
  let center = load(x, y);

  var sum = center * 4.0;
  var wt = 4.0;

  let off = array<vec2i, 4>(vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
  for (var i = 0; i < 4; i++) {
    let nx = clamp(x + off[i].x, 0, w - 1);
    let ny = clamp(y + off[i].y, 0, h - 1);
    let nOpacity = probeOpacity(nx, ny, spacing);
    if (abs(nOpacity - centerOpacity) < 0.5) {
      sum += load(nx, ny);
      wt += 1.0;
    }
  }

  textureStore(dst, vec2i(x, y), vec4f(toRGB(sum / wt, x, y), 1.0));
}
`;


// ── Ratio (tier 1, NOTES §44–§47): exact free-space transport × cascade visibility ──
//
// F(p) = Σ_dir F_free,dir(p) · blur(F_occ,dir)(p) / blur(F_freeH,dir)(p)
//   F_free,dir  exact unoccluded fluence from the direction's 90° quadrant (closed-form kernel, §47.1)
//   F_occ,dir   the cascade on the scene              (per-direction level-0 fluence)
//   F_freeH,dir the cascade on the scene with every cell transparent
// Light model: volumetric emitters (opacity 0, emission per unit length). Opaque emitters are seen
// by the free channel as volumetric ones; the ratio absorbs the difference approximately.

// Emitter compaction: one entry per probe cell with non-zero emission (world at the probe centre + bounce).
const freeEmitShader = /*wgsl*/ `
struct EmitParams { probesX: u32, probesY: u32, screenW: f32, screenH: f32, spacing: f32, capacity: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var worldTex: texture_2d<f32>;
@group(0) @binding(1) var bounceTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> emitters: array<vec4f>;   // 2 entries per emitter: (x, y, 0, 0), (r, g, b, 0)
@group(0) @binding(4) var<uniform> params: EmitParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (p.x >= i32(params.probesX) || p.y >= i32(params.probesY)) { return; }
  let wp = (vec2f(p) + 0.5) * params.spacing;
  let px = vec2i(floor(wp));
  if (px.x < 0 || px.y < 0 || px.x >= i32(params.screenW) || px.y >= i32(params.screenH)) { return; }
  let e = textureLoad(worldTex, px, 0).rgb + textureLoad(bounceTex, p, 0).rgb;
  if (e.r <= 0.0 && e.g <= 0.0 && e.b <= 0.0) { return; }
  let i = atomicAdd(&counter, 1u);
  if (i >= params.capacity) { return; }
  emitters[2u * i] = vec4f(vec2f(p), 0.0, 0.0);
  emitters[2u * i + 1u] = vec4f(e, 0.0);
}
`;

// Closed-form fluence kernel of a unit-square volumetric emitter (NOTES §47.1), with a multipole far field
// (1/r + 1/(24 r³)) beyond 8 px where the closed form cancels catastrophically in f32.
const wgslFreeKernel = /*wgsl*/ `
fn kG(u: f32, v: f32) -> f32 {
  let r = sqrt(u * u + v * v);
  var g = 0.0;
  if (abs(u) > 1e-6) { g += u * log(max(v + r, 1e-30)); }
  if (abs(v) > 1e-6) { g += v * log(max(u + r, 1e-30)); }
  return g;
}
fn freeKernel(x: f32, y: f32) -> f32 {
  let r2 = x * x + y * y;
  if (r2 > 64.0) { let r = sqrt(r2); return 1.0 / r + 1.0 / (24.0 * r2 * r); }
  return kG(x + 0.5, y + 0.5) - kG(x - 0.5, y + 0.5) - kG(x + 0.5, y - 0.5) + kG(x - 0.5, y - 0.5);
}
// Quadrant weights of the offset (dx, dy) = emitter − receiver for the 4 cascade directions E, N, W, S
// (production receiver convention: the emitter's own column/row is excluded, DX ≥ 1; diagonals split ½/½;
// the receiver's own cell is split ¼ each so the exact self-fluence is kept).
fn quadrantWeights(dx: i32, dy: i32) -> vec4f {
  if (dx == 0 && dy == 0) { return vec4f(0.25); }
  let ax = abs(dx); let ay = abs(dy);
  let diag = select(1.0, 0.5, ax == ay);
  var w = vec4f(0.0);
  if (dx >= 1 && ay <= dx) { w[0] = diag; }
  if (dy >= 1 && ax <= dy) { w[1] = diag; }
  if (dx <= -1 && ay <= -dx) { w[2] = diag; }
  if (dy <= -1 && ax <= -dy) { w[3] = diag; }
  return w;
}
`;

// Direct-sum free field (prototype, O(pixels × emitters) — scene dependent; the FFT path replaces it).
const freeFieldDirectShader = (mode: ChannelMode) =>
  wgslChannel[mode] +
  wgslFreeKernel +
  /*wgsl*/ `
struct FreeParams { probesX: u32, probesY: u32, stride: u32, capacity: u32, spacing: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> counter: u32;
@group(0) @binding(1) var<storage, read> emitters: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> freeField: array<MergePacked>;   // (y * stride + x) * 4 + dir
@group(0) @binding(3) var<uniform> params: FreeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (p.x >= i32(params.probesX) || p.y >= i32(params.probesY)) { return; }
  let n = min(counter, params.capacity);
  var acc = array<vec3f, 4>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for (var i = 0u; i < n; i++) {
    let q = vec2i(emitters[2u * i].xy);
    let e = emitters[2u * i + 1u].rgb;
    let d = q - p;
    let k = freeKernel(f32(d.x), f32(d.y)) * params.spacing;   // emission per unit world length
    let w = quadrantWeights(d.x, d.y);
    for (var dir = 0u; dir < 4u; dir++) { acc[dir] += e * (k * w[dir]); }
  }
  let base = (u32(p.y) * params.stride + u32(p.x)) * 4u;
  for (var dir = 0u; dir < 4u; dir++) { freeField[base + dir] = packMerge(toRad(acc[dir])); }
}
`;


// ── FFT free field (scene-independent cost): F_dir = E ∗ Kc_dir on a 2× zero-padded grid ──
// One workgroup transforms one row (axis 0) or column (axis 1) of the P×P complex grid in workgroup memory.
// mode 0: copy src; 1: load the emission image (pack 0 → (r, g), pack 1 → (b, 0)); 2: src × kernel spectrum of
// direction `dir` (E = K̂, W = conj K̂, N = K̂ᵀ, S = conj K̂ᵀ — the quadrant kernels are reflections of one another);
// 3: load the East convolution kernel image. `inverse` flips the twiddle sign; `scale` is applied on store.
const fftShader = (P: number) => {
  const logP = Math.log2(P);
  return (
    wgslFreeKernel +
    /*wgsl*/ `
struct FFTParams { axis: u32, mode: u32, dir: u32, pack: u32, inverse: u32, coarseX: u32, coarseY: u32, D: u32, scale: f32, cutR: f32, cutRho: f32, sig2: f32 };
@group(0) @binding(0) var<storage, read> src: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2f>;
@group(0) @binding(2) var<storage, read> kspec: array<vec2f>;
@group(0) @binding(3) var<uniform> params: FFTParams;
@group(0) @binding(4) var emissionTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> twiddle: array<vec2f>;   // exp(−2πi j / P), j < P/2

const P = ${P}u;
const LOGP = ${logP}u;
const WG = 256u;
var<workgroup> sh: array<vec2f, ${P}>;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn idx(line: u32, i: u32) -> u32 { return select(line * P + i, i * P + line, params.axis == 1u); }

// coarse cell (x, y) = 2× tent splat of the fine emission (cells 2x−1 … 2x+2, weights ¼ ¾ ¾ ¼): emission-preserving
fn emission(x: u32, y: u32) -> vec3f {
  if (x >= params.coarseX || y >= params.coarseY) { return vec3f(0.0); }
  let dim = vec2i(textureDimensions(emissionTex));
  let w = array<f32, 4>(0.25, 0.75, 0.75, 0.25);
  var e = vec3f(0.0);
  for (var j = 0; j < 4; j++) {
    let fy = 2 * i32(y) - 1 + j;
    if (fy < 0 || fy >= dim.y) { continue; }
    for (var i = 0; i < 4; i++) {
      let fx = 2 * i32(x) - 1 + i;
      if (fx < 0 || fx >= dim.x) { continue; }
      e += textureLoad(emissionTex, vec2i(fx, fy), 0).rgb * (w[i] * w[j]);
    }
  }
  return e;
}
fn cutFar(r: f32) -> f32 {
  let t = clamp((r - params.cutR * (1.0 - params.cutRho)) / (2.0 * params.cutRho * params.cutR), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn load(line: u32, i: u32) -> vec2f {
  switch (params.mode) {
    case 1u: {
      let x = select(i, line, params.axis == 1u);
      let y = select(line, i, params.axis == 1u);
      let e = emission(x, y);
      return select(vec2f(e.b, 0.0), vec2f(e.r, e.g), params.pack == 0u);
    }
    case 2u: {
      let kx = select(i, line, params.axis == 1u);
      let ky = select(line, i, params.axis == 1u);
      let transposed = (params.dir & 1u) == 1u;
      var k = kspec[select(ky * P + kx, kx * P + ky, transposed)];
      if (params.dir >= 2u) { k.y = -k.y; }
      return cmul(src[idx(line, i)], k);
    }
    case 3u: {
      let x = select(i, line, params.axis == 1u);
      let y = select(line, i, params.axis == 1u);
      // signed coarse offsets on the periodic grid; far convolution kernel Kc_E(d) = K(D·d) · w_E(−d) · cut(|D·d|)
      let dx = i32(x) - select(0, i32(P), x >= P / 2u);
      let dy = i32(y) - select(0, i32(P), y >= P / 2u);
      let w = quadrantWeights(-dx, -dy)[0];
      let fx = f32(dx * i32(params.D));
      let fy = f32(dy * i32(params.D));
      let r = length(vec2f(fx, fy));
      let c = cutFar(r);
      // −(σ²/2)∇²K ≈ −(σ²/2)/r³ cancels the second-order smoothing of the tent splat + bilinear gather (σ² = 1.5 px²)
      let k = freeKernel(fx, fy) - (params.sig2 * 0.5) / max(r * r * r, 1e-9);
      return vec2f(select(0.0, k * w * c, w > 0.0 && c > 0.0), 0.0);
    }
    default: { return src[idx(line, i)]; }
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let line = wid.x;
  for (var i = lid.x; i < P; i += WG) { sh[reverseBits(i) >> (32u - LOGP)] = load(line, i); }
  workgroupBarrier();
  let sign = select(1.0, -1.0, params.inverse == 1u);
  for (var s = 1u; s <= LOGP; s++) {
    let m = 1u << s;
    let half = m >> 1u;
    for (var t = lid.x; t < P / 2u; t += WG) {
      let k = t & (half - 1u);
      let base = (t / half) * m;
      var w = twiddle[k * (P / m)];
      w.y *= sign;
      let a = sh[base + k];
      let b = cmul(sh[base + k + half], w);
      sh[base + k] = a + b;
      sh[base + k + half] = a - b;
    }
    workgroupBarrier();
  }
  for (var i = lid.x; i < P; i += WG) { dst[idx(line, i)] = sh[i] * params.scale; }
}
`
  );
};

// Transport world for the ratio (NOTES §50.2, §51): an opaque emitter becomes a volumetric body of the same total
// emission — emission per unit length ρ = E·P/(πA) (Cauchy: mean chord = πA/P), with P/A = 2(1/c_x + 1/c_y) from the
// body's horizontal and vertical run lengths through the pixel (exact for rectangles, close for discs) — so the scene
// cascade, the free channel, the exact free field and the path tracer see one convolution-compatible model. The
// earlier 1-px shell was wrong (3× too bright, and its rim rows are "light tubes" for the line lattice). Display
// keeps the original world. Bodies no longer block light (model error of this proxy, ±17 % angular, §50.2).
const transportWorldShader = /*wgsl*/ `
@group(0) @binding(0) var worldTex: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
const MAXRUN = 96;
fn isBody(p: vec2i, dim: vec2i) -> bool {
  if (p.x < 0 || p.y < 0 || p.x >= dim.x || p.y >= dim.y) { return false; }
  let w = textureLoad(worldTex, p, 0);
  return w.a >= 0.99 && (w.r > 0.0 || w.g > 0.0 || w.b > 0.0);
}
fn run(p: vec2i, step: vec2i, dim: vec2i) -> i32 {
  var n = 0;
  for (var i = 1; i <= MAXRUN; i++) {
    if (!isBody(p + step * i, dim)) { break; }
    n++;
  }
  return n;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = vec2i(textureDimensions(worldTex));
  let p = vec2i(gid.xy);
  if (p.x >= dim.x || p.y >= dim.y) { return; }
  var w = textureLoad(worldTex, p, 0);
  if (isBody(p, dim)) {
    let cx = f32(1 + run(p, vec2i(1, 0), dim) + run(p, vec2i(-1, 0), dim));
    let cy = f32(1 + run(p, vec2i(0, 1), dim) + run(p, vec2i(0, -1), dim));
    let rho = 2.0 * (1.0 / cx + 1.0 / cy) / 3.14159265;
    w = vec4f(w.rgb * rho, 0.0);
  }
  textureStore(dst, p, w);
}
`;

// Emission image at probe resolution (world sampled at probe centres + bounce), the input of the free field.
const emissionImageShader = /*wgsl*/ `
struct EmParams { probesX: u32, probesY: u32, screenW: f32, screenH: f32, spacing: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var worldTex: texture_2d<f32>;
@group(0) @binding(1) var bounceTex: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: EmParams;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (p.x >= i32(params.probesX) || p.y >= i32(params.probesY)) { return; }
  let wp = (vec2f(p) + 0.5) * params.spacing;
  let px = vec2i(floor(wp));
  var e = vec3f(0.0);
  if (px.x >= 0 && px.y >= 0 && px.x < i32(params.screenW) && px.y < i32(params.screenH)) {
    let w = textureLoad(worldTex, px, 0);
    // emission per unit length as the scene cascade would see it unoccluded: E for a volumetric light (α = 0),
    // E·α for an emissive medium (first order of E·(1 − (1−α)^len)); opaque emitters were shelled upstream
    let ePerLen = w.rgb * select(w.a, 1.0, w.a < 1e-6);
    e = (ePerLen + textureLoad(bounceTex, p, 0).rgb) * params.spacing;
  }
  textureStore(dst, p, vec4f(e, 1.0));
}
`;

// Near field (E77b): per receiver phase (p mod 2), tap weights = exact K·w_dir − what the coarse far path (tent
// splat → coarse kernel → bilinear gather) already contributes for that emitter, so near + far is exact inside the
// window. One dispatch per phase: a workgroup covers 8×8 same-phase receivers (a 16×16 fine block) and stages the
// block plus apron of the emission image in workgroup memory (f16 rgb).
const NEAR_W = 12;
const NEAR_MAX_TAPS = 1024; // per phase (all 4 directions)
const freeNearShader = (mode: ChannelMode) =>
  wgslChannel[mode] +
  /*wgsl*/ `
struct NearParams { probesX: u32, probesY: u32, stride: u32, phase: u32, tapBase: vec4u, tapEnd: vec4u };
@group(0) @binding(0) var emissionTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> taps: array<vec4f>;   // (dx, dy, weight, dir), grouped by phase then direction
@group(0) @binding(2) var<storage, read_write> freeField: array<MergePacked>;
@group(0) @binding(3) var<uniform> params: NearParams;

// One dispatch per receiver phase; a thread handles one receiver (fine pixel 2·gid + phase). Plain texture loads:
// a workgroup-memory tile was tried and lost (occupancy), see NOTES §48.6.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = vec2i(i32(params.probesX), i32(params.probesY));
  let p = vec2i(gid.xy) * 2 + vec2i(i32(params.phase & 1u), i32(params.phase >> 1u));
  if (p.x >= dim.x || p.y >= dim.y) { return; }
  let base = (u32(p.y) * params.stride + u32(p.x)) * 4u;
  for (var dir = 0u; dir < 4u; dir++) {
    var a = vec3f(0.0);
    for (var t = params.tapBase[dir]; t < params.tapEnd[dir]; t++) {
      let tp = taps[t];
      let q = p + vec2i(tp.xy);
      if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) { continue; }
      a += textureLoad(emissionTex, q, 0).rgb * tp.z;
    }
    // the far field is already there (FFT gather); add the signed near correction and pack once
    freeField[base + dir] = packMerge(toRad(max(vec3f(unpackMerge(freeField[base + dir])) + a, vec3f(0.0))));
  }
}
`;

// Gather an inverse transform into the packed per-direction free field: pack 0 writes (r, g), pack 1 adds b.
const fftGatherShader = (mode: ChannelMode) =>
  wgslChannel[mode] +
  /*wgsl*/ `
struct GatherParams { probesX: u32, probesY: u32, stride: u32, P: u32, dir: u32, pack: u32, coarseX: u32, coarseY: u32 };
@group(0) @binding(0) var<storage, read> work: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> freeField: array<MergePacked>;
@group(0) @binding(2) var<uniform> params: GatherParams;

fn coarse(cx: i32, cy: i32) -> vec2f {
  let x = clamp(cx, 0, i32(params.coarseX) - 1);
  let y = clamp(cy, 0, i32(params.coarseY) - 1);
  return work[u32(y) * params.P + u32(x)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.probesX || gid.y >= params.probesY) { return; }
  // bilinear gather at the fine centre: coarse coordinate (p + ½)/2 − ½
  let cc = (vec2f(gid.xy) + 0.5) * 0.5 - 0.5;
  let c0 = vec2i(floor(cc));
  let f = cc - vec2f(c0);
  let v = max(mix(mix(coarse(c0.x, c0.y), coarse(c0.x + 1, c0.y), f.x), mix(coarse(c0.x, c0.y + 1), coarse(c0.x + 1, c0.y + 1), f.x), f.y), vec2f(0.0));
  let i = (gid.y * params.stride + gid.x) * 4u + params.dir;
  if (params.pack == 0u) {
    freeField[i] = packMerge(toRad(vec3f(v.x, v.y, 0.0)));
  } else {
    let prev = vec3f(unpackMerge(freeField[i]));
    freeField[i] = packMerge(toRad(vec3f(prev.r, prev.g, v.x)));
  }
}
`;

// Combine: per direction V = blur(F_occ)/blur(F_freeH) (the opacity-aware cross blur of the production
// pipeline on both), output F = Σ_dir (F_free,dir + sky_dir) · V_dir into the fluence texture.
// view: 0 = F (the method), 1 = Σ F_free (exact free field), 2 = mean V, 3 = Σ blur(F_occ) (= production HRC),
//       4 = Σ blur(F_freeH) (the cascade's own free estimate).
const combineShader = (mode: ChannelMode) =>
  wgslChannel[mode] +
  /*wgsl*/ `
struct CombineParams { w: u32, h: u32, stride: u32, screenW: u32, screenH: u32, view: u32, pad0: u32, pad1: u32, sky: array<vec4f, 4> };
@group(0) @binding(0) var<storage, read> occ: array<MergePacked>;
@group(0) @binding(1) var<storage, read> freeH: array<MergePacked>;
@group(0) @binding(2) var<storage, read> freeField: array<MergePacked>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: CombineParams;
@group(0) @binding(5) var worldTex: texture_2d<f32>;

fn probeOpacity(px: i32, py: i32, spacing: f32) -> f32 {
  let wp = vec2i(vec2f(f32(px) + 0.5, f32(py) + 0.5) * spacing);
  return textureLoad(worldTex, wp, 0).a;
}
fn toRGB(r: Rad) -> vec3f { return vec3f(r); }

@compute @workgroup_size(${WG_BLUR[0]}, ${WG_BLUR[1]})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(params.w);
  let h = i32(params.h);
  if (x >= w || y >= h) { return; }
  let spacing = f32(params.screenW) / f32(w);
  let centerOpacity = probeOpacity(x, y, spacing);
  // cross-blur taps (production fluenceBlurShader): centre ×4, 4 neighbours of similar opacity ×1
  var taps = array<i32, 5>(y * i32(params.stride) + x, -1, -1, -1, -1);
  var wt = 4.0;
  let off = array<vec2i, 4>(vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
  for (var i = 0; i < 4; i++) {
    let nx = clamp(x + off[i].x, 0, w - 1);
    let ny = clamp(y + off[i].y, 0, h - 1);
    if (abs(probeOpacity(nx, ny, spacing) - centerOpacity) < 0.5) { taps[i + 1] = ny * i32(params.stride) + nx; wt += 1.0; }
  }
  var out = vec3f(0.0);
  var vsum = vec3f(0.0);
  let own = (y * i32(params.stride) + x) * 4;
  var nums: array<vec3f, 4>;
  var dens: array<vec3f, 4>;
  var numT = vec3f(0.0);
  var denT = vec3f(0.0);
  for (var dir = 0; dir < 4; dir++) {
    var num = toRGB(unpackMerge(occ[taps[0] * 4 + dir])) * 4.0;
    var den = toRGB(unpackMerge(freeH[taps[0] * 4 + dir])) * 4.0;
    for (var i = 1; i < 5; i++) {
      if (taps[i] >= 0) {
        num += toRGB(unpackMerge(occ[taps[i] * 4 + dir]));
        den += toRGB(unpackMerge(freeH[taps[i] * 4 + dir]));
      }
    }
    nums[dir] = num / wt; dens[dir] = den / wt; numT += nums[dir]; denT += dens[dir];
  }
  // total visibility: the fallback for a quadrant the cascade sees as (nearly) dark, where the coarse far
  // field's quadrant split may still have put a little energy (NOTES §48.6)
  let vT = select(vec3f(0.0), numT / max(denT, vec3f(1e-7)), denT > vec3f(1e-7));
  for (var dir = 0; dir < 4; dir++) {
    let num = nums[dir];
    let den = dens[dir];
    let ff = toRGB(unpackMerge(freeField[own + dir])) + params.sky[dir].rgb;
    let v = select(vT, num / max(den, vec3f(1e-7)), den > 1e-3 * denT);
    switch (params.view) {
      case 1u: { out += ff; }
      case 2u: { vsum += v; }
      case 3u: { out += num; }
      case 4u: { out += den; }
      default: { out += ff * v; }
    }
  }
  if (params.view == 2u) { out = vsum * 0.25; }
  textureStore(dst, vec2i(x, y), vec4f(out, 1.0));
}
`;

// ── Bounce compute shader ──
//
// Computes diffuse re-emission at probe resolution (ps × ps). Each frame,
// the previous frame's fluence is read and converted to bounce emission
// that feeds back into the cascade via the seed shaders.
//
// Key design: wall probes (opacity≈1) sample fluence from exterior cardinal
// neighbors rather than their own position, because the cascade's fluence
// inside solid objects is zero. This solves the probe-surface misalignment
// problem where some of the probes used for the interpolation will likely
// be inside the object, where no light arrives.
//
// The blend `ownFluence * (1-opacity) + exteriorFluence * opacity` is
// continuous: vacuum probes use their own fluence, wall probes use exterior
// neighbors', and deep interior probes (no exterior neighbors) get zero.

const bounceComputeShader = /*wgsl*/ `
struct BounceParams {
  screenW: u32,
  screenH: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var prevFluence: texture_2d<f32>;
@group(0) @binding(1) var worldTex: texture_2d<f32>;
@group(0) @binding(2) var materialTex: texture_2d<f32>;
@group(0) @binding(3) var bounceOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BounceParams;

@compute @workgroup_size(${WG_BOUNCE[0]}, ${WG_BOUNCE[1]})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let probe = vec2i(i32(gid.x), i32(gid.y));
  let pdim = vec2i(textureDimensions(bounceOut));
  if (probe.x >= pdim.x || probe.y >= pdim.y) { return; }

  let spacing = f32(params.screenW) / f32(pdim.x);
  let worldPos = vec2i((vec2f(probe) + 0.5) * spacing);

  let opacity = textureLoad(worldTex, worldPos, 0).a;
  let albedo = textureLoad(materialTex, worldPos, 0).r;
  if (albedo == 0.0) {
    textureStore(bounceOut, probe, vec4f(0.0));
    return;
  }

  let ownFluence = textureLoad(prevFluence, probe, 0).rgb;

  var extFluence = vec3f(0.0);
  var extWeight = 0.0;
  let pMax = pdim - 1;
  let off = array<vec2i, 4>(vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
  for (var i = 0; i < 4; i++) {
    let d = off[i];
    let ni = clamp(probe + d, vec2i(0), pMax);
    let ni2 = clamp(probe + d * 2, vec2i(0), pMax);
    let nOpacity = textureLoad(worldTex, vec2i((vec2f(ni) + 0.5) * spacing), 0).a;
    let nFluence = (textureLoad(prevFluence, ni, 0).rgb
                  + textureLoad(prevFluence, ni2, 0).rgb) * 0.5;
    let w = 1.0 - nOpacity;
    extFluence += nFluence * w;
    extWeight += w;
  }
  if (extWeight > 0.001) {
    extFluence /= extWeight;
  }

  // Bounce emission = fluence × ω₀ / 2π (isotropic re-emission).
  let fluence = ownFluence * (1.0 - opacity) + extFluence * opacity;
  const TWO_PI = 6.2831853;
  textureStore(bounceOut, probe, vec4f(fluence * albedo / TWO_PI, 0.0));
}
`;

// ── 2D Path Tracer (ground truth reference) ──
//
// Progressive Monte Carlo path tracer at screen resolution. Each frame adds
// N stratified samples per pixel, blended into the accumulation buffer.
//
// Uses Amanatides & Woo DDA to walk every grid cell the ray crosses,
// computing Beer-Lambert with the exact Euclidean path length through
// each cell. This eliminates the directional bias of unit-step marching
// where diagonal rays traversed more cells than axis-aligned ones.
//
// Transport model — deterministic emission + stochastic interaction:
//   At each pixel:
//     1. Deterministic emission: radiance += throughput × emission × interactProb
//        where interactProb = 1 − pow(1−opacity, pathLen)
//     2. Stochastic interaction (three-way, bounces on):
//        - pass-through (prob 1−interactProb): continue, throughput unchanged
//        - scatter (prob interactProb×albedo): new direction, throughput unchanged
//        - absorb (prob interactProb×(1−albedo)): ray terminates
//     3. Deterministic extinction (bounces off):
//        throughput *= pow(1 − opacity, pathLen)

const pathTraceShader = /*wgsl*/ `
struct PTParams {
  screenW: u32,
  screenH: u32,
  frameIndex: u32,
  samplesPerPixel: u32,
  maxBounces: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var worldTex: texture_2d<f32>;
@group(0) @binding(1) var materialTex: texture_2d<f32>;
@group(0) @binding(2) var accumTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> params: PTParams;
@group(0) @binding(5) var skyTex: texture_2d<f32>;

fn sampleSky(dir: vec2f) -> vec3f {
  let angle = atan2(dir.y, dir.x);
  let u = (angle + 3.14159265) / 6.28318530;
  let skyW = f32(textureDimensions(skyTex, 0).x);
  let skyI = clamp(i32(u * skyW), 0, i32(skyW) - 1);
  return textureLoad(skyTex, vec2i(skyI, 0), 0).rgb;
}

fn pcgHash(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn randomFloat(seed: ptr<function, u32>) -> f32 {
  *seed = pcgHash(*seed);
  return f32(*seed) / 4294967295.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = i32(gid.x);
  let py = i32(gid.y);
  if (px >= i32(params.screenW) || py >= i32(params.screenH)) { return; }

  var seed = (u32(px) + u32(py) * params.screenW) * 1099u + params.frameIndex * 6971u;

  var sampleSum = vec3f(0.0);

  for (var s = 0u; s < params.samplesPerPixel; s++) {
    var rayPos = vec2f(f32(px) + 0.5, f32(py) + 0.5);
    let sectorSize = 6.2831853 / f32(params.samplesPerPixel);
    let angle = (f32(s) + randomFloat(&seed)) * sectorSize;
    var rayDir = vec2f(cos(angle), sin(angle));
    var throughput = 1.0;
    var radiance = vec3f(0.0);
    var surfaceEntry = rayPos;
    var inSurface = false;
    var bounceCount = 0u;

    // Camera pixel: deterministic emission + extinction to match HRC blit
    {
      let w0 = textureLoad(worldTex, vec2i(px, py), 0);
      radiance += throughput * w0.rgb * w0.a;
      // volumetric emitter at the camera pixel: its own half-chord (centre to the cell boundary along the ray)
      if (w0.a < 1e-6) { radiance += w0.rgb * 0.5 / max(abs(rayDir.x), abs(rayDir.y)); }
      throughput *= (1.0 - w0.a);
    }

    let sw = i32(params.screenW);
    let sh = i32(params.screenH);

    var rayOrigin = rayPos;
    var cell = vec2i(i32(floor(rayPos.x)), i32(floor(rayPos.y)));
    var absDX = abs(rayDir.x);
    var absDY = abs(rayDir.y);
    var sX = select(-1, 1, rayDir.x >= 0.0);
    var sY = select(-1, 1, rayDir.y >= 0.0);
    var tDX = select(1e20, 1.0 / absDX, absDX > 1e-12);
    var tDY = select(1e20, 1.0 / absDY, absDY > 1e-12);
    var tMX: f32; var tMY: f32;
    if (absDX > 1e-12) { tMX = (select(f32(cell.x), f32(cell.x + 1), rayDir.x >= 0.0) - rayOrigin.x) / rayDir.x; } else { tMX = 1e20; }
    if (absDY > 1e-12) { tMY = (select(f32(cell.y), f32(cell.y + 1), rayDir.y >= 0.0) - rayOrigin.y) / rayDir.y; } else { tMY = 1e20; }
    var tPrev = 0.0;

    // Advance past camera pixel (already handled above)
    tPrev = min(tMX, tMY);
    if (tMX < tMY) { cell.x += sX; tMX += tDX; } else { cell.y += sY; tMY += tDY; }

    for (var step = 0u; step < 8192u; step++) {
      if (throughput < 1e-6) { break; }
      if (cell.x < 0 || cell.y < 0 || cell.x >= sw || cell.y >= sh) {
        radiance += throughput * sampleSky(rayDir);
        break;
      }

      let tExit = min(tMX, tMY);
      let pathLen = tExit - tPrev;
      let w = textureLoad(worldTex, cell, 0);
      let opacity = w.a;

      if (opacity < 1e-6 || pathLen <= 0.0) {
        inSurface = false;
        radiance += throughput * w.rgb * pathLen;   // volumetric emitter: emission per unit length (matches the seed rule)
      } else {
        if (!inSurface) {
          surfaceEntry = rayOrigin + rayDir * tPrev;
          inSurface = true;
        }

        let stepTrans = pow(1.0 - opacity, pathLen);
        let interactProb = 1.0 - stepTrans;

        radiance += throughput * w.rgb * interactProb;

        if (params.maxBounces > 0u) {
          let albedo = textureLoad(materialTex, cell, 0).r;
          let r = randomFloat(&seed);
          if (r < interactProb * (1.0 - albedo)) {
            break;
          } else if (r < interactProb) {
            if (bounceCount >= params.maxBounces) { break; }
            bounceCount++;
            if (opacity > 0.9) {
              rayPos = surfaceEntry;
              inSurface = false;
            } else {
              rayPos = rayOrigin + rayDir * tPrev;
            }
            let newAngle = randomFloat(&seed) * 6.2831853;
            rayDir = vec2f(cos(newAngle), sin(newAngle));

            rayOrigin = rayPos;
            cell = vec2i(i32(floor(rayPos.x)), i32(floor(rayPos.y)));
            absDX = abs(rayDir.x); absDY = abs(rayDir.y);
            sX = select(-1, 1, rayDir.x >= 0.0); sY = select(-1, 1, rayDir.y >= 0.0);
            tDX = select(1e20, 1.0 / absDX, absDX > 1e-12);
            tDY = select(1e20, 1.0 / absDY, absDY > 1e-12);
            if (absDX > 1e-12) { tMX = (select(f32(cell.x), f32(cell.x + 1), rayDir.x >= 0.0) - rayOrigin.x) / rayDir.x; } else { tMX = 1e20; }
            if (absDY > 1e-12) { tMY = (select(f32(cell.y), f32(cell.y + 1), rayDir.y >= 0.0) - rayOrigin.y) / rayDir.y; } else { tMY = 1e20; }
            tPrev = min(tMX, tMY);
            if (tMX < tMY) { cell.x += sX; tMX += tDX; } else { cell.y += sY; tMY += tDY; }
            continue;
          }
        } else {
          throughput *= stepTrans;
          if (throughput < 1e-6) { break; }
        }
      }

      tPrev = tExit;
      if (tMX < tMY) { cell.x += sX; tMX += tDX; } else { cell.y += sY; tMY += tDY; }
    }

    sampleSum += radiance;
  }

  let newSample = sampleSum / f32(params.samplesPerPixel);

  let coord = vec2i(px, py);
  if (params.frameIndex == 0u) {
    textureStore(outTex, coord, vec4f(newSample, 1.0));
  } else {
    let prev = textureLoad(accumTex, coord, 0).rgb;
    let weight = 1.0 / f32(params.frameIndex + 1u);
    let blended = mix(prev, newSample, weight);
    textureStore(outTex, coord, vec4f(blended, 1.0));
  }
}
`;

// ── Exact reference (deterministic angular quadrature, bounces off) ──
//
// The path tracer's transport model without its randomness: per pixel, the mean over angle of the radiance arriving at
// the pixel centre along the DDA chord model (the same rule as pathTraceShader with maxBounces 0). The angle is
// sampled uniformly at M directions; each interval [θk, θk+1] is integrated by the trapezoid rule, and where the two
// rays end on different cells AND their radiance jumps (> 1 %), the jump is located by bisection (depth steps) and both
// sides are integrated separately: the research reference's rule (research/rc-theory/e64_volumetric.py ref_vol_sector).
// Work is split into chunks of C consecutive intervals visited in a scrambled order and into row bands, so each
// dispatch is short and the partial image (rescaled by intervals done) refines in place.
const exactShader = /*wgsl*/ `
struct ExactParams { w: u32, h: u32, m: u32, chunkLen: u32, chunk: u32, rowStart: u32, rowCount: u32, depth: u32, scale: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var worldTex: texture_2d<f32>;
@group(0) @binding(1) var skyTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> params: ExactParams;

fn sampleSky(dir: vec2f) -> vec3f {
  let u = (atan2(dir.y, dir.x) + 3.14159265) / 6.28318530;
  let skyW = f32(textureDimensions(skyTex, 0).x);
  return textureLoad(skyTex, vec2i(clamp(i32(u * skyW), 0, i32(skyW) - 1), 0), 0).rgb;
}

struct Hit { rad: vec3f, id: i32 };

fn trace(px: i32, py: i32, theta: f32) -> Hit {
  let dir = vec2f(cos(theta), sin(theta));
  let sw = i32(params.w);
  let sh = i32(params.h);
  let w0 = textureLoad(worldTex, vec2i(px, py), 0);
  var rad = w0.rgb * w0.a;
  if (w0.a < 1e-6) { rad += w0.rgb * 0.5 / max(abs(dir.x), abs(dir.y)); }
  var thr = 1.0 - w0.a;
  var id = py * sw + px;
  if (thr < 1e-6) { return Hit(rad, id); }
  let o = vec2f(f32(px) + 0.5, f32(py) + 0.5);
  var cell = vec2i(px, py);
  let ax = abs(dir.x);
  let ay = abs(dir.y);
  let sx = select(-1, 1, dir.x >= 0.0);
  let sy = select(-1, 1, dir.y >= 0.0);
  let tdx = select(1e20, 1.0 / ax, ax > 1e-12);
  let tdy = select(1e20, 1.0 / ay, ay > 1e-12);
  var tmx = select(1e20, (select(f32(cell.x), f32(cell.x + 1), dir.x >= 0.0) - o.x) / dir.x, ax > 1e-12);
  var tmy = select(1e20, (select(f32(cell.y), f32(cell.y + 1), dir.y >= 0.0) - o.y) / dir.y, ay > 1e-12);
  var tPrev = min(tmx, tmy);
  if (tmx < tmy) { cell.x += sx; tmx += tdx; } else { cell.y += sy; tmy += tdy; }
  for (var step = 0u; step < 16384u; step++) {
    if (cell.x < 0 || cell.y < 0 || cell.x >= sw || cell.y >= sh) {
      rad += thr * sampleSky(dir);
      id = -1;
      break;
    }
    let tExit = min(tmx, tmy);
    let len = tExit - tPrev;
    let w = textureLoad(worldTex, cell, 0);
    if (w.a < 1e-6 || len <= 0.0) {
      rad += thr * w.rgb * len;
    } else {
      let st = pow(1.0 - w.a, len);
      rad += thr * w.rgb * (1.0 - st);
      thr *= st;
      if (thr < 1e-6) { id = cell.y * sw + cell.x; break; }
    }
    tPrev = tExit;
    if (tmx < tmy) { cell.x += sx; tmx += tdx; } else { cell.y += sy; tmy += tdy; }
  }
  return Hit(rad, id);
}

fn jumps(a: Hit, b: Hit) -> bool {
  if (a.id == b.id) { return false; }
  let d = abs(a.rad - b.rad);
  let m = max(abs(a.rad), abs(b.rad));
  return any(d > m * 0.01 + vec3f(1e-6));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = i32(gid.x);
  let py = i32(params.rowStart + gid.y);
  if (gid.x >= params.w || gid.y >= params.rowCount || u32(py) >= params.h) { return; }
  let dth = 6.28318530 / f32(params.m);
  let k0 = params.chunk * params.chunkLen;
  var th0 = (f32(k0) + 0.5) * dth;
  var h0 = trace(px, py, th0);
  var sum = vec3f(0.0);
  for (var i = 0u; i < params.chunkLen; i++) {
    let th1 = (f32(k0 + i + 1u) + 0.5) * dth;
    let h1 = trace(px, py, th1);
    if (jumps(h0, h1)) {
      var a = th0; var b = th1; var ha = h0; var hb = h1;
      for (var d = 0u; d < params.depth; d++) {
        let m = 0.5 * (a + b);
        let hm = trace(px, py, m);
        if (hm.id == ha.id) { a = m; ha = hm; } else { b = m; hb = hm; }
      }
      let ts = 0.5 * (a + b);
      sum += (a - th0) * 0.5 * (h0.rad + ha.rad) + (ts - a) * ha.rad + (b - ts) * hb.rad + (th1 - b) * 0.5 * (hb.rad + h1.rad);
    } else {
      sum += dth * 0.5 * (h0.rad + h1.rad);
    }
    th0 = th1;
    h0 = h1;
  }
  let idx = u32(py) * params.w + u32(px);
  let total = acc[idx].rgb + sum;
  acc[idx] = vec4f(total, 1.0);
  textureStore(outTex, vec2i(px, py), vec4f(total * params.scale, 1.0));
}
`;

// ── Direction definitions ──
// Each direction is a 2×3 transform matrix mapping (probeIdx, sliceIdx) → world.
// Isotropic probes: spacing = max(W,H)/ps for both axes, so the probe grid
// covers a square region. The shorter screen axis is oversampled (probes
// outside the screen see vacuum). This eliminates all aspect-ratio math.

function dirTransform(
  dir: number,
  w: number,
  h: number,
  ps: number,
): { transformX: [number, number, number]; transformY: [number, number, number]; probeSpacing: number } {
  const s = Math.max(w, h) / ps;
  switch (dir) {
    case 0: // East: probeIdx→X, sliceIdx→Y
      return { transformX: [s, 0, 0], transformY: [0, s, 0], probeSpacing: s };
    case 1: // North: probeIdx→Y, sliceIdx→X
      return { transformX: [0, s, 0], transformY: [s, 0, 0], probeSpacing: s };
    case 2: // West: probeIdx→X reversed, sliceIdx→Y
      return { transformX: [-s, 0, w], transformY: [0, s, 0], probeSpacing: s };
    case 3: // South: probeIdx→Y reversed, sliceIdx→X
      return { transformX: [0, s, 0], transformY: [-s, 0, h], probeSpacing: s };
    default:
      return dirTransform(0, w, h, ps);
  }
}

// ── Cascade ──
// Everything that depends on a cascade's probe resolution and channel mode.
// Full color runs one rgb cascade at probe resolution. Split color runs a
// low-res rgb cascade, then a full-res mono cascade whose blur pass
// reconstructs RGB from the two.

interface Cascade {
  mode: ChannelMode;
  probesX: number;
  probesY: number;
  numCascades: number;
  mergeStride: number;
  fluenceStride: number;
  rayBuffers: GPUBuffer[];
  mergeBuffers: GPUBuffer[];
  // All 4 directions accumulate into this buffer, then blur writes it to `outputView`.
  fluenceBuffer: GPUBuffer;
  // Low-res color fluence (split mode only); otherwise output is the shared fluence texture.
  ownedOutput?: GPUTexture;
  outputView: GPUTextureView;
  // Sky prefix sums for O(1) sky integration at any level. Width = mergeStride+1, height = 4 directions.
  skyPrefixSumTexture: GPUTexture;
  skyPrefixSumTextureView: GPUTextureView;
  coneArcBuffer: GPUBuffer;
  seedParamsBuffer: GPUBuffer;
  extendParamsBuffer: GPUBuffer;
  mergeParamsBuffer: GPUBuffer;
  blurParamsBuffer: GPUBuffer;
  seedBindGroups: GPUBindGroup[];
  extendBindGroups: GPUBindGroup[];
  mergeBindGroups: GPUBindGroup[][];
  // Merge-side read (mergeRead > 0): stencil weights per (level, branch, sub-bin), 3 slices x 5 ray offsets.
  readWeightBuffer: GPUBuffer;
  mergeReadBindGroups: Record<number, GPUBindGroup[][]>;
  blurBindGroup: GPUBindGroup;
  // Ratio mode: 4 = level-0 fluence kept per direction; 1 = summed (production).
  fluenceDirs: number;
  // Seed every cell as transparent (the ratio's free channel).
  freeWorld: boolean;
  // Ray/merge buffers belong to another cascade (not destroyed here).
  sharesBuffers: boolean;
}

interface CascadePipelines {
  seed: GPUComputePipeline[]; // per seeded level (0, 1)
  extend: GPUComputePipeline;
  merge: GPUComputePipeline;
  mergeRead: Record<number, GPUComputePipeline>;
  blur: GPUComputePipeline;
}

// ── Component ──

/** Options that differ from the defaults; any reactive property can be given. */
export type HRCRatioOptions = Partial<Pick<HRCRatio, (typeof REACTIVE_PROPS)[number]>>;

/** Scene units across the displayed square; shapes are in these units, centred on the square. */
export const SCENE_SPAN = 1024;

/** Properties whose assignment triggers an update (uploads, rebuilds), batched per microtask. */
const REACTIVE_PROPS = [
  'exposure',
  'probeCount',
  'bounces',
  'debugMode',
  'falseColor',
  'banded',
  'pixelPerfect',
  'colorScale',
  'ratio',
  'ratioView',
  'freeMethod',
  'mergeRead',
  'parityKappa',
  'parityEdge',
  'parityRead',
  'equalise',
  'equalTaps',
  'equalEdge',
  'equalAxis',
  'evenWeight',
  'evenWeights',
  'mergeReadParams',
] as const;

/**
 * The HRC testbed renderer. It owns a canvas inside `host`, renders `shapes` (plain objects in scene units)
 * every frame, and reacts to assignments to its properties (see REACTIVE_PROPS).
 */
export class HRCRatio {
  readonly host: HTMLElement;
  /** Scene shapes, read every frame; mutate the array and the objects freely. */
  shapes: Shape[] = [];
  /** Resolves once WebGPU is initialised and the render loop runs. */
  readonly ready: Promise<void>;

  exposure = 2.0;
  probeCount = 1024;
  bounces = true;
  debugMode = 0;
  falseColor = false;
  // Banded display: log10 fluence × exposure in 10 bands per decade over [-2.2, 0], viridis (overrides falseColor).
  banded = false;
  pixelPerfect = 0;
  // Color resolution relative to probe resolution. Below 1, luminance is
  // computed at full probe resolution and color at this scale.
  colorScale = 1;
  // Tier-1 ratio (NOTES §44–§47): exact free-space transport × cascade visibility. Full-colour mode only.
  ratio = false;
  // Ratio debug view: 0 = result, 1 = exact free field, 2 = visibility, 3 = plain HRC (same run), 4 = cascade free estimate
  ratioView = 0;
  // Free-field method: 'fft' (exact, scene-independent cost) or 'direct' (sum over emitters; validation stand-in).
  freeMethod: 'fft' | 'direct' = 'fft';
  // Merge-side read (research/tree/notes/merge_footprints.md): 0 = production merge; 1 rows, 2 slopes, 3 both (weights: setMergeReadWeights).
  mergeRead = 0;
  // Parity read (merge-read 5, research/tree/notes/focus_and_extension.md §13–14): kappa per level (index 0 = finest; the last
  // entry repeats for deeper levels) and the shift of the sub-bin next to the quadrant edge (levels >= 1).
  parityKappa: number[] = [0, 0, 0, 0.16, 0.21, 0.24]; // research/tree/lab/kappa_levels.py
  parityEdge = 0.25;
  // Use the parity read (overrides mergeRead): subgroup halo variant (merge-read 10) where subgroups are 32 wide, else 8.
  parityRead = false;
  // Equal footprints (research/tree/notes/merge_read_mechanism.md §7): even-ray emission blur in the extension (tap .16) plus the
  // merge's quadrant-edge drain (.4 rows, the parity read with kappa 0). Derived, no fitting.
  equalise = false;
  // Per-level taps for equalise (index = parent level; null = uniform .16 on even rays). F1 of §10: even +.08, odd -.04..-.08.
  equalTaps: { even: number[]; odd: number[] } | null = null;
  // Per-level quadrant-edge drain for equalise (index = level; null = .4 at every level >= 1).
  // Derived per level on a free light at 512^2 (research/tree/lab/edge_levels.py): the drain that zeroes diagonal-minus-rest in each
  // level's distance band.
  equalEdge: number[] | null = [0, 0.4, 0.352, 0.433, 0.44, 0.44, 0.417, 0.33];
  // Per-level tap for the axis ray under equalise (index = parent level; null = the even tap); derived like the drain.
  equalAxis: number[] | null = [0, 0, 0.032, 0.08, 0.141, 0.14, 0.12, 0.102, 0.102, 0.102, 0.102];
  // Weight of the coarser datum in the even probes' merge (HRC: 1/2). 0.55 lowers error on every scene of the research suite
  // (research/tree/notes/read_implications.md §9); evenWeights, when set, gives one weight per level (index 0 = finest).
  evenWeight = 0.5;
  evenWeights: number[] | null = null;
  // Constants of the compact rows read (merge-read 4) per level class: '0'..'3' and 'int' (levels 4 .. top-1); the top level keeps
  // HRC's read. Each entry [aEdge, aInt, dEdge, dInt, R] for the odd and the even branch.
  mergeReadParams: Record<string, { o: number[]; e: number[] }> = MERGE_READ_PARAMS;
  #mergeReadWeights: Float32Array | null = null;

  /** Debug: suspend the internal render loop (frames can still be rendered with step()). */
  paused = false;

  constructor(host: HTMLElement, options: HRCRatioOptions = {}) {
    this.host = host;
    Object.assign(this, options);
    reactive(this, REACTIVE_PROPS, (changes) => this.#update(changes));
    this.ready = this.#start();
  }

  /** The canvas the scene is displayed on (exists once `ready` resolves). */
  get canvas(): HTMLCanvasElement | undefined {
    return this.#canvas;
  }

  get #pp(): number | null {
    return this.pixelPerfect > 0 ? this.pixelPerfect : null;
  }

  #canvas!: HTMLCanvasElement;
  #device!: GPUDevice;
  #context!: GPUCanvasContext;
  #presentationFormat!: GPUTextureFormat;

  // World textures (world + material, MRT)
  #worldTexture!: GPUTexture;
  #worldTextureView!: GPUTextureView;
  #materialTexture!: GPUTexture;
  #materialTextureView!: GPUTextureView;
  #worldRenderPipeline!: GPURenderPipeline;

  // Shape / line / mouse light rendering
  #shapeDataBuffer?: GPUBuffer;
  #shapeCount = 0;
  #lines: Line[] = [];
  #lineInstanceBuffer?: GPUBuffer;
  #lineInstanceCapacity = 0;
  #lineCount = 0;
  #lineBufferDirty = false;
  #lineRenderPipeline!: GPURenderPipeline;
  #lineUBO?: GPUBuffer;
  #lineBindGroup?: GPUBindGroup;
  #mousePosition = { x: -1e6, y: -1e6 }; // offscreen until the pointer moves
  #mouseDirty = true;
  #mouseLightColor = { r: 0.8, g: 0.6, b: 0.3 };
  #mouseLightRadius = 10;
  #mouseLightOpacity = 1;
  #mouseLightAlbedo = 0;
  #mouseLightBuffer?: GPUBuffer;
  #mouseLightVertexCount = 0;

  // Paint layer (pixel-perfect mode): the CPU-side scene, float RGBA (rgb = emission in linear units, a = opacity; f16)
  // plus albedo (8 bit). Strokes, PNG scenes and loadSceneData write it; dirty rectangles are uploaded to the paint
  // textures, which are copied under the world/material textures every frame (shapes, lines and the cursor light draw
  // on top). Float storage keeps the emission of opacity-0 (volumetric) strokes, which a 2D canvas would zero.
  #paint: { size: number; rgba: Uint16Array; albedo: Uint8Array } | null = null;
  #paintDirty: [number, number, number, number] | null = null; // x0, y0, x1, y1 (exclusive)
  #paintTexture: GPUTexture | null = null;
  #paintMaterialTexture: GPUTexture | null = null;

  // Cascades in execution order (color before the mono pass that reads it).
  #cascades: Cascade[] = [];
  // Ratio mode: the same cascade run on the transparent world (shares ray/merge buffers with #cascades[0]).
  #freeCascade: Cascade | null = null;
  #emitterCounter!: GPUBuffer;
  #emitterBuffer!: GPUBuffer;
  #freeFieldBuffer!: GPUBuffer;
  #emitParamsBuffer!: GPUBuffer;
  #freeParamsBuffer!: GPUBuffer;
  #combineParamsBuffer!: GPUBuffer;
  #emitPipeline!: GPUComputePipeline;
  #freeFieldPipeline!: GPUComputePipeline;
  #combinePipeline!: GPUComputePipeline;
  #emitBindGroup?: GPUBindGroup;
  #freeFieldBindGroup?: GPUBindGroup;
  #combineBindGroup?: GPUBindGroup;
  #fftP = 0;
  #fftPipeline?: GPUComputePipeline;
  #fftGatherPipeline!: GPUComputePipeline;
  #fftBuffers: Record<'kspec' | 'specA' | 'specB' | 'scratch' | 'work' | 'twiddle', GPUBuffer> | null = null;
  #fftParamsBuffer!: GPUBuffer;
  #fftParamsView = uboView(fftShader(256), 'params');
  #fftGatherParamsBuffer!: GPUBuffer;
  #fftGatherParamsView = uboView(fftGatherShader('rgb'), 'params');
  #fftPasses: { bg: GPUBindGroup; lines: number }[] = [];
  #fftGathers: GPUBindGroup[] = [];
  #fftKernelPasses: { bg: GPUBindGroup; lines: number }[] = [];
  #fftKernelReady = false;
  #transportTexture!: GPUTexture;
  #transportTextureView!: GPUTextureView;
  #transportPipeline!: GPUComputePipeline;
  #transportBindGroup?: GPUBindGroup;
  #emissionTexture!: GPUTexture;
  #emissionTextureView!: GPUTextureView;
  #emissionPipeline!: GPUComputePipeline;
  #emissionParamsBuffer!: GPUBuffer;
  #emissionParamsView = uboView(emissionImageShader, 'params');
  #emissionBindGroup?: GPUBindGroup;
  #nearPipeline!: GPUComputePipeline;
  #nearTapsBuffer!: GPUBuffer;
  #nearTapCount = 0;
  #nearParamsBuffer!: GPUBuffer;
  #nearParamsView = uboView(freeNearShader('rgb'), 'params');
  #nearBindGroups: GPUBindGroup[] = [];
  #emitParamsView = uboView(freeEmitShader, 'params');
  #freeParamsView = uboView(freeFieldDirectShader('rgb'), 'params');
  #combineParamsView = uboView(combineShader('rgb'), 'params');
  static readonly EMITTER_CAPACITY = 65536;

  get #allCascades(): Cascade[] {
    return this.#freeCascade ? [...this.#cascades, this.#freeCascade] : this.#cascades;
  }

  // Final RGB fluence at probe resolution, read by blit and bounce.
  #fluenceTexture!: GPUTexture;
  #fluenceTextureView!: GPUTextureView;

  // Bounce texture (screen resolution) for diffuse light bounces
  #bounceTexture!: GPUTexture;
  #bounceTextureView!: GPUTextureView;
  #lastFluenceReady = false;
  #bounceZeroBuffer!: GPUBuffer;

  // Sky circle texture (1D radiance from every angle, stored as 2D with height=1)
  #skyTexture!: GPUTexture;
  #skyTextureView!: GPUTextureView;
  #skyCircleData = new Float32Array(SKY_CIRCLE_SIZE * 4);

  // Path tracer accumulation (screen resolution, rgba32float for precision)
  #ptAccumTextures!: GPUTexture[];
  #ptAccumTextureViews!: GPUTextureView[];
  #ptFrameIndex = 0;
  // Exact reference (exactShader): cached results keyed by a hash of the snapshot it was computed for.
  #exactPipeline?: GPUComputePipeline;
  #exactParamsView = uboView(exactShader, 'params');
  #exactCache = new Map<string, ExactEntry>();
  #exactCurrent: ExactEntry | null = null;
  #ptPipeline!: GPUComputePipeline;
  #ptParamsBuffer!: GPUBuffer;
  #ptParamsView!: StructuredView;

  // Pipelines
  #bounceComputePipeline!: GPUComputePipeline;
  #cascadePipelines!: Record<ChannelMode, CascadePipelines>;
  #renderPipeline!: GPURenderPipeline;

  // Pre-created bind groups
  #blitBindGroup!: GPUBindGroup;
  #bounceBindGroup!: GPUBindGroup;

  // Sampler
  #linearSampler!: GPUSampler;

  // Uniform buffers + structured views (struct layouts are identical across channel modes)
  #seedParamsView = uboView(raySeedShader('rgb', 0), 'params');
  #extendParamsView = uboView(rayExtendShader('rgb'), 'params');
  #mergeParamsView = uboView(cascadeMergeShader('rgb'), 'params');
  #blitParamsBuffer!: GPUBuffer;
  #blitParamsView = uboView(blitShader, 'params');
  #bounceParamsBuffer!: GPUBuffer;
  #bounceParamsView = uboView(bounceComputeShader, 'params');

  // Full probe resolution (fluence / bounce textures)
  #probesX = 0;
  #probesY = 0;

  #animationFrame = 0;
  #isRunning = false;
  #resizing = false;

  #smoothedFrameTime = 0;
  #lastFrameTimestamp = 0;

  // GPU timestamp profiling (null when timestamp-query unavailable)
  #tsQuerySet: GPUQuerySet | null = null;
  #tsResolveBuffer: GPUBuffer | null = null;
  #tsResultBuffer: GPUBuffer | null = null;
  #tsNextIdx = 0;
  #tsPassCount = 0;
  #gpuTimeMs = 0;
  #jsTimeMs = 0;
  #gpuPassTimings: { label: string; ms: number }[] = [];

  static readonly TS_MAX_PASSES = 96;

  #initTimestampQueries() {
    const device = this.#device;
    const count = HRCRatio.TS_MAX_PASSES * 2;
    this.#tsQuerySet = device.createQuerySet({ type: 'timestamp', count });
    this.#tsResolveBuffer = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.#tsResultBuffer = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  #tsLabels: string[] = [];

  #tsBeginFrame() {
    this.#tsNextIdx = 0;
    this.#tsLabels = [];
  }

  #tsPass(label = ''): GPUComputePassTimestampWrites | GPURenderPassTimestampWrites | undefined {
    if (!this.#tsQuerySet) return undefined;
    const idx = this.#tsNextIdx;
    if (idx + 1 >= HRCRatio.TS_MAX_PASSES * 2) return undefined;
    this.#tsNextIdx = idx + 2;
    this.#tsLabels.push(label);
    return { querySet: this.#tsQuerySet, beginningOfPassWriteIndex: idx, endOfPassWriteIndex: idx + 1 };
  }

  #resolveTimestamps(encoder: GPUCommandEncoder) {
    if (!this.#tsQuerySet || !this.#tsResultBuffer) return;
    this.#tsPassCount = this.#tsNextIdx / 2;
    encoder.resolveQuerySet(this.#tsQuerySet, 0, this.#tsNextIdx, this.#tsResolveBuffer!, 0);
    if (this.#tsResultBuffer.mapState === 'unmapped') {
      encoder.copyBufferToBuffer(this.#tsResolveBuffer!, 0, this.#tsResultBuffer, 0, this.#tsNextIdx * 8);
    }
  }

  #readTimestamps() {
    if (!this.#tsResultBuffer || this.#tsResultBuffer.mapState !== 'unmapped') return;
    const passCount = this.#tsPassCount;
    const labels = [...this.#tsLabels];
    this.#tsResultBuffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const times = new BigInt64Array(this.#tsResultBuffer!.getMappedRange());
        if (passCount > 0) {
          this.#gpuTimeMs = Number(times[passCount * 2 - 1] - times[0]) / 1e6;
        }
        const aggregated = new Map<string, number>();
        let totalGpu = 0;
        for (let i = 0; i < passCount; i++) {
          const ms = (Number(times[i * 2 + 1]) - Number(times[i * 2])) / 1e6;
          const raw = labels[i] || `pass${i}`;
          const dashIdx = raw.indexOf('-');
          const key = dashIdx >= 0 && /^[ENSW]$/.test(raw.substring(0, dashIdx)) ? raw.substring(dashIdx + 1) : raw;
          aggregated.set(key, (aggregated.get(key) ?? 0) + ms);
          totalGpu += ms;
        }
        this.#gpuTimeMs = totalGpu;
        const passTimings: { label: string; ms: number }[] = [];
        for (const [label, ms] of aggregated) {
          passTimings.push({ label, ms });
        }
        this.#gpuPassTimings = passTimings;
        this.#tsResultBuffer!.unmap();
      })
      .catch(() => {});
  }

  async #start() {
    await this.#initWebGPU();
    this.#initResources();
    this.#initPipelines();
    this.#uploadStaticParams(this.#canvas.width, this.#canvas.height);
    this.#lineBindGroup = undefined;
    window.addEventListener('resize', this.#handleResize);
    window.addEventListener('mousemove', this.#handleMouseMove);
    this.#isRunning = true;
    this.#startAnimationLoop();
    this.#update(new Map());
  }

  /** Stops rendering, removes the canvas and frees GPU resources. */
  destroy() {
    this.#isRunning = false;
    if (this.#animationFrame) cancelAnimationFrame(this.#animationFrame);
    window.removeEventListener('resize', this.#handleResize);
    window.removeEventListener('mousemove', this.#handleMouseMove);
    this.#destroyResources();
    this.#canvas?.remove();
  }

  #updateCanvasLayout() {
    const pp = this.#pp;
    if (pp) {
      if (this.#canvas.width !== pp) this.#canvas.width = pp;
      if (this.#canvas.height !== pp) this.#canvas.height = pp;
      const containerW = this.host.clientWidth || 800;
      const containerH = this.host.clientHeight || 600;
      const displaySize = Math.min(containerW - 64, containerH - 64);
      Object.assign(this.#canvas.style, {
        position: 'absolute',
        left: `${(containerW - displaySize) / 2}px`,
        top: `${(containerH - displaySize) / 2}px`,
        width: `${displaySize}px`,
        height: `${displaySize}px`,
        pointerEvents: 'none',
        imageRendering: 'pixelated',
      });
    } else {
      this.#canvas.width = this.host.clientWidth || 800;
      this.#canvas.height = this.host.clientHeight || 600;
      Object.assign(this.#canvas.style, {
        position: 'absolute',
        inset: '0',
        left: '',
        top: '',
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        imageRendering: '',
      });
    }
  }

  #rebuild() {
    this.#destroyResources();
    this.#initResources();
    this.#createStaticBindGroups();
    this.#uploadStaticParams(this.#canvas.width, this.#canvas.height);
    this.#lineBindGroup = undefined;
  }

  #mapToCanvas(x: number, y: number): [number, number] {
    const rect = this.#canvas.getBoundingClientRect();
    if (!this.#pp) return [x - rect.left, y - rect.top];
    return [(x - rect.left) * (this.#canvas.width / rect.width), (y - rect.top) * (this.#canvas.height / rect.height)];
  }

  #scaleToCanvas(v: number): number {
    if (!this.#pp) return v;
    const rect = this.#canvas.getBoundingClientRect();
    return v * (this.#canvas.width / rect.width);
  }

  #ensurePaint() {
    const pp = this.#pp;
    if (!pp) return null;
    if (!this.#paint || this.#paint.size !== pp) {
      this.#paint = { size: pp, rgba: new Uint16Array(pp * pp * 4), albedo: new Uint8Array(pp * pp) };
      this.#markPaint(0, 0, pp, pp);
    }
    return this.#paint;
  }

  #markPaint(x0: number, y0: number, x1: number, y1: number) {
    const n = this.#paint?.size ?? 0;
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(n, x1); y1 = Math.min(n, y1);
    if (x1 <= x0 || y1 <= y0) return;
    const d = this.#paintDirty;
    this.#paintDirty = d ? [Math.min(d[0], x0), Math.min(d[1], y0), Math.max(d[2], x1), Math.max(d[3], y1)] : [x0, y0, x1, y1];
  }

  /** Paint a bs × bs block (top-left x, y) with one f16 RGBA value and albedo byte (replaces what is there). */
  #paintBlock(x: number, y: number, bs: number, px: Uint16Array, alb: number) {
    const p = this.#paint!;
    const n = p.size;
    const x0 = Math.max(0, x), y0 = Math.max(0, y), x1 = Math.min(n, x + bs), y1 = Math.min(n, y + bs);
    for (let yy = y0; yy < y1; yy++) {
      for (let xx = x0; xx < x1; xx++) {
        const i = yy * n + xx;
        p.rgba.set(px, i * 4);
        p.albedo[i] = alb;
      }
    }
    this.#markPaint(x0, y0, x1, y1);
  }

  addLine(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: [number, number, number] = [0, 0, 0],
    thickness = 20,
    opacity = 1,
    albedo = 0,
  ) {
    if (this.#pp) {
      this.#ensurePaint();
      [x1, y1] = this.#mapToCanvas(x1, y1);
      [x2, y2] = this.#mapToCanvas(x2, y2);
      // same material rule as the shape/line shaders: linear emission = sRGB-decoded colour, alpha = opacity
      const lin = (v: number) => {
        const c = Math.max(0, v);
        return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      };
      const px = new Uint16Array([floatToHalf(lin(color[0])), floatToHalf(lin(color[1])), floatToHalf(lin(color[2])), floatToHalf(Math.max(0, Math.min(1, opacity)))]);
      const alb = Math.round(Math.max(0, Math.min(1, albedo)) * 255);
      const bs = Math.max(1, Math.round(thickness));
      const blo = -Math.floor((bs - 1) / 2);
      const dx = Math.abs(Math.floor(x2) - Math.floor(x1));
      const dy = Math.abs(Math.floor(y2) - Math.floor(y1));
      const sx = x1 < x2 ? 1 : -1;
      const sy = y1 < y2 ? 1 : -1;
      let err = dx - dy;
      let x = Math.floor(x1),
        y = Math.floor(y1);
      const ex = Math.floor(x2),
        ey = Math.floor(y2);
      for (;;) {
        this.#paintBlock(x + blo, y + blo, bs, px, alb);
        if (x === ex && y === ey) break;
        const e2 = 2 * err;
        if (e2 > -dy) {
          err -= dy;
          x += sx;
        }
        if (e2 < dx) {
          err += dx;
          y += sy;
        }
      }
      return;
    }
    const [r, g, b] = color;
    this.#lines.push([x1, y1, x2, y2, r, g, b, thickness, opacity, albedo]);
    this.#lineBufferDirty = true;
  }

  clearLines() {
    this.#lines = [];
    this.#lineBufferDirty = true;
    if (this.#paint) {
      this.#paint.rgba.fill(0);
      this.#paint.albedo.fill(0);
      this.#markPaint(0, 0, this.#paint.size, this.#paint.size);
    }
  }

  loadSceneImage(img: HTMLImageElement) {
    const size = img.width;
    if (img.width !== img.height) {
      console.warn('Scene PNG must be square, got', img.width, 'x', img.height);
      return;
    }
    // PNG pixels are world values: rgb = emission (linear, as stored), alpha = opacity (saveScenePNG writes the same)
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, size, size).data;
    const rgba = new Float32Array(size * size * 4);
    for (let i = 0; i < rgba.length; i++) rgba[i] = d[i] / 255;
    this.loadSceneData(size, rgba);
  }

  /** Load a square scene as float RGBA (r, g, b = emission, a = opacity) into the paint layer (albedo 0). */
  loadSceneData(size: number, rgba: Float32Array) {
    if (rgba.length !== size * size * 4) throw new Error('loadSceneData: expected size² × 4 floats');
    this.pixelPerfect = size;
    this.clearLines();
    const f16 = new Uint16Array(rgba.length);
    for (let i = 0; i < rgba.length; i++) f16[i] = floatToHalf(rgba[i]);
    this.#paint = { size, rgba: f16, albedo: new Uint8Array(size * size) };
    this.#paintDirty = [0, 0, size, size];
  }

  /** Save the paint layer as PNG (rgb = emission clamped to [0, 1], alpha = opacity). Volumetric (alpha 0) pixels lose
   *  their colour: PNG/canvas storage is premultiplied. */
  saveScenePNG(filename = 'scene.png') {
    const p = this.#paint;
    if (!p) {
      console.warn('saveScenePNG: no painted scene (pixel-perfect mode only)');
      return;
    }
    const c = document.createElement('canvas');
    c.width = c.height = p.size;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(p.size, p.size);
    for (let i = 0; i < p.rgba.length; i++) img.data[i] = Math.round(Math.max(0, Math.min(1, halfToFloat(p.rgba[i]))) * 255);
    ctx.putImageData(img, 0, 0);
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png');
    a.download = filename;
    a.click();
  }

  setMouseLightColor(r: number, g: number, b: number) {
    this.#mouseLightColor = { r, g, b };
    this.#mouseDirty = true;
  }

  setMouseLightRadius(radius: number) {
    this.#mouseLightRadius = radius;
    this.#mouseDirty = true;
  }

  setMouseLightMaterial(opacity: number, albedo: number) {
    this.#mouseLightOpacity = opacity;
    this.#mouseLightAlbedo = albedo;
    this.#mouseDirty = true;
  }

  eraseAt(x: number, y: number, radius: number) {
    if (this.#pp) {
      const p = this.#paint;
      if (!p) return;
      [x, y] = this.#mapToCanvas(x, y);
      radius = this.#scaleToCanvas(radius);
      const x0 = Math.max(0, Math.floor(x - radius)), x1 = Math.min(p.size, Math.ceil(x + radius));
      const y0 = Math.max(0, Math.floor(y - radius)), y1 = Math.min(p.size, Math.ceil(y + radius));
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 > radius * radius) continue;
          const i = yy * p.size + xx;
          p.rgba.fill(0, i * 4, i * 4 + 4);
          p.albedo[i] = 0;
        }
      }
      this.#markPaint(x0, y0, x1, y1);
      return;
    }
    this.#lines = this.#lines.filter((line) => {
      const [x1, y1, x2, y2] = line;
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len2 = dx * dx + dy * dy;
      if (len2 === 0) return Math.hypot(x - x1, y - y1) > radius;
      const t = Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / len2));
      return Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) > radius;
    });
    this.#lineBufferDirty = true;
  }

  setSkyColor(r: number, g: number, b: number) {
    for (let i = 0; i < SKY_CIRCLE_SIZE; i++) {
      this.#skyCircleData[i * 4] = r;
      this.#skyCircleData[i * 4 + 1] = g;
      this.#skyCircleData[i * 4 + 2] = b;
      this.#skyCircleData[i * 4 + 3] = 1;
    }
    this.#uploadSkyCircle();
  }

  setSkyCircle(data: Float32Array) {
    if (data.length !== SKY_CIRCLE_SIZE * 4) {
      throw new Error(`Sky circle data must have ${SKY_CIRCLE_SIZE * 4} elements (${SKY_CIRCLE_SIZE} RGBA texels)`);
    }
    this.#skyCircleData.set(data);
    this.#uploadSkyCircle();
  }

  clearSky() {
    this.#skyCircleData.fill(0);
    this.#uploadSkyCircle();
  }

  #uploadSkyCircle() {
    if (!this.#device || !this.#skyTexture) return;
    this.#device.queue.writeTexture(
      { texture: this.#skyTexture },
      this.#skyCircleData,
      { bytesPerRow: SKY_CIRCLE_SIZE * 16 },
      { width: SKY_CIRCLE_SIZE, height: 1 },
    );
    for (const c of this.#allCascades) this.#computeSkyPrefixSums(c);
  }

  // Sky fluence over each cascade direction's 90° quadrant (same angle convention as #computeSkyPrefixSums and the
  // path tracer's sampleSky): E = (−π/4, π/4), N = (π/4, 3π/4), W = around π, S = (−3π/4, −π/4).
  #skyQuadrants(): number[] {
    const out = new Array<number>(16).fill(0);
    const n = SKY_CIRCLE_SIZE;
    const dTheta = (2 * Math.PI) / n;
    for (let i = 0; i < n; i++) {
      const angle = -Math.PI + (i + 0.5) * dTheta;
      const a = Math.abs(angle);
      let dir: number;
      if (a < Math.PI / 4) dir = 0;
      else if (a > (3 * Math.PI) / 4) dir = 2;
      else dir = angle > 0 ? 1 : 3;
      for (let c = 0; c < 3; c++) out[dir * 4 + c] += this.#skyCircleData[i * 4 + c] * dTheta;
    }
    return out;
  }

  #computeSkyPrefixSums(c: Cascade) {
    const ms = c.mergeStride;
    const rowLen = ms + 1;
    const data = new Float32Array(rowLen * 4 * 4);

    for (let dir = 0; dir < 4; dir++) {
      const rowOff = dir * rowLen * 4;

      data[rowOff] = 0;
      data[rowOff + 1] = 0;
      data[rowOff + 2] = 0;
      data[rowOff + 3] = 0;

      for (let s = 0; s < ms; s++) {
        const N = ms;
        const slope = (2 * s - N + 1) / N;
        let angle: number;
        switch (dir) {
          case 0:
            angle = Math.atan2(slope, 1);
            break;
          case 1:
            angle = Math.atan2(1, slope);
            break;
          case 2:
            angle = Math.atan2(slope, -1);
            break;
          default:
            angle = Math.atan2(-1, slope);
            break;
        }
        const u = (angle + Math.PI) / (Math.PI * 2);
        const skyIdx = Math.max(0, Math.min(SKY_CIRCLE_SIZE - 1, Math.floor(u * SKY_CIRCLE_SIZE)));
        const skyR = this.#skyCircleData[skyIdx * 4];
        const skyG = this.#skyCircleData[skyIdx * 4 + 1];
        const skyB = this.#skyCircleData[skyIdx * 4 + 2];

        const cW = Math.atan2(2 * s - N + 2, N) - Math.atan2(2 * s - N, N);

        const prevOff = rowOff + s * 4;
        const currOff = rowOff + (s + 1) * 4;
        data[currOff] = data[prevOff] + skyR * cW;
        data[currOff + 1] = data[prevOff + 1] + skyG * cW;
        data[currOff + 2] = data[prevOff + 2] + skyB * cW;
        data[currOff + 3] = 0;
      }
    }

    const bytesPerRow = rowLen * 16;
    const alignedBytesPerRow = Math.ceil(bytesPerRow / 256) * 256;
    const padded = new Float32Array((alignedBytesPerRow / 4) * 4);
    for (let dir = 0; dir < 4; dir++) {
      const srcOff = dir * rowLen * 4;
      const dstOff = dir * (alignedBytesPerRow / 4);
      padded.set(data.subarray(srcOff, srcOff + rowLen * 4), dstOff);
    }

    this.#device.queue.writeTexture(
      { texture: c.skyPrefixSumTexture },
      padded,
      { bytesPerRow: alignedBytesPerRow, rowsPerImage: 4 },
      { width: rowLen, height: 4 },
    );
  }

  #update(changedProperties: Changes) {
    if (!this.#device) return;
    const probeChanged = changedProperties.has('probeCount') && changedProperties.get('probeCount') !== undefined;
    const ppChanged = changedProperties.has('pixelPerfect') && changedProperties.get('pixelPerfect') !== undefined;
    const colorChanged = changedProperties.has('colorScale') && changedProperties.get('colorScale') !== undefined;
    if (changedProperties.has('evenWeight') || changedProperties.has('evenWeights') || changedProperties.has('mergeReadParams') ||
        changedProperties.has('parityKappa') || changedProperties.has('parityEdge') || changedProperties.has('equalise') ||
        changedProperties.has('equalEdge')) {
      for (const c of this.#allCascades) this.#uploadMergeParams(c);
    }
    if (changedProperties.has('equalise') || changedProperties.has('equalTaps') || changedProperties.has('equalAxis')) for (const c of this.#allCascades) this.#uploadExtendParams(c);
    const ratioChanged = changedProperties.has('ratio') && changedProperties.get('ratio') !== undefined;
    if (probeChanged || ppChanged || colorChanged || ratioChanged) {
      if (ppChanged) {
        this.#updateCanvasLayout();
        this.#context.configure({ device: this.#device, format: this.#presentationFormat, alphaMode: 'premultiplied' });
      }
      this.#rebuild();
    }
    if (
      probeChanged ||
      ppChanged ||
      changedProperties.has('exposure') ||
      changedProperties.has('bounces') ||
      changedProperties.has('debugMode') ||
      changedProperties.has('falseColor') ||
      changedProperties.has('ratio') ||
      changedProperties.has('ratioView') ||
      changedProperties.has('freeMethod')
    ) {
      this.#ptFrameIndex = 0;
    }
  }

  // ── WebGPU init ──

  async #initWebGPU() {
    if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('Failed to get GPU adapter.');
    const canTimestamp = adapter.features.has('timestamp-query');
    const canFloat32Filter = adapter.features.has('float32-filterable');
    const features: GPUFeatureName[] = [];
    if (canTimestamp) features.push('timestamp-query' as GPUFeatureName);
    if (canFloat32Filter) features.push('float32-filterable' as GPUFeatureName);
    if (adapter.features.has('subgroups')) features.push('subgroups' as GPUFeatureName);
    this.#device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize },
    });
    if (canTimestamp) {
      this.#initTimestampQueries();
    }

    this.#canvas = document.createElement('canvas');
    this.#updateCanvasLayout();
    this.host.prepend(this.#canvas);

    const context = this.#canvas.getContext('webgpu');
    if (!context) throw new Error('Failed to get WebGPU context.');
    this.#context = context;
    this.#presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    // Chrome adds 1-2 frames of input lag because WebGPU doesn't support
    // desynchronized canvas contexts yet (gpuweb/gpuweb#1224, Milestone 4+).
    // WebGL can bypass the compositor with desynchronized:true, WebGPU can't.
    this.#context.configure({ device: this.#device, format: this.#presentationFormat, alphaMode: 'premultiplied' });
    this.#linearSampler = this.#device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  #initResources() {
    const device = this.#device;
    const { width, height } = this.#canvas;
    const maxDim = Math.max(width, height);
    const ps = Math.max(2, this.probeCount);
    const pp = this.#pp;
    const probesX = pp ? pp : Math.max(2, ceilDiv(width * ps, maxDim));
    const probesY = pp ? pp : Math.max(2, ceilDiv(height * ps, maxDim));
    const probesMax = Math.max(probesX, probesY);
    this.#probesX = probesX;
    this.#probesY = probesY;

    const ubo = (label: string, size: number) =>
      device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    [this.#worldTexture, this.#worldTextureView] = tex(
      device,
      'World',
      width,
      height,
      'rgba16float',
      TEX_RENDER | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    );
    [this.#materialTexture, this.#materialTextureView] = tex(device, 'Material', width, height, 'r8unorm', TEX_RENDER | GPUTextureUsage.COPY_DST);
    if (pp) {
      // paint layer textures (see #paint); a rebuild re-uploads the whole layer
      [this.#paintTexture] = tex(device, 'Paint', pp, pp, 'rgba16float', GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST);
      [this.#paintMaterialTexture] = tex(device, 'PaintMaterial', pp, pp, 'r8unorm', GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST);
      if (this.#paint && this.#paint.size === pp) this.#paintDirty = [0, 0, pp, pp];
    } else {
      this.#paintTexture = this.#paintMaterialTexture = null;
    }

    const [ft, fv] = tex(
      device,
      'Fluence',
      probesX,
      probesY,
      'rgba16float',
      GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    );
    this.#fluenceTexture = ft;
    this.#fluenceTextureView = fv;

    const colorScale = Math.min(1, Math.max(0, this.colorScale || 1));
    this.#freeCascade = null;
    if (this.ratio) {
      // Tier 1: one cascade on the scene, one on the transparent scene (shared ray/merge buffers), 4 directions kept
      // apart in the fluence buffers; the combine pass writes the fluence texture.
      const occ = this.#createCascade('rgb', probesX, probesY, width, height, fv, 0, { fluenceDirs: 4 });
      this.#cascades = [occ];
      this.#freeCascade = this.#createCascade('rgb', probesX, probesY, width, height, fv, 0, {
        fluenceDirs: 4,
        freeWorld: true,
        shareWith: occ,
      });
      const cap = HRCRatio.EMITTER_CAPACITY;
      this.#emitterCounter = device.createBuffer({ label: 'Ratio-EmitterCount', size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.#emitterBuffer = device.createBuffer({ label: 'Ratio-Emitters', size: cap * 32, usage: GPUBufferUsage.STORAGE });
      this.#freeFieldBuffer = device.createBuffer({
        label: 'Ratio-FreeField',
        size: occ.fluenceStride * probesY * 4 * 4,
        usage: GPUBufferUsage.STORAGE,
      });
      this.#emitParamsBuffer = ubo('Ratio-EmitParams', this.#emitParamsView.arrayBuffer.byteLength);
      this.#freeParamsBuffer = ubo('Ratio-FreeParams', this.#freeParamsView.arrayBuffer.byteLength);
      this.#combineParamsBuffer = ubo('Ratio-CombineParams', this.#combineParamsView.arrayBuffer.byteLength);
      const s = Math.max(width, height) / probesMax;
      this.#emitParamsView.set({ probesX, probesY, screenW: width, screenH: height, spacing: s, capacity: cap, pad0: 0, pad1: 0 });
      device.queue.writeBuffer(this.#emitParamsBuffer, 0, this.#emitParamsView.arrayBuffer);
      this.#freeParamsView.set({ probesX, probesY, stride: occ.fluenceStride, capacity: cap, spacing: s, pad0: 0, pad1: 0, pad2: 0 });
      device.queue.writeBuffer(this.#freeParamsBuffer, 0, this.#freeParamsView.arrayBuffer);
      [this.#transportTexture, this.#transportTextureView] = tex(device, 'Ratio-TransportWorld', width, height, 'rgba16float', TEX_STORAGE | GPUTextureUsage.COPY_SRC); // COPY_SRC: exact reference snapshot
      [this.#emissionTexture, this.#emissionTextureView] = tex(device, 'Ratio-Emission', probesX, probesY, 'rgba16float', TEX_STORAGE);
      this.#emissionParamsBuffer = ubo('Ratio-EmissionParams', this.#emissionParamsView.arrayBuffer.byteLength);
      this.#emissionParamsView.set({ probesX, probesY, screenW: width, screenH: height, spacing: s, pad0: 0, pad1: 0, pad2: 0 });
      device.queue.writeBuffer(this.#emissionParamsBuffer, 0, this.#emissionParamsView.arrayBuffer);
      const taps = nearTaps();
      this.#nearTapCount = taps.data.length / 4;
      this.#nearTapsBuffer = device.createBuffer({ label: 'Ratio-NearTaps', size: taps.data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(this.#nearTapsBuffer, 0, taps.data);
      this.#nearParamsBuffer = ubo('Ratio-NearParams', 4 * 256);
      for (let phase = 0; phase < 4; phase++) {
        this.#nearParamsView.set({ probesX, probesY, stride: occ.fluenceStride, phase, tapBase: taps.offsets[phase], tapEnd: taps.ends[phase] });
        device.queue.writeBuffer(this.#nearParamsBuffer, phase * 256, this.#nearParamsView.arrayBuffer);
      }
      // FFT grid for the far field on the 2× tent-downsampled emission: 2× zero padding of the coarse square
      const P = 2 * nextPow2(Math.ceil(probesMax / 2));
      const maxShared = device.limits.maxComputeWorkgroupStorageSize;
      this.#fftBuffers = null;
      this.#fftKernelReady = false;
      if (P * 8 <= maxShared) {
        this.#fftP = P;
        const cbuf = (label: string) =>
          device.createBuffer({ label: `Ratio-FFT-${label}`, size: P * P * 8, usage: GPUBufferUsage.STORAGE });
        const tw = new Float32Array(P);
        for (let j = 0; j < P / 2; j++) {
          tw[2 * j] = Math.cos((-2 * Math.PI * j) / P);
          tw[2 * j + 1] = Math.sin((-2 * Math.PI * j) / P);
        }
        const twiddle = device.createBuffer({ label: 'Ratio-FFT-Twiddle', size: tw.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(twiddle, 0, tw);
        this.#fftBuffers = { kspec: cbuf('KSpec'), specA: cbuf('SpecA'), specB: cbuf('SpecB'), scratch: cbuf('Scratch'), work: cbuf('Work'), twiddle };
        this.#fftParamsBuffer = ubo('Ratio-FFTParams', 32 * 256);
        this.#fftGatherParamsBuffer = ubo('Ratio-GatherParams', 8 * 256);
      } else {
        console.warn(`hrc-ratio: FFT line of ${P} complex values exceeds workgroup storage (${maxShared} B); using the direct free field`);
      }
    } else if (colorScale < 1) {
      const colorMax = Math.max(2, Math.round(probesMax * colorScale));
      const cx = Math.max(2, ceilDiv(probesX * colorMax, probesMax));
      const cy = Math.max(2, ceilDiv(probesY * colorMax, probesMax));
      this.#cascades = [
        this.#createCascade('rgb', cx, cy, width, height, null, 0),
        this.#createCascade('mono', probesX, probesY, width, height, fv, colorMax / probesMax),
      ];
    } else {
      this.#cascades = [this.#createCascade('rgb', probesX, probesY, width, height, fv, 0)];
    }

    [this.#bounceTexture, this.#bounceTextureView] = tex(
      device,
      'Bounce',
      probesX,
      probesY,
      'rgba16float',
      TEX_STORAGE | GPUTextureUsage.COPY_DST,
    );
    // bytesPerRow must be 256-aligned for WebGPU copy operations.
    const alignedBounceRow = Math.ceil((probesX * 8) / 256) * 256;
    const bounceZeroSize = alignedBounceRow * probesY;
    this.#bounceZeroBuffer = device.createBuffer({ size: bounceZeroSize, usage: GPUBufferUsage.COPY_SRC });
    device.queue.writeTexture(
      { texture: this.#bounceTexture },
      new Uint8Array(bounceZeroSize),
      { bytesPerRow: alignedBounceRow, rowsPerImage: probesY },
      { width: probesX, height: probesY },
    );
    [this.#skyTexture, this.#skyTextureView] = tex(
      device,
      'Sky',
      SKY_CIRCLE_SIZE,
      1,
      'rgba32float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    );
    this.#uploadSkyCircle();

    this.#lastFluenceReady = false;

    this.#blitParamsBuffer = ubo('BlitParams', this.#blitParamsView.arrayBuffer.byteLength);
    this.#bounceParamsBuffer = ubo('BounceParams', this.#bounceParamsView.arrayBuffer.byteLength);

    const ptUsage = TEX_STORAGE | GPUTextureUsage.COPY_SRC;
    const [ptA0, ptV0] = tex(device, 'PT-Accum-0', width, height, 'rgba32float', ptUsage);
    const [ptA1, ptV1] = tex(device, 'PT-Accum-1', width, height, 'rgba32float', ptUsage);
    this.#ptAccumTextures = [ptA0, ptA1];
    this.#ptAccumTextureViews = [ptV0, ptV1];
    this.#ptParamsView = uboView(pathTraceShader, 'params');
    this.#ptParamsBuffer = ubo('PT-Params', this.#ptParamsView.arrayBuffer.byteLength);
  }

  // `output` null → allocate a private RGB fluence texture (low-res color in split mode).
  #createCascade(
    mode: ChannelMode,
    probesX: number,
    probesY: number,
    width: number,
    height: number,
    output: GPUTextureView | null,
    colorRatio: number,
    opts: { freeWorld?: boolean; fluenceDirs?: number; shareWith?: Cascade } = {},
  ): Cascade {
    const device = this.#device;
    const probesMax = Math.max(probesX, probesY);
    const numCascades = ceilLog2(probesMax);
    const mergeStride = nextPow2(probesMax);
    const fluenceDirs = opts.fluenceDirs ?? 1;
    const tag = opts.freeWorld ? 'Free' : mode === 'mono' ? 'Mono' : 'RGB';
    const ubo = (label: string, size: number) =>
      device.createBuffer({ label: `${tag}-${label}`, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Ray/merge buffers are shared across all 4 directions, sized for worst case.
    // Level 0 stores one entry per probe (no sub-probe duplication).
    const rayBuffers: GPUBuffer[] = opts.shareWith ? opts.shareWith.rayBuffers : [];
    for (let i = 0; i < numCascades && !opts.shareWith; i++) {
      const w = ceilDiv(probesMax, 1 << i) * ((1 << i) + 1);
      rayBuffers.push(
        device.createBuffer({
          label: `${tag}-Ray-${i}`,
          size: w * probesMax * RAY_BYTES[mode],
          usage: GPUBufferUsage.STORAGE,
        }),
      );
    }

    const mergeBuffers = opts.shareWith
      ? opts.shareWith.mergeBuffers
      : [0, 1].map((i) =>
          device.createBuffer({
            label: `${tag}-Merge-${i}`,
            size: mergeStride * probesMax * 4,
            usage: GPUBufferUsage.STORAGE,
          }),
        );
    const alignedFluenceRow = Math.ceil((probesX * 4) / 256) * 256;
    const fluenceBuffer = device.createBuffer({
      label: `${tag}-Fluence-SSBO`,
      size: alignedFluenceRow * probesY * fluenceDirs,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const fluenceStride = alignedFluenceRow / 4;

    let ownedOutput: GPUTexture | undefined;
    let outputView = output;
    if (!outputView) {
      [ownedOutput, outputView] = tex(device, `${tag}-ColorFluence`, probesX, probesY, 'rgba16float', TEX_STORAGE);
    }

    const [skyPrefixSumTexture, skyPrefixSumTextureView] = tex(
      device,
      `${tag}-SkyPrefixSum`,
      mergeStride + 1,
      4,
      'rgba32float',
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    );

    const totalConeArcs = 2 * (1 << numCascades) - 2;
    const coneArcBuffer = device.createBuffer({
      label: `${tag}-ConeArcs`,
      size: Math.max(4, totalConeArcs * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // identity read: weight 1 at (slice 0, ray offset 0) for every (level, branch, sub-bin); readBase(level) = 60 (2^level - 1)
    const readFloats = 60 * ((1 << numCascades) - 1);
    const readWeightBuffer = device.createBuffer({ label: `${tag}-ReadWeights`, size: Math.max(4, readFloats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    {
      const id = new Float32Array(readFloats);
      for (let i = 7; i < readFloats; i += 15) id[i] = 1;
      if (this.#mergeReadWeights) id.set(this.#mergeReadWeights.subarray(0, Math.min(readFloats, this.#mergeReadWeights.length)));
      device.queue.writeBuffer(readWeightBuffer, 0, id);
    }
    const blurParamsBuffer = ubo('FluenceBlurParams', 32);
    const blurParams = new ArrayBuffer(32);
    new Uint32Array(blurParams).set([probesX, probesY, fluenceStride, width, height]);
    new Float32Array(blurParams)[5] = colorRatio;
    device.queue.writeBuffer(blurParamsBuffer, 0, blurParams);

    return {
      mode,
      probesX,
      probesY,
      numCascades,
      mergeStride,
      fluenceStride,
      rayBuffers,
      mergeBuffers,
      fluenceBuffer,
      ownedOutput,
      outputView,
      skyPrefixSumTexture,
      skyPrefixSumTextureView,
      coneArcBuffer,
      seedParamsBuffer: ubo('SeedParams', 8 * 256),
      extendParamsBuffer: ubo('ExtendParams', 2 * Math.max(1, numCascades - 1) * 256),
      mergeParamsBuffer: ubo('MergeParams', 4 * numCascades * 256),
      blurParamsBuffer,
      seedBindGroups: [],
      extendBindGroups: [],
      mergeBindGroups: [],
      readWeightBuffer,
      mergeReadBindGroups: {},
      blurBindGroup: undefined!,
      fluenceDirs,
      freeWorld: !!opts.freeWorld,
      sharesBuffers: !!opts.shareWith,
    };
  }

  #initPipelines() {
    const device = this.#device;
    const MRT_TARGETS: GPUColorTargetState[] = [{ format: 'rgba16float' }, { format: 'r8unorm' }];

    const attr = (loc: number, off: number, fmt: GPUVertexFormat): GPUVertexAttribute => ({
      shaderLocation: loc,
      offset: off,
      format: fmt,
    });

    const fullscreenBlit = (label: string, code: string, format: GPUTextureFormat) => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        label,
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
      });
    };

    const worldModule = device.createShaderModule({ code: worldRenderShader });
    this.#worldRenderPipeline = device.createRenderPipeline({
      label: 'HRC-WorldRender',
      layout: 'auto',
      vertex: {
        module: worldModule,
        entryPoint: 'vertex_main',
        buffers: [
          {
            arrayStride: 28,
            attributes: [
              attr(0, 0, 'float32x2'),
              attr(1, 8, 'float32x3'),
              attr(2, 20, 'float32'),
              attr(3, 24, 'float32'),
            ],
          },
        ],
      },
      fragment: { module: worldModule, entryPoint: 'fragment_main', targets: MRT_TARGETS },
      primitive: { topology: 'triangle-list' },
    });

    const lineModule = device.createShaderModule({ code: lineRenderShader });
    this.#lineRenderPipeline = device.createRenderPipeline({
      label: 'HRC-LineRender',
      layout: 'auto',
      vertex: {
        module: lineModule,
        entryPoint: 'vertex_main',
        buffers: [
          {
            arrayStride: 40,
            stepMode: 'instance',
            attributes: [
              attr(0, 0, 'float32x2'),
              attr(1, 8, 'float32x2'),
              attr(2, 16, 'float32x3'),
              attr(3, 28, 'float32'),
              attr(4, 32, 'float32'),
              attr(5, 36, 'float32'),
            ],
          },
        ],
      },
      fragment: { module: lineModule, entryPoint: 'fragment_main', targets: MRT_TARGETS },
      primitive: { topology: 'triangle-list' },
    });

    this.#bounceComputePipeline = computePipeline(device, 'HRC-BounceCompute', bounceComputeShader);
    const cascadePipelines = (mode: ChannelMode): CascadePipelines => ({
      seed: [0, 1].map((l) => computePipeline(device, `HRC-RaySeed${l}-${mode}`, raySeedShader(mode, l))),
      extend: computePipeline(device, `HRC-RayExtend-${mode}`, rayExtendShader(mode)),
      merge: computePipeline(device, `HRC-CascadeMerge-${mode}`, cascadeMergeShader(mode)),
      mergeRead: Object.fromEntries(
        Object.entries(MERGE_READ_SHAPES)
          .filter(([, r]) => !r.sg || device.features.has('subgroups'))
          // the subgroup halo layout assumes 32-wide subgroups with lanes in invocation order
          .filter(([, r]) => r.parity !== 'sghalo' || (device.adapterInfo?.subgroupMinSize === 32 && device.adapterInfo?.subgroupMaxSize === 32))
          .map(([k, r]) => [k, computePipeline(device, `HRC-CascadeMergeRead${k}-${mode}`, cascadeMergeShader(mode, r))]),
      ),
      blur: computePipeline(device, `HRC-FluenceBlur-${mode}`, fluenceBlurShader(mode)),
    });
    this.#cascadePipelines = { rgb: cascadePipelines('rgb'), mono: cascadePipelines('mono') };
    this.#renderPipeline = fullscreenBlit('HRC-Blit', blitShader, this.#presentationFormat);
    this.#ptPipeline = computePipeline(device, 'PT-PathTrace', pathTraceShader);
    this.#exactPipeline = computePipeline(device, 'Exact-Reference', exactShader);
    this.#emitPipeline = computePipeline(device, 'Ratio-Emit', freeEmitShader);
    this.#freeFieldPipeline = computePipeline(device, 'Ratio-FreeField', freeFieldDirectShader('rgb'));
    this.#combinePipeline = computePipeline(device, 'Ratio-Combine', combineShader('rgb'));
    this.#fftGatherPipeline = computePipeline(device, 'Ratio-FFTGather', fftGatherShader('rgb'));
    this.#emissionPipeline = computePipeline(device, 'Ratio-Emission', emissionImageShader);
    this.#transportPipeline = computePipeline(device, 'Ratio-TransportWorld', transportWorldShader);
    this.#nearPipeline = computePipeline(device, 'Ratio-Near', freeNearShader('rgb'));

    this.#createStaticBindGroups();
  }

  #createStaticBindGroups() {
    const device = this.#device;
    for (const c of this.#allCascades) this.#createCascadeBindGroups(c);
    if (this.#freeCascade) {
      const occ = this.#cascades[0];
      this.#emitBindGroup = bg(
        device,
        this.#emitPipeline.getBindGroupLayout(0),
        this.#worldTextureView,
        this.#bounceTextureView,
        { buffer: this.#emitterCounter },
        { buffer: this.#emitterBuffer },
        { buffer: this.#emitParamsBuffer },
      );
      this.#freeFieldBindGroup = bg(
        device,
        this.#freeFieldPipeline.getBindGroupLayout(0),
        { buffer: this.#emitterCounter },
        { buffer: this.#emitterBuffer },
        { buffer: this.#freeFieldBuffer },
        { buffer: this.#freeParamsBuffer },
      );
      this.#combineBindGroup = bg(
        device,
        this.#combinePipeline.getBindGroupLayout(0),
        { buffer: occ.fluenceBuffer },
        { buffer: this.#freeCascade.fluenceBuffer },
        { buffer: this.#freeFieldBuffer },
        occ.outputView,
        { buffer: this.#combineParamsBuffer },
        this.#worldTextureView,
      );
      this.#transportBindGroup = bg(device, this.#transportPipeline.getBindGroupLayout(0), this.#worldTextureView, this.#transportTextureView);
      this.#emissionBindGroup = bg(
        device,
        this.#emissionPipeline.getBindGroupLayout(0),
        this.#transportTextureView,
        this.#bounceTextureView,
        this.#emissionTextureView,
        { buffer: this.#emissionParamsBuffer },
      );
      this.#nearBindGroups = [0, 1, 2, 3].map((phase) =>
        bg(
          device,
          this.#nearPipeline.getBindGroupLayout(0),
          this.#emissionTextureView,
          { buffer: this.#nearTapsBuffer },
          { buffer: this.#freeFieldBuffer },
          { buffer: this.#nearParamsBuffer, offset: phase * 256, size: this.#nearParamsView.arrayBuffer.byteLength },
        ),
      );
      if (this.#fftBuffers) this.#createFFTPasses(occ);
    }

    this.#blitBindGroup = bg(
      device,
      this.#renderPipeline.getBindGroupLayout(0),
      this.#fluenceTextureView,
      this.#worldTextureView,
      { buffer: this.#blitParamsBuffer },
      this.#linearSampler,
      this.#ptAccumTextureViews[0],
    );

    this.#bounceBindGroup = bg(
      device,
      this.#bounceComputePipeline.getBindGroupLayout(0),
      this.#fluenceTextureView,
      this.#worldTextureView,
      this.#materialTextureView,
      this.#bounceTextureView,
      { buffer: this.#bounceParamsBuffer },
    );
  }

  #createCascadeBindGroups(c: Cascade) {
    const device = this.#device;
    const nc = c.numCascades;
    const pipes = this.#cascadePipelines[c.mode];
    const extLayout = pipes.extend.getBindGroupLayout(0);
    const mergeLayout = pipes.merge.getBindGroupLayout(0);
    const seedPS = this.#seedParamsView.arrayBuffer.byteLength;
    const extPS = this.#extendParamsView.arrayBuffer.byteLength;
    const mergePS = this.#mergeParamsView.arrayBuffer.byteLength;

    // seed bind groups per (direction, level ∈ {0, 1}) — slot dir * 2 + level
    const seedWorld = c.fluenceDirs === 4 ? this.#transportTextureView : this.#worldTextureView;
    c.seedBindGroups = [0, 1, 2, 3].flatMap((dir) =>
      [0, 1].map((level) =>
        bg(
          device,
          pipes.seed[level].getBindGroupLayout(0),
          seedWorld,
          this.#bounceTextureView,
          { buffer: c.rayBuffers[Math.min(level, c.rayBuffers.length - 1)] },
          { buffer: c.seedParamsBuffer, offset: (dir * 2 + level) * 256, size: seedPS },
        ),
      ),
    );

    // Extend params depend only on the frustum shape (pc, sc), not the direction.
    // cfg 0 = E/W (probesX, probesY), cfg 1 = N/S (probesY, probesX). Indexed in #runCascade as dir & 1.
    c.extendBindGroups = [];
    for (let cfg = 0; cfg < 2; cfg++) {
      for (let level = 1; level < nc; level++) {
        c.extendBindGroups.push(
          bg(
            device,
            extLayout,
            { buffer: c.rayBuffers[level - 1] },
            { buffer: c.rayBuffers[level] },
            { buffer: c.extendParamsBuffer, offset: (cfg * (nc - 1) + (level - 1)) * 256, size: extPS },
          ),
        );
      }
    }

    c.mergeBindGroups = [];
    for (let dir = 0; dir < 4; dir++) {
      const dirBGs: GPUBindGroup[] = [];
      let readIdx = 1,
        writeIdx = 0;
      for (let k = 0; k < nc; k++) {
        const level = nc - 1 - k;
        dirBGs.push(
          bg(
            device,
            mergeLayout,
            { buffer: c.rayBuffers[level] },
            { buffer: c.mergeBuffers[readIdx] },
            { buffer: c.mergeBuffers[writeIdx] },
            { buffer: c.mergeParamsBuffer, offset: (dir * nc + level) * 256, size: mergePS },
            { buffer: c.fluenceBuffer },
            c.skyPrefixSumTextureView,
            { buffer: c.coneArcBuffer },
          ),
        );
        [readIdx, writeIdx] = [writeIdx, readIdx];
      }
      c.mergeBindGroups.push(dirBGs);
    }
    c.mergeReadBindGroups = {};
    for (const shape of Object.keys(pipes.mergeRead).map(Number)) {
    const readLayout = pipes.mergeRead[shape].getBindGroupLayout(0);
    const groups: GPUBindGroup[][] = [];
    for (let dir = 0; dir < 4; dir++) {
      const dirBGs: GPUBindGroup[] = [];
      let readIdx = 1,
        writeIdx = 0;
      for (let k = 0; k < nc; k++) {
        const level = nc - 1 - k;
        dirBGs.push(
          bg(
            device,
            readLayout,
            { buffer: c.rayBuffers[level] },
            { buffer: c.mergeBuffers[readIdx] },
            { buffer: c.mergeBuffers[writeIdx] },
            { buffer: c.mergeParamsBuffer, offset: (dir * nc + level) * 256, size: mergePS },
            { buffer: c.fluenceBuffer },
            c.skyPrefixSumTextureView,
            { buffer: c.coneArcBuffer },
            { buffer: c.rayBuffers[Math.min(level + 1, nc - 1)] },
            { buffer: c.readWeightBuffer },
          ),
        );
        [readIdx, writeIdx] = [writeIdx, readIdx];
      }
      groups.push(dirBGs);
    }
    c.mergeReadBindGroups[shape] = groups;
    }

    // Mono blur reconstructs RGB from the preceding (low-res) color cascade.
    const colorView = c.mode === 'mono' ? this.#cascades.find((o) => o.mode === 'rgb')!.outputView : null;
    c.blurBindGroup = bg(
      device,
      pipes.blur.getBindGroupLayout(0),
      { buffer: c.fluenceBuffer },
      c.outputView,
      { buffer: c.blurParamsBuffer },
      this.#worldTextureView,
      ...(colorView ? [colorView, this.#linearSampler] : []),
    );
  }

  // The FFT pass list (NOTES §47): forward (emission → specA, specB), then per direction and pack an inverse with the
  // kernel multiply fused into its row pass, gathered into the packed free field. Params live in 256-B slots.
  #createFFTPasses(occ: Cascade) {
    const device = this.#device;
    const P = this.#fftP;
    const b = this.#fftBuffers!;
    const coarseX = Math.ceil(occ.probesX / 2);
    const coarseY = Math.ceil(occ.probesY / 2);
    if (!this.#fftPipeline || this.#fftPipeline.label !== `Ratio-FFT-${P}`) {
      this.#fftPipeline = computePipeline(device, `Ratio-FFT-${P}`, fftShader(P));
    }
    const layout = this.#fftPipeline.getBindGroupLayout(0);
    const PS = this.#fftParamsView.arrayBuffer.byteLength;
    let slot = 0;
    const pass = (
      src: GPUBuffer,
      dst: GPUBuffer,
      kspec: GPUBuffer,
      p: { axis: number; mode: number; dir?: number; pack?: number; inverse?: number; scale?: number },
    ) => {
      this.#fftParamsView.set({
        axis: p.axis,
        mode: p.mode,
        dir: p.dir ?? 0,
        pack: p.pack ?? 0,
        inverse: p.inverse ?? 0,
        coarseX,
        coarseY,
        D: 2,
        scale: p.scale ?? 1,
        cutR: NEAR_R,
        cutRho: NEAR_RHO,
        sig2: NEAR_SIG2,
      });
      device.queue.writeBuffer(this.#fftParamsBuffer, slot * 256, this.#fftParamsView.arrayBuffer);
      const bgp = bg(
        device,
        layout,
        { buffer: src },
        { buffer: dst },
        { buffer: kspec },
        { buffer: this.#fftParamsBuffer, offset: slot * 256, size: PS },
        this.#emissionTextureView,
        { buffer: b.twiddle },
      );
      slot++;
      return { bg: bgp, lines: P };
    };
    // kernel spectrum (once): kernel image → rows → cols
    this.#fftKernelPasses = [pass(b.work, b.scratch, b.work, { axis: 0, mode: 3 }), pass(b.scratch, b.kspec, b.work, { axis: 1, mode: 0 })];
    // per frame
    const passes: { bg: GPUBindGroup; lines: number }[] = [];
    passes.push(pass(b.work, b.scratch, b.kspec, { axis: 0, mode: 1, pack: 0 }), pass(b.scratch, b.specA, b.kspec, { axis: 1, mode: 0 }));
    passes.push(pass(b.work, b.scratch, b.kspec, { axis: 0, mode: 1, pack: 1 }), pass(b.scratch, b.specB, b.kspec, { axis: 1, mode: 0 }));
    const gathers: GPUBindGroup[] = [];
    const GPS = this.#fftGatherParamsView.arrayBuffer.byteLength;
    let gslot = 0;
    for (let dir = 0; dir < 4; dir++) {
      for (let pack = 0; pack < 2; pack++) {
        passes.push(
          pass(pack === 0 ? b.specA : b.specB, b.scratch, b.kspec, { axis: 0, mode: 2, dir, inverse: 1 }),
          pass(b.scratch, b.work, b.kspec, { axis: 1, mode: 0, inverse: 1, scale: 1 / (P * P) }),
        );
        this.#fftGatherParamsView.set({ probesX: occ.probesX, probesY: occ.probesY, stride: occ.fluenceStride, P, dir, pack, coarseX, coarseY });
        device.queue.writeBuffer(this.#fftGatherParamsBuffer, gslot * 256, this.#fftGatherParamsView.arrayBuffer);
        gathers.push(
          bg(
            device,
            this.#fftGatherPipeline.getBindGroupLayout(0),
            { buffer: b.work },
            { buffer: this.#freeFieldBuffer },
            { buffer: this.#fftGatherParamsBuffer, offset: gslot * 256, size: GPS },
          ),
        );
        gslot++;
      }
    }
    this.#fftPasses = passes;
    this.#fftGathers = gathers;
  }

  #runFFTFreeField(encoder: GPUCommandEncoder) {
    const pipe = this.#fftPipeline!;
    {
      const occ = this.#cascades[0];
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('emission') });
      pass.setPipeline(this.#emissionPipeline);
      pass.setBindGroup(0, this.#emissionBindGroup!);
      pass.dispatchWorkgroups(Math.ceil(occ.probesX / 16), Math.ceil(occ.probesY / 16));
      pass.end();
    }
    if (!this.#fftKernelReady) {
      this.#fftKernelReady = true;
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('fft-kernel') });
      pass.setPipeline(pipe);
      for (const p of this.#fftKernelPasses) {
        pass.setBindGroup(0, p.bg);
        pass.dispatchWorkgroups(p.lines);
      }
      pass.end();
    }
    const passes = this.#fftPasses;
    const occ = this.#cascades[0];
    // forward: rows (emission load) + cols, twice; then per (dir, pack): rows (× kernel) + cols + gather.
    // Rows and columns are timed separately (the column pass is strided).
    const run = (label: string, idx: number[], gather: number) => {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass(label) });
      pass.setPipeline(pipe);
      for (const i of idx) {
        pass.setBindGroup(0, passes[i].bg);
        pass.dispatchWorkgroups(passes[i].lines);
      }
      if (gather >= 0) {
        pass.setPipeline(this.#fftGatherPipeline);
        pass.setBindGroup(0, this.#fftGathers[gather]);
        pass.dispatchWorkgroups(Math.ceil(occ.probesX / 16), Math.ceil(occ.probesY / 16));
      }
      pass.end();
    };
    run('fft-rows', [0], -1);
    run('fft-cols', [1], -1);
    run('fft-rows', [2], -1);
    run('fft-cols', [3], -1);
    for (let g = 0; g < 8; g++) {
      run('fft-rows', [4 + g * 2], -1);
      run('fft-cols', [5 + g * 2], g);
    }
    {
      const occ = this.#cascades[0];
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('near') });
      pass.setPipeline(this.#nearPipeline);
      for (let phase = 0; phase < 4; phase++) {
        pass.setBindGroup(0, this.#nearBindGroups[phase]);
        pass.dispatchWorkgroups(Math.ceil(occ.probesX / 32), Math.ceil(occ.probesY / 32));
      }
      pass.end();
    }
  }

  #destroyCascade(c: Cascade) {
    if (!c.sharesBuffers) {
      c.rayBuffers.forEach((b) => b.destroy());
      c.mergeBuffers.forEach((b) => b.destroy());
    }
    c.fluenceBuffer.destroy();
    c.ownedOutput?.destroy();
    c.skyPrefixSumTexture.destroy();
    c.coneArcBuffer.destroy();
    c.readWeightBuffer.destroy();
    c.seedParamsBuffer.destroy();
    c.extendParamsBuffer.destroy();
    c.mergeParamsBuffer.destroy();
    c.blurParamsBuffer.destroy();
  }

  #destroyResources() {
    this.#worldTexture?.destroy();
    this.#materialTexture?.destroy();
    this.#paintTexture?.destroy();
    this.#paintMaterialTexture?.destroy();
    this.#cascades.forEach((c) => this.#destroyCascade(c));
    this.#cascades = [];
    if (this.#freeCascade) {
      this.#destroyCascade(this.#freeCascade);
      this.#freeCascade = null;
      this.#emitterCounter?.destroy();
      this.#emitterBuffer?.destroy();
      this.#freeFieldBuffer?.destroy();
      this.#emitParamsBuffer?.destroy();
      this.#freeParamsBuffer?.destroy();
      this.#combineParamsBuffer?.destroy();
      this.#emissionTexture?.destroy();
      this.#transportTexture?.destroy();
      this.#emissionParamsBuffer?.destroy();
      this.#nearTapsBuffer?.destroy();
      this.#nearParamsBuffer?.destroy();
      if (this.#fftBuffers) {
        Object.values(this.#fftBuffers).forEach((b) => b.destroy());
        this.#fftBuffers = null;
        this.#fftParamsBuffer?.destroy();
        this.#fftGatherParamsBuffer?.destroy();
      }
    }
    this.#fluenceTexture?.destroy();
    this.#bounceZeroBuffer?.destroy();
    this.#bounceTexture?.destroy();
    this.#skyTexture?.destroy();
    this.#ptAccumTextures?.forEach((t) => t.destroy());
    this.#ptParamsBuffer?.destroy();
    this.#blitParamsBuffer?.destroy();
    this.#bounceParamsBuffer?.destroy();
  }

  // ── Shape / line data ──

  #shapeVertexData?: Float32Array<ArrayBuffer>;

  #updateShapeData() {
    const shapes = this.shapes;
    const count = shapes.length;
    this.#shapeCount = count;
    if (count === 0) return;

    const FLOATS_PER_VERTEX = 7;
    const VERTICES_PER_SHAPE = 6;
    const STRIDE = FLOATS_PER_VERTEX * VERTICES_PER_SHAPE;
    const needed = count * STRIDE;
    if (!this.#shapeVertexData || this.#shapeVertexData.length < needed) {
      this.#shapeVertexData = new Float32Array(needed);
    }
    const verts = this.#shapeVertexData;

    const cw = this.#canvas.width;
    const ch = this.#canvas.height;
    const invCw2 = 2 / cw;
    const invCh2 = 2 / ch;
    // Scene units -> canvas pixels: the scene square spans the canvas's shorter side, centred.
    const k = Math.min(cw, ch) / SCENE_SPAN;
    const ox = cw / 2;
    const oy = ch / 2;

    let i = 0;
    for (const shape of shapes) {
      // At coarse N a shape still covers at least one pixel.
      const sw = Math.max(shape.width * k, 1);
      const sh = Math.max(shape.height * k, 1);
      const sx = ox + shape.x * k;
      const sy = oy + shape.y * k;
      const rot = shape.rotation ?? 0;
      const [r, g, b] = shape.rgb ?? [0, 0, 0];
      const opacity = shape.opacity ?? 1;
      const albedo = shape.albedo ?? 0;

      const mx = sx + sw * 0.5;
      const my = sy + sh * 0.5;
      const cos = Math.cos(rot);
      const sin = Math.sin(rot);

      const hlx = -sw * 0.5,
        hly = -sh * 0.5;
      const hrx = sw * 0.5,
        hry = -sh * 0.5;
      const blx = -sw * 0.5,
        bly = sh * 0.5;
      const brx = sw * 0.5,
        bry = sh * 0.5;

      const tlX = (mx + hlx * cos - hly * sin) * invCw2 - 1;
      const tlY = 1 - (my + hlx * sin + hly * cos) * invCh2;
      const trX = (mx + hrx * cos - hry * sin) * invCw2 - 1;
      const trY = 1 - (my + hrx * sin + hry * cos) * invCh2;
      const blX = (mx + blx * cos - bly * sin) * invCw2 - 1;
      const blY = 1 - (my + blx * sin + bly * cos) * invCh2;
      const brX = (mx + brx * cos - bry * sin) * invCw2 - 1;
      const brY = 1 - (my + brx * sin + bry * cos) * invCh2;

      const off = i * STRIDE;
      verts[off] = tlX;
      verts[off + 1] = tlY;
      verts[off + 2] = r;
      verts[off + 3] = g;
      verts[off + 4] = b;
      verts[off + 5] = opacity;
      verts[off + 6] = albedo;
      verts[off + 7] = trX;
      verts[off + 8] = trY;
      verts[off + 9] = r;
      verts[off + 10] = g;
      verts[off + 11] = b;
      verts[off + 12] = opacity;
      verts[off + 13] = albedo;
      verts[off + 14] = blX;
      verts[off + 15] = blY;
      verts[off + 16] = r;
      verts[off + 17] = g;
      verts[off + 18] = b;
      verts[off + 19] = opacity;
      verts[off + 20] = albedo;
      verts[off + 21] = trX;
      verts[off + 22] = trY;
      verts[off + 23] = r;
      verts[off + 24] = g;
      verts[off + 25] = b;
      verts[off + 26] = opacity;
      verts[off + 27] = albedo;
      verts[off + 28] = brX;
      verts[off + 29] = brY;
      verts[off + 30] = r;
      verts[off + 31] = g;
      verts[off + 32] = b;
      verts[off + 33] = opacity;
      verts[off + 34] = albedo;
      verts[off + 35] = blX;
      verts[off + 36] = blY;
      verts[off + 37] = r;
      verts[off + 38] = g;
      verts[off + 39] = b;
      verts[off + 40] = opacity;
      verts[off + 41] = albedo;
      i++;
    }

    this.#shapeDataBuffer = uploadVertexData(this.#device, this.#shapeDataBuffer, verts, needed);
  }

  #updateLineBuffer() {
    if (!this.#device || this.#lines.length === 0) {
      this.#lineCount = 0;
      return;
    }
    const count = this.#lines.length;
    const FPL = 10;
    if (!this.#lineInstanceBuffer || this.#lineInstanceCapacity < count) {
      this.#lineInstanceBuffer?.destroy();
      this.#lineInstanceCapacity = Math.max(count, 256);
      this.#lineInstanceBuffer = this.#device.createBuffer({
        size: this.#lineInstanceCapacity * FPL * 4,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    const data = new Float32Array(count * FPL);
    for (let i = 0; i < count; i++) data.set(this.#lines[i], i * FPL);
    this.#device.queue.writeBuffer(this.#lineInstanceBuffer, 0, data);
    this.#lineCount = count;
  }

  #updateMouseLightBuffer() {
    if (!this.#device) return;
    let { x, y } = this.#mousePosition;
    const { r, g, b } = this.#mouseLightColor;
    const op = this.#mouseLightOpacity;
    const al = this.#mouseLightAlbedo;
    const verts: number[] = [];

    const pp = this.#pp;
    if (pp) {
      const cw = this.#canvas.width,
        ch = this.#canvas.height;
      const cx = Math.floor(x),
        cy = Math.floor(y);
      const bs = Math.max(1, this.#mouseLightRadius * 2);
      const blo = -Math.floor((bs - 1) / 2),
        bhi = Math.floor(bs / 2);
      for (let by = blo; by <= bhi; by++) {
        for (let bx = blo; bx <= bhi; bx++) {
          const px = cx + bx,
            py = cy + by;
          if (px >= 0 && py >= 0 && px < pp && py < pp) {
            const x0 = (px / cw) * 2 - 1,
              y0 = 1 - (py / ch) * 2;
            const x1 = ((px + 1) / cw) * 2 - 1,
              y1 = 1 - ((py + 1) / ch) * 2;
            verts.push(x0, y0, r, g, b, op, al, x1, y0, r, g, b, op, al, x0, y1, r, g, b, op, al);
            verts.push(x1, y0, r, g, b, op, al, x1, y1, r, g, b, op, al, x0, y1, r, g, b, op, al);
          }
        }
      }
    } else {
      const SEGS = 12;
      const rad = this.#mouseLightRadius;
      const toClipX = (px: number) => (px / this.#canvas.width) * 2 - 1;
      const toClipY = (py: number) => 1 - (py / this.#canvas.height) * 2;
      const rx = (rad / this.#canvas.width) * 2;
      const ry = (rad / this.#canvas.height) * 2;
      const cx = toClipX(x);
      const cy = toClipY(y);
      for (let i = 0; i < SEGS; i++) {
        const a0 = (i / SEGS) * Math.PI * 2;
        const a1 = ((i + 1) / SEGS) * Math.PI * 2;
        verts.push(cx, cy, r, g, b, op, al);
        verts.push(cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, r, g, b, op, al);
        verts.push(cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry, r, g, b, op, al);
      }
    }

    this.#mouseLightVertexCount = verts.length / 7;
    this.#mouseLightBuffer = uploadVertexData(this.#device, this.#mouseLightBuffer, new Float32Array(verts));
  }

  // ── Render loop ──

  #startAnimationLoop() {
    const render = (now: number) => {
      if (!this.#isRunning) return;
      if (this.paused) {
        this.#animationFrame = requestAnimationFrame(render);
        return;
      }
      this.#updateFrameTiming(now);
      const jsStart = performance.now();
      this.#renderFrame();
      this.#jsTimeMs = performance.now() - jsStart;
      this.#animationFrame = requestAnimationFrame(render);
    };
    this.#animationFrame = requestAnimationFrame(render);
  }

  #renderFrame() {
    this.#updateShapeData();
    if (this.#lineBufferDirty) {
      this.#lineBufferDirty = false;
      this.#updateLineBuffer();
    }
    if (this.#mouseDirty) {
      this.#mouseDirty = false;
      this.#updateMouseLightBuffer();
    }

    const { width, height } = this.#canvas;
    const device = this.#device;

    this.#blitParamsView.set({
      exposure: this.exposure,
      screenW: width,
      screenH: height,
      debugMode: this.debugMode >= 3 ? this.debugMode - 2 : this.debugMode, // 3, 4: exact reference in place of PT
      falseColor: this.falseColor ? 1.0 : 0.0,
      banded: this.banded ? 1.0 : 0.0,
      pad1: 0,
      pad2: 0,
    });
    device.queue.writeBuffer(this.#blitParamsBuffer, 0, this.#blitParamsView.arrayBuffer);

    const encoder = device.createCommandEncoder();
    this.#tsBeginFrame();

    // ── Step 1: Render world textures (world + material) ──
    const paint = this.#paint;
    const hasScene = !!(paint && this.#paintTexture && paint.size === width && paint.size === height);
    if (hasScene) {
      const d = this.#paintDirty;
      if (d) {
        this.#paintDirty = null;
        const [x0, y0, x1, y1] = d;
        const size = { width: x1 - x0, height: y1 - y0 };
        device.queue.writeTexture({ texture: this.#paintTexture!, origin: { x: x0, y: y0 } }, paint!.rgba,
          { offset: (y0 * width + x0) * 8, bytesPerRow: width * 8 }, size);
        device.queue.writeTexture({ texture: this.#paintMaterialTexture!, origin: { x: x0, y: y0 } }, paint!.albedo,
          { offset: y0 * width + x0, bytesPerRow: width }, size);
      }
      encoder.copyTextureToTexture({ texture: this.#paintTexture! }, { texture: this.#worldTexture }, { width, height });
      encoder.copyTextureToTexture({ texture: this.#paintMaterialTexture! }, { texture: this.#materialTexture }, { width, height });
    }
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.#worldTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: hasScene ? 'load' : 'clear',
            storeOp: 'store',
          },
          {
            view: this.#materialTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: hasScene ? 'load' : 'clear',
            storeOp: 'store',
          },
        ],
        timestampWrites: this.#tsPass('world'),
      });
      pass.setPipeline(this.#worldRenderPipeline);
      if (this.#shapeDataBuffer && this.#shapeCount > 0) {
        pass.setVertexBuffer(0, this.#shapeDataBuffer);
        pass.draw(this.#shapeCount * 6);
      }
      if (this.#lineInstanceBuffer && this.#lineCount > 0) {
        pass.setPipeline(this.#lineRenderPipeline);
        if (!this.#lineBindGroup) {
          this.#lineUBO?.destroy();
          this.#lineUBO = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(this.#lineUBO, 0, new Float32Array([width, height]));
          this.#lineBindGroup = bg(device, this.#lineRenderPipeline.getBindGroupLayout(0), { buffer: this.#lineUBO });
        }
        pass.setBindGroup(0, this.#lineBindGroup);
        pass.setVertexBuffer(0, this.#lineInstanceBuffer);
        pass.draw(6, this.#lineCount);
        pass.setPipeline(this.#worldRenderPipeline);
      }
      const { r: mr, g: mg, b: mb } = this.#mouseLightColor;
      if (
        this.#mouseLightBuffer &&
        this.#mouseLightVertexCount > 0 &&
        (mr > 0 || mg > 0 || mb > 0 || this.#mouseLightOpacity > 0)
      ) {
        pass.setVertexBuffer(0, this.#mouseLightBuffer);
        pass.draw(this.#mouseLightVertexCount);
      }
      pass.end();
    }

    // ── Step 1.5: Bounce compute (reads previous frame's fluence at probe resolution) ──
    if (!this.bounces && this.#lastFluenceReady) {
      this.#lastFluenceReady = false;
      encoder.copyBufferToTexture(
        {
          buffer: this.#bounceZeroBuffer,
          bytesPerRow: Math.ceil((this.#probesX * 8) / 256) * 256,
          rowsPerImage: this.#probesY,
        },
        { texture: this.#bounceTexture },
        { width: this.#probesX, height: this.#probesY },
      );
    }
    if (this.bounces && this.#lastFluenceReady) {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('bounce') });
      pass.setPipeline(this.#bounceComputePipeline);
      pass.setBindGroup(0, this.#bounceBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.#probesX / WG_BOUNCE[0]), Math.ceil(this.#probesY / WG_BOUNCE[1]));
      pass.end();
    }

    // ── Step 2: HRC cascade processing ──
    // Run the per-frustum algorithm once per cardinal direction, then
    // blur the fluence SSBO into a texture. In split color mode the low-res
    // color cascade runs first so the mono blur can reconstruct from it.
    // E/W share frustum shape (probesX × probesY), N/S share (probesY × probesX).
    const split = this.#cascades.length > 1;
    if (this.#freeCascade) {
      this.#runRatio(encoder, width, height);
    } else {
      for (const c of this.#cascades) {
        const tag = split ? `-${c.mode}` : '';
        encoder.clearBuffer(c.fluenceBuffer);
        for (let dir = 0; dir < 4; dir++) {
          const pc = dir & 1 ? c.probesY : c.probesX;
          const sc = dir & 1 ? c.probesX : c.probesY;
          this.#runCascade(encoder, c, dir, pc, sc, tag);
        }

        const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass(`blur${tag}`) });
        pass.setPipeline(this.#cascadePipelines[c.mode].blur);
        pass.setBindGroup(0, c.blurBindGroup);
        pass.dispatchWorkgroups(Math.ceil(c.probesX / WG_BLUR[0]), Math.ceil(c.probesY / WG_BLUR[1]));
        pass.end();
      }
    }

    this.#lastFluenceReady = true;

    // ── Step 2.75: Run PT alongside HRC for PT or PT-diff modes ──
    if (this.debugMode === 1 || this.debugMode === 2) {
      this.#runPathTraceCompute(encoder, width, height);
    }

    // ── Step 3: Final blit ──
    let blitBG = this.#blitBindGroup;
    if (this.debugMode >= 1 && this.debugMode <= 4) {
      const ptWriteIdx = this.#ptFrameIndex % 2;
      const ex = this.#exactCurrent;
      const refView = this.debugMode >= 3 && ex && ex.w === width && ex.h === height ? ex.outView : this.#ptAccumTextureViews[ptWriteIdx];
      blitBG = bg(
        device,
        this.#renderPipeline.getBindGroupLayout(0),
        this.#fluenceTextureView,
        this.#worldTextureView,
        { buffer: this.#blitParamsBuffer },
        this.#linearSampler,
        refView,
      );
    }
    this.#blitToScreen(encoder, this.#renderPipeline, blitBG);

    this.#resolveTimestamps(encoder);
    this.#submitAndCapture(device, encoder);
    this.#readTimestamps();
  }

  // Tier 1 (NOTES §47): the cascade twice (scene, transparent scene), the exact free field, then the combine pass.
  #runRatio(encoder: GPUCommandEncoder, width: number, height: number) {
    const occ = this.#cascades[0];
    const free = this.#freeCascade!;
    const device = this.#device;
    {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('transport') });
      pass.setPipeline(this.#transportPipeline);
      pass.setBindGroup(0, this.#transportBindGroup!);
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      pass.end();
    }
    for (const c of [occ, free]) {
      encoder.clearBuffer(c.fluenceBuffer);
      const tag = c.freeWorld ? '-free' : '-occ';
      for (let dir = 0; dir < 4; dir++) {
        const pc = dir & 1 ? c.probesY : c.probesX;
        const sc = dir & 1 ? c.probesX : c.probesY;
        this.#runCascade(encoder, c, dir, pc, sc, tag);
      }
    }
    // exact free field: FFT convolution with the quadrant kernels, or (validation) the direct sum over emitters
    if (this.#fftBuffers && this.freeMethod !== 'direct') {
      this.#runFFTFreeField(encoder);
    } else {
      encoder.clearBuffer(this.#emitterCounter);
      {
        const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('emit') });
        pass.setPipeline(this.#emitPipeline);
        pass.setBindGroup(0, this.#emitBindGroup!);
        pass.dispatchWorkgroups(Math.ceil(occ.probesX / 16), Math.ceil(occ.probesY / 16));
        pass.end();
      }
      {
        const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('freefield') });
        pass.setPipeline(this.#freeFieldPipeline);
        pass.setBindGroup(0, this.#freeFieldBindGroup!);
        pass.dispatchWorkgroups(Math.ceil(occ.probesX / 16), Math.ceil(occ.probesY / 16));
        pass.end();
      }
    }
    const sky = this.#skyQuadrants();
    this.#combineParamsView.set({
      w: occ.probesX,
      h: occ.probesY,
      stride: occ.fluenceStride,
      screenW: width,
      screenH: height,
      view: this.ratioView,
      pad0: 0,
      pad1: 0,
      sky: [sky.slice(0, 4), sky.slice(4, 8), sky.slice(8, 12), sky.slice(12, 16)],
    });
    device.queue.writeBuffer(this.#combineParamsBuffer, 0, this.#combineParamsView.arrayBuffer);
    {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass('combine') });
      pass.setPipeline(this.#combinePipeline);
      pass.setBindGroup(0, this.#combineBindGroup!);
      pass.dispatchWorkgroups(Math.ceil(occ.probesX / WG_BLUR[0]), Math.ceil(occ.probesY / WG_BLUR[1]));
      pass.end();
    }
  }

  // HRC is a per-frustum algorithm: given (probeCount × sliceCount), run
  // seed → extend → merge to produce fluence contributions. The `dir`
  // parameter (0=E, 1=N, 2=W, 3=S) selects pre-built bind groups but
  // does not affect the algorithm structure.
  #runCascade(encoder: GPUCommandEncoder, c: Cascade, dir: number, pc: number, sc: number, tag: string) {
    const nc = c.numCascades;
    const pipes = this.#cascadePipelines[c.mode];
    const dn = ['E', 'N', 'W', 'S'][dir];
    const cfg = dir & 1;

    // Every pass walks slices (64-wide workgroups along x) over the slice-fastest buffers.
    // Phase A: Seed — trace T_0 and T_1 directly as strips (paper §4.2, Alg. 1 line 3), one invocation per probe
    const seedLevels = nc > 1 ? 2 : 1;
    {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass(`${dn}-seed${tag}`) });
      for (let level = 0; level < seedLevels; level++) {
        pass.setPipeline(pipes.seed[level]);
        pass.setBindGroup(0, c.seedBindGroups[dir * 2 + level]);
        pass.dispatchWorkgroups(Math.ceil(sc / 64), ceilDiv(pc, 1 << level));
      }
      pass.end();
    }

    // Phase B: Extend / "Merge Up" — build T_2..T_{N-1} bottom-up (Eq. 18, 20), one invocation per ray pair
    {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass(`${dn}-extend${tag}`) });
      pass.setPipeline(pipes.extend);
      for (let level = seedLevels; level < nc; level++) {
        pass.setBindGroup(0, c.extendBindGroups[cfg * (nc - 1) + (level - 1)]);
        pass.dispatchWorkgroups(Math.ceil(sc / 64), ceilDiv(pc, 1 << level) * ((1 << (level - 1)) + 1));
      }
      pass.end();
    }

    // Phase C: "Merge Down" — resolve R_{N-1}..R_0 top-down (Eq. 14-15), write fluence at level 0;
    // one invocation per cone pair (per probe at level 0)
    {
      const pass = encoder.beginComputePass({ timestampWrites: this.#tsPass(`${dn}-merge${tag}`) });
      const read = this.parityRead ? (10 in pipes.mergeRead ? 10 : 8) : this.mergeRead in pipes.mergeRead ? this.mergeRead : 0;
      pass.setPipeline(read ? pipes.mergeRead[read] : pipes.merge);
      for (let k = 0; k < nc; k++) {
        const level = nc - 1 - k;
        pass.setBindGroup(0, (read ? c.mergeReadBindGroups[read] : c.mergeBindGroups)[dir][k]);
        pass.dispatchWorkgroups(Math.ceil(sc / (MERGE_READ_SHAPES[read]?.parity === 'halo' ? 62 : MERGE_READ_SHAPES[read]?.parity === 'sghalo' ? 60 : 64)), ceilDiv(pc, 1 << level) * (level > 0 ? 1 << (level - 1) : 1));
      }
      pass.end();
    }
  }

  #blitToScreen(encoder: GPUCommandEncoder, pipeline: GPURenderPipeline, bindGroup: GPUBindGroup, tsLabel = '') {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.#context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear' as const,
          storeOp: 'store' as const,
        },
      ],
      ...(tsLabel ? { timestampWrites: this.#tsPass(tsLabel) } : {}),
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setViewport(0, 0, this.#canvas.width, this.#canvas.height, 0, 1);
    pass.draw(4);
    pass.end();
  }

  #submitAndCapture(device: GPUDevice, encoder: GPUCommandEncoder) {
    device.queue.submit([encoder.finish()]);

    if (this.#pendingScreenshot) {
      const filename = this.#pendingScreenshot;
      this.#pendingScreenshot = '';
      const tmp = document.createElement('canvas');
      tmp.width = this.#canvas.width;
      tmp.height = this.#canvas.height;
      const ctx2d = tmp.getContext('2d')!;
      ctx2d.drawImage(this.#canvas, 0, 0);
      tmp.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      });
    }
  }

  #runPathTraceCompute(encoder: GPUCommandEncoder, width: number, height: number) {
    const device = this.#device;
    const readIdx = this.#ptFrameIndex % 2;
    const writeIdx = 1 - readIdx;

    this.#ptParamsView.set({
      screenW: width,
      screenH: height,
      frameIndex: this.#ptFrameIndex,
      samplesPerPixel: 16,
      maxBounces: this.bounces ? 2000 : 0,
      pad0: 0,
      pad1: 0,
      pad2: 0,
    });
    device.queue.writeBuffer(this.#ptParamsBuffer, 0, this.#ptParamsView.arrayBuffer);

    // in ratio mode the reference sees the same transport world (shelled opaque emitters) as the cascades
    const ptBG = bg(
      device,
      this.#ptPipeline.getBindGroupLayout(0),
      this.#freeCascade ? this.#transportTextureView : this.#worldTextureView,
      this.#materialTextureView,
      this.#ptAccumTextureViews[readIdx],
      this.#ptAccumTextureViews[writeIdx],
      { buffer: this.#ptParamsBuffer },
      this.#skyTextureView,
    );

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#ptPipeline);
    pass.setBindGroup(0, ptBG);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    this.#ptFrameIndex++;
  }

  // TS/shader coupling: the direction-to-slope formula (2s - N + 1) / N in
  // #computeSkyPrefixSums and the angular weight formula atan2(2s-N+2, N) -
  // atan2(2s-N, N) below must stay in sync with the extend/merge shaders'
  // dirToSliceOffset mapping. If direction indexing changes in a shader,
  // both this method and #computeSkyPrefixSums must be updated to match.
  #uploadStaticParams(width: number, height: number) {
    for (const c of this.#allCascades) this.#uploadCascadeParams(c, width, height);

    this.#bounceParamsView.set({ screenW: width, screenH: height, pad0: 0, pad1: 0 });
    this.#device.queue.writeBuffer(this.#bounceParamsBuffer, 0, this.#bounceParamsView.arrayBuffer);
  }

  #uploadCascadeParams(c: Cascade, width: number, height: number) {
    const device = this.#device;
    const { probesX, probesY } = c;
    const nc = c.numCascades;

    // Seed params: per direction (4) — includes transform matrix
    for (let dir = 0; dir < 4; dir++) {
      const pc = dir & 1 ? probesY : probesX;
      const sc = dir & 1 ? probesX : probesY;
      const dt = dirTransform(dir, width, height, Math.max(probesX, probesY));
      for (let level = 0; level < 2; level++) {
        this.#seedParamsView.set({
          probeCount: pc,
          sliceCount: sc,
          screenW: width,
          screenH: height,
          probeSpacing: dt.probeSpacing,
          freeWorld: c.freeWorld ? 1 : 0,
          transformX: [...dt.transformX, 0],
          transformY: [...dt.transformY, 0],
        });
        device.queue.writeBuffer(c.seedParamsBuffer, (dir * 2 + level) * 256, this.#seedParamsView.arrayBuffer);
      }
    }

    this.#uploadExtendParams(c);

    // Cone arc sizes A_n(i) (paper Eq. 13): shared across all directions.
    // coneArcBase for level l = 2^(l+1) - 2.
    const totalConeArcs = 2 * (1 << nc) - 2;
    const arcData = new Float32Array(totalConeArcs);
    {
      let off = 0;
      for (let level = 0; level < nc; level++) {
        const N = 2 << level;
        for (let s = 0; s < N; s++) {
          arcData[off + s] = Math.atan2(2 * s - N + 2, N) - Math.atan2(2 * s - N, N);
        }
        off += N;
      }
    }
    device.queue.writeBuffer(c.coneArcBuffer, 0, arcData);

    this.#uploadMergeParams(c);

    this.#computeSkyPrefixSums(c);
  }

  // Merge params: per direction (4) × per level (also re-uploaded when the merge's weights change)
  // Extend params: per frustum config (2) — E/W share (probesX, probesY), N/S share (probesY, probesX)
  #uploadExtendParams(c: Cascade) {
    const device = this.#device;
    const { probesX, probesY } = c;
    const nc = c.numCascades;
    for (let cfg = 0; cfg < 2; cfg++) {
      const pc = cfg === 0 ? probesX : probesY;
      const sc = cfg === 0 ? probesY : probesX;
      for (let level = 1; level < nc; level++) {
        const et = this.equalTaps;
        this.#extendParamsView.set({
          probeCount: pc,
          level,
          sliceCount: sc,
          equalTap: this.equalise ? (et ? et.even[Math.min(level, et.even.length - 1)] : 0.16) : 0,
          equalTapOdd: this.equalise && et ? et.odd[Math.min(level, et.odd.length - 1)] : 0,
          equalTapAxis: this.equalise
            ? this.equalAxis
              ? this.equalAxis[Math.min(level, this.equalAxis.length - 1)]
              : et
                ? et.even[Math.min(level, et.even.length - 1)]
                : 0.16
            : 0,
        });
        device.queue.writeBuffer(
          c.extendParamsBuffer,
          (cfg * (nc - 1) + (level - 1)) * 256,
          this.#extendParamsView.arrayBuffer,
        );
      }
    }
  }

  #uploadMergeParams(c: Cascade) {
    const device = this.#device;
    const { probesX, probesY } = c;
    const nc = c.numCascades;
    for (let dir = 0; dir < 4; dir++) {
      const pc = dir & 1 ? probesY : probesX;
      const sc = dir & 1 ? probesX : probesY;
      const fNc = ceilLog2(pc);
      const fMs = nextPow2(pc);
      for (let level = 0; level < nc; level++) {
        const nextNumDirections = 2 << level;
        const top = level === fNc - 1;
        const rp = top ? null : this.mergeReadParams[level < 4 ? String(level) : 'int'];
        const rv = (v?: number[]) => (v ? [v[0], v[1], v[2], v[3], 1 / v[4]] : [0, 0, 0, 0, 1]);
        const [o0, o1, o2, o3, o4] = rv(rp?.o);
        const [e0, e1, e2, e3, e4] = rv(rp?.e);
        this.#mergeParamsView.set({
          probeCount: pc,
          levelProbes: ceilDiv(pc, 1 << level),
          numRays: (1 << level) + 1,
          isTopCascade: top ? 1 : 0,
          fluenceW: probesX,
          fluenceStride: c.fluenceStride,
          skyShift: Math.log2(fMs) - Math.log2(nextNumDirections),
          level,
          coneArcBase: (1 << (level + 1)) - 2,
          skyRow: dir,
          sliceCount: sc,
          mergeInWidth: level < fNc - 1 ? ceilDiv(pc, 1 << (level + 1)) * (1 << (level + 1)) : 0,
          direction: dir,
          fluenceH: probesY,
          fluenceDirs: c.fluenceDirs,
          readBase: 60 * ((1 << level) - 1),
          nextProbes: ceilDiv(pc, 1 << (level + 1)),
          evenWeight: this.evenWeights?.[level] ?? this.evenWeight,
          rpO0: o0, rpO1: o1, rpO2: o2, rpO3: o3, rpO4: o4,
          rpE0: e0, rpE1: e1, rpE2: e2, rpE3: e3, rpE4: e4,
          kappa: this.parityKappa[Math.min(level, this.parityKappa.length - 1)],
          // parity read: its own edge shift; production merge: the edge drain only when equalising
          edgeShift: level < 1 ? 0 : this.parityRead ? this.parityEdge : this.equalise ? (this.equalEdge ? this.equalEdge[Math.min(level, this.equalEdge.length - 1)] : 0.4) : 0,
        });
        device.queue.writeBuffer(c.mergeParamsBuffer, (dir * nc + level) * 256, this.#mergeParamsView.arrayBuffer);
      }
    }
  }

  // ── Event handlers ──

  #handleResize = async () => {
    if (this.#pp) {
      this.#updateCanvasLayout(); // re-centre the square; the probe grid is fixed in pixel-perfect mode
      return;
    }
    if (this.#resizing) return;
    this.#resizing = true;
    this.#isRunning = false;
    cancelAnimationFrame(this.#animationFrame);
    await this.#device.queue.onSubmittedWorkDone();

    this.#updateCanvasLayout();
    this.#context.configure({ device: this.#device, format: this.#presentationFormat, alphaMode: 'premultiplied' });
    this.#rebuild();
    this.#updateShapeData();

    this.#resizing = false;
    this.#isRunning = true;
    this.#startAnimationLoop();
  };

  #handleMouseMove = (e: MouseEvent) => {
    const [mx, my] = this.#mapToCanvas(e.clientX, e.clientY);
    this.#mousePosition.x = mx;
    this.#mousePosition.y = my;
    this.#mouseDirty = true;
  };

  #pendingScreenshot = '';

  saveScreenshot(filename = 'hrc-screenshot.png') {
    this.#pendingScreenshot = filename;
  }

  /** Debug: render n frames synchronously (independent of requestAnimationFrame), waiting for the GPU. */
  async step(n = 1) {
    for (let i = 0; i < n; i++) {
      this.#renderFrame();
      await this.#device.queue.onSubmittedWorkDone();
    }
  }

  /** Debug: path-traced radiance accumulation (rgb, screen resolution; fluence ≈ 2π × this in vacuum). */
  async readPathTrace(): Promise<{ width: number; height: number; frames: number; data: Float32Array }> {
    const { width: w, height: h } = this.#canvas;
    const bytesPerRow = Math.ceil((w * 16) / 256) * 256;
    const buf = this.#device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.#device.createCommandEncoder();
    const frames = this.#ptFrameIndex;
    enc.copyTextureToBuffer({ texture: this.#ptAccumTextures[frames % 2] }, { buffer: buf, bytesPerRow }, { width: w, height: h });
    this.#device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(buf.getMappedRange());
    const data = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < 3; c++) data[(y * w + x) * 3 + c] = raw[y * (bytesPerRow / 4) + x * 4 + c];
    buf.unmap();
    buf.destroy();
    return { width: w, height: h, frames, data };
  }

  /** Debug: the current exact reference (mean radiance rgb, same units as readPathTrace), or null. */
  async readExact(): Promise<{ width: number; height: number; fraction: number; data: Float32Array } | null> {
    const e = this.#exactCurrent;
    if (!e) return null;
    const bytes = await this.#readTextureBytes(e.out, e.w, e.h, 16);
    const raw = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
    const stride = Math.ceil((e.w * 16) / 256) * 64;
    const data = new Float32Array(e.w * e.h * 3);
    for (let y = 0; y < e.h; y++)
      for (let x = 0; x < e.w; x++)
        for (let c = 0; c < 3; c++) data[(y * e.w + x) * 3 + c] = raw[y * stride + x * 4 + c];
    return { width: e.w, height: e.h, fraction: e.workDone / (e.w * e.h * e.m), data };
  }

  /** Debug: world opacity (screen resolution). */
  async readOpacity(): Promise<{ width: number; height: number; data: Float32Array }> {
    const { width: w, height: h } = this.#canvas;
    const bytesPerRow = Math.ceil((w * 8) / 256) * 256;
    const buf = this.#device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.#device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.#worldTexture }, { buffer: buf, bytesPerRow }, { width: w, height: h });
    this.#device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(buf.getMappedRange());
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = raw[y * (bytesPerRow / 2) + x * 4 + 3] ? 1 : 0;
    buf.unmap();
    buf.destroy();
    return { width: w, height: h, data };
  }

  /** Merge-side read weights: per level (0 ..), branch (odd probe, then even probe), sub-bin (2^(level+1)): 3 slices x 5 ray
   * offsets, row-major; levels beyond the array keep their current values. */
  setMergeReadWeights(w: Float32Array) {
    this.#mergeReadWeights = w.slice();
    for (const c of this.#allCascades) {
      const n = Math.min(w.length, c.readWeightBuffer.size / 4);
      this.#device.queue.writeBuffer(c.readWeightBuffer, 0, w.subarray(0, n));
    }
  }

  /** Debug: read back the probe-resolution fluence (rgb, row-major, probesX × probesY). */
  // ── Exact reference API ──

  /** The texture the path tracer (and so the exact reference) reads: the transport world in ratio mode. */
  get #referenceSource(): GPUTexture {
    return this.#freeCascade ? this.#transportTexture : this.#worldTexture;
  }

  async #readTextureBytes(t: GPUTexture, w: number, h: number, bpp: number): Promise<Uint8Array> {
    const bytesPerRow = Math.ceil((w * bpp) / 256) * 256;
    const buf = this.#device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.#device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: t }, { buffer: buf, bytesPerRow }, { width: w, height: h });
    this.#device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(buf.getMappedRange()).slice();
    buf.unmap();
    buf.destroy();
    return out;
  }

  /**
   * Make the exact reference for the scene as it is now (the path tracer's input, sky included; bounces off) current:
   * reuse a cached result for identical input, otherwise snapshot it and start computing incrementally. m = directions
   * per pixel (default 64 · max(w, h), the research reference's density), depth = bisection steps at jumps.
   */
  async exactEnsure(opts: { m?: number; depth?: number } = {}): Promise<{ key: string; cached: boolean }> {
    const device = this.#device;
    await this.step(1); // the world (and transport world) reflect the current scene
    const { width: w, height: h } = this.#canvas;
    const m = opts.m ?? 64 * Math.max(w, h);
    const depth = opts.depth ?? 20;
    const src = this.#referenceSource;
    const [worldBytes, skyBytes] = await Promise.all([
      this.#readTextureBytes(src, w, h, 8),
      this.#readTextureBytes(this.#skyTexture, SKY_CIRCLE_SIZE, 1, 16),
    ]);
    let hsh = 2166136261 >>> 0;
    for (const bytes of [worldBytes, skyBytes]) {
      const u = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
      for (let i = 0; i < u.length; i++) hsh = Math.imul(hsh ^ u[i], 16777619) >>> 0;
    }
    const key = `${w}x${h}:m${m}:d${depth}:${hsh.toString(16)}`;
    for (const e of this.#exactCache.values()) if (e !== this.#exactCache.get(key)) e.running = false;
    const hit = this.#exactCache.get(key);
    if (hit) {
      this.#exactCache.delete(key);
      this.#exactCache.set(key, hit); // most recent last
      this.#exactCurrent = hit;
      if (hit.nextChunk < hit.nChunks && !hit.running) this.#exactRun(hit);
      return { key, cached: true };
    }
    // evict beyond three entries (oldest first)
    while (this.#exactCache.size >= 3) {
      const [k, e] = this.#exactCache.entries().next().value!;
      e.running = false;
      e.acc.destroy();
      e.out.destroy();
      e.world?.destroy();
      e.sky?.destroy();
      e.params.destroy();
      this.#exactCache.delete(k);
    }
    const chunkLen = 16;
    const nChunks = Math.ceil(m / chunkLen);
    const order = new Uint32Array(nChunks);
    if ((nChunks & (nChunks - 1)) === 0) {
      const bits = Math.log2(nChunks);
      for (let i = 0; i < nChunks; i++) {
        let r = 0;
        for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        order[i] = r;
      }
    } else {
      let step = Math.round(nChunks * 0.618034) | 1;
      const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
      while (gcd(step, nChunks) !== 1) step += 2;
      for (let i = 0; i < nChunks; i++) order[i] = (i * step) % nChunks;
    }
    const world = device.createTexture({ size: { width: w, height: h }, format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const sky = device.createTexture({ size: { width: SKY_CIRCLE_SIZE, height: 1 }, format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: world }, worldBytes, { bytesPerRow: Math.ceil((w * 8) / 256) * 256 }, { width: w, height: h });
    device.queue.writeTexture({ texture: sky }, skyBytes, { bytesPerRow: Math.ceil((SKY_CIRCLE_SIZE * 16) / 256) * 256 }, { width: SKY_CIRCLE_SIZE, height: 1 });
    const acc = device.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const out = device.createTexture({ size: { width: w, height: h }, format: 'rgba32float', usage: TEX_STORAGE | GPUTextureUsage.COPY_SRC });
    const entry: ExactEntry = {
      key, w, h, m, chunkLen, depth, nChunks, order, nextChunk: 0, nextRow: 0, rows: Math.max(1, Math.round(h / 16)),
      acc, out, outView: out.createView(), world, sky,
      params: device.createBuffer({ size: 256 * 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      started: performance.now(), elapsedMs: 0, workDone: 0, running: false,
    };
    this.#exactCache.set(key, entry);
    this.#exactCurrent = entry;
    this.#exactRun(entry);
    return { key, cached: false };
  }

  /** Stop computing the current exact reference (the partial result stays cached and resumes on exactEnsure). */
  exactPause() {
    if (this.#exactCurrent) this.#exactCurrent.running = false;
  }

  /** Progress of the current exact reference: fraction done, elapsed and estimated remaining time (ms). */
  get exactState() {
    const e = this.#exactCurrent;
    if (!e) return null;
    const total = e.w * e.h * e.m;
    const fraction = e.workDone / total;
    const rate = e.elapsedMs > 0 ? e.workDone / e.elapsedMs : 0;
    return {
      key: e.key,
      m: e.m,
      fraction,
      done: e.nextChunk >= e.nChunks,
      running: e.running,
      elapsedMs: e.elapsedMs,
      etaMs: rate > 0 ? (total - e.workDone) / rate : NaN,
      totalMs: rate > 0 ? total / rate : NaN,
    };
  }

  async #exactRun(e: ExactEntry) {
    const device = this.#device;
    e.running = true;
    const TARGET_MS = 40;
    while (e.running && e.nextChunk < e.nChunks && this.#exactCache.get(e.key) === e) {
      // several bands per submit, each with its own uniform slot
      const enc = device.createCommandEncoder();
      const t0 = performance.now();
      let work = 0;
      let slots = 0;
      const bg0 = (slot: number) =>
        bg(device, this.#exactPipeline!.getBindGroupLayout(0), e.world!.createView(), e.sky!.createView(), { buffer: e.acc },
          e.outView, { buffer: e.params, offset: slot * 256, size: 64 });
      const rows = Math.min(e.rows, e.h - e.nextRow);
      this.#exactParamsView.set({
        w: e.w, h: e.h, m: e.m, chunkLen: e.chunkLen, chunk: e.order[e.nextChunk], rowStart: e.nextRow, rowCount: rows,
        depth: e.depth, scale: e.nChunks / (e.nextChunk + 1) / (2 * Math.PI), pad0: 0, pad1: 0, pad2: 0,
      });
      device.queue.writeBuffer(e.params, slots * 256, this.#exactParamsView.arrayBuffer);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.#exactPipeline!);
      pass.setBindGroup(0, bg0(slots));
      pass.dispatchWorkgroups(Math.ceil(e.w / 8), Math.ceil(rows / 8));
      pass.end();
      slots++;
      work += rows * e.w * e.chunkLen;
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const dt = performance.now() - t0;
      e.elapsedMs += dt;
      e.workDone += work;
      e.nextRow += rows;
      if (e.nextRow >= e.h) {
        e.nextRow = 0;
        e.nextChunk++;
      }
      // adapt the band height to the time budget (keeps each submit well under GPU watchdog limits)
      e.rows = Math.max(1, Math.min(e.h, Math.round(e.rows * Math.min(2, Math.max(0.5, TARGET_MS / Math.max(1, dt))))));
    }
    e.running = false;
    if (e.nextChunk >= e.nChunks) {
      e.world?.destroy();
      e.sky?.destroy();
      e.world = e.sky = null;
    }
  }

  async readFluence(): Promise<{ width: number; height: number; data: Float32Array }> {
    const w = this.#probesX;
    const h = this.#probesY;
    const bytesPerRow = Math.ceil((w * 8) / 256) * 256;
    const buf = this.#device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.#device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.#fluenceTexture }, { buffer: buf, bytesPerRow }, { width: w, height: h });
    this.#device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(buf.getMappedRange());
    const f16 = (x: number) => {
      const e = (x >> 10) & 31, m = x & 1023, sgn = x >> 15 ? -1 : 1;
      if (e === 0) return sgn * m * 2 ** -24;
      if (e === 31) return m ? NaN : sgn * Infinity;
      return sgn * (1 + m / 1024) * 2 ** (e - 15);
    };
    const data = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < 3; c++) data[(y * w + x) * 3 + c] = f16(raw[y * (bytesPerRow / 2) + x * 4 + c]);
    buf.unmap();
    buf.destroy();
    return { width: w, height: h, data };
  }

  get fps() {
    return this.#smoothedFrameTime > 0 ? Math.round(1000 / this.#smoothedFrameTime) : 0;
  }

  get frameTimeMs() {
    return this.#smoothedFrameTime;
  }

  get resolution() {
    return { width: this.#canvas?.width ?? 0, height: this.#canvas?.height ?? 0 };
  }

  get gpuTimeMs() {
    return this.#gpuTimeMs;
  }

  get jsTimeMs() {
    return this.#jsTimeMs;
  }

  get gpuPassTimings() {
    return this.#gpuPassTimings;
  }

  get debugInfo() {
    if ((this.debugMode === 1 || this.debugMode === 2) && this.#ptFrameIndex > 0) {
      return ` PT f${this.#ptFrameIndex} ${this.#ptFrameIndex * 16}spp`;
    }
    if (!this.#tsQuerySet) return '';
    if (this.#gpuTimeMs <= 0) return ' gpu:...';
    return ` gpu:${this.#gpuTimeMs.toFixed(1)}ms`;
  }

  #updateFrameTiming(now: number) {
    if (this.#lastFrameTimestamp > 0) {
      const dt = now - this.#lastFrameTimestamp;
      const alpha = 0.05;
      this.#smoothedFrameTime =
        this.#smoothedFrameTime === 0 ? dt : this.#smoothedFrameTime + alpha * (dt - this.#smoothedFrameTime);
    }
    this.#lastFrameTimestamp = now;
  }
}
