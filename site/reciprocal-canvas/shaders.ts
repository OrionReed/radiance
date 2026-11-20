// The canvas's own passes: the raster, the exact trace for "error against exact", and the draw.

import { DISPLAY_WGSL } from '../_shared/display';
import { DRAW_WGSL } from '../_shared/draw';
import { commonWgsl, skyWgsl } from './producer';

// One thread per domain pixel, written to the engine's scene ([rise class][axis][column][row], both halves alike). A pixel
// takes the boxes over its centre; with coverage (world.y = 1), each box by the share c of the pixel it covers, the
// pixel letting 1 - c through and emitting c E as a thin sheet (each half lets √(1 - c) through).
export const rasterWgsl = (N: number) => /* wgsl */ `
${commonWgsl(N)}
@group(0) @binding(0) var<storage, read> shapes: array<Shape>;
@group(0) @binding(1) var<uniform> world: vec4u;   // box count, coverage
@group(0) @binding(2) var<storage, read_write> scene: array<vec4f>;
@compute @workgroup_size(64)
fn raster(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = i32(id.x + id.y * groups.x * 64u);
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N; let c = vec2f(f32(x) + 0.5, f32(y) + 0.5);
  var pixel = vec4f(0.0, 0.0, 0.0, 1.0);
  if (world.y == 1u) {
    var through = 1.0; var emitted = vec3f(0.0);
    for (var j = 0u; j < world.x; j++) {
      let b = shapes[j].box;
      let overlap = max(vec2f(0.0), min(b.zw, c + 0.5) - max(b.xy, c - 0.5));
      let covered = min(1.0, overlap.x * overlap.y);
      if (covered <= 0.0) { continue; }
      emitted += covered * shapes[j].light.rgb * shapes[j].light.w;
      through *= 1.0 - covered;
    }
    if (through < 1.0) {
      let half = sqrt(max(through, 0.0));
      pixel = vec4f(emitted / (1.0 - through) * (1.0 - half), half);
    }
  }
  for (var j = 0u; j < world.x && world.y == 0u; j++) {
    let b = shapes[j].box;
    if (c.x < b.x || c.x > b.z || c.y < b.y || c.y > b.w) { continue; }
    if (shapes[j].light.w > 0.5) { pixel = vec4f(shapes[j].light.rgb, 0.0); }
    else if (pixel.w > 0.0) { pixel = vec4f(0.0); }
  }
  for (var k = 0; k < 2; k++) {
    scene[k * 2 * N * N + x * N + y] = pixel;
    scene[k * 2 * N * N + N * N + y * N + x] = pixel;
  }
}`;

// A domain pixel's exact fluence, the mean over 4 × 4 points of the angle-integrated radiance of the lights and the sky
// with first-hit visibility through every box (lightsOnly: through the lights only). A light's angle is sampled in
// proportion to its size, the sky's at 1024 angles and the sun's at 256 across it (one, if it has no size).
export const referenceWgsl = (N: number) => /* wgsl */ `
${commonWgsl(N)}
@group(0) @binding(0) var<storage, read> shapes: array<Shape>;
@group(0) @binding(1) var<uniform> world: vec4u;   // box count, first column, columns, lights only
@group(0) @binding(2) var<storage, read_write> exact: array<vec4f>;
@group(0) @binding(3) var<storage, read> sky: Sky;
${skyWgsl}
fn firstBox(q: vec2f, d: vec2f) -> i32 {
  var best = 1e30; var which = -1;
  for (var j = 0u; j < world.x; j++) {
    if (world.w == 1u && shapes[j].light.w < 0.5) { continue; }
    let b = shapes[j].box;
    let ta = (b.xy - q) / d; let tb = (b.zw - q) / d;
    let t0 = max(min(ta.x, tb.x), min(ta.y, tb.y)); let t1 = min(max(ta.x, tb.x), max(ta.y, tb.y));
    if (t0 <= t1 && t1 > 0.0 && max(t0, 0.0) < best) { best = max(t0, 0.0); which = i32(j); }
  }
  return which;
}
fn exactAt(q: vec2f) -> vec3f {
  var sum = vec3f(0.0);
  for (var j = 0u; j < world.x; j++) {
    if (shapes[j].light.w < 0.5) { continue; }
    let box = shapes[j].box; let centre = 0.5 * (box.xy + box.zw) - q;
    let base = atan2(centre.y, centre.x);
    var lo = 10.0; var hi = -10.0;
    for (var c = 0; c < 4; c++) {
      let corner = vec2f(select(box.x, box.z, (c & 1) == 1), select(box.y, box.w, (c & 2) == 2)) - q;
      var d = atan2(corner.y, corner.x) - base;
      d = d - 6.2831853 * round(d / 6.2831853);
      lo = min(lo, d); hi = max(hi, d);
    }
    let m = 32 + i32(2000.0 * (hi - lo));
    let step = (hi - lo) / f32(m);
    var seen = 0.0;
    for (var k = 0; k < m; k++) {
      let theta = base + lo + (f32(k) + 0.5) * step;
      if (firstBox(q, vec2f(cos(theta), sin(theta))) == i32(j)) { seen += step; }
    }
    sum += seen * shapes[j].light.rgb;
  }
  if (skyOn()) {
    let step = 6.2831853 / 1024.0;
    for (var k = 0; k < 1024; k++) {
      let a = f32(k) * step;
      if (firstBox(q, vec2f(cos(a + 0.5 * step), sin(a + 0.5 * step))) == -1) { sum += skySmooth(a, a + step); }
    }
    let r = sky.sun.w; let fine = (2.0 * r) / 256.0;
    for (var k = 0; k < 256 && r > 0.0; k++) {
      let a = sky.angles.y - r + f32(k) * fine;
      if (firstBox(q, vec2f(cos(a + 0.5 * fine), sin(a + 0.5 * fine))) == -1) { sum += skySun(a, a + fine); }
    }
    if (r <= 0.0 && firstBox(q, vec2f(cos(sky.angles.y), sin(sky.angles.y))) == -1) { sum += sky.sun.rgb; }
  }
  return sum;
}
@compute @workgroup_size(64)
fn reference(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x);
  if (i >= i32(world.z) * N) { return; }
  let x = i32(world.y) + i / N; let y = i % N;
  var sum = vec3f(0.0);
  for (var a = 0; a < 4; a++) {
    for (var b = 0; b < 4; b++) { sum += exactAt(vec2f(f32(x) + (f32(a) + 0.5) / 4.0, f32(y) + (f32(b) + 0.5) / 4.0)); }
  }
  let slot = x * N + y;
  let value = sum / 16.0;
  if (world.w == 1u) { exact[slot].w = dot(value, vec3f(0.2126, 0.7152, 0.0722)); } else { exact[slot] = vec4f(value, exact[slot].w); }
}`;

// The screen: a solve's pixel under each fragment (or, smooth, the bilinear mean of the free pixels around it); when
// cross-fading, mixed with the coarser solve's by its weight.
const solves = [
  ['fine', 'fluence', 'scene', 'draw.screen.w', 'draw.camera.zw', 'draw.view.x'],
  ['coarse', 'coarseFluence', 'coarseScene', 'draw.display.y', 'draw.coarse.xy', 'draw.coarse.z'],
];
export const drawWgsl = /* wgsl */ `
@group(0) @binding(0) var<storage, read> fluence: array<vec4f>;
@group(0) @binding(1) var<storage, read> scene: array<vec4f>;
@group(0) @binding(2) var<storage, read> exact: array<vec4f>;
@group(0) @binding(4) var<storage, read> coarseFluence: array<vec4f>;
@group(0) @binding(5) var<storage, read> coarseScene: array<vec4f>;
struct Draw {
  screen: vec4f,   // canvas size (device px), device px per world unit, N
  camera: vec4f,   // camera centre (world), domain origin (world)
  view: vec4f,     // cascade pixel (world units), exposure, view (0 fluence, 1 error against the exact trace), error clip
  coarse: vec4f,   // the coarser solve's domain origin (world), its pixel, its weight
  display: vec4f,  // smooth display, the coarser solve's N
}
@group(0) @binding(3) var<uniform> draw: Draw;
${DISPLAY_WGSL}
${DRAW_WGSL}
${solves.map(([which, fl, sc, size, origin, pixel]) => `
fn ${which}Smooth(at: vec2f) -> vec3f {
  let n = i32(${size});
  let base = vec2i(floor(at - 0.5)); let f = at - 0.5 - floor(at - 0.5);
  var sum = vec3f(0.0); var weight = 0.0;
  for (var k = 0; k < 4; k++) {
    let p = base + vec2i(k & 1, k >> 1);
    if (any(p < vec2i(0)) || any(p >= vec2i(n))) { continue; }
    let i = p.x * n + p.y;
    if (${sc}[i].w <= 0.0) { continue; }
    let w = select(1.0 - f.x, f.x, (k & 1) == 1) * select(1.0 - f.y, f.y, (k >> 1) == 1);
    sum += w * ${fl}[i].rgb; weight += w;
  }
  return select(vec3f(0.0), sum / weight, weight > 0.0);
}
fn ${which}Colour(fragment: vec2f) -> vec4f {
  let n = i32(${size});
  let world = draw.camera.xy + vec2f(fragment.x - 0.5 * draw.screen.x, 0.5 * draw.screen.y - fragment.y) / draw.screen.z;
  let at = (world - ${origin}) / ${pixel};
  let pixel = vec2i(floor(at));
  if (any(pixel < vec2i(0)) || any(pixel >= vec2i(n))) { return vec4f(0.3, 0.0, 0.3, 1.0); }
  let i = pixel.x * n + pixel.y;
  let material = ${sc}[i];
  if (material.w <= 0.0) {
    if (any(material.rgb > vec3f(0.0))) { return vec4f(srgbEncode(min(vec3f(1.0), material.rgb * 0.15)), 1.0); }
    return vec4f(0.11, 0.115, 0.13, 1.0);
  }
  var value = ${fl}[i].rgb;
  if (draw.display.x > 0.5) { value = ${which}Smooth(at); }
  // a light covering part of a pixel: its colour over its share c, where each half emits E (1 - √(1 - c))
  let covered = 1.0 - material.w * material.w;
  let shown = displayed(value * draw.view.y / 6.2831853, 0.0);
  if (covered > 0.0 && any(material.rgb > vec3f(0.0)) && draw.view.z < 0.5) {
    let light = srgbEncode(min(vec3f(1.0), material.rgb / (1.0 - material.w) * 0.15));
    return dither(mix(shown, light, covered), fragment);
  }${which === 'fine' ? `
  if (draw.view.z > 0.5) {
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    return vec4f(errorColour(dot(value, lum), dot(exact[i].rgb, lum), draw.view.y / 6.2831853, draw.view.w), 1.0);
  }` : ''}
  return dither(shown, fragment);
}`).join('')}
@fragment fn fragmentMain(@builtin(position) fragment: vec4f) -> @location(0) vec4f {
  let fine = fineColour(fragment.xy);
  if (draw.coarse.w <= 0.0 || draw.view.z > 0.5) { return fine; }
  return mix(fine, coarseColour(fragment.xy), draw.coarse.w);
}`;
