// The light entering a square domain from the world around it, every frame, as ReciprocalHRC's producer. The engine
// fills block 0 of every length from the producer, so every outside cell starts on a back edge's line: one sweep thread
// per start point there, Q per row, rows -back ... N + back - 1. A point's lines within ±45° of the travel come out as
// pieces of angle, each lit by the first box the line meets beyond the domain's edge or, past every box, by the sky (the
// sun is a disc of it). The engine's fill takes a row per workgroup and integrates each cell over the row's pieces.

import { outsideLayout, FILL_THREADS, type OutsideProducer } from '../reciprocal-hrc/reciprocal-hrc';

const HELD = 16; // pieces a sweep thread keeps in registers; a point with more walks again to write them
const CACHE = 960; // pieces a fill workgroup keeps in workgroup memory (15 KiB); a row with more reads the pool

export function pointLayout(N: number, Q: number) {
  const back = Math.max(...outsideLayout(N).map((l) => l.pad0));
  return { back, perFrustum: (N + 2 * back) * Q };
}

export const commonWgsl = (N: number, Q = 1) => /* wgsl */ `
const N: i32 = ${N};
const Q: i32 = ${Q};
struct Shape { box: vec4f, light: vec4f };   // box (x0, y0, x1, y1) in domain pixels; light (rgb, 1 if a light)
// a box in frustum f's frame: column u along the travel, row v across it (0 / 1: travel +x / -x, 2 / 3: +y / -y)
fn frustumBox(f: i32, b: vec4f) -> vec4f {   // (u0, v0, u1, v1)
  let n = f32(N);
  if (f == 0) { return b; }
  if (f == 1) { return vec4f(n - b.z, b.y, n - b.x, b.w); }
  if (f == 2) { return vec4f(b.y, b.x, b.w, b.z); }
  return vec4f(n - b.w, b.x, n - b.y, b.z);
}`;

/** Radiance arriving from world angle φ (0 from +x, counter-clockwise) at infinite distance: ambient + gradient
 * cos(φ - towards) + bands cos 2(φ - towards), and a sun: irradiance rgb from angle `at`, spread evenly over a disc of
 * that radius (0: a single direction). */
export interface Sky {
  ambient: [number, number, number];
  gradient?: [number, number, number];
  bands?: [number, number, number];
  towards?: number;
  sun?: { rgb: [number, number, number]; at: number; radius: number };
}

function skyData(sky: Sky | null) {
  if (!sky) return new Float32Array(20);
  const { ambient, gradient = [0, 0, 0], bands = [0, 0, 0], towards = 0, sun = { rgb: [0, 0, 0], at: 0, radius: 0 } } = sky;
  const smooth = [...ambient, ...gradient, ...bands].some((v) => v !== 0) ? 1 : 0;
  return new Float32Array([...ambient, 0, ...gradient, 0, ...bands, 0, ...sun.rgb, sun.radius, towards, sun.at, 1, smooth]);
}

export const skyWgsl = /* wgsl */ `
struct Sky { ambient: vec4f, gradient: vec4f, bands: vec4f, sun: vec4f, angles: vec4f }   // angles: towards, the sun's, on, smooth
fn skySmooth(a: f32, b: f32) -> vec3f {   // over world angles [a, b]
  let p = sky.angles.x;
  var sum = sky.ambient.rgb * (b - a);
  if (any(sky.gradient.rgb != vec3f(0.0))) { sum += sky.gradient.rgb * (sin(b - p) - sin(a - p)); }
  if (any(sky.bands.rgb != vec3f(0.0))) { sum += sky.bands.rgb * 0.5 * (sin(2.0 * (b - p)) - sin(2.0 * (a - p))); }
  return sum;
}
fn skySun(a: f32, b: f32) -> vec3f {   // a disc's
  if (sky.sun.w <= 0.0) { return vec3f(0.0); }
  var seen = 0.0;
  for (var k = -1; k <= 1; k++) {
    let c = sky.angles.y + 6.2831853 * f32(k);
    seen += max(0.0, min(b, c + sky.sun.w) - max(a, c - sky.sun.w));
  }
  return seen / (2.0 * sky.sun.w) * sky.sun.rgb;
}
// frustum f's angle 0 as the world angle the light comes from (its travel's opposite), and which way its angles turn
fn skyBack(f: i32) -> f32 { return select(select(1.5707964, 4.712389, f == 2), select(3.1415927, 0.0, f == 1), f < 2); }
fn skyTurn(f: i32) -> f32 { return select(-1.0, 1.0, f == 0 || f == 3); }
// over frustum f's angles [a, b]: the smooth sky (what 1), the sun (2) or both (3)
fn skyOver(f: i32, a: f32, b: f32, what: u32) -> vec3f {
  let back = skyBack(f); let turn = skyTurn(f);
  let lo = back + turn * select(b, a, turn > 0.0); let hi = back + turn * select(a, b, turn > 0.0);
  var sum = vec3f(0.0);
  if ((what & 1u) != 0u) { sum += skySmooth(lo, hi); }
  if ((what & 2u) != 0u) { sum += skySun(lo, hi); }
  return sum;
}
fn skyOn() -> bool { return sky.angles.z > 0.5; }
fn smoothSky() -> bool { return sky.angles.w > 0.5; }   // light from every angle, not only the sun's
fn sunOn() -> bool { return skyOn() && any(sky.sun.rgb > vec3f(0.0)); }
// the sun's angle in frustum f's frame
fn sunAt(f: i32) -> f32 {
  let at = skyTurn(f) * (sky.angles.y - skyBack(f));
  return at - 6.2831853 * round(at / 6.2831853);
}
// the sun disc's span of slope in frustum f (empty if it is outside ±45°, or a single direction)
fn sunSlopes(f: i32) -> vec2f {
  if (!sunOn() || sky.sun.w <= 0.0) { return vec2f(1.0, -1.0); }
  let at = sunAt(f);
  let lo = max(-0.78539816, at - sky.sun.w); let hi = min(0.78539816, at + sky.sun.w);
  return select(vec2f(1.0, -1.0), vec2f(tan(lo), tan(hi)), hi > lo);
}`;

const producerWgsl = (N: number, Q: number) => {
  const { back, perFrustum } = pointLayout(N, Q);
  return /* wgsl */ `
@group(0) @binding(20) var<storage, read> pool: array<vec4u>;
@group(0) @binding(21) var<storage, read> lists: array<vec2u>;   // per start point: first piece in the pool, pieces
@group(0) @binding(22) var<storage, read> sky: Sky;
${skyWgsl}
var<workgroup> cache: array<vec4u, ${CACHE}>;
var<workgroup> rowLists: array<vec2u, ${Q}>;   // per point: first piece, in the cache or else the pool; pieces, flags
const CACHED = 0x80000000u;
const SORTED = 0x40000000u;   // the pieces are in order of angle and do not overlap
const SUN_SEEN = 0x20000000u;  // a sun of no size: the point's line toward it meets nothing
const COUNT = 0x1fffffffu;
fn outsidePrepare(length: i32, row: i32, local: u32) {
  let first = u32(params.frustum * ${perFrustum} + (row + ${back}) * ${Q});
  var own: array<vec2u, ${Q}>;
  var total = 0u;
  for (var k = 0u; k < ${Q}u; k++) { own[k] = lists[first + k]; total += own[k].y & COUNT; }
  let cached = total <= ${CACHE}u;
  var at = 0u;
  for (var k = 0u; k < ${Q}u; k++) {
    let n = own[k].y & COUNT;
    if (cached) { for (var j = local; j < n; j += ${FILL_THREADS}u) { cache[at + j] = pool[own[k].x + j]; } }
    if (local == 0u) { rowLists[k] = vec2u(select(own[k].x, at, cached), own[k].y | select(0u, CACHED, cached)); }
    at += n;
  }
}
fn pieceAt(list: vec2u, m: u32) -> vec4u {
  if ((list.y & CACHED) != 0u) { return cache[list.x + m]; }
  return pool[list.x + m];
}
// The sun is irradiance over a disc. A cell takes it over its lines: over the start offsets a of a point's share of the
// start span, the lines' slopes [(rise - a) / L, (rise + 1 - a) / L] within the part of the disc the point sees, its
// slopes [σ0, σ1] (each slope weighted as the disc's angle per slope there). The overlap is a trapezoid in the lines'
// lowest slope, so its integral over a is four squared ramps. With no size, the share of those offsets whose lines
// hold the sun's slope (over a whole start span, the two-rise split).
fn rampSquared(x: f32) -> f32 { let r = max(0.0, x); return r * r; }
fn sunInCell(L: f32, rise: f32, a0: f32, a1: f32, s0: f32, s1: f32) -> f32 {
  let w = 1.0 / L;
  let lowest = (rise - vec2f(a0, a1)) / L;   // the lines' lowest slope at a0 and a1
  let ramps = 0.5 * (vec2f(rampSquared(lowest.x + w - s0), rampSquared(lowest.y + w - s0)) - vec2f(rampSquared(lowest.x - s0), rampSquared(lowest.y - s0))
    - vec2f(rampSquared(lowest.x + w - s1), rampSquared(lowest.y + w - s1)) + vec2f(rampSquared(lowest.x - s1), rampSquared(lowest.y - s1)));
  return L * (ramps.x - ramps.y);
}
fn sunDirectionInCell(L: f32, rise: f32, a0: f32, a1: f32, at: f32) -> f32 {
  let t = tan(at);
  return max(0.0, min(a1, rise + 1.0 - L * t) - max(a0, rise - L * t));
}
fn outsideValue(length: i32, block: i32, rise: i32, startRow: i32) -> vec3f {
  var sum = vec3f(0.0);
  let L = f32(length); let r = f32(rise);
  let sunAngle = sunAt(params.frustum); let disc = vec2f(sunAngle - sky.sun.w, sunAngle + sky.sun.w);
  for (var k = 0u; k < ${Q}u; k++) {
    let a = (f32(k) + 0.5) / ${Q}.0; let share = vec2f(f32(k), f32(k + 1u)) / ${Q}.0;
    let lo = atan(max(-1.0, (f32(rise) - a) / f32(length))); let hi = atan(min(1.0, (f32(rise) + 1.0 - a) / f32(length)));
    let reach = atan(clamp(vec2f(r - share.y, r + 1.0 - share.x) / L, vec2f(-1.0), vec2f(1.0)));   // the lines' angles over the share
    let list = rowLists[k]; let n = list.y & COUNT; let sorted = (list.y & SORTED) != 0u;
    if ((list.y & SUN_SEEN) != 0u && abs(sunAngle) < 0.78539816) { sum += sky.sun.rgb * ${Q}.0 * sunDirectionInCell(L, r, share.x, share.y, sunAngle); }
    var m = 0u;
    if (sorted) {   // the first piece that ends after the lines' lowest angle
      var end = n;
      while (m < end) { let mid = (m + end) / 2u; if (bitcast<f32>(pieceAt(list, mid).y) <= reach.x) { m = mid + 1u; } else { end = mid; } }
    }
    for (; m < n; m++) {
      let p = pieceAt(list, m);
      if (sorted && bitcast<f32>(p.x) >= reach.y) { break; }
      let start = bitcast<f32>(p.x); let end = bitcast<f32>(p.y);
      let blue = unpack2x16float(p.w); let what = u32(blue.y);
      if ((what & 2u) != 0u) {
        let s0 = max(start, disc.x); let s1 = min(end, disc.y);
        if (s1 > s0) {
          let slopes = vec2f(tan(s0), tan(s1));
          sum += sky.sun.rgb / (2.0 * sky.sun.w) * (s1 - s0) / (slopes.y - slopes.x) * ${Q}.0 * sunInCell(L, r, share.x, share.y, slopes.x, slopes.y);
        }
      }
      let a0 = max(lo, start); let a1 = min(hi, end);
      if (a1 <= a0) { continue; }
      if (blue.y > 0.5 && (a0 > start || a1 < end)) { sum += skyOver(params.frustum, a0, a1, what & 1u); }
      else { sum += (a1 - a0) * vec3f(unpack2x16float(p.z), blue.x); }
    }
  }
  return sum / ${Q}.0;
}`;
};

/** A start point's budget: at most `rays` lines cast and `pieces` pieces stored, in a pool of fixed size */
export interface Budget {
  rays: number;
  pieces: number;
}

// One thread per start point. Between consecutive breakpoints (slopes to box corners, the domain's corners and the ends
// of the lights' and the sun's spans) the box a line meets first going back from where it enters the domain does not
// change, so one line per interval decides it: a light's radiance, the sky, or nothing. The walk covers the lights' and
// the sun's spans, or all of ±45° when the sky has light from every angle. Without a budget every point walks its
// breakpoints. With one:
// - a point with more breakpoints than rays shares its rays among the lights, the sky and the sun by the light each
//   could bring, at least one each; a ray lights its share of angle if what it meets first is its own light (or the sky);
// - a point with more pieces than it may store joins consecutive pieces, keeping their energy.
const sweepWgsl = (N: number, Q: number, budget: Budget | null) => {
  const { back, perFrustum } = pointLayout(N, Q);
  return /* wgsl */ `
${commonWgsl(N, Q)}
@group(0) @binding(0) var<storage, read> shapes: array<Shape>;
@group(0) @binding(1) var<uniform> world: vec4u;   // box count, (the raster's), pool capacity
@group(0) @binding(2) var<storage, read_write> pool: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> lists: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>, 4>;   // pieces, points shared, points joined
@group(0) @binding(5) var<storage, read> sky: Sky;
${skyWgsl}
const BUDGET = ${budget !== null};
const RAYS = ${budget?.rays ?? 1};
const PIECES = ${budget?.pieces ?? 1}u;
const FAST_LIGHTS = 128;
const FAST_BREAKS = 512;
const LUMA = vec3f(0.2126, 0.7152, 0.0722);
const SORTED = 0x40000000u;
const SUN_SEEN = 0x20000000u;
// what lights an interval: a light's box index, or one of these
const DARK = -3;
fn skyKey(what: u32) -> i32 { return -10 - i32(what); }

var<private> f: i32;     // the frustum
var<private> p: vec2f;   // the start point, in its frame
var<private> spans: array<vec2f, FAST_LIGHTS>;
var<private> spanBox: array<u32, FAST_LIGHTS>;
var<private> lightCount: i32;
var<private> spansHeld: bool;   // every light's span is in spans
var<private> breaks: array<f32, FAST_BREAKS>;
var<private> breakCount: i32;
var<private> fast: bool;        // the breakpoints are in breaks, sorted
var<private> first: f32;        // the walk's slopes
var<private> last: f32;
var<private> sun: vec2f;        // the sun's span of slope

// a piece: angle from, angle to (f32 bits), its mean radiance (f16), what lights it (0: a box; else skyOver's what, for
// a cell that covers only part of it)
fn piece(lo: f32, hi: f32, rgb: vec3f, what: u32) -> vec4u {
  return vec4u(bitcast<u32>(lo), bitcast<u32>(hi), pack2x16float(rgb.rg), pack2x16float(vec2f(rgb.b, f32(what))));
}

// the box the line p + t (1, s) meets first going back from where it enters the domain (the largest min(t out, entry)
// over boxes it meets before the entry); -1: none, the sky; -2: the line never enters the domain
fn firstHit(s: f32) -> i32 {
  let n = f32(N);
  var tin = -p.x; var tout = n - p.x;
  if (abs(s) < 1e-12) { if (p.y < 0.0 || p.y > n) { return -2; } }
  else { let a = -p.y / s; let b = (n - p.y) / s; tin = max(tin, min(a, b)); tout = min(tout, max(a, b)); }
  if (tin > tout || tout < 0.0) { return -2; }
  let entry = max(tin, 0.0);
  var best = -1e30; var which = -1;
  for (var j = 0u; j < world.x; j++) {
    let b = frustumBox(f, shapes[j].box);
    var t0 = b.x - p.x; var t1 = b.z - p.x;
    if (abs(s) < 1e-12) { if (p.y < b.y || p.y > b.w) { continue; } }
    else { let a = (b.y - p.y) / s; let c = (b.w - p.y) / s; t0 = max(t0, min(a, c)); t1 = min(t1, max(a, c)); }
    if (t0 > t1 || t0 >= entry) { continue; }
    let h = min(t1, entry);
    if (h > best) { best = h; which = i32(j); }
  }
  return which;
}
// past the domain's far side: every line enters the domain before it
fn beyond(j: u32) -> bool { return frustumBox(f, shapes[j].box).x >= f32(N); }
fn isLight(j: i32) -> bool { return j >= 0 && shapes[j].light.w > 0.5; }

// a light's span of slope (a box across the column through p: all of them); empty for a wall
fn span(j: u32) -> vec2f {
  if (shapes[j].light.w < 0.5) { return vec2f(1.0, -1.0); }
  let box = frustumBox(f, shapes[j].box);
  if (box.x <= p.x && box.z >= p.x) { return vec2f(-1.0, 1.0); }
  let near = select(box.z, box.x, box.x > p.x) - p.x; let far = select(box.x, box.z, box.x > p.x) - p.x;
  let a = (box.y - p.y) / near; let b = (box.w - p.y) / near; let c = (box.y - p.y) / far; let d = (box.w - p.y) / far;
  return vec2f(max(-1.0, min(min(a, b), min(c, d))), min(1.0, max(max(a, b), max(c, d))));
}
fn inSomeSpan(s: f32) -> bool {
  if (s > sun.x && s < sun.y) { return true; }
  if (fast) {
    for (var l = 0; l < lightCount; l++) { if (s > spans[l].x && s < spans[l].y) { return true; } }
    return false;
  }
  for (var j = 0u; j < world.x; j++) { let sp = span(j); if (s > sp.x && s < sp.y) { return true; } }
  return false;
}
fn corner(box: vec4f, c: i32) -> vec2f { return vec2f(select(box.x, box.z, (c & 1) == 1), select(box.y, box.w, (c & 2) == 2)) - p; }
// the first breakpoint after s (at most end), by a scan
fn nextBreak(s: f32, end: f32) -> f32 {
  var next = end;
  let n = f32(N);
  if (sun.x > s && sun.x < next) { next = sun.x; }
  if (sun.y > s && sun.y < next) { next = sun.y; }
  for (var j = 0u; j <= world.x; j++) {
    var box = vec4f(0.0, 0.0, n, n);   // j = count: the domain
    if (j < world.x) {
      box = frustumBox(f, shapes[j].box);
      let sp = span(j);
      if (sp.x > s && sp.x < next) { next = sp.x; }
      if (sp.y > s && sp.y < next) { next = sp.y; }
    }
    for (var c = 0; c < 4; c++) {
      let d = corner(box, c);
      if (abs(d.x) <= 1e-9) { continue; }
      let slope = d.y / d.x;
      if (slope > s + 1e-9 && slope < next) { next = slope; }
    }
  }
  return next;
}

// Pieces are counted on a first walk (the first HELD kept) and written on a second if they did not all fit. Consecutive
// intervals with one key make one piece; with a join of k, k consecutive pieces make one stored piece.
var<private> held: array<vec4u, ${HELD}>;
var<private> writing: bool;
var<private> made: u32;     // pieces so far in this walk
var<private> total: u32;    // pieces in all, once counted
var<private> base: u32;
var<private> join: u32;
var<private> joinedFrom: f32;
var<private> joinedTo: f32;
var<private> joinedEnergy: vec3f;
var<private> openKey: i32;  // the piece being extended (DARK: none)
var<private> openFrom: f32;
var<private> openTo: f32;
var<private> openWhat: u32;
var<private> openRgb: vec3f;

fn store(lo: f32, hi: f32, light: vec3f, what: u32) {
  let i = made; made++;
  let rgb = select(light, skyOver(f, lo, hi, what & 1u) / max(1e-12, hi - lo), what > 0u);   // a sky piece: its smooth sky's mean
  if (!writing) { if (i < ${HELD}u) { held[i] = piece(lo, hi, rgb, what); } return; }
  if (join == 1u) { pool[base + i] = piece(lo, hi, rgb, what); return; }
  if (i % join == 0u) { joinedFrom = lo; joinedTo = hi; joinedEnergy = vec3f(0.0); }
  joinedFrom = min(joinedFrom, lo); joinedTo = max(joinedTo, hi);
  if (what > 0u) { joinedEnergy += skyOver(f, lo, hi, what); } else { joinedEnergy += rgb * (hi - lo); }
  if (i % join == join - 1u || i + 1u == total) { pool[base + i / join] = piece(joinedFrom, joinedTo, joinedEnergy / max(1e-12, joinedTo - joinedFrom), 0u); }
}
fn close() {
  if (openKey != DARK) { store(openFrom, openTo, openRgb, openWhat); }
  openKey = DARK;
}
fn interval(lo: f32, hi: f32, key: i32, what: u32, rgb: vec3f) {
  if (key != DARK && key == openKey) { openTo = hi; return; }
  close();
  openKey = key; openFrom = lo; openTo = hi; openWhat = what; openRgb = rgb;
}
fn dark() { interval(0.0, 0.0, DARK, 0u, vec3f(0.0)); }

fn walkBreaks() {
  var s = first;
  if (fast && breakCount > 0) { s = breaks[0]; }
  var at = 0;
  loop {
    var next: f32;
    if (fast) { if (at + 1 >= breakCount) { break; } next = breaks[at + 1]; at++; }
    else { if (s >= last) { break; } next = nextBreak(s, last); }
    let middle = 0.5 * (s + next);
    var hit = -2;
    if (next - s >= 1e-7 && (smoothSky() || inSomeSpan(middle))) { hit = firstHit(middle); }
    let what = select(0u, 1u, smoothSky()) | select(0u, 2u, middle > sun.x && middle < sun.y);
    if (isLight(hit)) { interval(atan(s), atan(next), hit, 0u, shapes[hit].light.rgb); }
    else if (hit == -1 && what != 0u) { interval(atan(s), atan(next), skyKey(what), what, vec3f(0.0)); }
    else { dark(); }
    s = next;
  }
}

// the l-th light (in angle order while the spans are held): its span of slope and its box; an empty span for none
fn lightAt(l: i32) -> vec3f {
  if (spansHeld) { return vec3f(spans[l], f32(spanBox[l])); }
  let j = u32(l);
  if (beyond(j)) { return vec3f(1.0, -1.0, 0.0); }
  return vec3f(span(j), f32(j));
}
// rays evenly over angles [lo, hi], each lighting its share if what it meets first is the light j (-1: the sky's what)
fn spread(lo: f32, hi: f32, rays: i32, j: i32, what: u32) {
  let width = (hi - lo) / f32(rays);
  for (var k = 0; k < rays; k++) {
    let a = lo + f32(k) * width;
    if (firstHit(tan(a + 0.5 * width)) != j) { dark(); continue; }
    if (j >= 0) { interval(a, a + width, j, 0u, shapes[j].light.rgb); } else { interval(a, a + width, skyKey(what), what, vec3f(0.0)); }
  }
  dark();
}
fn walkShares() {
  let n = select(i32(world.x), lightCount, spansHeld);
  let sunAngles = atan(sun);
  let skyEnergy = select(0.0, dot(skyOver(f, atan(first), atan(last), 1u), LUMA), smoothSky());
  var energy = skyEnergy + select(0.0, dot(skyOver(f, sunAngles.x, sunAngles.y, 2u), LUMA), sun.y > sun.x);
  for (var l = 0; l < n; l++) {
    let light = lightAt(l);
    if (light.x < light.y) { energy += dot(shapes[u32(light.z)].light.rgb, LUMA) * (atan(light.y) - atan(light.x)); }
  }
  let perEnergy = f32(RAYS) / max(energy, 1e-30);
  for (var l = 0; l < n; l++) {
    let light = lightAt(l);
    if (light.x >= light.y) { continue; }
    let lo = atan(light.x); let hi = atan(light.y); let j = u32(light.z);
    spread(lo, hi, max(1, i32(round(perEnergy * dot(shapes[j].light.rgb, LUMA) * (hi - lo)))), i32(j), 0u);
  }
  if (smoothSky()) { spread(atan(first), atan(last), max(1, i32(round(perEnergy * skyEnergy))), -1, 1u); }
  if (sun.y > sun.x) { spread(sunAngles.x, sunAngles.y, max(1, i32(round(perEnergy * dot(skyOver(f, sunAngles.x, sunAngles.y, 2u), LUMA)))), -1, 2u); }
}

fn shellSortBreaks() {
  for (var gap = 121; gap > 0; gap = select(gap / 3, 0, gap == 1)) {   // gaps 121, 40, 13, 4, 1
    for (var i = gap; i < breakCount; i++) {
      let v = breaks[i]; var m = i - gap;
      while (m >= 0 && breaks[m] > v) { breaks[m + gap] = breaks[m]; m -= gap; }
      breaks[m + gap] = v;
    }
  }
}
fn shellSortSpans() {
  for (var gap = 121; gap > 0; gap = select(gap / 3, 0, gap == 1)) {
    for (var i = gap; i < lightCount; i++) {
      let v = spans[i]; let box = spanBox[i]; var m = i - gap;
      while (m >= 0 && spans[m].x > v.x) { spans[m + gap] = spans[m]; spanBox[m + gap] = spanBox[m]; m -= gap; }
      spans[m + gap] = v; spanBox[m + gap] = box;
    }
  }
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = i32(id.x + id.y * groups.x * 64u);
  if (index >= 4 * ${perFrustum}) { return; }
  f = index / ${perFrustum};
  let j = index % ${perFrustum};
  p = vec2f(0.0, f32(j / Q - ${back}) + (f32(j % Q) + 0.5) / f32(Q));
  lightCount = 0; breakCount = 0; fast = true;
  sun = sunSlopes(f);
  first = select(2.0, sun.x, sun.y > sun.x); last = select(-2.0, sun.y, sun.y > sun.x);
  for (var b = 0u; b < world.x; b++) {
    let sp = span(b);
    if (sp.x >= sp.y || (BUDGET && beyond(b))) { continue; }
    first = min(first, sp.x); last = max(last, sp.y);
    if (lightCount < FAST_LIGHTS) { spans[lightCount] = sp; spanBox[lightCount] = b; } else { fast = false; }
    lightCount++;
  }
  spansHeld = fast;
  if (smoothSky()) { first = -1.0; last = 1.0; }
  let breakLimit = select(FAST_BREAKS, min(FAST_BREAKS, RAYS + 1), BUDGET);
  if (fast) {
    if (smoothSky()) { breaks[0] = -1.0; breaks[1] = 1.0; breakCount = 2; }
    if (sun.y > sun.x) { breaks[breakCount] = sun.x; breaks[breakCount + 1] = sun.y; breakCount += 2; }
    for (var l = 0; l < lightCount && breakCount + 2 <= FAST_BREAKS; l++) { breaks[breakCount] = spans[l].x; breaks[breakCount + 1] = spans[l].y; breakCount += 2; }
    let n = f32(N);
    for (var b = 0u; b <= world.x && fast; b++) {   // box corners, and the domain's (b = count)
      var box = vec4f(0.0, 0.0, n, n);
      if (b < world.x) { box = frustumBox(f, shapes[b].box); if (BUDGET && beyond(b)) { continue; } }
      for (var c = 0; c < 4; c++) {
        let d = corner(box, c);
        if (abs(d.x) <= 1e-9) { continue; }
        let slope = d.y / d.x;
        if (!(smoothSky() && slope > -1.0 && slope < 1.0) && !inSomeSpan(slope)) { continue; }
        if (breakCount < breakLimit) { breaks[breakCount] = slope; breakCount++; } else { fast = false; }
      }
    }
    if (breakCount > breakLimit) { fast = false; }
    if (fast) { shellSortBreaks(); }
  }
  let overRays = BUDGET && !fast;
  if (overRays) {
    atomicAdd(&counters[1], 1u);
    if (spansHeld) { shellSortSpans(); }
  }
  writing = false; join = 1u;
  for (var walk = 0; walk < 2; walk++) {
    if (walk == 1) {
      total = made;
      if (BUDGET) { join = max(1u, (total + PIECES - 1u) / PIECES); }
      let stored = (total + join - 1u) / join;
      if (BUDGET) { base = u32(index) * PIECES; atomicAdd(&counters[0], stored); } else { base = atomicAdd(&counters[0], stored); }
      if (join > 1u) { atomicAdd(&counters[2], 1u); }
      let sunSeen = sunOn() && sky.sun.w <= 0.0 && abs(sunAt(f)) < 0.78539816 && firstHit(tan(sunAt(f))) == -1;
      lists[index] = vec2u(base, stored | select(0u, SORTED, !overRays) | select(0u, SUN_SEEN, sunSeen));
      if (base + stored > world.z) { return; }
      if (join == 1u) {
        for (var m = 0u; m < min(total, ${HELD}u); m++) { pool[base + m] = held[m]; }
        if (total <= ${HELD}u) { return; }
      }
      writing = true;
    }
    made = 0u; openKey = DARK;
    if (overRays) { walkShares(); } else { walkBreaks(); }
    close();
  }
}`;
};

/** The sweep's buffers and pipeline for an N² domain, q start points per row and room for `boxes` boxes. Without a
 * budget the pool holds `pieces` (0: 4 per point) and the page grows it when a frame needs more; with one, it holds
 * `budget.pieces` per point. */
export class OutsideSweep {
  readonly points: number;
  readonly pieces: number;
  readonly shapes: GPUBuffer;
  readonly world: GPUBuffer;
  readonly sky: GPUBuffer;
  readonly producer: OutsideProducer;
  readonly #device: GPUDevice;
  readonly #pipeline: GPUComputePipeline;
  readonly #group: GPUBindGroup;
  readonly #counters: GPUBuffer;
  readonly #readback: GPUBuffer;
  readonly #buffers: GPUBuffer[];

  constructor(device: GPUDevice, readonly N: number, readonly q: number, readonly budget: Budget | null, readonly boxes = 256, pieces = 0) {
    this.#device = device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.points = 4 * pointLayout(N, q).perFrustum;
    this.pieces = budget ? budget.pieces * this.points : pieces || 4 * this.points;
    this.shapes = device.createBuffer({ size: boxes * 32, usage });
    this.sky = device.createBuffer({ size: 80, usage });
    const pool = device.createBuffer({ size: this.pieces * 16, usage });
    const lists = device.createBuffer({ size: this.points * 8, usage });
    this.#counters = device.createBuffer({ size: 16, usage });
    this.#readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.world = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.#pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: sweepWgsl(N, q, budget) }), entryPoint: 'sweep' } });
    this.#group = device.createBindGroup({ layout: this.#pipeline.getBindGroupLayout(0), entries: [this.shapes, this.world, pool, lists, this.#counters, this.sky].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.producer = { wgsl: producerWgsl(N, q), buffers: [pool, lists, this.sky] };
    this.#buffers = [this.shapes, this.sky, pool, lists, this.#counters, this.#readback, this.world];
  }

  /** boxes in domain pixels, 8 floats each: x0, y0, x1, y1, r, g, b, 1 if a light; false if it holds fewer */
  setShapes(data: Float32Array, count: number) {
    if (count > this.boxes) return false;
    this.#device.queue.writeBuffer(this.shapes, 0, data as Float32Array<ArrayBuffer>, 0, count * 8);
    this.#device.queue.writeBuffer(this.world, 0, new Uint32Array([count, 0, this.pieces, 0]));
    this.#device.queue.writeBuffer(this.#counters, 0, new Uint32Array(4));
    return true;
  }

  setSky(sky: Sky | null) {
    this.#device.queue.writeBuffer(this.sky, 0, skyData(sky));
  }

  encode(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#group);
    const groups = Math.ceil(this.points / 64);
    pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
  }

  /** after the sweep, in its encoder */
  encodeCounts(encoder: GPUCommandEncoder) {
    encoder.copyBufferToBuffer(this.#counters, 0, this.#readback, 0, 16);
  }

  /** the last encoded sweep's pieces (without a budget, more than `pieces` did not fit), points that shared their rays,
   * points that joined their pieces */
  async counts() {
    await this.#readback.mapAsync(GPUMapMode.READ);
    const [pieces, shared, joined] = new Uint32Array(this.#readback.getMappedRange().slice(0));
    this.#readback.unmap();
    return { pieces, shared, joined };
  }

  destroy() {
    this.#buffers.forEach((b) => b.destroy());
  }
}
