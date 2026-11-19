// Polygonal HRC (WebGPU).
//
// The scene is solved in four rotations by 90 degrees. In each, frustums look along +x, and frustums of direction >= 0 and
// < 0 are solved apart (view 1 and -1). Pixel p is the square [p, p + 1]; probes sit on pixel corners. Pixels are solid
// (transmittance <= 0: a surface of radiance E, 1 or 3 channels) or free; see-through pixels are read as free.
// An extension of size d holds what lies between two columns: a line ends at the near face of one pixel (end offset
// (d, k)), a cone at two (end pixels (d, k) and (d, k + 1)). Size 2d is built from size d and stored at start columns that
// are multiples of it. Transmittance is kept on lines only.
// Light is counted where it arrives at a face, faces normal to x and to y kept apart, and read at its exact angle as G, a
// complex value per channel whose real part is the light. A frustum (size s, apex f, direction i: slopes i / s .. (i + 1) / s)
// keeps G at two points on its axis and is read elsewhere by fitting 1 / G linearly in the point's complex position.

const WORKGROUP = 64;
const PARAM_STRIDE = 256;
const SAMPLE_SPACING = 1 / (2 - 1 / Math.SQRT2) / Math.SQRT2;

// floats per stored extension: light 2 x (channels + 2), runs 2, stairGuards 2, positionalGates 1 (y squared); lines
// also T 1 and, with positionalGates, two pixels 4
function widths({ channels, runs, stairGuards, positionalGates }) {
  const light = 2 * (channels + 2) + (runs ? 2 : 0) + (stairGuards ? 2 : 0) + (positionalGates ? 1 : 0);
  return { WL: 1 + light + (positionalGates ? 4 : 0), WC: light };
}
const cellFloats = (d, o) => { const { WL, WC } = widths(o); return (d + 1) * WL + d * WC; };
const rows = (N, d) => N + d + 4;

function shaderSource({ N, channels, runs, stairGuards, readSize2, positionalGates, sideCorners }) {
  const o = { channels, runs, stairGuards, positionalGates };
  const K = channels, COL = K === 3 ? 'vec3f' : 'f32';
  const { WL, WC } = widths(o);
  const levelOffsets = [0, 0];
  for (let d = 2; d < N; d *= 2) levelOffsets.push(levelOffsets.at(-1) + (N / d) * rows(N, d) * cellFloats(d, o));
  // an extension's light from offset base: ar (K), ap (2), br (K), bp (2), [q (2)], [g (2)], [yy (1)]; a line's pixels
  // follow at its offset C
  const AP = K, BR = K + 2, BP = 2 * K + 2, Q = 2 * K + 4, G = Q + (runs ? 2 : 0), Y = G + (stairGuards ? 2 : 0), C = Y + (positionalGates ? 1 : 0);
  const at = (base, k) => `extensions[${base} + ${k}]`;
  const col = (base, k) => (K === 3 ? `vec3f(${at(base, k)}, ${at(base, k + 1)}, ${at(base, k + 2)})` : at(base, k));
  const pair = (base, k) => `vec2f(${at(base, k)}, ${at(base, k + 1)})`;
  const readLight = (base) => `${col(base, 0)}, ${pair(base, AP)}, ${col(base, BR)}, ${pair(base, BP)},
    ${runs ? pair(base, Q) : 'vec2f(0.0)'}, ${stairGuards ? pair(base, G) : 'vec2f(0.0)'}, ${positionalGates ? at(base, Y) : '0.0'}`;
  const putCol = (base, k, v) => (K === 3 ? [0, 1, 2].map((c) => `${at(base, k + c)} = ${v}[${c}];`).join(' ') : `${at(base, k)} = ${v};`);
  const putPair = (base, k, v) => `${at(base, k)} = ${v}.x; ${at(base, k + 1)} = ${v}.y;`;
  const writeLight = (base) => [
    `${putCol(base, 0, 'e.ar')} ${putPair(base, AP, 'e.ap')}`,
    `${putCol(base, BR, 'e.br')} ${putPair(base, BP, 'e.bp')}`,
    runs ? putPair(base, Q, 'e.q') : '',
    stairGuards ? putPair(base, G, 'e.g') : '',
    positionalGates ? `${at(base, Y)} = e.yy;` : '',
  ].join('\n  ');

  // the merge of a frustum, written once for stored children and once for children evaluated at the probe (readSize2)
  const merge = (name, child) => /* wgsl */ `
fn ${name}(f: vec2i, s: i32, i: i32) -> Value {
  if (f.x + s >= N) { return zeroValue(); }
  let start = f + vec2i(s, i); let s2 = 2 * s; let o = vec2f(f32(s), f32(i));
  // even (f.x / s even): the near cone, from the start pixel, plus the two children of size 2s at the same apex, each
  // behind the line from the start pixel to its first pixel
  if (((f.x >> sizeShift(s)) & 1) == 0) {
    var r = recordValue(coneExt(s, start.x, start.y, i), start, f, s, i);
    for (var k = 0; k < 2; k++) {
      let j = 2 * i + k;
      let c = ${child('f', 'j')};
      r = addValue(r, c, gate(o, lineExt(s, start.x, start.y, i + k), f32(j) / f32(s2), f32(j + 1) / f32(s2), c.m));
    }
    return r;
  }
  // odd: half the short merge (the children one size back) plus half the long merge (children one size ahead)
  let b0 = f - vec2i(s, i); let b1 = f - vec2i(s, i + 1);
  var shortMerge = addValue(${child('b0', '2 * i')}, ${child('b1', '2 * i + 1')}, 1.0);
  // the long merge's near part, from the stored size-2s extensions at the start pixel (an odd frustum's start column is a
  // multiple of 2s): bottom cone (end offset (2s, 2i)) + top cone ((2s, 2i + 1)) - 2 x middle line ((2s, 2i + 1)), since
  // the cones overlap on that line and both short children count it
  var bottom = coneExt(s2, start.x, start.y, 2 * i); var top = coneExt(s2, start.x, start.y, 2 * i + 1);
  let r0 = f32(i) / f32(s); let rm = f32(2 * i + 1) / f32(s2); let r1 = f32(i + 1) / f32(s);
  if (SIDE_CORNERS) {
    // the two cones seen from f instead of from their own apex a size behind, built from their nine size-s extensions; the
    // short children hold the same two cones as their near parts, so they are swapped there too
    let lower = start + vec2i(s, i); let upper = start + vec2i(s, i + 1);
    let dl = vec2f(f32(s), f32(i)); let du = vec2f(f32(s), f32(i + 1));
    let l00 = lineExt(s, start.x, start.y, i); let l01 = lineExt(s, start.x, start.y, i + 1);
    let ll0 = lineExt(s, lower.x, lower.y, i); let ll1 = lineExt(s, lower.x, lower.y, i + 1);
    let lu0 = lineExt(s, upper.x, upper.y, i); let lu1 = lineExt(s, upper.x, upper.y, i + 1);
    let c0 = coneExt(s, start.x, start.y, i); let cl = coneExt(s, lower.x, lower.y, i); let cu = coneExt(s, upper.x, upper.y, i);
    let bottomF = halfCone(l00, ll0, c0, cl, lu0, l00, l01, dl, du, o, r0, r1);
    let topF = halfCone(l01, lu1, c0, cu, ll1, l01, l00, du, dl, o, r0, r1);
    shortMerge = addValue(shortMerge, recordValue(bottom, start, f, s, i), -1.0);
    shortMerge = addValue(shortMerge, recordValue(top, start, f, s, i), -1.0);
    shortMerge = addValue(shortMerge, recordValue(bottomF, start, f, s, i), 1.0);
    shortMerge = addValue(shortMerge, recordValue(topF, start, f, s, i), 1.0);
    bottom = bottomF; top = topF;
  }
  let middle = lineExt(s2, start.x, start.y, 2 * i + 1);
  var longMerge = addValue(addValue(recordValue(bottom, start, f, s, i), recordValue(top, start, f, s, i), 1.0), recordValue(middle, start, f, s, i), -2.0);
  let a0 = f + vec2i(s, i); let a1 = f + vec2i(s, i + 1);
  let v0 = ${child('a0', '2 * i')}; let v1 = ${child('a1', '2 * i + 1')};
  longMerge = addValue(longMerge, v0, gate(o, lineExt(s2, start.x, start.y, 2 * i), r0, rm, v0.m));
  longMerge = addValue(longMerge, v1, gate(o, lineExt(s2, start.x, start.y, 2 * i + 2), rm, r1, v1.m));
  return scaleValue(addValue(shortMerge, longMerge, 1.0), 0.5);
}`;

  return /* wgsl */ `
const N: i32 = ${N};
const WL: i32 = ${WL};
const WC: i32 = ${WC};
const RUNS: bool = ${!!runs};
const STAIR: bool = ${!!stairGuards};
const POSITIONAL: bool = ${!!positionalGates};
const SIDE_CORNERS: bool = ${!!sideCorners};
const K: i32 = ${K};   // channels
alias Col = ${COL};
const VS: i32 = ${K + (positionalGates ? 1 : 0)};   // vec4s per stored frustum value: one per channel, then moments (positionalGates)
const SAMPLE_SPACING: f32 = ${SAMPLE_SPACING.toPrecision(17)};
const QUARTER_PI: f32 = 0.78539816339744831;
// frustums of size pinS put both samples at pinPoint (from their apex), for the probe's reads; 0 = none
var<private> pinS: i32 = 0;
var<private> pinPoint: vec2f = vec2f(0.0);

struct PassParams { level: i32, rotation: i32, view: i32, first: i32 }

@group(0) @binding(0) var<storage, read> emission: array<f32>;
@group(0) @binding(1) var<storage, read> transmittance: array<f32>;
@group(0) @binding(2) var<storage, read_write> extensions: array<f32>;
@group(0) @binding(3) var<storage, read> childValues: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> frustumValues: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> fluence: array<f32>;
@group(0) @binding(6) var<uniform> params: PassParams;

fn threadIndex(id: vec3u, groups: vec3u) -> i32 { return i32(id.x + id.y * groups.x * ${WORKGROUP}u); }
// sizes are powers of two: x / d is x >> sizeShift(d) for x >= 0
fn sizeShift(d: i32) -> u32 { return countTrailingZeros(u32(d)); }

// the scene in a rotation
fn scenePixelIndex(p: vec2i) -> i32 {
  var r = p;
  for (var k = 0; k < params.rotation; k++) { r = vec2i(r.y, N - 1 - r.x); }
  return r.x * N + r.y;
}
fn inside(p: vec2i) -> bool { return p.x >= 0 && p.x < N && p.y >= 0 && p.y < N; }
fn solid(p: vec2i) -> bool { return inside(p) && transmittance[scenePixelIndex(p)] <= 0.0; }
fn pixelE(p: vec2i) -> Col {
  if (!inside(p)) { return Col(0.0); }
  let k = scenePixelIndex(p) * K;
  return ${K === 3 ? 'vec3f(emission[k], emission[k + 1], emission[k + 2])' : 'emission[k]'};
}
fn lum(c: Col) -> f32 { return ${K === 3 ? 'dot(c, vec3f(0.2126, 0.7152, 0.0722))' : 'c'}; }
fn channel(c: Col, k: i32) -> f32 { return ${K === 3 ? 'c[k]' : 'c'}; }

// complex numbers
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }
fn clogRatio(a: vec2f, b: vec2f) -> vec2f { return vec2f(0.5 * log(dot(a, a) / dot(b, b)), atan2(a.y * b.x - a.x * b.y, a.x * b.x + a.y * b.y)); }   // log(a / b)

// extension values
// t: T (lines); ar, br: radiance of faces normal to x and to y; ap, bp: their luminance x midpoint from the start pixel;
// q (runs): ar's luminance x y^2, br's luminance x x^2; g (stairGuards): br's luminance with a stair corner at gap 2, at
// gap 3; yy (positionalGates): luminance x y^2 over all faces; cn (positionalGates, lines): the most intruding solid pixels
struct Cn { lo: vec2f, hi: vec2f }   // lower-left corners from the start pixel; |y| > 1e3: none
struct Rec { t: f32, ar: Col, ap: vec2f, br: Col, bp: vec2f, q: vec2f, g: vec2f, yy: f32, cn: Cn }
fn noCn() -> Cn { return Cn(vec2f(0.0, -1e4), vec2f(0.0, 1e4)); }
fn emptyRec() -> Rec { return Rec(1.0, Col(0.0), vec2f(0.0), Col(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0), 0.0, noCn()); }
// e's light plus T x the light of b, which starts at offset d from e's start
fn behind(e: Rec, T: f32, b: Rec, d: vec2f) -> Rec {
  var r = e;
  let la = lum(b.ar); let lb = lum(b.br);
  r.ar += T * b.ar; r.ap += T * (b.ap + la * d);
  r.br += T * b.br; r.bp += T * (b.bp + lb * d);
  if (RUNS) { r.q += T * vec2f(b.q.x + 2.0 * d.y * b.ap.y + d.y * d.y * la, b.q.y + 2.0 * d.x * b.bp.x + d.x * d.x * lb); }
  r.g += T * b.g;
  if (POSITIONAL) { r.yy += T * (b.yy + 2.0 * d.y * (b.ap.y + b.bp.y) + d.y * d.y * (la + lb)); }
  return r;
}
fn scaleLight(e: Rec, k: f32) -> Rec {
  var r = e; r.ar *= k; r.ap *= k; r.br *= k; r.bp *= k; r.q *= k; r.g *= k; r.yy *= k; return r;
}
fn addLight(e: Rec, f: Rec) -> Rec {
  var r = e; r.ar += f.ar; r.ap += f.ap; r.br += f.br; r.bp += f.bp; r.q += f.q; r.g += f.g; r.yy += f.yy; return r;
}
// lines: a then b (b starts at offset d)
fn over(a: Rec, b: Rec, d: vec2f) -> Rec { var e = behind(a, a.t, b, d); e.t = a.t * b.t; return e; }

// stairGuards: the share of an extension's y-face light hidden by a stair corner from a reader at offset -z from the faces'
// mean position; at gap g the hidden length is clamp(1 / |slope| - g, 0, 1)
fn stairHidden(e: Rec, z: vec2f) -> f32 {
  let lb = lum(e.br);
  if (!STAIR || lb == 0.0) { return 0.0; }
  let r = abs(z.x) / max(abs(z.y), 1e-6);   // 1 / |slope|
  return (e.g.x * clamp(r - 1.0, 0.0, 1.0) + e.g.y * clamp(r - 2.0, 0.0, 1.0)) / lb;
}
// an extension's light seen from offset p from its start, as G per channel
fn readG(e: Rec, p: vec2f) -> array<vec2f, K> {
  var G: array<vec2f, K>;
  let la = lum(e.ar);
  if (la != 0.0) {
    let m = e.ap / la; let z = m - p;
    var k = 1.0;
    if (RUNS) { k = sqrt(max(12.0 * (e.q.x / la - m.y * m.y), 0.0) + 1.0); }
    let d = clogRatio(z + vec2f(0.0, 0.5 * k), z - vec2f(0.0, 0.5 * k));
    for (var c = 0; c < K; c++) { G[c] += (channel(e.ar, c) / k) * vec2f(d.y, -d.x); }   // W = 1, faces along y
  }
  let lb = lum(e.br);
  if (lb != 0.0) {
    let m = e.bp / lb; let z = m - p;
    var k = 1.0;
    if (RUNS) { k = sqrt(max(12.0 * (e.q.y / lb - m.x * m.x), 0.0) + 1.0); }
    let d = clogRatio(z + vec2f(0.5 * k, 0.0), z - vec2f(0.5 * k, 0.0));
    let h = 1.0 - stairHidden(e, z);
    for (var c = 0; c < K; c++) { G[c] += (f32(params.view) * channel(e.br, c) / k * h) * vec2f(-d.y, d.x); }   // W = i view, faces along x
  }
  return G;
}

// corner transmittance (positionalGates)
// the corner of pixel q (lower-left corner) that rays must pass above (cLo) or below (cHi); rays rise in view 1
fn cLo(q: vec2f) -> vec2f { return select(q + vec2f(1.0, 1.0), q + vec2f(0.0, 1.0), params.view > 0); }
fn cHi(q: vec2f) -> vec2f { return select(q, q + vec2f(1.0, 0.0), params.view > 0); }
fn chordY(d: f32, k: f32, x: f32) -> f32 { return 0.5 + k * x / d; }
// keep the more intruding of c's pixels and q, relative to the chord of a line of size d and end offset k
fn addC(c: Cn, q: vec2f, d: f32, k: f32) -> Cn {
  if (abs(q.y) > 1e3) { return c; }
  var r = c;
  if (q.y + 0.5 - chordY(d, k, q.x + 0.5) < 0.0) {
    let n = cLo(q); let p = cLo(c.lo);
    if (abs(c.lo.y) > 1e3 || n.y - chordY(d, k, n.x) > p.y - chordY(d, k, p.x)) { r.lo = q; }
  } else {
    let n = cHi(q); let p = cHi(c.hi);
    if (abs(c.hi.y) > 1e3 || chordY(d, k, n.x) - n.y > chordY(d, k, p.x) - p.y) { r.hi = q; }
  }
  return r;
}
// c with both pixels of x (whose start pixel is at s) as candidates
fn joinCn(c: Cn, x: Cn, s: vec2f, d: f32, k: f32) -> Cn { return addC(addC(c, x.lo + s, d, k), x.hi + s, d, k); }
// an extension's light as (mass, mass x centroid, mass x y^2) from a reader at its start - o
fn lightOf(r: Rec, o: vec2f) -> vec4f {
  let m = lum(r.ar) + lum(r.br);
  return vec4f(m, r.ap + r.bp + m * o, r.yy + 2.0 * o.y * (r.ap.y + r.bp.y) + o.y * o.y * m);
}
// the transmittance of line e (start pixel at o from the reader) for light L behind it (moments from the reader): the share
// of the light's slopes that passes the line's most intruding solid pixels. [r0, r1]: the slopes the light can have, used
// when it has no interval of its own
fn gate(o: vec2f, e: Rec, r0: f32, r1: f32, L: vec4f) -> f32 {
  if (!POSITIONAL) { return e.t; }
  if (e.t <= 0.0) { return 0.0; }
  var l = r0; var h = r1;
  if (L.x > 1e-12 && L.y > 1e-3 * L.x) {
    let c = L.yz / L.x;
    let w = sqrt(max(12.0 * (L.w / L.x - c.y * c.y), 0.0)) / c.x;
    l = c.y / c.x - 0.5 * w; h = c.y / c.x + 0.5 * w;
  }
  var slo = -1e30; var shi = 1e30;
  if (abs(e.cn.lo.y) < 1e3) { let p = o + cLo(e.cn.lo); if (p.x > 1e-6) { slo = p.y / p.x; } }
  if (abs(e.cn.hi.y) < 1e3) { let p = o + cHi(e.cn.hi); if (p.x > 1e-6) { shi = p.y / p.x; } }
  if (h - l < 1e-7) { return select(0.0, 1.0, l >= slo && l <= shi); }
  return clamp((min(h, shi) - max(l, slo)) / (h - l), 0.0, 1.0);
}

// extensions
// size 1, start pixel p, end offset (1, k). A solid start pixel shows its near face and blocks; a free one shows the side
// face of a solid pixel beside it (row y + view), which blocks lines of slope +-1.
fn leaf(p: vec2i, k: i32, cone: bool) -> Rec {
  if (solid(p)) {
    let e = pixelE(p); let l = lum(e);
    return Rec(0.0, e, vec2f(0.0, 0.5 * l), Col(0.0), vec2f(0.0), vec2f(0.25 * l, 0.0), vec2f(0.0), l * (0.25 + 1.0 / 12.0), addC(noCn(), vec2f(0.0), 1.0, f32(k)));
  }
  var r = emptyRec();
  let side = p + vec2i(0, params.view);
  if (solid(side)) {
    let e = pixelE(side); let l = lum(e);
    r.br = e; r.bp = vec2f(0.5 * l, select(0.0, l, params.view > 0)); r.q = vec2f(0.0, 0.25 * l); r.yy = r.bp.y;
    if (STAIR && l > 0.0 && !solid(p - vec2i(1, 0))) {
      if (solid(p - vec2i(2, 0))) { r.g = vec2f(l, 0.0); } else if (solid(p - vec2i(3, 0))) { r.g = vec2f(0.0, l); }
    }
    if (!cone && k != 0) { r.t = 0.0; }
    if (!cone && (k != 0 || SIDE_CORNERS)) { r.cn = addC(r.cn, vec2f(0.0, f32(params.view)), 1.0, f32(k)); }
  }
  return r;
}

// store: level n (size d = 2^n, n >= 1): [x / d][y - firstRow], then the d + 1 lines by |k| and the d cones
fn rowsAt(d: i32) -> i32 { return N + d + 4; }
fn firstRow(d: i32) -> i32 { return select(-2, -d - 3, params.view > 0); }
fn cellFloats(d: i32) -> i32 { return (d + 1) * WL + d * WC; }
const LEVEL_OFFSETS = array<i32, ${levelOffsets.length}>(${levelOffsets.join(', ')});   // where each size's store begins
fn cellOffset(d: i32, x: i32, y: i32) -> i32 {
  let r = y - firstRow(d);
  if (x < 0 || x >= N || r < 0 || r >= rowsAt(d)) { return -1; }
  let n = sizeShift(d);
  return LEVEL_OFFSETS[n] + ((x >> n) * rowsAt(d) + r) * cellFloats(d);
}
fn lineExt(d: i32, x: i32, y: i32, k: i32) -> Rec {
  if (d == 1) { return leaf(vec2i(x, y), k, false); }
  let c = cellOffset(d, x, y);
  if (c < 0) { return emptyRec(); }
  let o = c + abs(k) * WL;
  return Rec(extensions[o], ${readLight('o + 1')},
    ${positionalGates ? `Cn(vec2f(extensions[o + ${1 + C}], extensions[o + ${2 + C}]), vec2f(extensions[o + ${3 + C}], extensions[o + ${4 + C}]))` : 'noCn()'});
}
fn coneExt(d: i32, x: i32, y: i32, k: i32) -> Rec {
  if (d == 1) { return leaf(vec2i(x, y), k, true); }
  let c = cellOffset(d, x, y);
  if (c < 0) { return emptyRec(); }
  let o = c + (d + 1) * WL + select(-1 - k, k, params.view > 0) * WC;
  return Rec(1.0, ${readLight('o')}, noCn());
}
fn storeLine(o: i32, e: Rec) {
  extensions[o] = e.t;
  ${writeLight('o + 1')}${positionalGates ? `
  extensions[o + ${1 + C}] = e.cn.lo.x; extensions[o + ${2 + C}] = e.cn.lo.y; extensions[o + ${3 + C}] = e.cn.hi.x; extensions[o + ${4 + C}] = e.cn.hi.y;` : ''}
}
fn storeCone(o: i32, e: Rec) {
  ${writeLight('o')}
}

// a line of size s2 and end offset k made of halves p and q (q starts at offset d)
fn line2From(p: Rec, q: Rec, d: vec2f, s2: i32, k: i32) -> Rec {
  var e = over(p, q, d);
  if (POSITIONAL) { e.cn = joinCn(joinCn(noCn(), p.cn, vec2f(0.0), f32(s2), f32(k)), q.cn, d, f32(s2), f32(k)); }
  return e;
}
// the line of size 2h from pixel (x, y) to offset (2h, k), from lines of size h; for odd k, the mean of passing below and
// above the middle pixel
fn line2(h: i32, x: i32, y: i32, k: i32) -> Rec {
  let lo = k >> 1u;
  if ((k & 1) == 0) { return line2From(lineExt(h, x, y, lo), lineExt(h, x + h, y + lo, lo), vec2f(f32(h), f32(lo)), 2 * h, k); }
  let hi = lo + 1;
  let b1 = lineExt(h, x, y, lo); let b2 = lineExt(h, x + h, y + lo, hi);
  let a1 = lineExt(h, x, y, hi); let a2 = lineExt(h, x + h, y + hi, lo);
  let below = over(b1, b2, vec2f(f32(h), f32(lo))); let above = over(a1, a2, vec2f(f32(h), f32(hi)));
  var e = scaleLight(addLight(below, above), 0.5);
  e.t = 0.5 * (below.t + above.t);
  if (POSITIONAL) {
    let d = f32(2 * h); let kk = f32(k);
    e.cn = joinCn(joinCn(joinCn(joinCn(noCn(), b1.cn, vec2f(0.0), d, kk), b2.cn, vec2f(f32(h), f32(lo)), d, kk), a1.cn, vec2f(0.0), d, kk), a2.cn, vec2f(f32(h), f32(hi)), d, kk);
  }
  return e;
}
// a cone of size 2h from extensions of size h: half of the edge line (edge1 then edge2), the first cone, the second cone
// (behind the line toFull) and the narrow line (behind the line toHalf). dFull, dHalf: the full and the half middle
// pixel; o: the cone's start pixel from its reader; [r0, r1]: the slopes the cone covers from there
fn halfCone(edge1: Rec, edge2: Rec, first: Rec, second: Rec, narrow: Rec, toFull: Rec, toHalf: Rec, dFull: vec2f, dHalf: vec2f, o: vec2f, r0: f32, r1: f32) -> Rec {
  var e = behind(edge1, gate(o, edge1, r0, r1, lightOf(edge2, o + dFull)), edge2, dFull);
  e = behind(e, 1.0, first, vec2f(0.0));
  e = behind(e, gate(o, toFull, r0, r1, lightOf(second, o + dFull)), second, dFull);
  e = behind(e, gate(o, toHalf, r0, r1, lightOf(narrow, o + dHalf)), narrow, dHalf);
  return scaleLight(e, 0.5);
}
// the cone of size 2h from pixel (x, y) to offset (2h, k), seen from its own apex
fn cone2(h: i32, x: i32, y: i32, k: i32) -> Rec {
  let lo = k >> 1u; let even = (k & 1) == 0;
  let full = select(lo + 1, lo, even); let half = select(lo, lo + 1, even);   // the rows of the full and the half middle pixel
  let dFull = vec2f(f32(h), f32(full)); let dHalf = vec2f(f32(h), f32(half));
  let toFull = lineExt(h, x, y, full);
  return halfCone(toFull, lineExt(h, x + h, y + full, full), coneExt(h, x, y, lo), coneExt(h, x + h, y + full, lo),
                  lineExt(h, x + h, y + half, full), toFull, lineExt(h, x, y, half), dFull, dHalf,
                  vec2f(f32(2 * h), f32(k)), f32(k) / f32(2 * h), f32(k + 1) / f32(2 * h));
}

@compute @workgroup_size(${WORKGROUP})
fn extend(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let d = 1 << u32(params.level); let h = d >> 1u; let view = params.view;
  let rows = rowsAt(d); let slots = 2 * d + 1;
  let index = threadIndex(id, groups);
  if (index >= (N / d) * rows * slots) { return; }
  let slot = index % slots; let r = (index / slots) % rows; let column = index / (slots * rows);
  let x = column * d; let y = r + firstRow(d);
  let base = LEVEL_OFFSETS[params.level] + (column * rows + r) * cellFloats(d);
  if (slot <= d) { storeLine(base + slot * WL, line2(h, x, y, view * slot)); }
  else { let c = slot - d - 1; storeCone(base + (d + 1) * WL + c * WC, cone2(h, x, y, select(-1 - c, c, view > 0))); }
}

// frustums
// a frustum value: g, per channel G at the two samples (offsets +u and -u from the apex) as (G+, G-); m (positionalGates):
// its light's (mass, mass x centroid, mass x y^2) from the apex, by luminance
struct Value { g: array<vec4f, K>, m: vec4f }
fn zeroValue() -> Value { var v: Value; return v; }
fn addValue(a: Value, b: Value, w: f32) -> Value {
  var r = a;
  for (var c = 0; c < K; c++) { r.g[c] += w * b.g[c]; }
  r.m += w * b.m;
  return r;
}
fn scaleValue(a: Value, w: f32) -> Value {
  var r = a;
  for (var c = 0; c < K; c++) { r.g[c] *= w; }
  r.m *= w;
  return r;
}
// frustum (size s, apex f, direction i): axis (s, i + 1/2), samples f +- SAMPLE_SPACING s (1, (i + 1/2) / s)
fn sampleOffset(s: i32, i: i32) -> vec2f { return SAMPLE_SPACING * f32(s) * vec2f(1.0, (f32(i) + 0.5) / f32(s)); }
fn sampleA(s: i32, i: i32) -> vec2f { if (s == pinS) { return pinPoint; } return sampleOffset(s, i); }
fn sampleB(s: i32, i: i32) -> vec2f { if (s == pinS) { return pinPoint; } return -sampleOffset(s, i); }
// G of a frustum value at offset p from its apex: 1 / G linear in p
fn readValue(v: vec4f, s: i32, i: i32, p: vec2f) -> vec2f {
  let u = sampleOffset(s, i);
  let w0 = cdiv(p + u, 2.0 * u); let w1 = cdiv(u - p, 2.0 * u);
  let G0 = v.xy; let G1 = v.zw; let m0 = length(G0); let m1 = length(G1);
  if (min(m0, m1) > 1e-12 * max(max(m0, m1), 1e-30)) { return cdiv(vec2f(1.0, 0.0), cdiv(w0, G0) + cdiv(w1, G1)); }
  return cmul(w0, G0) + cmul(w1, G1);
}
// moments from an apex moved by -d (d = old apex - new apex)
fn shiftM(m: vec4f, d: vec2f) -> vec4f { return vec4f(m.x, m.yz + m.x * d, m.w + 2.0 * d.y * m.z + d.y * d.y * m.x); }
// a child (apex c, size s2, direction j) at the samples of frustum (f, s, i)
fn reproject(v: Value, c: vec2i, s2: i32, j: i32, f: vec2i, s: i32, i: i32) -> Value {
  let o = vec2f(f - c);
  var r: Value;
  for (var k = 0; k < K; k++) { r.g[k] = vec4f(readValue(v.g[k], s2, j, o + sampleA(s, i)), readValue(v.g[k], s2, j, o + sampleB(s, i))); }
  r.m = shiftM(v.m, vec2f(c - f));
  return r;
}
// an extension's light (from pixel start) at the samples of frustum (f, s, i)
fn recordValue(e: Rec, start: vec2i, f: vec2i, s: i32, i: i32) -> Value {
  let o = vec2f(f - start);
  let ga = readG(e, o + sampleA(s, i)); let gb = readG(e, o + sampleB(s, i));
  var r: Value;
  for (var c = 0; c < K; c++) { r.g[c] = vec4f(ga[c], gb[c]); }
  if (POSITIONAL) { r.m = lightOf(e, -o); }
  return r;
}

// stored frustums of size s: [f.x / s][f.y - firstApexRow][|direction|], f.x a multiple of s
fn apexRows(s: i32) -> i32 { return N + s + 2; }
fn firstApexRow(s: i32) -> i32 { return select(-2, 1 - s, params.view > 0); }
fn valueIndex(s: i32, c: vec2i, j: i32) -> i32 {
  let r = c.y - firstApexRow(s);
  if (c.x < 0 || c.x + s >= N || r < 0 || r >= apexRows(s)) { return -1; }
  return (((c.x >> sizeShift(s)) * apexRows(s) + r) * s + select(-1 - j, j, params.view > 0)) * VS;
}
fn childValue(s2: i32, c: vec2i, j: i32) -> Value {
  let k = valueIndex(s2, c, j);
  var v: Value;
  if (k < 0) { return v; }
  // a size-2 frustum evaluated at the probe reads its size-4 children from the other buffer, which still holds them
  for (var n = 0; n < VS; n++) {
    var x: vec4f;
    if (pinS == 2 && s2 == 4) { x = frustumValues[k + n]; } else { x = childValues[k + n]; }
    if (n < K) { v.g[n] = x; } else { v.m = x; }
  }
  return v;
}
${merge('frustumValue', (c, j) => `reproject(childValue(s2, ${c}, ${j}), ${c}, s2, ${j}, f, s, i)`)}
${readSize2 ? `
// a size-2 child (apex c, direction j) evaluated with both samples at the point the probe f reads it from
fn childAtProbe(c: vec2i, j: i32, f: vec2i) -> Value {
  if (valueIndex(2, c, j) < 0) { return zeroValue(); }
  pinS = 2; pinPoint = vec2f(f - c);
  let v = frustumValue(c, 2, j);
  pinS = 1; pinPoint = vec2f(0.0);
  return Value(v.g, shiftM(v.m, vec2f(c - f)));
}
${merge('probeValue', (c, j) => `childAtProbe(${c}, ${j}, f)`)}` : ''}

@compute @workgroup_size(${WORKGROUP})
fn mergeCascade(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let s = 1 << u32(params.level); let rows = apexRows(s);
  let index = threadIndex(id, groups);
  if (index >= (N / s) * rows * s) { return; }
  let slot = index % s; let r = (index / s) % rows; let column = index / (s * rows);
  let v = frustumValue(vec2i(column * s, r + firstApexRow(s)), s, select(-1 - slot, slot, params.view > 0));
  for (var c = 0; c < K; c++) { frustumValues[index * VS + c] = v.g[c]; }
  if (POSITIONAL) { frustumValues[index * VS + K] = v.m; }
}

// gather at the probes (all corners 0 .. N), added up over rotations and views in scene coordinates; [corner * K + channel]
@compute @workgroup_size(${WORKGROUP})
fn gatherProbes(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = threadIndex(id, groups);
  if (index >= (N + 1) * (N + 1)) { return; }
  let probe = vec2i(index / (N + 1), index % (N + 1));
  let i = select(-1, 0, params.view > 0);
  let near = probe + vec2i(0, i);
  var v: array<f32, K>;
  if (solid(near)) {   // the probe is on its surface: it fills the frustum
    let e = pixelE(near) * QUARTER_PI;
    for (var c = 0; c < K; c++) { v[c] = channel(e, c); }
  } else {
    pinS = 1; pinPoint = vec2f(0.0);   // both samples of the probe's size-1 frustum at the probe
    let f = ${readSize2 ? 'probeValue' : 'frustumValue'}(probe, 1, i);
    for (var c = 0; c < K; c++) { v[c] = f.g[c].x; }
    pinS = 0;
  }
  var q = probe;
  for (var k = 0; k < params.rotation; k++) { q = vec2i(q.y, N - q.x); }
  let o = (q.x * (N + 1) + q.y) * K;
  for (var c = 0; c < K; c++) {
    if (params.first == 1) { fluence[o + c] = v[c]; } else { fluence[o + c] += v[c]; }
  }
}
`;
}

export class PolygonalHRC {
  // runs: also keep each face direction's spread, so several faces read as one segment of the right length
  // stairGuards: take off side-face light hidden by the corner of a solid pixel 2 or 3 pixels before it in the row
  // readSize2: at the probe, evaluate each size-2 child at the point the probe reads it from
  // positionalGates: in place of a line's transmittance, the share of the light's slopes that passes the solid pixels
  //   intruding most into the line from below and above
  // sideCorners (with positionalGates): lines also keep their solid side pixel, and the odd merge's near cones are seen
  //   from the frustum's apex
  constructor(device, N, { channels = 3, runs = false, stairGuards = false, readSize2 = false, positionalGates = false, sideCorners = false } = {}) {
    const levels = Math.round(Math.log2(N));
    if (1 << levels !== N || levels < 3 || levels > 10) throw new Error('N must be a power of two, 8 .. 1024');
    if (channels !== 1 && channels !== 3) throw new Error('channels must be 1 or 3');
    if (!positionalGates) sideCorners = false;
    const o = { channels, runs, stairGuards, readSize2, positionalGates, sideCorners };
    Object.assign(this, { device, N, levels, ...o });
    this.profiler = null;   // a page may set a Profiler (site/_shared/profiler.ts)
    this.valueBytes = 16 * (channels + (positionalGates ? 1 : 0));
    const storage = GPUBufferUsage.STORAGE, copyDst = GPUBufferUsage.COPY_DST, copySrc = GPUBufferUsage.COPY_SRC;
    const buffer = (label, size, usage = storage) => device.createBuffer({ label, size: Math.max(size, 16), usage });
    this.extendThreads = []; this.mergeThreads = []; this.extensionBytes = [];
    let extensionFloats = 0;
    for (let n = 1; n < levels; n++) {
      const d = 1 << n;
      this.extendThreads[n] = (N / d) * rows(N, d) * (2 * d + 1);
      this.mergeThreads[n] = (N / d) * (N + d + 2) * d;
      this.extensionBytes[n] = (N / d) * rows(N, d) * cellFloats(d, o) * 4;
      extensionFloats += (N / d) * rows(N, d) * cellFloats(d, o);
    }
    const frustumCount = Math.max(...this.mergeThreads.slice(1));
    this.emissionBuffer = buffer('emission', N * N * channels * 4, storage | copyDst);
    this.transmittanceBuffer = buffer('transmittance', N * N * 4, storage | copyDst);
    this.extensionBuffer = buffer('extensions', extensionFloats * 4);
    this.frustumValueBuffers = [0, 1].map((p) => buffer(`frustum values ${p}`, frustumCount * this.valueBytes, storage | copyDst));
    this.fluenceBuffer = buffer('fluence', (N + 1) * (N + 1) * channels * 4, storage | copySrc | copyDst);
    this.bytes = { extensions: extensionFloats * 4, frustumValues: 2 * frustumCount * this.valueBytes };

    // pass parameters, one block per (rotation, view, level)
    this.paramData = new Int32Array(8 * (levels + 1) * (PARAM_STRIDE / 4));
    for (let r = 0; r < 4; r++) for (const view of [1, -1]) for (let n = 0; n <= levels; n++)
      this.paramData.set([n, r, view, r === 0 && view === 1 ? 1 : 0], this.paramOffset(r, view, n) / 4);
    this.paramBuffer = buffer('params', this.paramData.byteLength, GPUBufferUsage.UNIFORM | copyDst);
    device.queue.writeBuffer(this.paramBuffer, 0, this.paramData);

    this.module = device.createShaderModule({ code: shaderSource({ N, ...o }) });
    const readOnly = { type: 'read-only-storage' }, readWrite = { type: 'storage' };
    const kinds = [readOnly, readOnly, readWrite, readOnly, readWrite, readWrite, { type: 'uniform', hasDynamicOffset: true }];
    const layout = device.createBindGroupLayout({ entries: kinds.map((b, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: b })) });
    // bind group p reads children from buffers [p] and writes buffers [1 - p]
    this.bindGroups = [0, 1].map((p) => device.createBindGroup({ layout, entries: [this.emissionBuffer, this.transmittanceBuffer, this.extensionBuffer,
      this.frustumValueBuffers[p], this.frustumValueBuffers[1 - p], this.fluenceBuffer, this.paramBuffer]
      .map((b, binding) => ({ binding, resource: binding === 6 ? { buffer: b, size: 16 } : { buffer: b } })) }));
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint) => device.createComputePipeline({ layout: pipelineLayout, compute: { module: this.module, entryPoint } });
    this.pipelines = { extend: pipeline('extend'), mergeCascade: pipeline('mergeCascade'), gatherProbes: pipeline('gatherProbes') };
  }

  destroy() {
    for (const b of [this.emissionBuffer, this.transmittanceBuffer, this.extensionBuffer, ...this.frustumValueBuffers, this.fluenceBuffer, this.paramBuffer]) b.destroy();
  }
  paramOffset(rotation, view, level) { return ((rotation * 2 + (view > 0 ? 0 : 1)) * (this.levels + 1) + level) * PARAM_STRIDE; }
  async compileInfo() { return (await this.module.getCompilationInfo()).messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`); }
  // E: emission [(x * N + y) * channels + c], A: transmittance across one pixel (solid: A <= 0) [x * N + y]
  upload(E, A) { this.device.queue.writeBuffer(this.emissionBuffer, 0, E); this.device.queue.writeBuffer(this.transmittanceBuffer, 0, A); }
  dispatch(pass, pipeline, paramOffset, threads, bindGroup = 0) {
    if (threads <= 0) return;
    const groups = Math.ceil(threads / WORKGROUP), r = Math.ceil(groups / 65535);
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.bindGroups[bindGroup], [paramOffset]); pass.dispatchWorkgroups(Math.ceil(groups / r), r);
  }
  encodeExtensions(pass, r, view) {
    for (let n = 1; n < this.levels; n++) {
      this.profiler?.name(pass, `extend ${n}`);
      this.dispatch(pass, this.pipelines.extend, this.paramOffset(r, view, n), this.extendThreads[n]);
    }
  }
  // merges from size 2^from down to size 2, alternating between the buffer pairs from bind group p; returns the bind group
  // whose input holds size 2. The largest size reads buffers [0] but never uses them (its children have size N, all zero).
  encodeMerges(pass, r, view, from = this.levels - 1, p = 0) {
    for (let n = from; n >= 1; n--) {
      this.profiler?.name(pass, `merge ${n}`);
      this.dispatch(pass, this.pipelines.mergeCascade, this.paramOffset(r, view, n), this.mergeThreads[n], p);
      p = 1 - p;
    }
    return p;
  }
  encodeGather(pass, r, view, bindGroup) {
    this.profiler?.name(pass, 'gather');
    this.dispatch(pass, this.pipelines.gatherProbes, this.paramOffset(r, view, 0), (this.N + 1) * (this.N + 1), bindGroup);
  }
  // with a profiler (site/_shared/profiler.ts), each rotation and view is timed as one pass, or per level
  encodeRotation(enc, r, view) {
    const pass = this.profiler ? this.profiler.computePass(enc, 'rotation') : enc.beginComputePass();
    this.encodeExtensions(pass, r, view);
    this.encodeGather(pass, r, view, this.encodeMerges(pass, r, view));
    pass.end();
  }
  encodeAll(enc) { for (let r = 0; r < 4; r++) for (const view of [1, -1]) this.encodeRotation(enc, r, view); }
  // bytes each pass reads and writes, per rotation and view: every store it touches, counted once (a lower bound on
  // memory traffic). Level n is size 2^n.
  traffic() {
    const { N, levels, channels } = this, out = {};
    const scene = N * N * (channels + 1) * 4, fluence = (N + 1) * (N + 1) * channels * 4;
    const values = (n) => (n < levels ? this.mergeThreads[n] * this.valueBytes : 0);
    for (let n = 1; n < levels; n++) out[`extend ${n}`] = { read: n === 1 ? scene : this.extensionBytes[n - 1], write: this.extensionBytes[n] };
    // a merge of size d reads the extensions starting a size ahead or behind it: all columns of the store but one
    for (let n = levels - 1; n >= 1; n--) out[`merge ${n}`] = { read: this.extensionBytes[n] * (1 - (1 << n) / N) + values(n + 1), write: values(n) };
    out.gather = { read: values(1) + scene + fluence, write: fluence };
    out.rotation = Object.values(out).reduce((a, t) => ({ read: a.read + t.read, write: a.write + t.write }), { read: 0, write: 0 });
    return out;
  }
  async read(buf, bytes) {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, staging, 0, bytes); this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const out = staging.getMappedRange().slice(0); staging.unmap(); staging.destroy(); return out;
  }
  // GPU time of each stage alone (one rotation and view): reps dispatches in one submit, after one warm-up
  async profile(reps = 10, r = 0, view = 1) {
    const dev = this.device, out = {};
    const stage = async (name, encode) => {
      const run = (n) => { const enc = dev.createCommandEncoder(); const pass = enc.beginComputePass(); for (let k = 0; k < n; k++) encode(pass); pass.end(); dev.queue.submit([enc.finish()]); };
      run(1); await dev.queue.onSubmittedWorkDone(); const t0 = performance.now();
      run(reps); await dev.queue.onSubmittedWorkDone(); out[name] = +((performance.now() - t0) / reps).toFixed(3);
    };
    for (let n = 1; n < this.levels; n++) await stage(`extensions, size ${1 << n}`, (pass) => this.dispatch(pass, this.pipelines.extend, this.paramOffset(r, view, n), this.extendThreads[n]));
    let p = 0;
    for (let n = this.levels - 1; n >= 1; n--) { const bg = p; await stage(`merge, size ${1 << n}`, (pass) => this.dispatch(pass, this.pipelines.mergeCascade, this.paramOffset(r, view, n), this.mergeThreads[n], bg)); p = 1 - p; }
    await stage('gather', (pass) => this.encodeGather(pass, r, view, p));
    return out;
  }
  // corner fluence [(cx * (N + 1) + cy) * channels + c], all rotations and views
  async solve(E, A) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); enc.clearBuffer(this.fluenceBuffer);
    this.encodeAll(enc); this.device.queue.submit([enc.finish()]);
    return { F: new Float32Array(await this.read(this.fluenceBuffer, (this.N + 1) ** 2 * this.channels * 4)) };
  }
  // one frustum on its own: the frustums that can hold an injected value. Sizes 2 .. N / 2 are stored; apex columns 0, s, ...
  // with column + s < N; rows 1 - s .. N - 1; directions -s .. s - 1.
  injectable(level) {
    const s = 1 << level, N = this.N;
    return { cascades: [1, this.levels - 1], columns: Array.from({ length: N / s - 1 }, (_, k) => k * s), rows: [1 - s, N - 1], directions: [-s, s - 1] };
  }
  // no light anywhere except one frustum's value, set to value at both samples; rotation 0 and the view of its direction
  // merge from that size and gather. A: the scene's transmittance.
  async inject(A, { cascade: level, X, y, direction }, value = 1) {
    const N = this.N, s = 1 << level, view = direction >= 0 ? 1 : -1;
    this.upload(new Float32Array(N * N * this.channels), A);
    const clear = this.device.createCommandEncoder();
    for (const b of [...this.frustumValueBuffers, this.fluenceBuffer]) clear.clearBuffer(b);
    this.device.queue.submit([clear.finish()]);
    const firstApexRow = view > 0 ? 1 - s : -2, slot = view > 0 ? direction : -1 - direction, r = y - firstApexRow;
    if (X % s || X < 0 || X + s >= N || r < 0 || r >= N + s + 2) throw new Error(`no stored frustum at size ${s}, apex (${X}, ${y})`);
    // the merge of size s writes buffer 1 - p, p = (levels - 1 - level) mod 2 (see encodeMerges)
    const p = 1 - ((this.levels - 1 - level) & 1);
    const v = new Float32Array(this.valueBytes / 4);
    for (let c = 0; c < this.channels; c++) v.set([value, 0, value, 0], 4 * c);
    this.device.queue.writeBuffer(this.frustumValueBuffers[p], (((X / s) * (N + s + 2) + r) * s + slot) * this.valueBytes, v);
    const enc = this.device.createCommandEncoder(); const pass = enc.beginComputePass();
    this.encodeExtensions(pass, 0, view);
    this.encodeGather(pass, 0, view, this.encodeMerges(pass, 0, view, level - 1, p));
    pass.end(); this.device.queue.submit([enc.finish()]);
    return new Float32Array(await this.read(this.fluenceBuffer, (N + 1) ** 2 * this.channels * 4));
  }
}
