// Moment HRC on WebGPU.
//
// ray (m, X, y, j): the digital line of 2^m columns after column X, from row y, slope j / 2^m. It stores its transmittance T
//   and mu_k = sum of emission * transmittance before * (t / 2^m)^k, k = 0, 1, 2.
// node (l, X, y, c): the cone between lines c - n and c + 1 - n of level l from (X, y), n = 2^l. Its near part is those two
//   lines over the first n columns, far = the rest of it.
//   odd X / n: far = the two level-(l + 1) children rooted where its lines end, at X + n.
//   even X / n: far = the second halves of lines A, B, C of level l + 1, plus
//     mix: half the children at X and half the children rooted at the ends of A and C, at X + 2n;
//     near: the children at X, minus line B's strip beyond X + 2n.
// carrier: a node's far part at two distances along its centre line, read by interpolating 1/F.
// reader (d, eta): columns and rows back from a line's start.

const WORKGROUP = 64;
const PARAM_STRIDE = 256;

function shaderSource({ N, LT, f16, fam, grammar }) {
  return /* wgsl */ `
const N: i32 = ${N};
const LT: i32 = ${LT};
const MIX: bool = ${grammar === 'mix'};   // else 'near'
const EMPTY: f32 = 1e-14;                    // a record with mu0 below this has no light
const SAMPLE_RATIO_MIN: f32 = 1e-12;         // carriers interpolate 1/F only when both samples are above this fraction of the larger

struct Params { level: i32, frustum: i32, subtractDiagonals: i32, unused: i32 }

@group(0) @binding(0) var<storage, read> emission: array<f32>;
@group(0) @binding(1) var<storage, read> cellTransmittance: array<f32>;
@group(0) @binding(2) var<storage, read_write> rays: array<${f16 ? 'vec2u' : 'vec4f'}>;   // (mu0, mu1, mu2, T)
@group(0) @binding(3) var<storage, read_write> farCarriers: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> fluence: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> lineRows: array<vec2i>;         // (row at t, row at t + 1) of every line
@group(0) @binding(7) var<storage, read> sharedPrefixes: array<vec2i>;   // (count, 0) then (level, index) per piece
@group(0) @binding(8) var<storage, read_write> level1Far: array<vec2f>;  // pass 3

fn pow2(m: i32) -> i32 { return 1 << u32(m); }
// linear thread index; large passes use a 2-D grid of workgroups, at most 65535 per dimension
fn threadIndex(id: vec3u, groups: vec3u) -> i32 { return i32(id.x + id.y * groups.x * ${WORKGROUP}u); }
fn xSlots(m: i32) -> i32 { return ((N - 2) >> u32(m)) + 1; }   // lattice positions X = 0, 2^m, ... before the last column

// cell (x, y) of the current frustum's frame -> index into the scene and the fluence
fn sceneIndex(x: i32, y: i32) -> i32 {
  switch (params.frustum) {
    case 1: { return (N - 1 - y) * N + x; }
    case 2: { return (N - 1 - x) * N + (N - 1 - y); }
    case 3: { return y * N + (N - 1 - x); }
    default: { return x * N + y; }
  }
}

// the first-half index of line (m, j), rounding odd j away from or toward zero.
// lbal: away at even levels when bit 2 of |j| is set, at odd levels when bit 1 is clear.
// lsym: away at even levels, toward at odd levels.
fn split(m: i32, j: i32) -> i32 {
  if ((j & 1) == 0) { return j >> 1u; }
  let aj = abs(j);
  ${fam === 'lsym'
    ? 'let away = (m & 1) == 0;'
    : 'let away = select(((aj >> 1u) & 1) == 0, ((aj >> 2u) & 1) == 1, (m & 1) == 0);'}
  let a = (aj >> 1u) + select(0, 1, away);
  return select(-a, a, j > 0);
}

// cells are squares: a line of slope s crosses sqrt(1 + s^2) of a cell per column
fn secant(m: i32, j: i32) -> f32 { let s = f32(j) / f32(pow2(m)); return sqrt(1.0 + s * s); }
// a transmittance re-expressed for the parent line's path length per column
fn rescale(T: f32, mFrom: i32, jFrom: i32, mTo: i32, jTo: i32) -> f32 {
  if (T <= 0.0) { return T; }
  return pow(T, secant(mTo, jTo) / secant(mFrom, jFrom));
}

// rays: [level m][j + 2^m][X / 2^m][y], the X count padded to 2^(LT - m), so level m holds 2^(LT - m) (2^(m+1) + 1) N records
fn raySlot(m: i32, X: i32, y: i32, j: i32) -> u32 {
  let levelStart = u32(N) * (((u32(m) + 1u) << u32(LT + 1)) - (1u << u32(LT - m + 1)));
  return levelStart + u32(((((j + pow2(m)) << u32(LT - m)) + (X >> u32(m))) * N) + y);
}
// carriers: levels 2 .. LT - 1, [X / 2^l][c][y], 2^(LT + 1) N per level
fn carrierSlot(l: i32, X: i32, y: i32, c: i32) -> u32 {
  return u32((l - 2) << u32(LT + 1)) * u32(N) + u32(((((X >> u32(l)) << u32(l + 1)) + c) * N) + y);
}
// line rows: [level m][j + 2^m][t - 1]
fn rowSlot(m: i32, j: i32, t: i32) -> u32 {
  let L = u32(pow2(m));
  return (2u * (L * L - 1u)) / 3u + L - 1u + u32(((j + pow2(m)) << u32(m)) + t - 1);
}
// shared prefixes: [level m >= 1][j + 2^m][count, pieces...] with LT + 1 entries per line pair
fn prefixSlot(m: i32, j: i32) -> u32 { return u32((pow2(m + 1) - 4 + j + pow2(m)) * (LT + 1)); }
// level-1 far parts: [reader pair][X / 2][c][y + 2], rows -2 .. N + 1
fn level1Nodes() -> i32 { return xSlots(1) * 4 * (N + 4); }
fn level1Slot(X: i32, y: i32, c: i32, pair: i32) -> u32 { return u32(pair * level1Nodes() + (((X >> 1u) * 4 + c) * (N + 4)) + y + 2); }

${f16 ? `fn unpackRay(p: vec2u) -> vec4f { return vec4f(unpack2x16float(p.x), unpack2x16float(p.y)); }
fn packRay(r: vec4f) -> vec2u { return vec2u(pack2x16float(r.xy), pack2x16float(r.zw)); }`
      : `fn unpackRay(p: vec4f) -> vec4f { return p; }
fn packRay(r: vec4f) -> vec4f { return r; }`}
fn loadRay(m: i32, X: i32, y: i32, j: i32) -> vec4f {
  if (X >= N - 1 || y < 0 || y >= N) { return vec4f(0.0, 0.0, 0.0, 1.0); }   // outside the domain: empty
  return unpackRay(rays[raySlot(m, X, y, j)]);
}

// pass 1: extension
@compute @workgroup_size(${WORKGROUP})
fn extend(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  // threads in (X, j, y) order, so neighbouring lines share their halves in cache
  let m = params.level; let L = pow2(m); let lines = 2 * L + 1;
  let i = threadIndex(id, groups);
  if (i >= xSlots(m) * lines * N) { return; }
  let y = i % N; let j = (i / N) % lines - L; let X = (i / (N * lines)) * L;
  var ray = vec4f(0.0, 0.0, 0.0, 1.0);
  if (m == 0) {
    let x = X + 1; let r = y + j;
    if (x < N && r >= 0 && r < N) {
      let cell = sceneIndex(x, r); let a = cellTransmittance[cell]; var e = emission[cell];
      var T = 0.0;
      if (a > 0.0) { T = pow(a, secant(0, j)); }   // a is the transmittance across one cell
      if (e > 0.0 && T < 1.0) { e *= select(0.0, (1.0 - T) / -log(T), T > 0.0); }   // the cell's own emission, attenuated inside it
      ray = vec4f(e, e, e, T);
    }
  } else {
    let a = split(m, j); let b = j - a; let h = pow2(m - 1);
    let first = loadRay(m - 1, X, y, a); let second = loadRay(m - 1, X + h, y + a, b);
    let Ta = rescale(first.w, m - 1, a, m, j); let Tb = rescale(second.w, m - 1, b, m, j);
    ray = vec4f(first.x + Ta * second.x,
                0.5 * (first.y + Ta * (second.x + second.y)),
                0.25 * (first.z + Ta * (second.x + 2.0 * second.y + second.z)),
                Ta * Tb);
  }
  rays[raySlot(m, X, y, j)] = packRay(ray);
}

// a record read as a point at its centroid: mu0 / D * (1 + var (1 + s^2) / D^2), D the reader's distance to the centroid
fn atom(ray: vec4f, m: i32, j: i32, d: vec2f, eta: vec2f, on: vec2<bool>) -> vec2f {
  if (!any(on)) { return vec2f(0.0); }
  let L = pow2(m); let Lf = f32(L);
  let mu0 = max(ray.x, EMPTY); let q = ray.y / mu0; let tb = Lf * q;
  let t = clamp(i32(floor(tb)), 1, L); let f = tb - f32(t);
  let rows = vec2f(lineRows[rowSlot(m, j, t)]);
  let rb = rows.x * (1.0 - f) + rows.y * f;
  let s = f32(j) / Lf;
  let variance = max(Lf * Lf * (ray.z / mu0 - q * q), 0.0) * (1.0 + s * s);
  let dx = d + tb; let dy = eta + rb; let r = inverseSqrt(dx * dx + dy * dy);
  return select(vec2f(0.0), ray.x * r * (1.0 + variance * r * r), on & vec2<bool>(ray.x > EMPTY));
}

// line (m, X, y, j); readers closer than its length read its two halves. No reader is closer than half a record, so one
// split is enough.
fn readRay(m: i32, X: i32, y: i32, j: i32, d: vec2f, eta: vec2f) -> vec2f {
  if (X >= N - 1) { return vec2f(0.0); }
  let close = vec2<bool>(m > 0) & (d + 1.0 < vec2f(f32(pow2(m))));
  var v = vec2f(0.0);
  if (!all(close)) { v = atom(loadRay(m, X, y, j), m, j, d, eta, !close); }
  if (any(close)) {
    let a = split(m, j); let h = pow2(m - 1); let first = loadRay(m - 1, X, y, a);
    v += atom(first, m - 1, a, d, eta, close);
    if (X + h < N - 1) {
      v += first.w * atom(loadRay(m - 1, X + h, y + a, j - a), m - 1, j - a, d + f32(h), eta + f32(a), close);
    }
  }
  return v;
}

// the prefix shared by lines j and j + 1 of level m, read as its pieces (none of which splits), each gating the next
fn readSharedPrefix(m: i32, X0: i32, y0: i32, j: i32, d0: vec2f, eta0: vec2f) -> vec2f {
  if (m == 0 || X0 >= N - 1) { return vec2f(0.0); }
  var X = X0; var y = y0; var d = d0; var eta = eta0; var v = vec2f(0.0); var T = 1.0;
  let base = prefixSlot(m, j);
  let pieces = u32(sharedPrefixes[base].x);
  for (var i = 1u; i <= pieces; i += 1u) {
    let piece = sharedPrefixes[base + i]; let k = piece.x; let jp = piece.y;
    let ray = loadRay(k, X, y, jp);
    v += T * atom(ray, k, jp, d, eta, vec2<bool>(true));
    T *= ray.w;
    X += pow2(k); y += jp; d += f32(pow2(k)); eta += f32(jp);
    if (X >= N - 1) { break; }
  }
  return v;
}

// the strip of level-m0 lines of slope j0 from (X, y) to the domain end, for one reader: the largest aligned records
// with 2^k <= d + 1
fn readStrip(m0: i32, X0: i32, y0: i32, j0: i32, d0: f32, eta0: f32) -> f32 {
  var X = X0; var y = y0; var d = d0; var eta = eta0; var v = 0.0; var T = 1.0;
  loop {
    if (X >= N - 1) { break; }
    let k = max(m0, min(min(i32(countTrailingZeros(u32(X))), i32(firstLeadingBit(u32(d + 1.0)))), LT)); let j = j0 << u32(k - m0);
    let ray = loadRay(k, X, y, j);
    v += T * atom(ray, k, j, vec2f(d), vec2f(eta), vec2<bool>(true)).x;
    T *= ray.w;
    X += pow2(k); y += j; d += f32(pow2(k)); eta += f32(j);
  }
  return v;
}

fn slope(l: i32, c: i32) -> f32 { let n = f32(pow2(l)); return (f32(c) + 0.5 - n) / n; }
// the carrier sample distances of level l: Chebyshev nodes over [0, 2n] (mix: readers are at most 2n - 2 back) or [0, n] (near)
fn sampleDistances(l: i32) -> vec2f { let h = select(0.5, 1.0, MIX) * f32(pow2(l)); return h * vec2f(1.0 + 0.70710678118654752, 1.0 - 0.70710678118654752); }

// a node's far part from its carrier: readers projected onto the centre line, 1/F interpolated linearly in distance
fn readCarrier(l: i32, X: i32, y: i32, c: i32, d: vec2f, eta: vec2f) -> vec2f {
  if (X >= N - 1 || y < 0 || y >= N || l >= LT) { return vec2f(0.0); }
  let F = farCarriers[carrierSlot(l, X, y, c)];
  let at = sampleDistances(l); let s = slope(l, c); let x = (d + s * eta) / (1.0 + s * s);
  let w0 = (x - at.y) / (at.x - at.y); let w1 = (x - at.x) / (at.y - at.x);
  if (min(F.x, F.y) > SAMPLE_RATIO_MIN * max(max(F.x, F.y), 1e-30)) { return 1.0 / (w0 / F.x + w1 / F.y); }
  return w0 * F.x + w1 * F.y;
}
// a child's far part: from its carrier, or at level 1 from pass 3. Level-1 nodes are read only by receivers, at distance
// 0, 1 or 2 (2 in mix only), with the same reader in both lanes; pair 0 holds distances 0 and 1, pair 1 distance 2.
fn readChildFar(l: i32, X: i32, y: i32, c: i32, d: vec2f, eta: vec2f) -> vec2f {
  if (l == 1) {
    if (X >= N - 1) { return vec2f(0.0); }
    let k = i32(round(d.x));
    let F = level1Far[level1Slot(X, y, c, k >> 1u)];
    return vec2f(select(F.x, F.y, k == 1));
  }
  return readCarrier(l, X, y, c, d, eta);
}

// the node's own two lines over its first 2^l columns, their shared prefix counted once
fn readNear(l: i32, X: i32, y: i32, c: i32, d: vec2f, eta: vec2f) -> vec2f {
  let j0 = c - pow2(l);
  return readRay(l, X, y, j0, d, eta) + readRay(l, X, y, j0 + 1, d, eta) - readSharedPrefix(l, X, y, j0, d, eta);
}
fn readChild(l: i32, X: i32, y: i32, c: i32, d: vec2f, eta: vec2f) -> vec2f {
  if (X >= N - 1) { return vec2f(0.0); }
  return readNear(l, X, y, c, d, eta) + readChildFar(l, X, y, c, d, eta);
}

// even node: the second half of line j of level l + 1, gated by the first half; and its part of the shared prefix of j, j + 1
fn readSecondHalf(l: i32, X: i32, y: i32, j: i32, d: vec2f, eta: vec2f) -> vec2f {
  let n = pow2(l); let a = split(l + 1, j); let T = loadRay(l, X, y, a).w;
  return T * readRay(l, X + n, y + a, j - a, d + f32(n), eta + f32(a));
}
fn readSecondHalfPrefix(l: i32, X: i32, y: i32, j: i32, d: vec2f, eta: vec2f) -> vec2f {
  let n = pow2(l); let a = split(l + 1, j);
  if (a != split(l + 1, j + 1)) { return vec2f(0.0); }
  let T = loadRay(l, X, y, a).w;
  return T * readSharedPrefix(l, X + n, y + a, j - a, d + f32(n), eta + f32(a));
}

// a node's far part, exactly, from records and its children's far parts
fn exactFar(l: i32, X: i32, y: i32, c: i32, d: vec2f, eta: vec2f) -> vec2f {
  let n = pow2(l);
  if (X + n >= N - 1 || l >= LT) { return vec2f(0.0); }
  var v = vec2f(0.0);
  if (((X >> u32(l)) & 1) == 0) {
    // even: the second halves of lines A, B, C of level l + 1 (shared by both branches)
    let jA = 2 * (c - n); let jB = jA + 1; let jC = jA + 2;
    v += readSecondHalf(l, X, y, jA, d, eta) + readSecondHalf(l, X, y, jB, d, eta) + readSecondHalf(l, X, y, jC, d, eta);
    v -= readSecondHalfPrefix(l, X, y, jA, d, eta) + readSecondHalfPrefix(l, X, y, jB, d, eta);
    // near branch: the children at the same X
    v += select(1.0, 0.5, MIX) * (readChildFar(l + 1, X, y, 2 * c, d, eta) + readChildFar(l + 1, X, y, 2 * c + 1, d, eta));
    if (!MIX && X + 2 * n < N - 1) {   // near grammar: both children contain line B's continuation beyond X + 2n
      let T = loadRay(l + 1, X, y, jB).w; let dB = d + f32(2 * n); let etaB = eta + f32(jB);
      v -= T * vec2f(readStrip(l + 1, X + 2 * n, y + jB, jB, dB.x, etaB.x), readStrip(l + 1, X + 2 * n, y + jB, jB, dB.y, etaB.y));
    }
    // far branch (mix): the children re-anchored at the ends of A and C
    if (MIX && X + 2 * n < N - 1) {
      let TA = 0.5 * loadRay(l + 1, X, y, jA).w; let TC = 0.5 * loadRay(l + 1, X, y, jC).w;
      v += TA * readChild(l + 1, X + 2 * n, y + jA, 2 * c, d + f32(2 * n), eta + f32(jA));
      v += TC * readChild(l + 1, X + 2 * n, y + jC, 2 * c + 1, d + f32(2 * n), eta + f32(jC));
    }
  } else {
    // odd: the children start where the node's own lines end
    let j0 = c - n; let j1 = j0 + 1;
    let T0 = loadRay(l, X, y, j0).w; let T1 = loadRay(l, X, y, j1).w;
    v += T0 * readChild(l + 1, X + n, y + j0, 2 * c, d + f32(n), eta + f32(j0));
    v += T1 * readChild(l + 1, X + n, y + j1, 2 * c + 1, d + f32(n), eta + f32(j1));
  }
  return v;
}

// pass 2: merge-down, even and odd nodes as two kernels
fn mergeNode(i: i32, parity: i32) {
  let l = params.level; let cones = pow2(l + 1);
  if (i >= (xSlots(l) + 1 - parity) / 2 * cones * N) { return; }
  let y = i % N; let c = (i / N) % cones; let X = (2 * (i / (N * cones)) + parity) * pow2(l);
  let d = sampleDistances(l);
  farCarriers[carrierSlot(l, X, y, c)] = exactFar(l, X, y, c, d, slope(l, c) * d);
}
@compute @workgroup_size(${WORKGROUP})
fn mergeEven(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) { mergeNode(threadIndex(id, groups), 0); }
@compute @workgroup_size(${WORKGROUP})
fn mergeOdd(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) { mergeNode(threadIndex(id, groups), 1); }

// pass 3: level-1 nodes. Node (1, X, y, c) is read by receivers (X - k, y - k e) at (k, k e), k = 0, 1, 2, e = -1, 0, 0, 1
// for c = 0..3. Pair 0 evaluates k = 0 and 1, pair 1 k = 2. Rows -2 .. N + 1 are stored, since edge receivers read them.
@compute @workgroup_size(${WORKGROUP})
fn level1(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= 2 * level1Nodes()) { return; }
  let pair = i / level1Nodes(); let node = i % level1Nodes();
  let y = node % (N + 4) - 2; let c = (node / (N + 4)) & 3; let X = 2 * (node / (4 * (N + 4)));
  let e = select(select(0.0, 1.0, c == 3), -1.0, c == 0);
  let d = select(vec2f(0.0, 1.0), vec2f(2.0), pair == 1);
  level1Far[i] = exactFar(1, X, y, c, d, d * e);
}

// pass 4: receivers, with the receiver as both readers
fn addFluence(x: i32, y: i32, v: f32, overwrite: bool) {
  let cell = sceneIndex(x, y); let a = cellTransmittance[cell];
  let w = select(0.0, v * pow(a, 0.561), a > 0.0);   // the receiver sits at its cell's centre, so every path first crosses half the cell
  if (overwrite) { fluence[cell] = w; } else { fluence[cell] += w; }
}
// the seam (the line both cones contain) and, in the +-x frames, the diagonals the +-y frames also contain
@compute @workgroup_size(${WORKGROUP})
fn receiveStrips(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N;
  var v = -readStrip(0, x, y, 0, 0.0, 0.0);
  if (params.subtractDiagonals == 1) { v -= readStrip(0, x, y, 1, 0.0, 0.0) + readStrip(0, x, y, -1, 0.0, 0.0); }
  addFluence(x, y, v, params.frustum == 0);
}
@compute @workgroup_size(${WORKGROUP})
fn receiveCones(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N;
  var v = 0.0;
  for (var c = 0; c < 2; c += 1) { v += (readNear(0, x, y, c, vec2f(0.0), vec2f(0.0)) + exactFar(0, x, y, c, vec2f(0.0), vec2f(0.0))).x; }
  addFluence(x, y, v, false);
}
`;
}

// tables, built once on the CPU
function splitRule(fam) {
  return (m, j) => {
    if ((j & 1) === 0) return j >> 1;
    const aj = Math.abs(j);
    const away = fam === 'lsym' ? (m & 1) === 0 : (m & 1) === 0 ? ((aj >> 2) & 1) === 1 : ((aj >> 1) & 1) === 0;
    const a = (aj >> 1) + (away ? 1 : 0);
    return j > 0 ? a : -a;
  };
}
// rows of line (m, j) at columns 1 .. 2^m
function lineRowsOf(split, m, j) {
  if (m === 0) return [j];
  const a = split(m, j);
  return [...lineRowsOf(split, m - 1, a), ...lineRowsOf(split, m - 1, j - a).map((r) => r + a)];
}
// every line's (row at t, row at t + 1), t = 1 .. 2^m; the last t repeats its row
function buildLineRows(split, LT) {
  const out = [];
  for (let m = 0; m <= LT; m++) {
    const L = 1 << m;
    for (let j = -L; j <= L; j++) { const r = lineRowsOf(split, m, j); r.forEach((row, i) => out.push(row, r[Math.min(i + 1, L - 1)])); }
  }
  return new Int32Array(out);
}
// pieces (level, index) of the prefix shared by lines j, j + 1 of level m, in order along the line,
// as (count, 0) followed by LT entries per line pair
function buildSharedPrefixes(split, LT) {
  const out = [];
  for (let m = 1; m <= LT; m++) {
    const L = 1 << m;
    for (let j = -L; j < L; j++) {
      const pieces = [];
      for (let k = m, jj = j; k > 0; k--) {
        const a = split(k, jj);
        if (a === split(k, jj + 1)) { pieces.push(k - 1, a); jj -= a; } else jj = a;
      }
      const entry = new Array(2 * (LT + 1)).fill(0); entry[0] = pieces.length / 2; entry.splice(2, pieces.length, ...pieces);
      out.push(...entry.slice(0, 2 * (LT + 1)));
    }
  }
  return new Int32Array(out);
}

export class MomentHRC {
  constructor(device, N, { f16 = false, fam = 'lbal', grammar = 'mix' } = {}) {
    const LT = Math.max(1, Math.ceil(Math.log2(N)));
    Object.assign(this, { device, N, LT });
    const xSlots = (m) => ((N - 2) >> m) + 1;
    const storage = GPUBufferUsage.STORAGE, copyDst = GPUBufferUsage.COPY_DST;
    const buffer = (size, usage = storage) => device.createBuffer({ size: Math.max(size, 16), usage });
    const filled = (data) => { const b = buffer(data.byteLength, storage | copyDst); device.queue.writeBuffer(b, 0, data); return b; };

    // sizes (match the shader's slot functions)
    this.extendThreads = Array.from({ length: LT + 1 }, (_, m) => xSlots(m) * (2 * (1 << m) + 1) * N);
    const rayCount = N * ((LT + 2) * 2 ** (LT + 1) - 1);   // where level LT + 1 would start
    const carrierCount = Math.max(LT - 2, 0) * 2 ** (LT + 1) * N;
    this.level1Threads = (grammar === 'mix' ? 2 : 1) * xSlots(1) * 4 * (N + 4);   // reader pairs per level-1 node

    this.bufE = buffer(N * N * 4, storage | copyDst);
    this.bufA = buffer(N * N * 4, storage | copyDst);
    this.bufRays = buffer(rayCount * (f16 ? 8 : 16));
    this.bufCar = buffer(carrierCount * 8);
    this.bufLevel1 = buffer(2 * xSlots(1) * 4 * (N + 4) * 8);
    this.bufOut = buffer(N * N * 4, storage | GPUBufferUsage.COPY_SRC);
    this.bufRead = buffer(N * N * 4, GPUBufferUsage.MAP_READ | copyDst);
    const split = splitRule(fam);
    this.bufLineRows = filled(buildLineRows(split, LT)); this.bufSharedPrefixes = filled(buildSharedPrefixes(split, LT));

    // parameter slots (one uniform buffer, dynamic offsets): per frustum, one per level and two for the receivers
    // (with / without the diagonals); slot (frustum, level, subtractDiagonals)
    const slots = [];
    this.slot = (frustum, level, diagonals = 0) => {
      const key = `${frustum}:${level}:${diagonals}`;
      let s = slots.findIndex((p) => p.key === key);
      if (s < 0) { s = slots.length; slots.push({ key, values: [level, frustum, diagonals, 0] }); }
      return s * PARAM_STRIDE;
    };
    for (let k = 0; k < 4; k++) { for (let l = 0; l <= LT; l++) this.slot(k, l); this.slot(k, 0, 1); }
    const paramData = new Int32Array((slots.length * PARAM_STRIDE) / 4);
    slots.forEach((p, s) => paramData.set(p.values, (s * PARAM_STRIDE) / 4));
    this.bufParams = buffer(paramData.byteLength, GPUBufferUsage.UNIFORM | copyDst); device.queue.writeBuffer(this.bufParams, 0, paramData);

    this.module = device.createShaderModule({ code: shaderSource({ N, LT, f16, fam, grammar }) });
    const ro = { type: 'read-only-storage' }, rw = { type: 'storage' };
    const kinds = [ro, ro, rw, rw, rw, { type: 'uniform', hasDynamicOffset: true }, ro, ro, rw];
    const layout = device.createBindGroupLayout({ entries: kinds.map((b, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: b })) });
    // every buffer is a field: the page destroys them all when it rebuilds the cascade
    const buffers = [this.bufE, this.bufA, this.bufRays, this.bufCar, this.bufOut, this.bufParams, this.bufLineRows, this.bufSharedPrefixes, this.bufLevel1];
    this.bindGroup = device.createBindGroup({ layout, entries: buffers.map((b, binding) => ({ binding, resource: binding === 5 ? { buffer: b, size: 16 } : { buffer: b } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint) => device.createComputePipeline({ layout: pipelineLayout, compute: { module: this.module, entryPoint } });
    this.pipelines = { extend: pipeline('extend'), merge: [pipeline('mergeEven'), pipeline('mergeOdd')], level1: pipeline('level1'),
                       receivers: [pipeline('receiveStrips'), pipeline('receiveCones')] };
  }

  async compileInfo() { return (await this.module.getCompilationInfo()).messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`); }

  upload(E, A) { this.device.queue.writeBuffer(this.bufE, 0, E); this.device.queue.writeBuffer(this.bufA, 0, A); }

  dispatch(pass, pipeline, slot, threads) {
    if (threads <= 0) return;
    const groups = Math.ceil(threads / WORKGROUP), rows = Math.ceil(groups / 65535);
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.bindGroup, [slot]); pass.dispatchWorkgroups(Math.ceil(groups / rows), rows);
  }
  encodeExtend(pass, k) {
    for (let m = 0; m <= this.LT; m++) this.dispatch(pass, this.pipelines.extend, this.slot(k, m), this.extendThreads[m]);
  }
  encodeMerge(pass, k, l) {
    const xSlots = ((this.N - 2) >> l) + 1, perX = (1 << (l + 1)) * this.N;
    [0, 1].forEach((parity) => this.dispatch(pass, this.pipelines.merge[parity], this.slot(k, l), ((xSlots + 1 - parity) >> 1) * perX));
  }
  encodeLevel1(pass, k) {
    this.dispatch(pass, this.pipelines.level1, this.slot(k, 1), this.level1Threads);
  }
  encodeReceivers(pass, k, diagonals) {
    for (const p of this.pipelines.receivers) this.dispatch(pass, p, this.slot(k, 0, diagonals), this.N * this.N);
  }
  // frustum k: extension, merge-down, level 1, receivers (accumulating into the fluence in scene coordinates)
  encode(enc, k = 0, diagonals = k % 2 === 0 ? 1 : 0) {
    const pass = enc.beginComputePass();
    this.encodeExtend(pass, k);
    for (let l = this.LT - 1; l > 1; l--) this.encodeMerge(pass, k, l);
    this.encodeLevel1(pass, k);
    this.encodeReceivers(pass, k, diagonals);
    pass.end();
  }
  // all four frustums into the fluence buffer (no readback)
  encodeAll(enc) { for (let k = 0; k < 4; k++) this.encode(enc, k); }

  async readOut() {
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(this.bufOut, 0, this.bufRead, 0, this.N * this.N * 4);
    this.device.queue.submit([enc.finish()]);
    await this.bufRead.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.bufRead.getMappedRange().slice(0)); this.bufRead.unmap(); return out;
  }
  // frustum 0 alone, without diagonal subtraction
  async run(E, A) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); this.encode(enc, 0, 0); this.device.queue.submit([enc.finish()]);
    return this.readOut();
  }
  // all four frustums
  async runAll(E, A) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); this.encodeAll(enc); this.device.queue.submit([enc.finish()]);
    return this.readOut();
  }

  // GPU time of each stage alone (frustum k), averaged over reps submits
  async profile(reps = 10, k = 0) {
    const dev = this.device, out = {};
    const stage = async (name, encode) => {
      await dev.queue.onSubmittedWorkDone(); const t0 = performance.now();
      for (let r = 0; r < reps; r++) { const enc = dev.createCommandEncoder(); const pass = enc.beginComputePass(); encode(pass); pass.end(); dev.queue.submit([enc.finish()]); }
      await dev.queue.onSubmittedWorkDone(); out[name] = (performance.now() - t0) / reps;
    };
    await stage('extension', (pass) => this.encodeExtend(pass, k));
    for (let l = this.LT - 1; l > 1; l--) await stage(`merge L${l}`, (pass) => this.encodeMerge(pass, k, l));
    await stage('level 1', (pass) => this.encodeLevel1(pass, k));
    await stage('receivers', (pass) => this.encodeReceivers(pass, k, k % 2 === 0 ? 1 : 0));
    return out;
  }
}
