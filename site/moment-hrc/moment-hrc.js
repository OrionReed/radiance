// Moment HRC (WebGPU).
//
// quadrant: the scene is solved four times, rotated by 90 degrees; in each, lines go to +x with slopes in [-1, 1].
// ray (cascade n, first column X, start row y, rise r): the digital line of 2^n columns after column X from row y, rising r.
//   It holds its transmittance T and light moments mu_k = sum of emission x transmittance before x (t / 2^n)^k, k = 0, 1, 2,
//   built by extension from its two halves; the split rule says how an odd rise splits.
// cone (cascade n, apex (X, y), direction d): the lines between the rays of rise d and d + 1. Its own interval is its two
//   rays over the first 2^n columns; the rest ("beyond") comes from cascade n + 1:
//   odd X / 2^n: the two child cones re-rooted where its rays end (forward);
//   even X / 2^n: the second halves of rays A, B, C of cascade n + 1, plus either 1/2 the same-apex children and 1/2 those
//   re-rooted at the ends of A and C (merge pattern sameApexAndForward), or the same-apex children minus ray B's
//   continuation (sameApex).
// samples: a cone's beyond-value at two distances along its axis, read by interpolating 1 / value.
// offset (back, across): a point's position, in columns behind a ray's start and rows across.
// probes: cell centres.

const WORKGROUP = 64;
const PARAM_STRIDE = 256;

function shaderSource({ N, lastCascade, halfFloatRays, splitRule, mergePattern, averagedRounding, onePass, opaqueLightKernel, pointCells, rayCount }) {
  return /* wgsl */ `
const N: i32 = ${N};
const LAST_CASCADE: i32 = ${lastCascade};
const SAME_APEX_AND_FORWARD: bool = ${mergePattern === 'sameApexAndForward'};   // else 'sameApex'
const NO_LIGHT: f32 = 1e-14;                 // a ray with mu0 below this has no light
const SAMPLE_RATIO_MIN: f32 = 1e-12;         // samples interpolate 1/F only when both are above this fraction of the larger

struct PassParams { cascade: i32, quadrant: i32, subtractSharedDiagonals: i32, unused: i32 }

@group(0) @binding(0) var<storage, read> emission: array<f32>;
@group(0) @binding(1) var<storage, read> transmittance: array<f32>;
@group(0) @binding(2) var<storage, read_write> rays: array<${halfFloatRays ? 'vec2u' : 'vec4f'}>;   // (mu0, mu1, mu2, T)
@group(0) @binding(3) var<storage, read_write> beyondSamples: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> fluence: array<f32>;
@group(0) @binding(5) var<uniform> params: PassParams;
@group(0) @binding(6) var<storage, read> rayPathRows: array<vec2i>;     // (row at t, row at t + 1) of every ray
@group(0) @binding(7) var<storage, read> sharedPrefixes: array<vec2i>;  // (count, 0) then (cascade, rise) per piece
@group(0) @binding(8) var<storage, read_write> cascade1Beyond: array<vec2f>;  // pass 3
${onePass ? `const SHARED_PREFIX_OFFSET: u32 = ${rayCount}u;   // one-pass: the expected shared prefixes of rays r, r + 1 (mu0, mu1, mu2, -) sit after the rays\n` : ''}
fn pow2(n: i32) -> i32 { return 1 << u32(n); }
// linear thread index; large passes use a 2-D grid of workgroups, at most 65535 per dimension
fn threadIndex(id: vec3u, groups: vec3u) -> i32 { return i32(id.x + id.y * groups.x * ${WORKGROUP}u); }
fn apexColumns(cascade: i32) -> i32 { return ((N - 2) >> u32(cascade)) + 1; }   // lattice positions X = 0, 2^n, ... before the last column

// cell (x, y) of the current quadrant -> index into the scene and the fluence
fn scenePixelIndex(x: i32, y: i32) -> i32 {
  switch (params.quadrant) {
    case 1: { return (N - 1 - y) * N + x; }
    case 2: { return (N - 1 - x) * N + (N - 1 - y); }
    case 3: { return y * N + (N - 1 - x); }
    default: { return x * N + y; }
  }
}

// the first-half rise of ray (cascade, rise), rounding odd rises away from or toward zero.
// balanced: away at even cascades when bit 2 of |rise| is set, at odd cascades when bit 1 is clear.
// alternating: away at even cascades, toward at odd cascades.
fn splitRise(cascade: i32, rise: i32) -> i32 {
  if ((rise & 1) == 0) { return rise >> 1u; }
  let absRise = abs(rise);
  ${splitRule === 'alternating'
    ? 'let away = (cascade & 1) == 0;'
    : 'let away = select(((absRise >> 1u) & 1) == 0, ((absRise >> 2u) & 1) == 1, (cascade & 1) == 0);'}
  let a = (absRise >> 1u) + select(0, 1, away);
  return select(-a, a, rise > 0);
}

// cells are squares: a line of slope s crosses sqrt(1 + s^2) of a cell per column
${pointCells ? `fn pathLengthPerColumn(cascade: i32, rise: i32) -> f32 { return 1.0; }   // point model: transmittance per cell crossed
fn transmittanceForPath(T: f32, cascadeFrom: i32, riseFrom: i32, cascadeTo: i32, riseTo: i32) -> f32 { return T; }`
       : `fn pathLengthPerColumn(cascade: i32, rise: i32) -> f32 { let s = f32(rise) / f32(pow2(cascade)); return sqrt(1.0 + s * s); }
// a transmittance re-expressed for the parent ray's path length per column
fn transmittanceForPath(T: f32, cascadeFrom: i32, riseFrom: i32, cascadeTo: i32, riseTo: i32) -> f32 {
  if (T <= 0.0) { return T; }
  return pow(T, pathLengthPerColumn(cascadeTo, riseTo) / pathLengthPerColumn(cascadeFrom, riseFrom));
}`}

// rays: [cascade][rise + 2^n][X / 2^n][y], the X count padded to 2^(LAST_CASCADE - n), so cascade n holds
// 2^(LAST_CASCADE - n) (2^(n+1) + 1) N rays
fn rayIndex(cascade: i32, X: i32, y: i32, rise: i32) -> u32 {
  let cascadeStart = u32(N) * (((u32(cascade) + 1u) << u32(LAST_CASCADE + 1)) - (1u << u32(LAST_CASCADE - cascade + 1)));
  return cascadeStart + u32(((((rise + pow2(cascade)) << u32(LAST_CASCADE - cascade)) + (X >> u32(cascade))) * N) + y);
}
// beyond samples: cascades 2 .. LAST_CASCADE - 1, [X / 2^n][direction + 2^n][y], 2^(LAST_CASCADE + 1) N per cascade
fn beyondSampleIndex(cascade: i32, X: i32, y: i32, direction: i32) -> u32 {
  let directionSlot = direction + pow2(cascade);
  return u32((cascade - 2) << u32(LAST_CASCADE + 1)) * u32(N) + u32(((((X >> u32(cascade)) << u32(cascade + 1)) + directionSlot) * N) + y);
}
// ray path rows: [cascade][rise + 2^n][t - 1]
fn rayPathIndex(cascade: i32, rise: i32, t: i32) -> u32 {
  let L = u32(pow2(cascade));
  return (2u * (L * L - 1u)) / 3u + L - 1u + u32(((rise + pow2(cascade)) << u32(cascade)) + t - 1);
}
// shared prefixes: [cascade >= 1][rise + 2^n][count, pieces...] with LAST_CASCADE + 1 entries per ray pair
fn sharedPrefixIndex(cascade: i32, rise: i32) -> u32 { return u32((pow2(cascade + 1) - 4 + rise + pow2(cascade)) * (LAST_CASCADE + 1)); }
// cascade-1 beyond values: [probe pair][X / 2][direction + 2][y + 2], rows -2 .. N + 1
fn cascade1Cones() -> i32 { return apexColumns(1) * 4 * (N + 4); }
fn cascade1BeyondIndex(X: i32, y: i32, direction: i32, pair: i32) -> u32 { return u32(pair * cascade1Cones() + (((X >> 1u) * 4 + direction + 2) * (N + 4)) + y + 2); }

${halfFloatRays ? `fn unpackRay(p: vec2u) -> vec4f { return vec4f(unpack2x16float(p.x), unpack2x16float(p.y)); }
fn packRay(r: vec4f) -> vec2u { return vec2u(pack2x16float(r.xy), pack2x16float(r.zw)); }`
      : `fn unpackRay(p: vec4f) -> vec4f { return p; }
fn packRay(r: vec4f) -> vec4f { return r; }`}
fn loadRay(cascade: i32, X: i32, y: i32, rise: i32) -> vec4f {
  if (X >= N - 1 || y < 0 || y >= N) { return vec4f(0.0, 0.0, 0.0, 1.0); }   // outside the domain: empty
  return unpackRay(rays[rayIndex(cascade, X, y, rise)]);
}

${averagedRounding ? `// averaged rounding: 0 when the junction of ray (cascade, rise)'s halves (first-half rise a) steps a row and the corner cell
// its chord crosses there is opaque. A straight line through square cells touches both cells of the column pair at a row
// step; the digital path misses one of them. Opaque cells only: media keep the secant model.
fn stepCornerClear(cascade: i32, X: i32, y: i32, rise: i32, a: i32) -> f32 {
  let h = pow2(cascade - 1); let firstStep = rayPathRows[rayPathIndex(cascade - 1, rise - a, 1)].x;
  if (firstStep == 0 || X + h + 1 >= N) { return 1.0; }
  // the chord crosses the boundary between rows a and a + firstStep at t = (a + firstStep / 2) 2^n / rise; before the column boundary h + 1/2?
  let lhs = (2 * a + firstStep) * pow2(cascade); let rhs = (2 * h + 1) * rise;
  let early = select((lhs > rhs), (lhs < rhs), (rise > 0));
  let cx = select(X + h + 1, X + h, early); let cy = select(y + a, y + a + firstStep, early);
  if (cy < 0 || cy >= N) { return 1.0; }   // rows outside the domain are clear
  return select(1.0, 0.0, transmittance[scenePixelIndex(cx, cy)] <= 0.0);
}
// the ray's transmittance as the expectation over both roundings of an odd split (stochastic rounding of the path), each
// with its corner cell; light stays on the split rule's own path
fn averagedTransmittance(cascade: i32, X: i32, y: i32, rise: i32, a: i32, T: f32) -> f32 {
  let own = T * stepCornerClear(cascade, X, y, rise, a);
  if ((rise & 1) == 0) { return own; }
  let a2 = rise - a; let h = pow2(cascade - 1);
  let other = transmittanceForPath(loadRay(cascade - 1, X, y, a2).w, cascade - 1, a2, cascade, rise)
            * transmittanceForPath(loadRay(cascade - 1, X + h, y + a2, a).w, cascade - 1, a, cascade, rise);
  return 0.5 * (own + other * stepCornerClear(cascade, X, y, rise, a2));
}

` : ''}${onePass ? `// one-pass rounding. Rows outside the domain and columns beyond it are clear.
fn clearAt(x: i32, y: i32) -> f32 {
  if (x >= N || y < 0 || y >= N) { return 1.0; }
  return select(1.0, 0.0, transmittance[scenePixelIndex(x, y)] <= 0.0);
}
// a path in column x, row y, stepping to row y + rowStep in column x + 1, slope s: the detour through the target's own column
// with probability 1 / (1 + |s|) (it hides the target's side face), else through the source's
fn stepCornerClearExpected(x: i32, y: i32, rowStep: i32, s: f32) -> f32 {
  let q = 1.0 / (1.0 + s);
  return q * clearAt(x + 1, y) + (1.0 - q) * clearAt(x, y + rowStep);
}
// the expected corner at the junction of ray (cascade, X, y, rise) split at a: the second half steps a row first with
// probability |its slope|
fn junctionCornerClear(cascade: i32, X: i32, y: i32, rise: i32, a: i32) -> f32 {
  let b = rise - a; let h = pow2(cascade - 1);
  if (b == 0 || X + h >= N - 1) { return 1.0; }
  let p = f32(abs(b)) / f32(h);
  return 1.0 - p + p * stepCornerClearExpected(X + h, y + a, select(-1, 1, b > 0), f32(abs(rise)) / f32(pow2(cascade)));
}
// a shared prefix ends where either ray's junction does: the smaller of the two corners
fn sharedPrefixCornerClear(cascade: i32, X: i32, y: i32, rise: i32, a: i32) -> f32 { return min(junctionCornerClear(cascade, X, y, rise, a), junctionCornerClear(cascade, X, y, rise + 1, a)); }
// moments of A followed by B, B gated by T, at the parent's scale
fn followedBy(A: vec4f, T: f32, B: vec4f) -> vec3f {
  return vec3f(A.x + T * B.x, 0.5 * (A.y + T * (B.x + B.y)), 0.25 * (A.z + T * (B.x + 2.0 * B.y + B.z)));
}
// ray (cascade, X, y, rise) from its halves split at a, the junction gated by its expected corner
fn composeHalves(cascade: i32, X: i32, y: i32, rise: i32, a: i32) -> vec4f {
  let b = rise - a; let h = pow2(cascade - 1);
  let first = loadRay(cascade - 1, X, y, a); let second = loadRay(cascade - 1, X + h, y + a, b);
  let Ta = transmittanceForPath(first.w, cascade - 1, a, cascade, rise) * junctionCornerClear(cascade, X, y, rise, a); let Tb = transmittanceForPath(second.w, cascade - 1, b, cascade, rise);
  return vec4f(followedBy(first, Ta, second), Ta * Tb);
}
fn expectedRay(cascade: i32, X: i32, y: i32, rise: i32) -> vec4f {
  let a = rise >> 1u;   // floor(rise / 2)
  if ((rise & 1) == 0) { return composeHalves(cascade, X, y, rise, a); }
  return 0.5 * (composeHalves(cascade, X, y, rise, a) + composeHalves(cascade, X, y, rise, a + 1));
}
fn loadSharedPrefix(cascade: i32, X: i32, y: i32, rise: i32) -> vec4f {
  if (cascade == 0 || X >= N - 1 || y < 0 || y >= N) { return vec4f(0.0); }
  return rays[SHARED_PREFIX_OFFSET + rayIndex(cascade, X, y, rise)];
}
// rays r, r + 1 share their first halves (e = ceil(r / 2)) with probability 1/2, then share a prefix of the second halves;
// otherwise they share a prefix of the first halves (r >> 1, r >> 1 + 1)
fn expectedSharedPrefix(cascade: i32, X: i32, y: i32, rise: i32) -> vec4f {
  let h = pow2(cascade - 1); let e = (rise + 1) >> 1u;
  let first = loadRay(cascade - 1, X, y, e); let T = transmittanceForPath(first.w, cascade - 1, e, cascade, rise) * sharedPrefixCornerClear(cascade, X, y, rise, e);
  let apart = loadSharedPrefix(cascade - 1, X, y, rise >> 1u);
  return vec4f(0.5 * (followedBy(first, T, loadSharedPrefix(cascade - 1, X + h, y + e, rise - e)) + vec3f(apart.x, 0.5 * apart.y, 0.25 * apart.z)), 0.0);
}

` : ''}// pass 1: extension
@compute @workgroup_size(${WORKGROUP})
fn extendRays(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  // threads in (X, rise, y) order, so neighbouring rays share their halves in cache
  let cascade = params.cascade; let L = pow2(cascade); let rises = 2 * L + 1;
  let i = threadIndex(id, groups);
  if (i >= apexColumns(cascade) * rises * N) { return; }
  let y = i % N; let rise = (i / N) % rises - L; let X = (i / (N * rises)) * L;
  var ray = vec4f(0.0, 0.0, 0.0, 1.0);
  if (cascade == 0) {
    let x = X + 1; let row = y + rise;
    if (x < N && row >= 0 && row < N) {
      let cell = scenePixelIndex(x, row); let a = transmittance[cell]; var e = emission[cell];
      var T = 0.0;
      if (a > 0.0) { T = pow(a, pathLengthPerColumn(0, rise)); }   // a is the transmittance across one cell
${pointCells ? '' : onePass ? `      if (e > 0.0 && T < 1.0) { e *= select(1.0, (1.0 - T) / -log(T), T > 0.0); }   // attenuated inside the cell; opaque: an opaque light's surface
`
         : `      if (e > 0.0 && T < 1.0) { e *= select(1.0, (1.0 - T) / -log(T), T > 0.0); }   // the cell's own emission, attenuated inside it; opaque: an opaque light's surface
`}      ray = vec4f(e, e, e, T);
    }
  } else {
${onePass ? `    ray = expectedRay(cascade, X, y, rise);
    if (rise < L) { rays[SHARED_PREFIX_OFFSET + rayIndex(cascade, X, y, rise)] = expectedSharedPrefix(cascade, X, y, rise); }
` : `    let a = splitRise(cascade, rise); let b = rise - a; let h = pow2(cascade - 1);
    let first = loadRay(cascade - 1, X, y, a); let second = loadRay(cascade - 1, X + h, y + a, b);
    let Ta = transmittanceForPath(first.w, cascade - 1, a, cascade, rise); let Tb = transmittanceForPath(second.w, cascade - 1, b, cascade, rise);
    ray = vec4f(first.x + Ta * second.x,
                0.5 * (first.y + Ta * (second.x + second.y)),
                0.25 * (first.z + Ta * (second.x + 2.0 * second.y + second.z)),
                ${averagedRounding ? 'averagedTransmittance(cascade, X, y, rise, a, Ta * Tb)' : 'Ta * Tb'});
`}  }
  rays[rayIndex(cascade, X, y, rise)] = packRay(ray);
}

${onePass ? `// a ray read as a point source at its light's centroid on its chord of slope s (one-pass: the expected path's rows):
// mu0 / D * (1 + var (1 + s^2) / D^2), D the distance from the sample to the centroid
fn readAsPointOnChord(ray: vec4f, cascade: i32, s: f32, back: vec2f, across: vec2f, on: vec2<bool>) -> vec2f {
  if (!any(on)) { return vec2f(0.0); }
  let Lf = f32(pow2(cascade));
  let mu0 = max(ray.x, NO_LIGHT); let q = ray.y / mu0; let tb = Lf * q;
  let variance = max(Lf * Lf * (ray.z / mu0 - q * q), 0.0) * (1.0 + s * s);
  let dx = back + tb; let dy = across + s * tb; let r = inverseSqrt(dx * dx + dy * dy);
  ${opaqueLightKernel ? 'let kernel = (1.0 + abs(s)) * inverseSqrt(1.0 + s * s);   // an opaque emitter shows its projected width' : 'let kernel = 1.0;'}
  return select(vec2f(0.0), kernel * ray.x * r * (1.0 + variance * r * r), on & vec2<bool>(ray.x > NO_LIGHT));
}
fn readAsPoint(ray: vec4f, cascade: i32, rise: i32, back: vec2f, across: vec2f, on: vec2<bool>) -> vec2f { return readAsPointOnChord(ray, cascade, f32(rise) / f32(pow2(cascade)), back, across, on); }

` : `// a ray read as a point source at its light's centroid: mu0 / D * (1 + var (1 + s^2) / D^2), D the distance from the sample
// to the centroid
fn readAsPoint(ray: vec4f, cascade: i32, rise: i32, back: vec2f, across: vec2f, on: vec2<bool>) -> vec2f {
  if (!any(on)) { return vec2f(0.0); }
  let L = pow2(cascade); let Lf = f32(L);
  let mu0 = max(ray.x, NO_LIGHT); let q = ray.y / mu0; let tb = Lf * q;
  let t = clamp(i32(floor(tb)), 1, L); let f = tb - f32(t);
  let rows = vec2f(rayPathRows[rayPathIndex(cascade, rise, t)]);
  let rb = rows.x * (1.0 - f) + rows.y * f;
  let s = f32(rise) / Lf;
  let variance = max(Lf * Lf * (ray.z / mu0 - q * q), 0.0) * (1.0 + s * s);
  let dx = back + tb; let dy = across + rb; let r = inverseSqrt(dx * dx + dy * dy);
  return select(vec2f(0.0), ray.x * r * (1.0 + variance * r * r), on & vec2<bool>(ray.x > NO_LIGHT));
}

`}${onePass ? `// ray (cascade, X, y, rise) read as its halves split at a, the second gated by the junction's expected corner
fn readHalves(cascade: i32, X: i32, y: i32, rise: i32, a: i32, back: vec2f, across: vec2f, on: vec2<bool>) -> vec2f {
  let h = pow2(cascade - 1); let first = loadRay(cascade - 1, X, y, a);
  var v = readAsPoint(first, cascade - 1, a, back, across, on);
  if (X + h < N - 1) {
    v += first.w * junctionCornerClear(cascade, X, y, rise, a) * readAsPoint(loadRay(cascade - 1, X + h, y + a, rise - a), cascade - 1, rise - a, back + f32(h), across + f32(a), on);
  }
  return v;
}
` : ''}// ray (cascade, X, y, rise); samples closer than its length read its two halves. No sample is closer than half a ray, so one
// split is enough.
fn readRay(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> vec2f {
  if (X >= N - 1) { return vec2f(0.0); }
  let close = vec2<bool>(cascade > 0) & (back + 1.0 < vec2f(f32(pow2(cascade))));
  var v = vec2f(0.0);
  if (!all(close)) { v = readAsPoint(loadRay(cascade, X, y, rise), cascade, rise, back, across, !close); }
${onePass ? `  if (any(close)) {   // both roundings of an odd split
    let a = rise >> 1u;
    if ((rise & 1) == 0) { v += readHalves(cascade, X, y, rise, a, back, across, close); }
    else { v += 0.5 * (readHalves(cascade, X, y, rise, a, back, across, close) + readHalves(cascade, X, y, rise, a + 1, back, across, close)); }
  }
` : `  if (any(close)) {
    let a = splitRise(cascade, rise); let h = pow2(cascade - 1); let first = loadRay(cascade - 1, X, y, a);
    v += readAsPoint(first, cascade - 1, a, back, across, close);
    if (X + h < N - 1) {
      v += first.w * readAsPoint(loadRay(cascade - 1, X + h, y + a, rise - a), cascade - 1, rise - a, back + f32(h), across + f32(a), close);
    }
  }
`}  return v;
}

${onePass ? `// the prefix shared by rays r and r + 1 of cascade n: its stored expected moments on the mid-slope chord; a sample closer
// than its length reads one cascade of the mixture
fn readSharedPrefixAsPoint(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f, on: vec2<bool>) -> vec2f {
  if (cascade == 0 || X >= N - 1) { return vec2f(0.0); }
  return readAsPointOnChord(loadSharedPrefix(cascade, X, y, rise), cascade, (f32(rise) + 0.5) / f32(pow2(cascade)), back, across, on);
}
fn readSharedPrefix(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> vec2f {
  if (cascade == 0 || X >= N - 1) { return vec2f(0.0); }
  let close = back + 1.0 < vec2f(f32(pow2(cascade)));
  var v = vec2f(0.0);
  if (!all(close)) { v = readSharedPrefixAsPoint(cascade, X, y, rise, back, across, !close); }
  if (any(close)) {
    let h = pow2(cascade - 1); let e = (rise + 1) >> 1u; let first = loadRay(cascade - 1, X, y, e);
    v += 0.5 * (readAsPoint(first, cascade - 1, e, back, across, close) + readSharedPrefixAsPoint(cascade - 1, X, y, rise >> 1u, back, across, close)
                + first.w * sharedPrefixCornerClear(cascade, X, y, rise, e) * readSharedPrefixAsPoint(cascade - 1, X + h, y + e, rise - e, back + f32(h), across + f32(e), close));
  }
  return v;
}

` : `// the prefix shared by rays r and r + 1 of cascade n, read as its pieces (none of which splits), each gating the next
fn readSharedPrefix(cascade: i32, X0: i32, y0: i32, rise: i32, back0: vec2f, across0: vec2f) -> vec2f {
  if (cascade == 0 || X0 >= N - 1) { return vec2f(0.0); }
  var X = X0; var y = y0; var back = back0; var across = across0; var v = vec2f(0.0); var T = 1.0;
  let base = sharedPrefixIndex(cascade, rise);
  let pieces = u32(sharedPrefixes[base].x);
  for (var i = 1u; i <= pieces; i += 1u) {
    let piece = sharedPrefixes[base + i]; let pieceCascade = piece.x; let pieceRise = piece.y;
    let ray = loadRay(pieceCascade, X, y, pieceRise);
    v += T * readAsPoint(ray, pieceCascade, pieceRise, back, across, vec2<bool>(true));
    T *= ray.w;
    X += pow2(pieceCascade); y += pieceRise; back += f32(pow2(pieceCascade)); across += f32(pieceRise);
    if (X >= N - 1) { break; }
  }
  return v;
}

`}// the ray chain of cascade-n0 rays of rise r0 from (X, y) to the domain end, for one sample: the largest aligned rays with
// 2^k <= back + 1${onePass ? '; every junction inside it gated by its expected corner, so the cut does not matter' : ''}
fn readRayChain(cascade0: i32, X0: i32, y0: i32, rise0: i32, back0: f32, across0: f32) -> f32 {
  var X = X0; var y = y0; var back = back0; var across = across0; var v = 0.0; var T = 1.0;
${onePass ? '  let s = f32(abs(rise0)) / f32(pow2(cascade0));\n' : ''}  loop {
    if (X >= N - 1) { break; }
${onePass ? `    if (X != X0 && rise0 != 0) { T *= 1.0 - s + s * stepCornerClearExpected(X, y, select(-1, 1, rise0 > 0), s); }   // the junction's expected corner
` : ''}    let k = max(cascade0, min(min(i32(countTrailingZeros(u32(X))), i32(firstLeadingBit(u32(back + 1.0)))), LAST_CASCADE)); let rise = rise0 << u32(k - cascade0);
    let ray = loadRay(k, X, y, rise);
    v += T * readAsPoint(ray, k, rise, vec2f(back), vec2f(across), vec2<bool>(true)).x;
    T *= ray.w;
    X += pow2(k); y += rise; back += f32(pow2(k)); across += f32(rise);
  }
  return v;
}

fn axisSlope(cascade: i32, direction: i32) -> f32 { let n = f32(pow2(cascade)); return (f32(direction) + 0.5) / n; }
// the sample distances of cascade n: Chebyshev nodes over [0, 2s] (same apex + forward: samples are at most 2s - 2 back)
// or [0, s] (same apex)
fn sampleDistances(cascade: i32) -> vec2f { let h = select(0.5, 1.0, SAME_APEX_AND_FORWARD) * f32(pow2(cascade)); return h * vec2f(1.0 + 0.70710678118654752, 1.0 - 0.70710678118654752); }

// a cone's beyond-value from its samples: the point projected onto the axis, 1/F interpolated linearly in distance
fn readBeyondFromSamples(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> vec2f {
  if (X >= N - 1 || y < 0 || y >= N || cascade >= LAST_CASCADE) { return vec2f(0.0); }
  let F = beyondSamples[beyondSampleIndex(cascade, X, y, direction)];
  let at = sampleDistances(cascade); let s = axisSlope(cascade, direction); let x = (back + s * across) / (1.0 + s * s);
  let w0 = (x - at.y) / (at.x - at.y); let w1 = (x - at.x) / (at.y - at.x);
  if (min(F.x, F.y) > SAMPLE_RATIO_MIN * max(max(F.x, F.y), 1e-30)) { return 1.0 / (w0 / F.x + w1 / F.y); }
  return w0 * F.x + w1 * F.y;
}
// a child's beyond-value: from its samples, or at cascade 1 from pass 3. Cascade-1 cones are read only by probes, at
// distance 0, 1 or 2 (2 only with same apex + forward), with the same point in both lanes; pair 0 holds distances 0 and 1,
// pair 1 distance 2.
fn readChildBeyond(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> vec2f {
  if (cascade == 1) {
    if (X >= N - 1) { return vec2f(0.0); }
    let k = i32(round(back.x));
    let F = cascade1Beyond[cascade1BeyondIndex(X, y, direction, k >> 1u)];
    return vec2f(select(F.x, F.y, k == 1));
  }
  return readBeyondFromSamples(cascade, X, y, direction, back, across);
}

// the cone's own interval: its two rays over its first 2^n columns, their shared prefix counted once
fn readOwnInterval(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> vec2f {
  return readRay(cascade, X, y, direction, back, across) + readRay(cascade, X, y, direction + 1, back, across) - readSharedPrefix(cascade, X, y, direction, back, across);
}
fn readChild(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> vec2f {
  if (X >= N - 1) { return vec2f(0.0); }
  return readOwnInterval(cascade, X, y, direction, back, across) + readChildBeyond(cascade, X, y, direction, back, across);
}

${onePass ? `// even apex column: the second half of ray r of cascade n + 1 over both roundings, gated by the first half and the
// junction's expected corner; and its part of the shared prefix of r, r + 1 (first halves equal with probability 1/2)
fn secondHalfOf(cascade: i32, X: i32, y: i32, rise: i32, a: i32, back: vec2f, across: vec2f) -> vec2f {
  let n = pow2(cascade); let T = loadRay(cascade, X, y, a).w * junctionCornerClear(cascade + 1, X, y, rise, a);
  return T * readRay(cascade, X + n, y + a, rise - a, back + f32(n), across + f32(a));
}
fn readSecondHalf(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> vec2f {
  let a = rise >> 1u;
  if ((rise & 1) == 0) { return secondHalfOf(cascade, X, y, rise, a, back, across); }
  return 0.5 * (secondHalfOf(cascade, X, y, rise, a, back, across) + secondHalfOf(cascade, X, y, rise, a + 1, back, across));
}
fn readSecondHalfPrefix(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> vec2f {
  let n = pow2(cascade); let e = (rise + 1) >> 1u;
  let T = 0.5 * loadRay(cascade, X, y, e).w * sharedPrefixCornerClear(cascade + 1, X, y, rise, e);
  return T * readSharedPrefix(cascade, X + n, y + e, rise - e, back + f32(n), across + f32(e));
}

` : `// even apex column: the second half of ray r of cascade n + 1, gated by the first half; and its part of the shared prefix
// of r, r + 1
fn readSecondHalf(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> vec2f {
  let n = pow2(cascade); let a = splitRise(cascade + 1, rise); let T = loadRay(cascade, X, y, a).w;
  return T * readRay(cascade, X + n, y + a, rise - a, back + f32(n), across + f32(a));
}
fn readSecondHalfPrefix(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> vec2f {
  let n = pow2(cascade); let a = splitRise(cascade + 1, rise);
  if (a != splitRise(cascade + 1, rise + 1)) { return vec2f(0.0); }
  let T = loadRay(cascade, X, y, a).w;
  return T * readSharedPrefix(cascade, X + n, y + a, rise - a, back + f32(n), across + f32(a));
}

`}// a cone's beyond-value, exactly, from rays and its children's beyond-values
fn readBeyondExactly(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> vec2f {
  let n = pow2(cascade);
  if (X + n >= N - 1 || cascade >= LAST_CASCADE) { return vec2f(0.0); }
  var v = vec2f(0.0);
  if (((X >> u32(cascade)) & 1) == 0) {
    // even apex column: the second halves of rays A, B, C of cascade n + 1 (shared by both patterns)
    let riseA = 2 * direction; let riseB = riseA + 1; let riseC = riseA + 2;
    v += readSecondHalf(cascade, X, y, riseA, back, across) + readSecondHalf(cascade, X, y, riseB, back, across) + readSecondHalf(cascade, X, y, riseC, back, across);
    v -= readSecondHalfPrefix(cascade, X, y, riseA, back, across) + readSecondHalfPrefix(cascade, X, y, riseB, back, across);
    // same apex: the children at the same X
    v += select(1.0, 0.5, SAME_APEX_AND_FORWARD) * (readChildBeyond(cascade + 1, X, y, 2 * direction, back, across) + readChildBeyond(cascade + 1, X, y, 2 * direction + 1, back, across));
    if (!SAME_APEX_AND_FORWARD && X + 2 * n < N - 1) {   // same-apex pattern: both children contain ray B's continuation beyond X + 2s
      let T = loadRay(cascade + 1, X, y, riseB).w; let backB = back + f32(2 * n); let acrossB = across + f32(riseB);
      v -= T * vec2f(readRayChain(cascade + 1, X + 2 * n, y + riseB, riseB, backB.x, acrossB.x), readRayChain(cascade + 1, X + 2 * n, y + riseB, riseB, backB.y, acrossB.y));
    }
    // forward (same apex + forward): the children re-rooted at the ends of A and C
    if (SAME_APEX_AND_FORWARD && X + 2 * n < N - 1) {
      let TA = 0.5 * loadRay(cascade + 1, X, y, riseA).w; let TC = 0.5 * loadRay(cascade + 1, X, y, riseC).w;
      v += TA * readChild(cascade + 1, X + 2 * n, y + riseA, 2 * direction, back + f32(2 * n), across + f32(riseA));
      v += TC * readChild(cascade + 1, X + 2 * n, y + riseC, 2 * direction + 1, back + f32(2 * n), across + f32(riseC));
    }
  } else {
    // odd apex column, forward re-rooted: the children start where the cone's own rays end
    let rise0 = direction; let rise1 = rise0 + 1;
    let T0 = loadRay(cascade, X, y, rise0).w; let T1 = loadRay(cascade, X, y, rise1).w;
    v += T0 * readChild(cascade + 1, X + n, y + rise0, 2 * direction, back + f32(n), across + f32(rise0));
    v += T1 * readChild(cascade + 1, X + n, y + rise1, 2 * direction + 1, back + f32(n), across + f32(rise1));
  }
  return v;
}

// pass 2: merge, cones at even and odd apex columns as two kernels
fn mergeCone(i: i32, parity: i32) {
  let cascade = params.cascade; let cones = pow2(cascade + 1);
  if (i >= (apexColumns(cascade) + 1 - parity) / 2 * cones * N) { return; }
  let y = i % N; let direction = (i / N) % cones - pow2(cascade); let X = (2 * (i / (N * cones)) + parity) * pow2(cascade);
  let back = sampleDistances(cascade);
  beyondSamples[beyondSampleIndex(cascade, X, y, direction)] = readBeyondExactly(cascade, X, y, direction, back, axisSlope(cascade, direction) * back);
}
@compute @workgroup_size(${WORKGROUP})
fn mergeEvenApexColumns(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) { mergeCone(threadIndex(id, groups), 0); }
@compute @workgroup_size(${WORKGROUP})
fn mergeOddApexColumns(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) { mergeCone(threadIndex(id, groups), 1); }

// pass 3: cascade-1 cones. Cone (1, X, y, d) is read by probes (X - k, y - k e) at (k, k e), k = 0, 1, 2, e = -1, 0, 0, 1 for
// d = -2 .. 1. Pair 0 evaluates k = 0 and 1, pair 1 k = 2. Rows -2 .. N + 1 are stored, since edge probes read them.
@compute @workgroup_size(${WORKGROUP})
fn evaluateCascade1Beyond(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= 2 * cascade1Cones()) { return; }
  let pair = i / cascade1Cones(); let cone = i % cascade1Cones();
  let y = cone % (N + 4) - 2; let direction = ((cone / (N + 4)) & 3) - 2; let X = 2 * (cone / (4 * (N + 4)));
  let e = select(select(0.0, 1.0, direction == 1), -1.0, direction == -2);
  let back = select(vec2f(0.0, 1.0), vec2f(2.0), pair == 1);
  cascade1Beyond[i] = readBeyondExactly(1, X, y, direction, back, back * e);
}

// pass 4: gather, with the probe as both samples
fn addFluence(x: i32, y: i32, v: f32, overwrite: bool) {
  let cell = scenePixelIndex(x, y); let a = transmittance[cell];
${pointCells ? '  let w = v;   // point model: no half cell\n' : `  let w = select(0.0, v * pow(a, 0.561), a > 0.0);   // the probe sits at its cell's centre, so every path first crosses half the cell
`}  if (overwrite) { fluence[cell] = w; } else { fluence[cell] += w; }
}
// the seam (the ray both cones contain) and, in the +-x quadrants, the diagonals the +-y quadrants also contain
@compute @workgroup_size(${WORKGROUP})
fn gatherSeams(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N;
  var v = -readRayChain(0, x, y, 0, 0.0, 0.0);
  if (params.subtractSharedDiagonals == 1) { v -= readRayChain(0, x, y, 1, 0.0, 0.0) + readRayChain(0, x, y, -1, 0.0, 0.0); }
  addFluence(x, y, v, params.quadrant == 0);
}
@compute @workgroup_size(${WORKGROUP})
fn gatherCones(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N;
  var v = 0.0;
  for (var direction = -1; direction < 1; direction += 1) { v += (readOwnInterval(0, x, y, direction, vec2f(0.0), vec2f(0.0)) + readBeyondExactly(0, x, y, direction, vec2f(0.0), vec2f(0.0))).x; }
  addFluence(x, y, v, false);
}
`;
}

// tables, built once on the CPU
function splitRuleOf(splitRule) {
  return (cascade, rise) => {
    if ((rise & 1) === 0) return rise >> 1;
    const absRise = Math.abs(rise);
    const away = splitRule === 'alternating' ? (cascade & 1) === 0 : (cascade & 1) === 0 ? ((absRise >> 2) & 1) === 1 : ((absRise >> 1) & 1) === 0;
    const a = (absRise >> 1) + (away ? 1 : 0);
    return rise > 0 ? a : -a;
  };
}
// rows of ray (cascade, rise) at columns 1 .. 2^n
function rayPathRowsOf(split, cascade, rise) {
  if (cascade === 0) return [rise];
  const a = split(cascade, rise);
  return [...rayPathRowsOf(split, cascade - 1, a), ...rayPathRowsOf(split, cascade - 1, rise - a).map((r) => r + a)];
}
// every ray's (row at t, row at t + 1), t = 1 .. 2^n; the last t repeats its row
function buildRayPathRows(split, lastCascade) {
  const out = [];
  for (let n = 0; n <= lastCascade; n++) {
    const L = 1 << n;
    for (let rise = -L; rise <= L; rise++) { const r = rayPathRowsOf(split, n, rise); r.forEach((row, i) => out.push(row, r[Math.min(i + 1, L - 1)])); }
  }
  return new Int32Array(out);
}
// pieces (cascade, rise) of the prefix shared by rays r, r + 1 of cascade n, in order along the ray,
// as (count, 0) followed by lastCascade entries per ray pair
function buildSharedPrefixes(split, lastCascade) {
  const out = [];
  for (let n = 1; n <= lastCascade; n++) {
    const L = 1 << n;
    for (let rise = -L; rise < L; rise++) {
      const pieces = [];
      for (let k = n, r = rise; k > 0; k--) {
        const a = split(k, r);
        if (a === split(k, r + 1)) { pieces.push(k - 1, a); r -= a; } else r = a;
      }
      const entry = new Array(2 * (lastCascade + 1)).fill(0); entry[0] = pieces.length / 2; entry.splice(2, pieces.length, ...pieces);
      out.push(...entry.slice(0, 2 * (lastCascade + 1)));
    }
  }
  return new Int32Array(out);
}

export class MomentHRC {
  // splitRule: 'balanced' | 'alternating': how odd rises split into halves
  // mergePattern: 'sameApexAndForward' | 'sameApex': the merge at even apex columns
  // stochasticRounding: 'off' | 'averaged' (each ray's transmittance averaged over both roundings of its odd split, each path
  //   blocked by the corner cell it crosses at the junction) | 'onePass' (every odd split rounds either way with 1/2, so rays
  //   hold expected moments; a junction that steps a row passes one of its two corner cells)
  // opaqueLightKernel (onePass only): point reads count an emitting cell's projected width, for scenes whose lights are all
  //   opaque
  // cellModel: 'square' | 'point' (transmittance per cell crossed)
  // halfFloatRays: store rays as f16
  constructor(device, N, { halfFloatRays = false, splitRule = 'balanced', mergePattern = 'sameApexAndForward', stochasticRounding = 'off',
                           opaqueLightKernel = false, cellModel = 'square' } = {}) {
    const lastCascade = Math.max(1, Math.ceil(Math.log2(N)));
    const averagedRounding = stochasticRounding === 'averaged', onePass = stochasticRounding === 'onePass';
    if (onePass) halfFloatRays = false;   // the one-pass shared prefixes sit after the plain rays as vec4f
    if (!onePass) opaqueLightKernel = false;
    Object.assign(this, { device, N, lastCascade, splitRule, mergePattern, stochasticRounding, opaqueLightKernel });
    const apexColumns = (n) => ((N - 2) >> n) + 1;
    const storage = GPUBufferUsage.STORAGE, copyDst = GPUBufferUsage.COPY_DST;
    const buffer = (size, usage = storage) => device.createBuffer({ size: Math.max(size, 16), usage });
    const filled = (data) => { const b = buffer(data.byteLength, storage | copyDst); device.queue.writeBuffer(b, 0, data); return b; };

    // sizes (match the shader's index functions)
    this.extensionThreads = Array.from({ length: lastCascade + 1 }, (_, n) => apexColumns(n) * (2 * (1 << n) + 1) * N);
    const rayCount = N * ((lastCascade + 2) * 2 ** (lastCascade + 1) - 1);   // where cascade lastCascade + 1 would start
    const beyondSampleCount = Math.max(lastCascade - 2, 0) * 2 ** (lastCascade + 1) * N;
    this.cascade1Threads = (mergePattern === 'sameApexAndForward' ? 2 : 1) * apexColumns(1) * 4 * (N + 4);   // probe pairs per cascade-1 cone

    this.emissionBuffer = buffer(N * N * 4, storage | copyDst);
    this.transmittanceBuffer = buffer(N * N * 4, storage | copyDst);
    const cascade1Count = 2 * apexColumns(1) * 4 * (N + 4);
    // one-pass: one shared-prefix ray per ray slot after the plain ones
    this.rayBuffer = buffer(rayCount * (halfFloatRays ? 8 : 16) * (onePass ? 2 : 1));
    this.beyondSampleBuffer = buffer(beyondSampleCount * 8);
    this.cascade1BeyondBuffer = buffer(cascade1Count * 8);
    this.fluenceBuffer = buffer(N * N * 4, storage | GPUBufferUsage.COPY_SRC);
    this.readbackBuffer = buffer(N * N * 4, GPUBufferUsage.MAP_READ | copyDst);
    const split = splitRuleOf(splitRule);
    this.rayPathRowsBuffer = filled(buildRayPathRows(split, lastCascade)); this.sharedPrefixBuffer = filled(buildSharedPrefixes(split, lastCascade));

    // pass parameters (one uniform buffer, dynamic offsets): per quadrant, one per cascade and two for the gather
    // (with / without the shared diagonals); offset (quadrant, cascade, subtractSharedDiagonals)
    const blocks = [];
    this.paramOffset = (quadrant, cascade, diagonals = 0) => {
      const key = `${quadrant}:${cascade}:${diagonals}`;
      let b = blocks.findIndex((p) => p.key === key);
      if (b < 0) { b = blocks.length; blocks.push({ key, values: [cascade, quadrant, diagonals, 0] }); }
      return b * PARAM_STRIDE;
    };
    for (let q = 0; q < 4; q++) { for (let n = 0; n <= lastCascade; n++) this.paramOffset(q, n); this.paramOffset(q, 0, 1); }
    const paramData = new Int32Array((blocks.length * PARAM_STRIDE) / 4);
    blocks.forEach((p, b) => paramData.set(p.values, (b * PARAM_STRIDE) / 4));
    this.paramBuffer = buffer(paramData.byteLength, GPUBufferUsage.UNIFORM | copyDst); device.queue.writeBuffer(this.paramBuffer, 0, paramData);

    this.module = device.createShaderModule({ code: shaderSource({ N, lastCascade, halfFloatRays, splitRule, mergePattern, averagedRounding, onePass,
      opaqueLightKernel, pointCells: cellModel === 'point', rayCount }) });
    const readOnly = { type: 'read-only-storage' }, readWrite = { type: 'storage' };
    const kinds = [readOnly, readOnly, readWrite, readWrite, readWrite, { type: 'uniform', hasDynamicOffset: true }, readOnly, readOnly, readWrite];
    const layout = device.createBindGroupLayout({ entries: kinds.map((b, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: b })) });
    // every buffer is a field: the page destroys them all when it rebuilds the cascade
    const buffers = [this.emissionBuffer, this.transmittanceBuffer, this.rayBuffer, this.beyondSampleBuffer, this.fluenceBuffer, this.paramBuffer,
      this.rayPathRowsBuffer, this.sharedPrefixBuffer, this.cascade1BeyondBuffer];
    this.bindGroup = device.createBindGroup({ layout, entries: buffers.map((b, binding) => ({ binding, resource: binding === 5 ? { buffer: b, size: 16 } : { buffer: b } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint) => device.createComputePipeline({ layout: pipelineLayout, compute: { module: this.module, entryPoint } });
    this.pipelines = { extendRays: pipeline('extendRays'), merge: [pipeline('mergeEvenApexColumns'), pipeline('mergeOddApexColumns')],
                       evaluateCascade1Beyond: pipeline('evaluateCascade1Beyond'), gather: [pipeline('gatherSeams'), pipeline('gatherCones')] };
  }

  async compileInfo() { return (await this.module.getCompilationInfo()).messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`); }

  upload(E, A) { this.device.queue.writeBuffer(this.emissionBuffer, 0, E); this.device.queue.writeBuffer(this.transmittanceBuffer, 0, A); }

  dispatch(pass, pipeline, paramOffset, threads) {
    if (threads <= 0) return;
    const groups = Math.ceil(threads / WORKGROUP), rows = Math.ceil(groups / 65535);
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.bindGroup, [paramOffset]); pass.dispatchWorkgroups(Math.ceil(groups / rows), rows);
  }
  encodeExtension(pass, quadrant) {
    for (let n = 0; n <= this.lastCascade; n++) this.dispatch(pass, this.pipelines.extendRays, this.paramOffset(quadrant, n), this.extensionThreads[n]);
  }
  encodeMerge(pass, quadrant, cascade) {
    const apexColumns = ((this.N - 2) >> cascade) + 1, perColumn = (1 << (cascade + 1)) * this.N;
    [0, 1].forEach((parity) => this.dispatch(pass, this.pipelines.merge[parity], this.paramOffset(quadrant, cascade), ((apexColumns + 1 - parity) >> 1) * perColumn));
  }
  encodeCascade1Beyond(pass, quadrant) {
    this.dispatch(pass, this.pipelines.evaluateCascade1Beyond, this.paramOffset(quadrant, 1), this.cascade1Threads);
  }
  encodeGather(pass, quadrant, diagonals) {
    for (const p of this.pipelines.gather) this.dispatch(pass, p, this.paramOffset(quadrant, 0, diagonals), this.N * this.N);
  }
  // quadrant q: extension, merges top-down, cascade 1, gather (accumulating into the fluence in scene coordinates)
  encodeQuadrant(enc, quadrant = 0, diagonals = quadrant % 2 === 0 ? 1 : 0) {
    const pass = enc.beginComputePass();
    this.encodeExtension(pass, quadrant);
    for (let n = this.lastCascade - 1; n > 1; n--) this.encodeMerge(pass, quadrant, n);
    this.encodeCascade1Beyond(pass, quadrant);
    this.encodeGather(pass, quadrant, diagonals);
    pass.end();
  }
  // all four quadrants into the fluence buffer (no readback)
  encodeAll(enc) { for (let q = 0; q < 4; q++) this.encodeQuadrant(enc, q); }

  async readFluence() {
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(this.fluenceBuffer, 0, this.readbackBuffer, 0, this.N * this.N * 4);
    this.device.queue.submit([enc.finish()]);
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readbackBuffer.getMappedRange().slice(0)); this.readbackBuffer.unmap(); return out;
  }
  // quadrant 0 alone, without subtracting the shared diagonals
  async solveQuadrant0(E, A) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); this.encodeQuadrant(enc, 0, 0); this.device.queue.submit([enc.finish()]);
    return this.readFluence();
  }
  // all four quadrants
  async solve(E, A) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); this.encodeAll(enc); this.device.queue.submit([enc.finish()]);
    return this.readFluence();
  }

  // GPU time of each stage alone (quadrant q), averaged over reps submits
  async profile(reps = 10, quadrant = 0) {
    const dev = this.device, out = {};
    const stage = async (name, encode) => {
      await dev.queue.onSubmittedWorkDone(); const t0 = performance.now();
      for (let r = 0; r < reps; r++) { const enc = dev.createCommandEncoder(); const pass = enc.beginComputePass(); encode(pass); pass.end(); dev.queue.submit([enc.finish()]); }
      await dev.queue.onSubmittedWorkDone(); out[name] = (performance.now() - t0) / reps;
    };
    await stage('extension', (pass) => this.encodeExtension(pass, quadrant));
    for (let n = this.lastCascade - 1; n > 1; n--) await stage(`merge, cascade ${n}`, (pass) => this.encodeMerge(pass, quadrant, n));
    await stage('cascade 1', (pass) => this.encodeCascade1Beyond(pass, quadrant));
    await stage('gather', (pass) => this.encodeGather(pass, quadrant, quadrant % 2 === 0 ? 1 : 0));
    return out;
  }
}
