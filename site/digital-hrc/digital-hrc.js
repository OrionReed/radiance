// Digital HRC (WebGPU).
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
// solid lights (option): an opaque emitting cell only blocks; each of its faces next to a free cell (its skin cell) puts its
//   light into that cell, in the channel of the face's normal: x (normal -x), ym (normal -y), yp (normal +y). Rays carry
//   moments per channel; a cone reads x plus the y faces it can see (ym above its lines' start rows for upper cones, yp for
//   lower), each ray read as the exact angle of a unit x face or of a run of y faces on one row. Cone samples hold the
//   complex face field G (light = Re G), read as a one-pole fit in the point's complex position.

const WORKGROUP = 64;
const PARAM_STRIDE = 256;

function shaderSource({ N, lastCascade, splitRule, mergePattern, pointCells, rayCount, solidLights, coverageTest }) {
  return /* wgsl */ `
const N: i32 = ${N};
const LAST_CASCADE: i32 = ${lastCascade};
const SAME_APEX_AND_FORWARD: bool = ${mergePattern === 'sameApexAndForward'};   // else 'sameApex'
const NO_LIGHT: f32 = 1e-14;                 // a ray with mu0 below this has no light
const SAMPLE_RATIO_MIN: f32 = 1e-12;         // samples interpolate 1/F only when both are above this fraction of the larger

struct PassParams { cascade: i32, quadrant: i32, subtractSharedDiagonals: i32, unused: i32 }

@group(0) @binding(0) var<storage, read> emission: array<f32>;
@group(0) @binding(1) var<storage, read> transmittance: array<f32>;
@group(0) @binding(2) var<storage, read_write> rays: array<vec4f>;   // (mu0, mu1, mu2, T)
@group(0) @binding(3) var<storage, read_write> beyondSamples: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> fluence: array<f32>;
@group(0) @binding(5) var<uniform> params: PassParams;
@group(0) @binding(6) var<storage, read> rayPathRows: array<vec2i>;     // (row at t, row at t + 1) of every ray
@group(0) @binding(7) var<storage, read> sharedPrefixes: array<vec2i>;  // (count, 0) then (cascade, rise) per piece
@group(0) @binding(8) var<storage, read_write> cascade1Beyond: array<vec2f>;  // pass 3
${solidLights ? `// solid lights: after the rays, each ray's faces as 2 slots, (x mu0, mu1, mu2, y mu0), (y mu1, mu2, -, -), y being ym for
// rise > 0 and yp for rise < 0; rise-0 rays hold ym there and yp in a row ray after the faces. A cone sample is 3 slots: the
// volume light at both distances, then G at each (complex).
const FACE_OFFSET: u32 = ${rayCount}u;
const ROW_RAY_OFFSET: u32 = ${3 * rayCount}u;
alias Light = mat3x2f;   // columns: volume light at both points, G at the first point, G at the second
alias ChainLight = vec3f;   // one point: volume light, G
var<private> side: i32;   // the y faces a cone reads: 1 ym (upper cones), -1 yp (lower cones), 0 none (x faces only)
` : `alias Light = vec2f;   // light at two points
alias ChainLight = f32;
`}
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

fn loadRay(cascade: i32, X: i32, y: i32, rise: i32) -> vec4f {
  if (X >= N - 1 || y < 0 || y >= N) { return vec4f(0.0, 0.0, 0.0, 1.0); }   // outside the domain: empty
  return rays[rayIndex(cascade, X, y, rise)];
}
${solidLights ? `
// a light cell: opaque and emitting. Cells outside the domain are not lights.
fn lightAt(x: i32, y: i32) -> f32 {
  if (x < 0 || x >= N || y < 0 || y >= N) { return 0.0; }
  let cell = scenePixelIndex(x, y); let e = emission[cell];
  return select(0.0, e, transmittance[cell] <= 0.0 && e > 0.0);
}
struct Faces { x: vec3f, y: vec3f }
${coverageTest ? `// coverage test: the emission input holds the face light of every cell itself, for world normals -x, +x, -y, +y; the
// local channels x, ym, yp of each quadrant are world normals rotated with it
fn coverageFaces(cell: i32) -> vec3f {
  var normals = array<vec3i, 4>(vec3i(0, 2, 3), vec3i(2, 1, 0), vec3i(1, 3, 2), vec3i(3, 0, 1)); let n = normals[params.quadrant];
  return vec3f(emission[4 * cell + n.x], emission[4 * cell + n.y], emission[4 * cell + n.z]);
}
` : ''}   // moments mu0, mu1, mu2 of a ray's x faces and of the y faces of one side
// row rays: [cascade][X / 2^n][y], the X count padded as for rays
fn rowRayIndex(cascade: i32, X: i32, y: i32) -> u32 {
  return ROW_RAY_OFFSET + u32(N) * ((1u << u32(LAST_CASCADE + 1)) - (1u << u32(LAST_CASCADE + 1 - cascade)) + u32(X >> u32(cascade))) + u32(y);
}
// the faces of ray (cascade, X, y, rise) with the y faces of side ySide (1 ym, -1 yp, 0 none)
fn loadFaces(cascade: i32, X: i32, y: i32, rise: i32, ySide: i32) -> Faces {
  if (X >= N - 1 || y < 0 || y >= N) { return Faces(); }
  let i = FACE_OFFSET + 2u * rayIndex(cascade, X, y, rise); let a = rays[i];
  var f = Faces(a.xyz, vec3f(0.0));
  if (ySide == 0) { return f; }
  if (rise == 0 && ySide < 0) { f.y = rays[rowRayIndex(cascade, X, y)].xyz; } else { f.y = vec3f(a.w, rays[i + 1u].xy); }
  return f;
}
fn faceFollowedBy(A: vec3f, T: f32, B: vec3f) -> vec3f {
  return vec3f(A.x + T * B.x, 0.5 * (A.y + T * (B.x + B.y)), 0.25 * (A.z + T * (B.x + 2.0 * B.y + B.z)));
}
` : ''}

// pass 1: extension
@compute @workgroup_size(${WORKGROUP})
fn extendRays(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  // threads in (X, rise, y) order, so neighbouring rays share their halves in cache
  let cascade = params.cascade; let L = pow2(cascade); let rises = 2 * L + 1;
  let i = threadIndex(id, groups);
  if (i >= apexColumns(cascade) * rises * N) { return; }
  let y = i % N; let rise = (i / N) % rises - L; let X = (i / (N * rises)) * L;
  var ray = vec4f(0.0, 0.0, 0.0, 1.0);
${solidLights ? `  // faces: x, the side's y (ym for rise >= 0, yp for rise < 0), and yp of rise-0 rays
  var fx = vec3f(0.0); var fy = vec3f(0.0); var fyp = vec3f(0.0);
  if (cascade == 0) {
    let x = X + 1; let row = y + rise;
    if (x < N && row >= 0 && row < N) {
      let cell = scenePixelIndex(x, row); let a = transmittance[cell]; var e = ${coverageTest ? '0.0' : 'emission[cell]'};
      var T = 0.0;
      if (a > 0.0) {
        T = pow(a, pathLengthPerColumn(0, rise));
        if (e > 0.0 && T < 1.0) { e *= (1.0 - T) / -log(T); }
        // a free cell holds the light of the faces beside it; it does not dim that light, which enters at its boundary
${coverageTest ? `        let c = coverageFaces(cell); fx = vec3f(c.x); fy = vec3f(select(c.z, c.y, rise >= 0)); fyp = vec3f(c.z);`
               : `        fx = vec3f(lightAt(x + 1, row)); fy = vec3f(select(lightAt(x, row - 1), lightAt(x, row + 1), rise >= 0)); fyp = vec3f(lightAt(x, row - 1));`}
      } else { e = 0.0; }   // an opaque cell only blocks: a solid light's light is on its faces
      ray = vec4f(e, e, e, T);
    }
  } else {
    let a = splitRise(cascade, rise); let b = rise - a; let h = pow2(cascade - 1);
    let first = loadRay(cascade - 1, X, y, a); let second = loadRay(cascade - 1, X + h, y + a, b);
    let Ta = transmittanceForPath(first.w, cascade - 1, a, cascade, rise); let Tb = transmittanceForPath(second.w, cascade - 1, b, cascade, rise);
    ray = vec4f(faceFollowedBy(first.xyz, Ta, second.xyz), Ta * Tb);
    let ySide = select(-1, 1, rise >= 0);
    let firstFaces = loadFaces(cascade - 1, X, y, a, ySide); let secondFaces = loadFaces(cascade - 1, X + h, y + a, b, ySide);
    fx = faceFollowedBy(firstFaces.x, Ta, secondFaces.x); fy = faceFollowedBy(firstFaces.y, Ta, secondFaces.y);
    if (rise == 0) { fyp = faceFollowedBy(loadFaces(cascade - 1, X, y, 0, -1).y, Ta, loadFaces(cascade - 1, X + h, y, 0, -1).y); }
  }
  let r = rayIndex(cascade, X, y, rise); let f = FACE_OFFSET + 2u * r;
  rays[r] = ray; rays[f] = vec4f(fx, fy.x); rays[f + 1u] = vec4f(fy.yz, 0.0, 0.0);
  if (rise == 0) { rays[rowRayIndex(cascade, X, y)] = vec4f(fyp, 0.0); }
}
` : `  if (cascade == 0) {
    let x = X + 1; let row = y + rise;
    if (x < N && row >= 0 && row < N) {
      let cell = scenePixelIndex(x, row); let a = transmittance[cell]; var e = emission[cell];
      var T = 0.0;
      if (a > 0.0) { T = pow(a, pathLengthPerColumn(0, rise)); }   // a is the transmittance across one cell
${pointCells ? '' : `      if (e > 0.0 && T < 1.0) { e *= select(1.0, (1.0 - T) / -log(T), T > 0.0); }   // the cell's own emission, attenuated inside it; opaque: an opaque light's surface
`}      ray = vec4f(e, e, e, T);
    }
  } else {
    let a = splitRise(cascade, rise); let b = rise - a; let h = pow2(cascade - 1);
    let first = loadRay(cascade - 1, X, y, a); let second = loadRay(cascade - 1, X + h, y + a, b);
    let Ta = transmittanceForPath(first.w, cascade - 1, a, cascade, rise); let Tb = transmittanceForPath(second.w, cascade - 1, b, cascade, rise);
    ray = vec4f(first.x + Ta * second.x,
                0.5 * (first.y + Ta * (second.x + second.y)),
                0.25 * (first.z + Ta * (second.x + 2.0 * second.y + second.z)),
                Ta * Tb);
  }
  rays[rayIndex(cascade, X, y, rise)] = ray;
}
`}
// a ray read as a point source at its light's centroid: mu0 / D * (1 + var (1 + s^2) / D^2), D the distance from the sample
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

${solidLights ? `fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }
fn clogRatio(a: vec2f, b: vec2f) -> vec2f { return vec2f(0.5 * log(dot(a, a) / dot(b, b)), atan2(a.y * b.x - a.x * b.y, a.x * b.x + a.y * b.y)); }   // log(a / b)
// the row of ray (cascade, rise) at distance tb along it, interpolated between columns
fn rowOnRay(cascade: i32, rise: i32, tb: f32) -> f32 {
  let t = clamp(i32(floor(tb)), 1, pow2(cascade)); let f = tb - f32(t);
  let rows = vec2f(rayPathRows[rayPathIndex(cascade, rise, t)]);
  return rows.x * (1.0 - f) + rows.y * f;
}
// a ray's faces as the field G at one point (light = Re G). Exact for straight edges: a line crosses a column once, so its
// x faces are one unit face across the row; its y faces lie on one row as a run of k faces, k^2 = 12 var + 1. G of a face from
// a to b is W (log(b - p) - log(a - p)) / (b - a), W the face's normal turned inward (x: 1, ym: i, yp: -i).
fn readFacesAt(f: Faces, cascade: i32, rise: i32, back: f32, across: f32) -> vec2f {
${coverageTest ? '  return vec2f(f.x.x + select(0.0, f.y.x, side != 0), 0.0);   // coverage test: the light of the faces read, at no distance\n' : ''}  let Lf = f32(pow2(cascade)); var G = vec2f(0.0);
  if (f.x.x > NO_LIGHT) {
    let q = f.x.y / f.x.x; let tb = Lf * q;
    let z = vec2f(back + tb + 0.5, across + rowOnRay(cascade, rise, tb));   // the face's centre, from the point
    let variance = max(Lf * Lf * (f.x.z / f.x.x - q * q), 0.0);
    // one face, up to float rounding, which grows as tb^2; the two reads differ by O(var / distance^2) there
    if (cascade == 0 || variance < 1e-6 * max(1.0, tb * tb)) {
      let d = clogRatio(z + vec2f(0.0, 0.5), z - vec2f(0.0, 0.5)); G += f.x.x * vec2f(d.y, -d.x);
    } else {   // several x faces (not on one straight edge): a run along the ray's chord, plus the faces' own extent
      let h = 0.5 * sqrt(12.0 * variance) * vec2f(1.0, f32(rise) / Lf);
      G += f.x.x * (cdiv(clogRatio(z + h, z - h), 2.0 * h) - cdiv(vec2f(1.0 / 12.0, 0.0), cmul(z, cmul(z, z))));
    }
  }
  if (side != 0 && f.y.x > NO_LIGHT) {
    let q = f.y.y / f.y.x; let tb = Lf * q; let w = f32(side);
    let z = vec2f(back + tb, across + rowOnRay(cascade, rise, tb) + 0.5 * w);
    let k = select(sqrt(12.0 * max(Lf * Lf * (f.y.z / f.y.x - q * q), 0.0) + 1.0), 1.0, cascade == 0);
    let d = clogRatio(z + vec2f(0.5 * k, 0.0), z - vec2f(0.5 * k, 0.0)) / k;
    var g = w * vec2f(-d.y, d.x);
    if (cascade == 0 && w * z.y <= 0.0) { g.x = 0.0; }   // a single face seen from behind shows no angle
    G += f.y.x * g;
  }
  return G;
}
// a loaded ray (volume light) read with its faces
fn readLoaded(ray: vec4f, cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f, on: vec2<bool>) -> Light {
  var v = Light(readAsPoint(ray, cascade, rise, back, across, on), vec2f(0.0), vec2f(0.0));
  if (!any(on)) { return v; }
  let f = loadFaces(cascade, X, y, rise, side);
  if (f.x.x <= NO_LIGHT && f.y.x <= NO_LIGHT) { return v; }
  if (on.x) { v[1] = readFacesAt(f, cascade, rise, back.x, across.x); }
  if (on.y) { v[2] = select(readFacesAt(f, cascade, rise, back.y, across.y), v[1], on.x && back.x == back.y && across.x == across.y); }   // probes: one point twice
  return v;
}
fn firstPoint(v: Light) -> ChainLight { return vec3f(v[0].x, v[1]); }
fn chainPair(a: ChainLight, b: ChainLight) -> Light { return Light(vec2f(a.x, b.x), a.yz, b.yz); }
fn chainTotal(c: ChainLight) -> f32 { return c.x + c.y; }
fn lightTotal(v: Light) -> f32 { return v[0].x + v[1].x; }   // at the first point

` : `fn readLoaded(ray: vec4f, cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f, on: vec2<bool>) -> Light { return readAsPoint(ray, cascade, rise, back, across, on); }
fn firstPoint(v: Light) -> ChainLight { return v.x; }
fn chainPair(a: ChainLight, b: ChainLight) -> Light { return vec2f(a, b); }
fn chainTotal(c: ChainLight) -> f32 { return c; }
fn lightTotal(v: Light) -> f32 { return v.x; }

`}// ray (cascade, X, y, rise); samples closer than its length read its two halves. No sample is closer than half a ray, so one
// split is enough.
fn readRay(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> Light {
  if (X >= N - 1) { return Light(); }
  let close = vec2<bool>(cascade > 0) & (back + 1.0 < vec2f(f32(pow2(cascade))));
  var v = Light();
  if (!all(close)) { v = readLoaded(loadRay(cascade, X, y, rise), cascade, X, y, rise, back, across, !close); }
  if (any(close)) {
    let a = splitRise(cascade, rise); let h = pow2(cascade - 1); let first = loadRay(cascade - 1, X, y, a);
    v += readLoaded(first, cascade - 1, X, y, a, back, across, close);
    if (X + h < N - 1) {
      v += first.w * readLoaded(loadRay(cascade - 1, X + h, y + a, rise - a), cascade - 1, X + h, y + a, rise - a, back + f32(h), across + f32(a), close);
    }
  }
  return v;
}

// the prefix shared by rays r and r + 1 of cascade n, read as its pieces (none of which splits), each gating the next
fn readSharedPrefix(cascade: i32, X0: i32, y0: i32, rise: i32, back0: vec2f, across0: vec2f) -> Light {
  if (cascade == 0 || X0 >= N - 1) { return Light(); }
  var X = X0; var y = y0; var back = back0; var across = across0; var v = Light(); var T = 1.0;
  let base = sharedPrefixIndex(cascade, rise);
  let pieces = u32(sharedPrefixes[base].x);
  for (var i = 1u; i <= pieces; i += 1u) {
    let piece = sharedPrefixes[base + i]; let pieceCascade = piece.x; let pieceRise = piece.y;
    let ray = loadRay(pieceCascade, X, y, pieceRise);
    v += T * readLoaded(ray, pieceCascade, X, y, pieceRise, back, across, vec2<bool>(true));
    T *= ray.w;
    X += pow2(pieceCascade); y += pieceRise; back += f32(pow2(pieceCascade)); across += f32(pieceRise);
    if (X >= N - 1) { break; }
  }
  return v;
}

// the ray chain of cascade-n0 rays of rise r0 from (X, y) to the domain end, for one sample: the largest aligned rays with
// 2^k <= back + 1
fn readRayChain(cascade0: i32, X0: i32, y0: i32, rise0: i32, back0: f32, across0: f32) -> ChainLight {
  var X = X0; var y = y0; var back = back0; var across = across0; var v = ChainLight(); var T = 1.0;
  loop {
    if (X >= N - 1) { break; }
    let k = max(cascade0, min(min(i32(countTrailingZeros(u32(X))), i32(firstLeadingBit(u32(back + 1.0)))), LAST_CASCADE)); let rise = rise0 << u32(k - cascade0);
    let ray = loadRay(k, X, y, rise);
    v += T * firstPoint(readLoaded(ray, k, X, y, rise, vec2f(back), vec2f(across), vec2<bool>(true${solidLights ? ', false' : ''})));
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
fn readBeyondFromSamples(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> Light {
  if (X >= N - 1 || y < 0 || y >= N || cascade >= LAST_CASCADE) { return Light(); }
  let F = beyondSamples[${solidLights ? '3u * ' : ''}beyondSampleIndex(cascade, X, y, direction)];
  let at = sampleDistances(cascade); let s = axisSlope(cascade, direction); let x = (back + s * across) / (1.0 + s * s);
  let w0 = (x - at.y) / (at.x - at.y); let w1 = (x - at.x) / (at.y - at.x);
${solidLights ? `  var volume = w0 * F.x + w1 * F.y;
  if (min(F.x, F.y) > SAMPLE_RATIO_MIN * max(max(F.x, F.y), 1e-30)) { volume = 1.0 / (w0 / F.x + w1 / F.y); }
  // the face field: 1/G linear in the point's complex position p, the samples at P0, P1 on the axis (exact for one face
  // anywhere in the cone)
  let i = 3u * beyondSampleIndex(cascade, X, y, direction);
  let G0 = beyondSamples[i + 1u]; let G1 = beyondSamples[i + 2u];
  let P0 = -at.x * vec2f(1.0, s); let P1 = -at.y * vec2f(1.0, s);
  let g0 = faceFromSamples(G0, G1, P0, P1, -vec2f(back.x, across.x));
  if (back.x == back.y && across.x == across.y) { return Light(volume, g0, g0); }   // probes: one point twice
  return Light(volume, g0, faceFromSamples(G0, G1, P0, P1, -vec2f(back.y, across.y)));
}
fn faceFromSamples(G0: vec2f, G1: vec2f, P0: vec2f, P1: vec2f, p: vec2f) -> vec2f {
  let w0 = cdiv(p - P1, P0 - P1); let w1 = cdiv(p - P0, P1 - P0);
  let m0 = length(G0); let m1 = length(G1);
  if (min(m0, m1) > SAMPLE_RATIO_MIN * max(max(m0, m1), 1e-30)) { return cdiv(vec2f(1.0, 0.0), cdiv(w0, G0) + cdiv(w1, G1)); }
  return cmul(w0, G0) + cmul(w1, G1);
}` : `  if (min(F.x, F.y) > SAMPLE_RATIO_MIN * max(max(F.x, F.y), 1e-30)) { return 1.0 / (w0 / F.x + w1 / F.y); }
  return w0 * F.x + w1 * F.y;
}`}
// a child's beyond-value: from its samples, or at cascade 1 from pass 3. Cascade-1 cones are read only by probes, at
// distance 0, 1 or 2 (2 only with same apex + forward), with the same point in both lanes; pair 0 holds distances 0 and 1,
// pair 1 distance 2.
fn readChildBeyond(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> Light {
  if (cascade == 1) {
    if (X >= N - 1) { return Light(); }
    let k = i32(round(back.x));
    let F = cascade1Beyond[cascade1BeyondIndex(X, y, direction, k >> 1u)];
    return ${solidLights ? 'Light(vec2f(select(F.x, F.y, k == 1)), vec2f(0.0), vec2f(0.0))' : 'vec2f(select(F.x, F.y, k == 1))'};   // probes take the light only
  }
  return readBeyondFromSamples(cascade, X, y, direction, back, across);
}

// the cone's own interval: its two rays over its first 2^n columns, their shared prefix counted once
fn readOwnInterval(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> Light {
  return readRay(cascade, X, y, direction, back, across) + readRay(cascade, X, y, direction + 1, back, across) - readSharedPrefix(cascade, X, y, direction, back, across);
}
fn readChild(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> Light {
  if (X >= N - 1) { return Light(); }
  return readOwnInterval(cascade, X, y, direction, back, across) + readChildBeyond(cascade, X, y, direction, back, across);
}

// even apex column: the second half of ray r of cascade n + 1, gated by the first half; and its part of the shared prefix
// of r, r + 1
fn readSecondHalf(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> Light {
  let n = pow2(cascade); let a = splitRise(cascade + 1, rise); let T = loadRay(cascade, X, y, a).w;
  return T * readRay(cascade, X + n, y + a, rise - a, back + f32(n), across + f32(a));
}
fn readSecondHalfPrefix(cascade: i32, X: i32, y: i32, rise: i32, back: vec2f, across: vec2f) -> Light {
  let n = pow2(cascade); let a = splitRise(cascade + 1, rise);
  if (a != splitRise(cascade + 1, rise + 1)) { return Light(); }
  let T = loadRay(cascade, X, y, a).w;
  return T * readSharedPrefix(cascade, X + n, y + a, rise - a, back + f32(n), across + f32(a));
}

// a cone's beyond-value, exactly, from rays and its children's beyond-values
fn readBeyondExactly(cascade: i32, X: i32, y: i32, direction: i32, back: vec2f, across: vec2f) -> Light {
  let n = pow2(cascade);
  if (X + n >= N - 1 || cascade >= LAST_CASCADE) { return Light(); }
  var v = Light();
  if (((X >> u32(cascade)) & 1) == 0) {
    // even apex column: the second halves of rays A, B, C of cascade n + 1 (shared by both patterns)
    let riseA = 2 * direction; let riseB = riseA + 1; let riseC = riseA + 2;
    v += readSecondHalf(cascade, X, y, riseA, back, across) + readSecondHalf(cascade, X, y, riseB, back, across) + readSecondHalf(cascade, X, y, riseC, back, across);
    v -= readSecondHalfPrefix(cascade, X, y, riseA, back, across) + readSecondHalfPrefix(cascade, X, y, riseB, back, across);
    // same apex: the children at the same X
    v += select(1.0, 0.5, SAME_APEX_AND_FORWARD) * (readChildBeyond(cascade + 1, X, y, 2 * direction, back, across) + readChildBeyond(cascade + 1, X, y, 2 * direction + 1, back, across));
    if (!SAME_APEX_AND_FORWARD && X + 2 * n < N - 1) {   // same-apex pattern: both children contain ray B's continuation beyond X + 2s
      let T = loadRay(cascade + 1, X, y, riseB).w; let backB = back + f32(2 * n); let acrossB = across + f32(riseB);
      v -= T * chainPair(readRayChain(cascade + 1, X + 2 * n, y + riseB, riseB, backB.x, acrossB.x), readRayChain(cascade + 1, X + 2 * n, y + riseB, riseB, backB.y, acrossB.y));
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
${solidLights ? `  side = select(-1, 1, direction >= 0);   // the children of an upper cone are upper
  let v = readBeyondExactly(cascade, X, y, direction, back, axisSlope(cascade, direction) * back);
  let j = 3u * beyondSampleIndex(cascade, X, y, direction);
  beyondSamples[j] = v[0]; beyondSamples[j + 1u] = v[1]; beyondSamples[j + 2u] = v[2];` : `  beyondSamples[beyondSampleIndex(cascade, X, y, direction)] = readBeyondExactly(cascade, X, y, direction, back, axisSlope(cascade, direction) * back);`}
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
${solidLights ? `  side = select(-1, 1, direction >= 0);
  let v = readBeyondExactly(1, X, y, direction, back, back * e);
  cascade1Beyond[i] = v[0] + vec2f(v[1].x, v[2].x);   // probes read only the light, Re G` : `  cascade1Beyond[i] = readBeyondExactly(1, X, y, direction, back, back * e);`}
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
${solidLights ? `  // the seam in x faces only (both cones read them; each reads its own y faces), the diagonals with their cone's side
  side = 0; var v = -chainTotal(readRayChain(0, x, y, 0, 0.0, 0.0));
  if (params.subtractSharedDiagonals == 1) {
    side = 1; v -= chainTotal(readRayChain(0, x, y, 1, 0.0, 0.0));
    side = -1; v -= chainTotal(readRayChain(0, x, y, -1, 0.0, 0.0));
  }
  // the probe's own cell, never on a ray: each face it shares with a light subtends pi / 2 (once, in quadrant 0)
  if (params.quadrant == 0${coverageTest ? ' && false' : ''}) { v += 1.5707963267948966 * (lightAt(x - 1, y) + lightAt(x + 1, y) + lightAt(x, y - 1) + lightAt(x, y + 1)); }` : `  var v = -readRayChain(0, x, y, 0, 0.0, 0.0);
  if (params.subtractSharedDiagonals == 1) { v -= readRayChain(0, x, y, 1, 0.0, 0.0) + readRayChain(0, x, y, -1, 0.0, 0.0); }`}
  addFluence(x, y, v, params.quadrant == 0);
}
@compute @workgroup_size(${WORKGROUP})
fn gatherCones(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let i = threadIndex(id, groups);
  if (i >= N * N) { return; }
  let x = i / N; let y = i % N;
  var v = 0.0;
  for (var direction = -1; direction < 1; direction += 1) {
${solidLights ? '    side = select(-1, 1, direction >= 0);\n' : ''}    v += lightTotal(readOwnInterval(0, x, y, direction, vec2f(0.0), vec2f(0.0)) + readBeyondExactly(0, x, y, direction, vec2f(0.0), vec2f(0.0)));
  }
  addFluence(x, y, v, false);
}
`;
}

// tables, built once on the CPU
export function splitRuleOf(splitRule) {
  return (cascade, rise) => {
    if ((rise & 1) === 0) return rise >> 1;
    const absRise = Math.abs(rise);
    const away = splitRule === 'alternating' ? (cascade & 1) === 0 : (cascade & 1) === 0 ? ((absRise >> 2) & 1) === 1 : ((absRise >> 1) & 1) === 0;
    const a = (absRise >> 1) + (away ? 1 : 0);
    return rise > 0 ? a : -a;
  };
}
// rows of ray (cascade, rise) at columns 1 .. 2^n
export function rayPathRowsOf(split, cascade, rise) {
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

export class DigitalHRC {
  // splitRule: 'balanced' | 'alternating': how odd rises split into halves
  // mergePattern: 'sameApexAndForward' | 'sameApex': the merge at even apex columns
  // cellModel: 'square' | 'point' (transmittance per cell crossed)
  // solidLights: opaque emitting cells light through their faces (directional emission, exact face reads, complex cone
  //   samples); about 2.8x memory. Off, a solid light's cells emit as the cells they are, at no extra cost.
  constructor(device, N, { splitRule = 'balanced', mergePattern = 'sameApexAndForward', cellModel = 'square', solidLights = false, coverageTest = false } = {}) {
    const lastCascade = Math.max(1, Math.ceil(Math.log2(N)));
    Object.assign(this, { device, N, lastCascade, splitRule, mergePattern, solidLights });
    const apexColumns = (n) => ((N - 2) >> n) + 1;
    const storage = GPUBufferUsage.STORAGE, copyDst = GPUBufferUsage.COPY_DST;
    const buffer = (size, usage = storage) => device.createBuffer({ size: Math.max(size, 16), usage });
    const filled = (data) => { const b = buffer(data.byteLength, storage | copyDst); device.queue.writeBuffer(b, 0, data); return b; };

    // sizes (match the shader's index functions)
    this.extensionThreads = Array.from({ length: lastCascade + 1 }, (_, n) => apexColumns(n) * (2 * (1 << n) + 1) * N);
    const rayCount = N * ((lastCascade + 2) * 2 ** (lastCascade + 1) - 1);   // where cascade lastCascade + 1 would start
    const beyondSampleCount = Math.max(lastCascade - 2, 0) * 2 ** (lastCascade + 1) * N;
    this.cascade1Threads = (mergePattern === 'sameApexAndForward' ? 2 : 1) * apexColumns(1) * 4 * (N + 4);   // probe pairs per cascade-1 cone

    // coverageTest (solid lights): the emission input is 4 face lights per cell, read with no distance, to check which faces
    //   each probe counts
    if (!solidLights) coverageTest = false;
    this.emissionBuffer = buffer(N * N * 4 * (coverageTest ? 4 : 1), storage | copyDst);
    this.transmittanceBuffer = buffer(N * N * 4, storage | copyDst);
    const cascade1Count = 2 * apexColumns(1) * 4 * (N + 4);
    // solid lights: each ray, its faces (2 slots), then the row rays
    const rowRayCount = N * (2 ** (lastCascade + 1) - 1);
    this.rayBuffer = buffer(solidLights ? (3 * rayCount + rowRayCount) * 16 : rayCount * 16);
    this.beyondSampleBuffer = buffer(beyondSampleCount * 8 * (solidLights ? 3 : 1));
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

    this.module = device.createShaderModule({ code: shaderSource({ N, lastCascade, splitRule, mergePattern, pointCells: cellModel === 'point', rayCount, solidLights,
      coverageTest }) });
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
