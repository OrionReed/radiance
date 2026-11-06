// Polygonal HRC (WebGPU): a cone cascade with visibility from occluder silhouettes
//
// quadrant: the scene is solved four times, rotated by 90 degrees; in each, lines go to +x with slopes in [-1, 1]. Negative
//   slopes are stored as the mirror image (rows flipped) of positive ones.
// cone tile (cascade n, first column X, start row y, direction d >= 0), L = 2^n: the lines from the apex (X - L, y - d) with
//   slopes [d / L, (d + 1) / L], over columns X .. X + L. It holds up to three occluder silhouettes.
// occluder silhouette: directions of a tile blocked by opaque pixels, bounded by two whole pixels; from any viewpoint it
//   covers [lowest corner of the two, highest corner]. It carries its light: radiance x the share whose first hit is light.
//   Every silhouette spans a whole pixel, more than half of any tile's directions, so three always suffice.
// extension: a tile of length L from three of length L / 2, its first half and two second halves. Blocking is a union
//   light counts where nothing nearer has it. Joins are decided in exact integer arithmetic.
// cone (cascade n, apex, direction d), d in [-s, s), s = 2^n: valued at two samples on its axis, read by interpolating
//   1 / value. It merges cascade n + 1 by a same-apex merge or a re-rooted merge (1/2 backward + 1/2 forward), chosen by the
//   apex column; children are gated by their transmittance (unblocked share of their directions).
// gather: probes at pixel corners; the four quadrants accumulate into the (N + 1)^2 corner fluence.
//
// runs = false (no corner runs): a tile holds two numbers instead of silhouettes, its transmittance (unblocked share of its
//   directions) and its light (mean radiance over its directions, where the first hit is light). Extension: first half, then
//   the two second halves weighted by the share of the tile's lines crossing each one's start segment (2/3 and 1/3);
//   transmittances multiply, light counts behind the transmittance in front. The merge reads tiles the same way, with each
//   tile's light spread evenly over its directions and placed at the tile's centre.

const WORKGROUP = 64;
const PARAM_STRIDE = 256;
const SILHOUETTE_WORDS = 2;              // both silhouette pixels (16 bits each, offsets from the tile), light (f32 bits)
const TILE_WORDS = 3 * SILHOUETTE_WORDS;

const range = (n) => [...Array(n).keys()];
const unrolled = (n, f, separator = '\n') => range(n).map(f).join(separator);   // generated, unrolled code

function shaderSource({ N, lastCascade, cull, runs }) {
  return /* wgsl */ `
const N: i32 = ${N};
const LAST_CASCADE: i32 = ${lastCascade};       // tiles go up to length N = 2^LAST_CASCADE
const FIRST_APEX_COLUMN: i32 = -N;
const FIRST_APEX_ROW: i32 = -N - 4;
const APEX_ROWS: i32 = 3 * N + 8;
const LIGHT_CULL: bool = ${cull};
const EMPTY_SLOT: u32 = 0xffffffffu;
const SAMPLE_SPACING: f32 = ${(1 / (2 - 1 / Math.SQRT2) / Math.SQRT2).toPrecision(17)};   // samples at +- SAMPLE_SPACING s columns from the apex
const QUARTER_PI: f32 = 0.78539816339744831;
const SECOND_PIXEL_SHARE: f32 = ${(1 - Math.atan(0.5) / (Math.PI / 4)).toPrecision(17)};   // length-1 tile: share of its angle that meets pixel y + 1 first

struct PassParams { cascade: i32, quadrant: i32, lightCount: i32, unused: i32 }

@group(0) @binding(0) var<storage, read> emission: array<f32>;
@group(0) @binding(1) var<storage, read> transmittance: array<f32>;
@group(0) @binding(2) var<storage, read_write> coneTiles: array<u32>;
@group(0) @binding(3) var<storage, read> childValues: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> coneValues: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> fluence: array<f32>;
@group(0) @binding(6) var<uniform> params: PassParams;
@group(0) @binding(7) var<storage, read> lightList: array<vec2i>;
@group(0) @binding(8) var<storage, read_write> diagnostics: array<atomic<u32>>;   // [0] silhouette limit exceeded, [1] pixel out of packing range

fn threadIndex(id: vec3u, groups: vec3u) -> i32 { return i32(id.x + id.y * groups.x * ${WORKGROUP}u); }

// ------------------------------------------------------------------------------------------------ the scene in a quadrant
// pixel (x, y) of the current quadrant -> scene pixel index; quadrant k = np.rot90(scene, k)
fn scenePixelIndex(x: i32, y: i32) -> i32 {
  switch (params.quadrant) {
    case 1: { return y * N + (N - 1 - x); }
    case 2: { return (N - 1 - x) * N + (N - 1 - y); }
    case 3: { return (N - 1 - y) * N + x; }
    default: { return x * N + y; }
  }
}
fn sceneCornerIndex(x: i32, y: i32) -> i32 {
  switch (params.quadrant) {
    case 1: { return y * (N + 1) + (N - x); }
    case 2: { return (N - x) * (N + 1) + (N - y); }
    case 3: { return (N - y) * (N + 1) + x; }
    default: { return x * (N + 1) + y; }
  }
}
// scene pixel -> pixel of the current quadrant (for the light list)
fn quadrantPixel(p: vec2i) -> vec2i {
  switch (params.quadrant) {
    case 1: { return vec2i(N - 1 - p.y, p.x); }
    case 2: { return vec2i(N - 1 - p.x, N - 1 - p.y); }
    case 3: { return vec2i(p.y, N - 1 - p.x); }
    default: { return p; }
  }
}
// pixel of the quadrant, or of its mirror image (rows flipped) when mirrored == 1
fn isOpaque(mirrored: i32, x: i32, y: i32) -> bool {
  if (x < 0 || x >= N || y < 0 || y >= N) { return false; }
  return transmittance[scenePixelIndex(x, select(y, N - 1 - y, mirrored == 1))] <= 0.0;
}
fn emissionAt(mirrored: i32, x: i32, y: i32) -> f32 { return emission[scenePixelIndex(x, select(y, N - 1 - y, mirrored == 1))]; }

// ------------------------------------------------------------------------------------------------ exact slopes
// a corner is the integer vector c from the viewpoint (c.x > 0); its slope c.y / c.x
fn slopeLess(u: vec2i, v: vec2i) -> bool { return u.y * v.x < v.y * u.x; }
fn slopeEq(u: vec2i, v: vec2i) -> bool { return u.y * v.x == v.y * u.x; }
// slope(v) - slope(u), exact up to one rounding
fn slopeDiff(u: vec2i, v: vec2i) -> f32 { return f32(v.y * u.x - u.y * v.x) / f32(u.x * v.x); }
// the pixel's lowest / highest corner from a viewpoint; offset = pixel - viewpoint, offset.x >= 1
fn lowestCorner(offset: vec2i) -> vec2i { return select(vec2i(offset.x, offset.y), vec2i(offset.x + 1, offset.y), offset.y >= 0); }
fn highestCorner(offset: vec2i) -> vec2i { return select(vec2i(offset.x + 1, offset.y + 1), vec2i(offset.x, offset.y + 1), offset.y + 1 >= 0); }
// the tie-break between pixels: x, then y
fn pixelBefore(p: vec2i, q: vec2i) -> bool { return p.x < q.x || (p.x == q.x && p.y < q.y); }

// a silhouette pixel inside tile (X, y, d) of cascade n: column offset X..X + 1023 (10 bits) and row offset from the tile's
// lower edge at that column, y + floor(d dx / L), in [-32, 31] (6 bits); diagnostics[1] counts pixels out of range
fn packSilhouettePixel(p: vec2i, X: i32, y: i32, direction: i32, cascade: i32) -> u32 {
  let columnOffset = p.x - X; let rowOffset = p.y - y - ((direction * columnOffset) >> u32(cascade));
  if (columnOffset < 0 || columnOffset >= 1024 || rowOffset < -32 || rowOffset > 31) { atomicAdd(&diagnostics[1], 1u); }
  return u32(columnOffset & 1023) | (u32((rowOffset + 32) & 63) << 10u);
}
fn unpackSilhouettePixel(word: u32, X: i32, y: i32, direction: i32, cascade: i32) -> vec2i {
  let columnOffset = i32(word & 1023u);
  return vec2i(X + columnOffset, y + ((direction * columnOffset) >> u32(cascade)) + i32((word >> 10u) & 63u) - 32);
}
fn packSilhouette(lower: vec2i, upper: vec2i, X: i32, y: i32, direction: i32, cascade: i32) -> u32 {
  return packSilhouettePixel(lower, X, y, direction, cascade) | (packSilhouettePixel(upper, X, y, direction, cascade) << 16u);
}

// ------------------------------------------------------------------------------------------------ cone tile store
// [cascade][mirrored][X / L][direction][y + L + 1], y in [-(L + 1), N); cascade n holds 2 N (N + L + 1) tiles
fn cascadeOffset(cascade: i32) -> i32 { return 2 * N * (N + 1) * cascade + 2 * N * ((1 << u32(cascade)) - 1); }
fn coneTileIndex(mirrored: i32, cascade: i32, X: i32, y: i32, direction: i32) -> i32 {
  let L = 1 << u32(cascade);
  if (X < 0 || X + L > N || y < -(L + 1) || y >= N) { return -1; }
  return cascadeOffset(cascade) + ((mirrored * (N >> u32(cascade)) + (X >> u32(cascade))) * L + direction) * (N + L + 1) + y + L + 1;
}

// ------------------------------------------------------------------------------------------------ pass 1: extension
fn slopeMin(u: vec2i, v: vec2i) -> vec2i { return select(u, v, slopeLess(v, u)); }
fn slopeMax(u: vec2i, v: vec2i) -> vec2i { return select(u, v, slopeLess(u, v)); }
// From the apex, a pixel's lowest and highest corner are integer vectors that determine the pixel, so a silhouette's ends
// are kept as corner vectors. Ties between corners on one line from the apex go to the smaller pixel.
fn lowerSilhouettePixel(corner: vec2i, apex: vec2i) -> vec2i { return apex + corner - select(vec2i(0, 0), vec2i(1, 0), corner.y >= 0); }
fn upperSilhouettePixel(corner: vec2i, apex: vec2i) -> vec2i { return apex + corner - select(vec2i(1, 1), vec2i(0, 1), corner.y >= 0); }
fn lowerOf(u: vec2i, v: vec2i, apex: vec2i) -> vec2i {
  return select(u, v, slopeLess(v, u) || (slopeEq(u, v) && pixelBefore(lowerSilhouettePixel(v, apex), lowerSilhouettePixel(u, apex))));
}
fn upperOf(u: vec2i, v: vec2i, apex: vec2i) -> vec2i {
  return select(u, v, slopeLess(u, v) || (slopeEq(u, v) && pixelBefore(upperSilhouettePixel(v, apex), upperSilhouettePixel(u, apex))));
}

// Length 1: the column's two pixels y, y + 1, seen from (X - 1, y): pixel y spans slopes [0, 1], pixel y + 1 spans [1/2, 2]
// and counts only where pixel y doesn't. One silhouette; its light is the first pixel's.
@compute @workgroup_size(${WORKGROUP})
fn extendLength1(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let rows = N + 2;
  let index = threadIndex(id, groups);
  if (index >= 2 * N * rows) { return; }
  let y = index % rows - 2; let X = (index / rows) % N; let mirrored = index / (rows * N);
  let firstOpaque = isOpaque(mirrored, X, y); let secondOpaque = isOpaque(mirrored, X, y + 1);
  let out = u32(index) * ${TILE_WORDS}u;
  coneTiles[out] = select(EMPTY_SLOT, packSilhouette(vec2i(X, select(y + 1, y, firstOpaque)), vec2i(X, select(y, y + 1, secondOpaque)), X, y, 0, 0), firstOpaque || secondOpaque);
  coneTiles[out + 1u] = bitcast<u32>(select(select(0.0, emissionAt(mirrored, X, y + 1), secondOpaque), emissionAt(mirrored, X, y), firstOpaque));
${unrolled(2, (g) => { const o = (g + 1) * SILHOUETTE_WORDS; return `  coneTiles[out + ${o}u] = EMPTY_SLOT; coneTiles[out + ${o + 1}u] = 0u;`; })}
}

// Length >= 2. The code is unrolled when generated, so every per-thread array is indexed by constants and stays in
// registers. Candidate silhouettes: slots 0-2 from the first half, 3-5 and 6-8 from the two second halves.
@compute @workgroup_size(${WORKGROUP})
fn extendConeTiles(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let cascade = params.cascade; let L = 1 << u32(cascade); let rows = N + L + 1;
  let index = threadIndex(id, groups);
  if (index >= 2 * N * rows) { return; }
  let rowIndex = index % rows; let rest = index / rows; let direction = rest % L; let rest2 = rest / L;
  let columnIndex = rest2 % (N >> u32(cascade)); let mirrored = rest2 / (N >> u32(cascade));
  let X = columnIndex * L; let y = rowIndex - L - 1;
  let apex = vec2i(X - L, y - direction);
  let tileLowerEdge = vec2i(L, direction); let tileUpperEdge = vec2i(L, direction + 1);
  // each candidate: its lower and upper edge from the apex, its light, whether it meets the tile
  var lowerEdge: array<vec2i, 9>; var upperEdge: array<vec2i, 9>; var light: array<f32, 9>; var meetsTile: array<bool, 9>;
  {
    let halfLength = L >> 1u; let halfDirection = direction >> 1u;
${unrolled(3, (c) => { const at = ['X, y', 'X + halfLength, y + halfDirection', 'X + halfLength, y + halfDirection + 1'][c]; return `    { let slot = coneTileIndex(mirrored, cascade - 1, ${at}, halfDirection);
      if (slot >= 0) {
        let base = u32(slot) * ${TILE_WORDS}u;
${unrolled(3, (j) => { const k = 3 * c + j, o = j * SILHOUETTE_WORDS; return `        { let word = coneTiles[base + ${o}u];
          if (word != EMPTY_SLOT) {
            let lower = unpackSilhouettePixel(word & 0xffffu, ${at}, halfDirection, cascade - 1); let upper = unpackSilhouettePixel(word >> 16u, ${at}, halfDirection, cascade - 1);
            lowerEdge[${k}] = lowerOf(lowestCorner(lower - apex), lowestCorner(upper - apex), apex); upperEdge[${k}] = upperOf(highestCorner(lower - apex), highestCorner(upper - apex), apex);
            meetsTile[${k}] = slopeLess(tileLowerEdge, upperEdge[${k}]) && slopeLess(lowerEdge[${k}], tileUpperEdge);
            light[${k}] = bitcast<f32>(coneTiles[base + ${o + 1}u]);
          } }`; })}
      } }`; })}
  }
  // light measures (radiance x visible slope length inside the tile): first-half silhouettes by their clipped length;
  // second-half ones only in the gaps between earlier silhouettes. Each gap is an exact slope difference.
  var lightMeasure: array<f32, 9>;
${unrolled(3, (k) => `  if (meetsTile[${k}]) { lightMeasure[${k}] = light[${k}] * slopeDiff(slopeMax(lowerEdge[${k}], tileLowerEdge), slopeMin(upperEdge[${k}], tileUpperEdge)); }`)}
${unrolled(6, (kk) => { const k = kk + 3, earlier = range(k); return `  if (meetsTile[${k}] && light[${k}] > 0.0) {
    let clipLower = slopeMax(lowerEdge[${k}], tileLowerEdge); let clipUpper = slopeMin(upperEdge[${k}], tileUpperEdge);
${earlier.map((j) => `    let pieceLower${j} = slopeMax(lowerEdge[${j}], clipLower); let pieceUpper${j} = slopeMin(upperEdge[${j}], clipUpper); let pieceExists${j} = meetsTile[${j}] && slopeLess(pieceLower${j}, pieceUpper${j});`).join('\n')}
    var visibleLength = 0.0; var furthestEnd = clipLower; var coveredUpTo: vec2i;
${earlier.map((j) => `    if (pieceExists${j}) {
      coveredUpTo = clipLower;${earlier.filter((i) => i !== j).map((i) => ` if (pieceExists${i} && ${i < j ? `!slopeLess(pieceLower${j}, pieceLower${i})` : `slopeLess(pieceLower${i}, pieceLower${j})`}) { coveredUpTo = slopeMax(coveredUpTo, pieceUpper${i}); }`).join('')}
      if (slopeLess(coveredUpTo, pieceLower${j})) { visibleLength += slopeDiff(coveredUpTo, pieceLower${j}); }
      furthestEnd = slopeMax(furthestEnd, pieceUpper${j});
    }`).join('\n')}
    if (slopeLess(furthestEnd, clipUpper)) { visibleLength += slopeDiff(furthestEnd, clipUpper); }
    lightMeasure[${k}] = light[${k}] * visibleLength;
  }`; })}
  // join, as a sweep by lower edge (ties in slot order): a candidate starts a join when every candidate before it ends below
  // its lower edge (touching joins); a candidate's join = the number of starts up to it, minus one
  var startsJoin = 0u;
${unrolled(9, (k) => `  if (meetsTile[${k}]${range(9).filter((j) => j !== k).map((j) => ` && !(meetsTile[${j}] && ${j < k ? `!slopeLess(lowerEdge[${k}], lowerEdge[${j}])` : `slopeLess(lowerEdge[${j}], lowerEdge[${k}])`} && !slopeLess(upperEdge[${j}], lowerEdge[${k}]))`).join('')}) { startsJoin |= ${1 << k}u; }`)}
  var joinUsed: array<bool, 3>; var joinLower: array<vec2i, 3>; var joinUpper: array<vec2i, 3>; var joinLightMeasure: array<f32, 3>;
${unrolled(9, (k) => `  if (meetsTile[${k}]) {
    let joinIndex = select(-1, 0, (startsJoin & ${1 << k}u) != 0u)${range(9).filter((j) => j !== k).map((j) => ` + select(0, 1, (startsJoin & ${1 << j}u) != 0u && ${j < k ? `!slopeLess(lowerEdge[${k}], lowerEdge[${j}])` : `slopeLess(lowerEdge[${j}], lowerEdge[${k}])`})`).join('')};
${unrolled(3, (g) => `    ${g ? 'else ' : ''}if (joinIndex == ${g}) {
      if (!joinUsed[${g}]) { joinUsed[${g}] = true; joinLower[${g}] = lowerEdge[${k}]; joinUpper[${g}] = upperEdge[${k}]; }
      joinLower[${g}] = lowerOf(joinLower[${g}], lowerEdge[${k}], apex); joinUpper[${g}] = upperOf(joinUpper[${g}], upperEdge[${k}], apex); joinLightMeasure[${g}] += lightMeasure[${k}];
    }`)}
    else { atomicAdd(&diagnostics[0], 1u); }   // a fourth silhouette cannot happen; counted as a check
  }`)}
  let out = u32(cascadeOffset(cascade) + index) * ${TILE_WORDS}u;
${unrolled(3, (g) => { const o = g * SILHOUETTE_WORDS; return `  if (joinUsed[${g}]) {
    let joinWidth = slopeDiff(slopeMax(joinLower[${g}], tileLowerEdge), slopeMin(joinUpper[${g}], tileUpperEdge));
    coneTiles[out + ${o}u] = packSilhouette(lowerSilhouettePixel(joinLower[${g}], apex), upperSilhouettePixel(joinUpper[${g}], apex), X, y, direction, cascade);
    coneTiles[out + ${o + 1}u] = bitcast<u32>(select(0.0, joinLightMeasure[${g}] / joinWidth, joinWidth > 0.0));
  } else { coneTiles[out + ${o}u] = EMPTY_SLOT; coneTiles[out + ${o + 1}u] = 0u; }`; })}
}

// ------------------------------------------------------------------------------------------------ reading tiles from an apex
fn atanNearZero(z: f32) -> f32 {   // |z| <= tan(pi / 8)
  let zz = z * z;
  return (((8.05374449538e-2 * zz - 1.38776856032e-1) * zz + 1.99777106478e-1) * zz - 3.33329491539e-1) * zz * z + z;
}
fn atan2Accurate(y: f32, x: f32) -> f32 {
  let ax = abs(x); let ay = abs(y); let largest = max(ax, ay);
  if (largest == 0.0) { return 0.0; }
  let t = min(ax, ay) / largest;
  var r = select(atanNearZero(t), QUARTER_PI + atanNearZero((t - 1.0) / (t + 1.0)), t > 0.41421356);
  if (ay > ax) { r = 2.0 * QUARTER_PI - r; }
  if (x < 0.0) { r = 4.0 * QUARTER_PI - r; }
  return select(r, -r, y < 0.0);
}
// angle of the integer vector c (from the apex), measured from the axis
fn angleFromAxis(c: vec2i, axis: vec2i) -> f32 { return atan2Accurate(f32(axis.x * c.y - axis.y * c.x), f32(axis.x * c.x + axis.y * c.y)); }
// a silhouette's angles from the apex (from the axis): the lowest and highest corner of its two pixels
fn silhouetteAngles(lower: vec2i, upper: vec2i, apex: vec2i, axis: vec2i) -> vec2f {
  return vec2f(angleFromAxis(slopeMin(lowestCorner(lower - apex), lowestCorner(upper - apex)), axis),
               angleFromAxis(slopeMax(highestCorner(lower - apex), highestCorner(upper - apex)), axis));
}

// a tile's silhouettes as seen from an apex: their angles (an empty slot: an empty interval far away), light, and the centre
// of their two pixels. lightOnly: angles only for lit silhouettes (interval light reads).
const NO_SILHOUETTE = vec2f(3e38, 3e38);
struct SilhouettesFromApex { angles: array<vec2f, 3>, light: array<f32, 3>, centre: array<vec2f, 3> }
fn silhouettesFromApex(cascade: i32, X: i32, y: i32, direction: i32, apex: vec2i, axis: vec2i, lightOnly: bool) -> SilhouettesFromApex {
  var r: SilhouettesFromApex;
${unrolled(3, (j) => `  r.angles[${j}] = NO_SILHOUETTE;`)}
  let mirrored = direction < 0;   // negative directions read the mirror image
  let yStored = select(y, N - 1 - y, mirrored); let directionStored = select(direction, -direction - 1, mirrored);
  let slot = coneTileIndex(select(0, 1, mirrored), cascade, X, yStored, directionStored);
  if (slot < 0) { return r; }
  let base = u32(slot) * ${TILE_WORDS}u;
${unrolled(3, (j) => { const o = j * SILHOUETTE_WORDS; return `  { let word = coneTiles[base + ${o}u]; let l = bitcast<f32>(coneTiles[base + ${o + 1}u]);
    if (word != EMPTY_SLOT && (!lightOnly || l > 0.0)) {
      var lower = unpackSilhouettePixel(word & 0xffffu, X, yStored, directionStored, cascade); var upper = unpackSilhouettePixel(word >> 16u, X, yStored, directionStored, cascade);
      if (mirrored) { lower.y = N - 1 - lower.y; upper.y = N - 1 - upper.y; }
      r.angles[${j}] = silhouetteAngles(lower, upper, apex, axis); r.light[${j}] = l; r.centre[${j}] = vec2f(lower + upper) * 0.5 + 0.5;
    } }`; })}
  return r;
}
${[3, 6, 9].map((n) => `
// measure of the union of ${n} angle intervals clipped to [lowerAngle, upperAngle], without sorting: each interval counts beyond the furthest
// end of the intervals that start before it (ties by index)
fn unionLength${n}(intervals: array<vec2f, ${n}>, lowerAngle: f32, upperAngle: f32) -> f32 {
${unrolled(n, (k) => `  let start${k} = max(intervals[${k}].x, lowerAngle); let end${k} = min(intervals[${k}].y, upperAngle);`)}
  var total = 0.0; var coveredUpTo: f32;
${unrolled(n, (k) => `  coveredUpTo = start${k};${range(n).filter((j) => j !== k).map((j) => ` if (start${j} <${j < k ? '=' : ''} start${k}) { coveredUpTo = max(coveredUpTo, end${j}); }`).join('')} total += max(0.0, end${k} - coveredUpTo);`)}
  return total;
}`).join('\n')}
${[0, 3, 6].map((n) => `
// light of a tile's silhouettes inside [lowerAngle, upperAngle]${n ? `, minus what the ${n} silhouettes in front hide` : ''}, from the apex,
// rescaled to the two samples by distance to the silhouette's pixels
fn lightBehind${n}(r: SilhouettesFromApex${n ? `, inFront: array<vec2f, ${n}>` : ''}, lowerAngle: f32, upperAngle: f32, apex: vec2i, forwardSample: vec2f, backwardSample: vec2f) -> vec2f {
  var out = vec2f(0.0); let apexPoint = vec2f(apex);
${unrolled(3, (j) => `  if (r.light[${j}] > 0.0 && r.angles[${j}].y > lowerAngle && r.angles[${j}].x < upperAngle) {
    let clipFrom = max(r.angles[${j}].x, lowerAngle); let clipTo = min(r.angles[${j}].y, upperAngle);
    let distanceFromApex = length(r.centre[${j}] - apexPoint);
    out += r.light[${j}] * ((clipTo - clipFrom)${n ? ` - unionLength${n}(inFront, clipFrom, clipTo)` : ''}) * vec2f(distanceFromApex / length(r.centre[${j}] - forwardSample), distanceFromApex / length(r.centre[${j}] - backwardSample));
  }`)}
  return out;
}`).join('\n')}

// ------------------------------------------------------------------------------------------------ cone values
fn coneValueIndex(cascade: i32, apex: vec2i, direction: i32) -> i32 {
  let s = 1 << u32(cascade); let apexColumns = (2 * N) / s + 1;
  let columnIndex = (apex.x - FIRST_APEX_COLUMN) / s; let rowIndex = apex.y - FIRST_APEX_ROW;
  if (apex.x < FIRST_APEX_COLUMN || columnIndex >= apexColumns || rowIndex < 0 || rowIndex >= APEX_ROWS) { return -1; }
  return (columnIndex * APEX_ROWS + rowIndex) * (2 * s) + (direction + s);
}
// falloff interpolation between the two samples: 1 / value, linear in distance
fn falloffInterpolate(forwardValue: f32, backwardValue: f32, w: f32) -> f32 {
  if (min(forwardValue, backwardValue) > 1e-12 * max(max(forwardValue, backwardValue), 1e-30)) { return 1.0 / (w / forwardValue + (1.0 - w) / backwardValue); }
  return w * forwardValue + (1.0 - w) * backwardValue;
}
// a child cone's value at a point: the point projected onto the child's axis, between its two samples
fn readChildAtPoint(samples: vec2f, childApex: vec2i, childScale: i32, childDirection: i32, point: vec2f) -> f32 {
  let axisX = f32(childScale); let axisY = f32(childDirection) + 0.5; let offset = point - vec2f(childApex);
  let along = (offset.x * axisX + offset.y * axisY) / (axisX * axisX + axisY * axisY) * axisX; let spacing = SAMPLE_SPACING * f32(childScale);
  return falloffInterpolate(samples.x, samples.y, (along + spacing) / (2.0 * spacing));
}
fn readChild(childCascade: i32, childApex: vec2i, childDirection: i32, forwardSample: vec2f, backwardSample: vec2f) -> vec2f {
  let index = coneValueIndex(childCascade, childApex, childDirection);
  if (index < 0) { return vec2f(0.0); }
  let samples = childValues[index];
  if (samples.x == 0.0 && samples.y == 0.0) { return vec2f(0.0); }
  let childScale = 1 << u32(childCascade);
  return vec2f(readChildAtPoint(samples, childApex, childScale, childDirection, forwardSample),
               readChildAtPoint(samples, childApex, childScale, childDirection, backwardSample));
}

// the cone's value at two points (its two samples, or the probe itself at cascade 0)
fn coneValue(cascade: i32, isTopCascade: bool, apex: vec2i, direction: i32, forwardSample: vec2f, backwardSample: vec2f) -> vec2f {
  let s = 1 << u32(cascade); let axis = vec2i(2 * s, 2 * direction + 1);
  var out = vec2f(0.0);
  let ownTile = silhouettesFromApex(cascade, apex.x + s, apex.y + direction, direction, apex, axis, false);
  if (((apex.x >> u32(cascade)) & 1) == 0) {
    // same-apex merge
    for (var whichHalf = 0; whichHalf < 2; whichHalf += 1) {
      let childDirection = 2 * direction + whichHalf;
      let lowerAngle = angleFromAxis(vec2i(2 * s, childDirection), axis); let upperAngle = angleFromAxis(vec2i(2 * s, childDirection + 1), axis);
      let childTransmittance = 1.0 - unionLength3(ownTile.angles, lowerAngle, upperAngle) / (upperAngle - lowerAngle);
      if (childTransmittance > 0.0 && !isTopCascade) { out += childTransmittance * readChild(cascade + 1, apex, childDirection, forwardSample, backwardSample); }
      let childTile = silhouettesFromApex(cascade + 1, apex.x + 2 * s, apex.y + childDirection, childDirection, apex, axis, true);
      out += lightBehind3(childTile, ownTile.angles, lowerAngle, upperAngle, apex, forwardSample, backwardSample);
    }
  } else {
    // re-rooted merge: the next interval [2s, 3s) is covered by two half-length tiles
    let nextLower = silhouettesFromApex(cascade, apex.x + 2 * s, apex.y + 2 * direction, direction, apex, axis, false);
    let nextUpper = silhouettesFromApex(cascade, apex.x + 2 * s, apex.y + 2 * direction + 1, direction, apex, axis, false);
    let ownAndNextLower = array<vec2f, 6>(${unrolled(3, (j) => `ownTile.angles[${j}]`, ', ')}, ${unrolled(3, (j) => `nextLower.angles[${j}]`, ', ')});
    let ownAndNext = array<vec2f, 9>(${unrolled(3, (j) => `ownTile.angles[${j}]`, ', ')}, ${unrolled(3, (j) => `nextLower.angles[${j}]`, ', ')}, ${unrolled(3, (j) => `nextUpper.angles[${j}]`, ', ')});
    for (var whichHalf = 0; whichHalf < 2; whichHalf += 1) {
      let childDirection = 2 * direction + whichHalf;
      // backward: re-rooted one step back along the cone's edge; its own tiles gate it
      if (!isTopCascade) { out += 0.5 * readChild(cascade + 1, vec2i(apex.x - s, apex.y - direction - whichHalf), childDirection, forwardSample, backwardSample); }
      // forward: re-rooted where the cone's edge leaves the next interval, through it
      let lowerAngle = angleFromAxis(vec2i(3 * s, 3 * direction + 2 * whichHalf), axis); let upperAngle = angleFromAxis(vec2i(3 * s, 3 * direction + 2 * whichHalf + 1), axis);
      let childTransmittance = 1.0 - unionLength9(ownAndNext, lowerAngle, upperAngle) / (upperAngle - lowerAngle);
      if (childTransmittance <= 0.0) { continue; }
      let forwardApex = vec2i(apex.x + s, apex.y + direction + whichHalf); let forwardAxis = vec2i(4 * s, 2 * childDirection + 1);
      if (!isTopCascade) { out += 0.5 * childTransmittance * readChild(cascade + 1, forwardApex, childDirection, forwardSample, backwardSample); }
      let forwardChildTile = silhouettesFromApex(cascade + 1, apex.x + 3 * s, apex.y + 3 * direction + 2 * whichHalf, childDirection, forwardApex, forwardAxis, true);
      out += 0.5 * childTransmittance * lightBehind0(forwardChildTile, angleFromAxis(vec2i(2 * s, childDirection), forwardAxis),
                                                     angleFromAxis(vec2i(2 * s, childDirection + 1), forwardAxis), forwardApex, forwardSample, backwardSample);
    }
    let lowerAngle = angleFromAxis(vec2i(s, direction), axis); let upperAngle = angleFromAxis(vec2i(s, direction + 1), axis);
    out += lightBehind3(nextLower, ownTile.angles, lowerAngle, upperAngle, apex, forwardSample, backwardSample)
         + lightBehind6(nextUpper, ownAndNextLower, lowerAngle, upperAngle, apex, forwardSample, backwardSample);
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ without corner runs
// a tile is (transmittance, light) in its first two words
fn storedTileValue(slot: i32) -> vec2f {
  if (slot < 0) { return vec2f(1.0, 0.0); }
  let base = u32(slot) * ${TILE_WORDS}u;
  return vec2f(bitcast<f32>(coneTiles[base]), bitcast<f32>(coneTiles[base + 1u]));
}
fn storeTileValue(slot: u32, value: vec2f) {
  let base = slot * ${TILE_WORDS}u;
  coneTiles[base] = bitcast<u32>(value.x); coneTiles[base + 1u] = bitcast<u32>(value.y);
}

// length 1: pixel y meets every line first, pixel y + 1 the share SECOND_PIXEL_SHARE (slopes [1/2, 1])
@compute @workgroup_size(${WORKGROUP})
fn extendLength1Values(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let rows = N + 2;
  let index = threadIndex(id, groups);
  if (index >= 2 * N * rows) { return; }
  let y = index % rows - 2; let X = (index / rows) % N; let mirrored = index / (rows * N);
  var value = vec2f(1.0, 0.0);
  if (isOpaque(mirrored, X, y)) { value = vec2f(0.0, emissionAt(mirrored, X, y)); }
  else if (isOpaque(mirrored, X, y + 1)) { value = vec2f(1.0 - SECOND_PIXEL_SHARE, SECOND_PIXEL_SHARE * emissionAt(mirrored, X, y + 1)); }
  storeTileValue(u32(index), value);
}

// length >= 2: the first half, then the second halves; at column X + L / 2 the tile's lines cross rows y + d / 2 + [0, 3/2],
// so the lower second half (rows y + floor(d / 2) + [0, 1]) takes 2/3 of them for even d, 1/3 for odd d
@compute @workgroup_size(${WORKGROUP})
fn extendConeTilesValues(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let cascade = params.cascade; let L = 1 << u32(cascade); let rows = N + L + 1;
  let index = threadIndex(id, groups);
  if (index >= 2 * N * rows) { return; }
  let rowIndex = index % rows; let rest = index / rows; let direction = rest % L; let rest2 = rest / L;
  let columnIndex = rest2 % (N >> u32(cascade)); let mirrored = rest2 / (N >> u32(cascade));
  let X = columnIndex * L; let y = rowIndex - L - 1;
  let halfLength = L >> 1u; let halfDirection = direction >> 1u;
  let first = storedTileValue(coneTileIndex(mirrored, cascade - 1, X, y, halfDirection));
  let secondLower = storedTileValue(coneTileIndex(mirrored, cascade - 1, X + halfLength, y + halfDirection, halfDirection));
  let secondUpper = storedTileValue(coneTileIndex(mirrored, cascade - 1, X + halfLength, y + halfDirection + 1, halfDirection));
  let lowerShare = select(2.0 / 3.0, 1.0 / 3.0, (direction & 1) == 1);
  let second = lowerShare * secondLower + (1.0 - lowerShare) * secondUpper;
  storeTileValue(u32(cascadeOffset(cascade) + index), vec2f(first.x * second.x, first.y + first.x * second.y));
}

// tile (X, y, direction) of the quadrant (negative directions read the mirror image), and its centre: 3L / 2 along its axis
fn tileValue(cascade: i32, X: i32, y: i32, direction: i32) -> vec2f {
  let mirrored = direction < 0;
  return storedTileValue(coneTileIndex(select(0, 1, mirrored), cascade, X, select(y, N - 1 - y, mirrored), select(direction, -direction - 1, mirrored)));
}
fn tileCentre(cascade: i32, X: i32, y: i32, direction: i32) -> vec2f {
  let L = f32(1 << u32(cascade));
  return vec2f(f32(X) + 0.5 * L, f32(y - direction) + 1.5 * (f32(direction) + 0.5));
}
// a tile's light over [lowerAngle, upperAngle] from the apex, behind transmittance inFront, rescaled to the two samples by
// distance to the tile's centre
fn tileLight(cascade: i32, X: i32, y: i32, direction: i32, lowerAngle: f32, upperAngle: f32, inFront: f32, apex: vec2i, forwardSample: vec2f, backwardSample: vec2f) -> vec2f {
  let value = tileValue(cascade, X, y, direction);
  if (value.y <= 0.0 || inFront <= 0.0) { return vec2f(0.0); }
  let centre = tileCentre(cascade, X, y, direction); let distanceFromApex = length(centre - vec2f(apex));
  return value.y * (upperAngle - lowerAngle) * inFront * vec2f(distanceFromApex / length(centre - forwardSample), distanceFromApex / length(centre - backwardSample));
}

// coneValue without corner runs: the same merges, every gate a product of tile transmittances
fn coneValueValues(cascade: i32, isTopCascade: bool, apex: vec2i, direction: i32, forwardSample: vec2f, backwardSample: vec2f) -> vec2f {
  let s = 1 << u32(cascade); let axis = vec2i(2 * s, 2 * direction + 1);
  var out = vec2f(0.0);
  let own = tileValue(cascade, apex.x + s, apex.y + direction, direction).x;
  if (((apex.x >> u32(cascade)) & 1) == 0) {
    // same-apex merge
    for (var whichHalf = 0; whichHalf < 2; whichHalf += 1) {
      let childDirection = 2 * direction + whichHalf;
      let lowerAngle = angleFromAxis(vec2i(2 * s, childDirection), axis); let upperAngle = angleFromAxis(vec2i(2 * s, childDirection + 1), axis);
      if (own > 0.0 && !isTopCascade) { out += own * readChild(cascade + 1, apex, childDirection, forwardSample, backwardSample); }
      out += tileLight(cascade + 1, apex.x + 2 * s, apex.y + childDirection, childDirection, lowerAngle, upperAngle, own, apex, forwardSample, backwardSample);
    }
  } else {
    // re-rooted merge: the next interval [2s, 3s) is covered by two half-length tiles; each forward half crosses only one
    let nextLower = tileValue(cascade, apex.x + 2 * s, apex.y + 2 * direction, direction).x;
    let nextUpper = tileValue(cascade, apex.x + 2 * s, apex.y + 2 * direction + 1, direction).x;
    for (var whichHalf = 0; whichHalf < 2; whichHalf += 1) {
      let childDirection = 2 * direction + whichHalf;
      if (!isTopCascade) { out += 0.5 * readChild(cascade + 1, vec2i(apex.x - s, apex.y - direction - whichHalf), childDirection, forwardSample, backwardSample); }
      let childTransmittance = own * select(nextLower, nextUpper, whichHalf == 1);
      if (childTransmittance <= 0.0) { continue; }
      let forwardApex = vec2i(apex.x + s, apex.y + direction + whichHalf); let forwardAxis = vec2i(4 * s, 2 * childDirection + 1);
      if (!isTopCascade) { out += 0.5 * childTransmittance * readChild(cascade + 1, forwardApex, childDirection, forwardSample, backwardSample); }
      out += 0.5 * tileLight(cascade + 1, apex.x + 3 * s, apex.y + 3 * direction + 2 * whichHalf, childDirection, angleFromAxis(vec2i(2 * s, childDirection), forwardAxis),
                             angleFromAxis(vec2i(2 * s, childDirection + 1), forwardAxis), childTransmittance, forwardApex, forwardSample, backwardSample);
    }
    let lowerAngle = angleFromAxis(vec2i(s, direction), axis); let middleAngle = angleFromAxis(vec2i(2 * s, 2 * direction + 1), axis);
    let upperAngle = angleFromAxis(vec2i(s, direction + 1), axis);
    out += tileLight(cascade, apex.x + 2 * s, apex.y + 2 * direction, direction, lowerAngle, middleAngle, own, apex, forwardSample, backwardSample)
         + tileLight(cascade, apex.x + 2 * s, apex.y + 2 * direction + 1, direction, middleAngle, upperAngle, own, apex, forwardSample, backwardSample);
  }
  return out;
}

// no pixel centre of the domain in the cone beyond 2s - 2 columns, padded by 2.5 rows (conservative)
fn seesNoPixel(s: i32, apex: vec2i, direction: i32) -> bool {
  var nearest = max(f32(2 * s) - 2.0, 0.5 - f32(apex.x)); var farthest = f32(N) - 0.5 - f32(apex.x);
  let lowerSlope = f32(direction) / f32(s); let upperSlope = f32(direction + 1) / f32(s);
  let rowLimitBelow = f32(N + 2 - apex.y); let rowLimitAbove = f32(-2 - apex.y);   // lowerSlope x <= rowLimitBelow and upperSlope x >= rowLimitAbove
  if (lowerSlope > 0.0) { farthest = min(farthest, rowLimitBelow / lowerSlope); } else if (lowerSlope < 0.0) { nearest = max(nearest, rowLimitBelow / lowerSlope); } else if (rowLimitBelow < 0.0) { return true; }
  if (upperSlope > 0.0) { nearest = max(nearest, rowLimitAbove / upperSlope); } else if (upperSlope < 0.0) { farthest = min(farthest, rowLimitAbove / upperSlope); } else if (rowLimitAbove > 0.0) { return true; }
  return nearest > farthest + 1e-3;
}

// no light pixel centre in the cone beyond 2s - 2 columns, padded by 2.5 rows
fn lightCull(s: i32, apex: vec2i, direction: i32) -> bool {
  for (var n = 0; n < params.lightCount; n += 1) {
    let p = quadrantPixel(lightList[n]);
    let cx = f32(p.x) + 0.5 - f32(apex.x); let cy = f32(p.y) + 0.5 - f32(apex.y);
    if (cx > f32(2 * s) - 2.0 && cy >= (f32(direction) / f32(s)) * cx - 2.5 && cy <= (f32(direction + 1) / f32(s)) * cx + 2.5) { return false; }
  }
  return true;
}

// pass 2: the cones of one cascade, top-down; coneValues[(apex column index APEX_ROWS + apex row index) 2s + direction + s]
// = the values at the two samples
@compute @workgroup_size(${WORKGROUP})
fn mergeCascade(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let cascade = params.cascade; let s = 1 << u32(cascade); let apexColumns = (2 * N) / s + 1;
  let index = threadIndex(id, groups);
  if (index >= apexColumns * APEX_ROWS * 2 * s) { return; }
  let direction = index % (2 * s) - s; let rest = index / (2 * s);
  let apex = vec2i(FIRST_APEX_COLUMN + (rest / APEX_ROWS) * s, FIRST_APEX_ROW + rest % APEX_ROWS);
  var out = vec2f(0.0);
  if (apex.x + 2 * s < N && !seesNoPixel(s, apex, direction) && !(LIGHT_CULL && lightCull(s, apex, direction))) {
    let axisSlope = (f32(direction) + 0.5) / f32(s); let spacing = SAMPLE_SPACING * f32(s);
    out = ${runs ? 'coneValue' : 'coneValueValues'}(cascade, 2 * s >= N, apex, direction, vec2f(f32(apex.x) + spacing, f32(apex.y) + axisSlope * spacing),
                    vec2f(f32(apex.x) - spacing, f32(apex.y) - axisSlope * spacing));
  }
  coneValues[index] = out;
}

// pass 3: gather at the probes (corners 1 .. N - 1), accumulated over the quadrants in scene coordinates
@compute @workgroup_size(${WORKGROUP})
fn gatherProbes(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = threadIndex(id, groups);
  if (index >= (N - 1) * (N - 1)) { return; }
  let probe = vec2i(1 + index / (N - 1), 1 + index % (N - 1)); let probePoint = vec2f(probe);
  var fluenceAtProbe = 0.0;
  for (var direction = -1; direction <= 0; direction += 1) {
    if (isOpaque(0, probe.x, probe.y + direction)) { fluenceAtProbe += emissionAt(0, probe.x, probe.y + direction) * QUARTER_PI; continue; }   // the pixel next to the probe
    let axis = vec2i(2, 2 * direction + 1);
${runs ? `    fluenceAtProbe += lightBehind0(silhouettesFromApex(0, probe.x + 1, probe.y + direction, direction, probe, axis, true),
                                   angleFromAxis(vec2i(1, direction), axis), angleFromAxis(vec2i(1, direction + 1), axis), probe, probePoint, probePoint).x;
    fluenceAtProbe += coneValue(0, false, probe, direction, probePoint, probePoint).x;` : `    fluenceAtProbe += tileLight(0, probe.x + 1, probe.y + direction, direction, angleFromAxis(vec2i(1, direction), axis),
                                angleFromAxis(vec2i(1, direction + 1), axis), 1.0, probe, probePoint, probePoint).x;
    fluenceAtProbe += coneValueValues(0, false, probe, direction, probePoint, probePoint).x;`}
  }
  let o = sceneCornerIndex(probe.x, probe.y);
  if (params.quadrant == 0) { fluence[o] = fluenceAtProbe; } else { fluence[o] += fluenceAtProbe; }
}
`;
}

export class PolygonalHRC {
  // cull: also skip cones with no light pixel nearby (scans the light list per cone; changes nothing)
  // runs: tiles hold corner runs (occluder silhouettes); false: transmittance and light only
  constructor(device, N, { cull = false, runs = true } = {}) {
    const lastCascade = Math.round(Math.log2(N));
    if (1 << lastCascade !== N || lastCascade < 2 || lastCascade > 10) throw new Error('N must be a power of two, 4 .. 1024 (packed column offsets: 10 bits)');
    Object.assign(this, { device, N, lastCascade, cull, runs });
    const storage = GPUBufferUsage.STORAGE, copyDst = GPUBufferUsage.COPY_DST, copySrc = GPUBufferUsage.COPY_SRC;
    const buffer = (size, usage = storage) => device.createBuffer({ size: Math.max(size, 16), usage });

    // sizes (match the shader's index functions)
    this.extensionThreads = Array.from({ length: lastCascade + 1 }, (_, n) => 2 * N * (N + (1 << n) + 1));
    this.coneTileCount = this.extensionThreads.reduce((a, b) => a + b, 0);
    this.mergeThreads = Array.from({ length: lastCascade }, (_, n) => ((2 * N) / (1 << n) + 1) * (3 * N + 8) * 2 * (1 << n));
    const coneValueCount = Math.max(...this.mergeThreads.slice(1));

    this.emissionBuffer = buffer(N * N * 4, storage | copyDst);
    this.transmittanceBuffer = buffer(N * N * 4, storage | copyDst);
    this.coneTileBuffer = buffer(this.coneTileCount * TILE_WORDS * 4, storage | copySrc);
    this.coneValueBuffers = [buffer(coneValueCount * 8, storage | copySrc), buffer(coneValueCount * 8, storage | copySrc)];
    this.fluenceBuffer = buffer((N + 1) * (N + 1) * 4, storage | copySrc | copyDst);
    this.lightListBuffer = buffer(N * N * 8, storage | copyDst);
    this.diagnosticsBuffer = buffer(16, storage | copySrc | copyDst);
    this.bytes = { coneTiles: this.coneTileCount * TILE_WORDS * 4, coneValues: 2 * coneValueCount * 8 };

    // pass parameters, one block per (quadrant, cascade)
    this.lightCount = 0;
    this.paramData = new Int32Array(4 * (lastCascade + 1) * (PARAM_STRIDE / 4));
    this.paramBuffer = buffer(this.paramData.byteLength, GPUBufferUsage.UNIFORM | copyDst);
    this.writeParams();

    this.module = device.createShaderModule({ code: shaderSource({ N, lastCascade, cull, runs }) });
    const readOnly = { type: 'read-only-storage' }, readWrite = { type: 'storage' };
    const kinds = [readOnly, readOnly, readWrite, readOnly, readWrite, readWrite, { type: 'uniform', hasDynamicOffset: true }, readOnly, readWrite];
    const layout = device.createBindGroupLayout({ entries: kinds.map((b, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: b })) });
    // bind group p reads child values from coneValueBuffers[p] and writes coneValueBuffers[1 - p]
    this.bindGroups = [0, 1].map((p) => device.createBindGroup({ layout, entries: [this.emissionBuffer, this.transmittanceBuffer, this.coneTileBuffer,
      this.coneValueBuffers[p], this.coneValueBuffers[1 - p], this.fluenceBuffer, this.paramBuffer, this.lightListBuffer, this.diagnosticsBuffer]
      .map((b, binding) => ({ binding, resource: binding === 6 ? { buffer: b, size: 16 } : { buffer: b } })) }));
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint) => device.createComputePipeline({ layout: pipelineLayout, compute: { module: this.module, entryPoint } });
    this.pipelines = { extendLength1: pipeline(runs ? 'extendLength1' : 'extendLength1Values'), extendConeTiles: pipeline(runs ? 'extendConeTiles' : 'extendConeTilesValues'),
                       mergeCascade: pipeline('mergeCascade'),
                       gatherProbes: pipeline('gatherProbes') };
  }

  destroy() {
    for (const b of [this.emissionBuffer, this.transmittanceBuffer, this.coneTileBuffer, ...this.coneValueBuffers, this.fluenceBuffer,
      this.lightListBuffer, this.diagnosticsBuffer, this.paramBuffer]) b.destroy();
  }

  paramOffset(quadrant, cascade) { return (quadrant * (this.lastCascade + 1) + cascade) * PARAM_STRIDE; }
  writeParams() {
    for (let q = 0; q < 4; q++) for (let n = 0; n <= this.lastCascade; n++) this.paramData.set([n, q, this.lightCount, 0], this.paramOffset(q, n) / 4);
    this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramData);
  }

  async compileInfo() { return (await this.module.getCompilationInfo()).messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`); }

  // E: emission, A: transmittance (opaque: A <= 0), both [x * N + y]
  upload(E, A) {
    this.device.queue.writeBuffer(this.emissionBuffer, 0, E); this.device.queue.writeBuffer(this.transmittanceBuffer, 0, A);
    if (this.cull) {
      const list = []; for (let x = 0; x < this.N; x++) for (let y = 0; y < this.N; y++) if (E[x * this.N + y] > 0) list.push(x, y);
      if (list.length) this.device.queue.writeBuffer(this.lightListBuffer, 0, new Int32Array(list));
      if (list.length / 2 !== this.lightCount) { this.lightCount = list.length / 2; this.writeParams(); }
    }
  }

  dispatch(pass, pipeline, paramOffset, threads, bindGroup = 0) {
    if (threads <= 0) return;
    const groups = Math.ceil(threads / WORKGROUP), rows = Math.ceil(groups / 65535);
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.bindGroups[bindGroup], [paramOffset]); pass.dispatchWorkgroups(Math.ceil(groups / rows), rows);
  }
  extensionPipeline(cascade) { return cascade === 0 ? this.pipelines.extendLength1 : this.pipelines.extendConeTiles; }
  encodeExtension(pass, quadrant) {
    for (let n = 0; n <= this.lastCascade; n++) this.dispatch(pass, this.extensionPipeline(n), this.paramOffset(quadrant, n), this.extensionThreads[n]);
  }
  // cascades lastCascade - 1 .. 1 alternate between the two cone-value buffers; returns the bind group whose input holds cascade 1
  encodeMerges(pass, quadrant) {
    let p = 1;   // the top cascade writes coneValueBuffers[0] (bind group 1 writes buffer 0)
    for (let n = this.lastCascade - 1; n >= 1; n--) { this.dispatch(pass, this.pipelines.mergeCascade, this.paramOffset(quadrant, n), this.mergeThreads[n], p); p = 1 - p; }
    return p;
  }
  encodeGather(pass, quadrant, bindGroup) { this.dispatch(pass, this.pipelines.gatherProbes, this.paramOffset(quadrant, 0), (this.N - 1) * (this.N - 1), bindGroup); }
  encodeQuadrant(enc, quadrant) {
    const pass = enc.beginComputePass();
    this.encodeExtension(pass, quadrant);
    const bindGroup = this.encodeMerges(pass, quadrant);
    this.encodeGather(pass, quadrant, bindGroup);
    pass.end();
  }
  encodeAll(enc) { for (let q = 0; q < 4; q++) this.encodeQuadrant(enc, q); }

  async read(buf, bytes) {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, staging, 0, bytes); this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const out = staging.getMappedRange().slice(0); staging.unmap(); staging.destroy(); return out;
  }
  // corner fluence [cx * (N + 1) + cy], all four quadrants, and the diagnostics counters
  async solve(E, A) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); enc.clearBuffer(this.fluenceBuffer); enc.clearBuffer(this.diagnosticsBuffer);
    this.encodeAll(enc); this.device.queue.submit([enc.finish()]);
    const F = new Float32Array(await this.read(this.fluenceBuffer, (this.N + 1) ** 2 * 4));
    const counters = new Uint32Array(await this.read(this.diagnosticsBuffer, 16));
    return { F, silhouetteLimitExceeded: counters[0], packingOutOfRange: counters[1] };
  }
  // the raw cone-tile store of quadrant q (debug)
  async readConeTiles(E, A, quadrant = 0) {
    this.upload(E, A);
    const enc = this.device.createCommandEncoder(); const pass = enc.beginComputePass(); this.encodeExtension(pass, quadrant); pass.end();
    this.device.queue.submit([enc.finish()]);
    return new Uint32Array(await this.read(this.coneTileBuffer, this.coneTileCount * TILE_WORDS * 4));
  }
  // the silhouettes of tiles { quadrant, cascade, X, y, direction }, rebuilt for their quadrants (the store holds only the
  // last one): per query [{ lower: [x, y], upper: [x, y], light }] in the quadrant's pixels
  async silhouettesAt(queries) {
    const { N, device } = this, B = TILE_WORDS * 4;
    const slots = queries.map(({ cascade, X, y, direction }) => {
      const L = 1 << cascade, mirrored = direction < 0, yStored = mirrored ? N - 1 - y : y, directionStored = mirrored ? -direction - 1 : direction;
      if (X < 0 || X + L > N || yStored < -(L + 1) || yStored >= N) return -1;
      return 2 * N * (N + 1) * cascade + 2 * N * (L - 1) + ((mirrored ? N >> cascade : 0) + (X >> cascade)) * L * (N + L + 1) + directionStored * (N + L + 1) + yStored + L + 1;
    });
    const staging = device.createBuffer({ size: Math.max(16, slots.filter((s) => s >= 0).length * B), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder(), at = [];
    let o = 0;
    for (let q = 0; q < 4; q++) {
      const picked = queries.map((_, i) => i).filter((i) => queries[i].quadrant === q && slots[i] >= 0);
      if (!picked.length) continue;
      const pass = enc.beginComputePass(), top = Math.max(...picked.map((i) => queries[i].cascade));
      for (let n = 0; n <= top; n++) this.dispatch(pass, this.extensionPipeline(n), this.paramOffset(q, n), this.extensionThreads[n]);
      pass.end();
      for (const i of picked) { enc.copyBufferToBuffer(this.coneTileBuffer, slots[i] * B, staging, o, B); at[i] = o / 4; o += B; }
    }
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = staging.getMappedRange().slice(0); staging.unmap(); staging.destroy();
    const words = new Uint32Array(raw), lights = new Float32Array(raw);
    return queries.map(({ cascade, X, y, direction }, i) => {
      if (at[i] === undefined) return [];
      const mirrored = direction < 0, yStored = mirrored ? N - 1 - y : y, directionStored = mirrored ? -direction - 1 : direction, silhouettes = [];
      const pixel = (w) => { const dx = w & 1023, py = yStored + ((directionStored * dx) >> cascade) + ((w >> 10) & 63) - 32; return [X + dx, mirrored ? N - 1 - py : py]; };
      for (let j = 0; j < 3; j++) {
        const w = words[at[i] + j * SILHOUETTE_WORDS];
        if (w === 0xffffffff) break;
        silhouettes.push({ lower: pixel(w & 0xffff), upper: pixel(w >>> 16), light: lights[at[i] + j * SILHOUETTE_WORDS + 1] });
      }
      return silhouettes;
    });
  }
  // GPU time of each stage alone (quadrant q): reps dispatches in one submit, after one warm-up
  async profile(reps = 10, quadrant = 0) {
    const dev = this.device, out = {};
    const stage = async (name, encode) => {
      const run = (n) => { const enc = dev.createCommandEncoder(); const pass = enc.beginComputePass(); for (let r = 0; r < n; r++) encode(pass); pass.end(); dev.queue.submit([enc.finish()]); };
      run(1); await dev.queue.onSubmittedWorkDone(); const t0 = performance.now();
      run(reps); await dev.queue.onSubmittedWorkDone(); out[name] = +((performance.now() - t0) / reps).toFixed(3);
    };
    for (let n = 0; n <= this.lastCascade; n++) await stage(`extension, length ${1 << n}`, (pass) => this.dispatch(pass, this.extensionPipeline(n), this.paramOffset(quadrant, n), this.extensionThreads[n]));
    let p = 1;
    for (let n = this.lastCascade - 1; n >= 1; n--) { const bindGroup = p; await stage(`merge, cascade ${n}`, (pass) => this.dispatch(pass, this.pipelines.mergeCascade, this.paramOffset(quadrant, n), this.mergeThreads[n], bindGroup)); p = 1 - p; }
    await stage('gather', (pass) => this.encodeGather(pass, quadrant, p));
    return out;
  }
}
