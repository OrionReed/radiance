// Overlay presets: code run with `api` (see runOverlay in main.js). Each is a starting point to edit.
export const PRESETS = {
  'line tiles of one block per level': `// Line space ("at column" chart reads best). Every ray's line tile in the block holding the filter column,
// start rows from the filter, one colour per level.
const X = api.state.col + 0.5, f = api.frusta[0], t = api.tree, s = api.state;
for (let n = 0; n <= t.top; n++) {
  const l = t.lv[n], [u] = api.toLocal(api.N, t.scheme, f, f < 2 ? X : 0, f < 2 ? 0 : X);
  const j = Math.min(l.B - 1, Math.floor(u / l.L)), flat = [];
  for (let ri = 0; ri < l.R; ri++) for (let y = s.y0; y < Math.min(api.N, s.y0 + s.rows); y++) if (t.inGrid(n, ri, y)) flat.push(...api.embedTile(f, t.id(n, j, ri, y)));
  api.rawSegments(flat, api.levelColor(n), 1);
}`,
  'join posts': `// Joins. A post at every column, as high as its join level: where rays of that level meet their siblings.
const segs = [];
for (let x = 0; x <= api.tree.W; x++) {
  const h = api.joinAt(x) * api.G;
  segs.push([x, -1, 0, x, -1, h], [x, api.N + 1, 0, x, api.N + 1, h]);
}
api.segments(segs, [0.7, 0.7, 0.7], 1.5);`,
  'pick-up level staircase': `// Stacked or joins, frustum +x, Reciprocal frame. z = the level at which the merge picks up light from column x
// for the pixel's column: floor(log2(x XOR x_pixel)), the highest join level between them.
const xr = api.state.rx;
api.surface((x) => {
  const a = Math.min(api.N - 1, Math.floor(x));
  if (a === xr) return -api.G * 0.5;
  let n = 0; while ((a >> (n + 1)) !== (xr >> (n + 1))) n++;
  return n * api.G;
}, { res: api.N * 2, color: [0.4, 0.8, 1], opacity: 0.18 });`,
  'log-distance cone': `// z = G · log2(distance to the pixel): the level a distance-based cascade would use at distance D.
const rx = api.state.rx + 0.5, ry = api.state.ry + 0.5;
api.surface((x, y) => api.G * Math.max(0, Math.log2(Math.max(1, Math.hypot(x - rx, y - ry)))), { res: 96, color: [1, 0.6, 0.3], opacity: 0.16 });`,
  'block boundaries per level': `// Stacked. At each level the edges of its nodes (frustum +x; Fibonacci nodes of one length do not tile).
const t = api.tree, segs = [];
for (let n = 0; n <= t.top; n++) {
  const { xs, L } = t.lv[n];
  for (const x of xs) for (const e of [x, x + L]) segs.push([e, 0, n * api.G, e, api.N, n * api.G]);
}
api.segments(segs, [0.9, 0.9, 0.9], 1);`,
  'HRC probes': `// Stacked. HRC probe points (p 2^n, y + ½) at each level, every 4th row.
const t = api.tree, f = api.frusta[0], pts = [];
for (let n = 0; n <= t.top; n++) {
  const L = 1 << n;
  for (let p = 0; p <= api.N / L; p++) for (let y = 0; y < api.N; y += 4) {
    const [x, yy] = api.toWorld(api.N, 'hrc', f, p * L, y + 0.5);
    pts.push([x, yy, n * api.G]);
  }
}
api.points(pts, [1, 1, 0.4], 3);`,
  'ℓ’s vertical sheet': `// World. The vertical sheet over line ℓ (turn ℓ on): rays lying in it carry ℓ.
const { A, B } = api.line, top = (api.tree.top + 0.5) * api.G, segs = [];
for (let k = 0; k <= 16; k++) {
  const x = A[0] + (B[0] - A[0]) * k / 16, y = A[1] + (B[1] - A[1]) * k / 16;
  segs.push([x, y, -api.G * 0.5, x, y, top]);
}
segs.push([A[0], A[1], top, B[0], B[1], top]);
api.segments(segs, [0.6, 1, 0.5], 1);`,
};
