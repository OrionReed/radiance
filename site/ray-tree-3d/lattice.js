// The arrangement that drawn rays make on one flat sheet: where they cross (exact), how many distinct lines pass through
// each crossing, and the faces between the lines. Every ray end point lies on the half-integer grid, so doubled
// coordinates are integers and a crossing is an exact rational point, keyed by its reduced numerators and denominator.

const gcd = (a, b) => {
  a = Math.abs(a); b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
};

/** segs: [[x0, y0, x1, y1, isEdge]] in world units. Points: {key, x, y, lines} (lines: distinct non-edge lines through it).
 * Faces: bounded cells, each a list of corner [x, y] (collinear boundary points dropped). */
export function arrangement(segs, { faces: wantFaces = true } = {}) {
  const n = segs.length;
  const S = segs.map(([x0, y0, x1, y1]) => [Math.round(2 * x0), Math.round(2 * y0), Math.round(2 * x1), Math.round(2 * y1)]);
  const lineId = S.map(([x0, y0, x1, y1]) => {
    let a = y1 - y0, b = x0 - x1, c = a * x0 + b * y0;
    const g = gcd(gcd(a, b), c) || 1;
    a /= g; b /= g; c /= g;
    if (a < 0 || (a === 0 && b < 0)) { a = -a; b = -b; c = -c; }
    return `${a},${b},${c}`;
  });
  const pts = new Map(), on = Array.from({ length: n }, () => []);
  const add = (i, s, X, Y, D) => {
    if (D < 0) { X = -X; Y = -Y; D = -D; }
    const g = gcd(gcd(X, Y), D);
    X /= g; Y /= g; D /= g;
    const key = `${X},${Y},${D}`;
    let p = pts.get(key);
    if (!p) pts.set(key, (p = { key, x: X / (2 * D), y: Y / (2 * D), lines: new Set() }));
    if (!segs[i][4]) p.lines.add(lineId[i]);
    on[i].push([s, key]);
  };
  for (let i = 0; i < n; i++) { add(i, 0, S[i][0], S[i][1], 1); add(i, 1, S[i][2], S[i][3], 1); }
  const order = [...S.keys()].sort((a, b) => Math.min(S[a][0], S[a][2]) - Math.min(S[b][0], S[b][2]));
  for (let oi = 0; oi < n; oi++) {
    const i = order[oi], [ax, ay, bx, by] = S[i], dxi = bx - ax, dyi = by - ay;
    const maxX = Math.max(ax, bx), minY = Math.min(ay, by), maxY = Math.max(ay, by);
    for (let oj = oi + 1; oj < n; oj++) {
      const j = order[oj], [cx, cy, ex_, ey_] = S[j];
      if (Math.min(cx, ex_) > maxX) break;
      if (Math.max(cy, ey_) < minY || Math.min(cy, ey_) > maxY) continue;
      const dxj = ex_ - cx, dyj = ey_ - cy, den = dxi * dyj - dyi * dxj, ex = cx - ax, ey = cy - ay;
      if (den !== 0) {
        let sn = ex * dyj - ey * dxj, un = ex * dyi - ey * dxi, d = den;
        if (d < 0) { d = -d; sn = -sn; un = -un; }
        if (sn < 0 || sn > d || un < 0 || un > d) continue;
        const X = ax * d + sn * dxi, Y = ay * d + sn * dyi;
        add(i, sn / d, X, Y, d); add(j, un / d, X, Y, d);
      } else if (ex * dyi - ey * dxi === 0) {
        // collinear: each takes the other's end points that lie on it
        for (const [a, b] of [[i, j], [j, i]]) {
          const [px, py, qx, qy] = S[a], ux = qx - px, uy = qy - py, len = ux * ux + uy * uy;
          for (const [X, Y] of [[S[b][0], S[b][1]], [S[b][2], S[b][3]]]) {
            const tn = (X - px) * ux + (Y - py) * uy;
            if (tn >= 0 && tn <= len) add(a, tn / len, X, Y, 1);
          }
        }
      }
    }
  }
  const points = [...pts.values()];
  if (!wantFaces) return { points, faces: [] };
  // edges: consecutive points along each segment (collinear overlaps give the same edge twice; kept once)
  const adj = new Map(), seen = new Set();
  const link = (u, v) => {
    const k = u < v ? `${u}|${v}` : `${v}|${u}`;
    if (seen.has(k)) return;
    seen.add(k);
    for (const [a, b] of [[u, v], [v, u]]) {
      const A = pts.get(a), B = pts.get(b);
      if (!adj.has(a)) adj.set(a, []);
      adj.get(a).push([Math.atan2(B.y - A.y, B.x - A.x), b]);
    }
  };
  for (const list of on) {
    list.sort((p, q) => p[0] - q[0]);
    for (let k = 1; k < list.length; k++) if (list[k][1] !== list[k - 1][1]) link(list[k - 1][1], list[k][1]);
  }
  const index = new Map();
  for (const [a, list] of adj) { list.sort((p, q) => p[0] - q[0]); index.set(a, new Map(list.map(([, b], i) => [b, i]))); }
  const done = new Set(), faces = [];
  for (const [a0, list] of adj)
    for (const [, b0] of list) {
      if (done.has(`${a0}|${b0}`)) continue;
      const cyc = [];
      let a = a0, b = b0;
      while (!done.has(`${a}|${b}`)) {
        done.add(`${a}|${b}`);
        cyc.push(a);
        const lb = adj.get(b), i = index.get(b).get(a);
        [a, b] = [b, lb[(i - 1 + lb.length) % lb.length][1]];
      }
      let area = 0;
      for (let k = 0; k < cyc.length; k++) { const P = pts.get(cyc[k]), Q = pts.get(cyc[(k + 1) % cyc.length]); area += P.x * Q.y - Q.x * P.y; }
      if (area <= 1e-9) continue;
      const corners = [];
      for (let k = 0; k < cyc.length; k++) {
        const A = pts.get(cyc[(k - 1 + cyc.length) % cyc.length]), B = pts.get(cyc[k]), C = pts.get(cyc[(k + 1) % cyc.length]);
        if (Math.abs((B.x - A.x) * (C.y - B.y) - (B.y - A.y) * (C.x - B.x)) > 1e-9) corners.push([B.x, B.y]);
      }
      faces.push(corners);
    }
  return { points, faces };
}

/** a colour per number of lines through a point, and its legend */
export const LINE_BUCKETS = [
  [2, 2, '#9a9a9a'], [3, 3, '#5aa9ff'], [4, 4, '#4ed8c9'], [5, 5, '#9be564'], [6, 8, '#ffd166'], [9, 16, '#ff8a5c'], [17, 1e9, '#ff7ac6'],
];
export const bucketOf = (lines) => LINE_BUCKETS.findIndex(([lo, hi]) => lines >= lo && lines <= hi);
export const FACE_COLOURS = { 3: '#5aa9ff', 4: '#ffd166', 5: '#ff5c5c' };
