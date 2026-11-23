// Plain Reciprocal HRC (every row offset, no Wu split), kept per level so that every ray of every cascade can be drawn as a
// line. Port of wide-rays/wide.ts at spacing 1 (which equals folkjs plain_port/plain.py exactly).
//
// One frustum carries light toward +x. Level L has rays of length L between probe planes x = bL and (b+1)L: a ray
// goes from (bL, row + 1/2) to ((b+1)L, row + 1/2 + d), d = -L … L. Per ray: T (transmittance), R (radiance gathered
// along it) and M (light arriving at its start from behind, inside its cone: what the merge hands it).
// Arrays are [x][y] flattened as x * N + y.

export interface Level {
  L: number;
  T: Float32Array;
  R: Float32Array;
  M: Float32Array;
}

export interface Frustum {
  rows: number;
  pad: number;
  levels: Map<number, Level>;
  out: Float32Array;
}

export const coneWeight = (L: number, d: number) =>
  Math.atan(Math.min(1, (d + 0.5) / L)) - Math.atan(Math.max(-1, (d - 0.5) / L));

/** a ray with row offset D as halves [first offset a, second offset b, weight]; the second half starts at row + a */
export const splits = (D: number): [number, number, number][] =>
  D % 2 === 0 ? [[D / 2, D / 2, 1]] : [[(D - 1) / 2, (D + 1) / 2, 0.5], [(D + 1) / 2, (D - 1) / 2, 0.5]];

/** where one ray sits inside a frustum: level k (length 2^k), start plane, start row (with the padding), row offset d */
export interface RayId { k: number; plane: number; row: number; d: number }

export interface Options {
  /** emission from the scene's lights feeds the length-1 rays */
  lights: boolean;
  /** the extension covers cascades upFrom … upTo: R below upFrom is only what is injected (no emission enters when
   * upFrom > 0), R is extended over (upFrom, upTo], and R above upTo is zero. upTo < 0: no extension. T, geometry, is
   * always kept. */
  upFrom: number;
  upTo: number;
  /** the merge starts at cascade downFrom (log2 N: from nothing above the top) and merges down to downTo; M
   * outside that range is zero apart from what is injected */
  downFrom: number;
  downTo: number;
  /** values added to a ray's R right after the extension computes it, or to its M right after the merge does */
  injectR: (RayId & { value: number })[];
  injectM: (RayId & { value: number })[];
}

export function frustum(N: number, emission: Float32Array, free: Float32Array, opt: Options): Frustum {
  const pad = N, rows = N + 2 * pad;
  const e = (x: number, y: number) => (!opt.lights || opt.upFrom > 0 || opt.upTo < 0 || y < pad || y >= pad + N ? 0 : emission[x * N + y - pad]);
  const f = (x: number, y: number) => (y < pad || y >= pad + N ? 0 : free[x * N + y - pad]);
  const at = (L: number, plane: number, row: number, d: number) => (plane * rows + row) * (2 * L + 1) + d + L;
  const inject = (list: Options['injectR'], L: number, arr: Float32Array) => {
    for (const r of list) if (1 << r.k === L && r.plane < N / L && r.row >= 0 && r.row < rows) arr[at(L, r.plane, r.row, r.d)] += r.value;
  };
  const T = new Map<number, Float32Array>(), R = new Map<number, Float32Array>();

  const T1 = new Float32Array(N * rows * 3), R1 = new Float32Array(N * rows * 3);
  for (let x = 0; x < N; x++)
    for (let y = 0; y < rows; y++)
      for (let d = -1; d <= 1; d++) {
        const i = at(1, x, y, d), fd = f(x, y + d);
        T1[i] = f(x, y) * fd;
        R1[i] = e(x, y + d) + fd * e(x, y);
      }
  inject(opt.injectR, 1, R1);
  T.set(1, T1); R.set(1, R1);

  // extension: each ray from its two halves
  for (let L = 2; L <= N / 2; L *= 2) {
    const h = L / 2, planes = N / L, Th = T.get(h)!, Rh = R.get(h)!, extend = Math.log2(L) <= opt.upTo && Math.log2(L) > opt.upFrom;
    const t = new Float32Array(planes * rows * (2 * L + 1)), r = new Float32Array(t.length);
    for (let D = -L; D <= L; D++)
      for (const [a, b, w] of splits(D))
        for (let p = 0; p < planes; p++)
          for (let y = 0; y < rows; y++) {
            const yb = y + a;
            if (yb < 0 || yb >= rows) continue;
            const A = at(h, 2 * p, y, a), B = at(h, 2 * p + 1, yb, b), o = at(L, p, y, D);
            t[o] += w * Th[A] * Th[B];
            if (extend) r[o] += w * (Rh[B] + Th[B] * Rh[A]);
          }
    inject(opt.injectR, L, r);
    T.set(L, t); R.set(L, r);
  }

  // merge: the light arriving at each ray's start from behind it, longest rays first
  const levels = new Map<number, Level>();
  const Ms = new Map<number, Float32Array>();
  for (let L = 1; L <= N; L *= 2) Ms.set(L, new Float32Array((N / L) * rows * (2 * L + 1)));
  const top = 1 << Math.min(opt.downFrom, Math.log2(N)), bottom = 1 << opt.downTo;
  for (let L = N; L >= 1; L /= 2) {
    const M = Ms.get(L)!;
    if (L < top && L >= bottom) {
      const P = 2 * L, Gp = Ms.get(P)!, Th = T.get(L)!, Rh = R.get(L)!;
      for (let D = -P; D <= P; D++) {
        const c = coneWeight(P, D);
        for (const [a, b, w] of splits(D))
          for (let p = 0; p < N / P; p++)
            for (let y = 0; y < rows; y++) {
              const far = Gp[at(P, p, y, D)], A = at(L, 2 * p, y, a);
              M[A] += w * far;
              const yb = y + a;
              if (yb < 0 || yb >= rows) continue;
              M[at(L, 2 * p + 1, yb, b)] += w * (Th[A] * far + c * Rh[A]);
            }
      }
    }
    inject(opt.injectM, L, M);
    if (L < N) levels.set(L, { L, T: T.get(L)!, R: R.get(L)!, M });
  }
  const M = Ms.get(1)!;

  // the pixel readout: per row offset, the two length-1 rays whose middles bracket the pixel centre
  const out = new Float32Array(N * N);
  for (let x = 0; x < N; x++)
    for (let y = pad; y < pad + N; y++) {
      let sum = 0;
      for (let d = -1; d <= 1; d++) {
        sum += 0.5 * M[at(1, x, y, d)];
        const ys = y - d;
        if (ys >= 0 && ys < rows) sum += 0.5 * f(x, ys) * M[at(1, x, ys, d)];
      }
      out[x * N + y - pad] = sum * f(x, y);
    }
  return { rows, pad, levels, out };
}

/** frustum views: index into the world image, and frustum-local (column edge px, row coordinate py) to world (x, y) */
export const views = [
  { src: (N: number, x: number, y: number) => x * N + y, world: (N: number, px: number, py: number) => [px, py] },
  { src: (N: number, x: number, y: number) => (N - 1 - x) * N + y, world: (N: number, px: number, py: number) => [N - px, py] },
  { src: (N: number, x: number, y: number) => y * N + x, world: (_: number, px: number, py: number) => [py, px] },
  { src: (N: number, x: number, y: number) => y * N + (N - 1 - x), world: (N: number, px: number, py: number) => [py, N - px] },
] as const;

/** floats per line: x0 y0 x1 y1, M, M at end, c·R, cone weight c, level, frustum, R, T, start plane, start row, row offset */
export const STRIDE = 15;

export interface Solution {
  N: number;
  fluence: Float32Array;
  lines: Float32Array;
  count: number;
  levels: number;
}

/** every frustum, every level: the pixel fluence and one line per ray that touches the image */
export type Injection = RayId & { frustum: number; value: number; pass: 'R' | 'M' };

export function solve(N: number, emission: Float32Array, solid: Uint8Array, opt: Omit<Options, 'injectR' | 'injectM'>, injections: Injection[]): Solution {
  const free = new Float32Array(N * N).map((_, i) => (solid[i] ? 0 : 1));
  const fluence = new Float32Array(N * N);
  const nLevels = Math.log2(N);
  let capacity = 0;
  for (let h = 1; h <= N / 2; h *= 2) capacity += 4 * (N / h) * (N + 2 * h + 4) * (2 * h + 1);
  const lines = new Float32Array(capacity * STRIDE);
  let count = 0;
  views.forEach((view, v) => {
    const em = new Float32Array(N * N), fr = new Float32Array(N * N), src = new Int32Array(N * N);
    for (let x = 0; x < N; x++)
      for (let y = 0; y < N; y++) {
        const i = view.src(N, x, y);
        src[x * N + y] = i; em[x * N + y] = emission[i]; fr[x * N + y] = free[i];
      }
    const mine = injections.filter((j) => j.frustum === v);
    const F = frustum(N, em, fr, { ...opt, injectR: mine.filter((j) => j.pass === 'R'), injectM: mine.filter((j) => j.pass === 'M') });
    for (let i = 0; i < N * N; i++) fluence[src[i]] += F.out[i];
    for (const [h, lv] of F.levels) {
      const level = Math.log2(h), nd = 2 * h + 1;
      for (let d = -h; d <= h; d++) {
        const cone = coneWeight(h, d);
        for (let y = 0; y < F.rows; y++) {
          const y0 = y - F.pad + 0.5, y1 = y0 + d;
          if (Math.max(y0, y1) < -1 || Math.min(y0, y1) > N + 1) continue;
          for (let q = 0; q < N / h; q++) {
            const i = (q * F.rows + y) * nd + d + h;
            const start = lv.M[i], own = cone * lv.R[i], end = lv.T[i] * start + own;
            const [ax, ay] = view.world(N, q * h, y0), [bx, by] = view.world(N, (q + 1) * h, y1);
            const o = count++ * STRIDE;
            lines[o] = ax; lines[o + 1] = ay; lines[o + 2] = bx; lines[o + 3] = by;
            lines[o + 4] = start; lines[o + 5] = end; lines[o + 6] = own; lines[o + 7] = cone;
            lines[o + 8] = level; lines[o + 9] = v;
            lines[o + 10] = lv.R[i]; lines[o + 11] = lv.T[i];
            lines[o + 12] = q; lines[o + 13] = y; lines[o + 14] = d;
          }
        }
      }
    }
  });
  return { N, fluence, lines, count, levels: nLevels };
}

/** exact fluence at a point: the angle the union of the light squares covers (radiance 1); NaN inside a light */
export function exactAt(px: number, py: number, lights: readonly (readonly [number, number])[]): number {
  const spans: [number, number][] = [];
  for (const [lx, ly] of lights) {
    if (px >= lx && px <= lx + 1 && py >= ly && py <= ly + 1) return NaN;
    const c = Math.atan2(ly + 0.5 - py, lx + 0.5 - px);
    let lo = Infinity, hi = -Infinity;
    for (const [cx, cy] of [[lx, ly], [lx + 1, ly], [lx, ly + 1], [lx + 1, ly + 1]]) {
      let a = Math.atan2(cy - py, cx - px) - c;
      a -= 2 * Math.PI * Math.round(a / (2 * Math.PI));
      lo = Math.min(lo, a); hi = Math.max(hi, a);
    }
    let s = c + lo;
    s -= 2 * Math.PI * Math.floor(s / (2 * Math.PI));
    spans.push([s, s + hi - lo]);
  }
  if (spans.length === 1) return spans[0][1] - spans[0][0];
  spans.sort((p, q) => p[0] - q[0]);
  const merged: [number, number][] = [];
  for (const [s, e] of spans) {
    const last = merged[merged.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  let total = 0;
  for (const [s, e] of merged) total += e - s;
  // the last span may wrap past 2π onto the first ones
  const wrap = merged[merged.length - 1][1] - 2 * Math.PI;
  if (wrap > 0)
    for (let i = 0; i < merged.length - 1; i++) total -= Math.max(0, Math.min(wrap, merged[i][1]) - merged[i][0]);
  return total;
}

/** a ray's place in the tree of halves: its descendants (the halves it is made of, down to length 1) and its
 * ancestors (the rays it is a half of, up to the top), each with the summed product of split weights on the way */
export function relatives(N: number, id: RayId) {
  const rows = 3 * N, K = Math.log2(N);
  const key = (r: RayId) => `${r.k} ${r.plane} ${r.row} ${r.d}`;
  const out: { id: RayId; w: number; up: boolean }[] = [];
  let level = new Map([[key(id), { id, w: 1 }]]);
  for (let k = id.k; k > 0; k--) {
    const next = new Map<string, { id: RayId; w: number }>();
    const add = (r: RayId, w: number) => {
      if (r.row < 0 || r.row >= rows) return;
      const e = next.get(key(r));
      if (e) e.w += w; else next.set(key(r), { id: r, w });
    };
    for (const { id: p, w } of level.values())
      for (const [a, b, s] of splits(p.d)) {
        add({ k: k - 1, plane: 2 * p.plane, row: p.row, d: a }, w * s);
        add({ k: k - 1, plane: 2 * p.plane + 1, row: p.row + a, d: b }, w * s);
      }
    for (const e of next.values()) out.push({ ...e, up: false });
    level = next;
  }
  level = new Map([[key(id), { id, w: 1 }]]);
  for (let k = id.k; k < K - 1; k++) {
    const next = new Map<string, { id: RayId; w: number }>(), L = 2 << k;
    for (const { id: c, w } of level.values()) {
      const second = c.plane % 2 === 1, plane = c.plane >> 1;
      for (let D = -L; D <= L; D++)
        for (const [a, b, s] of splits(D)) {
          if ((second ? b : a) !== c.d) continue;
          const r = { k: k + 1, plane, row: second ? c.row - a : c.row, d: D };
          if (r.row < 0 || r.row >= rows) continue;
          const e = next.get(key(r));
          if (e) e.w += w * s; else next.set(key(r), { id: r, w: w * s });
        }
    }
    for (const e of next.values()) out.push({ ...e, up: true });
    level = next;
  }
  return out;
}
