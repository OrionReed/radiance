// Plain Reciprocal HRC (every row offset, no Wu split), kept per level so that every ray of every cascade can be drawn as a
// line. Port of wide-rays/wide.ts at spacing 1 (which equals folkjs plain_port/plain.py exactly).
//
// One frustum carries light toward +x. Level L has rays of length L between probe planes x = bL and (b+1)L: a ray
// goes from (bL, row + 1/2) to ((b+1)L, row + 1/2 + d), d = -L … L (… L + 1 with the rule's guard band). Per ray: T (transmittance), R (radiance gathered
// along it) and M (light arriving at its start from behind, inside its cone: what the merge hands it).
// A Rule switches the split and merge away from plain: other pairs of halves, a linear merge split, and a half-row tile
// phase for the longer levels (their rays then run from row + 1/2 + phase).
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

/** a ray's share of the directions: its offset bin [d − 1/2, d + 1/2] / L clipped to 45° both ways, so a guard ray
 * (steeper than 45°) gets exactly 0; inside the frustum the clip changes nothing */
const clip1 = (v: number) => Math.max(-1, Math.min(1, v));
export const coneWeight = (L: number, d: number) => Math.atan(clip1((d + 0.5) / L)) - Math.atan(clip1((d - 0.5) / L));

/** the cascade's split and merge rule; PLAIN is plain Reciprocal HRC */
export interface Rule {
  /** plain; plain plus 1/8 to the two neighbouring pairs of an even parent from length 8 on; or, for an odd parent,
   * two pairs of straight halves (equal offsets) whose second half starts one row off: a one-row jog at the junction */
  split: 'plain' | 'jitter' | 'jog';
  /** merge only: an odd parent shares its M between its two pairs by the slope of M across start rows */
  mergeLinear: boolean;
  /** tiles of length 8 and up sit half a row up: windows [j + 1/2, j + 3/2] at both planes */
  phase: boolean;
  /** a one-offset guard band at every level: offsets |d| <= L + 1 are stored (rises of ±2 across one column at c0).
   * Guard rays are steeper than 45°: they carry light (jitter can send them some) but have cone weight 0 and the
   * readout ignores them; jitter falls back to plain only where a partner leaves this extended range */
  guard: boolean;
}
export const PLAIN: Rule = { split: 'plain', mergeLinear: false, phase: false, guard: false };
/** the offsets beyond ±L a level stores: 1 with the guard band, else 0 */
export const guardOf = (rule: Rule) => (rule.guard ? 1 : 0);
const JITTER = 1 / 8, JITTER_FROM = 8, PHASE_FROM = 8;

/** the row phase of level L's tiles: its rays start and end at row + 1/2 + phase */
export const phaseOf = (rule: Rule, L: number) => (rule.phase && L >= PHASE_FROM ? 0.5 : 0);

/** one way to make a ray (start row u, offset D) from two halves: the first half starts at row u + dA with offset a,
 * the second at row u + dj with offset b; w is its share. Plain: dA = 0, dj = a, a + b = D. */
export interface Pair { dA: number; a: number; dj: number; b: number; w: number }

/** the pairs of halves of a ray of length L with row offset D under a rule */
export function pairsOf(rule: Rule, L: number, D: number): Pair[] {
  const h = L / 2, php = phaseOf(rule, L), phc = phaseOf(rule, h), G = guardOf(rule);
  // the sampled pairs of a ray inside the frustum keep to the halves' own range, as without the guard band (so plain
  // stays unchanged by it); only a guard ray's may use the guard offsets
  if (php !== phc) return sampledPairs(L, D, php, phc, Math.abs(D) > L ? h + G : h);
  // aligned tiles: plain's pairs, or the split law's
  const a = Math.floor(D / 2);
  if (D % 2 === 0) {
    if (rule.split === 'jitter' && L >= JITTER_FROM && Math.abs(a) + 1 <= h + G)
      return [
        { dA: 0, a, dj: a, b: a, w: 1 - 2 * JITTER },
        { dA: 0, a: a - 1, dj: a - 1, b: a + 1, w: JITTER }, // second half starts one row lower
        { dA: 0, a: a + 1, dj: a + 1, b: a - 1, w: JITTER }, // one row higher
      ];
    return [{ dA: 0, a, dj: a, b: a, w: 1 }];
  }
  if (rule.split === 'jog')
    return [{ dA: 0, a, dj: a + 1, b: a, w: 0.5 }, { dA: 0, a: a + 1, dj: a, b: a + 1, w: 0.5 }];
  return [{ dA: 0, a, dj: a, b: a + 1, w: 0.5 }, { dA: 0, a: a + 1, dj: a + 1, b: a, w: 0.5 }];
}

/** a parent's pairs when its tiles and its halves' tiles have different phases: the parent's lines sampled uniformly in
 * its tile (start and end in its windows), each assigned to the half tiles holding its start, middle and end */
function sampledPairs(L: number, D: number, php: number, phc: number, Kh: number): Pair[] {
  const S = 64, counts = new Map<string, number>();
  for (let i = 0; i < S; i++)
    for (let j = 0; j < S; j++) {
      const y0 = php + (i + 0.37) / S, y2 = D + php + (j + 0.37) / S, ym = (y0 + y2) / 2;
      const i0 = Math.floor(y0 - phc), jm = Math.floor(ym - phc), i2 = Math.floor(y2 - phc);
      const k = `${i0} ${jm - i0} ${jm} ${i2 - jm}`;
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
  // the true shares are small rationals; sampling ties on row boundaries bias them by ~1/S, so snap to sixteenths, and
  // keep only children inside the half's offset range
  const out: Pair[] = [];
  for (const [k, n] of counts) {
    const [dA, a, dj, b] = k.split(' ').map(Number), w = Math.round((16 * n) / (S * S)) / 16;
    if (w > 0 && Math.abs(a) <= Kh && Math.abs(b) <= Kh) out.push({ dA, a, dj, b, w });
  }
  const total = out.reduce((s, p) => s + p.w, 0);
  for (const p of out) p.w /= total;
  return out;
}

/** every level's pairs, by parent length L and D + L + G (G: the guard band, guardOf(rule)) */
const tables = new Map<string, Map<number, Pair[][]>>();
export function pairTable(N: number, rule: Rule) {
  const key = `${N} ${rule.split} ${rule.phase} ${rule.guard}`, G = guardOf(rule);
  let t = tables.get(key);
  if (!t) {
    t = new Map();
    for (let L = 2; L <= N; L *= 2) t.set(L, Array.from({ length: 2 * (L + G) + 1 }, (_, i) => pairsOf(rule, L, i - L - G)));
    tables.set(key, t);
  }
  return t;
}

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
  rule: Rule;
}

export function frustum(N: number, emission: Float32Array, free: Float32Array, opt: Options): Frustum {
  const pad = N, rows = N + 2 * pad;
  const e = (x: number, y: number) => (!opt.lights || opt.upFrom > 0 || opt.upTo < 0 || y < pad || y >= pad + N ? 0 : emission[x * N + y - pad]);
  const f = (x: number, y: number) => (y < pad || y >= pad + N ? 0 : free[x * N + y - pad]);
  const G = guardOf(opt.rule);
  const at = (L: number, plane: number, row: number, d: number) => (plane * rows + row) * (2 * (L + G) + 1) + d + L + G;
  const inject = (list: Options['injectR'], L: number, arr: Float32Array) => {
    for (const r of list) if (1 << r.k === L && r.plane < N / L && r.row >= 0 && r.row < rows && Math.abs(r.d) <= L + G) arr[at(L, r.plane, r.row, r.d)] += r.value;
  };
  const T = new Map<number, Float32Array>(), R = new Map<number, Float32Array>();
  const pairs = pairTable(N, opt.rule);

  const T1 = new Float32Array(N * rows * (2 * (1 + G) + 1)), R1 = new Float32Array(T1.length);
  for (let x = 0; x < N; x++)
    for (let y = 0; y < rows; y++) {
      for (let d = -1; d <= 1; d++) {
        const i = at(1, x, y, d), fd = f(x, y + d);
        T1[i] = f(x, y) * fd;
        R1[i] = e(x, y + d) + fd * e(x, y);
      }
      // guard rays, rise ±2: they cross the pixels y, y + s, y + 2s of their column, front to back
      if (G)
        for (const s of [-1, 1]) {
          const i = at(1, x, y, 2 * s), f1 = f(x, y + s), f2 = f(x, y + 2 * s);
          T1[i] = f(x, y) * f1 * f2;
          R1[i] = e(x, y + 2 * s) + f2 * (e(x, y + s) + f1 * e(x, y));
        }
    }
  inject(opt.injectR, 1, R1);
  T.set(1, T1); R.set(1, R1);

  // extension: each ray from its two halves
  for (let L = 2; L <= N / 2; L *= 2) {
    const h = L / 2, planes = N / L, Th = T.get(h)!, Rh = R.get(h)!, extend = Math.log2(L) <= opt.upTo && Math.log2(L) > opt.upFrom;
    const t = new Float32Array(planes * rows * (2 * (L + G) + 1)), r = new Float32Array(t.length);
    for (let D = -L - G; D <= L + G; D++)
      for (const { dA, a, dj, b, w } of pairs.get(L)![D + L + G])
        for (let p = 0; p < planes; p++)
          for (let y = 0; y < rows; y++) {
            const ya = y + dA, yb = y + dj;
            if (ya < 0 || ya >= rows || yb < 0 || yb >= rows) continue;
            const A = at(h, 2 * p, ya, a), B = at(h, 2 * p + 1, yb, b), o = at(L, p, y, D);
            t[o] += w * Th[A] * Th[B];
            if (extend) r[o] += w * (Rh[B] + Th[B] * Rh[A]);
          }
    inject(opt.injectR, L, r);
    T.set(L, t); R.set(L, r);
  }

  // merge: the light arriving at each ray's start from behind it, longest rays first
  const levels = new Map<number, Level>();
  const Ms = new Map<number, Float32Array>();
  for (let L = 1; L <= N; L *= 2) Ms.set(L, new Float32Array((N / L) * rows * (2 * (L + G) + 1)));
  const top = 1 << Math.min(opt.downFrom, Math.log2(N)), bottom = 1 << opt.downTo;
  for (let L = N; L >= 1; L /= 2) {
    const M = Ms.get(L)!;
    if (L < top && L >= bottom) {
      const P = 2 * L, Gp = Ms.get(P)!, Th = T.get(L)!, Rh = R.get(L)!;
      for (let D = -P - G; D <= P + G; D++) {
        const c = coneWeight(P, D), ps = pairs.get(P)![D + P + G];
        // the linear merge rule: of an odd parent's two pairs, the one whose second half starts lower gets
        // M (1/2 + delta) and the other M (1/2 - delta); the injected c·R_A keeps the pairs' own shares
        const linear = opt.rule.mergeLinear && ps.length === 2 && ps.every((q) => q.dA === 0);
        const lower = linear && ps[1].dj < ps[0].dj ? ps[1] : ps[0];
        for (const pair of ps) {
          const { dA, a, dj, b, w } = pair;
          for (let p = 0; p < N / P; p++)
            for (let y = 0; y < rows; y++) {
              const far = Gp[at(P, p, y, D)];
              let wf = w;
              if (linear) {
                const up = y + 1 < rows ? Gp[at(P, p, y + 1, D)] : 0, dn = y > 0 ? Gp[at(P, p, y - 1, D)] : 0;
                const delta = far > 0 ? Math.max(-0.5, Math.min(0.5, -(up - dn) / (24 * far))) : 0;
                wf = pair === lower ? 0.5 + delta : 0.5 - delta;
              }
              const ya = y + dA, yb = y + dj;
              if (ya < 0 || ya >= rows) continue;
              const A = at(L, 2 * p, ya, a);
              M[A] += wf * far;
              if (yb < 0 || yb >= rows) continue;
              M[at(L, 2 * p + 1, yb, b)] += wf * Th[A] * far + w * c * Rh[A];
            }
        }
      }
    }
    inject(opt.injectM, L, M);
    if (L < N) levels.set(L, { L, T: T.get(L)!, R: R.get(L)!, M });
  }
  const M = Ms.get(1)!;

  // the pixel readout (guard rays left out): per row offset, the two length-1 rays whose middles bracket the pixel centre
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

/** floats per line: x0 y0 x1 y1, M, M at end, c·R, cone weight c, level, frustum, R, T, start plane, start row, row offset,
 * guard (1 for a guard ray, steeper than 45°) */
export const STRIDE = 16;

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
  const nLevels = Math.log2(N), G = guardOf(opt.rule);
  let capacity = 0;
  for (let h = 1; h <= N / 2; h *= 2) capacity += 4 * (N / h) * (N + 2 * (h + G) + 4) * (2 * (h + G) + 1);
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
      const level = Math.log2(h), nd = 2 * (h + G) + 1, phase = phaseOf(opt.rule, h);
      for (let d = -h - G; d <= h + G; d++) {
        const cone = coneWeight(h, d);
        for (let y = 0; y < F.rows; y++) {
          const y0 = y - F.pad + 0.5 + phase, y1 = y0 + d;
          if (Math.max(y0, y1) < -1 || Math.min(y0, y1) > N + 1) continue;
          for (let q = 0; q < N / h; q++) {
            const i = (q * F.rows + y) * nd + d + h + G;
            const start = lv.M[i], own = cone * lv.R[i], end = lv.T[i] * start + own;
            const [ax, ay] = view.world(N, q * h, y0), [bx, by] = view.world(N, (q + 1) * h, y1);
            const o = count++ * STRIDE;
            lines[o] = ax; lines[o + 1] = ay; lines[o + 2] = bx; lines[o + 3] = by;
            lines[o + 4] = start; lines[o + 5] = end; lines[o + 6] = own; lines[o + 7] = cone;
            lines[o + 8] = level; lines[o + 9] = v;
            lines[o + 10] = lv.R[i]; lines[o + 11] = lv.T[i];
            lines[o + 12] = q; lines[o + 13] = y; lines[o + 14] = d; lines[o + 15] = Math.abs(d) > h ? 1 : 0;
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

/** a ray's place in the tree of halves under a rule: its descendants (the halves it is made of, down to length 1) and
 * its ancestors (the rays it is a half of, up to the top), each with the summed product of the pairs' shares on the way
 * (the linear merge rule's per-row shares are left out: these are the rule's fixed shares) */
export function relatives(N: number, id: RayId, rule: Rule) {
  const rows = 3 * N, K = Math.log2(N), pairs = pairTable(N, rule), G = guardOf(rule);
  const key = (r: RayId) => `${r.k} ${r.plane} ${r.row} ${r.d}`;
  const out: { id: RayId; w: number; up: boolean }[] = [];
  const add = (into: Map<string, { id: RayId; w: number }>, r: RayId, w: number) => {
    if (r.row < 0 || r.row >= rows) return;
    const e = into.get(key(r));
    if (e) e.w += w; else into.set(key(r), { id: r, w });
  };
  let level = new Map([[key(id), { id, w: 1 }]]);
  for (let k = id.k; k > 0; k--) {
    const next = new Map<string, { id: RayId; w: number }>(), L = 1 << k;
    for (const { id: p, w } of level.values())
      for (const { dA, a, dj, b, w: s } of pairs.get(L)![p.d + L + G]) {
        add(next, { k: k - 1, plane: 2 * p.plane, row: p.row + dA, d: a }, w * s);
        add(next, { k: k - 1, plane: 2 * p.plane + 1, row: p.row + dj, d: b }, w * s);
      }
    for (const e of next.values()) out.push({ ...e, up: false });
    level = next;
  }
  level = new Map([[key(id), { id, w: 1 }]]);
  for (let k = id.k; k < K - 1; k++) {
    const next = new Map<string, { id: RayId; w: number }>(), L = 2 << k;
    for (const { id: c, w } of level.values()) {
      const second = c.plane % 2 === 1, plane = c.plane >> 1;
      for (let D = -L - G; D <= L + G; D++)
        for (const { dA, a, dj, b, w: s } of pairs.get(L)![D + L + G]) {
          if ((second ? b : a) !== c.d) continue;
          add(next, { k: k + 1, plane, row: c.row - (second ? dj : dA), d: D }, w * s);
        }
    }
    for (const e of next.values()) out.push({ ...e, up: true });
    level = next;
  }
  return out;
}

/** a solid rectangle [x0, x1] × [y0, y1]: a run of light (emitting, radiance 1) or wall (opaque, dark) pixels */
export interface Box { x0: number; y0: number; x1: number; y1: number; light: boolean }

/** pixels as few disjoint rectangles: rows of each kind merged greedily into runs, runs grown downward while whole */
export function boxesOf(N: number, lights: readonly (readonly [number, number])[], walls: readonly (readonly [number, number])[]): Box[] {
  const out: Box[] = [];
  for (const [list, light] of [[lights, true], [walls, false]] as const) {
    const on = new Uint8Array(N * N);
    for (const [x, y] of list) if (x >= 0 && y >= 0 && x < N && y < N) on[y * N + x] = 1;
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        if (!on[y * N + x]) continue;
        let x1 = x;
        while (x1 + 1 < N && on[y * N + x1 + 1]) x1++;
        let y1 = y;
        const whole = (r: number) => { for (let i = x; i <= x1; i++) if (!on[r * N + i]) return false; return true; };
        while (y1 + 1 < N && whole(y1 + 1)) y1++;
        for (let r = y; r <= y1; r++) for (let i = x; i <= x1; i++) on[r * N + i] = 0;
        out.push({ x0: x, y0: y, x1: x1 + 1, y1: y1 + 1, light });
      }
  }
  return out;
}

/** the directions toward the sources each frustum gathers: frustum v carries light toward +x, −x, +y, −y, so it sees
 * the directions (from the receiver) around π, 0, 3π/2, π/2, a quarter turn each */
const FRUSTUM_CENTRE = [Math.PI, 0, 1.5 * Math.PI, 0.5 * Math.PI];

const TAU = 2 * Math.PI;
const wrapAngle = (a: number) => a - TAU * Math.floor(a / TAU);

/** exact fluence at a point among lights (radiance 1) and walls (radiance 0, opaque): the measure of the directions
 * whose first hit is a light, optionally only the directions of the frustums in `frustums` (a bit mask); NaN inside a
 * solid. The circle is cut at every box's silhouette (and the frustum edges); within each piece the set of boxes a ray
 * meets is fixed, and since the boxes are disjoint and convex their order along the ray is fixed too, so the box nearest
 * along the piece's middle direction is the first hit for the whole piece. */
export function exactOccludedAt(px: number, py: number, boxes: readonly Box[], frustums = 0b1111): number {
  const n = boxes.length;
  if (lo.length < n) { lo = new Float64Array(2 * n); hi = new Float64Array(2 * n); }
  /** box i's span of directions [lo, hi] (hi may run past 2π); the point is outside it */
  const span = (i: number) => {
    const b = boxes[i], c = Math.atan2((b.y0 + b.y1) / 2 - py, (b.x0 + b.x1) / 2 - px);
    let l = Infinity, h = -Infinity;
    for (let k = 0; k < 4; k++) {
      let a = Math.atan2((k < 2 ? b.y0 : b.y1) - py, (k & 1 ? b.x1 : b.x0) - px) - c;
      a -= TAU * Math.round(a / TAU);
      if (a < l) l = a;
      if (a > h) h = a;
    }
    lo[i] = wrapAngle(c + l); hi[i] = lo[i] + h - l;
  };
  const overlaps = (i: number, j: number) => {
    for (let k = -1; k <= 1; k++) if (lo[i] < hi[j] + k * TAU && lo[j] + k * TAU < hi[i]) return true;
    return false;
  };
  // the lights, and the square of the farthest distance to any of them
  const keep: number[] = [];
  let far2 = 0;
  for (let i = 0; i < n; i++) {
    const b = boxes[i];
    if (px >= b.x0 && px <= b.x1 && py >= b.y0 && py <= b.y1) return NaN;
    if (!b.light) continue;
    span(i); keep.push(i);
    const fx = Math.max(Math.abs(b.x0 - px), Math.abs(b.x1 - px)), fy = Math.max(Math.abs(b.y0 - py), Math.abs(b.y1 - py));
    far2 = Math.max(far2, fx * fx + fy * fy);
  }
  const nLights = keep.length;
  if (!nLights) return 0;
  // the walls that can matter: nearer than the farthest light, and crossing some light's span
  for (let i = 0; i < n; i++) {
    const b = boxes[i];
    if (b.light) continue;
    const nx = Math.max(b.x0 - px, 0, px - b.x1), ny = Math.max(b.y0 - py, 0, py - b.y1);
    if (nx * nx + ny * ny >= far2) continue;
    span(i);
    for (let j = 0; j < nLights; j++) if (overlaps(i, keep[j])) { keep.push(i); break; }
  }
  const restrict = (frustums & 15) !== 15, cuts: number[] = [0, TAU];
  for (const i of keep) cuts.push(lo[i], wrapAngle(hi[i]));
  if (restrict) for (const f of FRUSTUM_CENTRE) cuts.push(wrapAngle(f - Math.PI / 4));
  cuts.sort((a, b) => a - b);
  let total = 0;
  for (let k = 0; k + 1 < cuts.length; k++) {
    const a = cuts[k], e = cuts[k + 1];
    if (e - a < 1e-12) continue;
    const m = (a + e) / 2;
    if (restrict) {
      // the frustum whose quarter holds m
      let v = -1;
      for (let f = 0; f < 4; f++) {
        let d = m - FRUSTUM_CENTRE[f];
        d -= TAU * Math.round(d / TAU);
        if (Math.abs(d) <= Math.PI / 4) { v = f; break; }
      }
      if (v < 0 || !((frustums >> v) & 1)) continue;
    }
    const dx = Math.cos(m), dy = Math.sin(m);
    let best = Infinity, lit = false;
    for (const i of keep) {
      // is m inside this box's span?
      if (!((m >= lo[i] && m <= hi[i]) || m + TAU <= hi[i])) continue;
      const b = boxes[i];
      // entry distance by slabs; the ray is known to meet the box
      const tx0 = (b.x0 - px) / dx, tx1 = (b.x1 - px) / dx, ty0 = (b.y0 - py) / dy, ty1 = (b.y1 - py) / dy;
      const t = Math.max(Math.min(tx0, tx1), Math.min(ty0, ty1));
      if (t < best) { best = t; lit = b.light; }
    }
    if (lit) total += e - a;
  }
  return total;
}
let lo = new Float64Array(64), hi = new Float64Array(64);
