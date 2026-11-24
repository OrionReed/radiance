// Plain Reciprocal HRC run in a fan of lattice cones (CPU). The page solves on the GPU (./gpu.ts); this is its
// reference, kept for the agreement check and the cost law.
//
// A cone is plain's tree in an integer frame of the pixel lattice: pixel p = X u + Y w with det(u, w) = 1, so (X, Y)
// is integer exactly when p is a pixel. Level L holds rays from plane X = bL to (b + 1)L, a frame ray (L, D) being the
// pixel vector L u + D w, D in [lo L, hi L]. The tree is plain's and never sees the metric: offsets pair floor / ceil
// of D / 2 (both orders at 1/2 when D is odd), R = R_B + T_B R_A and T = T_A T_B, and the merge is its transpose. Only
// the cone weights (the true angle of a ray's bin) and the length-1 rays see the plane.
//
// Length-1 rays join a start pixel S and an end pixel E: T = f(S) f(E), R = e(E) + f(E) e(S). With centre endpoints E
// is S + u + d w (a pixel step); with edge endpoints (plain's frame only) the ray crosses one column, through S and
// S + d w. The readout is the transpose of that injection: 1/2 M of the length-1 rays starting at a pixel plus
// 1/2 f(S) M of those ending at it.
//
// A fan of unimodular cones [g_k, g_k+1] (consecutive generators with det +1) runs each cone with u = g_k,
// w = g_k+1 − g_k, D in [0, L] and centre endpoints. Plain is the exception: four axis frames, D in [−L, L], edge
// endpoints (site/line-field/lines.ts).
//
// Arrays are [x][y] flattened as x * N + y, y up. Outside the image is opaque and dark.

export type V = [number, number];
export const det = (a: V, b: V) => a[0] * b[1] - a[1] * b[0];

export interface Cone {
  u: V;
  w: V;
  lo: number;
  hi: number;
  edge: boolean;
}

export interface FanSpec {
  id: string;
  name: string;
  /** the generators, counter-clockwise; plain lists its frames' corners */
  generators: V[];
  cones: Cone[];
  /** the frames' level lines are drawn from these segments (k u to k v per cone) */
  segments: [V, V][];
  /** one line for the sidebar */
  line: string;
}

function unimodular(generators: V[]): Cone[] {
  return generators.map((u, k) => {
    const v = generators[(k + 1) % generators.length], w: V = [v[0] - u[0], v[1] - u[1]];
    if (det(u, w) !== 1) throw new Error(`cone ${u} → ${v} is not unimodular`);
    return { u, w, lo: 0, hi: 1, edge: false };
  });
}
const pairsOf = (generators: V[]) => generators.map((g, k): [V, V] => [g, generators[(k + 1) % generators.length]]);

const fan = (id: string, name: string, generators: V[], line: string): FanSpec => ({ id, name, generators, cones: unimodular(generators), segments: pairsOf(generators), line });

const PLAIN_FRAMES: Cone[] = [
  { u: [1, 0], w: [0, 1], lo: -1, hi: 1, edge: true },
  { u: [0, 1], w: [-1, 0], lo: -1, hi: 1, edge: true },
  { u: [-1, 0], w: [0, -1], lo: -1, hi: 1, edge: true },
  { u: [0, -1], w: [1, 0], lo: -1, hi: 1, edge: true },
];
const OCTANT: V[] = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
// Farey order 2: the octant fan with every mediant added
const FAREY: V[] = [[1, 0], [2, 1], [1, 1], [1, 2], [0, 1], [-1, 2], [-1, 1], [-2, 1], [-1, 0], [-2, -1], [-1, -1], [-1, -2], [0, -1], [1, -2], [1, -1], [2, -1]];

export const FANS: FanSpec[] = [
  {
    id: 'plain', name: 'plain', generators: OCTANT, cones: PLAIN_FRAMES,
    segments: [[[1, -1], [1, 1]], [[1, 1], [-1, 1]], [[-1, 1], [-1, -1]], [[-1, -1], [1, -1]]],
    line: "Four axis frustums: the octant's directions. Its level is max(|x|, |y|), not W, but edge endpoints count each ray 1 + |t| times, which is W.",
  },
  fan('quadrant', 'quadrant', [[1, 0], [0, 1], [-1, 0], [0, -1]],
    "W's normal fan, with generators of width 1: exact at every light size, at half plain's cones and √2 its widest spacing. The gap-free 1×."),
  fan('octant', 'octant', OCTANT,
    "Plain's directions, probed at pixel centres. (1, 1) has width 2, so a cone counts a pixel ½ to 1 times: about −28 % at a one-pixel light."),
  fan('three', '3 cones', [[1, 0], [0, 1], [-1, -1]],
    'The fewest cones. Two of them bend across an axis, inside the diamond: about +29 % at a one-pixel light.'),
  fan('farey', '16 cones', FAREY,
    'The octant with its mediants: finer directions, but every new generator is off the diamond: about −45 %.'),
];

/** the cone weight of frame ray (L, D): the angle of directions L u + t w, t in [D − 1/2, D + 1/2] ∩ [lo L, hi L] */
export function coneWeight(c: Cone, L: number, D: number) {
  const t0 = Math.max(D - 0.5, c.lo * L), t1 = Math.min(D + 0.5, c.hi * L);
  const a = [L * c.u[0] + t0 * c.w[0], L * c.u[1] + t0 * c.w[1]], b = [L * c.u[0] + t1 * c.w[0], L * c.u[1] + t1 * c.w[1]];
  return Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
}

type Pair = [a: number, b: number, w: number];
const pairs = (D: number): Pair[] => {
  const a = Math.floor(D / 2);
  return D % 2 === 0 ? [[a, a, 1]] : [[a, a + 1, 0.5], [a + 1, a, 0.5]];
};

/** one cone's frame without its pixel map: plane count Xn, rows Yn, and frame point (X, Y) is frame coordinates
 *  (X − off + xmin, Y − padLo + ymin) */
export function frameGeometry(N: number, c: Cone) {
  const { u, w } = c;
  const xp = (i: number, j: number) => w[1] * i - w[0] * j, yp = (i: number, j: number) => -u[1] * i + u[0] * j;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const i of [0, N - 1])
    for (const j of [0, N - 1]) {
      xmin = Math.min(xmin, xp(i, j)); xmax = Math.max(xmax, xp(i, j));
      ymin = Math.min(ymin, yp(i, j)); ymax = Math.max(ymax, yp(i, j));
    }
  const xspan = xmax - xmin + 1;
  let Xn = 2;
  while (Xn < xspan) Xn *= 2;
  // centre endpoints: planes one past the image's first when there is room, so that opposite cones share planes
  const off = !c.edge && xspan < Xn ? 1 : 0;
  const padLo = Xn * Math.max(0, c.hi), padHi = Xn * Math.max(0, -c.lo), Yn = ymax - ymin + 1 + padLo + padHi;
  return { Xn, Yn, off, padLo, xmin, ymin };
}

/** one cone's frame: plane count, rows, and the pixel at each frame point */
export function frameOf(N: number, c: Cone) {
  const { u, w } = c, g = frameGeometry(N, c), { Xn, Yn, off, padLo, xmin, ymin } = g;
  const pix = new Int32Array((Xn + 1) * Yn).fill(-1);
  for (let X = 0; X <= Xn; X++)
    for (let Y = 0; Y < Yn; Y++) {
      const x = X - off + xmin, y = Y - padLo + ymin, i = x * u[0] + y * w[0], j = x * u[1] + y * w[1];
      if (i >= 0 && j >= 0 && i < N && j < N) pix[X * Yn + Y] = i * N + j;
    }
  return { ...g, pix };
}

/** stored rays of one cone, all levels (including the padding rows) */
export function storedRays(N: number, c: Cone) {
  const { Xn, Yn } = frameGeometry(N, c);
  let n = 0;
  for (let L = 1; L <= Xn / 2; L *= 2) n += (Xn / L) * Yn * ((c.hi - c.lo) * L + 1);
  return n;
}

/** adds one cone's fluence (rgb) to out[3 (x N + y)] (not yet masked by the free field) */
export function solveCone(N: number, emission: Float32Array, free: Float32Array, c: Cone, out: Float64Array) {
  const { Xn, Yn, pix } = frameOf(N, c);
  const span = c.hi - c.lo, nd = (L: number) => span * L + 1, base = (L: number) => c.lo * L;
  const at = (L: number, b: number, y: number, D: number) => (b * Yn + y) * nd(L) + D - base(L);
  const fAt = (X: number, Y: number) => { const k = Y >= 0 && Y < Yn ? pix[X * Yn + Y] : -1; return k < 0 ? 0 : free[k]; };
  const eAt = (X: number, Y: number, ch: number) => { const k = Y >= 0 && Y < Yn ? pix[X * Yn + Y] : -1; return k < 0 ? 0 : emission[3 * k + ch]; };
  const endX = (X: number) => (c.edge ? X : X + 1);

  const T = new Map<number, Float32Array>(), R = new Map<number, Float32Array>();
  {
    const n = Xn * Yn * nd(1), t = new Float32Array(n), r = new Float32Array(3 * n);
    for (let X = 0; X < Xn; X++)
      for (let Y = 0; Y < Yn; Y++) {
        const fS = fAt(X, Y);
        for (let d = c.lo; d <= c.hi; d++) {
          const XE = endX(X), YE = Y + d, fE = fAt(XE, YE), o = at(1, X, Y, d);
          t[o] = fS * fE;
          for (let ch = 0; ch < 3; ch++) r[3 * o + ch] = eAt(XE, YE, ch) + fE * eAt(X, Y, ch);
        }
      }
    T.set(1, t); R.set(1, r);
  }
  // extension
  for (let L = 2; L <= Xn / 2; L *= 2) {
    const h = L / 2, P = Xn / L, Th = T.get(h)!, Rh = R.get(h)!, n = P * Yn * nd(L), t = new Float32Array(n), r = new Float32Array(3 * n);
    for (let D = c.lo * L; D <= c.hi * L; D++)
      for (const [a, bb, wt] of pairs(D))
        for (let b = 0; b < P; b++)
          for (let y = 0; y < Yn; y++) {
            if (y + a < 0 || y + a >= Yn) continue;
            const A = at(h, 2 * b, y, a), B = at(h, 2 * b + 1, y + a, bb), o = at(L, b, y, D), tB = Th[B];
            t[o] += wt * Th[A] * tB;
            r[3 * o] += wt * (Rh[3 * B] + tB * Rh[3 * A]);
            r[3 * o + 1] += wt * (Rh[3 * B + 1] + tB * Rh[3 * A + 1]);
            r[3 * o + 2] += wt * (Rh[3 * B + 2] + tB * Rh[3 * A + 2]);
          }
    T.set(L, t); R.set(L, r);
  }
  // merge, longest first: the top level (length Xn) has M = 0
  let Mp = new Float32Array(3 * Yn * nd(Xn));
  for (let P = Xn; P >= 2; P /= 2) {
    const L = P / 2, Th = T.get(L)!, Rh = R.get(L)!, M = new Float32Array(3 * (Xn / L) * Yn * nd(L));
    for (let D = c.lo * P; D <= c.hi * P; D++) {
      const cw = coneWeight(c, P, D);
      for (const [a, bb, wt] of pairs(D))
        for (let b = 0; b < Xn / P; b++)
          for (let y = 0; y < Yn; y++) {
            const F = 3 * at(P, b, y, D), A = at(L, 2 * b, y, a);
            const f0 = Mp[F], f1 = Mp[F + 1], f2 = Mp[F + 2];
            M[3 * A] += wt * f0; M[3 * A + 1] += wt * f1; M[3 * A + 2] += wt * f2;
            if (y + a < 0 || y + a >= Yn) continue;
            const B = 3 * at(L, 2 * b + 1, y + a, bb), tA = Th[A];
            M[B] += wt * (tA * f0 + cw * Rh[3 * A]);
            M[B + 1] += wt * (tA * f1 + cw * Rh[3 * A + 1]);
            M[B + 2] += wt * (tA * f2 + cw * Rh[3 * A + 2]);
          }
    }
    Mp = M;
    T.delete(L); R.delete(L);
  }
  // readout: the transpose of the length-1 injection
  for (let X = 0; X < Xn; X++)
    for (let Y = 0; Y < Yn; Y++) {
      const S = pix[X * Yn + Y], fS = S < 0 ? 0 : free[S];
      for (let d = c.lo; d <= c.hi; d++) {
        const o = 3 * at(1, X, Y, d), YE = Y + d, E = YE >= 0 && YE < Yn ? pix[endX(X) * Yn + YE] : -1;
        for (let ch = 0; ch < 3; ch++) {
          const m = Mp[o + ch];
          if (S >= 0) out[3 * S + ch] += 0.5 * m;
          if (E >= 0) out[3 * E + ch] += 0.5 * fS * m;
        }
      }
    }
}

/** the fan's fluence per pixel as (rgb, luminance), [x N + y]; solid pixels (lights too) read 0 */
export function solveFan(N: number, emission: Float32Array, solid: ArrayLike<number>, cones: Cone[]) {
  const free = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) free[i] = solid[i] || emission[3 * i] > 0 || emission[3 * i + 1] > 0 || emission[3 * i + 2] > 0 ? 0 : 1;
  const acc = new Float64Array(3 * N * N);
  for (const c of cones) solveCone(N, emission, free, c, acc);
  const out = new Float32Array(4 * N * N);
  for (let i = 0; i < N * N; i++) {
    const r = acc[3 * i] * free[i], g = acc[3 * i + 1] * free[i], b = acc[3 * i + 2] * free[i];
    out[4 * i] = r; out[4 * i + 1] = g; out[4 * i + 2] = b;
    out[4 * i + 3] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return out;
}

/** The cost law (derived): rays per pixel per octave = Σ over cones of ∫ dθ / s(θ)², s(θ) = |det(w, θ̂)| the spacing of
 *  a direction's parallel rays; merit = cost · s_max² / 2π ≥ 1. */
export function costLaw(f: FanSpec, steps = 4096) {
  let cost = 0, smax = 0;
  for (const c of f.cones) {
    const a0 = Math.atan2(c.u[1] + c.lo * c.w[1], c.u[0] + c.lo * c.w[0]);
    const span = angleOf(c);
    for (let k = 0; k < steps; k++) {
      const th = a0 + ((k + 0.5) / steps) * span, s = Math.abs(c.w[0] * Math.sin(th) - c.w[1] * Math.cos(th));
      cost += span / steps / (s * s);
      smax = Math.max(smax, s);
    }
  }
  return { cost, smax, merit: (cost * smax * smax) / (2 * Math.PI) };
}
const angleOf = (c: Cone) => {
  const a = [c.u[0] + c.lo * c.w[0], c.u[1] + c.lo * c.w[1]], b = [c.u[0] + c.hi * c.w[0], c.u[1] + c.hi * c.w[1]];
  return Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
};
