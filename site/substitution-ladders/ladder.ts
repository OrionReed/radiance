// Reciprocal HRC on a substitution ladder, with the Christoffel law or box windows at its junctions.
//
// A substitution σ sends each letter to a word (Fibonacci: a → ab, b → a). The ladder of depth n cuts the top tile,
// letter a of length N = |σⁿ(a)|, into the tiles of σ(a), one level down, and so on: a tile of letter x at level i has
// length |σⁱ(x)| and is cut at the interior planes s_c (the partial lengths of σ(x) one level down). Level 0 is the
// pixel. Binary a → aa is plain reciprocal HRC; a letter whose image is one letter is a copy (one child, no cut).
//
// A law gives the paths of a parent of length P and rise D: the tuples of junction rows j_c at its cuts, and weights.
//   christoffel: the parent's own line, slope D / P, through a uniform intercept U of its start row:
//                j_c = floor(s_c D / P + U). Exact over U: a path per interval of U where the tuple is constant.
//   box:         a line from a uniform point of the start row to a uniform point of the end row (rise D + v − u):
//                j_c = floor((1 − t_c) u + t_c (D + v)), t_c = s_c / P, weighted by exact areas in (u, v).
// Both are plain at the midpoint of radix 2.
//
// Extension, merge, c0, cone weights, the four frustum views, padding and readout are lab 4 deep-3's (fam.ts), which
// are plain's (site/line-field/lines.ts). The merge reads the same path objects as the extension, so the transport is
// reciprocal by construction. Arrays are [x][y] flattened as x · N + y.

// ── substitutions ──

export interface Substitution {
  /** letters in order; the first is the top tile's */
  letters: string[];
  images: Map<string, string>;
}

/** "a → ab, b → a" (also ->, =, :; separated by commas, semicolons or new lines). Letters are single characters. */
export function parse(text: string): Substitution | string {
  const parts = text.split(/[,;\n]+/).map((s) => s.trim()).filter(Boolean);
  if (!parts.length) return 'empty rule';
  const letters: string[] = [], images = new Map<string, string>();
  for (const part of parts) {
    const m = part.match(/^(\S)\s*(?:→|->|=|:|↦)\s*(\S+)$/u);
    if (!m) return `cannot read “${part}”: write x → word`;
    const [, x, w] = m;
    if (images.has(x)) return `${x} has two rules`;
    letters.push(x);
    images.set(x, w);
  }
  for (const [x, w] of images) for (const y of w) if (!images.has(y)) return `${x} → ${w}: ${y} has no rule`;
  return { letters, images };
}

export const format = (s: Substitution) => s.letters.map((x) => `${x} → ${s.images.get(x)}`).join(', ');

/** incidence matrix: M[x][y] = number of y in σ(x) */
export function incidence(s: Substitution) {
  const ix = new Map(s.letters.map((x, i) => [x, i]));
  return s.letters.map((x) => {
    const row = s.letters.map(() => 0);
    for (const y of s.images.get(x)!) row[ix.get(y)!]++;
    return row;
  });
}

/** primitive: some power of the incidence matrix is positive (Wielandt: power (n − 1)² + 1 suffices) */
export function primitive(M: number[][]) {
  const n = M.length;
  let P: number[][] = M.map((r) => r.map((v) => (v > 0 ? 1 : 0)));
  const B = P.map((r) => [...r]);
  for (let k = 1; k <= (n - 1) ** 2 + 1; k++) {
    if (P.every((r) => r.every((v) => v > 0))) return true;
    P = P.map((r) => B[0].map((_, j) => (r.some((v, i) => v && B[i][j]) ? 1 : 0)));
  }
  return false;
}

type C = [number, number];
const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a: C, b: C): C => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };

/** eigenvalues of M: characteristic polynomial by Faddeev–LeVerrier, roots by Durand–Kerner */
export function eigenvalues(M: number[][]): C[] {
  const n = M.length;
  // c[k]: coefficient of λ^(n − k), c[0] = 1
  const c = [1];
  // M_1 = M, M_k = M (M_{k−1} + c_{k−1} I), c_k = −tr(M_k) / k
  let Mk: number[][] = M.map((r) => r.map(() => 0));
  for (let k = 1; k <= n; k++) {
    const S = Mk.map((r, i) => r.map((v, j) => v + (i === j ? c[k - 1] : 0)));
    Mk = M.map((r) => S[0].map((_, j) => r.reduce((s, v, i) => s + v * S[i][j], 0)));
    c.push(-Mk.reduce((s, r, i) => s + r[i], 0) / k);
  }
  const p = (z: C): C => c.reduce<C>((acc, ck) => { const m = cmul(acc, z); return [m[0] + ck, m[1]]; }, [0, 0]);
  let roots: C[] = Array.from({ length: n }, (_, i) => { const r = 1 + Math.max(...c.map(Math.abs)); const a = (2 * Math.PI * i) / n + 0.4; return [r * 0.5 * Math.cos(a), r * 0.5 * Math.sin(a)]; });
  for (let it = 0; it < 500; it++) {
    roots = roots.map((z, i) => {
      let den: C = [1, 0];
      roots.forEach((w, j) => { if (j !== i) den = cmul(den, [z[0] - w[0], z[1] - w[1]]); });
      const q = cdiv(p(z), den);
      return [z[0] - q[0], z[1] - q[1]];
    });
  }
  return roots;
}

export interface Analysis {
  primitive: boolean;
  /** the Perron root: the ratio of successive lengths */
  lambda: number;
  /** the largest modulus of the other eigenvalues */
  second: number;
  /** λ > 1 and every other eigenvalue inside the unit circle */
  pisot: boolean;
}
export function analyse(s: Substitution): Analysis {
  const M = incidence(s), ev = eigenvalues(M).map(([a, b]) => Math.hypot(a, b)).sort((a, b) => b - a);
  const lambda = ev[0], second = ev.length > 1 ? ev[1] : 0;
  return { primitive: primitive(M), lambda, second, pisot: lambda > 1 + 1e-9 && second < 1 - 1e-6 };
}

/** |σⁱ(x)| for every letter, i = 0 … depth */
export function lengths(s: Substitution, depth: number) {
  const out: Map<string, number>[] = [new Map(s.letters.map((x) => [x, 1]))];
  for (let i = 1; i <= depth; i++) {
    const prev = out[i - 1];
    out.push(new Map(s.letters.map((x) => [x, [...s.images.get(x)!].reduce((a, y) => a + prev.get(y)!, 0)])));
  }
  return out;
}

/** the top lengths |σⁿ(a)| within [lo, hi], with their depth n */
export function tops(s: Substitution, lo: number, hi: number) {
  const out: { N: number; depth: number }[] = [];
  const ls = lengths(s, 64);
  for (let n = 1; n < ls.length; n++) {
    const N = ls[n].get(s.letters[0])!;
    if (N > hi) break;
    if (N >= lo && N !== out.at(-1)?.N) out.push({ N, depth: n });
  }
  return out;
}

// ── the ladder ──

/** junction rows j_0 = 0, j_1 … j_{k−1}, j_k = D, relative to the parent's start row; w its share */
export interface Path { j: number[]; w: number }
export type Law = 'christoffel' | 'box';

export interface TileClass {
  level: number;
  letter: string;
  L: number;
  /** start columns of its tiles */
  xs: number[];
  /** child classes and, per tile, its children's tile indices */
  kids: number[];
  kidIndex: Int32Array;
  /** interior cut planes s_1 … s_{k−1}, relative to the tile's start */
  cuts: number[];
  /** paths per rise, [D + L] */
  paths: Path[][];
}

export interface Ladder {
  N: number;
  depth: number;
  substitution: Substitution;
  law: Law;
  /** children before parents; the last is the top tile */
  classes: TileClass[];
}

const cone = (L: number, d: number) => { const c = (v: number) => Math.max(-1, Math.min(1, v)); return Math.atan(c((d + 0.5) / L)) - Math.atan(c((d - 0.5) / L)); };

/** Christoffel law at integer cuts s of a parent of length P: exact over the intercept U */
export function christoffel(P: number, D: number, s: number[]): Path[] {
  // cut c steps up where U crosses 1 − frac(s_c D / P), a multiple of 1 / P: thresholds in units of 1 / P
  const th = new Set<number>([0, P]);
  for (const sc of s) { const r = (((sc * D) % P) + P) % P; if (r) th.add(P - r); }
  const t = [...th].sort((a, b) => a - b), out: Path[] = [];
  for (let i = 0; i + 1 < t.length; i++) {
    const mid = (t[i] + t[i + 1]) / 2;   // U · P, never on a threshold
    out.push({ j: [0, ...s.map((sc) => Math.floor((sc * D + mid) / P)), D], w: (t[i + 1] - t[i]) / P });
  }
  return out;
}

type Poly = [number, number][];
const area = (p: Poly) => { let a = 0; for (let i = 0; i < p.length; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length]; a += x0 * y1 - y0 * x1; } return Math.abs(a) / 2; };
/** clip a convex polygon to a u + b v − c ≥ 0 (keep 1) or < 0 (keep −1) */
function clip(p: Poly, a: number, b: number, c: number, keep: 1 | -1): Poly {
  const out: Poly = [], g = (q: [number, number]) => keep * (a * q[0] + b * q[1] - c);
  for (let i = 0; i < p.length; i++) {
    const P = p[i], Q = p[(i + 1) % p.length], gp = g(P), gq = g(Q);
    if (gp >= 0) out.push(P);
    if ((gp >= 0) !== (gq >= 0)) { const k = gp / (gp - gq); out.push([P[0] + k * (Q[0] - P[0]), P[1] + k * (Q[1] - P[1])]); }
  }
  return out;
}
/** box windows at cut fractions t (lab 4 ladders' pathsOf): start u and end v uniform in their rows */
export function box(D: number, t: number[]): Path[] {
  let faces: { poly: Poly; j: number[] }[] = [{ poly: [[0, 0], [1, 0], [1, 1], [0, 1]], j: [0] }];
  for (const tc of t) {
    // g = (1 − tc) u + tc v + tc D in [base, base + 1): floor takes m or m + 1, split at g = m + 1
    const base = tc * D, m = Math.floor(base + 1e-12), next: typeof faces = [];
    for (const f of faces) {
      if (Math.abs(base - Math.round(base)) < 1e-12) { next.push({ poly: f.poly, j: [...f.j, Math.round(base)] }); continue; }
      const lo = clip(f.poly, 1 - tc, tc, m + 1 - base, -1), hi = clip(f.poly, 1 - tc, tc, m + 1 - base, 1);
      if (lo.length > 2 && area(lo) > 1e-14) next.push({ poly: lo, j: [...f.j, m] });
      if (hi.length > 2 && area(hi) > 1e-14) next.push({ poly: hi, j: [...f.j, m + 1] });
    }
    faces = next;
  }
  const out = faces.map((f) => ({ j: [...f.j, D], w: area(f.poly) }));
  const sum = out.reduce((a, p) => a + p.w, 0);
  for (const p of out) p.w /= sum;
  return out;
}

export function lawPaths(law: Law, P: number, D: number, s: number[]): Path[] {
  if (!s.length) return [{ j: [0, D], w: 1 }];
  // at the midpoint of radix 2 the laws agree; use the exact table so binary is plain to the bit
  if (law === 'christoffel' || (s.length === 1 && 2 * s[0] === P)) return christoffel(P, D, s);
  return box(D, s.map((sc) => sc / P));
}

export function buildLadder(sub: Substitution, depth: number, law: Law): Ladder {
  const ls = lengths(sub, depth), byKey = new Map<string, number>(), classes: (TileClass & { idx: Map<number, number>; kidLists: number[][] })[] = [];
  const classOf = (level: number, letter: string) => {
    const key = level === 0 ? '0' : `${level} ${letter}`;
    let id = byKey.get(key);
    if (id === undefined) {
      id = classes.length;
      byKey.set(key, id);
      const L = ls[level].get(letter)!, word = level ? sub.images.get(letter)! : '';
      const cuts: number[] = [];
      let at = 0;
      for (let c = 0; c + 1 < word.length; c++) cuts.push((at += ls[level - 1].get(word[c])!));
      classes.push({ level, letter: level ? letter : '·', L, xs: [], kids: [], kidIndex: new Int32Array(0), cuts, paths: [], idx: new Map(), kidLists: [] });
    }
    return id;
  };
  const visit = (x: number, level: number, letter: string): [number, number] => {
    const id = classOf(level, letter), c = classes[id], ti = c.xs.length;
    c.xs.push(x);
    c.idx.set(x, ti);
    if (level > 0) {
      const word = sub.images.get(letter)!, list: number[] = [], kinds: number[] = [];
      let at = x;
      for (const y of word) {
        const [kid, kti] = visit(at, level - 1, y);
        kinds.push(kid); list.push(kti);
        at += ls[level - 1].get(y)!;
      }
      c.kids = kinds;
      c.kidLists.push(list);
    }
    return [id, ti];
  };
  visit(0, depth, sub.letters[0]);
  const N = ls[depth].get(sub.letters[0])!;
  for (const c of classes) {
    c.kidIndex = Int32Array.from(c.kidLists.flat());
    if (c.level === 0) continue;
    c.paths = Array.from({ length: 2 * c.L + 1 }, (_, q) => lawPaths(law, c.L, q - c.L, c.cuts));
    for (const ps of c.paths) for (const p of ps) for (let k = 0; k < c.kids.length; k++)
      if (Math.abs(p.j[k + 1] - p.j[k]) > classes[c.kids[k]].L) throw new Error(`child rise out of range: ${c.letter} at level ${c.level}, ${JSON.stringify(p)}`);
  }
  // children before parents: by level (a class's children are one level down)
  const order = classes.map((_, i) => i).sort((a, b) => classes[a].level - classes[b].level);
  const remap = new Map(order.map((o, i) => [o, i]));
  const sorted = order.map((o) => { const { idx: _i, kidLists: _k, ...c } = classes[o]; return { ...c, kids: c.kids.map((k) => remap.get(k)!) }; });
  return { N, depth, substitution: sub, law, classes: sorted };
}

// ── the solve ──
// The page solves on the GPU (./gpu.ts, the same steps from tables). This CPU solve is its reference.

/** one frustum toward +x: per channel, the fluence of its rays at each pixel (frustum layout [x][y]) */
function frustum(lad: Ladder, emissions: Float32Array[], free: Float32Array): Float32Array[] {
  const N = lad.N, pad = N, rows = N + 2 * pad, cls = lad.classes, top = cls.length - 1;
  const f = (x: number, y: number) => (y < pad || y >= pad + N ? 0 : free[x * N + y - pad]);
  const size = (c: TileClass) => c.xs.length * rows * (2 * c.L + 1);
  const at = (L: number, tile: number, row: number, d: number) => (tile * rows + row) * (2 * L + 1) + d + L;

  // T: geometry only, shared by the channels
  const T: Float32Array[] = cls.map((c, i) => new Float32Array(i === top ? 0 : size(c)));
  {
    const T1 = T[0];
    for (let x = 0; x < N; x++) for (let y = 0; y < rows; y++) for (let d = -1; d <= 1; d++) T1[at(1, x, y, d)] = f(x, y) * f(x, y + d);
  }
  for (let i = 1; i < top; i++) {
    const c = cls[i], t = T[i], k = c.kids.length, L = c.L;
    for (let ti = 0; ti < c.xs.length; ti++)
      for (let D = -L; D <= L; D++)
        for (const { j, w } of c.paths[D + L])
          for (let y = 0; y < rows; y++) {
            let Tw = w, ok = true;
            for (let q = 0; q < k; q++) {
              const yc = y + j[q];
              if (yc < 0 || yc >= rows) { ok = false; break; }
              const kc = cls[c.kids[q]];
              Tw *= T[c.kids[q]][at(kc.L, c.kidIndex[ti * k + q], yc, j[q + 1] - j[q])];
            }
            if (ok) t[at(L, ti, y, D)] += Tw;
          }
  }

  return emissions.map((emission) => {
    const e = (x: number, y: number) => (y < pad || y >= pad + N ? 0 : emission[x * N + y - pad]);
    const R: Float32Array[] = cls.map((c, i) => new Float32Array(i === top ? 0 : size(c)));
    const R1 = R[0];
    for (let x = 0; x < N; x++) for (let y = 0; y < rows; y++) for (let d = -1; d <= 1; d++) R1[at(1, x, y, d)] = e(x, y + d) + f(x, y + d) * e(x, y);
    for (let i = 1; i < top; i++) {
      const c = cls[i], r = R[i], k = c.kids.length, L = c.L;
      for (let ti = 0; ti < c.xs.length; ti++)
        for (let D = -L; D <= L; D++)
          for (const { j, w } of c.paths[D + L])
            for (let y = 0; y < rows; y++) {
              let Ra = 0, ok = true;
              for (let q = 0; q < k; q++) {
                const yc = y + j[q];
                if (yc < 0 || yc >= rows) { ok = false; break; }
                const kc = cls[c.kids[q]], C = at(kc.L, c.kidIndex[ti * k + q], yc, j[q + 1] - j[q]);
                Ra = R[c.kids[q]][C] + T[c.kids[q]][C] * Ra;
              }
              if (ok) r[at(L, ti, y, D)] += w * Ra;
            }
    }
    // merge, parents first: child q gets the light arriving through children 0 … q − 1 and their own light at the
    // parent's angle
    const M: Float32Array[] = cls.map((c) => new Float32Array(size(c)));
    for (let i = top; i >= 1; i--) {
      const c = cls[i], G = M[i], k = c.kids.length, P = c.L;
      for (let ti = 0; ti < c.xs.length; ti++)
        for (let D = -P; D <= P; D++) {
          const cw = cone(P, D);
          for (const { j, w } of c.paths[D + P])
            for (let y = 0; y < rows; y++) {
              const far = G[at(P, ti, y, D)];
              let Tw = w, Ra = 0;
              for (let q = 0; q < k; q++) {
                const yc = y + j[q];
                if (yc < 0 || yc >= rows) break;
                const kid = c.kids[q], kc = cls[kid], C = at(kc.L, c.kidIndex[ti * k + q], yc, j[q + 1] - j[q]);
                M[kid][C] += q === 0 ? w * far : Tw * far + w * cw * Ra;
                Ra = R[kid][C] + T[kid][C] * Ra;
                Tw *= T[kid][C];
              }
            }
        }
    }
    // readout: per rise, the two length-1 rays whose middles bracket the pixel centre
    const M1 = M[0], out = new Float32Array(N * N);
    for (let x = 0; x < N; x++)
      for (let y = pad; y < pad + N; y++) {
        let sum = 0;
        for (let d = -1; d <= 1; d++) {
          sum += 0.5 * M1[at(1, x, y, d)];
          const ys = y - d;
          if (ys >= 0 && ys < rows) sum += 0.5 * f(x, ys) * M1[at(1, x, ys, d)];
        }
        out[x * N + y - pad] = sum * f(x, y);
      }
    return out;
  });
}

const views = [
  (N: number, x: number, y: number) => x * N + y,
  (N: number, x: number, y: number) => (N - 1 - x) * N + y,
  (N: number, x: number, y: number) => y * N + x,
  (N: number, x: number, y: number) => y * N + (N - 1 - x),
];

/** fluence per pixel per channel: emission is channels × N² (pixel i at i), solid N² (a pixel with emission is solid) */
export function solveChannels(lad: Ladder, emissions: Float32Array[], solid: ArrayLike<number>): Float32Array[] {
  const N = lad.N, free = new Float32Array(N * N).map((_, i) => (solid[i] ? 0 : 1));
  const fluence = emissions.map(() => new Float32Array(N * N));
  for (const src of views) {
    const fr = new Float32Array(N * N), at = new Int32Array(N * N);
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) { const i = src(N, x, y); at[x * N + y] = i; fr[x * N + y] = free[i]; }
    const ems = emissions.map((em) => { const v = new Float32Array(N * N); for (let i = 0; i < N * N; i++) v[i] = em[at[i]]; return v; });
    const F = frustum(lad, ems, fr);
    F.forEach((Fc, ch) => { for (let i = 0; i < N * N; i++) fluence[ch][at[i]] += Fc[i]; });
  }
  return fluence;
}

/** (rgb, luminance) fluence per pixel from rgb emission (3 per pixel); a grey scene is solved once */
export function solveRGB(lad: Ladder, emission: Float32Array, solid: ArrayLike<number>): Float32Array {
  const n = lad.N * lad.N, ch = [0, 1, 2].map((c) => { const v = new Float32Array(n); for (let i = 0; i < n; i++) v[i] = emission[3 * i + c]; return v; });
  let grey = true;
  for (let i = 0; i < n && grey; i++) grey = ch[0][i] === ch[1][i] && ch[1][i] === ch[2][i];
  const F = grey ? solveChannels(lad, [ch[0]], solid) : solveChannels(lad, ch, solid);
  const [r, g, b] = grey ? [F[0], F[0], F[0]] : F, out = new Float32Array(4 * n);
  for (let i = 0; i < n; i++) {
    out[4 * i] = r[i]; out[4 * i + 1] = g[i]; out[4 * i + 2] = b[i];
    out[4 * i + 3] = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
  }
  return out;
}

export const PRESETS: { id: string; name: string; rule: string; title: string }[] = [
  { id: 'binary', name: 'binary', rule: 'a → aa', title: 'radix 2: plain reciprocal HRC, lengths 2ⁿ' },
  { id: 'ternary', name: 'ternary', rule: 'a → aaa', title: 'radix 3, lengths 3ⁿ' },
  { id: 'fibonacci', name: 'Fibonacci', rule: 'a → ab, b → a', title: 'lengths 1, 2, 3, 5, 8, …; λ = φ' },
  { id: 'tribonacci', name: 'tribonacci', rule: 'a → ab, b → ac, c → a', title: 'lengths 1, 2, 4, 7, 13, …; λ ≈ 1.839' },
  { id: 'silver', name: 'metallic 2', rule: 'a → aab, b → a', title: 'the silver mean: a → aᵐb with m = 2; λ = 1 + √2' },
  { id: 'bronze', name: 'metallic 3', rule: 'a → aaab, b → a', title: 'the bronze mean: a → aᵐb with m = 3; λ = (3 + √13) / 2' },
  { id: 'plastic', name: 'plastic', rule: 'a → bc, b → c, c → a', title: 'the plastic ratio λ ≈ 1.325, the smallest Pisot number; lengths 1, 2, 3, 4, 5, 7, 9, 12, …' },
  { id: 'mixed', name: 'φ², mixed', rule: 'a → aab, b → ab', title: 'three children per a, two per b, no copies; λ = φ², lengths 1, 3, 8, 21, 55, 144' },
];
