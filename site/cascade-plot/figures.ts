// The figures: each answers one question, owns its settings and draws onto a Diagram.
import { Diagram, risesAt, spacingOf, type Pt, type Stroke, type Pen } from './plot';
import { splits } from '../wide-rays/wide';
import { pairFlows, exactAngle, type Flows } from './paths';
import { defineFigure, clampInt, PROJ, projOptions, marksOption, dirsOption, sizeOption, type Figure, type Option, type Proj, type Marks, type Dirs, type Size } from './figure';
import { extension, merge, forms } from './flow';
import { dither } from './junctions';

type Common = Proj & Marks & Dirs;
const COMMON: Common = { ...PROJ, marks: 'probes', dirs: 'all' };
const common = <T extends Common>(): Option<T>[] => [dirsOption<T>(), ...projOptions<T>(), marksOption<T>()];

/** a ray: its line, and for a wide end the two edges of its fan */
function ray(D: Diagram, s: number, L: number, x0: number, u: number, d: number, sp: number, wide: boolean, stroke: Stroke, pen: Pen = 'ink') {
  D.line(s, [x0, u + 0.5], [x0 + L, u + 0.5 + d], stroke, pen);
  if (wide && sp > 1) {
    D.line(s, [x0, u], [x0 + L, u + 0.5 + d - sp / 2], 'fine', pen);
    D.line(s, [x0, u + 1], [x0 + L, u + 0.5 + d + sp / 2], 'fine', pen);
  }
}

const cascades = defineFigure<Common & Size>({
  id: 'cascades',
  name: 'Cascades',
  question: 'What rays does each cascade have?',
  about: 'Every ray of every cascade, probe to probe. Cascade cK has rays of length 2^K; wide rays are drawn by their centre line.',
  defaults: { ...COMMON, n: 8 },
  options: [sizeOption(), ...common()],
  N: (s) => s.n,
  sheets: (c) => c.K + 1,
  draw(D, { N, K, sch }) {
    for (let k = 0; k <= K; k++) {
      const L = 1 << k, sp = spacingOf(sch, L);
      D.frame(k, L, `c${k}`);
      for (let b = 0; b < N / L; b++)
        for (let u = 0; u < N; u++)
          for (const d of risesAt(L, sp)) if (u + d >= 0 && u + d < N) D.line(k, [b * L, u + 0.5], [(b + 1) * L, u + 0.5 + d], 'thin');
    }
  },
});

const halves = defineFigure<Common & Size & { level: number; block: number; start: number; rise: number }>({
  id: 'halves',
  name: 'Halves',
  question: 'Which pairs of halves is one ray built from, and with what weights?',
  about: 'One ray (red) and the pairs of halves it is built from, one cascade down. Line weight shows the weight: normal 3/4 or more, thin 1/4 to 3/4, dashed below. Rings mark junction probes. The merge hands light back down the same pairs with the same weights.',
  defaults: { ...COMMON, n: 8, level: 2, block: 0, start: 2, rise: 5 },
  options: [
    sizeOption(), dirsOption(),
    { key: 'level', label: 'Cascade', kind: 'slider', min: 1, max: (s) => Math.log2(s.n), step: 1 },
    { key: 'block', label: 'Interval', kind: 'slider', min: 0, max: (s) => s.n / (1 << Math.max(1, Math.min(Math.log2(s.n), s.level))) - 1, step: 1 },
    { key: 'start', label: 'Start row', kind: 'slider', min: 0, max: (s) => s.n - 1, step: 1 },
    { key: 'rise', label: 'Direction', kind: 'slider', min: 0, max: (s) => 2 * (1 << Math.max(1, Math.min(Math.log2(s.n), s.level))), step: 1 },
    ...projOptions(), marksOption(),
  ],
  N: (s) => s.n,
  sheets: () => 2,
  draw(D, { N, K, sch, s }) {
    const k = clampInt(s.level, 1, K), L = 1 << k, h = L / 2, sP = spacingOf(sch, L), sC = spacingOf(sch, h), wide = sch.kind === 'wide';
    const R = risesAt(L, sP), d = R[clampInt(s.rise, 0, R.length - 1)], b = clampInt(s.block, 0, N / L - 1), u = clampInt(s.start, 0, N - 1);
    D.frame(0, h, `c${k - 1}`);
    D.frame(1, L, `c${k}`);
    const x0 = b * L, xm = x0 + h;
    ray(D, 1, L, x0, u, d, sP, wide, 'normal', 'red');
    D.pin(1, 0, x0, u + 0.5); D.pin(1, 0, x0 + L, u + 0.5 + d); D.pin(1, 0, xm, u + 0.5 + d / 2);
    const H = risesAt(h, sC);
    for (const [a, j, bb, w] of splits(L, d, sP, sC, wide ? 'wide' : 'one row')) {
      if (!H.includes(a) || !H.includes(bb)) continue;
      const stroke: Stroke = w >= 0.75 ? 'normal' : w >= 0.25 ? 'thin' : 'dashed';
      ray(D, 0, h, x0, u, a, sC, wide, stroke);
      ray(D, 0, h, xm, u + j, bb, sC, wide, stroke);
      D.ring(0, [xm, u + j + 0.5]);
    }
  },
});

const coverage = defineFigure<Common & Size & { y0: number; y1: number }>({
  id: 'coverage',
  name: 'Coverage',
  question: 'Which ray holds a given line in each cascade, and where are the cracks?',
  about: 'One line (thin) and, in every cascade and interval, the ray that holds it (normal). Red dashes: an interval where no ray holds the line, a crack.',
  defaults: { ...COMMON, n: 8, y0: 1.3, y1: 4.6 },
  options: [
    sizeOption(), dirsOption(),
    { key: 'y0', label: 'Line starts at row', kind: 'slider', min: 0, max: (s) => s.n - 0.01, step: 0.01, format: (v) => v.toFixed(2) },
    { key: 'y1', label: 'Line ends at row', kind: 'slider', min: 0, max: (s) => s.n - 0.01, step: 0.01, format: (v) => v.toFixed(2) },
    ...projOptions(), marksOption(),
  ],
  N: (s) => s.n,
  sheets: (c) => c.K + 1,
  draw(D, { N, K, sch, s }) {
    const y0 = Math.min(s.y0, N - 0.01), y1 = Math.min(s.y1, N - 0.01), y = (x: number) => y0 + ((y1 - y0) * x) / N;
    for (let k = 0; k <= K; k++) {
      const L = 1 << k, sp = spacingOf(sch, L);
      D.frame(k, L, `c${k}`);
      D.line(k, [0, y0], [N, y1], 'thin');
      for (let b = 0; b < N / L; b++) {
        const ya = y(b * L), yb = y((b + 1) * L), u = Math.floor(ya);
        let d: number, held: boolean;
        if (sch.kind === 'wide' || sp === 1) { d = sp * Math.floor((yb - u - 0.5) / sp + 0.5); held = Math.abs(d) <= L; }
        else { d = Math.floor(yb) - u; held = risesAt(L, sp).includes(d); }
        if (held) ray(D, k, L, b * L, u, d, sp, sch.kind === 'wide', 'normal');
        else D.line(k, [b * L, ya], [(b + 1) * L, yb], 'dashed', 'red');
      }
    }
  },
});

const fan = defineFigure<Marks & Dirs & Size & { probe: number }>({
  id: 'fan',
  name: 'One probe',
  question: "Where do one probe's rays reach in every cascade?",
  about: 'The rays leaving one probe in every cascade: cK reaches 2^K columns. Their ends are where the next cascade starts. With fewer directions the fans thin out; wide rays mark their end window.',
  defaults: { marks: 'probes', dirs: 'all', n: 8, probe: 3 },
  options: [sizeOption(), dirsOption(), { key: 'probe', label: 'Probe row', kind: 'slider', min: 0, max: (s) => s.n - 1, step: 1 }, marksOption()],
  N: (s) => s.n,
  sheets: () => 1,
  draw(D, { N, K, sch, s }) {
    const r = clampInt(s.probe, 0, N - 1), wide = sch.kind === 'wide';
    D.frame(0, N, '');
    for (let x = 1; x < N; x *= 2) D.line(0, [x, 0], [x, N], 'fine');
    D.dot(0, [0, r + 0.5], 2.6, 'red');
    for (let k = 0; k <= K; k++) {
      const L = 1 << k, sp = spacingOf(sch, L);
      for (const d of risesAt(L, sp)) {
        const e = r + 0.5 + d;
        if (e < 0 || e > N) continue;
        D.line(0, [0, r + 0.5], [L, e], k === K ? 'normal' : 'thin');
        D.dot(0, [L, e], 1.5);
        if (wide && sp > 1) {   // the end window as a bracket: its span, with a tick at each end
          const lo = Math.max(0, e - sp / 2) + 0.08, hi = Math.min(N, e + sp / 2) - 0.08, t = 0.18 * Math.min(4, L / 4);
          D.path(0, [[L - t, lo], [L, lo], [L, hi], [L - t, hi]], 'thin');
        }
      }
      D.text(0, [L, 0], `c${k}`, 'middle', 0, 17);
    }
  },
});

const handoff = defineFigure<Proj & Marks>({
  id: 'handoff',
  name: 'Hand-off',
  question: 'What does a wide ray hand to its junction when a wall sits just past it?',
  about: 'Left: one wide ray (dotted fan) from row 0 to a 4-row window at the junction, a light (cross-hatched) and a wall just past the junction (hatched); every line through the light ends on the wall. Right: the cascade keeps one value per ray and hands it to the junction rows (marked on the junction) by its lines, about 1/8, 1/4, 1/4, 1/4, 1/8 from the bottom; the free rows carry about half of it past the wall (red, clipped at the top of the sheet). Truly, none gets past.',
  defaults: { ...PROJ, marks: 'probes' },
  options: [...projOptions(), marksOption()],
  sheets: () => 2,
  N: () => 16,
  rows: () => 8,
  draw(D) {
    for (const s of [0, 1]) {
      D.frame(s, 8, s === 0 ? 'lines' : 'cascade');
      D.area(s, [[0, 0], [0, 1], [8, 6.5], [8, 2.5]], 'dots', 'ink', 'thin');
      D.cell(s, 7, 4, 'cross');
      D.cell(s, 8, 4, 'hatch'); D.cell(s, 8, 5, 'hatch');
    }
    for (const [y0, px, py] of [[0.15, 7.0, 4.05], [0.5, 7.5, 4.5], [0.85, 7.02, 4.95], [0.2, 7.9, 4.95], [0.8, 7.95, 4.1]]) {
      D.line(0, [0, y0], [px, py], 'thin');
      D.line(0, [px, py], [8, y0 + ((py - y0) * 8) / px], 'normal');
    }
    const shares: [number, number][] = [[2.5, 3], [3, 4], [4, 5], [5, 6], [6, 6.5]];   // its lines' shares: 1/8, 1/4, 1/4, 1/4, 1/8
    for (const [lo, hi] of shares) {
      D.line(1, [8, lo + 0.06], [8, hi - 0.06], 'normal');
      const blocked = lo >= 4 && hi <= 6;
      if (!blocked) {
        const m = (lo + hi) / 2;
        for (const rise of [2, 6]) {   // the edges of the second half's fan, clipped at the top of the sheet
          const x = m + rise <= 8 ? 16 : 8 + ((8 - m) * 8) / rise;
          D.path(1, [[8, m], [x, m + ((x - 8) * rise) / 8]], 'dashed', 'red');
        }
      }
    }
  },
});


// ── The line arrangement of one block (lattice_geometry/NOTES.md) ──

/** rows shown for a block of length L: as many as its columns, so the sheet is square */
const rowsFor = (L: number) => Math.max(2, L);

/** a convex polygon clipped to the half-plane a x + b y <= c */
function clip(poly: Pt[], a: number, b: number, c: number): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i < poly.length; i++) {
    const P = poly[i], Q = poly[(i + 1) % poly.length], fp = a * P[0] + b * P[1] - c, fq = a * Q[0] + b * Q[1] - c;
    if (fp <= 1e-12) out.push(P);
    if ((fp < -1e-12 && fq > 1e-12) || (fp > 1e-12 && fq < -1e-12)) { const t = fp / (fp - fq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
  }
  return out;
}
/** corners of a convex polygon: drop repeats and points on a straight run */
function corners(poly: Pt[]): Pt[] {
  const q = poly.filter((p, i) => Math.hypot(p[0] - poly[(i + 1) % poly.length][0], p[1] - poly[(i + 1) % poly.length][1]) > 1e-9);
  return q.filter((p, i) => {
    const a = q[(i + q.length - 1) % q.length], b = q[(i + 1) % q.length];
    return Math.abs((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) > 1e-9;
  });
}
/** the block's rays (length L, rises D) clipped to rows [0, W], and its faces with their corner counts */
export function arrangement(L: number, D: number[], WINDOW = rowsFor(L)) {
  const rays: [Pt, Pt][] = [];
  for (const d of D)
    for (let a = -L - 2; a <= WINDOW + L + 2; a++) {
      let s: Pt = [0, a + 0.5], e: Pt = [L, a + 0.5 + d];
      const y = (x: number) => a + 0.5 + (d * x) / L, xAt = (yy: number) => ((yy - a - 0.5) * L) / d;
      if (Math.max(s[1], e[1]) < 0 || Math.min(s[1], e[1]) > WINDOW) continue;
      if (d !== 0) {
        for (const yy of [0, WINDOW]) {
          const x = xAt(yy);
          if (x > 0 && x < L) { if (s[1] < 0 || s[1] > WINDOW) s = [x, y(x)]; else if (e[1] < 0 || e[1] > WINDOW) e = [x, y(x)]; }
        }
      }
      rays.push([s, e]);
    }
  const seen = new Set<string>(), faces: { pts: Pt[]; n: number; whole: boolean }[] = [];
  const nx = 60 * L, ny = 140 * WINDOW;
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      const x = ((i + 0.37) / nx) * L, y = ((j + 0.61) / ny) * WINDOW;
      const k = D.map((d) => Math.floor(y - 0.5 - (x / L) * d));
      const key = k.join(',');
      if (seen.has(key)) continue;
      seen.add(key);
      let poly: Pt[] = [[0, 0], [L, 0], [L, WINDOW], [0, WINDOW]];
      D.forEach((d, n) => {   // k_n + 1/2 <= y - (d/L) x <= k_n + 3/2
        poly = clip(poly, d / L, -1, -(k[n] + 0.5));
        poly = clip(poly, -d / L, 1, k[n] + 1.5);
      });
      const c = corners(poly);
      if (c.length >= 3) faces.push({ pts: c, n: c.length, whole: c.every((p) => p[1] > 1e-9 && p[1] < WINDOW - 1e-9) });
    }
  return { rays, faces };
}

/** Euler's totient */
function totient(n: number) { let r = n, m = n; for (let p = 2; p * p <= m; p++) if (m % p === 0) { while (m % p === 0) m /= p; r -= r / p; } if (m > 1) r -= r / m; return r; }
/** faces per row of a block of length L with 2L+1 rises: all faces, triangles, crossings (lattice_geometry/NOTES.md 1.3) */
export function faceCounts(L: number) {
  let F = 1, T = 2, V = 0;
  for (let m = 1; m <= 2 * L; m++) F += totient(m) * (2 * L + 1 - m);
  for (let m = 1; m <= L; m++) T += 2 * totient(m) * (2 * L + 1 - 2 * m);
  for (let m = 2; m <= 2 * L; m++) V += m <= L ? m * totient(m) : totient(m) * (2 * L + 1 - m);
  return { F, T, Q: F - T, V };
}
/** p/m in lowest terms with m <= maxDen, for a number that is such a fraction */
function frac(v: number, maxDen: number) {
  for (let m = 1; m <= maxDen; m++) { const p = Math.round(v * m); if (Math.abs(v - p / m) < 1e-7) return m === 1 ? `${p}` : `${p}/${m}`; }
  return v.toFixed(4);
}
const arrangementCache = new Map<string, ReturnType<typeof arrangement>>();
function cachedArrangement(L: number, D: number[]) {
  const key = `${L}:${D.join(',')}`;
  if (!arrangementCache.has(key)) arrangementCache.set(key, arrangement(L, D));
  return arrangementCache.get(key)!;
}
/** the faces of one row (the row just below the middle), ordered left to right */
function rowFaces(L: number, D: number[]) {
  const W = rowsFor(L), r0 = Math.floor(W / 2) - (W % 2 === 0 ? 1 : 0);
  const cy = (f: { pts: Pt[] }) => f.pts.reduce((s, p) => s + p[1], 0) / f.pts.length;
  const cx = (f: { pts: Pt[] }) => f.pts.reduce((s, p) => s + p[0], 0) / f.pts.length;
  return cachedArrangement(L, D).faces.filter((f) => f.whole && cy(f) >= r0 - 1e-6 && cy(f) < r0 + 1 - 1e-6).sort((a, b) => cx(a) - cx(b) || cy(a) - cy(b));
}

const faces = defineFigure<Marks & Dirs & { blk: number; face: number }>({
  id: 'faces',
  name: 'Faces',
  question: "What are the gaps between one interval's rays?",
  about: 'One interval of one cascade, as many rows tall as it is wide. Between its rays every gap is a triangle (blank) or a quad (hatched). Each gap is addressed by its digitisation: for a point at fraction t across the interval and height y above a row centre, the ray just below it in direction d starts in row k_d = floor(y - t d). Every point of one gap has the same k_d for all d, so a gap is one list of start rows, and its steps k_d - k_(d+1) form a balanced word of 0s and 1s. The red gap is the one picked below.',
  defaults: { marks: 'probes', dirs: 'all', blk: 2, face: 0 },
  options: [
    dirsOption(),
    { key: 'blk', label: 'Cascade (interval length 2^n)', kind: 'slider', min: 1, max: 3, step: 1 },
    { key: 'face', label: 'Face in the row', kind: 'slider', min: 0, max: (s) => faceCounts(1 << clampInt(s.blk, 1, 3)).F - 1, step: 1 },
    marksOption(),
  ],
  sheets: () => 1,
  N: (s) => 1 << clampInt(s.blk, 1, 3),
  rows: (s) => rowsFor(1 << clampInt(s.blk, 1, 3)),
  draw(D, { sch, s: p }) {
    const L = 1 << clampInt(p.blk, 1, 3), R = risesAt(L, spacingOf(sch, L));
    const { rays, faces: fs } = cachedArrangement(L, R);
    for (const f of fs) if (f.whole && f.n === 4) D.area(0, f.pts, 'hatch');
    D.frame(0, L, `c${Math.log2(L)}, one interval`);
    for (const [s, e] of rays) D.line(0, s, e, 'thin');
    const row = rowFaces(L, R), pick = row[clampInt(p.face, 0, row.length - 1)];
    if (pick) { D.area(0, pick.pts, 'cross', 'red', 'normal'); }
  },
  info({ sch, s: p }) {
    const L = 1 << clampInt(p.blk, 1, 3), R = risesAt(L, spacingOf(sch, L));
    const row = rowFaces(L, R), tri = row.filter((f) => f.n === 3).length;
    const lines: string[] = [];
    if (sch.kind === 'all') {
      const c = faceCounts(L);
      lines.push(`per row, by formula (2L+1 = ${2 * L + 1} rises):`,
        `  F = 1 + sum_{m=1..${2 * L}} phi(m)(${2 * L + 1} - m) = ${c.F}`,
        `  T = 2 + 2 sum_{m=1..${L}} phi(m)(${2 * L + 1} - 2m) = ${c.T}`,
        `  Q = F - T = ${c.Q}`,
        `  crossings V = ${c.V}`);
    }
    lines.push(`drawn, in one row: ${row.length} faces = ${tri} triangles + ${row.length - tri} quads`);
    const pick = row[clampInt(p.face, 0, row.length - 1)];
    if (pick) {
      const cx = pick.pts.reduce((s, q) => s + q[0], 0) / pick.n, cy = pick.pts.reduce((s, q) => s + q[1], 0) / pick.n;
      const t = cx / L, y = cy - 0.5;
      const k = R.map((d) => Math.floor(y - t * d));
      const word = k.slice(0, -1).map((v, i) => v - k[i + 1]).join('');
      const area = Math.abs(pick.pts.reduce((s, q, i) => { const n = pick.pts[(i + 1) % pick.n]; return s + q[0] * n[1] - n[0] * q[1]; }, 0)) / 2;
      lines.push('', `face ${clampInt(p.face, 0, row.length - 1)}: a ${pick.n === 3 ? 'triangle' : 'quad'}`,
        `  start rows k_d, d = ${R[0]}..${R[R.length - 1]}: ${k.join(' ')}`,
        `  steps (balanced word): ${word}`,
        `  corners (t across the interval, y above a row centre):`,
        ...pick.pts.map((q) => `    (${frac(q[0] / L, 2 * L)}, ${frac(q[1] - 0.5, 2 * L)})`),
        `  area: ${area.toFixed(5)} px² (a row of the interval has area ${L})`);
    }
    return lines.join('\n');
  },
});

const crossings = defineFigure<Marks & Dirs & { blk: number }>({
  id: 'crossings',
  name: 'Crossings',
  question: 'Where do the rays of one interval cross?',
  about: 'Where the rays of one interval cross. A crossing at t = p/m across the interval (p/m in lowest terms) sits on a row offset by a multiple of 1/m, and the rays through it are the rises d with p d = s (mod m): about (2L+1)/m of them. So crossings line up on the Farey columns of order 2L (fine lines, all of them; labelled up to m = 4), denser where m is larger, with fewer rays each. Ring size: rays through the point. Pixel centres are never crossings.',
  defaults: { marks: 'probes', dirs: 'all', blk: 2 },
  options: [dirsOption(), { key: 'blk', label: 'Cascade (interval length 2^n)', kind: 'slider', min: 1, max: 3, step: 1 }, marksOption()],
  sheets: () => 1,
  N: (s) => 1 << clampInt(s.blk, 1, 3),
  rows: (s) => rowsFor(1 << clampInt(s.blk, 1, 3)),
  draw(D, { sch, s: p }) {
    const L = 1 << clampInt(p.blk, 1, 3), sp = spacingOf(sch, L), R = risesAt(L, sp);
    const WINDOW = rowsFor(L);
    const { rays } = arrangement(L, R);
    D.frame(0, L, `c${Math.log2(L)}, one interval`);
    const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
    for (let m = 2; m <= 2 * L; m++)   // every Farey column of order 2L; labels up to m = 4
      for (let q = 1; q < m; q++)
        if (gcd(q, m) === 1) {
          D.line(0, [(L * q) / m, 0], [(L * q) / m, WINDOW], 'fine');
          if (m <= 4) D.text(0, [(L * q) / m, 0], `${q}/${m}`, 'middle', 0, 30);
        }
    for (const [s, e] of rays) D.line(0, s, e, 'thin');
    for (let m = 2; m <= 2 * L; m++)
      for (let q = 1; q < m; q++) {
        if (gcd(q, m) !== 1) continue;
        for (let s = 0; s <= WINDOW * m; s++) {
          const mu = R.filter((d) => (((s - q * d) % m) + m) % m === 0).length;
          const y = 0.5 + s / m;
          if (mu < 2 || y > WINDOW) continue;
          if (mu === 2) D.dot(0, [(L * q) / m, y], 1.3);
          else D.ring(0, [(L * q) / m, y], 1.6 + 1.1 * (mu - 2));
        }
      }
  },
});


const bitlen = (v: number) => (v ? 32 - Math.clz32(v) : 0);
type Pair = { lx: number; ly: number; rx: number; ry: number };
const pairOptions = <T extends Pair>(rowMax: number): Option<T>[] => [
  { key: 'lx', label: 'Light column', kind: 'slider', min: 0, max: 15, step: 1 },
  { key: 'ly', label: 'Light row', kind: 'slider', min: 0, max: rowMax, step: 1 },
  { key: 'rx', label: 'Receiver column', kind: 'slider', min: 0, max: 15, step: 1 },
  { key: 'ry', label: 'Receiver row', kind: 'slider', min: 0, max: rowMax, step: 1 },
];
const sharedOf = (p: Pair, N: number) => {
  const lx = clampInt(p.lx, 0, N - 1), rx = clampInt(p.rx, 0, N - 1), k = Math.max(1, bitlen(lx ^ rx));
  const L = 1 << k, B = (lx >> k) << k;
  return { lx, rx, ly: clampInt(p.ly, 0, 7), ry: clampInt(p.ry, 0, 7), k, L, B, J: B + L / 2 };
};

const shared = defineFigure<Common & Pair>({
  id: 'shared',
  name: 'Hand-over',
  question: 'Where does the merge hand a light across to a receiver?',
  about: 'A light pixel (hatched) and a receiver pixel (cross-hatched), pinned through every sheet. The top sheet is the first where one interval holds both. One sheet down that interval is two halves, the light in one and the receiver in the other, and the merge passes the light across the column between them: the hand-over column (rings). Its ray along the line from light to receiver is red; below it, the pairs of halves it is built from. The dashed line is the straight line from light to receiver.',
  defaults: { ...COMMON, lx: 3, ly: 4, rx: 12, ry: 7 },
  options: [dirsOption(), ...pairOptions(7), ...projOptions(), marksOption()],
  N: () => 16,
  rows: () => 8,
  sheets: (c) => sharedOf(c.s, c.N).k + 1,
  draw(D, { N, sch, s: p }) {
    const { lx, rx, ly, ry, k, L, B, J } = sharedOf(p, N), h = L / 2;
    const slope = (ry - ly) / (rx - lx || 1), yAt = (x: number) => ly + 0.5 + slope * (x - lx - 0.5);
    for (let s = 0; s <= k; s++) {
      D.frame(s, 1 << s, `c${s}`);
      D.cell(s, lx, ly, 'hatch');
      D.cell(s, rx, ry, 'cross');
    }
    D.pin(0, k, lx + 0.5, ly + 0.5);
    D.pin(0, k, rx + 0.5, ry + 0.5);
    // the parent ray along the line, and the pairs of halves it is built from, one sheet down
    const u = Math.round(yAt(B) - 0.5), d = Math.round(yAt(B + L) - 0.5) - u;
    const sP = spacingOf(sch, L), sC = spacingOf(sch, h), H = risesAt(h, sC);
    D.line(k, [B, u + 0.5], [B + L, u + 0.5 + d], 'normal', 'red');
    for (const [a, j, b, w] of splits(L, d, sP, sC, 'one row')) {
      if (!H.includes(a) || !H.includes(b)) continue;
      const stroke: Stroke = w >= 0.75 ? 'normal' : w >= 0.25 ? 'thin' : 'dashed';
      D.line(k - 1, [B, u + 0.5], [J, u + 0.5 + a], stroke);
      D.line(k - 1, [J, u + j + 0.5], [B + L, u + j + 0.5 + b], stroke);
      D.ring(k - 1, [J, u + j + 0.5]);
      D.pin(k, k - 1, J, u + j + 0.5);
    }
    D.pin(k, 0, J, yAt(J));
    D.line(0, [lx + 0.5, ly + 0.5], [rx + 0.5, ry + 0.5], 'dashed', 'red');
  },
  info({ N, K, s: p }) {
    const { lx, rx, k, B, J } = sharedOf(p, N), bin = (v: number) => v.toString(2).padStart(K, '0');
    const before = J - (lx + 0.5), after = rx + 0.5 - J;
    return [
      `light column    ${bin(lx)}  (${lx})`,
      `receiver column ${bin(rx)}  (${rx})`,
      `they first differ in bit ${k - 1}, so the smallest interval holding both is c${k}'s [${B}, ${B + (1 << k)})`,
      `hand-over column (its middle): ${J}`,
      `light ${before} before it, receiver ${after} after it`,
      before === after ? 'halfway: the two are mirrored across the hand-over column' : `halfway receiver: column ${2 * J - lx - 1}`,
    ].join('\n');
  },
});

const flowCache = new Map<string, Flows>();
const flowsOf = (N: number, lx: number, ly: number, rx: number, ry: number) => {
  const key = [N, lx, ly, rx, ry].join(' ');
  let F = flowCache.get(key);
  if (!F) { F = pairFlows(N, lx, ly, rx, ry); flowCache.set(key, F); }
  return F;
};
const pathPair = (p: Pair, N: number) => {
  const lx = clampInt(p.lx, 0, N - 3), rx = clampInt(p.rx, lx + 2, N - 1);
  return { lx, rx, ly: clampInt(p.ly, 0, N - 1), ry: clampInt(p.ry, 0, N - 1), k: bitlen(lx ^ rx) };
};
/** the outline of every straight line from one pixel to another: the hull of the two squares */
function hull(a: Pt, b: Pt): Pt[] {
  const pts: Pt[] = [];
  for (const [x, y] of [a, b]) pts.push([x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]);
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const cross = (o: Pt, p: Pt, q: Pt) => (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
  const half = (list: Pt[]) => { const h: Pt[] = []; for (const p of list) { while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop(); h.push(p); } h.pop(); return h; };
  return [...half(pts), ...half([...pts].reverse())];
}

const paths = defineFigure<Proj & Marks & Pair>({
  id: 'paths',
  name: 'Light paths',
  question: 'Which rays carry one light pixel to one receiver, and how much each?',
  about: "Where one receiver's light from one light pixel travels. Each ray is drawn as wide as its share of that light, on its own cascade: on the light's side as the light climbs (records), on the receiver's side as the merge hands it down (far values), and in red on the top sheet, the rays that hand it across the hand-over column. Each sheet's shares add up to the receiver's whole value. The red dashed outline holds every straight line from the light pixel (hatched) to the receiver pixel (cross-hatched). Rays under 1 % are left out. The faint lines are all of each cascade's rays near the pair; the gaps between them are its faces.",
  defaults: { ...PROJ, marks: 'probes', lx: 3, ly: 4, rx: 12, ry: 7 },
  options: [...pairOptions(15), ...projOptions(), marksOption()],
  N: () => 16,
  sheets: (c) => pathPair(c.s, c.N).k + 1,
  draw(D, { N, s: p }) {
    const { lx, ly, rx, ry, k } = pathPair(p, N), F = flowsOf(N, lx, ly, rx, ry);
    const outline = hull([lx, ly], [rx, ry]);
    const width = (frac: number) => 0.35 + 7 * frac;
    for (let s = 0; s <= k; s++) {
      const L = 1 << s;
      D.frame(s, L, `c${s}`);
      D.path(s, [...outline, outline[0]], 'fine', 'red');
      D.cell(s, lx, ly, 'hatch');
      D.cell(s, rx, ry, 'cross');
      // the sheet's own rays, faint, between the rows the hull spans (their gaps are the faces)
      const y0 = Math.min(ly, ry) - 1, y1 = Math.max(ly, ry) + 2;
      for (let iv = 0; iv < N / L; iv++)
        for (let d = -L; d <= L; d++)
          for (let u = y0 - L - 1; u <= y1 + L; u++) {
            const ya = u + 0.5;
            let ta = 0, tb = 1;
            if (d === 0) { if (ya < y0 || ya > y1) continue; }
            else {
              const t0 = (y0 - ya) / d, t1 = (y1 - ya) / d;
              ta = Math.max(0, Math.min(t0, t1)); tb = Math.min(1, Math.max(t0, t1));
              if (ta >= tb) continue;
            }
            D.line(s, [iv * L + ta * L, ya + ta * d], [iv * L + tb * L, ya + tb * d], 'thin', 'grey');
          }
      const draw = (parts: Float32Array, pen: 'ink' | 'red') => {
        const R = F.D[L], nd = R.length;
        for (let i = 0; i < parts.length; i++) {
          const frac = parts[i] / F.value;
          if (frac < 0.01) continue;
          const kk = i % nd, y = Math.floor(i / nd) % F.rows - F.pad, b = Math.floor(i / nd / F.rows), d = R[kk];
          if (y < 0 || y >= N || y + d < 0 || y + d >= N) continue;
          D.weighted(s, [b * L, y + 0.5], [(b + 1) * L, y + 0.5 + d], width(frac), pen);
        }
      };
      if (s < k) { draw(F.up[L], 'ink'); draw(F.down[L], 'ink'); }
      else draw(F.hand[L], 'red');
    }
    D.pin(0, k, lx + 0.5, ly + 0.5);
    D.pin(0, k, rx + 0.5, ry + 0.5);
  },
  info({ N, K, s: p }) {
    const { lx, ly, rx, ry, k } = pathPair(p, N), F = flowsOf(N, lx, ly, rx, ry);
    const E = exactAngle(rx + 0.5, ry + 0.5, lx, ly), J = ((lx >> k) << k) + (1 << (k - 1)), h = 1 << (k - 1);
    const bits = (v: number) => (k > 1 ? v.toString(2).padStart(k - 1, '0') : '-');
    const spread = (L: number) => { let n = 0; for (const a of [F.up[L], F.down[L]]) for (const v of a) if (v / F.value >= 0.05) n++; return n; };
    return [
      `cascade ${F.value.toFixed(4)}, exact ${E.toFixed(4)}, error ${((100 * (F.value - E)) / E).toFixed(1)} %`,
      `hand-over column ${J} (c${k}); halves of ${h}`,
      `light ${J - lx - 1} columns before it (${bits(J - lx - 1)}), receiver ${rx - J} after (${bits(rx - J)})`,
      `rays carrying 5 % or more, per cascade: ${Array.from({ length: k }, (_, s) => `c${s} ${spread(1 << s)}`).join(', ')}`,
    ].join('\n');
  },
});

export const FIGURES: Figure[] = [cascades, extension, merge, forms, dither, halves, coverage, fan, handoff, faces, crossings, shared, paths];
