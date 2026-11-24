// Extension, merge, and primal against dual, drawn the way the HRC paper draws its extension: probe positions along the
// bottom, one panel per direction, side by side, so each panel shows which values one value is made from.
//
// Every panel is one frustum with light travelling up. A panel's x is position across the frustum (one probe per
// position), its y the distance along it: the start plane at 0, the junction plane at L/2, the end plane at L. A ray of
// length L and rise D runs from (x, 0) to (x + D, L).
//
// Marks: a line is a ray's own value (its light and transmittance); a wedge at a point is light arriving there from
// behind, over a cone of directions (HRC's cones, Reciprocal HRC's merged values). Orange: a value read. Red: the value
// written. Fine lines: construction (the parent a merged value belongs to). Arrowheads: the way light travels.
//
// Pairs (site/hrc/hrc.ts, site/reciprocal-napkin/napkin.ts, site/reciprocal-dither/dither.ts):
//   HRC: rises -L .. L in steps of 2. D/2 is a child rise: one straight pair; otherwise the two chains bent by a row.
//   Reciprocal: every rise. Even D one straight pair (D/2, D/2); odd D (floor, ceil) and (ceil, floor), ½ each.
//   Dithered: as Reciprocal, but an even D with 2 <= |D| <= L - 2 is (D/2, D/2) at ¾ and the pairs bent by a row at ⅛.
// The merge reads the same table backwards: a half of rise r takes, from every parent it is a half of, the parent's
// arriving light, through its sibling when it is the second half (with the sibling's own light at the parent's angle).
import type { Diagram, Fill, Pen, Pt } from './plot';
import { defineFigure, PROJ, projOptions, type Option, type Proj } from './figure';

export type Scheme = 'hrc' | 'plain' | 'dither';
export interface Pair { a: number; b: number; w: number }

/** the directions a frustum keeps for rays of length L */
export const risesOf = (scheme: Scheme, L: number) => {
  const out: number[] = [];
  for (let d = -L; d <= L; d += scheme === 'hrc' ? 2 : 1) out.push(d);
  return out;
};
/** the pairs of halves (rises a then b) a ray of length L and rise D is built from, with their weights */
export function pairsOf(scheme: Scheme, L: number, D: number): Pair[] {
  const h = L / 2;
  if (scheme === 'hrc') {
    const child = (r: number) => Math.abs(r) <= h && (((r - h) % 2) + 2) % 2 === 0;
    if (child(D / 2)) return [{ a: D / 2, b: D / 2, w: 1 }];
    return [{ a: D / 2 - 1, b: D / 2 + 1, w: 0.5 }, { a: D / 2 + 1, b: D / 2 - 1, w: 0.5 }];
  }
  if (D % 2 !== 0) { const f = Math.floor(D / 2); return [{ a: f, b: f + 1, w: 0.5 }, { a: f + 1, b: f, w: 0.5 }]; }
  if (scheme === 'dither' && Math.abs(D) >= 2 && Math.abs(D) <= L - 2)
    return [{ a: D / 2, b: D / 2, w: 0.75 }, { a: D / 2 - 1, b: D / 2 + 1, w: 0.125 }, { a: D / 2 + 1, b: D / 2 - 1, w: 0.125 }];
  return [{ a: D / 2, b: D / 2, w: 1 }];
}
/** the parents a half of rise r is the first (or second) half of: the parent's rise, the sibling's rise, the weight */
export function termsOf(scheme: Scheme, L: number, r: number, second: boolean) {
  const out: { D: number; sib: number; w: number }[] = [];
  for (const D of risesOf(scheme, L))
    for (const p of pairsOf(scheme, L, D)) if ((second ? p.b : p.a) === r) out.push({ D, sib: second ? p.a : p.b, w: p.w });
  return out;
}
const FRAC: Record<string, string> = { 0: '0', 1: '1', 0.5: '½', 0.75: '¾', 0.25: '¼', 0.125: '⅛' };
export const frac = (w: number) => FRAC[String(w)] ?? w.toFixed(3);

// ── marks ──

/** a panel: planes at heights ys across W positions with a probe at each position, a title above, captions below */
export function panel(D: Diagram, s: number, W: number, ys: number[], title: string, caption: string[], bottom = 0, names?: string[]) {
  const top = Math.max(...ys);
  if (D.v.mode === 'stack') D.path(s, [[0, Math.min(0, bottom)], [W, Math.min(0, bottom)], [W, top], [0, top], [0, Math.min(0, bottom)]], 'thin', 'grey');
  for (const y of ys) {
    D.line(s, [0, y], [W, y], 'fine', 'grey');
    for (let i = 0; i < W; i++) D.dot(s, [i + 0.5, y], 1.1);
  }
  if (D.v.mode === 'stack') {   // stacked: one label by the sheet's near corner, as frame() does
    const both = [title, caption[0]].filter(Boolean).join(' · ');
    D.text(s, [0, 0], both.length <= 22 ? both : title, 'end', -10, 4);
    return;
  }
  if (title) D.text(s, [W / 2, top], title, 'middle', 0, -9);
  caption.forEach((c, i) => c && D.text(s, [W / 2, bottom], c, 'middle', 0, 15 + 13 * i, i ? 'grey' : 'ink'));
  if (names) ys.forEach((y, i) => names[i] && D.text(s, [0, y], names[i], 'end', -6, 4, 'grey'));
}
/** a ray's value, read: a wide orange stroke (wider for a heavier weight) with an arrowhead the way light goes */
export function readRay(D: Diagram, s: number, a: Pt, b: Pt, w = 1) {
  D.trail(s, [a, b], 2.2 + 4.2 * w, 'amber', 0.5 + 0.45 * w);
  D.head(s, a, b, 0.38, 4.2, 'ink');
}
/** a ray's value, written: a red line (a polyline for a chain of halves) */
export function writeRay(D: Diagram, s: number, pts: Pt[], head = false) {
  D.path(s, pts, 'normal', 'red');
  if (head) D.head(s, pts[pts.length - 2], pts[pts.length - 1], 0.55, 4.2, 'red');
}
const unit = ([x, y]: Pt): Pt => { const n = Math.hypot(x, y) || 1; return [x / n, y / n]; };
/** light at p over the directions between v0 and v1 (vectors pointing away from p, toward where the wedge opens) */
export function wedge(D: Diagram, s: number, p: Pt, v0: Pt, v1: Pt, len: number, pen: Pen, fill: Fill = 'hatch', min = pen === 'red' ? 0.4 : 0.26) {
  let a = unit(v0), b = unit(v1);
  const ang = Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1])));
  if (ang < min) {   // too narrow to see: open it about its middle
    const m = unit([a[0] + b[0], a[1] + b[1]]), t = min / 2, c = Math.cos(t), sn = Math.sin(t);
    a = [m[0] * c - m[1] * sn, m[0] * sn + m[1] * c]; b = [m[0] * c + m[1] * sn, -m[0] * sn + m[1] * c];
  }
  D.area(s, [p, [p[0] + len * a[0], p[1] + len * a[1]], [p[0] + len * b[0], p[1] + len * b[1]]], fill, pen, pen === 'red' ? 'normal' : 'thin');
}
/** light arriving at p from behind along rays of rises (d ± ½) over length L: a wedge opening backwards */
const behind = (D: Diagram, s: number, p: Pt, d: number, L: number, len: number, pen: Pen, fill: Fill = 'hatch') =>
  wedge(D, s, p, [-(d - 0.5) / L, -1], [-(d + 0.5) / L, -1], len, pen, fill);

// ── the passes, one value written per panel ──

/** extension: a ray of length L and rise d from x0 (red) built from its pairs of halves (orange) */
export function paintExtension(D: Diagram, s: number, scheme: Scheme, L: number, d: number, x0: number, weights = true) {
  const h = L / 2, P = pairsOf(scheme, L, d);
  for (const p of P) { readRay(D, s, [x0, 0], [x0 + p.a, h], p.w); readRay(D, s, [x0 + p.a, h], [x0 + d, L], p.w); }
  for (const p of P) writeRay(D, s, [[x0, 0], [x0 + p.a, h], [x0 + d, L]]);
  for (const p of P) {
    D.ring(s, [x0 + p.a, h], 2.6, 'red');
    if (weights && P.length > 1) {
      const side = p.a < d / 2 ? -1 : p.a > d / 2 ? 1 : 0;
      D.text(s, [x0 + p.a, h], frac(p.w), side < 0 ? 'end' : 'start', side < 0 ? -6 : 6, side === 0 ? -5 : 4, 'red');
    }
  }
  return 2 * P.length;
}

/** Reciprocal merge: what a half of rise r reads (as the second half, starting at the junction at xj; as the first,
 * starting at the start plane at xj) */
export function paintRecipMerge(D: Diagram, s: number, scheme: Scheme, L: number, r: number, second: boolean, xj: number) {
  const h = L / 2, T = termsOf(scheme, L, r, second), big = 0.5 * h, small = 0.6 * h;
  for (const t of T) {
    const S: Pt = second ? [xj - t.sib, 0] : [xj, 0], E: Pt = second ? [xj + r, L] : [xj + t.D, L];
    D.line(s, S, E, 'fine', 'ink');                       // the parent
    behind(D, s, S, t.D, L, big, 'ink', 'cross');           // its arriving light, read
    if (second) {
      readRay(D, s, S, [xj, h], t.w);                       // through its first half, with that half's own light
    }
  }
  const A: Pt = second ? [xj, h] : [xj, 0];
  writeRay(D, s, [A, [A[0] + r, A[1] + h]]);
  behind(D, s, A, r, h, small, 'red', 'cross');
  D.dot(s, A, 2.2, 'red');
  return second ? 2 * T.length : T.length;
}

/** HRC's ray rises at length h: rays 0 .. h, rise 2i - h */
const hrcRise = (h: number, i: number) => 2 * i - h;
/** HRC merge: cone c of a probe (red wedge) reads its two rays (orange) and, where they end, cascade k's sub-cones
 * (black wedges); on an even plane the rays run on through the next half, and half the cone is read where it stands */
export function paintHrcMerge(D: Diagram, s: number, L: number, c: number, even: boolean, xq: number) {
  const h = L / 2, Q: Pt = [xq, even ? L : h], r0 = hrcRise(h, c), r1 = hrcRise(h, c + 1), mid = (r0 + r1) / 2;
  const big = 0.6 * h, small = 0.75 * h, far = even ? 2 : 1;
  for (const [i, lo, hi] of [[c, r0, mid], [c + 1, mid, r1]] as const) {
    const rr = hrcRise(h, i), F: Pt = [xq + far * rr, 0];
    if (even) { const M: Pt = [xq + rr, h]; readRay(D, s, F, M); readRay(D, s, M, Q); }
    else readRay(D, s, F, Q);
    wedge(D, s, F, [lo / h, -1], [hi / h, -1], big, 'ink', 'cross');
    if (even) D.text(s, F, '½', 'start', 6, 4, 'red');
  }
  if (even) {
    wedge(D, s, Q, [r0 / h, -1], [mid / h, -1], 0.62 * h, 'ink', 'cross');
    wedge(D, s, Q, [mid / h, -1], [r1 / h, -1], 0.62 * h, 'ink', 'cross');
    D.text(s, Q, '½ read here', 'start', 10, 14, 'red');
  }
  wedge(D, s, Q, [r0 / h, -1], [r1 / h, -1], small, 'red', 'cross');
  D.dot(s, Q, 2.2, 'red');
  return even ? 8 : 4;
}

/** Dual HRC, its extension: a source's cone c at the start plane (black) hands its light to its two rays (red) and
 * carries the rest to cascade k's sub-cones where they end (red) */
function paintDualExt(D: Diagram, s: number, L: number, c: number, xq: number) {
  const h = L / 2, Q: Pt = [xq, 0], r0 = hrcRise(h, c), r1 = hrcRise(h, c + 1), mid = (r0 + r1) / 2;
  wedge(D, s, Q, [r0 / h, 1], [r1 / h, 1], 0.42 * h, 'ink', 'cross');
  for (const [i, lo, hi] of [[c, r0, mid], [c + 1, mid, r1]] as const) {
    const F: Pt = [xq + hrcRise(h, i), h];
    writeRay(D, s, [Q, F], true);
    wedge(D, s, F, [lo / h, 1], [hi / h, 1], 0.5 * h, 'red', 'cross');
  }
  D.dot(s, Q, 2.2, 'ink');
}
/** Dual HRC, its merge: a ray's carried light (orange) handed to its halves (red), the second through the first */
function paintDualMerge(D: Diagram, s: number, L: number, d: number, x0: number) {
  const h = L / 2;
  readRay(D, s, [x0, 0], [x0 + d, L]);
  writeRay(D, s, [[x0 + 0.4, 0], [x0 + 0.4 + d / 2, h]], true);
  writeRay(D, s, [[x0 + 0.4 + d / 2, h], [x0 + 0.4 + d, L]], true);
  D.ring(s, [x0 + d / 2, h], 2.6, 'red');
}

// ── reads and writes ──

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
/** values read per value written, per pass, for each scheme at ray length L (one frustum, rows away from the edges) */
export function readsTable(L: number) {
  const h = L / 2, R: number[] = [];
  for (let r = -h; r <= h; r++) R.push(r);
  const up = (sc: Scheme) => {
    const n = risesOf(sc, L).map((d) => 2 * pairsOf(sc, L, d).length), v = [...new Set(n)].sort();
    return `${v.join(v.length > 2 ? ', ' : ' or ')}, mean ${fmt(mean(n))}`;
  };
  const down = (sc: Scheme, second: boolean) => fmt(Math.max(...R.map((r) => (second ? 2 : 1) * termsOf(sc, L, r, second).length)));
  return [
    `Values read per value written, L = ${L}`,
    `(one frustum, away from its edges):`,
    `extension`,
    `  HRC         ${up('hrc')}`,
    `  Reciprocal  ${up('plain')}`,
    `  Dithered    ${up('dither')}`,
    `merge`,
    `  HRC         a cone: 4 (odd plane),`,
    `              8 (even plane)`,
    `  Reciprocal  first half ${down('plain', false)}, second ${down('plain', true)}`,
    `  Dithered    first half ${down('dither', false)}, second ${down('dither', true)}`,
    `Writes: 1 per value per pass.`,
    `Two-way records (the napkin and dither`,
    `pages) read the sibling for a first half`,
    `too: 2 per parent in the merge.`,
  ].join('\n');
}

// ── figures ──

type Base = Proj & { scheme: Scheme; L: number };
const schemeOption = <T extends Base>(): Option<T> => ({
  key: 'scheme', label: 'Scheme', kind: 'choice', choices: [['hrc', 'HRC'], ['plain', 'Reciprocal'], ['dither', 'Dithered', 'Reciprocal HRC with dithered junctions']],
});
const lengthOption = <T extends Base>(): Option<T> => ({ key: 'L', label: 'Long ray length', kind: 'choice', choices: [[4, '4'], [8, '8']] });
/** panels per row: about square overall on a landscape page */
const gridCols = (n: number) => (n <= 4 ? n : Math.ceil(Math.sqrt(1.6 * n)));
const lg = (L: number) => Math.round(Math.log2(L));

export const extension = defineFigure<Base>({
  id: 'extension',
  name: 'Extension',
  question: 'Which shorter rays is each longer ray built from?',
  about: 'One panel per direction: the long ray (red) is chained from two shorter rays (orange) that meet at the junction. Each orange arrow is one value read.',
  defaults: { ...PROJ, layer: 0.8, tilt: 50, scheme: 'plain', L: 4 },
  options: [schemeOption(), lengthOption(), ...projOptions()],
  N: (s) => s.L + 2,
  rows: (s) => 1.45 * s.L,   // the panel is L tall; the rest leaves room for captions between rows
  sheets: (c) => risesOf(c.s.scheme, c.s.L).length,
  cols: (c) => gridCols(risesOf(c.s.scheme, c.s.L).length),
  draw(D, { s }) {
    const L = s.L, h = L / 2, W = L + 2, k = lg(L);
    risesOf(s.scheme, L).forEach((d, i) => {
      const P = pairsOf(s.scheme, L, d), x0 = Math.round(W / 2 - 0.5 - d / 2) + 0.5;
      panel(D, i, W, [0, h, L], `D = ${d}`, [`reads ${2 * P.length}`, P.length > 1 ? P.map((p) => frac(p.w)).join(' + ') : ''], 0, i === 0 ? [`c${k - 1} start`, 'junction', `c${k} end`] : undefined);
      paintExtension(D, i, s.scheme, L, d, x0, false);
    });
  },
  info: ({ s }) => readsTable(s.L),
});

type MS = Base & { half: 'first' | 'second'; plane: 'odd' | 'even' };
const mergeWidth = (s: MS) => (s.scheme === 'hrc' ? (s.plane === 'even' ? 2 * s.L + 3 : s.L + 3) : s.L + 3);
export const merge = defineFigure<MS>({
  id: 'merge',
  name: 'Merge',
  question: 'Where does each shorter ray get the light arriving at it?',
  about: 'One panel per direction of a value written (red wedge). Reciprocal: it reads each parent\'s arriving light (black wedge) through its sibling (orange), over the extension\'s own pairs. HRC: a cone reads its two rays and the cones where they end.',
  defaults: { ...PROJ, layer: 0.8, tilt: 50, scheme: 'plain', L: 4, half: 'second', plane: 'odd' },
  options: [
    schemeOption(), lengthOption(),
    { key: 'half', label: 'Written half', kind: 'choice', choices: [['first', 'First'], ['second', 'Second']], enabled: (s) => s.scheme !== 'hrc' },
    { key: 'plane', label: 'Probe plane', kind: 'choice', choices: [['odd', 'Odd'], ['even', 'Even']], enabled: (s) => s.scheme === 'hrc' },
    ...projOptions(),
  ],
  N: mergeWidth,
  rows: (s) => 1.45 * (s.scheme === 'hrc' && s.plane === 'odd' ? s.L / 2 : s.L),
  sheets: (c) => (c.s.scheme === 'hrc' ? c.s.L / 2 : c.s.L + 1),
  cols: (c) => gridCols(c.s.scheme === 'hrc' ? c.s.L / 2 : c.s.L + 1),
  draw(D, { s }) {
    const L = s.L, h = L / 2, W = mergeWidth(s), k = lg(L), low = -0.45 * h;
    if (s.scheme === 'hrc') {
      const even = s.plane === 'even';
      for (let c = 0; c < h; c++) {
        panel(D, c, W, even ? [0, h, L] : [0, h], `cone ${c}`, [`reads ${even ? 8 : 4}`], low,
          c === 0 ? (even ? [`c${k}`, '', `c${k - 1}, c${k}`] : [`c${k}`, `c${k - 1}`]) : undefined);
        paintHrcMerge(D, c, L, c, even, Math.floor(W / 2) + 0.5);
      }
      return;
    }
    const second = s.half === 'second';
    for (let r = -h; r <= h; r++) {
      const i = r + h, xj = second ? Math.floor(W / 2) + 0.5 : Math.round(W / 2 - 0.5 - r) + 0.5;
      const T = termsOf(s.scheme, L, r, second), n = (second ? 2 : 1) * T.length;
      panel(D, i, W, [0, h, L], `r = ${r}`, [`reads ${n}`], low, i === 0 ? [`c${k} start`, 'junction', `c${k} end`] : undefined);
      paintRecipMerge(D, i, s.scheme, L, r, second, xj);
    }
  },
  info: ({ s }) => readsTable(s.L),
});

export const forms = defineFigure<Proj>({
  id: 'forms',
  name: 'Primal and dual',
  question: 'Primal or dual: what does each pass read and write?',
  about: 'Primal HRC gathers light at each receiver\'s cones; dual HRC, its transpose, carries light out from each light\'s cones. Reciprocal HRC is the primal\'s extension and the dual\'s merge, joined at each junction, with no cones.',
  defaults: { ...PROJ, layer: 0.6, tilt: 50 },
  options: projOptions(),
  N: () => 13,
  rows: () => 12.5,
  sheets: () => 6,
  cols: () => 2,
  draw(D) {
    const L = 8, h = 4, W = 13, low = -0.6 * h, xc = Math.floor(W / 2) + 0.5, x0 = Math.round(W / 2 - 0.5 - 2) + 0.5, ys = [0, h, L];
    const names = ['c1 start', 'junction', 'c2 end'];
    panel(D, 0, W, ys, 'HRC primal · extension', ['reads 2 rays · writes 1'], low, names);
    paintExtension(D, 0, 'hrc', L, 4, x0);
    panel(D, 1, W, ys, 'HRC primal · merge', ['reads 2 rays, 2 c2 cones', 'writes 1 c1 cone'], low);
    paintHrcMerge(D, 1, L, 2, false, xc - 1);
    panel(D, 2, W, ys, 'HRC dual · extension', ['reads 1 c1 cone', 'writes 2 rays, 2 c2 cones'], low, ['c1 source', 'c2', '']);
    paintDualExt(D, 2, L, 2, xc - 1);
    panel(D, 3, W, ys, 'HRC dual · merge', ['reads 1 ray', 'writes 2 halves'], low);
    paintDualMerge(D, 3, L, 4, x0);
    panel(D, 4, W, ys, 'Reciprocal · extension', ['reads 2 rays · writes 1'], low, names);
    paintExtension(D, 4, 'plain', L, 4, x0);
    panel(D, 5, W, ys, 'Reciprocal · merge', ['reads 3 parents, 3 siblings', 'writes 1'], low);
    paintRecipMerge(D, 5, 'plain', L, 2, true, xc - 1);
  },
  info: () => [
    'Primal: the receivers\' side. The merge gathers, top cascade down, the light arriving at each probe\'s cones.',
    'Dual: the transpose. Light leaves each source\'s cones and is carried out (dual extension), then rays are halved back to c0 and splat into pixels (dual merge). Both are written as gathers on the GPU (site/dual-hrc).',
    '',
    'Reciprocal HRC stores no cones. Its extension is the primal\'s; its merge is the transposed extension (the dual\'s merge), plus the sibling\'s own light at the parent\'s angle where the halves meet. So it is its own transpose: light from p to q equals light from q to p, and the dual form is the same program on the opposite frustum.',
    '',
    readsTable(8),
  ].join('\n'),
});
