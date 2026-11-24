// A small diagram kit for cascades, drawn as a pen plotter would: one ink, a red pen kept for what a figure is about,
// three strokes (normal, thin, dashed, plus a fine dashed one for construction lines), and hatching instead of fills.
//
// Geometry lives on sheets: flat N x N pixel grids, x in columns [0, N], y in rows [0, N] (row r spans [r, r + 1]).
// A figure draws onto sheets; the diagram records every mark and fits them tightly into the page at the end, so
// strokes, dots and text keep their size whatever the scale.
//
// Views: 'plan' lays the sheets side by side, seen from straight above. 'stack' stacks them and looks with a true
// orthographic camera (turned about the vertical, tilted down): parallel lines stay parallel, nothing is sheared.
//
// Look (shared by every figure): a theme (plotter ink on paper, or chalk on a blackboard), a line weight scale and
// labels on or off. Figures name pens, not colours; the theme picks the colours.

export type Pt = readonly [number, number];
export type Stroke = 'normal' | 'thin' | 'dashed' | 'fine' | 'dotted';
export type Pen = 'ink' | 'red' | 'grey' | 'blue' | 'violet' | 'amber';
export type Fill = 'hatch' | 'cross' | 'dots';
export type Grid = 'probes' | 'corners' | 'none';
export interface ViewSpec {
  N: number; rows?: number; sheets: number; mode: 'plan' | 'stack'; turn: number; tilt: number; layer: number; grid: Grid;
  /** plan only: sheets per row (all in one row when unset); later rows sit below */
  cols?: number;
}

/** the shared view: how every figure is drawn */
export interface Look { theme: ThemeName; weight: number; labels: boolean }
export type ThemeName = 'paper' | 'chalk';
interface Theme { bg: string; pens: Record<Pen, string>; font: string; size: number; charW: number; chalk: boolean; /** stroke width factor */ stroke: number }
export const THEMES: Record<ThemeName, Theme> = {
  // one ink, a red pen kept for what a figure is about, and a few spare pens for figures that tell parts apart
  paper: {
    bg: '#f4f1e9', pens: { ink: '#1d1d1b', red: '#c23a2b', grey: '#b9b3a7', blue: '#2b5c9e', violet: '#6a3fa0', amber: '#e9a866' },
    font: "ui-monospace, 'SF Mono', monospace", size: 11, charW: 6.7, chalk: false, stroke: 1,
  },
  // chalk on a green-black board: chalk white and pastel chalks, grainy strokes, a hand-drawn label face
  chalk: {
    bg: '#1f2b27', pens: { ink: '#ecebe3', red: '#f3a29a', grey: '#5f716a', blue: '#9fc9ec', violet: '#cdb2f0', amber: '#b97a40' },
    font: "Chalkduster, 'Chalkboard SE', 'Bradley Hand', 'Comic Sans MS', 'Marker Felt', cursive", size: 11.5, charW: 7.6, chalk: true, stroke: 1.45,
  },
};
const WIDTH: Record<Stroke, number> = { normal: 1.15, thin: 0.5, dashed: 1.15, fine: 0.5, dotted: 2.2 };
const DASH: Record<Stroke, number[]> = { normal: [], thin: [], dashed: [4.5, 3], fine: [1.6, 2.4], dotted: [0.1, 4] };

type Item =
  | { kind: 'line'; pts: Pt[]; stroke: Stroke; pen: Pen; w?: number; a?: number }
  | { kind: 'area'; pts: Pt[]; fill: Fill; pen: Pen; outline?: Stroke }
  | { kind: 'dot'; p: Pt; r: number; pen: Pen }
  | { kind: 'ring'; p: Pt; r: number; pen: Pen }
  | { kind: 'head'; p: Pt; d: Pt; r: number; pen: Pen }
  | { kind: 'text'; p: Pt; s: string; anchor: 'start' | 'middle' | 'end'; dx: number; dy: number; pen: Pen };

function camera(v: ViewSpec) {
  const { N } = v, M = v.rows ?? N;
  if (v.mode === 'plan') {
    const cols = v.cols ?? v.sheets, gx = N + Math.max(2, N * 0.22), gy = M + Math.max(3, M * 0.3);
    return (s: number, x: number, y: number): Pt => [(s % cols) * gx + x, Math.floor(s / cols) * gy - y];
  }
  const t = (v.turn * Math.PI) / 180, p = (v.tilt * Math.PI) / 180, gap = N * v.layer;
  return (s: number, x: number, y: number): Pt => {
    const X = x - N / 2, Z = y - M / 2;
    const xr = X * Math.cos(t) - Z * Math.sin(t), zr = X * Math.sin(t) + Z * Math.cos(t);
    return [xr, -(s * gap * Math.cos(p) + zr * Math.sin(p))];
  };
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export class Diagram {
  readonly items: Item[] = [];
  readonly P: (s: number, x: number, y: number) => Pt;
  constructor(readonly v: ViewSpec) { this.P = camera(v); }

  /** points the fit includes but nothing draws: in the isometric view, the stack's extent at every turn, so turning
   * neither zooms nor shifts the drawing */
  #extent(): Pt[] {
    if (this.v.mode === 'plan') return [];
    const { N, sheets, tilt, layer } = this.v, M = this.v.rows ?? N, R = Math.hypot(N, M) / 2, p = (tilt * Math.PI) / 180, gap = N * layer;
    const out: Pt[] = [];
    for (const s of [0, sheets - 1]) for (const z of [-R, R]) for (const x of [-R, R]) out.push([x, -(s * gap * Math.cos(p) + z * Math.sin(p))]);
    return out;
  }

  line(s: number, a: Pt, b: Pt, stroke: Stroke = 'normal', pen: Pen = 'ink') { this.path(s, [a, b], stroke, pen); }
  path(s: number, pts: Pt[], stroke: Stroke = 'normal', pen: Pen = 'ink') {
    this.items.push({ kind: 'line', pts: pts.map(([x, y]) => this.P(s, x, y)), stroke, pen });
  }
  /** a line drawn as wide as `w` (px): for amounts carried along it; `a` its opacity */
  weighted(s: number, a: Pt, b: Pt, w: number, pen: Pen = 'ink', alpha?: number) { this.trail(s, [a, b], w, pen, alpha); }
  /** a polyline drawn as wide as `w` (px), with opacity `alpha` */
  trail(s: number, pts: Pt[], w: number, pen: Pen = 'ink', alpha?: number) {
    this.items.push({ kind: 'line', pts: pts.map(([x, y]) => this.P(s, x, y)), stroke: 'normal', pen, w, a: alpha });
  }
  /** a construction line between two sheets at the same (x, y); only when they are stacked */
  pin(s0: number, s1: number, x: number, y: number) {
    if (this.v.mode === 'plan') return;
    this.items.push({ kind: 'line', pts: [this.P(s0, x, y), this.P(s1, x, y)], stroke: 'fine', pen: 'ink' });
  }
  /** a region shaded by hatching; outline optional */
  area(s: number, pts: Pt[], fill: Fill = 'hatch', pen: Pen = 'ink', outline?: Stroke) {
    this.items.push({ kind: 'area', pts: pts.map(([x, y]) => this.P(s, x, y)), fill, pen, outline });
  }
  cell(s: number, x: number, y: number, fill: Fill = 'hatch', pen: Pen = 'ink') {
    this.area(s, [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], fill, pen, 'thin');
  }
  dot(s: number, p: Pt, r = 1.5, pen: Pen = 'ink') { this.items.push({ kind: 'dot', p: this.P(s, p[0], p[1]), r, pen }); }
  ring(s: number, p: Pt, r = 3.4, pen: Pen = 'ink') { this.items.push({ kind: 'ring', p: this.P(s, p[0], p[1]), r, pen }); }
  /** an arrowhead (filled, r px long) at fraction `at` of the segment a-b, pointing from a to b */
  head(s: number, a: Pt, b: Pt, at = 0.5, r = 4.5, pen: Pen = 'ink') {
    const A = this.P(s, a[0], a[1]), B = this.P(s, b[0], b[1]), dx = B[0] - A[0], dy = B[1] - A[1], n = Math.hypot(dx, dy);
    if (n < 1e-9) return;
    this.items.push({ kind: 'head', p: [A[0] + at * dx, A[1] + at * dy], d: [dx / n, dy / n], r, pen });
  }
  text(s: number, p: Pt, text: string, anchor: 'start' | 'middle' | 'end' = 'start', dx = 0, dy = 0, pen: Pen = 'ink') {
    this.items.push({ kind: 'text', p: this.P(s, p[0], p[1]), s: text, anchor, dx, dy, pen });
  }

  /** a sheet: outline, block edges every L columns, probes or pixel corners, and its name (rows: the sheet's height) */
  frame(s: number, L: number, name: string) {
    const N = this.v.N, M = this.v.rows ?? N;
    this.path(s, [[0, 0], [N, 0], [N, M], [0, M], [0, 0]], 'thin');
    for (let x = L; x < N; x += L) this.line(s, [x, 0], [x, M], 'fine');
    if (this.v.grid === 'probes') for (let x = 0; x <= N; x += L) for (let r = 0; r < M; r++) this.dot(s, [x, r + 0.5], 1.5);
    if (this.v.grid === 'corners') for (let x = 0; x <= N; x++) for (let y = 0; y <= M; y++) this.dot(s, [x, y], 0.9);
    if (!name) return;
    if (this.v.mode === 'plan') this.text(s, [N / 2, 0], name, 'middle', 0, 17);
    else this.text(s, [0, 0], name, 'end', -10, 4);
  }

  /** the page: every mark fitted into W x H (px), centred, with a margin, drawn in the shared look */
  svg(W: number, H: number, look: Look) {
    const th = THEMES[look.theme], m = 26, ws = look.weight * th.stroke;
    const items = look.labels ? this.items : this.items.filter((it) => it.kind !== 'text');
    const colourOf = (pen: Pen) => th.pens[pen];
    const textExtent = (it: Item): [number, number, number, number] | null => {
      if (it.kind !== 'text') return null;
      const w = it.s.length * th.charW, l = it.anchor === 'start' ? 0 : it.anchor === 'middle' ? -w / 2 : -w;
      return [l + it.dx, it.dy - 10, l + w + it.dx, it.dy + 3];
    };
    const bounds = (scale: number) => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const add = (x: number, y: number) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); };
      // the stack's extent at every turn, widened by the longest label (labels sit by the sheets' corners)
      const label = scale > 0 ? Math.max(0, ...items.map((it) => (it.kind === 'text' ? it.s.length * th.charW + Math.abs(it.dx) : 0))) / scale : 0;
      for (const [x, y] of this.#extent()) (add(x - label, y), add(x + label, y));
      for (const it of items) {
        if ('pts' in it) for (const [x, y] of it.pts) add(x, y);
        else {
          add(it.p[0], it.p[1]);
          const e = scale > 0 ? textExtent(it) : null;
          if (e) { add(it.p[0] + e[0] / scale, it.p[1] + e[1] / scale); add(it.p[0] + e[2] / scale, it.p[1] + e[3] / scale); }
        }
      }
      if (!Number.isFinite(x0)) return [0, 1, 0, 1];
      return [x0, x1, y0, y1];
    };
    const fit = ([x0, x1, y0, y1]: number[]) => Math.min((W - 2 * m) / Math.max(1e-6, x1 - x0), (H - 2 * m) / Math.max(1e-6, y1 - y0));
    let scale = fit(bounds(0));
    scale = fit(bounds(scale));
    const [x0, x1, y0, y1] = bounds(scale);
    const u = 1 / scale, f = (n: number) => n.toFixed(3);
    const g = 4.2 * u, hw = f(0.55 * u * ws);
    const defs: string[] = [];
    for (const pen of Object.keys(th.pens) as Pen[]) {
      const c = colourOf(pen), open = (id: string) => `<pattern id="${id}-${pen}" patternUnits="userSpaceOnUse" width="${f(g)}" height="${f(g)}" patternTransform="rotate(45)">`;
      defs.push(`${open('hatch')}<line x1="0" y1="0" x2="0" y2="${f(g)}" stroke="${c}" stroke-width="${hw}"/></pattern>`);
      defs.push(`${open('cross')}<line x1="0" y1="0" x2="0" y2="${f(g)}" stroke="${c}" stroke-width="${hw}"/><line x1="0" y1="0" x2="${f(g)}" y2="0" stroke="${c}" stroke-width="${hw}"/></pattern>`);
      defs.push(`${open('dots')}<circle cx="${f(g / 2)}" cy="${f(g / 2)}" r="${hw}" fill="${c}"/></pattern>`);
    }
    if (th.chalk) {
      // chalk: strokes wobble a little (displacement by low noise) and break up into grain (alpha from fine noise);
      // frequencies are per user unit, so they are set from the scale to stay the same on screen
      defs.push(`<filter id="chalk" x="-2%" y="-2%" width="104%" height="104%" color-interpolation-filters="sRGB">` +
        `<feTurbulence type="fractalNoise" baseFrequency="${f(0.06 / u)}" numOctaves="2" seed="4" result="wob"/>` +
        `<feDisplacementMap in="SourceGraphic" in2="wob" scale="${f(1.6 * u)}" xChannelSelector="R" yChannelSelector="G" result="moved"/>` +
        `<feTurbulence type="fractalNoise" baseFrequency="${f(0.9 / u)}" numOctaves="1" seed="9" result="grain"/>` +
        `<feColorMatrix in="grain" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  2.4 0 0 0 -0.3" result="mask"/>` +
        `<feComposite in="moved" in2="mask" operator="in"/></filter>`);
    }
    const strokeAttrs = (width: number, dash: number[], pen: Pen, alpha?: number) =>
      `stroke="${colourOf(pen)}" stroke-width="${f(width * u * ws)}"${dash.length ? ` stroke-dasharray="${dash.map((d) => f(d * u * ws)).join(' ')}"` : ''}${alpha !== undefined && alpha < 1 ? ` stroke-opacity="${alpha.toFixed(3)}"` : ''} stroke-linecap="round" stroke-linejoin="round" fill="none"`;
    const out: string[] = [];
    for (const it of items) {
      if (it.kind === 'line') out.push(`<polyline points="${it.pts.map((p) => p.map(f).join(',')).join(' ')}" ${strokeAttrs(it.w ?? WIDTH[it.stroke], DASH[it.stroke], it.pen, it.a)}/>`);
      else if (it.kind === 'area') {
        const pts = it.pts.map((p) => p.map(f).join(',')).join(' ');
        out.push(`<polygon points="${pts}" fill="url(#${it.fill}-${it.pen})" stroke="none"/>`);
        if (it.outline) out.push(`<polygon points="${pts}" ${strokeAttrs(WIDTH[it.outline], DASH[it.outline], it.pen)}/>`);
      } else if (it.kind === 'dot') out.push(`<circle cx="${f(it.p[0])}" cy="${f(it.p[1])}" r="${f(it.r * u * Math.sqrt(ws))}" fill="${colourOf(it.pen)}"/>`);
      else if (it.kind === 'head') {
        const r = it.r * u * Math.sqrt(ws), [x, y] = it.p, [dx, dy] = it.d;
        const pts: Pt[] = [[x + r * dx, y + r * dy], [x - 0.6 * r * dx - 0.55 * r * dy, y - 0.6 * r * dy + 0.55 * r * dx], [x - 0.6 * r * dx + 0.55 * r * dy, y - 0.6 * r * dy - 0.55 * r * dx]];
        out.push(`<polygon points="${pts.map((p) => p.map(f).join(',')).join(' ')}" fill="${colourOf(it.pen)}"/>`);
      } else if (it.kind === 'ring') out.push(`<circle cx="${f(it.p[0])}" cy="${f(it.p[1])}" r="${f(it.r * u)}" ${strokeAttrs(0.9, [], it.pen)}/>`);
      else out.push(`<text x="${f(it.p[0] + it.dx * u)}" y="${f(it.p[1] + it.dy * u)}" fill="${colourOf(it.pen)}" font-size="${f(th.size * u)}" font-family="${th.font.replace(/"/g, "'")}" text-anchor="${it.anchor}">${esc(it.s)}</text>`);
    }
    const vb = [x0 - m * u, y0 - m * u, x1 - x0 + 2 * m * u, y1 - y0 + 2 * m * u].map(f).join(' ');
    const body = th.chalk ? `<g filter="url(#chalk)">${out.join('')}</g>` : out.join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb}" width="${W}" height="${H}" preserveAspectRatio="xMidYMid meet" style="background:${th.bg}"><defs>${defs.join('')}</defs>${body}</svg>`;
  }
}

/** direction schedules: 'all' (2L+1 rises), 'half' (L+1, one-row ends), 'wide' (a spacing function, wide ends) */
export type Schedule = { kind: 'all' | 'half' | 'wide'; spacing: (L: number) => number };
export const risesAt = (L: number, s: number) => {
  const out: number[] = [];
  for (let d = -Math.floor(L / s) * s; d <= L; d += s) out.push(d);
  return out;
};
export const spacingOf = (sch: Schedule, L: number) => (L === 1 ? 1 : sch.kind === 'all' ? 1 : sch.kind === 'half' ? 2 : sch.spacing(L));
export const SCHEDULES: Record<string, Schedule> = {
  all: { kind: 'all', spacing: () => 1 },
  half: { kind: 'half', spacing: () => 2 },
  halfwide: { kind: 'wide', spacing: () => 2 },
  wide8: { kind: 'wide', spacing: (L) => Math.max(1, (2 * L) / 8) },
  wide4: { kind: 'wide', spacing: (L) => Math.max(1, (2 * L) / 4) },
};
