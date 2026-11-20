// The Reciprocal HRC page's figures, in one frustum's frame: light travels toward +x, columns along x, rows up. Pixels
// are cells with gaps between them; a plane between columns is its gap's centre. A ray piece is drawn once, parents blue
// and children orange; a probe's 1-px span is a bar on its plane, light blue where read and yellow where written. HRC is
// drawn half a pixel left, its probes on pixel edges like reciprocal HRC's; the merge is the same.
// Each function takes { scale } (CSS px per unit) and returns SVG.

/** the figures' colours; the captions' keys use the same */
export const C = { cell: '#222222', parent: '#2d5c99', child: '#ed7b39', half: '#d61a88', read: '#78d7ff', write: '#fff540', midpoint: '#d8d8d8', text: '#d8d8d8', dim: '#8c8c8c' };
const G = 2; // the gap between cells
const W = 2; // a 1-px span's width
const LINE = 3; // every ray's stroke
const TEXT = { name: 11, label: 10 };
const WIDE = 274; // the explainer column at desktop
const SANS = 'font-family="system-ui, sans-serif"';
const HALF = '½';

type Point = [number, number];
type Piece = [Point, Point];
const f = (v: number) => +v.toFixed(3);

/** columns cols[0] ... cols[1] - 1 and rows rows[0] ... rows[1] - 1, cell size U, top-left at (left, top) */
function grid(cols: [number, number], rows: [number, number], U: number, left: number, top: number, id: string) {
  const X = (x: number) => left + (x - cols[0]) * U;
  const Y = (y: number) => top + (rows[1] - y) * U;
  const out: string[] = [];
  let cells = '';
  for (let c = cols[0]; c < cols[1]; c++)
    for (let r = rows[0]; r < rows[1]; r++) cells += `<rect x="${X(c) + G / 2}" y="${Y(r + 1) + G / 2}" width="${U - G}" height="${U - G}" fill="${C.cell}"/>`;
  out.push(`<g shape-rendering="crispEdges">${cells}</g>`);
  // an end on the grid's edge runs past it and is clipped square at the cells
  const ex = (x: number) => (x === cols[0] ? X(x) - G : x === cols[1] ? X(x) + G : X(x));
  const line = ([[xa, ya], [xb, yb]]: Piece, colour: string) =>
    `<line x1="${f(ex(xa))}" y1="${f(Y(ya))}" x2="${f(ex(xb))}" y2="${f(Y(yb))}" stroke="${colour}" stroke-width="${LINE}" stroke-linecap="butt"/>`;
  const rowTop = (r: number) => Y(r + 1) + G / 2, rowBottom = (r: number) => Y(r) - G / 2;
  return {
    X, Y, out,
    /** later groups on top; where two pieces of one colour meet, a dot of the line's width closes the joint */
    rays(groups: [Piece[], string][]) {
      out.push(`<clipPath id="${id}"><rect x="${X(cols[0]) + G / 2}" y="${Y(rows[1])}" width="${X(cols[1]) - X(cols[0]) - G}" height="${(rows[1] - rows[0]) * U}"/></clipPath>`);
      out.push(`<g clip-path="url(#${id})">`);
      for (const [pieces, colour] of groups) out.push(...pieces.map((p) => line(p, colour)));
      for (const [pieces, colour] of groups) {
        const count = new Map<string, [Point, number]>();
        for (const p of pieces) for (const q of p) count.set(q.join(), [q, (count.get(q.join())?.[1] ?? 0) + 1]);
        for (const [[x, y], n] of count.values()) if (n > 1 && x !== cols[0] && x !== cols[1]) out.push(`<circle cx="${f(X(x))}" cy="${f(Y(y))}" r="${LINE / 2}" fill="${colour}"/>`);
      }
      out.push('</g>');
    },
    /** a 1-px span on the plane x over row r: in the gap, or flush inside the cell on the grid's edge */
    span(x: number, r: number, colour: string) {
      const x0 = x === cols[0] ? X(x) + G / 2 : x === cols[1] ? X(x) - G / 2 - W : X(x) - G / 2;
      out.push(`<rect x="${f(x0)}" y="${f(rowTop(r))}" width="${W}" height="${f(rowBottom(r) - rowTop(r))}" fill="${colour}" shape-rendering="crispEdges"/>`);
    },
    label([x, y]: Point, text: string, colour = C.text, anchor = 'middle') {
      out.push(`<text x="${f(X(x))}" y="${f(Y(y) + TEXT.label * 0.36)}" font-size="${TEXT.label}" fill="${colour}" text-anchor="${anchor}" ${SANS}>${text}</text>`);
    },
    name(text: string, centred = false) {
      const x = centred ? (X(cols[0]) + X(cols[1])) / 2 : X(cols[0]) + G / 2;
      out.push(`<text x="${f(x)}" y="${top - 5}" font-size="${TEXT.name}" fill="${C.dim}" text-anchor="${centred ? 'middle' : 'start'}" ${SANS}>${text}</text>`);
    },
  };
}
const svg = (height: number, scale: number, description: string, body: string) =>
  `<svg viewBox="0 0 ${WIDE} ${height}" width="${WIDE * scale}" height="${height * scale}" role="img" aria-label="${description}">${body}</svg>`;

// Ray extension and the merge at a 2× base ray count, for a straight and an odd direction. A cN+1 ray is two cN rays end
// to end, or for an odd direction the mean of its two bent paths; the merge runs the same paths backwards.
/** a cN+1 ray of length L and rise d as its cN paths: (rise to the odd plane, weight) */
const pathsOf = (L: number, d: number): [number, number][] =>
  d % 2 === 0 && Math.abs(d / 2) <= L / 2 ? [[d / 2, 1]] : [[(d - 1) / 2, 0.5], [(d + 1) / 2, 0.5]];
const at = (x: number, row: number): Point => [x, row + 0.5];

/** two columns (extension, merge) by two rows (a straight direction, an odd one) */
export function mergeDiagrams({ scale = 1 } = {}) {
  const U = 24, top = TEXT.name + 5, side = 14, between = WIDE - 2 * side - 8 * U, rowGap = 10;
  const height = 2 * (4 * U) + rowGap + top + 2;
  const body: string[] = [];
  [2, 3].forEach((d, row) => {
    ['extension', 'merge'].forEach((name, slot) => {
      const g = grid([0, 4], [0, 4], U, side + slot * (4 * U + between), top + row * (4 * U + rowGap), `merge-${name}-${d}`);
      const paths = pathsOf(4, d);
      g.out.push(`<line x1="${f(g.X(0))}" y1="${f(g.Y(0.5))}" x2="${f(g.X(4))}" y2="${f(g.Y(d + 0.5))}" stroke="${C.parent}" stroke-opacity="0.45" stroke-width="${3 * LINE}" stroke-linecap="round"/>`);
      g.rays([[paths.map(([m]): Piece => [at(0, 0), at(2, m)]).concat(paths.map(([m]): Piece => [at(2, m), at(4, d)])), C.child]]);
      for (const [m, w] of paths) if (w !== 1) g.label([2, m + 0.5 + (m > d / 2 ? 0.8 : -0.8)], HALF, slot === 0 ? C.dim : C.write);
      g.out.push(`<circle cx="${f(g.X(4))}" cy="${f(g.Y(d + 0.5))}" r="3.5" fill="${C.text}"/>`);
      if (slot === 1) {
        g.span(0, 0, C.read);
        g.out.push(`<circle cx="${f(g.X(0) + 5)}" cy="${f(g.Y(0.5))}" r="2.5" fill="${C.write}"/>`);
        for (const [m] of paths) g.span(2, m, C.write);
      }
      if (row === 0) g.name(name);
      body.push(g.out.join(''));
    });
  });
  return svg(height, scale, 'ray extension builds a cN+1 ray from 2 cN rays, or from 4 (two bent paths, ½ each) for an odd direction; the merge sends the cN+1 merged radiance back along the same paths', body.join(''));
}

/** a ray of length 4 and rise 2 and its halves A and B: the extensions write the ray at its right end, the merge each
 * half at its left end */
export function halvesDiagram({ scale = 1 } = {}) {
  const U = 32, outer = 2, between = WIDE - 8 * U - 2 * outer, top = TEXT.name + 5;
  const a: Piece = [[0, 0.5], [2, 1.5]], b: Piece = [[2, 1.5], [4, 2.5]];
  const body = ['extend', 'merge'].map((name, slot) => {
    const g = grid([0, 4], [0, 3], U, outer + slot * (4 * U + between), top, `halves-${name}`);
    g.rays([[[a], C.child], [[b], C.half]]);
    if (slot === 0) (g.span(2, 1, C.midpoint), g.span(4, 2, C.write));
    else (g.span(0, 0, C.write), g.span(2, 1, C.write));
    g.label([1, 1.6], 'A', C.child);
    g.label([3, 2.6], 'B', C.half);
    g.name(name);
    return g.out.join('');
  });
  return svg(top + 3 * U + 2, scale, 'a ray of length 4 and its halves A and B: the extension writes the ray at its right end; the merge writes each half at its left end', body.join(''));
}

/** the rays of length 4 from one probe at a 2× base ray count; a dot past the far end marks the rises 1× keeps */
export function raysDiagram({ scale = 1 } = {}) {
  const U = 24, L = 4, top = TEXT.name + 5;
  const g = grid([0, L], [-4, 5], U, (WIDE - L * U) / 2, top, 'rays');
  const rises = Array.from({ length: 2 * L + 1 }, (_, i) => i - L);
  const ray = (d: number): Piece => [[0, 0.5], [L, d + 0.5]];
  g.rays([[rises.map(ray), C.parent]]);
  g.span(0, 0, C.read);
  for (const d of rises.filter((d) => d % 2 === 0)) g.out.push(`<circle cx="${f(g.X(L) + 6)}" cy="${f(g.Y(d + 0.5))}" r="2" fill="${C.dim}"/>`);
  g.name('c2 rays (length 4), every slope', true);
  return svg(top + 9 * U + 2, scale, 'the c2 rays (length 4) from one probe at a 2× base ray count, every slope; dots mark the slopes a 1× count keeps', g.out.join(''));
}

/** a caption's key: per mark, a colour dot, a small marker dot or the mark's letter */
export function legendKey(items: [kind: 'dot' | 'mark' | 'letter', colour: string, text: string, letter?: string][]) {
  return `<ul class='key'>${items.map(([kind, colour, text, letter]) => `<li><i class='${kind}' style='--c: ${colour}'>${kind === 'letter' ? letter : ''}</i><span>${text}</span></li>`).join('')}</ul>`;
}
