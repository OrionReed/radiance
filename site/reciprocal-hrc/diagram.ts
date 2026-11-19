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

// The merge step: a child of length 4 and rise 0 from parents of length 8, bent parents in both orders. A parent piece
// that is also a child ray is drawn once, as the child.
const chainsToPieces = (chains: Point[][]) => chains.flatMap((c) => c.slice(1).map((q, i): Piece => [c[i], q]));
const key = ([[a, b], [c, d]]: Piece) => [a, b, c, d].join();
function pieces(parentChains: Point[][], childChains: Point[][]) {
  const child = chainsToPieces(childChains), used = new Set(child.map(key)), seen = new Set<string>();
  const parent = chainsToPieces(parentChains).filter((p) => !used.has(key(p)) && !seen.has(key(p)) && seen.add(key(p)));
  return { parent, child };
}
const MERGE_COLS: [number, number] = [-8, 8], MERGE_ROWS: [number, number] = [-2, 3], MERGE_U = 16, MERGE_LEFT = 14;
// each parent's share, by the row it is read on
const SHARES_1X: [number, string][] = [[2.5, HALF], [0.5, '1'], [-1.5, HALF]];
const SHARES_2X: [number, string][] = [[1.5, HALF], [0.5, '1'], [-0.5, HALF]];
const MERGE = {
  // P's own ray runs from P (plane 4) to Q (plane 0); all three parents are read at Q
  hrc: {
    name: 'Normal HRC merge',
    ...pieces(
      [
        [[0, 0.5], [-4, 0.5], [-8, 0.5]],
        [[0, 0.5], [-4, 0.5], [-8, 2.5]],
        [[0, 0.5], [-4, 2.5], [-8, 2.5]],
        [[0, 0.5], [-4, 0.5], [-8, -1.5]],
        [[0, 0.5], [-4, -1.5], [-8, -1.5]],
      ],
      [[[0, 0.5], [4, 0.5]]],
    ),
  },
  // B (columns 4-7, row 0) is the right half of all three parents, each read at its own start through its own left half
  reciprocal: {
    name: 'Reciprocal HRC merge, 1×',
    ...pieces(
      [
        [[0, 0.5], [4, 0.5], [8, 0.5]],
        [[0, -1.5], [4, 0.5], [8, 0.5]],
        [[0, -1.5], [4, -1.5], [8, 0.5]],
        [[0, 2.5], [4, 0.5], [8, 0.5]],
        [[0, 2.5], [4, 2.5], [8, 0.5]],
      ],
      [[[0, 0.5], [4, 0.5]], [[0, -1.5], [4, 0.5]], [[0, 2.5], [4, 0.5]], [[4, 0.5], [8, 0.5]]],
    ),
  },
  // at 2× B's parents have rises 0 and ±1; a parent of rise ±1 is the mean of halves (±1, 0) and (0, ±1)
  reciprocal2x: {
    name: 'Reciprocal HRC merge, 2× (default)',
    ...pieces(
      [
        [[0, 0.5], [4, 0.5], [8, 0.5]],
        [[0, -0.5], [4, 0.5], [8, 0.5]],
        [[0, -0.5], [4, -0.5], [8, 0.5]],
        [[0, 1.5], [4, 0.5], [8, 0.5]],
        [[0, 1.5], [4, 1.5], [8, 0.5]],
      ],
      [[[0, 0.5], [4, 0.5]], [[0, -0.5], [4, 0.5]], [[0, 1.5], [4, 0.5]], [[4, 0.5], [8, 0.5]]],
    ),
  },
};

/** the merge step for HRC and reciprocal HRC at 1× and 2×, on one column grid so the read and write planes line up */
export function mergeDiagrams({ scale = 1 } = {}) {
  const top = TEXT.name + 5, height = top + (MERGE_ROWS[1] - MERGE_ROWS[0]) * MERGE_U + 2;
  const panel = (which: 'hrc' | 'reciprocal' | 'reciprocal2x') => {
    const step = MERGE[which], g = grid(MERGE_COLS, MERGE_ROWS, MERGE_U, MERGE_LEFT, top, `merge-${which}`);
    const isB = (p: Piece) => which !== 'hrc' && key(p) === key([[4, 0.5], [8, 0.5]]);
    g.rays([[step.parent, C.parent], [step.child.filter((p) => !isB(p)), C.child], [step.child.filter(isB), C.half]]);
    if (which === 'hrc') {
      g.span(0, 0, C.read);
      g.span(4, 0, C.write);
      g.label([0, 1.5], 'Q', C.read);
      g.label([4, 1.5], 'P', C.write);
    } else {
      for (const r of which === 'reciprocal' ? [0, -2, 2] : [0, -1, 1]) g.span(0, r, C.read);
      g.span(4, 0, C.write);
      g.label([7.5, 1.5], 'B', C.half);
    }
    for (const [y, text] of which === 'reciprocal2x' ? SHARES_2X : SHARES_1X) g.out.push(`<text x="${f(g.X(MERGE_COLS[0]) - 3)}" y="${f(g.Y(y) + TEXT.label * 0.36)}" font-size="${TEXT.label}" fill="${C.dim}" text-anchor="end" ${SANS}>${text}</text>`);
    g.name(step.name);
    return g.out.join('');
  };
  return {
    hrc: svg(height, scale, 'HRC: P reads all three parents at Q, the end of its own ray', panel('hrc')),
    reciprocal: svg(height, scale, 'reciprocal HRC: B is the right half of three parents, each read at its own start through its own left half', panel('reciprocal')),
    reciprocal2x: svg(height, scale, 'reciprocal HRC at a 2× base ray count: B is the right half of parents of rises 0 and ±1, each read at its own start', panel('reciprocal2x')),
  };
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
