// Dithered junctions (site/reciprocal-dither): which pairs of halves a ray is built from, plain against dithered, and
// the junction row read as a rounding.
//
// Plain: an even rise D is one straight pair (D/2, D/2); an odd one is (floor, ceil) and (ceil, floor) at ½. That puts
// the junction at round(D/2 + u), u uniform on a row (box dither): even rays never bend, odd rays always do, so the bend
// depends on the binary digits of D (the spokes). Dithered: an even D with 2 <= |D| <= L - 2 is (D/2, D/2) at ¾ and
// (D/2 ∓ 1, D/2 ± 1) at ⅛ each: round(D/2 + u - u') (triangular dither). Odd rises are unchanged. The bend b - a then
// has mean square 1 for every rise but 0 and ±L, which stay straight.
import { defineFigure, PROJ, projOptions, type Proj } from './figure';
import { pairsOf, panel, paintExtension, readsTable, frac, type Scheme } from './flow';

type S = Proj & { L: number; D: number };
const evenD = (s: S) => { const m = s.L - 2, d = 2 * Math.round(s.D / 2); return Math.max(2, Math.min(m, d)); };

export const dither = defineFigure<S>({
  id: 'dither',
  name: 'Dithered junctions',
  question: 'How does dithering the junction make every ray bend alike?',
  about: 'Each ray\'s pairs of halves with their weights, and below, where its junction lands. Plain rounds D/2 + u, so even rays never bend; dithered rounds D/2 + u − u′, so every ray bends with the same spread.',
  defaults: { ...PROJ, layer: 0.8, tilt: 50, L: 8, D: 4 },
  options: [
    { key: 'L', label: 'Ray length L', kind: 'choice', choices: [[8, '8'], [16, '16']] },
    { key: 'D', label: 'Even rise D (the odd one is D + 1)', kind: 'slider', min: 2, max: (s) => s.L - 2, step: 2 },
    ...projOptions(),
  ],
  N: (s) => s.L + 4,
  rows: (s) => 1.75 * s.L,   // the panel is L tall; the rest holds the junction bars and captions
  sheets: () => 4,
  cols: () => 2,
  draw(Dg, { s }) {
    const L = s.L, h = L / 2, W = L + 4, d0 = evenD(s);
    const cases: [Scheme, number][] = [['plain', d0], ['plain', d0 + 1], ['dither', d0], ['dither', d0 + 1]];
    cases.forEach(([sc, d], i) => {
      const P = pairsOf(sc, L, d), x0 = Math.round(W / 2 - 0.5 - d / 2) + 0.5;
      const spread = P.reduce((t, p) => t + p.w * (p.b - p.a) ** 2, 0);
      // where the junction lands: one bar per junction row, as tall as its weight, under the panel
      const base = -0.14 * L, tall = 0.32 * L, bottom = base - tall;
      panel(Dg, i, W, [0, h, L], `${sc === 'plain' ? 'plain' : 'dithered'}, D = ${d}`,
        [sc === 'plain' ? 'junction row = round(D/2 + u)' : 'junction row = round(D/2 + u − u′)', `bend spread ${frac(spread)} · reads ${2 * P.length}`], bottom,
        i % 2 === 0 ? ['start', 'junction', 'end'] : undefined);
      Dg.line(i, [x0, 0], [x0 + d, L], 'dashed', 'ink');
      paintExtension(Dg, i, sc, L, d, x0, false);
      Dg.ring(i, [x0 + d / 2, h], 1.6, 'ink');
      Dg.line(i, [0.2, base], [W - 0.2, base], 'fine', 'grey');
      Dg.line(i, [x0 + d / 2, base + 0.06 * L], [x0 + d / 2, bottom], 'fine', 'ink');
      for (const p of P) {
        const x = x0 + p.a, y = base - tall * p.w;
        Dg.area(i, [[x - 0.32, base], [x + 0.32, base], [x + 0.32, y], [x - 0.32, y]], 'hatch', 'red', 'thin');
        Dg.text(i, [x + 0.32, y], frac(p.w), 'start', 3, 9, 'red');
      }
    });
  },
  info: ({ s }) => [
    'Bend spread: the mean square of b − a, the second half\'s rise minus the first\'s.',
    '  plain: 0 for even D, 1 for odd D, so it follows the binary digits of D: spokes.',
    '  dithered: 1 for every D but 0 and ±L (those stay straight, so no path leaves 45°).',
    'The ring on the junction plane is where the straight ray crosses it.',
    '',
    readsTable(s.L),
  ].join('\n'),
});
