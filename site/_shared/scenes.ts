// The scenes every scheme's page shows, so schemes can be compared on the same input.
//
// Coordinates are scene units: 1024 across the square, origin at its centre, y down, independent of the resolution N.
// A shape is a rectangle. Brightness is its radiance (mono): what a solid shows, and what a thick enough volume glows
// with; a light also has a colour, linear rgb whose luminance is its brightness, for schemes that render colour. Opacity
// is per reference cell, 1/256 of the square: 1 is solid, below 1 a volume. No shared scene uses opacity 0 (a see-through
// light), which not every scheme can model.
//
// Tests are drawn in pixels at a resolution of their own, which the page imposes, so results compare exactly across
// schemes and with other implementations.

export interface Shape {
  x: number;
  y: number;
  w: number;
  h: number;
  brightness: number;
  opacity: number;
  color?: [number, number, number];
}
export interface Scene {
  id: string;
  group: string;
  name: string;
  shapes: Shape[];
  N?: number; // tests: the resolution they are drawn at
}

export const SPAN = 1024;
export const REFERENCE_CELLS = 256;

type RGB = [number, number, number];
type Row = [x: number, y: number, w: number, h: number, light?: number | RGB, opacity?: number];

export const scenes: Scene[] = [];
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
function scene(group: string, name: string, rows: Row[], N?: number) {
  const shapes = rows.map(([x, y, w, h, light = 0, opacity = 1]): Shape =>
    typeof light === 'number' ? { x, y, w, h, brightness: light, opacity } : { x, y, w, h, brightness: lum(...light), opacity, color: light });
  scenes.push({ id: slug(name), group, name, shapes, ...(N ? { N } : {}) });
}

// light colours
function lum(r: number, g: number, b: number) { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
const RED: RGB = [1, 0.25, 0.25], ORANGE: RGB = [1, 0.5, 0.2], YELLOW: RGB = [0.75, 0.75, 0.2], GREEN: RGB = [0.25, 0.8, 0.35];
const CYAN: RGB = [0.25, 0.75, 0.75], BLUE: RGB = [0.3, 0.4, 1], PURPLE: RGB = [0.65, 0.3, 1], WHITE: RGB = [0.8, 0.8, 0.8];

// ── lights ──
scene('lights', 'Point + wall', [[168, -12, 24, 24, WHITE], [-200, -60, 40, 160], [-60, 40, 14, 14]]);
scene('lights', 'Three lights', [[172, -10, 16, 16, WHITE], [194, -88, 14, 14, ORANGE], [232, 62, 18, 18, CYAN], [-30, -10, 40, 20]]);
scene('lights', 'Room', [
  [-250, -200, 500, 12], [-250, 188, 500, 12], [-250, -200, 12, 400], [238, -200, 12, 400],
  [-126, -126, 20, 20, WHITE], [114, 84, 16, 16, ORANGE], [-20, -20, 60, 40], [60, -140, 8, 120],
]);
// a light behind a wall with a gap, an absorbing box, a second light
scene('lights', 'Gap + box', [
  [-8, -8, 16, 16, WHITE], [192, -320, 16, 272], [192, 48, 16, 272], [-352, -240, 192, 16],
  [-256, 128, 160, 160, 0, 0.2], [352, -328, 16, 16, [0.6, 0.6, 0.6]],
]);
// a thin glowing volume beside a solid light of the same size, and a larger dim one
scene('lights', 'Volume lights', [[-200, -100, 24, 24, 40, 0.02], [-200, 76, 24, 24, WHITE], [-60, -200, 12, 400], [140, -60, 60, 120, ORANGE, 0.05]]);

// ── occluders ──
scene('occluders', 'Pinhole', [
  [-200, -84, 16, 24, RED], [-200, -60, 16, 24, ORANGE], [-200, -36, 16, 24, YELLOW], [-200, -12, 16, 24, GREEN],
  [-200, 12, 16, 24, CYAN], [-200, 36, 16, 24, BLUE], [-200, 60, 16, 24, PURPLE],
  [-5, -500, 8, 488], [-5, 12, 8, 500],
]);
scene('occluders', 'Center block', [[-60, -60, 120, 120]]);
scene('occluders', 'Thin walls', [[200, -10, 20, 20, WHITE], [-200, -80, 400, 1], [-200, -20, 400, 2], [-200, 40, 400, 4], [-200, 110, 400, 8]]);
scene('occluders', 'Two lights', [[-200, -100, 40, 40, RED], [-200, 60, 40, 40, BLUE], [-40, -80, 20, 160], [100, -120, 10, 240, WHITE]]);
scene('occluders', 'Slats', [[-300, -30, 30, 60, WHITE], ...Array.from({ length: 9 }, (_, i): Row => [-40, -200 + i * 48, 12, 24]), [150, -60, 4, 120]]);

// ── media ──
scene('media', 'Glowing fog', [
  [-200, -40, 60, 60, WHITE], [-80, -100, 60, 200, RED, 0.015], [-10, -100, 60, 200, GREEN, 0.015], [60, -100, 60, 200, BLUE, 0.015], [180, -40, 40, 80],
]);
scene('media', 'Fog room', [
  [-250, -200, 500, 15], [-250, 185, 500, 15], [-250, -200, 15, 400], [235, -200, 15, 400],
  [-200, -185, 30, 30, ORANGE], [170, 155, 30, 30, CYAN], [-100, -50, 200, 100, 0, 0.01],
]);
scene('media', 'Smoke', [[-30, 80, 60, 30, ORANGE], [-80, -100, 160, 160, 0, 0.05], [-200, -60, 30, 30], [170, -60, 30, 30]]);
scene('media', 'Absorbers', [[-260, -20, 30, 40, WHITE], ...[0.02, 0.05, 0.1, 0.2, 0.4, 0.7].map((o, i): Row => [-180 + i * 70, -120, 40, 240, 0, o])]);

scene('empty', 'Empty', []);

// ── tests ──
// A small standard set, one case of an error class each, from single cases to combinations: a light alone, a light and one
// occluder edge, an aperture, corner-touching and scattered occluders, media, then the hard cases together. Tests are drawn
// in pixels at 128² (Occlusion stress at 512²), rows counted down from the top. Lights sit off the centre and off the axes,
// so errors that are symmetric about them do not cancel.
// pixels [x0, x1) × [y0, y1) at resolution N
const pixels = (N: number) => (x0: number, x1: number, y0: number, y1: number, brightness = 0, opacity = 1): Row => {
  const u = SPAN / N;
  return [x0 * u - SPAN / 2, y0 * u - SPAN / 2, (x1 - x0) * u, (y1 - y0) * u, brightness, opacity];
};
const px = pixels(128);
const light = (x: number, y: number, size: number, brightness = 1) => px(x, x + size, y, y + size, brightness);
// opacity per reference cell for a transmittance t across one pixel at 128²
const across = (t: number) => 1 - t ** (128 / REFERENCE_CELLS);
const rng = (seed: number) => () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

// a one-pixel light in free space: falloff and directional artefacts with nothing in the way
scene('tests', 'Pixel light', [light(44, 52, 1)], 128);
// an 8×8 light: an extended source in free space
scene('tests', 'Square light', [light(40, 48, 8)], 128);
// an irregular light with one-pixel steps along its edge: 61 pixels grown at random from one
scene('tests', 'Blob light', (() => {
  const r = rng(5), grown = new Set(['44,60']);
  while (grown.size < 61) {
    const [x, y] = [...grown][Math.floor(r() * grown.size)].split(',').map(Number), d = [[1, 0], [-1, 0], [0, 1], [0, -1]][Math.floor(r() * 4)];
    grown.add(`${x + d[0]},${y + d[1]}`);
  }
  return [...grown].map((k) => { const [x, y] = k.split(',').map(Number); return light(x, y, 1); });
})(), 128);
// RC's scene: a one-pixel light of radiance 5 at (36, 32) and a wall along row 60 over columns 0 .. 49
scene('tests', 'RC: pixel', [light(36, 32, 1, 5), px(0, 50, 60, 61)], 128);
// RC's offline scene: an 8×8 light at (36, 32) and the same wall
scene('tests', 'RC: 8×8', [light(36, 32, 8), px(0, 50, 60, 61)], 128);
// a light behind a one-pixel wall with a one-pixel hole: a narrow beam
scene('tests', 'Pixel hole', [light(20, 63, 2), px(48, 49, 0, 64), px(48, 49, 65, 128)], 128);
// a light beside a diagonal wall of single pixels that touch only at their corners: leaks between the corners
scene('tests', 'Diagonal wall', [light(24, 56, 4), ...Array.from({ length: 64 }, (_, i) => px(52 + i, 53 + i, 28 + i, 29 + i))], 128);
// a light among single-pixel occluders, 4 % of the pixels to its right
scene('tests', 'Clutter', (() => {
  const r = rng(3), list = [light(20, 62, 4)];
  for (let x = 36; x < 128; x++) for (let y = 0; y < 128; y++) if (r() < 0.04) list.push(px(x, x + 1, y, y + 1));
  return list;
})(), 128);
// media: an absorbing block lets about a quarter through; a thin glowing block, and a dense one that is nearly a solid light
scene('tests', 'Fog shadow', [light(28, 62, 4), px(52, 64, 50, 80, 0, across(0.89))], 128);
scene('tests', 'Thin glow', [px(52, 76, 52, 76, 1, across(0.95))], 128);
scene('tests', 'Dense glow', [px(52, 76, 52, 76, 1, across(0.55))], 128);
// the hard cases for visibility together: a light boxed in behind a slit, a grating of one-pixel slats, a diagonal of blocks
// touching only at corners, a wall with a one-pixel hole, a checkerboard, a long wall one pixel thick seen along its length,
// a field of one-pixel dots, a light touching a block, and a light one pixel thick
scene('tests', 'Stress test', [
  [-380, -8, 16, 16, WHITE],
  [-420, -60, 100, 6], [-420, 54, 100, 6], [-420, -60, 6, 120], [-326, -60, 6, 58], [-326, 2, 6, 58],
  ...Array.from({ length: 10 }, (_, i): Row => [-240 + i * 8, -40, 2, 80]),
  [200, -420, 12, 12, ORANGE], ...Array.from({ length: 25 }, (_, i): Row => [40 + i * 4, -300 + i * 4, 4, 4]),
  [250, -110, 6, 6, GREEN], [300, -200, 4, 96], [300, -102, 4, 96],
  [320, 170, 10, 10, CYAN],
  ...Array.from({ length: 64 }, (_, k) => [k % 8, k >> 3]).filter(([i, j]) => (i + j) % 2 === 0).map(([i, j]): Row => [260 + 4 * i, 160 + 4 * j, 4, 4]),
  [-150, 100, 500, 2],
  ...Array.from({ length: 36 }, (_, k): Row => [-300 + (k % 6) * 20 + (Math.floor(k / 6) & 1) * 10, 200 + Math.floor(k / 6) * 20, 2, 2]),
  [-120, 250, 60, 60], [-60, 270, 8, 20, BLUE],
  [0, -480, 200, 2, PURPLE],
], 512);

// the scenes a scheme can draw: one that models only solids drops every scene with a volume
export const scenesFor = ({ volumes }: { volumes: boolean }) => scenes.filter((s) => volumes || s.shapes.every((sh) => sh.opacity >= 1));
// transmittance across one pixel at resolution N
export const pixelTransmittance = (opacity: number, N: number) => (opacity >= 1 ? 0 : (1 - opacity) ** (REFERENCE_CELLS / N));
