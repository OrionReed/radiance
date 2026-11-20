// The world (world units, y up): boxes that may move, the skies beyond them, and how boxes meet the cascade's pixels.

import type { Sky } from './producer';

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  light?: [number, number, number];
  /** offset at time t (seconds) */
  move?: (t: number) => [number, number];
}

const orbit = (r: number, period: number, phase = 0) => (t: number): [number, number] => [r * Math.cos((2 * Math.PI * t) / period + phase), r * Math.sin((2 * Math.PI * t) / period + phase)];
const slide = (dx: number, dy: number, period: number, phase = 0) => (t: number): [number, number] => {
  const k = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / period + phase);
  return [dx * k, dy * k];
};
const wall = (x0: number, y0: number, x1: number, y1: number, move?: Box['move']): Box => ({ x0, y0, x1, y1, move });
const light = (x: number, y: number, size: number, rgb: [number, number, number], move?: Box['move']): Box => ({ x0: x - size / 2, y0: y - size / 2, x1: x + size / 2, y1: y + size / 2, light: rgb, move });

export const WORLD: Box[] = [
  // room A, with a doorway east
  wall(-600, -400, -580, 400), wall(-600, 380, 0, 400), wall(-600, -400, 0, -380), wall(-20, -400, 0, -60), wall(-20, 60, 0, 400),
  // a corridor to the hall
  wall(0, 40, 700, 60), wall(0, -60, 700, -40),
  // the hall, with gaps and pillars
  wall(700, -500, 720, -40), wall(700, 60, 720, 500), wall(700, 480, 1500, 500), wall(700, -500, 1500, -480), wall(1480, -500, 1500, 100),
  ...[0, 1, 2].flatMap((i) => [wall(860 + 200 * i, -240, 900 + 200 * i, -200), wall(860 + 200 * i, 200, 900 + 200 * i, 240)]),
  // a sliding door in the corridor, a sliding screen in room A
  wall(340, -40, 360, 40, slide(0, 70, 6)),
  wall(-320, -150, -300, 150, slide(160, 0, 9, 1)),
  light(-450, 250, 12, [8, 6, 3.5]), light(1300, 300, 10, [3, 6, 9]), light(1300, -300, 10, [9, 4, 3]),
  light(-300, 0, 10, [6, 9, 5], orbit(180, 7)), light(1100, 0, 12, [9, 8, 6], orbit(220, 11, 2)), light(500, 0, 6, [9, 9, 9], slide(150, 0, 5)),
  // far to the west, a bright source behind a slatted fence
  light(-4200, 0, 160, [60, 52, 40]), ...[-3, -2, -1, 0, 1, 2, 3].map((i) => wall(-2400, 120 * i - 30, -2380, 120 * i + 30)),
];

export function worldAt(t: number) {
  return WORLD.map((b) => {
    const [dx, dy] = b.move ? b.move(t) : [0, 0];
    return { ...b, x0: b.x0 + dx, y0: b.y0 + dy, x1: b.x1 + dx, y1: b.y1 + dy };
  });
}

/** a sun: where its light comes from and its size across (degrees, 0 from the east), its irradiance, and its colour
 * from cool (-1) through white to warm (1); with a faint blue sky */
export interface Sun {
  angle: number;
  size: number;
  intensity: number;
  warmth: number;
}
export function sunSky({ angle, size, intensity, warmth }: Sun): Sky {
  const toward = warmth >= 0 ? [1, 0.82, 0.6] : [0.72, 0.84, 1], w = Math.abs(warmth);
  const rgb = toward.map((c) => intensity * (1 + w * (c - 1))) as [number, number, number];
  return { ambient: [0.012, 0.018, 0.03], sun: { rgb, at: (angle * Math.PI) / 180, radius: (size * Math.PI) / 360 } };
}
export const SUN: Sun = { angle: 143, size: 0.5, intensity: 1.5, warmth: 0.6 };

/** by the world angle the light comes from (0: from the east) */
export const SKIES: Record<string, Sky | null> = {
  none: null,
  overcast: { ambient: [0.11, 0.12, 0.13] },
  sun: sunSky(SUN),
  dusk: { ambient: [0.07, 0.05, 0.06], gradient: [0.06, 0.025, -0.03], towards: Math.PI },
  bands: { ambient: [0.05, 0.065, 0.09], bands: [0.045, 0.06, 0.085], towards: Math.PI / 2 },
};

/** How boxes meet the cascade's pixels:
 * - size: the size rounded (at least one pixel), placed at the pixel edge nearest its own;
 * - edges: each edge rounded to a pixel edge (at least one pixel across);
 * - lights: lights as they are, each pixel taking the share it covers; walls as in size;
 * - coverage: every box as it is, each pixel taking the share it covers. */
export const RASTERS = ['size', 'edges', 'lights', 'coverage'] as const;
export type Raster = (typeof RASTERS)[number];

export interface Domain {
  pixel: number;
  x: number;
  y: number;
}

/** boxes in domain pixels, 8 floats each (x0, y0, x1, y1, rgb, 1 if a light), as `raster` places them; exact: as they are */
export function placed(boxes: Box[], domain: Domain, raster: Raster | 'exact') {
  const data = new Float32Array(Math.max(1, boxes.length) * 8);
  boxes.forEach((b, i) => {
    const to = (v: number, o: number) => (v - o) / domain.pixel;
    let x0 = to(b.x0, domain.x), x1 = to(b.x1, domain.x), y0 = to(b.y0, domain.y), y1 = to(b.y1, domain.y);
    const keep = raster === 'exact' || raster === 'coverage' || (raster === 'lights' && b.light);
    if (raster === 'edges') {
      (x0 = Math.round(x0)), (x1 = Math.round(x1)), (y0 = Math.round(y0)), (y1 = Math.round(y1));
      if (x1 <= x0) x1 = x0 + 1;
      if (y1 <= y0) y1 = y0 + 1;
    } else if (!keep) {
      const w = Math.max(1, Math.round(x1 - x0)), h = Math.max(1, Math.round(y1 - y0));
      (x0 = Math.round(0.5 * (x0 + x1 - w))), (y0 = Math.round(0.5 * (y0 + y1 - h))), (x1 = x0 + w), (y1 = y0 + h);
    }
    data.set([x0, y0, x1, y1, ...(b.light ?? [0, 0, 0]), b.light ? 1 : 0], 8 * i);
  });
  return data;
}
