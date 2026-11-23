// Shapes, pen strokes and the pen's cursor preview, drawn into pixels the same way on every page.
//
// Positions are scene units (SPAN across, origin at the centre). A pixel is covered when its centre is inside the box,
// and a box thinner than a pixel is widened to one about its middle, so a thin wall is never lost (`snap` rounds the
// edges instead). Pixel (gx, gy), gy up, is i = gx·N + (N − 1 − gy). Later items overwrite earlier ones.

import { SPAN, pixelTransmittance, type Shape } from './scenes';

export interface Box { x: number; y: number; w: number; h: number }
export interface Stroke { x0: number; y0: number; x1: number; y1: number; size: number }   // size in scene units
export interface Cursor<M> { x: number; y: number; size: number; m: M }                    // size in pixels
export type Put<M> = (i: number, m: M) => void;

/** Every pixel of a box given in pixel units, gy up. */
export function box<M>(N: number, x0: number, x1: number, y0: number, y1: number, m: M, put: Put<M>, snap = false) {
  if (!(x1 - x0 >= 1)) [x0, x1] = [(x0 + x1) / 2 - 0.5, (x0 + x1) / 2 + 0.5];
  if (!(y1 - y0 >= 1)) [y0, y1] = [(y0 + y1) / 2 - 0.5, (y0 + y1) / 2 + 0.5];
  const edge = snap ? Math.round : (a: number) => Math.ceil(a - 0.5);
  const gx1 = Math.min(N, edge(x1)), gy0 = Math.max(0, edge(y0)), gy1 = Math.min(N, edge(y1));
  for (let gx = Math.max(0, edge(x0)); gx < gx1; gx++) for (let gy = gy0; gy < gy1; gy++) put(gx * N + (N - 1 - gy), m);
}

/** A square brush of `size` pixels swept from a to b, in pixel units. */
export function sweep<M>(N: number, ax: number, ay: number, bx: number, by: number, size: number, m: M, put: Put<M>, snap = false) {
  const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) * 4)), h = size / 2;
  for (let t = 0; t <= n; t++) {
    const cx = ax + ((bx - ax) * t) / n, cy = ay + ((by - ay) * t) / n;
    box(N, cx - h, cx + h, cy - h, cy + h, m, put, snap);
  }
}

/** Shapes, then strokes, then the cursor preview: each item is its own material. */
export function cover<B extends Box, S extends Stroke, C>(
  N: number,
  { shapes, strokes = [], cursor = null, snap = false }: { shapes: Iterable<B>; strokes?: Iterable<S>; cursor?: Cursor<C> | null; snap?: boolean },
  put: Put<B | S | C>,
) {
  const k = N / SPAN, px = (u: number) => (u + SPAN / 2) * k;
  for (const s of shapes) box(N, px(s.x), px(s.x + s.w), px(s.y), px(s.y + s.h), s, put, snap);
  for (const s of strokes) sweep(N, px(s.x0), px(s.y0), px(s.x1), px(s.y1), s.size * k, s, put, snap);
  if (cursor) sweep(N, px(cursor.x), px(cursor.y), px(cursor.x), px(cursor.y), cursor.size, cursor.m, put, snap);
}

// The editor's material: rgb is the emitted colour (already scaled by brightness b), o < 1 a medium's opacity per
// reference pixel. Without an absorption field every item is solid.
export interface Material { b: number; rgb: ArrayLike<number>; o?: number }
export interface Fields { emission: Float32Array; solid: Float32Array; absorption?: Float32Array }

/** Clears the fields and paints the editor's shapes, strokes and cursor into them. */
export function paint<B extends Box & Material, S extends Stroke & Material>(
  N: number,
  { emission, solid, absorption }: Fields,
  layer: { shapes: Iterable<B>; strokes?: Iterable<S>; cursor?: Cursor<Material> | null },
) {
  emission.fill(0);
  solid.fill(0);
  absorption?.fill(0);
  let last: Material | null = null, sigma = 0;
  cover(N, layer, (i, m) => {
    const volume = absorption !== undefined && m.o !== undefined && m.o < 1;
    if (m !== last) ((last = m), (sigma = volume ? -Math.log(pixelTransmittance(m.o!, N)) : 0));
    solid[i] = volume ? 0 : 1;
    if (absorption) absorption[i] = sigma;
    const lit = m.b > 0, k = 3 * i;
    emission[k] = lit ? m.rgb[0] : 0;
    emission[k + 1] = lit ? m.rgb[1] : 0;
    emission[k + 2] = lit ? m.rgb[2] : 0;
  });
}

/** A shared scene's shapes as (emission, solid), and with `media` its absorption: a volume (opacity < 1) absorbs −ln
 *  of its pixel transmittance and is not solid. Without `media` every shape is solid. */
export function rasteriseScene(shapes: Shape[], N: number, { media = false } = {}) {
  const emission = new Float32Array(N * N * 3), solid = new Float32Array(N * N), absorption = media ? new Float32Array(N * N) : undefined;
  const items = shapes.map((s) => ({ ...s, b: s.brightness, rgb: s.color ?? [s.brightness, s.brightness, s.brightness], o: s.opacity }));
  paint(N, { emission, solid, absorption }, { shapes: items });
  return absorption ? { emission, solid, absorption, media: absorption.some((a) => a > 0) } : { emission, solid };
}
