// Scene uploads run on every edit, so they write into one reused array element by element: no per-pixel arrays, views
// or divides. writeBuffer copies the data when called, so the array can be refilled at once.

/** A Float32Array kept between calls and reallocated only when the length changes. It keeps its old contents: every
 *  element must be written. */
export function staging() {
  let data = new Float32Array(0);
  return (length: number) => (data.length === length ? data : (data = new Float32Array(length)));
}

/** Copies the N×N pixels at pixel offset `from`, (x, y) at x·N + y, to pixel offset `to` with (x, y) at y·N + x;
 *  `stride` floats per pixel. */
export function transpose(data: Float32Array, N: number, stride: number, from: number, to: number) {
  for (let x = 0; x < N; x++)
    for (let y = 0; y < N; y++) {
      const s = stride * (from + x * N + y), d = stride * (to + y * N + x);
      for (let c = 0; c < stride; c++) data[d + c] = data[s + c];
    }
}

/** (rgb, w) per pixel into `data` from emission (rgb per pixel): w is `opaque` where the pixel is solid or emits in any
 *  channel, else `clear`. With `litOnly`, rgb is 0 where no channel emits. */
export function rgba<T extends Float32Array>(data: T, { emission, solid }: { emission: Float32Array; solid: ArrayLike<number> }, opaque: number, clear: number, litOnly = false) {
  for (let i = 0, n = solid.length; i < n; i++) {
    const r = emission[3 * i], g = emission[3 * i + 1], b = emission[3 * i + 2], k = 4 * i, lit = r > 0 || g > 0 || b > 0, keep = lit || !litOnly;
    data[k] = keep ? r : 0;
    data[k + 1] = keep ? g : 0;
    data[k + 2] = keep ? b : 0;
    data[k + 3] = solid[i] || lit ? opaque : clear;
  }
  return data;
}
