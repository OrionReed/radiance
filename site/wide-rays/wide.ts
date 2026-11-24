// Reciprocal HRC with fewer directions. A ray of length L starts in one row; with direction spacing s it ends either
// in one row (today's rays: lines between the windows of neighbouring rises fall in no ray when s > 1) or in a window
// s rows wide (wide rays: neighbouring windows tile the end column, so every line is in exactly one ray).
// Port of folkjs research/community/dual_rc/few_directions/wide.py. One frustum carries light toward +x; arrays are
// [x][y] flattened as x * N + y.

export type Windows = 'wide' | 'one row';
export type Split = readonly [a: number, j: number, b: number, w: number];

export const risesOf = (L: number, s: number) => {
  const out: number[] = [];
  for (let d = -Math.floor(L / s) * s; d <= L; d += s) out.push(d);
  return out;
};

const cache = new Map<string, Split[]>();

/** a ray of length L2 and rise D, spacing sP, as its halves (spacing sC): first half rise a from the same row, second
 * half from row + j with rise b, weighted by the share of its lines going that way */
export function splits(L2: number, D: number, sP: number, sC: number, windows: Windows): Split[] {
  const key = `${L2} ${D} ${sP} ${sC} ${windows}`;
  let out = cache.get(key);
  if (out) return out;
  if (windows === 'one row') {
    // today's rule: equal halves when they exist, else 1/2 each order of the two nearest rises
    const half = risesOf(L2 / 2, sC);
    if (D % 2 === 0 && half.includes(D / 2)) out = [[D / 2, D / 2, D / 2, 1]];
    else {
      const lo = (D - sC) / 2, hi = (D + sC) / 2;
      out = [[Math.floor(lo), Math.floor(lo), Math.ceil(hi), 0.5], [Math.ceil(hi), Math.ceil(hi), Math.floor(lo), 0.5]];
    }
  } else {
    const counts = new Map<string, number>();
    const n0 = 48, n2 = 48 * sP;
    let total = 0;
    for (const shift of [0.25, 0.75])
      for (let i = 0; i < n0; i++) {
        const y0 = (i + shift) / n0;
        for (let k = 0; k < n2; k++) {
          const y2 = D + 0.5 - sP / 2 + ((k + shift) / n2) * sP;
          if (Math.abs(y2 - y0) > L2) continue; // steeper than 45 degrees: the next frustum's line
          const y1 = 0.5 * (y0 + y2), j = Math.floor(y1);
          const a = sC * Math.floor((y1 - 0.5) / sC + 0.5), b = sC * Math.floor((y2 - j - 0.5) / sC + 0.5);
          const id = `${a} ${j} ${b}`;
          counts.set(id, (counts.get(id) ?? 0) + 1);
          total++;
        }
      }
    out = [...counts].map(([id, c]) => {
      const [a, j, b] = id.split(' ').map(Number);
      return [a, j, b, c / total] as const;
    });
  }
  cache.set(key, out);
  return out;
}

const coneWeight = (L: number, d: number, s: number) =>
  Math.atan(Math.min(1, (d + s / 2) / L)) - Math.atan(Math.max(-1, (d - s / 2) / L));

export interface Work { rays: number; pairs: number }

/** one frustum's fluence (light toward +x), multiplied by free; emission and free are [x * N + y] */
/** keep: receives the far values of the length-1 rays, [(x * rows + row) * 3 + rise + 1] with row = y + pad */
export function frustum(N: number, emission: Float32Array, free: Float32Array, spacing: (L: number) => number, windows: Windows, work?: Work, keep?: (G: Float32Array, rows: number, pad: number) => void) {
  const pad = N, rows = N + 2 * pad;
  const e = (x: number, y: number) => (y < pad || y >= pad + N ? 0 : emission[x * N + y - pad]);
  const f = (x: number, y: number) => (y < pad || y >= pad + N ? 0 : free[x * N + y - pad]);
  // ray arrays per length: [block][row][rise]
  const T: Float32Array[] = [], R: Float32Array[] = [], D: number[][] = [], S: number[] = [];
  const at = (L: number, block: number, row: number, k: number) => (block * rows + row) * D[L].length + k;
  D[1] = [-1, 0, 1]; S[1] = 1;
  T[1] = new Float32Array(N * rows * 3); R[1] = new Float32Array(N * rows * 3);
  for (let x = 0; x < N; x++)
    for (let y = 0; y < rows; y++)
      for (let k = 0; k < 3; k++) {
        const d = k - 1, i = at(1, x, y, k), fy = f(x, y), fd = f(x, y + d);
        T[1][i] = fy * fd;
        R[1][i] = e(x, y + d) + fd * e(x, y);
      }
  const ids = (L: number) => new Map(D[L].map((d, k) => [d, k]));
  for (let L = 2; L <= N / 2; L *= 2) {
    const h = L / 2, sP = spacing(L), sC = S[h];
    D[L] = risesOf(L, sP); S[L] = sP;
    const H = ids(h), blocks = N / L, nd = D[L].length;
    const t = new Float32Array(blocks * rows * nd), r = new Float32Array(blocks * rows * nd);
    let pairs = 0;
    D[L].forEach((d, k) => {
      for (const [a, j, b, w] of splits(L, d, sP, sC, windows)) {
        const ia = H.get(a), ib = H.get(b);
        if (ia === undefined || ib === undefined) continue; // lines leaving the frustum
        pairs++;
        for (let p = 0; p < blocks; p++)
          for (let y = 0; y < rows; y++) {
            const yb = y + j;
            if (yb < 0 || yb >= rows) continue;
            const A = at(h, 2 * p, y, ia), B = at(h, 2 * p + 1, yb, ib), o = (p * rows + y) * nd + k;
            const tB = T[h][B];
            t[o] += w * T[h][A] * tB;
            r[o] += w * (R[h][B] + tB * R[h][A]);
          }
      }
    });
    T[L] = t; R[L] = r;
    if (work) { work.rays += nd / L; work.pairs += pairs / L; }
  }
  // merge: the light arriving at each ray's start from behind it, longest rays first
  S[N] = spacing(N); D[N] = risesOf(N, S[N]);
  let G = new Float32Array(rows * D[N].length);
  for (let L = N; L >= 2; L /= 2) {
    const h = L / 2, sP = S[L], sC = S[h], H = ids(h), nh = D[h].length, nd = D[L].length;
    const g = new Float32Array((N / h) * rows * nh);
    D[L].forEach((d, k) => {
      const c = coneWeight(L, d, sP);
      for (const [a, j, b, w] of splits(L, d, sP, sC, windows)) {
        const ia = H.get(a), ib = H.get(b);
        if (ia === undefined || ib === undefined) continue;
        for (let p = 0; p < N / L; p++)
          for (let y = 0; y < rows; y++) {
            const far = G[(p * rows + y) * nd + k];
            const A = at(h, 2 * p, y, ia);
            g[((2 * p) * rows + y) * nh + ia] += w * far;
            const yb = y + j;
            if (yb < 0 || yb >= rows) continue;
            g[((2 * p + 1) * rows + yb) * nh + ib] += w * (T[h][A] * far + c * R[h][A]);
          }
      }
    });
    G = g;
  }
  keep?.(G, rows, pad);
  const out = new Float32Array(N * N);
  for (let x = 0; x < N; x++)
    for (let y = pad; y < pad + N; y++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        const d = k - 1;
        sum += 0.5 * G[(x * rows + y) * 3 + k];
        const ys = y - d;
        if (ys >= 0 && ys < rows) sum += 0.5 * f(x, ys) * G[(x * rows + ys) * 3 + k];
      }
      out[x * N + y - pad] = sum * f(x, y);
    }
  return out;
}

/** fluence over all four frustums: emission and solid are [x * N + y] */
export function solve(N: number, emission: Float32Array, solid: Uint8Array, spacing: (L: number) => number, windows: Windows, work?: Work) {
  const free = new Float32Array(N * N).map((_, i) => (solid[i] ? 0 : 1));
  const out = new Float32Array(N * N);
  const map = (fn: (x: number, y: number) => number) => {
    const em = new Float32Array(N * N), fr = new Float32Array(N * N), src = new Int32Array(N * N);
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) { const i = fn(x, y); src[x * N + y] = i; em[x * N + y] = emission[i]; fr[x * N + y] = free[i]; }
    return { em, fr, src };
  };
  const views = [(x: number, y: number) => x * N + y, (x: number, y: number) => (N - 1 - x) * N + y,
    (x: number, y: number) => y * N + x, (x: number, y: number) => y * N + (N - 1 - x)];
  views.forEach((fn, v) => {
    const { em, fr, src } = map(fn);
    const F = frustum(N, em, fr, spacing, windows, v === 0 ? work : undefined);
    for (let i = 0; i < N * N; i++) out[src[i]] += F[i];
  });
  return out;
}

/** exact fluence at pixel centres by casting `dirs` rays through the pixel grid; rows [y0, y1) of x-columns */
export function castRows(N: number, emission: Float32Array, solid: Uint8Array, dirs: number, out: Float32Array, x0: number, x1: number) {
  for (let x = x0; x < x1; x++)
    for (let y = 0; y < N; y++) {
      if (solid[x * N + y]) { out[x * N + y] = 0; continue; }
      let sum = 0;
      for (let k = 0; k < dirs; k++) {
        const th = ((k + 0.5) / dirs) * 2 * Math.PI, dx = Math.cos(th), dy = Math.sin(th);
        let cx = x, cy = y;
        const sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1;
        const tdx = Math.abs(1 / dx), tdy = Math.abs(1 / dy);
        let tx = (dx > 0 ? 0.5 : 0.5) * tdx, ty = 0.5 * tdy;
        for (;;) {
          if (tx < ty) { cx += sx; tx += tdx; } else { cy += sy; ty += tdy; }
          if (cx < 0 || cy < 0 || cx >= N || cy >= N) break;
          if (solid[cx * N + cy]) { sum += emission[cx * N + cy]; break; }
        }
      }
      out[x * N + y] = (sum * 2 * Math.PI) / dirs;
    }
}
