// Where one pair's light goes: a solid 1-px light at (lx, ly), a receiver pixel at (rx, ry), today's merge (Reciprocal
// HRC, every rise, one frustum toward +x, free space). The receiver's value is linear in the light's emission, so every
// ray's part in it is its value from this light times how much the receiver reads it (the adjoint). Summed over one
// cascade's rays, the parts give the receiver's whole value: once on the light's side (the up-pass records), once on
// the receiver's side (the merge's far values), and once over the rays that hand the light across (the parent rays'
// cone weights).
import { splits } from '../wide-rays/wide';

export interface Flows {
  N: number; pad: number; rows: number;
  /** rises per length, and per length the parts: [block][row][rise] with row = y + pad */
  D: Record<number, number[]>;
  up: Record<number, Float32Array>; down: Record<number, Float32Array>; hand: Record<number, Float32Array>;
  value: number;
}

const coneWeight = (L: number, d: number) => Math.atan(Math.min(1, (d + 0.5) / L)) - Math.atan(Math.max(-1, (d - 0.5) / L));
const rises = (L: number) => Array.from({ length: 2 * L + 1 }, (_, i) => i - L);

export function pairFlows(N: number, lx: number, ly: number, rx: number, ry: number): Flows {
  const pad = N, rows = N + 2 * pad;
  const light = (x: number, y: number) => x === lx && y - pad === ly;
  const e = (x: number, y: number) => (light(x, y) ? 1 : 0);
  const f = (x: number, y: number) => (y < pad || y >= pad + N || light(x, y) ? 0 : 1);
  const D: Record<number, number[]> = { 1: [-1, 0, 1] };
  const at = (L: number, p: number, y: number, k: number) => (p * rows + y) * D[L].length + k;
  const T: Record<number, Float32Array> = {}, R: Record<number, Float32Array> = {};
  T[1] = new Float32Array(N * rows * 3); R[1] = new Float32Array(N * rows * 3);
  for (let x = 0; x < N; x++)
    for (let y = 0; y < rows; y++)
      for (let k = 0; k < 3; k++) {
        const d = k - 1, i = at(1, x, y, k);
        T[1][i] = f(x, y) * f(x, y + d);
        R[1][i] = e(x, y + d) + f(x, y + d) * e(x, y);
      }
  const index = (L: number) => new Map(D[L].map((d, k) => [d, k]));
  // each pair of halves of a ray of length L: [rise k, first half index, row shift, second half index, weight]
  const pairs: Record<number, [number, number, number, number, number][]> = {};
  for (let L = 2; L <= N; L *= 2) {
    D[L] = rises(L);
    const H = index(L / 2);
    pairs[L] = [];
    D[L].forEach((d, k) => {
      for (const [a, j, b, w] of splits(L, d, 1, 1, 'one row')) {
        const ia = H.get(a), ib = H.get(b);
        if (ia !== undefined && ib !== undefined) pairs[L].push([k, ia, j, ib, w]);
      }
    });
  }
  // up: records of every length below N
  for (let L = 2; L < N; L *= 2) {
    const h = L / 2, t = new Float32Array((N / L) * rows * D[L].length), r = new Float32Array(t.length);
    for (const [k, ia, j, ib, w] of pairs[L])
      for (let p = 0; p < N / L; p++)
        for (let y = 0; y < rows; y++) {
          const yb = y + j;
          if (yb < 0 || yb >= rows) continue;
          const A = at(h, 2 * p, y, ia), B = at(h, 2 * p + 1, yb, ib), o = at(L, p, y, k);
          t[o] += w * T[h][A] * T[h][B];
          r[o] += w * (R[h][B] + T[h][B] * R[h][A]);
        }
    T[L] = t; R[L] = r;
  }
  // down: far values, longest first
  const G: Record<number, Float32Array> = { [N]: new Float32Array(rows * D[N].length) };
  for (let L = N; L >= 2; L /= 2) {
    const h = L / 2, g = new Float32Array((N / h) * rows * D[h].length);
    for (const [k, ia, j, ib, w] of pairs[L]) {
      const c = coneWeight(L, D[L][k]);
      for (let p = 0; p < N / L; p++)
        for (let y = 0; y < rows; y++) {
          const far = G[L][at(L, p, y, k)], A = at(h, 2 * p, y, ia);
          g[at(h, 2 * p, y, ia)] += w * far;
          const yb = y + j;
          if (yb < 0 || yb >= rows) continue;
          g[at(h, 2 * p + 1, yb, ib)] += w * (T[h][A] * far + c * R[h][A]);
        }
    }
    G[h] = g;
  }
  // the receiver's read of the length-1 rays, then the adjoint back through the merge and the records
  const lamG: Record<number, Float32Array> = { 1: new Float32Array(N * rows * 3) };
  const Y = ry + pad;
  let value = 0;
  for (let k = 0; k < 3; k++) {
    const d = k - 1, ys = Y - d;
    lamG[1][at(1, rx, Y, k)] += 0.5 * f(rx, Y);
    if (ys >= 0 && ys < rows) lamG[1][at(1, rx, ys, k)] += 0.5 * f(rx, ys) * f(rx, Y);
  }
  for (let i = 0; i < lamG[1].length; i++) value += lamG[1][i] * G[1][i];
  const lamR: Record<number, Float32Array> = {}, hand: Record<number, Float32Array> = {};
  for (let L = 1; L < N; L *= 2) lamR[L] = new Float32Array(R[L].length);
  for (let L = 2; L <= N; L *= 2) {
    const h = L / 2, lg = new Float32Array(G[L].length), hd = new Float32Array(G[L].length);
    for (const [k, ia, j, ib, w] of pairs[L]) {
      const c = coneWeight(L, D[L][k]);
      for (let p = 0; p < N / L; p++)
        for (let y = 0; y < rows; y++) {
          const o = at(L, p, y, k), A = at(h, 2 * p, y, ia);
          lg[o] += w * lamG[h][A];
          const yb = y + j;
          if (yb < 0 || yb >= rows) continue;
          const back = lamG[h][at(h, 2 * p + 1, yb, ib)];
          lg[o] += w * T[h][A] * back;
          lamR[h][A] += w * c * back;
          hd[o] += w * c * R[h][A] * back;
        }
    }
    lamG[L] = lg; hand[L] = hd;
  }
  for (let L = N / 2; L >= 2; L /= 2) {
    const h = L / 2;
    for (const [k, ia, j, ib, w] of pairs[L])
      for (let p = 0; p < N / L; p++)
        for (let y = 0; y < rows; y++) {
          const yb = y + j;
          if (yb < 0 || yb >= rows) continue;
          const back = lamR[L][at(L, p, y, k)], A = at(h, 2 * p, y, ia), B = at(h, 2 * p + 1, yb, ib);
          lamR[h][B] += w * back;
          lamR[h][A] += w * T[h][B] * back;
        }
  }
  const up: Record<number, Float32Array> = {}, down: Record<number, Float32Array> = {};
  for (let L = 1; L < N; L *= 2) {
    up[L] = R[L].map((v, i) => v * lamR[L][i]);
    down[L] = G[L].map((v, i) => v * lamG[L][i]);
  }
  return { N, pad, rows, D, up, down, hand, value };
}

/** the angle light pixel (lx, ly) covers seen from (qx, qy), within |dy| <= dx */
export function exactAngle(qx: number, qy: number, lx: number, ly: number) {
  const q = Math.PI / 4;
  const a = Math.atan2(qy - ly, qx - lx), b = Math.atan2(qy - ly, qx - lx - 1);
  const c = Math.atan2(qy - ly - 1, qx - lx), d = Math.atan2(qy - ly - 1, qx - lx - 1);
  return Math.max(0, Math.min(q, Math.max(a, b, c, d)) - Math.max(-q, Math.min(a, b, c, d)));
}
