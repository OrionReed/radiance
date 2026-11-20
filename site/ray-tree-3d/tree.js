// The ray tree of one frustum of 2-D HRC or Reciprocal HRC, as index arithmetic plus free-space linear passes.
// Local frame: u along the frustum axis, v across; pixel (i, r) is [i, i+1] × [r, r+1]. Level n holds nodes of one length
// L; a ray (n, j, d, y) spans node j, [x, x + L], from row y to row y + d: segment (x, y + ½) → (x + L, y + d + ½).
// A node splits into a first piece A and a second piece B (dyadic: two halves; Fibonacci: the long piece first). A ray is
// one or two chains A (rise q) then B (rise d − q), weighted. Reciprocal light travels +u; HRC probes look +u (light
// travels −u). HRC merged radiance lives on cones M(n, p, c, y) and needs the dyadic tree.

export const SCHEMES = {
  hrc: 'HRC',
  rec1: 'Reciprocal 1×',
  rec2: 'Reciprocal 2×',
  fib: 'Reciprocal Fibonacci',
};

const LN2 = Math.log(2);

/** the smallest Fibonacci length (1, 2, 3, 5, …) at least N */
export function fibRoot(N) {
  let a = 1, b = 2;
  while (b < N) [a, b] = [b, a + b];
  return b;
}

export class Tree {
  /** mirror: the Fibonacci tree with its pieces swapped, for the frusta whose light travels −x or −y, so that every
   * frustum splits at the same world columns */
  constructor(N, scheme, { evenWeight = 0.5, mirror = false } = {}) {
    this.N = N;
    this.scheme = scheme;
    this.hrc = scheme === 'hrc';
    this.fib = scheme === 'fib';
    this.mirror = mirror;
    this.evenWeight = evenWeight;
    const lengths = [1], pieces = {};
    if (this.fib) {
      const W = fibRoot(N);
      let a = 1, b = 2;
      pieces[2] = [1, 1];
      lengths.push(2);
      while (b < W) { [a, b] = [b, a + b]; pieces[b] = [a, b - a]; lengths.push(b); }
    } else for (let L = 2; L <= N; L *= 2) { pieces[L] = [L / 2, L / 2]; lengths.push(L); }
    if (mirror) for (const L in pieces) pieces[L] = [pieces[L][1], pieces[L][0]];
    this.W = lengths[lengths.length - 1]; // frame width: the root's length
    this.top = lengths.length - 1;
    const rank = new Map(lengths.map((L, n) => [L, n]));
    const starts = lengths.map(() => []);
    starts[this.top].push(0);
    for (let n = this.top; n >= 1; n--) {
      const [l1, l2] = pieces[lengths[n]];
      for (const x of starts[n]) { starts[rank.get(l1)].push(x); starts[rank.get(l2)].push(x + l1); }
    }
    this.lv = [];
    let off = 0, moff = 0;
    for (let n = 0; n <= this.top; n++) {
      const L = lengths[n], s = this.stepOf(L), R = (2 * L) / s + 1, xs = Int32Array.from(starts[n].sort((p, q) => p - q)), B = xs.length;
      this.lv.push({ n, L, s, R, B, xs, off, count: B * R * N, moff, mcount: this.hrc ? B * L * N : 0, at: new Map([...xs].map((x, j) => [x, j])) });
      off += B * R * N;
      moff += this.hrc ? B * L * N : 0;
    }
    // children of every node, and every node's parent (its level, node and side: 0 first piece, 1 second)
    for (const l of this.lv) { l.parLevel = new Int32Array(l.B).fill(-1); l.par = new Int32Array(l.B).fill(-1); l.side = new Uint8Array(l.B); }
    this.joinH = new Float32Array(this.W + 1).fill(this.top);
    for (let n = 1; n <= this.top; n++) {
      const l = this.lv[n], [l1, l2] = pieces[l.L];
      l.l1 = l1; l.nA = rank.get(l1); l.nB = rank.get(l2);
      const A = this.lv[l.nA], B = this.lv[l.nB];
      l.cA = new Int32Array(l.B); l.cB = new Int32Array(l.B);
      for (let j = 0; j < l.B; j++) {
        const x = l.xs[j], ja = A.at.get(x), jb = B.at.get(x + l1);
        l.cA[j] = ja; l.cB[j] = jb;
        A.parLevel[ja] = n; A.par[ja] = j; A.side[ja] = 0;
        B.parLevel[jb] = n; B.par[jb] = j; B.side[jb] = 1;
        this.joinH[x + l1] = n - 1;
      }
    }
    this.total = off;
    this.mtotal = this.hrc ? moff : off;
    this.#buildPairs();
  }

  stepOf(L) {
    if (this.scheme === 'rec2' || this.fib) return 1;
    if (this.scheme === 'rec1') return L === 1 ? 1 : 2;
    return 2;
  }
  riseAt(n, ri) { const l = this.lv[n]; return ri * l.s - l.L; }
  riseIndex(n, d) { const l = this.lv[n]; return (d + l.L) / l.s; }
  hasRise(n, d) { const l = this.lv[n]; return Math.abs(d) <= l.L && (((d + l.L) % l.s) + l.s) % l.s === 0; }
  id(n, j, ri, y) { const l = this.lv[n]; return l.off + (j * l.R + ri) * this.N + y; }
  levelOf(id) { let n = 0; while (n < this.top && id >= this.lv[n + 1].off) n++; return n; }
  decode(id) {
    const n = this.levelOf(id), l = this.lv[n], k = id - l.off, y = k % this.N, jr = (k - y) / this.N;
    const ri = jr % l.R, j = (jr - ri) / l.R;
    return { n, L: l.L, j, ri, d: ri * l.s - l.L, y, x: l.xs[j] };
  }
  /** the node of level n holding column u (u in [x, x + L)), or −1 where that level has no node */
  nodeAt(n, u) {
    const { xs, L } = this.lv[n];
    let lo = 0, hi = xs.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (xs[mid] <= u) lo = mid; else hi = mid - 1; }
    return xs[lo] <= u && u < xs[lo] + L ? lo : -1;
  }
  /** the height of column u in the joins picture: the level of the first piece of the node that splits there */
  joinAt(u) { const k = Math.round(u); return k <= 0 || k >= this.W ? this.top : this.joinH[k]; }
  /** a ray holds light: Reciprocal needs both end rows on the grid (off the rows is solid); HRC only its start */
  alive(n, ri, y) {
    if (y < 0 || y >= this.N) return false;
    if (this.hrc) return true;
    const e = y + this.riseAt(n, ri);
    return e >= 0 && e < this.N;
  }
  inGrid(n, ri, y) { const e = y + this.riseAt(n, ri); return y >= 0 && y < this.N && e >= 0 && e < this.N; }

  /** Reciprocal cone weight (the GPU page's coneWeight); HRC rays use their two sub-cones */
  cone(n, d) {
    const L = this.lv[n].L, s = this.stepOf(L);
    if (L === 1 && !this.hrc) return d === 0 ? Math.PI / 2 - LN2 : LN2 / 2;
    return Math.atan(Math.min(1, (d + s / 2) / L)) - Math.atan(Math.max(-1, (d - s / 2) / L));
  }
  /** HRC sub-cone i of level n: slopes [(i − L)/L, (i + 1 − L)/L] */
  arc(n, i) { const L = this.lv[n].L; return Math.atan((i + 1 - L) / L) - Math.atan((i - L) / L); }
  /** slope interval a node holds: ray (Reciprocal merged node too), or HRC cone */
  raySlopes(n, d) { const L = this.lv[n].L, s = this.stepOf(L); return [(d - s / 2) / L, (d + s / 2) / L]; }

  #buildPairs() {
    // pairs[n][ri] = [[aRi, bRi, q, w], …]: ray (n, j, d, y) = Σ w · (A (nA, cA[j], q, y) then B (nB, cB[j], d − q, y + q)),
    // the lower chain first. parentRef[n][side][child ri] = [[Dri, q, w]]: the same entries, seen from a piece.
    this.pairs = [null];
    this.parentRef = [null];
    for (let n = 1; n <= this.top; n++) {
      const l = this.lv[n], A = l.nA, B = l.nB, list = [];
      const ref = [Array.from({ length: this.lv[A].R }, () => []), Array.from({ length: this.lv[B].R }, () => [])];
      for (let ri = 0; ri < l.R; ri++) {
        const d = this.riseAt(n, ri);
        let ps;
        if (this.fib) {
          // the linear rule: the centre line crosses the junction at θd (θ = l1 / L), weights 1 − f and f on the rows
          // below and above it
          const num = l.l1 * d, fl = Math.floor(num / l.L), rem = num - fl * l.L;
          ps = rem === 0 ? [[fl, d - fl, 1]] : [[fl, d - fl - 0, 1 - rem / l.L], [fl + 1, d - fl - 1, rem / l.L]];
        } else {
          const sh = this.lv[A].s;
          if (d % 2 === 0 && this.hasRise(A, d / 2)) ps = [[d / 2, d / 2, 1]];
          else {
            const lo = (d - sh) / 2, up = (d + sh) / 2;
            ps = [[lo, up, 0.5], [up, lo, 0.5]];
          }
        }
        ps = ps.filter(([a, b]) => this.hasRise(A, a) && this.hasRise(B, b));
        const entry = ps.map(([a, b, w]) => [this.riseIndex(A, a), this.riseIndex(B, b), a, w]);
        list.push(entry);
        for (const [aRi, bRi, a, w] of entry) { ref[0][aRi].push([ri, a, w]); ref[1][bRi].push([ri, a, w]); }
      }
      this.pairs.push(list);
      this.parentRef.push(ref);
    }
  }

  // ── structure queries ──
  extChildren(id) {
    const { n, j, ri, y } = this.decode(id);
    if (n === 0) return [];
    const l = this.lv[n], out = [];
    for (const [aRi, bRi, a, w] of this.pairs[n][ri]) {
      if (y >= 0 && y < this.N) out.push({ id: this.id(l.nA, l.cA[j], aRi, y), w, side: 0 });
      if (y + a >= 0 && y + a < this.N) out.push({ id: this.id(l.nB, l.cB[j], bRi, y + a), w, side: 1 });
    }
    return out;
  }
  extParents(id) {
    const { n, j, ri, y } = this.decode(id), l = this.lv[n], pn = l.parLevel[j];
    if (pn < 0) return [];
    const out = [], J = l.par[j], right = l.side[j];
    for (const [Dri, a, w] of this.parentRef[pn][right][ri]) {
      const Y = right ? y - a : y;
      if (Y >= 0 && Y < this.N) out.push({ id: this.id(pn, J, Dri, Y), w, a, right });
    }
    return out;
  }
  /** HRC merged node index and decode */
  mid(n, p, c, y) { const l = this.lv[n]; return l.moff + (p * l.L + c) * this.N + y; }
  mdecode(mid) {
    if (!this.hrc) return this.decode(mid);
    let n = 0; while (n < this.top && mid >= this.lv[n + 1].moff) n++;
    const l = this.lv[n], k = mid - l.moff, y = k % this.N, pc = (k - y) / this.N, c = pc % l.L, p = (pc - c) / l.L;
    return { n, L: l.L, p, c, y };
  }

  // ── line tiles ──
  /** the ray of level n, node j whose tile holds the local line v = b + m u, or −1 */
  cellOf(n, j, b, m) {
    const l = this.lv[n], u0 = l.xs[j], v0 = b + m * u0, y = Math.floor(v0);
    if (y < 0 || y >= this.N) return -1;
    const e = b + m * (u0 + l.L) - y - 0.5, s = l.s;
    const d = s * Math.floor((e + l.L + s / 2) / s) - l.L;
    if (Math.abs(d) > l.L) return -1;
    return this.id(n, j, this.riseIndex(n, d), y);
  }
  /** corners of a ray's tile as lines (v at u0, v at u1) */
  cellCorners(id) {
    const { n, j, d, y } = this.decode(id), l = this.lv[n];
    const lo = y + d + 0.5 - l.s / 2, hi = y + d + 0.5 + l.s / 2;
    return { u0: l.xs[j], u1: l.xs[j] + l.L, c: [[y, lo], [y + 1, lo], [y + 1, hi], [y, hi]] };
  }

  // ── free-space linear passes (transmittance 1): arrays over ray ids (e, ρR, u) and merged ids (m, ρM) ──
  /** a source pixel's light in the leaves: ½ per half pixel (Reciprocal) or per row crossed (HRC level 0) */
  seedPixel(e, col, row, w = 1) {
    if (col < 0 || col >= this.W) return;
    const l = this.lv[0];
    for (let ri = 0; ri < l.R; ri++) {
      const d = this.riseAt(0, ri);
      for (const y of [row, row - d]) if (this.alive(0, ri, y)) e[this.id(0, col, ri, y)] += 0.5 * w;
    }
  }
  /** extensions: e(parent) += Σ w (e(A) + e(B)) */
  extUp(e) {
    const N = this.N;
    for (let n = 1; n <= this.top; n++) {
      const l = this.lv[n], A = this.lv[l.nA], B = this.lv[l.nB];
      for (let j = 0; j < l.B; j++) {
        const ja = l.cA[j], jb = l.cB[j];
        for (let ri = 0; ri < l.R; ri++) {
          const ps = this.pairs[n][ri], base = l.off + (j * l.R + ri) * N;
          for (let y = 0; y < N; y++) {
            if (!this.alive(n, ri, y)) continue;
            let v = 0;
            for (const [aRi, bRi, a, w] of ps) {
              const yb = y + a;
              v += w * (e[A.off + (ja * A.R + aRi) * N + y] + (yb >= 0 && yb < N ? e[B.off + (jb * B.R + bRi) * N + yb] : 0));
            }
            e[base + y] += v;
          }
        }
      }
    }
    return e;
  }
  /** the transpose: ρ on parents handed to their pieces */
  extDown(u, keep = null) {
    const N = this.N;
    for (let n = this.top; n >= 1; n--) {
      const l = this.lv[n], A = this.lv[l.nA], B = this.lv[l.nB];
      for (let j = 0; j < l.B; j++) {
        const ja = l.cA[j], jb = l.cB[j];
        for (let ri = 0; ri < l.R; ri++) {
          const ps = this.pairs[n][ri], base = l.off + (j * l.R + ri) * N;
          for (let y = 0; y < N; y++) {
            const up = u[base + y];
            if (!up) continue;
            for (const [aRi, bRi, a, w] of ps) {
              if (this.alive(l.nA, aRi, y) && (!keep || keep(l.nA, this.riseAt(l.nA, aRi)))) u[A.off + (ja * A.R + aRi) * N + y] += w * up;
              const yb = y + a;
              if (this.alive(l.nB, bRi, yb) && (!keep || keep(l.nB, this.riseAt(l.nB, bRi)))) u[B.off + (jb * B.R + bRi) * N + yb] += w * up;
            }
          }
        }
      }
    }
    return u;
  }
  /** the merge, top down: merged far radiance m from the rays' radiance e; returns per-pixel fluence too. split(id),
   * if given, is the lower chain's share of a two-chain parent's far light (Reciprocal; default: the chain weights) */
  mergeDown(e, split = null) {
    const N = this.N, m = new Float32Array(this.mtotal), F = new Float32Array(this.W * N); // F[local column * N + row]
    if (this.hrc) {
      const ew = this.evenWeight;
      for (let n = this.top; n >= 0; n--) {
        const l = this.lv[n], up = n < this.top ? this.lv[n + 1] : null;
        const M1 = (q, i, y) => (up && q < up.B && y >= 0 && y < N ? m[up.moff + (q * up.L + i) * N + y] : 0);
        for (let p = 0; p < l.B; p++)
          for (let c = 0; c < l.L; c++)
            for (let y = 0; y < N; y++) {
              let v = 0;
              for (let i = 2 * c; i <= 2 * c + 1; i++) {
                const k = (i + 1) >> 1, d = 2 * k - l.L, a = this.arc(n, i);
                const ray = e[l.off + (p * l.R + k) * N + y];
                if (p % 2) v += a * ray + M1((p + 1) / 2, i, y + d);
                else {
                  const yx = y + d, ext = p + 1 < l.B && yx >= 0 && yx < N ? e[l.off + ((p + 1) * l.R + k) * N + yx] : 0;
                  v += (1 - ew) * (a * (ray + ext) + M1((p + 2) / 2, i, y + 2 * d)) + ew * M1(p / 2, i, y);
                }
              }
              m[l.moff + (p * l.L + c) * N + y] = v;
            }
      }
      for (let x = 0; x + 1 < N; x++) for (let y = 0; y < N; y++) F[x * N + y] = m[this.mid(0, x + 1, 0, y)];
      return { m, F };
    }
    for (let n = this.top; n >= 1; n--) {
      const l = this.lv[n], A = this.lv[l.nA], B = this.lv[l.nB];
      for (let j = 0; j < l.B; j++) {
        const ja = l.cA[j], jb = l.cB[j];
        for (let ri = 0; ri < l.R; ri++) {
          const ps = this.pairs[n][ri], base = l.off + (j * l.R + ri) * N, cp = this.cone(n, this.riseAt(n, ri));
          for (let y = 0; y < N; y++) {
            const mp = m[base + y], s = split && ps.length === 2 && mp ? split(base + y) : null;
            ps.forEach(([aRi, bRi, a, w], q) => {
              const wf = s === null ? w : q === 0 ? s : 1 - s;
              const left = A.off + (ja * A.R + aRi) * N + y;
              m[left] += wf * mp;
              const yb = y + a;
              if (yb >= 0 && yb < N) {
                const before = this.alive(l.nA, aRi, y) ? e[left] : 0;
                m[B.off + (jb * B.R + bRi) * N + yb] += wf * mp + w * cp * before;
              }
            });
          }
        }
      }
    }
    for (let x = 0; x < this.W; x++) for (let y = 0; y < N; y++) {
      let v = 0;
      for (const [d, yy, w] of this.receiverTaps(y)) v += w * m[this.id(0, x, this.riseIndex(0, d), yy)];
      F[x * N + y] = v;
    }
    return { m, F };
  }
  /** Reciprocal: the merged 1-px nodes a pixel reads, with weights (pixels shader, free space) */
  receiverTaps(y) {
    const t = [[-1, y, 0.5], [0, y, 1], [1, y, 0.5], [1, y - 1, 0.5], [-1, y + 1, 0.5]];
    return t.filter(([, yy]) => yy >= 0 && yy < this.N);
  }
  /** receiver x's read weights, back up the merge. keep(n, lo, hi): follow only cones that pass (e.g. hold a slope) */
  readBack(receivers, keep = null) {
    const N = this.N, rM = new Float32Array(this.mtotal), rR = new Float32Array(this.total);
    if (this.hrc) {
      const ew = this.evenWeight;
      for (const [x, y, w] of receivers) if (x + 1 < N) rM[this.mid(0, x + 1, 0, y)] += w;
      for (let n = 0; n <= this.top; n++) {
        const l = this.lv[n], up = n < this.top ? this.lv[n + 1] : null;
        const add = (q, i, y, v) => { if (up && q < up.B && y >= 0 && y < N) rM[up.moff + (q * up.L + i) * N + y] += v; };
        for (let p = 0; p < l.B; p++)
          for (let c = 0; c < l.L; c++)
            for (let y = 0; y < N; y++) {
              const r = rM[l.moff + (p * l.L + c) * N + y];
              if (!r) continue;
              for (let i = 2 * c; i <= 2 * c + 1; i++) {
                if (keep && !keep(n, (i - l.L) / l.L, (i + 1 - l.L) / l.L)) continue;
                const k = (i + 1) >> 1, d = 2 * k - l.L, a = this.arc(n, i);
                if (p % 2) {
                  rR[l.off + (p * l.R + k) * N + y] += r * a;
                  add((p + 1) / 2, i, y + d, r);
                } else {
                  rR[l.off + (p * l.R + k) * N + y] += r * (1 - ew) * a;
                  const yx = y + d;
                  if (p + 1 < l.B && yx >= 0 && yx < N) rR[l.off + ((p + 1) * l.R + k) * N + yx] += r * (1 - ew) * a;
                  add((p + 2) / 2, i, y + 2 * d, r * (1 - ew));
                  add(p / 2, i, y, r * ew);
                }
              }
            }
      }
      return { rM, rR };
    }
    for (const [x, y, w] of receivers)
      for (const [d, yy, tw] of this.receiverTaps(y)) {
        if (keep && !keep(0, ...this.raySlopes(0, d))) continue;
        rM[this.id(0, x, this.riseIndex(0, d), yy)] += w * tw;
      }
    // a node's parent is higher than both its pieces, so increasing levels finish every parent's inputs first
    for (let n = 0; n < this.top; n++) {
      const l = this.lv[n];
      for (let j = 0; j < l.B; j++) {
        const pn = l.parLevel[j];
        if (pn < 0) continue;
        const P = this.lv[pn], J = l.par[j], right = l.side[j], A = this.lv[P.nA], jaP = P.cA[J];
        for (let ri = 0; ri < l.R; ri++) {
          const base = l.off + (j * l.R + ri) * N, list = this.parentRef[pn][right][ri];
          for (let y = 0; y < N; y++) {
            const r = rM[base + y];
            if (!r) continue;
            for (const [Dri, a, w] of list) {
              const D = this.riseAt(pn, Dri);
              if (keep && !keep(pn, ...this.raySlopes(pn, D))) continue;
              const Y = right ? y - a : y;
              if (Y < 0 || Y >= N) continue;
              rM[P.off + (J * P.R + Dri) * N + Y] += w * r;
              if (right) {
                const aRi = this.riseIndex(P.nA, a);
                if (this.alive(P.nA, aRi, Y)) rR[A.off + (jaP * A.R + aRi) * N + Y] += w * this.cone(pn, D) * r;
              }
            }
          }
        }
      }
    }
    return { rM, rR };
  }
  /** HRC merged node → its local segment (probe, cone centre); Reciprocal merged node = its ray */
  mSegment(mid) {
    if (!this.hrc) return this.segment(mid);
    const { n, p, c, y } = this.mdecode(mid), L = this.lv[n].L;
    return [p * L, y + 0.5, (p + 1) * L, y + 0.5 + 2 * c + 1 - L];
  }
  segment(id) {
    const { L, d, y, x } = this.decode(id);
    return [x, y + 0.5, x + L, y + d + 0.5];
  }
}

// ── frames: world (x, y) ↔ local (u, v), W the frame width. Frustum f: 0 light +x, 1 −x, 2 +y, 3 −y. HRC is mirrored
// (probes look +u). ──
export function toLocal(W, scheme, f, x, y) {
  let u, v;
  if (f === 0) [u, v] = [x, y];
  else if (f === 1) [u, v] = [W - x, y];
  else if (f === 2) [u, v] = [y, x];
  else [u, v] = [W - y, x];
  if (scheme === 'hrc') u = W - u;
  return [u, v];
}
export function toWorld(W, scheme, f, u, v) {
  if (scheme === 'hrc') u = W - u;
  if (f === 0) return [u, v];
  if (f === 1) return [W - u, v];
  if (f === 2) return [v, u];
  return [v, W - u];
}
/** a world line through A, B in a frustum's local frame: v = b + m u, or null if steeper than 45° there */
export function localLine(W, scheme, f, A, B) {
  const [ua, va] = toLocal(W, scheme, f, A[0], A[1]), [ub, vb] = toLocal(W, scheme, f, B[0], B[1]);
  if (Math.abs(ub - ua) < 1e-12) return null;
  const m = (vb - va) / (ub - ua);
  if (Math.abs(m) > 1 + 1e-12) return null;
  return { b: va - m * ua, m };
}
/** pixels the local line passes through, one per column of the scene [0, N): [col, row]; xs: the scene's local columns */
export function linePixels(N, ln, cols = null) {
  const out = [];
  for (const x of cols ?? Array.from({ length: N }, (_, i) => i)) {
    const y = Math.floor(ln.b + ln.m * (x + 0.5));
    if (y >= 0 && y < N) out.push([x, y]);
  }
  return out;
}

/** the dyadic join level of column u (u an odd multiple of 2^n sits at n; the frame's ends at the top) */
export function joinLevel(N, u) {
  let k = Math.round(u);
  if (k <= 0 || k >= N) return Math.log2(N);
  let h = 0;
  while (!(k & 1)) { k >>= 1; h++; }
  return h;
}
/** for a local line: at each level, how many neighbouring line tiles (nodes that touch) share their end point, and how
 * many parents share both end points with the line's tiles in their two pieces */
export function lineJoins(t, b, m) {
  const out = [];
  for (let n = 0; n <= t.top; n++) {
    let side = 0, sideOk = 0, up = 0, upOk = 0;
    const l = t.lv[n], tiles = [];
    for (let j = 0; j < l.B; j++) {
      const id = t.cellOf(n, j, b, m);
      tiles.push(id >= 0 && t.inGrid(n, t.decode(id).ri, t.decode(id).y) ? t.decode(id) : null);
    }
    for (let j = 0; j + 1 < l.B; j++) {
      const a = tiles[j], c = tiles[j + 1];
      if (!a || !c || l.xs[j] + l.L !== l.xs[j + 1]) continue;
      side++;
      if (a.y + a.d === c.y) sideOk++;
    }
    if (n > 0) {
      for (let J = 0; J < l.B; J++) {
        const P = tiles[J];
        if (!P) continue;
        const A = t.cellOf(l.nA, l.cA[J], b, m), C = t.cellOf(l.nB, l.cB[J], b, m);
        if (A < 0 || C < 0) continue;
        const a = t.decode(A), c = t.decode(C);
        up++;
        if (a.y === P.y && c.y + c.d === P.y + P.d) upOk++;
      }
    }
    out.push({ side, sideOk, up, upOk });
  }
  return out;
}
/** the lower chain's share of a two-chain parent's far light, for a 1-px light at local pixel (sx, sy): the fraction of
 * the parent tile's lines through the light that cross the junction in the lower chain's row; ½ when none cross it */
export function forkShare(t, id, sx, sy, G = 16) {
  const { n, d, y, x } = t.decode(id), l = t.lv[n], L = l.L, th = l.l1 / L, fr = th * d - Math.floor(th * d);
  let hit = 0, lo = 0;
  for (let a = 0; a < G; a++)
    for (let b = 0; b < G; b++) {
      const h0 = (a + 0.5) / G, h1 = (b + 0.5) / G, s = (d + h1 - h0) / L, v0 = y + h0;
      const va = v0 + s * (sx - x), vb = v0 + s * (sx + 1 - x);
      if (Math.max(va, vb) > sy && Math.min(va, vb) < sy + 1) { hit++; if (fr + (1 - th) * h0 + th * h1 < 1) lo++; }
    }
  return hit ? lo / hit : 0.5;
}
