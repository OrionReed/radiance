// Ray tree 3D: every ray of a frustum of HRC or Reciprocal HRC in 3-D, with picking, line following and pixel traces.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { Tree, SCHEMES, toLocal, toWorld, localLine, linePixels, lineJoins, forkShare } from './tree.js';
import { PRESETS } from './presets.js';

const LEVEL_COLORS = ['#f0f0f0', '#ff8a5c', '#ffd166', '#9be564', '#4ed8c9', '#5aa9ff', '#a98bff', '#ff7ac6', '#c8c8a0', '#9adfff'];
const HL = {
  sel: '#ffffff', from: '#4ed8c9', into: '#ffd166', light: '#a98bff', on: '#9be564', off: '#ff5c5c', root: '#ffffff',
  reads: '#ff7ac6', readsOn: '#ffd0ea', pair: '#ffd166', pick: '#ffffff', route: '#a98bff', foot: '#ff7ac6',
};
const rgb = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };

// ── state (URL query: only what differs from the defaults) ──
const DEFAULTS = {
  a: 'rec2', b: 'none', N: 32, fr: '0',
  rep: 'joins', z: 'level', chart: 'centre', cam: 'persp', zs: 1, zexpr: 'n * 6 + t * 4',
  hide: '', y0: 0, rows: 256, stride: 1, block: 'all', col: 16, rises: 'all', clip: true,
  color: 'level', alpha: 0.16, width: 2.5, pt: 2, additive: true,
  grid: true, heat: true, fill: true,
  desc: true, anc: false, light: false, tri: true,
  line: false, ax: 0.3, ay: 6.37, bx: 31.7, by: 19.13, shift: 0, lTiles: true, lReads: false, lAll: false, lTri: true, viewIdx: -1,
  rx: 20, ry: 16, sx: 5, sy: 12, foot: false, pair: false, click: 'ray', pTri: false, triColor: 'error',
};
const S = { ...DEFAULTS };
{
  const p = new URLSearchParams(location.search);
  for (const [k, v] of p) if (k in S) S[k] = typeof DEFAULTS[k] === 'number' ? Number(v) : typeof DEFAULTS[k] === 'boolean' ? v === '1' : v;
}
function writeURL() {
  const p = new URLSearchParams();
  for (const k of Object.keys(DEFAULTS)) if (S[k] !== DEFAULTS[k]) p.set(k, typeof S[k] === 'boolean' ? (S[k] ? '1' : '0') : String(S[k]));
  const qs = p.toString();
  history.replaceState(null, '', qs ? '?' + qs : location.pathname);
}
const hidden = () => new Set(S.hide ? S.hide.split(',').map(Number) : []);
const shown = (n) => !hidden().has(n);
const worldRep = () => S.rep === 'joins' || S.rep === 'stacked';
const schemesOf = () => (S.b === 'none' || S.b === S.a ? [S.a] : [S.a, S.b]);
const frustaList = () => ({ 0: [0], 1: [1], 2: [2], 3: [3], x: [0, 1], all: [0, 1, 2, 3] })[S.fr];
function clampToN() {
  const N = S.N, c = (k, hi) => (S[k] = Math.max(0, Math.min(hi, S[k])));
  c('ax', N); c('bx', N); c('ay', N - 0.1); c('by', N - 0.1);
  for (const k of ['rx', 'ry', 'sx', 'sy', 'col', 'y0']) c(k, N - 1);
}

// ── renderer, cameras ──
const view = document.getElementById('view'), area = document.getElementById('canvas-area'), busy = document.getElementById('busy');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
view.appendChild(renderer.domElement);
const persp = new THREE.PerspectiveCamera(40, 1, 0.1, 1e5);
const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e5, 1e5);
persp.up.set(0, 0, 1);
ortho.up.set(0, 0, 1);
let camera = S.cam === 'ortho' ? ortho : persp;
const controls = new OrbitControls(camera, renderer.domElement);
controls.addEventListener('change', () => (needsRender = true));
let needsRender = true, aspect = 1;
function setOrthoExtent(h) {
  ortho.top = h; ortho.bottom = -h; ortho.left = -h * aspect; ortho.right = h * aspect;
  ortho.updateProjectionMatrix();
}
function useCamera(kind) {
  const next = kind === 'ortho' ? ortho : persp;
  if (next === camera) return;
  next.position.copy(camera.position);
  if (next === ortho) { ortho.zoom = 1; setOrthoExtent(camera.position.distanceTo(controls.target) * Math.tan((persp.fov * Math.PI) / 360)); }
  camera = next;
  controls.object = camera;
  controls.update();
  needsRender = true;
}

const trees = new Map();
function tree(N, scheme, mirror = false) {
  const key = `${N}:${scheme}:${mirror}`;
  if (!trees.has(key)) trees.set(key, new Tree(N, scheme, { mirror }));
  return trees.get(key);
}
/** a panel's tree for frustum f: the Fibonacci tree is mirrored where light travels −x or −y (same world columns) */
const treeOf = (P, f) => tree(S.N, P.scheme, P.scheme === 'fib' && f % 2 === 1);
/** the scene's columns in a frustum's local frame (the Fibonacci frame is wider than the scene) */
const sceneCols = (t, f) => (t.W === S.N ? null : Array.from({ length: S.N }, (_, i) => (f % 2 ? t.W - S.N + i : i)));

let panels = [];
function makePanels() {
  for (const p of panels) p.label.remove();
  panels = schemesOf().map((scheme) => {
    const scene = new THREE.Scene(), pickScene = new THREE.Scene();
    const label = Object.assign(document.createElement('div'), { className: 'plabel', textContent: SCHEMES[scheme] });
    area.appendChild(label);
    const hl = new THREE.Group(), overlay = new THREE.Group(), floor = new THREE.Group();
    scene.add(hl, overlay, floor);
    return { scheme, scene, pickScene, label, hl, overlay, floor, base: null, pick: null, vis: null, weights: {}, stats: [] };
  });
}

// ── embeddings ──
let LEVELS = Math.log2(S.N) + 1; // the most levels among the panels' trees
const G = () => (S.zs * S.N) / LEVELS; // height of one level
let zFn = () => 0;
function compileZ() {
  try { zFn = new Function('r', 't', `const {n, L, j, d, y, s, f, w, x, wy, u, v, slope} = r; return (${S.zexpr});`); }
  catch { zFn = () => 0; }
}
compileZ();
const R = { n: 0, L: 1, j: 0, d: 0, y: 0, s: 1, f: 0, w: 0, x: 0, wy: 0, u: 0, v: 0, slope: 0 };
/** height of one vertex of a level-n ray, at local (u, v), tt ∈ [0, 1] along it */
function zAt(t, n, L, j, d, y, f, wx, wy, u, v, w, tt) {
  const g = G(), k = (g * (t.top + 1)) / S.N;
  if (S.rep === 'joins') return t.joinAt(u) * g;
  switch (S.z) {
    case 'length': return L * k;
    case 'depth': return (u - j * L) * k;
    case 'leveldepth': return n * g + ((u - j * L) / L) * g * 0.85;
    case 'dist': return Math.hypot(wx - S.rx - 0.5, wy - S.ry - 0.5) * k;
    case 'weight': return w > 0 ? Math.max(0, Math.log10(w) + 5) * g : -g * 0.5;
    case 'custom':
      Object.assign(R, { n, L, j, d, y, s: t.lv[n].s, f, w, x: wx, wy, u, v, slope: d / L });
      return zFn(R, tt);
  }
  return n * g;
}
const curved = () => S.rep === 'stacked' && (S.z === 'dist' || S.z === 'custom');
/** a world line through (x0, y0), (x1, y1) as a point of the line chart (x frustums left, y frustums offset right) */
function chart(f, x0, y0, x1, y1) {
  const N = S.N, xc = S.chart === 'column' ? S.col + 0.5 : N / 2;
  let m, c, off = 0;
  if (f < 2) { m = (y1 - y0) / (x1 - x0); c = y0 + m * (xc - x0); }
  else { m = (x1 - x0) / (y1 - y0); c = x0 + m * (xc - y0); off = 1.3 * N; }
  if (S.chart === 'hough') { const th = Math.atan(m); return [th * ((2 * N) / Math.PI) + off, (c - N / 2) * Math.cos(th)]; }
  return [(m * N) / 2 + off, c - N / 2];
}
function segVerts(t, f, seg, n, j, d, y, w, out, pieces, zOffset = 0) {
  const N = S.N, W = t.W, [u0, v0, u1, v1] = seg, L = t.lv[n].L;
  if (worldRep()) {
    for (let k = 0; k < pieces; k++)
      for (const tt of [k / pieces, (k + 1) / pieces]) {
        const u = u0 + (u1 - u0) * tt, v = v0 + (v1 - v0) * tt, [wx, wy] = toWorld(W, t.scheme, f, u, v);
        out.push(wx - N / 2, wy - N / 2, zAt(t, n, L, j, d, y, f, wx, wy, u, v, w, tt) + zOffset);
      }
    return;
  }
  const [ax, ay] = toWorld(W, t.scheme, f, u0, v0), [bx, by] = toWorld(W, t.scheme, f, u1, v1);
  const [h, vv] = chart(f, ax, ay, bx, by);
  if (S.rep === 'lines') { out.push(h, vv, n * G() + zOffset); return; }
  const X0 = f < 2 ? ax : ay, X1 = f < 2 ? bx : by;
  out.push(h, vv, X0 * S.zs + zOffset, h, vv, X1 * S.zs + zOffset);
}
/** a line tile's outline in the chart (corners as (row at u0, row at u1)); fillOut gets two triangles */
function tileVerts(t, f, tile, n, out, fillOut) {
  const N = S.N, { u0, u1, c } = tile, pcs = S.chart === 'hough' ? 5 : 1, z = S.rep === 'linecol' ? (u0 + u1) * 0.5 * S.zs : n * G();
  const corner = (va, vb) => {
    const [ax, ay] = toWorld(t.W, t.scheme, f, u0, va), [bx, by] = toWorld(t.W, t.scheme, f, u1, vb);
    return [...chart(f, ax, ay, bx, by), z];
  };
  for (let e = 0; e < 4; e++) {
    const A = c[e], B = c[(e + 1) % 4];
    for (let k = 0; k < pcs; k++)
      for (const tt of [k / pcs, (k + 1) / pcs]) out.push(...corner(A[0] + (B[0] - A[0]) * tt, A[1] + (B[1] - A[1]) * tt));
  }
  if (fillOut) { const Q = c.map(([a, b]) => corner(a, b)); fillOut.push(...Q[0], ...Q[1], ...Q[2], ...Q[0], ...Q[2], ...Q[3]); }
}
function mTile(t, mid) {
  if (!t.hrc) return t.cellCorners(mid);
  const { n, p, c, y } = t.mdecode(mid), L = t.lv[n].L, lo = 2 * c - L, hi = 2 * c + 2 - L;
  return { u0: p * L, u1: (p + 1) * L, c: [[y, y + 0.5 + lo], [y + 1, y + 0.5 + lo], [y + 1, y + 0.5 + hi], [y, y + 0.5 + hi]] };
}

// ── colours ──
const levelRGB = LEVEL_COLORS.map(rgb);
const lc = (n) => levelRGB[n % levelRGB.length];
function baseColor(n, j, ri, d, w, wmax, L) {
  switch (S.color) {
    case 'rise': { const s = d / L; return [0.5 + 0.5 * s, 0.45 - 0.2 * Math.abs(s), 0.5 - 0.5 * s]; }
    case 'block': return j & 1 ? [1, 0.55, 0.2] : [0.3, 0.7, 1];
    case 'kpar': return ri & 1 ? [1, 0.45, 0.45] : [0.45, 0.9, 0.5];
    case 'weight': { const a = w > 0 ? Math.pow(w / wmax, 0.35) : 0; return [0.08 + 0.92 * a, 0.08 + 0.62 * a, 0.12 + 0.2 * a]; }
  }
  return lc(n);
}

// ── the rays shown ──
function forEachVisible(t, f, cb) {
  const N = S.N, hide = hidden();
  for (let n = 0; n <= t.top; n++) {
    if (hide.has(n)) continue;
    const l = t.lv[n];
    let j0 = 0, j1 = l.B - 1;
    if (S.block === 'column') {
      const [u] = toLocal(t.W, t.scheme, f, f < 2 ? S.col + 0.5 : 0, f < 2 ? 0 : S.col + 0.5);
      j0 = j1 = t.nodeAt(n, Math.floor(u));
      if (j0 < 0) continue;
    }
    for (let j = j0; j <= j1; j++)
      for (let ri = 0; ri < l.R; ri++) {
        if (S.rises === 'even' && ri & 1) continue;
        if (S.rises === 'odd' && !(ri & 1)) continue;
        const d = ri * l.s - l.L;
        for (let y = Math.max(0, S.y0); y <= Math.min(N - 1, S.y0 + S.rows - 1); y += Math.max(1, S.stride)) {
          if (S.clip ? !t.inGrid(n, ri, y) : !t.alive(n, ri, y)) continue;
          cb(l.off + (j * l.R + ri) * N + y, n, j, ri, d, y);
        }
      }
  }
}

const pickMat = new THREE.ShaderMaterial({
  glslVersion: THREE.GLSL3,
  uniforms: { size: { value: 4 } },
  vertexShader: `in float pid; flat out float vId; uniform float size;
    void main() { vId = pid; gl_PointSize = size; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `flat in float vId; out vec4 pickOut;
    void main() { float i = vId; pickOut = vec4(mod(i, 256.0), mod(floor(i / 256.0), 256.0), floor(i / 65536.0), 255.0) / 255.0; }`,
});

const baseOpacity = (P) => (P.hl.children.length ? S.alpha * 0.4 : S.alpha);
function buildBase(P) {
  const frs = frustaList(), pts = S.rep === 'lines', pieces = curved() ? 6 : 1;
  let count = 0;
  for (const f of frs) forEachVisible(treeOf(P, f), f, () => count++);
  const cap = 3e6, keepEvery = count > cap ? Math.ceil(count / cap) : 1;
  const vertsPer = pts ? 1 : 2 * pieces, nKeep = Math.ceil(count / keepEvery);
  const pos = new Float32Array(nKeep * vertsPer * 3), col = new Float32Array(nKeep * vertsPer * 3), pid = new Float32Array(nKeep * vertsPer);
  const visF = new Uint8Array(nKeep), visId = new Uint32Array(nKeep);
  let i = 0, seen = 0;
  const tmp = [];
  for (const f of frs) {
    const t = treeOf(P, f), W = P.weights[f];
    let wmax = 0;
    if (W) for (let q = 0; q < W.length; q++) if (W[q] > wmax) wmax = W[q];
    forEachVisible(t, f, (id, n, j, ri, d, y) => {
      if (seen++ % keepEvery) return;
      tmp.length = 0;
      const w = W ? W[id] : 0;
      segVerts(t, f, t.segment(id), n, j, d, y, w, tmp, pts ? 1 : pieces);
      const c = baseColor(n, j, ri, d, w, wmax || 1, t.lv[n].L);
      for (let k = 0; k < vertsPer; k++) {
        const o = (i * vertsPer + k) * 3;
        pos[o] = tmp[3 * k]; pos[o + 1] = tmp[3 * k + 1]; pos[o + 2] = tmp[3 * k + 2];
        col[o] = c[0]; col[o + 1] = c[1]; col[o + 2] = c[2];
        pid[i * vertsPer + k] = i + 1;
      }
      visF[i] = f; visId[i] = id; i++;
    });
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('pid', new THREE.BufferAttribute(pid, 1));
  const common = { vertexColors: true, transparent: true, opacity: baseOpacity(P), depthWrite: false, blending: S.additive ? THREE.AdditiveBlending : THREE.NormalBlending };
  if (P.base) { P.scene.remove(P.base); P.pickScene.remove(P.pick); P.base.geometry.dispose(); P.base.material.dispose(); }
  P.base = pts ? new THREE.Points(geo, new THREE.PointsMaterial({ ...common, size: S.pt, sizeAttenuation: false })) : new THREE.LineSegments(geo, new THREE.LineBasicMaterial(common));
  P.pick = pts ? new THREE.Points(geo, pickMat) : new THREE.LineSegments(geo, pickMat);
  P.base.frustumCulled = P.pick.frustumCulled = false;
  P.scene.add(P.base);
  P.pickScene.add(P.pick);
  P.vis = { visF, visId, count, shown: i, keepEvery };
}

// ── highlights: fat lines, filled tiles and triangles ──
const fatMats = [];
function fat(positions, colors, width) {
  const g = new LineSegmentsGeometry();
  g.setPositions(positions);
  g.setColors(colors);
  const m = new LineMaterial({ linewidth: width, vertexColors: true, transparent: true, depthTest: false, worldUnits: false });
  fatMats.push(m);
  const o = new LineSegments2(g, m);
  o.renderOrder = 10;
  o.frustumCulled = false;
  return o;
}
function mesh(positions, colors, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const o = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
  o.renderOrder = 5;
  o.frustumCulled = false;
  return o;
}
/** items {f, id, w, m (merged node), rgb}: fat segments in world space, tiles in line space */
function addItems(P, items, hex, { width = S.width, cap = 60000, zOffset = 0, minA = 0.3 } = {}) {
  if (!items.length) return;
  const t = treeOf(P, items[0].f), base = rgb(hex);
  if (items.length > cap) items = items.sort((a, b) => b.w - a.w).slice(0, cap);
  let wmax = 0;
  for (const it of items) wmax = Math.max(wmax, it.w);
  const pos = [], col = [], tmp = [], fill = [], fcol = [], pieces = curved() ? 6 : 1;
  for (const it of items) {
    tmp.length = 0;
    const dec = it.m && t.hrc ? t.mdecode(it.id) : t.decode(it.id);
    const a = wmax > 0 && it.w > 0 ? minA + (1 - minA) * Math.pow(it.w / wmax, 0.4) : 1, c = it.rgb || base;
    if (worldRep()) {
      const seg = it.m ? t.mSegment(it.id) : t.segment(it.id), d = it.m && t.hrc ? 2 * dec.c + 1 - dec.L : dec.d;
      segVerts(t, it.f, seg, dec.n, dec.j ?? dec.p, d, dec.y, it.w, tmp, pieces, zOffset);
    } else {
      const before = fill.length;
      tileVerts(t, it.f, it.m ? mTile(t, it.id) : t.cellCorners(it.id), dec.n, tmp, S.fill ? fill : null);
      for (let k = before; k < fill.length; k += 3) fcol.push(c[0] * a, c[1] * a, c[2] * a);
    }
    for (let k = 0; k < tmp.length / 3; k++) { pos.push(tmp[3 * k], tmp[3 * k + 1], tmp[3 * k + 2]); col.push(c[0] * a, c[1] * a, c[2] * a); }
  }
  P.hl.add(fat(new Float32Array(pos), new Float32Array(col), width));
  if (fill.length) P.hl.add(mesh(fill, fcol, 0.28));
}
/** extension triangles: a ray (top edge) and one pair of its halves (lower edges), meeting where the halves join */
function addTriangles(P, f, list, opacity = 0.22) {
  if (!worldRep() || !list.length) return;
  const t = treeOf(P, f), pos = [], col = [], tmp = [];
  for (const item of list.slice(0, 40000)) {
    const { id, aRi, w } = item;
    const p = t.decode(id);
    if (p.n === 0) continue;
    tmp.length = 0;
    segVerts(t, f, t.segment(id), p.n, p.j, p.d, p.y, 0, tmp, 1);
    const lp = t.lv[p.n], ch = t.id(lp.nA, lp.cA[p.j], aRi, p.y), dc = t.decode(ch);
    segVerts(t, f, t.segment(ch), dc.n, dc.j, dc.d, dc.y, 0, tmp, 1);
    const c = list[0].rgb ? item.rgb : lc(p.n), k = list[0].rgb ? 1 : 0.35 + 0.65 * w;
    pos.push(tmp[0], tmp[1], tmp[2], tmp[9], tmp[10], tmp[11], tmp[3], tmp[4], tmp[5]);
    for (let q = 0; q < 3; q++) col.push(c[0] * k, c[1] * k, c[2] * k);
  }
  if (pos.length) P.hl.add(mesh(pos, col, opacity));
}
const pairsOf = (t, id) => {
  const { n, ri } = t.decode(id);
  return n === 0 ? [] : t.pairs[n][ri].map(([aRi, bRi, a, w]) => ({ id, aRi, bRi, a, w }));
};
function itemsFrom(arr, f, total, { m = false, skip = -1, min = 1e-9 } = {}) {
  const out = [];
  for (let i = 0; i < total; i++) if (arr[i] > min && i !== skip) out.push({ f, id: i, w: arr[i], m });
  return out;
}

// ── selection ──
let sel = null; // { f, a, b }: world end points of the picked ray, start first
function idInTree(t, f, a, b) {
  let [u0, v0] = toLocal(t.W, t.scheme, f, a[0], a[1]), [u1, v1] = toLocal(t.W, t.scheme, f, b[0], b[1]);
  if (u1 < u0) [u0, v0, u1, v1] = [u1, v1, u0, v0];
  const L = Math.round(u1 - u0), n = t.lv.findIndex((l) => l.L === L);
  const j = n < 0 ? undefined : t.lv[n].at.get(Math.round(u0));
  if (j === undefined) return -1;
  const y = Math.round(v0 - 0.5), d = Math.round(v1 - v0);
  if (!t.hasRise(n, d) || y < 0 || y >= S.N || !t.alive(n, t.riseIndex(n, d), y)) return -1;
  return t.id(n, j, t.riseIndex(n, d), y);
}
const worldLine = () => ({ A: [S.ax, S.ay + S.shift], B: [S.bx, S.by + S.shift] });
function localPixel(t, f, x, y) {
  const [u, v] = toLocal(t.W, t.scheme, f, x + 0.5, y + 0.5);
  return [Math.floor(u), Math.floor(v)];
}

// ── traces ──
function buildTraces(P) {
  const N = S.N, lines = [], ray = [];
  P.weights = {};
  const heat = new Float32Array(N * N);
  let heatOn = false;
  const addHeat = (f, F, t) => {
    heatOn = true;
    for (let i = 0; i < t.W; i++) for (let r = 0; r < N; r++) {
      const v = F[i * N + r];
      if (!v) continue;
      const [wx, wy] = toWorld(t.W, t.scheme, f, i + 0.5, r + 0.5), X = Math.floor(wx), Y = Math.floor(wy);
      if (X >= 0 && X < N && Y >= 0 && Y < N) heat[X * N + Y] += v;
    }
  };
  for (const f of frustaList()) {
    const t = treeOf(P, f);
    if (sel && f === sel.f) {
      const id = idInTree(t, f, sel.a, sel.b);
      if (id >= 0) {
        addItems(P, [{ f, id, w: 1 }], HL.sel, { width: S.width + 2 });
        const dec = t.decode(id), ch = t.extChildren(id), pa = t.extParents(id);
        ray.push(`level ${dec.n} · length ${dec.L} · ${t.fib ? `node at column ${dec.x}` : `block ${dec.j}`}`, `rise ${dec.d} · start row ${dec.y}`);
        if (t.hrc) ray.push(`probe ${dec.j} (${dec.j & 1 ? 'odd' : 'even'}) · ray ${dec.ri} (${dec.ri & 1 ? 'odd: two bent pairs' : 'even: one straight pair'})`);
        else {
          const ps = dec.n ? t.pairs[dec.n][dec.ri] : [], lp = t.lv[dec.n];
          const how = !dec.n ? 'level 0' : t.fib
            ? `pieces ${lp.l1} + ${lp.L - lp.l1}: ${ps.map(([, , q, w]) => `${w.toFixed(3)} via row +${q}`).join(', ')}`
            : ps.length === 2 ? 'two bent pairs of halves' : 'one straight pair of halves';
          ray.push(`cone weight ${t.cone(dec.n, dec.d).toFixed(4)} · ${how}`);
        }
        ray.push(`built from ${ch.length} halves · part of ${pa.length} rays one level up`);
        if (S.desc || S.tri) {
          const u = new Float32Array(t.total); u[id] = 1; t.extDown(u);
          const items = itemsFrom(u, f, t.total).filter((it) => shown(t.levelOf(it.id)));
          if (S.desc) addItems(P, items.filter((it) => it.id !== id), HL.from);
          if (S.tri) addTriangles(P, f, items.flatMap((it) => pairsOf(t, it.id).map((p) => ({ ...p, w: p.w * Math.min(1, it.w) }))));
        }
        if (S.anc) { const e = new Float32Array(t.total); e[id] = 1; t.extUp(e); addItems(P, itemsFrom(e, f, t.total, { skip: id }).filter((it) => shown(t.levelOf(it.id))), HL.into); }
        if (S.light) {
          const e = new Float32Array(t.total); e[id] = 1; t.extUp(e);
          const { m, F } = t.mergeDown(e);
          addItems(P, itemsFrom(m, f, t.mtotal, { m: true }), HL.light, { width: Math.max(1, S.width - 1), zOffset: G() * 0.06 });
          addHeat(f, F, t);
          P.weights[f] = e;
          let nRec = 0; for (const v of F) if (v > 1e-9) nRec++;
          ray.push(`its light reaches ${nRec} pixels`);
        }
      } else ray.push(`not a ray of ${SCHEMES[t.scheme]}`);
    }
    const { A, B } = worldLine(), ln = S.line ? localLine(t.W, t.scheme, f, A, B) : null;
    if (S.line && !ln) lines.push(`ℓ is steeper than 45° in frustum ${f}`);
    if (ln) {
      lines.push(`<b>line ℓ</b> f${f} · slope ${ln.m.toFixed(3)}`);
      if (S.lTiles || S.lTri) {
        const items = [], tri = [], perLevel = [];
        for (let n = 0; n <= t.top; n++) {
          let tot = 0, disc = 0;
          const [uc] = toLocal(t.W, t.scheme, f, f < 2 ? S.col + 0.5 : 0, f < 2 ? 0 : S.col + 0.5), jc = t.nodeAt(n, Math.floor(uc)), l = t.lv[n];
          for (let j = 0; j < t.lv[n].B; j++) {
            const id = t.cellOf(n, j, ln.b, ln.m);
            if (id < 0) continue;
            const { ri, y } = t.decode(id);
            if (!t.inGrid(n, ri, y)) continue;
            const drawn = worldRep() ? S.block === 'all' || j === jc : j === jc;
            let c = HL.root;
            if (l.parLevel[j] >= 0) {
              const pid = t.cellOf(l.parLevel[j], l.par[j], ln.b, ln.m), ok = pid >= 0 && t.extChildren(pid).some((q) => q.id === id);
              c = ok ? HL.on : HL.off;
              tot++; if (!ok) disc++;
            }
            if (n > 0) {
              const A0 = t.cellOf(l.nA, l.cA[j], ln.b, ln.m), C0 = t.cellOf(l.nB, l.cB[j], ln.b, ln.m);
              for (const p of pairsOf(t, id)) if (t.id(l.nA, l.cA[j], p.aRi, y) === A0 && t.id(l.nB, l.cB[j], p.bRi, y + p.a) === C0) tri.push({ ...p, w: 1 });
            }
            if (shown(n) && drawn) items.push({ f, id, w: 1, rgb: rgb(c) });
          }
          perLevel.push(tot ? `${disc}/${tot}` : '–');
        }
        if (S.lTiles) addItems(P, items, HL.on);
        if (S.lTri) addTriangles(P, f, tri.filter((p) => shown(t.levelOf(p.id))), 0.3);
        const J = lineJoins(t, ln.b, ln.m);
        if (!worldRep()) lines.push(`  drawn: the block holding column ${S.col} only`);
        lines.push(`  not a half of ℓ's ray above: ${perLevel.join(' ')}`);
        lines.push(`  neighbours that do not meet: ${J.map((r) => (r.side ? `${r.side - r.sideOk}/${r.side}` : '–')).join(' ')}`);
      }
      const pix = linePixels(N, ln, sceneCols(t, f));
      if (S.lReads && pix.length) {
        const recv = S.viewIdx >= 0 ? [pix[Math.min(S.viewIdx, pix.length - 1)]] : pix;
        const eps = 1e-9, keep = (n, lo, hi) => ln.m >= lo - eps && ln.m <= hi + eps;
        const { rR } = t.readBack(recv.map(([x, y]) => [x, y, 1]), keep);
        const items = [], cnt = new Array(t.top + 1).fill(0);
        for (let i = 0; i < t.total; i++) {
          if (!(rR[i] > 1e-9)) continue;
          const { n, j } = t.decode(i), holds = t.cellOf(n, j, ln.b, ln.m) === i;
          cnt[n]++;
          if (shown(n)) items.push({ f, id: i, w: rR[i], rgb: rgb(holds ? HL.readsOn : HL.reads) });
        }
        addItems(P, items, HL.reads, { zOffset: G() * 0.03 });
        P.weights[f] = rR;
        lines.push(`  rays read per level by ${recv.length} pixel${recv.length > 1 ? 's' : ''} on ℓ: ${cnt.join(' ')}`);
      }
      if (S.lAll && pix.length) {
        const e = new Float32Array(t.total);
        for (const [x, y] of pix) t.seedPixel(e, x, y);
        t.extUp(e);
        const { rR } = t.readBack(pix.map(([x, y]) => [x, y, 1]));
        const u = t.extDown(rR.slice()), flow = new Float32Array(t.total);
        for (let i = 0; i < t.total; i++) flow[i] = e[i] * u[i];
        addItems(P, itemsFrom(flow, f, t.total).filter((it) => shown(t.levelOf(it.id))), HL.pair, { width: Math.max(1, S.width - 0.5) });
        P.weights[f] = flow;
      }
    }
    if (S.foot || S.pair) {
      const [rx, ry] = localPixel(t, f, S.rx, S.ry);
      const { rM, rR } = t.readBack([[rx, ry, 1]]);
      if (S.foot) {
        addItems(P, itemsFrom(rR, f, t.total).filter((it) => shown(t.levelOf(it.id))), HL.foot);
        P.weights[f] = rR;
        let c = 0; for (const v of rR) if (v > 1e-9) c++;
        lines.push(`<b>pixel (${S.rx}, ${S.ry})</b> f${f} reads ${c} rays`);
      }
      if (S.pair) {
        const [sx, sy] = localPixel(t, f, S.sx, S.sy);
        const e = new Float32Array(t.total); t.seedPixel(e, sx, sy);
        const e0 = e.slice();
        t.extUp(e);
        const u = t.extDown(rR.slice()), flow = new Float32Array(t.total), pick = new Float32Array(t.total);
        let T = 0;
        const byLevel = new Array(t.top + 1).fill(0);
        for (let i = 0; i < t.total; i++) { flow[i] = e[i] * u[i]; pick[i] = e[i] * rR[i]; T += e0[i] * u[i]; if (pick[i] > 0) byLevel[t.levelOf(i)] += pick[i]; }
        const { m, F } = t.mergeDown(e), mflow = new Float32Array(t.mtotal);
        for (let i = 0; i < t.mtotal; i++) mflow[i] = m[i] * rM[i];
        addItems(P, itemsFrom(flow, f, t.total).filter((it) => shown(t.levelOf(it.id))), HL.pair, { width: Math.max(1, S.width - 0.5) });
        addItems(P, itemsFrom(pick, f, t.total), HL.pick, { width: S.width + 1.5 });
        addItems(P, itemsFrom(mflow, f, t.mtotal, { m: true }), HL.route, { width: Math.max(1, S.width - 1), zOffset: G() * 0.06 });
        addHeat(f, F, t);
        P.weights[f] = flow;
        lines.push(`<b>source (${S.sx}, ${S.sy}) → pixel (${S.rx}, ${S.ry})</b> f${f}: ${T.toExponential(2)}`);
        if (S.pTri) lines.push(...routeTriangles(P, f, m, mflow, rM, flow, sx, sy, T));
        lines.push(`  share the merge picks up per level: ${byLevel.map((v) => (v ? (v / T).toFixed(2) : '0')).join(' ')}`);
      }
    }
  }
  P.stats = lines;
  P.ray = ray;
  P.heat = heatOn && S.heat ? heat : null;
}

/** the pair route's extension triangles, coloured by a per-extension quantity. Reciprocal: merged radiance m sits on
 * the rays, so the route's parents are rays with merged flow; a fork (odd parent) gives its far light ½ / ½ where the
 * light's geometry gives s / 1 − s. error: (½ − s) m (ρ_lo − ρ_hi), the first-order change in the pixel's value if the
 * fork were split by geometry (ρ: the pixel's read weight of each fork's two halves). HRC: flow only. */
function routeTriangles(P, f, m, mflow, rM, flow, sx, sy, T) {
  const t = treeOf(P, f), list = [], vals = [], out = [];
  const mode = t.hrc && (S.triColor === 'share' || S.triColor === 'error') ? 'flow' : S.triColor;
  let sumErr = 0, forks = 0;
  const route = t.hrc ? flow : mflow;
  for (let id = 0; id < t.total; id++) {
    if (!(route[id] > 1e-12)) continue;
    const { n, j, y } = t.decode(id);
    if (n === 0 || !shown(n)) continue;
    const ps = pairsOf(t, id);
    let s = 0.5, dErr = 0;
    if (ps.length === 2 && !t.hrc) {
      s = forkShare(t, id, sx, sy);
      const l = t.lv[n], rho = (p) => rM[t.id(l.nA, l.cA[j], p.aRi, y)] + (y + p.a >= 0 && y + p.a < S.N ? rM[t.id(l.nB, l.cB[j], p.bRi, y + p.a)] : 0);
      dErr = (0.5 - s) * m[id] * (rho(ps[0]) - rho(ps[1]));
      sumErr += dErr; forks++;
    }
    ps.forEach((p, q) => {
      if (mode === 'share' && q === 1) return; // the two forks nearly coincide; the lower one's value says both
      const v = mode === 'level' ? 0 : mode === 'merged' ? m[id] : mode === 'flow' ? route[id] : mode === 'share' ? (q === 0 ? 0.5 - s : s - 0.5) * m[id] : dErr;
      list.push({ ...p, w: 1 }); vals.push(v);
    });
  }
  if (mode === 'level') addTriangles(P, f, list, 0.35);
  else {
    let mx = 0;
    for (const v of vals) mx = Math.max(mx, Math.abs(v));
    const pos = rgb('#ff8a5c'), neg = rgb('#5aa9ff');
    list.forEach((it, i) => {
      const a = mx > 0 ? Math.pow(Math.abs(vals[i]) / mx, 0.5) : 0, c = vals[i] >= 0 ? pos : neg;
      it.rgb = [0.12 + c[0] * a, 0.12 + c[1] * a, 0.12 + c[2] * a];
    });
    addTriangles(P, f, list, 0.55);
  }
  if (!t.hrc) out.push(`  forks on the route ${forks}; first-order error the forks give the pixel: ${(sumErr / T).toExponential(2)} of the transfer`);
  else if (S.triColor !== mode) out.push('  triangles: flow (HRC has no far-light split at its forks)');
  return out;
}

// ── floor: the pixel grid, markers and the lit pixels ──
const floorZ = () => (S.rep === 'joins' ? -G() * 0.6 : -G() * 0.5);
function buildFloor(P) {
  if (!worldRep()) return;
  const N = S.N, z = floorZ();
  if (S.grid) {
    const g = [], step = N <= 64 ? 1 : N / 64;
    for (let i = 0; i <= N; i += step) g.push(i - N / 2, -N / 2, z, i - N / 2, N / 2, z, -N / 2, i - N / 2, z, N / 2, i - N / 2, z);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(g, 3));
    P.floor.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x3a3a3a, transparent: true, opacity: 0.7, depthWrite: false })));
  }
  const mk = [];
  if (S.line) { const { A, B } = worldLine(); mk.push([A[0], A[1], HL.on], [B[0], B[1], HL.on]); }
  if (S.foot || S.pair) mk.push([S.rx + 0.5, S.ry + 0.5, HL.foot]);
  if (S.pair) mk.push([S.sx + 0.5, S.sy + 0.5, HL.pair]);
  const mp = [], mc = [];
  for (const [x, y, hex] of mk) {
    const c = rgb(hex);
    for (const [dx, dy] of [[-0.6, 0], [0.6, 0], [0, -0.6], [0, 0.6]]) { mp.push(x - N / 2, y - N / 2, z, x - N / 2 + dx, y - N / 2 + dy, z); mc.push(...c, ...c); }
  }
  if (S.line) { const { A, B } = worldLine(), c = rgb(HL.on); mp.push(A[0] - N / 2, A[1] - N / 2, z, B[0] - N / 2, B[1] - N / 2, z); mc.push(...c, ...c); }
  if (mp.length) P.floor.add(fat(new Float32Array(mp), new Float32Array(mc), 2));
  if (P.heat) {
    const data = new Uint8Array(N * N * 4);
    let mx = 0;
    for (const v of P.heat) mx = Math.max(mx, v);
    for (let X = 0; X < N; X++) for (let Y = 0; Y < N; Y++) {
      const a = mx > 0 ? Math.pow(P.heat[X * N + Y] / mx, 0.35) : 0, o = 4 * (Y * N + X);
      data[o] = 255 * Math.min(1, a * 1.4); data[o + 1] = 255 * a * a; data[o + 2] = 60 * a; data[o + 3] = a > 0 ? 200 : 0;
    }
    const tex = new THREE.DataTexture(data, N, N);
    tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(N, N), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    plane.position.z = z - 0.01;
    P.floor.add(plane);
  }
}

// ── overlay: user geometry ──
let overlayCode = '';
function runOverlay(P) {
  if (!overlayCode.trim()) return;
  const t = treeOf(P, frustaList()[0]);
  const api = {
    N: S.N, THREE, tree: t, scheme: t.scheme, frusta: frustaList(), state: S, G: G(), toLocal, toWorld, localLine, joinAt: (u) => t.joinAt(u),
    line: worldLine(),
    segments(list, color = [1, 1, 1], width = 1.5) {
      const p = [], c = [];
      for (const s of list) { p.push(s[0] - S.N / 2, s[1] - S.N / 2, s[2], s[3] - S.N / 2, s[4] - S.N / 2, s[5]); c.push(...color, ...color); }
      if (p.length) P.overlay.add(fat(new Float32Array(p), new Float32Array(c), width));
    },
    rawSegments(flat, color = [1, 1, 1], width = 1.5) {
      const c = [];
      for (let i = 0; i < flat.length / 3; i++) c.push(...color);
      if (flat.length) P.overlay.add(fat(new Float32Array(flat), new Float32Array(c), width));
    },
    triangles(flat, color = [1, 1, 1], opacity = 0.3) {
      const c = [];
      for (let i = 0; i < flat.length / 3; i++) c.push(...color);
      if (flat.length) P.overlay.add(mesh(flat, c, opacity));
    },
    points(list, color = [1, 1, 1], size = 4) {
      const p = [];
      for (const q of list) p.push(q[0] - S.N / 2, q[1] - S.N / 2, q[2]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
      P.overlay.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: new THREE.Color(...color), size, sizeAttenuation: false })));
    },
    surface(fn, { x0 = 0, x1 = S.N, y0 = 0, y1 = S.N, res = 64, color = [0.4, 0.7, 1], opacity = 0.35, wire = true } = {}) {
      const geo = new THREE.PlaneGeometry(x1 - x0, y1 - y0, res, res), a = geo.attributes.position;
      for (let i = 0; i < a.count; i++) {
        const x = a.getX(i) + (x0 + x1) / 2, y = a.getY(i) + (y0 + y1) / 2;
        a.setXYZ(i, x - S.N / 2, y - S.N / 2, fn(x, y));
      }
      P.overlay.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: new THREE.Color(...color), transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false })));
      if (wire) P.overlay.add(new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: new THREE.Color(...color), transparent: true, opacity: Math.min(1, opacity * 1.6), depthWrite: false })));
    },
    embedRay(f, id) { const tmp = [], d = t.decode(id); segVerts(t, f, t.segment(id), d.n, d.j, d.d, d.y, 0, tmp, 1); return tmp; },
    embedTile(f, id) { const tmp = []; tileVerts(t, f, t.cellCorners(id), t.levelOf(id), tmp, null); return tmp; },
    levelColor: (n) => lc(n),
  };
  try { new Function('api', overlayCode)(api); }
  catch (e) { P.stats.push(`overlay error: ${e.message}`); }
}

// ── rebuild, render ──
let rebuildTimer = 0, pendingFrame = false;
function rebuild({ base = true, reframe = false } = {}) {
  clearTimeout(rebuildTimer);
  pendingFrame ||= reframe;
  busy.textContent = 'building…';
  rebuildTimer = setTimeout(() => {
    const t0 = performance.now();
    clampToN();
    if (schemesOf().join() !== panels.map((p) => p.scheme).join()) { makePanels(); layout(); }
    LEVELS = Math.max(...panels.map((P) => treeOf(P, 0).top + 1));
    fatMats.length = 0;
    const weighted = S.color === 'weight' || (S.rep === 'stacked' && S.z === 'weight');
    for (const P of panels) {
      P.tree = treeOf(P, 0);
      for (const g of [P.hl, P.floor, P.overlay]) { g.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); }); g.clear(); }
      buildTraces(P);
      if (base || weighted || !P.base) buildBase(P);
      else P.base.material.opacity = baseOpacity(P);
      buildFloor(P);
      runOverlay(P);
    }
    if (pendingFrame) { pendingFrame = false; frame(); }
    writeReadout();
    syncAll();
    busy.textContent = `${(performance.now() - t0).toFixed(0)} ms`;
    writeURL();
    needsRender = true;
  }, 10);
}
function layout() {
  const w = area.clientWidth, h = area.clientHeight;
  renderer.setSize(w, h);
  aspect = w / Math.max(1, panels.length) / Math.max(1, h);
  persp.aspect = aspect;
  persp.updateProjectionMatrix();
  setOrthoExtent(ortho.top || 30);
  panels.forEach((P, i) => (P.label.style.left = `${(i * w) / panels.length + 10}px`));
  if (w > 0 && h > 0 && !Number.isFinite(camera.position.x)) frame(); // loaded while the page had no size
  needsRender = true;
}
function frame(top = false) {
  const N = S.N, g = G(), levels = LEVELS - 1;
  const tg = S.rep === 'lines' ? [S.fr === 'all' || S.fr === '2' || S.fr === '3' ? 0.65 * N : 0, 0, (g * levels) / 2] : S.rep === 'linecol' ? [0, 0, (N * S.zs) / 2] : [0, 0, top ? 0 : (g * levels) / 2];
  controls.target.set(...tg);
  const dir = top ? new THREE.Vector3(0, -0.001, 1) : S.rep === 'lines' ? new THREE.Vector3(0.1, -0.5, 1) : new THREE.Vector3(-0.15, -1, 0.75);
  const asp = Number.isFinite(aspect) && aspect > 0.05 ? aspect : 1, dist = 1.9 * N * Math.max(1, 1.1 / asp);
  camera.position.copy(controls.target).addScaledVector(dir.normalize(), dist);
  if (camera === ortho) { ortho.zoom = 1; setOrthoExtent(dist * Math.tan((persp.fov * Math.PI) / 360)); }
  controls.update();
  needsRender = true;
}
function draw() {
  const w = area.clientWidth, h = area.clientHeight, pw = w / panels.length;
  for (const m of fatMats) m.resolution.set(pw, h);
  renderer.setScissorTest(true);
  renderer.setClearColor(0x0e0e0e, 1);
  panels.forEach((P, i) => {
    renderer.setViewport(i * pw, 0, pw, h);
    renderer.setScissor(i * pw, 0, pw, h);
    renderer.render(P.scene, camera);
  });
  renderer.setScissorTest(false);
}
function render() {
  requestAnimationFrame(render);
  if (!needsRender) return;
  needsRender = false;
  draw();
}

// ── picking ──
const pickRT = new THREE.WebGLRenderTarget(15, 15), pickBuf = new Uint8Array(15 * 15 * 4);
function pickAt(cx, cy) {
  const rect = renderer.domElement.getBoundingClientRect(), pw = rect.width / panels.length, h = rect.height;
  const x = cx - rect.left, y = cy - rect.top, pi = Math.min(panels.length - 1, Math.floor(x / pw)), P = panels[pi];
  if (!P?.vis) return null;
  pickMat.uniforms.size.value = S.pt + 4;
  camera.setViewOffset(pw, h, x - pi * pw - 7, y - 7, 15, 15);
  renderer.setRenderTarget(pickRT);
  renderer.setClearColor(0, 0);
  renderer.clear();
  renderer.render(P.pickScene, camera);
  renderer.readRenderTargetPixels(pickRT, 0, 0, 15, 15, pickBuf);
  renderer.setRenderTarget(null);
  camera.clearViewOffset();
  let best = 0, bd = 1e9;
  for (let iy = 0; iy < 15; iy++) for (let ix = 0; ix < 15; ix++) {
    const o = 4 * (iy * 15 + ix), v = pickBuf[o] + 256 * pickBuf[o + 1] + 65536 * pickBuf[o + 2], dd = (ix - 7) ** 2 + (iy - 7) ** 2;
    if (v && dd < bd) { bd = dd; best = v; }
  }
  needsRender = true;
  return best ? { P, f: P.vis.visF[best - 1], id: P.vis.visId[best - 1] } : { P, none: true };
}
function floorPoint(cx, cy) {
  const rect = renderer.domElement.getBoundingClientRect(), pw = rect.width / panels.length;
  const x = cx - rect.left, pi = Math.floor(x / pw), lx = x - pi * pw;
  const rc = new THREE.Raycaster();
  rc.setFromCamera(new THREE.Vector2((lx / pw) * 2 - 1, -((cy - rect.top) / rect.height) * 2 + 1), camera);
  const pt = new THREE.Vector3();
  return rc.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -floorZ()), pt) ? [pt.x + S.N / 2, pt.y + S.N / 2] : null;
}
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || e.button !== 0) return;
  if (S.click !== 'ray' && worldRep()) {
    const p = floorPoint(e.clientX, e.clientY);
    if (!p) return;
    const r = (v) => Math.round(v * 10) / 10, c = (v) => Math.max(0, Math.min(S.N - 1, Math.floor(v)));
    if (S.click === 'A') { S.ax = r(p[0]); S.ay = r(p[1]); S.line = true; }
    if (S.click === 'B') { S.bx = r(p[0]); S.by = r(p[1]); S.line = true; }
    if (S.click === 'pixel') { S.rx = c(p[0]); S.ry = c(p[1]); if (!S.pair) S.foot = true; }
    if (S.click === 'source') { S.sx = c(p[0]); S.sy = c(p[1]); S.pair = true; }
    rebuild({ base: S.z === 'dist' });
    return;
  }
  const hit = pickAt(e.clientX, e.clientY);
  if (!hit) return;
  if (hit.none) sel = null;
  else {
    const t = treeOf(hit.P, hit.f), [u0, v0, u1, v1] = t.segment(hit.id);
    sel = { f: hit.f, a: toWorld(t.W, t.scheme, hit.f, u0, v0), b: toWorld(t.W, t.scheme, hit.f, u1, v1) };
  }
  rebuild({ base: false });
});
addEventListener('keydown', (e) => { if (e.key === 'Escape') { sel = null; rebuild({ base: false }); } });

// ── controls, after the radiance pages' toggle / segmented / slider ──
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const syncs = [];
function syncAll() { for (const s of syncs) s(); }
function changed(kind) {
  if (kind === 'none') { syncAll(); writeURL(); return; }
  rebuild({ base: kind !== 'traces', reframe: kind === 'frame' });
}
function section(parent, title, small = '') {
  const s = el('section', {}, el('h2', {}, el('span', { textContent: title }), el('small', { textContent: small })));
  parent.append(s);
  return s;
}
function field(parent, label, child, title = '') {
  const f = el('div', { className: 'field', title }, ...(label ? [el('div', { className: 'label' }, el('span', { textContent: label }))] : []), child);
  parent.append(f);
  return f;
}
function seg(parent, label, key, options, { kind = 'base', cols, title, enabled } = {}) {
  const s = el('div', { className: 'seg' + (cols ? ' wrap' : '') });
  if (cols) s.style.setProperty('--cols', String(cols));
  const bs = options.map(([v, text, tip]) => {
    const b = el('button', { textContent: text, title: tip ?? text });
    b.onclick = () => { S[key] = v; changed(kind); };
    s.append(b);
    return [v, b];
  });
  const f = field(parent, label, s, title);
  syncs.push(() => { bs.forEach(([v, b]) => b.classList.toggle('on', S[key] === v)); if (enabled) f.classList.toggle('disabled', !enabled()); });
}
function toggle(parent, label, key, { kind = 'traces', color, title, enabled } = {}) {
  const b = el('button', { className: 'toggle', title: title ?? '' }, el('span', { className: 'sw' }), el('span', { className: 'name', textContent: label }));
  if (color) b.append(el('i', { style: `width:9px;height:9px;border-radius:2px;flex:none;background:${color}` }));
  b.onclick = () => { S[key] = !S[key]; changed(kind); };
  const f = el('div', { className: 'field' }, b);
  parent.append(f);
  syncs.push(() => { b.classList.toggle('on', !!S[key]); if (enabled) f.classList.toggle('disabled', !enabled()); });
}
function slider(parent, label, key, { min, max, step = 1, kind = 'traces', fmt, title, enabled, live = false } = {}) {
  const val = el('span', { className: 'val' }), input = el('input', { type: 'range', min, step });
  const show = () => (val.textContent = fmt ? fmt(S[key]) : String(S[key]));
  input.oninput = () => { S[key] = Number(input.value); show(); if (live) changed(kind); };
  input.onchange = () => { S[key] = Number(input.value); changed(kind); };
  const f = el('div', { className: 'field', title: title ?? '' }, el('div', { className: 'label' }, el('span', { textContent: label }), val), input);
  parent.append(f);
  syncs.push(() => { input.max = typeof max === 'function' ? max() : max; input.value = S[key]; show(); if (enabled) f.classList.toggle('disabled', !enabled()); });
}
function buttons(parent, list) {
  const s = el('div', { className: 'seg' });
  for (const [text, fn, tip] of list) { const b = el('button', { textContent: text, title: tip ?? text }); b.onclick = fn; s.append(b); }
  field(parent, '', s);
}

const left = document.getElementById('left'), right = document.getElementById('right');
const SCHEME_OPTS = [['hrc', 'HRC'], ['rec1', 'Recip 1×', 'Reciprocal HRC, base ray count 1×'], ['rec2', 'Recip 2×', 'Reciprocal HRC, base ray count 2×'], ['fib', 'Recip Fib', 'Reciprocal HRC on the Fibonacci column tree: lengths 1, 2, 3, 5, …, long piece first, two chains with linear weights']];
{
  const s = section(left, 'schemes');
  seg(s, 'left', 'a', SCHEME_OPTS, { kind: 'frame' });
  seg(s, 'right', 'b', [['none', 'none'], ...SCHEME_OPTS], { kind: 'frame' });
  seg(s, 'resolution (N × N pixels)', 'N', [8, 16, 32, 64, 128, 256].map((n) => [n, String(n)]), { kind: 'frame', cols: 6 });
  seg(s, 'frustum (the way light travels)', 'fr', [['0', '+x'], ['1', '−x'], ['2', '+y'], ['3', '−y'], ['x', '±x'], ['all', 'all']], { cols: 6 });
}
{
  const s = section(left, 'representation');
  seg(s, '', 'rep', [['joins', 'joins', 'each column at its join level: rays climb from where their halves meet'], ['stacked', 'stacked', 'each level flat at its own height'], ['lines', 'line space', 'each ray at its central line (slope, offset)'], ['linecol', 'line + column', 'each ray over its central line, height = column along the frustum axis']], { kind: 'frame', cols: 2 });
  seg(s, 'height (stacked)', 'z', [['level', 'level'], ['length', 'length'], ['depth', 'depth', 'distance along the ray from its start'], ['leveldepth', 'lvl + depth', 'level, rising along the ray'], ['dist', 'distance', 'distance to the pixel set under pixels'], ['weight', 'weight', 'log of the active trace weight'], ['custom', 'custom']], { cols: 4, enabled: () => S.rep === 'stacked' });
  const zi = el('input', { value: S.zexpr, title: 'fields n L j d y s f w x wy u v slope; t along the ray' });
  Object.assign(zi.style, { width: '100%', background: 'var(--ctl)', color: 'var(--text)', border: '1px solid var(--line)', borderRadius: '4px', font: '11px var(--mono)', padding: '3px 5px' });
  zi.onchange = () => { S.zexpr = zi.value; compileZ(); rebuild(); };
  const zf = field(s, 'custom height (n L j d y s f w x wy u v slope, t)', zi);
  syncs.push(() => zf.classList.toggle('disabled', !(S.rep === 'stacked' && S.z === 'custom')));
  seg(s, 'line chart (line space)', 'chart', [['centre', 'offset at centre'], ['column', 'at column', 'the offset measured at the filter column'], ['hough', 'angle, dist.', 'Hough: angle θ and distance ρ from the centre']], { enabled: () => !worldRep() });
  slider(s, 'height scale', 'zs', { min: 0.1, max: 4, step: 0.05, kind: 'frame', fmt: (v) => v.toFixed(2) });
  seg(s, 'camera', 'cam', [['persp', 'perspective'], ['ortho', 'orthographic']], { kind: 'none' });
}
{
  const s = section(left, 'levels', 'click to hide');
  const box = el('div', { id: 'levels' });
  s.append(box);
  syncs.push(() => {
    const hide = hidden(), ts = panels.map((P) => treeOf(P, 0)), top = LEVELS - 1;
    const lengthsAt = (n) => [...new Set(ts.map((t) => t.lv[n]?.L ?? '–'))].join('/');
    const key = `${top}:${ts.map((t) => t.scheme).join()}`;
    if (box.dataset.key !== key) {
      box.dataset.key = key;
      box.replaceChildren();
      for (let n = 0; n <= top; n++) {
        const b = el('button', { title: `level ${n}: length ${lengthsAt(n)}` }, el('i', { style: `background:${LEVEL_COLORS[n % LEVEL_COLORS.length]}` }), el('span', { textContent: String(n) }), el('span', { className: 'n', textContent: lengthsAt(n) }));
        b.onclick = () => { const h = hidden(); if (h.has(n)) h.delete(n); else h.add(n); S.hide = [...h].sort((a, c) => a - c).join(','); rebuild(); };
        box.append(b);
      }
    }
    [...box.children].forEach((b, n) => b.classList.toggle('off', hide.has(n)));
  });
}
{
  const s = section(left, 'filter');
  slider(s, 'first start row', 'y0', { min: 0, max: () => S.N - 1, kind: 'base' });
  slider(s, 'rows shown', 'rows', { min: 1, max: () => S.N, kind: 'base', fmt: (v) => String(Math.min(v, S.N)) });
  slider(s, 'row stride', 'stride', { min: 1, max: 16, kind: 'base' });
  seg(s, 'blocks', 'block', [['all', 'all'], ['column', 'holding the column']]);
  slider(s, 'column', 'col', { min: 0, max: () => S.N - 1, kind: 'base', enabled: () => S.block === 'column' || S.chart === 'column' });
  seg(s, 'rays', 'rises', [['all', 'all'], ['even', 'even index'], ['odd', 'odd index']]);
  toggle(s, 'only rays with both ends on the grid', 'clip', { kind: 'base' });
}
{
  const s = section(left, 'look');
  seg(s, 'colour', 'color', [['level', 'level'], ['rise', 'rise'], ['block', 'block', 'block parity (HRC: probe parity)'], ['kpar', 'ray parity', 'rise index parity (HRC: even / odd ray)'], ['weight', 'weight', 'the active trace weight']], { cols: 3 });
  slider(s, 'opacity', 'alpha', { min: 0, max: 1, step: 0.01, kind: 'traces', fmt: (v) => v.toFixed(2) });
  slider(s, 'highlight width', 'width', { min: 0.5, max: 8, step: 0.5, kind: 'traces' });
  slider(s, 'point size (line space)', 'pt', { min: 1, max: 8, step: 0.5, kind: 'base', enabled: () => S.rep === 'lines' });
  toggle(s, 'additive blending', 'additive', { kind: 'base' });
  toggle(s, 'filled tiles (line space)', 'fill', { enabled: () => !worldRep() });
  toggle(s, 'pixel grid', 'grid', { enabled: worldRep });
  toggle(s, 'lit pixels', 'heat', { enabled: worldRep, title: 'where the light of the selected ray or the source arrives' });
}
const selInfo = el('div', { className: 'readout' });
{
  const s = section(right, 'selected ray', 'click · Esc');
  s.append(selInfo);
  toggle(s, 'rays it is built from', 'desc', { color: HL.from, title: 'its halves, their halves, … (extensions)' });
  toggle(s, 'extension triangles', 'tri', { color: '#888', title: 'each ray with one pair of its halves: the ray is the top edge' });
  toggle(s, 'rays it is part of', 'anc', { color: HL.into });
  toggle(s, 'where its light goes', 'light', { color: HL.light, title: 'merged radiance its light reaches, and the pixels lit' });
}
{
  const s = section(right, 'follow a line');
  toggle(s, 'line ℓ', 'line');
  const on = () => S.line;
  toggle(s, 'ℓ’s ray in every block', 'lTiles', { color: HL.on, enabled: on, title: 'green: a half of ℓ’s ray one level up; red: not' });
  toggle(s, 'triangles joining them', 'lTri', { color: '#888', enabled: on });
  toggle(s, 'rays the pixels on ℓ read', 'lReads', { color: HL.reads, enabled: on, title: 'through the merge, keeping cones that hold ℓ’s slope' });
  slider(s, 'one pixel along ℓ (−1: all)', 'viewIdx', { min: -1, max: () => S.N - 1, enabled: () => S.line && S.lReads });
  toggle(s, 'light between all pixels on ℓ', 'lAll', { color: HL.pair, enabled: on });
  slider(s, 'A x', 'ax', { min: 0, max: () => S.N, step: 0.1, enabled: on });
  slider(s, 'A y', 'ay', { min: 0, max: () => S.N, step: 0.1, enabled: on });
  slider(s, 'B x', 'bx', { min: 0, max: () => S.N, step: 0.1, enabled: on });
  slider(s, 'B y', 'by', { min: 0, max: () => S.N, step: 0.1, enabled: on });
  slider(s, 'shift across rows', 'shift', { min: -2, max: 2, step: 0.02, enabled: on, fmt: (v) => v.toFixed(2), live: true });
  buttons(s, [
    ['ℓ from the ray', () => { if (!sel) return; S.ax = sel.a[0]; S.ay = sel.a[1] + 0.031; S.bx = sel.b[0]; S.by = sel.b[1] + 0.031; S.shift = 0; S.line = true; rebuild({ base: false }); }, 'the selected ray’s central line, nudged off the tile edges'],
    ['random ℓ', () => { const N = S.N, r = (v) => Math.round(v * 10) / 10; S.ax = 0.3; S.ay = r(Math.random() * N); S.bx = N - 0.3; S.by = r(Math.min(N - 0.1, Math.max(0, S.ay + (Math.random() * 2 - 1) * N * 0.9))); S.line = true; rebuild({ base: false }); }],
  ]);
}
{
  const s = section(right, 'pixels');
  toggle(s, 'rays a pixel reads', 'foot', { color: HL.foot });
  toggle(s, 'light from a source pixel to it', 'pair', { color: HL.pair });
  slider(s, 'pixel x', 'rx', { min: 0, max: () => S.N - 1 });
  slider(s, 'pixel y', 'ry', { min: 0, max: () => S.N - 1 });
  slider(s, 'source x', 'sx', { min: 0, max: () => S.N - 1, enabled: () => S.pair });
  slider(s, 'source y', 'sy', { min: 0, max: () => S.N - 1, enabled: () => S.pair });
  toggle(s, 'route triangles', 'pTri', { color: '#888', enabled: () => S.pair, title: 'the extension triangles on the light’s route, coloured below' });
  seg(s, 'triangle colour', 'triColor', [['error', 'fork error', '(½ − s) m (ρ_lo − ρ_hi), first order: orange, the fork makes the pixel too bright; blue, too dark'], ['share', 'fork share', '(½ − s) m, drawn on the lower fork only: the light plain gives it beyond its geometric share (orange: too much, blue: too little; the upper fork has the opposite)'], ['merged', 'merged', 'the parent’s merged radiance m'], ['flow', 'flow', 'the light of this pair through the parent'], ['level', 'level']], { kind: 'traces', cols: 3, enabled: () => S.pair && S.pTri });
  seg(s, 'a click on the floor sets', 'click', [['ray', 'nothing', 'clicks pick rays'], ['pixel', 'pixel'], ['source', 'source'], ['A', 'ℓ: A'], ['B', 'ℓ: B']], { kind: 'none' });
}
const readout = el('div', { className: 'readout' });
section(right, 'readout').append(readout);
const guide = el('div', { id: 'guide' });
section(right, 'how to read it').append(guide);
{
  const s = section(right, 'overlay geometry');
  const pick = el('select');
  pick.append(el('option', { value: '', textContent: '— preset —' }), ...Object.keys(PRESETS).map((k) => el('option', { value: k, textContent: k })));
  const ta = el('textarea', { spellcheck: false });
  pick.onchange = () => { if (pick.value) ta.value = PRESETS[pick.value]; };
  s.append(pick, ta);
  buttons(s, [['run', () => { overlayCode = ta.value; rebuild({ base: false }); }], ['clear', () => { overlayCode = ''; rebuild({ base: false }); }]]);
  s.append(el('div', { className: 'hint', textContent: 'api: segments, rawSegments, triangles, points, surface, embedRay, embedTile, joinAt, tree, line, G, N, state' }));
}
{
  const s = el('section', { id: 'tools' });
  right.append(s);
  for (const [text, fn] of [['reset view', () => frame()], ['top view', () => frame(true)], ['save PNG', () => saveShot('ray-tree-3d')]]) {
    const b = el('button', { textContent: text });
    b.onclick = fn;
    s.append(b);
  }
}

const GUIDE = {
  joins: `<p><b>Joins.</b> Each column sits at its <b>join level</b>: the level whose two halves meet there (an odd multiple of 2ⁿ sits at height n; the frame's two ends at the top). A ray of level n runs from the column where it meets its sibling, at height n, up to its outer end. A ray and its two halves are then a triangle: the ray is the top edge, the halves are the lower edges, meeting at the bottom corner.</p>`,
  stacked: `<p><b>Stacked.</b> Each level is drawn flat at its own height (or the height picked above). Level n has N/2ⁿ blocks; each block holds every rise at every row.</p>`,
  lines: `<p><b>Line space.</b> Each ray is a point at the slope and offset of its central line, one level per height. Highlights draw the ray's <b>line tile</b>: the lines that enter its block in its start row and leave it in its end row (two rows wide at 1×). One level's tiles in one block cover every line once.</p>`,
  linecol: `<p><b>Line + column.</b> Each ray stands over its central line's point in the line chart, from the column where it starts to the column where it ends. All levels of one line stand over nearly one point.</p>`,
  passes: `<p><b>Extensions</b> build a ray from two halves: its radiance is the first half's plus the second half's seen through the first. A selected ray's <b>extension triangles</b> show this, one per pair of halves (a ray of odd rise has two bent pairs, each at half weight).</p><p><b>Merge.</b> Reciprocal HRC reads the same triangles downwards: a ray's merged radiance (light arriving at its start) passes to its first half unchanged and, carried through the first half with the cone weight, to its second half. HRC keeps merged radiance on cones at its probes (drawn from the probe along the cone's centre, between two rays) and reads the next level's probe where the ray ends, or, for an even probe, where the ray and the next probe's ray end.</p>`,
  line: `<p><b>Line ℓ</b> is one straight line of the scene. Each block holds it in exactly one ray per level, the ray whose line tile contains ℓ. A selected ray shows one ray's family; ℓ shows the one ray per block that carries ℓ, whichever ray you picked: green where it is a half of ℓ's ray one level up, red where it is not. Pink rays are what the pixels on ℓ actually read for ℓ's direction, which need not be ℓ's own rays.</p>`,
  fib: `<p><b>Fibonacci tree.</b> Lengths are 1, 2, 3, 5, 8, …, up to a frame at least N wide (the frame runs past the scene). A node of length F<sub>k</sub> splits at F<sub>k−1</sub>, the long piece first (light toward −x or −y: the mirrored tree, so the same columns split). The junction row is not a whole row: the ray's centre line crosses the junction at θd (θ = F<sub>k−1</sub>/F<sub>k</sub>), so the ray is two chains, via the row below with weight 1 − f and via the row above with weight f (f the fraction of θd). Its extension triangles are those two chains; nodes of one length do not tile the frame.</p>`,
  forks: `<p><b>Forks.</b> A ray of odd rise has two bent pairs of halves: two triangles on one top edge whose inner corners are a row apart. A straight line passes through only one of them, but plain gives each half the far light. With "route triangles" on, "fork share" colours each fork triangle by the light plain gives it beyond its geometric share (orange: too much, blue: too little), and "fork error" by what that does to the pixel, to first order (orange: too bright, blue: too dark).</p>`,
  model: `<p class="hint">Index arithmetic as on the HRC and Reciprocal HRC pages. Traces are free space (transmittance 1). Rays whose start row is off the grid are left out.</p>`,
};
function writeReadout() {
  const selParts = [], rest = [];
  for (const P of panels) {
    const v = P.vis;
    if (P.ray.length) selParts.push(`<b>${SCHEMES[P.scheme]}</b>\n${P.ray.join('\n')}`);
    rest.push([`<b>${SCHEMES[P.scheme]}</b> ${P.tree.total.toLocaleString()} rays per frustum · ${v.shown.toLocaleString()} drawn${v.keepEvery > 1 ? ` (1 in ${v.keepEvery})` : ''}`, ...P.stats].join('\n'));
  }
  selInfo.innerHTML = sel ? selParts.join('\n\n') : '<span style="color:var(--faint)">none</span>';
  readout.innerHTML = rest.join('\n\n');
  guide.innerHTML = GUIDE[S.rep] + GUIDE.passes + (schemesOf().includes('fib') ? GUIDE.fib : '') + (S.line ? GUIDE.line : '') + (S.pair && S.pTri ? GUIDE.forks : '') + GUIDE.model;
}

/** POST to a shot server when one is given (scripted captures), else download */
async function saveShot(name, url) {
  draw();
  const data = renderer.domElement.toDataURL('image/png');
  if (url) { await fetch(`${url}?name=${encodeURIComponent(name)}`, { method: 'POST', body: data }); return; }
  Object.assign(document.createElement('a'), { download: `${name}.png`, href: data }).click();
}

new ResizeObserver(layout).observe(area);
syncs.push(() => useCamera(S.cam));
clampToN();
makePanels();
layout();
frame();
rebuild();
render();
window.viz = { S, rebuild, frame, draw, pickAt, saveShot, panels: () => panels, controls, camera: () => camera, setSel: (s) => { sel = s; rebuild({ base: false }); }, setOverlay: (c) => { overlayCode = c; rebuild({ base: false }); }, syncAll };
