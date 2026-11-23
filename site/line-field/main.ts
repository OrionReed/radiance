import { $, el, createControls } from '../_shared/ui';
import { exactAt, relatives, solve, STRIDE, type Injection, type Solution } from './lines';

// ── rendering: lines summed into a float buffer, then shown; one Renderer per canvas ──

/** one drawing of the line field into a viewport */
interface Pass {
  vp: [number, number, number, number]; // device pixels, origin bottom left
  view: { x: number; y: number; span: number };
  N: number;
  extMask: number; mergeMask: number; frustumMask: number;
  extMode: number; mergeMode: number;
  profile: number; width: number; alongW: number; alongV: number;
  gain: number;
  mode: number; // 0 lines, 1 pixels, 2 exact, 3 lines ÷ exact, 4 pixels ÷ exact, 5 coverage
  tone: number; top: number; grid: number; coverageNorm: number; latticeMask: number;
  selPix: number[] | null;
  /** solid lines drawn on top: x0 y0 x1 y1, r g b a, width in CSS pixels */
  overlay: number[];
}

const LINES_VS = /* glsl */ `#version 300 es
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 seg;
layout(location = 2) in vec4 vals;  // M, M at end, c·R, cone weight c
layout(location = 3) in vec2 meta;  // level, frustum
layout(location = 4) in vec2 rt;    // R, T
uniform vec4 view;                  // x0, y0, span, world units per device pixel
uniform int extMask, mergeMask, latticeMask, frustumMask, profile;
uniform float width;
out float vT, vPerp;
flat out float vCos;
flat out vec4 vVals;
flat out vec2 vRT;
flat out int vExt, vMerge, vLat;
void main() {
  int lv = int(meta.x + 0.5), fr = int(meta.y + 0.5);
  int ext = (extMask >> lv) & 1, merge = (mergeMask >> lv) & 1, lat = (latticeMask >> lv) & 1;
  if (ext + merge + lat == 0 || ((frustumMask >> fr) & 1) == 0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 d = seg.zw - seg.xy, dir = normalize(d), n = vec2(-dir.y, dir.x);
  float c = max(abs(dir.x), abs(dir.y));
  float halfW = profile == 3 ? 1.5 * view.w : 0.5 * width * c + view.w;
  vec2 p = mix(seg.xy, seg.zw, corner.x) + n * corner.y * halfW;
  vec2 uv = (p - view.xy) / view.z;
  gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  vT = corner.x; vPerp = corner.y * halfW; vCos = c; vVals = vals; vRT = rt; vExt = ext; vMerge = merge; vLat = lat;
}`;

// Across a line the profile is taken in rows (the frustum's row axis), v = perp / cos θ: lines with one row offset sit one
// row apart, so a profile of unit integral makes them tile. Along it, the width and the weight can vary with t.
const LINES_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
uniform int profile, extMode, mergeMode, alongW, alongV;
uniform float width, gain;
uniform vec4 view;
in float vT, vPerp;
flat in float vCos;
flat in vec4 vVals;
flat in vec2 vRT;
flat in int vExt, vMerge, vLat;
out vec4 o;
void main() {
  float t = vT, tent = 1.0 - abs(2.0 * t - 1.0);
  float fw = alongW == 0 ? 1.0 : alongW == 1 ? tent : alongW == 2 ? t : 1.0 - t;
  float px = view.w / vCos, w = max(width * fw, px), v = vPerp / vCos, prof;
  if (profile == 0) prof = clamp((0.5 * w - abs(v)) / px + 0.5, 0.0, 1.0) / w;
  else if (profile == 1) prof = max(0.0, 1.0 - abs(2.0 * v / w)) * 2.0 / w;
  else if (profile == 2) { float s = w / 4.0; prof = exp(-v * v / (2.0 * s * s)) / (s * 2.5066283); }
  else prof = max(0.0, 1.0 - abs(vPerp) / view.w); // a tent one device pixel in radius: the same total wherever the line falls
  if (profile != 3 && abs(v) > 0.5 * w + px) prof = 0.0;
  float weight = alongV == 1 ? 2.0 * tent : 1.0;
  float extV = extMode == 0 ? vRT.x : extMode == 1 ? vRT.y : vVals.z;
  float mergeV = mergeMode == 0 ? vVals.x : mergeMode == 1 ? vVals.y : mix(vVals.x, vVals.y, t);
  float val = float(vExt) * extV + float(vMerge) * mergeV;
  float lattice = 0.6 * float(vLat) * max(0.0, 1.0 - abs(vPerp) / view.w);
  o = vec4(gain * weight * prof * val, gain * weight * prof * vVals.w, lattice, 1.0);
}`;

const DISPLAY_VS = /* glsl */ `#version 300 es
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const DISPLAY_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D accum, pix, exact;
uniform vec4 view, vp;
uniform float N, top, coverageNorm;
uniform int mode, tone, grid;
uniform vec2 selPix; // the focused pixel (x < 0: none)
out vec4 o;
vec3 ratioColour(float r) {
  if (!(r > 0.0)) return vec3(0.0);
  float s = clamp(log(r) / log(1.25), -1.0, 1.0);
  vec3 base = vec3(0.13);
  return s > 0.0 ? mix(base, vec3(1.0, 0.3, 0.2), s) : mix(base, vec3(0.25, 0.5, 1.0), -s);
}
vec3 toneMap(float v) {
  float t = tone == 1 ? 1.0 + log2(max(v, 1e-30) / top) / 10.0 : pow(clamp(v / top, 0.0, 1.0), 1.0 / 2.2);
  return vec3(clamp(t, 0.0, 1.0));
}
void main() {
  vec2 uv = (gl_FragCoord.xy - vp.xy) / vp.zw;
  uv.y = 1.0 - uv.y;
  vec2 w = view.xy + uv * view.z;
  if (w.x < 0.0 || w.y < 0.0 || w.x >= N || w.y >= N) { o = vec4(0.04, 0.04, 0.04, 1.0); return; }
  vec3 acc = texelFetch(accum, ivec2(gl_FragCoord.xy), 0).rgb;
  vec2 px = texelFetch(pix, ivec2(floor(w)), 0).rg;
  float ex = texture(exact, w / N).r;
  vec3 c;
  if (mode == 0) c = toneMap(acc.r);
  else if (mode == 1) c = toneMap(px.r);
  else if (mode == 2) c = toneMap(ex);
  else if (mode == 3) c = ratioColour(acc.r / ex);
  else if (mode == 4) c = ratioColour(px.r / ex);
  else c = ratioColour(acc.g / coverageNorm);
  vec3 structure = vec3(0.3, 0.48, 0.85); // the pixel grid, and a cascade's own lines in the stack
  if (grid == 1) {
    vec2 f = abs(w - round(w)) / view.w;
    float g = 1.0 - clamp(min(f.x, f.y) - 0.5, 0.0, 1.0);
    float k = clamp((1.0 / view.w - 4.0) / 8.0, 0.0, 1.0); // fades out when pixels get small on screen
    c = mix(c, structure, 0.35 * g * k);
  }
  if (grid == 2) c += structure * 0.3 * min(acc.b, 1.0);
  if (px.g > 0.5) c = vec3(1.0, 0.85, 0.4);
  vec3 accent = vec3(1.0, 0.31, 0.85);
  if (selPix.x >= 0.0) {
    vec2 q = (w - selPix) / view.w; // device pixels from the focused pixel's corner
    float side = 1.0 / view.w, e = min(min(q.x, q.y), min(side - q.x, side - q.y));
    if (e > -1.5 && e < 1.0) c = accent;
  }
  o = vec4(c, 1.0);
}`;

const OVERLAY_VS = /* glsl */ `#version 300 es
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 seg;
layout(location = 2) in vec4 col;
layout(location = 3) in float widthPx;
uniform vec4 view; // x0, y0, span, world units per device pixel
uniform float dpr;
out float vAcross;
flat out float vHalf;
flat out vec4 vCol;
void main() {
  vec2 d = seg.zw - seg.xy, dir = normalize(d), n = vec2(-dir.y, dir.x);
  float hw = 0.5 * widthPx * dpr + 1.0;
  vec2 p = mix(seg.xy, seg.zw, corner.x) + n * corner.y * hw * view.w;
  vec2 uv = (p - view.xy) / view.z;
  gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  vAcross = corner.y * hw; vHalf = hw - 1.0; vCol = col;
}`;
const OVERLAY_FS = /* glsl */ `#version 300 es
precision highp float;
in float vAcross;
flat in float vHalf;
flat in vec4 vCol;
out vec4 o;
void main() { o = vec4(vCol.rgb, vCol.a * clamp(vHalf - abs(vAcross) + 0.5, 0.0, 1.0)); }`;

class Renderer {
  gl: WebGL2RenderingContext;
  lines; display; overlayProg; overlayVao: WebGLVertexArrayObject; overlayBuf: WebGLBuffer; dpr = 1;
  vao: WebGLVertexArrayObject; instances: WebGLBuffer;
  accum: WebGLTexture; pix: WebGLTexture; exact: WebGLTexture; fbo: WebGLFramebuffer;
  size = [0, 0]; count = 0; floatBlend: boolean;

  constructor(public canvas: HTMLCanvasElement) {
    const gl = (this.gl = canvas.getContext('webgl2', { antialias: false })!);
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('needs EXT_color_buffer_float');
    this.floatBlend = !!gl.getExtension('EXT_float_blend');
    this.lines = this.program(LINES_VS, LINES_FS);
    this.display = this.program(DISPLAY_VS, DISPLAY_FS);
    const corner = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, corner);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    this.instances = gl.createBuffer()!;
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instances);
    for (const [loc, n, off] of [[1, 4, 0], [2, 4, 4], [3, 2, 8], [4, 2, 10]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, STRIDE * 4, off * 4);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    this.overlayProg = this.program(OVERLAY_VS, OVERLAY_FS);
    this.overlayBuf = gl.createBuffer()!;
    this.overlayVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.overlayVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, corner);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.overlayBuf);
    for (const [loc, n, off] of [[1, 4, 0], [2, 4, 4], [3, 1, 8]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 9 * 4, off * 4);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    this.accum = this.texture(gl.NEAREST); this.pix = this.texture(gl.NEAREST); this.exact = this.texture(gl.LINEAR);
    this.fbo = gl.createFramebuffer()!;
  }

  program(vs: string, fs: string) {
    const gl = this.gl, p = gl.createProgram()!;
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? '');
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? '');
    const cache: Record<string, WebGLUniformLocation | null> = {};
    return { p, u: (k: string) => (k in cache ? cache[k] : (cache[k] = gl.getUniformLocation(p, k))) };
  }

  texture(filter: number) {
    const gl = this.gl, t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  upload(sol: Solution, pix: Float32Array, exact: Float32Array, M: number) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instances);
    gl.bufferData(gl.ARRAY_BUFFER, sol.lines.subarray(0, sol.count * STRIDE), gl.DYNAMIC_DRAW);
    this.count = sol.count;
    gl.bindTexture(gl.TEXTURE_2D, this.pix);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, sol.N, sol.N, 0, gl.RG, gl.FLOAT, pix);
    gl.bindTexture(gl.TEXTURE_2D, this.exact);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, M, M, 0, gl.RED, gl.FLOAT, exact);
  }

  /** match the canvas to its CSS size and clear it; returns device pixels per CSS pixel */
  resize() {
    const gl = this.gl, r = this.canvas.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (w !== this.size[0] || h !== this.size[1]) {
      this.canvas.width = w; this.canvas.height = h; this.size = [w, h];
      gl.bindTexture(gl.TEXTURE_2D, this.accum);
      gl.texImage2D(gl.TEXTURE_2D, 0, this.floatBlend ? gl.RGBA32F : gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.accum, 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    return (this.dpr = w / Math.max(1, r.width));
  }

  pass(p: Pass) {
    const gl = this.gl, [x, y, w, h] = p.vp, perPx = p.view.span / w;
    gl.enable(gl.SCISSOR_TEST);
    gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (p.mode === 0 || p.mode === 3 || p.mode === 5) {
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      const { p: prog, u } = this.lines;
      gl.useProgram(prog);
      gl.uniform4f(u('view'), p.view.x, p.view.y, p.view.span, perPx);
      for (const k of ['extMask', 'mergeMask', 'latticeMask', 'frustumMask', 'profile', 'extMode', 'mergeMode', 'alongW', 'alongV'] as const) gl.uniform1i(u(k), p[k]);
      gl.uniform1f(u('width'), p.width); gl.uniform1f(u('gain'), p.gain);
      gl.bindVertexArray(this.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const { p: prog, u } = this.display;
    gl.useProgram(prog);
    [this.accum, this.pix, this.exact].forEach((t, i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); });
    gl.uniform1i(u('accum'), 0); gl.uniform1i(u('pix'), 1); gl.uniform1i(u('exact'), 2);
    gl.uniform4f(u('view'), p.view.x, p.view.y, p.view.span, perPx);
    gl.uniform4f(u('vp'), x, y, w, h);
    gl.uniform1f(u('N'), p.N); gl.uniform1f(u('top'), p.top); gl.uniform1f(u('coverageNorm'), p.coverageNorm);
    gl.uniform1i(u('mode'), p.mode); gl.uniform1i(u('tone'), p.tone); gl.uniform1i(u('grid'), p.grid);
    gl.uniform2f(u('selPix'), p.selPix ? p.selPix[0] : -1, p.selPix ? p.selPix[1] : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (p.overlay.length) {
      const o = this.overlayProg;
      gl.useProgram(o.p);
      gl.uniform4f(o.u('view'), p.view.x, p.view.y, p.view.span, perPx);
      gl.uniform1f(o.u('dpr'), this.dpr);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.overlayBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(p.overlay), gl.STREAM_DRAW);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.overlayVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, p.overlay.length / 9);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
    gl.disable(gl.SCISSOR_TEST);
  }
}

const SELECTED = [1.0, 0.31, 0.85, 1.0];
const DESCENDANT = [1.0, 0.7, 0.28];
const ANCESTOR = [0.66, 0.55, 1.0];

// ── state ──

type Range = [number, number] | null;
const S = {
  N: 32,
  tool: 'move' as 'move' | 'paint' | 'erase',
  /** the cascades each pass computes */
  ext: [0, 4] as Range,
  merge: [0, 4] as Range,
  /** the cascades drawn, and which pass's values they show */
  draw: 0b11111,
  drawExt: false, drawMerge: true,
  extValue: 0, mergeValue: 0,
  frustums: 0b0001,
  profile: 0, width: 1, alongW: 0, alongV: 0,
  average: false,
  show: 0, tone: 0, grid: true, exposure: 0,
};
const K = () => Math.log2(S.N);
const VIEW = () => ({ x: 0, y: 0, span: S.N });
let lights: [number, number][] = [];
let injections: Injection[] = [];
type LineId = { frustum: number; k: number; plane: number; row: number; d: number };
let focus: { line: LineId } | { pixel: [number, number] } | null = null;
let sol: Solution | null = null, autoTop = 1;

const setLight = (size: number) => {
  lights = [];
  const cx = Math.floor(S.N / 4), cy = Math.floor(S.N / 2) - Math.floor(size / 2);
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) lights.push([cx + i, cy + j]);
};
const inRange = (r: Range, k: number) => !!r && k >= r[0] && k <= r[1];
const bits = (m: number) => { let n = 0; for (; m; m &= m - 1) n++; return n; };
const allLevels = () => (1 << K()) - 1;

const main = new Renderer($('view') as HTMLCanvasElement);
const stack = new Renderer($('stack') as HTMLCanvasElement);

// ── solve ──
let solvedKey = '';
const passOptions = () => {
  const k = K();
  return {
    lights: true,
    upFrom: S.ext ? S.ext[0] : 0, upTo: S.ext ? S.ext[1] : -1,
    downFrom: S.merge ? S.merge[1] + 1 : 0, downTo: S.merge ? S.merge[0] : k - 1,
  };
};
const solveKey = () => JSON.stringify([S.N, lights, injections, passOptions()]);
function compute() {
  const N = S.N;
  solvedKey = solveKey();
  const emission = new Float32Array(N * N), solid = new Uint8Array(N * N);
  for (const [x, y] of lights) { emission[x * N + y] = 1; solid[x * N + y] = 1; }
  sol = solve(N, emission, solid, passOptions(), injections);

  const pix = new Float32Array(N * N * 2), pos: number[] = [];
  for (let x = 0; x < N; x++)
    for (let y = 0; y < N; y++) {
      const f = sol.fluence[x * N + y];
      pix[(y * N + x) * 2] = f; pix[(y * N + x) * 2 + 1] = solid[x * N + y];
      if (f > 0 && !solid[x * N + y]) pos.push(f);
    }
  // exposure: the 98th percentile of the pixels, or the largest line value when the pixels are dark
  pos.sort((a, b) => a - b);
  autoTop = pos.length ? pos[Math.floor(0.98 * (pos.length - 1))] : 0;
  if (!(autoTop > 0)) {
    for (let i = 0; i < sol.count; i++) for (const o of [4, 5, 10]) autoTop = Math.max(autoTop, Math.abs(sol.lines[i * STRIDE + o]));
    if (!(autoTop > 0)) autoTop = 1;
  }

  const sub = Math.max(2, Math.floor((lights.length > 16 ? 256 : 512) / N)), M = N * sub;
  const reference = new Float32Array(M * M);
  for (let j = 0; j < M; j++)
    for (let i = 0; i < M; i++) {
      const e = lights.length ? exactAt((i + 0.5) / sub, (j + 0.5) / sub, lights) : 0;
      reference[j * M + i] = Number.isNaN(e) ? -1 : e;
    }
  indexLines();
  main.upload(sol, pix, reference, M);
  stack.upload(sol, pix, reference, M);
}

// ── controls ──
const { segmented, slider, toggle, syncControls } = createControls(() => changed());
const field = (name: keyof typeof S) => ({ get: () => S[name], set: (v: unknown) => { (S as Record<string, unknown>)[name] = v; } });
const FRUSTUMS = ['+x', '−x', '+y', '−y'];
const fieldWrap = (label: string, body: HTMLElement, title = '') =>
  el('div', { className: 'field', title }, el('div', { className: 'label' }, el('span', { textContent: label })), body);

segmented($('scene'), { label: 'resolution', options: [16, 32, 64, 128].map((n) => [n, String(n)]), get: () => S.N, set: setN });
segmented($('scene'), { label: 'light', options: [['move', 'move'], ['paint', 'paint'], ['erase', 'erase']], ...field('tool') });
segmented($('scene'), { options: [[1, '1 px'], [2, '2×2'], [3, '3×3'], [0, 'none']], get: () => -1, set: (n: number) => setLight(n) });

/** a strip of cascades; drag across it to set a range, drag an end to move it, shift-click to extend */
function rangeStrip(get: () => Range, set: (r: Range) => void) {
  const seg = el('div', { className: 'seg strip' });
  const buttons: HTMLButtonElement[] = [];
  let anchor = -1;
  for (let k = 0; k < K(); k++) {
    const b = el('button', { textContent: `c${k}`, title: `rays of length ${1 << k}` });
    b.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      b.releasePointerCapture(e.pointerId);
      const r = get();
      if (r && e.shiftKey) { const n: Range = [Math.min(r[0], k), Math.max(r[1], k)]; set(n); anchor = k === n[0] ? n[1] : n[0]; }
      else if (r && r[0] !== r[1] && (k === r[0] || k === r[1])) anchor = k === r[0] ? r[1] : r[0];
      else { anchor = k; set([k, k]); }
      changed();
    });
    b.addEventListener('pointerenter', () => { if (anchor >= 0) { set([Math.min(anchor, k), Math.max(anchor, k)]); changed(); } });
    seg.append(b); buttons.push(b);
  }
  const all = el('button', { textContent: 'all', className: 'end' });
  all.onclick = () => { set([0, K() - 1]); changed(); };
  const none = el('button', { textContent: 'none', className: 'end' });
  none.onclick = () => { set(null); changed(); };
  seg.append(all, none);
  addEventListener('pointerup', () => (anchor = -1));
  return { seg, sync: () => { const r = get(); buttons.forEach((b, k) => b.classList.toggle('on', inRange(r, k))); none.classList.toggle('on', !r); all.classList.toggle('on', !!r && r[0] === 0 && r[1] === K() - 1); } };
}
/** a strip of cascades as independent switches; drag to paint the same state across */
function setStrip(get: () => number, set: (m: number) => void) {
  const seg = el('div', { className: 'seg strip' });
  const buttons: HTMLButtonElement[] = [];
  let paint: boolean | null = null;
  const apply = (k: number) => { const m = get(); set(paint ? m | (1 << k) : m & ~(1 << k)); changed(); };
  for (let k = 0; k < K(); k++) {
    const b = el('button', { textContent: `c${k}`, title: `rays of length ${1 << k}` });
    b.addEventListener('pointerdown', (e) => { e.preventDefault(); b.releasePointerCapture(e.pointerId); paint = !((get() >> k) & 1); apply(k); });
    b.addEventListener('pointerenter', () => { if (paint !== null) apply(k); });
    seg.append(b); buttons.push(b);
  }
  const all = el('button', { textContent: 'all', className: 'end' });
  all.onclick = () => { set(allLevels()); changed(); };
  seg.append(all);
  addEventListener('pointerup', () => (paint = null));
  return { seg, sync: () => { const m = get(); buttons.forEach((b, k) => b.classList.toggle('on', !!((m >> k) & 1))); all.classList.toggle('on', m === allLevels()); } };
}
const stripHost = el('div');
$('passes').append(stripHost);
let strips: { sync: () => void }[] = [];
function buildStrips() {
  stripHost.innerHTML = '';
  const ext = rangeStrip(() => S.ext, (r) => (S.ext = r));
  const merge = rangeStrip(() => S.merge, (r) => (S.merge = r));
  const draw = setStrip(() => S.draw, (m) => (S.draw = m));
  stripHost.append(
    fieldWrap('extension computes', ext.seg, 'R and T, from the bottom of the range up to its top; below the range only lit lines carry R'),
    fieldWrap('merge computes', merge.seg, 'M, from above the top of the range (nothing arriving) down to its bottom'),
    fieldWrap('draw', draw.seg, 'the cascades the main view sums'),
  );
  strips = [ext, merge, draw];
}
segmented($('passes'), { label: 'values drawn', options: [['ext', 'extension'], ['merge', 'merge'], ['both', 'both']], get: () => (S.drawExt && S.drawMerge ? 'both' : S.drawExt ? 'ext' : 'merge'), set: (v: string) => { S.drawExt = v !== 'merge'; S.drawMerge = v !== 'ext'; } });
segmented($('passes'), {
  label: 'extension value',
  options: [[0, 'R', 'radiance along the ray'], [1, 'T', 'transmittance along the ray'], [2, 'c·R', 'R times the cone weight c']],
  ...field('extValue'),
});
segmented($('passes'), {
  label: 'merge value',
  options: [[0, 'M', 'merged radiance: the light reaching the ray\'s start from behind it, in its cone'], [1, 'M at end', 'T·M + c·R: the light leaving the ray\'s end'], [2, 'M along', 'from M at the start to M at end, along the ray']],
  ...field('mergeValue'),
});
const frustumSeg = el('div', { className: 'seg' });
const frustumButtons = FRUSTUMS.map((name, k) => {
  const b = el('button', { textContent: name, title: `the frustum carrying light toward ${name}` });
  b.onclick = () => { S.frustums ^= 1 << k; changed(); };
  frustumSeg.append(b);
  return b;
});
$('passes').append(fieldWrap('frustums', frustumSeg));
segmented($('passes'), { label: 'cascades drawn', options: [[false, 'sum'], [true, 'average']], ...field('average') });

segmented($('lines'), { label: 'profile across', title: 'in rows: lines with one row offset sit one row apart', options: [[0, 'none'], [1, 'tent'], [2, 'gauss']], ...field('profile') });
slider($('lines'), { label: 'width', min: 0.05, max: 4, step: 0.05, ...field('width'), format: (v: number) => `${v.toFixed(2)} rows` });
segmented($('lines'), { label: 'width along', options: [[0, 'even'], [1, 'diamond'], [2, 'opens'], [3, 'closes']], ...field('alongW') });
segmented($('lines'), { label: 'weight along', options: [[0, 'even'], [1, 'tent']], ...field('alongV') });

segmented($('viewc'), {
  label: 'show', cols: 3,
  options: [[0, 'lines'], [1, 'pixels'], [2, 'reference'], [3, 'lines ÷ ref'], [4, 'pixels ÷ ref'], [5, 'coverage', 'Σ profile × cone weight ÷ 2π: 1 where the lines tile']],
  ...field('show'),
});
segmented($('viewc'), { label: 'tone', options: [[0, 'linear'], [1, 'log']], ...field('tone') });
slider($('viewc'), { label: 'exposure', key: '[ ]', min: -6, max: 6, step: 0.25, ...field('exposure'), format: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}` });
toggle($('viewc'), { label: 'pixel grid', ...field('grid') });

// ── cascade stack: the top cascade first, c0 last, then the pixels ──
const grid = $('stack-grid');
let tiles: { k: number; el: HTMLElement; sq: HTMLElement }[] = [];
function buildStack() {
  grid.innerHTML = '';
  tiles = [];
  for (let k = K() - 1; k >= -1; k--) {
    const pixels = k < 0;
    const sq = el('div', { className: 'sq' });
    const name = el('div', { className: 'name' }, el('b', { textContent: pixels ? 'pixels' : `c${k}` }), el('span', { textContent: pixels ? 'fluence' : `length ${1 << k}` }));
    const t = el('div', { className: 'tile' }, name, sq);
    sq.addEventListener('pointerdown', (e) => {
      const r = sq.getBoundingClientRect(), wx = ((e.clientX - r.left) / r.width) * S.N, wy = ((e.clientY - r.top) / r.height) * S.N;
      if (pixels) focus = { pixel: [Math.floor(wx), Math.floor(wy)] };
      else pickAt(wx, wy, k);
      changed();
    });
    grid.append(t);
    tiles.push({ k, el: t, sq });
  }
}

// ── focus, and lighting lines ──
/** every line's index in sol.lines, by its id */
let lineIndex = new Map<number, number>();
const keyOf = (v: number, k: number, plane: number, row: number, d: number) =>
  ((((v * 16 + k) * S.N + plane) * 3 * S.N + row) * (2 * S.N + 1)) + d + S.N;
function indexLines() {
  lineIndex = new Map();
  if (!sol) return;
  const L = sol.lines;
  for (let i = 0; i < sol.count; i++) {
    const o = i * STRIDE;
    lineIndex.set(keyOf(L[o + 9], L[o + 8], L[o + 12], L[o + 13], L[o + 14]), i);
  }
}
const indexOf = (id: LineId) => lineIndex.get(keyOf(id.frustum, id.k, id.plane, id.row, id.d)) ?? -1;
/** a ray by its cascade, frustum and the image positions of its start and end probes */
const idText = (p: LineId) => {
  const i = indexOf(p);
  if (i < 0 || !sol) return `c${p.k} ${FRUSTUMS[p.frustum]}  off the image`;
  const L = sol.lines, o = i * STRIDE, q = (v: number) => String(+v.toFixed(1));
  return `c${p.k} ${FRUSTUMS[p.frustum]}  (${q(L[o])}, ${q(L[o + 1])}) → (${q(L[o + 2])}, ${q(L[o + 3])})`;
};
function pickAt(wx: number, wy: number, k: number) {
  if (!sol) return;
  const L = sol.lines;
  let best = -1, bestD = Infinity;
  for (let i = 0; i < sol.count; i++) {
    const o = i * STRIDE;
    if (L[o + 8] !== k || !((S.frustums >> L[o + 9]) & 1)) continue;
    const ax = L[o], ay = L[o + 1], bx = L[o + 2] - ax, by = L[o + 3] - ay;
    const t = Math.max(0, Math.min(1, ((wx - ax) * bx + (wy - ay) * by) / (bx * bx + by * by)));
    const dist = Math.hypot(wx - ax - t * bx, wy - ay - t * by);
    if (dist < bestD) { bestD = dist; best = i; }
  }
  if (best < 0) return;
  const o = best * STRIDE;
  focus = { line: { k: L[o + 8], frustum: L[o + 9], plane: L[o + 12], row: L[o + 13], d: L[o + 14] } };
}
const lightFocus = (pass: 'R' | 'M') => {
  if (!focus || !('line' in focus)) return;
  injections.push({ ...focus.line, pass, value: Number(($('inj-value') as HTMLInputElement).value) });
  changed();
};
$('inj-r').onclick = () => lightFocus('R');
$('inj-g').onclick = () => lightFocus('M');
$('inj-clear').onclick = () => { injections = []; changed(); };

function showFocus(i: number) {
  const f = (v: number) => (Number.isFinite(v) ? v.toPrecision(3) : '—');
  const box = $('focus');
  if (focus && 'line' in focus) {
    if (i < 0 || !sol) box.innerHTML = `<b>${idText(focus.line)}</b>`;
    else {
      const L = sol.lines, o = i * STRIDE;
      box.innerHTML = `<b>${idText(focus.line)}</b>\nR ${f(L[o + 10])}   T ${f(L[o + 11])}   c ${f(L[o + 7])}\nM ${f(L[o + 4])}   M at end ${f(L[o + 5])}`;
    }
  } else if (focus && 'pixel' in focus && sol) {
    const [x, y] = focus.pixel, v = sol.fluence[x * S.N + y], e = lights.length ? exactAt(x + 0.5, y + 0.5, lights) : 0;
    box.innerHTML = `<b>pixel (${x}, ${y})</b>\nfluence ${f(v)}   reference ${f(e)}\nratio ${f(v / e)}`;
  } else box.innerHTML = 'click a line in a cascade, or a pixel';
  $('injections').textContent = injections.map((j) => `${j.pass} + ${j.value}   ${idText(j)}`).join('\n');
  $('inject').classList.toggle('disabled', !(focus && 'line' in focus));
}

// ── main view: move or paint the light ──
const canvas = $('view') as HTMLCanvasElement;
const pixelAt = (e: { clientX: number; clientY: number }) => {
  const r = canvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * S.N, ((e.clientY - r.top) / r.height) * S.N];
};
let drag: { start: number[]; lights: [number, number][] } | null = null;
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  const p = pixelAt(e);
  drag = { start: p, lights: lights.map((l) => [...l] as [number, number]) };
  if (S.tool !== 'move') paintAt(p[0], p[1]);
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const [wx, wy] = pixelAt(e);
  if (S.tool !== 'move') { paintAt(wx, wy); return; }
  const dx = Math.round(wx - drag.start[0]), dy = Math.round(wy - drag.start[1]);
  const moved = drag.lights.map(([x, y]) => [x + dx, y + dy] as [number, number]);
  if (moved.every(([x, y]) => x >= 0 && y >= 0 && x < S.N && y < S.N) && moved.some(([x, y], i) => x !== lights[i][0] || y !== lights[i][1])) {
    lights = moved; changed();
  }
});
addEventListener('pointerup', () => (drag = null));
function paintAt(wx: number, wy: number) {
  const x = Math.floor(wx), y = Math.floor(wy);
  if (x < 0 || y < 0 || x >= S.N || y >= S.N) return;
  const i = lights.findIndex(([a, b]) => a === x && b === y);
  if (S.tool === 'paint' && i < 0) { lights.push([x, y]); changed(); }
  if (S.tool === 'erase' && i >= 0) { lights.splice(i, 1); changed(); }
}
addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'INPUT') return;
  if (e.key === '[' || e.key === ']') { S.exposure = Math.max(-6, Math.min(6, S.exposure + (e.key === ']' ? 0.25 : -0.25))); changed(); }
});

function setN(n: number) {
  const old = K(), scale = n / S.N;
  S.N = n;
  const k = K(), fit = (r: Range): Range => r && [Math.min(r[0], k - 1), r[1] === old - 1 ? k - 1 : Math.min(r[1], k - 1)];
  lights = [...new Map(lights.map(([x, y]) => [Math.floor(x * scale), Math.floor(y * scale)] as [number, number]).map((l) => [l.join(), l])).values()];
  S.ext = fit(S.ext); S.merge = fit(S.merge);
  S.draw = S.draw === (1 << old) - 1 ? allLevels() : S.draw & allLevels();
  injections = []; focus = null;
  buildStrips(); buildStack();
}

// ── draw ──
function changed() {
  syncControls();
  for (const s of strips) s.sync();
  frustumButtons.forEach((b, k) => b.classList.toggle('on', !!(S.frustums & (1 << k))));
  if (solveKey() !== solvedKey) compute();
  draw();
}
let pending = false;
function draw() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}

/** the largest value a cascade's lines show, for the stack's own exposure */
function levelTop(k: number) {
  if (!sol) return 1;
  const L = sol.lines;
  let m = 0;
  for (let i = 0; i < sol.count; i++) {
    const o = i * STRIDE;
    if (L[o + 8] !== k || !((S.frustums >> L[o + 9]) & 1)) continue;
    const u = S.extValue === 0 ? L[o + 10] : S.extValue === 1 ? L[o + 11] : L[o + 6];
    const d = S.mergeValue === 0 ? L[o + 4] : S.mergeValue === 1 ? L[o + 5] : Math.max(L[o + 4], L[o + 5]);
    m = Math.max(m, Math.abs((S.drawExt ? u : 0) + (S.drawMerge ? d : 0)));
  }
  return m > 0 ? m : 1;
}

function render() {
  if (!sol) return;
  const shown = bits(S.draw);
  const i = focus && 'line' in focus ? indexOf(focus.line) : -1;
  const selPix = focus && 'pixel' in focus ? focus.pixel : null;
  const base = { N: S.N, view: VIEW(), frustumMask: S.frustums, extMode: S.extValue, mergeMode: S.mergeValue, selPix };
  // the focused line, and per cascade its relatives in the tree of halves
  const segOf = (j: number) => Array.from(sol!.lines.subarray(j * STRIDE, j * STRIDE + 4));
  const selected = i >= 0 ? [...segOf(i), ...SELECTED, 3] : [];
  const perLevel: number[][] = Array.from({ length: K() }, () => []);
  if (focus && 'line' in focus) {
    const f = focus.line;
    if (i >= 0) perLevel[f.k].push(...selected);
    for (const r of relatives(S.N, f)) {
      const j = indexOf({ ...r.id, frustum: f.frustum });
      if (j < 0) continue;
      const [cr, cg, cb] = r.up ? ANCESTOR : DESCENDANT;
      perLevel[r.id.k].push(...segOf(j), cr, cg, cb, 0.3 + 0.7 * Math.min(1, r.w), 2);
    }
  }

  main.resize();
  const [W, H] = main.size;
  main.pass({
    ...base, vp: [0, 0, W, H],
    extMask: S.drawExt ? S.draw : 0, mergeMask: S.drawMerge ? S.draw : 0, latticeMask: 0,
    profile: S.profile, width: S.width, alongW: S.alongW, alongV: S.alongV,
    gain: S.average && shown ? 1 / shown : 1,
    mode: S.show, tone: S.tone, top: autoTop * 2 ** S.exposure * (S.average || S.show !== 0 ? 1 : Math.max(1, shown)), grid: S.grid ? 1 : 0,
    coverageNorm: 2 * Math.PI * (bits(S.frustums) / 4) * (S.average ? 1 : Math.max(1, shown)),
    overlay: selected,
  });

  // the stack: each cascade alone as hairlines at its own exposure over its own lines, then the pixels
  const dpr = stack.resize(), cr = stack.canvas.getBoundingClientRect(), Hs = stack.size[1];
  const dim = focus && 'line' in focus ? 0.3 : 1;
  for (const t of tiles) {
    const r = t.sq.getBoundingClientRect();
    const vp: Pass['vp'] = [Math.round((r.left - cr.left) * dpr), Math.round(Hs - (r.bottom - cr.top) * dpr), Math.round(r.width * dpr), Math.round(r.height * dpr)];
    const isPixels = t.k < 0, bit = isPixels ? 0 : 1 << t.k, drawn = !!(S.draw & bit);
    t.el.classList.toggle('on', drawn || isPixels);
    t.el.classList.toggle('focus', (!!focus && 'line' in focus && focus.line.k === t.k) || (isPixels && !!selPix));
    stack.pass({
      ...base, vp,
      extMask: S.drawExt ? bit : 0, mergeMask: S.drawMerge ? bit : 0, latticeMask: bit,
      profile: 3, width: 1, alongW: 0, alongV: 0,
      gain: (drawn ? 1 : 0.4) * dim,
      mode: isPixels ? 1 : 0, tone: 0, top: isPixels ? autoTop * 2 ** S.exposure : levelTop(t.k),
      grid: isPixels ? (S.grid ? 1 : 0) : 2, coverageNorm: 1,
      overlay: isPixels ? [] : perLevel[t.k],
    });
  }
  showFocus(i);
}

new ResizeObserver(draw).observe(canvas);
new ResizeObserver(draw).observe(grid);
setLight(1);
buildStrips();
buildStack();
changed();
