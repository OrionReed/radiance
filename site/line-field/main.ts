import { $, el, createControls, writeControls, readControls } from '../_shared/ui';
import { boxesOf, exactAt, exactOccludedAt, relatives, solve, STRIDE, type Box, type Injection, type Rule, type Solution } from './lines';

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
  /** the weight of guard rays (steeper than 45°, cone weight 0): 0 hides them */
  guardGain: number;
  /** the fluence above which a pixel in full shadow (reference 0) counts as leak */
  leak: number;
  mode: number; // 0 lines, 1 pixels, 2 exact, 3 lines ÷ exact, 4 pixels ÷ exact, 5 coverage
  tone: number; top: number; grid: number; coverageNorm: number; latticeMask: number;
  /** 0: the continuous field; 1: each image pixel's mean over RASTER² samples; 2: its centre */
  raster: number;
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
layout(location = 5) in float guard; // 1: a guard ray, steeper than 45°
uniform vec4 view;                  // x0, y0, span, world units per device pixel
uniform int extMask, mergeMask, latticeMask, frustumMask, profile;
uniform float width, guardGain;     // guard rays' weight: 0 hides them
out float vT, vPerp;
flat out float vCos, vGain;
flat out vec4 vVals;
flat out vec2 vRT;
flat out int vExt, vMerge, vLat;
void main() {
  int lv = int(meta.x + 0.5), fr = int(meta.y + 0.5);
  int ext = (extMask >> lv) & 1, merge = (mergeMask >> lv) & 1, lat = (latticeMask >> lv) & 1;
  vGain = guard > 0.5 ? guardGain : 1.0;
  if (ext + merge + lat == 0 || ((frustumMask >> fr) & 1) == 0 || vGain == 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
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
flat in float vCos, vGain;
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
  float val = vGain * (float(vExt) * extV + float(vMerge) * mergeV);
  float lattice = 0.6 * vGain * float(vLat) * max(0.0, 1.0 - abs(vPerp) / view.w);
  o = vec4(gain * weight * prof * val, gain * weight * prof * vVals.w, lattice, 1.0);
}`;

const DISPLAY_VS = /* glsl */ `#version 300 es
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const DISPLAY_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D accum, pix, exact, raster;
uniform int rasterMode, S;
uniform vec4 view, vp;
uniform float N, top, coverageNorm, leak;
uniform int mode, tone, grid;
uniform vec2 selPix; // the focused pixel (x < 0: none)
out vec4 o;
vec3 ratioColour(float r) {
  if (!(r > 0.0)) return vec3(0.0);
  float s = clamp(log(r) / log(1.25), -1.0, 1.0);
  vec3 base = vec3(0.13);
  return s > 0.0 ? mix(base, vec3(1.0, 0.3, 0.2), s) : mix(base, vec3(0.25, 0.5, 1.0), -s);
}
// light reaching a pixel the reference has in full shadow: green, brighter with more of it
vec3 leakColour(float v) {
  float s = clamp(log(v / leak) / log(100.0), 0.0, 1.0);
  return mix(vec3(0.1, 0.35, 0.15), vec3(0.35, 1.0, 0.45), s);
}
// the error views: masked inside solids (reference < 0), leak or dark in full shadow (reference 0), else the ratio
vec3 errorColour(float v, float ex) {
  if (ex < 0.0) return vec3(0.0);
  if (ex == 0.0) return v > leak ? leakColour(v) : vec3(0.0);
  return ratioColour(v / ex);
}
// the reference, bilinear over the texels outside the solids (whose texels hold -1); -1 where all four are inside
float exactAt(vec2 w) {
  ivec2 sz = textureSize(exact, 0);
  vec2 t = w / N * vec2(sz) - 0.5, f = fract(t);
  ivec2 i0 = ivec2(floor(t));
  float sum = 0.0, wsum = 0.0;
  for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
    float v = texelFetch(exact, clamp(i0 + ivec2(i, j), ivec2(0), sz - 1), 0).r;
    float k = (i == 1 ? f.x : 1.0 - f.x) * (j == 1 ? f.y : 1.0 - f.y);
    if (v >= 0.0) { sum += k * v; wsum += k; }
  }
  return wsum > 0.0 ? sum / wsum : -1.0;
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
  if (rasterMode != 0) {
    // the field rendered at S samples per image pixel (texture row 0 is the image's bottom), reduced to one value
    ivec2 p0 = ivec2(floor(w)) * S;
    int top = int(N) * S - 1;
    vec3 sum = vec3(0.0);
    if (rasterMode == 1) {
      for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) if (i < S && j < S) sum += texelFetch(raster, ivec2(p0.x + i, top - p0.y - j), 0).rgb;
      acc = sum / float(S * S);
    } else {
      int h = S / 2;
      for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) sum += texelFetch(raster, ivec2(p0.x + h - 1 + i, top - p0.y - h + 1 - j), 0).rgb;
      acc = sum / 4.0;
    }
  }
  vec2 px = texelFetch(pix, ivec2(floor(w)), 0).rg;
  float ex = exactAt(w);
  vec3 c;
  if (mode == 0) c = toneMap(acc.r);
  else if (mode == 1) c = toneMap(px.r);
  else if (mode == 2) c = toneMap(ex);
  else if (mode == 3) c = errorColour(acc.r, ex);
  else if (mode == 4) c = errorColour(px.r, ex);
  else c = ratioColour(acc.g / coverageNorm);
  vec3 structure = vec3(0.3, 0.48, 0.85); // the pixel grid, and a cascade's own lines in the stack
  if (grid == 1) {
    vec2 f = abs(w - round(w)) / view.w;
    float g = 1.0 - clamp(min(f.x, f.y) - 0.5, 0.0, 1.0);
    float k = clamp((1.0 / view.w - 4.0) / 8.0, 0.0, 1.0); // fades out when pixels get small on screen
    c = mix(c, structure, 0.35 * g * k);
  }
  if (grid == 2) c += structure * 0.3 * min(acc.b, 1.0);
  if (px.g > 1.5) c = vec3(0.42, 0.45, 0.52); // a wall
  else if (px.g > 0.5) c = vec3(1.0, 0.85, 0.4); // a light
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
  rasterTex: WebGLTexture; rasterFbo: WebGLFramebuffer; rasterSize = 0;
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
    for (const [loc, n, off] of [[1, 4, 0], [2, 4, 4], [3, 2, 8], [4, 2, 10], [5, 1, 15]]) {
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
    this.rasterTex = this.texture(gl.NEAREST);
    this.rasterFbo = gl.createFramebuffer()!;
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

  upload(sol: Solution, pix: Float32Array) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instances);
    gl.bufferData(gl.ARRAY_BUFFER, sol.lines.subarray(0, sol.count * STRIDE), gl.DYNAMIC_DRAW);
    this.count = sol.count;
    gl.bindTexture(gl.TEXTURE_2D, this.pix);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, sol.N, sol.N, 0, gl.RG, gl.FLOAT, pix);
  }
  uploadExact(exact: Float32Array, M: number) {
    const gl = this.gl;
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
    const needsLines = p.mode === 0 || p.mode === 3 || p.mode === 5;
    if (needsLines) this.drawLines(p, perPx);
    if (needsLines && p.raster) {
      // the same field over the whole image at RASTER samples per image pixel
      const size = p.N * RASTER;
      if (size !== this.rasterSize) {
        this.rasterSize = size;
        gl.bindTexture(gl.TEXTURE_2D, this.rasterTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, this.floatBlend ? gl.RGBA32F : gl.RGBA16F, size, size, 0, gl.RGBA, gl.FLOAT, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.rasterFbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.rasterTex, 0);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.rasterFbo);
      gl.disable(gl.SCISSOR_TEST);
      gl.viewport(0, 0, size, size);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      this.drawLines({ ...p, view: { x: 0, y: 0, span: p.N } }, 1 / RASTER);
      gl.enable(gl.SCISSOR_TEST);
      gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const { p: prog, u } = this.display;
    gl.useProgram(prog);
    [this.accum, this.pix, this.exact, this.rasterTex].forEach((t, i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); });
    gl.uniform1i(u('accum'), 0); gl.uniform1i(u('pix'), 1); gl.uniform1i(u('exact'), 2); gl.uniform1i(u('raster'), 3);
    gl.uniform1i(u('rasterMode'), needsLines ? p.raster : 0); gl.uniform1i(u('S'), RASTER);
    this.display_(p, x, y, w, h, perPx);
    gl.disable(gl.SCISSOR_TEST);
  }

  drawLines(p: Pass, perPx: number) {
    const gl = this.gl;
    {
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      const { p: prog, u } = this.lines;
      gl.useProgram(prog);
      gl.uniform4f(u('view'), p.view.x, p.view.y, p.view.span, Math.abs(perPx));
      for (const k of ['extMask', 'mergeMask', 'latticeMask', 'frustumMask', 'profile', 'extMode', 'mergeMode', 'alongW', 'alongV'] as const) gl.uniform1i(u(k), p[k]);
      gl.uniform1f(u('width'), p.width); gl.uniform1f(u('gain'), p.gain); gl.uniform1f(u('guardGain'), p.guardGain);
      gl.bindVertexArray(this.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
  }

  /** the display pass (its program bound, textures set) and the overlay */
  display_(p: Pass, x: number, y: number, w: number, h: number, perPx: number) {
    const gl = this.gl, u = this.display.u;
    gl.uniform4f(u('view'), p.view.x, p.view.y, p.view.span, perPx);
    gl.uniform4f(u('vp'), x, y, w, h);
    gl.uniform1f(u('N'), p.N); gl.uniform1f(u('top'), p.top); gl.uniform1f(u('coverageNorm'), p.coverageNorm); gl.uniform1f(u('leak'), p.leak);
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
  }
}

/** samples per image pixel, each way, when the field is rasterised */
const RASTER = 8;
const SELECTED = [1.0, 0.31, 0.85, 1.0];
const DESCENDANT = [1.0, 0.7, 0.28];
const ANCESTOR = [0.66, 0.55, 1.0];

// ── state ──

type Range = [number, number] | null;
const S = {
  N: 64,
  /** move the lights, paint lights or walls, or erase either */
  tool: 'move' as 'move' | 'paint' | 'wall' | 'erase',
  /** the reference counts every direction, or only those of the frustums shown */
  refFrustums: false,
  /** the cascades each pass computes */
  ext: [0, 5] as Range,
  merge: [0, 5] as Range,
  /** the cascades drawn, and which pass's values they show */
  draw: 0b1,
  drawExt: false, drawMerge: true,
  extValue: 0, mergeValue: 0,
  frustums: 0b0001,
  profile: 0, width: 1, alongW: 0, alongV: 0,
  raster: 0,
  average: false,
  show: 0, tone: 0, grid: false, exposure: 0,
  /** the split and merge rule (defaults: plain) */
  split: 'plain' as Rule['split'], mergeLinear: false, phase: false, guard: false,
};
const rule = (): Rule => ({ split: S.split, mergeLinear: S.mergeLinear, phase: S.phase, guard: S.guard });
/** guard rays in the stack: drawn faintly, at this weight; the main view leaves them out */
const GUARD_GAIN = 0.35;
const K = () => Math.log2(S.N);
const VIEW = () => ({ x: 0, y: 0, span: S.N });
let lights: [number, number][] = [];
/** opaque pixels that emit nothing */
let walls: [number, number][] = [];
let injections: Injection[] = [];
type LineId = { frustum: number; k: number; plane: number; row: number; d: number };
let focus: { line: LineId } | { pixel: [number, number] } | null = null;
let sol: Solution | null = null, autoTop = 1;

const setLight = (size: number) => {
  lights = [];
  const cx = 0, cy = Math.floor(S.N / 2) - Math.floor(size / 2);
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) lights.push([cx + i, cy + j]);
};
/** wall presets beside the light (its bounding box), a column N/8 to its right: an edge over the rows below the light's
 * middle, a full column with a gap on the light's middle row(s), or a square N/16 across on its middle */
const setWalls = (kind: 'none' | 'edge' | 'slit' | 'block') => {
  const N = S.N, ls = lights.length ? lights : [[0, N / 2] as [number, number]];
  const x1 = Math.max(...ls.map((l) => l[0])) + 1, cy = (Math.min(...ls.map((l) => l[1])) + Math.max(...ls.map((l) => l[1])) + 1) / 2;
  const wx = Math.min(N - 1, x1 + Math.max(1, N / 8) - 1), s = Math.max(2, N / 16), out: [number, number][] = [];
  if (kind === 'edge') for (let y = Math.ceil(cy); y < N; y++) out.push([wx, y]);
  if (kind === 'slit') for (let y = 0; y < N; y++) if (Math.abs(y + 0.5 - cy) >= 1) out.push([wx, y]);
  if (kind === 'block') for (let i = 0; i < s; i++) for (let j = 0; j < s; j++) out.push([wx + i, Math.round(cy - s / 2) + j]);
  const lit = new Set(lights.map((l) => l.join()));
  walls = out.filter(([x, y]) => x < N && y >= 0 && y < N && !lit.has(`${x},${y}`));
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
    rule: rule(),
  };
};
const solveKey = () => JSON.stringify([S.N, lights, walls, injections, passOptions()]);
function compute() {
  const N = S.N;
  solvedKey = solveKey();
  const emission = new Float32Array(N * N), solid = new Uint8Array(N * N);
  for (const [x, y] of walls) solid[x * N + y] = 2;
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
    for (let i = 0; i < sol.count; i++) if (!sol.lines[i * STRIDE + 15]) for (const o of [4, 5, 10]) autoTop = Math.max(autoTop, Math.abs(sol.lines[i * STRIDE + o]));
    if (!(autoTop > 0)) autoTop = 1;
  }

  indexLines();
  main.upload(sol, pix);
  stack.upload(sol, pix);
}

// ── the reference: exact, on a sub-grid of the image; with walls (or only the shown frustums' directions) it is
// computed coarse at once and then refined in slices, so painting and dragging stay live ──
let referencedKey = '', refBoxes: Box[] = [], refJob = 0;
const refMask = () => (S.refFrustums ? S.frustums : 15);
const refKey = () => JSON.stringify([S.N, lights, walls, refMask()]);
/** the exact fluence at a point of the image (NaN inside a light or wall) */
const referenceAt = (x: number, y: number) =>
  !lights.length ? 0 : !walls.length && refMask() === 15 ? exactAt(x, y, lights) : exactOccludedAt(x, y, refBoxes, refMask());
function startReference() {
  const N = S.N, job = ++refJob;
  referencedKey = refKey();
  refBoxes = boxesOf(N, lights, walls);
  const sub = Math.max(2, Math.floor((lights.length > 16 || refBoxes.length > 16 ? 256 : 512) / N));
  /** rows [j0, j1) of the reference at sub samples per pixel */
  const rows = (out: Float32Array, s: number, j0: number, j1: number) => {
    const M = N * s;
    for (let j = j0; j < j1; j++)
      for (let i = 0; i < M; i++) {
        const e = referenceAt((i + 0.5) / s, (j + 0.5) / s);
        out[j * M + i] = Number.isNaN(e) ? -1 : e;
      }
  };
  const upload = (out: Float32Array, s: number) => { main.uploadExact(out, N * s); stack.uploadExact(out, N * s); draw(); };
  const quick = !walls.length && refMask() === 15;
  const first = quick ? sub : 2, coarse = new Float32Array((N * first) ** 2);
  rows(coarse, first, 0, N * first);
  upload(coarse, first);
  if (first === sub) return;
  const fine = new Float32Array((N * sub) ** 2);
  let j = 0;
  const step = () => {
    if (job !== refJob) return;
    const t0 = performance.now();
    while (j < N * sub && performance.now() - t0 < 30) { rows(fine, sub, j, Math.min(N * sub, j + 8)); j += 8; }
    if (j < N * sub) setTimeout(step, 0);
    else upload(fine, sub);
  };
  setTimeout(step, 60);
}

// ── controls ──
const { controls, segmented, slider, toggle, pill, syncControls } = createControls(() => changed());
/** a binding of S[name], saved in the URL as `url` when it differs from its value now (the default) */
const field = (name: keyof typeof S, url: string = name) => ({ get: () => S[name], set: (v: unknown) => { (S as Record<string, unknown>)[name] = v; }, url: [url, S[name]] as [string, unknown] });
const FRUSTUMS = ['+x', '−x', '+y', '−y'];
const fieldWrap = (label: string, body: HTMLElement, title = '') =>
  el('div', { className: 'field', title }, el('div', { className: 'label' }, el('span', { textContent: label })), body);

segmented($('scene'), { label: 'resolution', options: [16, 32, 64, 128].map((n) => [n, String(n)]), get: () => S.N, set: setN });
segmented($('scene'), {
  label: 'tool',
  options: [['move', 'move', 'drag the lights'], ['paint', 'light', 'paint light pixels'], ['wall', 'wall', 'paint wall pixels: opaque, emitting nothing'], ['erase', 'erase', 'erase light and wall pixels']],
  ...field('tool'),
});
segmented($('scene'), { label: 'light', options: [[1, '1 px'], [2, '2×2'], [3, '3×3'], [0, 'none']], get: () => -1, set: (n: number) => { setLight(n); const lit = new Set(lights.map((l) => l.join())); walls = walls.filter((w) => !lit.has(w.join())); } });
segmented($('scene'), {
  label: 'walls',
  options: [['none', 'none'], ['edge', 'edge', 'a wall column beside the light over the rows below its middle'], ['slit', 'slit', 'a wall column with a gap on the light\'s middle row(s)'], ['block', 'block', 'a small square in front of the light']],
  get: () => -1, set: setWalls,
});

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
segmented($('passes'), { label: 'values drawn', options: [['ext', 'extension'], ['merge', 'merge'], ['both', 'both']], get: () => (S.drawExt && S.drawMerge ? 'both' : S.drawExt ? 'ext' : 'merge'), set: (v: string) => { S.drawExt = v !== 'merge'; S.drawMerge = v !== 'ext'; }, url: ['values', 'merge'] });
segmented($('passes'), {
  label: 'extension value',
  options: [[0, 'R', 'radiance along the ray'], [1, 'T', 'transmittance along the ray'], [2, 'c·R', 'R times the cone weight c']],
  ...field('extValue', 'extv'),
});
segmented($('passes'), {
  label: 'merge value',
  options: [[0, 'M', 'merged radiance: the light reaching the ray\'s start from behind it, in its cone'], [1, 'M at end', 'T·M + c·R: the light leaving the ray\'s end'], [2, 'M along', 'from M at the start to M at end, along the ray']],
  ...field('mergeValue', 'mergev'),
});
const frustumSeg = el('div', { className: 'seg' });
const frustumButtons = FRUSTUMS.map((name, k) => {
  const b = el('button', { textContent: name, title: `the frustum carrying light toward ${name}` });
  b.onclick = () => { S.frustums ^= 1 << k; changed(); };
  frustumSeg.append(b);
  return b;
});
$('passes').append(fieldWrap('frustums', frustumSeg));
segmented($('passes'), { label: 'cascades drawn', options: [[false, 'sum'], [true, 'average']], ...field('average', 'avg') });

segmented($('rule'), {
  label: 'split law',
  options: [
    ['plain', 'plain', 'an even offset 2a splits into (a, a); an odd one 2a + 1 into (a, a + 1) and (a + 1, a), 1/2 each'],
    ['jitter', 'jitter', 'from length 8, an even parent also gives 1/8 each to (a − 1, a + 1) and (a + 1, a − 1), keeping 3/4 on (a, a)'],
    ['jog', 'jog', 'an odd parent splits into straight halves (a, a) and (a + 1, a + 1), the second half started one row off'],
  ],
  ...field('split'),
});
pill($('rule'), { caption: 'merge split', label: 'linear merge rule', title: 'merge only: an odd parent gives its lower pair M (1/2 + δ) and its upper pair M (1/2 − δ), δ = −(M(u+1) − M(u−1)) / 24M', ...field('mergeLinear', 'mlin') });
pill($('rule'), { caption: 'tile phase', label: 'half row from L ≥ 8', title: 'tiles of length 8 and up sit half a row up: windows [j + 1/2, j + 3/2] at both planes', ...field('phase') });
pill($('rule'), {
  caption: 'guard band', label: 'one offset past 45°',
  title: 'every level also stores offsets ±(L + 1): rays steeper than 45° with cone weight 0, which carry light but never deliver it; jitter then falls back to plain only where a partner leaves this range, so the diagonals are jittered too. Plain is unchanged. The stack draws guard rays faintly; the main view leaves them out',
  ...field('guard'),
});

segmented($('lines'), { label: 'profile across', title: 'in rows: lines with one row offset sit one row apart', options: [[0, 'none'], [1, 'tent'], [2, 'gauss']], ...field('profile') });
slider($('lines'), { label: 'width', min: 0.05, max: 4, step: 0.05, ...field('width'), format: (v: number) => `${v.toFixed(2)} rows` });
segmented($('lines'), { label: 'width along', options: [[0, 'even'], [1, 'diamond'], [2, 'opens'], [3, 'closes']], ...field('alongW', 'alongw') });
segmented($('lines'), { label: 'weight along', options: [[0, 'even'], [1, 'tent']], ...field('alongV', 'alongv') });
segmented($('raster'), {
  label: 'to pixels', title: `the line field drawn at ${RASTER}×${RASTER} samples per image pixel, then one value per pixel`,
  options: [[0, 'off', 'the continuous field'], [1, 'box', 'the mean over the pixel'], [2, 'centre', 'the value at the pixel centre']],
  ...field('raster'),
});

segmented($('viewc'), {
  label: 'show', cols: 3,
  options: [[0, 'lines'], [1, 'pixels'], [2, 'reference'], [3, 'lines ÷ ref'], [4, 'pixels ÷ ref'], [5, 'coverage', 'Σ profile × cone weight ÷ 2π: 1 where the lines tile']],
  ...field('show'),
});
segmented($('viewc'), {
  label: 'reference directions',
  options: [[false, 'all', 'the whole circle of directions'], [true, 'frustums shown', 'only the directions of the frustums shown: a quarter turn each']],
  ...field('refFrustums', 'reffr'),
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
    const [x, y] = focus.pixel, v = sol.fluence[x * S.N + y], e = referenceAt(x + 0.5, y + 0.5);
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
  const wall = new Set(walls.map((w) => w.join()));
  if (moved.every(([x, y]) => x >= 0 && y >= 0 && x < S.N && y < S.N && !wall.has(`${x},${y}`)) && moved.some(([x, y], i) => x !== lights[i][0] || y !== lights[i][1])) {
    lights = moved; changed();
  }
});
addEventListener('pointerup', () => (drag = null));
function paintAt(wx: number, wy: number) {
  const x = Math.floor(wx), y = Math.floor(wy);
  if (x < 0 || y < 0 || x >= S.N || y >= S.N) return;
  const i = lights.findIndex(([a, b]) => a === x && b === y), w = walls.findIndex(([a, b]) => a === x && b === y);
  // a light painted over a wall replaces it, and a wall over a light
  if (S.tool === 'paint' && i < 0) { if (w >= 0) walls.splice(w, 1); lights.push([x, y]); changed(); }
  if (S.tool === 'wall' && w < 0) { if (i >= 0) lights.splice(i, 1); walls.push([x, y]); changed(); }
  if (S.tool === 'erase' && (i >= 0 || w >= 0)) { if (i >= 0) lights.splice(i, 1); if (w >= 0) walls.splice(w, 1); changed(); }
}
addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'INPUT') return;
  if (e.key === '[' || e.key === ']') { S.exposure = Math.max(-6, Math.min(6, S.exposure + (e.key === ']' ? 0.25 : -0.25))); changed(); }
});

function setN(n: number) {
  const old = K(), scale = n / S.N;
  S.N = n;
  const k = K(), fit = (r: Range): Range => r && [Math.min(r[0], k - 1), r[1] === old - 1 ? k - 1 : Math.min(r[1], k - 1)];
  const rescale = (ps: [number, number][]) => [...new Map(ps.map(([x, y]) => [Math.floor(x * scale), Math.floor(y * scale)] as [number, number]).map((l) => [l.join(), l])).values()];
  lights = rescale(lights);
  const lit = new Set(lights.map((l) => l.join()));
  walls = rescale(walls).filter((w) => !lit.has(w.join()));
  S.ext = fit(S.ext); S.merge = fit(S.merge);
  S.draw = S.draw === (1 << old) - 1 ? allLevels() : S.draw & allLevels();
  injections = []; focus = null;
  buildStrips(); buildStack();
}

// ── URL: every setting round-trips; params at their default are left out ──
// ext=0-5 | none, draw=0,2 | all | none, frustums=px,ny, lights=x.y,x.y | none, walls=x.y,x.y, lit=R_px_c3_p5_r2_d-1_v1,…
const SIZES = [16, 32, 64, 128];
const FR_URL = ['px', 'nx', 'py', 'ny'];
const isInt = (v: number, lo: number, hi: number) => Number.isInteger(v) && v >= lo && v <= hi;
const rangeText = (r: Range) => (r ? `${r[0]}-${r[1]}` : 'none');
const levelsText = (m: number) => (m === allLevels() ? 'all' : m ? [...Array(K()).keys()].filter((k) => (m >> k) & 1).join(',') : 'none');
const litText = (j: Injection) => `${j.pass}_${FR_URL[j.frustum]}_c${j.k}_p${j.plane}_r${j.row}_d${j.d}_v${+j.value.toFixed(4)}`;
function parseRange(s: string): Range | undefined {
  if (s === 'none') return null;
  if (s === 'all') return [0, K() - 1];
  const [a, b = a] = s.split('-').map(Number);
  return isInt(a, 0, K() - 1) && isInt(b, a, K() - 1) ? [a, b] : undefined;
}
function parseLevels(s: string): number | undefined {
  if (s === 'all') return allLevels();
  if (s === 'none') return 0;
  const ks = s.split(',').map(Number);
  return ks.every((k) => isInt(k, 0, K() - 1)) ? ks.reduce((m, k) => m | (1 << k), 0) : undefined;
}
function parseLit(s: string): Injection | undefined {
  const m = /^([RM])_(px|nx|py|ny)_c(\d+)_p(\d+)_r(\d+)_d(-?\d+)_v(-?[\d.e+-]+)$/.exec(s);
  if (!m) return;
  const [k, plane, row, d, value] = m.slice(3).map(Number);
  if (!isInt(k, 0, K() - 1) || !isInt(plane, 0, (S.N >> k) - 1) || !isInt(row, 0, 3 * S.N - 1) || !isInt(d, -(1 << k) - (S.guard ? 1 : 0), (1 << k) + (S.guard ? 1 : 0)) || !Number.isFinite(value)) return;
  return { pass: m[1] as 'R' | 'M', frustum: FR_URL.indexOf(m[2]), k, plane, row, d, value };
}
const defaultLights = () => { const keep = lights; setLight(1); const d = lights; lights = keep; return d; };

let urlTimer = 0;
function writeURL() {
  const p = new URLSearchParams();
  if (S.N !== 64) p.set('N', String(S.N));
  writeControls(p, controls);
  const all: Range = [0, K() - 1];
  if (rangeText(S.ext) !== rangeText(all)) p.set('ext', rangeText(S.ext));
  if (rangeText(S.merge) !== rangeText(all)) p.set('merge', rangeText(S.merge));
  if (S.draw !== 0b1) p.set('draw', levelsText(S.draw));
  if (S.frustums !== 0b0001) p.set('frustums', S.frustums ? FR_URL.filter((_, v) => (S.frustums >> v) & 1).join(',') : 'none');
  const lightsText = (ls: [number, number][]) => (ls.length ? ls.map((l) => l.join('.')).join(',') : 'none');
  if (lightsText(lights) !== lightsText(defaultLights())) p.set('lights', lightsText(lights));
  if (walls.length) p.set('walls', lightsText(walls));
  if (injections.length) p.set('lit', injections.map(litText).join(','));
  const qs = p.toString().replace(/%2C/g, ','); // commas are legal in a query; keep the lists legible
  history.replaceState(null, '', qs ? '?' + qs : location.pathname);
}
/** apply the URL: N first, since it sets the cascades the rest refer to; anything invalid keeps its default */
function readURL() {
  const p = new URLSearchParams(location.search);
  const n = Number(p.get('N'));
  if (SIZES.includes(n) && n !== S.N) { setN(n); setLight(1); }
  readControls(p, controls);
  for (const { spec } of controls) {
    if (!spec.url) continue;
    const v = spec.get(), def = spec.url[1];
    if (spec.options ? !spec.options.some((o) => o[0] === v) : typeof v === 'number' && !Number.isFinite(v)) spec.set(def);
    else if (spec.min !== undefined && spec.max !== undefined) spec.set(Math.max(spec.min, Math.min(spec.max, v)));
  }
  const get = <T>(name: string, parse: (s: string) => T | undefined) => (p.has(name) ? parse(p.get(name)!) : undefined);
  const ext = get('ext', parseRange), merge = get('merge', parseRange), draw = get('draw', parseLevels);
  if (ext !== undefined) S.ext = ext;
  if (merge !== undefined) S.merge = merge;
  if (draw !== undefined) S.draw = draw;
  const fr = get('frustums', (s) => (s === 'none' ? 0 : s.split(',').every((t) => FR_URL.includes(t)) ? s.split(',').reduce((m, t) => m | (1 << FR_URL.indexOf(t)), 0) : undefined));
  if (fr !== undefined) S.frustums = fr;
  const pixels = (s: string) => {
    if (s === 'none') return [];
    const out = s.split(',').map((t) => t.split('.').map(Number) as [number, number]);
    return out.every((l) => l.length === 2 && l.every((c) => isInt(c, 0, S.N - 1))) ? [...new Map(out.map((l) => [l.join(), l])).values()] : undefined;
  };
  const ls = get('lights', pixels), ws = get('walls', pixels);
  if (ls) lights = ls;
  if (ws) { const lit = new Set(lights.map((l) => l.join())); walls = ws.filter((w) => !lit.has(w.join())); }
  const lit = get('lit', (s) => s.split(',').map(parseLit).filter((j): j is Injection => !!j));
  if (lit) injections = lit;
}

// ── draw ──
function changed() {
  clearTimeout(urlTimer);
  urlTimer = window.setTimeout(writeURL, 100); // replaceState is rate-limited by browsers; a drag writes once it settles
  syncControls();
  for (const s of strips) s.sync();
  frustumButtons.forEach((b, k) => b.classList.toggle('on', !!(S.frustums & (1 << k))));
  if (solveKey() !== solvedKey) compute();
  if (refKey() !== referencedKey) startReference();
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
    if (L[o + 8] !== k || L[o + 15] || !((S.frustums >> L[o + 9]) & 1)) continue;
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
    for (const r of relatives(S.N, f, rule())) {
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
    gain: S.average && shown ? 1 / shown : 1, guardGain: 0,
    leak: 1e-3 * autoTop, mode: S.show, tone: S.tone, top: autoTop * 2 ** S.exposure * (S.average || S.show !== 0 ? 1 : Math.max(1, shown)), grid: S.grid ? 1 : 0,
    coverageNorm: 2 * Math.PI * (bits(S.frustums) / 4) * (S.average ? 1 : Math.max(1, shown)),
    raster: S.raster,
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
      gain: (drawn ? 1 : 0.4) * dim, guardGain: GUARD_GAIN,
      leak: 1e-3 * autoTop, mode: isPixels ? 1 : 0, tone: 0, top: isPixels ? autoTop * 2 ** S.exposure : levelTop(t.k),
      grid: isPixels ? (S.grid ? 1 : 0) : 2, coverageNorm: 1, raster: 0,
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
readURL();
changed();
