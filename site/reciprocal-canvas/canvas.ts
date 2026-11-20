// A window onto a larger world that moves. Reciprocal HRC solves a square domain of N cascade pixels around the view;
// the producer (producer.ts) brings in, every frame, the light arriving along each line that enters the domain from
// outside it: from the first box the line meets, or from the sky.

import { ReciprocalHRC } from '../reciprocal-hrc/reciprocal-hrc';
import { EXPOSURE } from '../_shared/display';
import { $, el, createControls, fieldFor, writeControls, readControls, replaceURL } from '../_shared/ui';
import { OutsideSweep, type Budget } from './producer';
import { WORLD, worldAt, SKIES, SUN, sunSky, RASTERS, placed, type Box, type Raster, type Domain } from './world';
import { rasterWgsl, referenceWgsl, drawWgsl } from './shaders';

const fail = (message: string) => (($('error').textContent = message), $('error').classList.add('open'), console.error(message));

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) throw fail('WebGPU is not available in this browser.');
const device = await adapter.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
});
device.addEventListener('uncapturederror', (e) => fail((e as GPUUncapturedErrorEvent).error.message));
const view = $('view') as HTMLCanvasElement;
const context = view.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque' });

/** rays per start point, with a quarter as many pieces; 0: no budget, the pool grown to fit */
const BUDGETS = [0, 64, 128, 256];
const budgetFor = (rays: number): Budget | null => (rays ? { rays, pieces: rays / 4 } : null);

/** How the lattice follows the camera:
 * - octaves: fixed to the world, its pixel a power of two in world units;
 * - cross-fade: octaves, mixed with the next coarser octave's solve by the zoom's place in the octave;
 * - continuous: its pixel follows the zoom, its origin steps with the world;
 * - screen: its pixel follows the zoom, its origin the camera. */
const ZOOMS = ['octaves', 'cross-fade', 'continuous', 'screen'] as const;
type Zoom = (typeof ZOOMS)[number];

const next = <T,>(list: readonly T[], value: T) => list[(list.indexOf(value) + 1) % list.length];

const settings = { N: 512, q: 8, wuSplit: true, raster: 'size' as Raster, budget: 128, sky: 'none', sun: { ...SUN }, zoom: 'octaves' as Zoom, smooth: false, paused: false, stops: 0, errorView: false };
const camera = { x: 300, y: 0, zoom: 1 }; // device px per world unit
let time = 0;

// ---------------------------------------------------------------- solving

interface Solver {
  N: number;
  q: number;
  budget: number;
  engine: ReciprocalHRC;
  sweep: OutsideSweep;
  raster: GPUComputePipeline;
  rasterGroup: GPUBindGroup;
  reference: GPUComputePipeline;
  referenceGroup: GPUBindGroup;
  referenceWorld: GPUBuffer;
  referenceShapes: GPUBuffer;
  exact: GPUBuffer;
}
/** the shown solve, and the cross-fade's coarser one */
const solvers: [Solver | null, Solver | null] = [null, null];
/** room for boxes and, without a budget, pieces */
const capacity = { boxes: 256, pieces: 0 };

const compute = (code: string, entryPoint: string) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint } });
function bindGroup(p: GPUComputePipeline | GPURenderPipeline, buffers: GPUBuffer[]) {
  return device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
}

function makeSolver(N: number): Solver {
  const { q, wuSplit, budget } = settings;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const sweep = new OutsideSweep(device, N, q, budgetFor(budget), capacity.boxes, capacity.pieces);
  if (!budget) capacity.pieces = Math.max(capacity.pieces, sweep.pieces);
  const engine = new ReciprocalHRC(device, N, { wuSplit, producer: sweep.producer });
  const referenceWorld = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const referenceShapes = device.createBuffer({ size: capacity.boxes * 32, usage });
  const exact = device.createBuffer({ size: N * N * 16, usage });
  const raster = compute(rasterWgsl(N), 'raster'), reference = compute(referenceWgsl(N), 'reference');
  return {
    N, q, budget, engine, sweep, raster, reference, referenceWorld, referenceShapes, exact,
    rasterGroup: bindGroup(raster, [sweep.shapes, sweep.world, engine.sceneBuffer]),
    referenceGroup: bindGroup(reference, [referenceShapes, referenceWorld, exact, sweep.sky]),
  };
}
/** the solver for slot 0 (N) or 1 (the cross-fade's, N / 2), made again when the settings or the room it needs change */
function solverNow(boxes = 0, slot: 0 | 1 = 0) {
  if (boxes > capacity.boxes) capacity.boxes = 2 ** Math.ceil(Math.log2(boxes));
  const s = solvers[slot], N = slot === 1 ? settings.N / 2 : settings.N;
  const sameSweep = s?.N === N && s.q === settings.q && s.budget === settings.budget;
  if (!sameSweep || s.engine.wuSplit !== settings.wuSplit || s.sweep.boxes < capacity.boxes || (!s.budget && s.sweep.pieces < capacity.pieces)) {
    if (s) [s.engine, s.sweep, s.exact, s.referenceWorld, s.referenceShapes].forEach((x) => x.destroy());
    if (!sameSweep && slot === 0) capacity.pieces = 0;
    solvers[slot] = makeSolver(N);
  }
  return solvers[slot]!;
}

/** The domain: its pixel, the view fitting N / 2 - 8 of them (in octaves, the next power of two in world units), and
 * its origin, a multiple of N / 2 pixels with the view at least 4 pixels in from each side (on the screen lattice, the
 * view centred). */
function domainFor(N: number): Domain {
  const across = Math.max(view.width, view.height) / camera.zoom;
  const fit = across / (N / 2 - 8);
  const pixel = settings.zoom === 'octaves' || settings.zoom === 'cross-fade' ? 2 ** Math.ceil(Math.log2(fit)) : fit;
  if (settings.zoom === 'screen') return { pixel, x: camera.x - (N / 2) * pixel, y: camera.y - (N / 2) * pixel };
  const step = (N / 2) * pixel;
  const origin = (c: number) => Math.floor((c - across / 2 - 4 * pixel) / step) * step;
  return { pixel, x: origin(camera.x), y: origin(camera.y) };
}
/** the cross-fade's weight on the coarser octave: 0 at the finer end of the octave, 1 where the next octave takes over */
function coarseWeight(N: number) {
  if (settings.zoom !== 'cross-fade') return 0;
  const octave = Math.log2(Math.max(view.width, view.height) / camera.zoom / (N / 2 - 8));
  return octave - Math.ceil(octave) + 1;
}

function encodeSolve(encoder: GPUCommandEncoder, s: Solver, boxes: Box[], domain: Domain) {
  s.sweep.setShapes(placed(boxes, domain, settings.raster), boxes.length);
  s.sweep.setSky(settings.sky === 'sun' ? sunSky(settings.sun) : SKIES[settings.sky]);
  if (settings.raster === 'lights' || settings.raster === 'coverage') device.queue.writeBuffer(s.sweep.world, 4, new Uint32Array([1]));
  const pass = encoder.beginComputePass();
  pass.setPipeline(s.raster);
  pass.setBindGroup(0, s.rasterGroup);
  const groups = Math.ceil((s.N * s.N) / 64);
  pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
  s.sweep.encode(pass);
  pass.end();
  s.sweep.encodeCounts(encoder);
  s.engine.encodeAll(encoder);
}

/** a frame's solves: the shown one and, cross-fading, the coarser octave's on N / 2 pixels over the same domain */
interface Solves {
  s: Solver;
  domain: Domain;
  coarse: { s: Solver; domain: Domain; weight: number } | null;
}
function encodeSolves(encoder: GPUCommandEncoder, boxes: Box[]): Solves {
  const s = solverNow(boxes.length), domain = domainFor(s.N), weight = coarseWeight(s.N);
  encodeSolve(encoder, s, boxes, domain);
  if (weight <= 0) return { s, domain, coarse: null };
  const coarse = { s: solverNow(boxes.length, 1), domain: { ...domain, pixel: 2 * domain.pixel }, weight };
  encodeSolve(encoder, coarse.s, boxes, coarse.domain);
  return { s, domain, coarse };
}

let sweepCounts = { pieces: 0, shared: 0, joined: 0, points: 1 };

/** solve this frame (and draw it); without a budget, again with more room while a sweep's pieces do not fit */
async function solveFrame(draw: boolean) {
  for (;;) {
    const boxes = worldAt(time), encoder = device.createCommandEncoder();
    const solves = encodeSolves(encoder, boxes);
    if (draw) encodeDraw(encoder, solves);
    device.queue.submit([encoder.finish()]);
    const all = solves.coarse ? [solves.s, solves.coarse.s] : [solves.s];
    const counts = await Promise.all(all.map((s) => s.sweep.counts()));
    sweepCounts = { ...counts[0], points: solves.s.sweep.points };
    if (solves.s.budget || all.every((s, i) => counts[i].pieces <= s.sweep.pieces)) return { ...solves, boxes };
    capacity.pieces = Math.ceil(Math.max(...counts.map((c) => c.pieces)) * 1.5);
  }
}

// ---------------------------------------------------------------- drawing

const drawModule = device.createShaderModule({ code: drawWgsl });
const drawPipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: { module: drawModule, entryPoint: 'vertexMain' },
  fragment: { module: drawModule, entryPoint: 'fragmentMain', targets: [{ format }] },
  primitive: { topology: 'triangle-list' },
});
const drawUniforms = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

function encodeDraw(encoder: GPUCommandEncoder, { s, domain, coarse }: Solves) {
  const c = coarse?.s ?? s;
  device.queue.writeBuffer(drawUniforms, 0, new Float32Array([
    view.width, view.height, camera.zoom, s.N,
    camera.x, camera.y, domain.x, domain.y,
    domain.pixel, EXPOSURE * 2 ** settings.stops, settings.errorView ? 1 : 0, 0.3,
    coarse?.domain.x ?? 0, coarse?.domain.y ?? 0, coarse?.domain.pixel ?? 1, coarse?.weight ?? 0,
    settings.smooth ? 1 : 0, c.N, 0, 0,
  ]));
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
  pass.setPipeline(drawPipeline);
  pass.setBindGroup(0, bindGroup(drawPipeline, [s.engine.outputBuffer, s.engine.sceneBuffer, s.exact, drawUniforms, c.engine.outputBuffer, c.engine.sceneBuffer]));
  pass.draw(3);
  pass.end();
}

// ---------------------------------------------------------------- frames, input, controls

let frameMs = 0, last = performance.now(), busy = false;
/** run f while the page's own frames wait */
async function alone<T>(f: () => Promise<T>) {
  while (busy) await new Promise((r) => setTimeout(r, 5));
  busy = true;
  try {
    return await f();
  } finally {
    busy = false;
  }
}

async function frame() {
  requestAnimationFrame(frame);
  if (busy) return;
  busy = true;
  const now = performance.now();
  if (!settings.paused && !settings.errorView) time += Math.min(0.1, (now - last) / 1000);
  last = now;
  if (settings.errorView) {
    const s = solverNow(), encoder = device.createCommandEncoder();
    encodeDraw(encoder, { s, domain: domainFor(s.N), coarse: null });
    device.queue.submit([encoder.finish()]);
  } else {
    await solveFrame(true);
    frameMs += (performance.now() - now - frameMs) * 0.1;
  }
  busy = false;
}
setInterval(() => {
  const s = solvers[0];
  if (!s) return;
  $('fps').textContent = `${s.N}² · ${frameMs.toFixed(1)} ms per frame`;
  const share = (n: number) => `${((100 * n) / sweepCounts.points).toFixed(1)} %`;
  $('sweep-counts').textContent = s.budget
    ? `${share(sweepCounts.shared)} of start points over the ray budget, ${share(sweepCounts.joined)} over the piece budget`
    : `${(sweepCounts.pieces / 1e6).toFixed(2)} M pieces`;
}, 250);

function layout() {
  const r = $('canvas-area').getBoundingClientRect(), side = Math.max(64, Math.floor(Math.min(r.width, r.height) - 48));
  Object.assign(view.style, { width: `${side}px`, height: `${side}px`, left: `${Math.floor((r.width - side) / 2)}px`, top: `${Math.floor((r.height - side) / 2)}px` });
  view.width = view.height = Math.round(side * devicePixelRatio);
}
window.addEventListener('resize', layout);
layout();

let drag: { x: number; y: number } | null = null;
view.addEventListener('pointerdown', (e) => ((drag = { x: e.clientX, y: e.clientY }), view.setPointerCapture(e.pointerId), view.classList.add('grabbing')));
view.addEventListener('pointermove', (e) => {
  if (!drag) return;
  camera.x -= ((e.clientX - drag.x) * devicePixelRatio) / camera.zoom;
  camera.y += ((e.clientY - drag.y) * devicePixelRatio) / camera.zoom;
  drag = { x: e.clientX, y: e.clientY };
  leaveErrorView();
});
view.addEventListener('pointerup', () => ((drag = null), view.classList.remove('grabbing')));
view.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = view.getBoundingClientRect();
  const fx = (e.clientX - r.left) * devicePixelRatio - view.width / 2, fy = view.height / 2 - (e.clientY - r.top) * devicePixelRatio;
  const before = { x: camera.x + fx / camera.zoom, y: camera.y + fy / camera.zoom };
  camera.zoom = Math.min(8, Math.max(1 / 32, camera.zoom * Math.exp(-e.deltaY * 0.0015)));
  camera.x = before.x - fx / camera.zoom;
  camera.y = before.y - fy / camera.zoom;
  leaveErrorView();
}, { passive: false });

const { controls, toggle, segmented, slider, syncControls } = createControls(() => changed());
const field = fieldFor(() => {});
function changed() {
  syncControls();
  const p = new URLSearchParams();
  writeControls(p, controls);
  replaceURL(p);
}
segmented($('domain'), { label: 'domain', title: 'cascade pixels across the domain; the view is half of it', options: [256, 512, 1024].map((n) => [n, String(n)]), ...field(settings, 'N'), url: ['N', 512] });
segmented($('domain'), {
  label: 'boxes on pixels', key: 'r', cols: 2, ...field(settings, 'raster'), url: ['raster', 'size'],
  options: [
    ['size', 'size', 'each box keeps its size in whole pixels (at least one) and sits on the nearest pixel edge'],
    ['edges', 'edges', 'each edge on the nearest pixel edge: a small light changes size as it moves'],
    ['lights', 'lights', 'lights by the share of each pixel they cover (a pixel lets the rest through); walls as in size'],
    ['coverage', 'coverage', 'every box by the share of each pixel it covers: walls thinner than a pixel let light through'],
  ],
});
segmented($('domain'), { label: 'start points per row', title: 'start points per row on the domain\'s back edges', options: [4, 8, 16].map((n) => [n, String(n)]), ...field(settings, 'q'), url: ['q', 8] });
segmented($('domain'), {
  label: 'zoom', key: 'z', cols: 2, ...field(settings, 'zoom'), url: ['zoom', 'octaves'],
  options: [
    ['octaves', 'octaves', 'the lattice fixed to the world, its pixel a power of two in world units: a zoom across an octave re-solves at the other pixel'],
    ['cross-fade', 'cross-fade', 'octaves, mixed with the next coarser octave\'s solve by the zoom\'s place in the octave: a second solve on N / 2 pixels'],
    ['continuous', 'continuous', 'the pixel follows the zoom, the origin steps with the world: a zoom moves the world across the lattice'],
    ['screen', 'screen', 'the pixel follows the zoom and the origin the camera: zooms and pans move the world across the lattice'],
  ],
});
const pair = el('div', { className: 'pair' });
$('domain').append(pair);
toggle(pair, { label: 'Wu split', key: 'w', title: 'the merge splits an odd parent\'s far light between its two bent orders by its slope across rows', ...field(settings, 'wuSplit'), url: ['wu', true] });
toggle(pair, { label: 'smooth display', key: 'f', title: 'each screen pixel takes the bilinear mean of the free cascade pixels around it', ...field(settings, 'smooth'), url: ['smooth', false] });
toggle(pair, { label: 'pause', key: 'space', title: 'stop the world\'s motion', ...field(settings, 'paused') });
segmented($('outside'), {
  label: 'sky', key: 'k', cols: 3, ...field(settings, 'sky'), url: ['sky', 'none'],
  options: [
    ['none', 'none', 'nothing beyond the boxes'],
    ['overcast', 'overcast', 'the same radiance from every direction'],
    ['sun', 'sun', 'a sun and a faint blue sky'],
    ['dusk', 'dusk', 'warm from the west, cooler to the east'],
    ['bands', 'bands', 'brighter from north and south than from east and west'],
  ],
});
const sunOn = () => settings.sky === 'sun';
slider($('outside'), { label: 'sun direction', title: 'where the sunlight comes from (0°: the east)', min: 0, max: 360, step: 1, ...field(settings.sun, 'angle'), format: (v) => `${v}°`, enabled: sunOn, url: ['sunAngle', SUN.angle] });
slider($('outside'), { label: 'sun size', title: 'the sun\'s angular size across: 0 casts sharp shadows, a larger sun wider penumbrae', min: 0, max: 10, step: 0.1, ...field(settings.sun, 'size'), format: (v) => `${v}°`, enabled: sunOn, url: ['sunSize', SUN.size] });
slider($('outside'), { label: 'sun intensity', title: 'the sun\'s irradiance', min: 0, max: 6, step: 0.05, ...field(settings.sun, 'intensity'), format: (v) => v.toFixed(2), enabled: sunOn, url: ['sunIntensity', SUN.intensity] });
slider($('outside'), { label: 'sun colour', title: 'cool (−1), white (0) or warm (1)', min: -1, max: 1, step: 0.05, ...field(settings.sun, 'warmth'), format: (v) => v.toFixed(2), enabled: sunOn, url: ['sunWarmth', SUN.warmth] });
segmented($('outside'), {
  label: 'budget per start point', key: 'b', ...field(settings, 'budget'), url: ['budget', 128],
  title: 'exact: every piece of angle, the memory grown to fit. A number: at most that many lines and a quarter as many pieces per start point, in fixed memory; a point over budget shares its lines by the light each light brings, or joins neighbouring pieces',
  options: BUDGETS.map((n) => [n, n ? String(n) : 'exact']),
});
slider($('output'), { label: 'exposure (stops)', min: -4, max: 4, step: 0.25, ...field(settings, 'stops'), format: (v) => (v > 0 ? `+${v}` : String(v)), url: ['stops', 0] });
toggle($('output'), {
  label: 'error against exact', key: 'e', title: 'trace the world exactly (a few seconds) and show this frame\'s error, (F − exact) / exact; the world pauses',
  get: () => settings.errorView, set: (on: boolean) => (on ? enterErrorView() : leaveErrorView()),
});

async function enterErrorView() {
  if (settings.errorView) return;
  $('measure').textContent = 'tracing the world exactly…';
  const result = await measure();
  settings.errorView = true;
  syncControls();
  $('measure').textContent = `rms e ${result.rms.toExponential(1)} · ${(100 * result.above1e2).toFixed(0)} % of pixels above 1 %, in units of the light with nothing in the way`;
}
function leaveErrorView() {
  if (!settings.errorView) return;
  settings.errorView = false;
  $('measure').textContent = '';
  syncControls();
}

const KEYS = [['drag', 'pan'], ['scroll', 'zoom'], ['space', 'pause'], ['r', 'boxes on pixels'], ['z', 'zoom'], ['f', 'smooth display'], ['k', 'sky'], ['b', 'budget'], ['w', 'Wu split'], ['e', 'error against exact'], ['s', 'screenshot'], ['?', 'this help']];
KEYS.forEach(([k, d]) => $('help').append(el('div', { className: 'row' }, el('span', {}, el('kbd', { textContent: k })), el('span', { textContent: d }))));
const toggleHelp = (force?: boolean) => ($('help').classList.toggle('open', force), $('help-btn').classList.toggle('on', $('help').classList.contains('open')));
$('help-btn').addEventListener('click', () => toggleHelp());
function screenshot() {
  view.toBlob((b) => b && Object.assign(document.createElement('a'), { href: URL.createObjectURL(b), download: `reciprocal-canvas-${Date.now()}.png` }).click());
}
$('shot-btn').addEventListener('click', screenshot);
document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const keys: Record<string, () => unknown> = {
    ' ': () => (settings.paused = !settings.paused),
    w: () => (settings.wuSplit = !settings.wuSplit),
    r: () => (settings.raster = next(RASTERS, settings.raster)),
    z: () => (settings.zoom = next(ZOOMS, settings.zoom)),
    f: () => (settings.smooth = !settings.smooth),
    k: () => (settings.sky = next(Object.keys(SKIES), settings.sky)),
    b: () => (settings.budget = next(BUDGETS, settings.budget)),
    e: () => (settings.errorView ? leaveErrorView() : enterErrorView()),
    s: screenshot,
    '?': () => toggleHelp(),
    Escape: () => toggleHelp(false),
  };
  if (!keys[e.key]) return;
  e.preventDefault();
  keys[e.key]();
  changed();
});
readControls(new URLSearchParams(location.search), controls);
syncControls();

// ---------------------------------------------------------------- measurements (also window.canvasHRC)

async function read(buffer: GPUBuffer, bytes: number) {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return out;
}

/** this frame against the exact trace of the world (its boxes as they are), over the view's free pixels (also both
 * fluences, world [x * N + y] rgba, and the domain):
 * e = (F - exact) / exact_free, exact_free the same lights with nothing else in the way (pixels where it is below 1e-4 of
 * the view's largest are left out) */
const measure = () => alone(async () => {
  const { s, domain, boxes } = await solveFrame(false), N = s.N;
  device.queue.writeBuffer(s.referenceShapes, 0, placed(boxes, domain, 'exact'));
  const half = Math.max(view.width, view.height) / camera.zoom / 2;
  const to = (c: number, o: number, side: number) => Math.max(0, Math.min(N, Math.floor((c + side * half - o) / domain.pixel)));
  const x0 = to(camera.x, domain.x, -1), x1 = to(camera.x, domain.x, 1) + 1, y0 = to(camera.y, domain.y, -1), y1 = to(camera.y, domain.y, 1) + 1;
  for (const lightsOnly of [0, 1])
    for (let x = x0; x < x1; x += 4) {
      const columns = Math.min(4, x1 - x);
      device.queue.writeBuffer(s.referenceWorld, 0, new Uint32Array([boxes.length, x, columns, lightsOnly]));
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(s.reference);
      pass.setBindGroup(0, s.referenceGroup);
      pass.dispatchWorkgroups(Math.ceil((columns * N) / 64));
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
  const fluence = await read(s.engine.outputBuffer, N * N * 16), exact = await read(s.exact, N * N * 16), scene = await read(s.engine.sceneBuffer, N * N * 16);
  const lum = (a: Float32Array, i: number) => 0.2126 * a[4 * i] + 0.7152 * a[4 * i + 1] + 0.0722 * a[4 * i + 2];
  let largest = 0;
  for (let x = x0; x < x1; x++) for (let y = y0; y < y1; y++) largest = Math.max(largest, exact[4 * (x * N + y) + 3]);
  const errors: number[] = [];
  for (let x = x0; x < x1; x++)
    for (let y = y0; y < y1; y++) {
      const i = x * N + y, free = exact[4 * i + 3];
      if (scene[4 * i + 3] > 0 && free >= 1e-4 * largest) errors.push((lum(fluence, i) - lum(exact, i)) / free);
    }
  const abs = errors.map(Math.abs).sort((a, b) => a - b);
  return {
    N, domain, fluence, exact,
    pixels: errors.length,
    rms: Math.sqrt(errors.reduce((a, e) => a + e * e, 0) / errors.length),
    mean: errors.reduce((a, e) => a + e, 0) / errors.length,
    above1e2: abs.filter((e) => e > 1e-2).length / abs.length,
    p99: abs[Math.floor(0.99 * abs.length)],
  };
});

/** GPU ms per frame with everything moving (frames encoded back to back, the median of rounds): the frame, the shown
 * solve's sweep alone, and its fill and solve alone */
const timeFrames = (frames = 10, rounds = 7) => alone(async () => {
  await solveFrame(false);
  const s = solverNow(), results: Record<string, number[]> = { frame: [], sweep: [], solve: [] };
  const run = async (what: string) => {
    const encoder = device.createCommandEncoder();
    for (let f = 0; f < frames; f++) {
      const boxes = worldAt(time + f / 60);
      if (what === 'frame') encodeSolves(encoder, boxes);
      else if (what === 'solve') s.engine.encodeAll(encoder);
      else {
        s.sweep.setShapes(placed(boxes, domainFor(s.N), settings.raster), boxes.length);
        const pass = encoder.beginComputePass();
        s.sweep.encode(pass);
        pass.end();
      }
    }
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / frames;
  };
  for (let r = 0; r < rounds; r++) for (const what of Object.keys(results)) results[what].push(await run(what));
  const median = (a: number[]) => +[...a].sort((x, y) => x - y)[a.length >> 1].toFixed(2);
  return Object.fromEntries(Object.entries(results).map(([k, v]) => [k, median(v)]));
});

/** solve this frame and read its fluence back */
const solveHere = () => alone(async () => {
  const { s, domain } = await solveFrame(false);
  return { N: s.N, domain, fluence: await read(s.engine.outputBuffer, s.N * s.N * 16) };
});
const drawHere = () => alone(() => solveFrame(true));

(window as unknown as Record<string, unknown>).canvasHRC = { settings, camera, world: WORLD, skies: SKIES, setTime: (t: number) => (time = t), solveHere, drawHere, measure, timeFrames };
requestAnimationFrame(frame);
