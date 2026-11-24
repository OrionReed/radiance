import { castRows, solve, type Windows, type Work } from './wide';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const controls = ['dirs', 'windows', 'scene', 'size', 'n', 'right', 'exposure'].map((id) => $<HTMLInputElement>(id));
const left = $<HTMLCanvasElement>('left'), right = $<HTMLCanvasElement>('rightc');
let light = { x: 0.17, y: 0.5 };

const spacingOf = (dirs: string) =>
  dirs === 'all' ? () => 1 : dirs === 'half' ? (L: number) => (L === 1 ? 1 : 2) : (L: number) => Math.max(1, (2 * L) / Number(dirs));

function scene(N: number) {
  const emission = new Float32Array(N * N), solid = new Uint8Array(N * N);
  const k = N / 64, set = (x: number, y: number, e = 0) => {
    if (x < 0 || y < 0 || x >= N || y >= N) return;
    solid[x * N + y] = 1; emission[x * N + y] = e;
  };
  const kind = $<HTMLSelectElement>('scene').value;
  if (kind === 'walls') {
    for (let y = Math.round(20 * k); y < Math.round(29 * k); y++) set(Math.round(24 * k), y);
    for (let y = Math.round(34 * k); y < Math.round(44 * k); y++) set(Math.round(24 * k), y);
    for (let x = Math.round(40 * k); x < Math.round(52 * k); x++) set(x, Math.round(12 * k));
  } else if (kind === 'slats') {
    for (let y = 0; y < N; y++) if (y % 4 < 2) set(Math.round(0.45 * N), y);
  }
  const size = Number($<HTMLSelectElement>('size').value), cx = Math.floor(light.x * N), cy = Math.floor(light.y * N);
  const lo = -Math.floor((size - 1) / 2);
  for (let dx = 0; dx < size; dx++)
    for (let dy = 0; dy < size; dy++) {
      const x = cx + lo + dx, y = cy + lo + dy;
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      emission[x * N + y] = 1;
      solid[x * N + y] = 1;
    }
  return { emission, solid };
}

/** the reference, cast only when the right image asks for it */
let exact: Float32Array | null = null, exactKey = '', exactJob = 0;
function startExact(N: number, emission: Float32Array, solid: Uint8Array, key: string) {
  if (key === exactKey) return;
  exactKey = key; exact = null;
  const job = ++exactJob, out = new Float32Array(N * N), dirs = N <= 64 ? 1440 : 2048;
  let x = 0;
  const step = () => {
    if (job !== exactJob) return;
    const x1 = Math.min(N, x + Math.max(1, Math.floor(4096 / N)));
    castRows(N, emission, solid, dirs, out, x, x1);
    x = x1;
    if (x < N) { $('rightcap').textContent = `exact: ${Math.round((100 * x) / N)} %`; setTimeout(step, 0); }
    else { exact = out; render(); }
  };
  setTimeout(step, 0);
}

let current: Float32Array | null = null, baseline: Float32Array | null = null, sceneNow: ReturnType<typeof scene> | null = null;
let work: Work = { rays: 0, pairs: 0 }, ms = 0;
function compute() {
  const N = Number($<HTMLSelectElement>('n').value), dirs = $<HTMLSelectElement>('dirs').value;
  const windows = $<HTMLSelectElement>('windows').value as Windows;
  sceneNow = scene(N);
  work = { rays: 0, pairs: 0 };
  const t0 = performance.now();
  current = solve(N, sceneNow.emission, sceneNow.solid, spacingOf(dirs), windows, work);
  ms = performance.now() - t0;
  baseline = $<HTMLSelectElement>('right').value === 'all' ? solve(N, sceneNow.emission, sceneNow.solid, () => 1, 'one row') : null;
  const key = `${N} ${$<HTMLSelectElement>('scene').value} ${$<HTMLSelectElement>('size').value} ${light.x} ${light.y}`;
  if ($<HTMLSelectElement>('right').value !== 'all') startExact(N, sceneNow.emission, sceneNow.solid, key);
  render();
}

function draw(canvas: HTMLCanvasElement, N: number, colour: (i: number) => [number, number, number]) {
  canvas.width = canvas.height = N;
  const ctx = canvas.getContext('2d')!, img = ctx.createImageData(N, N);
  for (let x = 0; x < N; x++)
    for (let y = 0; y < N; y++) {
      const [r, g, b] = colour(x * N + y), o = ((N - 1 - y) * N + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}

function render() {
  if (!current || !sceneNow) return;
  const N = Number($<HTMLSelectElement>('n').value), { emission, solid } = sceneNow;
  const E = 0.6 * Math.pow(2, Number($<HTMLInputElement>('exposure').value));
  $('expv').textContent = `x${(E / 0.6).toFixed(2)}`;
  const tone = (F: Float32Array) => (i: number): [number, number, number] => {
    if (solid[i]) return emission[i] > 0 ? [255, 214, 140] : [60, 64, 74];
    const v = 255 * Math.pow(1 - Math.exp(-E * F[i]), 1 / 2.2);
    return [v, v, v];
  };
  draw(left, N, tone(current));
  const dirs = $<HTMLSelectElement>('dirs').selectedOptions[0].textContent, windows = $<HTMLSelectElement>('windows').value;
  $('leftcap').textContent = `${dirs}, ${windows} ends`;
  const mode = $<HTMLSelectElement>('right').value;
  if (mode === 'all' && baseline) { draw(right, N, tone(baseline)); $('rightcap').textContent = '2L+1, one-row ends (today)'; }
  else if (exact && mode === 'exact') { draw(right, N, tone(exact)); $('rightcap').textContent = 'exact (ray cast)'; }
  else if (exact && mode === 'error') {
    let scale = 0;
    for (let i = 0; i < N * N; i++) scale = Math.max(scale, exact[i]);
    draw(right, N, (i) => {
      if (solid[i]) return [60, 64, 74];
      const d = (current![i] - exact![i]) / (exact![i] + 0.02 * scale), a = Math.min(1, Math.abs(d) / 0.5);
      return d > 0 ? [40 + 215 * a, 40, 40] : [40, 40 + 80 * a, 40 + 215 * a];
    });
    $('rightcap').textContent = 'error vs exact: red too bright, blue too dark (full colour at 50 %)';
  }
  let stats = `solve        ${ms.toFixed(0)} ms (CPU)\nrays/px      ${work.rays.toFixed(2)}  (levels >= 2, one frustum)\npairs/px     ${work.pairs.toFixed(2)}`;
  if (exact && $<HTMLSelectElement>('right').value !== 'all') {
    let tot = 0, ref = 0, err = 0;
    for (let i = 0; i < N * N; i++) if (!solid[i]) { tot += current[i]; ref += exact[i]; err += Math.abs(current[i] - exact[i]); }
    stats += `\ntotal light  ${(tot / ref).toFixed(3)} x exact\nL1 error     ${(err / ref).toFixed(3)}`;
  }
  $('stats').textContent = stats;
}

let pending = false;
const schedule = () => { if (!pending) { pending = true; setTimeout(() => { pending = false; compute(); }, 0); } };
for (const c of controls) c.addEventListener('input', () => (c.id === 'exposure' ? render() : schedule()));
for (const canvas of [left, right]) {
  const move = (ev: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    light = { x: Math.min(0.999, Math.max(0, (ev.clientX - r.left) / r.width)), y: Math.min(0.999, Math.max(0, 1 - (ev.clientY - r.top) / r.height)) };
    schedule();
  };
  canvas.addEventListener('pointerdown', (ev) => { canvas.setPointerCapture(ev.pointerId); move(ev); });
  canvas.addEventListener('pointermove', (ev) => { if (ev.buttons) move(ev); });
}
compute();
