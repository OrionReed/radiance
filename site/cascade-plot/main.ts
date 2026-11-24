// The page: a list of figures, each with its own settings, and one shared look (theme, line weight, labels).
// Every setting is in the URL: the figure as `fig`, a figure's settings as `<figure>.<setting>`, the look by name.
import { Diagram, SCHEDULES, THEMES, type Look, type ThemeName } from './plot';
import { FIGURES } from './figures';
import type { Figure, Option, Value } from './figure';
import { $, el, createControls, writeControls, readControls, replaceURL } from '../_shared/ui';

type State = Record<string, Value>;
const V = { fig: FIGURES[0].id, theme: 'paper' as ThemeName, weight: 1, labels: true };
/** the isometric view turning on its own: the turn's own control is off meanwhile */
const spin = { on: false, last: 0 };
const states = new Map<string, State>(FIGURES.map((f) => [f.id, { ...f.defaults }]));

const figureOf = () => FIGURES.find((f) => f.id === V.fig) ?? FIGURES[0];
const ui = createControls(() => render());
const bind = (obj: Record<string, unknown>, name: string) => ({ get: () => obj[name], set: (v: unknown) => { obj[name] = v; } });

function section(title: string) {
  const sec = el('section', {}, el('h2', { textContent: title }));
  $('left').append(sec);
  return sec;
}

const figSec = section('Figures');
ui.segmented(figSec, { ...bind(V, 'fig'), options: FIGURES.map((f) => [f.id, f.name, f.question]), cols: 2, url: ['fig', FIGURES[0].id] });
const about = el('p', { className: 'about' });
figSec.append(about);

// each figure's controls, right below its description; only the shown figure's are visible
const sliders: { fig: Figure; opt: Option<State>; input: HTMLInputElement }[] = [];
const figControls = new Map<string, HTMLElement>();
for (const fig of FIGURES) {
  const st = states.get(fig.id)!, box = el('div', { className: 'controls' });
  figSec.append(box);
  figControls.set(fig.id, box);
  for (const opt of fig.options as Option<State>[]) {
    const own = opt.enabled;
    const enabled = opt.key === 'turn' ? () => (own ? own(st) : true) && !spin.on : own ? () => own(st) : undefined;
    const spec = { label: opt.label, ...bind(st, opt.key), title: opt.title, url: [`${fig.id}.${opt.key}`, fig.defaults[opt.key]] as [string, unknown], enabled };
    if (opt.kind === 'toggle') ui.toggle(box, spec);
    else if (opt.kind === 'choice') ui.segmented(box, { ...spec, options: opt.choices!, cols: opt.cols ?? (opt.choices!.length > 3 ? 2 : undefined) });
    else {
      const step = opt.step ?? 1;
      ui.slider(box, { ...spec, min: typeof opt.min === 'number' ? opt.min : 0, max: typeof opt.max === 'number' ? opt.max : 64, step, format: opt.format ?? ((v: number) => (step < 1 ? v.toFixed(2) : String(v))) });
      sliders.push({ fig, opt, input: ui.controls[ui.controls.length - 1].wrap.querySelector('input')! });
    }
    if (opt.key === 'turn')
      ui.pill(box, { label: 'Spin', title: 'turn the stack continuously', get: () => spin.on, set: (v: boolean) => toggleSpin(v), enabled: () => (own ? own(st) : true) });
  }
}
const info = el('pre', { className: 'info' });
figSec.append(info);

const viewSec = section('View');
ui.segmented(viewSec, { label: 'Theme', ...bind(V, 'theme'), options: [['paper', 'Plotter on paper'], ['chalk', 'Chalk on blackboard']], url: ['theme', 'paper'] });
ui.slider(viewSec, { label: 'Line weight', ...bind(V, 'weight'), min: 0.4, max: 2.5, step: 0.05, format: (v: number) => `${v.toFixed(2)}x`, url: ['weight', 1] });
ui.toggle(viewSec, { label: 'Labels', ...bind(V, 'labels'), url: ['labels', true] });
const save = el('button', { className: 'action', textContent: 'Save SVG' });
save.addEventListener('click', () => {
  const blob = new Blob([$('figure').innerHTML], { type: 'image/svg+xml' });
  el('a', { href: URL.createObjectURL(blob), download: `${V.fig}.svg` }).click();
});
viewSec.append(el('div', { className: 'field' }, save));

/** the stack turns by SPIN_RATE degrees a second while spinning */
const SPIN_RATE = 30;
function toggleSpin(on: boolean) {
  spin.on = on;
  spin.last = performance.now();
  if (on) requestAnimationFrame(step);
  else render();
}
function step(now: number) {
  if (!spin.on) return;
  const fig = figureOf(), st = states.get(fig.id)!;
  if (!('turn' in st) || st.mode !== 'stack') return toggleSpin(false), render();
  st.turn = ((((Number(st.turn) + (SPIN_RATE * (now - spin.last)) / 1000 + 180) % 360) + 360) % 360) - 180;
  spin.last = now;
  render();
  requestAnimationFrame(step);
}

readControls(new URLSearchParams(location.search), ui.controls);

/** keep a figure's settings inside their ranges (which may depend on its other settings) */
function clampSettings(fig: Figure, st: State) {
  for (const opt of fig.options as Option<State>[]) {
    if (opt.kind === 'choice' && !opt.choices!.some(([v]) => v === st[opt.key])) st[opt.key] = fig.defaults[opt.key];
    if (opt.kind === 'toggle') st[opt.key] = !!st[opt.key];
  }
  for (let pass = 0; pass < 2; pass++)   // twice: a range may depend on a setting clamped after it
    for (const { fig: f, opt, input } of sliders) {
      if (f !== fig) continue;
      const lo = typeof opt.min === 'function' ? opt.min(st) : opt.min ?? 0, hi = typeof opt.max === 'function' ? opt.max(st) : opt.max ?? 64;
      input.min = String(lo); input.max = String(hi);
      const v = Number(st[opt.key]);
      st[opt.key] = Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : (fig.defaults[opt.key] as number);
    }
}

function render() {
  const fig = figureOf(), st = states.get(fig.id)!;
  clampSettings(fig, st);
  for (const [id, box] of figControls) box.style.display = id === fig.id ? '' : 'none';
  const N = fig.N(st), K = Math.log2(N), sch = SCHEDULES[String(st.dirs ?? 'all')] ?? SCHEDULES.all;
  const c = { N, K, sch, s: st };
  about.textContent = fig.about;
  info.textContent = fig.info ? fig.info(c) : '';
  info.style.display = fig.info ? '' : 'none';
  const plot = $('plot');
  plot.dataset.theme = V.theme;
  plot.style.setProperty('--fig-bg', THEMES[V.theme].bg);
  $('fig-title').textContent = fig.name;
  $('fig-question').textContent = fig.question;
  const D = new Diagram({
    N, rows: fig.rows?.(st), sheets: fig.sheets(c), cols: fig.cols?.(c),
    mode: (st.mode as 'plan' | 'stack') ?? 'plan', turn: Number(st.turn ?? 20), tilt: Number(st.tilt ?? 35), layer: Number(st.layer ?? 1.15),
    grid: (st.marks as 'probes' | 'corners' | 'none') ?? 'probes',
  });
  fig.draw(D, c);
  const box = $('figure'), look: Look = { theme: V.theme, weight: V.weight, labels: V.labels };
  box.innerHTML = D.svg(box.clientWidth, box.clientHeight, look);
  ui.syncControls();
  if (spin.on) return;   // the URL keeps the turn it stopped at
  const q = new URLSearchParams();
  writeControls(q, ui.controls);
  replaceURL(q);
}

window.addEventListener('resize', () => render());
render();
