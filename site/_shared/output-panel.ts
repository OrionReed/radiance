// The scheme pages' output panel: the reference views, the map, the scene overlay and the exposure, with their legends
// and the reference's status line. The page owns the reference and the scene lock.
import { el, type Spec } from './ui';
import { BANDED_LEGEND, EXPOSURE_CONTROL, MAP_CONTROL } from './display';

type Make = (parent: HTMLElement, spec: Spec) => void;
/** reference views: 0 the scheme, 1 path trace, 2 exact, 3 error against the path trace, 4 against exact */
export interface OutputOptions {
  controls: { segmented: Make; slider: Make; pill: Make };
  /** the display settings: map ('tone' or 'banded'), scene overlay, exposure in stops */
  display: { map: string; scene: boolean; stops: number };
  /** after a display setting changes (a redraw, no solve) */
  redraw: () => void;
  /** the reference view; set gets the view clicked, or 0 when the selected one is clicked again */
  reference: { get: () => number; set: (mode: number) => void };
  referenceTitle: string;
  sceneTitle: string;
  /** the scheme's name in the error legend */
  errorScheme?: () => string;
  /** whether the error is of luminance (the scheme's output is rgb) */
  luminance?: boolean;
}
/** the exact reference's directions per pixel, per pixel of N */
export const EXACT_DENSITY = 32;

const sup = (s: number) => String(s).replace(/[-.0-9]/g, (c) => ({ '-': '⁻', '.': '·' } as Record<string, string>)[c] ?? '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]);
const fmtTime = (ms: number) => (!isFinite(ms) ? '…' : ms < 60e3 ? `${(ms / 1000).toFixed(ms < 10e3 ? 1 : 0)} s` : `${Math.floor(ms / 60e3)} m ${Math.round((ms % 60e3) / 1000)} s`);
const ticksRow = (ticks: [string, number][]) => {
  const row = el('div', { className: 'ticks' });
  Object.assign(row.style, { position: 'relative', height: '12px' });
  for (const [label, x] of ticks) {
    const s = el('span', { textContent: label });
    Object.assign(s.style, { position: 'absolute', left: `${x}%`, transform: `translateX(${x === 0 ? '0' : x === 100 ? '-100%' : '-50%'})` });
    row.append(s);
  }
  return row;
};

export function outputPanel(parent: HTMLElement, o: OutputOptions) {
  const { segmented, slider, pill } = o.controls;
  const bind = (name: 'map' | 'scene' | 'stops') => ({ get: () => o.display[name], set: (v: unknown) => ((o.display as Record<string, unknown>)[name] = v, o.redraw()) });
  // errors of at least 10^errorClip show magenta; 0 is off
  let errorClip = 0;

  segmented(parent, {
    label: 'reference',
    title: o.referenceTitle,
    cols: 2,
    options: [[1, 'path trace'], [2, 'exact'], [3, 'Δ path trace', 'error against the path trace'], [4, 'Δ exact', 'error against the exact reference']],
    get: o.reference.get,
    set: (v) => o.reference.set(o.reference.get() === v ? 0 : v),
    url: ['ref', 0],
  });
  parent.lastElementChild!.classList.add('reference');
  const refStatus = el('div', { className: 'hint' });
  parent.append(refStatus);
  // the map and the scene overlay share a row
  const mapRow = el('div', { className: 'map-row' });
  parent.append(mapRow);
  segmented(mapRow, { ...MAP_CONTROL, ...bind('map'), url: ['map', 'tone'] });
  pill(mapRow, { label: 'scene', key: 'v', title: o.sceneTitle, ...bind('scene'), url: ['showScene', false] });
  slider(parent, { ...EXPOSURE_CONTROL, ...bind('stops'), format: (v) => (v > 0 ? `+${v}` : String(v)) });

  // the error legend: the draw shader's errorColour from −1 through 0 to +1. Dragging the bar sets errorClip, mirrored
  // about 0; at the ends (errorClip 0) nothing is magenta
  function errorLegend(L: HTMLElement) {
    const BRIGHTER = [[1, 1, 1], [0.957, 0.647, 0.51], [0.839, 0.376, 0.302], [0.404, 0, 0.122]];
    const DARKER = [[1, 1, 1], [0.573, 0.773, 0.871], [0.263, 0.576, 0.765], [0.02, 0.188, 0.38]];
    const colour = (t: number) => {   // t in [0, 1]: error −1 at 0, 0 at ½, +1 at 1
      const d = Math.abs(2 * t - 1);
      if (errorClip < 0 && d >= (errorClip + 5) / 5) return 'rgb(217,51,199)';
      const s = t < 0.5 ? DARKER : BRIGHTER, m = (Math.round(d * 50) / 50) * 3, i = Math.min(Math.floor(m), 2), f = m - i;
      return `rgb(${s[i].map((a, k) => Math.round((a + (s[i + 1][k] - a) * f) * 255)).join(',')})`;
    };
    const b = el('div', { className: 'bar error', title: 'drag to mark errors of at least a size in magenta; double-click to clear' });
    const marks = [el('i'), el('i')];
    b.append(...marks);
    const ticks = ticksRow([['−1', 0], ['−10⁻²', 20], ['−10⁻⁴', 40], ['0', 50], ['10⁻⁴', 60], ['10⁻²', 80], ['1', 100]]);
    const caption = el('div', { className: 'caption' });
    const scheme = o.errorScheme?.() ?? 'scheme';
    const draw = () => {
      b.style.background = `linear-gradient(to right, ${Array.from({ length: 201 }, (_, k) => `${colour(k / 200)} ${k / 2}%`).join(',')})`;
      const d = (errorClip + 5) / 5;
      marks[0].style.left = `${50 - 50 * d}%`;
      marks[1].style.left = `${50 + 50 * d}%`;
      caption.textContent = `error: (${scheme} − reference) / reference in L · exposure${o.luminance ? ', luminance' : ''}, reference floored at 0.01; log scale, 10 bands per decade; white at 10⁻⁵ or less, red brighter, blue darker` +
        (errorClip < 0 ? `; magenta at least 10${sup(errorClip)}` : '');
    };
    const set = (e: PointerEvent) => {
      const r = b.getBoundingClientRect(), d = Math.min(1, Math.abs((2 * (e.clientX - r.left)) / r.width - 1));
      errorClip = Math.min(0, Math.round((5 * d - 5) * 10) / 10);
      draw();
      o.redraw();
    };
    b.addEventListener('pointerdown', (e) => (b.setPointerCapture(e.pointerId), set(e)));
    b.addEventListener('pointermove', (e) => b.hasPointerCapture(e.pointerId) && set(e));
    b.addEventListener('dblclick', () => ((errorClip = 0), draw(), o.redraw()));
    draw();
    L.append(b, ticks, caption);
  }

  return {
    /** the legend for the current view: the error bar in the error views, the bands with the banded map, else none */
    legend(L: HTMLElement) {
      L.innerHTML = '';
      if (o.reference.get() >= 3) return errorLegend(L);
      if (o.display.map !== 'banded') return;
      const b = el('div', { className: 'bar' });
      b.style.background = BANDED_LEGEND.background;
      L.append(b, ticksRow(BANDED_LEGEND.ticks.map(([t, x]) => [t, x])), el('div', { className: 'caption', textContent: BANDED_LEGEND.caption }));
    },
    /** the draw shader's error clip: errors of at least it show magenta */
    clip: () => (errorClip < 0 ? 10 ** errorClip : 1e9),
    /** the status line under the reference views */
    status(reference: { pathTraceFrames: number; exactProgress?: { done: boolean; m: number; elapsedMs: number; fraction: number; etaMs: number } | null } | null) {
      const mode = o.reference.get();
      const st = mode === 2 || mode === 4 ? reference?.exactProgress : null;
      refStatus.textContent = !mode ? ''
        : mode === 1 || mode === 3 ? `path trace · ${reference!.pathTraceFrames * 16} samples per pixel`
        : !st ? 'exact · starting…'
        : st.done ? `exact · ${st.m} directions per pixel · took ${fmtTime(st.elapsedMs)}`
        : `exact · ${(st.fraction * 100).toFixed(1)} % · ${fmtTime(st.etaMs)} left`;
    },
  };
}
