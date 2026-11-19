// The scheme pages' controls (toggle, segmented, slider, pill), element helpers and URL state. Every control the page
// makes is in `controls`; syncControls() re-reads them all, so keyboard shortcuts and URL loads show on the controls.

export const $ = (id: string) => document.getElementById(id) as HTMLElement;

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, unknown> = {}, ...children: (Node | string)[]) {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export interface Spec {
  label?: string;
  get: () => any;
  set: (v: any) => void;
  enabled?: () => boolean;
  key?: string;
  title?: string;
  url?: [string, unknown];
  options?: [any, string, string?][];
  cols?: number;
  min?: number;
  max?: number;
  step?: number;
  format?: (v: any) => string;
}
export interface Control {
  spec: Spec;
  sync: () => void;
  wrap: HTMLElement;
}

/** a binding of obj[name] that calls `after` when set (the page's touch() by default) */
export const fieldFor = (fallback: () => void) => (obj: Record<string, any>, name: string, after: () => void = fallback) => ({
  get: () => obj[name],
  set: (v: unknown) => ((obj[name] = v), after()),
});

/** the page's controls; `changed` runs after any control is used */
export function createControls(changed: () => void) {
  const controls: Control[] = [];

  function toggle(parent: HTMLElement, spec: Spec) {
    const sw = el('span', { className: 'sw' });
    const b = el('button', { className: 'toggle', title: spec.title ?? '' }, sw, el('span', { className: 'name', textContent: spec.label }));
    if (spec.key) b.append(el('kbd', { textContent: spec.key }));
    b.addEventListener('click', () => {
      spec.set(!spec.get());
      changed();
    });
    const wrap = el('div', { className: 'field' }, b);
    parent.append(wrap);
    controls.push({ spec, sync: () => b.classList.toggle('on', !!spec.get()), wrap });
  }

  function segmented(parent: HTMLElement, spec: Spec) {
    const seg = el('div', { className: 'seg' + (spec.cols ? ' wrap' : '') });
    if (spec.cols) seg.style.setProperty('--cols', String(spec.cols));
    const buttons = spec.options!.map(([value, text, title]) => {
      const b = el('button', { textContent: text, title: title ?? text });
      b.addEventListener('click', () => {
        spec.set(value);
        changed();
      });
      seg.append(b);
      return [value, b] as const;
    });
    let wrap: HTMLElement = seg;
    if (spec.label) {
      const label = el('div', { className: 'label' }, el('span', { textContent: spec.label }));
      if (spec.key) label.append(el('kbd', { textContent: spec.key }));
      wrap = el('div', { className: 'field', title: spec.title ?? '' }, label, seg);
    }
    parent.append(wrap);
    controls.push({ spec, sync: () => buttons.forEach(([v, b]) => b.classList.toggle('on', v === spec.get())), wrap });
  }

  function slider(parent: HTMLElement, spec: Spec) {
    const val = el('span', { className: 'val' });
    const input = el('input', { type: 'range', min: spec.min, max: spec.max, step: spec.step });
    input.addEventListener('input', () => {
      spec.set(Number(input.value));
      changed();
    });
    const wrap = el('div', { className: 'field', title: spec.title ?? '' }, el('div', { className: 'label' }, el('span', { textContent: spec.label }), val), input);
    parent.append(wrap);
    controls.push({
      spec,
      sync: () => {
        input.value = spec.get();
        val.textContent = spec.format ? spec.format(spec.get()) : spec.get();
      },
      wrap,
    });
  }

  /** an on/off button styled as a one-option segmented control, with a label row (caption, key) above it */
  function pill(parent: HTMLElement, spec: Spec & { caption?: string }) {
    const b = el('button', { textContent: spec.label, title: spec.title ?? spec.label });
    b.addEventListener('click', () => {
      spec.set(!spec.get());
      changed();
    });
    const label = el('div', { className: 'label' }, el('span', { textContent: spec.caption ?? '' }));
    if (spec.key) label.append(el('kbd', { textContent: spec.key }));
    const wrap = el('div', { className: 'field', title: spec.title ?? '' }, label, el('div', { className: 'seg pill' }, b));
    parent.append(wrap);
    controls.push({ spec, sync: () => b.classList.toggle('on', !!spec.get()), wrap });
  }

  function syncControls() {
    for (const c of controls) {
      c.sync();
      if (c.spec.enabled) c.wrap.classList.toggle('disabled', !c.spec.enabled());
    }
  }

  return { controls, toggle, segmented, slider, pill, syncControls };
}

// ── URL state: every control with url: [name, default] round-trips; the page adds the rest (its scene) ──

export const encodeValue = (v: unknown) => (v === true ? '1' : v === false ? '0' : typeof v === 'number' && !Number.isInteger(v) ? String(+v.toFixed(3)) : String(v));
export const decodeValue = (s: string, def: unknown) => (typeof def === 'boolean' ? s === '1' : typeof def === 'number' ? Number(s) : s);

/** put every control that differs from its default into p; valueOf(name, value) may substitute what is saved */
export function writeControls(p: URLSearchParams, controls: Control[], valueOf = (_name: string, v: unknown) => v) {
  for (const { spec } of controls) {
    if (!spec.url) continue;
    const [name, def] = spec.url, v = valueOf(name, spec.get());
    if (v !== def) p.set(name, encodeValue(v));
  }
}
/** set every control from p, or to its default */
export function readControls(p: URLSearchParams, controls: Control[]) {
  for (const { spec } of controls) {
    if (!spec.url) continue;
    const [name, def] = spec.url;
    spec.set(p.has(name) ? decodeValue(p.get(name)!, def) : def);
  }
}
/** the scene a URL names, by id or index (0 when none) */
export function sceneFromURL(p: URLSearchParams, scenes: { id: string }[]) {
  const s = p.get('scene') ?? '0';
  const byId = scenes.findIndex((x) => x.id === s);
  return byId >= 0 ? byId : Number.isFinite(Number(s)) ? Number(s) : 0;
}
export const replaceURL = (p: URLSearchParams) => {
  const qs = p.toString();
  history.replaceState(null, '', qs ? '?' + qs : location.pathname);
};
