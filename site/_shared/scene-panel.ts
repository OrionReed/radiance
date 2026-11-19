// The scheme pages' scene panel: the scene list (tests impose their resolution while shown) and the pen on the canvas
// (drag a shape, draw or erase strokes, a cursor that previews the pen). What a shape or stroke is made of, and how the
// scene is rasterised, stays with the page.
import { el } from './ui';

export interface SceneEntry {
  id: string;
  name: string;
  group: string;
  /** a test's own resolution */
  N?: number;
}

/** the scene list; load(scene) copies the scene into the page, after() runs after any switch (the page's touch and URL) */
export function sceneList(o: {
  list: HTMLElement;
  label: HTMLElement;
  scenes: SceneEntry[];
  resolution: { get: () => number; set: (n: number) => void };
  load: (scene: SceneEntry) => void;
  after: () => void;
  changed: () => void;
}) {
  let current = 0, freeN: number | null = null;   // freeN: the resolution to go back to after a test
  const buttons: HTMLElement[] = [];
  function switchTo(index: number) {
    current = (index + o.scenes.length) % o.scenes.length;
    (document.activeElement as HTMLElement | null)?.blur();
    const s = o.scenes[current];
    if (s.N) ((freeN ??= o.resolution.get()), o.resolution.set(s.N));
    else if (freeN) (o.resolution.set(freeN), (freeN = null));
    o.load(s);
    buttons.forEach((b, i) => b.classList.toggle('on', i === current));
    o.label.textContent = s.id;
    o.after();
  }
  let group: string | null = null;
  o.scenes.forEach((s, i) => {
    if (s.group !== group) o.list.append(el('div', { className: 'group', textContent: (group = s.group) }));
    const b = el('button', { title: s.id }, el('span', { className: 'idx', textContent: i }), el('span', { textContent: s.name }));
    b.addEventListener('click', () => {
      switchTo(i);
      o.changed();
    });
    o.list.append(b);
    buttons.push(b);
  });
  return {
    switchTo,
    /** the scene shown */
    get current() {
      return current;
    },
    /** while a test is shown: the resolution to go back to (the one to save) */
    get freeN() {
      return freeN;
    },
  };
}

type Point = [number, number];
export interface Stroke {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** in scene units */
  size: number;
}

/** pen sizes, in pixels of the N × N grid */
export const PEN_SIZES = [1, 2, 4, 8, 16, 32];
export const nextPenSize = (size: number, d: number, fallback = 2) => PEN_SIZES[Math.max(0, Math.min(PEN_SIZES.length - 1, PEN_SIZES.indexOf(size) + d))] ?? fallback;

/**
 * The pen on the canvas (scene units: the square is `span` across, origin at its centre, y down). Pressing on a shape
 * (topmost, with half a pixel of slack) selects and drags it; elsewhere it draws strokes of the pen's material, or erases
 * the strokes within max(size, 2) pixels. `paint` replaces the default strokes list (a page whose engine keeps strokes).
 */
export function penPointer<S>(view: HTMLElement, o: {
  span: number;
  N: () => number;
  locked: () => boolean;
  /** the page's shapes and a shape's rectangle (x, y, w, h: top-left and size in scene units); move sets its top-left */
  shapes: () => S[];
  rect: (s: S) => [number, number, number, number];
  move: (s: S, x: number, y: number) => void;
  select: (s: S | null) => void;
  eraser: () => boolean;
  /** pen size in pixels */
  size: () => number;
  /** the default strokes list and the material a new stroke takes */
  strokes?: { get: () => Stroke[]; set: (s: Stroke[]) => void; material: () => object };
  paint?: (a: Point, b: Point, size: number, eraser: boolean) => void;
  cursor: (p: Point | null) => void;
  touch: () => void;
}) {
  const toScene = (e: PointerEvent): Point => {
    const r = view.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width - 0.5) * o.span, ((e.clientY - r.top) / r.height - 0.5) * o.span];
  };
  const cellUnits = () => o.span / o.N();
  const shapeAt = ([u, v]: Point) => {
    const pad = cellUnits() / 2; // shapes thinner than a pixel still pick
    const shapes = o.shapes();
    for (let i = shapes.length - 1; i >= 0; i--) {
      const [x, y, w, h] = o.rect(shapes[i]);
      if (u >= x - pad && u <= x + w + pad && v >= y - pad && v <= y + h + pad) return shapes[i];
    }
    return null;
  };
  function paint(a: Point, b: Point) {
    if (o.paint) o.paint(a, b, o.size() * cellUnits(), o.eraser());
    else if (o.strokes && o.eraser()) {
      const r = Math.max(o.size(), 2) * cellUnits();
      const near = (s: Stroke) => {
        const vx = s.x1 - s.x0, vy = s.y1 - s.y0, L = vx * vx + vy * vy;
        const t = L > 0 ? Math.max(0, Math.min(1, ((b[0] - s.x0) * vx + (b[1] - s.y0) * vy) / L)) : 0;
        return Math.hypot(s.x0 + t * vx - b[0], s.y0 + t * vy - b[1]) < r;
      };
      o.strokes.set(o.strokes.get().filter((s) => !near(s)));
    } else if (o.strokes) o.strokes.get().push({ x0: a[0], y0: a[1], x1: b[0], y1: b[1], size: o.size() * cellUnits(), ...o.strokes.material() });
    o.touch();
  }
  let drawing = false, drag: { s: S; dx: number; dy: number } | null = null, last: Point = [0, 0];
  view.addEventListener('pointerdown', (e) => {
    if (o.locked()) return;
    const p = toScene(e);
    view.setPointerCapture(e.pointerId);
    const s = shapeAt(p);
    if (s && !o.eraser()) {
      o.select(s);
      const [x, y] = o.rect(s);
      drag = { s, dx: x - p[0], dy: y - p[1] };
      view.className = 'grabbing';
    } else {
      o.select(null);
      drawing = true;
      last = p;
      paint(p, p);
    }
    e.preventDefault();
  });
  view.addEventListener('pointermove', (e) => {
    if (o.locked()) return;
    const p = toScene(e);
    o.cursor(p);
    if (drag) o.move(drag.s, p[0] + drag.dx, p[1] + drag.dy);
    else if (drawing && Math.hypot(p[0] - last[0], p[1] - last[1]) >= cellUnits()) {
      paint(last, p);
      last = p;
    } else if (!drawing) view.className = shapeAt(p) && !o.eraser() ? 'grab' : '';
    o.touch();
  });
  const end = () => {
    drawing = false;
    drag = null;
    view.className = '';
    o.touch();
  };
  view.addEventListener('pointerup', end);
  view.addEventListener('pointercancel', end);
  view.addEventListener('pointerleave', () => {
    o.cursor(null);
    o.touch();
  });
  return {
    /** a shape is being dragged (the cursor preview hides) */
    dragging: () => drag !== null,
  };
}
