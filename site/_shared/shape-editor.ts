/** A scene rectangle in scene units: (x, y) is its top-left before rotation; rotation (radians) is about its centre. */
export interface Shape {
  x: number;
  y: number;
  width: number;
  height: number;
  rotation?: number;
  /** Linear RGB emission (default black). */
  rgb?: number[];
  /** 1 = opaque, 0 = volumetric light, between = absorbing medium (default 1). */
  opacity?: number;
  /** Diffuse reflectance (default 0). */
  albedo?: number;
}

/** The displayed scene square in viewport pixels. */
export interface View {
  left: number;
  top: number;
  size: number;
}

type Edge = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';
const EDGES: Record<Edge, [number, number, string]> = {
  n: [0.5, 0, 'ns-resize'],
  s: [0.5, 1, 'ns-resize'],
  e: [1, 0.5, 'ew-resize'],
  w: [0, 0.5, 'ew-resize'],
  ne: [1, 0, 'nesw-resize'],
  sw: [0, 1, 'nesw-resize'],
  nw: [0, 0, 'nwse-resize'],
  se: [1, 1, 'nwse-resize'],
};

/**
 * Edits a shapes array with plain DOM: one div per shape in `layer` (a viewport-sized, fixed layer).
 * Click to select, drag to move, drag a handle of the selected shape to resize (in screen axes).
 * It reconciles with `shapes` every frame, so code can add, remove and mutate shapes directly.
 * Scene units map onto `view()` with the scene square [-span/2, span/2]² filling it.
 */
export class ShapeEditor {
  selected: Shape | null = null;
  /** While locked, shapes ignore the pointer. */
  locked = false;
  onSelect?: (shape: Shape | null) => void;

  #divs = new Map<Shape, HTMLDivElement>();
  #handles: HTMLDivElement[] = [];
  #frame = 0;

  constructor(
    readonly layer: HTMLElement,
    readonly shapes: Shape[],
    readonly view: () => View,
    readonly span: number,
  ) {
    for (const [edge, [fx, fy, cursor]] of Object.entries(EDGES)) {
      const h = document.createElement('div');
      h.className = 'shape-handle';
      h.dataset.edge = edge;
      Object.assign(h.style, { left: `${fx * 100}%`, top: `${fy * 100}%`, cursor });
      this.#handles.push(h);
    }
    const loop = () => {
      this.sync();
      this.#frame = requestAnimationFrame(loop);
    };
    loop();
  }

  /** True if the element belongs to the editor (a shape or a handle). */
  owns(target: EventTarget | null) {
    return target instanceof Element && !!target.closest('.shape');
  }

  select(shape: Shape | null) {
    if (shape === this.selected) return;
    this.selected = shape;
    this.onSelect?.(shape);
    this.sync();
  }

  remove(shape: Shape) {
    const i = this.shapes.indexOf(shape);
    if (i >= 0) this.shapes.splice(i, 1);
    if (this.selected === shape) this.select(null);
  }

  destroy() {
    cancelAnimationFrame(this.#frame);
    for (const div of this.#divs.values()) div.remove();
    this.#divs.clear();
  }

  /** Brings the divs in line with the shapes (called every frame). */
  sync() {
    if (this.selected && !this.shapes.includes(this.selected)) this.select(null);
    const live = new Set(this.shapes);
    for (const [shape, div] of this.#divs) {
      if (!live.has(shape)) {
        div.remove();
        this.#divs.delete(shape);
      }
    }
    const v = this.view();
    const k = v.size / this.span;
    const cx = v.left + v.size / 2;
    const cy = v.top + v.size / 2;
    for (const shape of this.shapes) {
      let div = this.#divs.get(shape);
      if (!div) {
        div = document.createElement('div');
        div.className = 'shape';
        div.addEventListener('pointerdown', (e) => this.#onPointerDown(e, shape));
        this.layer.append(div);
        this.#divs.set(shape, div);
      }
      const s = div.style;
      s.left = `${cx + shape.x * k}px`;
      s.top = `${cy + shape.y * k}px`;
      s.width = `${Math.max(shape.width * k, 1)}px`;
      s.height = `${Math.max(shape.height * k, 1)}px`;
      s.rotate = shape.rotation ? `${shape.rotation}rad` : '';
      s.pointerEvents = this.locked ? 'none' : 'auto';
      const selected = shape === this.selected;
      div.classList.toggle('selected', selected);
      if (selected && this.#handles[0].parentElement !== div) div.append(...this.#handles);
    }
    if (!this.selected) for (const h of this.#handles) h.remove();
  }

  #onPointerDown(e: PointerEvent, shape: Shape) {
    if (e.button !== 0 || this.locked) return;
    e.preventDefault();
    e.stopPropagation();
    this.select(shape);
    const edge = (e.target as HTMLElement).dataset.edge as Edge | undefined;
    const div = e.currentTarget as HTMLElement;
    div.setPointerCapture(e.pointerId);
    const k = this.span / this.view().size; // scene units per screen pixel
    const start = { px: e.clientX, py: e.clientY, x: shape.x, y: shape.y, w: shape.width, h: shape.height };
    const min = k; // one screen pixel

    const move = (ev: PointerEvent) => {
      const dx = (ev.clientX - start.px) * k;
      const dy = (ev.clientY - start.py) * k;
      if (!edge) {
        shape.x = start.x + dx;
        shape.y = start.y + dy;
      } else {
        if (edge.includes('e')) shape.width = Math.max(min, start.w + dx);
        if (edge.includes('s')) shape.height = Math.max(min, start.h + dy);
        if (edge.includes('w')) {
          shape.width = Math.max(min, start.w - dx);
          shape.x = start.x + start.w - shape.width;
        }
        if (edge.includes('n')) {
          shape.height = Math.max(min, start.h - dy);
          shape.y = start.y + start.h - shape.height;
        }
      }
      this.sync();
    };
    const up = () => {
      div.removeEventListener('pointermove', move);
      div.removeEventListener('pointerup', up);
      div.removeEventListener('pointercancel', up);
    };
    div.addEventListener('pointermove', move);
    div.addEventListener('pointerup', up);
    div.addEventListener('pointercancel', up);
  }
}
