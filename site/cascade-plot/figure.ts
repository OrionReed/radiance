// What a figure is: one question, its own settings (with their controls and URL keys), and how it draws.
// Figures share nothing but the look (theme, line weight, labels), which the page owns.
import type { Diagram, Grid, Schedule } from './plot';

export type Value = number | string | boolean;

/** one setting of a figure: its control, and its range (a maximum may depend on the figure's other settings) */
export interface Option<T> {
  key: keyof T & string;
  label: string;
  kind: 'slider' | 'choice' | 'toggle';
  min?: number | ((s: T) => number);
  max?: number | ((s: T) => number);
  step?: number;
  choices?: [Value, string, string?][];
  cols?: number;
  enabled?: (s: T) => boolean;
  format?: (v: number) => string;
  title?: string;
}

export interface Ctx<T> { N: number; K: number; sch: Schedule; s: T }

export interface Figure<T extends object = Record<string, Value>> {
  id: string;
  name: string;
  /** the one question the figure answers */
  question: string;
  /** how to read it */
  about: string;
  defaults: T;
  options: Option<T>[];
  /** sheet size in columns, and its height in rows (N when unset) */
  N: (s: T) => number;
  rows?: (s: T) => number;
  sheets: (c: Ctx<T>) => number;
  /** plan view: sheets per row (one row when unset) */
  cols?: (c: Ctx<T>) => number;
  draw: (D: Diagram, c: Ctx<T>) => void;
  /** text for the sidebar: numbers, formulas, the selected thing */
  info?: (c: Ctx<T>) => string;
}

/** a figure with its own settings type, stored with the others */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const defineFigure = <T extends object>(f: Figure<T>) => f as unknown as Figure<any>;

export const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));

// ── setting groups several figures have (each figure keeps its own copy of the values) ──

/** how the sheets are laid out: side by side and flat, or stacked and seen at an angle (isometric) */
export interface Proj { mode: 'plan' | 'stack'; layer: number; turn: number; tilt: number }
export const PROJ: Proj = { mode: 'plan', layer: 1.15, turn: 20, tilt: 35 };
export const projOptions = <T extends Proj>(): Option<T>[] => [
  { key: 'mode', label: 'Sheets', kind: 'choice', choices: [['plan', 'Flat'], ['stack', 'Isometric']] },
  { key: 'layer', label: 'Layer height', kind: 'slider', min: 0.3, max: 2.5, step: 0.05, format: (v) => v.toFixed(2), enabled: (s) => s.mode === 'stack' },
  { key: 'turn', label: 'Turn', kind: 'slider', min: -180, max: 180, step: 1, format: (v) => `${Math.round(v)}°`, enabled: (s) => s.mode === 'stack' },
  { key: 'tilt', label: 'Tilt', kind: 'slider', min: 10, max: 85, step: 1, format: (v) => `${v}°`, enabled: (s) => s.mode === 'stack' },
];

/** what marks the sheet's lattice: probes, pixel corners, or nothing */
export interface Marks { marks: Grid }
export const marksOption = <T extends Marks>(): Option<T> => ({
  key: 'marks', label: 'Marks', kind: 'choice', choices: [['probes', 'Probes'], ['corners', 'Pixel corners'], ['none', 'None']],
});

/** which directions each cascade keeps */
export interface Dirs { dirs: string }
export const dirsOption = <T extends Dirs>(): Option<T> => ({
  key: 'dirs', label: 'Directions', kind: 'choice', cols: 3,
  choices: [['all', '2L+1', 'every rise'], ['half', 'L+1', 'every other rise, one-row ends'], ['halfwide', 'L+1 wide', 'every other rise, wide ends'], ['wide8', '8 wide', '8 directions, wide ends'], ['wide4', '4 wide', '4 directions, wide ends']],
});

/** the image size */
export interface Size { n: number }
export const sizeOption = <T extends Size>(sizes = [8, 16]): Option<T> => ({
  key: 'n', label: 'Image size', kind: 'choice', choices: sizes.map((n) => [n, `${n} x ${n}`] as [Value, string]),
});
