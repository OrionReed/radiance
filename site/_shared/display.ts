// The display every scheme page shares. A pixel shows its mean radiance L = fluence / 2π (lights have emission 1)
// times the exposure E · 2^stops, as
//   tone map  sRGB(1 − e^(−E L)), per channel
//   banded    log10(E L) of the luminance in bands of 0.1 decade over [−2.2, 0], viridis
// E is one constant for every scene, size and page; the slider moves it in stops.

/** the exposure E at 0 stops */
export const EXPOSURE = 1;

/** viridis (matplotlib), 17 samples, sRGB */
const VIRIDIS = [
  [0.267, 0.0049, 0.3294], [0.2823, 0.095, 0.4173], [0.2788, 0.1755, 0.4834], [0.259, 0.2515, 0.5247],
  [0.2297, 0.3224, 0.5457], [0.1994, 0.3876, 0.5546], [0.1727, 0.4488, 0.5579], [0.149, 0.5081, 0.5573],
  [0.1276, 0.5669, 0.5506], [0.1206, 0.6258, 0.5335], [0.1579, 0.6838, 0.5017], [0.2461, 0.7389, 0.452],
  [0.3692, 0.7889, 0.3829], [0.516, 0.8312, 0.2943], [0.6785, 0.8637, 0.1895], [0.8456, 0.8873, 0.0997],
  [0.9932, 0.9062, 0.1439],
];
const BANDS = 22; // bands of 0.1 decade from 10^−2.2 to 1

/** WGSL: `displayed(x, map)` for x = E · L (rgb) and map 0 (tone map) or 1 (banded); `srgbEncode` */
export const DISPLAY_WGSL = /* wgsl */ `
fn srgbEncode(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308)); }
const VIRIDIS = array<vec3f, 17>(${VIRIDIS.map((c) => `vec3f(${c.join(', ')})`).join(', ')});
fn viridis(t: f32) -> vec3f {
  let scaled = clamp(t, 0.0, 1.0) * 16.0; let i = min(u32(scaled), 15u);
  return mix(VIRIDIS[i], VIRIDIS[i + 1u], scaled - f32(i));
}
fn displayed(x: vec3f, map: f32) -> vec3f {
  if (map > 0.5) {
    let luminance = dot(max(x, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    let band = clamp(floor((log2(max(luminance, 1e-12)) * 0.30103 + 2.2) * 10.0), 0.0, ${BANDS - 1}.0);
    return viridis(band / ${BANDS - 1}.0);   // viridis is already sRGB
  }
  return srgbEncode(1.0 - exp(-max(x, vec3f(0.0))));
}
`;

const viridisCss = (t: number) => {
  const s = Math.max(0, Math.min(1, t)) * 16, i = Math.min(Math.floor(s), 15), f = s - i;
  return `rgb(${VIRIDIS[i].map((a, k) => Math.round((a + (VIRIDIS[i + 1][k] - a) * f) * 255)).join(',')})`;
};
/** the banded view's legend: the bar's CSS background, ticks [value, position in %], caption */
export const BANDED_LEGEND = {
  background: `linear-gradient(to right, ${Array.from({ length: BANDS }, (_, i) => `${viridisCss(i / (BANDS - 1))} ${(i / BANDS) * 100}% ${((i + 1) / BANDS) * 100}%`).join(',')})`,
  ticks: [-2.2, -2, -1.5, -1, -0.5, 0].map((v) => [String(v).replace('-', '−'), ((v + 2.2) / 2.2) * 100] as [string, number]),
  caption: 'log10(E L), 10 bands per decade',
};

/** the map control, as every page shows it */
export const MAP_CONTROL = {
  label: 'map',
  key: 'b',
  title: 'tone map: sRGB(1 − e^(−E L)); banded: log10(E L), 10 bands per decade. L = fluence / 2π, E = 2^stops',
  options: [['tone', 'tone map'], ['banded', 'banded']] as [string, string][],
};
export const MAP_INDEX: Record<string, number> = { tone: 0, banded: 1 };

/** the exposure slider, in stops from E = 1 */
export const EXPOSURE_CONTROL = { label: 'exposure (stops)', min: -4, max: 10, step: 0.25 };
