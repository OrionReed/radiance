// WGSL every scheme page's draw pass shares, next to DISPLAY_WGSL (./display.ts): the full-screen triangle, the display
// dither and the error view's colours. Each page writes its own fragment shader.
export const DRAW_WGSL = /* wgsl */ `
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)); return vec4f(p[i], 0.0, 1.0);
}
// display dither: triangular noise of ±1 LSB of the 8-bit output, per device pixel (removes banding, mean 0)
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453); }
fn dither(colour: vec3f, at: vec2f) -> vec4f {
  let noise = (hash(at) + hash(at + vec2f(17.3, 5.1)) - 1.0) / 255.0;
  return vec4f(colour + noise, 1.0);
}
// (scheme − reference) / reference in L · exposure, the reference floored at 0.01, in log bands of 0.1 decade: white at
// 1e-5 or less, darkest at 1; red where the scheme is brighter, blue where darker (ColorBrewer RdBu); magenta at the clip
fn errorColour(scheme: f32, exact: f32, exposure: f32, clip: f32) -> vec3f {
  let error = (scheme - exact) * exposure / max(exact * exposure, 0.01);
  if (abs(error) >= clip) { return vec3f(0.85, 0.2, 0.78); }
  let band = 3.0 * clamp((floor(log(max(abs(error), 1e-9)) / log(10.0) * 10.0) / 10.0 + 5.0) / 5.0, 0.0, 1.0);
  var stops = array<vec3f, 4>(vec3f(1.0), vec3f(0.957, 0.647, 0.510), vec3f(0.839, 0.376, 0.302), vec3f(0.404, 0.0, 0.122));
  if (error < 0.0) { stops = array<vec3f, 4>(vec3f(1.0), vec3f(0.573, 0.773, 0.871), vec3f(0.263, 0.576, 0.765), vec3f(0.020, 0.188, 0.380)); }
  let i = min(u32(band), 2u);
  return mix(stops[i], stops[i + 1u], band - f32(i));
}
`;
