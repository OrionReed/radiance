# Profiling the WebGPU pages

How to use the shared profiler (`site/_shared/profiler.ts`), and what the browser can measure. Measured on a MacBook Pro
with an Apple M2 Pro (19-core GPU), macOS 15.6 and Chrome 157 (WebGPU through Dawn on Metal).

## What can be measured

| source | per pass | inside a pass | GPU counters | notes |
| --- | --- | --- | --- | --- |
| WebGPU `timestamp-query` | yes | no | no | rounded to 65.5 µs unless Chrome is started with a developer switch |
| Chrome tracing (Perfetto) | no | no | no | CPU side only: submits, map callbacks, shader compiles |
| Instruments, Metal System Trace | CPU-side encoder intervals only | no | no | the GPU timeline is empty |
| Xcode Metal frame capture | – | – | – | can't attach to Chrome's GPU process |

In Chrome on this Mac, GPU time per pass is all there is. Time per dispatch needs one pass per dispatch, which the
profiler's per-dispatch mode does.

## Timestamps

By default every timestamp is a multiple of 65 536 ns (2¹⁶), so a pass of a few µs reads as 0 or 65.5 µs. For full
resolution, quit Chrome fully and start it with one of these switches:

| Chrome switches | timestamps |
| --- | --- |
| none | multiples of 65 536 ns |
| `--enable-webgpu-developer-features` | full resolution |
| `--enable-unsafe-webgpu` | full resolution |
| `--disable-dawn-features=timestamp_quantization` | full resolution |
| `--enable-dawn-features=allow_unsafe_apis` | multiples of 65 536 ns |
| `--enable-unsafe-webgpu --enable-webgpu-developer-features` | multiples of 65 536 ns (not understood) |

```bash
open -a "Google Chrome" --args --enable-webgpu-developer-features
```

At full resolution the clock ticks every 41.67 ns (24 MHz). The profiler shows "65 µs steps" when every timestamp it
reads is rounded. Not checked: whether `chrome://flags/#enable-webgpu-developer-features` does the same.

**Passes only.** Metal on this GPU samples timestamps at stage boundaries only (not at dispatch, draw or blit
boundaries), and Chrome's `chromium-experimental-timestamp-query-inside-passes` feature isn't offered. A pass with several
dispatches gets one time.

**Render passes after compute.** A render pass that follows compute passes in the same command buffer has its begin
stamped when the command buffer starts, likely because its vertex stage starts at once. The profiler times each pass from
the later of its own begin and the previous pass's end.

## Using the profiler

A page makes one `Profiler` right after `requestDevice`, with `Profiler.features(adapter)` in `requiredFeatures`. Then,
per frame:

```ts
profiler.beginFrame();
const pass = profiler.computePass(enc, 'merge');            // or { timestampWrites: profiler.timestampWrites('draw') }
profiler.name(pass, 'merge 3');                             // names the next dispatch (per-dispatch mode only)
profiler.endFrame(enc);
device.queue.submit([enc.finish()]);
profiler.collect();
```

It reports:

- **per pass**: mean and minimum GPU time per label over the last 60 frames, summed over the label's passes in a frame;
- **per frame**: the sum of pass times (GPU), first begin to last end (span, which adds gaps and work outside passes),
  and CPU time from `beginFrame` to `endFrame` (encode);
- **per buffer**: bytes of every live buffer made on the device after the profiler (it wraps `device.createBuffer`);
- **bandwidth**: bytes per frame and GB/s, where a page sets `profiler.traffic.set(label, { read, write })` (bytes per
  pass).

It's off by default. While off, each call is one branch and it holds no GPU resources.

**Per-dispatch mode** runs each dispatch in a timed pass of its own, named by the latest `name()`, which gives per-level
times. Each extra pass costs about 12–15 µs of wall time, so it's off by default: turn it on with the panel's "per
dispatch" switch or `perDispatch = true`. Read per-dispatch times as proportions between levels; for totals, use whole
passes.

**The panel** graphs the GPU time of the last 120 frames as stacked bars, newest on the right, one colour per label
(levels of a stage alternate shades of its colour).

- **Scale:** the 90th percentile of the latest 30 frames, or the mean if that is taller. Taller frames are cut off.
- **Legend:** each label's mean ms over the last 60 frames, plus GB/s where traffic is declared, next to its band.
- **Header:** GPU ms, CPU encode ms, and the bytes of all buffers (hover for the largest).
- **Page line:** a line the page sets (`profiler.info`), such as its dispatches per frame.
- **CPU figure:** the mean encode time of the last 10 frames. It turns orange above 2 ms, or above twice its median plus
  0.5 ms. `performance.now()` has 0.1 ms steps on these pages (they aren't cross-origin isolated), so encode times under
  a millisecond are coarse.

### Pages

| page | how to open | labels |
| --- | --- | --- |
| `polygonal-hrc` | click the frame time (bottom right), or `?profile=1` | `rotation`, or per dispatch `extend n`, `merge n`, `gather` (level n is size 2ⁿ); `draw` |
| `hrc` | click the fps, or `?perf=1`; `t` prints a table to the console | `world`, `seed`, `extend`, `merge`, `blur`, `blit` (and the ratio and FFT passes when on), summed over the four directions; per dispatch `seed n`, `extend n`, `merge n` |
| `digital-hrc` | click the frame time (bottom right), or `?perf=1` | `quadrant`, or per dispatch `extend n`, `merge n` (`merge 1` is cascade 1's evaluation), `gather`; `draw` |
| `dual-hrc` | click the frame time (bottom right), or `?profile=1` | `sources`, `direction d`, `finish`; per dispatch the primal's `seed`, `extend n`, `merge n`, `blur`, or the dual's `near faces`, `spots`, `transmittance n`, `extensions n`, `gathering n`, `deposit seed`, `end mix`; `draw` |

While the profiler is open, Polygonal HRC, Digital HRC and Dual HRC solve the scene every frame and show GPU ms in place
of the frame time. Polygonal HRC and Dual HRC declare their traffic per level.

### Reading the numbers

- On a page the GPU idles between frames, so these are per-frame latencies at whatever clock the GPU is at, not
  throughput.
- Declared bytes count every array element a pass touches once: a lower bound on traffic, not DRAM traffic. GB/s can
  exceed the memory bandwidth when data stays in cache.

## Checks against wall time

Dense lights (25 % of pixels), default options, three channels; medians of 5 alternated rounds.

**Polygonal HRC** (`site/_harness/polygonal-hrc.html`, ten solves per submit), ms per solve:

| N | wall, off | wall, per pass | GPU sum, per pass | wall, per dispatch | GPU sum, per dispatch |
| --- | --- | --- | --- | --- | --- |
| 256 | 14.18 | 14.38 | 13.98 | 15.53 | 14.33 |
| 512 | 59.72 | 59.58 | 59.20 | 61.55 | 59.69 |

**HRC** (`HRC.burst`, 40 frames back to back, every frame read back), ms per frame:

| N | wall, off | GPU sum, per pass | GPU sum, per dispatch |
| --- | --- | --- | --- |
| 256 | 0.99 | 1.03 | 1.05 |
| 512 | 3.69 | 3.58 | 3.68 |
| 1024 | 15.99 | 16.34 | 16.50 |

- Pass timestamps agree with wall time to within 1–4 %.
- Per-level times sum to 1–3 % above the per-pass times, and once to 13 % above: the extra pass boundaries also slow the
  GPU work a little, by a varying amount.
- Timing each pass costs 0–1.4 % of wall time for Polygonal HRC, and 1.6–7 % for HRC (more at small N). Reading back
  every frame of a burst costs much more (+47 % at 256²), so only `readbacks` frames (6) are in flight; later ones are
  not read.
- Per-dispatch mode adds 3–12 % for Polygonal HRC (120–136 passes per solve instead of 8). For HRC at 256², where a level
  takes 25–60 µs, it's a large share.

## Other tools

**Chrome tracing.** Traced over the DevTools protocol with the `gpu`, `gpu.dawn`, `disabled-by-default-gpu.dawn`,
`gpu.service`, `disabled-by-default-gpu.service`, `disabled-by-default-gpu.device`, `viz`, `toplevel` and
`devtools.timeline` categories, the GPU process shows submits (40–200 µs of CPU each), map calls, pipeline creation by
WebGPU label, and shader compiles (`tint::msl::writer::Generate`, `MTLDevice::newLibraryWithSource`). No event carries
GPU execution time. Use it for CPU-side, IPC and shader-compile costs.

**Instruments.** `xcrun xctrace record --template 'Metal System Trace' --attach <pid>` attaches to Chrome's GPU process
(the `--type=gpu-process` helper) without admin rights, but records only CPU-side intervals: the GPU tables are empty,
also for a native Metal program, so the cause is this machine's setup (developer mode off, no admin rights). Encoders
show as "Compute Command N", without Chrome's labels.

**Xcode frame capture.** Chrome's GPU helper is signed with the hardened runtime and without `get-task-allow`, so Xcode
can't attach while System Integrity Protection is on.

Not tried: Instruments with admin rights or developer mode on; a Chromium or Dawn development build for frame capture
and labelled encoders.
