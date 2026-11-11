# The engine in a web browser, and on the GPU

The same C code runs natively (Linux, Windows) and in a web browser. In the browser it is
compiled to WebAssembly with **Emscripten**, draws on an HTML canvas, and uses the GPU
through **WebGPU**. The CPU backends are always there; the GPU takes over what it does
(rasterizing the scene, path tracing it, raymarching clouds and fog, simulating the air and
the cloud), and whatever it does not do yet (the rigid bodies) stays on the CPU.

| | native (Linux / Windows) | browser |
|---|---|---|
| build | `make` (gcc) | `make web` (Emscripten's `emcc`) |
| window | X11 / Win32 (`platform/window.c`, `window_win32.c`) | the `<canvas id="canvas">` (`platform/window_web.c`) |
| main loop | `while (frame())` | `emscripten_set_main_loop`: the browser calls `frame()` once per display refresh |
| threads | OpenMP | one thread (the CPU paths are slower than native) |
| GPU | WebGPU through wgpu-native after `make deps` (section 3), otherwise the CPU | WebGPU (`gpu/`), on by default |

## 1. Building and running

Emscripten, once (Windows: in the MSYS2 UCRT64 shell; Linux: any shell):

```sh
git clone https://github.com/emscripten-core/emsdk.git /f/Tools/emsdk   # or ~/emsdk on Linux
cd /f/Tools/emsdk && ./emsdk install latest && ./emsdk activate latest
```

Every new shell:

```sh
source /f/Tools/emsdk/emsdk_env.sh        # Linux: source ~/emsdk/emsdk_env.sh
cd /f/Projects/little_renderer
make web                                  # -> web/build/index.html, .js, .wasm, .data (the models)
make serve                                # http://localhost:8000 (python3, or uv)
```

Open **http://localhost:8000** in Chrome or Edge (WebGPU is on by default there; Firefox
and Safari have it in recent versions). Browsers do not run WebAssembly from `file://`:
use the server. `make web WEB_DRAGON=1` also packs the full 870k-triangle dragon.

Click the image, then the keys of `bin/view` work (USAGE.md section 4). The page's links
are presets; **any option of `bin/view` goes in the address**, `-name value` becoming
`name=value`:

```
http://localhost:8000/?scene=cloud&backend=raster&rastercloud=on        the cloud, GPU-raymarched
http://localhost:8000/?scene=spheres&cloud=on&fog=0.002&backend=raster  two media at once
http://localhost:8000/?sim=on&backend=raster                            physics: air and cloud on the GPU
http://localhost:8000/?sim=on&demo=blast&backend=raster                 the explosion
http://localhost:8000/?backend=rt&n=320                                 the path tracer (CPU)
http://localhost:8000/?scene=cloud&backend=raster&gpu=off               everything on the CPU
```

The page's "console" box shows what the program prints (the GPU it found, timings,
errors); the title line under the heading is the window title of `bin/view`.

## 2. The GPU: what runs where

| | GPU (`-gpu on`, the browser's default) | CPU |
|---|---|---|
| the rasterizer (`-gpuraster on`) | the same triangles, lamp, materials and 4 shading modes as `render_scene`, by a WebGPU render pipeline (`gpu/raster_gpu.c`) | `render_scene` |
| clouds and fog in the rasterizer (`-rastercloud on`) | the light grid (one compute thread per cell) and the march (one thread per pixel), the same model as `raster/volume_march.c` | `raster_draw_volumes` |
| the path tracer (`-gputrace on`) | the light transport of `trace_path_ex` (NEE + MIS, mirror, glass, Russian roulette, media by delta and ratio tracking) in one compute kernel over the scene's BVH (`gpu/trace_gpu.c`) | `trace_path_ex` |
| the air and the cloud of `-sim on` (`-gpufluid on`) | every kernel of `fluid_step`: obstacles, walls, forces, vorticity, advection, the multigrid pressure solve, MacCormack for the cloud | `fluid_step` |
| showing a CPU frame | the framebuffer is uploaded and drawn by one full-screen pass | `putImageData` on a 2D canvas |
| rigid bodies | | yes |

Measured at 640×640 in Chrome on an NVIDIA Ampere GPU: the path tracer takes 6 ms per pass
(one sample per pixel) on the Cornell box, 10 ms on the 100k-triangle dragon, 20 ms with
the cloud, 48 ms with the mirror and glass spheres in cloud and fog (the browser's CPU path
tracer: about 700 ms per pass on the box, on one thread). The physics demo runs the air,
the cloud and the drawing on the GPU in real time.

On a *software* GPU (SwiftShader, llvmpipe: headless browsers, machines without a GPU
driver) the simple kernels still win (cloud raster 159 → 42 ms per frame, cloud + fog
322 → 43 ms, the air 199 ms per step → real time), but the path tracer's long, branching
kernel is slower than the CPU's own code, so `-gputrace` is off by default there.

### 2.1 Options

| option | meaning | default |
|---|---|---|
| `-gpu on` | WebGPU draws on the canvas | on in the browser |
| `-gpu offscreen` | WebGPU draws into a texture that is read back and shown on a 2D canvas: one frame of latency, but works in browsers that can compute with WebGPU and not show it (some headless ones) | |
| `-gpu off` | the CPU does everything | builds without a GPU library |
| `-gpufluid on\|off` | with `-sim on`: the air and the cloud are simulated on the GPU | on |
| `-gpuraster on\|off` | the rasterizer's triangles are drawn on the GPU | on |
| `-gputrace on\|off` | the GPU path traces (while the scene stands still; during the physics the CPU does, at low resolution) | on, off for a software GPU |

If the browser takes the GPU away (a driver reset, the tab moved to another GPU) the
program says `GPU lost (...)` and the CPU carries on; with `-gpu on` the canvas cannot be
used for 2D drawing any more, so reload the page with `?gpu=off` or `?gpu=offscreen`.

### 2.2 How it is built (`engine/gpu/`)

| file | what it does |
|---|---|
| `gpu.h`, `gpu.c` | `Gpu`: the adapter and device (asked for asynchronously: `gpu_create`, then `gpu_state` every frame until `GPU_READY`), the canvas surface, device loss; helpers for compute kernels (`gpu_layout`, `gpu_compute`, `gpu_bind`, `gpu_dispatch`) |
| `view_gpu.h`, `view_gpu.c` | `GpuView`: shows a `Framebuffer` (or the GPU rasterizer's frame, or the path tracer's image) and raymarches the scene's media over it; computes each medium's bricks, box and maximum on the GPU; `gpu_view_capture` reads a frame back (screenshots, offscreen mode) |
| `raster_gpu.c` | the rasterizer: the triangles of `render_scene` (raster/render.h hands them out) pulled from storage buffers by the vertex shader, depth as the CPU's z-buffer, so the clouds composite the same way |
| `trace_gpu.c` | the path tracer: the scene's BVH, primitives, triangles and normals flattened into buffers, one thread per pixel per sample, samples summed in an accumulation buffer; bands of rows sized to keep each submission under ~45 ms |
| `fluid_gpu.h`, `fluid_gpu.c` | `GpuFluid`: takes over a `PhysWorld`'s air (`PhysWorld.air_step`); the state stays on the GPU, the view draws the cloud straight from there; only the air around each body (the drag) comes back every step |

The shaders are WGSL, written inside the C files with the `WGSL(...)` macro, so the engine
stays a C library with nothing to load at run time. Every kernel has an explicit binding
layout. The `gpu` layer may use `core`, `raster` (the raymarching model) and `phys`
(`make check-layers`), and nothing uses `gpu` except the apps.

Rules the GPU code follows, because a browser is asynchronous:

- nothing waits for the GPU: results come back through callbacks (`mapAsync`), a frame
  later; the drag on the bodies uses the air of the previous step;
- a slow GPU cannot fall behind without limit: at most 2 air steps are in flight (the time
  of a skipped step goes into the next one), one band of the path tracer, and offscreen at
  most one frame;
- a callback that arrives after its object was destroyed does nothing (live lists).

### 2.3 Checking the GPU against the CPU

```sh
make webtest      # -> web/build/test_gpu_fluid.html; open it through make serve
```

The path tracer and the rasterizer are checked image against image (headless Chromium,
`?gpu=offscreen`, the GPU against `&gputrace=off` / `&gpuraster=off`): the path traced
images agree in mean radiance within 1% on all the presets (boxes, spheres, cloud, cloud +
fog + glass, dragon), the remaining differences being noise; the rasterized ones agree
within 2/255 except on the pixels whose center lies exactly on an edge (the two break that
tie differently).

`examples/test_gpu_fluid.c` steps two identical worlds (a cloud, wind, buoyancy,
vorticity, a moving obstacle, an explosion) for 40 steps, one on the CPU and one on the
GPU, and compares the clouds voxel by voxel: they agree to about 1e-4 of the largest
density (the GPU and the CPU do not round alike, so not to the last bit). The clouds of
`-rastercloud` drawn by the GPU and by the CPU differ by 0.2 levels out of 255 on average.

## 3. The GPU natively (Windows, Linux)

The native programs (`bin/view`, `bin/render`, `bin/sim`) run the same GPU code as the
browser. The engine's `gpu/` layer is written against the standard WebGPU C API
(`webgpu.h`), so natively it only needs a library that implements that API. The Makefile
uses **wgpu-native** (the implementation inside Firefox, a plain C library). It reaches the
GPU through the driver's Vulkan or Direct3D 12, and nothing else gets installed.

```sh
make deps      # once: downloads wgpu-native v29.0.1.1 for this system into third_party/wgpu (git-ignored)
make           # from now on every build has the GPU (objects in build/float-gpu/)
bin/view       # the window is drawn by the GPU; bin/render and bin/sim render on it too
make NOGPU=1   # a CPU-only build even with the library there
make deps-clean   # remove it: builds are CPU-only again, exactly as before
```

The library is linked statically, so the programs need no DLL or `.so` beside them. They
grow from about 1.3 MB to 16 MB.

| program | on the GPU | measured (RTX 3070, Windows) |
|---|---|---|
| `bin/view` | everything the browser does: rasterizer, path tracer, clouds, the air | path tracing 5.7 ms per pass at 640×640 (the box), as fast as Chrome on the same GPU |
| `bin/render` | the path tracer and the rasterizer, offscreen, read back at full precision | box, 640×640, 256 spp: 2.0 s against 22.4 s on 8 CPU threads; cloud 2.0 s against 15.7 s; glass spheres in cloud and fog 2.6 s against 24.6 s |
| `bin/sim` | the air, the cloud, the frames | 60 rasterized frames: under 1 s against 7 s; 30 path traced frames (320², 64 spp): 11 s against 82 s |

The GPU images agree with the CPU's: average brightness within 0.05% on the path traced
scenes, the rasterizer identical except 0.04% of pixels (triangle edge ties), and the
physics film after 60 frames visually identical.

Natively there is no browser to deliver the GPU's callbacks, so the engine does it:
`gpu_state` (called every frame by the viewer) and `gpu_poll` run them, and `gpu_wait`
blocks until the GPU is done (`bin/render` and `bin/sim`, which wait for their images;
`bin/sim` also makes every air step run, `gpu_fluid_set_wait`, so a film does not depend on
the GPU's speed). With `-gpu off`, or if no GPU is found, everything runs on the CPU.

Two notes:
- **One rasterized image:** the CPU is faster than the GPU for `bin/render -backend raster`,
  3 ms against 55 ms, most of which is the GPU making its pipelines once.
- **Linker warnings on Windows:** with MSYS2's binutils 2.38 the link prints many
  `.drectve ... -exclude-symbols ... unrecognized` warnings. They are harmless: the library
  was built with a newer linker, and the old one ignores those hints. Updating MSYS2
  (`pacman -Syu`) makes them go away.

Later the engine can bring its own implementation of the same `webgpu.h` (a Vulkan backend),
and drop the library without touching the GPU code.

## 4. Using the engine in your own web page

The build is an ordinary Emscripten module. `web/shell.html` is the page around it: a
`<canvas id="canvas">`, a `#title` element (the window title) and a `Module` object whose
`arguments` come from the address. In your page, keep the canvas id, set `Module` before
loading `index.js`, and call into the program:

```js
var Module = { arguments: ['-scene', 'cloud', '-backend', 'raster'],
               canvas: document.getElementById('canvas'),
               print: console.log, printErr: console.warn };
// later: a screenshot of what the GPU shows, as a PNG data URL in window.lrCapture
Module._lr_capture();
```

To write a different program, compile your own C file with the engine like `apps/view.c`
(see the `web` target of the Makefile): the engine's API is the same as natively
(SDK.md), plus `gpu/` for the GPU.

## 5. Troubleshooting

| problem | cause / fix |
|---|---|
| `emcc: command not found` | `source /f/Tools/emsdk/emsdk_env.sh` in this shell |
| the page stays on "loading..." | open it through `make serve`, not as a file |
| `GPU not used (no adapter ...)` | the browser has no WebGPU: a recent Chrome or Edge; on Linux Chrome may need `--enable-unsafe-webgpu --enable-features=Vulkan` |
| black image, `GPU lost (device destroyed by the browser)` | the browser cannot show WebGPU on this canvas (seen in headless browsers): `?gpu=offscreen` |
| everything slow | the CPU paths run on one thread in the browser: use `backend=raster`, a smaller `n=`, and keep the GPU on |
