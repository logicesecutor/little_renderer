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
| rigid bodies of `-sim on` (`-gpurigid on`) | forces, pairs, contacts (balls, boxes, hulls, fixed meshes, walls), the contact solver with mass splitting (2.8) | `rigid_step` |

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
| `-gpurigid on\|off` | with `-sim on`: the rigid bodies are simulated on the GPU (2.8) | on |
| `-gpuraster on\|off` | the rasterizer's triangles are drawn on the GPU | on |
| `-gputrace on\|off` | the GPU path traces; in `rt` mode every frame, the camera and the physics moving or not (2.5) | on, off for a software GPU |

If the browser takes the GPU away (a driver reset, the tab moved to another GPU) the
program says `GPU lost (...)` and the CPU carries on; with `-gpu on` the canvas cannot be
used for 2D drawing any more, so reload the page with `?gpu=off` or `?gpu=offscreen`.

### 2.2 How it is built (`engine/gpu/`)

| file | what it does |
|---|---|
| `gpu.h`, `gpu.c` | `Gpu`: the adapter and device (asked for asynchronously: `gpu_create`, then `gpu_state` every frame until `GPU_READY`), the canvas surface, device loss; helpers for compute kernels (`gpu_layout`, `gpu_compute`, `gpu_bind`, `gpu_dispatch`) |
| `view_gpu.h`, `view_gpu.c` | `GpuView`: shows a `Framebuffer` (or the GPU rasterizer's frame, or the path tracer's image) and raymarches the scene's media over it; computes each medium's bricks, box and maximum on the GPU; `gpu_view_capture` reads a frame back (screenshots, offscreen mode) |
| `raster_gpu.c` | the rasterizer: the triangles of `render_scene` (raster/render.h hands them out) pulled from storage buffers by the vertex shader, depth as the CPU's z-buffer, so the clouds composite the same way |
| `trace_gpu.c` | the path tracer: the scene's BVH, primitives, triangles and normals flattened into buffers, one thread per pixel per sample, samples summed in an accumulation buffer; bands of rows sized to keep each submission under ~45 ms; with the samples, the denoiser's moments and guide |
| `denoise_gpu.c` | the denoiser (2.4) |
| (in `trace_gpu.c`) | real time: each image's guide pass and the reprojection of the last image's samples (2.5) |
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

### 2.4 The denoiser (`-denoise`, key `X`)

The GPU path tracer's image can be denoised on the GPU (`gpu/denoise_gpu.c`): an
edge-avoiding à-trous wavelet filter (Dammertz et al. 2010) steered by each pixel's noise,
as in SVGF (Schied et al. 2017) without its temporal part.

- **What the path tracer adds** (`trace_gpu.c`, when the GPU's buffers allow 64 bytes per
  pixel, so up to about 1920×1080 at the default 128 MB limit): per pixel, the sums of `l`
  and `l²` (`l` the luminance of a sample over the pixel's albedo), and a guide traced once
  along the ray through the pixel's centre: the normal, distance and albedo of the first
  diffuse surface, *through* mirrors and glass, so reflections and refractions keep their
  edges. Lamps and the sky are not filtered; a pixel whose surface a medium hides
  (transmittance under ½) is filtered on luminance only.
- **prep:** the pixel's mean over its albedo (the irradiance: smooth where the colour is
  not) and the variance of that mean, from the pixel's own samples once it has 4, before
  that from its 7×7 neighbours on the same surface.
- **5 à-trous passes**, taps 1, 2, 4, 8, 16 pixels apart (a 65×65 footprint), each tap
  moved at random inside its hole (otherwise a grid prints on smooth walls). A neighbour
  counts by how alike it is: normal (cosine to the power 128), distance (relative to how
  fast it changes there) and luminance, relative to the noise of the *noisier* of the two
  pixels. With the centre's noise alone, a bright outlier keeps its energy to itself and
  the image darkens by several %; with the larger one, outliers are spread and the mean
  stays within ±0.6% of the reference.
- **last:** times the albedo again. As samples add up the variance falls, the luminance
  weights tighten, and the filter fades out by itself.

Measured natively (512×512, RTX 3070; RMSE in 8-bit levels against 4096 spp):

| scene | 4 spp noisy → denoised | 64 spp noisy → denoised |
|---|---|---|
| boxes | 13.9 → 3.0 | 4.7 → 2.2 |
| spheres (mirror, glass) | 17.0 → 6.2 | 11.9 → 4.3 |
| cloud | 32.1 → 4.7 | 13.3 → 3.2 |

The 5 passes cost well under a millisecond at 640×640 on that GPU, and run only when a band
of samples has finished. `view` and `sim` denoise by default, `render` does not (a final
image stays unbiased unless `-denoise on`); `gpu_view_trace_denoise` turns it on in your
own program.

### 2.5 Real-time path tracing (`-backend rt`, `-history`)

In `rt` mode with the GPU, `view` path traces every frame, also while the camera moves and
while the physics runs (bodies falling, the cloud blown by the air), and each frame is
denoised. This is the temporal half of SVGF (Schied et al. 2017), written for this engine;
no outside denoiser is used.

- **Each frame** that something moved, `gpu_view_trace_move` begins a new image: a guide
  pass traces every pixel's guide (2.4) for the new camera and the new geometry, then a
  reprojection pass gives each pixel the samples its surface had in the previous image.
  The pixel's guide point (along its ray, at its first diffuse surface's distance: through a
  mirror, the point behind it, which moves as the reflection does; in a medium, the mean of
  4 collision distances) is projected into the old camera. Of the 4 old pixels around it,
  those that saw the same thing count, with bilinear weights: same kind, normal within ~25°,
  distance from the old eye within 2% (25% in a medium) plus how fast the distance changes
  there. A pixel that matches none (it was hidden, or off screen) starts from nothing and
  shows the rasterized scene until its first sample.
- **History:** a pixel keeps at most `-history` samples (32), 12 while the physics runs.
  The cap ages old samples: a change of light (a moving shadow, the cloud) shows within a
  few frames, and the blur of blending 4 pixels again every frame stays short. Taking the
  nearest old pixel instead does not blur, but it duplicates pixels under subpixel motion
  and the noise lines up in streaks; it measured worse.
- **Then** the frame's new samples are added (as many passes as fit in ~10 ms) and the
  denoiser smooths the result. A band of samples never waits for another: on a GPU too slow
  for a whole pass per frame, the bands go on round the image, and a move waits until the
  running band ends, so the frame shows the last image together with its own camera.
- The physics moves its bodies every step: they are instances (2.7), so only their
  transforms are uploaded and the top level rebuilt; the scene's own BVH stays. The cloud
  stays on the GPU (`-gpufluid`), and so do the bodies (`-gpurigid`, 2.8).

Measured on an RTX 3070 (Windows, native), with the camera swinging at 15°/s
(`-orbit 15`); RMSE in 8-bit levels against 1024 spp at the screenshot's camera, 400×400:

| | reprojection, raw | reprojection + denoiser |
|---|---|---|
| spheres (mirror, glass) | 13.0 | **8.0**, 104 frames/s |
| cloud | 12.5 | **6.2**, 102 frames/s |

Before (every move started the image over), at 640×640: 22.8 on the spheres, 69.3 on the
cloud, whose lower rows never got a sample while moving.

At 640×640: spheres 106 frames/s, cloud 47 (the cloud's passes cost ~20 ms each); the
physics demo (`-sim on`, 11 bodies, air and cloud on the GPU) 50 frames/s, 35 at
1024×1024, against 12 frames/s for the CPU's quarter-resolution preview. The denoiser costs
~2 ms per frame at 640×640; the guide and reprojection passes ~2 ms in the cloud scene.

Try it: `bin/view -backend rt -scene spheres`, `bin/view -backend rt -sim on`, or in a
browser `?backend=rt&sim=on`. In `hybrid` mode (the default) the rasterizer still draws while
the camera moves, and the path tracer takes over once it settles.

### 2.6 Caustics on the GPU (`-caustics lt`, `-photoncache`)

- **The light tracer** (`ltrace` in `trace_gpu.c`) is `light_trace_caustics` of
  `rt/tracer.c` as a compute kernel, one thread per light path. A path leaves the lamp,
  passes mirrors and glass, and each diffuse surface it reaches is connected to the camera
  and added to its pixel. The sums are 64-bit 32.32 fixed point made of two `u32` atomics
  with a carry (WGSL has no 64-bit atomics), so the image does not depend on the order the
  threads add in.
- **No double counting:** the path tracer leaves out exactly those paths (`skip_caustics`).
  The caustics are added after the denoiser, so they stay sharp. One pass of light paths
  (`-photons` per pixel, 1 by default) follows each pass of the path tracer.
- **Measured** (the spheres, 400×400, RMSE against 8192 spp): 64 spp path traced 11.8,
  light traced on the GPU 5.0 (the CPU's 5.0), for 0.32 s instead of 0.24 s. The mean is
  within 0.1%. `render`, `view` (and `bin/sim -caustics lt`) light-trace them by default
  on the GPU; the CPU viewer still path traces them.
- **The photon cache** (`view`, while moving): light paths leave their power at every
  diffuse surface they reach, in a world-space hash grid. Each slot holds 3 × 64-bit sums,
  a photon count and its cell's key. Cells are claimed with an atomic compare-exchange and
  linear probing, so two places never share a slot.
- **Cell sizes:** cells are the scene's size / 256, ×4 and ×16, per main normal axis. A
  pixel that sees a diffuse surface directly reads, bilinearly along that surface, the
  finest size holding at least 16 photons (adaptive density estimation). The grid fades by
  0.9 each frame something moves, so what moved fades out.
- **Standing still,** the exact light tracer takes over (weight P/(P+8) after P passes),
  and the image converges to the unbiased one.
- **While moving** (`-orbit 15`, raw / denoised RMSE): photon cache 6.5 / 6.7, exact light
  tracer each frame 9.0 / 8.9, path traced caustics (`-caustics pt`) 8.8 / 6.0. The cache
  gives by far the best raw frame; once denoised, path traced caustics come out ~10% ahead,
  because the denoiser smooths them, while the cache's cells keep a little bias (a few %
  near the lamp, a slightly blurred caustic). `-photoncache off` traces the exact caustics
  every frame instead.

### 2.7 Instances: meshes that move (`-instances n`)

A mesh can be kept in its own space and placed in the world by transforms (SDK.md, section
3); `-instances n` puts n copies of the model in any scene (`-scene spheres`: beside the
spheres). Moving one changes only its transform, never its triangles.

- **The CPU path tracer** has a BVH per mesh; a ray is moved into an instance's space when
  it meets the instance's world box (`intersect_instances` in `rt/accel.c`).
- **The GPU path tracer** keeps everything in its existing buffers (no new bindings):
  `nodes` = [the meshes' BVHs | the top level | the world's BVH], `aux` = [their leaves'
  shape ids | the top level's | the vertex normals | the world's], and after the
  primitives two records per instance (its inverse and forward transforms, its mesh's BVH
  root, material, world box). A ray walks the world's BVH, then the top-level BVH over the
  instances, moving into each instance it reaches.
- **The top level** is the CPU's SAH tree when it is current (`scene_tlas`); otherwise it
  is built on the GPU every frame, by a linear BVH builder (Karras 2012, `BUILD` in
  `trace_gpu.c`): 30-bit Morton codes of the box centres, a bitonic sort (one dispatch per
  step), the hierarchy from the codes' common prefixes (each internal node independently),
  then the boxes from the leaves up, one pass per tree level. A dynamic mesh's BVH is
  rebuilt on the GPU after its triangles changed (`scene_mesh_changed`), in place of the
  CPU's, by PLOC (Meister & Bittner 2018, `PLOC` in `trace_gpu.c`): from the same sorted
  leaves, clusters merge with their nearest of 16 neighbours on each side, round after
  round (PERFORMANCE.md R4b). Either way the image is the same (only how fast rays find
  the triangles changes).
- **One traversal loop** walks all of it: a stack entry is a node and the instance whose
  space it is in, the ray is moved into an instance's space when its entries come up. So the
  hit tests exist once in the compiled kernel: written as two nested walks, the browser's
  first compile of the path tracer (D3D12 on Windows) took ~50 s; as one loop, ~15 s (then
  it is cached), and the instanced dragon traces 20% faster natively.
- **The GPU rasterizer** draws each instance as its mesh's triangles (uploaded once) with
  the transform picked by the draw's first instance.
- **Physics** (PHYSICS.md 1.2b): `bin/sim -obj <model> -meshes n -collider on` drops convex
  hulls of the model on a fixed copy of it; they are instances, so every frame only their
  transforms move to the GPU. The GPU air sees them through each shape's 48³ occupancy
  bits, uploaded once (the obstacle kernel's 5th binding).
- **Measured** (RTX 3070, `view -orbit 15`, 400²): the dragon as an instance 119 frames/s,
  as world triangles 131. Against the CPU: the same images (RMSE 2.6–3.0 at 64 spp, equal
  means; 40 instances too).
- **Shapes as instances.** An instance can also be the unit sphere (`INST_SPHERE`) or the
  unit box [−1, 1]³ (`INST_BOX`), scaled, turned and moved by its transform: every ball
  and box of the physics is one (PHYSICS.md 4.1). The path tracers hit them exactly (the
  GPU's top-level leaf recognises the two shapes by their kind and tests them in the
  instance's space, no BVH below); the rasterizers draw a tessellated unit sphere and box,
  uploaded once. The CPU path tracer has a top-level BVH over the instances too
  (binned SAH, rebuilt by `scene_update_instances` when they moved). With 64 bodies the
  rasterized `view -sim on` went from 21 to 6 ms per frame (before, every body's primitives
  were tessellated again each frame), and the GPU scene upload from 0.16 to 0.02 ms.

### 2.8 Rigid bodies on the GPU (`-gpurigid`, on by default)

With the GPU, the physics' bodies are stepped there as well as the air (`gpu/rigid_gpu.c`,
PHYSICS.md 1.4): about 15 compute passes per substep (pairs, contacts of balls, boxes,
hulls and fixed meshes, the contact solver with mass splitting), the state kept on the GPU
and read back a step late, so the CPU never waits for it in the viewer. `-gpurigid off`
keeps them on the CPU. In the browser it runs as natively (the dragons button too).

| | CPU | GPU (RTX 3070) |
|---|---|---|
| 1024 bodies, one 1/60 s step | 17.9 ms | 1.7 ms |
| `view -sim on`, 600 bodies, raster | 142 frames/s (19 ms of physics on the CPU) | 164 (the screen's limit; 0.6 ms) |
| `sim -demo blast`, 300 bodies, physics of the film | 3.4 s | 1.7 s |

A note for writing WGSL kernels that browsers accept (Tint is stricter than wgpu-native's
naga): a `workgroupBarrier` must be in uniform control flow, so a loop whose count lives in
workgroup memory reads it with `workgroupUniformLoad`; an array or matrix indexed by a
variable must be a `var`, not a `let`; and a kernel may bind at most 8 storage buffers, so
the layouts skip the bindings a kernel does not use (`-` in `gpu_layout`).

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

**The path tracer's kernel and its memory.** What one GPU thread keeps (its registers and
local arrays) decides how many threads the GPU can run at once. The BVH walk's stack was
three arrays of 64 entries (node, instance, distance); it is now one array of 40 node
entries (entering an instance pushes an EXIT mark and the mesh's root, since instances are
never nested; no distance is kept, the children's boxes are tested against the closest hit
anyway). That alone made every scene 2-5x faster on an RTX 3070 (docs/PERFORMANCE.md).
`-wavefront on` splits the tracer into small kernels per bounce (extend, shade, shadow
rays); it gives the same image but is slower here, and it is not yet tried in browsers.

To see where the GPU's time goes, set `LR_GPU_TIMING=1` before starting a native program:
it asks the adapter for timestamp queries, and the GPU rigid bodies print the time of each
of their passes when they are freed (the `GpuTimer` in `gpu/gpu_internal.h`: one query per
pass, read back without waiting, averaged over the run). Browsers hide timestamps by
default, so it is native only.

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
