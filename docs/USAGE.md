# little_renderer: how to use everything

This is the reference for every program, `make` target, command-line option, key and
tunable constant in this repository. How the code works is explained elsewhere:
`RASTERIZER.md` covers the rasterizer, `RAYTRACER.md` the path tracer, `VOLUMES.md` the
clouds, `PHYSICS.md` the physics engine, and `SDK.md` how to use the engine in your own
programs (all in `docs/`).

**Run every command from the top folder** (`~/Projects/little_renderer`): the programs look
for `models/` and write into `out/` relative to it.

| program | source | what it is |
|---|---|---|
| `bin/render` | `apps/render.c` | offline: renders the Cornell box into a `.png`, with the path tracer (`-backend rt`, default) or the rasterizer (`-backend raster`) |
| `bin/view` | `apps/view.c` | real time: a window you fly through; rasterizer while moving, path tracer when still; `-sim on` runs the physics live |
| `bin/sim` | `apps/sim.c` | offline: a physics simulation (rigid bodies, air, the cloud it carries), path traced frame by frame into `out/sim/` and a GIF |
| `bin/raster_milestones` | `examples/raster_milestones.c` | the rasterizer step by step: one image per milestone into `out/*.png` |
| `bin/hello_rt`, `bin/hello_raster`, `bin/custom_scene` | `examples/` | small programs using the library (see `SDK.md`); write into `out/` |
| `bin/mesh_rain` | `examples/mesh_rain.c` | random meshes rain into the room as convex hulls, without end (the oldest are reused past `-max`); the frame rate and the bodies asleep in the window's title. Options: `-rate meshes/s` (8), `-max bodies`, `-size units` (70), `-obj model`, `-gpu on\|off`, `-sleep on\|off` (on: bodies at rest fall asleep, PHYSICS.md 1.5), `-endless on\|off` (off: the rain stops at `-max` and the pile settles), `-seed n`, `-autoquit seconds`, `-n`/`-h` the window |
| `bin/test_medium` | `examples/test_medium.c` | unit test of the volume sampling (`make test`) |
| `bin/test_physics` | `examples/test_physics.c` | the physics against known answers (`make test`) |
| `tools/frames_to_gif.py` | `tools/` | a folder of `frame_NNNN.png` into an animated GIF (Pillow: `uv run` brings it, see 2.1) |
| `tools/gen_models.py` | `tools/` | writes the test models (cube, sphere, torus, dense torus) |

Everything is written from scratch in C. The only libraries are `libm`, `libX11` (for the
window) and, optionally, OpenMP, which spreads both the path tracer and the rasterizer
over the CPU cores.

---

## 1. Quick start

```sh
cd ~/Projects/little_renderer
make models                      # test models into models/ (once)
make                             # library, apps and examples into build/ and bin/
bin/view                         # fly around the Cornell box (window)
bin/render                       # path trace it to out/cornell_boxes.png (~7 s)
bin/render -backend raster       # rasterize it to out/cornell_boxes_raster.png (milliseconds)

bin/sim                          # physics: things fall through the cloud -> out/sim.gif (~10 min)
bin/view -sim on                 # the same, live in the window

make dragon-download             # Stanford dragon, 11 MB (once; already in the repo)
bin/view -scene mesh -obj models/dragon_recon/dragon_vrip.ply -rot -35
```

Every program prints its options with `-h`. The images are PNG files: any image viewer
or browser opens them.

---

## 2. Building (one `Makefile` in the top folder)

The same Makefile builds on Linux (gcc, make, libX11) and on Windows (section 2.1).

| command | what it does |
|---|---|
| `make` or `make all` | the library, the three apps, all examples |
| `make lib` | only `build/float/liblr.a` |
| `make apps` / `make examples` | only the three programs / only the examples |
| `make bin/render` (or any `bin/...`) | just that one program (and the library); on Windows `make bin/render.exe` |
| `make check-layers` | checks that `core` includes nothing else and `raster`, `rt`, `phys`, `platform` only `core` |
| `make models` | runs `tools/gen_models.py` |
| `make dragon-download` | downloads the Stanford dragon into `models/dragon_recon/` |
| `make clean` | deletes `build/` and `bin/` |

Shortcuts that build and run:

| command | what it does |
|---|---|
| `make demo` | rasterizer: `bin/raster_milestones`, one image per milestone into `out/` |
| `make run` | quick previews of the scenes into `out/`, path traced and rasterized |
| `make view` | opens `bin/view`; pass options with `make view ARGS="-scene spheres -backend rt"` |
| `make view-dragon` | opens `bin/view` on the dragon (`ARGS=...` works here too) |
| `make final` | slow, clean 600×600 / 1024 spp renders of every scene |
| `make dragon` | renders the dragon (orange and glass) into `out/` |
| `make bench` | none vs octree vs BVH on the torus; compares the 3 images |
| `make bench-dragon` | octree vs BVH on the Stanford dragon; compares the images |
| `make test` | `bin/test_medium` and `bin/test_physics`: the cloud's sampling and the physics against exact answers |
| `make sim` | the drop demo: 120 frames, 320×320, 64 spp → `out/sim/` and `out/sim.gif` (~20 min) |
| `make sim-blast` | the explosion demo: 90 frames → `out/sim_blast/` and `out/sim_blast.gif` |
| `make cloud` | the cloud, 600×600, 1024 spp → `out/cornell_cloud_final.png` |
| `make cloud-frames` | 48 animation frames (`-time` 0 to 23.5 s, 300×300, 64 spp) into `out/cloud_frames/` (`FRAMES=n` to change) |

Build flags, combinable with any target:

| flag | effect |
|---|---|
| `make DOUBLE=1` | double precision everywhere: `build/double/liblr.a` and `bin/double/...` (the float build stays in `bin/`) |
| `make SINGLE=1` | no OpenMP: both backends run on one thread (easier to debug) |
| `make SAN=1` | address + undefined-behaviour sanitizers: catches bad memory access; ~10× slower |
| `make CFLAGS="-O2 -DOCT_LEAF_SIZE=8"` | your own flags, e.g. to change a compile-time constant |

Run `make clean` first when you change `SINGLE`, `SAN` or `CFLAGS` (`DOUBLE` has its own
folders and needs no clean).

**float or double?** float (the default) is 15–25% faster for the path tracer and matches
double up to noise. In double the three acceleration structures give bit-identical
images; in float a handful of pixels can differ (ties, see RAYTRACER.md section 3), which
`make bench` reports.

What gets built:

| output | from |
|---|---|
| `build/float/liblr.a` | `engine/*/*.c` (headers next to them; include `engine/lr.h`) |
| `bin/render`, `bin/view`, `bin/sim` | `apps/*.c` + the library |
| `bin/hello_raster`, `bin/hello_rt`, `bin/custom_scene`, `bin/raster_milestones`, `bin/test_medium`, `bin/test_physics`, `bin/hello_physics` | `examples/*.c` + the library |

---

### 2.1 Windows (MSYS2)

The engine builds natively on Windows with MSYS2's gcc: same Makefile, same commands. The window is plain Win32 (`engine/platform/window_win32.c`); programs get a
`.exe` (`bin/render.exe`, ...), and you can still type `bin/render` in the MSYS2 shell.

1. Install MSYS2 (msys2.org), then in its terminal, once:
   `pacman -S --needed mingw-w64-ucrt-x86_64-gcc make`
2. Open the **MSYS2 UCRT64** shell (not "MSYS2 MSYS"), go to the folder and build:
   ```sh
   cd /f/Projects/little_renderer       # F:\Projects\little_renderer
   make
   make models                          # test models (with uv, see below)
   bin/view                             # the window
   make test
   ```
3. Python tools (`make models`, the GIF of `bin/sim`, `make gif`) run through **uv**
   when it is installed: `uv run tools/frames_to_gif.py ...` reads the script's header
   (PEP 723) and installs Pillow into a cached environment by itself. Without uv they use
   `python3` (then `pip install pillow`). uv installs itself into `%USERPROFILE%\.local\bin`,
   which the MSYS2 shell does not have in its PATH; the Makefile and `bin/sim` look there
   anyway. To type `uv` yourself in that shell, add it to the PATH
   (`export PATH="$PATH:$(cygpath -u "$USERPROFILE")/.local/bin"`, e.g. in `~/.bashrc`),
   or start MSYS2 with `MSYS2_PATH_TYPE=inherit`.

Visual Studio (MSVC) is not supported: the code needs OpenMP 3 (tasks in the BVH build)
and gcc builtins.

### 2.2 The GPU (natively: make deps)

```sh
make deps      # once: wgpu-native (a WebGPU library) into third_party/wgpu
make           # every build now uses the GPU when one is found; NOGPU=1 to build without
```

`bin/view`, `bin/render` and `bin/sim` then rasterize, path trace and simulate the air on
the GPU. `-gpu off` runs everything on the CPU. `make deps-clean` removes the library.
Details and measurements are in docs/WEB.md, section 3.

### 2.3 Web browser (Emscripten, WebGPU)

`bin/view` also runs in a browser, on the GPU (docs/WEB.md has everything):

```sh
source /f/Tools/emsdk/emsdk_env.sh      # once per shell (Linux: ~/emsdk/emsdk_env.sh)
make web                                # -> web/build/
make serve                              # then open http://localhost:8000
make webtest                            # the GPU air against the CPU's: web/build/test_gpu_fluid.html
```

The options go in the address: `http://localhost:8000/?scene=cloud&backend=raster`.

---

## 3. `render`, `view` and `sim` options

The three programs accept the same options; the defaults differ where noted. Values are
given as `-option value`. The physics options are in section 3.1.

### Backend

| option | meaning | default |
|---|---|---|
| `-backend rt` | path tracer | `render` ✓ |
| `-backend raster` | rasterizer: local lighting only (no shadows, reflections or bounced light), but milliseconds per image | |
| `-backend hybrid` | `view` only: rasterizer while the camera moves, path tracer once it stops | `view` ✓ |
| `-shading phong\|gouraud\|flat\|wireframe` | rasterizer shading | phong |
| `-rastercloud on\|off` | rasterizer: draw clouds and fog by raymarching over the frame (VOLUMES.md 7.1) | on |
| `-march n` | raymarching: most steps per ray (4..4096) | 128 |
| `-lightres n` | raymarching: light grid cells along a medium's longest side (2..512) | 48 |
| `-marchscale 1\|2` | raymarching: one ray per pixel, or per 2×2 block upsampled (3–4× faster) | 1 (view: 2) |

With `-backend raster`, `render` ignores the path tracer options (`-s`, `-d`, `-clamp`,
`-accel` ...) and builds no acceleration structure; it prints the raymarching times.

### Image and quality

| option | meaning | default | tips |
|---|---|---|---|
| `-n size` | width = height in pixels | 400 (view: 640) | time grows with the pixel count |
| `-width px`, `-height px` | set width and height separately | | the camera adapts to the aspect ratio |
| `-s samples` | render: light paths per pixel. view: stop refining after this many | 128 (view: 4096) | noise falls like 1/√s: 4× the samples = half the grain |
| `-d bounces` | maximum bounces per path | 50 | 3–5 is fast but darker (less indirect light); Russian roulette ends most paths early anyway |
| `-clamp max` | caps the brightness of a single sample | 0 (off) | `-clamp 4` removes fireflies (glass, mirrors) but darkens caustics: biased |
| `-wavefront on\|off` | path tracing on the GPU: small kernels, each doing one step (extend, shade, shadow rays) of all paths at once, instead of one kernel per path. The same image; slower on today's scenes (docs/PERFORMANCE.md R1), kept for heavier shading. Native builds only for now | off |
| `-denoise on\|off` | path tracing on the GPU: the image is denoised (an edge-avoiding filter guided by each pixel's noise, normal, distance and albedo: WEB.md 2.4) | `view`, `sim`: on; `render`: off | at 4–16 samples it looks like 256–1024; it blurs a little and is slightly biased (±1%), so a final `render` stays unbiased unless asked; the CPU path tracer does not denoise |
| `-threads n` | CPU threads, for both backends | 0 = all cores | same as the `OMP_NUM_THREADS` environment variable; the image is the same for any value |
| `-sampler sobol\|random` | where a path's random numbers come from: the Sobol sequence, Owen-scrambled, shifted per pixel by blue noise; or plain PCG32 (RAYTRACER.md 5.10) | sobol | sobol: less noise for the same samples (up to 1.35×, more while moving in the viewer), and finer-grained noise |
| `-mis on\|off` | multiple importance sampling of the lamp: direct sampling and bounces both count, weighted | on | helps near the lamp and in fog; `off` = the old estimator |
| `-caustics lt\|pt` | caustics by light tracing (traced from the lamp and connected to the camera) or by path tracing alone; on the GPU too (WEB.md 2.6) | `render`, `view`: lt (the CPU viewer: pt), `sim`: pt | lt removes most fireflies of glass and mirrors, unbiased (RAYTRACER.md 5.14); `-mis off -caustics pt -sampler random` reproduces the old images bit for bit |
| `-photons x` | light paths per camera path for `-caustics lt` | 1 | more = smoother caustics, slower |
| `-o file` | render: output PNG. view: prefix for screenshots (`<prefix>_001.png`, ...) | `out/cornell_<scene>.png`, `..._raster.png` for the rasterizer (view: `out/view`) | the folder must exist (`make` targets create `out/`) |

### Scene

| option | meaning | default |
|---|---|---|
| `-scene boxes` | the classic Cornell box: a tall box and a short box | ✓ |
| `-scene showcase` | one model of each material: a mirror Stanford dragon (the full 871k triangles), a diffuse blue cube and a glass torus, under a cloud (the cloud options shape it, `-wind` blows it); needs `models/` | |
| `-scene spheres` | a mirror sphere and a glass sphere | |
| `-scene mesh` | a model from a file, standing in the middle of the box | |
| `-scene cloud` | the empty box with a volumetric cloud floating in it (path tracer, see VOLUMES.md) | |
| `-scene empty` | just the room and the lamp | |
| `-obj file` | the model for `-scene mesh`: `.obj`, or ASCII `.ply` (binary PLY is not supported) | `models/torus.obj` |
| `-mat diffuse\|mirror\|glass` | material of the model | diffuse (warm orange) |
| `-rot degrees` | turns the model around the vertical axis | 0 |
| `-instances n` | n instances of the model (one mesh placed n times by transforms) in any scene: in a row (`-scene spheres`: beside the spheres first); on the GPU too (WEB.md 2.7) | 0: the model as world triangles, `-scene mesh` only |

The model is always scaled so its largest side is 330 units (the box is 555), centered,
and stood on the floor. Files without normals get smooth normals computed automatically.

### Cloud (path tracer)

A procedural cumulus floating in the middle of the box, below the lamp (between y = 150
and 440). How it works and what the numbers mean: `VOLUMES.md`. The rasterizer draws it
too, by raymarching (section "Backend": `-rastercloud`, `-march`, `-lightres`,
`-marchscale`), so it shows in `view` while moving.

| option | meaning | default |
|---|---|---|
| `-cloud on\|off` | add the cloud to any scene (`-scene cloud` always has it) | off |
| `-density sigma` | extinction at full density, per scene unit (thicker = more opaque, slower) | 0.08 |
| `-albedo a` or `-albedo r,g,b` | fraction of the light scattered at each interaction (the rest is absorbed) | 0.99 |
| `-g value` | Henyey-Greenstein anisotropy, −1..1: 0 uniform, > 0 forward (real droplets ≈ 0.85) | 0.3 |
| `-detail value` | 0 = smooth blobs .. 1 = wispy, eroded edges | 0.6 |
| `-seed n` | which cloud (puff layout and noise) | 1 |
| `-time seconds` | animation: the billows drift with the wind | 0 |
| `-wind x,y,z` | (`render`, `view` without `-sim`) a wind blows the cloud: its top is pushed downwind by a quarter of its height per 100 units/s, its detail drawn out into streaks along it (with `-sim` the air itself moves, 3.1) | 0,0,0 |
| `-cloudres n` | voxels along the longest side of the cloud's box (4..1024) | 128 |
| `-fog sigma` | fill the room with a uniform haze of this extinction (works with or without a cloud; try 0.002) | 0 = off |

### Camera

| option | meaning | default |
|---|---|---|
| `-cam x,y,z` | camera position | `278,278,-800` (in front of the open side) |
| `-look x,y,z` | the point the camera looks at | `278,278,0` |
| `-fov degrees` | vertical field of view | 40 |

Coordinates: the box spans 0..555 on every axis, y is up, the open side faces −z, and
x = 555 (the red wall) appears on the **left**. The lamp is under the ceiling at
y = 554. The easiest way to get numbers is to fly there in `view` and press **C**.

### Acceleration structure (path tracer)

| option | meaning | default |
|---|---|---|
| `-accel auto` | BVH from 64 primitives up, otherwise none | ✓ |
| `-accel none` | test every primitive for every ray (fine for the 18-quad box) | |
| `-accel octree` | the octree (`rt/octree.c`) | |
| `-accel bvh` | the BVH (`rt/bvh.c`) | |
| `-octdepth n` | octree max depth; 0 = automatic (log₄(n/16) + 2, clamped to 6..14) | 0 |
| `-bvhleaf n` | BVH: max primitives per leaf (SAH may stop earlier) | 4 |
| `-bvhbins n` | BVH: buckets per axis when evaluating SAH splits (2..256) | 16 |
| `-bvhsplit sah\|middle` | BVH: surface area heuristic, or cut the longest axis in the middle | sah |
| `-bvhwidth 2\|4` | BVH: children per node for the CPU path tracer; 4 = half as deep, 4 boxes tested per step (the dragon 1.9x faster). The GPU uses the binary tree, faster there (`LR_GPU_WIDE=1` to try the 4-wide one) | 4 |

The three structures give the same image (bit for bit in the double build); only the
speed changes (`make bench`, `make bench-dragon`). The program prints the build
statistics (nodes, leaves, depth, build time) at start-up. The BVH is built in parallel
(the dragon in about 0.25 s on 8 threads), and gives the same tree for any number of threads.

### view only

| option | meaning | default |
|---|---|---|
| `-preview n` | `rt` mode: while moving, path trace at 1/n resolution (1 = always full resolution) | 4 |
| `-movespp n` | samples per pixel of each low-resolution frame while moving | 1 |
| `-speed units` | movement speed in scene units per second | 300 |
| `-autoquit seconds` | quit after this time and save a screenshot (scripts, benchmarks); prints the frame rate, and with `-orbit` the `render` command of the screenshot's camera | 0 = never |
| `-photoncache on\|off` | GPU light-traced caustics while moving: from a world-space photon cache (on), or traced exactly every frame (off) | on |
| `-orbit deg/s` | the camera swings by itself, ±25° around the point it looks at, at this speed: benchmarks and tests of the image while moving | 0 = off |
| `-history n` | GPU path tracing while moving: samples per pixel carried from one frame to the next (reprojected); 12 at most while the physics runs; 0 = start over on every move | 32 |
| `-gpu on\|off\|offscreen` | render and simulate on the GPU (WebGPU: browser builds, or natively after `make deps`). `offscreen`: draw into a texture read back to the window (WEB.md 2.1) | on when the build has a GPU |
| `-gpufluid on\|off` | with `-sim on` and the GPU: simulate the air and the cloud there | on |
| `-gpurigid on\|off` | with `-sim on` (and `sim`) and the GPU: simulate the rigid bodies there (PHYSICS.md 1.4) | on |
| `-gpuraster on\|off` | with the GPU: rasterize the scene there | on |
| `-gputrace on\|off` | with the GPU: path trace there; in `rt` mode every frame, while the camera or the physics move too (reprojected and denoised: WEB.md 2.5) | on, off for a software GPU |

### 3.1 Physics (`sim`, and `view -sim on`)

How it works: `PHYSICS.md`. The demos always use the empty room (`-scene` is ignored);
the cloud options (`-density`, `-detail`, `-seed`, `-cloudres` ...) shape the cloud the air
carries, `-fog` adds haze, the camera options work as usual.

| option | meaning | default |
|---|---|---|
| `-sim on\|off` | `view` only: run the physics demo live | off |
| `-demo drop` | balls and boxes fall through the cloud one after another, balloons rise through it | ✓ |
| `-demo blast` | everything rests on the floor under the cloud; an explosion at t = 1 s | |
| `-balls n`, `-boxes n`, `-balloons n` | how many of each (4096 bodies in all at most); in the drop demo one appears every 0.4 s, or, with more than 50, all of them within 20 s | 5, 3, 3 |
| `-meshes n` | n copies of the model (`-obj`, 120 units) fall too (`-demo blast`: rest on the floor), as their convex hulls (PHYSICS.md 1.2b); `-mat` their material | 0 |
| `-collider on\|off` | the model (`-obj`, 260 units, turned by `-rot`) stands in the middle of the room, fixed: things bounce off and rest on its triangles, the air flows around it | off |
| `-wind x,y,z` | the air drifts towards this velocity (units/s). The room is closed but for its front, so a wind mostly swirls | 0,0,0 |
| `-fluidres n` | air cells along the room's side (8..256); cost grows with n³ | 48 (view: 32, and `-cloudres` 96) |
| `-buoyancy a` | the cloud rises with acceleration a × density (units/s²); try 100 | 0 |
| `-vorticity e` | vorticity confinement: keeps the swirls alive, 0 = off | 0.25 |
| `-physseed n` | where things start | 1 |
| `-frames n` | `sim`: frames to render | 120 |
| `-fps n` | `sim`: frames per second of the animation (the physics steps are at most 1/60 s) | 30 |
| `-gif on\|off` | `sim`: also write `<-o>.gif` (with `uv run`, else python3 + Pillow) | on |
| `-o folder` | `sim`: where the frames go (`frame_0000.png` ...) | `out/sim` |

`sim`'s image defaults are smaller than `render`'s: `-n 320 -s 32`, `-caustics pt`.
`-backend raster` renders the frames with the rasterizer and the cloud raymarched: the
whole animation in about 40 s, a quick look before the 20-minute path traced version.
`view -sim on` uses a coarser grid unless told otherwise (`-fluidres 32 -cloudres 96`)
and one physics step of up to 1/30 s per frame; the hybrid mode rasterizes while the
simulation runs and path traces when it is paused; in `rt` mode the preview resolution
is 1/2 (`-preview`) and `-movespp 2` gives less grain per frame.
With the GPU the bodies are simulated there (`-gpurigid`), so hundreds of them cost the
CPU nothing: `bin/view -sim on -backend raster -balls 400 -boxes 200 -balloons 0` stays at the
screen's frame rate (164 frames/s on an RTX 3070; 142 with `-gpurigid off`). Set
`LR_GPU_TIMING=1` to have the GPU passes timed (WEB.md 3).

Environment variables for measuring (docs/PERFORMANCE.md keeps what they showed):

| variable | effect |
|---|---|
| `LR_GPU_TIMING=1` | native GPU builds: time every GPU pass (path tracer per pass, rigid bodies per stage), printed at the end |
| `LR_WF_PROFILE=1` | with `-wavefront on` and `LR_GPU_TIMING=1`: the time of each wavefront kernel |
| `LR_GPU_WIDE=1` | the GPU path tracer walks the CPU's 4-wide trees (default: binary) |
| `LR_GPU_TLAS=lbvh` | the GPU builds the top level over the instances itself every frame (default: it takes the CPU's SAH tree) |
| `LR_TLAS_REFIT=1` | the top level is refit while the instances only move, instead of rebuilt |

---

## 4. `view` controls

| key | action |
|---|---|
| `W` `A` `S` `D` | move forward / left / back / right |
| `E` / `Q` | move up / down |
| left mouse drag | look around |
| arrow keys | look around |
| mouse wheel | zoom (field of view ±2°) |
| `Shift` (hold) | move 4× faster |
| `+` / `-` | movement speed ×1.5 / ÷1.5 |
| `M` | mode: hybrid → rt → raster |
| `V` | raster shading: Phong → Gouraud → flat → wireframe |
| `[` / `]` | one bounce less / more (restarts refining) |
| `B` | cycle the acceleration structure: none → octree → BVH (rebuilds it, prints stats) |
| `F` | `rt` mode: low-resolution preview while moving on / off |
| `1` `2` `3` `4` | `rt` mode: preview resolution 1/1, 1/2, 1/4, 1/8 |
| `R` | back to the starting camera |
| `X` | path tracing on the GPU: denoiser on / off |
| `P` | save what the window shows as PNG (the path traced image, or the raster frame) |
| `C` | print a `render` command that path traces exactly this view |
| `H` | print the controls in the terminal |
| `Esc` | quit |

With `-sim on`:

| key | action |
|---|---|
| `Space` | pause / resume the simulation (paused, the image refines as usual) |
| `K` | an explosion in the cloud |
| `N` | drop a steel ball |
| `T` | restart the demo |

The title bar always starts with the frame rate (frames shown per second, and the time of
one), then shows the mode and either the raster shading and frame time, or (path
tracing) the acceleration structure, samples per pixel so far (/ the `-s` limit), time
per full pass, current resolution (1/1 when still), bounces and speed.

**How it works.**
- `hybrid` (default): while you move, the rasterizer draws every frame. About 0.1 s after
  you stop, the path tracer starts and paints over it, one sample per pixel per pass, top
  to bottom, so the image gets cleaner the longer you wait.
- `rt`: while you move, every frame is one path traced sample per pixel at 1/4 resolution,
  so it follows the mouse but looks grainy. When you stop, it restarts at full resolution.
- `raster`: the rasterizer only.

Path tracing stops at `-s` samples per pixel and then sits idle.

With `-sim on`, every frame advances the simulation by the time that passed (at most
1/20 s, so a slow machine shows slow motion). In hybrid mode the rasterizer draws it (the
cloud raymarched) until you pause with Space; then the path tracer refines. In `rt` mode
every frame is one path traced pass, and the window shows a running average of the last
passes (motion blur, but much less grain).

Clouds and fog are raymarched in the raster frames (at half resolution, `-marchscale`);
the title bar then shows the light grid and march times.

**Display.** It uses X11. On a GNOME/Wayland desktop it runs through Xwayland, which is
already there. Start it from a terminal in your desktop session. Over SSH or from a
script, set `DISPLAY=:0`.

---

## 5. Recipes

**Quick look vs final quality**
```sh
bin/render -backend raster                 # milliseconds, local lighting: check the camera
bin/render -n 200 -s 16                    # < 1 s, very grainy
bin/render -n 400 -s 128                   # ~7 s, the default
bin/render -n 800 -s 1024 -o out/big.png   # ~4 min, clean
```
(times for the boxes scene on 8 cores; glass, mirrors and big meshes are slower)
```sh
# the loop to learn what a parameter does: change ONE thing, compare the images
bin/render -s 64 -d 2  -o out/d2.png       # direct light + one bounce: dark, hard
bin/render -s 64 -d 50 -o out/d50.png      # full global illumination
```

**Raster vs path traced, same view**
```sh
bin/render -scene spheres -backend raster -o out/a.png
bin/render -scene spheres -o out/b.png     # shadows, reflections, refraction, color bleeding
```

**Find a view in the window, render it offline**
1. `bin/view` (add your `-scene ...` options), fly to a nice spot, press **C**.
2. Copy the printed line, e.g.
   `bin/render -scene boxes -cam 278.0,278.0,-497.0 -look 278.0,278.0,3.0 -fov 40 -n 640 -s 512 -o out/from_viewer.png`,
   and change `-n` / `-s` as you like.

**Your own model**
```sh
bin/view   -scene mesh -obj ~/Downloads/bunny.obj -rot 30
bin/render -scene mesh -obj ~/Downloads/bunny.obj -rot 30 -mat glass -clamp 4 -s 512
```

**Compare acceleration structures**
```sh
bin/render -scene mesh -obj models/torus_dense.obj -accel none   -n 100 -s 8
bin/render -scene mesh -obj models/torus_dense.obj -accel octree -n 100 -s 8
bin/render -scene mesh -obj models/torus_dense.obj -accel bvh    -n 100 -s 8
bin/render -scene mesh -obj models/torus_dense.obj -accel bvh -bvhsplit middle -n 100 -s 8
```
Or press **B** in `view` (path tracing) and watch "ms/pass" in the title bar.

**float vs double**
```sh
make && make DOUBLE=1
time bin/render -scene mesh -n 256 -s 256 -o out/f.png
time bin/double/render -scene mesh -n 256 -s 256 -o out/d.png
```

**A cloud**
```sh
bin/render -scene cloud                                   # the cloud alone
bin/render -scene boxes -cloud on -density 0.05           # in the classic box
bin/render -scene cloud -g 0.85 -density 0.2 -s 1024      # physical droplets, denser
bin/render -scene cloud -albedo 0.9,0.6,0.35 -detail 0.9  # tinted, ragged
bin/view -scene cloud -backend rt                         # fly around it
```

**Less noise from glass and mirrors**: the default `render` already traces caustics from
the lamp (`-caustics lt`) and uses MIS. For more: more samples (`-s 2048`), more light paths
(`-photons 4`), or `-clamp 4` (fast, slightly darker caustics, biased).

**Fog and light**
```sh
bin/render -scene spheres -fog 0.002 -s 512          # haze glowing around the lamp
bin/render -scene cloud -fog 0.002 -s 512            # a cloud in haze: two media at once
```

**Physics**
```sh
bin/sim -frames 30 -s 8 -n 200            # a quick draft (about a minute)
bin/sim                                   # the drop demo, ~10 min -> out/sim.gif
bin/sim -demo blast -frames 90 -o out/sim_blast
bin/sim -balls 10 -boxes 0 -balloons 6 -physseed 7
bin/sim -buoyancy 100 -vorticity 0.5      # the cloud rises and curls
bin/view -sim on                          # live; Space pauses, K = boom, N = another ball
bin/view -sim on -fluidres 48 -cloudres 128 # the offline grids (slower)
bin/sim -backend raster                   # the drop demo in ~40 s, cloud raymarched
bin/render -scene cloud -backend raster   # the raymarched cloud vs ...
bin/render -scene cloud                   # ... the path traced one
make gif FRAMES_DIR=out/sim FPS=30        # (re)make the GIF (uv run tools/frames_to_gif.py ...)
```

**Use fewer cores** (to keep the computer responsive): `-threads 4`.

**Debug a crash**: `make clean && make SAN=1 SINGLE=1`, then run the same command with
a small `-n` and `-s`.

---

## 6. Constants you can tweak in the code

These have no command-line option. Change them and run `make` again.

| what | where | value |
|---|---|---|
| wall colors, light color/strength (15), camera default | `engine/core/scene.c`, `scene_cornell_box` | Cornell / Shirley values |
| box sizes and angles, sphere positions | `engine/core/scene.c` | see the comments |
| model size (330) and position, model color | `engine/core/scene.c`, `SCENE_MESH` branch | |
| ray offsets against self-hits (`RT_EPS`, `RT_SHADOW_MARGIN`) | `engine/rt/tracer.c` (or `-DRT_EPS=`) | float 2e-3 / 2e-2, double 1e-4 / 1e-3 |
| BVH: subtrees bigger than this are built in parallel (`BVH_PAR_MIN`) | `engine/rt/bvh.c` | 32768 |
| octree mailbox size (`OCT_MAILBOX`), or `-DOCT_NO_MAILBOX` to turn it off | `engine/rt/octree.c` | 64 |
| majorant grid brick size (`VOLUME_BRICK`, or `-DVOLUME_BRICK=n`), `-DMEDIUM_NO_BRICKS` to turn it off | `engine/core/volume.h`, `engine/rt/medium.c` | 8 |
| fog material (albedo, g) and extent | `engine/core/scene.c`, `scene_cornell_box` | 0.95, 0.4 |
| light paths per random-number stream (`LT_BATCH`) | `engine/rt/tracer.c` | 4096 |
| cloud position in the box | `engine/core/scene.c`, `scene_cornell_box` | (70,150,110) .. (485,440,450) |
| cloud shape: puff sizes, base height, warp, erosion, contrast, wind | `engine/core/volume.c`, `cloud_init`, `cloud_density`, `cloud_default_params` | |
| bounces before Russian roulette (`RR_START`) | `engine/rt/tracer.c` | 3 |
| octree leaf size (`OCT_LEAF_SIZE`) | `engine/rt/octree.h` (or `-DOCT_LEAF_SIZE=`) | 16 |
| octree depth limit / node limit | `engine/rt/octree.h` | 14 / 8,000,000 |
| BVH depth limit (`BVH_MAX_DEPTH`) | `engine/rt/bvh.h` | 100 |
| SAH costs (traversal 1, intersection 1) | `engine/rt/bvh.c`, `sah_split` | 1 / 1 |
| `-accel auto` threshold (`AUTO_ACCEL_MIN_PRIMS`) | `engine/app/options.h` | 64 |
| raster scene lighting: ambient, lamp, camera fill | `engine/raster/render.c` (`SCENE_AMBIENT`, `SCENE_LAMP`, `SCENE_FILL`) | 0.08 / 0.70 / 0.22 |
| sphere tessellation for the rasterizer | `engine/raster/render.c` (`SPHERE_STACKS`, `SPHERE_SLICES`) | 16 × 32 |
| mirror / glass look in the rasterizer | `engine/raster/render.c`, `render_scene` | |
| `render_mesh` light and ambient | `engine/raster/render.c` (`AMBIENT`, `render_default_params`) | 0.12 |
| hybrid: still time before path tracing starts | `apps/view.c` (`HYBRID_SETTLE_S`) | 0.12 s |
| turn speed / mouse sensitivity | `apps/view.c` | 1.5 rad/s / 0.004 rad per pixel |
| milestone image size | `examples/raster_milestones.c` | 640×480 |
| contact slop, Baumgarte factor, bounce threshold | `engine/phys/rigid.c` (`CONTACT_SLOP`, `BAUMGARTE`, `BOUNCE_THRESHOLD`) | 0.5 / 0.2 / 60 |
| speculative contacts: smallest margin, fastest push out | `engine/phys/rigid.c` (`SPECULATIVE_MIN`, `MAX_PUSH`; the GPU's copies in `gpu/rigid_gpu.c`) | 0.1 / 300 |
| sleeping: still below this speed, for this long; the speed that wakes; on or off | `engine/phys/rigid.c` (`SLEEP_SPEED`, `SLEEP_TIME`, `WAKE_SPEED`; the GPU's copies in `gpu/rigid_gpu.c`), `RigidWorld.sleep` | 5 units/s, 0.5 s, 20 units/s, on |
| solver iterations, damping, wall restitution / friction | `engine/phys/rigid.c`, `rigid_init` | 16, angular 0.1 /s, 0.4 / 0.5 |
| rigid substeps, air density, drag coefficients | `engine/phys/world.c`, `phys_init` | 4, 0.0012, sphere 0.47 / box 1.05 |
| pressure V-cycles, wind rate, which faces are open | `engine/phys/fluid.c`, `fluid_create` | 5, 0.3 /s, front |
| multigrid smoothing sweeps | `engine/phys/poisson.c`, `poisson_init` | 3 + 3 |
| the demos: bodies, sizes, densities, colors, the explosion | `engine/app/simdemo.c`, `sim_setup` | |
| largest physics step | `engine/app/simdemo.c` (`MAX_PHYS_DT`; the viewer sets 1/30) | 1/60 s |
| raymarching: multiple-scattering octave weight / depth | `engine/raster/volume_march.c` (`OCT_WEIGHT`, `OCT_DEPTH`) | 0.7 / 0.3 |
| raymarching: octaves, brightness, ambient, step length | `engine/raster/volume_march.c`, `march_default_params` | 3, 2, estimated, 1 voxel |

---

## 7. The rasterizer milestones

```sh
bin/raster_milestones [model.obj] [texture.png]   # default: models/torus.obj + a checkerboard
```

Writes `out/0_gradient.png` ... `out/9_near_clipping.png`, one per milestone, and prints
the render time of each shading mode. For the rasterizer in real time, use
`bin/view -backend raster` (key `V` cycles the shading).

---

## 8. Troubleshooting

| problem | cause / fix |
|---|---|
| `make: gcc: No such file or directory` (Windows) | open the **MSYS2 UCRT64** shell, not "MSYS2 MSYS"; install gcc: `pacman -S mingw-w64-ucrt-x86_64-gcc make` |
| `could not make the GIF` | install uv (docs.astral.sh/uv), or `pip install pillow` for python3; the frames are in the folder anyway |
| `cannot open X display` | run `bin/view` from a terminal inside your desktop, or `export DISPLAY=:0` |
| `could not load models/torus.obj` | run `make models`, and run the programs from the top folder (or pass `-obj`) |
| `could not write out/...` | run from the top folder, or create the folder: `mkdir -p out` |
| `ply: only ASCII PLY files are supported` | convert the file to ASCII PLY or OBJ (e.g. with Blender or MeshLab) |
| white speckles | fireflies from glass/mirror paths: more `-s`, or `-clamp 4` |
| image too dark with small `-d` | fewer bounces = less indirect light; use the default 50 |
| `view` feels slow on big models | hybrid: the rasterizer draws the full dragon at ~12 fps on 8 threads; use a smaller window (`-n 400`), a decimated model (`dragon_vrip_res2.ply`), or press `M` for `rt` mode (1/4-resolution preview) |
| everything is black | the camera is outside the box looking away; press `R` in `view`, or check `-cam`/`-look` |
| `view -sim on` is slow / jerky | the physics takes ~30 ms per step at the viewer's defaults: `-fluidres 24 -cloudres 64`, a smaller window, or `Space` to pause and look |
| the raymarched cloud looks banded or blocky | more steps (`-march 256`), full resolution (`-marchscale 1`), a finer light grid (`-lightres 96`) |
| the browser build: `GPU not used`, `GPU lost`, a black canvas | docs/WEB.md, section 5 |
| `GPU not used (no GPU adapter ...)` natively | no Vulkan / Direct3D 12 driver for the GPU: update the GPU driver; `-gpu off` meanwhile |
| Windows link: many `.drectve ... unrecognized` warnings | harmless (old binutils); `pacman -Syu` in MSYS2 updates them |
| linker errors / garbage after changing `DOUBLE` | apps and library must use the same precision; the Makefile keeps them in separate folders, so use `bin/double/...` for `DOUBLE=1` |
