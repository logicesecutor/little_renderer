# little_renderer

A small rendering engine written from scratch in plain C, with two backends that draw the
same scene: a **rasterizer** (fast, local lighting) and a physically based **path tracer**
(slow, global illumination), plus a **physics engine** (rigid bodies, and the air that
carries a volumetric cloud). One static library, one header, apps and examples on top.
The only dependencies are `libm`, the window system (`libX11` on Linux, Win32 on Windows)
and optionally OpenMP. It builds with gcc on Linux and with MSYS2's gcc on Windows
(docs/USAGE.md, section 2.1), and with Emscripten for **web browsers**. Everywhere it can
also run on the **GPU** through WebGPU: in the browser by itself, natively after `make deps`
(docs/WEB.md).

## Layout

```
little_renderer/
├── engine/                    the engine -> build/float/liblr.a, everything via lr.h
│   ├── lr.h                   the one header to include
│   ├── core/                  shared by everything: math (float or double), random numbers,
│   │                          images, camera, meshes (.obj/.ply), scenes, textures,
│   │                          noise and volumes (procedural clouds)
│   ├── raster/                backend 1: the rasterizer (triangles -> pixels, z-buffer;
│   │                          clouds and fog by raymarching)
│   ├── rt/                    backend 2: the path tracer (rays, octree, BVH, light transport,
│   │                          participating media: clouds)
│   ├── phys/                  physics: rigid bodies, the air (stable fluids, multigrid),
│   │                          the cloud it carries, the coupling between them
│   ├── gpu/                   the GPU backend (WebGPU, shaders in WGSL): rasterizer, path
│   │                          tracer, clouds raymarched, the air and the cloud simulated
│   ├── platform/              the window: X11, Win32, or an HTML canvas
│   └── app/                   what the programs share: options, the physics demos, frames
├── apps/
│   ├── render.c               offline: one image into a PPM, -backend rt|raster -> bin/render
│   ├── view.c                 real time: raster while moving, path tracing when still -> bin/view
│   └── sim.c                  physics simulation, path traced frame by frame -> bin/sim
├── examples/                  small programs using the library (start here for the SDK)
│   ├── hello_raster.c         the Cornell box, rasterized           -> bin/hello_raster
│   ├── hello_rt.c             the Cornell box, path traced          -> bin/hello_rt
│   ├── custom_scene.c         your own objects, both backends       -> bin/custom_scene
│   ├── raster_milestones.c    the rasterizer step by step           -> bin/raster_milestones
│   ├── hello_physics.c        a ball falling through a cloud        -> bin/hello_physics
│   ├── mesh_rain.c            endless random meshes, fps in the title -> bin/mesh_rain
│   ├── test_medium.c          checks the cloud sampling (make test) -> bin/test_medium
│   ├── test_physics.c         checks the physics (make test)        -> bin/test_physics
│   └── test_gpu_fluid.c       the GPU air against the CPU's (make webtest, in a browser)
├── docs/
│   ├── USAGE.md               every command, option, key and tunable constant
│   ├── SDK.md                 how to use the engine in your own program
│   ├── RAYTRACER.md           how the path tracer works (code + maths)
│   ├── VOLUMES.md             volumetric clouds: physics, algorithms, next steps
│   ├── PHYSICS.md             the physics engine: rigid bodies, air, coupling (code + maths)
│   ├── RASTERIZER.md          how the rasterizer works (files + milestones)
│   ├── WEB.md                 the engine in a browser, and on the GPU
│   └── PERFORMANCE.md         ideas and plans for speed, how to measure, what was measured
├── web/                       shell.html (the page around the program); make web -> web/build/
├── tools/                     gen_models.py (test models), frames_to_gif.py (animations)
├── models/                    .obj test models + the Stanford dragon
├── out/                       rendered images
└── Makefile                   builds everything; `make clean` removes build/ and bin/
```

Each layer only uses the ones above it in this list: `core` uses nothing, `raster`, `rt`,
`phys` and `platform` use only `core`, and the backends and the physics never include each
other; `gpu` may use `core`, `raster`, `rt` and `phys` (`make check-layers` verifies it).

## Quick start

```sh
make models                    # test models into models/ (needs uv or python3, once)
make                           # the library, both apps, the examples
bin/view                       # fly around the Cornell box (WASD + mouse, M = mode, Esc quits)
bin/render                     # path trace it to out/cornell_boxes.ppm
bin/render -backend raster     # rasterize it to out/cornell_boxes_raster.ppm
bin/render -scene cloud        # a volumetric cloud in the box -> out/cornell_cloud.ppm
bin/sim                        # physics: balls fall through the cloud -> out/sim.gif
bin/view -sim on               # the same, live (Space pause, K explosion, N another ball)
bin/sim -backend raster        # the whole animation in ~40 s: rasterized, cloud raymarched
bin/hello_rt                   # the 40-line SDK example -> out/hello_rt.ppm
make DOUBLE=1                  # the same in double precision, into bin/double/
make deps && make              # the GPU natively (WebGPU through wgpu-native, docs/WEB.md)
make web && make serve         # in a browser, on the GPU: http://localhost:8000 (docs/WEB.md)
```

Every program prints its options with `-h`. All commands and options are in
**[docs/USAGE.md](docs/USAGE.md)**. To write your own program on the engine, see
**[docs/SDK.md](docs/SDK.md)**.

## One engine, two backends

| | rasterizer (`raster/`) | path tracer (`rt/`) |
|---|---|---|
| question it answers | which pixels does each triangle cover? | what does each pixel see? |
| lighting | local: ambient + lamp + highlights, no shadows; clouds and fog raymarched with a light grid | global: soft shadows, color bleeding, reflection, refraction, caustics (light traced from the lamp, with MIS), volumetric clouds and fog |
| speed (Cornell box) | milliseconds per frame | seconds per clean image |
| used for | the real-time preview, quick looks | final images |

Both draw the same `Scene` (`core/scene.h`), built from the same meshes and cameras with
the same math. `real` is `float` by default (faster), `double` with `make DOUBLE=1`.

## Physics

`phys/` moves rigid spheres and boxes (separating axes and face clipping, sequential
impulses with friction and warm starting) and the air of the room (Stable Fluids on a MAC
grid, multigrid pressure, vorticity confinement), which carries the cloud's density
(MacCormack advection) in the very `Volume` the path tracer renders. Bodies push the air;
the air drags bodies and lifts balloons; an explosion makes the air expand. See
**[docs/PHYSICS.md](docs/PHYSICS.md)**.

## Credits

The Stanford dragon (`models/dragon_recon/`, `make dragon-download`) is from the Stanford
3D Scanning Repository, © Stanford University Computer Graphics Laboratory. It may be
used for non-commercial purposes with credit; please don't animate, morph or break it
apart.
