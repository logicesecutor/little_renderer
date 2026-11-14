# Performance: ideas, plans and measurements

The single place where ideas for making the engine faster are kept, with what each one is
expected to give, what has been measured, and what was decided. **Keep it up to date as the
engine develops:** add new ideas when they come up (from profiling, a paper, a benchmark),
move an idea to *Done* with its measured result when it is implemented, and write down when
an idea was tried and rejected (and why), so it is not tried again blindly.

How each part works is in the other docs (RAYTRACER.md, PHYSICS.md, WEB.md); this file is
only about speed.

## How to measure

Measure before and after every change, on the same machine, with the same command:

| what | command | prints |
|---|---|---|
| the path tracer, every test scene | `bash third_party/claude/bench_rt.sh bin TAG` (a script kept outside git; the commands it runs are below) | per scene: GPU kernel ms per pass, CPU M paths/s, viewer fps |
| one GPU image | `LR_GPU_TIMING=1 bin/render ... -n 640 -s 256 -caustics pt -denoise off` | `GPU: x ms per pass` (wall clock) and, at the end, `GPU time, path tracer: x ms per pass` (timestamps, the kernel alone) |
| the live viewer | `bin/view ... -autoquit 20` | frames per second, the CPU time of each stage of a frame (sim, accel, move, more, wait, raster, present) |
| a camera in motion | `bin/view ... -orbit 15 -autoquit 20` | the same, with the camera swinging (real-time path tracing) |
| the GPU's passes | `LR_GPU_TIMING=1 bin/...` (native only) | the GPU time of every pass (today: the rigid bodies) when the program ends |
| rigid bodies, CPU vs GPU | `bin/test_gpu_rigid` (`make test`) | one 1/60 s step for 64 to 2048 bodies on both |
| many meshes, physics + drawing | `bin/mesh_rain -rate 60 -max N -autoquit 35` | frames per second and the CPU time of physics and drawing with N hulls in a pile |
| an offline film | `bin/sim ...` | physics and rendering time of the whole film |

Reference machine for the numbers below: Windows, RTX 3070, 8-core CPU, native build
(`make deps && make`), unless written otherwise. The GPU's speed varies by 10-20% from one
run (or hour) to the next: compare two versions in the same session, alternating, and take
the fastest of a few runs (the benchmark script does 3).

## Status of the ideas

`next` = chosen as the next step · `planned` = agreed, waiting its turn · `assess` = do it
only if a benchmark shows it pays · `later` / `last` = agreed to wait · `idea` = not yet
discussed · `done` · `rejected`.

### Rendering (the path tracer, mostly on the GPU)

| ID | idea | status | expected gain / why |
|---|---|---|---|
| R1 | **Wavefront path tracing**: the single path tracing kernel split into stages (camera rays, extend, shade, sizes, shadow rays, into the image) with queues of paths between them | done, measured: slower here, off by default (`-wavefront on`) | the same images (surfaces: bit for bit; media: the same mean). But 1.2-2.4x slower than the single kernel once its stack was fixed (table below): each path's state goes through memory at every bounce, and every bounce costs 4 dispatches (~7 us each with wgpu). It would pay with heavy materials (textures, layered BRDFs) or with hardware ray queries (R6), where the single kernel gets long; `LR_WF_PROFILE=1` times each of its kernels |
| R2 | Upscaler / temporal SVGF (render fewer pixels or samples, reconstruct; no neural network) | later | fewer samples per frame for the same image |
| R3 | Wide BVH (4 children per node) | done (CPU); GPU: measured, off | CPU: the dragon 1.9x faster. GPU: 10% slower than binary in the single kernel (`LR_GPU_WIDE=1` turns it on); to try again inside the wavefront tracer (R1), and with compressed (quantized) nodes |
| R4 | Refit the trees of moving instances instead of rebuilding; a higher-quality GPU builder than the LBVH | done, differently | the GPU now takes the CPU's SAH top level (it is built anyway): 2-12% faster rays than its own LBVH; the SAH builder made 2.5x faster and parallel (2500 bodies: 4.2 → 0.8 ms per step); refit measured and kept off (5% slower rays); PLOC not needed for now (R4b) |
| R4b | PLOC (a GPU builder close to SAH quality) for when the CPU has no top level, and for dynamic meshes | planned (ROADMAP.md step 4) | only matters once the CPU's tree is too slow or absent (no dynamic meshes are used yet) |
| R5 | Ray reordering for coherence | measured, no gain | in the single kernel: 8x8 pixel tiles per workgroup instead of rows of 64, 0-10% slower (rays of a row are as coherent at the first bounce, and diffuse bounces scatter both). Sorting the wavefront's queues was not tried: the wavefront loses even at 3 bounces, where sorting cannot help its memory traffic |
| R6 | Hardware ray tracing | last (the user decides) | the aim is to be fast without it first, so its gain is clear when it comes |

R3-R5 were "points 4-6" of the list discussed on 2026-11-13; their wording here is
reconstructed and may be refined.

### Physics (rigid bodies; CPU in `phys/`, GPU in `gpu/rigid_gpu.c`)

Where the time goes today (`mesh_rain`, 1500 hulls piled in the room, GPU, `LR_GPU_TIMING=1`,
after P3): 4.1 ms of GPU per physics step, of which 1.8 ms are the hull–hull contacts,
2.0 ms solve + apply, 0.1 ms the single-thread contact-list scan, 0.02 ms the pairs (broad
phase). Frames per second with N hulls: 500: 165 (the screen's limit), 1000: 118, 1500: 20
(1500 hulls of size 70 overfill the room: new ones appear inside the pile). (Before P1:
9.9 ms, 7.9 of them the hulls; 1000: 17, 1500: 7–9. After P1: 4.7 ms, 2.6 the hulls.)

| ID | idea | status | expected gain / why |
|---|---|---|---|
| P3b | GJK (or MPR) for the distance and depth of hulls instead of the separating axes, if the full test (still all faces and edge pairs) becomes the cost again | idea | since P3 the full test runs only for pairs that moved or newly met |
| P4 | Choose CPU or GPU automatically by the number of bodies | planned | the crossover is about 64 bodies since P1: 64 bodies 0.67 ms CPU / 0.67 ms GPU, 256 bodies 2.9 / 0.80 |
| P10 | **The GPU's fixed cost**: about 140 dispatches per step (16 iterations of solve + apply in each of 4 substeps), each launched for the room for contacts, not for the contacts there are; with 893 of 1024 bodies asleep a step still costs 1.07 ms, as with none. Indirect dispatches sized by the awake contacts, no solve when nothing is awake, fewer passes (solve and apply fused through graph coloring, P6) | later (very good to have) | small and sleeping scenes on the GPU: 1 ms → a fraction |
| P11 | **Tall piles that never rest**: 1500 small hulls (`mesh_rain -max 1500 -size 45 -endless off`) are still restless on the GPU after 70 s (36 asleep), while 600 fall asleep in 20 s; the bodies jitter, faster than 20 units/s now and then, and wake their neighbours. Better kept contacts (P3), a stiffer solver (P5) | idea | the big piles get the gain of P2 too |
| P9 | Fewer speculative contacts on the CPU: a tighter margin (only the approach along the normal, or Box2D's fixed 4 cm plus continuous collision for the fast ones), or contacts reused while the pair barely moves | idea | P1 made the CPU about 8% slower (1024 bodies 18.8 → 20.3 ms): its narrow phase was cheap, and the gaps add solver work; a dense, restless pile has twice the contacts it had |
| P5 | **AVBD solver** (Augmented Vertex Block Descent, SIGGRAPH 2025): bodies colored, a 6x6 system per body, 3–4 iterations, hard contacts through an augmented Lagrangian | later (very good to have; a bigger rewrite) | paper, RTX 4090: 110k bodies 3.5 ms of solve vs 14.1 ms sequential impulses, 19.5 ms XPBD; stable at mass ratios where the others crush; a WebGPU port exists. Even there collision detection cost as much as the solve, so P1–P3 come first |
| P6 | Faster CPU path as in Box2D v3: graph coloring of the contacts, SIMD (AVX2 natively, WASM SIMD in browsers), threads | planned | more than 2x on one thread in Box2D v3, and it scales with cores; matters for builds without a GPU and in the browser (one thread) |
| P7 | Broad phase with a tree (reuse the tracer's GPU LBVH builder) or sweep-and-prune | when needed | all pairs (n²) still cost only 0.05 ms at 1500 bodies; needed past about 10k |
| P8 | Parallel prefix sum for the contact lists ("contact lists 1" is one GPU thread) | planned (ROADMAP.md step 3) | 0.36 ms at 1500 bodies |

### Air and cloud

| ID | idea | status | expected gain / why |
|---|---|---|---|
| A1 | Sparse bricks for the cloud's MacCormack pass instead of its bounding box | idea | the pass touches only voxels with cloud |
| A2 | Run the air and the cloud at a lower rate than the bodies when nothing violent happens | idea | the air is the most expensive part of a CPU physics step (VOLUMES.md, PHYSICS.md 7) |

### Web (to do once the native optimizations are settled)

| ID | idea | status | expected gain / why |
|---|---|---|---|
| W1 | Bring this round's GPU changes to the browser: the new traversal stack is in the web build (a GPU path traced image checked in Chrome, 5 ms per pass on the spheres); the wavefront kernels (`?wavefront=on`) were not checked: their first compile in Chrome had not finished after more than a minute (stopped there) | stopped (ROADMAP.md: no web work for now) | the same speedups in browsers; the wavefront needs one shader module per kernel (or smaller ones) before it can be used there |

### Platform

| ID | idea | status | expected gain / why |
|---|---|---|---|
| X1 | Own Vulkan backend behind the same `webgpu.h` (drop wgpu-native) | later | control over memory, barriers and timestamps; smaller programs |

## Done

Path tracer benchmark (RTX 3070; GPU: 640x640, ms per pass of 1 sample per pixel, kernel time; CPU: 320x320, 8 threads):

| scene | GPU before | GPU after R3 | CPU before | CPU after R3 |
|---|---|---|---|---|
| boxes | 8.65 | 1.91 | 3.14 M paths/s | 3.12 |
| spheres (mirror, glass) | 9.62 | 1.73 | 4.61 | 4.53 |
| dragon (871k triangles) | 15.07 | 8.09 | 1.87 | 3.38 |
| cloud | 27.47 | 7.60 | 2.00 | 1.98 |
| spheres in cloud and fog | 35.53 | 10.70 | 1.56 | 1.55 |
| 40 instanced dragons (100k each) | 10.33 | 2.92 | 3.99 | 3.96 |
| `view` dragon `-backend rt -orbit 15` | 57 frames/s | 79 | | |

| when | what | measured |
|---|---|---|
| 2026-11-14 | **P3: hull pairs remembered**, CPU and GPU: each pair of hulls keeps the axis that separated it, or the feature of least overlap with its pose; next step a separating axis is tried first, and a touching pair that barely moved (1% of its size, 1°) takes its contacts from the same feature (PHYSICS.md 1.2b). Fewer corners per hull measured and not taken (the hull leaves too much of round shapes out) | one CPU thread, 250 hulls settling: 30.5 → 9.4 ms per step; `mesh_rain -gpu off -max 150 -endless off -sleep off`: physics 49.8 → 13.0 ms per frame. GPU: `mesh_rain -max 1000`, hull contacts 2.3 → 1.4 ms, 3.9 → 3.1 ms per step, 52 → 118 frames/s at the end; 1500 hulls 2.6 → 1.8 ms (4.7 → 4.1); 600 settled, sleeping off, 2.5 → 1.9 ms |
| 2026-11-13 | **P2: sleeping**, CPU and GPU: a body still (5 units/s) for 0.5 s falls asleep and is fixed to the others; woken by a body faster than 20 units/s near it, by the program (`Body.dirty`, impulses) or by a change of the air's force; pairs of sleeping bodies not tested (PHYSICS.md 1.5) | settled pile of 1024 balls and boxes (`test_gpu_rigid`), CPU: 22.7 → 3.5–3.7 ms per step (907 asleep); `mesh_rain -gpu off -max 150 -endless off`: physics 49.8 → 0.07 ms per frame, 13 → 40 frames/s (drawing). GPU: `mesh_rain -max 600 -endless off`, 60 s (a third of it filling): 2.52 → 0.90 ms per step on average, hull contacts 1.49 → 0.24 ms; the settled 1024 balls and boxes 1.07–1.33 → 1.07 ms (P10) |
| 2026-11-13 | **P1: contacts found once per step**, CPU and GPU: each contact's points kept in its bodies' frames and followed through the substeps (Box2D v3's soft step), speculative contacts (a margin of how far a pair can close in the step), the bounce from the speed when the step began, the push out capped at 300 units/s | `mesh_rain` 1500 hulls: 8.9 → 4.7 ms of GPU per step, hull contacts 7.1 → 2.6 ms, 9 → 19 frames/s; 1000 hulls 17 → 52 frames/s. `test_gpu_rigid` GPU: 64 bodies 1.20 → 0.67 ms, 1024 1.73 → 1.07 ms, 2048 2.80 → 1.60 ms; CPU 1024 18.8 → 20.3 ms (P9) |
| 2026-11-13 | The GPU takes the CPU's top level (SAH) when it is current (`scene_tlas`), instead of building an LBVH every frame (`LR_GPU_TLAS=lbvh` for the old way) | rays 12% faster on 40 instanced dragons (3.49 → 3.08 ms), 6% on a pile of 700 falling bodies (5.48 → 5.15), 2-3% on static scenes of 400-2000 instances; no GPU build pass (0.13-0.18 ms per frame) |
| 2026-11-13 | SAH builder: plain comparisons instead of `fminf`/`fmaxf` in the box code, one multiply instead of a division per binned item, subtrees in parallel from 1024 items (was 32768) | 2500 boxes 5.3 → 2.1 ms on one thread, 1.15 ms on 2; the viewer's "scene trees" with 2500 bodies 4.2 → 0.8 ms per frame; the 871k-triangle dragon's BVH 515 → 226 ms |
| 2026-11-13 | **GPU traversal stack: one array of 40 u32** instead of three arrays of 64 (node, instance, distance). Entering an instance pushes EXIT and its mesh's root (instances are never nested), so no instance per entry; no distance per entry (children are culled against the closest hit when tested). What a thread keeps in memory decides how many threads run at once | 1.9x to 5.6x on every GPU scene (table above); the stack size itself (24 to 64 entries) matters little, the three arrays did |
| 2026-11-13 | 4-wide BVH on the CPU (`bvh_make_wide`, `-bvhwidth 2\|4`, default 4): 4 child boxes side by side, tested in one loop, half the depth | dragon 1.87 → 3.38 M paths/s; the same images (double build: identical) |
| 2026-11-13 | Physics bodies as exact shape instances (unit sphere / unit box) in the top-level tree, a CPU top-level tree over instances (`scene_update_instances`) | `view -sim on` raster, 64 bodies: 21 → 6.1 ms per frame; GPU scene upload 0.16 → 0.02 ms |
| 2026-11-13 | Rigid bodies on the GPU (`gpu/rigid_gpu.c`, mass splitting) | one 1/60 s step: 1024 bodies 17.9 ms CPU → 1.7 ms GPU, 2048 bodies 2.8 ms; viewer with 600 bodies 142 → 164 frames/s (physics on the CPU 19 ms → 0.6 ms); blast film with 300 bodies, physics 3.4 → 1.7 s |
| 2026-11-13 | GPU rigid bodies: mesh contacts with one workgroup per pair walking the mesh's tree breadth first | the dragon's near triangles 5.74 → 0.23 ms per step |
| 2026-11-12 | One traversal loop for instances in the GPU tracer | first browser compile ~50 s → ~15 s, instanced dragon 20% faster |
| 2026-11-12 | Denoiser passes read packed data | ~3 → ~2 ms at 640x640 |
| 2026-11-12 | Owen-scrambled Sobol + blue noise sampler | same error with up to 1.35x fewer samples (Cornell box, 64 spp) |

## Tried and rejected

| when | what | why |
|---|---|---|
| 2026-11-13 | Wavefront path tracer (R1), 640x640, ms per pass: boxes 1.94 → 3.53 (single kernel → wavefront), dragon 9.0 → 13.3; with 3 bounces at most (19 dispatches instead of 207) still 1.3 → 2.2 and 6.3 → 7.5; its kernels' own times add up to less than the single kernel on simple scenes (boxes 0.76 vs 1.0 ms at 400x400), the rest is the dispatches and the paths' state in memory | kept as an option (`-wavefront on`) for heavier shading later |
| 2026-11-13 | 8x8 pixel tiles per workgroup in the single kernel (R5) | 0-10% slower than rows of 64 |
| 2026-11-13 | Refit of the top level instead of rebuilding it (`bvh_refit_boxes`; `LR_TLAS_REFIT=1`, rebuilt when its cost grows 10%) | with the faster builder the rebuild costs under 1 ms, and refit trees made rays 5% slower on the falling pile (5.41 vs 5.15 ms) |
| 2026-11-13 | 4-wide BVH in the GPU's single kernel (4 slots of 32 bytes per node, children sorted by distance; also unsorted) | 5-10% slower than the binary tree there (dragon 8.8-9.0 vs 8.1 ms); kept behind `LR_GPU_WIDE=1` to try again with R1 and compressed nodes |
| 2026-11-12 | Nearest-pixel reprojection for the real-time path tracer | duplicated pixels under subpixel motion, noise in streaks: measured worse than 2x2 bilinear |
| 2026-11-13 | The GPU rigid bodies' heavy tests inside the all-pairs kernel | one slow pair stalled the whole kernel: replaced by pair lists and indirect dispatches |
| 2026-11-13 | P1 variants: bouncing a speculative contact whose gap closes within the substep | in a pile the bounces fed on each other: bodies at 10 000 units/s, then NaN |
| 2026-11-13 | P1 variants: the bounce as a separate pass after the substeps (Box2D v3) | with only the bouncing contacts the pile kept jumping; with all touching ones (target 0) GPU hull stacks slid apart (mass splitting with 4 iterations): replaced by a bounce in the substeps from the speed when the step began |
| 2026-11-13 | P1 variants: mass splitting counting only the "active" contacts of a body on the GPU | no gain and more contacts per step: all of a body's contacts count |
| 2026-11-13 | P2 variants: a body sleeps only with all it touches (islands, as Box2D and PhysX) | in a pile of 250 hulls some body always jitters: about 10% asleep after 40 s; each body alone: all asleep after 35 s |

## Sources

- C. Yuksel et al., "Augmented Vertex Block Descent", SIGGRAPH 2025:
  https://graphics.cs.utah.edu/research/projects/avbd/
- E. Catto, "Solver2D" (soft step, contacts not recomputed in substeps):
  https://box2d.org/posts/2024/02/solver2d/ ; "Releasing Box2D 3.0":
  https://box2d.org/posts/2024/08/releasing-box2d-3.0 ; "SIMD matters":
  https://box2d.org/posts/2024/08/simd-matters/
- Newton physics, collisions (MPR/GJK, contact reduction, collide once per frame):
  https://newton-physics.github.io/newton/1.2.0/concepts/collisions.html
- NVIDIA PhysX 5, GPU rigid bodies (hull limits, persistent contacts):
  https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/GPURigidBodies.html
- R. Tonge et al., "Mass Splitting for Jitter-Free Parallel Rigid Body Simulation", SIGGRAPH 2012
- T. Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", HPG 2012
