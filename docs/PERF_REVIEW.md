# Performance review (2026-11-18, branch `perf`)

What is slow, why, and what to do about it: a ranked list of findings, each with its
evidence (a profile or a measurement), a fix, the gain measured with a prototype on this
branch or estimated, and the risk. Every prototype was one commit on `perf` (from `main` at
37b805c), to take or leave one by one.

**Taken on 2026-11-18** (branch `development`, merged into `main`): P1, P2, P4, P5, P6, P7, P8,
P9, the profiler (`tools/lrprof`) and the `bench -compare` fix. **Not taken: P3**, which only
concerned wgpu-native, removed from the engine the same day (docs/STATUS.md 1.8: the native
GPU is our Vulkan layer alone). The wgpu columns below were measured before that and stay as
they were.

**N1-N8 built the same day** (branch `development`), measured against the build before each
(medians of 3 alternated rounds), every one the same results to the bit:

| | what was built | measured |
|---|---|---|
| N1 | the air's finest multigrid level: one parallel region per half V-cycle, the threads meeting at a spinning barrier (`base/spin.h`) instead of 14 regions; the bodies' substep regions left (serial solver passes between them) | G3 15.08 -> 8.24 ms |
| N2 | subtrees of 128 primitives (1024 left threads waiting at the joins), each node array reserved once, the children's bounds from the partition | E1 11.28 -> 9.73, E2 4.55 -> 3.57, H4 320 -> 312 |
| N3 | a depth pass from the corners, then the shading of each pixel once; hidden triangles skip the vertex stage | C5 20.73 -> 18.49, C6 37.18 -> 34.47 |
| N4 | the samples' half vectors, weights and levels in a table per level (the fast atan2 for the lookups too: 1% more, other bits: not taken) | H3 560.1 -> 507.9 |
| N5 | after a JPEG's scans, the inverse DCT, upsampling and colours in tasks of 64 rows; texmap_make's levels too; the threads idle at the textures' loop take them | H5 187.8 -> 97.9 |
| N6 | the tests inline (`shape_test`), the Hit made once for the shape kept (`shape_hit`); the sampler's seeds not touched | B1 3.913 -> 5.159 M paths/s, B2 0.509 -> 0.625, B3 3.905 -> 4.093 |
| N7 | hulls placed on the stack, the inverse inertia once per substep, `mark_solids` in one region | F1 7.149 -> 6.974, F2 23.43 -> 23.10 |
| N8 | `make NATIVE=1`: `-O3 -march=x86-64-v3 -ffp-contract=off`, opt-in, folders of its own; the same bits as the default build (10 shots). With FMA (contraction on) it was faster still (C5 13.85, F1 6.20) but other bits. Its failing test_physics check with gcc 13 (hulls' momentum, 31164 for 0.001) was the check's, not the collision's: the two momenta cancel and it divided by their sum, 0 (fixed: STATUS.md step 15) | against the default: C5 18.56 -> 15.71, C6 34.47 -> 30.36, E1 9.66 -> 8.79, G3 8.49 -> 7.56, B1 5.12 -> 5.38, F1 7.06 -> 6.83 |

Machine: i5-10600K (6 cores, 12 threads), RTX 3070, Windows, MSYS2 gcc 16.2, a 60 Hz display
that day. Numbers are medians of `bin/bench` cases (docs/STATUS.md 1.3), the two builds run
in turn in the same session (`out/perf/ab.ps1`, 3 rounds of 3 to 5 repetitions); the spread
between rounds was under 1% for the CPU cases once nothing else ran on the machine (with
another build running it was 5 to 15%: such runs were thrown away).

## 1. How it was profiled

- **CPU: `tools/lrprof.c` + `tools/lrprof.mjs`** (new, 80c123d and 65b78e4; Windows has no
  perf, and no profiler was installed). A sampling profiler in 200 lines, linked into the
  program to profile: a thread stops every other thread 1000 times a second, reads its
  instruction pointer, and for time spent in a system library or the C runtime unwinds the
  stack to the engine's function that called it. Only threads that used CPU time since the
  last sample count. The report (addr2line, the build's `-g`) is by function, by inlined
  function, by caller of library time, and by source line, for all threads or one
  (`-thread 0`: the main thread, which is where the wall-clock time of a step is).
  `gcc ... apps/bench.c tools/lrprof.c build/float-gpu/liblr.a ... -o bin/bench_prof.exe`,
  then `LRPROF=out.prof bin/bench_prof -only F1` and `node tools/lrprof.mjs bin/bench_prof.exe out.prof`.
- **GPU:** the timers the engine already has (`LR_GPU_TIMING=1`, the bench's GPU column).
- **Before and after:** `bin/bench -only ...` of the two builds in turn; the images by
  `bin/bench -shots` and `-imgdiff` (10 images: CPU and GPU, path traced, denoised,
  rasterized), the tests of `make test`.

## 2. The findings, ranked

Rank is by what a user of the viewer, the page or the CPU fallback gains for the risk taken.
`done` = a prototype is on the branch with its measurement; `est.` = estimated from the
profile, not built.

| # | where | what is slow, and the evidence | fix | gain | risk / effort | prototype |
|---|---|---|---|---|---|---|
| P1 | `phys/rigid.c: broad_phase` | The sweep pairs bodies whose intervals meet on **one axis**. A pile on the floor pairs each body with every body of its slab: ~100 times the pairs that touch, then `qsort`ed. Main thread of F1: **46% of a step in `qsort` + `cmp_pair`** (10 of 21.7 ms); in the page worse (qsort's calls through a table). | List a pair only when the grown bounding spheres meet in space (the test `test_pair` does next anyway: the same pairs survive, in the same order). | F1 21.7 -> 12.1 ms, F2 47 -> 33, F4 7.7 -> 6.7. **Page, threaded: F1 46.0 -> (with P4-P6) 9.8 ms; one thread 70.0 -> 28.6.** The same contacts (3976, 252). | none seen: 12 lines, results identical | done, fc33ffd |
| P3 | `gpu/gpu.c: create` | `wgpuCreateInstance(NULL)` starts every backend wgpu-native has to list adapters: Direct3D 12 and OpenGL beside Vulkan (the profile of a start shows `wgpu_hal::dx12::create_device` and the WGL instance). | Vulkan alone (the one it chose anyway); all of them only if that finds no adapter, or `LR_WGPU_BACKENDS=all`. | **H1 231 -> 41 ms, H2 330 -> 133 ms** (wgpu; our layer: 62 and 122). Passes unchanged (A8 8.44 / 8.42). | low: a machine with Direct3D 12 but no Vulkan takes the fallback (one more instance, ~40 ms) | done, f436082 |
| P8 | `scene/mesh.c: read_ply` | 1.3 M lines through `fgets` (the stream's lock each), 4.8 M numbers through `strtod` / `strtol` (the locale each): **26% of H4's samples are parsing, on one thread, ~230 of 562 ms**. | The file read at once, lines and numbers parsed from memory; the float parse is exact (digits as an integer times an exact power of ten, Clinger's fast path; `strtod` for anything else). | **H4 562 -> 331 ms.** The same mesh: the same tree (963187 nodes), the same image. | low: 80 lines; OBJ files still go the old way (the same change would apply) | done, b9a5878 |
| P2 | `scene/env.c: pixel_weight` | The sky's sampling weight of a pixel (largest luminance of its 3 x 3) was computed at **every** use: 9 luminances + 9 `fmaxf` calls + a `sin`, for every sample of the sky, every density, every pixel of `env_prepare`'s tables (serial: 26% of H3's main thread, ~90 ms for a 2048 x 1024 sky) and again for every pixel of the GPU's copy (`trace_gpu.c:1467`, another ~90 ms at the first path traced frame of a sky scene, not in H3). | A table made once, on all cores (separable 3 + 3 maximum, the row's sine once). | **H3 721 -> 569 ms**; page H3 802 -> 663 (threaded), 4255 -> 3728 (one thread); and ~90 ms less before the first path traced frame under a sky. Images identical to the bit. | low; +4 bytes per sky pixel (8 MB for a 2k sky) | done, 9fd4d1a |
| P5 | `phys/rigid.c: solve_contact` | Each visit of a contact (10 iterations + 1 relax pass, in each of 4 substeps) redid 4 cross products and 2 matrix-vector products per direction and copied two 3x3 matrices. After P1 the solver is the largest serial part of F1. | Rows made once per substep: `d`, `ra x d`, `rb x d`, `I^-1 (r x d)`; a visit is 4 dot products and 4 scaled additions. | F1 10.74 -> 8.34 ms, F2 31.5 -> 29.5, F4 6.26 -> 4.05. | medium: the same mathematics with another rounding, so trajectories differ (4136 contacts for 3976 at the end of F1); `test_physics` and `test_gpu_rigid` pass | done, f53fe1b |
| P6 | `phys/convex.c: convex_collide_cached` | The full hull test visits every pair of edges (17 000 for two 46-corner hulls); per pair it negated two normals and took two dot products that depend on b's **faces** only, then a cross product that depends on b's **edge** only. 40% of F2 before P1, 60% after. | Per edge of a: one dot product per face of b into a table; b's edge cross products once per test. The same numbers. | F1 8.31 -> 7.50 ms, **F2 29.5 -> 23.5**, F4 4.05 -> 3.79. The same contacts. | low (identical results); 22 KB of stack in the test | done, 3f55720 |
| P4 | `base/lrmath.h: r_fmin, r_fmax` | With this compiler `fminf` / `fmaxf` are **calls into ucrtbase.dll** (it may not inline them: NaN and signed zero rules). They sit in inner loops: the solver's clamps and the swept test (533 of F1's samples), the rasterizer's shading, the sky's weights, `v_max`. | Inline, with the library's NaN rule (`b != b ? a : (a < b ? a : b)`). | C5 22.1 -> 20.9 ms, C6 38.6 -> 37.5, F1 11.08 -> 10.68, F2 32.4 -> 31.5, B1 3.85 -> 3.91 M paths/s. Images identical to the bit. | low (`fmin(-0, +0)` may give the other zero, as the library's may) | done, 5262c55 |
| N1 | every small `#pragma omp parallel for` (air, physics, top level) | **libgomp on Windows never spins**: a region's threads sleep in the kernel between regions (`OMP_WAIT_POLICY=active` changes nothing: measured), so each region costs a wake-up of 11 threads, 50-100 us. G3 (air 32^3): the main thread is **39% of a step waiting in `smooth`'s regions**, and making those loops serial costs the same time (15.79 -> 16.07 ms): 12 threads give G3 only 2.5x (table in 3.6). F1's main thread: 14% in fork and join after P1-P6. The idle threads show as 50-67% of all CPU samples (`NtWaitForMultipleObjects` returning). | (a) fewer, larger regions: one `parallel` around a V-cycle or a substep, `omp for nowait` inside; (b) natively the engine's own small runtime (`base/omp_web.c`, already written for the page) with workers that spin ~100 us before they sleep; (c) thresholds by measured work, not cell counts. | est. G3 15.8 -> 10-11 ms, F1 -10%, E1 -10% | medium: (b) is a runtime to maintain on one more platform; (a) is local edits | no |
| P7 | `gpu/shaders/trace/denoise.wgsl: atrous` | Two integer remainders (`%`) per tap for the jitter of the holes: 48 divisions a pixel a pass, in 4 of the 5 passes. | A product and a shift from 16 random bits each. | D1 1.87 -> 1.73 ms, D2 1.35 -> 1.25 (wgpu); page D1 1.68 -> 1.58. | another pattern of taps (the same statistics): the denoised shots keep their mean (+-0.01 levels) and differ by RMSE 3.5 levels at 8 samples per pixel. To judge by eye | done, 3de6c36 |
| N2 | `rt/bvh.c: build_par, new_node, flatten` | The builder's time is spread, not in one loop: every small subtree grows its own node array by `realloc` (heap calls under `new_node`: 5% of E1's main thread), a `calloc` per big node, `memset` per node, then `flatten` copies every node once more, `bvh_cost` walks them again; the top nodes' partition is one thread's. 12 threads give E1 3.8x (43.6 -> 11.4 ms). P9 below (one pass for the 3 axes, the top nodes' bins in chunks) gave only 3-6%. | Nodes allocated once (2n - 1 at most) and written in place by ranges (each subtree's node count is known when it is done: build into a private arena per task, as now, but sized once and `memcpy`ed), the cost summed while building, the top partition in parallel by a stable two-pass scatter. | est. E1 11.0 -> 7-8 ms, H4 -60 ms (wgpu's viewer spends ~1 ms a frame here with 1000 moving bodies) | medium: the tree must stay the same whatever the thread count | P9 only: cc59b20 (H4 330 -> 309, E1 11.33 -> 11.04) |
| N3 | `raster/render.c`, `raster/raster.c` (the CPU rasterizer) | C5: **75% of the time is the per-pixel shading** (`shade_point_g` 30%, `irr_grid_blend` 17%, `bsdf_spec_lobe` 6.5%, `env_level` + `bilerp` + `env_box_dir` 8%), run for every fragment that passes the depth test at that moment, then overwritten by nearer ones. C6 (871k triangles): **27% is per-triangle set-up done twice** (the binning pass and the drawing pass each transform the corners: `triangle_rows` 8.4%, `scene_vertices` 5.7%, `scene_tri` 4.2%, `scene_tri_of`'s bisection 3.5%, `poly_area_ok` 2.8%, `to_screen` 2.1%). | Shade after visibility: the drawing pass writes depth, the triangle and its barycentrics; one shading pass over the pixels at the end (same bands, same threads). Keep the binning pass's clip-space corners for the drawing pass. | est. C5 -15 to -25% (the boxes' overdraw), C6 -30% | medium: 150 lines; the image stays the same (the same fragment wins) | no |
| N4 | `scene/env.c: prefilter_level` | 128 samples per texel per level, each with `atan2f` + `acosf` (`map_uv`: exact, where `map_uv_fast` would do), `log2f`, `sincosf`: all calls into the C runtime, 2500 of H3's 9200 samples are libm (`sinf` 555, `atan2f` 507, `cosf` 354, `acosf` 295, `log2f` 201). Run twice per start (the sky, the probe). | The 128 samples' `cos`, `sin`, half-vector and level depend on the level only: a table per level; `map_uv_fast` for the lookups. | est. H3 -30 to -50 ms of the 557 left (the rest: the probe's 4.2 M paths, 300 ms on 12 threads; procedural textures, hulls) | low; the prefiltered levels change in the last bits (the fast `atan2`) | no |
| N5 | `formats/jpeg.c`, `scene/texture.c: texmap_make` (glTF) | H5: `jpeg_decode` 38% + `texmap_make` 35% of the samples (`clamp8`, the sRGB decode per texel, the mip chain); the textures already decode on separate threads, each one serially (4 threads busy of 12). | Rows of one image in parallel for the colour conversion, the upsampling and the mip levels (the Huffman stream stays serial); the sRGB decode by its 256-entry table throughout. | est. H5 190 -> 120 ms; page H5 too | low | no |
| N6 | `rt/accel.c`, `rt/intersect.c`, `base/sampler.c` (the CPU path tracer) | B2 (boxes): **`hit_quad` 33% + `intersect_shape` 10%**: the 18 primitives are tested through two calls each into another file (no inlining across files: `-O2`, no LTO); `sampler_2d` 10% (three hashes and three scrambles per pair of numbers); B4 (dragon): `wide_intersect` 23%, `hit_triangle` + normal for every candidate hit, not the last. | The hit tests `static inline` in a header (or `-flto`); the geometric and shading normal only for the hit kept; the sampler's per-pixel seeds once per path. | est. B1/B2 +20-30% | low; a path's numbers must stay the same (bit-exact images today) | no |
| N7 | `phys/world.c: mark_solids`; `phys/rigid.c: collide_*`, `rigid_steps` | Small things seen on the way: a `parallel for` **per body** over a handful of cells (the CPU air with many bodies: one region each, see N1); `malloc` of 3-6 KB per pair and per hull against the walls (`Placed`); `body_inv_inertia_world` twice per body per substep (4% of F1 now). | Scratch per thread; the inertia kept from the first loop; one region over the bodies. | est. F1 -5%; the CPU air with bodies more | low | no |
| N8 | `Makefile: CFLAGS = -O2` | `-O3 -march=x86-64-v3` (AVX2, FMA): C5 -15%, C6 -12%, F2 -17%, B1 +8% in a noisy session; `-fno-math-errno -fno-trapping-math`: nothing measurable; `-O3` alone: C5 -8%, C6 -5%. | An opt-in `make NATIVE=1`, not the default. | 10-17% on the CPU rasterizer and the bodies | FMA changes rounding (images no longer equal to the page's and to the tests' references); needs a 2013+ CPU | no (measured with `make CFLAGS=...`) |

Tried here and dropped: the air's multigrid loops on one thread below 100 000 cells (G3
15.79 -> 16.07 ms: the regions cost as much as they save, see N1); `OMP_WAIT_POLICY=active`
(no effect with this libgomp); the top level's bins in chunks alone (P9: 3%).


## 3. The profiles behind them

Shares are of the samples of the thread or threads named; `main` is the thread whose time is
the step's wall-clock time. All at 37b805c unless said.

### 3.1 The bodies on the CPU (F1: 1000 falling, 12 threads; F2: one thread)

| F1, main thread | before (21.7 ms a step) | after P1-P6 (7.5 ms) |
|---|---|---|
| `qsort` of the pairs + `cmp_pair` (`broad_phase`) | 33.5% + 12.4% | 2.3% + 1.0% |
| the solver (`solve_contact`, `apply`, `rel_velocity`) | 21% | 24% |
| `convex_collide_cached` (its share of the parallel part) | 9.7% | 19% |
| `rigid_steps` itself (contacts set up, warm start, lists) | 5% | 14% |
| waiting for the other threads (fork and join of its regions) | 6% | 14% |
| `rigid_ccd` | 2.5% | 6.2% |
| `body_inv_inertia_world` | 1.4% | 4.2% |

F2 before (one thread, 47 ms): `convex_collide_cached` 40.5% (its edge-pair loop: lines
411-418), `qsort` + `cmp_pair` 25%, the solver 9.5%, `rigid_ccd` 6.8%, `convex_place` 2.4%,
`pair_margin` + `test_pair` 4.2%, `fminf` / `fmaxf` about 3%.
In F1 before, the 11 other threads were each busy 41% of the time, 27% of it going to sleep
and waking up (N1) and 14% doing work.

### 3.2 The start (H)

| what | wall | where it goes |
|---|---|---|
| H1, a GPU ready (wgpu) | 231 ms | all of wgpu-native's backends started (P3): 41 ms with Vulkan alone |
| H3, the sky demo made | 721 ms | main thread busy 90% of it: `pixel_weight` 26% (P2), libm through `sincosf` 4%, `bilerp` 5.5%, `env_prepare` 5.4%, `convex_hull` + `occupancy_convex` 9.4% (the props' hulls, serial), `hdr_decode` + `hdr_load` 2.7%. All threads: `prefilter_level` 14.5%, `pixel_weight` 12.4%, `bilerp` 12.2%, libm 27% (N4), the probe's paths ~20%, `fbm` + `texmap_bilinear` 8% |
| H4, the dragon loaded | 562 ms | `read_ply` + `read_line` and the C runtime under them 26% of the samples, all on the main thread (P8); the tree 250 ms: `sah_split` 42%, `choose_split` 7%, `partition` 3.6%; while it is built the main thread is busy throughout and the 11 others about a third of the time (N2) |
| H5, the helmet (glTF) | 187 ms | `jpeg_decode` 38%, `texmap_make` 35% (N5) |

### 3.3 The air on the CPU (G3, 15.8 ms a step)

Main thread: `smooth` 25% + `vcycle` 21% (of which 39 points are waiting for `smooth`'s
`parallel for`: N1), `sample3` 12.4% (the cloud's advection), `fluid_step` 8.7%,
`volume_update` 3.2%. All threads: 67% of the samples are threads going to sleep or waking,
`sample3` 12.4%, `smooth` 6.2%.

### 3.4 The CPU rasterizer and path tracer

C5 (boxes, 22 ms): `shade_point_g` 29.8%, `irr_grid_blend` 17.2% (already 4 floats at a
time), `bsdf_spec_lobe` 6.5%, `draw_triangle_rows` 3.8%, `irr_sh_eval` 3.3%, `env_box_dir` 2.7%,
`map_uv_fast` 2.7%, `bilerp` 2.6%, `fminf` / `fmaxf` and other C runtime calls 4.8% (P4).
C6 (dragon, 38 ms): `shade_point_g` 14.7%, `draw_triangle_rows` 9.9%, `irr_grid_blend` 9%,
`triangle_rows` 8.4%, `scene_vertices` 5.7%, `scene_tri` 4.2%, `scene_tri_of` 3.5% (N3).
B2 (path tracer, boxes, one thread): `hit_quad` 33%, `intersect_shape` 10.4%, `sampler_2d`
10.1%, `trace_path_sampled` 8.1%, `bsdf_eval` 7%, `scene_intersect` 4.5%. B4 (dragon):
`wide_intersect` 22.5%, `intersect_shape` + `hit_triangle` 8.3%, `sampler_2d` 8.1% (N6).

### 3.5 The viewer (I6: the sky demo, 1020 bodies falling, path traced, 1280 x 720, `-vsync off`)

| | wgpu | our layer (hardware rays) |
|---|---|---|
| a frame | 21.1 ms | 9.2 ms |
| of it the CPU waiting for the GPU | 18.1 ms | 8.7 ms |
| CPU: physics (encoding the bodies' step) / scene trees / scene update / trace submit / present | 1.25 / 1.07 / 0.13 / 0.18 / 0.30 ms | 0.17 / 0 / 0.03 / 0.06 / 0.19 ms |
| GPU: a path tracing pass (1 sample per pixel) | **14.7 ms** | **4.8 ms** |
| GPU: the guide | 1.69 ms | 0.35 ms |
| GPU: the bodies' step (its kernels; first pass to last) | 0.96; 1.43 ms | 0.94; 1.14 ms |

The viewer is bound by the GPU, and on the GPU by the path tracing kernel: with wgpu (the
tree walked in the shader) a pass over the pile of moving bodies costs 14.7 ms against 8.4 ms
for the same scene with its props alone (A8), with hardware rays 4.8 against 3.2. Nothing on
the CPU side holds a frame up (its 2.9 ms with wgpu run while the GPU works). The kernel
itself was not reworked here (docs/STATUS.md 1.7 and 7 list what was tried); the denoiser's
remainders (P7) are the one GPU change.

### 3.6 Threads (the same build, `OMP_NUM_THREADS`)

| threads | C5 ms | E1 ms | G3 ms | H3 ms |
|---|---|---|---|---|
| 1 | 141.5 | 43.6 | 39.6 | 3325 |
| 2 | 71.5 | 24.3 | 26.1 | 1722 |
| 4 | 37.2 | 15.4 | 18.5 | 926 |
| 6 | 26.4 | 13.4 | 16.2 | 677 |
| 12 | 20.7 | 11.6 | 15.8 | 558 |

The rasterizer scales (6.8x on 6 cores with 2 threads each); the top level's build 3.8x (N2);
the air 2.5x, and no better beyond 4 threads (N1).


## 4. The whole suite, before and after (all prototypes together)

`bin/bench -full` of `main` (37b805c) and of `perf` (cc59b20), one after the other, wgpu and
our layer; the cases that changed by more than the noise (the others, all of A, C1-C4, F3,
F5, F6, G, H5, I1-I5, are within 2%; the CSV files are `out/perf/final_*.csv`):

| case | what | unit | wgpu before | wgpu after | vk before | vk after | from |
|---|---|---|---|---|---|---|---|
| F1 | bodies on the CPU: 1000 falling, all threads | ms/step | 21.6 | **7.58** | 21.5 | **7.32** | P1, P4, P5, P6 |
| F2 | bodies on the CPU: 1000 falling, 1 thread | ms/step | 44.3 | **23.6** | 43.9 | **23.4** | P1, P4, P5, P6 |
| F4 | bodies on the CPU: tall pile of 400 | ms/step | 7.66 | **3.80** | 7.64 | **3.78** | P1, P5, P6 |
| H1 | start: a GPU ready | ms | 236 | **37.6** | 59.7 | 60.4 | P3 |
| H2 | start: a new GPU to the first frames | ms | 320 | **129** | 120 | 119 | P3 |
| H3 | start: the sky demo made (props, light probe) | ms | 720 | **551** | 722 | **555** | P2, P4 |
| H4 | start: the dragon loaded, its tree built | ms | 565 | **307** | 562 | **308** | P8, P9 |
| D1 | denoiser: 1280x720 | ms/run | 1.90 | 1.68 | 1.46 | 1.35 | P7 |
| D2 | denoiser: 640x640 | ms/run | 1.34 | 1.24 | 1.05 | 0.976 | P7 |
| C5 | CPU rasterizer: boxes | ms/frame | 22.2 | 20.7 | 22.1 | 20.7 | P4 |
| C6 | CPU rasterizer: dragon | ms/frame | 38.2 | 37.5 | 38.2 | 36.7 | P4 |
| B1 | CPU path tracer: boxes, all threads | M paths/s | 3.85 | 3.90 | 3.82 | 3.91 | P4 |
| E3 | path traced frame, 2000 moving: 48k | ms/frame | 14.8 | 14.6 | 5.83 | 5.30 | P4 (the moved boxes) |
| E4 | path traced frame, 2000 moving: 18k | ms/frame | 17.5 | 17.1 | 4.94 | 4.58 | P4 |
| E5 | path traced frame, 2000 moving: 3k | ms/frame | 11.0 | 10.8 | 4.29 | 4.07 | P4 |
| I6 | viewer: sky demo, bodies falling, path traced | ms/frame | 20.6 | 20.7 | 9.66 | 8.74 | P4, P7 (one run each: to confirm) |

E1 / E2 (the top level on the CPU) vary by 5% from run to run (11.2-11.6 and 4.0-4.6 ms
before, 11.0-11.3 and 3.9-4.4 after): P9's gain there is inside that.

## 5. The page (Chrome, headless, `tools/webbench.mjs`; `main` against `perf` at P7)

| case | threaded page before | after | one thread before | after |
|---|---|---|---|---|
| F1, bodies on the CPU: 1000 falling | 46.0 ms | **9.78** | 70.0 | **28.6** |
| F2, the same on one thread | 72.0 | **29.1** | | |
| F4, tall pile of 400 | 10.1 | **5.22** | 15.4 | **9.16** |
| H3, the sky demo made | 802 | **663** | 4255 | **3728** |
| H5, the helmet loaded (no change made: its spread) | 215 | 188 | | |
| D1, denoiser 1280x720 | 1.68 | 1.58 | | |
| E1, CPU top level, 48k (with P9; E2 unchanged) | 12.4 | 11.2 | | |

The gap of docs/STATUS.md 2 (F1 in the threaded page 45.7 ms against 21.6 natively) was the
broad phase: a hundred times too many pairs, sorted by a `qsort` that in the page calls its
comparison through a function table. With P1-P6 the threaded page runs F1 at 9.8 ms against
7.5 natively.

## 6. Notes

- **In which order to take them:** P1, P3, P8, P2, P6, P4 change no result (identical
  contacts, meshes, trees, images) and are small; P5 changes the bodies' rounding (tests
  pass; the piles look the same); P7 changes the denoiser's tap pattern (to judge by eye);
  P9 is correct but hardly worth its lines until N2 is done.
- **A bug found on the way:** `bin/bench -compare` took `ms/step` for a rate (the unit
  holds `/s`), so it called faster bodies `slower`; fixed on the branch with this file.
- **Not looked into:** the GPU path tracing kernel itself (the viewer's bound: 3.5), the GPU
  bodies' encoding cost with wgpu (F3: 3.1 ms a step for 1.2 ms of GPU work, known as P10),
  the page's first compile (STATUS 1.7), Linux (perf is there: `tools/lrprof.c` is for
  Windows only).
- **Checks run on the branch:** `make test` and `make VK=1 test` pass at its head; the double
  build (`make DOUBLE=1`) builds and runs; the threaded page builds and runs its cases;
  `bin/bench -shots`
  against `main`: all 10 images identical to the bit up to P6 and with P8, P9; after P7 the
  two denoised ones differ as said; the full dragon rendered before and after P8 and P9:
  identical.
