# Status: what comes next, and how fast it runs

The one file that says **what to do next** and **how the engine performs now**, kept up to
date as the work goes (it joins the former ROADMAP.md and PERFORMANCE.md, 2026-11-17).
How each part works is in the other docs (RAYTRACER.md, RASTERIZER.md, PHYSICS.md,
VOLUMES.md, PBR.md, WEB.md, VULKAN.md); this file is about order and speed.

**Keep it up to date:** after every change, update section 2's numbers (from the
performance suite, once it exists) and section 1's steps; add an idea to section 4 when it
comes up, move it to the history (section 6) with its measurement when it lands, and write
down in section 7 what was tried and rejected, and why, so it is not tried again blindly.
The IDs (P*, R*, A*, W*, X*, M*) are used across the docs and the commits.

Status words: `next` · `planned` · `proposed` · `assess` (only if a benchmark shows it
pays) · `later` / `last` · `idea` (not yet discussed) · `stopped` · `done` · `rejected`.

## 1. Next steps

### 1.1 The order (decided 2026-11-17)

| # | step | status | what it is |
|---|---|---|---|
| 7 | **Physics robustness**: continuous collision checks for very fast bodies, tall piles that come to rest, the CPU solver on all cores | done (f9ca27f … 8b3a18f) | PHYSICS.md 1.3b, 1.3c, 1.5; P6, P11 |
| 8 | **Bigger scenes** (20–50k instances): culling and simpler distant copies of meshes in the rasterizer, the top level built on the GPU, instances updated in place | done (4f0edd0 … 824333e) | R8 |
| 9 | **The browser build again**: the sky scene, the denoiser and the fast physics in a browser (without hardware ray tracing) | W2 (its tools) and W4, W6 done on branch `webgpu-perf` (1.7); W3, W5, W7, W8 next | WEB.md |
| 10 | **The GPU's load**: what the GPU's time goes to in the path traced viewer, and more of it to path tracing | G1-G4, G6 done (branches `gpu-load`, `gpu-load-2`); G5 proposed | 1.5 |
| 11 | **Adaptive sampling**: the samples where the image is still noisy, not one per pixel per pass everywhere | proposed 2026-11-18, plan in 1.6 | R9 |
| 12 | **One native GPU**: wgpu-native removed, our Vulkan layer the default build, its X11 window | done (branch `vk-only`, merged 2026-11-18) | 1.8; VULKAN.md |
| 13 | **The performance review's speedups**: P1 (the bodies' broad phase in 3D: F1 21.7 -> 12.1 ms), P2 (the sky's weights in a table: H3 721 -> 569), P4 (min and max inline), P5 (the solver's rows once per substep; the bodies' rounding changes), P6 (the hulls' edge tables), P7 (the denoiser's taps without remainders; its pattern changes), P8 (PLY parsed in memory: H4 562 -> 331), P9 (SAH binning in one pass); all together F1 21.6 -> 7.6 ms natively, 46 -> 9.8 in the threaded page. P3 (wgpu-native only) not taken | done (branch `development`, 2026-11-18) | [PERF_REVIEW.md](PERF_REVIEW.md) |
| 14 | **The performance review's N1-N8**, each the same results to the bit: N1 the air's multigrid level in one region per half V-cycle, threads meeting at a spinning barrier (`base/spin.h`; G3 15.08 -> 8.24 ms); N2 the SAH builder's subtrees of 128 with their nodes reserved at once (E1 11.28 -> 9.73, E2 4.55 -> 3.57); N3 the CPU rasterizer's depth pass, then each pixel shaded once (C5 20.73 -> 18.49, C6 37.18 -> 34.47); N4 the sky prefilter's samples from a table per level (H3 560 -> 508); N5 a texture's JPEG decode and mips in row tasks (H5 188 -> 98); N6 the CPU path tracer's tests inline, the hit made once (B1 3.91 -> 5.16 M paths/s, B2 0.509 -> 0.625); N7 the physics' small things (F1 7.15 -> 6.97); N8 `make NATIVE=1`, opt-in (`-O3 -march=x86-64-v3 -ffp-contract=off`, the same bits: C5 18.6 -> 15.7, C6 34.5 -> 30.4, G3 8.5 -> 7.6, E1 9.7 -> 8.8; with FMA faster still, but other bits) | done (branch `development`, 2026-11-18) | [PERF_REVIEW.md](PERF_REVIEW.md); the commits' messages have each one's checks |
| 15 | **The bug round** (REVIEW.md, picked from it): every kind of build in folders of its own (`build/float-vk`, `build/float-cpu`, `build/float-vk-native` ...; `bin/`, `bin/cpu/`, `bin/native/`, `bin/cpu-native/` ...: B4, and after `make NOGPU=1` a plain `make` no longer keeps its CPU-only programs in `bin/`); test_physics' hull momentum check against each body's momentum (the two cancel: it divided by their sum, 0, so `NATIVE` with FMA's rounding of 1e-9 read as 31164; the collision was right). Then **B1-B16**, a commit each: B1 `-photoncache` off by default as the code has it (the docs said on), B2 a GPU path tracer whose pipelines fail no longer waited for (render, sim, bench spun for ever), B3 `-h`'s defaults printed from where they are set, B5 an explicit `-sim on -preview 4` kept, B6 serve.mjs (a sibling folder, a malformed escape), B7 numbers whole and in range (`-fov 0`, `12x` refused), B8 glTF indices checked, B9 one pixel limit in the decoders (2^28: a 32-bit `size_t` could wrap), B10 media too big for the GPU marched on the CPU, B11 a surface Outdated or Lost configured again, B12 the web thread team, B13 X11 releasing the keys on FocusOut, B14 `gpu_wait`'s flags off its stack, B15 an over-full submission an error, B16 the docs (79 shaders, `bench`, `LR_VK_DUMP`). Same images to the bit (10 shots), every build and test on Windows and Linux, `SAN=1` tests clean, the pages without GPU errors | done (branch `development`, 2026-11-18) | REVIEW.md |

`next` merged into `main` on 2026-11-17 (877237c, with the user's removal of the dragon
from the repository; the dragon stays on disk, untracked: `make dragon-download`).

### 1.2 The viewer's frame rate (fixed 2026-11-17)

The viewer showed a frame only when something changed, and its frames/s counted only those:
with a still camera the rasterizer drew nothing (2 frames/s on the counter), the GPU path
tracer waited for bands of 12-45 ms and showed at most ~30 images a second (26), and once
its samples were done it idled (0-3); moving it showed every frame (165, the display's
rate). Now (`-maxfps on`, the default) a frame is rendered and shown every time round, the
path tracer's work per frame is kept within the display's period (85% of it in a window in
step with the display, where its band runs while the frame waits for the display; 60%
elsewhere; `-band ms` to choose), sized from the GPU's own timer of a pass (the wall time,
which counts what waited on the queue, made the bands ever smaller once frames came every
few ms), and the readout gives the passes per second next to the frames. `-maxfps off`
keeps the old behaviour (less power), `-vsync off` shows frames as soon as they are ready.
The timestamps are now asked for whenever the GPU has them (Chrome too).

| Cornell box, 1280x720, window in step with the 165 Hz display | before: frames/s | after: frames/s, passes/s |
|---|---|---|
| raster, still / moving | 2 / 165 | 165 / 165 |
| path traced (wgpu), still / moving | 22-26 (288-347 passes/s) / 110 | 165, 165 / 165, 182 |
| path traced (our layer), still / moving | 21 / 102 | 165, 330 / 165, 338 |
| hybrid (wgpu), still / moving | 22-26 / 165 | 165, 165 / 165 |
| `-vsync off` (wgpu): raster / path traced still / moving | | 4016 / 211, 206 passes/s / 138, 228 passes/s |
| Chrome (headless, its own pace of 125): the sky demo path traced | 30-33 | 126, 30 passes/s |

The cost: with wgpu a still image gets half the passes per second it got before (165
against 290-350: each frame also pays the denoiser, 1.8 ms, and the composite); our layer
keeps them (330). `-band 30` or `-maxfps off` give them back at fewer frames.

### 1.3 The performance suite (done 2026-11-17)

`bin/bench` (apps/bench.c) times every case itself: the CPU's clock around work it waited
for, and the GPU's own timers (timestamp queries) where the GPU has them; nothing is read
from the viewer's title or images. Offscreen, no vsync, the same scenes, cameras, seeds,
time steps and counts every time; warm-up rounds left out; each case repeated (3 times in
the quick set, 5 in the full) and its median kept with the fastest and slowest.

| command | what |
|---|---|
| `make perf-quick` / `make perf` | the quick set (~35 s) / the full set (~3.5 min, the viewer's group I 1.5 of them) on our Vulkan layer, appended to `out/perf/<commit>_vk.csv` (the full set adds A1c, A3c, A8c: without hardware rays; `NOGPU=1`: `_cpu.csv`). The `_wgpu.csv` files of before 2026-11-18 were wgpu-native's |
| `make web-bench`, `make serve`, `node tools/webbench.mjs quick\|full -tag <commit> -o out/perf/<commit>_web.csv` | the same program in a page (built with JSPI so it can wait for the GPU), run by a headless Chrome of its own on this machine's GPU (~2 / ~6 min) |
| `bin/bench -compare old.csv new.csv` | each case before and after, faster or slower by more than 5% flagged |
| `bin/bench -table a.csv b.csv c.csv` | the Markdown table of section 2 |
| `bin/bench -list`, `-only A,F3` | the cases; some of them |

Groups: **A** the GPU path tracer (ms a pass of 1 sample per pixel), **B** the CPU path
tracer (M paths/s), **C** the rasterizers (ms a frame), **D** the denoiser (ms a run), **E**
the trees and the scene's updates (the CPU's top level; a path traced frame where 2000
instances moved), **F** the rigid bodies (ms a step at 1/60 s, a pile falling or a tall pile
settled; CPU on all threads and one, GPU), **G** the air and the cloud (ms a step), **H**
the start (a GPU ready, its first frames, a scene made), **I** the viewer itself in its
window with `-vsync off` (ms a frame, from its own counters: `-stats`; the passes per second
in the note; natively only). In a page the full dragon (35 MB) is not loaded, so its cases
are skipped there.

### 1.5 The GPU's load (branch `gpu-load`, 2026-11-17)

The question: `bin/vk/view` (today `bin/view`) `-sim on -demo sky -props 48000 ... -width 1920 -height 1080
-vsync off` never showed the GPU above 80%. The analysis (a parallel agent, its scripts and
logs in `out/gpuload/`): the GPU was not idle (nvidia-smi 95-96%, its timestamps busy 18.3
of every 19.1 ms); the 61-80% was Windows' "GPU 3D" counter (Task Manager), which reads low.
What was wrong is what the busy time went to: path tracing 39%, the denoiser 24%, the scene
rasterized under the traced image 10%, the bodies 9%, reprojection and guide 11%, the top
level 3%, idle 4%.

| # | cause | fix | status, measured (1080p, our layer, alternating A/B) |
|---|---|---|---|
| G1 | the scene rasterized under every image until its first whole pass, which never came while things moved (each frame carries the image over and restarts the count): ~3 ms a frame, hidden | only under an image begun from nothing; a carried-over image's holes (uncovered, in rows not yet traced) take their traced neighbours' mean in the composite | done (3fb6542): -vsync off 54 -> 58-63 frames/s, 27.8 -> 32.3 passes/s; vsync on 90-93 -> 138, 19 -> 27 |
| G2 | the guide timed as path tracing: a pass seemed 2.7x cheaper while things moved (5.3 ms against ~14), the bands ran 2-3x over budget | its own timer stage | done (7331657); alone: 60 -> 68 frames/s, 32 -> 19 passes/s (the bands now what the budget says) |
| G3 | the band 60% of the display's period (3.6 ms) beside ~9 ms of fixed work a frame | frames not waiting for the display (`-vsync off`, offscreen): at least a whole pass, up to 33 ms (`gpu_view_trace_budget_pass`) | done (68f41ab): -vsync off 63 -> 42-44 frames/s, 32 -> 42-44 passes/s; suite I (720p, -vsync off) our layer / wgpu: Cornell box still 339 -> 399 / 178 -> 207 passes/s, sky demo 101 -> 108 / 30 -> 47 |
| G4 | the denoiser after every band: 4.6 ms at 1080p | while things move 3 a-trous passes instead of 5 (a 17-pixel footprint instead of 65; `-denoisemove n`, still images keep 5) | done (c731866, branch `gpu-load-2`): -vsync off 39.5 -> 44 frames and passes/s (4 passes 43.7, 2 passes 46.4); vsync on 14.4 -> 17.3 passes/s |
| G5 | with vsync off the CPU waits for the band, then encodes the frame while the GPU has nothing (0.8 ms, 4%) | submit the frame behind the band, wait once a frame | proposed (up to +4%) |
| G6 | the bodies step every frame: at 165 frames/s 165 steps a second, each restarting the image from a reprojection (guide, top level, ~150 dispatches of the bodies) | at most 60 steps a second (`-simhz n`); frames between refine the same image; each step takes all the time since the last, so slower frames step every frame as before | done (4d439bc): vsync on 14.4 -> 17.5 passes/s, with G4 19.2; -vsync off (~40 frames/s) no change. The bodies move in steps of 1/55-1/60 s on a faster display |

With vsync on (the default, a 165 Hz display) the frames come at the display's rate (165,
before G1-G3 137) with fewer passes than before G2 (27): G2 made the band the 85% of the
period it was meant to be, and at 165 frames/s the fixed work of every frame takes most of
the GPU. G4 and G6 give back part of it (14.4 -> 19.2 passes/s); `-band 12` trades frames
for passes. Where the rest of a 165 Hz frame goes now is not measured yet (the agent's
per-submission timestamps, `out/gpuload/experiments.diff`, would tell); the next candidates
are G5 and adaptive sampling (1.6).

### 1.6 Step 11: adaptive sampling (proposed 2026-11-18)

The question: send more rays to the objects in the scene than to the empty parts of the
screen? A survey (a parallel agent: the code read, the literature checked) says that this is
the crude form of **adaptive sampling**, samples where the error is still high, and that the
renderer does not do it: every pixel gets one sample per pass (`trace.wgsl` `main`, one
thread per pixel of a band of rows; the CPU path tracer a fixed count too), and nothing in
the code or these docs tried it.

Two ideas to keep apart: **(a) less work on empty pixels**: a pixel that sees the sky
directly is exact after one sample (but at silhouettes, for their anti-aliasing), and its path
is already cheap (it ends at the first miss), so the saving is small, and on a GPU it comes
only where whole workgroups (64 pixels of a row) are sky; **(b) adaptive sampling proper**:
samples go where the error is (edges, glossy surfaces, soft shadows, caustics, glass); "object
or empty" is only a proxy for it, since a flat wall in direct light converges at once too.

What is there already: the moments of each pixel's samples (`accum[N+k]`: the sum of the
luminance and of its square), which the denoiser turns into a variance; the guide's `kind` per
pixel (sky, lamp, surface, medium: `accum[3N+k].w`), a ready mask; the block prefix sum (to
compact a list); the band and budget machinery (1.2, 1.5); and per sample already the
"smarter rays": the lamp and the sky importance sampled with MIS, Russian roulette after 3
bounces, Owen-scrambled Sobol with a blue-noise shift (1.35x), light tracing and the photon
cache for caustics, reprojection while moving. Missing there: emissive glTF surfaces are
found only when a path hits them, not sampled as lights.

| # | step | what | expected |
|---|---|---|---|
| AS1 | **Still images: per-tile adaptive sampling** | every 16-32 passes the relative error of each 8x8 tile, `sqrt(var/n) / (mean + eps)`, the tile's largest, after at least 8-16 samples; the tiles still above a threshold compacted into a list (in row order: 8x8 workgroups measured 0-10% slower, R5) and traced by an indirect dispatch inside the band; the viewer stops when no tile is left. The sample index per pixel (the count, `acc.w`) instead of the global one (`T.cnt.z`), which weakens the blue-noise pairing of neighbours; the stop decided on two half-buffers (even and odd samples), since one fed by its own samples is slightly biased. Prototype first in the CPU path tracer (a loop per pixel with a threshold) | 1.5-2x for the same noise in the literature, more in views with sky or plain walls; the cost one small pass and an indirect dispatch per round |
| AS2 | **Sky and lamp pixels** (the special case asked about) | stopped after a few samples where the whole 3x3 around is sky or lamp with no medium in front (edges keep their anti-aliasing), from the guide's `kind`; an early return frees whole sky rows | small, the sky demo only; a part of AS1 |
| AS3 | **While moving: pixels with little history** | after the reprojection, pixels with 0-2 samples (uncovered by what moved) get a second sample through the same list (as ATAA, Hasselgren 2020); not the variance: 12-32 samples are too few to rank pixels by it | the holes that G1 now fills from neighbours, traced instead |
| AS4 | **Better rays per sample** | emissive triangles sampled as lights; then ReSTIR DI for many emitters; ReSTIR GI or path guiding for hard indirect light | the last two are large projects |

Not to do: per-pixel atomics or loops of varying length inside the main kernel (divergence
eats the gain); per-pixel decisions under ~8 samples; "the pixel hits an object" as the
measure of error.

Literature (each checked by the agent): Mitchell 1987, the first adaptive sampling
(https://history.siggraph.org/?p=116060); Dammertz et al. 2010, a hierarchical stop from
two half-buffers, the basis of Cycles' noise threshold
(https://dspace.zcu.cz/items/1663e1e8-27b7-4fcc-bded-32801c34b5b2); Rousselle, Knaus, Zwicker
2011, samples placed to minimise the error with the filter
(https://www.cs.umd.edu/~zwicker/publications/AdaptiveSamplingGreedyError-SIGA11.pdf);
Zwicker et al. 2015, the survey (https://cs.dartmouth.edu/~wjarosz/publications/zwicker15star.html);
RenderMan's `pixelvariance` and Cycles' noise threshold (a threshold between a least and a
most samples per pixel); Kuznetsov et al. 2018, a network's sampling map: ~4 samples per
pixel match 6 uniform (https://visualcomp.ucsd.edu/projects/dasr/); Marrs et al. 2018, ATAA,
rays for ~5-6% of the pixels chosen by a mask
(https://research.nvidia.com/sites/default/files/pubs/2018-08_Adaptive-Temporal-Antialiasing/adaptive-temporal-antialiasing-preprint.pdf);
Schied et al. 2018, A-SVGF (https://cg.ivd.kit.edu/atf.php); Hasselgren et al. 2020, temporal
adaptive sampling learned with the denoiser
(https://research.nvidia.com/labs/rtr/publication/hasselgren2020neuraltemp); Firmino, Frisvad,
Jensen 2023, sampling from the denoised image's variance: 1.49x on average, 2.77x at best,
one scene 0.81x (https://graphics.ucsd.edu/~henrik/papers/daas.pdf); ReSTIR, Bitterli et al.
2020, 6-60x for many lights (https://cs.dartmouth.edu/~wjarosz/publications/bitterli20spatiotemporal.html);
ReSTIR GI, Ouyang et al. 2021; GRIS / ReSTIR PT, Lin et al. 2022; practical path guiding,
Müller et al. 2017 (https://www.jannovak.info/publications/PathGuide/index.html).

### 1.4 Step 9: the browser (branch `web`)

**Where it stands** (Chrome 1280x800, RTX 3070, the page measured by a headless Chrome of
its own on the GPU, `out_next/cdp.mjs`): the web build is up to date and loads; one kernel
was refused by Chrome's WGSL compiler (the bodies' sweep: a barrier after a return that
depends on the thread; fixed, 24ed363), so until then the GPU bodies did not step in the
page. Measured since:

| page | browser | natively (wgpu, same GPU) |
|---|---|---|
| sky demo rasterized, GPU bodies | 125 frames/s (the display's rate) | 126+ |
| sky demo path traced, denoised | 30-33 frames/s, 22-24 ms a pass | 10 ms a pass |
| the first ~20 s of the path traced page | ~10 frames/s (light probe baked on the CPU, 3.2 s; pipelines compiled when first used) | the probe in 0.37 s |
| sky demo, bodies on the CPU (`?gpurigid=off`) | 97-127 ms a physics step at 800-900 bodies (1-2 frames/s) | 45 ms on one core, 30 on 16 |
| blast: air and cloud on the GPU | 125 frames/s | |

The steps, in order (each its own commits; measured before and after with W2's tool):

| # | step | what |
|---|---|---|
| W2 | **Measure and check in a real browser, automatically** | `tools/webbench.mjs` (Node 22+, no packages): the page in a headless Chrome of its own (a temporary profile) on this machine's GPU: console, frame rate, the viewer's line, `Module._lr_prof()`, in a log; `make web-bench`. `make web-check`: every pipeline the page makes compiled by Chrome, failing on any GPU error (the check that caught the sweep kernel) |
| W3 | **Every pipeline right in the browser** | the image set rendered in Chrome (GPU raster, GPU path traced, denoised) against the native wgpu images; `test_gpu_rigid` and `test_gpu_fluid` as pages; the wavefront kernels (their first compile in Chrome took over a minute, W1): one module per kernel, or kept off in browsers |
| W4 | **The path tracer as fast as natively** (done 2026-11-18, 1.7: the stack in workgroup memory, the 4-wide walk left out, pipelines made asynchronously; subgroups and f16 measured, no gain; the light probe's bake on the page's threads instead of the GPU; a seam for ray queries) | find the 2.3x: Chrome's own timestamps (`timestamp-query`), the kernel's code from Chrome's compiler (the traversal stack in private memory, bounds checks), workgroup sizes. The start: pipelines made asynchronously (`createComputePipelineAsync`) instead of on first use; the light probe baked on the GPU (it already path traces) or made once and kept |
| W5 | **The denoiser in the page** | on by default with the path tracer: its cost per frame there, its look while the camera and the bodies move, its options in the page |
| W6 | **The bodies on the CPU in a browser** (no WebGPU, or `?gpurigid=off`) (done in part 2026-11-18, 1.7: SIMD by the compiler, the threaded page; written SIMD for the contacts, fewer substeps and P4 left) | 2.5x slower than natively on one core: WASM SIMD (`-msimd128`) for the contacts' math; threads (Emscripten pthreads, which need a cross-origin isolated page: `make serve` can send the headers, a static host cannot, so optional); fewer substeps when a step falls behind; P4 (CPU or GPU chosen by the number of bodies) |
| W7 | **The page** | the sky scene a link with its sky (today 20 MB of data to load: a 1k sky for the web, or fetched when that link is opened), a bigger-scene link (`props`), loading progress, the frame-rate HUD |
| W8 | **Docs and publishing** | WEB.md, README, W1; optionally the page on GitHub Pages (with the GitHub Actions item below) |

### 1.7 W4 and W6: the page against the machine (branch `webgpu-perf`, 2026-11-18)

The goal: the page (Emscripten, the browser's own WebGPU, no server but a static one, no
helper process) as close to the machine as it goes. The steps, each measured before and after
and committed on its own: (1) where the page loses, (2) subgroups, (3) the start, (4) the
CPU's code in the page (SIMD, threads), (5) a seam for hardware ray queries when browsers
get them. Chrome 157 (Dawn on Direct3D 12, DXC), Emscripten 7.0.0, RTX 3070, 12 threads.

**Tools added for it:** `node tools/webbench.mjs ... -serve` serves `web/build` itself
(`tools/serve.mjs`, also `make serve`: cross-origin isolated), `-flags "..."` passes switches
to Chrome for measuring (Dawn's toggles), `-q "env=NAME=value"` sets an environment variable
in the page (`bin/bench -env`), `-log file` keeps every console line; `make web-check` runs
one round of the cases that make every pipeline of the page and fails on any GPU error. The
bench page now preloads `dragon_vrip_res4.ply`: before, the sky demo in the page had no
dragon props (`not found`), so A8/A9, C1-C3 and E3-E5 measured a smaller scene than natively
(A8 in the page: 12.6 ms a pass without them, 14.4 with). The GPU timers of the bodies and
the air run in the page too now (`gpu_set_timing`: Chrome has timestamps), so F and G have
their GPU column there.

**Step 1, where the page loses** (before any change; GPU = the kernel's own timestamps; the
suite at this point is `out/webperf/perf/base_table.md`, from the code of 4197a1a):

| what | measured | so |
|---|---|---|
| the path tracer's kernel | A1 3.52 / 2.03 ms (page / wgpu natively), A4 15.3 / 10.1, A5 29.3 / 15.0, A7 8.7 / 4.86, A8 13.0 / 8.18, A9 25.9 / 17.9: 1.45-1.95x | the loss is in the kernel itself, as compiled by Chrome |
| a pass's time besides the kernel (wall clock - GPU) | A1 0.14 ms, A4 0.9, A8 1.4 (natively 0.01-0.1) | small next to the kernel |
| the simple kernels | denoiser D1 1.58 / 1.82 ms, rasterizer C1 1.36 / 1.13, bodies' kernels F3 1.36 / 1.19, air G1 0.88 / 0.75 (GPU) | within 0.85-1.25x: the browser is not slow at GPU work as such |
| a step of the bodies, the air (wall clock) | F3 3.72 / 3.07, G1 1.93 / 0.98 (GPU 1.36 and 0.88 of it) | the page pays ~1-2.4 ms a step to encode ~150 passes and wait for the readback (the bench waits every step; the viewer does not) |
| bounds checks (`--enable-dawn-features=allow_unsafe_apis,disable_robustness`) | A1 3.92 -> 4.32, A4 15.1 -> 15.3, A5 29.1 -> 30.9, A8 11.8 -> 10.2 ms | 0 to 15%, only A8; not the cause. (Also off: workgroup init, validation, lazy clears: the same) |
| Chrome's shader compiler: FXC instead of DXC (`--disable-dawn-features=use_dxc`) | A1 3.9 -> 5.3 ms, A4 15.1 -> 24.8; the first compile 70 s instead of 9 | DXC is Chrome's default already |
| the HLSL Chrome makes (`--enable-dawn-features=dump_shaders,disable_symbol_renaming`) | every index into the stack clamped (`min(uint(sp), 39u)`), every loop given a 64-bit counter that stops it (`tint_loop_idx`), the stack zeroed at each walk; the 4-wide node walk (off by default: `LR_GPU_WIDE`) compiled in with its two private arrays | the 4-wide walk now left out when it is off (below) |
| the limits asked for | the adapter's own (`requiredLimits` = its limits) already | nothing to gain |
| workgroup sizes of the path tracer (32, 64, 128 threads) | A1 3.52 / 3.69 / 3.89 ms, A4 14.6 / 13.9 / 14.1, A5 27.5 / 26.0 / 26.3 | 64 stays |
| the walk's stack in private memory | in workgroup memory instead (64 stacks interleaved): A1 3.40 -> 3.16, A4 13.9 -> 11.6, A5 26.2 -> 17.7, A8 11.6 -> 10.1; natively (wgpu) the same within the noise (A1 2.08 -> 2.30, A6 3.06 -> 2.76, the rest +-2%) | the main loss: private arrays on Direct3D 12 here; workgroup memory in the page (below) |
| a smaller stack in workgroup memory (40 -> 32 -> 24 entries) | A8 9.9 -> 9.4 -> 7.8 ms, A5 17.9 -> 17.3 -> 17.2 | faster, but a full stack drops entries (missing geometry in deep trees, the 48k props): kept at 40 |
| the first compile of the path tracer | the first pass of a new page 10.3 s (H2's first run; natively 0.32 s with wgpu, 0.12 with our layer, from the driver's cache); 9 s of A1's first case | the start's stall: step 3 |
| the CPU's code | B2 0.44 / 0.50 M paths/s on one thread; F2 71 / 44 ms; G3 32 / 15.7; H3 4164 / 728 ms (natively on 12 threads); E1 79.5 / 10.6 | one thread and no SIMD: step 4 |
| Direct3D 12 natively (wgpu-native on DX12, to compare the same driver path) | wgpu's DX12 backend did not finish compiling the path tracer in 8 minutes (FXC) | not measurable here |

**The changes**, in order (GPU ms a pass; before -> after, the same session):

| # | change | page (Chrome) | wgpu natively | our layer |
|---|---|---|---|---|
| 1 | the 4-wide walk compiled only with `LR_GPU_WIDE=1` (`WIDE_ON`, a constant of the module: its private arrays and insertion sort were in every kernel) | A4 15.3 -> 14.1, A5 29.3 -> 26.2, A7 8.7 -> 8.3, A8 13.0 -> 12.6; A1, A2, A6, A9 within the noise | A4 10.1 -> 9.06, A5 15.0 -> 12.9; the rest within the noise | the same (its kernels trace with ray queries; A1c 1.94 -> 1.93) |
| 2 | the walk's stack in workgroup memory in the page (`LR_GPU_STACK=wg`, the page's default; natively `private` stays: no gain there) | A1 3.81 -> 3.16, A2 3.32 -> 2.64, A4 14.1 -> 11.9, A5 26.2 -> 17.6, A6 4.56 -> 3.64, A7 8.30 -> 6.24, A8 12.6 -> 11.0, A9 26.1 -> 23.4 | unchanged (the default); with `wg`: the same within the noise | unchanged |

The images: `bin/bench -shots dir` and `node tools/webbench.mjs -serve -shots dir` make the
same 8 images natively and in the page (path traced: the boxes, spheres in cloud and fog,
the helmet, the sky demo; denoised: the boxes, the sky demo; rasterized: the boxes, the sky
demo), `bin/bench -imgdiff` compares them. After change 2, page against wgpu: the boxes path
traced RMSE 0.19 levels (mean +0.01), denoised 0.35, the helmet 0.67, rasterized 0.01; the
spheres in media 8.0 (mean -0.01: the same noise drawn differently, the GPUs' float rounding
in the media's tracking), the sky demo 7.1 path traced, 7.0 rasterized (mean -0.07: a few
props stand a little elsewhere: the page's CPU code computes its constants in double, the
native float build in float, `-fsingle-precision-constant`). The same numbers as with the
private stack in the page.

**Step 2, subgroups and f16** (Chrome 157 has both on this GPU: `subgroups`, sizes 32 to 32,
and `shader-f16`; tried behind feature detection, the plain kernels kept for GPUs without;
none kept: no gain):

| tried | page, GPU ms (before -> after) | why not kept |
|---|---|---|
| the tree walk as "while-while" (Aila and Laine 2009): inner nodes until the thread holds a leaf, then the leaf; with subgroups the threads wait (`subgroupAny`) until all hold one, so the leaves' tests run side by side | while-while alone: A1 3.16 -> 2.87, A2 2.64 -> 2.44, A5 17.6 -> 16.6, A6 3.64 -> 3.31, A8 11.0 -> 10.6, A9 23.4 -> 22.7 (3-9% faster); with the subgroup vote: A1 2.98, A5 17.0, A9 23.2 (no better than without) | the vote gains nothing; while-while alone is 40-90% slower natively (wgpu: A1 2.3 -> 3.9, A4 9.1 -> 13.6), so it would need a second copy of the walk for the page alone, for 3-9% |
| the bodies' 8-thread sums of a body's contacts by `subgroupShuffleXor` instead of workgroup memory and two barriers | not built: in the page the 44 solve passes of a step take 0.35 ms of the GPU's 1.4 (F3), the step 3.8 ms of wall clock (the rest is encoding ~150 passes and waiting for them) | at most ~0.1 ms of 3.8 |
| the prefix sums (the bodies' grid and contact lists) and the media's box reductions by subgroup operations | not built: 0.005-0.023 ms a pass (F3: grid scan 0.017, contact lists 0.005 + 0.023) | nothing to gain |
| the denoiser's neighbour weights (the normal's power 128, the luminance's exp) in f16 (`enable f16;`) | D1 1.70 -> 1.75-1.77 ms, D2 1.13 -> 1.17 | slower; the passes are memory bound (their inputs are already half floats), and the page's denoiser is already as fast as natively (D1 1.58-1.70 against 1.82) |

**Step 3, the start.** The path tracer's three big pipelines (the kernel, the guide, the light
tracer; and the kernel with the sky when a scene has one) are made with
`createComputePipelineAsync` (`gpu_compute_async`), asked for as soon as the viewer has its GPU
view (`gpu_view_trace_prepare`), after every small pipeline (the rasterizer's first: in Chrome
a pipeline made at once while the big ones compile waits for them, and the frames with it);
until they are ready the viewer shows the rasterized scene. Natively nothing changes (made at
once, from the driver's cache). Measured in a new headless Chrome each time (nothing
compiled before; `LR_GPU_ASYNC=0` makes them at once as before, to compare):

| case (first run of a page) | before (made at once) | after (asynchronous) |
|---|---|---|
| H6: a new GPU to the first path traced pass of the boxes, frames rasterized meanwhile | 10.3 s, 8 frames, the longest 10.2 s | 9.0 s, 495 frames, the longest 4.8 s |
| H2: a new GPU to the first frames (rasterized and path traced), its first run | 10.3 s | 8.6 s |

What is left: after ~4 s of frames one frame still waits 4.8 s. The compile to DXIL runs on
Chrome's worker threads; the rest (seen as one frame held while the pipelines arrive, most
likely the driver's own compile of the big kernel) is outside the page's reach. The later
runs of a page find the pipelines in Chrome's cache (H2's next runs: 74-90 ms).

**The light probe on the GPU (H3) was assessed and not built:** its kernel would be the path
tracer's (the probe is path traced from one point), so on a page's first run it would wait
for that same 5-9 s compile, against 4.2 s for the CPU's bake on one thread; the CPU's bake
on the page's threads (step 4) is the way taken.

**Step 4, the CPU's code in the page.** (a) WebAssembly SIMD: `-msimd128` for the whole web
build (`NOSIMD=1` without), the compiler vectorizing what it can (no intrinsics written).
Single-threaded page, before -> after:

| case | before | -msimd128 | -msimd128 -fno-slp-vectorize | -msimd128 -fno-vectorize |
|---|---|---|---|---|
| B2 CPU path tracer, 1 thread (M paths/s) | 0.444 | 0.459 | 0.443 | 0.464 |
| C5 CPU rasterizer (ms/frame) | 173 | 146 | 150 | 149 |
| E1 CPU top level, 48k (ms/build) | 79.6 | 52.6 | 79.2 | 52.6 |
| E2 CPU top level, 18k (ms/build) | 28.4 | 18.6 | | |
| F2 bodies on the CPU, 1 thread (ms/step) | 71.3 | 70.9 | 71.2 | 71.2 |
| F4 tall pile on the CPU (ms/step) | 15.5 | 15.4 | | |
| G3 air and cloud on the CPU (ms/step) | 32.2 | 34.8 | 32.1 | 34.7 |
| H3 the sky demo made (ms) | 4191 | 4131 | | |

Kept: all of it (the SLP vectorizer makes the top level's build 1.5x faster and the air's
step 8% slower; the loop vectorizer the rasterizer 15% faster); the bodies' contacts did not
vectorize by themselves (F2 the same): that would take written SIMD.

(b) Threads: the threaded page (`make web THREADS=1`: `index_mt.js`; `make web-bench
THREADS=1`: `bench_mt.html`), Emscripten's pthreads, one per hardware thread (12 here). The
engine's ~50 OpenMP pragmas stay as they are: the compiler turns them into calls of LLVM's
OpenMP runtime (`-Xclang -fopenmp`), and `engine/base/omp_web.c` (~390 lines, empty
otherwise) is the part of such a runtime they need, on pthreads: parallel regions (with
`if`, nested ones alone), static and dynamic loops, reductions, `critical`, `single`,
barriers, tasks and `taskwait` (the BVH builder), `omp_*` (its header `web/include/omp.h`).
It needs a cross-origin isolated page (SharedArrayBuffer): `make serve` and `webbench
-serve` send the headers; the page (`web/shell.html`) loads `index_mt.js` when it is
isolated and the file is there, `index.js` otherwise (a static host without the headers, or
`?threads=1`). The single-threaded page is built and works as before. Page with SIMD, one
thread -> threaded (native, 12 threads, for comparison):

| case | one thread | threaded | native |
|---|---|---|---|
| B1 CPU path tracer, all threads (M paths/s) | 0.446 | 3.56 | 3.56-3.84 |
| C5 CPU rasterizer (ms/frame) | 146 | 21.5 | 22.2 |
| E1 CPU top level, 48k (ms/build) | 52.6 | 11.4 | 10.6 |
| E2 CPU top level, 18k (ms/build) | 18.6 | 4.39 | 4.03 |
| F1 bodies on the CPU, all threads (ms/step) | 70.9 | 45.9 | 21.6 |
| G3 air and cloud on the CPU (ms/step) | 34.8 | 17.5 | 15.7 |
| H3 the sky demo made: props, light probe (ms) | 4131 | 794 | 728 |
| the viewer, `?sim=on&gpurigid=off&gpufluid=off&backend=raster&balls=300&boxes=300` | 3 frames/s (physics 182 ms a frame) | 8 frames/s (79 ms) | |

The bodies on the CPU (F1) keep half their native speed: their iterations stay on one thread
(7: colours lost to barriers natively too), and one thread of the page runs them 1.6x slower
than natively (F2 71 against 44). The images: the same as natively and as the
single-threaded page (`-shots`, both pages: the CPU's path traced and rasterized boxes, and
the GPU's 8, the same RMSE to the last digit as the single-threaded page's).

**Step 5, ready for hardware ray queries.** No browser has them; the proposals (gpuweb #11054
acceleration structures, feature `"acceleration-structure"`; #11056 WGSL's `ray_query`) are
under review. The page now looks for the feature by its name on the adapter (it has no
`WGPUFeatureName` in webgpu.h), reports it when a browser offers it (`gpu_rt_offered`) and
traces with its own tree walk either way; checked by looking for a name Chrome has
(`timestamp-query`: found). What is missing before the hardware path (`Gpu.rt`, as our Vulkan
layer sets it) can switch on in a page is written down in WEB.md 2.10: a web implementation
of webgpu_rt.h's 9 functions, the top level as a bind group entry (Emscripten's WebGPU port
knows none), and `trace_hw_traverse.wgsl` in the proposal's syntax (it differs from wgpu's
naga syntax our layer takes). Nothing fake is shipped and nothing needs a browser flag.

### 1.8 Step 12: one native GPU (branch `vk-only`, 2026-11-18)

The user's decision: keep two WebGPUs, our Vulkan layer natively and the browser's in a
page, and drop wgpu-native. What changed:

- `make` builds the layer (`-DLR_GPU -DLR_VK`, objects in `build/float-vk/`, binaries in
  `bin/`); `NOGPU=1` builds without the GPU. Gone: `make deps`, `deps-clean`, `VK=1`,
  `bin/vk/`, `third_party/wgpu` (the folder can be deleted), `LR_WGPU_NATIVE`, about 16 MB
  per program (wgpu-native was linked statically).
- `webgpu/include/webgpu/webgpu.h`: our copy of the standard header (BSD-3-Clause, the
  version wgpu-native v29.0.1.1 shipped). The browser build keeps Dawn's
  (`--use-port=emdawnwebgpu`).
- wgpu.h's extras the layer imitated now have names of our own, in
  `webgpu/vk/webgpu_lr.h`: `lrQueueSubmitForIndex`, `lrDevicePoll`, `LRSubmissionIndex`,
  `LRFeatureName_RayQuery` (with `lrDeviceGetComputeQueue`, moved there from webgpu_rt.h).
- The layer's X11 surface (`VK_KHR_xlib_surface`): before, Linux drew offscreen (each
  frame read back by the CPU). Built and warning-free; not yet run on a Linux GPU.
- `bench`'s native column is `vk` (`cpu` with `NOGPU=1`); `-shots` names its images
  `<name>_vk.png`.

Checked: Windows `make`, `make test` (hardware ray tracing on), `make DOUBLE=1` and its
tests, `make NOGPU=1` and its tests, `make check-layers`, `make web`, `make web THREADS=1`,
`make web-check THREADS=1` (both pages, no GPU error); Linux (gcc 13, no `third_party` at
all, a Vulkan loader but no GPU driver): `make`, `make test` (the GPU tests skip and the
programs fall back to the CPU, as on a machine without Vulkan 1.3), `make DOUBLE=1` and its
tests, `make NOGPU=1` and its tests, `make check-layers`; every build warning-free (two
warnings the layer's code had on Linux fixed: `surface_free` unused without a surface,
`failed` clobbered by `longjmp` in the WGSL compiler).

What it costs: MSAA waits for multisampled targets in the layer (R7); a machine without
Vulkan 1.3 runs natively on the CPU (its browser still has WebGPU); the WGSL is no longer
compiled by a second native compiler (naga): the page's (Tint) checks it, `make web-check`
and the images against native (`bench -shots`, `-imgdiff`).

## 2. Performance now

The full suite (1.3) of 2026-11-18 on the branch `webgpu-perf` at b296213 (W4 and W6, 1.7):
RTX 3070, 12 hardware threads, Windows, a 165 Hz display; wgpu and our layer natively,
Chrome 157 (headless, its own profile) for the pages: "web (4197a1a)" is the page before W4
and W6 (`main`, measured the same day), "web" the default page now, "web-mt" the threaded
page (`make web-bench THREADS=1`, bench_mt.html, 12 threads). Nothing waits for the
display: the cases run offscreen, and the viewer's (I) in its window with `-vsync off`, so
each number is as fast as the machine goes. Times in ms (less is better), the CPU path
tracer in M paths/s (more is better); "GPU" is the GPU's own timer for the same unit (Chrome
has them too). **Run the suite after each change and replace this table** (`bin/bench
-table`); compare with `bin/bench -compare`. The GPU's clock varies by 10-20% from one run
to the next, and the cases of whole frames (E3-E5, I) more: in the same hour E3 read 14.5,
19.4 and 31 ms, with the commit before the change as slow as after it; compare such cases
within one session, one build after the other.

| case | what | unit | wgpu | vk | web (4197a1a) | web | web-mt |
|---|---|---|---|---|---|---|---|
| A1 | GPU path tracer: boxes | ms/pass | 2.07 (GPU 2.06) | 1.28 (GPU 1.27) | 3.66 (GPU 3.52) | 2.94 (GPU 2.77) | 2.84 (GPU 2.75) |
| A2 | GPU path tracer: spheres (mirror, glass) | ms/pass | 1.97 (GPU 1.95) | 1.47 (GPU 1.46) | 3.7 (GPU 3.43) | 2.72 (GPU 2.64) | 2.78 (GPU 2.66) |
| A3 | GPU path tracer: dragon (871k triangles) | ms/pass | 8.18 (GPU 8.1) | 1.34 (GPU 1.33) | - | - | - |
| A4 | GPU path tracer: cloud | ms/pass | 9.17 (GPU 9.06) | 7.31 (GPU 7.26) | 16.2 (GPU 15.3) | 12.2 (GPU 11.5) | 12.8 (GPU 11.5) |
| A5 | GPU path tracer: spheres in cloud and fog | ms/pass | 13.1 (GPU 12.9) | 9.95 (GPU 9.87) | 30.4 (GPU 29.3) | 19.3 (GPU 17.8) | 19.1 (GPU 17.8) |
| A6 | GPU path tracer: 40 instanced dragons | ms/pass | 3.09 (GPU 3.05) | 1.37 (GPU 1.36) | 4.98 (GPU 4.76) | 3.89 (GPU 3.62) | 4.02 (GPU 3.64) |
| A7 | GPU path tracer: helmet (glTF, textures) | ms/pass | 5 (GPU 4.94) | 1.77 (GPU 1.76) | 9.17 (GPU 8.7) | 6.9 (GPU 6.2) | 6.91 (GPU 6.23) |
| A8 | GPU path tracer: sky demo, 3k props | ms/pass | 8.46 (GPU 8.38) | 3.21 (GPU 3.18) | 14.4 (GPU 13.0) | 11.6 (GPU 11.0) | 11.7 (GPU 11.1) |
| A9 | GPU path tracer: sky demo, 48k props | ms/pass | 17.3 (GPU 17.1) | 3.92 (GPU 3.89) | 27.1 (GPU 25.9) | 24.5 (GPU 23.2) | 24.4 (GPU 23.2) |
| B1 | CPU path tracer: boxes, all threads | M paths/s | 3.84 | 3.87 | 0.446 | 0.458 | 3.63 |
| B2 | CPU path tracer: boxes, 1 thread | M paths/s | 0.495 | 0.498 | 0.44 | 0.461 | 0.464 |
| B3 | CPU path tracer: dragon, all threads | M paths/s | 3.84 | 3.82 | - | - | - |
| B4 | CPU path tracer: dragon, 1 thread | M paths/s | 0.494 | 0.485 | - | - | - |
| C1 | GPU rasterizer: sky demo, 3k props | ms/frame | 1.12 | 1.15 | 1.36 | 1.34 | 1.34 |
| C2 | GPU rasterizer: sky demo, 18k props | ms/frame | 1.04 | 1.08 | 1.31 | 1.28 | 1.29 |
| C3 | GPU rasterizer: sky demo, 48k props | ms/frame | 1.6 | 1.66 | 1.9 | 1.88 | 1.87 |
| C4 | GPU rasterizer: dragon (871k triangles) | ms/frame | 0.62 | 0.598 | - | - | - |
| C5 | CPU rasterizer: boxes | ms/frame | 22.2 | 22.1 | 172 | 147 | 20.6 |
| C6 | CPU rasterizer: dragon | ms/frame | 38.3 | 38.3 | - | - | - |
| D1 | denoiser: 1280x720 | ms/run | 1.96 | 1.52 | 1.58 | 1.71 | 1.58 |
| D2 | denoiser: 640x640 | ms/run | 1.34 | 1.04 | 1.14 | 1.22 | 1.14 |
| E1 | CPU top level (SAH), all moved: 48k | ms/build | 10.7 | 10.8 | 79.5 | 52.9 | 11.5 |
| E2 | CPU top level (SAH), all moved: 18k | ms/build | 4.06 | 4 | 28.4 | 18.7 | 4.47 |
| E3 | path traced frame, 2000 moving: 48k | ms/frame | 14.7 (GPU 0.404) | 5.97 | 36.5 (GPU 0.423) | 19.6 (GPU 0.42) | 19.8 (GPU 0.425) |
| E4 | path traced frame, 2000 moving: 18k | ms/frame | 17.5 | 4.92 | 48.3 | 35.3 | 21.7 |
| E5 | path traced frame, 2000 moving: 3k | ms/frame | 11.0 | 4.28 | 19.0 | 16.6 | 15.5 |
| F1 | bodies on the CPU: 1000 falling, all threads | ms/step | 21.6 | 21.5 | 71.2 | 70.9 | 45.7 |
| F2 | bodies on the CPU: 1000 falling, 1 thread | ms/step | 44.3 | 44.2 | 71.1 | 70.9 | 71.4 |
| F3 | bodies on the GPU: 1000 falling | ms/step | 3.07 (GPU 1.17) | 1.68 (GPU 1.14) | 3.72 (GPU 1.36) | 3.74 (GPU 1.36) | 3.8 (GPU 1.37) |
| F4 | bodies on the CPU: tall pile of 400 | ms/step | 7.64 | 7.64 | 15.4 | 15.4 | 9.98 |
| F5 | bodies on the GPU: tall pile of 400 | ms/step | 3.09 (GPU 1.15) | 1.62 (GPU 1.12) | 3.73 (GPU 1.31) | 3.62 (GPU 1.29) | 3.72 (GPU 1.29) |
| F6 | bodies on the GPU: 2000 falling | ms/step | 3.57 (GPU 1.63) | 2.17 (GPU 1.56) | 4.06 (GPU 1.94) | 4.09 (GPU 1.92) | 4.19 (GPU 1.96) |
| G1 | air and cloud on the GPU: 32^3, cloud 96^3 | ms/step | 0.973 (GPU 0.751) | 0.971 (GPU 0.751) | 1.93 (GPU 0.878) | 1.98 (GPU 0.864) | 1.9 (GPU 0.787) |
| G2 | air and cloud on the GPU: 32^3, cloud 192^3 | ms/step | 1.79 (GPU 1.46) | 1.65 (GPU 1.41) | 2.41 (GPU 1.72) | 2.41 (GPU 1.71) | 2.38 (GPU 1.58) |
| G3 | air and cloud on the CPU: 32^3, cloud 96^3 | ms/step | 15.8 | 15.7 | 32.0 | 34.8 | 17.5 |
| H1 | start: a GPU ready | ms | 233 | 62.9 | 32.9 | 33.6 | 38.0 |
| H2 | start: a new GPU to the first frames | ms | 318 | 120 | 85.4 | 110 | 84.5 |
| H3 | start: the sky demo made (props, light probe) | ms | 730 | 731 | 4164 | 4094 | 806 |
| H4 | start: the dragon loaded, its tree built | ms | 565 | 567 | - | - | - |
| H5 | start: the helmet loaded (glTF) | ms | 186 | 188 | 480 | 468 | 211 |
| H6 | start: the longest frame until path traced | ms | 1.33 | 4.24 | - | 31.4 | 30.1 |
| I1 | viewer: rasterized, still | ms/frame | 0.25 | 0.244 | - | - | - |
| I2 | viewer: rasterized, moving | ms/frame | 0.25 | 0.243 | - | - | - |
| I3 | viewer: path traced, still | ms/frame | 10.3 | 11.5 | - | - | - |
| I4 | viewer: path traced, moving | ms/frame | 10.3 | 11.4 | - | - | - |
| I5 | viewer: hybrid, still | ms/frame | 10.3 | 11.5 | - | - | - |
| I6 | viewer: sky demo, bodies falling, path traced | ms/frame | 18.7 | 9.18 | - | - | - |
| A1c | GPU path tracer, no hardware rays: boxes | ms/pass | - | 1.96 (GPU 1.95) | - | - | - |
| A3c | GPU path tracer, no hardware rays: dragon | ms/pass | - | 7.4 (GPU 7.35) | - | - | - |
| A8c | GPU path tracer, no hardware rays: sky 3k | ms/pass | - | 7.84 (GPU 7.79) | - | - | - |

What it shows: our layer path traces 1.4-6x faster than wgpu (hardware rays: the dragon
1.34 against 8.18 ms, the sky demo with 48k props 3.92 against 17.3; without them,
A1c/A3c/A8c, it is close to wgpu). The page path traces 1.2-1.5x slower than wgpu natively
(A1 2.77 against 2.06 GPU ms, A5 17.8 against 12.9, A9 23.2 against 17.1), where it was
1.5-2.3x before W4 (1.7); natively only A4 and A5 changed (WIDE_ON, 10.1 -> 9.06 and 15.0 ->
12.9 GPU ms). The threaded page runs the CPU's code at native speed or close (B1 3.63
against 3.84 M paths/s, C5 20.6 against 22.2 ms, E1 11.5 against 10.7, G3 17.5 against
15.8, H3 806 against 730 ms); the bodies on the CPU (F1) stay 2.1x slower (45.7 against
21.6 ms), and on one thread every CPU case is as before W6 apart from SIMD (E1 79.5 ->
52.9, C5 172 -> 147; G3 32.0 -> 34.8). The rasterizers, the denoiser and the GPU bodies
cost about the same everywhere. H6 here is warm (the pipelines are in the driver's and
Chrome's caches from the cases before it); cold, alone, a page's first start has its
longest frame at 4.8 s, 10.2 before (1.7). The viewer's path traced frames (I3-I5) read 10
ms on both native builds in this session, the same on `main` (4197a1a) measured the same
day, against 5 ms on 2026-11-17: the display's state, not the code.

Not in the suite (yet): tall piles fall fully asleep on the CPU in 11 s, on the GPU in 2 of
3 runs within 60 s (P11); `mesh_rain` 1500 hulls 5.3 / 5.1 ms a step (wgpu / our layer).

## 3. How to measure

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
(`make`; before 2026-11-18 wgpu-native through `make deps && make`, our layer `make VK=1`), unless written otherwise. The GPU's speed varies by 10-20% from one
run (or hour) to the next: compare two versions in the same session, alternating, and take
the fastest of a few runs (the benchmark script does 3).

Before 2026-11-15, `view -gpu offscreen` counted a frame even when it drew nothing (the
previous frame still coming back from the GPU), so the frames per second it printed were too
high, sometimes absurdly ("318998 frames/s"); the per-stage CPU times were right. Frame rates
recorded that way above should be measured again; from then on only frames shown count
(the helmet rasterized, orbiting: 63 frames/s; 1000 bodies in the viewer: 65).

## 4. Ideas and their status

### Rendering (the path tracer, mostly on the GPU)

| ID | idea | status | expected gain / why |
|---|---|---|---|
| R1 | **Wavefront path tracing**: the single path tracing kernel split into stages (camera rays, extend, shade, sizes, shadow rays, into the image) with queues of paths between them | done, measured: slower here, off by default (`-wavefront on`) | the same images (surfaces: bit for bit until PBR2; since, the two kernels' compiled float rounding differs, so a few paths take the other side of a decision: the same mean); But 1.2-2.4x slower than the single kernel once its stack was fixed (table below): each path's state goes through memory at every bounce, and every bounce costs 4 dispatches (~7 us each with wgpu). It would pay with heavy materials (textures, layered BRDFs) or with hardware ray queries (R6), where the single kernel gets long; `LR_WF_PROFILE=1` times each of its kernels |
| R2 | Upscaler / temporal SVGF (render fewer pixels or samples, reconstruct; no neural network) | later | fewer samples per frame for the same image |
| R3 | Wide BVH (4 children per node) | done (CPU); GPU: measured, off | CPU: the dragon 1.9x faster. GPU: 10% slower than binary in the single kernel (`LR_GPU_WIDE=1` turns it on); to try again inside the wavefront tracer (R1), and with compressed (quantized) nodes |
| R4 | Refit the trees of moving instances instead of rebuilding; a higher-quality GPU builder than the LBVH | done, differently | the GPU now takes the CPU's SAH top level (it is built anyway): 2-12% faster rays than its own LBVH; the SAH builder made 2.5x faster and parallel (2500 bodies: 4.2 → 0.8 ms per step); refit measured and kept off (5% slower rays); PLOC not needed for now (R4b) |
| R4b | PLOC (a GPU builder close to SAH quality) for when the CPU has no top level, and for dynamic meshes | done | meshes' trees 8-9% cheaper (bvh_cost) than the LBVH's, rays 3-4% faster, built 1.3-1.8x slower; the top level as good either way, so it keeps the LBVH there (log below); the CPU's SAH trees (several triangles per leaf) stay the fastest to trace |
| R5 | Ray reordering for coherence | measured, no gain | in the single kernel: 8x8 pixel tiles per workgroup instead of rows of 64, 0-10% slower (rays of a row are as coherent at the first bounce, and diffuse bounces scatter both). Sorting the wavefront's queues was not tried: the wavefront loses even at 3 bounces, where sorting cannot help its memory traffic |
| R6 | Hardware ray tracing | done natively on our Vulkan layer (2026-11-15, `vulkan` branch, VULKAN.md): ray queries in the existing kernels (traverse() has a hardware twin), the compute BVH kept for wgpu, the browser and `LR_GPU_HWRT=0` | 1.4-2.7x faster per pass (table of 2026-11-15 below); wgpu-native's C API cannot build acceleration structures, and browsers do not have them yet (a WebGPU proposal since 2026-10) |
| R7 | **Anti-aliasing** where there is none yet. Today only the path tracer has it (each sample's camera ray jittered inside its pixel, a box filter); both rasterizers draw one sample per pixel (WebGPU `sampleCount` 1), so the viewer's edges stair-step while moving and in raster mode. Candidates, cheapest first: (a) **MSAA 4x in the GPU rasterizer** (every WebGPU device has 4 samples; a resolve after the pass, the cloud composited after the resolve); (b) **FXAA or SMAA 1x** as a post pass on any frame, CPU or GPU (edges from luminance, no geometry; SMAA's area tables made by our own tool); (c) **coverage-mask MSAA in the CPU rasterizer**: the edge functions tested at 4 sample positions, the pixel shaded once and weighted by its covered samples (depth per sample); (d) **TAA in the viewer**: the projection jittered each frame by a Halton (2, 3) subpixel offset, the history reprojected (the path tracer's reprojection, WEB.md 2.5) and clamped to the neighbourhood's colour box; it also smooths the switch from raster to path tracing, and shares its history with R2; (e) the path tracer: **the denoiser's guide jittered** with the samples (it is traced through the pixel centre, so a denoised edge can keep a stair step) and a better pixel filter (tent or Blackman-Harris by filter importance sampling, instead of the box) | (b) done (2026-11-18, branch `antialias`): FXAA, the rest ideas. MSAA (a) waits for our Vulkan layer, which has no multisampled targets yet | (b): the viewer's rasterized frames (GPU raster and CPU framebuffers, not path traced ones) go through FXAA after the composite (`view_fxaa.wgsl`, the quality variant of FXAA 3.11; `-aa fxaa\|off`, on by default). Near-horizontal and near-vertical edges come out smooth, 45-degree ones only a little (FXAA's known weak case), textures barely touched. Cost with `-vsync off`: 0.04-0.05 ms a frame at 1280x720, 0.06-0.07 ms at 1600x900 (boxes), 0.07-0.13 ms on the sky demo with 48k props, the same on wgpu and our layer (identical pixels) |
| R8 | **Bigger scenes** (20-50k instances; `-demo sky -props 18000` / `48000`, the props in a ring that grows with them): the rasterizer drew one call per run of the same mesh, about one per instance with mixed props; the GPU path tracer took the CPU's SAH top level, built on the CPU every frame something moved; every frame all the instances' records were packed and sent again | done (2026-11-17, branch `next`) | (a) the GPU rasterizer leaves out the instances outside the frustum and draws one call per mesh and level of detail through an order buffer (`ORD`); from 4096 instances on, simpler copies of each mesh (vertex clustering, `scene_mesh_simplify`) for those under 40 and 12 pixels of radius (`LR_GPU_LOD`); (b) the top level built on the GPU from 32768 instances on (`LR_GPU_TLAS`); (c) each instance knows when it last changed, so only the records of those moved or added are sent; (d) our layer's top level updated in place (a refit) while at most a quarter moved. Measurements in Done, 2026-11-17 |
| R9 | **Adaptive sampling** (researched 2026-11-18): samples where the image is still noisy instead of one per pixel per pass; the skys pixels its special case | planned: step 11, plan and sources in 1.6 | 1.5-2x for the same noise on still images in the literature |

R3-R5 were "points 4-6" of the list discussed on 2026-11-13; their wording here is
reconstructed and may be refined.

### Physics (rigid bodies; CPU in `phys/`, GPU in `gpu/rigid_gpu.c`)

Where the time goes today (`mesh_rain`, 1500 hulls piled in the room, GPU, `LR_GPU_TIMING=1`,
after P3): 4.1 ms of GPU per physics step, of which 1.8 ms are the hull–hull contacts,
2.0 ms solve + apply, 0.004 ms the contact-list prefix sum (P8), 0.02 ms the pairs (broad
phase). Frames per second with N hulls: 500: 165 (the screen's limit), 1000: 118, 1500: 20
(1500 hulls of size 70 overfill the room: new ones appear inside the pile). (Before P1:
9.9 ms, 7.9 of them the hulls; 1000: 17, 1500: 7–9. After P1: 4.7 ms, 2.6 the hulls.)

| ID | idea | status | expected gain / why |
|---|---|---|---|
| P3b | GJK (or MPR) for the distance and depth of hulls instead of the separating axes, if the full test (still all faces and edge pairs) becomes the cost again | idea | since P3 the full test runs only for pairs that moved or newly met |
| P4 | Choose CPU or GPU automatically by the number of bodies | planned | the crossover was about 64 bodies after P1 (64 bodies 0.67 ms CPU / 0.67 ms GPU, 256 bodies 2.9 / 0.80); since P10 the GPU is ahead at 64 too (0.53 / 0.27), so it is somewhere below |
| P10 | **The GPU's fixed cost**: about 150 dispatches per step (16 iterations of solve + apply in each of 4 substeps); with 893 of 1024 bodies asleep a step still cost 1.07 ms, as with none | done (2026-11-15): a step is one submission; an iteration is one pass (each body, 8 threads, works out its contacts' impulses and adds them up in its list's order: 92 dispatches per step instead of 156); the warm start 8 threads a body too; a world all asleep, unchanged since, is not stepped at all. Not taken: dispatches sized by the contacts there are (indirect: wgpu checks each indirect dispatch, the prepare pass got slower). Left: a world partly asleep still pays the dispatches | 1024 bodies 0.80 → 0.40 ms per step, the settled pile (88% asleep) 1.07 → 0.80, all asleep: 0; `mesh_rain` 1500 hulls 37 → 45 frames/s (table below) |
| P11 | **Tall piles that never rest**: 1500 small hulls (`mesh_rain -max 1500 -size 45 -endless off`) were still restless on the GPU after 70 s (36 asleep), while 600 fell asleep in 20 s; the bodies jitter, faster than 20 units/s now and then, and wake their neighbours | partly done (2026-11-17) | settling damping, sleeping when going nowhere, no sleep on a moving neighbour, no wake by a body jittering in place, the push out taken back (PHYSICS.md 1.3c, 1.5): on the CPU a 400-body tall pile asleep in 11 s (6 of 6 runs); on the GPU 2 of 3 within 60 s (before: none). The rest needs a Gauss-Seidel GPU solver (by colours, P5's colouring) |
| P9 | Fewer speculative contacts on the CPU: a tighter margin (only the approach along the normal, or Box2D's fixed 4 cm plus continuous collision for the fast ones), or contacts reused while the pair barely moves | idea | P1 made the CPU about 8% slower (1024 bodies 18.8 → 20.3 ms): its narrow phase was cheap, and the gaps add solver work; a dense, restless pile has twice the contacts it had |
| P5 | **AVBD solver** (Augmented Vertex Block Descent, SIGGRAPH 2025): bodies colored, a 6x6 system per body, 3–4 iterations, hard contacts through an augmented Lagrangian | later (very good to have; a bigger rewrite) | paper, RTX 4090: 110k bodies 3.5 ms of solve vs 14.1 ms sequential impulses, 19.5 ms XPBD; stable at mass ratios where the others crush; a WebGPU port exists. Even there collision detection cost as much as the solve, so P1–P3 come first |
| P6 | Faster CPU path as in Box2D v3: graph coloring of the contacts, SIMD (AVX2 natively, WASM SIMD in browsers), threads | partly done (2026-11-17) | the contacts found on all cores, the same ones as on one (1000 falling bodies 59.7 → 29.5 ms a step on 16 threads); the iterations by colours on all cores lost to the barriers (see Tried); SIMD still to do; the browser has one thread |
| P7 | Broad phase with a tree (reuse the tracer's GPU LBVH builder) or sweep-and-prune | when needed | all pairs (n²) still cost only 0.05 ms at 1500 bodies; needed past about 10k |

### Air and cloud

| ID | idea | status | expected gain / why |
|---|---|---|---|
| A1 | Sparse bricks for the cloud's MacCormack pass instead of its bounding box | idea | the pass touches only voxels with cloud |
| A2 | Run the air and the cloud at a lower rate than the bodies when nothing violent happens | idea | the air is the most expensive part of a CPU physics step (VOLUMES.md, PHYSICS.md 7) |

### Web

| ID | idea | status | expected gain / why |
|---|---|---|---|
| W1 | Bring this round's GPU changes to the browser: the new traversal stack is in the web build (a GPU path traced image checked in Chrome, 5 ms per pass on the spheres); the wavefront kernels (`?wavefront=on`) were not checked: their first compile in Chrome had not finished after more than a minute (stopped there) | stopped (no web work for now then) | the same speedups in browsers; the wavefront needs one shader module per kernel (or smaller ones) before it can be used there |

### Platform

| ID | idea | status | expected gain / why |
|---|---|---|---|
| X1 | Own Vulkan backend behind the same `webgpu.h` (drop wgpu-native), with our own WGSL → SPIR-V compiler | done: the layer since 2026-11-16 (step 6), the only native GPU since 2026-11-18 (step 12, 1.8) | no per-call validation or conservative barriers (P10), the hardware's features (subgroups, 64-bit atomics, bindless, more bindings, a second queue), hardware ray tracing (R6); less foreign code |

## 5. Backlog (everything else to do, gathered 2026-11-17)

Not scheduled; the IDs point to section 4 (the ideas, with their measurements).

Agreed for later (2026-11-17):

- **A Linux window for our Vulkan layer**: the X11 surface is written (2026-11-18, 1.8) but
  not yet run on a Linux machine with a GPU; Wayland not yet.
- **Automated builds and tests**: GitHub Actions building and running `make test` on Linux
  and Windows (float and double, our layer where a runner has a GPU) for every push.

Physics:

- **P5, AVBD solver** ("very good to have"; started on the branch `physics-avbd`, which the user rebases).
- **P11, the rest**: tall piles on the GPU fall fully asleep in 2 of 3 runs; the rest needs a
  Gauss-Seidel GPU solver by colours (P5's colouring).
- **P6, the rest**: SIMD in the CPU solver (AVX2 natively, WASM SIMD: W6).
- **P4**: CPU or GPU chosen by the number of bodies. **P9**: fewer speculative contacts on
  the CPU. **P3b**: GJK/MPR for hulls if the full test costs again. **P7**: a broad-phase tree
  past ~10k bodies (the CPU's is sort and sweep now).
- **P10, the rest**: a world partly asleep still pays all the GPU dispatches.

Rendering:

- **R7, anti-aliasing, the rest** (FXAA done): MSAA in the GPU rasterizer (needs multisampled
  targets in our Vulkan layer), SMAA, coverage MSAA in the CPU rasterizer, TAA in the viewer,
  the path tracer's denoiser guide jittered and a better pixel filter.
- **R2**: upscaler / temporal SVGF.
- **R1 in browsers**: the wavefront kernels (W3).
- **R8, the rest**: the wgpu path tracer's GPU top level traces 10-25% slower than the CPU's
  SAH tree (PLOC or a refit could close it); simpler copies of the unit sphere for the
  physics' balls; levels of detail for the path tracer's preview.
- **M1**: surface depth from textures (height maps and parallax, detail maps; below).
- Mixed materials for the falling meshes (`-meshmat mirror,diffuse`).

Air and cloud: **A1** (sparse bricks for the cloud's advection), **A2** (the air at a lower
rate when nothing violent happens).

Our Vulkan layer (VULKAN.md "Order"): **remote rendering** to browsers (native rendering
streamed where the browser lacks a technology, e.g. hardware ray tracing); the browsers'
new features (subgroups, ray tracing) as they ship.

Tests and tools: the web checks of W2 in the GitHub Actions; the GPU physics tests are
timing-sensitive on piles (fixed with tolerances; a seeded, waited mode would make them
exact).

### M1: surface depth from textures (an idea, agreed 2026-11-15)

Make surfaces look deep and detailed without more triangles: the model stays simple and
textures carry the relief. Normal maps (PBR5) already bend the light; what they cannot do is
move what is seen (a brick's edge does not hide the mortar behind it), cast the relief's own
shadows or change the silhouette. The textures to add, each with a fallback when a model
lacks it:

- **height maps and parallax**: a height (depth) texture; the rasterizers march the view ray
  through it (parallax occlusion mapping, Tatarchuk 2006; relief mapping) so the texture
  coordinates shift where the surface would really be, with the relief's self-shadowing
  towards the lamp from the same march. The path tracers the same at each hit (the ray
  marched in the height field, then the shading point and normal there), so all four
  pipelines still agree. The procedural bricks get their height map (it is already their
  normal map's source);
- **detail maps**: a second, finer normal and colour texture repeated many times over the
  first, so close-ups stay sharp without huge textures;
- the textures glTF already brings in, used further: the occlusion texture also on the
  lamp's light where the relief hides it (only the light from around today); and a
  displacement map as the step after, where the triangles are subdivided only near the camera
  (tessellation) for real silhouettes, if parallax is not enough.

glTF has no height map in its core material: an option to give one (`-heightmap`, as
`-normalmap`), and the extensions that carry one read when models use them.

## 6. History

### 6.1 The order of 2026-11-14

| # | step | status | what it is |
|---|---|---|---|
| 1 | **PNG, and the FPS counter on the image** | done (050baf2, 97091cd) | below |
| 2 | **P3: a cheaper hull–hull test** | done | below; P3 |
| 3 | **P8: a parallel prefix sum for the contact lists** | done | below; P8 |
| 4 | **R4b: PLOC, a GPU tree builder of SAH quality** | done | below; R4b |
| 5 | **PBR: one physically based material model for all four pipelines** | done: PBR1–PBR6 (2026-11-14/15) | [PBR.md](PBR.md), phases PBR1–PBR6 |
| 6 | **Our own Vulkan backend**: a WGSL → SPIR-V compiler and `webgpu.h` on Vulkan in place of wgpu-native, then native extensions, hardware ray tracing (R6) and remote rendering to browsers | done (merged into `main`, 2026-11-16; remote rendering not yet) | [VULKAN.md](VULKAN.md); X1, R6, P10 |

#### PNG, and the FPS counter on the image

PNG replaces PPM completely (the user's choice: the engine writes PNG itself, in C, and PPM
goes away), reading and writing, so that every image the engine makes or reads is one any
program opens.

- `formats/png.{c,h}`, from scratch, no library: CRC-32 and Adler-32; a deflate encoder (LZ77
  with hash chains and fixed Huffman codes: a good part of zlib's ratio in a few hundred
  lines); the PNG filters chosen per row (the smallest sum of absolute differences, as libpng
  does). A decoder too, needed by the PBR textures (PBR5, glTF): inflate (stored, fixed and
  dynamic Huffman blocks), all five filters, colour types grey, grey + alpha, RGB, RGBA and
  palette, bit depths 1–16 (to 8 bits), Adam7 interlacing.
- Everything that wrote `.ppm` writes `.png`: `render` and `sim` (`frame_NNNN.png`), the
  viewer's screenshots, the examples and `make` targets; `tools/frames_to_gif.py` reads PNG;
  `texture_load_ppm` becomes `texture_load_png`; the tests and `make bench` compare PNGs.
  The default output names change (`out/cornell_boxes.png` ...), the docs with them.
- A test: write and read back random images of every colour type and size (round trip
  exact), and decode reference files written by another encoder (Pillow, via uv, once, kept
  in the repo).
- **FPS on the image**: a small readout (frames per second and ms per frame) in a corner of
  the frame, drawn with a built-in 5×7 pixel font, on the CPU framebuffer and in the GPU's
  compositing pass; on by default, a key toggles it. The title keeps it too.

#### P3: a cheaper hull–hull test

The hulls' contacts are the biggest cost of a pile (2.6 of 4.7 ms per step with 1500 hulls
on the GPU). The separating-axis test tries every pair of edges (48 corners per hull: about
19k pairs per pair of hulls).

- Fewer corners where they do not show: measure the hull's error against the mesh for
  24, 32 and 48 corners.
- GJK for the distance (and EPA or the SAT on the touching faces only for the depth) instead
  of all edge pairs; the contact points from clipping the two touching faces, as now.
- Kept contacts (PhysX's persistent contact manifold): each pair remembers its separating
  axis or touching faces from the last step; a pair that barely moved starts there and most
  of the time needs one test.
- CPU and GPU together; measured with `mesh_rain` and the tall piles of P11.

Done (2026-11-14): the remembered pairs (the third point), CPU and GPU; the corners measured
and kept at 48 (32 would be 1.6x faster but leave up to 7% of a round shape out); GJK not
needed for now (P3b). Results in 6.2.

#### P8: a parallel prefix sum for the contact lists

"Contact lists 1" is one GPU thread adding up the contacts of every body (0.1–0.4 ms).
A work-efficient scan (Blelloch; or decoupled look-back, Merrill & Garland 2016) over
workgroups, with a fallback when subgroup operations are missing.

Done (2026-11-14): a single workgroup of 256 threads is enough for the bodies there are
(each sums a run of them, the 256 sums are scanned together): 0.10–0.14 ms → 0.004 ms.

#### R4b: PLOC

Meister & Bittner, "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy
Construction" (2018): a GPU builder close to SAH quality, for when the CPU has no top-level
tree and for meshes that change. Compared with today's LBVH and the CPU's SAH tree on build
time, tree cost (bvh_cost) and ray speed.

Done (2026-11-14): meshes' trees 8-9% cheaper than the LBVH's and rays 3-4% faster, built
1.3-1.8x slower (the 871k-triangle dragon: 10.5 ms instead of 8.4); the top level as good
either way. So PLOC builds meshes' trees and the LBVH the top level (`LR_GPU_BVH`); the
CPU's SAH trees are still the fastest to trace, and are used whenever they are current.

#### PBR

The design and its six phases are in [PBR.md](PBR.md): the glTF metallic-roughness model
with transmission, one shading code for the CPU and GPU rasterizers and path tracers, an
environment for the rasterizer (light probes from the scene and HDR skies), textures, and
glTF files with PNG and JPEG textures. All six phases are done (`-gltf`, `make
gltf-download` for Khronos's sample models).

### 6.2 What was done, and what it measured

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
| 2026-11-17 | **The viewer's frame rate** (1.2; branch `web`): a frame every time round (`-maxfps`), the path tracer's work per frame within the display's period, sized from the GPU's own timer of a pass (`gpu_view_trace_budget`), overlapping the wait for the display in a window; `-vsync off`, `-band ms`, `-stats file`; timestamps asked for whenever the GPU has them; group I of the suite | 1.2's table; section 2 |
| 2026-11-17 | **The performance suite** (1.3; branch `web`): `bin/bench`, groups A-H, quick and full, natively on wgpu and our layer and in a page (JSPI, run by `tools/webbench.mjs` in a headless Chrome); the engine's GPU timers readable by programs (`gpu_set_timing`, `gpu_view_gpu_ms`, `gpu_rigid_gpu_ms`, `gpu_fluid_gpu_ms`); a page built to wait gives the browser its turns by a message to itself (setTimeout's 4 ms clamp made every wait 4 ms). Found on the way: our layer kept one table of device functions, and a second device (here one without hardware rays) cleared the first one's ray-tracing functions: a crash at its next build of an acceleration structure (fixed: the first device's are kept) | section 2 |
| 2026-11-17 | **Bigger scenes** (R8, branch `next`). The GPU rasterizer culls the instances to the frustum and draws one call per mesh and level of detail (an order buffer between the draw's instance and its record); from 4096 instances on, a mesh instance small on the screen is drawn from a simpler copy (vertex clustering on grids of 16 and 6 cells: `scene_mesh_simplify`). The GPU path tracer builds the top level itself (LBVH) from 32768 instances on, where the CPU's SAH tree cost 12 ms a frame. Each instance carries the scene's change number of its last move (`Instance.version`), so the rasterizer's and the path tracer's records and our layer's top-level records go to the GPU again only for those moved or added; our layer then updates its top level in place (`lrCommandEncoderUpdateTlas`, at most a quarter moved, a build every 32). Found on the way: the GPU bodies' readback could be asked for without its copy (the last one's callback landing during the submit), so the newest body came back as zeros, and when the buffers grew right then it went back to the GPU as such: NaN, and the contacts filled their buffer (2 of 6 runs of the 18k scene at 150 frames/s; fixed) | RTX 3070, `-demo sky`, 1280x720 offscreen, 2000 bodies falling, wgpu / our layer. **Rasterized**, frames/s: 18k props 34 / 43 → 136 / 150; 48k props 16 / 20 → 121 / 132 (wgpu, step by step: culling alone about 65, with the levels of detail 103, with the records sent by change 121). **Path traced**: 48k props, wgpu 23 → 30 frames/s (the CPU's tree 12 ms a frame → none; rays slower through the LBVH: 12.5 → 14-15.6 ms a pass), 18k as fast either way (31-32, so the CPU's tree is kept there); our layer at 65-69 frames/s either way, its top level's update against its build no difference to measure (3.68 / 3.69 ms a pass). "GPU scene update" with 48k props 2.7 → 1.2 ms (wgpu), 2.3 → 1.0 (our layer). Images: the 64 of the set identical to the byte; the 48k scene with and without the levels of detail looks the same (out_next/lod*.png) |
| 2026-11-17 | **Physics robustness** (branch `next`; PHYSICS.md 1.2, 1.3b, 1.3c, 1.5). The CPU's pairs by sort and sweep (the same contacts); its contacts found on all cores; fast bodies swept (CPU and GPU); the margin from the whole relative speed; the push out taken back after each substep; slow resting bodies damped; sleeping when going nowhere, not on a moving neighbour | CPU, 1000 falling bodies: 12.9 → 5.3 ms a step (sort and sweep, one thread), then 59.7 → 29.5 ms on 16 threads in the pile's busiest second (more contacts by then: the larger margins); a 4-unit slab now stops a ball, a box and two balls head on at 200 to 50000 units/s (before: a box went through at 500); a tall pile of 400 asleep in 11 s (6 of 6 runs, before 11–29 s and 1 of 6 not after 30 s). GPU, `mesh_rain` 1500 hulls, RTX 3070: 4.5 → 5.1 ms a step on our Vulkan layer, 4.6 → 5.3 on wgpu (the sweep 0.21 ms, the relax pass ~0.3 ms, more pairs from the margins); the sky demo's 2000 bodies 1.0 → 1.5 ms (our layer), 1.4 → 1.9 ms (wgpu) |
| 2026-11-16 | **The review's last six** (branch `vulkan-review`). The cloud on the GPU no longer copied once per step: its two buffers take turns as the velocity's do, and a view asks the fluid where the cloud is each time it copies it (a link the two share, `GpuDensity`: either can be destroyed first). The tree builds' bind groups (one per pass, made every frame: 114 for an LBVH over 1500 instances) and the wavefront kernels' (six per wave and pass) are kept (`gpu_bind_cached`), their parameters chosen by a dynamic offset. A read back that fails is told to who waits for it (`render`, `sim`, the viewer's screenshot waited for ever); `test_gpu_fluid` runs natively in `make test`; meshes, volumes and the world's triangles are known by numbers of their own, not addresses; the air's step is timed (`LR_GPU_TIMING=1`), and every timer prints its round from the first pass's start to the last one's end. Found on the way: in a browser the bodies' kernels were not compiled (WGSL wants parentheses around `*` beside `^`, `<<` beside `|`), and the wavefront tracer with a medium does not give the same image twice (about 200 of 65536 pixels, before and after) | RTX 3070, before â†’ after, wgpu / our layer, two rounds each. **The air** (`view -sim on`, 32^3 cells), GPU ms per step from the first pass's start to the last's end: a cloud of 96^3 0.81-0.85 / 0.80-0.81 â†’ 0.81-0.82 / 0.79, of which between the passes (the copy) 0.03 / 0.03 â†’ 0.02 / 0.01; a cloud of 192^3 (`-cloudres 192`) 1.59-1.61 / 1.56-1.57 â†’ 1.51-1.52 / 1.43, between the passes 0.10 / 0.14 â†’ 0.015 / 0.008; the passes themselves unchanged (pressure 0.63-0.66, the cloud's advection 0.10, 0.77-0.82 at 192^3). **The top level built on the GPU** (`LR_GPU_TLAS=gpu`, the sky demo's 4988 instances, 1280x720, `-backend rt`; our layer with `LR_GPU_HWRT=0`), "trace submit" per frame: LBVH 0.76-0.79 / 0.26-0.27 â†’ 0.40-0.41 / 0.18-0.20 ms, PLOC (`LR_GPU_BVH=ploc`) 0.66-0.68 / 0.25-0.27 â†’ 0.40-0.42 / 0.19-0.21; the builds' GPU time as before (0.19-0.24 and 0.49-0.62 ms). **The wavefront** (`-wavefront on`): no gain to speak of, its submit is the dispatches themselves (about 40 per wave and pass): the sky demo 1.49-1.75 / 0.62-0.63 â†’ 1.30-1.76 / 0.53-0.63, the spheres with the camera orbiting 4.36-4.65 / 0.90-0.94 â†’ 4.28-4.38 / 0.86-1.15, the drop 1.30-1.96 / 0.62-0.65 â†’ 1.28-1.91 / 0.60-0.65. Unchanged: the sky demo's path tracer 10.19 / 3.20 â†’ 10.29 / 3.23 ms per pass, its "trace submit" 0.13 / 0.05 â†’ 0.13 / 0.06, rasterized offscreen 3.79 / 3.13 â†’ 3.81 / 3.15 ms per frame, `mesh_rain` 1500 hulls 4.50 / 4.42 â†’ 4.56 / 4.45 ms per step. Images: the 64 of the set (CPU, GPU raster, GPU path traced, `sim`) identical to the byte; 80 frames of `sim` with the air on the GPU (rasterized, path traced, `-demo blast`) identical on both; with the top level built by the LBVH and by PLOC, the meshes' trees built on the GPU and the wavefront without a medium, 35 of 40 identical (the other five: the wavefront with a cloud, which differed between two runs before too) |
| 2026-11-16 | **The review's leftovers** (branch `vulkan-review`). The viewer makes the CPU's top level over the moving instances when something reads it (the CPU path tracer, the probe's bake, the GPU path tracer where it copies it: `gpu_view_trace_reads_tlas`), not every simulated frame: never while the GPU rasterizes, or path traces with ray queries. The instances' records (path tracer and rasterizer) and the hardware's top level are sent and built again only when an instance changed (`Scene.inst_version`, now a number no two states share), from memory kept between frames; the bind groups of every frame (composite, text, the media's statistics and light, the tracer's, its resolve and reprojection, the denoiser's) are kept (`gpu_bind_cached`) until a buffer they bind changes. The air on the GPU advects its velocity into the buffers it does not read and goes on from those (three copies per step less; the cloud's one copy stays: the view draws from that buffer); on the CPU the cloud's array and its copy change places, and only the voxels the last step wrote are copied. `render -gpu on` and `sim` make exactly `-s` samples (`gpu_view_trace_limit`): the same file every time; and a buffer the path tracer had to enlarge is known by `gpu_buffer_reserve`'s answer, not its address (a new one at the old address was taken for the old one: a model in the box then lost its normals, about one run in four on wgpu) | RTX 3070, 1280×720, before → after, wgpu / our layer. `view -sim on -demo sky` (4988 instances), "scene trees" per frame: `-backend raster` and hybrid 1.43 / 1.40 → 0.00 / 0.00 ms, `-backend rt` 1.27 → 1.24 (wgpu still copies the tree) / 1.33 → 0.00; "trace submit" 0.19 → 0.14 / 0.08 → 0.06. The same with `-gpu offscreen -backend raster` (no display rate): 4.64 → 3.57 / 3.42 → 2.91 ms per frame (85 → 96 / 99 → 117 frames/s). Paused, the camera orbiting, `-backend rt`: "GPU scene update" 0.27 → 0.03 / 0.13 → 0.02 ms; rasterized offscreen 118 → 121 / 130 → 137 frames/s. `view -sim on -backend rt` (the drop, its cloud): "trace submit" 0.24 → 0.19 / 0.10 → 0.08. GPU ms unchanged: `mesh_rain` 1500 hulls 4.55 / 4.50 → 4.69 / 4.49 per step, the sky demo's path tracer 10.99 / 3.19 → 10.43 / 3.25 per pass. Images: the 40 CPU and GPU-raster ones and 40 frames of `sim -demo drop` rasterized on the GPU identical to the byte; CPU `sim` 90 frames identical; `render -gpu on` the same file in 10 runs of each scene on both (a model in the box: 22 and 8 of 30 runs before, 60 of 60 after) |
| 2026-11-16 | **A review's small items** (branch `vulkan-review`; nothing changes in what is computed). With ray queries the path tracer no longer packs and uploads the CPU's top level and the instances' boxes every frame (the hardware has its own top level). The float build uploads the CPU rasterizer's z-buffer as it is (it was copied into a second array of floats first, as the double build needs). The GPU rasterizer draws neighbouring instances of one mesh in one draw (the same triangles in the same order). The GPU bodies' buffers for body - mesh pairs (2.5 KB per pair, 32 pairs per body of capacity) are full size only once the world has a fixed mesh | RTX 3070, the two versions alternating. `view -sim on -demo sky -backend rt` on our layer, "GPU scene update": 0.28-0.32 → 0.17-0.22 ms per frame. `view -backend raster -gpuraster off -gpu offscreen -orbit 15`, 1280×720, "present": wgpu 2.53-2.56 → 2.33-2.36 ms, our layer 2.16-2.17 → 2.01-2.03. Encoding the raster pass, `view -demo sky` (4988 instances): wgpu 0.21 → 0.17 ms, our layer 0.32 → 0.26; `mesh_rain` 1500 hulls: 0.06 → 0.06 and 0.10 → 0.08. GPU memory of a world without a fixed mesh: 42 MB less with 500 bodies, 168 MB with 1500, 672 MB with 8000. `render`'s and `sim`'s images identical to the byte, `mesh_rain` 1500 hulls as before (GPU ms per step, wgpu / our layer: 4.65 / 4.47 → 4.69 / 4.51) |
| 2026-11-16 | **Physics for larger scenes** (branch `physics-speed`): over-relaxed Jacobi iterations, 10 instead of 16 (Box2D v3's soft step tried first: with the Jacobi solver piles no longer fell asleep, dropped); the hull pairs the memo settles (20000 of 32000 apart, 5000 the same feature) one thread each in `K_HULLPRE`, the rest queued for `K_HEAVY`; the pairs and wakes through a hashed grid, 27 threads per body, instead of a thread per pair of bodies | RTX 3070, GPU ms per step. `mesh_rain` 1500 hulls: solver 2.6-3.0 → 1.8-1.9, hull contacts 4.2 → 2.1-2.3, the step 6.8 → 4.5-4.7 (60 frames/s). 8000 small hulls (`-size 22 -max 8000`): pairs 1.5 → 0.13, wake 2.1 → under 0.05, the step 12.5 → 9.0-9.9 (30 → 37-43 frames/s). 600 hulls all asleep after 30 s as before |
| 2026-11-16 | **`mesh_rain`: fixed physics steps of 1/60 s**, as many as the time owed holds (the rest carried over, at most 2 a frame), instead of `ceil(dt / (1/60))` steps of `dt / n`: a frame a little over 1/60 s made two half steps, which at 1500 hulls kept every frame over 1/60 s. (The viewer, steps of up to 1/30 s, already makes one a frame; `sim`'s films unchanged) | RTX 3070, 1500 hulls: 55 → 60 frames/s (the display's rate) with wgpu and our layer; 300 hulls still all asleep after the rain stops |
| 2026-11-16 | **The hull test split in two, and its data in workgroup memory**: `K_HEAVY` (one workgroup per pair) now only chooses the feature; its contacts (clipping the incident face, one thread's work, the other 63 waiting) are made by `K_HULLC`, one thread per pair. b's edges' indices join the corners and normals in workgroup memory. The same tests and choices; tried and dropped: a's edges against quarters of b's (slower) | RTX 3070, `mesh_rain` 1500 hulls at steady state, GPU ms per step (runs vary by ±0.2): hull contacts 5.03 → 3.3-3.7 + 0.3 (the clipping); the step 8.1 → 6.6-7.1; 45 → 55 frames/s. The solver (64 passes, 2.6 ms) reads its contacts' 7 vec4 per body per pass: near the memory's speed |
| 2026-11-16 | **Profiling pass** (gprof on the CPU builds in a Linux VM, `LR_GPU_TIMING` and the GPU timers on Windows). The second Sobol dimension in 5 shifts instead of a loop over the index's bits (CPU and WGSL: the same points, the images identical to the byte); the blue-noise mask's update in two straight runs (vectorized); the procedural textures' rows in parallel; the multigrid's neighbour count without `__builtin_popcount` (a libgcc call without `-mpopcnt`); the hull test's corners and face normals in workgroup memory, made once per pair. Not taken: `-O3` (the same speed), `-march=native` (no faster, other images), assembly (the hot loops are already the compiler's best: the 4-wide BVH, binned SAH) | RTX 3070: GPU path tracer, spheres 640², 2.19 → 1.97 ms per pass; `mesh_rain` 1500 hulls at steady state, per physics step: hull contacts 5.45 → 5.03 ms (our layer 4.95), the step 8.52 → 8.10 ms. Linux VM, 2 threads: `render -scene spheres` 1.10 → 0.98 s, `test_physics` 0.73 → 0.69 s. At 1500 hulls the frame is the physics: two steps per frame (8 ms each) once a frame is over 1/60 s |
| 2026-11-16 | **A second queue tried, not taken by default** (`LR_VK_QUEUES=2`: VULKAN.md): the rigid bodies on a queue of the compute-only family (asynchronous compute), or a second queue of the main family (`=same`), beside the drawing | Windows, RTX 3070, `mesh_rain` 1500 hulls (about 4.7 ms of physics and 18 ms of drawing per frame), frames/s: one queue 44-45, compute family 34-35 (the physics' passes 4.6 → 7.4 ms), main family 42; in a 320x240 window 48 / 36 / 47; 3 steps in flight instead of 2: 45 / 36. Both workloads fill the GPU and slow each other down: off |
| 2026-11-16 | **Subgroups tried, not taken** (the compiler and our layer support them: VULKAN.md). The solver's 8 threads per body summed by `subgroupShuffleDown` instead of workgroup memory, and the hull test's largest of 64 values with 2 barriers instead of 7, the same additions and comparisons in the same order (the same physics to the bit, checked) | Windows, RTX 3070, `mesh_rain` 1500 hulls, GPU ms per step without / with: our layer the solver 1.40 / 1.41, the hull test 2.91 / 3.03; wgpu 1.32 / 1.34 and 3.01 / 3.00: no gain (the kernels wait on memory, not on their barriers), so left out |
| 2026-11-16 | **Lighter barriers in our Vulkan layer**: before a command, a barrier from the stages of the commands since the last one (and their writes) to its own stages, instead of a full `ALL_COMMANDS` / `MEMORY_*` barrier; pass timestamps start when the compute work before has finished | Windows, RTX 3070, `LR_GPU_TIMING=1`, GPU ms per physics step: `test_gpu_rigid` 1 body 0.71 → 0.36 (wgpu 0.26), `mesh_rain` 1500 hulls 4.98 → 4.63 (wgpu 4.66 the same run). With no barriers at all (wrong results, a bound): 0.29 for 1 body. The same images and physics (with `LR_GPU_HWRT=0`, as wgpu's within 1 level) |
| 2026-11-15 | **R6: hardware ray tracing** (our Vulkan layer, `make VK=1`, a GPU with ray queries). `traverse()` with ray queries over a top level built every frame from records the CPU writes: the world's triangles, its spheres and quads as boxes (hit by the kernel's own tests), each mesh (its first triangle in the instance's unused SBT offset), the unit box for balls and boxes; triangles opaque, so the hardware keeps the nearest. The same `Found`, so every kernel (path tracer, light tracer, guide, wavefront, photon cache) uses it unchanged. At one sample per pixel and the same random numbers the images differ in a few hundred pixels of 160 000 (hardware and `hit_tri` round t differently, so a path now and then takes the other side); at 256 they agree within noise (mean under 0.02 of 255, the showcase 1.0 with its cloud) | Windows, RTX 3070, 640×640, `LR_GPU_TIMING=1`, kernel ms per pass (wgpu / our layer with the BVH / with ray queries): boxes 2.28-2.37 / 2.08 / 1.35, spheres 2.14 / 1.97 / 1.40, a sphere mesh 3.59 / 3.36 / 1.25, the showcase (dragon, helmet, torus, cloud) 14.96 / 12.17 / 7.40, the helmet 5.07 / 4.82 / 1.80, the open scene under a sky 1.29 / 1.13 / 1.03. `view -sim on -backend rt` 55 → 58 frames/s (the display's rate) |
| 2026-11-15 | **P10: one pass per solver iteration; a world all asleep not stepped** (GPU rigid bodies). Each body's 8 threads read its contact list 8 entries at a time: each contact's impulse worked out from the velocities and impulses before the iteration (both in two buffers used in turns), the 8 results added to the body's velocities by its first thread in the list's order, in the workgroup's memory; a contact is worked out by both its bodies with the same instructions (the side applied after, by an exact sign), and its first body keeps the result. It was a pass over the contacts and a pass over the bodies, whose thread waited on one contact after another. The warm start the same way. A world whose bodies all slept after a step read back, with nothing changed since (no body new or changed, the same forces), is not stepped (`gpu_rigid_skipped`). Two first versions rested less often (the 8 sums added in another order; the impulse computed in two branches, rounded differently for the two bodies, so momentum was not kept): 600 hulls all asleep after 40 s in 3 of 6 runs | Windows, RTX 3070, `LR_GPU_TIMING=1`, GPU ms per step: `view -sim on` 1000 bodies (585 on the GPU) 0.78 → 0.51 (the iterations 0.65 → 0.36); `mesh_rain` 1500 hulls 4.9 → 4.2 (iterations and warm start 2.36 → 1.28). `test_gpu_rigid`: 64 bodies 0.40 → 0.27-0.40 ms, 1024 0.80 → 0.40, 2048 0.93 → 0.53-0.67, the settled pile 1.07 → 0.80; all asleep: no step (`mesh_rain -max 600 -endless off`, physics 0.6 → 0.15 ms of CPU per frame). `mesh_rain` 1500 hulls 37 → 45 frames/s (average 49 → 54), 1500 cubes 27 → 42 (42 → 50). Settling as before: 600 hulls all asleep in 40 s; 1500 small hulls after 60 s vary run to run before and after (before: 1500, 1500, 1499, 1498, 1496, 1496, 1196, 651; after: 1500, 1498, 1427, 1398, 1342, 1186; P11) |
| 2026-11-15 | **The review's speed items** (a review and profile of the whole engine, then the hot spots it found). The GPU rigid step in one submission: the parameters of every substep in one write, copied into place inside the command buffer, the bodies changed on the CPU uploaded in runs, a wait for the oldest step in flight only, the solver writing back only the impulses. One texture atlas per GPU, its levels written straight into it; sRGB mips through a table of the steps between bytes (the same bytes), integer averages for linear channels. The path tracer's passes of a band share one bind group (dynamic offsets). A quarter of the light paths while things move. The viewer's estimate of a pass's time no longer sticks low while the camera moves. The CPU cloud advection samples the air once per voxel. The sampler's bit reversal by a byte swap | Windows, RTX 3070, 12 threads; before = 84e20c1, runs alternating. `test_gpu_rigid`, GPU ms per step: 64 bodies 0.67 → 0.40, 256 0.80 → 0.67, 1024 0.93 → 0.80, 2048 1.20 → 0.93. `view -sim on`, 1000 bodies: raster, physics 2.15 → 1.12 ms per frame; path traced, light tracer 5.9 → 1.4 ms, 24.0 → 19.3 ms per frame (582 → 727 frames in 15 s). `mesh_rain` 1500 hulls 34 → 37 frames/s (average 47 → 49), 1500 cubes 17 → 28 (37 → 42). The helmet: loaded and drawn at 16×16 in 1020 → 805 ms; orbiting, path traced, light tracer 1.52 → 0.48 ms, 865 → 941 frames in 10 s. The showcase orbiting: light tracer 1.36 → 0.62 ms. Offline GPU path tracer as before (ms per pass: boxes 2.27-2.35 → 2.26, the showcase 15.0 → 15.0, the helmet 5.06 → 5.06). CPU: path tracer, boxes 400×400 at 64 spp 3.24 → 3.20 s; the cloud's air step (Linux VM, 2 threads, `sim -demo blast`) 213 → 165 ms, `view -gpufluid off` (12 threads) 52 → 50 ms. Images identical where the change is exact (CPU path tracer, `sim` frames, GPU path tracer, the mips) |
| 2026-11-15 | **PBR6: glTF models and JPEG** (a cost only for the rasterizers' probe normalization; the rest is new). Loading: a model's images decoded in parallel, one per thread (the PNG decoder's tables, made on first use, now made under a critical section). The CPU rasterizer's probe normalization: the luminances from the first 4 harmonics instead of 9 (65.0 → 57.7 ms on the boxes) | Windows, RTX 3070 and 12 threads. JPEG: a 2048×2048 progressive file in 120 ms on the Linux VM (libjpeg-turbo, SIMD: 73). Loading (more than an empty scene): the damaged helmet (5 JPEGs of 2048×2048) 760 → 370 ms with the parallel decoding, MetalRoughSpheres (500k triangles) 190 ms, the normal-map test (3 PNGs of 2048×2048) 610 → 385 ms. CPU rasterizer 1000×1000, Phong, ms: boxes 52.6 → 57.7, spheres 49.8 → 53.5, the open scene 77.6 → 83.5; the helmet 69.5, MetalRoughSpheres 79.5. GPU path tracer, 640×640, ms per pass: boxes 2.44 → 2.49, the helmet 6.71, MetalRoughSpheres 9.51, the showcase with the helmet 17.6. CPU path tracer: boxes 3.08 → 3.01 M paths/s, the helmet 2.70 |
| 2026-11-15 | **PBR5: textures, normal maps, the irradiance grid** (a cost where a scene has textures or a probe). The rasterizers: three trilinear lookups per textured pixel and the BSDF's albedos per pixel instead of the material's tables; the irradiance grid's 8 probes blended per pixel. Kept down by finding a bilinear lookup's texels without integer division (82 → 67 ms on the open scene, its probe off) and by blending the grid's probes without branches, 4 floats at a time (GCC's and clang's vector types; 140 → 80 ns per lookup) | Windows, RTX 3070 and 12 threads. CPU rasterizer 1000×1000, Phong, ms: boxes 40.2 → 53.2 (41.3 without the grid), spheres 37.9 → 49.8, the open scene 28.3 → 78.9 (its floor is now textured brick: a glossy PBR material with three maps where it was diffuse), a sphere mesh 51.9, with bricks 60.9. GPU rasterizer: no visible cost (15.6 ms per frame both, the display's rate). GPU path tracer, 640×640: boxes 149.7 M paths/s both, the open scene 225 → 202, a sphere mesh 5.09 ms per pass, with bricks 5.04 (wavefront 6.95). CPU path tracer: boxes 3.24 → 3.27 M paths/s, the open scene 6.42 → 5.36, a sphere mesh 3.81, with bricks 3.61. The probe and its grid: 250-350 ms in a room, 660 under a sky (PBR4, the probe alone: 220-470 and 620-700) |
| 2026-11-14 | **PBR4: the environment** (a cost where a scene has a sky or a probe; none elsewhere). The path tracers: a second light to sample at every rough vertex; the GPU's kernels compiled without the sky's code for scenes without one (with it in every kernel: boxes 151.8 → 134.2 M paths/s). The rasterizers: three lookups per pixel; the CPU's map coordinates by polynomial atan2 / acos (the library's: 680 → 110 ns per pixel for the environment) | Windows, RTX 3070 and 12 threads. GPU path tracer, ms per pass 640×640: spheres 2.19 (as PBR3: 151.5 against 151.8 M paths/s on the boxes), the same room with a sky through its front 4.07, the open scene under a 2k sky 1.08 (wavefront 2.82). CPU path tracer: boxes 3.25 M paths/s (PBR3 3.25), the open scene 5.97. CPU rasterizer 1000×1000, Phong, ms: boxes 25.7 → 40.2, spheres 24.3 → 37.5 with the probe (`-probe off`: as PBR3), the open scene 28.4. The probe: 220-470 ms in a room (256 × 128, 32 paths per texel), 620-700 under a sky (512 × 256); the levels and harmonics of a 2k sky: under 0.3 s |
| 2026-11-14 | **PBR3: the rasterizers shade with the BSDF and the quad lamp** (a cost: the old Blinn-Phong with a point light was a few multiplications). Kept down by the lamp's form factor without atan2 (Heitz and Hill's θ / sin θ fit, each corner normalized once), a single sample at the lamp's centre for lobes much wider than the lamp, and per material tables of what depends on one cosine (first version: 438 ns per pixel) | Windows, 12 threads, 1000×1000, `render -gpu off -backend raster`, ms: Phong boxes 10.8 → 25.7, spheres 10.0 → 24.3 (an empty frame: 6.4). Linux VM, 2 threads, the whole program with the PNG: Phong 76-83 → 141-146 ms. One pixel: 438 → 164 ns. The GPU rasterizer: no visible cost on the RTX 3070. (First recorded here as 53 → 63 ms: those were `render`'s GPU rasterizer, its default with a GPU, start-up included) |
| 2026-11-14 | **PBR2: the GPU path tracer shades with the BSDF** (WGSL copy of `scene/brdf.h`, `gpu/brdf_gpu.c`; a cost). The shadow ray is walked after the next direction is drawn, so the BSDF's state is no longer held through it: fewer registers (first version: boxes +36%, fog +31%) | RTX 3070, `LR_GPU_TIMING=1`, 640×640, ms per pass: boxes 1.91 → 2.23-2.42 (+20%), spheres 1.80 → 2.12 (+18%), dragon 871k 8.06 → 8.25 (+2%), spheres in cloud and fog 11.17 → 14.1-14.6 (+28%: its walls are hit many times per path) |
| 2026-11-14 | **PBR1: the CPU path tracer shades with the microfacet BSDF** (docs/PBR.md; a cost, not a gain): every surface is GGX + diffuse (+ multiple scattering, transmission), so a bounce evaluates the BSDF twice (next event estimation, and the sampled direction's density for one-sample MIS) where Lambert cost a few multiplications. Kept down by doing once per vertex what depends on the view alone (`bsdf_view`: the lobes' chances and albedos), drawing SAMPLE_MISC's pair once for Russian roulette and the lobe, and SAMPLE_DIR only for a direction that is not fixed (first version: -23 to -38%) | Linux (the VM on the RTX 3070 machine), float, all cores, M paths/s: boxes 1.13 → 0.96 (-15%), spheres (`-caustics pt`) 1.93 → 1.39 (-28%: few, cheap shapes, so the shading weighs most), dragon 48k triangles 1.18 → 0.97 (-18%); with light tracing (spheres) 1.63 → 1.14 |
| 2026-11-14 | **R4b: PLOC, a second GPU tree builder** (Meister & Bittner 2018; `PLOC` in `trace_gpu.c`): the leaves in Morton order, then rounds where each cluster finds the one of its 16 neighbours on each side whose box joined with its own is smallest, and mutual pairs merge (no atomics: the same tree every time). Rounds of 4 dispatches while more than 1024 clusters are left, then one workgroup does the rest with barriers (one dispatch for a top level of up to 1024 instances). Each node's two children are written side by side, as the LBVH's: written where they were made, the same trees traced 2-3% slower. Default: PLOC for meshes' trees (dynamic meshes), the LBVH for the top level (`LR_GPU_BVH`); `LR_GPU_BLAS=gpu` and `LR_GPU_BVH_CHECK=1` to measure | RTX 3070, `LR_GPU_BLAS=gpu`, rays per pass / tree cost / build: dragon 871k triangles LBVH 3.64 ms / 61.0 / 8.4 ms, PLOC 3.50 / 55.5 / 10.5 (CPU SAH tree 2.87 ms); dragon 202k LBVH 3.26 / 54.6 / 1.08, PLOC 3.15 / 50.3 / 1.97 (CPU 2.86). Top level (`LR_GPU_TLAS=gpu`): 40 dragons LBVH 1.54 ms / 0.09 ms build, PLOC 1.54 / 0.07 (CPU 1.56); `sim -demo blast`, 700 bodies, LBVH 2.78 / 0.15, PLOC 2.84 / 0.30 (CPU 2.71). The images match the CPU tree's (mean difference 0.1 of 255) |
| 2026-11-14 | **P8: the contact lists' prefix sum in parallel**: one workgroup of 256 threads (each sums a run of bodies, the 256 sums scanned together) instead of one thread | "contact lists 1" 0.10–0.14 → 0.004 ms per step (1000–1500 hulls); `test_gpu_rigid` 1024 bodies 1.17 → 0.98 ms, 2048 bodies 1.56 → 1.23 ms |
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

## 7. Tried and rejected

| when | what | why |
|---|---|---|
| 2026-11-18 | In the page (W4, 1.7): subgroups for the path tracer's tree walk (while-while with a `subgroupAny` vote), f16 for the denoiser's weights, workgroup sizes 32 and 128 for the path tracer, a 24- or 32-entry stack | the vote and f16 no faster; while-while 3-9% faster in the page but 40-90% slower natively; the other sizes no faster; a smaller stack is faster (A8 9.9 -> 7.8 ms at 24) but drops entries in deep trees |
| 2026-11-17 | The CPU solver's iterations by colours (no shared moving body) on all cores (OpenMP) | some 60 colours and 44 passes a step: 2600 barriers; 1000 bodies 60 → 111 ms a step on 16 threads, 49 ms on 4: the iterations stay on one thread |
| 2026-11-17 | Spin friction for balls (an angular impulse about the normal, limited as friction over a small patch) | stopped the spinning balls, but sleeping them earlier set off avalanches in tall piles; the settling damping (PHYSICS.md 1.5) does it without them |
| 2026-11-15 | The review's dead ends: the GPU's sRGB decode of texels by a 256-entry table instead of `pow` (the helmet 5.12 → 5.12 ms per pass); the photon cache at its finest level only (the light pass -7 to -9 %); skipping the instance upload when no instance moved (the GPU scene update is 0.02-0.06 ms per frame) | no gain worth it, or a loss of quality |
| 2026-11-15 | Skipping the light tracer when no specular object is in view | not taken: a mirror or glass behind the camera still throws caustics on what is in view |
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

## 8. Sources

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
- D. Meister, J. Bittner, "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy
  Construction", IEEE TVCG 2018
