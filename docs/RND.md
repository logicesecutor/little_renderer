# R&D: the engine in the browser at the hardware's full power, and hardware ray tracing on the web

This branch (`rnd`) is for experiments: new features of browsers and GPUs tried before
they are ready for `main`. Each experiment has its own section here (what it is, why, the
design, what was measured, what was decided). Nothing goes from here to `main` without a
measured gain and the owner's OK; experiments that do not pay stay here, written down, with
their numbers (as STATUS.md section 7 does for `main`).

Opened 2026-11-18. The decisions so far are in section 9.

Contents:

1. What "natively in the browser" means
2. Where the page stands today (measured)
3. What browsers offer today to use the hardware
4. Running: W4 and W6 on branch `webgpu-perf`
5. Hardware ray tracing on the web: can we unlock it?
6. Experiment RD1: a ray-query polyfill for WebGPU
7. Experiment RD2: ray queries in Dawn on Direct3D 12 (a patched Chromium)
8. Experiment RD3: feedback to the WebGPU group
9. Decisions
10. Sources

## 1. What "natively in the browser" means

A page that uses WebGPU already runs on the machine's GPU, natively. The browser is a
translation layer, not an emulator:

| browser | WebGPU implementation | the native API underneath | WGSL compiled to |
|---|---|---|---|
| Chrome, Edge, Brave, Opera | Dawn (C++) | Direct3D 12 on Windows; Vulkan on Linux, Android, ChromeOS; Metal on macOS | Tint: HLSL (then DXIL), SPIR-V, MSL |
| Firefox | wgpu (Rust) | Direct3D 12, Vulkan, Metal | naga |
| Safari | WebKit's own | Metal | MSL |

Our web build (`make web`, WEB.md section 1) is Emscripten with the `emdawnwebgpu` port: the
engine's `webgpu.h` calls go to the browser's WebGPU, and the same WGSL kernels run there
as natively on wgpu-native. So the rasterizer, the path tracer, the denoiser, the bodies
and the air already run on the RTX 3070 from the page.

What a page cannot reach, by the web's security model:

- Vulkan or Direct3D directly. Our own Vulkan layer (`make VK=1`, VULKAN.md) is native C
  that talks to the Vulkan driver; no page can load it.
- Any GPU feature the browser does not expose, vendor extensions included. Today that
  includes the ray tracing cores.
- More than the browser's limits (buffer sizes, bindings per stage and so on), which may be
  lower than the GPU's.

What costs time in a page and not natively:

- **The GPU process.** The page's commands are serialized and sent to a separate,
  sandboxed GPU process (Dawn's "wire"), which replays them on Direct3D 12. Every submit,
  map and callback crosses it.
- **Robustness.** Tint adds bounds checks (clamps) to buffer and array accesses, so that a
  page cannot read outside its memory. A path tracer's BVH walk and its stack are mostly
  such accesses.
- **A second compiler.** WGSL goes through Tint to HLSL, then through Microsoft's DXC (or
  FXC) to DXIL, then the driver. Natively wgpu goes through naga and the Vulkan driver; our
  layer through our own WGSL to SPIR-V compiler. Different compilers make different code
  out of the same kernel (register use, the stack in private memory).
- **The CPU code.** WebAssembly in a page runs on one thread unless the page is
  cross-origin isolated (section 3), and without SIMD unless built with it.

## 2. Where the page stands today (measured)

The full suite at 4197a1a (2026-11-18, RTX 3070, a 60 Hz display; STATUS.md 1.3 for how):

| case | unit | page (Chrome) | wgpu natively | our Vulkan layer | our layer, no hardware rays |
|---|---|---|---|---|---|
| A1 path tracer, boxes | ms/pass | 3.74 | 2.05 | 1.26 | 1.96 |
| A3 path tracer, dragon (871k triangles) | ms/pass | - | 8.15 | 1.32 | 7.41 |
| A4 path tracer, cloud | ms/pass | 16.0 | 10.1 | 7.29 | |
| A5 spheres in cloud and fog | ms/pass | 31.8 | 15.2 | 9.9 | |
| A8 sky demo, 3k props | ms/pass | 12.6 | 8.22 | 3.16 | 7.92 |
| A9 sky demo, 48k props | ms/pass | 25.6 | 17.6 | 3.87 | |
| C1 GPU rasterizer, sky 3k | ms/frame | 0.944 | 1.12 | 1.13 | |
| D1 denoiser, 1280x720 | ms/run | 1.70 | 1.97 | 1.48 | |
| E3 path traced frame, 2000 moving, 48k | ms/frame | 20.6 | 19.2 | 5.82 | |
| F2 bodies on the CPU, 1 thread | ms/step | 70.9 | 44.2 | 44.1 | |
| G3 air and cloud on the CPU | ms/step | 31.9 | 15.7 | 15.8 | |
| H1 a GPU ready | ms | 29.2 | 230 | 60.4 | |
| H3 the sky demo made (light probe baked on the CPU) | ms | 4040 | 730 | 723 | |

What it says:

- **Without hardware rays our layer is as fast as wgpu** (A1c 1.96 against 2.05, A3c 7.41
  against 8.15, A8c 7.92 against 8.22). Its lead, 1.4-6.4x, is the ray tracing cores.
- **The page path traces 1.6-2.1x slower than wgpu** on the same kernels (A1, A4, A5, A8,
  A9). That gap is the browser's (section 1) and can be closed in part: W4.
- **The rasterizer and the denoiser cost the same in the page** (C1, D1): they are not where
  the browser loses.
- **The CPU code is 1.6-5.5x slower in the page** (F2, G3, H3): one thread, no SIMD. W6.
- So "the hardware at full power in a browser" has two parts: the browser's own gap (W4,
  W6: section 4), and the ray tracing cores, which no browser exposes yet (section 5).

## 3. What browsers offer today to use the hardware

GPU (WebGPU):

| feature | where | what it is for here |
|---|---|---|
| `subgroups` | shipped in Chrome 134 (2025); `enable subgroups;` in WGSL | SIMD-level operations among the threads of a wave: reductions, prefix sums (the physics' contact lists, the stats), the BVH walk (a ballot to keep a wave's rays together), the denoiser. Our layer has it too (VULKAN.md); on the physics' kernels it gained nothing (2026-11-16: they wait on memory), so each use must be measured |
| `shader-f16` | Chrome; with `subgroups` only when both are asked for | half floats where precision allows (the denoiser's weights and colours): half the registers and bandwidth |
| `timestamp-query` | Chrome (coarse unless Chrome runs with unsafe APIs) | the GPU's own time per pass in the page: the tool to find the 2x |
| `requiredLimits` | everywhere | the adapter's real limits (storage buffer size, workgroup memory, invocations) instead of the defaults |
| `createComputePipelineAsync` / `createRenderPipelineAsync` | everywhere | pipelines compiled without stalling the first frames |
| immediates (push constants) | intent to ship in Chrome (2026) | small per-dispatch values without a uniform buffer write |
| non-uniform indexing, bindless | in the specification's work (a PR merged 2026-10-27) | textures and buffers chosen per ray: what ray tracing needs (section 5) |
| subgroup matrices | experimental (Chrome) | matrix tiles in shaders; for the path tracer, little |
| acceleration structures, ray queries | proposals opened 2026-10 (section 5) | the ray tracing cores |

CPU (WebAssembly):

| feature | where | what it is for here |
|---|---|---|
| SIMD128 (`-msimd128`) | all current browsers | the physics' and the BVH builder's vector maths |
| relaxed SIMD | Chrome, others in progress | fused multiply-add and faster min/max where exact results are not needed |
| threads (Emscripten pthreads over SharedArrayBuffer) | all current browsers, but only in a **cross-origin isolated** page: the server sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` | the light probe's bake, the bodies, the air, the top level's builds on all cores (natively OpenMP). `make serve` can send the headers; a static host such as GitHub Pages cannot (a service worker that adds them is the usual way round) |
| Memory64 | Chrome 133+, Firefox 134+, not Safari | more than 4 GB of memory (big scenes, big meshes) |
| WebGPU and OffscreenCanvas in a worker | all current browsers with WebGPU | the frame loop off the page's main thread |

## 4. Running: W4 and W6 on branch `webgpu-perf`

Started 2026-11-18 (STATUS.md 1.4 has W4 and W6), each step measured before and after,
kept only when it gains:

1. **Measure** with `timestamp-query` where the page loses its 1.6-2.1x against wgpu, kernel
   by kernel (and the submit, map and readback overhead per frame, the limits asked for, the
   workgroup sizes, the bounds checks, the traversal stack in private memory, the compile
   stalls), written down before anything is changed.
2. **Subgroups** in the traversal, the reductions and prefix sums, and the denoiser;
   `shader-f16` where precision allows. Detected, with today's code as the fallback.
3. **Startup**: pipelines made asynchronously; the light probe baked on the GPU (H3: 4040 ms
   in the page).
4. **The CPU code**: WASM SIMD, and threads in a cross-origin isolated page (optional: the
   single-threaded page stays for static hosts).
5. **A seam for hardware ray queries**: the web build detects a ray query feature by name and
   uses the same hardware path as our Vulkan layer when a browser has one, the software walk
   otherwise. No feature is faked.

Its results go into STATUS.md and WEB.md on that branch, and a summary here.

## 5. Hardware ray tracing on the web: can we unlock it?

### 5.1 Not from a page

A page has the GPU features its browser exposes, and nothing else: a page cannot add a GPU
API, a browser extension cannot either, and WebGL has no ray tracing. A program outside the
browser (a native helper the page talks to) could reach the cores, but that is a server,
ruled out (section 9). So the ray tracing cores can be unlocked only **inside a browser**
(its WebGPU implementation changed), or **in the standard** (so that every browser gets
them), and meanwhile **emulated** with the same API (a polyfill).

### 5.2 The standard

- **gpuweb issue #535, "Ray Tracing extension"** (since 2019): exploratory; the group said
  ray tracing comes after WebGPU 1.0, and estimated at least 1-2 years for hardware ray
  tracing. The concerns raised there:
  - **Portability.** Metal's intersection queries differ from DXR's and Vulkan's pipelines.
  - **Bindless.** A ray can hit anything, so a shader must be able to reach every
    material's textures and buffers. Bindless is a prerequisite.
  - **Security.** The build's scratch memory and the instance records must be managed and
    validated by the API, so that a page cannot make the driver read outside its buffers.
  - **Ray queries or ray tracing pipelines.** Benchmarks quoted there show pipelines faster
    on AMD by a lot and on NVIDIA by a little, but such results do not carry from one
    vendor to another.
  - **Adoption.** It is still uneven across devices.
- **2026-10-13: proposals opened.**
  - **PR #11054** (by dj2): acceleration structures, which ray queries and pipelines would
    share. During review a reviewer reported a high CPU cost when wgpu manages the instance
    records of a large scene's top level. Our R8 work answers that: the top level is built
    and refit on the GPU from records written by a kernel.
  - **PR #11056 and issue #11055:** `rayQuery` in WGSL.
  - **State:** open; the acceleration structure proposal was the year's most discussed
    item (34 comments by 2026-10-27).
- **2026-10-27, bindless work.** Non-uniform indexing was merged in the specification work,
  a step towards bindless.

### 5.3 What exists

| project | what | hardware? | API | state |
|---|---|---|---|---|
| maierfelix/dawn-ray-tracing, chromium-ray-tracing (2020) | Dawn and Chromium patched for ray tracing | yes, Vulkan (`VK_NV_ray_tracing`) | ray tracing pipelines (shader binding table) | a prototype of its time |
| codedhead/WebRTX | a JavaScript library: ray tracing on plain WebGPU | no: compute shaders, a BVH built on the CPU | Vulkan-like ray tracing pipelines; GLSL shaders compiled to SPIR-V (glslang.js), then WGSL (naga) | unstable, Chrome only |
| willwaltz/webgpu-ray-tracing | patches to Dawn, Tint and Blink (Brave 1.97.59, Chromium 155) | yes, Vulkan only (`VK_KHR_acceleration_structure`, `VK_KHR_ray_query`); other backends stubbed | `GPUAccelerationStructure`, `device.createAccelerationStructure()`, `encoder.buildAccelerationStructure()`, `queue.writeAccelerationStructureInstances()`; WGSL `enable chromium_experimental_ray_query;` with `acceleration_structure` and `ray_query` | an experiment: a custom build (about 80 minutes), only for listed origins, not hardened against hostile pages, tested on Linux with an RTX 4090 |
| wgpu (Rust, Firefox's WebGPU) | experimental ray queries natively | yes (Vulkan, Direct3D 12, Metal) | naga's WGSL `ray_query` syntax | native programs only; not exposed to pages |
| **our Vulkan layer** (`webgpu_rt.h`, VULKAN.md "Hardware ray tracing") | `webgpu.h` extended with acceleration structures and ray queries | yes, Vulkan (ray queries) | bottom levels of triangles or boxes, a top level of 64-byte instance records, update (refit), bound by chained bind group entries; the WGSL of wgpu and the proposal (`acceleration_structure`, `ray_query`, `rayQueryInitialize` / `Proceed` / `GenerateIntersection` / `GetCommittedIntersection`, `RayDesc`) | in use: the path tracer's `traverse()` has a hardware twin, 1.4-6.4x faster |

As far as found (2026-11-18), nobody implements ray queries in Dawn's **Direct3D 12**
backend, which is what Chrome uses on Windows; and nothing emulates the **ray query**
form that the proposals and our layer use (WebRTX emulates the older pipeline form).

### 5.4 Three ways, and which

| | A. a patched browser (RD2) | B. a polyfill (RD1) | C. feedback to the standard (RD3) |
|---|---|---|---|
| what | Dawn and Tint taught ray queries on Direct3D 12 (DXR 1.1), in a Chromium build of our own | the proposed API on plain WebGPU: acceleration structures built by compute, `rayQuery` in WGSL rewritten into a software walk | our layer's design and measurements given to the group writing the proposals |
| hardware cores | yes | no (software, as fast as our best WGSL walk) | in time, for everyone |
| who can use it | whoever runs that build | every page, every browser with WebGPU, today | everyone, when browsers ship it |
| cost | high: a Chromium checkout (tens of GB) and builds of hours; Dawn's Direct3D 12 backend and Tint's HLSL writer to learn | medium: mostly code the engine has (section 6) | low |
| the risk | patches that age with every Chromium; never shipped as is | the proposal's names and rules change (an adapter layer absorbs it) | none |
| useful to others | as a prototype for the proposal's Direct3D 12 side | very: the future API now, and a fallback later for devices without the cores | yes |

Chosen order: **B (RD1) and C (RD3)** first; A (RD2) only as an experiment, later, and
only with the owner's OK for the downloads.

## 6. Experiment RD1: a ray-query polyfill for WebGPU

### 6.1 The goal

A small library that gives any WebGPU page the ray query API of the proposals today, in
software, and gets out of the way when a browser has the real one:

- **Same API.** The same JavaScript and the same WGSL as the proposals (as close as their
  current text allows), so that code written against it runs unchanged on a future browser
  with ray queries.
- **Detection.** When the adapter has a native ray query feature, the library does nothing
  and the browser's hardware path runs.
- **Correct.** The same intersections as the hardware (checked against our Vulkan layer's
  ray queries, section 6.7).
- **As fast as a software walk can be.** Our tuned WGSL traversal, with subgroups where they
  help.
- **No dependencies, no build step.** An ES module and WGSL text; the engine's Boost licence.

Not goals: ray tracing pipelines (shader binding tables, recursion); hiding the speed
difference from hardware; Safari before it has WebGPU features we need.

### 6.2 The API it offers

JavaScript, after `installRayQueryPolyfill(device)` (or `requestDevice` wrapped):

| proposal / our layer | the polyfill |
|---|---|
| `device.createAccelerationStructure({ geometries | instances, ... })` (`lrDeviceCreateBlas`, `lrDeviceCreateTlas`) | an object holding its buffers (section 6.3), sized at creation |
| `encoder.buildAccelerationStructure(as, ...)` (`lrCommandEncoderBuildBlas` / `BuildTlas`) | compute passes encoded into the same encoder: the build on the GPU |
| an update of a top level (`lrCommandEncoderUpdateTlas`) | a refit pass (boxes made again, the shape kept) |
| instances written by the CPU (`queue.writeAccelerationStructureInstances`) or by a kernel (64-byte records, as `LRTlasInstance`) | the same 64-byte records, read by the build |
| a bind group layout entry and a bind group entry of type acceleration structure | expanded into the few storage buffers it stands for (section 6.4) |
| `device.createShaderModule({ code })` with `enable ...ray_query;` | the WGSL rewritten (section 6.5), then compiled by the browser |

The proposal's final names are not settled: the library keeps one table of names (ours and
the proposal's current ones) so a rename is a one-line change.

The engine's side: on the web build, `webgpu_rt.h` (today implemented by our Vulkan layer
only) gets a second implementation on top of the polyfill, so the path tracer's
hardware-twin `traverse()` and its build code run the same way in the page. With that, the
browser build and the native build share one ray tracing path, and the day a browser ships
ray queries the page uses the cores with no change to the engine.

### 6.3 Data on the GPU

Built entirely on the GPU, from the engine's code:

- **Bottom level (per mesh).** A binary BVH:
  - built by the GPU builder the engine already has (`engine/gpu/trace_gpu.c`: Morton codes,
    a radix sort, an LBVH, then PLOC for a better tree: R4b in STATUS.md);
  - nodes of 32 bytes (two child boxes, or a quantized wide node if RD1's measurements
    favour it: R3);
  - triangles stored in leaf order (positions only; the hit reports the primitive index and
    barycentrics, the shader fetches the rest itself);
  - boxes (for `LRGeometry_Boxes`, procedural primitives) stored as such, each one reported
    as a candidate.
- **Top level.**
  - **Instances:** the 64-byte records (object-to-world 3x4, 24-bit custom data and 8-bit
    mask, flags, the bottom level's reference).
  - **Instance tree:** a BVH over the instances' world boxes, built on the GPU (R8's GPU top
    level from 32768 instances on, the LBVH) or refit when they only moved.
- **References.** Where the proposal uses a device address for a bottom level, the polyfill
  uses an index into a table of bottom levels: one big node buffer and one big triangle
  buffer per device, sub-allocated, so that a top level reaches all its bottom levels
  through a fixed set of bindings.

### 6.4 Bindings

One acceleration structure binding stands for four storage buffers: top-level nodes and
instances, bottom-level nodes, bottom-level primitives, and the table of bottom levels.

- **Where they go.** The polyfill rewrites the bind group layout and the bind group so the
  four take the binding's place plus three free binding numbers in the same group. The
  numbers are chosen above the highest the shader uses and passed to the WGSL rewriter.
- **The limit.** This counts against `maxStorageBuffersPerShaderStage`. The default is 8, so
  the library asks the adapter for its real limit (10 or more on desktop GPUs) and reports
  clearly when a shader would go over it.
- **A merged variant.** All four packed into one buffer with offsets in a header. Fewer
  bindings, slightly more address arithmetic. To be measured.

### 6.5 Rewriting the WGSL

The shader keeps the proposal's syntax; the library rewrites it to plain WGSL before
`createShaderModule`. This is done with a WGSL tokenizer, not regular expressions, so
comments and strings stay safe.

| in the shader | becomes |
|---|---|
| `enable ...ray_query;` | removed |
| `var acc: acceleration_structure;` (a binding) | the four buffer bindings (section 6.4), in a struct the library's functions take |
| `var rq: ray_query;` | `var rq: RQState;` (the ray, the committed hit, the candidate, the walk's stack and position) |
| `RayDesc(flags, cull_mask, t_min, t_max, origin, dir)` | the same struct, defined by the library |
| `rayQueryInitialize(&rq, acc, desc)` | `rq_init(&rq, desc)` (the root pushed) |
| `rayQueryProceed(&rq)` | `rq_proceed(&rq)`: walks until the next candidate the shader must decide (a non-opaque triangle, a box) and returns true; or until the walk ends and returns false |
| `rayQueryGenerateIntersection(&rq, t)`, `rayQueryConfirmIntersection(&rq)`, `rayQueryTerminate(&rq)` | commit a candidate's hit (a box's `t`, or a non-opaque triangle); stop the walk |
| `rayQueryGetCandidateIntersection(&rq)`, `rayQueryGetCommittedIntersection(&rq)` | the library's `RayIntersection`: kind, `t`, instance custom data, instance index, geometry index, primitive index, barycentrics, front face, object-to-world and world-to-object |
| `RAY_FLAG_*`, `RAY_QUERY_INTERSECTION_*` constants | the same constants |

Semantics, as in Vulkan and DXR:

- **Opaque triangles.** The walk keeps the closest hit itself and never stops for them.
  So a scene of opaque triangles needs only one `rq_proceed`: it walks to the end and
  returns false, which is the fast case, today's `traverse()`.
- **Flags.**
  - `RAY_FLAG_TERMINATE_ON_FIRST_HIT` ends the walk at the first committed hit (shadow
    rays).
  - `CULL_BACK_FACING` / `CULL_FRONT_FACING`, `CULL_OPAQUE` / `CULL_NO_OPAQUE`, `FORCE_OPAQUE`
    / `FORCE_NO_OPAQUE` are honoured.
  - The cull mask is tested against each instance's mask.
  - Instance flags (no face culling, opaque, not opaque) are read from the record.
- **Resumable walk.** For candidates, the walk's state (stack, node, instance, the ray in
  object space) lives in the `RQState`, so `rq_proceed` can return in the middle and resume.
  The stack is a fixed-size private array (as today's walk); its depth is a library
  constant, with an overflow fallback (a restart from the root with the current `t_max`).
- **Error messages.** Errors the browser reports in the rewritten code are mapped back to
  the shader's own lines (the rewriter keeps a line table).

### 6.6 Speed

The polyfill is as fast as the engine's software walk in the page (A1 today 3.74 ms per
pass, better after W4). The work to make it faster is the engine's own:

- the walk's code from W4;
- subgroups to keep a wave's rays on the same nodes, measured;
- compressed nodes (R3);
- the tree's quality (PLOC);
- fewer bindings (section 6.4).

### 6.7 Correctness

**Hit tests.**
- **Hardware comparison.** For random rays into each test scene (the boxes, the dragon, the
  sky demo's instances, boxes as procedural primitives), the polyfill's committed hits must
  match our Vulkan layer's hardware ray queries: same instance and primitive, `t` and
  barycentrics within a small tolerance, same misses.
- **Ties.** Ties at shared edges may differ: they are counted, not failed.
- **How it runs.** Natively, through wgpu with the polyfill's WGSL, against `bin/vk`.

**Images.** The engine's test images are rendered through the polyfill in Chrome (the page)
and compared with the native ones (`make web-check` style).

**API tests.** In the spirit of WebGPU's conformance tests:
- builds with zero geometries;
- the maximum instance counts;
- refit after moves;
- every flag;
- the binding limit errors.

### 6.8 Packaging and use by others

- **Files.** `rnd/rayquery/` on this branch: `rayquery.js` (the API and the device wrapper),
  `rq_build.wgsl` (the builders, from the engine's kernels), `rq_walk.wgsl` (the walk and
  the query functions), `rq_rewrite.js` (the WGSL tokenizer and rewriter), a demo page (a
  Cornell box and the dragon, ray queried, with a switch to the hardware path where a
  browser has it), and the tests.
- **Published as.** Plain files on GitHub (no npm needed to use it; an npm package is
  optional), with a README that says plainly: software, the proposal's current API, and
  which proposal text it follows (dated).
- **For the engine.** `webgpu_rt.h` implemented on it in the web build (section 6.2).

### 6.9 Steps

| step | what | done when |
|---|---|---|
| RD1.0 | the proposals' current text (PRs #11054, #11056) read; the name table written; the differences from our `webgpu_rt.h` listed | a table here |
| RD1.1 | bottom level of opaque triangles: GPU build, `rayQueryProceed` to the closest hit | the hit tests pass on the boxes and the dragon |
| RD1.2 | top level: instances, transforms, masks, GPU build and refit | the sky demo's 48k instances, tested |
| RD1.3 | flags (first hit, culling, opaque forcing), shadow rays | every flag tested |
| RD1.4 | candidates: non-opaque triangles and boxes, the resumable walk | the engine's spheres and quads as boxes, tested |
| RD1.5 | the WGSL rewriter made robust (tokenizer, line table, errors) | the engine's kernels and the demo rewritten and compiled in Chrome |
| RD1.6 | `webgpu_rt.h` on the polyfill in the web build: the path tracer's hardware-twin path in the page | the page's images equal to native |
| RD1.7 | speed: W4's walk, subgroups, node layout, bindings | numbers against the engine's own walk |
| RD1.8 | the demo page, the README, publishing | a link |
| RD1.9 | feedback (RD3) with what RD1 taught | posted |

### 6.10 Risks

- **The proposal changes** (names, binding model, what an instance record is). The name
  table and a thin adapter absorb it; the README dates what is followed.
- **WGSL rewriting is fragile** if done with text tricks: a real tokenizer, and tests over
  real shaders.
- **Binding limits** on small devices (8 storage buffers per stage): the merged variant.
- **Speed**: software stays software; the polyfill's value is the API and the fallback, not
  the cores.

## 7. Experiment RD2: ray queries in Dawn on Direct3D 12 (a patched Chromium)

The idea. Chrome on Windows runs WebGPU on Direct3D 12, and Direct3D 12 has inline ray
tracing (DXR 1.1: `RayQuery` in HLSL, shader model 6.5). The willwaltz patches add the
proposal's API to Dawn, Tint and Blink on Vulkan only. The Direct3D 12 side is missing:

- **Dawn's Direct3D 12 backend:** the acceleration structure objects and their builds (the
  prebuild sizes, scratch buffers, the instance descriptors);
- **Tint's HLSL writer:** `ray_query` to HLSL `RayQuery<...>`;
- **the validation that keeps it safe:** instance records checked, scratch managed by
  Dawn.

What it would give: the cores in Chrome on Windows, for whoever runs the build, and
implementation experience for the proposal's Direct3D 12 part.

The cost:
- a Chromium (or Brave) checkout of tens of GB and builds of hours;
- learning Dawn's and Tint's code;
- patches that age with every release.

Downloads and the toolchain need the owner's OK first.

State: an idea; after RD1, if wanted.

## 8. Experiment RD3: feedback to the WebGPU group

What we can bring to PRs #11054 and #11056 and issue #535, from experience rather than
opinion:

- **The API, implemented.** Our layer implements an acceleration structure and ray query
  extension of `webgpu.h` on Vulkan, used by a real path tracer:
  - bottom levels of triangles and boxes;
  - top levels of 64-byte records written by the CPU or by kernels;
  - in-place updates (refit);
  - chained bind group entries.
- **Numbers.** Ray queries against a tuned compute BVH walk on the same kernels: 1.4-6.4x
  (STATUS.md section 2).
- **Large top levels.** The reviewer's report of CPU overhead for large top levels (wgpu
  managing instance descriptors). Our answer: instance records written by a kernel, the top
  level built on the GPU from 32768 instances, and refit while at most a quarter moved
  (R8). With measurements.
- **The polyfill (RD1).** That the proposal's API can be emulated on WebGPU 1.0, and what it
  costs: which proposal choices make a polyfill easy or hard (device addresses for bottom
  levels, binding counts).

How: comments and a short write-up linked from the PRs, posted by the owner from his
GitHub account. Claude drafts, the owner posts.

State: to do after RD1.1-RD1.2 (so it carries numbers).

## 9. Decisions

| date | decision |
|---|---|
| 2026-11-18 | The web build uses the browser's own WebGPU only: no servers and no native helper process (streaming the native engine's frames to a page, and replaying a page's WebGPU calls on our Vulkan layer, were both proposed and rejected) |
| 2026-11-18 | W4 and W6 (STATUS.md 1.4) run on branch `webgpu-perf`, in five measured steps (section 4) |
| 2026-11-18 | This R&D branch opened; hardware ray tracing on the web: the polyfill (RD1) and the feedback (RD3) first, the patched browser (RD2) maybe later |

## 10. Sources

- Chrome, "New in WebGPU 134" (subgroups shipped): https://developer.chrome.com/blog/new-in-webgpu-134
- GPU for the Web Community Group, activity in 2026 (acceleration structures, ray queries,
  bindless, subgroup matrices): https://www.w3.org/groups/cg/gpu/activity/2026/
- gpuweb PR #11054, acceleration structures: https://github.com/gpuweb/gpuweb/pull/11054
- gpuweb issue #535, Ray Tracing extension: https://github.com/gpuweb/gpuweb/issues/535
- Intent to Ship: WebGPU Immediates: https://groups.google.com/a/chromium.org/g/blink-dev/c/8LdEcW1CkNo/m/rrV3CUyGAAAJ
- willwaltz/webgpu-ray-tracing: https://github.com/willwaltz/webgpu-ray-tracing
- maierfelix/dawn-ray-tracing: https://github.com/maierfelix/dawn-ray-tracing ;
  chromium-ray-tracing: https://github.com/maierfelix/chromium-ray-tracing
- codedhead/WebRTX: https://github.com/codedhead/webrtx
- Can I use, WebAssembly Memory64: https://caniuse.com/wf-wasm-memory64 ; SIMD:
  https://caniuse.com/wasm-simd
- This repository: VULKAN.md ("Hardware ray tracing"), `webgpu/vk/webgpu_rt.h`, STATUS.md
  (sections 1.4, 2, 4), WEB.md
