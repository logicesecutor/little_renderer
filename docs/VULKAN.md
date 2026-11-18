# Our own Vulkan backend (`webgpu/vk`, `webgpu/wgsl`)

The native build's GPU: the `webgpu.h` API implemented on Vulkan 1.3, with its own WGSL →
SPIR-V compiler. What follows began as the plan agreed on 2026-11-14 for the engine's GPU layer natively, step 6 of
[STATUS.md](STATUS.md). Work happens on the `vulkan` branch (merged only when agreed).
Decided on 2026-11-15: `webgpu.h` on Vulkan first, natively only (the browser build is not
kept up meanwhile); hardware ray tracing is the aim; the WebGPU side for the web comes after.

## Status

**Since 2026-11-18 this layer is the only native GPU**: `make` builds it, wgpu-native is gone
(no `make deps`, no `VK=1`, no `bin/vk/`). The standard `webgpu.h` is our copy in
`webgpu/include/webgpu/` (the version wgpu-native v29.0.1.1 shipped); what we add to it has
names of our own: `webgpu/vk/webgpu_lr.h` (`lrQueueSubmitForIndex`, `lrDevicePoll`,
`LRFeatureName_RayQuery`, `lrDeviceGetComputeQueue`; until then wgpu.h's `wgpuDevicePoll`,
`wgpuQueueSubmitForIndex`, `WGPUNativeFeature_RayQuery`) and `webgpu/vk/webgpu_rt.h`. Why:
on Windows and Linux the layer reaches the same hardware as wgpu-native did (which ran on
Vulkan here; its Direct3D 12 backend never finished compiling the path tracer, its Metal one
serves Macs, not a target), faster, with ray tracing; the WGSL is checked by the browser
instead (the page's images against native: `bench -shots` / `-imgdiff`). What it costs:
MSAA waits for multisampled targets here (R7); machines without Vulkan 1.3 run on the CPU
natively (their browser still has WebGPU).

| part | state |
|---|---|
| `webgpu/vk/vk.h`: our declarations of the Vulkan API (no SDK) | `tools/vk_check.py` checks every size, offset and constant against the official headers (899 checks, 0 differ) |
| loader, instance, adapter, device, queue (timeline semaphore), submissions, polling, callbacks | done |
| buffers (dedicated memory each, for now), uploads (`vkCmdUpdateBuffer` up to 64 KB, staging memory above), copies, clears, readback by mapping | done: `bin/test_gpu_basic` passes |
| bind group layouts, bind groups (descriptor pools made as needed, dynamic offsets), pipeline layouts, compute pipelines, compute passes, timestamps | done |
| WGSL → SPIR-V (`webgpu/wgsl`): every kernel the engine has compiles, `spirv-val` finds nothing | done |
| textures (all in the GENERAL layout), texture writes, copies to and from buffers | done |
| render pipelines and passes (dynamic rendering; WebGPU's y up and front faces mirrored) | done |
| the window's surface (Win32 swapchain; X11 since 2026-11-18, `VK_KHR_xlib_surface`): an image waited for on the CPU (a fence), presented after a small submission moves it to PRESENT_SRC; the display's refresh rate from the monitor's mode (Win32; X11 by XRandR, loaded when there) | done (X11: built, not yet run on a Linux GPU) |
| acceleration structures and ray queries (R6): our extension of webgpu.h (`webgpu/vk/webgpu_rt.h`), the compiler's `ray_query`, `bin/test_gpu_rt`; the path tracer's `traverse()` on them | done (below) |

Checked so far, with the validation layer on and nothing reported: `make test` passes;
`render` (rasterizer: every shading mode, the helmet's textures, the sky; path tracer: the
box, the sky, the helmet) writes the same pixels as on wgpu, bit for bit; `sim` (drop,
blast: GPU bodies, air and cloud) differs by at most 2 levels of 255 after 12 frames;
`view` shows its window at 60 frames/s with the physics (rasterizer and hybrid), its
screenshot of the rasterizer the same as on wgpu. Measured in `view -sim on -autoquit 6`:
the physics' CPU time per frame 0.39 ms on ours, 0.80 ms on wgpu (GPU bodies 0.1 against
0.3 ms, the air 0.3 against 0.5 ms); the path tracer's pass in `view` 2.1-2.3 ms against
4.2 ms.

### Hardware ray tracing (R6)

webgpu.h has no acceleration structures (nor had wgpu-native's C API), so ours is an extension of webgpu.h of
our own (`webgpu/vk/webgpu_rt.h`), shaped after the WebGPU proposal: bottom levels of
triangles or boxes (`lrDeviceCreateBlas`, `lrCommandEncoderBuildBlas`), a top level of
64-byte instance records (`lrDeviceCreateTlas`, `lrCommandEncoderBuildTlas`), bound through
chained bind group entries. The WGSL is wgpu's: `acceleration_structure`,
`ray_query`, `rayQueryInitialize` / `Proceed` / `GenerateIntersection` /
`GetCommittedIntersection`, `RayDesc`, `RayIntersection`, the `RAY_*` constants. (The WebGPU
proposals of October 2026, gpuweb #11054 and #11056, differ from it in places; what a page
needs to use the same hardware path once a browser ships them: WEB.md 2.10.)

The device asks for ray queries when the GPU has them (`LR_GPU_HWRT=0`: not). The path
tracer then compiles a `traverse()` on ray queries in place of the BVH walk (the same
result, so every kernel uses it unchanged) and keeps, besides its buffers, bottom levels of
the world's triangles, its spheres and quads (boxes the kernel hits with the CPU's tests),
each mesh, and the unit box (balls and boxes); the top level is built every frame from
records written by the CPU with the instances. Kernel ms per pass, 640×640, RTX 3070 (the
BVH on our layer → ray queries): boxes 2.08 → 1.35, a sphere mesh 3.36 → 1.25, the helmet
4.82 → 1.80, the showcase 12.17 → 7.40 (STATUS.md).

The WGSL lives in files of its own (`engine/gpu/shaders/`, WEB.md 2.2). Next: the native
extensions (below) where measurements show a gain.

Barriers: a command waits only for the commands since the last barrier, at their stages
(compute, transfer, a build, a render pass) and for their writes, not for everything
(`ALL_COMMANDS`, as at first: a small physics step took 0.71 ms of GPU time where wgpu takes
0.26; now 0.32-0.36, and `mesh_rain`'s 1500 hulls 4.63 ms per step against wgpu's 4.66).
Skipping barriers between dispatches that touch different buffers would gain at most a
further 0.03 ms there (measured with no barriers at all), so it is not done.
Subgroups: the layer offers WebGPU's `subgroups` feature (with the sizes in the adapter's
info) and the compiler takes `subgroup_invocation_id`, `subgroup_size` and the subgroup
functions (`subgroupAdd`, the scans, `Min` / `Max`, `Broadcast`, `Shuffle` / `ShuffleXor` /
`Up` / `Down`, `Ballot`, `All` / `Any`, `Elect`). Tried on 2026-11-16 where the physics
combines its threads' values (the solver's 8 threads per body; the hull test's largest of 64:
2 barriers instead of 7), with exactly the same results: no gain, on our layer or on wgpu
(`mesh_rain` 1500 hulls: the solver 1.40 / 1.41 ms, the hull test 2.91 / 3.03 ms, without /
with), since those kernels wait on memory, not on their barriers. So the engine does not use
them; the support stays for kernels that will.
A second queue (`LR_VK_QUEUES=2`): `lrDeviceGetComputeQueue` (webgpu_rt.h), of a family with
compute only if the GPU has one (its asynchronous compute engine; buffers are then shared by
both families, `VK_SHARING_MODE_CONCURRENT`), else a second queue of the main family
(`LR_VK_QUEUES=same`). Each queue has its own timeline, uploads and command buffers; an
encoder is made for it with an `LRCommandEncoderQueue` chained to its descriptor, and a
submission index carries its queue in the top bits. The engine runs the rigid bodies there
(`gpu_encoder`, `gpu_submit_to`, `g->compute`); the air stays on the main queue, as the cloud
is drawn from its buffers. Measured on 2026-11-16 (RTX 3070, `mesh_rain` 1500 hulls, about
4.7 ms of physics and 18 ms of drawing per frame): one queue 44-45 frames/s, the compute
family 34-35, a second queue of the main family 42 (a smaller window and 3 steps in flight
instead of 2: the same). The two workloads fill the GPU each and slow each other down
rather than filling each other's gaps, so it is off by default; the main queue is then
`lrDeviceGetComputeQueue`'s answer and nothing changes.

A lost device: when a submission fails, or Vulkan answers `VK_ERROR_DEVICE_LOST` to a wait,
an acquire or a present, the layer calls the device-lost callback (once), as wgpu does, and
from then on fails what waits for the GPU instead of waiting for good: work-done callbacks
get an error, mappings are aborted, the surface reports itself lost. The engine's
`gpu_state` goes to `GPU_FAILED` and the programs carry on with the CPU (the viewer says
"GPU lost"). `LR_VK_LOSE_AFTER=n` makes the n-th submission fail that way, to test it.
The engine's waits keep their flags in the `Gpu`, not on the stack, so a callback that comes
after a wait gave up writes nowhere it should not. A surface that answers Outdated or Lost is
configured again by the engine, the frame skipped. A submission of more command buffers than
the layer holds at once is a validation error, not a silent drop.

`LR_VK_VALIDATE=sync` adds the validation layer's synchronization checks (hazards between
commands): the tests, `render`, `sim` and `view` run clean.

Building: `make` (the layer is the native build's GPU; `NOGPU=1` leaves it out). `LR_VK_VALIDATE=1` turns on the Khronos validation layer if it is installed
(MSYS2: `mingw-w64-ucrt-x86_64-vulkan-validation-layers`; then `VK_ADD_LAYER_PATH` to its
`bin`). Development tools only, nothing linked: SPIRV-Tools (`spirv-val`, `spirv-dis`) and
the validation layers, through MSYS2 (updated on 2026-11-15 for them: gcc 12.1 → 16.2, the
builds and `make test` unchanged).

## The goal

One engine that runs both natively and in web browsers, behind the user's own projects, as
fast as the hardware allows, with as little foreign code as possible (what is missing, we
write).

## How the pieces relate

- **WebAssembly** is how the engine's C code runs inside a web page (on the CPU). It has no
  GPU of its own: the page reaches the GPU only through the browser's WebGPU.
- **WebGPU** is a portable, safe GPU interface (shaders in WGSL). In a browser it is a
  JavaScript API, which our WebAssembly calls through Emscripten's bindings for `webgpu.h`;
  natively it is a C library implementing the same `webgpu.h` (ours, on Vulkan; until
  2026-11-18 wgpu-native, Rust).
  WebGPU itself runs on Vulkan, Direct3D 12 or Metal underneath.
- **Vulkan** is the low-level native GPU interface the drivers implement (Windows, Linux,
  Android; not Apple). Shaders in SPIR-V. Not reachable from a web page.

```
browser:  engine (C → WebAssembly) → webgpu.h → browser's WebGPU → (Vulkan/D3D12/Metal) → GPU
natively: engine (C) → webgpu.h → our implementation on Vulkan → GPU
(before 2026-11-18: engine (C) → webgpu.h → wgpu-native → Vulkan or D3D12 → GPU)
```

So Vulkan can make the native engine faster, never the page; the page gets what its
browser's WebGPU offers (and, where that is missing, frames rendered natively: below).

## The design

**The engine keeps speaking WebGPU** (`webgpu.h` and WGSL), unchanged: in a browser it
reaches the browser's WebGPU as today; natively it reaches **our own implementation of the
part of `webgpu.h` we use, on Vulkan**, which replaces wgpu-native. Not a rewrite of the
engine's GPU code against Vulkan: that would need a second WebGPU version of every GPU file
for the browser, kept in step forever.

What has to be written:

1. **A WGSL → SPIR-V compiler** (C, `webgpu/wgsl/`): Vulkan reads SPIR-V only, and WGSL stays
   our one shader source because it is the only one browsers accept. Only the WGSL we
   write: structs, arrays, loops, atomics, workgroup memory, pointers to function
   variables, the built-ins; then textures and samplers (PBR5) and ray queries. Parser,
   type checker, SPIR-V emitter (structured control flow, the memory model). About 6-10k
   lines. Checked kernel by kernel: the same results as wgpu's compiler on every test.
2. **`webgpu.h` on Vulkan** (`webgpu/vk/`): the about 60 functions the engine calls. Instance,
   adapter and device (the Vulkan loader opened at run time: no library linked, our own
   declarations of the part of the API we use), buffers (an allocator of our own, staging
   uploads, mapped readbacks), bind group layouts and groups (descriptor sets), compute and
   render pipelines, command encoders and passes, barriers (from the order of the dispatches:
   our code needs a memory barrier between dependent dispatches, which is what WebGPU's
   tracking ends up doing), the swapchain (Win32, X11), timestamp queries. No validation in
   release builds; the Khronos validation layers while developing.
3. **Native extensions**, asked for as features of our own (`LRFeatureName_*`, webgpu_lr.h), each with
   the plain WGSL path kept for the browser:
   - more than 8 storage buffers per stage (the BSDF tables would leave the blue-noise
     buffer), push constants;
   - 64-bit atomics (the caustics' sums, today two 32-bit halves);
   - subgroup operations (prefix sums, reductions, the contact lists, PLOC);
   - bindless textures (PBR5's atlases become a choice, not a need);
   - a second queue: the physics while the image renders (done, off by default: slower
     on `mesh_rain`, see above);
   - **hardware ray tracing (R6)**: acceleration structures and ray queries in the existing
     kernels (the shape WebGPU's own proposal of October 2026 takes): each mesh a bottom
     level, our instances the top level's (transform, a 24-bit index to the material and
     record), spheres, quads and the unit shapes as procedural boxes hit by our own tests.
     The compute BVH (PLOC, LBVH, the CPU's SAH tree) stays: the browser, GPUs without ray
     tracing, the reference.
4. **Remote rendering** (where the browser lacks a technology, we provide it): the native
   engine renders (Vulkan, ray tracing, every core) and streams its frames to a page, which
   decodes them with the browser's own video decoder (WebCodecs) and sends input back.
   Frames encoded by the GPU's video encoder through Vulkan Video (only the driver), or a
   simple codec of our own; a WebSocket server of our own (handshake, framing, SHA-1,
   base64). WebRTC is too large to write; `wss://` needs TLS of our own or a proxy in front.
5. **The browser side** keeps improving with what browsers ship: subgroups, f16, immediates
   (push constants), ray queries when they come; WebAssembly SIMD and threads (the page
   served cross-origin isolated). Later, optionally: the browser build by plain clang
   (`--target=wasm32`) with a small libc subset and our own JavaScript bridge to WebGPU,
   instead of Emscripten.

## What it gains, and what not

- Faster natively: no per-call validation and conservative barrier tracking (wgpu: about
  1.6 us more per indirect dispatch, 3-7 us per dispatch; the physics' 92 dispatches per
  step, P10, still cost 0.8 ms when the bodies sleep), the hardware's features above, ray
  tracing (an estimated 2-5x on scenes whose time goes to finding hits).
- Not faster by itself: a kernel's arithmetic compiles to about the same machine code as
  through wgpu.
- One engine, one shader source, the web unchanged; less foreign code than today (the
  driver's Vulkan loader instead of a 16 MB Rust library).
- Costs: no Vulkan on Apple (MoltenVK, or a Metal implementation of our own, later; Safari's
  WebGPU covers Macs in the browser); our layer does not catch our mistakes as wgpu does
  (validation layers while developing).

## Order

1. The compiler, checked against wgpu's on every kernel and test.
2. `webgpu.h` on Vulkan; wgpu-native stays until `make test` and the image comparisons
   pass on ours, then goes (gone on 2026-11-18: Status above).
3. The native extensions (P10 becomes cheap; then R6, ray tracing).
4. Remote rendering; the browser's new features as they ship.

## Open questions (to answer when it starts)

- Rendering on the visitor's GPU, frames streamed from a machine of the user's, or both,
  for the projects the engine runs behind? What the projects are (interactive pages,
  simulations, visualization) decides whether latency, image quality or the number of
  users matters most.
- Apple: MoltenVK, a Metal backend of our own, or the browser only.
- Emscripten as the browser build's tool, or plain clang and our own bridge.
