# Our own Vulkan backend: the plan (after PBR)

The plan agreed on 2026-11-14 for the engine's GPU layer natively, step 6 of
[ROADMAP.md](ROADMAP.md). Work happens on the `vulkan` branch (merged only when agreed).
Decided on 2026-11-15: `webgpu.h` on Vulkan first, natively only (the browser build is not
kept up meanwhile); hardware ray tracing is the aim; the WebGPU side for the web comes after.

## Status

| part | state |
|---|---|
| `engine/gpu/vk/vk.h`: our declarations of the Vulkan API (no SDK) | `tools/vk_check.py` checks every size, offset and constant against the official headers (838 checks, 0 differ) |
| loader, instance, adapter, device, queue (timeline semaphore), submissions, polling, callbacks | done |
| buffers (dedicated memory each, for now), uploads (`vkCmdUpdateBuffer` up to 64 KB, staging memory above), copies, clears, readback by mapping | done: `bin/vk/test_gpu_basic` passes |
| bind group layouts, bind groups (descriptor pools made as needed, dynamic offsets), pipeline layouts, compute pipelines, compute passes, timestamps | done |
| WGSL → SPIR-V (`engine/gpu/wgsl`): every kernel the engine has compiles, `spirv-val` finds nothing | done |
| textures (all in the GENERAL layout), texture writes, copies to and from buffers | done |
| render pipelines and passes (dynamic rendering; WebGPU's y up and front faces mirrored) | done |
| the window's surface (Win32 swapchain) | to do (`vk_todo.c` says so at run time; `view` draws offscreen meanwhile) |
| acceleration structures, ray queries (R6) | after |

Checked so far, with the validation layer on and nothing reported: `make VK=1 test` passes;
`render` (rasterizer: every shading mode, the helmet's textures, the sky; path tracer: the
box, the sky, the helmet) writes the same pixels as on wgpu, bit for bit; `sim` (drop,
blast: GPU bodies, air and cloud) differs by at most 2 levels of 255 after 12 frames.

Building: `make VK=1` (binaries in `bin/vk`; the webgpu.h header still comes from `make
deps`). `LR_VK_VALIDATE=1` turns on the Khronos validation layer if it is installed
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
  natively it is a C library implementing the same `webgpu.h` (today wgpu-native, Rust).
  WebGPU itself runs on Vulkan, Direct3D 12 or Metal underneath.
- **Vulkan** is the low-level native GPU interface the drivers implement (Windows, Linux,
  Android; not Apple). Shaders in SPIR-V. Not reachable from a web page.

```
browser:  engine (C → WebAssembly) → webgpu.h → browser's WebGPU → (Vulkan/D3D12/Metal) → GPU
today:    engine (C) → webgpu.h → wgpu-native → Vulkan or D3D12 → GPU
planned:  engine (C) → webgpu.h → our implementation on Vulkan → GPU
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

1. **A WGSL → SPIR-V compiler** (C, `gpu/wgsl/`): Vulkan reads SPIR-V only, and WGSL stays
   our one shader source because it is the only one browsers accept. Only the WGSL we
   write: structs, arrays, loops, atomics, workgroup memory, pointers to function
   variables, the built-ins; then textures and samplers (PBR5) and ray queries. Parser,
   type checker, SPIR-V emitter (structured control flow, the memory model). About 6-10k
   lines. Checked kernel by kernel: the same results as wgpu's compiler on every test.
2. **`webgpu.h` on Vulkan** (`gpu/vk/`): the about 60 functions the engine calls. Instance,
   adapter and device (the Vulkan loader opened at run time: no library linked, our own
   declarations of the part of the API we use), buffers (an allocator of our own, staging
   uploads, mapped readbacks), bind group layouts and groups (descriptor sets), compute and
   render pipelines, command encoders and passes, barriers (from the order of the dispatches:
   our code needs a memory barrier between dependent dispatches, which is what WebGPU's
   tracking ends up doing), the swapchain (Win32, X11), timestamp queries. No validation in
   release builds; the Khronos validation layers while developing.
3. **Native extensions**, asked for like wgpu-native's (`WGPUNativeFeature_*`), each with
   the plain WGSL path kept for the browser:
   - more than 8 storage buffers per stage (the BSDF tables would leave the blue-noise
     buffer), push constants;
   - 64-bit atomics (the caustics' sums, today two 32-bit halves);
   - subgroup operations (prefix sums, reductions, the contact lists, PLOC);
   - bindless textures (PBR5's atlases become a choice, not a need);
   - a second queue: the physics while the image renders;
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
   pass on ours, then goes.
3. The native extensions (P10 becomes cheap; then R6, ray tracing).
4. Remote rendering; the browser's new features as they ship.

## Open questions (to answer when it starts)

- Rendering on the visitor's GPU, frames streamed from a machine of the user's, or both,
  for the projects the engine runs behind? What the projects are (interactive pages,
  simulations, visualization) decides whether latency, image quality or the number of
  users matters most.
- Apple: MoltenVK, a Metal backend of our own, or the browser only.
- Emscripten as the browser build's tool, or plain clang and our own bridge.
