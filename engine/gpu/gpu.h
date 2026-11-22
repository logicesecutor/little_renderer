// gpu.h - the GPU backend of the engine, on WebGPU (webgpu.h): the same C code drives the
// GPU in a web browser (Emscripten + Dawn's "emdawnwebgpu" bindings over the browser's
// WebGPU) and, natively, through our own WebGPU on Vulkan (webgpu/vk, docs/VULKAN.md).
// Shaders are WGSL, kept as strings inside the engine.
//
// Builds define LR_GPU unless made with NOGPU=1 (`make web` always does). Without it
// these functions still exist: gpu_create says the GPU is not available, and everything
// stays on the CPU backends, which are always there.
//
// Starting a GPU is asynchronous (the browser answers later), so: gpu_create, then call
// gpu_state every frame until it is GPU_READY (or GPU_FAILED: keep using the CPU).
#ifndef LR_GPU_H
#define LR_GPU_H

#include <stdint.h>

typedef struct Gpu Gpu;

typedef enum { GPU_PENDING, GPU_READY, GPU_FAILED } GpuState;

// 1 if this build has a GPU backend (the browser's WebGPU, or natively our Vulkan layer; not
// with NOGPU=1), 0 if everything runs on the CPU
int gpu_available(void);

// canvas: the CSS selector of the HTML canvas the GPU draws on (web), e.g. "#canvas". NULL
// draws offscreen (natively too): the frames are read back (gpu_view_capture) and shown like
// CPU frames, or saved (bin/render, bin/sim). Returns NULL only if out of memory.
Gpu *gpu_create(const char *canvas, int width, int height);
// Natively, drawing in a window: X11 (display = the Display *, window = the X Window) or
// Win32 (display = the HINSTANCE, window = the HWND); see window_native in platform/window.h.
Gpu *gpu_create_window(void *display, uintptr_t window, int width, int height);
// GPU_FAILED also comes later if the browser takes the GPU away (device lost): then stop
// using it and draw on the CPU again. Natively it also delivers the callbacks that are ready
// (gpu_poll), so a program that calls it every frame needs nothing else.
GpuState gpu_state(Gpu *g);
// Natively: runs the callbacks of the work the GPU has finished (readbacks, finished steps).
// In a browser they arrive by themselves, and this does nothing.
void gpu_poll(Gpu *g);
// Natively: blocks until the GPU has done everything submitted so far, then gpu_poll. For
// programs that wait for their images (bin/render, bin/sim). In a browser, where nothing may
// block, it only polls.
void gpu_wait(Gpu *g);
const char *gpu_message(const Gpu *g);   // what happened: the adapter's name, or why it failed
int gpu_offscreen(const Gpu *g);         // 1 if created with no canvas
// 1 if the "GPU" is software (SwiftShader, llvmpipe...): it runs simple kernels well (the
// air, the cloud's raymarch) but long branching ones (the path tracer) far slower than the
// engine's own CPU code
int gpu_software(const Gpu *g);
// 1 = a browser's adapter offers hardware ray queries ("acceleration-structure", the WebGPU
// proposal's name); not used yet: the page traces with its own tree walk (docs/WEB.md 2.10)
int gpu_rt_offered(const Gpu *g);
void gpu_destroy(Gpu *g);
// Benchmarks (bin/bench): the GPU's own timers (timestamp queries; natively only, when the
// adapter has them). Asked for before gpu_create (or LR_GPU_TIMING=1 in the environment);
// gpu_timing says whether this GPU times its passes. Each part then reads its own
// (gpu_view_gpu_ms, gpu_rigid_gpu_ms, gpu_fluid_gpu_ms).
void gpu_set_timing(int on);
int gpu_timing(const Gpu *g);
// A window's frames in step with the display (1, the default: no tearing), or shown as soon
// as they are ready (0: Mailbox where the window system has it, else Immediate, which may
// tear). Set before gpu_create_window. In a browser the page is always in step.
void gpu_set_vsync(int on);
// The window's (or canvas's) new size: its surface made again at it (after window_resize).
// What was made for the old size (a GpuView) is made again by its owner.
void gpu_resize(Gpu *g, int width, int height);

#endif
