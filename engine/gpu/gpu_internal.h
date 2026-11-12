// gpu_internal.h - what the files of gpu/ share: the WebGPU objects and small helpers.
// Not part of the engine's public API.
#ifndef LR_GPU_INTERNAL_H
#define LR_GPU_INTERNAL_H

#ifdef LR_GPU
#include <stddef.h>
#include <stdint.h>
#include <webgpu/webgpu.h>
#include "core/scene.h"
#include "gpu/gpu.h"
#include "gpu/view_gpu.h"
#include "raster/render.h"

struct Gpu {
    GpuState state;
    char message[256];
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUSurface surface;
    WGPUTextureFormat surface_format;
    int width, height;
    uint64_t max_storage;   // largest storage buffer a shader can bind, in bytes
    int offscreen;       // no canvas: draw into textures only
    void *native_display;   // natively: the Display * (X11) or HINSTANCE (Win32) of the window
    uint64_t native_window; // the X Window or HWND, 0 = none
    int software;        // the adapter is a CPU emulating a GPU
    int timestamps;      // passes can be timed (LR_GPU_TIMING=1 natively, and the adapter can)
    char canvas[64];
};

// WGSL source kept in C: WGSL( ...code... ) turns the code into a string. The preprocessor
// removes the comments and joins the lines, which WGSL does not mind.
#define WGSL(...) #__VA_ARGS__

static inline WGPUStringView gpu_str(const char *s){ WGPUStringView v = {s, WGPU_STRLEN}; return v; }

WGPUShaderModule gpu_shader(Gpu *g, const char *label, const char *wgsl);
WGPUBuffer gpu_buffer(Gpu *g, const char *label, size_t size, WGPUBufferUsage usage);
// (re)creates *buf if it is missing or smaller than size; returns 1 if it was recreated
int gpu_buffer_reserve(Gpu *g, WGPUBuffer *buf, size_t *cap, size_t size, WGPUBufferUsage usage, const char *label);

// compute kernels: a layout of n buffers (binding i of types[i], 'U' uniform, 'R' read-only
// storage, 'W' read-write storage, '-' no binding i), a pipeline on it, bind groups, and 1-D
// dispatches
WGPUBindGroupLayout gpu_layout(Gpu *g, WGPUShaderStage stage, const char *types);
WGPUComputePipeline gpu_compute(Gpu *g, const char *label, const char *wgsl, const char *entry, WGPUBindGroupLayout bgl);
WGPUBindGroup gpu_bind(Gpu *g, WGPUBindGroupLayout bgl, const WGPUBuffer *bufs, int n);
// `threads` invocations of a kernel with @workgroup_size(64): the workgroups are spread over x
// and y (at most 65535 per axis); the kernel's index is gid.x + gid.y * num_workgroups.x * 64
void gpu_dispatch(WGPUComputePassEncoder pass, size_t threads);

// ---- GPU time per stage (timestamp queries; natively with LR_GPU_TIMING=1) ----
// Each timed compute pass gets the timestamps gpu_timer_next gives for its descriptor
// (NULL: not timed, run it as usual), counted under `stage`; gpu_timer_finish resolves those
// used since the last call and reads them back (asynchronously; a round is skipped while the
// last one is still on its way); gpu_timer_print shows the mean GPU time per stage and round.
typedef struct GpuTimer GpuTimer;
GpuTimer *gpu_timer_create(Gpu *g, int passes, int stages);   // NULL when timing is off
const WGPUPassTimestampWrites *gpu_timer_next(GpuTimer *t, int stage);
void gpu_timer_finish(GpuTimer *t, WGPUCommandEncoder enc);
void gpu_timer_submitted(GpuTimer *t);   // after submitting the encoder gpu_timer_finish wrote into
// the work the pass just given timestamps did (e.g. full-image samples), and the mean GPU
// time per unit of work of a stage (only the timed passes count; 0 if none was)
void gpu_timer_units(GpuTimer *t, double units);
double gpu_timer_ms_per_unit(GpuTimer *t, int stage);
void gpu_timer_print(GpuTimer *t, const char *title, const char *const *names);
void gpu_timer_destroy(GpuTimer *t);

// The GPU rasterizer (gpu/raster_gpu.c): the scene's surfaces into its own color (sRGB
// values in rgba8unorm) and depth (0 near .. 1 far, as the CPU's z-buffer) textures.
typedef struct GpuRaster GpuRaster;
GpuRaster *gpu_raster_create(Gpu *g, int width, int height);
// encodes the pass; 0 if the scene cannot be drawn on this GPU
int gpu_raster_draw(GpuRaster *r, WGPUCommandEncoder enc, const Scene *s, const RenderParams *p);
WGPUTextureView gpu_raster_color(GpuRaster *r);
WGPUTextureView gpu_raster_depth(GpuRaster *r);
void gpu_raster_destroy(GpuRaster *r);

// The GPU path tracer (gpu/trace_gpu.c): samples added to an accumulation buffer, w * h
// vec4 (rgb = sum of the samples, a = how many). It shares the view's media: the WGSL of the
// frame's types and of the media's densities, and the buffers passed to gpu_trace_encode.
typedef struct GpuTrace GpuTrace;
extern const char *const GPU_WGSL_TYPES, *const GPU_WGSL_MEDIA;
GpuTrace *gpu_trace_create(Gpu *g, int width, int height, const char *types_wgsl, const char *media_wgsl);
// the scene's geometry and BVH onto the GPU if they changed; 0 = this GPU cannot hold them
int gpu_trace_sync(GpuTrace *t, const Scene *s);
void gpu_trace_clear(GpuTrace *t, WGPUCommandEncoder enc);
// `repeats` more samples for every pixel of rows [y0, y0 + rows), samples number index,
// index + 1, ... of the sequence (core/sampler.h)
void gpu_trace_encode(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                      int y0, int rows, int repeats, int nvolumes, uint32_t index,
                      WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat);
// Before the first band of an image: each pixel's guide for the denoiser (with no samples),
// and with history > 0 the samples of the previous image reprojected (at most history each).
void gpu_trace_begin_image(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                           int history, int nvolumes,
                           WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat);
WGPUBuffer gpu_trace_accum(GpuTrace *t);   // the image's samples (4 planes when guided)
// Light-traced caustics (tp->photons > 0 and the scene has a mirror or glass): `passes` more
// passes of light paths added into the caustic sums; clear them for a new image; resolve: the
// image `base` (vec4 sum + count, or rgb + 1) plus the caustics, as rgb + 1 (0 where base has
// no sample).
int gpu_trace_caustics_on(const GpuTrace *t, const GpuTraceParams *tp);
// around the submit of the commands gpu_trace_encode / gpu_trace_light wrote (the GPU timer)
void gpu_trace_before_submit(GpuTrace *t, WGPUCommandEncoder enc);
void gpu_trace_after_submit(GpuTrace *t);
void gpu_trace_light(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                     int passes, int nvolumes, uint32_t index,
                     WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat);
void gpu_trace_caustics_clear(GpuTrace *t, WGPUCommandEncoder enc);
int gpu_trace_caustic_passes(const GpuTrace *t);
// cache = 1: blend in the photon cache (the viewer) while the light tracer has few passes
WGPUBuffer gpu_trace_resolve(GpuTrace *t, WGPUCommandEncoder enc, WGPUBuffer base, int cache);
// The photon cache: fade it (fade < 1: times fade), then add one pass of light paths
void gpu_trace_photons(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp, int nvolumes,
                       uint32_t index, double fade, WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks,
                       WGPUBuffer vstat);
void gpu_trace_photons_clear(GpuTrace *t, WGPUCommandEncoder enc);
double gpu_trace_photon_paths(const GpuTrace *t);
int gpu_trace_guided(const GpuTrace *t);    // 1 = the denoiser's planes follow the samples
// 1 = trace with the wavefront kernels (WAVE in trace_gpu.c), 0 = the single kernel
void gpu_trace_set_wavefront(GpuTrace *t, int on);
void gpu_trace_destroy(GpuTrace *t);

// The denoiser (gpu/denoise_gpu.c): from the path tracer's 4 planes to a w * h image of
// vec4 (rgb, 1), or (0, 0, 0, 0) where there is no sample yet.
typedef struct GpuDenoise GpuDenoise;
GpuDenoise *gpu_denoise_create(Gpu *g, int width, int height);
void gpu_denoise_encode(GpuDenoise *d, WGPUCommandEncoder enc, WGPUBuffer pixels);
WGPUBuffer gpu_denoise_output(GpuDenoise *d);
void gpu_denoise_destroy(GpuDenoise *d);
double gpu_now_ms(void);

// GPU views take the densities of some media straight from a GPU buffer (gpu/fluid_gpu.c)
#define GPU_MAX_SOURCES 4
struct GpuView;
struct Volume;
void gpu_view_volume_on_gpu(struct GpuView *v, const struct Volume *vol, WGPUBuffer density);
#endif

#endif
