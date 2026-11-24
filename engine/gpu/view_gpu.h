// view_gpu.h - showing frames with the GPU, and raymarching the clouds there.
//
// The CPU backends render into a Framebuffer (colors + depth). gpu_view_present uploads it,
// and one full-screen pass draws it onto the canvas; with volumes = 1 that same pass first
// raymarches every participating medium of the scene over it (up to GPU_MAX_VOLUMES), with
// the lighting model of raster/volume_march.c: the light grids are computed by a compute
// pass, one GPU thread per cell, and every pixel marches in parallel. The z-buffer stops the
// rays at the surfaces, as on the CPU.
#ifndef LR_VIEW_GPU_H
#define LR_VIEW_GPU_H

#include "scene/camera.h"
#include "scene/image.h"
#include "scene/scene.h"
#include "gpu/gpu.h"
#include "raster/render.h"
#include "raster/volume_march.h"

#define GPU_MAX_VOLUMES 2

typedef struct GpuView GpuView;

GpuView *gpu_view_create(Gpu *g, int width, int height);
// 1 = its pipelines are there (made at once, or asynchronously in a browser with
// gpu_set_async_views: 0 until they are), -1 = they could not be made
int gpu_view_ready(GpuView *v);   // NULL if the GPU is not ready
void gpu_view_present(GpuView *v, const Framebuffer *fb, const Scene *s, const RenderParams *p,
                      const MarchParams *mp, int volumes);
// The same without a CPU framebuffer: the scene's surfaces are drawn by the GPU rasterizer
// (gpu/raster_gpu.c: the triangles, the lamp and the shading of render_scene; uses p->view,
// p->proj, p->eye, p->mode, p->cull_backfaces), then the media over them. Returns 0 if this
// GPU cannot draw the scene (a mesh too big for one buffer): then draw it on the CPU and
// show it with gpu_view_present.
// A rectangle of text drawn over every frame shown on the canvas (not in screenshots):
// w x h pixels 0xAARRGGBB (app/hud.h makes them) at (x, y); argb = NULL removes it.
void gpu_view_hud(GpuView *v, const uint32_t *argb, int w, int h, int x, int y);
int gpu_view_present_scene(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes);
// 1 if the scene's media (densities, bricks, light grids: mp, or NULL for the path tracer,
// which has no light grid) fit this GPU's largest storage buffer, or there are none. If not
// (said once on stderr), the GPU draws no media: gpu_view_present_scene with volumes and the
// path tracer's start return 0, and the media are to be marched on the CPU.
int gpu_view_volumes_fit(GpuView *v, const Scene *s, const MarchParams *mp);
// Anti-aliasing of the rasterized frames shown (and their screenshots): on = FXAA after the
// composite (gpu/shaders/view/view_fxaa.wgsl), one more full-screen pass; 0 = none (default).
// Path traced images are left as they are (their samples already cover each pixel).
void gpu_view_aa(GpuView *v, int on);

// ---- path tracing on the GPU (gpu/trace_gpu.c: the light transport of rt/tracer.c) -------
// The image builds up sample by sample in a buffer on the GPU, a band of rows at a time so
// that a slow GPU stays responsive. The scene must not change while it is traced.
typedef struct {
    Camera cam;
    int max_depth;     // bounces per path
    int mis;           // multiple importance sampling
    real clamp;        // largest sample value, 0 = no limit
    int random;        // 1 = plain random numbers, 0 (default) = Sobol + blue noise (base/sampler.h)
    real photons;      // > 0: caustics by light tracing, this many light paths per pixel and pass
                       // (as -photons; the path tracer then leaves those paths out)
    int photon_cache;  // 1 = also a world-space photon cache, for real time (blurred a little;
                       // the light tracer's exact caustics take over as the view stands still)
    real indirect_clamp; // TraceParams.indirect_clamp (rt/tracer.h), 0 = off
    real indirect_clamp_move;   // the viewer: the same while things move, when lower (0: as indirect_clamp)
} GpuTraceParams;
// Starts a new image: the scene's geometry and BVH go to the GPU if they changed (the scene's
// own BVH is used when it has one, so call scene_set_accel after changing geometry). Returns 0
// if this GPU cannot hold the scene: then path trace on the CPU.
int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp);
// The path tracer's kernels asked for now, before the first image (a viewer that will path
// trace calls it once its view is made): in a browser they compile asynchronously, side by
// side, while frames go on (rasterized); natively they are made at once. Until they are ready
// gpu_view_trace_more does nothing (no passes: what is drawn is the rasterized scene).
void gpu_view_trace_prepare(GpuView *v);
// Goes on after something changed (the camera moved, the bodies, the media): a new image,
// whose pixels keep the samples their surface had in the last one (reprojected into the new
// camera; through mirrors and glass too), at most `history` each, so that a change of light
// shows within a few frames. Pixels that were hidden start from nothing (and show the
// rasterized scene until their first sample). Without the denoiser's planes (an image too
// big for this GPU) or with history 0, the same as gpu_view_trace_start. For real time: call
// it every frame something moved, then gpu_view_trace_more, then gpu_view_present_traced; the
// passes per frame stay within ~10 ms while it moves.
int gpu_view_trace_move(GpuView *v, const Scene *s, const GpuTraceParams *tp, int history);
// 1 = gpu_view_trace_move copies the scene's top level over its instances (scene_tlas: keep it
// up to date with scene_update_instances, or the GPU builds one, a little slower to trace);
// 0 = it never reads it (ray queries on our Vulkan layer, LR_GPU_TLAS=gpu, or 32768 instances
// and more: the GPU builds its own): moving instances then need no tree on the CPU at all.
int gpu_view_trace_reads_tlas(const GpuView *v, const Scene *s);
// Call every frame: unless the previous band is still running, submits the next one (p and mp
// as in gpu_view_present: the media are made ready the same way). Returns 0 when the path
// tracer cannot run (its pipelines failed): no pass will ever come, path trace on the CPU.
int gpu_view_trace_more(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp);
// No image gets more than spp samples per pixel (0, the default: no limit; a fast GPU is
// given several passes per submission, as many as its speed allows, so an image asked for
// until gpu_view_trace_passes reaches n could end with a few more, a different number from
// run to run). With the limit, gpu_view_trace_more does nothing once an image has them all.
void gpu_view_trace_limit(GpuView *v, int spp);
// The GPU time the path tracer's work may take per frame (bands of rows, or several passes),
// so that frames keep coming while it refines: ms > 0. 0 (the default): bands of 12-45 ms,
// the image cleaner sooner but the frames slower (while the camera moves: 10 ms either way).
void gpu_view_trace_budget(GpuView *v, double ms);
// With the budget: at least a whole pass per frame (when a pass takes up to max_ms; past it,
// max_ms), for frames that do not wait for the display: their fixed cost (the denoiser, the
// composite) beside a short band left the GPU mostly not path tracing. 0 (the default): no.
void gpu_view_trace_budget_pass(GpuView *v, double max_ms);
// While things move (a new image every frame): this budget instead (0: the one above)
void gpu_view_trace_budget_live(GpuView *v, double ms);
int gpu_view_trace_passes(const GpuView *v);        // samples per pixel finished
double gpu_view_trace_work(const GpuView *v);       // passes' worth of work done since the view was made, all images
int gpu_view_trace_busy(const GpuView *v);          // a band is still running on the GPU
// Reads the image back (asynchronously, like gpu_view_capture): fn gets w * h values of
// 4 floats, the sum of the samples (linear RGB) and their count. 0 if a read is already on
// its way. The image at full precision, for saving (bin/render, bin/sim). If the read fails
// (the buffer could not be mapped) fn is called with sum_count = NULL: once 1 was returned,
// fn is always called, unless the device is lost or the view destroyed first.
typedef void (*GpuTraceReadFn)(void *user, const float *sum_count, int w, int h);
int gpu_view_trace_read(GpuView *v, GpuTraceReadFn fn, void *user);
double gpu_view_trace_pass_ms(const GpuView *v);    // how long the last one took
// the band the path tracer runs now (rows, passes at a time) and its budget per frame (ms)
void gpu_view_trace_band(const GpuView *v, int *rows, int *reps, double *budget_ms);
// 1 = the path tracer runs as small kernels, one step of every path at a time (wavefront;
// docs/STATUS.md R1), 0 = as one kernel per path (the default)
void gpu_view_trace_wavefront(GpuView *v, int on);
// The denoiser (gpu/denoise_gpu.c): on = 1 shows and reads back the image denoised (the
// count of each pixel is then 1), 0 the samples as they are. Off by default. Returns 1 if it
// will be used: 0 when this GPU's buffers are too small for its guide at this image size.
int gpu_view_trace_denoise(GpuView *v, int on);
// While things move (a band that follows gpu_view_trace_move): the denoiser's a-trous passes,
// 1-5 (a footprint of 5, 9, 17, 33, 65 pixels); fewer cost less and smooth less. 0 (the
// default): all 5, as for a still image.
void gpu_view_trace_denoise_moving(GpuView *v, int passes);
// Shows the image so far. During the first pass, pixels with no sample yet show the scene
// rasterized (with its media if volumes = 1), so the picture is painted over, not blanked.
int gpu_view_present_traced(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes);
void gpu_view_destroy(GpuView *v);
// Benchmarks (bin/bench): the path tracer's GPU time from its own timers (gpu_set_timing)
// since the last reset: what = 0 a pass (1 sample per pixel, the whole image), 1 the tree
// builds of a frame that had some; 0 when not timed.
double gpu_view_gpu_ms(GpuView *v, int what);
void gpu_view_gpu_ms_reset(GpuView *v);
// The denoiser run n times on the samples there are now (gpu_view_trace_denoise on, a pass
// made), waited for: the milliseconds per run on the CPU's clock; < 0 if it cannot run.
double gpu_view_bench_denoise(GpuView *v, int n);

// Reads back the next frame gpu_view_present draws (a screenshot of the GPU's image):
// fn is called later, when the GPU has finished, with w * h pixels 0x00RRGGBB, or with
// pixels = NULL if they could not be read back. One read at a
// time: returns 0 (and does nothing) while the previous one is still on its way.
// Offscreen (gpu_create with no canvas) this is how frames are shown at all: ask for every one.
typedef void (*GpuCaptureFn)(void *user, const uint32_t *pixels, int w, int h);
int gpu_view_capture(GpuView *v, GpuCaptureFn fn, void *user);

#endif
