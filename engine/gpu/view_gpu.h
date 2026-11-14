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

#include "core/camera.h"
#include "core/image.h"
#include "core/scene.h"
#include "gpu/gpu.h"
#include "raster/render.h"
#include "raster/volume_march.h"

#define GPU_MAX_VOLUMES 2

typedef struct GpuView GpuView;

GpuView *gpu_view_create(Gpu *g, int width, int height);   // NULL if the GPU is not ready
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

// ---- path tracing on the GPU (gpu/trace_gpu.c: the light transport of rt/tracer.c) -------
// The image builds up sample by sample in a buffer on the GPU, a band of rows at a time so
// that a slow GPU stays responsive. The scene must not change while it is traced.
typedef struct {
    Camera cam;
    int max_depth;     // bounces per path
    int mis;           // multiple importance sampling
    real clamp;        // largest sample value, 0 = no limit
    int random;        // 1 = plain random numbers, 0 (default) = Sobol + blue noise (core/sampler.h)
    real photons;      // > 0: caustics by light tracing, this many light paths per pixel and pass
                       // (as -photons; the path tracer then leaves those paths out)
    int photon_cache;  // 1 = also a world-space photon cache, for real time (blurred a little;
                       // the light tracer's exact caustics take over as the view stands still)
} GpuTraceParams;
// Starts a new image: the scene's geometry and BVH go to the GPU if they changed (the scene's
// own BVH is used when it has one, so call scene_set_accel after changing geometry). Returns 0
// if this GPU cannot hold the scene: then path trace on the CPU.
int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp);
// Goes on after something changed (the camera moved, the bodies, the media): a new image,
// whose pixels keep the samples their surface had in the last one (reprojected into the new
// camera; through mirrors and glass too), at most `history` each, so that a change of light
// shows within a few frames. Pixels that were hidden start from nothing (and show the
// rasterized scene until their first sample). Without the denoiser's planes (an image too
// big for this GPU) or with history 0, the same as gpu_view_trace_start. For real time: call
// it every frame something moved, then gpu_view_trace_more, then gpu_view_present_traced; the
// passes per frame stay within ~10 ms while it moves.
int gpu_view_trace_move(GpuView *v, const Scene *s, const GpuTraceParams *tp, int history);
// Call every frame: unless the previous band is still running, submits the next one (p and mp
// as in gpu_view_present: the media are made ready the same way).
void gpu_view_trace_more(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp);
int gpu_view_trace_passes(const GpuView *v);        // samples per pixel finished
int gpu_view_trace_busy(const GpuView *v);          // a band is still running on the GPU
// Reads the image back (asynchronously, like gpu_view_capture): fn gets w * h values of
// 4 floats, the sum of the samples (linear RGB) and their count. 0 if a read is already on
// its way. The image at full precision, for saving (bin/render, bin/sim).
typedef void (*GpuTraceReadFn)(void *user, const float *sum_count, int w, int h);
int gpu_view_trace_read(GpuView *v, GpuTraceReadFn fn, void *user);
double gpu_view_trace_pass_ms(const GpuView *v);    // how long the last one took
// 1 = the path tracer runs as small kernels, one step of every path at a time (wavefront;
// docs/PERFORMANCE.md R1), 0 = as one kernel per path (the default)
void gpu_view_trace_wavefront(GpuView *v, int on);
// The denoiser (gpu/denoise_gpu.c): on = 1 shows and reads back the image denoised (the
// count of each pixel is then 1), 0 the samples as they are. Off by default. Returns 1 if it
// will be used: 0 when this GPU's buffers are too small for its guide at this image size.
int gpu_view_trace_denoise(GpuView *v, int on);
// Shows the image so far. During the first pass, pixels with no sample yet show the scene
// rasterized (with its media if volumes = 1), so the picture is painted over, not blanked.
int gpu_view_present_traced(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes);
void gpu_view_destroy(GpuView *v);

// Reads back the next frame gpu_view_present draws (a screenshot of the GPU's image):
// fn is called later, when the GPU has finished, with w * h pixels 0x00RRGGBB. One read at a
// time: returns 0 (and does nothing) while the previous one is still on its way.
// Offscreen (gpu_create with no canvas) this is how frames are shown at all: ask for every one.
typedef void (*GpuCaptureFn)(void *user, const uint32_t *pixels, int w, int h);
int gpu_view_capture(GpuView *v, GpuCaptureFn fn, void *user);

#endif
