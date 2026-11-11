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
int gpu_view_present_scene(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes);

// ---- path tracing on the GPU (gpu/trace_gpu.c: the light transport of rt/tracer.c) -------
// The image builds up sample by sample in a buffer on the GPU, a band of rows at a time so
// that a slow GPU stays responsive. The scene must not change while it is traced.
typedef struct {
    Camera cam;
    int max_depth;     // bounces per path
    int mis;           // multiple importance sampling
    real clamp;        // largest sample value, 0 = no limit
} GpuTraceParams;
// Starts a new image: the scene's geometry and BVH go to the GPU if they changed (the scene's
// own BVH is used when it has one, so call scene_set_accel after changing geometry). Returns 0
// if this GPU cannot hold the scene: then path trace on the CPU.
int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp);
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
