// gpu_frame.h - whole images on the GPU, for the programs that wait for them (bin/render and
// bin/sim): a GPU with no window, the path tracer run to a number of samples, the rasterizer,
// each read back to the CPU. Apps only; the engine's GPU API is gpu/gpu.h and gpu/view_gpu.h.
#ifndef LR_GPU_FRAME_H
#define LR_GPU_FRAME_H

#include "core/image.h"
#include "core/scene.h"
#include "gpu/gpu.h"
#include "gpu/view_gpu.h"

// A GPU without a window, and a view of w x h on it: waits until it is ready (a few seconds
// at most). 0 on success; otherwise prints why and the program uses the CPU.
int gpu_frame_open(int w, int h, Gpu **gpu, GpuView **view);
void gpu_frame_close(Gpu *gpu, GpuView *view);

// spp samples per pixel of the scene, path traced on the GPU, into img (linear RGB, the
// average). The scene must have its BVH up to date (scene_set_accel), or none (one is built).
// sampler: 1 = Sobol + blue noise, 0 = plain random numbers (core/sampler.h). photons > 0:
// caustics by light tracing, this many light paths per pixel and sample (0: path traced).
// progress = 1 prints a percentage. 0 on success, 1 if this GPU cannot hold the scene.
int gpu_frame_trace(Gpu *gpu, GpuView *view, const Scene *s, const Camera *cam, const RenderParams *rp,
                    const MarchParams *mp, int spp, int max_depth, int mis, real clamp, int sampler, real photons,
                    Image *img, int progress);

// The scene rasterized on the GPU (with its media raymarched over it if volumes = 1) into
// fb->color. 0 on success, 1 if this GPU cannot draw the scene.
int gpu_frame_raster(Gpu *gpu, GpuView *view, const Scene *s, const RenderParams *rp, const MarchParams *mp,
                     int volumes, Framebuffer *fb);

#endif
