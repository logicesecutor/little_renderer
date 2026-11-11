// volume_march.h - the rasterizer's clouds: participating media (core/volume.h) drawn in
// real time by raymarching, over a frame render_scene has already drawn.
//
// The path tracer follows random paths through a cloud, which takes seconds per clean
// image. Here every pixel marches ONE ray through the media in steps, from the camera to
// the opaque surface the z-buffer says it hits, and adds up, front to back:
//
//     for each step of length ds:       sigma = sigma_t * density(x)
//         L += T * (1 - exp(-sigma ds)) * albedo * S(x)        light scattered towards the eye
//         T *= exp(-sigma ds)                                  what is left of the light behind
//     pixel = L + T * pixel
//
// where S(x) is the light arriving at x, from the lamp (dimmed by the media between x and
// the lamp: a "light grid" of optical depths, precomputed once per change of the media) and
// from the walls (an ambient term). The many bounces of light inside a cloud, which make it
// bright and soft, are approximated by "octaves" (Wrenninge 2015): the lamp term is summed
// again with half the optical depth, half the weight and half the anisotropy, twice.
//
// Not a path tracer: no shadows of the media on the surfaces, no light bounced by the
// media onto the walls, approximate multiple scattering. But milliseconds per frame.
#ifndef LR_VOLUME_MARCH_H
#define LR_VOLUME_MARCH_H

#include "core/image.h"
#include "core/scene.h"
#include "raster/render.h"

typedef struct {
    int max_steps;     // most steps per ray (more = finer detail, slower)        (128)
    real step_voxels;  // step length in voxels of the finest medium                (1)
    int light_res;     // light grid: cells along the longest side of each medium   (48)
    real ambient;      // < 0: estimated from the scene's lamp and walls; else this radiance
    int octaves;       // multiple-scattering octaves, 1 = single scattering only     (3)
    // The rasterizer's lamp has no distance falloff and no shadows, so its rooms are about
    // twice as bright as the path tracer's; the media's lamp light is scaled by this to
    // look right next to them (1 = the physical value, as in the path tracer).       (2)
    real brightness;
    int scale;         // 1 = one ray per pixel; 2 = one per 2x2 block, upsampled (4x faster) (1)
} MarchParams;

MarchParams march_default_params(void);

// Keeps the light grids between frames: they are rebuilt only when a medium changes
// (Volume.version) or the lamp moves.
typedef struct RasterVolumes RasterVolumes;
RasterVolumes *raster_volumes_create(void);
void raster_volumes_free(RasterVolumes *rv);

// Composites every medium of s over fb, using fb's z-buffer to stop each ray at the opaque
// surface. Call after render_scene, with the same RenderParams (view, proj, eye). Does
// nothing if the scene has no media.
void raster_draw_volumes(RasterVolumes *rv, Framebuffer *fb, const Scene *s, const RenderParams *p,
                         const MarchParams *mp);

// timings of the last call (milliseconds): light grids (0 when reused), marching
void raster_volumes_timing(const RasterVolumes *rv, double *ms_light, double *ms_march);

// The pieces of the lighting model, shared with the GPU raymarcher (gpu/view_gpu.c) so that
// both draw the same cloud.
struct Volume;
typedef struct { Vec3 pos, normal, radiance; real area; } MarchLamp;   // normal 0 = all sides
MarchLamp march_lamp(const Scene *s);
Vec3 march_ambient(const Scene *s, const MarchParams *mp);              // the walls' light
void march_light_dims(const struct Volume *v, int light_res, int dims[3]);

#endif
