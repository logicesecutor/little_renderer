// frame.h - one whole image of a scene with the path tracer, for programs that render many
// (an animation). Apps only; the engine's API for tracing is rt/tracer.h.
#ifndef LR_FRAME_H
#define LR_FRAME_H

#include <stdint.h>
#include "core/camera.h"
#include "core/image.h"
#include "core/scene.h"
#include "rt/tracer.h"

// spp paths per pixel into img (already created at the size wanted), in parallel over
// rows. Every pixel's random numbers come from (pixel, seed): give each frame its own seed.
// clamp > 0 caps a sample's brightness. With caustics_photons > 0 the caustics come from
// light tracing (tp->skip_caustics must then be 1), that many light paths per pixel sample.
void frame_render_rt(const Scene *s, const Camera *cam, Image *img, int spp, const TraceParams *tp,
                     real clamp, double caustics_photons, uint64_t seed);

#endif
