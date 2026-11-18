// tracer.h - the path tracer: estimates the light arriving along one camera ray, and the
// light tracer that adds the caustics a path tracer finds badly.
#ifndef LR_TRACER_H
#define LR_TRACER_H

#include "scene/camera.h"
#include "scene/image.h"
#include "base/rng.h"
#include "base/sampler.h"
#include "scene/scene.h"

typedef struct {
    int max_depth;   // maximum number of bounces before a path is cut
    // Multiple importance sampling: the lamp can be found by sampling it directly (next
    // event estimation) or by a bounce that happens to hit it; with MIS both count, each
    // weighted by the power heuristic. 0 = only direct sampling (the classic estimator).
    int mis;
    // 1 = leave out the paths camera -> diffuse -> mirror/glass... -> lamp, because
    // light_trace_caustics adds them (much less noisily). Only set it if you run it.
    int skip_caustics;
    // For the pixel loops (and the GPU): 1 = the numbers of each path from the Sobol sequence
    // with blue noise (base/sampler.h), 0 = from the pixel's Rng (the images of before).
    int sampler;
    // 1 = the lamp is not counted where the ray itself (before any bounce) meets it: the
    // light probes (rt/probe.h), whose users light with the lamp themselves
    int no_lamp_seen;
    // 1 = the same for the sky: not counted where the ray itself leaves the scene (the
    // irradiance grid samples the sky's direct light on its own)
    int no_sky_seen;
} TraceParams;

TraceParams trace_default_params(void);   // max_depth 50, mis on, skip_caustics off, sampler on

// Returns linear RGB radiance (not clamped, not gamma corrected).
// Uses the scene's acceleration structure if scene_set_accel was called.
// Thread-safe: it only reads the scene (give every thread its own Rng).
Vec3 trace_path_ex(const Scene *s, Ray ray, Rng *rng, const TraceParams *p);
// the same with trace_default_params() and this max_depth
Vec3 trace_path(const Scene *s, Ray ray, Rng *rng, int max_depth);
// The same, the path's numbers from a Sampler (base/sampler.h): make one per pixel and
// sample, with sampler_make(&rng, x, y, sample_index, image_seed, on), and take the camera
// ray's jitter from it too: sampler_2d(&sm, SAMPLE_CAMERA, &jx, &jy).
Vec3 trace_path_sampled(const Scene *s, Ray ray, Sampler *sm, const TraceParams *p);

// A whole image: spp paths per pixel through cam into px (w * h, rows top to bottom; the
// mean of each pixel's samples), in parallel over rows. Every pixel's numbers come from
// (pixel, seed): an animation gives each frame its own seed. Samples that are not numbers
// (NaN) are left out; clamp > 0 caps a sample's brightness (biased). row_done (or NULL) is
// called after each row, from the thread that traced it (a progress line).
void trace_image(const Scene *s, const Camera *cam, const TraceParams *p, int w, int h, int spp, uint64_t seed,
                 real clamp, Vec3 *px, void (*row_done)(void *user), void *user);

// The sky (Scene.env) is a light like the lamp: next event estimation draws a direction by
// its luminance (env_sample) at every rough vertex, and a path leaving the scene sees it,
// the two weighted by MIS. Light tracing covers only the lamp's caustics: the sky's are path
// traced.

// ---- light tracing for caustics --------------------------------------------------------
//
// Traces `photons` light paths from the lamp. A path that bounces off at least one mirror or
// glass surface and then lands on a diffuse surface is connected to the camera, and its light
// is added to the pixel it lands in. These are exactly the paths trace_path leaves out with
// skip_caustics = 1, so the two together are an unbiased image: add the result to the path
// traced average. The image is the same for any number of threads (fixed-point atomic
// sums). Runs in parallel (OpenMP).
//
//   out:  w * h linear RGB values, ADDED to (so it can start as the path traced image)
//   seed: changes the random numbers (e.g. per progressive pass)
void light_trace_caustics(const Scene *s, const Camera *cam, int w, int h, long long photons,
                          const TraceParams *p, uint64_t seed, Vec3 *out);

#endif
