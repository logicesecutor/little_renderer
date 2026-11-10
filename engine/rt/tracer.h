// tracer.h - the path tracer: estimates the light arriving along one camera ray, and the
// light tracer that adds the caustics a path tracer finds badly.
#ifndef LR_TRACER_H
#define LR_TRACER_H

#include "core/camera.h"
#include "core/image.h"
#include "core/rng.h"
#include "core/scene.h"

typedef struct {
    int max_depth;   // maximum number of bounces before a path is cut
    // Multiple importance sampling: the lamp can be found by sampling it directly (next
    // event estimation) or by a bounce that happens to hit it; with MIS both count, each
    // weighted by the power heuristic. 0 = only direct sampling (the classic estimator).
    int mis;
    // 1 = leave out the paths camera -> diffuse -> mirror/glass... -> lamp, because
    // light_trace_caustics adds them (much less noisily). Only set it if you run it.
    int skip_caustics;
} TraceParams;

TraceParams trace_default_params(void);   // max_depth 50, mis on, skip_caustics off

// Returns linear RGB radiance (not clamped, not gamma corrected).
// Uses the scene's acceleration structure if scene_set_accel was called.
// Thread-safe: it only reads the scene (give every thread its own Rng).
Vec3 trace_path_ex(const Scene *s, Ray ray, Rng *rng, const TraceParams *p);
// the same with trace_default_params() and this max_depth
Vec3 trace_path(const Scene *s, Ray ray, Rng *rng, int max_depth);

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
