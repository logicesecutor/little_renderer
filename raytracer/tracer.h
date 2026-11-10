// tracer.h - the path tracer: estimates the light arriving along one camera ray.
#ifndef TRACER_H
#define TRACER_H

#include "geometry.h"
#include "rng.h"
#include "scene.h"

// Returns linear RGB radiance (not clamped, not gamma corrected).
// max_depth = maximum number of bounces before the path is cut.
Vec3 trace_path(const Scene *s, Ray ray, Rng *rng, int max_depth);

#endif
