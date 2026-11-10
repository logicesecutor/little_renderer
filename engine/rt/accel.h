// accel.h - path tracer: the acceleration structure of a scene (none, octree or BVH) and
// the scene-wide ray queries built on it.
#ifndef LR_ACCEL_H
#define LR_ACCEL_H

#include "core/scene.h"
#include "rt/bvh.h"
#include "rt/intersect.h"
#include "rt/octree.h"

typedef enum { ACCEL_NONE, ACCEL_OCTREE, ACCEL_BVH } AccelKind;

typedef struct {
    int oct_depth;          // octree max depth, <= 0 = automatic
    int bvh_leaf;           // BVH: max primitives per leaf
    int bvh_bins;           // BVH: SAH buckets per axis
    BvhSplit bvh_split;     // BVH: SAH or middle split
} AccelParams;

AccelParams accel_default_params(void);   // {automatic, 4, 16, SAH}

// (Re)builds the chosen structure for all primitives currently in the scene and stores it
// in s->accel (scene_free frees it). Call it again after adding primitives.
void scene_set_accel(Scene *s, AccelKind kind, const AccelParams *p);
AccelKind scene_accel_kind(const Scene *s);
const char *accel_name(AccelKind kind);
// one line of statistics about the current structure, e.g. for printing
void scene_accel_stats(const Scene *s, char *buf, int len);

// closest hit over all primitives
int scene_intersect(const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h);
// is there ANY hit in (tmin, tmax)? (shadow rays)
int scene_occluded(const Scene *s, const Ray *ray, real tmin, real tmax);

#endif
