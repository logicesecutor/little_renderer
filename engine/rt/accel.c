#define _POSIX_C_SOURCE 200809L   // clock_gettime
#include "rt/accel.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    AccelKind kind;
    Octree octree;
    Bvh bvh;
    double build_ms;
} RtAccel;

static double clock_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;   // (casts: see Makefile, PRECFLAGS)
}

static void rt_accel_free(void *p){
    RtAccel *a = p;
    octree_free(&a->octree);
    bvh_free(&a->bvh);
    free(a);
}

AccelParams accel_default_params(void){
    AccelParams p = {0, 4, 16, BVH_SPLIT_SAH};
    return p;
}

void scene_set_accel(Scene *s, AccelKind kind, const AccelParams *p){
    if (s->accel && s->accel_free) s->accel_free(s->accel);
    RtAccel *a = calloc(1, sizeof(*a));
    if (!a){ perror("calloc"); exit(1); }
    double t0 = clock_ms();
    if (kind == ACCEL_OCTREE) octree_build(&a->octree, s, p->oct_depth);
    if (kind == ACCEL_BVH) bvh_build(&a->bvh, s, p->bvh_leaf, p->bvh_bins, p->bvh_split);
    a->build_ms = clock_ms() - t0;
    a->kind = kind;
    s->accel = a;
    s->accel_free = rt_accel_free;
}

AccelKind scene_accel_kind(const Scene *s){
    const RtAccel *a = s->accel;
    return a ? a->kind : ACCEL_NONE;
}

const Bvh *scene_bvh(const Scene *s){
    const RtAccel *a = s->accel;
    return a && a->kind == ACCEL_BVH ? &a->bvh : NULL;
}

const char *accel_name(AccelKind kind){
    return kind == ACCEL_OCTREE ? "octree" : (kind == ACCEL_BVH ? "bvh" : "none");
}

void scene_accel_stats(const Scene *s, char *buf, int len){
    const RtAccel *a = s->accel;
    const int nshapes = scene_num_shapes(s);
    if (a && a->kind == ACCEL_OCTREE){
        const Octree *o = &a->octree;
        snprintf(buf, len, "octree: %d primitives -> %d nodes, %d leaves, depth %d (max %d), "
                 "%.2f refs/primitive, built in %.1f ms", nshapes, o->nnodes, o->leaves, o->depth,
                 o->max_depth, (double)o->nrefs / nshapes, a->build_ms);
    } else if (a && a->kind == ACCEL_BVH){
        const Bvh *b = &a->bvh;
        snprintf(buf, len, "bvh (%s, leaf <= %d, %d bins): %d primitives -> %d nodes, %d leaves, "
                 "depth %d, %.2f primitives/leaf, built in %.1f ms",
                 b->split == BVH_SPLIT_SAH ? "sah" : "middle", b->max_leaf, b->bins, nshapes,
                 b->nnodes, b->leaves, b->depth, (double)nshapes / b->leaves, a->build_ms);
    } else {
        snprintf(buf, len, "no acceleration: every ray tests all %d primitives", nshapes);
    }
}

int scene_intersect(const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h){
    const RtAccel *a = s->accel;
    if (a && a->kind == ACCEL_BVH) return bvh_intersect(&a->bvh, s, ray, tmin, tmax, h, 0);
    if (a && a->kind == ACCEL_OCTREE) return octree_intersect(&a->octree, s, ray, tmin, tmax, h, 0);

    // brute force: test everything, keep the closest
    int found = 0;
    Hit tmp;
    for (int i = 0; i < scene_num_shapes(s); i++){
        if (intersect_shape(s, i, ray, tmin, tmax, &tmp)){
            found = 1;
            tmax = tmp.t;          // only accept closer hits from now on
            tmp.prim = i;
            *h = tmp;
        }
    }
    return found;
}

int scene_occluded(const Scene *s, const Ray *ray, real tmin, real tmax){
    const RtAccel *a = s->accel;
    Hit tmp;
    if (a && a->kind == ACCEL_BVH) return bvh_intersect(&a->bvh, s, ray, tmin, tmax, &tmp, 1);
    if (a && a->kind == ACCEL_OCTREE) return octree_intersect(&a->octree, s, ray, tmin, tmax, &tmp, 1);
    for (int i = 0; i < scene_num_shapes(s); i++)
        if (intersect_shape(s, i, ray, tmin, tmax, &tmp)) return 1;
    return 0;
}
