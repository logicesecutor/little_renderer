// octree.h - spatial acceleration: split space into 8 cubes, recursively, so a ray only
// tests the primitives in the cells it actually passes through.
#ifndef OCTREE_H
#define OCTREE_H

#include "geometry.h"

// Tuned on the dense torus (37k triangles): smaller leaves were not faster, they only
// duplicated more triangle references. Override with -DOCT_LEAF_SIZE=...
#ifndef OCT_LEAF_SIZE
#define OCT_LEAF_SIZE 16      // a cell with this many primitives or fewer is not split
#endif
#define OCT_DEPTH_LIMIT 14     // hard upper limit for the maximum depth
#define OCT_MAX_NODES 8000000  // safety net against pathological scenes

typedef struct {
    Aabb box;
    int first_child;   // index of the first of 8 consecutive children, -1 for a leaf
    int prim_start;    // leaf only: range [prim_start, prim_start + prim_count) in refs
    int prim_count;
} OctNode;

typedef struct {
    OctNode *nodes;  int nnodes, cap_nodes;   // nodes[0] is the root
    int *refs;       int nrefs, cap_refs;     // primitive indices, grouped by leaf
    int max_depth;                            // depth limit used for this build
    int leaves, depth;                        // statistics
} Octree;

// A good maximum depth for n primitives. A surface mesh fills cells roughly like a 2D sheet,
// so the number of useful leaves grows ~4x per level: depth = log4(n / leaf size) + 2,
// clamped to [6, OCT_DEPTH_LIMIT]. (torus 2.3k -> 6, dense torus 37k -> 8, dragon 871k -> 10)
int octree_auto_depth(int nprims);

// max_depth <= 0 means octree_auto_depth(nprims)
void octree_build(Octree *o, const Primitive *prims, int nprims, int max_depth);
void octree_free(Octree *o);

// Closest hit in (tmin, tmax). With any_hit = 1 it returns as soon as ANY hit is found,
// which is all a shadow ray needs. h->prim is set.
int octree_intersect(const Octree *o, const Primitive *prims, const Ray *ray,
                     double tmin, double tmax, Hit *h, int any_hit);

#endif
