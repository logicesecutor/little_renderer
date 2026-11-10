// bvh.h - bounding volume hierarchy: a binary tree of boxes around groups of primitives.
//
// Unlike the octree (which cuts SPACE into fixed cubes), a BVH splits the list of
// PRIMITIVES into two groups again and again, and each node stores the box around its
// group. Every primitive is in exactly one leaf, and boxes adapt to where the geometry is.
#ifndef LR_BVH_H
#define LR_BVH_H

#include "rt/intersect.h"

typedef enum {
    BVH_SPLIT_SAH,      // surface area heuristic: pick the cheapest split (best trees)
    BVH_SPLIT_MIDDLE    // cut the longest axis in the middle (simple, faster to build)
} BvhSplit;

#define BVH_MAX_DEPTH 100   // hard limit; the traversal stack is sized from it

typedef struct {
    Aabb box;
    int left, right;    // children (inner node)
    int start, count;   // leaf: refs[start .. start + count); count == 0 for inner nodes
} BvhNode;

typedef struct {
    BvhNode *nodes;  int nnodes, cap;   // nodes[0] is the root
    int *refs;       int nrefs;         // primitive indices, reordered so leaves are ranges
    int leaves, depth;                  // statistics
    int max_leaf, bins;                 // build parameters used
    BvhSplit split;
} Bvh;

// max_leaf: a node with this many primitives or fewer MAY become a leaf (SAH decides);
// bins: number of buckets per axis for the binned SAH (8..64 is typical).
// Builds over all shapes of the scene (primitives and triangles).
void bvh_build(Bvh *b, const Scene *s, int max_leaf, int bins, BvhSplit split);
void bvh_free(Bvh *b);

// Closest hit in (tmin, tmax); any_hit = 1 stops at the first hit (shadow rays).
int bvh_intersect(const Bvh *b, const Scene *s, const Ray *ray,
                  real tmin, real tmax, Hit *h, int any_hit);

#endif
