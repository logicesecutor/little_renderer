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

// The same tree, 4 children per node (bvh_make_wide): each node holds its children's boxes
// side by side, so one step of a ray tests 4 boxes at once (in a loop the compiler turns
// into SIMD) and the tree is half as deep. Made by collapsing the binary tree: a node takes
// its two children, then again and again replaces the inner child with the largest box by
// that child's two children, until it has 4.
typedef struct {
    real b[6][4];     // child k's box: lo x, y, z = b[0..2][k], hi x, y, z = b[3..5][k]
    int child[4];     // inner child: its index in wide[]; leaf: its first ref
    int count[4];     // leaf: its refs (> 0); inner child: 0; an empty slot: -1
} BvhWideNode;

typedef struct {
    BvhNode *nodes;  int nnodes, cap;   // nodes[0] is the root
    int *refs;       int nrefs;         // primitive indices, reordered so leaves are ranges
    int leaves, depth;                  // statistics
    int max_leaf, bins;                 // build parameters used
    BvhSplit split;
    BvhWideNode *wide; int nwide, wide_cap;   // the 4-wide version (wide[0] the root), or none
} Bvh;

// max_leaf: a node with this many primitives or fewer MAY become a leaf (SAH decides);
// bins: number of buckets per axis for the binned SAH (8..64 is typical).
// Builds over all shapes of the scene (primitives and triangles).
void bvh_build(Bvh *b, const Scene *s, int max_leaf, int bins, BvhSplit split);
// the same over any n boxes (refs are their indices): e.g. the instances' world boxes
void bvh_build_boxes(Bvh *b, const Aabb *boxes, int n, int max_leaf, int bins, BvhSplit split);
void bvh_free(Bvh *b);
// After the items of bvh_build_boxes moved (the same items, new boxes): the same tree with
// its boxes made again from the leaves up (a "refit": O(n), no sorting). Returns the tree's
// cost, as bvh_cost. The tree gets worse as things move apart: rebuild when the cost grew.
real bvh_refit_boxes(Bvh *b, const Aabb *boxes);
// the expected cost of a random ray through the tree: the sum of its nodes' box areas over
// the root's (the SAH's measure of a tree)
real bvh_cost(const Bvh *b);
// adds the 4-wide version of the tree; from then on bvh_intersect walks that one
void bvh_make_wide(Bvh *b);

// Closest hit in (tmin, tmax); any_hit = 1 stops at the first hit (shadow rays).
int bvh_intersect(const Bvh *b, const Scene *s, const Ray *ray,
                  real tmin, real tmax, Hit *h, int any_hit);

#endif
