#include "rt/bvh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- small box helpers ----------------------------------------------------------------

static Aabb box_empty(void){
    Aabb b = {vec3(REAL_MAX, REAL_MAX, REAL_MAX), vec3(-REAL_MAX, -REAL_MAX, -REAL_MAX)};
    return b;
}

// (plain comparisons, not fmin / fmax: the boxes have no NaN, and these are the builder's
// innermost loop: a library call each made the build 3x slower)
static inline real mn(real a, real b){ return a < b ? a : b; }
static inline real mx(real a, real b){ return a > b ? a : b; }

static void box_add_box(Aabb *b, const Aabb *o){
    b->lo.x = mn(b->lo.x, o->lo.x); b->lo.y = mn(b->lo.y, o->lo.y); b->lo.z = mn(b->lo.z, o->lo.z);
    b->hi.x = mx(b->hi.x, o->hi.x); b->hi.y = mx(b->hi.y, o->hi.y); b->hi.z = mx(b->hi.z, o->hi.z);
}

static void box_add_point(Aabb *b, Vec3 p){
    b->lo.x = mn(b->lo.x, p.x); b->lo.y = mn(b->lo.y, p.y); b->lo.z = mn(b->lo.z, p.z);
    b->hi.x = mx(b->hi.x, p.x); b->hi.y = mx(b->hi.y, p.y); b->hi.z = mx(b->hi.z, p.z);
}

static real box_area(const Aabb *b){
    Vec3 d = v_sub(b->hi, b->lo);
    if (d.x < 0 || d.y < 0 || d.z < 0) return 0;   // empty box
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

static real comp(Vec3 v, int axis){ return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }

// ---- building -----------------------------------------------------------------------

typedef struct {
    Aabb *box;         // bounds of every primitive
    Vec3 *centroid;    // center of each of those bounds: what we sort / bin by
} BuildInfo;

static int new_node(Bvh *b){
    if (b->nnodes >= b->cap){
        b->cap = b->cap ? b->cap * 2 : 1024;
        b->nodes = realloc(b->nodes, (size_t)b->cap * sizeof(BvhNode));
        if (!b->nodes){ perror("realloc"); exit(1); }
    }
    memset(&b->nodes[b->nnodes], 0, sizeof(BvhNode));
    return b->nnodes++;
}

// Moves the refs in [start, start+count) whose test is true to the front; returns how many.
static int partition(int *refs, int start, int count, const BuildInfo *bi, int axis, real split_pos){
    int i = start, j = start + count - 1;
    while (i <= j){
        if (comp(bi->centroid[refs[i]], axis) < split_pos) i++;
        else { int t = refs[i]; refs[i] = refs[j]; refs[j] = t; j--; }
    }
    return i - start;
}

// Binned SAH. The expected cost of a node is
//   cost = C_trav + (A_left / A_node) * N_left * C_isect + (A_right / A_node) * N_right * C_isect
// because a random ray that hits the node hits a child with probability ~ (child area /
// node area). We try `bins - 1` split planes per axis, using buckets of centroids, and keep
// the cheapest. Returns the split position (or a NaN-free "no split" via *ok = 0).
static real sah_split(const Bvh *b, const BuildInfo *bi, int start, int count,
                        const Aabb *node_box, const Aabb *cbox, int *best_axis, real *best_cost){
    enum { MAX_BINS = 256 };
    int nb = b->bins < 2 ? 2 : (b->bins > MAX_BINS ? MAX_BINS : b->bins);
    real parent_area = box_area(node_box);
    real best_pos = 0;
    *best_cost = REAL_MAX;
    *best_axis = -1;

    for (int axis = 0; axis < 3; axis++){
        real lo = comp(cbox->lo, axis), hi = comp(cbox->hi, axis);
        real extent = hi - lo;
        if (extent <= 1e-12) continue;   // all centroids in one plane: nothing to split here

        int bin_n[MAX_BINS] = {0};
        Aabb bin_box[MAX_BINS];
        for (int k = 0; k < nb; k++) bin_box[k] = box_empty();
        const real scale = nb / extent;
        for (int i = start; i < start + count; i++){
            int p = b->refs[i];
            int k = (int)((comp(bi->centroid[p], axis) - lo) * scale);
            if (k >= nb) k = nb - 1;
            bin_n[k]++;
            box_add_box(&bin_box[k], &bi->box[p]);
        }

        // sweep from the right to get the area/count of everything right of each plane
        real right_area[MAX_BINS];
        int right_n[MAX_BINS];
        Aabb acc = box_empty();
        int n = 0;
        for (int k = nb - 1; k > 0; k--){
            box_add_box(&acc, &bin_box[k]);
            n += bin_n[k];
            right_area[k] = box_area(&acc);
            right_n[k] = n;
        }
        // sweep from the left and evaluate the cost of the plane between bin k-1 and k
        acc = box_empty();
        n = 0;
        for (int k = 1; k < nb; k++){
            box_add_box(&acc, &bin_box[k - 1]);
            n += bin_n[k - 1];
            if (n == 0 || right_n[k] == 0) continue;
            real cost = 1.0 + (box_area(&acc) * n + right_area[k] * right_n[k]) / parent_area;
            if (cost < *best_cost){
                *best_cost = cost;
                *best_axis = axis;
                best_pos = lo + extent * k / nb;
            }
        }
    }
    return best_pos;
}

// The box of refs[start, start+count) and how to split it: returns how many go to the left
// child (0 = make a leaf). Partitions refs in place. The same decisions whether the tree is
// built by one thread or many.
static int choose_split(const Bvh *b, const BuildInfo *bi, int start, int count, int depth, Aabb *box_out){
    Aabb box = box_empty(), cbox = box_empty();
    for (int i = start; i < start + count; i++){
        box_add_box(&box, &bi->box[b->refs[i]]);
        box_add_point(&cbox, bi->centroid[b->refs[i]]);
    }
    *box_out = box;

    int nleft = 0;
    if (count > 1 && depth < BVH_MAX_DEPTH){
        if (b->split == BVH_SPLIT_SAH){
            int axis;
            real cost;
            real pos = sah_split(b, bi, start, count, &box, &cbox, &axis, &cost);
            // Leaf if not splitting is cheaper (cost of a leaf = count tests), unless the
            // leaf would be too big.
            int want_split = axis >= 0 && (cost < count || count > b->max_leaf);
            if (want_split) nleft = partition(b->refs, start, count, bi, axis, pos);
        } else if (count > b->max_leaf){
            // middle split: longest axis of the centroid box, cut at its center
            Vec3 ext = v_sub(cbox.hi, cbox.lo);
            int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
            real mid = 0.5 * (comp(cbox.lo, axis) + comp(cbox.hi, axis));
            if (comp(ext, axis) > 1e-12) nleft = partition(b->refs, start, count, bi, axis, mid);
        }
        // Everything ended up on one side (e.g. identical centroids) but the node is too
        // big: split the list in half, in whatever order it is. Still a valid tree.
        if ((nleft == 0 || nleft == count) && count > b->max_leaf) nleft = count / 2;
        if (nleft == count) nleft = 0;
    }
    return nleft;
}

// Serial build of a subtree into b->nodes, in preorder (a node, then its whole left
// subtree, then its right subtree). Returns the node's index.
static int build(Bvh *b, const BuildInfo *bi, int start, int count, int depth){
    int node = new_node(b);
    if (depth > b->depth) b->depth = depth;
    Aabb box;
    int nleft = choose_split(b, bi, start, count, depth, &box);
    b->nodes[node].box = box;

    if (nleft == 0){                                      // leaf
        b->nodes[node].start = start;
        b->nodes[node].count = count;
        b->leaves++;
        return node;
    }
    int l = build(b, bi, start, nleft, depth + 1);        // (indices, not pointers: the
    int r = build(b, bi, start + nleft, count - nleft, depth + 1);   // array may move)
    b->nodes[node].left = l;
    b->nodes[node].right = r;
    b->nodes[node].count = 0;
    return node;
}

// ---- parallel build ------------------------------------------------------------------
//
// The two children of a node are independent: they own disjoint ranges of refs. So the top
// of the tree (nodes with more than BVH_PAR_MIN primitives) is split in parallel with
// OpenMP tasks, and each smaller subtree is built serially (build above) into its own node
// array. A final pass copies everything, in preorder, into one array: the layout comes out
// exactly as if one thread had built the whole tree, so the tree, the statistics and the
// images do not depend on the number of threads.

#define BVH_PAR_MIN 1024

typedef struct Sub Sub;
struct Sub {
    int big;               // 1: a node split in parallel, with two child subtrees
    BvhNode node;          // big: the node (its left/right are filled in by the final pass)
    Sub *left, *right;
    Bvh part;              // small: a serially built subtree (nodes, depth, leaves)
};

static Sub *build_par(const Bvh *b, const BuildInfo *bi, int start, int count, int depth){
    Sub *sb = calloc(1, sizeof(Sub));
    if (!sb){ perror("calloc"); exit(1); }
    if (count <= BVH_PAR_MIN || depth >= BVH_MAX_DEPTH){
        sb->part = *b;                                    // same parameters and refs...
        sb->part.nodes = NULL;                            // ...but its own nodes
        sb->part.nnodes = sb->part.cap = sb->part.leaves = sb->part.depth = 0;
        build(&sb->part, bi, start, count, depth);
        return sb;
    }
    Aabb box;
    int nleft = choose_split(b, bi, start, count, depth, &box);
    if (nleft == 0){                                      // (a huge leaf: only with odd parameters)
        sb->part = *b;
        sb->part.nodes = NULL;
        sb->part.nnodes = sb->part.cap = sb->part.leaves = sb->part.depth = 0;
        int n = new_node(&sb->part);
        sb->part.nodes[n].box = box;
        sb->part.nodes[n].start = start;
        sb->part.nodes[n].count = count;
        sb->part.leaves = 1;
        sb->part.depth = depth;
        return sb;
    }
    sb->big = 1;
    sb->node.box = box;
#ifdef _OPENMP
#pragma omp task shared(sb)
#endif
    sb->left = build_par(b, bi, start, nleft, depth + 1);
#ifdef _OPENMP
#pragma omp task shared(sb)
#endif
    sb->right = build_par(b, bi, start + nleft, count - nleft, depth + 1);
#ifdef _OPENMP
#pragma omp taskwait
#endif
    return sb;
}

// copies the subtree into b->nodes in preorder; returns the index of its root
static int flatten(Bvh *b, Sub *sb, int depth){
    int root;
    if (sb->big){
        root = new_node(b);
        b->nodes[root] = sb->node;
        if (depth > b->depth) b->depth = depth;
        int l = flatten(b, sb->left, depth + 1);
        int r = flatten(b, sb->right, depth + 1);
        b->nodes[root].left = l;
        b->nodes[root].right = r;
        b->nodes[root].count = 0;
    } else {
        // a serial part: its nodes are already in preorder with local indices
        root = b->nnodes;
        for (int i = 0; i < sb->part.nnodes; i++){
            int k = new_node(b);
            b->nodes[k] = sb->part.nodes[i];
            if (b->nodes[k].count == 0){                  // inner node: shift its children
                b->nodes[k].left += root;
                b->nodes[k].right += root;
            }
        }
        b->leaves += sb->part.leaves;
        if (sb->part.depth > b->depth) b->depth = sb->part.depth;
        free(sb->part.nodes);
    }
    free(sb);
    return root;
}

static void build_over(Bvh *b, const Scene *s, const Aabb *boxes, int nprims, int max_leaf, int bins, BvhSplit split){
    memset(b, 0, sizeof(*b));
    b->max_leaf = max_leaf < 1 ? 1 : max_leaf;
    b->bins = bins;
    b->split = split;
    if (nprims <= 0) return;

    BuildInfo bi;
    bi.box = malloc((size_t)nprims * sizeof(Aabb));
    bi.centroid = malloc((size_t)nprims * sizeof(Vec3));
    b->refs = malloc((size_t)nprims * sizeof(int));
    if (!bi.box || !bi.centroid || !b->refs){ perror("malloc"); exit(1); }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nprims; i++){
        bi.box[i] = boxes ? boxes[i] : scene_shape_bounds(s, i);
        bi.centroid[i] = v_scale(v_add(bi.box[i].lo, bi.box[i].hi), 0.5);
        b->refs[i] = i;
    }
    b->nrefs = nprims;

    if (nprims <= BVH_PAR_MIN){
        build(b, &bi, 0, nprims, 0);                       // small: no threads needed
    } else {
        Sub *root = NULL;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
#endif
        root = build_par(b, &bi, 0, nprims, 0);
        flatten(b, root, 0);
    }
    free(bi.box);
    free(bi.centroid);
}

void bvh_build(Bvh *b, const Scene *s, int max_leaf, int bins, BvhSplit split){
    build_over(b, s, NULL, scene_num_shapes(s), max_leaf, bins, split);
}

void bvh_build_boxes(Bvh *b, const Aabb *boxes, int n, int max_leaf, int bins, BvhSplit split){
    build_over(b, NULL, boxes, n, max_leaf, bins, split);
}

void bvh_free(Bvh *b){
    free(b->nodes);
    free(b->refs);
    free(b->wide);
    memset(b, 0, sizeof(*b));
}

// ---- refit ----------------------------------------------------------------------------

real bvh_cost(const Bvh *b){
    if (b->nnodes == 0) return 0;
    real root = box_area(&b->nodes[0].box), sum = 0;
    for (int i = 0; i < b->nnodes; i++) sum += box_area(&b->nodes[i].box);
    return root > 0 ? sum / root : 0;
}

real bvh_refit_boxes(Bvh *b, const Aabb *boxes){
    // preorder: a node's children come after it, so from the last node back every child is
    // done before its parent
    for (int i = b->nnodes - 1; i >= 0; i--){
        BvhNode *n = &b->nodes[i];
        Aabb box = box_empty();
        if (n->count > 0) for (int k = 0; k < n->count; k++) box_add_box(&box, &boxes[b->refs[n->start + k]]);
        else { box = b->nodes[n->left].box; box_add_box(&box, &b->nodes[n->right].box); }
        n->box = box;
    }
    if (b->wide) bvh_make_wide(b);
    return bvh_cost(b);
}

// ---- the 4-wide tree -------------------------------------------------------------------

static int new_wide(Bvh *b){
    if (b->nwide >= b->wide_cap){
        b->wide_cap = b->wide_cap ? b->wide_cap * 2 : 256;
        b->wide = realloc(b->wide, (size_t)b->wide_cap * sizeof(BvhWideNode));
        if (!b->wide){ perror("realloc"); exit(1); }
    }
    return b->nwide++;
}

// the wide node for binary node `node` (an inner node, or a leaf when it is the root)
static int collapse(Bvh *b, int node){
    int w = new_wide(b);
    int kids[4], n = 0;
    const BvhNode *bn = &b->nodes[node];
    if (bn->count > 0) kids[n++] = node;                  // a tree that is one leaf
    else { kids[n++] = bn->left; kids[n++] = bn->right; }
    while (n < 4){
        int best = -1;
        real area = -1;
        for (int k = 0; k < n; k++){
            const BvhNode *c = &b->nodes[kids[k]];
            real a = box_area(&c->box);
            if (c->count == 0 && a > area){ area = a; best = k; }
        }
        if (best < 0) break;                              // only leaves left
        const BvhNode *c = &b->nodes[kids[best]];
        kids[best] = c->left;
        kids[n++] = c->right;
    }
    int child[4], count[4];
    for (int k = 0; k < 4; k++){
        if (k >= n){ child[k] = 0; count[k] = -1; continue; }
        const BvhNode *c = &b->nodes[kids[k]];
        if (c->count > 0){ child[k] = c->start; count[k] = c->count; }
        else { child[k] = collapse(b, kids[k]); count[k] = 0; }      // (may move b->wide)
    }
    BvhWideNode *o = &b->wide[w];
    for (int k = 0; k < 4; k++){
        Aabb box = k < n ? b->nodes[kids[k]].box : box_empty();
        o->b[0][k] = box.lo.x; o->b[1][k] = box.lo.y; o->b[2][k] = box.lo.z;
        o->b[3][k] = box.hi.x; o->b[4][k] = box.hi.y; o->b[5][k] = box.hi.z;
        o->child[k] = child[k];
        o->count[k] = count[k];
    }
    return w;
}

void bvh_make_wide(Bvh *b){
    free(b->wide);
    b->wide = NULL;
    b->nwide = b->wide_cap = 0;
    if (b->nnodes > 0) collapse(b, 0);
}

// a ray against the wide tree: the 4 boxes of a node at once, the hit children visited
// nearest first (leaves are entries of the stack too, so they wait their turn as well)
static int wide_intersect(const Bvh *b, const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h, int any_hit){
    const real ox = ray->o.x, oy = ray->o.y, oz = ray->o.z;
    const real ix = 1.0 / ray->d.x, iy = 1.0 / ray->d.y, iz = 1.0 / ray->d.z;
    // per axis, which of a box's planes the ray meets first
    const int nx = ix < 0 ? 3 : 0, ny = iy < 0 ? 4 : 1, nz = iz < 0 ? 5 : 2;
    const int fx = 3 - nx, fy = 5 - ny, fz = 7 - nz;
    struct { int node, count; real t; } stack[4 * BVH_MAX_DEPTH + 8];
    int sp = 0, found = 0;
    real best = tmax;
    stack[sp].node = 0; stack[sp].count = 0; stack[sp].t = tmin; sp++;
    while (sp > 0){
        sp--;
        if (stack[sp].t > best) continue;
        if (stack[sp].count > 0){                          // a leaf
            const int start = stack[sp].node, count = stack[sp].count;
            for (int k = 0; k < count; k++){
                int id = b->refs[start + k];
                Hit tmp;
                if (intersect_shape(s, id, ray, tmin, best, &tmp)){
                    tmp.prim = id;
                    *h = tmp;
                    best = tmp.t;
                    found = 1;
                    if (any_hit) return 1;
                }
            }
            continue;
        }
        const BvhWideNode *n = &b->wide[stack[sp].node];
        real tn[4];
        int hit[4];
        for (int k = 0; k < 4; k++){                       // (vectorized: 4 lanes)
            real a = (n->b[nx][k] - ox) * ix, c = (n->b[fx][k] - ox) * ix;
            real lo = a > tmin ? a : tmin, hi = c < best ? c : best;
            a = (n->b[ny][k] - oy) * iy; c = (n->b[fy][k] - oy) * iy;
            lo = a > lo ? a : lo; hi = c < hi ? c : hi;
            a = (n->b[nz][k] - oz) * iz; c = (n->b[fz][k] - oz) * iz;
            lo = a > lo ? a : lo; hi = c < hi ? c : hi;
            tn[k] = lo;
            hit[k] = lo <= hi;
        }
        // the hit children, farthest first onto the stack (so the nearest comes off first)
        int order[4], m = 0;
        for (int k = 0; k < 4; k++){
            if (!hit[k] || n->count[k] < 0) continue;
            int j = m++;
            while (j > 0 && tn[order[j - 1]] < tn[k]){ order[j] = order[j - 1]; j--; }
            order[j] = k;
        }
        for (int q = 0; q < m; q++){
            int k = order[q];
            stack[sp].node = n->child[k];
            stack[sp].count = n->count[k];
            stack[sp].t = tn[k];
            sp++;
        }
    }
    return found;
}

// ---- traversal ----------------------------------------------------------------------

int bvh_intersect(const Bvh *b, const Scene *s, const Ray *ray,
                  real tmin, real tmax, Hit *h, int any_hit){
    if (b->nnodes == 0) return 0;
    if (b->wide) return wide_intersect(b, s, ray, tmin, tmax, h, any_hit);
    Vec3 inv_d = vec3(1.0 / ray->d.x, 1.0 / ray->d.y, 1.0 / ray->d.z);

    // explicit stack of (node, distance at which the ray enters its box)
    struct { int node; real t; } stack[2 * BVH_MAX_DEPTH + 8];
    int sp = 0;
    real best = tmax, te;
    int found = 0;

    if (!aabb_hit(&b->nodes[0].box, ray, inv_d, tmin, best, &te)) return 0;
    stack[sp].node = 0; stack[sp].t = te; sp++;

    while (sp > 0){
        sp--;
        if (stack[sp].t > best) continue;          // a closer hit was found meanwhile
        const BvhNode *n = &b->nodes[stack[sp].node];

        if (n->count > 0){                         // leaf
            for (int k = 0; k < n->count; k++){
                int id = b->refs[n->start + k];
                Hit tmp;
                if (intersect_shape(s, id, ray, tmin, best, &tmp)){
                    tmp.prim = id;
                    *h = tmp;
                    best = tmp.t;
                    found = 1;
                    if (any_hit) return 1;
                }
            }
            continue;
        }

        // Inner node: test both children, push the FARTHER first so the nearer one is
        // popped (visited) first; that finds close hits early and prunes more.
        real tl, tr;
        int hl = aabb_hit(&b->nodes[n->left].box, ray, inv_d, tmin, best, &tl);
        int hr = aabb_hit(&b->nodes[n->right].box, ray, inv_d, tmin, best, &tr);
        if (hl && hr){
            int near = tl <= tr ? n->left : n->right, far = tl <= tr ? n->right : n->left;
            real tnear = r_fmin(tl, tr), tfar = r_fmax(tl, tr);
            stack[sp].node = far;  stack[sp].t = tfar;  sp++;
            stack[sp].node = near; stack[sp].t = tnear; sp++;
        } else if (hl){
            stack[sp].node = n->left;  stack[sp].t = tl; sp++;
        } else if (hr){
            stack[sp].node = n->right; stack[sp].t = tr; sp++;
        }
    }
    return found;
}
