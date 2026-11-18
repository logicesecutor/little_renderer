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

// room for `need` nodes in b->nodes (a tree of n primitives has at most 2n - 1: the
// builders reserve that once, so new_node does not grow the array node by node)
static void reserve_nodes(Bvh *b, int need){
    if (need <= b->cap) return;
    int cap = b->cap ? 2 * b->cap : need;     // (the first time: exactly what was asked)
    if (cap < need) cap = need;
    b->nodes = realloc(b->nodes, (size_t)cap * sizeof(BvhNode));
    if (!b->nodes){ perror("realloc"); exit(1); }
    b->cap = cap;
}

static int new_node(Bvh *b){
    if (b->nnodes >= b->cap) reserve_nodes(b, b->nnodes + 1);
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

// partition, and the two sides' bounds on the way (kids: left box, left centroid box, right
// box, right centroid box): each primitive is added once, when its side is final, so the
// children need no pass of their own for their bounds (smallest and largest do not depend on
// the order: the same boxes as that pass)
static int partition_bounds(int *refs, int start, int count, const BuildInfo *bi, int axis, real split_pos, Aabb kids[4]){
    Aabb lb = box_empty(), lc = box_empty(), rb = box_empty(), rc = box_empty();
    int i = start, j = start + count - 1;
    while (i <= j){
        const int p = refs[i];
        if (comp(bi->centroid[p], axis) < split_pos){
            box_add_box(&lb, &bi->box[p]); box_add_point(&lc, bi->centroid[p]);
            i++;
        } else {
            refs[i] = refs[j]; refs[j] = p; j--;          // p is now right for good
            box_add_box(&rb, &bi->box[p]); box_add_point(&rc, bi->centroid[p]);
        }
    }
    kids[0] = lb; kids[1] = lc; kids[2] = rb; kids[3] = rc;
    return i - start;
}

// ---- a node's primitives, in chunks on several threads ------------------------------------
//
// The top of a big tree is a few nodes with very many primitives each: their bounds and
// their bins are a pass over all of them, which one thread did while the others waited
// (the full dragon: the thread that builds busy all the time, the others a third of it).
// Inside the parallel build those passes are cut into chunks, each a task; the chunks'
// boxes and counts are then put together, which gives what one pass gives (smallest,
// largest and sums do not depend on the order).

enum { MAX_BINS = 256 };
#define BVH_CHUNK 4096           // primitives per chunk
#define BVH_CHUNKS_MAX 64

typedef struct { int n[3][MAX_BINS]; Aabb box[3][MAX_BINS]; } Bins;
typedef struct { int nb, use[3]; real lo[3], scale[3]; } BinAxes;

static void range_bounds(const Bvh *b, const BuildInfo *bi, int start, int count, Aabb *box_out, Aabb *cbox_out){
    Aabb box = box_empty(), cbox = box_empty();
    for (int i = start; i < start + count; i++){
        box_add_box(&box, &bi->box[b->refs[i]]);
        box_add_point(&cbox, bi->centroid[b->refs[i]]);
    }
    *box_out = box;
    *cbox_out = cbox;
}

// The bins of the three axes filled in one pass over the primitives (each one's centre and
// box are somewhere else in memory: read once, not once per axis)
static void range_bins(const Bvh *b, const BuildInfo *bi, int start, int count, const BinAxes *ax, Bins *B){
    for (int axis = 0; axis < 3; axis++)
        for (int k = 0; k < ax->nb; k++){ B->n[axis][k] = 0; B->box[axis][k] = box_empty(); }
    for (int i = start; i < start + count; i++){
        int p = b->refs[i];
        const Aabb *pb = &bi->box[p];
        const real c[3] = {bi->centroid[p].x, bi->centroid[p].y, bi->centroid[p].z};
        for (int axis = 0; axis < 3; axis++){
            if (!ax->use[axis]) continue;
            real f = (c[axis] - ax->lo[axis]) * ax->scale[axis];
            int k = f > 0 ? (f < ax->nb ? (int)f : ax->nb - 1) : 0;   // (a NaN box, a body gone wild: bin 0, not a crash)
            B->n[axis][k]++;
            box_add_box(&B->box[axis][k], pb);
        }
    }
}

// how many chunks a node of `count` primitives is cut into (1: one pass, here)
static int chunks_of(int count, int par){
    if (!par || count < 2 * BVH_CHUNK) return 1;
    int k = count / BVH_CHUNK;
    return k > BVH_CHUNKS_MAX ? BVH_CHUNKS_MAX : k;
}

// Binned SAH. The expected cost of a node is
//   cost = C_trav + (A_left / A_node) * N_left * C_isect + (A_right / A_node) * N_right * C_isect
// because a random ray that hits the node hits a child with probability ~ (child area /
// node area). We try `bins - 1` split planes per axis, using buckets of centroids, and keep
// the cheapest. Returns the split position (or a NaN-free "no split" via *ok = 0).
// par: inside the parallel build (the bins of a big node in chunks, as tasks).
static real sah_split(const Bvh *b, const BuildInfo *bi, int start, int count,
                        const Aabb *node_box, const Aabb *cbox, int *best_axis, real *best_cost, int par){
    int nb = b->bins < 2 ? 2 : (b->bins > MAX_BINS ? MAX_BINS : b->bins);
    real parent_area = box_area(node_box);
    real best_pos = 0;
    *best_cost = REAL_MAX;
    *best_axis = -1;

    BinAxes ax;
    ax.nb = nb;
    for (int axis = 0; axis < 3; axis++){
        real lo = comp(cbox->lo, axis), extent = comp(cbox->hi, axis) - lo;
        ax.use[axis] = extent > 1e-12;      // (else all centroids in one plane: nothing to split here)
        ax.lo[axis] = lo;
        ax.scale[axis] = ax.use[axis] ? nb / extent : 0;
    }
    Bins all;
    const int nch = chunks_of(count, par);
    Bins *part = nch > 1 ? malloc((size_t)nch * sizeof(Bins)) : NULL;
    if (part){
        for (int c = 0; c < nch; c++){
            const int s0 = start + (int)((long long)count * c / nch), s1 = start + (int)((long long)count * (c + 1) / nch);
            Bins *to = &part[c];
#ifdef _OPENMP
#pragma omp task firstprivate(to, s0, s1, ax)
#endif
            range_bins(b, bi, s0, s1 - s0, &ax, to);
        }
#ifdef _OPENMP
#pragma omp taskwait
#endif
        for (int axis = 0; axis < 3; axis++)
            for (int k = 0; k < nb; k++){
                all.n[axis][k] = 0;
                all.box[axis][k] = box_empty();
                for (int c = 0; c < nch; c++){ all.n[axis][k] += part[c].n[axis][k]; box_add_box(&all.box[axis][k], &part[c].box[axis][k]); }
            }
        free(part);
    } else range_bins(b, bi, start, count, &ax, &all);
    const int *use = ax.use;
    const real *los = ax.lo;
    int (*bins_n)[MAX_BINS] = all.n;
    Aabb (*bins_box)[MAX_BINS] = all.box;

    for (int axis = 0; axis < 3; axis++){
        if (!use[axis]) continue;
        const real lo = los[axis], extent = comp(cbox->hi, axis) - lo;
        const int *bin_n = bins_n[axis];
        const Aabb *bin_box = bins_box[axis];

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
// have: the node's box and centroid box when its parent already knows them (NULL: a pass
// here); kids (not NULL): filled with the children's (as partition_bounds) when it splits.
static int choose_split(const Bvh *b, const BuildInfo *bi, int start, int count, int depth, Aabb *box_out, int par,
                        const Aabb *have, Aabb *kids){
    Aabb box = box_empty(), cbox = box_empty();
    const int nch = have ? 1 : chunks_of(count, par);
    if (have){ box = have[0]; cbox = have[1]; }
    else if (nch > 1){
        Aabb pb[BVH_CHUNKS_MAX], pc[BVH_CHUNKS_MAX];
        for (int c = 0; c < nch; c++){
            const int s0 = start + (int)((long long)count * c / nch), s1 = start + (int)((long long)count * (c + 1) / nch);
            Aabb *tb = &pb[c], *tc = &pc[c];
#ifdef _OPENMP
#pragma omp task firstprivate(tb, tc, s0, s1)
#endif
            range_bounds(b, bi, s0, s1 - s0, tb, tc);
        }
#ifdef _OPENMP
#pragma omp taskwait
#endif
        for (int c = 0; c < nch; c++){ box_add_box(&box, &pb[c]); box_add_box(&cbox, &pc[c]); }
    } else range_bounds(b, bi, start, count, &box, &cbox);
    *box_out = box;

    int nleft = 0, kids_done = 0;
    if (count > 1 && depth < BVH_MAX_DEPTH){
        int axis = -1;
        real pos = 0;
        if (b->split == BVH_SPLIT_SAH){
            real cost;
            pos = sah_split(b, bi, start, count, &box, &cbox, &axis, &cost, par);
            // Leaf if not splitting is cheaper (cost of a leaf = count tests), unless the
            // leaf would be too big.
            if (!(axis >= 0 && (cost < count || count > b->max_leaf))) axis = -1;
        } else if (count > b->max_leaf){
            // middle split: longest axis of the centroid box, cut at its center
            Vec3 ext = v_sub(cbox.hi, cbox.lo);
            axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
            pos = 0.5 * (comp(cbox.lo, axis) + comp(cbox.hi, axis));
            if (!(comp(ext, axis) > 1e-12)) axis = -1;
        }
        if (axis >= 0){
            if (kids){ nleft = partition_bounds(b->refs, start, count, bi, axis, pos, kids); kids_done = 1; }
            else nleft = partition(b->refs, start, count, bi, axis, pos);
        }
        // Everything ended up on one side (e.g. identical centroids) but the node is too
        // big: split the list in half, in whatever order it is. Still a valid tree.
        if ((nleft == 0 || nleft == count) && count > b->max_leaf){ nleft = count / 2; kids_done = 0; }
        if (nleft == count) nleft = 0;
    }
    if (kids && nleft > 0 && !kids_done){                 // (the halves: a pass each, rare)
        range_bounds(b, bi, start, nleft, &kids[0], &kids[1]);
        range_bounds(b, bi, start + nleft, count - nleft, &kids[2], &kids[3]);
    }
    return nleft;
}

// Serial build of a subtree into b->nodes, in preorder (a node, then its whole left
// subtree, then its right subtree). Returns the node's index.
// have: the node's box and centroid box, from its parent's partition (NULL at the top).
static int build(Bvh *b, const BuildInfo *bi, int start, int count, int depth, const Aabb *have){
    int node = new_node(b);
    if (depth > b->depth) b->depth = depth;
    Aabb box, kids[4];
    int nleft = choose_split(b, bi, start, count, depth, &box, 0, have, kids);
    b->nodes[node].box = box;

    if (nleft == 0){                                      // leaf
        b->nodes[node].start = start;
        b->nodes[node].count = count;
        b->leaves++;
        return node;
    }
    int l = build(b, bi, start, nleft, depth + 1, kids);          // (indices, not pointers:
    int r = build(b, bi, start + nleft, count - nleft, depth + 1, kids + 2);   // the array may move)
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
// The subtrees are small (128) so the threads stay busy to the end (1024 left them waiting
// at the joins: E1 12.7 -> 9.7 ms with 128, with each subtree's nodes reserved at once; 64 was
// no faster). A whole tree of up to 128 primitives is built on one thread; above that the
// parallel build was faster even for a few hundred (measured with E1, E2).

#define BVH_PAR_MIN 128

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
        sb->part.nodes = NULL;                            // ...but its own nodes, room for
        sb->part.nnodes = sb->part.cap = sb->part.leaves = sb->part.depth = 0;   // all at once
        reserve_nodes(&sb->part, 2 * count - 1);
        build(&sb->part, bi, start, count, depth, NULL);
        return sb;
    }
    Aabb box;
    int nleft = choose_split(b, bi, start, count, depth, &box, 1, NULL, NULL);
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
        const int n = sb->part.nnodes;
        reserve_nodes(b, root + n);
        memcpy(&b->nodes[root], sb->part.nodes, (size_t)n * sizeof(BvhNode));
        for (int k = root; k < root + n; k++)
            if (b->nodes[k].count == 0){                  // inner node: shift its children
                b->nodes[k].left += root;
                b->nodes[k].right += root;
            }
        b->nnodes += n;
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

    reserve_nodes(b, 2 * nprims - 1);                     // the most a tree of them has
    if (nprims <= BVH_PAR_MIN){
        build(b, &bi, 0, nprims, 0, NULL);                 // small: no threads needed
    } else {
        Sub *root = NULL;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
#endif
        root = build_par(b, &bi, 0, nprims, 0);
        flatten(b, root, 0);
    }
    if (b->nnodes < b->cap / 2){                          // (leaves of several: give back the rest)
        BvhNode *fit = realloc(b->nodes, (size_t)b->nnodes * sizeof(BvhNode));
        if (fit){ b->nodes = fit; b->cap = b->nnodes; }
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
    int hit_id = -1;                       // the shape kept, its Hit made once at the end
    Vec2 uv = {0, 0}, kept = {0, 0};
    real best = tmax;
    stack[sp].node = 0; stack[sp].count = 0; stack[sp].t = tmin; sp++;
    while (sp > 0){
        sp--;
        if (stack[sp].t > best) continue;
        if (stack[sp].count > 0){                          // a leaf
            const int start = stack[sp].node, count = stack[sp].count;
            for (int k = 0; k < count; k++){
                int id = b->refs[start + k];
                real t;
                if (shape_test(s, id, ray, tmin, best, &t, &uv)){
                    best = t;
                    hit_id = id;
                    kept = uv;
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
    if (found) shape_hit(s, hit_id, ray, best, kept, h);
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
    int hit_id = -1;                       // the shape kept, its Hit made once at the end
    Vec2 uv = {0, 0}, kept = {0, 0};

    if (!aabb_hit(&b->nodes[0].box, ray, inv_d, tmin, best, &te)) return 0;
    stack[sp].node = 0; stack[sp].t = te; sp++;

    while (sp > 0){
        sp--;
        if (stack[sp].t > best) continue;          // a closer hit was found meanwhile
        const BvhNode *n = &b->nodes[stack[sp].node];

        if (n->count > 0){                         // leaf
            for (int k = 0; k < n->count; k++){
                int id = b->refs[n->start + k];
                real t;
                if (shape_test(s, id, ray, tmin, best, &t, &uv)){
                    best = t;
                    hit_id = id;
                    kept = uv;
                    found = 1;
                    if (any_hit) return 1;
                }
            }
            continue;
        }

        // Inner node: test both children, push the FARTHER first so the nearer one is
        // popped (visited) first; that finds close hits early and prunes more.
        real tl = 0, tr = 0;                       // (set by aabb_hit when it hits; gcc 13 cannot tell)
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
    if (found) shape_hit(s, hit_id, ray, best, kept, h);
    return found;
}
