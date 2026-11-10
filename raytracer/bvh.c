#include "bvh.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- small box helpers ----------------------------------------------------------------

static Aabb box_empty(void){
    Aabb b = {vec3(DBL_MAX, DBL_MAX, DBL_MAX), vec3(-DBL_MAX, -DBL_MAX, -DBL_MAX)};
    return b;
}

static void box_add_box(Aabb *b, const Aabb *o){
    b->lo = vec3(fmin(b->lo.x, o->lo.x), fmin(b->lo.y, o->lo.y), fmin(b->lo.z, o->lo.z));
    b->hi = vec3(fmax(b->hi.x, o->hi.x), fmax(b->hi.y, o->hi.y), fmax(b->hi.z, o->hi.z));
}

static void box_add_point(Aabb *b, Vec3 p){
    b->lo = vec3(fmin(b->lo.x, p.x), fmin(b->lo.y, p.y), fmin(b->lo.z, p.z));
    b->hi = vec3(fmax(b->hi.x, p.x), fmax(b->hi.y, p.y), fmax(b->hi.z, p.z));
}

static double box_area(const Aabb *b){
    Vec3 d = v_sub(b->hi, b->lo);
    if (d.x < 0 || d.y < 0 || d.z < 0) return 0;   // empty box
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

static double comp(Vec3 v, int axis){ return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }

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
static int partition(int *refs, int start, int count, const BuildInfo *bi, int axis, double split_pos){
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
static double sah_split(const Bvh *b, const BuildInfo *bi, int start, int count,
                        const Aabb *node_box, const Aabb *cbox, int *best_axis, double *best_cost){
    enum { MAX_BINS = 256 };
    int nb = b->bins < 2 ? 2 : (b->bins > MAX_BINS ? MAX_BINS : b->bins);
    double parent_area = box_area(node_box);
    double best_pos = 0;
    *best_cost = DBL_MAX;
    *best_axis = -1;

    for (int axis = 0; axis < 3; axis++){
        double lo = comp(cbox->lo, axis), hi = comp(cbox->hi, axis);
        double extent = hi - lo;
        if (extent <= 1e-12) continue;   // all centroids in one plane: nothing to split here

        int bin_n[MAX_BINS] = {0};
        Aabb bin_box[MAX_BINS];
        for (int k = 0; k < nb; k++) bin_box[k] = box_empty();
        for (int i = start; i < start + count; i++){
            int p = b->refs[i];
            int k = (int)(nb * (comp(bi->centroid[p], axis) - lo) / extent);
            if (k >= nb) k = nb - 1;
            bin_n[k]++;
            box_add_box(&bin_box[k], &bi->box[p]);
        }

        // sweep from the right to get the area/count of everything right of each plane
        double right_area[MAX_BINS];
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
            double cost = 1.0 + (box_area(&acc) * n + right_area[k] * right_n[k]) / parent_area;
            if (cost < *best_cost){
                *best_cost = cost;
                *best_axis = axis;
                best_pos = lo + extent * k / nb;
            }
        }
    }
    return best_pos;
}

static int build(Bvh *b, const BuildInfo *bi, int start, int count, int depth){
    int node = new_node(b);
    if (depth > b->depth) b->depth = depth;

    Aabb box = box_empty(), cbox = box_empty();
    for (int i = start; i < start + count; i++){
        box_add_box(&box, &bi->box[b->refs[i]]);
        box_add_point(&cbox, bi->centroid[b->refs[i]]);
    }
    b->nodes[node].box = box;

    int nleft = 0;
    if (count > 1 && depth < BVH_MAX_DEPTH){
        if (b->split == BVH_SPLIT_SAH){
            int axis;
            double cost;
            double pos = sah_split(b, bi, start, count, &box, &cbox, &axis, &cost);
            // Leaf if not splitting is cheaper (cost of a leaf = count tests), unless the
            // leaf would be too big.
            int want_split = axis >= 0 && (cost < count || count > b->max_leaf);
            if (want_split) nleft = partition(b->refs, start, count, bi, axis, pos);
        } else if (count > b->max_leaf){
            // middle split: longest axis of the centroid box, cut at its center
            Vec3 ext = v_sub(cbox.hi, cbox.lo);
            int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
            double mid = 0.5 * (comp(cbox.lo, axis) + comp(cbox.hi, axis));
            if (comp(ext, axis) > 1e-12) nleft = partition(b->refs, start, count, bi, axis, mid);
        }
        // Everything ended up on one side (e.g. identical centroids) but the node is too
        // big: split the list in half, in whatever order it is. Still a valid tree.
        if ((nleft == 0 || nleft == count) && count > b->max_leaf) nleft = count / 2;
        if (nleft == count) nleft = 0;
    }

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

void bvh_build(Bvh *b, const Primitive *prims, int nprims, int max_leaf, int bins, BvhSplit split){
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
    for (int i = 0; i < nprims; i++){
        bi.box[i] = primitive_bounds(&prims[i]);
        bi.centroid[i] = v_scale(v_add(bi.box[i].lo, bi.box[i].hi), 0.5);
        b->refs[i] = i;
    }
    b->nrefs = nprims;
    build(b, &bi, 0, nprims, 0);
    free(bi.box);
    free(bi.centroid);
}

void bvh_free(Bvh *b){
    free(b->nodes);
    free(b->refs);
    memset(b, 0, sizeof(*b));
}

// ---- traversal ----------------------------------------------------------------------

int bvh_intersect(const Bvh *b, const Primitive *prims, const Ray *ray,
                  double tmin, double tmax, Hit *h, int any_hit){
    if (b->nnodes == 0) return 0;
    Vec3 inv_d = vec3(1.0 / ray->d.x, 1.0 / ray->d.y, 1.0 / ray->d.z);

    // explicit stack of (node, distance at which the ray enters its box)
    struct { int node; double t; } stack[2 * BVH_MAX_DEPTH + 8];
    int sp = 0;
    double best = tmax, te;
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
                if (intersect_primitive(&prims[id], ray, tmin, best, &tmp)){
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
        double tl, tr;
        int hl = aabb_hit(&b->nodes[n->left].box, ray, inv_d, tmin, best, &tl);
        int hr = aabb_hit(&b->nodes[n->right].box, ray, inv_d, tmin, best, &tr);
        if (hl && hr){
            int near = tl <= tr ? n->left : n->right, far = tl <= tr ? n->right : n->left;
            double tnear = fmin(tl, tr), tfar = fmax(tl, tr);
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
