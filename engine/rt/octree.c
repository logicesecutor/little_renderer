#include "rt/octree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- building -----------------------------------------------------------------------

static void *grow(void *p, int *cap, int need, size_t elem){
    if (need <= *cap) return p;
    int c = *cap ? *cap : 64;
    while (c < need) c *= 2;
    p = realloc(p, (size_t)c * elem);
    if (!p){ perror("realloc"); exit(1); }
    *cap = c;
    return p;
}

static int overlap(const Aabb *a, const Aabb *b){
    return a->lo.x <= b->hi.x && a->hi.x >= b->lo.x &&
           a->lo.y <= b->hi.y && a->hi.y >= b->lo.y &&
           a->lo.z <= b->hi.z && a->hi.z >= b->lo.z;
}

// child c of box b: bit 0 of c picks the x half, bit 1 the y half, bit 2 the z half
static Aabb octant(const Aabb *b, int c){
    Vec3 m = v_scale(v_add(b->lo, b->hi), 0.5);
    Aabb r;
    r.lo.x = (c & 1) ? m.x : b->lo.x;  r.hi.x = (c & 1) ? b->hi.x : m.x;
    r.lo.y = (c & 2) ? m.y : b->lo.y;  r.hi.y = (c & 2) ? b->hi.y : m.y;
    r.lo.z = (c & 4) ? m.z : b->lo.z;  r.hi.z = (c & 4) ? b->hi.z : m.z;
    return r;
}

static void make_leaf(Octree *o, int node, const int *ids, int n){
    o->refs = grow(o->refs, &o->cap_refs, o->nrefs + n, sizeof(int));
    o->nodes[node].first_child = -1;
    o->nodes[node].prim_start = o->nrefs;
    o->nodes[node].prim_count = n;
    memcpy(o->refs + o->nrefs, ids, (size_t)n * sizeof(int));
    o->nrefs += n;
    o->leaves++;
}

// pb = bounds of every primitive, ids/n = the primitives overlapping this node
static void build(Octree *o, const Aabb *pb, int node, const int *ids, int n, int depth){
    if (depth > o->depth) o->depth = depth;
    if (n <= OCT_LEAF_SIZE || depth >= o->max_depth || o->nnodes + 8 > OCT_MAX_NODES){
        make_leaf(o, node, ids, n);
        return;
    }

    // Hand every primitive to each child cell its box overlaps. A primitive that crosses a
    // split plane is referenced by several children (a wall, for instance, by many).
    Aabb box = o->nodes[node].box;
    Aabb cbox[8];
    int *cid[8], cn[8], useful = 0;
    for (int c = 0; c < 8; c++){
        cbox[c] = octant(&box, c);
        cid[c] = malloc((size_t)n * sizeof(int));
        if (!cid[c]){ perror("malloc"); exit(1); }
        cn[c] = 0;
        for (int i = 0; i < n; i++)
            if (overlap(&pb[ids[i]], &cbox[c])) cid[c][cn[c]++] = ids[i];
        if (cn[c] < n) useful = 1;
    }

    // If every child would still get every primitive, splitting only costs memory.
    if (!useful){
        for (int c = 0; c < 8; c++) free(cid[c]);
        make_leaf(o, node, ids, n);
        return;
    }

    int first = o->nnodes;
    o->nodes = grow(o->nodes, &o->cap_nodes, o->nnodes + 8, sizeof(OctNode));
    o->nnodes += 8;
    o->nodes[node].first_child = first;   // (index, not pointer: grow may move the array)
    o->nodes[node].prim_start = 0;
    o->nodes[node].prim_count = 0;

    for (int c = 0; c < 8; c++){
        o->nodes[first + c].box = cbox[c];
        o->nodes[first + c].first_child = -1;
        build(o, pb, first + c, cid[c], cn[c], depth + 1);
        free(cid[c]);
    }
}

int octree_auto_depth(int nprims){
    double leaves = (double)nprims / OCT_LEAF_SIZE;
    int d = leaves > 1 ? (int)ceil(log(leaves) / log(4.0)) + 2 : 6;
    if (d < 6) d = 6;
    if (d > OCT_DEPTH_LIMIT) d = OCT_DEPTH_LIMIT;
    return d;
}

void octree_build(Octree *o, const Scene *s, int max_depth){
    int nprims = scene_num_shapes(s);
    memset(o, 0, sizeof(*o));
    if (nprims <= 0) return;
    if (max_depth <= 0) max_depth = octree_auto_depth(nprims);
    if (max_depth > OCT_DEPTH_LIMIT) max_depth = OCT_DEPTH_LIMIT;
    o->max_depth = max_depth;

    Aabb *pb = malloc((size_t)nprims * sizeof(Aabb));
    int *ids = malloc((size_t)nprims * sizeof(int));
    if (!pb || !ids){ perror("malloc"); exit(1); }

    Aabb root = scene_shape_bounds(s, 0);
    for (int i = 0; i < nprims; i++){
        pb[i] = scene_shape_bounds(s, i);
        ids[i] = i;
        root.lo = vec3(r_fmin(root.lo.x, pb[i].lo.x), r_fmin(root.lo.y, pb[i].lo.y), r_fmin(root.lo.z, pb[i].lo.z));
        root.hi = vec3(r_fmax(root.hi.x, pb[i].hi.x), r_fmax(root.hi.y, pb[i].hi.y), r_fmax(root.hi.z, pb[i].hi.z));
    }

    o->nodes = grow(o->nodes, &o->cap_nodes, 1, sizeof(OctNode));
    o->nnodes = 1;
    o->nodes[0].box = root;
    o->nodes[0].first_child = -1;
    build(o, pb, 0, ids, nprims, 0);

    free(pb);
    free(ids);
}

void octree_free(Octree *o){
    free(o->nodes);
    free(o->refs);
    memset(o, 0, sizeof(*o));
}

// ---- traversal ----------------------------------------------------------------------

// (the ray-box slab test, aabb_hit, lives in rt/intersect.h: the BVH uses it too)

// Mailboxing. A primitive that crosses cell borders is referenced by several leaves (5 on
// average on the dragon), so one ray could test it several times. The mailbox remembers the
// last primitives this ray tested, in a small direct-mapped table (slot = id mod 64), and
// skips a repeat. Skipping is safe: the earlier test of the same primitive used a farther or
// equal tmax, so a repeat could only find the same hit again, never a closer one. If two ids
// share a slot the older one is forgotten and simply tested again.
#define OCT_MAILBOX 64

typedef struct {
    const Octree *o;
    const Scene *s;
    const Ray *ray;
    Vec3 inv_d;
    real tmin;
    real best;     // closest hit so far (starts at tmax)
    Hit *hit;
    int found, any_hit;
    int mailbox[OCT_MAILBOX];   // ids tested by this ray (-1 = empty)
} Traversal;

static void visit(Traversal *tr, int node){
    const OctNode *nd = &tr->o->nodes[node];

    if (nd->first_child < 0){                      // leaf: test its primitives
        for (int k = 0; k < nd->prim_count; k++){
            int id = tr->o->refs[nd->prim_start + k];
            int *slot = &tr->mailbox[id & (OCT_MAILBOX - 1)];
#ifndef OCT_NO_MAILBOX
            if (*slot == id) continue;               // already tested by this ray
#endif
            *slot = id;
            Hit tmp;
            if (intersect_shape(tr->s, id, tr->ray, tr->tmin, tr->best, &tmp)){
                tmp.prim = id;
                *tr->hit = tmp;
                tr->best = tmp.t;
                tr->found = 1;
                if (tr->any_hit) return;
            }
        }
        return;
    }

    // Collect the children the ray enters, sorted near -> far by entry distance.
    int idx[8];
    real tent[8];
    int m = 0;
    for (int c = 0; c < 8; c++){
        int child = nd->first_child + c;
        const OctNode *cn = &tr->o->nodes[child];
        if (cn->first_child < 0 && cn->prim_count == 0) continue;   // empty cell
        real te;
        if (!aabb_hit(&cn->box, tr->ray, tr->inv_d, tr->tmin, tr->best, &te)) continue;
        int j = m++;
        while (j > 0 && tent[j - 1] > te){ tent[j] = tent[j - 1]; idx[j] = idx[j - 1]; j--; }
        tent[j] = te;
        idx[j] = child;
    }

    for (int i = 0; i < m; i++){
        // Every remaining child starts beyond the best hit: nothing in it can be closer.
        if (tent[i] > tr->best) break;
        visit(tr, idx[i]);
        if (tr->any_hit && tr->found) return;
    }
}

int octree_intersect(const Octree *o, const Scene *s, const Ray *ray,
                     real tmin, real tmax, Hit *h, int any_hit){
    if (o->nnodes == 0) return 0;
    Traversal tr;
    tr.o = o;
    tr.s = s;
    tr.ray = ray;
    tr.inv_d = vec3(1.0 / ray->d.x, 1.0 / ray->d.y, 1.0 / ray->d.z);
    tr.tmin = tmin;
    tr.best = tmax;
    tr.hit = h;
    tr.found = 0;
    tr.any_hit = any_hit;
#ifndef OCT_NO_MAILBOX
    memset(tr.mailbox, 0xff, sizeof(tr.mailbox));   // all -1
#endif

    real te;
    if (!aabb_hit(&o->nodes[0].box, ray, tr.inv_d, tmin, tmax, &te)) return 0;
    visit(&tr, 0);
    return tr.found;
}
