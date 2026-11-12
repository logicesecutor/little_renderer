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
    Bvh tlas;                // over the instances' world boxes (the top level)
    unsigned tlas_version;   // Scene.inst_version it was built for
    int tlas_n;              // and the instances it holds
    real tlas_cost;          // its cost (bvh_cost) when it was built: refit until it grows
    int tlas_refits;
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
    bvh_free(&a->tlas);
    free(a);
}

AccelParams accel_default_params(void){
    AccelParams p = {0, 4, 16, BVH_SPLIT_SAH, 4};
    return p;
}

// ---- instances: each mesh's own BVH, over a "view" scene holding only its triangles ----
typedef struct { Bvh bvh; Scene view; } MeshBvh;

static void mesh_bvh_free(void *p){
    MeshBvh *m = p;
    bvh_free(&m->bvh);
    free(m);
}

static void prepare_meshes(Scene *s, const AccelParams *p){
    for (int i = 0; i < s->nmeshes; i++){
        SceneMesh *m = &s->meshes[i];
        if (m->bvh && m->bvh_version == m->version) continue;
        if (m->bvh && m->bvh_free) m->bvh_free(m->bvh);
        MeshBvh *mb = calloc(1, sizeof(*mb));
        if (!mb){ perror("calloc"); exit(1); }
        mb->view.tris = m->tris;
        mb->view.ntris = m->ntris;
        mb->view.normals = m->normals;
        mb->view.nnormals = m->nnormals;
        mb->view.light = -1;
        bvh_build(&mb->bvh, &mb->view, p->bvh_leaf, p->bvh_bins, p->bvh_split);
        if (p->bvh_width == 4) bvh_make_wide(&mb->bvh);
        m->bvh = mb;
        m->bvh_free = mesh_bvh_free;
        m->bvh_version = m->version;
    }
}

const Bvh *scene_mesh_bvh(const Scene *s, int mesh){
    const MeshBvh *mb = s->meshes[mesh].bvh;
    return mb && s->meshes[mesh].bvh_version == s->meshes[mesh].version ? &mb->bvh : NULL;
}

// the unit sphere and the unit box's 6 faces (u x v outwards), for shape instances
static Primitive unit_sphere, unit_faces[6];
static void unit_shapes_init(void){
    static int made;
    if (made) return;
    unit_sphere = make_sphere(vec3(0, 0, 0), 1, 0);
    for (int f = 0; f < 6; f++) unit_faces[f] = unit_box_face(f, 0);
    made = 1;
}

// instance i against a ray (world): the closest hit in (tmin, tmax), its normals in the world
static int intersect_instance(const Scene *s, int i, const Ray *ray, real tmin, real tmax, Hit *h, int any){
    const Instance *in = &s->insts[i];
    // the ray in the instance's space: o' = inv o, d' = inv d, the same t
    Ray r = {xform_point(&in->inv, ray->o), xform_dir(&in->inv, ray->d)};
    Hit tmp;
    int hit = 0;
    if (in->mesh >= 0){
        const SceneMesh *m = &s->meshes[in->mesh];
        const MeshBvh *mb = m->bvh;
        if (!mb || m->bvh_version != m->version) return 0;      // (scene_set_accel builds it)
        hit = bvh_intersect(&mb->bvh, &mb->view, &r, tmin, tmax, &tmp, any);
    } else if (in->mesh == INST_SPHERE){
        hit = intersect_primitive(&unit_sphere, &r, tmin, tmax, &tmp);
        tmp.prim = -1;
    } else {
        Hit f;
        for (int k = 0; k < 6; k++)
            if (intersect_primitive(&unit_faces[k], &r, tmin, tmax, &f)){ tmp = f; tmax = f.t; hit = 1; if (any) break; }
        tmp.prim = -1;
    }
    if (!hit) return 0;
    if (any) return 1;
    tmp.p = v_add(ray->o, v_scale(ray->d, tmp.t));
    tmp.n = xform_normal(&in->inv, tmp.n);
    tmp.ng = xform_normal(&in->inv, tmp.ng);
    tmp.inst = i;
    *h = tmp;
    return 1;
}

// the closest instance hit in (tmin, tmax) (any = 1: whether there is one): through the top
// level (scene_set_accel builds it), or, if instances moved since, all of them by their boxes
static int intersect_instances(const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h, int any){
    int found = 0;
    Vec3 inv_d = vec3(1.0 / ray->d.x, 1.0 / ray->d.y, 1.0 / ray->d.z);
    const RtAccel *a = s->accel;
    real te;
    if (a && a->tlas_n == s->ninsts && a->tlas_version == s->inst_version && a->tlas.nnodes > 0){
        const Bvh *b = &a->tlas;
        int stack[BVH_MAX_DEPTH + 2], sp = 0;
        stack[sp++] = 0;
        while (sp > 0){
            const BvhNode *nd = &b->nodes[stack[--sp]];
            if (!aabb_hit(&nd->box, ray, inv_d, tmin, tmax, &te)) continue;
            if (nd->count > 0){
                for (int k = 0; k < nd->count; k++){
                    int i = b->refs[nd->start + k];
                    if (nd->count > 1 && !aabb_hit(&s->insts[i].box, ray, inv_d, tmin, tmax, &te)) continue;
                    Hit tmp;
                    if (!intersect_instance(s, i, ray, tmin, tmax, &tmp, any)) continue;
                    if (any) return 1;
                    found = 1;
                    tmax = tmp.t;
                    *h = tmp;
                }
            } else if (sp + 2 <= BVH_MAX_DEPTH + 2){
                stack[sp++] = nd->left;
                stack[sp++] = nd->right;
            }
        }
        return found;
    }
    for (int i = 0; i < s->ninsts; i++){
        if (!aabb_hit(&s->insts[i].box, ray, inv_d, tmin, tmax, &te)) continue;
        Hit tmp;
        if (!intersect_instance(s, i, ray, tmin, tmax, &tmp, any)) continue;
        if (any) return 1;
        found = 1;
        tmax = tmp.t;
        *h = tmp;
    }
    return found;
}

// The top level over the instances' world boxes, built again after they moved (the
// physics: every step): binned SAH, in parallel, under 1 ms for 2500 bodies on 8 cores.
// LR_TLAS_REFIT=1 refits it instead while the same instances only moved (their boxes made
// again from the leaves up, O(n)) until its cost has grown by a tenth: measured on a pile of
// 700 falling bodies, the GPU then traced 5% slower than with a new tree each step, so the
// rebuild is the default (docs/PERFORMANCE.md).
#define TLAS_REFIT_GROWTH 1.1
static void build_tlas(RtAccel *a, const Scene *s, const AccelParams *p){
    static int refit_env = -1;
    if (refit_env < 0){ const char *e = getenv("LR_TLAS_REFIT"); refit_env = e && atoi(e) > 0; }
    int same = refit_env && a->tlas.nnodes > 0 && a->tlas_n == s->ninsts && s->ninsts > 0;
    a->tlas_n = s->ninsts;
    a->tlas_version = s->inst_version;
    if (s->ninsts <= 0){ bvh_free(&a->tlas); return; }
    Aabb *boxes = malloc((size_t)s->ninsts * sizeof(Aabb));
    if (!boxes){ bvh_free(&a->tlas); a->tlas_n = -1; return; }
    for (int i = 0; i < s->ninsts; i++) boxes[i] = s->insts[i].box;
    if (same && bvh_refit_boxes(&a->tlas, boxes) <= TLAS_REFIT_GROWTH * a->tlas_cost){
        a->tlas_refits++;
    } else {
        bvh_free(&a->tlas);
        bvh_build_boxes(&a->tlas, boxes, s->ninsts, p ? p->bvh_leaf : 4, p ? p->bvh_bins : 16, BVH_SPLIT_SAH);
        a->tlas_cost = bvh_cost(&a->tlas);
        a->tlas_refits = 0;
    }
    free(boxes);
}

void scene_set_accel(Scene *s, AccelKind kind, const AccelParams *p){
    AccelParams def = accel_default_params();
    if (!p) p = &def;
    unit_shapes_init();
    prepare_meshes(s, p);
    if (s->accel && s->accel_free) s->accel_free(s->accel);
    RtAccel *a = calloc(1, sizeof(*a));
    if (!a){ perror("calloc"); exit(1); }
    double t0 = clock_ms();
    if (kind == ACCEL_OCTREE) octree_build(&a->octree, s, p->oct_depth);
    if (kind == ACCEL_BVH){
        bvh_build(&a->bvh, s, p->bvh_leaf, p->bvh_bins, p->bvh_split);
        if (p->bvh_width == 4) bvh_make_wide(&a->bvh);
    }
    a->build_ms = clock_ms() - t0;
    a->kind = kind;
    build_tlas(a, s, p);
    s->accel = a;
    s->accel_free = rt_accel_free;
}

void scene_update_instances(Scene *s, const AccelParams *p){
    AccelParams def = accel_default_params();
    if (!p) p = &def;
    if (!s->accel || s->accel_free != rt_accel_free){ scene_set_accel(s, ACCEL_NONE, p); return; }
    prepare_meshes(s, p);
    RtAccel *a = s->accel;
    if (a->tlas_n != s->ninsts || a->tlas_version != s->inst_version) build_tlas(a, s, p);
}

const Bvh *scene_tlas(const Scene *s){
    const RtAccel *a = s->accel;
    if (!a || s->accel_free != rt_accel_free || s->ninsts <= 0) return NULL;
    return a->tlas_n == s->ninsts && a->tlas_version == s->inst_version && a->tlas.nnodes > 0 ? &a->tlas : NULL;
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
        char wide[48] = "";
        if (b->wide) snprintf(wide, sizeof(wide), " (%d 4-wide)", b->nwide);
        snprintf(buf, len, "bvh (%s, leaf <= %d, %d bins): %d primitives -> %d nodes%s, %d leaves, "
                 "depth %d, %.2f primitives/leaf, built in %.1f ms",
                 b->split == BVH_SPLIT_SAH ? "sah" : "middle", b->max_leaf, b->bins, nshapes,
                 b->nnodes, wide, b->leaves, b->depth, (double)nshapes / b->leaves, a->build_ms);
    } else {
        snprintf(buf, len, "no acceleration: every ray tests all %d primitives", nshapes);
    }
}

static int intersect_world(const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h);

int scene_intersect(const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h){
    int found = intersect_world(s, ray, tmin, tmax, h);
    if (found){ h->inst = -1; tmax = h->t; }
    if (s->ninsts > 0 && intersect_instances(s, ray, tmin, tmax, h, 0)) found = 1;
    return found;
}

static int intersect_world(const Scene *s, const Ray *ray, real tmin, real tmax, Hit *h){
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
    if (s->ninsts > 0 && intersect_instances(s, ray, tmin, tmax, &tmp, 1)) return 1;
    if (a && a->kind == ACCEL_BVH) return bvh_intersect(&a->bvh, s, ray, tmin, tmax, &tmp, 1);
    if (a && a->kind == ACCEL_OCTREE) return octree_intersect(&a->octree, s, ray, tmin, tmax, &tmp, 1);
    for (int i = 0; i < scene_num_shapes(s); i++)
        if (intersect_shape(s, i, ray, tmin, tmax, &tmp)) return 1;
    return 0;
}
