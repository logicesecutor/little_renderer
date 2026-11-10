#define _POSIX_C_SOURCE 200809L   // clock_gettime
#include "scene.h"

#include "meshfile.h"
#include "rng.h"   // PI
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int scene_add_material(Scene *s, Material m){
    if (s->nmats >= MAX_MATS){ fprintf(stderr, "too many materials\n"); exit(1); }
    s->mats[s->nmats] = m;
    return s->nmats++;
}

int scene_add_prim(Scene *s, Primitive p){
    if (s->nprims >= s->cap_prims){
        s->cap_prims = s->cap_prims ? s->cap_prims * 2 : 64;
        s->prims = realloc(s->prims, (size_t)s->cap_prims * sizeof(Primitive));
        if (!s->prims){ perror("realloc"); exit(1); }
    }
    s->prims[s->nprims] = p;
    return s->nprims++;
}

static Material diffuse(double r, double g, double b){
    Material m = {MAT_DIFFUSE, vec3(r, g, b), vec3(0, 0, 0), 1.0};
    return m;
}

// rotate around the vertical (y) axis
static Vec3 rotate_y(Vec3 p, double deg){
    double a = deg * PI / 180.0;
    double c = cos(a), s = sin(a);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

// An axis-aligned box from (0,0,0) to size, rotated around y by deg, then moved by offset.
// Made of 6 quads; we transform the quad corners and edges instead of the rays.
static void add_box(Scene *s, Vec3 size, double deg, Vec3 offset, int mat){
    Vec3 dx = vec3(size.x, 0, 0), dy = vec3(0, size.y, 0), dz = vec3(0, 0, size.z);
    Vec3 corner[6] = { vec3(0, 0, 0), vec3(0, 0, size.z), vec3(0, 0, 0),
                       vec3(size.x, 0, 0), vec3(0, 0, 0), vec3(0, size.y, 0) };
    Vec3 eu[6] = { dx, dx, dz, dz, dx, dx };
    Vec3 ev[6] = { dy, dy, dy, dy, dz, dz };
    for (int i = 0; i < 6; i++){
        Vec3 q = v_add(rotate_y(corner[i], deg), offset);
        scene_add_prim(s, make_quad(q, rotate_y(eu[i], deg), rotate_y(ev[i], deg), mat));
    }
}

// The Cornell box, 555 units on each side, with the dimensions and colors used by
// Peter Shirley's "Ray Tracing: The Next Week" (based on the Cornell measurements).
// The camera looks down +z, so x = 555 shows up on the LEFT of the image.
int scene_cornell_box(Scene *s, const SceneOptions *opt){
    int red   = scene_add_material(s, diffuse(0.65, 0.05, 0.05));
    int white = scene_add_material(s, diffuse(0.73, 0.73, 0.73));
    int green = scene_add_material(s, diffuse(0.12, 0.45, 0.15));
    Material lm = {MAT_LIGHT, vec3(0, 0, 0), vec3(15, 15, 15), 1.0};
    int light = scene_add_material(s, lm);

    scene_add_prim(s, make_quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));     // left wall
    scene_add_prim(s, make_quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));     // right wall
    scene_add_prim(s, make_quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));     // floor
    scene_add_prim(s, make_quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white)); // ceiling
    scene_add_prim(s, make_quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));   // back wall

    // ceiling light, 1 unit below the ceiling. u x v points DOWN (-y): that is the side that emits.
    s->light = scene_add_prim(s, make_quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light));

    if (opt->kind == SCENE_BOXES){
        add_box(s, vec3(165, 330, 165), 15.0, vec3(265, 0, 295), white);   // tall box, back left
        add_box(s, vec3(165, 165, 165), -18.0, vec3(130, 0, 65), white);   // short box, front right
    } else if (opt->kind == SCENE_SPHERES){
        Material mirror = {MAT_MIRROR, vec3(0.95, 0.95, 0.95), vec3(0, 0, 0), 1.0};
        Material glass  = {MAT_GLASS,  vec3(1, 1, 1),          vec3(0, 0, 0), 1.5};
        scene_add_prim(s, make_sphere(vec3(370, 100, 370), 100, scene_add_material(s, mirror)));
        scene_add_prim(s, make_sphere(vec3(185, 100, 170), 100, scene_add_material(s, glass)));
    } else {
        Material m = diffuse(0.80, 0.55, 0.25);                       // warm orange
        if (opt->mesh_mat == MAT_MIRROR) m = (Material){MAT_MIRROR, vec3(0.95, 0.95, 0.95), vec3(0, 0, 0), 1.0};
        if (opt->mesh_mat == MAT_GLASS)  m = (Material){MAT_GLASS,  vec3(1, 1, 1),          vec3(0, 0, 0), 1.5};
        // largest side 330 units, standing on the floor in the middle of the box
        int n = mesh_load(s, opt->obj_path, scene_add_material(s, m), 330.0, vec3(278, 0, 278), opt->rot_deg);
        if (n < 0){
            fprintf(stderr, "could not load %s\n", opt->obj_path);
            return 1;
        }
    }

    s->cam_from = vec3(278, 278, -800);
    s->cam_at = vec3(278, 278, 0);
    s->cam_up = vec3(0, 1, 0);
    s->cam_vfov_deg = 40.0;
    return 0;
}

static double clock_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void scene_set_accel(Scene *s, AccelKind kind, const AccelParams *p){
    octree_free(&s->octree);
    bvh_free(&s->bvh);
    double t0 = clock_ms();
    if (kind == ACCEL_OCTREE) octree_build(&s->octree, s->prims, s->nprims, p->oct_depth);
    if (kind == ACCEL_BVH) bvh_build(&s->bvh, s->prims, s->nprims, p->bvh_leaf, p->bvh_bins, p->bvh_split);
    s->accel_build_ms = clock_ms() - t0;
    s->accel = kind;
}

const char *accel_name(AccelKind kind){
    return kind == ACCEL_OCTREE ? "octree" : (kind == ACCEL_BVH ? "bvh" : "none");
}

void scene_accel_stats(const Scene *s, char *buf, int len){
    if (s->accel == ACCEL_OCTREE){
        const Octree *o = &s->octree;
        snprintf(buf, len, "octree: %d primitives -> %d nodes, %d leaves, depth %d (max %d), "
                 "%.2f refs/primitive, built in %.1f ms", s->nprims, o->nnodes, o->leaves, o->depth,
                 o->max_depth, (double)o->nrefs / s->nprims, s->accel_build_ms);
    } else if (s->accel == ACCEL_BVH){
        const Bvh *b = &s->bvh;
        snprintf(buf, len, "bvh (%s, leaf <= %d, %d bins): %d primitives -> %d nodes, %d leaves, "
                 "depth %d, %.2f primitives/leaf, built in %.1f ms",
                 b->split == BVH_SPLIT_SAH ? "sah" : "middle", b->max_leaf, b->bins, s->nprims,
                 b->nnodes, b->leaves, b->depth, (double)s->nprims / b->leaves, s->accel_build_ms);
    } else {
        snprintf(buf, len, "no acceleration: every ray tests all %d primitives", s->nprims);
    }
}

void scene_free(Scene *s){
    octree_free(&s->octree);
    bvh_free(&s->bvh);
    free(s->prims);
    s->prims = NULL;
    s->nprims = s->cap_prims = 0;
}

int scene_intersect(const Scene *s, const Ray *ray, double tmin, double tmax, Hit *h){
    if (s->accel == ACCEL_BVH) return bvh_intersect(&s->bvh, s->prims, ray, tmin, tmax, h, 0);
    if (s->accel == ACCEL_OCTREE) return octree_intersect(&s->octree, s->prims, ray, tmin, tmax, h, 0);

    // brute force: test everything, keep the closest
    int found = 0;
    Hit tmp;
    for (int i = 0; i < s->nprims; i++){
        if (intersect_primitive(&s->prims[i], ray, tmin, tmax, &tmp)){
            found = 1;
            tmax = tmp.t;          // only accept closer hits from now on
            tmp.prim = i;
            *h = tmp;
        }
    }
    return found;
}

int scene_occluded(const Scene *s, const Ray *ray, double tmin, double tmax){
    Hit tmp;
    if (s->accel == ACCEL_BVH) return bvh_intersect(&s->bvh, s->prims, ray, tmin, tmax, &tmp, 1);
    if (s->accel == ACCEL_OCTREE) return octree_intersect(&s->octree, s->prims, ray, tmin, tmax, &tmp, 1);
    for (int i = 0; i < s->nprims; i++)
        if (intersect_primitive(&s->prims[i], ray, tmin, tmax, &tmp)) return 1;
    return 0;
}
