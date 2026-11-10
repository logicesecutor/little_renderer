// scene.h - materials, the list of primitives, and the Cornell box builders.
#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"
#include "geometry.h"
#include "octree.h"

#define MAX_MATS 16

typedef enum {
    MAT_DIFFUSE,   // ideal Lambertian: scatters light equally in all directions
    MAT_MIRROR,    // perfect reflection
    MAT_GLASS,     // dielectric: reflects or refracts according to Fresnel
    MAT_LIGHT      // emits light, reflects nothing
} MatType;

typedef struct {
    MatType type;
    Vec3 albedo;     // fraction of light reflected per RGB channel
    Vec3 emission;   // radiance emitted (lights only)
    double ior;      // index of refraction (glass only)
} Material;

typedef enum { SCENE_BOXES, SCENE_SPHERES, SCENE_MESH } SceneKind;

typedef struct {
    SceneKind kind;
    const char *obj_path;   // SCENE_MESH: the .obj or .ply file to put in the box
    MatType mesh_mat;       // SCENE_MESH: MAT_DIFFUSE, MAT_MIRROR or MAT_GLASS
    double rot_deg;         // SCENE_MESH: turn the model around the vertical axis
} SceneOptions;

typedef enum { ACCEL_NONE, ACCEL_OCTREE, ACCEL_BVH } AccelKind;

typedef struct {
    int oct_depth;          // octree max depth, <= 0 = automatic
    int bvh_leaf;           // BVH: max primitives per leaf
    int bvh_bins;           // BVH: SAH buckets per axis
    BvhSplit bvh_split;     // BVH: SAH or middle split
} AccelParams;

typedef struct {
    Primitive *prims;       // growable array
    int nprims, cap_prims;
    Material mats[MAX_MATS];
    int nmats;
    int light;              // index of the one quad light (used for direct light sampling)

    AccelKind accel;        // which structure scene_intersect uses
    Octree octree;
    Bvh bvh;
    double accel_build_ms;  // how long the last scene_set_accel took

    // camera
    Vec3 cam_from, cam_at, cam_up;
    double cam_vfov_deg;
} Scene;

// fills s (which must start zeroed); returns 0 on success
int scene_cornell_box(Scene *s, const SceneOptions *opt);
int scene_add_prim(Scene *s, Primitive p);
int scene_add_material(Scene *s, Material m);
// (re)builds the chosen acceleration structure, freeing the previous one
void scene_set_accel(Scene *s, AccelKind kind, const AccelParams *p);
const char *accel_name(AccelKind kind);
// one line of statistics about the current structure, e.g. for printing
void scene_accel_stats(const Scene *s, char *buf, int len);
void scene_free(Scene *s);

// closest hit over all primitives
int scene_intersect(const Scene *s, const Ray *ray, double tmin, double tmax, Hit *h);
// is there ANY hit in (tmin, tmax)? (shadow rays)
int scene_occluded(const Scene *s, const Ray *ray, double tmin, double tmax);

#endif
