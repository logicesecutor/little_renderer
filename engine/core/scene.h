// scene.h - WHAT is rendered, shared by both backends: materials, primitives (spheres,
// quads, triangles), the light and the camera, plus ready-made Cornell box scenes.
// The path tracer intersects these primitives; the rasterizer turns them into triangles.
#ifndef LR_SCENE_H
#define LR_SCENE_H

#include "core/lrmath.h"
#include "core/mesh.h"

#define MAX_MATS 16
#define MAX_VOLUMES 8

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
    real ior;        // index of refraction (glass only)
} Material;

// axis-aligned bounding box
typedef struct { Vec3 lo, hi; } Aabb;

// A scene holds two kinds of geometry:
//   Primitive: spheres and quads (walls, lamps, boxes). Few, each with precomputed data.
//   Triangle:  mesh triangles, stored compactly (a mesh can have millions of them), with
//              vertex normals shared through one array.
// Together they are numbered as SHAPES: ids 0 .. nprims-1 are the primitives, then
// nprims .. nprims+ntris-1 the triangles. The acceleration structures, the hit records and
// the rasterizer all use shape ids.

typedef enum { PRIM_SPHERE, PRIM_QUAD } PrimType;

typedef struct {
    PrimType type;
    int material;          // index into Scene.mats

    // sphere
    Vec3 center;
    real radius;

    // quad: the points q + a*u + b*v with a, b in [0, 1]
    Vec3 q, u, v;
    Vec3 normal;           // unit normal = normalize(u x v); for lights: the side that emits
    real plane_d;          // plane equation: dot(normal, p) = plane_d
    Vec3 w;                // (u x v) / |u x v|^2, used to find a, b
    real area;
} Primitive;

typedef struct {
    Vec3 q, u, v;          // corners q, q + u, q + v (u, v = the two edges leaving q)
    int n[3];              // vertex normals: indices into Scene.normals, or -1 (flat)
    int material;
} Triangle;

Primitive make_sphere(Vec3 center, real radius, int material);
Primitive make_quad(Vec3 q, Vec3 u, Vec3 v, int material);
// box that contains the primitive (slightly padded so flat shapes have some thickness)
Aabb primitive_bounds(const Primitive *prim);
Aabb triangle_bounds(const Triangle *t);

// SCENE_EMPTY: just the room and the lamp (e.g. for a physics simulation to fill)
typedef enum { SCENE_BOXES, SCENE_SPHERES, SCENE_MESH, SCENE_CLOUD, SCENE_EMPTY } SceneKind;

struct CloudParams;   // core/volume.h
struct Volume;

typedef struct {
    SceneKind kind;
    const char *obj_path;   // SCENE_MESH: the .obj or .ply file to put in the box
    MatType mesh_mat;       // SCENE_MESH: MAT_DIFFUSE, MAT_MIRROR or MAT_GLASS
    real rot_deg;           // SCENE_MESH: turn the model around the vertical axis
    // A cloud floating in the box (any scene). NULL = no cloud, except SCENE_CLOUD (an
    // empty box with a cloud), which then uses cloud_default_params().
    const struct CloudParams *cloud;
    // > 0: the box is filled with a uniform fog of this extinction (per unit)
    real fog;
} SceneOptions;

typedef struct Scene {
    Primitive *prims;       // spheres and quads (growable array)
    int nprims, cap_prims;
    Triangle *tris;         // mesh triangles (growable array)
    int ntris, cap_tris;
    Vec3 *normals;          // unit vertex normals referenced by Triangle.n (growable array)
    int nnormals, cap_normals;
    Material mats[MAX_MATS];
    int nmats;
    int light;              // index of the one quad light

    // camera
    Vec3 cam_from, cam_at, cam_up;
    real cam_vfov_deg;

    // Participating media (clouds, smoke, fog), owned by the scene: scene_free frees them.
    // They may overlap each other and the surfaces (their extinctions add up); a medium has
    // no index of refraction.
    struct Volume *volumes[MAX_VOLUMES];
    int nvolumes;

    // Acceleration structure, owned by the backend that built it (the path tracer's
    // scene_set_accel). scene_free calls accel_free. Rebuild it after adding geometry.
    void *accel;
    void (*accel_free)(void *accel);
} Scene;

// fills s (which must start zeroed); returns 0 on success
int scene_cornell_box(Scene *s, const SceneOptions *opt);
int scene_add_prim(Scene *s, Primitive p);          // returns the new primitive's index
int scene_add_material(Scene *s, Material m);       // returns the new material's index
// corners a, b, c counter-clockwise seen from outside; normals = 3 vertex normals or NULL.
// Returns the new triangle's index in s->tris.
int scene_add_triangle(Scene *s, Vec3 a, Vec3 b, Vec3 c, const Vec3 *normals, int material);
// all faces of a mesh; its vertex normals are added once and shared. Returns the triangles added.
int scene_add_mesh(Scene *s, const Mesh *m, int material);
// mesh_load + mesh_place + scene_add_mesh. Returns the triangles added, or -1.
int scene_load_mesh(Scene *s, const char *path, int material, real size, Vec3 base, real rot_y_deg);
Vec3 scene_light_center(const Scene *s);            // center of the light quad

// shapes = primitives followed by triangles (see above)
static inline int scene_num_shapes(const Scene *s){ return s->nprims + s->ntris; }
static inline int scene_shape_material(const Scene *s, int id){
    return id < s->nprims ? s->prims[id].material : s->tris[id - s->nprims].material;
}
Aabb scene_shape_bounds(const Scene *s, int id);
// adds a medium and takes ownership of it; returns its index, or -1 if there are already
// MAX_VOLUMES (then v is freed)
int scene_add_volume(Scene *s, struct Volume *v);
void scene_free(Scene *s);

#endif
