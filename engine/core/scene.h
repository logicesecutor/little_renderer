// scene.h - WHAT is rendered, shared by both backends: materials, primitives (spheres,
// quads, triangles), the light and the camera, plus ready-made Cornell box scenes.
// The path tracer intersects these primitives; the rasterizer turns them into triangles.
#ifndef LR_SCENE_H
#define LR_SCENE_H

#include "core/lrmath.h"
#include "core/mesh.h"
#include "core/texture.h"

#define MAX_MATS 16
#define MAX_VOLUMES 8

// A material is glTF 2.0's metallic-roughness model with transmission (docs/PBR.md, the
// BSDF in core/brdf.h): base colour, metallic, roughness, index of refraction, transmission.
// MAT_PBR takes them as given; the other types are presets of it (material_resolved), so
// a material written as {MAT_GLASS, colour, emission, ior} keeps working:
//   MAT_DIFFUSE  a rough dielectric (roughness 1, ior 1.5): Lambert plus a faint sheen
//   MAT_MIRROR   a smooth metal: the base colour reflected (whiter at grazing angles)
//   MAT_GLASS    a smooth clear dielectric of index `ior`, tinted by the base colour
//   MAT_LIGHT    emits `emission`, reflects nothing
typedef enum {
    MAT_DIFFUSE,
    MAT_MIRROR,
    MAT_GLASS,
    MAT_LIGHT,
    MAT_PBR
} MatType;

typedef struct {
    MatType type;
    Vec3 albedo;        // base colour: the diffuse colour, a metal's reflectance, glass's tint
    Vec3 emission;      // radiance emitted: a MAT_LIGHT's; a MAT_PBR's glow (glTF's emissive:
                        // seen, and lighting by the paths that find it, not sampled as a lamp)
    real ior;           // index of refraction (dielectrics: glass, and the specular of the
                        // others; MAT_DIFFUSE uses 1.5)
    real metallic;      // MAT_PBR: 0 dielectric .. 1 metal
    real roughness;     // MAT_PBR: 0 smooth (mirror, clear) .. 1 rough
    real transmission;  // MAT_PBR: 0 opaque .. 1 clear: how much of the light a dielectric
                        // does not reflect goes through (refracted) instead of diffusing
    // textures (PBR5), each 1 + an index into Scene.texmaps (0 = none), as glTF's: the base
    // colour (times albedo), metallic in blue and roughness in green (times metallic and
    // roughness: they make a preset MAT_PBR), the normal map (tangent space, green up), the
    // emission (times `emission`; PBR6) and the occlusion (red: how much of the light from
    // around reaches the point; only the rasterizers use it, the path tracers find it)
    int tex_base, tex_mr, tex_normal, tex_emissive, tex_occlusion;
    real uv_scale;      // texture coordinates times this (0 = 1)
    real normal_scale;  // the normal map's x and y times this (0 = 1)
    real occlusion;     // the occlusion's strength (0 = 1)
} Material;

// the presets, and a MAT_PBR material
static inline Material material_diffuse(Vec3 color){
    Material m = {.type = MAT_DIFFUSE, .albedo = color, .ior = 1.5, .roughness = 1};
    return m;
}
static inline Material material_mirror(Vec3 color){
    Material m = {.type = MAT_MIRROR, .albedo = color, .ior = 1.5, .metallic = 1};
    return m;
}
static inline Material material_glass(Vec3 tint, real ior){
    Material m = {.type = MAT_GLASS, .albedo = tint, .ior = ior, .transmission = 1};
    return m;
}
static inline Material material_light(Vec3 emission){
    Material m = {.type = MAT_LIGHT, .emission = emission, .ior = 1, .roughness = 1};
    return m;
}
static inline Material material_pbr(Vec3 base, real metallic, real roughness, real ior, real transmission){
    Material m = {.type = MAT_PBR, .albedo = base, .ior = ior, .metallic = metallic, .roughness = roughness,
                  .transmission = transmission};
    return m;
}
// the material has textures
static inline int material_textured(const Material *m){
    return m->tex_base || m->tex_mr || m->tex_normal || m->tex_emissive || m->tex_occlusion;
}
// it glows: a MAT_PBR (or preset) with an emission (not a lamp)
static inline int material_glows(const Material *m){
    return m->type != MAT_LIGHT && (m->emission.x > 0 || m->emission.y > 0 || m->emission.z > 0);
}
// the material with all its parameters: a preset's filled in (type stays)
static inline Material material_resolved(const Material *m){
    Material r = *m;
    switch (m->type){
    case MAT_DIFFUSE: r.metallic = 0; r.roughness = 1; r.ior = 1.5; r.transmission = 0; break;
    case MAT_MIRROR:  r.metallic = 1; r.roughness = 0; r.ior = 1.5; r.transmission = 0; break;
    case MAT_GLASS:   r.metallic = 0; r.roughness = 0; r.transmission = 1; if (!(r.ior > 0)) r.ior = 1.5; break;
    case MAT_LIGHT:   r.metallic = 0; r.roughness = 1; r.transmission = 0; break;
    default:          if (!(r.ior > 0)) r.ior = 1.5; break;
    }
    return r;
}
// the preset closest to a material (MAT_PBR: glass if mostly clear, mirror if mostly metal,
// else diffuse), for the code that knows only the presets (the GPU path tracer until PBR2,
// the rasterizers until PBR3)
static inline MatType material_kind(const Material *m){
    if (m->type != MAT_PBR) return m->type;
    if (m->transmission >= (real)0.5) return MAT_GLASS;
    if (m->metallic >= (real)0.5) return MAT_MIRROR;
    return MAT_DIFFUSE;
}

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
    int t[3];              // texture coordinates (and their tangents): indices into
                           // Scene.uvs / .tangents, or -1 (none)
} Triangle;

Primitive make_sphere(Vec3 center, real radius, int material);
Primitive make_quad(Vec3 q, Vec3 u, Vec3 v, int material);
// box that contains the primitive (slightly padded so flat shapes have some thickness)
Aabb primitive_bounds(const Primitive *prim);
Aabb triangle_bounds(const Triangle *t);

// ---- instances: meshes placed in the world by a transform ------------------------------
// A SceneMesh holds triangles in its own (object) space, with its own vertex normals; an
// Instance places one in the world by an affine transform, and may override its material.
// Instances of one mesh share its triangles and its BVH (rt/accel.c builds it, and the GPU
// path tracer its copy). Moving an instance only changes its transform (and the top-level
// structure over the instances, rebuilt every frame on the GPU). A mesh whose triangles
// change (dynamic, e.g. a cloth) gets its BVH rebuilt when its version changes. Shape ids
// (above) do not cover instances: a hit on one has Hit.inst >= 0 and Hit.prim = the
// triangle's index in its mesh.
typedef struct { real m[3][4]; } Xform;   // world = m * (object, 1): rotation/scale, then translation
Xform xform_identity(void);
// scale, then rotate by r (3 x 3, row-major), then translate by t
Xform xform_make(Vec3 t, const real r[3][3], real scale);
Xform xform_rot_y(Vec3 t, real deg, real scale);
Xform xform_inverse(const Xform *x);
static inline Vec3 xform_point(const Xform *x, Vec3 p){
    return vec3(x->m[0][0] * p.x + x->m[0][1] * p.y + x->m[0][2] * p.z + x->m[0][3],
                x->m[1][0] * p.x + x->m[1][1] * p.y + x->m[1][2] * p.z + x->m[1][3],
                x->m[2][0] * p.x + x->m[2][1] * p.y + x->m[2][2] * p.z + x->m[2][3]);
}
static inline Vec3 xform_dir(const Xform *x, Vec3 d){
    return vec3(x->m[0][0] * d.x + x->m[0][1] * d.y + x->m[0][2] * d.z,
                x->m[1][0] * d.x + x->m[1][1] * d.y + x->m[1][2] * d.z,
                x->m[2][0] * d.x + x->m[2][1] * d.y + x->m[2][2] * d.z);
}
// a normal of the object, in the world: by the inverse transpose (inv = the inverse transform)
static inline Vec3 xform_normal(const Xform *inv, Vec3 n){
    return v_norm(vec3(inv->m[0][0] * n.x + inv->m[1][0] * n.y + inv->m[2][0] * n.z,
                       inv->m[0][1] * n.x + inv->m[1][1] * n.y + inv->m[2][1] * n.z,
                       inv->m[0][2] * n.x + inv->m[1][2] * n.y + inv->m[2][2] * n.z));
}
Aabb xform_box(const Xform *x, Aabb b);   // the world box around a transformed box

typedef struct {
    Triangle *tris; int ntris;      // object space; Triangle.n indexes this mesh's normals
    Vec3 *normals; int nnormals;
    Vec2 *uvs; Vec4 *tangents; int nuvs;   // ... and Triangle.t its texture coordinates
    Aabb box;                       // object-space bounds
    int dynamic;                    // its triangles change: bump version after changing them
    unsigned version;
    void *bvh;                      // the path tracer's BVH of it (rt/accel.c), built when needed
    void (*bvh_free)(void *bvh);
    unsigned bvh_version;           // the version it was built for
} SceneMesh;

// An instance is a mesh, or one of two exact shapes in their own space: the unit sphere
// (radius 1 at the origin) or the unit box [-1, 1]^3, placed (and sized) by the transform.
// The physics' balls and boxes are shape instances: moving one changes only its transform.
#define INST_SPHERE (-1)
#define INST_BOX    (-2)

typedef struct {
    int mesh;                       // index into Scene.meshes, or INST_SPHERE / INST_BOX
    Xform xf, inv;                  // object -> world, and back
    int material;                   // >= 0: instead of the triangles' own (shapes: required)
    Aabb box;                       // world bounds (scene_move_instance updates it)
} Instance;

// SCENE_EMPTY: just the room and the lamp (e.g. for a physics simulation to fill)
// SCENE_SKY: no room and no lamp: a wide floor (brick paving: textures) with five spheres
// (mirror, gold, glass, blue plastic, white) under the sky (SceneOptions.sky, a procedural
// one if none)
// SCENE_SHOWCASE: one model of each material, a mirror dragon, a diffuse cube and a glass
// torus (models/), under a cloud (cloud_default_params() unless SceneOptions.cloud)
typedef enum { SCENE_BOXES, SCENE_SPHERES, SCENE_MESH, SCENE_CLOUD, SCENE_EMPTY, SCENE_SHOWCASE, SCENE_SKY } SceneKind;

struct CloudParams;   // core/volume.h
struct Volume;
struct Env;           // core/env.h

typedef struct {
    SceneKind kind;
    const char *obj_path;   // SCENE_MESH: the .obj or .ply file to put in the box
    // a glTF model (core/gltf.h) instead: SCENE_MESH's model, or in SCENE_SKY in place of
    // the spheres; turned to face the camera (glTF's front is +z), then by rot_deg
    const char *gltf_path;
    MatType mesh_mat;       // SCENE_MESH: MAT_DIFFUSE, MAT_MIRROR or MAT_GLASS
    // ... and what changes it (-base, -metal, -rough, -ior, -transmission): the MESH_SET_*
    // bits say which of these are set; any makes it a MAT_PBR material (scene_mesh_material)
    int mesh_set;
    Vec3 mesh_base;
    real mesh_metal, mesh_rough, mesh_ior, mesh_trans;
    real rot_deg;           // SCENE_MESH: turn the model around the vertical axis
    // A cloud floating in the box (any scene). NULL = no cloud, except SCENE_CLOUD (an
    // empty box with a cloud), which then uses cloud_default_params().
    const struct CloudParams *cloud;
    // > 0: the box is filled with a uniform fog of this extinction (per unit)
    real fog;
    // > 0: also this many instances of obj_path (one mesh, mesh_mat), in a row on the floor
    // (any scene; SCENE_MESH then puts its model in as an instance too)
    int instances;
    // the model's textures (scene_texture's names: a .png file, or a procedural one; "bricks"
    // as mesh_tex also takes the bricks' normal and metal-roughness maps) and their scale
    const char *mesh_tex, *mesh_mr, *mesh_normal;
    real mesh_uv_scale;
    // the sky (any scene; the room's open front lets it in): a Radiance .hdr file, or
    // "procedural"; NULL = none (black), except SCENE_SKY (procedural). sky_scale multiplies
    // it (0 = 1), sky_rot turns it about the vertical axis (degrees)
    const char *sky;
    real sky_scale, sky_rot;
    int bare;               // SCENE_SKY: the floor alone (no spheres): for a program to fill
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
    int light;              // index of the one quad light (-1: none)
    // textures (PBR5): the materials' (Material.tex_*); the world triangles' texture
    // coordinates (Triangle.t), each with a tangent: xyz along +u, w = +-1 the side of the
    // bitangent w (n x t), which points up the image (towards -v)
    TexMap *texmaps; int ntexmaps, cap_texmaps;
    unsigned tex_version;   // a new number whenever a texture is added (the GPU's atlas follows it)
    Vec2 *uvs; Vec4 *tangents; int nuvs, cap_uvs;
    // The environment (core/env.h), owned by the scene (scene_free frees them):
    struct Env *env;        // the sky, at infinity: what a ray leaving the scene sees (NULL: black)
    struct Env *probe;      // the rasterizers' light probe of the room (rt/probe.h), NULL if none
    struct IrrGrid *irr;    // ... and their irradiance grid (its diffuse light), NULL if none
    // where that probe is made (scene_cornell_box sets it: the room's centre and its box; in
    // the open scene above the spheres, its "box" the floor and the sky far away)
    int probe_set;
    Vec3 probe_at;
    Aabb probe_box;

    // camera
    Vec3 cam_from, cam_at, cam_up;
    real cam_vfov_deg;

    // Participating media (clouds, smoke, fog), owned by the scene: scene_free frees them.
    // They may overlap each other and the surfaces (their extinctions add up); a medium has
    // no index of refraction.
    struct Volume *volumes[MAX_VOLUMES];
    int nvolumes;

    // instanced meshes (see above)
    SceneMesh *meshes;
    int nmeshes, cap_meshes;
    Instance *insts;
    int ninsts, cap_insts;
    unsigned inst_version;  // changes whenever an instance is added or moved

    // Acceleration structure, owned by the backend that built it (the path tracer's
    // scene_set_accel). scene_free calls accel_free. Rebuild it after adding geometry.
    void *accel;
    void (*accel_free)(void *accel);
} Scene;

enum { MESH_SET_BASE = 1, MESH_SET_METAL = 2, MESH_SET_ROUGH = 4, MESH_SET_IOR = 8, MESH_SET_TRANS = 16 };

// fills s (which must start zeroed); returns 0 on success
int scene_cornell_box(Scene *s, const SceneOptions *opt);
// the models' material: the preset opt->mesh_mat (diffuse: warm orange), changed as opt says
// (its textures are not loaded: scene_mesh_material_tex does that too)
Material scene_mesh_material(const SceneOptions *opt);
// the same with its textures (opt->mesh_tex, mesh_mr, mesh_normal) loaded into s
Material scene_mesh_material_tex(Scene *s, const SceneOptions *opt);
int scene_add_prim(Scene *s, Primitive p);          // returns the new primitive's index
int scene_add_material(Scene *s, Material m);       // returns the new material's index
// corners a, b, c counter-clockwise seen from outside; normals = 3 vertex normals or NULL.
// Returns the new triangle's index in s->tris.
int scene_add_triangle(Scene *s, Vec3 a, Vec3 b, Vec3 c, const Vec3 *normals, int material);
// all faces of a mesh; its vertex normals are added once and shared. Returns the triangles added.
int scene_add_mesh(Scene *s, const Mesh *m, int material);
// mesh_load + mesh_place + scene_add_mesh. Returns the triangles added, or -1.
int scene_load_mesh(Scene *s, const char *path, int material, real size, Vec3 base, real rot_y_deg);
// A new SceneMesh from a loaded mesh (copied; its triangles get `material`). Returns its index.
int scene_add_mesh_object(Scene *s, const Mesh *m, int material);
// A new SceneMesh from arrays the scene takes over (malloc'ed; Triangle.n and .t index
// normals and uvs; uvs, tangents may be NULL). With uvs but no tangents, the tangents are
// computed. Returns its index.
int scene_add_mesh_arrays(Scene *s, Triangle *tris, int ntris, Vec3 *normals, int nnormals,
                          Vec2 *uvs, Vec4 *tangents, int nuvs);
// mesh_load, scaled so its largest side is `size`, its bottom centre at the origin. -1 on failure.
int scene_load_mesh_object(Scene *s, const char *path, int material, real size);
// An instance of mesh `mesh` (material -1: the triangles' own), or of a shape (mesh =
// INST_SPHERE or INST_BOX, material >= 0). Returns its index.
int scene_add_instance(Scene *s, int mesh, Xform xf, int material);
// rotation r (columns: the shape's axes), scaled per axis, then moved to t: a box of half
// sizes `half`, or a sphere of radius r (scale r, r, r)
Xform xform_axes(Mat3 r, Vec3 scale, Vec3 t);
// a shape instance as world-space primitives: the sphere (part 0), or face `part` (0..5) of
// the box as a quad (u x v outwards) - what the rasterizers draw
Primitive instance_shape_prim(const Instance *in, int part);
// the unit box's face k (0..5) in its own space, u x v outwards
Primitive unit_box_face(int k, int material);
void scene_move_instance(Scene *s, int inst, Xform xf);
// after changing a dynamic mesh's triangles: its bounds, and a new version (its BVHs rebuild)
void scene_mesh_changed(Scene *s, int mesh);
// the material of an instance's triangle
static inline int scene_instance_material(const Scene *s, int inst, int tri){
    const Instance *in = &s->insts[inst];
    return in->material >= 0 || in->mesh < 0 ? in->material : s->meshes[in->mesh].tris[tri].material;
}
Vec3 scene_light_center(const Scene *s);            // center of the light quad

// shapes = primitives followed by triangles (see above)
static inline int scene_num_shapes(const Scene *s){ return s->nprims + s->ntris; }
static inline int scene_shape_material(const Scene *s, int id){
    return id < s->nprims ? s->prims[id].material : s->tris[id - s->nprims].material;
}
Aabb scene_shape_bounds(const Scene *s, int id);
// A texture by name, loaded (or made) once: a .png file, else a procedural one
// (texmap_procedural); srgb for colour. Returns 1 + its index (for Material.tex_*), 0 if it
// cannot be had.
int scene_texture(Scene *s, const char *name, int srgb);
// a texture made elsewhere (a glTF model's), taken over by the scene: 1 + its index
int scene_add_texmap(Scene *s, const TexMap *t);
// The material at texture coordinates uv (Material.uv_scale applied here): its textures
// applied, as a MAT_PBR (or MAT_LIGHT) material without textures. uv_lod: log2 of the
// footprint in texture coordinates (material_uv_lod; MATERIAL_LOD0: no filtering, level 0)
// - each texture reads the mip level of that footprint in its own texels
Material material_at(const Scene *s, const Material *m, Vec2 uv, real uv_lod);
// the normal n bent by the material's normal map at uv (tangent t: xyz, w its handedness)
Vec3 material_normal(const Scene *s, const Material *m, Vec2 uv, real uv_lod, Vec3 n, Vec4 t);
// the occlusion at uv: 1 (none) .. 0 (no light from around), its strength applied
real material_occlusion(const Scene *s, const Material *m, Vec2 uv, real uv_lod);
#define MATERIAL_LOD0 (-1000.0f)
// the footprint's log2 for a pixel whose texture coordinates change by (dudx, dvdx) and
// (dudy, dvdy) to the next ones (the longer side)
static inline real material_uv_lod(real dudx, real dvdx, real dudy, real dvdy){
    real m = r_fmax(dudx * dudx + dvdx * dvdx, dudy * dudy + dvdy * dvdy);
    return m > 0 ? 0.5f * r_log2(m) : MATERIAL_LOD0;
}
// the tangents of triangles' texture coordinates: for each uv index, the sum over the
// triangles using it of their dP/du (and the handedness from dP/dv), orthonormalised to
// nothing yet (the normal is the shading point's): tangents[base .. base + n)
void scene_compute_tangents(const Triangle *tris, int ntris, const Vec2 *uvs, Vec4 *tangents, int base, int n);

// adds a medium and takes ownership of it; returns its index, or -1 if there are already
// MAX_VOLUMES (then v is freed)
int scene_add_volume(Scene *s, struct Volume *v);
void scene_free(Scene *s);

#endif
