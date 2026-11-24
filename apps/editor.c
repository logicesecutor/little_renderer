// editor.c - the engine side of the web editor ("Kiln", web/editor/): the viewer
// (app/viewer.h) driven by the page's HTML interface through the functions below, which the
// page calls (Module._lr_...). The page owns the input; the engine draws the scene.
//
//   make web-editor && make serve        then open http://localhost:8000/editor/
//
// Natively (bin/editor) it opens the same scene in a window, flown with the viewer's keys.
//
// Units: 100 to a metre (the Cornell box's 555 units are 5.55 m), so a glTF model in metres
// keeps its real size; the page shows metres.
#define _POSIX_C_SOURCE 200809L
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "app/viewer.h"
#include "lr.h"
#include "base/mem.h"
#include "scene/env.h"
#include "rt/accel.h"
#include "rt/intersect.h"
#include "scene/camera.h"
#include "scene/gltf.h"
#include "scene/gltf_write.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "scene/pbr_maps.h"
#include "phys/world.h"
#include "formats/json.h"
#include "base/file.h"
#include "rt/probe.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

#define UNITS_PER_M 100.0

// ---- the catalogue (docs: the editor's spec, 7.2): the materials, glTF metallic-roughness ----

typedef struct {
    const char *key, *label;
    unsigned rgb;                  // base colour, sRGB
    real metallic, roughness, transmission;
    real glow;                     // emission: the base colour times this (KHR_materials_emissive_strength)
} EdMaterial;

static const EdMaterial MATERIALS[] = {
    {"wood", "Painted wood", 0xC9A27E, 0, 0.7, 0, 0},
    {"terracotta", "Terracotta", 0xC9694A, 0, 0.8, 0, 0},
    {"stone", "Stone", 0x8C857A, 0, 0.9, 0, 0},
    {"steel", "Brushed steel", 0xA9B0BA, 1, 0.35, 0, 0},
    {"gold", "Gold", 0xE3B04B, 1, 0.25, 0, 0},
    {"rubber", "Blue rubber", 0x3E6FA8, 0, 0.6, 0, 0},
    {"glass", "Glass", 0xFFFFFF, 0, 0, 1, 0},
    {"glow", "Emissive", 0xFFE2B8, 0, 1, 0, 8},
};
#define NMATERIALS ((int)(sizeof(MATERIALS) / sizeof(MATERIALS[0])))

static real srgb_channel(unsigned c){
    double v = c / 255.0;
    return (real)(v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4));
}
static Vec3 srgb_hex(unsigned rgb){
    return vec3(srgb_channel(rgb >> 16 & 255), srgb_channel(rgb >> 8 & 255), srgb_channel(rgb & 255));
}

// ---- the objects: what the page lists and moves, each one or more instances ----------------

typedef enum { ED_BOX, ED_SPHERE, ED_ROCK, ED_LAMP, ED_MODEL } EdKind;
static const char *const KIND_NAMES[] = {"box", "sphere", "rock", "lamp", "model"};

typedef struct {
    int id;
    EdKind kind;
    char name[64];
    int mat;                       // index into MATERIALS (-1: a model's own, MAT_OWN: its look)
    int mesh;                      // a rock's mesh (Scene.meshes)
    int model;                     // a model's index into Editor.models
    int dynamic;                   // its body in the physics (Simulate mode): 0 static, 1 dynamic
    real scale;                    // 1 = the shape's own size (a model: the file's)
    Vec3 pos;                      // the middle of its bottom, units
    Quat rot;                      // its turn, about the middle of its bottom
    Vec3 stretch;                  // its size along its own axes, times scale (the gizmo's Scale: 1, 1, 1 at first)
    int support;                   // the id of what it rests on (0: the ground)
    int *inst, ninst;              // its instances in the scene (its own array)
    // its own material (lr_set_look; mat == MAT_OWN): the scene material it has (-1: none yet;
    // kept once made, through undo), its base colour (sRGB), roughness, metallic, and an image
    // for the base colour (1 + an index into Scene.texmaps, 0: none; its file's name, to show)
    int look;
    unsigned look_rgb;
    real look_rough, look_metal;
    // its maps (PBR7: LOOK_BASE..., 1 + an index into Scene.texmaps, 0: none) and their files'
    // names (to show), the bump's strength, the normal map's, the occlusion's, the glow's
    // brightness (with an emissive map), whether the normal map's green points down; what was
    // made from them (pbr_apply: the normal from normal and bump, roughness and metallic
    // packed), kept in place
    int look_map[8];
    char look_map_name[8][48];
    real look_bump, look_normal_k, look_ao_k, look_glow;
    int look_flip;
    int look_made_normal, look_made_mr;
} EdObject;
enum { LOOK_BASE, LOOK_NORMAL, LOOK_BUMP, LOOK_ROUGH, LOOK_METAL, LOOK_ORM, LOOK_AO, LOOK_EMISSIVE, NLOOK };
static const char *const LOOK_KEYS[NLOOK] = {"base", "normal", "bump", "roughness", "metallic", "orm", "ao", "emissive"};
#define MAT_OWN (-2)

// a model read from a file: its parts as the file places them, at its own size, the middle
// of its bottom at the origin (kept while the editor runs, for undo)
typedef struct { int mesh, material; Xform xf; } EdPart;
typedef struct { EdPart *parts; int nparts; Aabb box; int clay; } EdModel;   // (clay: a mesh file's, of no material of its own)

// the sky presets (the spec, 7.3), and a file's
enum { SKY_DUSK, SKY_NOON, SKY_OVERCAST, NSKIES, SKY_FILE = -1, SKY_SUN = 10 };
// the Light tab's sky (SKY_SUN): made here from the time of day (t: 0 night .. 0.35 dusk ..
// 1 noon), the sun's intensity, its disc's size (degrees across), the haze (1 clear .. 10),
// its direction (degrees) and whether its colour follows its height (warm low down)
typedef struct { real t, intensity, size, haze, az; int warm; } SunSky;
static const char *const SKY_NAMES[NSKIES] = {"Dusk", "Noon", "Overcast"};

typedef struct {
    EdObject *objs;
    int nobjs, cap, next_id;
    EdModel *models;
    int nmodels, cap_models;
    int mats[NMATERIALS];          // the catalogue's materials in the scene
    int dark;                      // the lamp's stand
    int clay;                      // what a mesh file (.obj, .ply) is made of: it has no material
    int last_fitted;               // the last mesh file was fitted to 1 m (lr_import_mesh)
    int rock_mesh;                 // the rock's mesh (made here)
    int sky;                       // the sky: a preset, SKY_SUN (sun), or SKY_FILE (sky_path)
    SunSky sun;                    // SKY_SUN's
    int sun_w;                     // ... made this wide (a slider being dragged: smaller, quicker)
    char sky_name[96], sky_path[256];
    struct ProbeJob *relight;      // the rasterizer's light being made again (NULL: up to date)
    double relight_done;
    // the orbit camera: around a point, at a distance
    Vec3 pivot;
    double dist;
    int fly_keys;                  // W A S D E Q Shift (and Space: 128) held while flying or walking (lr_camera_fly)
    int walk;                      // Play: the camera walks (level, at eye height, falls, jumps) instead of flying
    double walk_vy;                // ... its vertical speed (units / s)
    View saved_view;               // lr_camera_save: the view to come back to (after Play)
    Vec3 saved_pivot;
    double saved_dist;
    int have_saved;
    // a built-in sample instead of the editor's own scene (-sim on: the engine's physics
    // demo exactly, bin/view -sim on -demo ...): shown and simulated, not edited
    int sample;
    double t_prev;
} Editor;

static Editor E;
static int editing_off(void);    // (a sample shows, or a simulation holds the objects)

static int object_parts(const EdObject *o){
    return o->kind == ED_LAMP ? 3 : o->kind == ED_MODEL ? E.models[o->model].nparts : 1;
}

// part i in the object's own space at its own size (units, the middle of its bottom at the
// origin)
static Xform object_part_local(const EdObject *o, int i){
    switch (o->kind){
    case ED_BOX: case ED_SPHERE:
        return xform_axes(mat3_identity(), vec3(50, 50, 50), vec3(0, 50, 0));
    case ED_ROCK:
        return xform_identity();
    case ED_LAMP:   // a base, a pole and the glowing ball on top
        if (i == 0) return xform_axes(mat3_identity(), vec3(18, 3, 18), vec3(0, 3, 0));
        if (i == 1) return xform_axes(mat3_identity(), vec3(3, 32, 3), vec3(0, 38, 0));
        return xform_axes(mat3_identity(), vec3(20, 20, 20), vec3(0, 88, 0));
    case ED_MODEL: default:
        return E.models[o->model].parts[i].xf;
    }
}

// part i in the world: its own place, sized (scale times stretch along the object's axes),
// turned, moved to pos
static Xform object_part_xform(const EdObject *o, int i){
    const Xform L = object_part_local(o, i);
    Mat3 r = quat_to_mat3(o->rot);
    Vec3 k = v_scale(o->stretch, o->scale);
    real ks[3] = {k.x, k.y, k.z}, p[3] = {o->pos.x, o->pos.y, o->pos.z};
    Xform w;
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 4; c++){
            real v = 0;
            for (int j = 0; j < 3; j++) v += r.m[a][j] * ks[j] * L.m[j][c];
            w.m[a][c] = c == 3 ? v + p[a] : v;
        }
    return w;
}

// the materials of an object's parts: the lamp's stand is dark, the rest its material (a
// model: its own, unless one of the catalogue's was put on it)
static int part_material(const EdObject *o, int part){
    if (o->kind == ED_LAMP && part < 2) return E.dark;
    if (o->mat >= 0) return E.mats[o->mat];
    if (o->mat == MAT_OWN && o->look >= 0) return o->look;
    return o->kind == ED_MODEL ? E.models[o->model].parts[part].material : -1;
}
static int part_mesh(const EdObject *o, int part){
    switch (o->kind){
    case ED_BOX: return INST_BOX;
    case ED_SPHERE: return INST_SPHERE;
    case ED_LAMP: return part < 2 ? INST_BOX : INST_SPHERE;
    case ED_MODEL: return E.models[o->model].parts[part].mesh;
    default: return o->mesh;
    }
}

// its instances made (an object new, or back by an undo)
static void object_instances(EdObject *o){
    Scene *s = viewer_scene();
    int n = object_parts(o);
    o->inst = malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!o->inst) lr_oom("editor");
    for (int i = 0; i < n; i++) o->inst[i] = scene_add_instance(s, part_mesh(o, i), object_part_xform(o, i), part_material(o, i));
    o->ninst = n;
}

// its instances where its fields say, of its material
static void object_update(EdObject *o){
    Scene *s = viewer_scene();
    for (int i = 0; i < o->ninst; i++){
        s->insts[o->inst[i]].material = part_material(o, i);
        scene_move_instance(s, o->inst[i], object_part_xform(o, i));   // (a new version: the copies take it again)
    }
}

// its instances removed (the scene's last ones take their places: whoever had those now has these)
static void object_drop_instances(EdObject *o){
    Scene *s = viewer_scene();
    for (int k = o->ninst - 1; k >= 0; k--){
        int i = o->inst[k];
        o->inst[k] = -1;
        int moved = scene_remove_instance(s, i);
        if (moved < 0) continue;
        for (int j = 0; j < E.nobjs; j++)
            for (int q = 0; q < E.objs[j].ninst; q++)
                if (E.objs[j].inst[q] == moved) E.objs[j].inst[q] = i;
    }
    free(o->inst);
    o->inst = NULL;
    o->ninst = 0;
}

// a new object, with its instances. label: the name before its number ("Glass ball"; NULL:
// the kind's); a model (model >= 0): its name as it is
static int object_add(EdKind kind, int mat, int model, Vec3 pos, real scale, int dynamic, int support, const char *label){
    E.objs = lr_grow(E.objs, sizeof(EdObject), &E.cap, E.nobjs + 1, 16);
    EdObject *o = &E.objs[E.nobjs];
    memset(o, 0, sizeof(*o));
    o->id = ++E.next_id;
    o->kind = kind;
    o->mat = mat;
    o->mesh = kind == ED_ROCK ? E.rock_mesh : -1;
    o->model = model;
    o->scale = scale;
    o->rot = quat_identity();
    o->stretch = vec3(1, 1, 1);
    o->pos = pos;
    o->dynamic = dynamic;
    o->support = support;
    o->look = -1;
    if (kind == ED_MODEL) snprintf(o->name, sizeof(o->name), "%s", label ? label : "Model");
    else if (label) snprintf(o->name, sizeof(o->name), "%s %d", label, o->id);
    else snprintf(o->name, sizeof(o->name), "%c%s %d", KIND_NAMES[kind][0] - 32, KIND_NAMES[kind] + 1, o->id);
    object_instances(o);
    E.nobjs++;
    return o->id;
}

static EdObject *object_find(int id){
    for (int i = 0; i < E.nobjs; i++) if (E.objs[i].id == id) return &E.objs[i];
    return NULL;
}
static EdObject *object_of_inst(int inst){
    for (int i = 0; i < E.nobjs; i++)
        for (int k = 0; k < E.objs[i].ninst; k++) if (E.objs[i].inst[k] == inst) return &E.objs[i];
    return NULL;
}
// id is root, or rests on it (on something resting on it...)
static int in_stack(int id, int root){
    for (int depth = 0; id > 0 && depth < 1000; depth++){
        if (id == root) return 1;
        const EdObject *o = object_find(id);
        id = o ? o->support : 0;
    }
    return 0;
}

// its world box: around its instances'
static Aabb object_box(const EdObject *o){
    const Scene *s = viewer_scene();
    Aabb box = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
    for (int k = 0; k < o->ninst; k++){
        Aabb b = s->insts[o->inst[k]].box;
        box.lo = vec3(r_fmin(box.lo.x, b.lo.x), r_fmin(box.lo.y, b.lo.y), r_fmin(box.lo.z, b.lo.z));
        box.hi = vec3(r_fmax(box.hi.x, b.hi.x), r_fmax(box.hi.y, b.hi.y), r_fmax(box.hi.z, b.hi.z));
    }
    return box;
}

// how high its top is above the middle of its bottom (where something put on it stands),
// and how far from its middle its top reaches (where on it something can stand), units:
// from its box in the world (turned or stretched by the gizmo too)
static real object_top(const EdObject *o){ return object_box(o).hi.y - o->pos.y; }
static real object_reach(const EdObject *o){
    Aabb b = object_box(o);
    return 0.3f * r_fmin(b.hi.x - b.lo.x, b.hi.z - b.lo.z);
}

// an object moved by d, and what rests on it with it
static void object_shift(int id, Vec3 d, int depth){
    EdObject *o = object_find(id);
    if (!o || depth > 1000) return;
    o->pos = v_add(o->pos, d);
    object_update(o);
    for (int i = 0; i < E.nobjs; i++) if (E.objs[i].support == id) object_shift(E.objs[i].id, d, depth + 1);
}

// ---- the scene -----------------------------------------------------------------------------

// a rock: a sphere of 162 points pushed in and out by noise, its bottom flattened; 110 units
// across, its bottom centre at the origin
static int make_rock(Scene *s){
    // an icosahedron split twice
    static const real t = 1.6180340f;
    Vec3 base[12] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                     {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    static const int faces[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                     {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                                     {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    int nv = 12, nf = 20;
    Vec3 *v = malloc(642 * sizeof(Vec3));
    int (*f)[3] = malloc(1280 * sizeof(*f));
    if (!v || !f) lr_oom("editor");
    for (int i = 0; i < 12; i++) v[i] = v_norm(base[i]);
    memcpy(f, faces, sizeof(faces));
    for (int level = 0; level < 2; level++){
        int (*g)[3] = malloc((size_t)nf * 4 * sizeof(*g));
        if (!g) lr_oom("editor");
        int ng = 0;
        for (int k = 0; k < nf; k++){
            int mid[3];
            for (int e = 0; e < 3; e++){
                int a = f[k][e], b = f[k][(e + 1) % 3];
                Vec3 m = v_norm(v_scale(v_add(v[a], v[b]), 0.5f));
                int found = -1;
                for (int i = 12; i < nv && found < 0; i++)
                    if (v_dot(v_sub(v[i], m), v_sub(v[i], m)) < 1e-8f) found = i;
                if (found < 0){ v[nv] = m; found = nv++; }
                mid[e] = found;
            }
            int a = f[k][0], b = f[k][1], c = f[k][2];
            int add[4][3] = {{a, mid[0], mid[2]}, {b, mid[1], mid[0]}, {c, mid[2], mid[1]}, {mid[0], mid[1], mid[2]}};
            memcpy(g[ng], add, sizeof(add));
            ng += 4;
        }
        free(f);
        f = g;
        nf = ng;
    }
    Noise nz;
    noise_init(&nz, 7);
    Mesh m = {0};
    m.v = malloc((size_t)nv * sizeof(Vec3));
    m.f = calloc((size_t)nf, sizeof(Face));
    if (!m.v || !m.f) lr_oom("editor");
    for (int i = 0; i < nv; i++){
        Vec3 p = v[i];
        real r = 1 + 0.22f * noise_fbm(&nz, v_scale(p, 1.7f), 3);
        p = vec3(p.x * r * 1.25f, p.y * r * 0.8f, p.z * r);   // wider than tall
        if (p.y < -0.35f) p.y = -0.35f;                         // a flat bottom to stand on
        m.v[i] = p;
    }
    for (int k = 0; k < nf; k++)
        for (int e = 0; e < 3; e++){ m.f[k].v[e] = f[k][e]; m.f[k].t[e] = -1; m.f[k].n[e] = -1; }
    m.nv = nv;
    m.nf = nf;
    free(v);
    free(f);
    mesh_compute_normals(&m);
    mesh_place(&m, 110, vec3(0, 0, 0), 0);
    int id = scene_add_mesh_object(s, &m, E.mats[2]);
    free(m.v);
    free(m.vn);
    free(m.f);
    return id;
}

// ---- the skies (the spec, 7.3; their exact looks belong to Light mode) -----------------------

// Dusk: a low warm sun (4 degrees up) under a deep blue sky, warm along the horizon; Noon: the
// engine's sky with the sun 60 degrees up; Overcast: no sun, a soft grey sky
static Env *make_sky(int preset){
    if (preset == SKY_NOON) return env_procedural_sky(1024, 60, 150);
    int w = 1024, h = 512;
    float *img = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (!img) return NULL;
    real el = 4 * (real)PI / 180, az = 150 * (real)PI / 180;
    Vec3 sun = vec3(r_cos(el) * r_sin(az), r_sin(el), r_cos(el) * r_cos(az));
    const real sun_cos = r_cos(1.2f * (real)PI / 180);
    Env flat = {0};                                       // (env_dir of an unturned map)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            Vec3 d = env_dir(&flat, (x + 0.5f) / w, (y + 0.5f) / h), c;
            if (preset == SKY_OVERCAST){
                if (d.y >= 0){
                    real t = r_pow(1 - d.y, 3);
                    c = v_add(v_scale(vec3(0.50f, 0.53f, 0.57f), 1 - t), v_scale(vec3(0.78f, 0.80f, 0.82f), t));
                } else {
                    real t = r_fmin(-d.y * 4, 1);
                    c = v_add(v_scale(vec3(0.55f, 0.55f, 0.55f), 1 - t), v_scale(vec3(0.16f, 0.15f, 0.14f), t));
                }
            } else {                                      // dusk
                real cs = v_dot(d, sun);
                if (d.y >= 0){
                    real t = 1 - d.y, t4 = t * t * t * t;
                    c = v_add(v_scale(vec3(0.05f, 0.08f, 0.22f), 1 - t4), v_scale(vec3(0.95f, 0.52f, 0.30f), t4));
                    c = v_add(c, v_scale(vec3(1.0f, 0.55f, 0.25f), 1.2f * r_pow(r_fmax(cs, 0), 24)));   // the glow around the sun
                    c = v_add(c, v_scale(vec3(0.9f, 0.45f, 0.25f), 0.25f * r_pow(r_fmax(cs, 0), 3)));
                    if (cs > sun_cos) c = v_add(c, vec3(1700, 850, 340));                         // the sun, red through the air
                } else {
                    real t = r_fmin(-d.y * 4, 1);
                    c = v_add(v_scale(vec3(0.45f, 0.30f, 0.22f), 1 - t), v_scale(vec3(0.07f, 0.06f, 0.05f), t));
                }
            }
            float *o = img + 3 * ((size_t)y * w + x);
            o[0] = (float)c.x; o[1] = (float)c.y; o[2] = (float)c.z;
        }
    return env_from_rgb(img, w, h);
}

static Vec3 mix3(Vec3 a, Vec3 b, real t){ return v_add(v_scale(a, 1 - t), v_scale(b, t)); }
static real smooth01(real x){ x = r_fmin(r_fmax(x, 0), 1); return x * x * (3 - 2 * x); }
static real sun_elev(const SunSky *p){ return -8 + 68 * p->t; }     // degrees: -8 (night) .. 60 (noon)

// the Light tab's sky: blue by day, warm at dusk, dark at night (zenith and horizon colours
// blended by the sun's height), whiter at the horizon with haze, a glow round the sun and its
// disc, whose light on the ground (radiance times its solid angle) is the intensity times 3
// whatever its size, reddened and dimmed low down when its colour follows its height
static Env *make_sun_sky(const SunSky *p, int w){
    int h = w / 2;
    float *img = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (!img) return NULL;
    real eld = sun_elev(p), el = eld * (real)PI / 180, az = p->az * (real)PI / 180;
    Vec3 sun = vec3(r_cos(el) * r_sin(az), r_sin(el), r_cos(el) * r_cos(az));
    real day = smooth01(eld / 25), dusk = smooth01((eld + 8) / 8), hk = r_fmin(r_fmax((p->haze - 1) / 9, 0), 1);
    Vec3 zn = mix3(mix3(vec3(0.003f, 0.005f, 0.012f), vec3(0.05f, 0.08f, 0.22f), dusk), vec3(0.10f, 0.22f, 0.55f), day);
    Vec3 hz = mix3(mix3(vec3(0.012f, 0.014f, 0.03f), vec3(0.95f, 0.52f, 0.30f), dusk), vec3(0.62f, 0.72f, 0.86f), day);
    real lum = 0.3f * hz.x + 0.6f * hz.y + 0.1f * hz.z;
    hz = mix3(hz, vec3(lum * 1.1f, lum * 1.1f, lum * 1.1f), 0.6f * hk);
    zn = mix3(zn, hz, 0.35f * hk);
    Vec3 sc = p->warm ? mix3(vec3(1.0f, 0.45f, 0.18f), vec3(1.0f, 0.96f, 0.90f), smooth01(eld / 35)) : vec3(1.0f, 0.97f, 0.93f);
    real through = p->warm ? 0.25f + 0.75f * smooth01(eld / 20) : 1;            // (more air low down)
    real rad = 0.5f * r_fmax(p->size, 0.05f) * (real)PI / 180, cos_r = r_cos(rad);
    real e_sun = p->intensity * 3 * through * smooth01((eld + 1) / 2) * (1 - 0.5f * hk);
    Vec3 disc = v_scale(sc, e_sun / ((real)PI * rad * rad));
    real glow = (0.9f * (1 - day) + 0.25f * day) * dusk * p->intensity;
    Env flat = {0};                                       // (env_dir of an unturned map)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            Vec3 d = env_dir(&flat, (x + 0.5f) / w, (y + 0.5f) / h), c;
            real cs = v_dot(d, sun);
            if (d.y >= 0){
                real t = 1 - d.y, tt = r_pow(t, 4 - 2 * hk);
                c = mix3(zn, hz, tt);
                c = v_add(c, v_scale(sc, glow * (1.2f * r_pow(r_fmax(cs, 0), 24) + 0.25f * r_pow(r_fmax(cs, 0), 3))));
                if (cs > cos_r && e_sun > 0) c = v_add(c, disc);
            } else {
                real t = r_fmin(-d.y * 4, 1);
                c = mix3(v_scale(hz, 0.5f), v_scale(vec3(0.07f, 0.06f, 0.05f), 0.2f + 0.8f * day + 0.4f * dusk), t);
            }
            float *o = img + 3 * ((size_t)y * w + x);
            o[0] = (float)c.x; o[1] = (float)c.y; o[2] = (float)c.z;
        }
    return env_from_rgb(img, w, h);
}
static void sun_name(char *out, size_t n){            // "Sun 18:24": the clock of the Light tab
    int mins = (int)lrint((21.5 - E.sun.t * 8.5) * 60);
    snprintf(out, n, "Sun %02d:%02d", mins / 60, mins % 60);
}

// the sky made: a preset, or a file (.hdr) when path is given; 0 = done (else nothing changed)
static int set_sky(int preset, const char *path, const char *name){
    Scene *s = viewer_scene();
    Env *e = path ? env_load_hdr(path) : preset == SKY_SUN ? make_sun_sky(&E.sun, E.sun_w > 0 ? E.sun_w : 1024) :
             preset >= 0 && preset < NSKIES ? make_sky(preset) : NULL;
    if (!e) return 1;
    env_free(s->env);
    s->env = e;
    E.sky = path ? SKY_FILE : preset;
    snprintf(E.sky_path, sizeof(E.sky_path), "%s", path ? path : "");
    if (preset == SKY_SUN && !path) sun_name(E.sky_name, sizeof(E.sky_name));
    else snprintf(E.sky_name, sizeof(E.sky_name), "%s", path ? name : SKY_NAMES[preset]);
    return 0;
}

static int build_scene(Scene *s){
    for (int i = 0; i < NMATERIALS; i++){
        const EdMaterial *d = &MATERIALS[i];
        Material m = material_pbr(srgb_hex(d->rgb), d->metallic, d->roughness, 1.5, d->transmission);
        if (d->glow > 0) m.emission = v_scale(srgb_hex(d->rgb), d->glow);
        E.mats[i] = scene_add_material(s, m);
    }
    E.dark = scene_add_material(s, material_pbr(srgb_hex(0x3A3530), 0, 0.6, 1.5, 0));
    E.clay = scene_add_material(s, material_pbr(srgb_hex(0xB8AFA3), 0, 0.6, 1.5, 0));
    // the ground: 200 m of plain earth-coloured floor
    int ground = scene_add_material(s, material_pbr(srgb_hex(0x8A7E6C), 0, 0.95, 1.5, 0));
    scene_add_prim(s, make_quad(vec3(-10000, 0, -10000), vec3(0, 0, 20000), vec3(20000, 0, 0), ground));
    s->light = -1;
    if (set_sky(SKY_DUSK, NULL, NULL)) return 1;
    s->probe_set = 1;
    s->probe_at = vec3(0, 150, 0);
    s->probe_box = (Aabb){vec3(-10000, 0, -10000), vec3(10000, 1e5f, 10000)};
    E.rock_mesh = make_rock(s);
    // the prototype's starting set: a rock, a box with a smaller steel one on it, a lamp,
    // a terracotta ball and a glass one (the camera looks down +z: +x is on the left)
    object_add(ED_ROCK, 2, -1, vec3(250, 0, 260), 1, 0, 0, NULL);
    int box = object_add(ED_BOX, 0, -1, vec3(40, 0, 120), 1, 0, 0, NULL);
    object_add(ED_BOX, 3, -1, vec3(43, 100, 118), 0.7f, 1, box, NULL);
    object_add(ED_LAMP, 7, -1, vec3(-240, 0, 200), 1, 0, 0, NULL);
    object_add(ED_SPHERE, 1, -1, vec3(-120, 0, -40), 1, 1, 0, NULL);
    object_add(ED_SPHERE, 6, -1, vec3(170, 0, -110), 1, 1, 0, "Glass ball");
    s->cam_from = vec3(0, 240, -680);
    s->cam_at = vec3(0, 50, 100);
    s->cam_up = vec3(0, 1, 0);
    s->cam_vfov_deg = 40;
    return 0;
}

// ---- the camera: orbit / pan / dolly around a point; fly with the keys ----------------------

static void camera_from_pivot(View *v){
    v->pos = v_sub(E.pivot, v_scale(view_forward(v), (real)E.dist));
}

EXPORT void lr_camera_orbit(double dx, double dy){   // a drag, in pixels
    View v = viewer_view();
    v.yaw -= dx * 0.005;
    v.pitch = fmax(-1.45, fmin(0.35, v.pitch - dy * 0.005));
    camera_from_pivot(&v);
    viewer_set_view(&v);
}

EXPORT void lr_camera_pan(double dx, double dy){     // a drag, in pixels of the page's view
    View v = viewer_view();
    ViewerStatus st;
    viewer_status(&st);
    double k = 2 * E.dist * tan(v.fov * PI / 360) / (st.height > 0 ? st.height : 1);
    Vec3 r = view_right(&v), u = v_cross(r, view_forward(&v));
    E.pivot = v_sub(E.pivot, v_add(v_scale(r, (real)(dx * k)), v_scale(u, (real)(-dy * k))));
    camera_from_pivot(&v);
    viewer_set_view(&v);
}

// an object dragged past the frame's edge (by dx, dy pixels of the page's view): the camera
// slides over the ground, its height kept, so that the point p (the object, at its depth)
// moves on the screen by about that much and the drag can go on as far as the mouse goes.
// Sideways along the view's right, up and down along its forward (both level); up and
// down a ground point moves less on the screen than it does on the ground (seen at a
// slant), so the slide there is longer by 1 / sin of the angle the ray meets it at.
EXPORT void lr_camera_slide(double dx, double dy, double x, double y, double z){
    View v = viewer_view();
    ViewerStatus st;
    viewer_status(&st);
    Vec3 f = view_forward(&v), to = v_sub(vec3((real)x, (real)y, (real)z), v.pos);
    double depth = fmax(30, v_dot(to, f));
    double k = 2 * depth * tan(v.fov * PI / 360) / (st.height > 0 ? st.height : 1);
    Vec3 r = view_right(&v);
    r.y = 0;
    f.y = 0;
    if (v_dot(r, r) < 1e-8f || v_dot(f, f) < 1e-8f) return;
    r = v_norm(r);
    f = v_norm(f);
    double slant = fmax(0.2, fabs(v_norm(to).y));
    Vec3 mv = v_add(v_scale(r, (real)(dx * k)), v_scale(f, (real)(-dy * k / slant)));
    E.pivot = v_add(E.pivot, mv);
    v.pos = v_add(v.pos, mv);
    viewer_set_view(&v);
}

EXPORT void lr_camera_dolly(double steps){           // wheel steps, + towards
    View v = viewer_view();
    E.dist = fmax(30, fmin(20000, E.dist * pow(0.88, steps)));
    camera_from_pivot(&v);
    viewer_set_view(&v);
}

// flying (the right mouse button held): the keys held (1 W, 2 A, 4 S, 8 D, 16 E, 32 Q,
// 64 Shift) move the camera every frame, the mouse turns it
EXPORT void lr_camera_fly(int keys){ E.fly_keys = keys; }
EXPORT void lr_camera_look(double dx, double dy){
    View v = viewer_view();
    v.yaw -= dx * 0.004;
    v.pitch -= dy * 0.004;
    E.pivot = v_add(v.pos, v_scale(view_forward(&v), (real)E.dist));
    viewer_set_view(&v);
}

static void walk_step(double dt);
static void fly_step(double dt){
    if (E.walk){ walk_step(dt); return; }
    if (!E.fly_keys) return;
    View v = viewer_view();
    Vec3 f = view_forward(&v), r = view_right(&v), mv = vec3(0, 0, 0);
    if (E.fly_keys & 1) mv = v_add(mv, f);
    if (E.fly_keys & 4) mv = v_sub(mv, f);
    if (E.fly_keys & 8) mv = v_add(mv, r);
    if (E.fly_keys & 2) mv = v_sub(mv, r);
    if (E.fly_keys & 16) mv = v_add(mv, vec3(0, 1, 0));
    if (E.fly_keys & 32) mv = v_sub(mv, vec3(0, 1, 0));
    if (v_dot(mv, mv) <= 0) return;
    double speed = 300 * (E.fly_keys & 64 ? 4 : 1) * dt;
    v.pos = v_add(v.pos, v_scale(v_norm(mv), (real)speed));
    E.pivot = v_add(v.pos, v_scale(view_forward(&v), (real)E.dist));
    viewer_set_view(&v);
}

// Play: the camera walks. W A S D along the ground (the view's forward and right, level),
// Shift faster, Space jumps; the eye EYE above whatever it stands on (the ground: 0), falling
// back under gravity.
#define EYE 165.0
static void walk_step(double dt){
    View v = viewer_view();
    Vec3 f = view_forward(&v), r = view_right(&v), mv = vec3(0, 0, 0);
    f.y = 0; r.y = 0;
    if (v_dot(f, f) > 1e-8f) f = v_norm(f);
    if (v_dot(r, r) > 1e-8f) r = v_norm(r);
    if (E.fly_keys & 1) mv = v_add(mv, f);
    if (E.fly_keys & 4) mv = v_sub(mv, f);
    if (E.fly_keys & 8) mv = v_add(mv, r);
    if (E.fly_keys & 2) mv = v_sub(mv, r);
    Vec3 was = v.pos;
    if (v_dot(mv, mv) > 0) v.pos = v_add(v.pos, v_scale(v_norm(mv), (real)(240 * (E.fly_keys & 64 ? 2.5 : 1) * dt)));
    int ground = v.pos.y <= EYE + 0.5;
    if (ground && (E.fly_keys & 128)) E.walk_vy = 420;               // (a jump of about 0.9 m)
    E.walk_vy -= 981 * dt;
    v.pos.y = (real)(v.pos.y + E.walk_vy * dt);
    if (v.pos.y < EYE){ v.pos.y = (real)EYE; E.walk_vy = 0; }
    if (!memcmp(&was, &v.pos, sizeof(was))) return;
    E.pivot = v_add(v.pos, v_scale(view_forward(&v), (real)E.dist));
    viewer_set_view(&v);
}
EXPORT void lr_camera_walk(int on){                  // Play: walk (1, from eye height) or fly (0)
    E.walk = on != 0;
    E.walk_vy = 0;
    if (!E.walk) return;
    View v = viewer_view();
    v.pos.y = (real)EYE;
    E.pivot = v_add(v.pos, v_scale(view_forward(&v), (real)E.dist));
    viewer_set_view(&v);
}
EXPORT void lr_camera_save(void){                    // the view now, to come back to
    E.saved_view = viewer_view();
    E.saved_pivot = E.pivot;
    E.saved_dist = E.dist;
    E.have_saved = 1;
}
EXPORT void lr_camera_restore(void){
    if (!E.have_saved) return;
    E.pivot = E.saved_pivot;
    E.dist = E.saved_dist;
    viewer_set_view(&E.saved_view);
}

// ---- what the page sets ---------------------------------------------------------------------

// at most fps frames a second (0: the display's). A page cannot turn vsync off (the browser
// shows its frames in step with the display); fewer frames give the path tracer more of the
// GPU each: at 60 on a 165 Hz screen a frame has time for a whole pass, and the image follows
// the camera in every frame shown. The frames in between are skipped (nothing drawn).
static double frame_cap_ms;
#ifdef __EMSCRIPTEN__
static double frame_last_ms;
#endif
EXPORT void lr_set_frame_cap(int fps){
    fps = fps < 0 ? 0 : fps > 360 ? 360 : fps;
    frame_cap_ms = fps > 0 ? 1000.0 / fps : 0;
    viewer_set_max_hz(fps);
}
EXPORT void lr_set_backend(int b){                   // 0 path traced, 1 raster, 2 auto (Backend)
    if (b >= BACKEND_RT && b <= BACKEND_HYBRID) viewer_set_backend((Backend)b);
}

EXPORT int lr_set_resolution(int w, int h){          // 0 = done
    if (w < 64 || h < 64 || w > 7680 || h > 4320) return 1;
    return viewer_resize(w, h);
}

// ---- the objects: what the page adds, moves, sizes, removes (positions in metres) --------------

// ---- relighting: the rasterizer's light probe made again after every change, in steps between
// the frames (rt/probe.h, probe_job_*): the page keeps drawing; the old light shows until
// the new one is done ----------------------------------------------------------------------

#define RELIGHT_MS 6.0     // of each frame: with the frame itself under a 60 Hz screen's 16.7 ms

static void relight_start(void){
    probe_job_free(E.relight);
    E.relight = NULL;
    E.relight_done = 1;
    Options *o = viewer_options();
    Scene *s = viewer_scene();
    if (!o->probe) return;
    scene_update_instances(s, &o->ap);              // (the bake traces the scene as it is now)
    E.relight = probe_job_start(s, o->probe_res, o->probe_spp, PROBE_GRID_PATHS);
    E.relight_done = 0;
}

static void relight_step(void){
    if (!E.relight) return;
    Scene *s = viewer_scene();
    scene_update_instances(s, &viewer_options()->ap);
    E.relight_done = probe_job_step(E.relight, s, RELIGHT_MS);
    if (E.relight_done < 1) return;
    Env *p = NULL;
    IrrGrid *g = NULL;
    int r = probe_job_finish(E.relight, &p, &g);
    if (r == 1) return;                             // (not yet: it says it is all done only when it is)
    E.relight = NULL;
    if (r == 0){
        env_free(s->probe);
        s->probe = p;
        irr_grid_free(s->irr);
        s->irr = g;
        viewer_redraw();
    }
}

// a change made: what shows follows (the path traced image keeps what did not change), the
// light is made again
static void changed(void){
    viewer_scene_edited();
    relight_start();
}

// shapes as the tray has them (the spec, 7.1), with their default materials (6.4)
enum { SHAPE_BOX, SHAPE_SPHERE, SHAPE_GLASS_BALL, SHAPE_ROCK, SHAPE_LAMP, NSHAPES };

static Vec3 metres(double x, double y, double z){
    return vec3((real)(x * UNITS_PER_M), (real)(y * UNITS_PER_M), (real)(z * UNITS_PER_M));
}

// a shape standing at (x, y, z) (the middle of its bottom), on `support` (0: the ground);
// static, at its own size. Returns its id (0: none).
// ---- placement: a new object never inside what is there ------------------------------------
// (A shape or model put on the ground next to another landed half inside it.) Boxes in world
// units, a centimetre smaller each way: touching is not inside.
static int boxes_overlap(Aabb a, Aabb b){
    const real e = 1;
    return a.lo.x + e < b.hi.x && b.lo.x + e < a.hi.x && a.lo.y + e < b.hi.y && b.lo.y + e < a.hi.y &&
           a.lo.z + e < b.hi.z && b.lo.z + e < a.hi.z;
}
// the box against every object but `skip` (0: none)
static int box_free(Aabb b, int skip){
    for (int i = 0; i < E.nobjs; i++)
        if (E.objs[i].id != skip && boxes_overlap(b, object_box(&E.objs[i]))) return 0;
    return 1;
}
// The nearest way along the ground (x, z) to move box b so that it is free: rings round where
// it is, a step apart (a quarter of its smaller side, at least 10 cm), 8 more points on each
// ring than the one before, up to 40 rings. (0, 0, 0) if it is free already or nothing is.
static Vec3 free_shift(Aabb b, int skip){
    if (box_free(b, skip)) return vec3(0, 0, 0);
    real step = r_fmax(10, 0.25f * r_fmin(b.hi.x - b.lo.x, b.hi.z - b.lo.z));
    for (int k = 1; k <= 40; k++){
        const int n = 8 * k;
        for (int j = 0; j < n; j++){
            real a = (real)(6.283185307179586 * j / n);
            Vec3 d = vec3(r_cos(a) * step * k, 0, r_sin(a) * step * k);
            Aabb m = {v_add(b.lo, d), v_add(b.hi, d)};
            if (box_free(m, skip)) return d;
        }
    }
    return vec3(0, 0, 0);
}
// a new object on the ground (support 0) moved off what it went into
static void place_free(int id){
    EdObject *o = object_find(id);
    if (!o || o->support) return;
    Vec3 d = free_shift(object_box(o), id);
    if (d.x != 0 || d.z != 0) object_shift(id, d, 0);
}
EXPORT int lr_add_shape(int shape, double x, double y, double z, int support){
    static const struct { EdKind kind; int mat; const char *label; } SH[NSHAPES] = {
        {ED_BOX, 0, NULL}, {ED_SPHERE, 1, NULL}, {ED_SPHERE, 6, "Glass ball"}, {ED_ROCK, 2, NULL}, {ED_LAMP, 7, NULL}};
    if (shape < 0 || shape >= NSHAPES || editing_off()) return 0;
    if (support && !object_find(support)) support = 0;
    int id = object_add(SH[shape].kind, SH[shape].mat, -1, metres(x, y, z), 1, 0, support, SH[shape].label);
    place_free(id);
    changed();
    return id;
}

// moved to stand at (x, y, z) on `support` (0: the ground), what rests on it with it; not onto
// itself or what rests on it. 0 = done.
EXPORT int lr_move_object(int id, double x, double y, double z, int support){
    if (editing_off()) return 1;
    EdObject *o = object_find(id);
    if (!o || (support && (!object_find(support) || in_stack(support, id)))) return 1;
    o->support = support;
    object_shift(id, v_sub(metres(x, y, z), o->pos), 0);
    changed();
    return 0;
}

// its size (1: its own), what rests on it raised or lowered with its top
EXPORT int lr_scale_object(int id, double scale){
    if (editing_off()) return 1;
    EdObject *o = object_find(id);
    if (!o || !(scale >= 0.05 && scale <= 20)) return 1;
    real top = object_top(o);
    o->scale = (real)scale;
    object_update(o);
    Vec3 d = vec3(0, object_top(o) - top, 0);
    for (int i = 0; i < E.nobjs; i++) if (E.objs[i].support == id) object_shift(E.objs[i].id, d, 0);
    changed();
    return 0;
}

// the gizmo (Move, Rotate, Scale): its place (the middle of its bottom, metres), its turn (a
// quaternion w x y z) and its stretch along its own axes, all at once. What rests on it comes
// along, as one with it (moved and turned with it, raised or lowered with its top). Lifted
// or lowered (by more than a centimetre) it no longer rests on what it stood on.
static void object_carry(int id, Vec3 from, Vec3 to, Quat turn, real lift, int depth){
    for (int i = 0; i < E.nobjs && depth < 1000; i++){
        EdObject *c = &E.objs[i];
        if (c->support != id) continue;
        Vec3 p = v_add(v_add(to, quat_rotate(turn, v_sub(c->pos, from))), vec3(0, lift, 0));
        Vec3 was = c->pos;
        c->pos = p;
        c->rot = quat_norm(quat_mul(turn, c->rot));
        object_update(c);
        object_carry(c->id, was, p, turn, 0, depth + 1);
    }
}
EXPORT int lr_set_transform(int id, double x, double y, double z, double qw, double qx, double qy, double qz,
                            double sx, double sy, double sz){
    if (editing_off()) return 1;
    EdObject *o = object_find(id);
    if (!o || !(sx > 0.01 && sy > 0.01 && sz > 0.01 && sx < 100 && sy < 100 && sz < 100)) return 1;
    Quat q = quat_norm(quat((real)qw, (real)qx, (real)qy, (real)qz));
    Quat turn = quat_mul(q, quat(o->rot.w, -o->rot.x, -o->rot.y, -o->rot.z));   // (the new turn after the old one undone)
    Vec3 from = o->pos;
    real top = object_top(o);
    o->pos = metres(x, y, z);
    o->rot = q;
    o->stretch = vec3((real)sx, (real)sy, (real)sz);
    object_update(o);
    if (r_fabs(o->pos.y - from.y) > 1) o->support = 0;
    // (the children: as one with it; a change of its size raises or lowers them with its top,
    // when it is not turned at the same time)
    real lift = r_fabs(turn.w) > 0.999999f ? object_top(o) - top : 0;     // (the move itself is in from -> pos)
    object_carry(id, from, o->pos, turn, lift, 0);
    changed();
    return 0;
}

// its look as JSON: {"rgb", "roughness", "metallic", "image" (the base colour's file),
// "maps" {slot: file}, "bump", "normal", "ao", "glow", "flipGreen"} (names: made safe for
// JSON when they were set)
static void look_json(char *b, size_t n, const EdObject *o){
    size_t k = (size_t)snprintf(b, n, "{\"rgb\":\"#%06X\",\"roughness\":%.3f,\"metallic\":%.3f,\"image\":\"%s\",\"maps\":{",
                                o->look_rgb, (double)o->look_rough, (double)o->look_metal, o->look_map[LOOK_BASE] ? o->look_map_name[LOOK_BASE] : "");
    int first = 1;
    for (int i = 0; i < NLOOK && k < n; i++){
        if (!o->look_map[i]) continue;
        k += (size_t)snprintf(b + k, n - k, "%s\"%s\":\"%s\"", first ? "" : ",", LOOK_KEYS[i], o->look_map_name[i]);
        first = 0;
    }
    if (k < n) snprintf(b + k, n - k, "},\"bump\":%.3f,\"normal\":%.3f,\"ao\":%.3f,\"glow\":%.3f,\"flipGreen\":%d}",
                        (double)(o->look_bump > 0 ? o->look_bump : 1), (double)(o->look_normal_k > 0 ? o->look_normal_k : 1),
                        (double)(o->look_ao_k > 0 ? o->look_ao_k : 1), (double)(o->look_glow > 0 ? o->look_glow : 1), o->look_flip);
}

// What each texture the looks made holds (E.made: its sources then), so that a look made again
// (a colour dragged, an undo) makes its normal or its packed roughness again only when what
// they come from changed: a 2048 x 2048 bump takes a moment
typedef struct { int tex; int k[4]; real strength; } Made;
static Made *made_list;
static int nmade, cap_made;
static Made *made_of(int tex){
    for (int i = 0; i < nmade; i++) if (made_list[i].tex == tex) return &made_list[i];
    return NULL;
}
static void made_note(int tex, const int k[4], real strength){
    Made *m = made_of(tex);
    if (!m){ made_list = lr_grow(made_list, sizeof(Made), &cap_made, nmade + 1, 16); m = &made_list[nmade++]; m->tex = tex; }
    memcpy(m->k, k, sizeof(m->k));
    m->strength = strength;
}
static int made_same(int tex, const int k[4], real strength){
    const Made *m = tex > 0 ? made_of(tex) : NULL;
    return m && !memcmp(m->k, k, sizeof(m->k)) && m->strength == strength;
}

// its own material made (or made again) from its look fields. 1 = no room for another
// material in the scene (MAX_MATS)
static int look_sync(EdObject *o){
    Scene *s = viewer_scene();
    Material m = material_pbr(srgb_hex(o->look_rgb), o->look_metal, o->look_rough, 1.5, 0);
    const int *mp = o->look_map;
    PbrMapSet ms = {.base = mp[LOOK_BASE], .normal = mp[LOOK_NORMAL], .bump = mp[LOOK_BUMP], .rough = mp[LOOK_ROUGH],
                    .metal = mp[LOOK_METAL], .mr = mp[LOOK_ORM], .ao = mp[LOOK_AO], .emissive = mp[LOOK_EMISSIVE],
                    .bump_strength = o->look_bump > 0 ? o->look_bump : 1, .flip_green = o->look_flip};
    // (what was made already, from the same sources: taken as it is)
    const int nk[4] = {mp[LOOK_NORMAL], mp[LOOK_BUMP], o->look_flip, 0}, mk[4] = {mp[LOOK_ROUGH], mp[LOOK_METAL], 0, 0};
    const int need_n = mp[LOOK_BUMP] || (mp[LOOK_NORMAL] && o->look_flip), need_mr = mp[LOOK_ROUGH] || mp[LOOK_METAL];
    if (need_n && made_same(o->look_made_normal, nk, ms.bump_strength)){ ms.normal = o->look_made_normal; ms.bump = 0; ms.flip_green = 0; }
    if (need_mr && made_same(o->look_made_mr, mk, 0)){ ms.mr = o->look_made_mr; ms.rough = ms.metal = 0; }
    int made_n = o->look_made_normal, made_mr = o->look_made_mr;
    pbr_apply(s, &m, &ms, &made_n, &made_mr);
    if (ms.bump || ms.flip_green){ o->look_made_normal = made_n; made_note(made_n, nk, ms.bump_strength); }
    if (ms.rough || ms.metal){ o->look_made_mr = made_mr; made_note(made_mr, mk, 0); }
    m.normal_scale = o->look_normal_k > 0 ? o->look_normal_k : 1;
    m.occlusion = o->look_ao_k > 0 ? o->look_ao_k : 1;
    if (m.tex_emissive){ real g = o->look_glow > 0 ? o->look_glow : 1; m.emission = vec3(g, g, g); }
    if (o->look < 0){
        if (s->nmats >= MAX_MATS) return 1;
        o->look = scene_add_material(s, m);
    } else s->mats[o->look] = m;
    return 0;
}
// the look starts from what it is made of now (a preset's colour, roughness, metallic)
static void look_from_now(EdObject *o){
    if (o->mat == MAT_OWN) return;
    const EdMaterial *d = &MATERIALS[o->mat >= 0 && o->mat < 6 ? o->mat : 0];    // (glass, glow: from wood)
    o->look_rgb = d->rgb;
    o->look_rough = d->roughness;
    o->look_metal = d->metallic;
}
// Its own material (the card's Own): base colour rgb (sRGB 0xRRGGBB), roughness and metallic
// (0..1); its image, if it has one, stays (the colour then tints it). 0 = done, 1 = no such
// object, 2 = no room for another material
EXPORT int lr_set_look(int id, int rgb, double roughness, double metallic){
    EdObject *o = object_find(id);
    if (!o || editing_off()) return 1;
    EdObject was = *o;
    o->look_rgb = (unsigned)rgb & 0xffffffu;
    o->look_rough = (real)fmin(1, fmax(0, roughness));
    o->look_metal = (real)fmin(1, fmax(0, metallic));
    if (look_sync(o)){ *o = was; return 2; }
    o->mat = MAT_OWN;
    object_update(o);
    changed();
    return 0;
}
// A map (.png, .jpg: a file the page put in its file system) for one of its look's slots (0
// base colour ... 7 emissive: LOOK_KEYS), on its own material (from its material now; a base
// colour image makes the colour white, so it shows as it is; a roughness, metallic or packed
// map makes those factors 1, so the map's values are as they are); "" takes the map off.
// 0 = done, 1 = no such object, slot or image, 2 = no room for another material
EXPORT int lr_set_look_map(int id, int slot, const char *path){
    EdObject *o = object_find(id);
    if (!o || editing_off() || slot < 0 || slot >= NLOOK) return 1;
    EdObject was = *o;
    int t = 0;
    const int colour = slot == LOOK_BASE || slot == LOOK_EMISSIVE;   // (sRGB; the rest are data)
    if (path && *path && !(t = scene_texture(viewer_scene(), path, colour))) return 1;
    if (o->mat != MAT_OWN) look_from_now(o);
    if (t && !o->look_map[slot]){
        if (slot == LOOK_BASE) o->look_rgb = 0xffffff;
        if (slot == LOOK_ROUGH || slot == LOOK_ORM) o->look_rough = 1;
        if (slot == LOOK_METAL || slot == LOOK_ORM) o->look_metal = 1;
    }
    // a metallic map taken off (and no packed one, or the other way round): not metal then
    // (its 1 was the map's, which said where)
    if (!t && o->look_map[slot] && ((slot == LOOK_METAL && !o->look_map[LOOK_ORM]) || (slot == LOOK_ORM && !o->look_map[LOOK_METAL])))
        o->look_metal = 0;
    o->look_map[slot] = t;
    const char *f = path ? strrchr(path, '/') : NULL;
    char *nm = o->look_map_name[slot];
    snprintf(nm, sizeof(o->look_map_name[slot]), "%s", t ? (f ? f + 1 : path) : "");
    for (char *c = nm; *c; c++) if (*c == '"' || *c == '\\' || (unsigned char)*c < 32) *c = '_';   // (it goes into JSON)
    if (look_sync(o)){ *o = was; return 2; }
    o->mat = MAT_OWN;
    object_update(o);
    changed();
    return 0;
}
// (the base colour's: as before)
EXPORT int lr_set_look_image(int id, const char *path){ return lr_set_look_map(id, LOOK_BASE, path); }
// The maps' strengths: the bump's (pbr_bump_to_normal's, 0 = 1), the normal map's (its tilt
// times k), the occlusion's (0..1), the glow's brightness (an emissive map's white), and
// whether the normal map's green points down (DirectX's). 0 = done
EXPORT int lr_set_look_strengths(int id, double bump, double normal_k, double ao_k, double glow, int flip){
    EdObject *o = object_find(id);
    if (!o || editing_off() || o->mat != MAT_OWN) return 1;
    o->look_bump = (real)fmin(32, fmax(0.01, bump));
    o->look_normal_k = (real)fmin(4, fmax(0.01, normal_k));
    o->look_ao_k = (real)fmin(1, fmax(0.01, ao_k));
    o->look_glow = (real)fmin(1000, fmax(0.01, glow));
    o->look_flip = flip != 0;
    look_sync(o);
    object_update(o);
    changed();
    return 0;
}

// static or dynamic (its body in Simulate mode)
// its material: one of the catalogue's (MATERIALS, 0..7), or -1: a model's own again
EXPORT int lr_set_material(int id, int mat){
    EdObject *o = object_find(id);
    if (!o || mat < -1 || mat >= NMATERIALS || (mat < 0 && o->kind != ED_MODEL)) return 1;
    o->mat = mat;
    object_update(o);
    changed();
    return 0;
}

// a glTF model (.gltf, .glb: the page puts the dropped files into its file system first) at its
// real size (glTF's metres), standing at (x, y, z) on `support`. Returns its id (0: it could
// not be read).
EXPORT int lr_import_gltf(const char *path, double x, double y, double z, int support){
    if (editing_off()) return 0;
    Scene *s = viewer_scene();
    int n0 = s->ninsts;
    int n = scene_load_gltf(s, path, (real)-UNITS_PER_M, vec3(0, 0, 0), 0);
    if (n <= 0) return 0;
    // its instances, as the file places them standing at the origin: the model's parts; they
    // become the object's (made again from the parts: the same, in the object's place)
    E.models = lr_grow(E.models, sizeof(EdModel), &E.cap_models, E.nmodels + 1, 4);
    EdModel *m = &E.models[E.nmodels];
    m->nparts = s->ninsts - n0;
    m->clay = 0;                                     // (its own materials: the slot is not cleared)
    m->parts = malloc((size_t)m->nparts * sizeof(EdPart));
    if (!m->parts) lr_oom("editor");
    m->box = (Aabb){vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
    for (int i = 0; i < m->nparts; i++){
        const Instance *in = &s->insts[n0 + i];
        m->parts[i] = (EdPart){in->mesh, in->material, in->xf};
        m->box.lo = vec3(r_fmin(m->box.lo.x, in->box.lo.x), r_fmin(m->box.lo.y, in->box.lo.y), r_fmin(m->box.lo.z, in->box.lo.z));
        m->box.hi = vec3(r_fmax(m->box.hi.x, in->box.hi.x), r_fmax(m->box.hi.y, in->box.hi.y), r_fmax(m->box.hi.z, in->box.hi.z));
    }
    while (s->ninsts > n0) scene_remove_instance(s, s->ninsts - 1);
    if (support && !object_find(support)) support = 0;
    const char *file = strrchr(path, '/');
    int id = object_add(ED_MODEL, -1, E.nmodels++, metres(x, y, z), 1, 0, support, file ? file + 1 : path);
    place_free(id);
    changed();
    return id;
}

// a mesh file (.obj, ASCII .ply) as an object of one part, of clay (the file has no material).
// Its units are taken as metres; one under 1 cm or over 100 m on its longest side would make
// no sense: it is fitted to 1 m (lr_last_import_fitted says so). Returns its id (0: it could
// not be read).
EXPORT int lr_import_mesh(const char *path, double x, double y, double z, int support){
    if (editing_off()) return 0;
    Mesh *mesh = mesh_load(path);
    if (!mesh) return 0;
    Vec3 lo = vec3(1e30f, 1e30f, 1e30f), hi = vec3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < mesh->nv; i++){
        Vec3 v = mesh->v[i];
        lo = vec3(r_fmin(lo.x, v.x), r_fmin(lo.y, v.y), r_fmin(lo.z, v.z));
        hi = vec3(r_fmax(hi.x, v.x), r_fmax(hi.y, v.y), r_fmax(hi.z, v.z));
    }
    real ext = r_fmax(hi.x - lo.x, r_fmax(hi.y - lo.y, hi.z - lo.z));
    E.last_fitted = !(ext >= 0.01f && ext <= 100);
    if (mesh->nv == 0 || !(ext > 0)){ mesh_destroy(mesh); return 0; }
    mesh_place(mesh, (real)UNITS_PER_M * (E.last_fitted ? 1 : ext), vec3(0, 0, 0), 0);   // (its own size: metres in units)
    Scene *s = viewer_scene();
    int mi = scene_add_mesh_object(s, mesh, E.clay);
    mesh_destroy(mesh);
    if (mi < 0) return 0;
    E.models = lr_grow(E.models, sizeof(EdModel), &E.cap_models, E.nmodels + 1, 4);
    EdModel *m = &E.models[E.nmodels];
    m->nparts = 1;
    m->parts = malloc(sizeof(EdPart));
    if (!m->parts) lr_oom("editor");
    m->parts[0] = (EdPart){mi, -1, xform_identity()};
    m->box = s->meshes[mi].box;
    m->clay = 1;
    if (support && !object_find(support)) support = 0;
    const char *file = strrchr(path, '/');
    int id = object_add(ED_MODEL, -1, E.nmodels++, metres(x, y, z), 1, 0, support, file ? file + 1 : path);
    place_free(id);
    changed();
    return id;
}
EXPORT int lr_last_import_fitted(void){ return E.last_fitted; }

// the sky: a preset (0 Dusk, 1 Noon, 2 Overcast), or a Radiance .hdr file (name: what the
// page shows); the path traced image starts over. 0 = done.
EXPORT int lr_set_sky(int preset){
    if (E.sample || set_sky(preset, NULL, NULL)) return 1;
    viewer_scene_changed(1);
    relight_start();
    return 0;
}
// the Light tab: the sun sky (SunSky) as these say; quick = 1 while a slider is dragged (a
// smaller map, quicker to make), 0 when let go. 0 = done.
EXPORT int lr_set_sun(double t, double intensity, double size, double haze, double az, int warm, int quick){
    if (E.sample) return 1;
    E.sun = (SunSky){(real)fmin(fmax(t, 0), 1), (real)fmax(intensity, 0), (real)fmin(fmax(size, 0.05), 10),
                     (real)fmin(fmax(haze, 1), 10), (real)az, warm != 0};
    E.sun_w = quick ? 256 : 1024;
    if (set_sky(SKY_SUN, NULL, NULL)) return 1;
    viewer_scene_changed(1);
    relight_start();
    return 0;
}

EXPORT int lr_load_sky(const char *path, const char *name){
    if (E.sample || set_sky(SKY_FILE, path, name)) return 1;
    viewer_scene_changed(1);
    relight_start();
    return 0;
}

EXPORT int lr_set_body(int id, int dynamic){
    if (editing_off()) return 1;
    EdObject *o = object_find(id);
    if (!o) return 1;
    o->dynamic = dynamic != 0;
    return 0;
}

// removed; what rested on it comes down to where it stood, onto what it stood on
EXPORT int lr_remove_object(int id){
    if (editing_off()) return 1;
    EdObject *o = object_find(id);
    if (!o) return 1;
    Vec3 base = o->pos;
    int under = o->support;
    for (int i = 0; i < E.nobjs; i++)
        if (E.objs[i].support == id){
            E.objs[i].support = under;
            object_shift(E.objs[i].id, vec3(0, base.y - E.objs[i].pos.y, 0), 0);
        }
    o = object_find(id);
    object_drop_instances(o);
    int k = (int)(o - E.objs);
    memmove(&E.objs[k], &E.objs[k + 1], (size_t)(E.nobjs - k - 1) * sizeof(EdObject));
    E.nobjs--;
    changed();
    return 0;
}

// ---- undo / redo: the objects as they were before each change -------------------------------

typedef struct { EdObject *objs; int n; int sky; SunSky sun; char sky_name[96], sky_path[256]; } Snapshot;
#define HISTORY 100
static Snapshot undo_st[HISTORY], redo_st[HISTORY];
static int nundo, nredo;

static Snapshot snapshot_take(void){
    Snapshot sn;
    memset(&sn, 0, sizeof(sn));
    sn.objs = malloc((size_t)(E.nobjs > 0 ? E.nobjs : 1) * sizeof(EdObject));
    sn.n = E.nobjs;
    if (!sn.objs) lr_oom("editor");
    memcpy(sn.objs, E.objs, (size_t)E.nobjs * sizeof(EdObject));
    sn.sky = E.sky;
    sn.sun = E.sun;
    memcpy(sn.sky_name, E.sky_name, sizeof(sn.sky_name));
    memcpy(sn.sky_path, E.sky_path, sizeof(sn.sky_path));
    return sn;
}
static void snapshot_push(Snapshot *st, int *n, Snapshot sn){
    if (*n == HISTORY){ free(st[0].objs); memmove(st, st + 1, (HISTORY - 1) * sizeof(Snapshot)); (*n)--; }
    st[(*n)++] = sn;
}

// the objects made what the snapshot has: those not in it removed, those back made again,
// the rest moved, sized, coloured as it says (their instances kept: their samples carry on)
static void snapshot_apply(const Snapshot *sn){
    for (int i = E.nobjs - 1; i >= 0; i--){
        int keep = 0;
        for (int j = 0; j < sn->n && !keep; j++) keep = sn->objs[j].id == E.objs[i].id;
        if (keep) continue;
        object_drop_instances(&E.objs[i]);
        memmove(&E.objs[i], &E.objs[i + 1], (size_t)(E.nobjs - i - 1) * sizeof(EdObject));
        E.nobjs--;
    }
    EdObject *now = malloc((size_t)(sn->n > 0 ? sn->n : 1) * sizeof(EdObject));
    if (!now) lr_oom("editor");
    for (int j = 0; j < sn->n; j++){
        now[j] = sn->objs[j];
        const EdObject *cur = object_find(now[j].id);
        now[j].inst = NULL;                      // (the snapshot's are not its: its instances are the live object's)
        now[j].ninst = 0;
        if (cur){ now[j].inst = cur->inst; now[j].ninst = cur->ninst; }
    }
    free(E.objs);
    E.objs = now;
    E.cap = sn->n > 0 ? sn->n : 1;
    E.nobjs = sn->n;
    for (int j = 0; j < E.nobjs; j++){
        if (E.objs[j].mat == MAT_OWN) look_sync(&E.objs[j]);     // (its own material as it was then)
        if (E.objs[j].ninst == 0) object_instances(&E.objs[j]);
        else object_update(&E.objs[j]);
    }
    if (sn->sky != E.sky || strcmp(sn->sky_path, E.sky_path) || (sn->sky == SKY_SUN && memcmp(&sn->sun, &E.sun, sizeof(SunSky)))){
        // (another sky: the path traced image starts over)
        E.sun = sn->sun;
        E.sun_w = 0;
        if (set_sky(sn->sky, sn->sky == SKY_FILE ? sn->sky_path : NULL, sn->sky_name) == 0) viewer_scene_changed(1);
    }
    changed();
}

// before a change (an add, a move, a size, a material, a body, a removal): the scene as it is
EXPORT void lr_checkpoint(void){
    if (editing_off()) return;
    snapshot_push(undo_st, &nundo, snapshot_take());
    while (nredo > 0) free(redo_st[--nredo].objs);
}
// the last checkpoint, when nothing changed after it after all (a press without a drag)
EXPORT void lr_forget_checkpoint(void){
    if (nundo > 0) free(undo_st[--nundo].objs);
}
EXPORT int lr_undo(void){
    if (editing_off()) return 0;
    if (nundo == 0) return 0;
    snapshot_push(redo_st, &nredo, snapshot_take());
    Snapshot sn = undo_st[--nundo];
    snapshot_apply(&sn);
    free(sn.objs);
    return 1;
}
EXPORT int lr_redo(void){
    if (editing_off()) return 0;
    if (nredo == 0) return 0;
    snapshot_push(undo_st, &nundo, snapshot_take());
    Snapshot sn = redo_st[--nredo];
    snapshot_apply(&sn);
    free(sn.objs);
    return 1;
}

// ---- the camera and the screen: picking, projecting, framing ---------------------------------

static Camera camera_now(void){
    View v = viewer_view();
    ViewerStatus st;
    viewer_status(&st);
    return camera_make(v.pos, v_add(v.pos, view_forward(&v)), vec3(0, 1, 0), (real)v.fov,
                       (real)(st.width > 0 ? st.width : 16) / (real)(st.height > 0 ? st.height : 9));
}

#define FLOOR_M 99.0     // (the floor reaches 100 m from the middle: a landing stays on it)

// ---- what the page reads: JSON text (valid until the next call) -----------------------------

static char *json_buf;
static size_t json_cap;
static size_t jlen;
static void jprintf(const char *fmt, ...){
    for (;;){
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(json_buf ? json_buf + jlen : NULL, json_buf ? json_cap - jlen : 0, fmt, ap);
        va_end(ap);
        if (n < 0) return;
        if (json_buf && jlen + (size_t)n < json_cap){ jlen += (size_t)n; return; }
        size_t want = (json_cap ? json_cap : 4096);
        while (want <= jlen + (size_t)n) want *= 2;
        char *b = realloc(json_buf, want);
        if (!b) lr_oom("editor");
        json_buf = b;
        json_cap = want;
    }
}

// the render's state: {"tracing":1,"gpu":1,...}
// a string for JSON: quotes and backslashes escaped, control characters dropped
static void jstring(const char *t){
    jprintf("\"");
    for (; *t; t++){
        if (*t == '"' || *t == '\\') jprintf("\\%c", *t);
        else if ((unsigned char)*t >= 32) jprintf("%c", *t);
    }
    jprintf("\"");
}

// ---- the editor's own simulation (the Simulate tab): its objects as rigid bodies -----------
// Play makes a world of them: a box a box, a sphere a ball (stretched: its hull), anything
// else (a lamp, a rock, a model) the convex hull of its parts, or, when it is static, its
// triangles. Static objects stay where they are; dynamic ones fall, roll and stack; the
// floor and the ground's edge (FLOOR_M) are the room's walls. Steps of a fixed SIM_DT, as
// many as the time that went by asks (at most SIM_MAX_STEPS a frame): the same scene gives
// the same motion at any frame rate. While a simulation is made the objects are its: the
// editing calls do nothing until Reset (back to the scene before Play) or Keep (its pose
// kept, one undo step).
#define SIM_DT (1.0 / 120)
#define SIM_MAX_STEPS 8
#define SIM_DENSITY 1.0f       // g / cm^3: everything weighs like water (the materials don't say)

typedef struct { int obj;        // the object's id
                 int body;       // its body
                 int hull;       // a hull or mesh body (its pose: body_mesh_xform), else a shape
                 Vec3 c;         // a shape's middle in the object's own frame, sized
               } SimLink;
static struct {
    int on, paused;
    PhysWorld pw;
    SimLink *links;
    int nlinks;
    Snapshot before;           // the scene at Play (Reset: back to it)
    double acc;                // time not stepped yet
    long steps;
} S;

// the CPU's time per part of the editor's frames (ms, summed: lr_profile)
static struct { double sim, apply, relight, viewer, frame; long frames, steps; } P;

static Quat quat_of_mat3(Mat3 m){              // (a rotation's quaternion: Shepperd's way)
    real tr = m.m[0][0] + m.m[1][1] + m.m[2][2];
    Quat q;
    if (tr > 0){
        real k = 0.5f / r_sqrt(tr + 1);
        q = quat(0.25f / k, (m.m[2][1] - m.m[1][2]) * k, (m.m[0][2] - m.m[2][0]) * k, (m.m[1][0] - m.m[0][1]) * k);
    } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]){
        real k = 2 * r_sqrt(1 + m.m[0][0] - m.m[1][1] - m.m[2][2]);
        q = quat((m.m[2][1] - m.m[1][2]) / k, 0.25f * k, (m.m[0][1] + m.m[1][0]) / k, (m.m[0][2] + m.m[2][0]) / k);
    } else if (m.m[1][1] > m.m[2][2]){
        real k = 2 * r_sqrt(1 + m.m[1][1] - m.m[0][0] - m.m[2][2]);
        q = quat((m.m[0][2] - m.m[2][0]) / k, (m.m[0][1] + m.m[1][0]) / k, 0.25f * k, (m.m[1][2] + m.m[2][1]) / k);
    } else {
        real k = 2 * r_sqrt(1 + m.m[2][2] - m.m[0][0] - m.m[1][1]);
        q = quat((m.m[1][0] - m.m[0][1]) / k, (m.m[0][2] + m.m[2][0]) / k, (m.m[1][2] + m.m[2][1]) / k, 0.25f * k);
    }
    return quat_norm(q);
}

typedef struct { Triangle *t; int n, cap; } TriList;
static void tri_add(TriList *l, Vec3 a, Vec3 b, Vec3 c){
    l->t = lr_grow(l->t, sizeof(Triangle), &l->cap, l->n + 1, 64);
    Triangle *t = &l->t[l->n++];
    memset(t, 0, sizeof(*t));
    t->q = a;
    t->u = v_sub(b, a);
    t->v = v_sub(c, a);
    t->n[0] = t->n[1] = t->n[2] = t->t[0] = t->t[1] = t->t[2] = -1;
}
// the object's parts as triangles in its own frame, sized (scale and stretch), not turned
static TriList object_tris(const EdObject *o){
    TriList l = {0};
    const Scene *s = viewer_scene();
    Vec3 k = v_scale(o->stretch, o->scale);
    for (int i = 0; i < object_parts(o); i++){
        Xform L = object_part_local(o, i);
        #define AT(p) v_mul(xform_point(&L, (p)), k)
        int m = part_mesh(o, i);
        if (m == INST_BOX){
            static const int F[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
            Vec3 c[8];
            for (int j = 0; j < 8; j++) c[j] = AT(vec3(j & 4 ? 1 : -1, j & 2 ? 1 : -1, j & 1 ? 1 : -1));
            for (int f = 0; f < 6; f++){ tri_add(&l, c[F[f][0]], c[F[f][1]], c[F[f][2]]); tri_add(&l, c[F[f][0]], c[F[f][2]], c[F[f][3]]); }
        } else if (m == INST_SPHERE){
            enum { LAT = 8, LON = 12 };
            for (int a = 0; a < LAT; a++)
                for (int b = 0; b < LON; b++){
                    Vec3 p[4];
                    for (int q = 0; q < 4; q++){
                        real th = (real)PI * (a + (q >> 1)) / LAT, ph = 2 * (real)PI * (b + ((q ^ (q >> 1)) & 1)) / LON;
                        p[q] = AT(vec3(r_sin(th) * r_cos(ph), r_cos(th), r_sin(th) * r_sin(ph)));
                    }
                    tri_add(&l, p[0], p[1], p[2]);
                    tri_add(&l, p[0], p[2], p[3]);
                }
        } else if (m >= 0 && m < s->nmeshes){
            const SceneMesh *sm = &s->meshes[m];
            for (int j = 0; j < sm->ntris; j++){
                const Triangle *t = &sm->tris[j];
                tri_add(&l, AT(t->q), AT(v_add(t->q, t->u)), AT(v_add(t->q, t->v)));
            }
        }
        #undef AT
    }
    return l;
}

static void sim_free_world(void){
    phys_free(&S.pw);
    free(S.links);
    S.links = NULL;
    S.nlinks = 0;
}

// the world made of the objects as they are now
static void sim_build(void){
    memset(&S.pw, 0, sizeof(S.pw));
    phys_init(&S.pw);
    real edge = (real)(FLOOR_M * UNITS_PER_M);
    rigid_set_room(&S.pw.rigid, (Aabb){vec3(-edge, 0, -edge), vec3(edge, 1e7f, edge)});
    S.links = calloc((size_t)(E.nobjs > 0 ? E.nobjs : 1), sizeof(SimLink));
    if (!S.links) lr_oom("editor");
    RigidWorld *w = &S.pw.rigid;
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        Vec3 k = v_scale(o->stretch, o->scale);
        Mat3 R = quat_to_mat3(o->rot);
        SimLink L = {o->id, -1, 0, vec3(0, 0, 0)};
        int uniform = r_fabs(k.x - k.y) < 1e-4f * k.x && r_fabs(k.x - k.z) < 1e-4f * k.x;
        if (o->kind == ED_BOX){
            L.c = vec3(0, 50 * k.y, 0);
            L.body = rigid_add_box(w, v_add(o->pos, mat3_mul_vec(R, L.c)), v_scale(k, 50), o->rot, SIM_DENSITY);
        } else if (o->kind == ED_SPHERE && uniform){
            L.c = vec3(0, 50 * k.y, 0);
            L.body = rigid_add_sphere(w, v_add(o->pos, mat3_mul_vec(R, L.c)), 50 * k.x, SIM_DENSITY);
            if (L.body >= 0) w->bodies[L.body].rot = o->rot;
        } else {
            TriList t = object_tris(o);
            Xform place = xform_make(o->pos, R.m, 1);
            if (o->dynamic){
                int g = t.n ? rigid_geom_hull(w, t.t, t.n, 0) : -1;
                L.body = g >= 0 ? rigid_add_hull_body(w, g, place, SIM_DENSITY) : -1;
            } else {
                int g = t.n ? rigid_geom_mesh(w, t.t, t.n) : -1;
                L.body = g >= 0 ? rigid_add_mesh_collider(w, g, place) : -1;
            }
            L.hull = 1;
            free(t.t);
        }
        if (L.body < 0) continue;
        Body *b = &w->bodies[L.body];
        if (!o->dynamic){ b->inv_mass = 0; b->inv_inertia = vec3(0, 0, 0); }   // (fixed: nothing moves it)
        S.links[S.nlinks++] = L;
    }
}

// the dynamic objects where their bodies are (those at rest left as they are)
static void sim_apply(void){
    int moved = 0;
    for (int i = 0; i < S.nlinks; i++){
        const SimLink *L = &S.links[i];
        const Body *b = &S.pw.rigid.bodies[L->body];
        EdObject *o = object_find(L->obj);
        if (!o || !o->dynamic || b->inv_mass == 0) continue;
        Vec3 pos;
        Quat rot;
        if (L->hull){
            Xform x = body_mesh_xform(b);
            pos = vec3(x.m[0][3], x.m[1][3], x.m[2][3]);
            Mat3 r;
            for (int u = 0; u < 3; u++) for (int v = 0; v < 3; v++) r.m[u][v] = x.m[u][v];
            rot = quat_of_mat3(r);
        } else {
            rot = b->rot;
            pos = v_sub(b->pos, mat3_mul_vec(quat_to_mat3(rot), L->c));
        }
        if (!memcmp(&pos, &o->pos, sizeof(pos)) && !memcmp(&rot, &o->rot, sizeof(rot))) continue;
        o->pos = pos;
        o->rot = rot;
        o->support = 0;                              // (what it rests on now is the physics' to say)
        object_update(o);
        moved = 1;
    }
    if (moved) viewer_scene_changed(0);
}

static void sim_frame(double dt){
    if (!S.on || S.paused) return;
    S.acc += dt;
    int n = 0;
    while (S.acc >= SIM_DT && n < SIM_MAX_STEPS){ phys_step(&S.pw, (real)SIM_DT); S.acc -= SIM_DT; S.steps++; n++; }
    if (n == SIM_MAX_STEPS) S.acc = 0;               // (a slow frame: the simulation slows down, it doesn't pile up)
    P.steps += n;
    double t0 = clock_s();
    if (n) sim_apply();
    P.apply += (clock_s() - t0) * 1000;
}
static int sim_dynamic_count(void){
    int n = 0;
    for (int i = 0; i < E.nobjs; i++) n += E.objs[i].dynamic != 0;
    return n;
}
// editing: not while a sample shows, nor while a simulation holds the objects
static int editing_off(void){ return E.sample || S.on; }

static void sim_end(void){
    sim_free_world();
    S.on = S.paused = 0;
    S.acc = 0;
    S.steps = 0;
}
static void sim_reset(void){                         // back to the scene at Play
    if (!S.on) return;
    sim_end();
    snapshot_apply(&S.before);
    free(S.before.objs);
    S.before.objs = NULL;
}
static void sim_keep(void){                          // its pose kept: undo goes back to before Play
    if (!S.on) return;
    sim_end();
    snapshot_push(undo_st, &nundo, S.before);
    S.before.objs = NULL;
    while (nredo > 0) free(redo_st[--nredo].objs);
    changed();
}

// ---- the simulation (the Simulate tab): the physics demo's, when it runs (-sim on) ----------
// (a sample: the physics demo's; the editor's own scene: its simulation, made at the first Play)
EXPORT int lr_sim_play(int play){                    // 1 runs, 0 pauses; 0 = done
    if (E.sample){
        if (!viewer_sim_on()) return 1;
        viewer_sim_pause(!play);
        return 0;
    }
    if (!S.on){
        if (!play) return 0;
        S.before = snapshot_take();
        sim_build();
        S.on = 1;
        S.acc = 0;
        S.steps = 0;
        if (E.relight){ probe_job_free(E.relight); E.relight = NULL; }   // (the light is made again after)
    }
    S.paused = !play;
    return 0;
}
EXPORT int lr_sim_reset(void){                       // from t = 0, as it was set up; 0 = done
    if (E.sample) return viewer_sim_restart();
    sim_reset();
    return 0;
}
// (tests: the sample paused at simulated time t, and where its instances are then, as hex)
EXPORT void lr_sim_pause_at(double t){ if (E.sample) viewer_sim_pause_at(t); }
EXPORT const char *lr_sim_digest(void){
    jlen = 0;
    jprintf("%016llx", E.sample ? viewer_sim_digest() : 0ULL);
    return json_buf;
}
// a ball (blue rubber, 30 cm) thrown from the camera into the running simulation (Play's B):
// an object of the scene while it lasts (Reset and Stop take it away with the rest). Its id.
EXPORT int lr_sim_throw(double speed){
    if (E.sample || !S.on) return 0;
    View v = viewer_view();
    Vec3 f = view_forward(&v), c = v_add(v.pos, v_scale(f, 60));
    const real k = 0.3f;
    int id = object_add(ED_SPHERE, 5, -1, v_sub(c, vec3(0, 50 * k, 0)), k, 1, 0, "Ball");
    int b = rigid_add_sphere(&S.pw.rigid, c, 50 * k, SIM_DENSITY);
    if (b < 0) return id;
    S.pw.rigid.bodies[b].vel = v_scale(f, (real)speed);
    S.links = realloc(S.links, sizeof(SimLink) * (size_t)(S.nlinks + 1));
    if (!S.links) lr_oom("editor");
    S.links[S.nlinks++] = (SimLink){id, b, 0, vec3(0, 50 * k, 0)};
    viewer_scene_changed(0);
    return id;
}
EXPORT int lr_sim_keep(void){ if (E.sample || !S.on) return 1; sim_keep(); return 0; }

EXPORT const char *lr_status(void){
    ViewerStatus st;
    viewer_status(&st);
    int emitters = 0;
    for (int i = 0; i < E.nobjs; i++) emitters += E.objs[i].mat >= 0 && MATERIALS[E.objs[i].mat].glow > 0;
    jlen = 0;
    jprintf("{\"backend\":%d,\"tracing\":%d,\"gpu\":%d,\"gpuStarting\":%d,\"gpuTracing\":%d,\"spp\":%d,\"target\":%d,"
            "\"fps\":%.1f,\"width\":%d,\"height\":%d,\"instances\":%d,\"emitters\":%d,\"relighting\":%d,\"relight\":%.3f",
            (int)viewer_backend(), st.tracing, st.on_gpu, st.gpu_starting, st.gpu_tracing, st.spp, st.spp_target,
            st.fps, st.width, st.height, viewer_scene()->ninsts, emitters, E.relight != NULL, E.relight_done);
    jprintf(",\"displayHz\":%.1f,\"frameCap\":%d", viewer_display_hz(), frame_cap_ms > 0 ? (int)(1000.0 / frame_cap_ms + 0.5) : 0);
    jprintf(",\"trace\":{\"starts\":%ld,\"moves\":%ld,\"passMs\":%.2f,\"rows\":%d,\"reps\":%d,\"budgetMs\":%.2f,\"work\":%.2f,\"history\":%d,\"rate\":%.1f,\"compiling\":%d}",
            st.trace_starts, st.trace_moves, st.trace_pass_ms, st.trace_rows, st.trace_reps, st.trace_budget_ms, st.trace_work,
            st.trace_history, st.trace_rate, st.trace_compiling);
    if (viewer_sim_on()){                            // the simulation: its time, its bodies so far
        char line[200];
        viewer_sim_line(line, sizeof(line));
        jprintf(",\"sim\":{\"own\":0,\"on\":1,\"paused\":%d,\"t\":%.3f,\"bodies\":%d,\"line\":", viewer_sim_paused(), viewer_sim_time(), viewer_sim_bodies());
        jstring(line);
        jprintf("}");
    } else if (!E.sample){                           // the editor's own: made at Play (on), its bodies
        int awake = 0;
        for (int i = 0; i < S.nlinks; i++){
            const Body *b = &S.pw.rigid.bodies[S.links[i].body];
            awake += b->inv_mass > 0 && !b->sleeping;
        }
        char line[200];
        snprintf(line, sizeof(line), "t %.2f s | %d bodies, %d moving, %d contacts | rigid %.2f ms",
                 S.steps * SIM_DT, S.nlinks, awake, S.pw.rigid.ncontacts, S.pw.ms_rigid);
        jprintf(",\"sim\":{\"own\":1,\"on\":%d,\"paused\":%d,\"t\":%.3f,\"bodies\":%d,\"moving\":%d,\"line\":",
                S.on, !S.on || S.paused, S.steps * SIM_DT, sim_dynamic_count(), awake);
        jstring(line);
        jprintf("}");
    }
    jprintf("}");
    return json_buf;
}

// the scene as the page lists it: {"sky":"Dusk","objects":[{"id":1,"name":"Rock 1",...},...]}
// (metres; pos: the middle of its bottom, top: its height)
EXPORT const char *lr_scene(void){
    jlen = 0;
    jprintf("{\"sky\":");
    jstring(E.sky_name);
    jprintf(",\"skyKey\":\"%s\"", E.sky == SKY_FILE ? "file" : E.sky == SKY_SUN ? "sun" : E.sky == SKY_NOON ? "noon" : E.sky == SKY_OVERCAST ? "overcast" : "dusk");
    if (E.sky == SKY_FILE){ jprintf(",\"skyPath\":"); jstring(E.sky_path); }   // (where its .hdr is: Publish packs it)
    jprintf(",\"sun\":{\"t\":%.4f,\"intensity\":%.4f,\"size\":%.4f,\"haze\":%.4f,\"az\":%.2f,\"warm\":%d}",
            (double)E.sun.t, (double)E.sun.intensity, (double)E.sun.size, (double)E.sun.haze, (double)E.sun.az, E.sun.warm);
    jprintf(",\"undo\":%d,\"redo\":%d", nundo, nredo);
    if (E.sample){                                   // (what the sample is made of: the engine's demo)
        const Options *op = viewer_options();
        jprintf(",\"sample\":{\"name\":\"Sky rain\",\"bodies\":%d,\"props\":%d,\"instances\":%d}",
                op->sim.bodies, op->sim.props, viewer_scene()->ninsts);
    }
    jprintf(",\"objects\":[");
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        Aabb bx = object_box(o);
        Vec3 c = v_scale(v_add(bx.lo, bx.hi), 0.5f);
        jprintf("%s{\"id\":%d,\"name\":", i ? "," : "", o->id);
        jstring(o->name);
        jprintf(",\"kind\":\"%s\",\"material\":\"%s\",\"dynamic\":%d,\"scale\":%.3f,"
                "\"pos\":[%.4f,%.4f,%.4f],\"top\":%.4f,\"rot\":[%.6f,%.6f,%.6f,%.6f],\"stretch\":[%.4f,%.4f,%.4f],"
                "\"centre\":[%.4f,%.4f,%.4f],\"support\":%d",
                KIND_NAMES[o->kind], o->mat >= 0 ? MATERIALS[o->mat].key : o->mat == MAT_OWN ? "own" : o->kind == ED_MODEL && E.models[o->model].clay ? "clay" : "",
                o->dynamic, (double)o->scale, (double)o->pos.x / UNITS_PER_M, (double)o->pos.y / UNITS_PER_M,
                (double)o->pos.z / UNITS_PER_M, (double)object_top(o) / UNITS_PER_M,
                (double)o->rot.w, (double)o->rot.x, (double)o->rot.y, (double)o->rot.z, (double)o->stretch.x, (double)o->stretch.y, (double)o->stretch.z,
                (double)c.x / UNITS_PER_M, (double)c.y / UNITS_PER_M, (double)c.z / UNITS_PER_M, o->support);
        if (o->mat == MAT_OWN){
            char lk[1024];
            look_json(lk, sizeof(lk), o);
            jprintf(",\"look\":%s", lk);
        }
        jprintf("}");
    }
    jprintf("]}");
    return json_buf;
}

// What is under the screen point (sx, sy) (0..1 of the frame, from the top left), not counting
// `exclude` and what rests on it (an object being dragged): {"hit":1,"id":2,"p":[x,y,z],
// "land":[x,y,z]}: the object (0: the ground), the point hit, and where something dropped
// there would stand (on an object: on its top, near the point; on the ground: there, kept
// on the floor). {"hit":0}: the sky.
EXPORT const char *lr_pick(double sx, double sy, int exclude){
    if (E.sample){ jlen = 0; jprintf("{\"hit\":0}"); return json_buf; }   // (a sample: nothing of its to pick)
    Scene *s = viewer_scene();
    scene_update_instances(s, &viewer_options()->ap);   // (the top level, as the instances are now)
    Camera c = camera_now();
    Ray ray = camera_ray(&c, (real)sx, (real)sy);
    Hit h;
    real tmin = 1e-3f;
    const EdObject *on = NULL;
    Vec3 p = vec3(0, 0, 0);
    int hit = 0;
    for (int tries = 0; tries < 16; tries++){
        if (!scene_intersect(s, &ray, tmin, 1e9f, &h)) break;
        on = h.inst >= 0 ? object_of_inst(h.inst) : NULL;
        if (on && exclude && in_stack(on->id, exclude)){ tmin = h.t + 0.01f; on = NULL; continue; }
        p = h.p;
        hit = 1;
        break;
    }
    if (!hit && ray.d.y < 0){                // (past the floor's end: where the ray meets its plane)
        p = v_add(ray.o, v_scale(ray.d, -ray.o.y / ray.d.y));
        hit = 1;
        on = NULL;
    }
    jlen = 0;
    if (!hit){ jprintf("{\"hit\":0}"); return json_buf; }
    Vec3 land;
    if (on){
        Aabb ob = object_box(on);
        Vec3 mid = v_scale(v_add(ob.lo, ob.hi), 0.5f);
        Vec3 d = v_sub(p, mid);
        d.y = 0;
        real r = object_reach(on), l = v_len(d);
        if (l > r) d = v_scale(d, r / l);
        land = vec3(mid.x + d.x, ob.hi.y, mid.z + d.z);
    } else {
        real f = (real)(FLOOR_M * UNITS_PER_M);
        land = vec3(r_fmax(-f, r_fmin(f, p.x)), 0, r_fmax(-f, r_fmin(f, p.z)));
    }
    jprintf("{\"hit\":1,\"id\":%d,\"p\":[%.3f,%.3f,%.3f],\"land\":[%.3f,%.3f,%.3f]}", on ? on->id : 0,
            (double)p.x / UNITS_PER_M, (double)p.y / UNITS_PER_M, (double)p.z / UNITS_PER_M,
            (double)land.x / UNITS_PER_M, (double)land.y / UNITS_PER_M, (double)land.z / UNITS_PER_M);
    return json_buf;
}

// where a point (metres) is on the screen: [sx,sy] (0..1 of the frame), or null behind the camera
EXPORT const char *lr_project(double x, double y, double z){
    Camera c = camera_now();
    real u, v, cs;
    jlen = 0;
    if (!camera_project(&c, metres(x, y, z), &u, &v, &cs)) jprintf("null");
    else jprintf("[%.5f,%.5f]", (double)u, (double)v);
    return json_buf;
}

// the camera, for the page's gizmo: {"pos":[m],"fwd":[...],"right":[...],"up":[...],"fov":40,
// "w":1920,"h":1080} (fov: the vertical, degrees)
// For the page (a stack of boxes before they are made): where on the ground a box of half
// sizes hx, hz and height h (metres, its middle at x, z) touches nothing, the nearest to there:
// [x, z] in metres
EXPORT const char *lr_free_spot(double x, double z, double hx, double hz, double h){
    Aabb b = {metres(x - hx, 0, z - hz), metres(x + hx, h, z + hz)};
    Vec3 d = free_shift(b, 0);
    jlen = 0;
    jprintf("[%.4f,%.4f]", x + (double)d.x / UNITS_PER_M, z + (double)d.z / UNITS_PER_M);
    return json_buf;
}

EXPORT const char *lr_camera(void){
    View v = viewer_view();
    ViewerStatus st;
    viewer_status(&st);
    Vec3 f = view_forward(&v), r = view_right(&v), u = v_cross(r, f);
    jlen = 0;
    jprintf("{\"pos\":[%.5f,%.5f,%.5f],\"fwd\":[%.6f,%.6f,%.6f],\"right\":[%.6f,%.6f,%.6f],\"up\":[%.6f,%.6f,%.6f],\"fov\":%.4f,\"w\":%d,\"h\":%d}",
            (double)v.pos.x / UNITS_PER_M, (double)v.pos.y / UNITS_PER_M, (double)v.pos.z / UNITS_PER_M,
            (double)f.x, (double)f.y, (double)f.z, (double)r.x, (double)r.y, (double)r.z, (double)u.x, (double)u.y, (double)u.z,
            v.fov, st.width, st.height);
    return json_buf;
}

// every object's box on the screen: {"2":[s0,t0,s1,t1,d],...} (0..1 of the frame; d: its
// distance, metres; what is all behind the camera is left out)
EXPORT const char *lr_boxes(void){
    Camera c = camera_now();
    jlen = 0;
    jprintf("{");
    int first = 1;
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        real s0 = 1e9f, t0 = 1e9f, s1 = -1e9f, t1 = -1e9f;
        int any = 0;
        Aabb box = object_box(o);
        for (int q = 0; q < 8; q++){
            Vec3 p = vec3(q & 1 ? box.hi.x : box.lo.x, q & 2 ? box.hi.y : box.lo.y, q & 4 ? box.hi.z : box.lo.z);
            real u, v, cs;
            if (!camera_project(&c, p, &u, &v, &cs)) continue;
            any = 1;
            s0 = r_fmin(s0, u); s1 = r_fmax(s1, u); t0 = r_fmin(t0, v); t1 = r_fmax(t1, v);
        }
        if (!any) continue;
        real d = v_len(v_sub(v_scale(v_add(box.lo, box.hi), 0.5f), c.origin));
        jprintf("%s\"%d\":[%.5f,%.5f,%.5f,%.5f,%.3f]", first ? "" : ",", o->id, (double)s0, (double)t0, (double)s1,
                (double)t1, (double)d / UNITS_PER_M);
        first = 0;
    }
    jprintf("}");
    return json_buf;
}

// the camera turned as it is, moved to show the object whole in the middle (F)
EXPORT int lr_frame_object(int id){
    const EdObject *o = object_find(id);
    if (!o) return 1;
    Aabb box = object_box(o);
    View v = viewer_view();
    double r = 0.5 * v_len(v_sub(box.hi, box.lo));
    E.pivot = v_scale(v_add(box.lo, box.hi), 0.5f);
    E.dist = fmax(30, 1.6 * r / sin(v.fov * PI / 360));
    camera_from_pivot(&v);
    viewer_set_view(&v);
    return 0;
}

// ---- saving: the scene as glTF 2.0 (.glb), in metres (the spec, 6.8) ------------------------
//
// Every object a node (named as in the list) over its parts, each a mesh of its material;
// physics as KHR_physics_rigid_bodies (a dynamic object's node has a motion; every part a
// collider: the unit box or sphere of KHR_collision_shapes under the part's transform, a
// mesh part its convex hull); what the standards have no place for in LR_scene: the sky,
// the renderer and the resolution (the file's), each object's id, kind, catalogue material,
// what it rests on and (version 2) its place, turn and stretch (its node's). Both physics extensions are Review Drafts: their names
// are here only, to change when they do.
#define EXT_BODIES "KHR_physics_rigid_bodies"
#define EXT_SHAPES "KHR_collision_shapes"
#define EXT_SCENE "LR_scene"

// ---- OpenUSD (.usda): the scene as text, for Blender, usdview, Omniverse... -----------------
// Units as the editor's (metersPerUnit 0.01: centimetres), Y up. Each object an Xform of its
// parts, each a Mesh placed by its instance's transform: the unit box [-1, 1]^3, the unit ball
// (32 x 16), or a mesh's triangles with their normals and texture coordinates (a part of
// several materials a GeomSubset for each). The materials UsdPreviewSurface
// (colour, roughness, metallic, emission, glass as opacity and ior); images are not written
// (the .glb has them). The ground a Mesh of its quad; the sky's file, if it has one, a DomeLight.
#define M_PI_LR 3.14159265358979323846
static void usd_name(char *out, size_t n, const char *name, int id){
    size_t k = 0;
    if (!(isalpha((unsigned char)name[0]) || name[0] == '_') && k + 1 < n) out[k++] = '_';
    for (const char *c = name; *c && k + 12 < n; c++) out[k++] = isalnum((unsigned char)*c) ? *c : '_';
    snprintf(out + k, n - k, "_%d", id);
}
static void usd_matrix(FILE *f, const Xform *x){
    // (USD's rows are the axes and then the place: the transpose of ours)
    fprintf(f, "( (%.6g, %.6g, %.6g, 0), (%.6g, %.6g, %.6g, 0), (%.6g, %.6g, %.6g, 0), (%.6g, %.6g, %.6g, 1) )",
            (double)x->m[0][0], (double)x->m[1][0], (double)x->m[2][0], (double)x->m[0][1], (double)x->m[1][1], (double)x->m[2][1],
            (double)x->m[0][2], (double)x->m[1][2], (double)x->m[2][2], (double)x->m[0][3], (double)x->m[1][3], (double)x->m[2][3]);
}
static void usd_mesh(FILE *f, const SceneMesh *m, const char *ind){
    fprintf(f, "%sint[] faceVertexCounts = [", ind);
    for (int t = 0; t < m->ntris; t++) fprintf(f, t ? ", 3" : "3");
    fprintf(f, "]\n%sint[] faceVertexIndices = [", ind);
    for (int t = 0; t < 3 * m->ntris; t++) fprintf(f, t ? ", %d" : "%d", t);
    fprintf(f, "]\n%spoint3f[] points = [", ind);
    for (int t = 0; t < m->ntris; t++){
        const Triangle *tr = &m->tris[t];
        Vec3 p[3] = {tr->q, v_add(tr->q, tr->u), v_add(tr->q, tr->v)};
        for (int k = 0; k < 3; k++) fprintf(f, "%s(%.6g, %.6g, %.6g)", t || k ? ", " : "", (double)p[k].x, (double)p[k].y, (double)p[k].z);
    }
    fprintf(f, "]\n");
    int nrm = m->ntris > 0 && m->nnormals > 0, uv = m->ntris > 0 && m->nuvs > 0;
    for (int t = 0; t < m->ntris && (nrm || uv); t++){
        if (m->tris[t].n[0] < 0) nrm = 0;
        if (m->tris[t].t[0] < 0) uv = 0;
    }
    if (nrm){
        fprintf(f, "%snormal3f[] normals = [", ind);
        for (int t = 0; t < m->ntris; t++)
            for (int k = 0; k < 3; k++){ Vec3 n = m->normals[m->tris[t].n[k]]; fprintf(f, "%s(%.5f, %.5f, %.5f)", t || k ? ", " : "", (double)n.x, (double)n.y, (double)n.z); }
        fprintf(f, "] (\n%s    interpolation = \"faceVarying\"\n%s)\n", ind, ind);
    }
    if (uv){
        // (glTF's v runs down the image, USD's up)
        fprintf(f, "%stexCoord2f[] primvars:st = [", ind);
        for (int t = 0; t < m->ntris; t++)
            for (int k = 0; k < 3; k++){ Vec2 c = m->uvs[m->tris[t].t[k]]; fprintf(f, "%s(%.5f, %.5f)", t || k ? ", " : "", (double)c.x, 1.0 - (double)c.y); }
        fprintf(f, "] (\n%s    interpolation = \"faceVarying\"\n%s)\n", ind, ind);
    }
    fprintf(f, "%suniform token subdivisionScheme = \"none\"\n", ind);
}
// the unit box [-1, 1]^3 (each face its own 4 points: flat normals) and the unit ball (32 x 16),
// as meshes: importers take a Mesh's material where some drop a Cube's or a Sphere's
static void usd_box(FILE *f, const char *ind){
    fprintf(f, "%sint[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n%sint[] faceVertexIndices = [", ind, ind);
    for (int i = 0; i < 24; i++) fprintf(f, i ? ", %d" : "%d", i);
    fprintf(f, "]\n%spoint3f[] points = [", ind);
    Vec3 nrm[6];
    for (int k = 0; k < 6; k++){
        Primitive q = unit_box_face(k, 0);
        Vec3 c[4] = {q.q, v_add(q.q, q.u), v_add(v_add(q.q, q.u), q.v), v_add(q.q, q.v)};
        for (int j = 0; j < 4; j++) fprintf(f, "%s(%g, %g, %g)", k || j ? ", " : "", (double)c[j].x, (double)c[j].y, (double)c[j].z);
        nrm[k] = q.normal;
    }
    fprintf(f, "]\n%snormal3f[] normals = [", ind);
    for (int k = 0; k < 6; k++) for (int j = 0; j < 4; j++) fprintf(f, "%s(%g, %g, %g)", k || j ? ", " : "", (double)nrm[k].x, (double)nrm[k].y, (double)nrm[k].z);
    fprintf(f, "] (\n%s    interpolation = \"faceVarying\"\n%s)\n%suniform token subdivisionScheme = \"none\"\n", ind, ind, ind);
}
static void usd_ball(FILE *f, const char *ind){
    enum { NU = 32, NV = 16 };
    fprintf(f, "%sint[] faceVertexCounts = [", ind);
    for (int i = 0; i < NU * NV; i++) fprintf(f, i ? ", 4" : "4");
    fprintf(f, "]\n%sint[] faceVertexIndices = [", ind);
    for (int v = 0; v < NV; v++)
        for (int u = 0; u < NU; u++){
            int a = v * (NU + 1) + u, b = a + NU + 1;
            fprintf(f, "%s%d, %d, %d, %d", v || u ? ", " : "", a, b, b + 1, a + 1);
        }
    fprintf(f, "]\n%spoint3f[] points = [", ind);
    for (int v = 0; v <= NV; v++)
        for (int u = 0; u <= NU; u++){
            double th = M_PI_LR * v / NV, ph = 2 * M_PI_LR * u / NU;
            fprintf(f, "%s(%.6f, %.6f, %.6f)", v || u ? ", " : "", sin(th) * sin(ph), cos(th), sin(th) * cos(ph));
        }
    fprintf(f, "]\n%snormal3f[] normals = [", ind);
    for (int v = 0; v <= NV; v++)
        for (int u = 0; u <= NU; u++){
            double th = M_PI_LR * v / NV, ph = 2 * M_PI_LR * u / NU;
            fprintf(f, "%s(%.6f, %.6f, %.6f)", v || u ? ", " : "", sin(th) * sin(ph), cos(th), sin(th) * cos(ph));
        }
    fprintf(f, "] (\n%s    interpolation = \"vertex\"\n%s)\n%suniform token subdivisionScheme = \"none\"\n", ind, ind, ind);
}
static void usd_bind(FILE *f, const char *ind, int mat){
    if (mat >= 0) fprintf(f, "%srel material:binding = </Scene/Looks/M%d>\n", ind, mat);
}
// 0 = written
EXPORT int lr_save_usda(const char *path){
    if (E.sample) return 1;
    Scene *s = viewer_scene();
    FILE *f = fopen(path, "w");
    if (!f) return 1;
    fprintf(f, "#usda 1.0\n(\n    defaultPrim = \"Scene\"\n    metersPerUnit = 0.01\n    upAxis = \"Y\"\n    doc = \"Kiln (little_renderer's editor)\"\n)\n\n");
    fprintf(f, "def Xform \"Scene\"\n{\n");
    // the materials the objects use, and the ground's
    int used[MAX_MATS] = {0};
    for (int i = 0; i < E.nobjs; i++)
        for (int k = 0; k < E.objs[i].ninst; k++){
            const Instance *in = &s->insts[E.objs[i].inst[k]];
            if (in->material >= 0 && in->material < MAX_MATS) used[in->material] = 1;
            else if (in->mesh >= 0)
                for (int t = 0; t < s->meshes[in->mesh].ntris; t++){
                    int m = s->meshes[in->mesh].tris[t].material;
                    if (m >= 0 && m < MAX_MATS) used[m] = 1;
                }
        }
    for (int p = 0; p < s->nprims; p++) if (s->prims[p].material >= 0 && s->prims[p].material < MAX_MATS) used[s->prims[p].material] = 1;
    fprintf(f, "    def Scope \"Looks\"\n    {\n");
    for (int m = 0; m < s->nmats; m++){
        if (!used[m]) continue;
        Material r = material_resolved(&s->mats[m]);
        fprintf(f, "        def Material \"M%d\"\n        {\n", m);
        fprintf(f, "            token outputs:surface.connect = </Scene/Looks/M%d/Surface.outputs:surface>\n", m);
        fprintf(f, "            def Shader \"Surface\"\n            {\n                uniform token info:id = \"UsdPreviewSurface\"\n");
        fprintf(f, "                color3f inputs:diffuseColor = (%.4f, %.4f, %.4f)\n", (double)r.albedo.x, (double)r.albedo.y, (double)r.albedo.z);
        fprintf(f, "                float inputs:roughness = %.3f\n                float inputs:metallic = %.3f\n", (double)r.roughness, (double)r.metallic);
        if (r.emission.x > 0 || r.emission.y > 0 || r.emission.z > 0)
            fprintf(f, "                color3f inputs:emissiveColor = (%.4f, %.4f, %.4f)\n", (double)r.emission.x, (double)r.emission.y, (double)r.emission.z);
        if (r.transmission > 0)
            fprintf(f, "                float inputs:opacity = %.3f\n                float inputs:ior = %.3f\n", 1.0 - 0.9 * (double)r.transmission, (double)r.ior);
        if (s->mats[m].tex_base) fprintf(f, "                # (its base colour has an image: in the .glb)\n");
        fprintf(f, "                token outputs:surface\n            }\n        }\n");
    }
    fprintf(f, "    }\n\n");
    // the objects
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        char nm[96];
        usd_name(nm, sizeof(nm), o->name, o->id);
        fprintf(f, "    def Xform \"%s\" (\n        customData = { string kilnKind = \"%s\" }\n    )\n    {\n", nm, KIND_NAMES[o->kind]);
        for (int k = 0; k < o->ninst; k++){
            const Instance *in = &s->insts[o->inst[k]];
            const char *type = "Mesh";
            fprintf(f, "        def %s \"part_%d\" (\n            prepend apiSchemas = [\"MaterialBindingAPI\"]\n        )\n        {\n"
                       "            matrix4d xformOp:transform = ", type, k + 1);
            usd_matrix(f, &in->xf);
            fprintf(f, "\n            uniform token[] xformOpOrder = [\"xformOp:transform\"]\n");
            if (in->mesh == INST_BOX) usd_box(f, "            ");
            else if (in->mesh == INST_SPHERE) usd_ball(f, "            ");
            if (in->mesh >= 0){
                const SceneMesh *m = &s->meshes[in->mesh];
                usd_mesh(f, m, "            ");
                if (in->material >= 0) usd_bind(f, "            ", in->material);
                else {
                    // (its triangles' materials: one, or a subset for each)
                    int first = m->ntris ? m->tris[0].material : -1, one = 1;
                    for (int t = 1; t < m->ntris; t++) one &= m->tris[t].material == first;
                    if (one) usd_bind(f, "            ", first);
                    else for (int mm = 0; mm < MAX_MATS; mm++){
                        int any = 0;
                        for (int t = 0; t < m->ntris && !any; t++) any = m->tris[t].material == mm;
                        if (!any) continue;
                        fprintf(f, "            def GeomSubset \"material_%d\" (\n                prepend apiSchemas = [\"MaterialBindingAPI\"]\n            )\n"
                                   "            {\n                uniform token elementType = \"face\"\n"
                                   "                uniform token familyName = \"materialBind\"\n                int[] indices = [", mm);
                        int c = 0;
                        for (int t = 0; t < m->ntris; t++) if (m->tris[t].material == mm) fprintf(f, c++ ? ", %d" : "%d", t);
                        fprintf(f, "]\n");
                        usd_bind(f, "                ", mm);
                        fprintf(f, "            }\n");
                    }
                }
            } else usd_bind(f, "            ", in->material);
            fprintf(f, "        }\n");
        }
        fprintf(f, "    }\n");
    }
    // the ground (the scene's quads)
    for (int p = 0; p < s->nprims; p++){
        const Primitive *q = &s->prims[p];
        if (q->type != PRIM_QUAD) continue;
        Vec3 a = q->q, b = v_add(q->q, q->u), c = v_add(v_add(q->q, q->u), q->v), d = v_add(q->q, q->v);
        fprintf(f, "    def Mesh \"Ground_%d\" (\n        prepend apiSchemas = [\"MaterialBindingAPI\"]\n    )\n    {\n        int[] faceVertexCounts = [4]\n        int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "        point3f[] points = [(%.6g, %.6g, %.6g), (%.6g, %.6g, %.6g), (%.6g, %.6g, %.6g), (%.6g, %.6g, %.6g)]\n"
                   "        uniform token subdivisionScheme = \"none\"\n", p, (double)a.x, (double)a.y, (double)a.z, (double)b.x, (double)b.y,
                (double)b.z, (double)c.x, (double)c.y, (double)c.z, (double)d.x, (double)d.y, (double)d.z);
        usd_bind(f, "        ", q->material);
        fprintf(f, "    }\n");
    }
    if (E.sky == SKY_FILE && E.sky_path[0]){
        const char *b = strrchr(E.sky_path, '/');
        fprintf(f, "    def DomeLight \"Sky\"\n    {\n        asset inputs:texture:file = @%s@\n        token inputs:texture:format = \"latlong\"\n    }\n", b ? b + 1 : E.sky_path);
    } else fprintf(f, "    # the sky: %s (made by the editor; not written)\n", E.sky_name);
    fprintf(f, "}\n");
    int bad = ferror(f);
    if (fclose(f) || bad) return 1;
    return 0;
}

EXPORT int lr_save_gltf(const char *path){
    if (E.sample) return 1;                          // (a sample is the engine's: nothing of the editor's to save)
    Scene *s = viewer_scene();
    GltfWriter *w = gltf_writer_new(s, (real)(1 / UNITS_PER_M));
    if (!w) return 1;
    for (int i = 0; i < NMATERIALS; i++) gltf_name_material(w, E.mats[i], MATERIALS[i].label);   // (as the catalogue calls them)
    gltf_name_material(w, E.dark, "Lamp stand");
    gltf_name_material(w, E.clay, "Clay");
    for (int i = 0; i < E.nobjs; i++)                   // (an object's own: by its name)
        if (E.objs[i].mat == MAT_OWN && E.objs[i].look >= 0){
            char nm[96];
            snprintf(nm, sizeof(nm), "%s (own)", E.objs[i].name);
            gltf_name_material(w, E.objs[i].look, nm);
        }
    int *geo = malloc(sizeof(int) * (size_t)(s->nmeshes + 2));             // (geometries: the meshes', then the box's and the sphere's)
    int nmesh = 0, cap = 0;
    struct { int geo, mat, mesh; } *meshes = NULL;
    int *kids = NULL, kcap = 0;
    if (!geo){ gltf_writer_free(w); return 1; }
    for (int i = 0; i < s->nmeshes + 2; i++) geo[i] = -1;
    int nodes = 0, ok = 1;
    char ext[2048], extras[256];
    for (int i = 0; i < E.nobjs && ok; i++){
        const EdObject *o = &E.objs[i];
        kids = lr_grow(kids, sizeof(int), &kcap, o->ninst, 8);
        for (int k = 0; k < o->ninst; k++){
            const Instance *in = &s->insts[o->inst[k]];
            int gi = in->mesh >= 0 ? in->mesh : s->nmeshes + (in->mesh == INST_BOX ? 0 : 1);
            if (geo[gi] < 0) geo[gi] = gltf_add_geometry(w, in->mesh);
            int m = -1;
            for (int j = 0; j < nmesh && m < 0; j++) if (meshes[j].geo == geo[gi] && meshes[j].mat == in->material) m = meshes[j].mesh;
            if (m < 0){
                meshes = lr_grow(meshes, sizeof(*meshes), &cap, nmesh + 1, 16);
                m = gltf_add_mesh(w, geo[gi], in->material);
                meshes[nmesh].geo = geo[gi]; meshes[nmesh].mat = in->material; meshes[nmesh++].mesh = m;
            }
            if (geo[gi] < 0 || m < 0){ ok = 0; break; }
            if (in->mesh >= 0) snprintf(ext, sizeof(ext), "\"" EXT_BODIES "\":{\"collider\":{\"geometry\":{\"node\":%d,\"convexHull\":true}}}", nodes);
            else snprintf(ext, sizeof(ext), "\"" EXT_BODIES "\":{\"collider\":{\"geometry\":{\"shape\":%d}}}", in->mesh == INST_BOX ? 0 : 1);
            char name[96];                       // (a name of 63 characters at most, and its part)
            snprintf(name, sizeof(name), "%s, part %d", o->name, k + 1);
            kids[k] = gltf_add_node(w, name, m, &in->xf, ext, NULL);
            nodes++;
        }
        if (!ok) break;
        const EdObject *sup = object_find(o->support);
        int n = 0;
        n += snprintf(ext + n, sizeof(ext) - (size_t)n, "\"" EXT_SCENE "\":{\"id\":%d,\"kind\":\"%s\"", o->id, KIND_NAMES[o->kind]);
        if (o->mat >= 0) n += snprintf(ext + n, sizeof(ext) - (size_t)n, ",\"material\":\"%s\"", MATERIALS[o->mat].key);
        else if (o->mat == MAT_OWN){
            char lk[1024];
            look_json(lk, sizeof(lk), o);
            n += snprintf(ext + n, sizeof(ext) - (size_t)n, ",\"material\":\"own\",\"look\":%s", lk);
        }
        else if (o->kind == ED_MODEL && E.models[o->model].clay) n += snprintf(ext + n, sizeof(ext) - (size_t)n, ",\"clay\":true");
        if (sup) n += snprintf(ext + n, sizeof(ext) - (size_t)n, ",\"restsOn\":%d", sup->id);
        n += snprintf(ext + n, sizeof(ext) - (size_t)n, ",\"pos\":[%.6f,%.6f,%.6f],\"rot\":[%.8f,%.8f,%.8f,%.8f],\"stretch\":[%.6f,%.6f,%.6f]}",
                      o->pos.x / UNITS_PER_M, o->pos.y / UNITS_PER_M, o->pos.z / UNITS_PER_M, (double)o->rot.w, (double)o->rot.x,
                      (double)o->rot.y, (double)o->rot.z, (double)o->stretch.x, (double)o->stretch.y, (double)o->stretch.z);
        if (o->dynamic) snprintf(ext + n, sizeof(ext) - (size_t)n, ",\"" EXT_BODIES "\":{\"motion\":{}}");
        snprintf(extras, sizeof(extras), "\"scale\":%.4f", (double)o->scale);
        gltf_add_group(w, o->name, kids, o->ninst, ext, extras);
        nodes++;
    }
    // the file's extensions: the two unit shapes; the scene's own
    gltf_use_extension(w, EXT_SHAPES, "{\"shapes\":[{\"type\":\"box\",\"box\":{\"size\":[2,2,2]}},{\"type\":\"sphere\",\"sphere\":{\"radius\":1}}]}", 0);
    gltf_use_extension(w, EXT_BODIES, NULL, 0);
    ViewerStatus st;
    viewer_status(&st);
    static const char *const RENDERERS[] = {"path traced", "raster", "auto"};
    char root[640], sky[320];
    if (E.sky == SKY_FILE){
        snprintf(sky, sizeof(sky), "{\"file\":\"");
        size_t k = strlen(sky);
        for (const char *c = E.sky_name; *c && k < sizeof(sky) - 4; c++) if (*c != '"' && *c != '\\' && (unsigned char)*c >= 32) sky[k++] = *c;
        snprintf(sky + k, sizeof(sky) - k, "\"}");
    } else if (E.sky == SKY_SUN){
        snprintf(sky, sizeof(sky), "{\"preset\":\"sun\",\"sun\":{\"t\":%.4f,\"intensity\":%.4f,\"size\":%.4f,\"haze\":%.4f,\"az\":%.2f,\"warm\":%d}}",
                 (double)E.sun.t, (double)E.sun.intensity, (double)E.sun.size, (double)E.sun.haze, (double)E.sun.az, E.sun.warm);
    } else snprintf(sky, sizeof(sky), "{\"preset\":\"%s\"}", E.sky == SKY_NOON ? "noon" : E.sky == SKY_OVERCAST ? "overcast" : "dusk");
    int b = (int)viewer_backend();
    snprintf(root, sizeof(root), "{\"version\":2,\"sky\":%s,\"renderer\":\"%s\",\"resolution\":[%d,%d],\"unitsPerMetre\":%g}",
             sky, b >= 0 && b < 3 ? RENDERERS[b] : "auto", st.width, st.height, UNITS_PER_M);
    gltf_use_extension(w, EXT_SCENE, root, 0);
    if (ok) ok = gltf_write_glb(w, path) == 0;
    gltf_writer_free(w);
    free(geo);
    free(meshes);
    free(kids);
    return ok ? 0 : 1;
}

// ---- a saved scene opened again ---------------------------------------------------------------

// the JSON of a .glb (its first chunk) or of a .gltf, parsed; NULL: neither
static Json *gltf_json(const char *path){
    size_t n = 0;
    unsigned char *f = file_read_all(path, &n);
    if (!f) return NULL;
    Json *j = NULL;
    #define LE32(p) ((uint32_t)(p)[0] | (uint32_t)(p)[1] << 8 | (uint32_t)(p)[2] << 16 | (uint32_t)(p)[3] << 24)
    if (n >= 20 && LE32(f) == 0x46546C67u){
        size_t len = LE32(f + 12);
        if (LE32(f + 16) == 0x4E4F534Au && len <= n - 20) j = json_parse((const char *)f + 20, len);
    } else j = json_parse((const char *)f, n);
    #undef LE32
    free(f);
    return j;
}
static Vec3 json_vec3(const Json *a, Vec3 def){
    if (json_len(a) != 3) return def;
    return vec3((real)json_num(json_at(a, 0), def.x), (real)json_num(json_at(a, 1), def.y), (real)json_num(json_at(a, 2), def.z));
}
static real v_at(Vec3 v, int i){ return i == 0 ? v.x : i == 1 ? v.y : v.z; }
static Aabb box_grow(Aabb b, Vec3 p){
    b.lo = vec3(r_fmin(b.lo.x, p.x), r_fmin(b.lo.y, p.y), r_fmin(b.lo.z, p.z));
    b.hi = vec3(r_fmax(b.hi.x, p.x), r_fmax(b.hi.y, p.y), r_fmax(b.hi.z, p.z));
    return b;
}

// A scene saved by lr_save_gltf (its root has LR_scene), opened in place of this one, as one
// undo step: its objects (names, kinds, catalogue materials, static or dynamic, what rests on
// what, places, turns and stretches; a model's parts its own meshes and materials), its sky
// when a preset (a file's .hdr is not in the file: its name is said), its renderer said.
// Files of version 1 (before places were saved): the shapes' places from their geometry, a
// model's from its box. The objects get new ids (undo goes back to the old ones). JSON:
// {"opened":0} not such a scene (an ordinary model: import it), {"opened":-1} could not be
// read; {"opened":1,"objects":n,"skyFile":name or null,"renderer":name}.
// The materials from m0 on that no object took (a saved scene has the catalogue's, which are
// the editor's already, and the lamp's stand, the clay) taken out again, those taken moved
// down: opening a scene does not fill the scene's room (MAX_MATS) with copies. A model's
// parts have their materials in their meshes' triangles (the meshes from k0 on: the file's)
static void materials_compact(int m0, int k0){
    Scene *s = viewer_scene();
    int used[MAX_MATS] = {0}, map[MAX_MATS];
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        if (o->mat == MAT_OWN && o->look >= m0) used[o->look] = 1;
        if (o->kind != ED_MODEL || o->mat >= 0) continue;          // (a catalogue material on it: not its own)
        for (int q = 0; q < E.models[o->model].nparts; q++){
            const EdPart *pt = &E.models[o->model].parts[q];
            if (pt->material >= m0 && pt->material < MAX_MATS) used[pt->material] = 1;
            if (pt->mesh >= k0 && pt->material < 0)
                for (int t = 0; t < s->meshes[pt->mesh].ntris; t++){
                    int m = s->meshes[pt->mesh].tris[t].material;
                    if (m >= m0 && m < MAX_MATS) used[m] = 1;
                }
        }
    }
    int k = m0;
    for (int i = m0; i < s->nmats; i++){
        map[i] = used[i] ? k : -1;
        if (used[i]) s->mats[k++] = s->mats[i];
    }
    for (int i = k0; i < s->nmeshes; i++)                          // (unused meshes: any material)
        for (int t = 0; t < s->meshes[i].ntris; t++){
            int *m = &s->meshes[i].tris[t].material;
            if (*m >= m0 && *m < s->nmats) *m = map[*m] >= 0 ? map[*m] : 0;
        }
    s->nmats = k;
    for (int i = 0; i < E.nobjs; i++){
        EdObject *o = &E.objs[i];
        if (o->look >= m0) o->look = map[o->look];
        if (o->kind == ED_MODEL)
            for (int q = 0; q < E.models[o->model].nparts; q++){
                int *m = &E.models[o->model].parts[q].material;
                if (*m >= m0 && *m < MAX_MATS) *m = map[*m];
            }
        object_update(o);
    }
}

EXPORT const char *lr_open_scene(const char *path){
    Json *j = gltf_json(path);
    const Json *root = json_get(json_get(j, "extensions"), EXT_SCENE);
    if (!root || editing_off()){ json_free(j); jlen = 0; jprintf("{\"opened\":%d}", root ? -1 : j ? 0 : -1); return json_buf; }
    Scene *s = viewer_scene();
    lr_checkpoint();
    const int m0 = s->nmats, k0 = s->nmeshes;       // (the file's materials and meshes come after: materials_compact)
    // its nodes' instances where the file has them: taken (a model's parts need them), then
    // removed (the shapes are made again as the editor makes them)
    int n0 = s->ninsts, *node_of = NULL;
    int n = scene_load_gltf_placed(s, path, (real)UNITS_PER_M, &node_of);
    if (n < 0){ lr_forget_checkpoint(); json_free(j); jlen = 0; jprintf("{\"opened\":-1}"); return json_buf; }
    Instance *got = malloc(sizeof(Instance) * (size_t)(n > 0 ? n : 1));
    if (!got) lr_oom("editor");
    memcpy(got, s->insts + n0, sizeof(Instance) * (size_t)n);
    while (s->ninsts > n0) scene_remove_instance(s, s->ninsts - 1);
    for (int i = E.nobjs - 1; i >= 0; i--) object_drop_instances(&E.objs[i]);   // (this scene's objects gone)
    E.nobjs = 0;
    const Json *nodes = json_get(j, "nodes");
    int nn = json_len(nodes), made = 0;
    int *new_id = calloc((size_t)(nn > 0 ? nn : 1), sizeof(int)), *file_id = calloc((size_t)(nn > 0 ? nn : 1), sizeof(int));
    if (!new_id || !file_id) lr_oom("editor");
    for (int ni = 0; ni < nn; ni++){
        const Json *nd = json_at(nodes, ni), *ls = json_get(json_get(nd, "extensions"), EXT_SCENE);
        if (!ls) continue;
        const char *kind = json_str(json_get(ls, "kind"), "");
        int k = -1;
        for (int q = 0; q < 5; q++) if (!strcmp(kind, KIND_NAMES[q])) k = q;
        if (k < 0) continue;
        int mat = -1;
        const char *mk = json_str(json_get(ls, "material"), NULL);
        for (int q = 0; mk && q < NMATERIALS; q++) if (!strcmp(mk, MATERIALS[q].key)) mat = q;
        const int own = mk && !strcmp(mk, "own");     // (its own material: the file's, and its look's numbers)
        if (k != ED_MODEL && mat < 0) mat = k == ED_ROCK ? 2 : k == ED_LAMP ? 7 : k == ED_SPHERE ? 1 : 0;
        int dynamic = json_get(json_get(json_get(nd, "extensions"), EXT_BODIES), "motion") != NULL;
        real scale = (real)json_num(json_get(json_get(nd, "extras"), "scale"), 1);
        if (!(scale > 0)) scale = 1;
        // the instances of its parts, in the order of its children
        const Json *kids = json_get(nd, "children");
        int nk = json_len(kids), np = 0, *pi = malloc(sizeof(int) * (size_t)(nk > 0 ? nk : 1));
        if (!pi) lr_oom("editor");
        for (int c = 0; c < nk; c++){
            int cn = json_int(json_at(kids, c), -1);
            for (int q = 0; q < n; q++) if (node_of[q] == cn){ pi[np++] = q; break; }
        }
        EdObject shape;                              // (its kind's parts: to read its place back)
        memset(&shape, 0, sizeof(shape));
        shape.kind = (EdKind)k;
        Vec3 pos = vec3(0, 0, 0), stretch = vec3(1, 1, 1);
        Quat rot = quat_identity();
        const Json *jp = json_get(ls, "pos");
        if (jp){                                      // version 2: as saved
            pos = v_scale(json_vec3(jp, vec3(0, 0, 0)), (real)UNITS_PER_M);
            const Json *jr = json_get(ls, "rot");
            if (json_len(jr) == 4)
                rot = quat_norm(quat((real)json_num(json_at(jr, 0), 1), (real)json_num(json_at(jr, 1), 0),
                                     (real)json_num(json_at(jr, 2), 0), (real)json_num(json_at(jr, 3), 0)));
            stretch = json_vec3(json_get(ls, "stretch"), vec3(1, 1, 1));
        } else if (k != ED_MODEL && np > 0){          // version 1: from its first part, W = pos + R diag(kk) L
            Xform L = object_part_local(&shape, 0), W = got[pi[0]].xf;
            Mat3 R;
            real kv[3];
            for (int c = 0; c < 3; c++){
                Vec3 col = vec3(W.m[0][c] / L.m[c][c], W.m[1][c] / L.m[c][c], W.m[2][c] / L.m[c][c]);
                kv[c] = v_len(col);
                for (int r = 0; r < 3; r++) R.m[r][c] = v_at(col, r) / (kv[c] > 0 ? kv[c] : 1);
            }
            Vec3 kk = vec3(kv[0], kv[1], kv[2]);
            rot = quat_of_mat3(R);
            pos = v_sub(vec3(W.m[0][3], W.m[1][3], W.m[2][3]), mat3_mul_vec(R, v_mul(kk, vec3(L.m[0][3], L.m[1][3], L.m[2][3]))));
            stretch = v_scale(kk, 1 / scale);
        } else if (np > 0){                           // version 1, a model: the middle of its bottom, no turn
            Aabb b = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
            for (int q = 0; q < np; q++){ b = box_grow(b, got[pi[q]].box.lo); b = box_grow(b, got[pi[q]].box.hi); }
            pos = vec3((b.lo.x + b.hi.x) / 2, b.lo.y, (b.lo.z + b.hi.z) / 2);
            stretch = vec3(1 / scale, 1 / scale, 1 / scale);
        }
        int model = -1;
        if (k == ED_MODEL){                           // its parts in its own frame: L = diag(1/kk) R^T (W - pos)
            if (np == 0){ free(pi); continue; }
            E.models = lr_grow(E.models, sizeof(EdModel), &E.cap_models, E.nmodels + 1, 4);
            EdModel *m = &E.models[E.nmodels];
            m->nparts = np;
            m->clay = json_get(ls, "clay") != NULL;      // (a mesh file's: of clay, no material of its own)
            m->parts = malloc(sizeof(EdPart) * (size_t)np);
            if (!m->parts) lr_oom("editor");
            Mat3 Rt = mat3_transpose(quat_to_mat3(rot));
            Vec3 kk = v_scale(stretch, scale);
            m->box = (Aabb){vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
            for (int q = 0; q < np; q++){
                const Instance *in = &got[pi[q]];
                Xform L;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 4; c++){
                        real v = 0;
                        for (int t = 0; t < 3; t++) v += Rt.m[r][t] * (in->xf.m[t][c] - (c == 3 ? v_at(pos, t) : 0));
                        L.m[r][c] = v / v_at(kk, r);
                    }
                m->parts[q] = (EdPart){in->mesh, in->material, L};
                Aabb mb = s->meshes[in->mesh].box;
                for (int c = 0; c < 8; c++)
                    m->box = box_grow(m->box, xform_point(&L, vec3(c & 1 ? mb.hi.x : mb.lo.x, c & 2 ? mb.hi.y : mb.lo.y, c & 4 ? mb.hi.z : mb.lo.z)));
            }
            model = E.nmodels++;
        }
        // (the material the file has for its first part: the instance's, or its mesh's)
        int file_mat = np > 0 ? got[pi[0]].material : -1;
        if (file_mat < 0 && np > 0 && got[pi[0]].mesh >= 0 && s->meshes[got[pi[0]].mesh].ntris > 0)
            file_mat = s->meshes[got[pi[0]].mesh].tris[0].material;
        free(pi);
        int id = object_add((EdKind)k, mat, model, pos, scale, dynamic, 0, NULL);
        EdObject *o = object_find(id);
        const char *name = json_str(json_get(nd, "name"), NULL);
        if (name && *name) snprintf(o->name, sizeof(o->name), "%s", name);
        o->rot = rot;
        o->stretch = stretch;
        if (own){
            const Json *lk = json_get(ls, "look");
            const char *hex = json_str(json_get(lk, "rgb"), "#C9A27E");
            o->look_rgb = (unsigned)strtoul(hex[0] == '#' ? hex + 1 : hex, NULL, 16) & 0xffffffu;
            o->look_rough = (real)json_num(json_get(lk, "roughness"), 0.7);
            o->look_metal = (real)json_num(json_get(lk, "metallic"), 0);
            o->look_bump = (real)json_num(json_get(lk, "bump"), 1);
            o->look_normal_k = (real)json_num(json_get(lk, "normal"), 1);
            o->look_ao_k = (real)json_num(json_get(lk, "ao"), 1);
            o->look_glow = (real)json_num(json_get(lk, "glow"), 1);
            o->look_flip = 0;                            // (the file's normal map is turned over already)
            // its maps as the file has them: what was made from a bump or roughness and metallic
            // apart is one normal map and glTF's packed map now (their files' names kept)
            const int fm = file_mat;
            const Material *f = fm >= 0 ? &s->mats[fm] : NULL;
            const Json *mn = json_get(lk, "maps");
            const char *base_name = json_str(json_get(lk, "image"), "");
            int got_map[NLOOK] = {0};
            if (f){
                got_map[LOOK_BASE] = f->tex_base; got_map[LOOK_NORMAL] = f->tex_normal; got_map[LOOK_ORM] = f->tex_mr;
                got_map[LOOK_AO] = f->tex_occlusion; got_map[LOOK_EMISSIVE] = f->tex_emissive;
            }
            for (int q = 0; q < NLOOK; q++){
                o->look_map[q] = got_map[q] > 0 && got_map[q] <= s->ntexmaps ? got_map[q] : 0;
                const char *nm = json_str(json_get(mn, LOOK_KEYS[q]), q == LOOK_BASE ? base_name : "");
                if (q == LOOK_NORMAL && !*nm) nm = json_str(json_get(mn, "bump"), "");
                if (q == LOOK_ORM && !*nm) nm = json_str(json_get(mn, "roughness"), json_str(json_get(mn, "metallic"), ""));
                snprintf(o->look_map_name[q], sizeof(o->look_map_name[q]), "%s", o->look_map[q] ? (*nm ? nm : "from the file") : "");
                for (char *c = o->look_map_name[q]; *c; c++) if (*c == '"' || *c == '\\' || (unsigned char)*c < 32) *c = '_';
            }
            o->look = fm;
            if (look_sync(o) == 0) o->mat = MAT_OWN;
        }
        object_update(o);
        new_id[ni] = id;
        file_id[ni] = json_int(json_get(ls, "id"), 0);
        made++;
    }
    materials_compact(m0, k0);
    // what rests on what: the file's ids, now the new ones
    for (int ni = 0; ni < nn; ni++){
        if (!new_id[ni]) continue;
        int on = json_int(json_get(json_get(json_get(json_at(nodes, ni), "extensions"), EXT_SCENE), "restsOn"), 0);
        for (int q = 0; on && q < nn; q++)
            if (new_id[q] && file_id[q] == on){ object_find(new_id[ni])->support = new_id[q]; break; }
    }
    // the sky: a preset as it says; a file's only by its name (its .hdr: dropped with it)
    const Json *sky = json_get(root, "sky");
    const char *preset = json_str(json_get(sky, "preset"), NULL), *file = json_str(json_get(sky, "file"), NULL);
    int sp = preset ? (!strcmp(preset, "noon") ? SKY_NOON : !strcmp(preset, "overcast") ? SKY_OVERCAST : !strcmp(preset, "sun") ? SKY_SUN : SKY_DUSK) : -2;
    if (sp == SKY_SUN){                                // (the Light tab's: as it was set)
        const Json *su = json_get(sky, "sun");
        E.sun = (SunSky){(real)json_num(json_get(su, "t"), 0.35), (real)json_num(json_get(su, "intensity"), 1.2), (real)json_num(json_get(su, "size"), 0.53),
                         (real)json_num(json_get(su, "haze"), 2.5), (real)json_num(json_get(su, "az"), 150), json_int(json_get(su, "warm"), 1)};
        E.sun_w = 0;
        if (set_sky(SKY_SUN, NULL, NULL) == 0) viewer_scene_changed(1);
    } else if (sp >= 0 && sp != E.sky && set_sky(sp, NULL, NULL) == 0) viewer_scene_changed(1);
    int need_sky = file && !(E.sky == SKY_FILE && !strcmp(E.sky_name, file));
    jlen = 0;
    jprintf("{\"opened\":1,\"objects\":%d,\"skyFile\":", made);
    if (need_sky) jstring(file); else jprintf("null");
    jprintf(",\"renderer\":");
    jstring(json_str(json_get(root, "renderer"), "auto"));
    jprintf("}");
    free(new_id);
    free(file_id);
    free(got);
    free(node_of);
    json_free(j);
    changed();
    return json_buf;
}

// ---- the loop --------------------------------------------------------------------------------

static int editor_frame(void){
    double t = clock_s(), dt = fmin(t - E.t_prev, 0.1);
    E.t_prev = t;
    fly_step(dt);
    double t1 = clock_s();
    sim_frame(dt);
    double t2 = clock_s();
    relight_step();
    double t3 = clock_s();
    int r = viewer_frame();
    double t4 = clock_s();
    P.sim += (t2 - t1) * 1000; P.relight += (t3 - t2) * 1000; P.viewer += (t4 - t3) * 1000; P.frame += (t4 - t) * 1000;
    P.frames++;
    return r;
}

// where the time goes, summed since the start (ms), for the page to take differences of:
// the editor's own parts on the CPU (its simulation: steps and moving the objects; relighting;
// the viewer's frame), the viewer's parts, and the GPU's (its timers: path tracer, light
// tracer, BVH builds, guide, reprojection, denoiser)
EXPORT const char *lr_profile(void){
    double cpu[8], gpu[6];
    long vf = viewer_profile(cpu, gpu);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    jlen = 0;
    jprintf("{\"threads\":%d,\"frames\":%ld,\"steps\":%ld,\"cpu\":{\"frame\":%.3f,\"sim\":%.3f,\"simApply\":%.3f,\"relight\":%.3f,\"viewer\":%.3f},"
            "\"viewerFrames\":%ld,\"viewer\":{\"physics\":%.3f,\"trees\":%.3f,\"gpuScene\":%.3f,\"traceSubmit\":%.3f,\"waitGpu\":%.3f,\"raster\":%.3f,\"present\":%.3f,\"frame\":%.3f},"
            "\"gpu\":{\"trace\":%.3f,\"light\":%.3f,\"builds\":%.3f,\"guide\":%.3f,\"reproject\":%.3f,\"denoise\":%.3f}}",
            threads, P.frames, P.steps, P.frame, P.sim, P.apply, P.relight, P.viewer,
            vf, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4], cpu[5], cpu[6], cpu[7], gpu[0], gpu[1], gpu[2], gpu[3], gpu[4], gpu[5]);
    return json_buf;
}

#ifdef __EMSCRIPTEN__
static void frame_web(void){
    // (a cap: this animation frame skipped when the last one shown is too recent; a tenth of
    // a display frame early is on time)
    if (frame_cap_ms > 0){
        double now = emscripten_get_now(), hz = viewer_display_hz();
        if (now - frame_last_ms < frame_cap_ms - (hz > 0 ? 100.0 / hz : 1)) return;
        frame_last_ms = now;
    }
    if (!editor_frame()){ viewer_close(); emscripten_cancel_main_loop(); }
}
#endif

int main(int argc, char **argv){
    // the editor's defaults before the page's own options (the address: ?n=...): Full HD,
    // path traced, 64 samples per pixel shown as converged (the page offers Raster and Path
    // traced; the engine's Auto, raster while things move, is left to the address)
    const char *pre[] = {"-width", "1920", "-height", "1080", "-backend", "rt", "-s", "64"};
    int npre = (int)(sizeof(pre) / sizeof(pre[0]));
    char **args = malloc((size_t)(argc + npre + 1) * sizeof(char *));
    if (!args) return 1;
    args[0] = argv[0];
    for (int i = 0; i < npre; i++) args[1 + i] = (char *)pre[i];
    for (int i = 1; i < argc; i++) args[npre + i] = argv[i];
    args[argc + npre] = NULL;
    Options *o = viewer_options();
    options_parse(o, PROG_VIEW, argc + npre, args);
    E.sun = (SunSky){0.35f, 1.2f, 0.53f, 2.5f, 150, 1};   // (the Light tab's sun until it is moved: dusk, as the design has it)

    Scene *s = viewer_scene();
    if (o->sim.on){
        // a sample (the page's ?sample=...: bin/view's options for the demo): the engine's
        // physics demo as bin/view sets it up, its seed, camera, materials and light
        E.sample = 1;
        snprintf(E.sky_name, sizeof(E.sky_name), "%s", o->scene.sky ? (strrchr(o->scene.sky, '/') ? strrchr(o->scene.sky, '/') + 1 : o->scene.sky) : "Procedural");
        E.sky = SKY_FILE;
        if (viewer_setup_scene()) return 1;
    } else {
        if (build_scene(s)) return 1;
        scene_set_accel(s, options_resolve_accel(o, s), &o->ap);
    }
    if (viewer_open("Kiln") != 0) return 1;
    // the rasterizer's light probe: made in steps between the frames, as after a change (the
    // sky lights the raster until it is done). Baked at once, before the first frame, it held
    // the page 3 s at every start (the browser has one thread for it)
    if (!E.sample) relight_start();

    E.pivot = s->cam_at;
    E.dist = v_len(v_sub(s->cam_at, s->cam_from));
    E.t_prev = clock_s();
#ifdef __EMSCRIPTEN__
    // the page: its own interface takes the keys and the mouse, and shows the state
    viewer_set_input(0);
    viewer_set_title(0);
    viewer_set_hud(0);
    viewer_set_settle(0.7);                      // "Auto": raster for 0.7 s after a change
    viewer_set_keep_trace(1);                    // ... then the path traced image carries on
    emscripten_set_main_loop(frame_web, 0, 1);
#else
    viewer_set_settle(0.7);
    viewer_set_keep_trace(1);
    viewer_print_controls();
    while (editor_frame()){}
    viewer_close();
#endif
    free(args);
    return 0;
}
