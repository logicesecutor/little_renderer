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

#include "app/viewer.h"
#include "lr.h"
#include "base/mem.h"
#include "scene/env.h"
#include "rt/accel.h"
#include "rt/intersect.h"
#include "scene/camera.h"
#include "scene/gltf.h"
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
    int mat;                       // index into MATERIALS (-1: a model's own)
    int mesh;                      // a rock's mesh (Scene.meshes)
    int model;                     // a model's index into Editor.models
    int dynamic;                   // its body in the physics (Simulate mode): 0 static, 1 dynamic
    real scale;                    // 1 = the shape's own size (a model: the file's)
    Vec3 pos;                      // the middle of its bottom, units
    Quat rot;                      // its turn, about the middle of its bottom
    Vec3 stretch;                  // its size along its own axes, times scale (the gizmo's Scale: 1, 1, 1 at first)
    int support;                   // the id of what it rests on (0: the ground)
    int *inst, ninst;              // its instances in the scene (its own array)
} EdObject;

// a model read from a file: its parts as the file places them, at its own size, the middle
// of its bottom at the origin (kept while the editor runs, for undo)
typedef struct { int mesh, material; Xform xf; } EdPart;
typedef struct { EdPart *parts; int nparts; Aabb box; int clay; } EdModel;   // (clay: a mesh file's, of no material of its own)

// the sky presets (the spec, 7.3), and a file's
enum { SKY_DUSK, SKY_NOON, SKY_OVERCAST, NSKIES, SKY_FILE = -1 };
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
    int sky;                       // the sky: a preset, or SKY_FILE (sky_path)
    char sky_name[96], sky_path[256];
    struct ProbeJob *relight;      // the rasterizer's light being made again (NULL: up to date)
    double relight_done;
    // the orbit camera: around a point, at a distance
    Vec3 pivot;
    double dist;
    int fly_keys;                  // W A S D E Q Shift held while flying (lr_camera_fly)
    double t_prev;
} Editor;

static Editor E;

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

// the sky made: a preset, or a file (.hdr) when path is given; 0 = done (else nothing changed)
static int set_sky(int preset, const char *path, const char *name){
    Scene *s = viewer_scene();
    Env *e = path ? env_load_hdr(path) : preset >= 0 && preset < NSKIES ? make_sky(preset) : NULL;
    if (!e) return 1;
    env_free(s->env);
    s->env = e;
    E.sky = path ? SKY_FILE : preset;
    snprintf(E.sky_path, sizeof(E.sky_path), "%s", path ? path : "");
    snprintf(E.sky_name, sizeof(E.sky_name), "%s", path ? name : SKY_NAMES[preset]);
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

static void fly_step(double dt){
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

// ---- what the page sets ---------------------------------------------------------------------

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
    viewer_scene_changed(0);
    relight_start();
}

// shapes as the tray has them (the spec, 7.1), with their default materials (6.4)
enum { SHAPE_BOX, SHAPE_SPHERE, SHAPE_GLASS_BALL, SHAPE_ROCK, SHAPE_LAMP, NSHAPES };

static Vec3 metres(double x, double y, double z){
    return vec3((real)(x * UNITS_PER_M), (real)(y * UNITS_PER_M), (real)(z * UNITS_PER_M));
}

// a shape standing at (x, y, z) (the middle of its bottom), on `support` (0: the ground);
// static, at its own size. Returns its id (0: none).
EXPORT int lr_add_shape(int shape, double x, double y, double z, int support){
    static const struct { EdKind kind; int mat; const char *label; } SH[NSHAPES] = {
        {ED_BOX, 0, NULL}, {ED_SPHERE, 1, NULL}, {ED_SPHERE, 6, "Glass ball"}, {ED_ROCK, 2, NULL}, {ED_LAMP, 7, NULL}};
    if (shape < 0 || shape >= NSHAPES) return 0;
    if (support && !object_find(support)) support = 0;
    int id = object_add(SH[shape].kind, SH[shape].mat, -1, metres(x, y, z), 1, 0, support, SH[shape].label);
    changed();
    return id;
}

// moved to stand at (x, y, z) on `support` (0: the ground), what rests on it with it; not onto
// itself or what rests on it. 0 = done.
EXPORT int lr_move_object(int id, double x, double y, double z, int support){
    EdObject *o = object_find(id);
    if (!o || (support && (!object_find(support) || in_stack(support, id)))) return 1;
    o->support = support;
    object_shift(id, v_sub(metres(x, y, z), o->pos), 0);
    changed();
    return 0;
}

// its size (1: its own), what rests on it raised or lowered with its top
EXPORT int lr_scale_object(int id, double scale){
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
    Scene *s = viewer_scene();
    int n0 = s->ninsts;
    int n = scene_load_gltf(s, path, (real)-UNITS_PER_M, vec3(0, 0, 0), 0);
    if (n <= 0) return 0;
    // its instances, as the file places them standing at the origin: the model's parts; they
    // become the object's (made again from the parts: the same, in the object's place)
    E.models = lr_grow(E.models, sizeof(EdModel), &E.cap_models, E.nmodels + 1, 4);
    EdModel *m = &E.models[E.nmodels];
    m->nparts = s->ninsts - n0;
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
    changed();
    return id;
}

// a mesh file (.obj, ASCII .ply) as an object of one part, of clay (the file has no material).
// Its units are taken as metres; one under 1 cm or over 100 m on its longest side would make
// no sense: it is fitted to 1 m (lr_last_import_fitted says so). Returns its id (0: it could
// not be read).
EXPORT int lr_import_mesh(const char *path, double x, double y, double z, int support){
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
    changed();
    return id;
}
EXPORT int lr_last_import_fitted(void){ return E.last_fitted; }

// the sky: a preset (0 Dusk, 1 Noon, 2 Overcast), or a Radiance .hdr file (name: what the
// page shows); the path traced image starts over. 0 = done.
EXPORT int lr_set_sky(int preset){
    if (set_sky(preset, NULL, NULL)) return 1;
    viewer_scene_changed(1);
    relight_start();
    return 0;
}
EXPORT int lr_load_sky(const char *path, const char *name){
    if (set_sky(SKY_FILE, path, name)) return 1;
    viewer_scene_changed(1);
    relight_start();
    return 0;
}

EXPORT int lr_set_body(int id, int dynamic){
    EdObject *o = object_find(id);
    if (!o) return 1;
    o->dynamic = dynamic != 0;
    return 0;
}

// removed; what rested on it comes down to where it stood, onto what it stood on
EXPORT int lr_remove_object(int id){
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

typedef struct { EdObject *objs; int n; int sky; char sky_name[96], sky_path[256]; } Snapshot;
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
        if (E.objs[j].ninst == 0) object_instances(&E.objs[j]);
        else object_update(&E.objs[j]);
    }
    if (sn->sky != E.sky || strcmp(sn->sky_path, E.sky_path)){     // (another sky: the path traced image starts over)
        if (set_sky(sn->sky, sn->sky == SKY_FILE ? sn->sky_path : NULL, sn->sky_name) == 0) viewer_scene_changed(1);
    }
    changed();
}

// before a change (an add, a move, a size, a material, a body, a removal): the scene as it is
EXPORT void lr_checkpoint(void){
    snapshot_push(undo_st, &nundo, snapshot_take());
    while (nredo > 0) free(redo_st[--nredo].objs);
}
// the last checkpoint, when nothing changed after it after all (a press without a drag)
EXPORT void lr_forget_checkpoint(void){
    if (nundo > 0) free(undo_st[--nundo].objs);
}
EXPORT int lr_undo(void){
    if (nundo == 0) return 0;
    snapshot_push(redo_st, &nredo, snapshot_take());
    Snapshot sn = undo_st[--nundo];
    snapshot_apply(&sn);
    free(sn.objs);
    return 1;
}
EXPORT int lr_redo(void){
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
EXPORT const char *lr_status(void){
    ViewerStatus st;
    viewer_status(&st);
    int emitters = 0;
    for (int i = 0; i < E.nobjs; i++) emitters += E.objs[i].mat >= 0 && MATERIALS[E.objs[i].mat].glow > 0;
    jlen = 0;
    jprintf("{\"backend\":%d,\"tracing\":%d,\"gpu\":%d,\"gpuStarting\":%d,\"gpuTracing\":%d,\"spp\":%d,\"target\":%d,"
            "\"fps\":%.1f,\"width\":%d,\"height\":%d,\"instances\":%d,\"emitters\":%d,\"relighting\":%d,\"relight\":%.3f}",
            (int)viewer_backend(), st.tracing, st.on_gpu, st.gpu_starting, st.gpu_tracing, st.spp, st.spp_target,
            st.fps, st.width, st.height, viewer_scene()->ninsts, emitters, E.relight != NULL, E.relight_done);
    return json_buf;
}

// a string for JSON: quotes and backslashes escaped, control characters dropped
static void jstring(const char *t){
    jprintf("\"");
    for (; *t; t++){
        if (*t == '"' || *t == '\\') jprintf("\\%c", *t);
        else if ((unsigned char)*t >= 32) jprintf("%c", *t);
    }
    jprintf("\"");
}

// the scene as the page lists it: {"sky":"Dusk","objects":[{"id":1,"name":"Rock 1",...},...]}
// (metres; pos: the middle of its bottom, top: its height)
EXPORT const char *lr_scene(void){
    jlen = 0;
    jprintf("{\"sky\":");
    jstring(E.sky_name);
    jprintf(",\"skyKey\":\"%s\"", E.sky == SKY_FILE ? "file" : E.sky == SKY_NOON ? "noon" : E.sky == SKY_OVERCAST ? "overcast" : "dusk");
    jprintf(",\"undo\":%d,\"redo\":%d,\"objects\":[", nundo, nredo);
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        Aabb bx = object_box(o);
        Vec3 c = v_scale(v_add(bx.lo, bx.hi), 0.5f);
        jprintf("%s{\"id\":%d,\"name\":", i ? "," : "", o->id);
        jstring(o->name);
        jprintf(",\"kind\":\"%s\",\"material\":\"%s\",\"dynamic\":%d,\"scale\":%.3f,"
                "\"pos\":[%.4f,%.4f,%.4f],\"top\":%.4f,\"rot\":[%.6f,%.6f,%.6f,%.6f],\"stretch\":[%.4f,%.4f,%.4f],"
                "\"centre\":[%.4f,%.4f,%.4f],\"support\":%d}",
                KIND_NAMES[o->kind], o->mat >= 0 ? MATERIALS[o->mat].key : o->kind == ED_MODEL && E.models[o->model].clay ? "clay" : "",
                o->dynamic, (double)o->scale, (double)o->pos.x / UNITS_PER_M, (double)o->pos.y / UNITS_PER_M,
                (double)o->pos.z / UNITS_PER_M, (double)object_top(o) / UNITS_PER_M,
                (double)o->rot.w, (double)o->rot.x, (double)o->rot.y, (double)o->rot.z, (double)o->stretch.x, (double)o->stretch.y, (double)o->stretch.z,
                (double)c.x / UNITS_PER_M, (double)c.y / UNITS_PER_M, (double)c.z / UNITS_PER_M, o->support);
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

// ---- the loop --------------------------------------------------------------------------------

static int editor_frame(void){
    double t = clock_s(), dt = fmin(t - E.t_prev, 0.1);
    E.t_prev = t;
    fly_step(dt);
    relight_step();
    return viewer_frame();
}

#ifdef __EMSCRIPTEN__
static void frame_web(void){
    if (!editor_frame()){ viewer_close(); emscripten_cancel_main_loop(); }
}
#endif

int main(int argc, char **argv){
    // the editor's defaults before the page's own options (the address: ?n=...): Full HD,
    // auto (hybrid), 64 samples per pixel shown as converged
    const char *pre[] = {"-width", "1920", "-height", "1080", "-backend", "hybrid", "-s", "64"};
    int npre = (int)(sizeof(pre) / sizeof(pre[0]));
    char **args = malloc((size_t)(argc + npre + 1) * sizeof(char *));
    if (!args) return 1;
    args[0] = argv[0];
    for (int i = 0; i < npre; i++) args[1 + i] = (char *)pre[i];
    for (int i = 1; i < argc; i++) args[npre + i] = argv[i];
    args[argc + npre] = NULL;
    Options *o = viewer_options();
    options_parse(o, PROG_VIEW, argc + npre, args);

    Scene *s = viewer_scene();
    if (build_scene(s)) return 1;
    scene_set_accel(s, options_resolve_accel(o, s), &o->ap);
    options_bake_probe(o, s);
    if (viewer_open("Kiln") != 0) return 1;

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
