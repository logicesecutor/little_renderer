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

#define ED_MAX_INST 3
typedef struct {
    int id;
    EdKind kind;
    char name[64];
    int mat;                       // index into MATERIALS (-1: a model's own)
    int mesh;                      // a rock's or a model's mesh (Scene.meshes)
    int dynamic;                   // its body in the physics (Simulate mode): 0 static, 1 dynamic
    real scale;                    // 1 = the shape's own size (a model: the file's)
    Vec3 pos;                      // the middle of its bottom, units
    real yaw;                      // degrees about the vertical
    int support;                   // the id of what it rests on (0: the ground)
    int inst[ED_MAX_INST], ninst;  // its instances in the scene
} EdObject;

typedef struct {
    EdObject *objs;
    int nobjs, cap, next_id;
    int mats[NMATERIALS];          // the catalogue's materials in the scene
    int dark;                      // the lamp's stand
    int rock_mesh;                 // the rock's mesh (made here)
    const char *sky;               // the sky's name
    // the orbit camera: around a point, at a distance
    Vec3 pivot;
    double dist;
    int fly_keys;                  // W A S D E Q Shift held while flying (lr_camera_fly)
    double t_prev;
} Editor;

static Editor E;

static Mat3 yaw_mat(real deg){
    real a = deg * (real)PI / 180, c = r_cos(a), s = r_sin(a);
    Mat3 r = {{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}};
    return r;
}

// the instance transforms of an object: its parts in its own space (units, bottom centre at
// the origin), scaled, turned, moved to pos
static void object_xforms(const EdObject *o, Xform *xf, int *n){
    real s = o->scale;
    Mat3 r = yaw_mat(o->yaw);
    Vec3 p = o->pos;
    switch (o->kind){
    case ED_BOX:
        xf[0] = xform_axes(r, vec3(50 * s, 50 * s, 50 * s), v_add(p, vec3(0, 50 * s, 0)));
        *n = 1;
        break;
    case ED_SPHERE:
        xf[0] = xform_axes(r, vec3(50 * s, 50 * s, 50 * s), v_add(p, vec3(0, 50 * s, 0)));
        *n = 1;
        break;
    case ED_ROCK:
    case ED_MODEL:
        xf[0] = xform_axes(r, vec3(s, s, s), p);
        *n = 1;
        break;
    case ED_LAMP:   // a base, a pole and the glowing ball on top
        xf[0] = xform_axes(r, vec3(18 * s, 3 * s, 18 * s), v_add(p, vec3(0, 3 * s, 0)));
        xf[1] = xform_axes(r, vec3(3 * s, 32 * s, 3 * s), v_add(p, vec3(0, 38 * s, 0)));
        xf[2] = xform_axes(r, vec3(20 * s, 20 * s, 20 * s), v_add(p, vec3(0, 88 * s, 0)));
        *n = 3;
        break;
    }
}

// the materials of an object's parts: the lamp's stand is dark, the rest its material (a
// model: its triangles' own)
static int part_material(const EdObject *o, int part){
    if (o->kind == ED_LAMP && part < 2) return E.dark;
    return o->mat >= 0 ? E.mats[o->mat] : -1;
}
static int part_mesh(const EdObject *o, int part){
    switch (o->kind){
    case ED_BOX: return INST_BOX;
    case ED_SPHERE: return INST_SPHERE;
    case ED_LAMP: return part < 2 ? INST_BOX : INST_SPHERE;
    default: return o->mesh;
    }
}

// its instances made (an object new, or back by an undo)
static void object_instances(EdObject *o){
    Scene *s = viewer_scene();
    Xform xf[ED_MAX_INST];
    int n = 0;
    object_xforms(o, xf, &n);
    for (int i = 0; i < n; i++) o->inst[i] = scene_add_instance(s, part_mesh(o, i), xf[i], part_material(o, i));
    o->ninst = n;
}

// its instances where its fields say, of its material
static void object_update(EdObject *o){
    Scene *s = viewer_scene();
    Xform xf[ED_MAX_INST];
    int n = 0;
    object_xforms(o, xf, &n);
    for (int i = 0; i < n && i < o->ninst; i++){
        s->insts[o->inst[i]].material = part_material(o, i);
        scene_move_instance(s, o->inst[i], xf[i]);          // (a new version: the copies take it again)
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
    o->ninst = 0;
}

// label: the name before its number ("Glass ball"; NULL: the kind's)
static int object_add(EdKind kind, int mat, Vec3 pos, real scale, int dynamic, int support, const char *label){
    E.objs = lr_grow(E.objs, sizeof(EdObject), &E.cap, E.nobjs + 1, 16);
    EdObject *o = &E.objs[E.nobjs];
    memset(o, 0, sizeof(*o));
    o->id = ++E.next_id;
    o->kind = kind;
    o->mat = mat;
    o->mesh = kind == ED_ROCK ? E.rock_mesh : -1;
    o->scale = scale;
    o->pos = pos;
    o->dynamic = dynamic;
    o->support = support;
    if (label) snprintf(o->name, sizeof(o->name), "%s %d", label, o->id);
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

// how tall it is (where something put on it stands), and how far from its middle its top
// reaches (where on it something can stand), units
static real object_top(const EdObject *o){
    switch (o->kind){
    case ED_BOX: case ED_SPHERE: return 100 * o->scale;
    case ED_LAMP: return 108 * o->scale;
    default: return viewer_scene()->meshes[o->mesh].box.hi.y * o->scale;
    }
}
static real object_reach(const EdObject *o){
    switch (o->kind){
    case ED_BOX: return 40 * o->scale;
    case ED_SPHERE: return 12 * o->scale;
    case ED_LAMP: return 6 * o->scale;
    default: {
        Aabb b = viewer_scene()->meshes[o->mesh].box;
        return 0.3f * r_fmin(b.hi.x - b.lo.x, b.hi.z - b.lo.z) * o->scale;
    }
    }
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

// the sky presets (Light mode will set them; v1: Dusk)
static int set_sky(Scene *s, const char *name){
    Env *e = NULL;
    if (!strcmp(name, "Dusk")) e = env_procedural_sky(1024, 4, 150);
    else if (!strcmp(name, "Noon")) e = env_procedural_sky(1024, 60, 150);
    if (!e) return 1;
    if (s->env) env_free(s->env);
    s->env = e;
    E.sky = name;
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
    // the ground: 200 m of plain earth-coloured floor
    int ground = scene_add_material(s, material_pbr(srgb_hex(0x8A7E6C), 0, 0.95, 1.5, 0));
    scene_add_prim(s, make_quad(vec3(-10000, 0, -10000), vec3(0, 0, 20000), vec3(20000, 0, 0), ground));
    s->light = -1;
    if (set_sky(s, "Dusk")) return 1;
    s->probe_set = 1;
    s->probe_at = vec3(0, 150, 0);
    s->probe_box = (Aabb){vec3(-10000, 0, -10000), vec3(10000, 1e5f, 10000)};
    E.rock_mesh = make_rock(s);
    // the prototype's starting set: a rock, a box with a smaller steel one on it, a lamp,
    // a terracotta ball and a glass one (the camera looks down +z: +x is on the left)
    object_add(ED_ROCK, 2, vec3(250, 0, 260), 1, 0, 0, NULL);
    int box = object_add(ED_BOX, 0, vec3(40, 0, 120), 1, 0, 0, NULL);
    object_add(ED_BOX, 3, vec3(43, 100, 118), 0.7f, 1, box, NULL);
    object_add(ED_LAMP, 7, vec3(-240, 0, 200), 1, 0, 0, NULL);
    object_add(ED_SPHERE, 1, vec3(-120, 0, -40), 1, 1, 0, NULL);
    object_add(ED_SPHERE, 6, vec3(170, 0, -110), 1, 1, 0, "Glass ball");
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

static void changed(void){ viewer_scene_changed(0); }   // (the path traced image keeps what did not change)

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
    int id = object_add(SH[shape].kind, SH[shape].mat, metres(x, y, z), 1, 0, support, SH[shape].label);
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

// static or dynamic (its body in Simulate mode)
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

typedef struct { EdObject *objs; int n; } Snapshot;
#define HISTORY 100
static Snapshot undo_st[HISTORY], redo_st[HISTORY];
static int nundo, nredo;

static Snapshot snapshot_take(void){
    Snapshot sn = {malloc((size_t)(E.nobjs > 0 ? E.nobjs : 1) * sizeof(EdObject)), E.nobjs};
    if (!sn.objs) lr_oom("editor");
    memcpy(sn.objs, E.objs, (size_t)E.nobjs * sizeof(EdObject));
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
        now[j].ninst = 0;
        if (cur){ memcpy(now[j].inst, cur->inst, sizeof(cur->inst)); now[j].ninst = cur->ninst; }
    }
    free(E.objs);
    E.objs = now;
    E.cap = sn->n > 0 ? sn->n : 1;
    E.nobjs = sn->n;
    for (int j = 0; j < E.nobjs; j++){
        if (E.objs[j].ninst == 0) object_instances(&E.objs[j]);
        else object_update(&E.objs[j]);
    }
    changed();
}

// before a change (an add, a move, a size, a material, a body, a removal): the scene as it is
EXPORT void lr_checkpoint(void){
    snapshot_push(undo_st, &nundo, snapshot_take());
    while (nredo > 0) free(redo_st[--nredo].objs);
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
            "\"fps\":%.1f,\"width\":%d,\"height\":%d,\"instances\":%d,\"emitters\":%d}",
            (int)viewer_backend(), st.tracing, st.on_gpu, st.gpu_starting, st.gpu_tracing, st.spp, st.spp_target,
            st.fps, st.width, st.height, viewer_scene()->ninsts, emitters);
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
    jstring(E.sky ? E.sky : "");
    jprintf(",\"undo\":%d,\"redo\":%d,\"objects\":[", nundo, nredo);
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        jprintf("%s{\"id\":%d,\"name\":", i ? "," : "", o->id);
        jstring(o->name);
        jprintf(",\"kind\":\"%s\",\"material\":\"%s\",\"dynamic\":%d,\"scale\":%.3f,"
                "\"pos\":[%.3f,%.3f,%.3f],\"top\":%.3f,\"yaw\":%.1f,\"support\":%d}",
                KIND_NAMES[o->kind], o->mat >= 0 ? MATERIALS[o->mat].key : "",
                o->dynamic, (double)o->scale, (double)o->pos.x / UNITS_PER_M, (double)o->pos.y / UNITS_PER_M,
                (double)o->pos.z / UNITS_PER_M, (double)object_top(o) / UNITS_PER_M, (double)o->yaw, o->support);
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
        Vec3 d = v_sub(p, on->pos);
        d.y = 0;
        real r = object_reach(on), l = v_len(d);
        if (l > r) d = v_scale(d, r / l);
        land = vec3(on->pos.x + d.x, on->pos.y + object_top(on), on->pos.z + d.z);
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
