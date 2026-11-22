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

static int object_add(EdKind kind, int mat, Vec3 pos, real scale, int dynamic, int support, const char *name){
    Scene *s = viewer_scene();
    E.objs = lr_grow(E.objs, sizeof(EdObject), &E.cap, E.nobjs + 1, 16);
    EdObject *o = &E.objs[E.nobjs];
    memset(o, 0, sizeof(*o));
    o->id = ++E.next_id;
    o->kind = kind;
    o->mat = mat;
    o->scale = scale;
    o->pos = pos;
    o->dynamic = dynamic;
    o->support = support;
    if (name) snprintf(o->name, sizeof(o->name), "%s", name);
    else snprintf(o->name, sizeof(o->name), "%c%s %d", KIND_NAMES[kind][0] - 32, KIND_NAMES[kind] + 1, o->id);
    Xform xf[ED_MAX_INST];
    int n = 0;
    object_xforms(o, xf, &n);
    int m = mat >= 0 ? E.mats[mat] : -1;
    for (int i = 0; i < n; i++){
        int mesh = kind == ED_BOX || kind == ED_LAMP ? INST_BOX : kind == ED_SPHERE ? INST_SPHERE : E.rock_mesh;
        int im = m;
        if (kind == ED_LAMP){
            mesh = i < 2 ? INST_BOX : INST_SPHERE;
            im = i < 2 ? E.dark : m;
        }
        o->inst[i] = scene_add_instance(s, mesh, xf[i], im);
    }
    o->ninst = n;
    E.nobjs++;
    return o->id;
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
    object_add(ED_SPHERE, 6, vec3(170, 0, -110), 1, 1, 0, NULL);
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

// the scene as the page lists it: {"sky":"Dusk","objects":[{"id":1,"name":"Rock 1",...},...]}
EXPORT const char *lr_scene(void){
    jlen = 0;
    jprintf("{\"sky\":\"%s\",\"objects\":[", E.sky ? E.sky : "");
    for (int i = 0; i < E.nobjs; i++){
        const EdObject *o = &E.objs[i];
        jprintf("%s{\"id\":%d,\"name\":\"%s\",\"kind\":\"%s\",\"material\":\"%s\",\"dynamic\":%d,\"scale\":%.3f,"
                "\"pos\":[%.3f,%.3f,%.3f],\"support\":%d}",
                i ? "," : "", o->id, o->name, KIND_NAMES[o->kind], o->mat >= 0 ? MATERIALS[o->mat].key : "",
                o->dynamic, (double)o->scale, (double)o->pos.x / UNITS_PER_M, (double)o->pos.y / UNITS_PER_M,
                (double)o->pos.z / UNITS_PER_M, o->support);
    }
    jprintf("]}");
    return json_buf;
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
