// bench.c - the performance suite (docs/STATUS.md 1.3): every number is timed by this
// program itself (the CPU's clock around work it waited for, and the GPU's own timers where
// the GPU has them), offscreen, no vsync, with fixed inputs: the same scenes, cameras,
// seeds, time steps and counts of passes, frames or steps every time. Warm-up rounds are
// left out; each case is repeated and its median kept, with the fastest and slowest.
//
//   bin/bench                   the quick set (about a minute)
//   bin/bench -full             every case (about ten minutes)
//   bin/bench -only A,F3        these groups or cases (with -full: of the full set)
//   bin/bench -list             the cases
//   bin/bench -o file.csv -tag label   append the results to a CSV file, labelled
//   bin/bench -compare old.csv new.csv the cases of both, before -> after
//   bin/bench -csv              also the CSV lines on the output ("CSV ...")
//   bin/bench -table a.csv b.csv ...   a Markdown table of their cases side by side (each
//                               file a column, named by its backend: docs/STATUS.md section 2)
//
// In a browser (make web-bench: web/build/bench.html, built with -sJSPI so it can wait
// for the GPU) the options come from the address (bench.html?full&only=A,F3) and the CSV
// lines from the output: tools/webbench.mjs runs it in a headless Chrome and keeps them.
//
// Groups: A the GPU path tracer, B the CPU path tracer, C the rasterizers, D the denoiser,
// E the trees and the scene's updates, F the rigid bodies, G the air and the cloud, H the
// start. Natively it measures our Vulkan layer (column vk; NOGPU=1: cpu).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "app/options.h"
#include "app/simdemo.h"
#include "app/gpu_frame.h"
#include "formats/png.h"
#include "gpu/gpu.h"
#include "gpu/view_gpu.h"
#include "gpu/rigid_gpu.h"
#include "gpu/fluid_gpu.h"
#include "lr.h"

#define MAXREP 16
#define DRAGON "models/dragon_recon/dragon_vrip.ply"
#define DRAGON2 "models/dragon_recon/dragon_vrip_res2.ply"
#define SKY "models/skies/kloofendal_48d_partly_cloudy_2k.hdr"
#define HELMET "models/gltf/DamagedHelmet.glb"

#ifdef LR_VK
static const char *BACKEND = "vk";
#elif defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__)
static const char *BACKEND = "web-mt";      // (make web-bench THREADS=1: bench_mt.html)
#elif defined(__EMSCRIPTEN__)
static const char *BACKEND = "web";
#else
static const char *BACKEND = "cpu";        // (make NOGPU=1)
#endif

// ---- results ----------------------------------------------------------------------------

typedef struct {
    double v[MAXREP];       // each repetition's value
    int n;
    double gpu[MAXREP];     // the GPU's own time for the same unit, when it has timers
    int ngpu;
    char note[160];         // what the case found besides (counts, sizes); or why it was skipped
    int skipped;
} Res;

static void add(Res *r, double v){ if (r->n < MAXREP) r->v[r->n++] = v; }
static void add_gpu(Res *r, double v){ if (v > 0 && r->ngpu < MAXREP) r->gpu[r->ngpu++] = v; }
static void skip(Res *r, const char *why){ r->skipped = 1; snprintf(r->note, sizeof(r->note), "%s", why); }

static int cmp_d(const void *a, const void *b){ double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n){
    if (n <= 0) return 0;
    double t[MAXREP];
    memcpy(t, v, (size_t)n * sizeof(double));
    qsort(t, (size_t)n, sizeof(double), cmp_d);
    return n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}
static double vmin(const double *v, int n){ double m = n ? v[0] : 0; for (int i = 1; i < n; i++) if (v[i] < m) m = v[i]; return m; }
static double vmax(const double *v, int n){ double m = n ? v[0] : 0; for (int i = 1; i < n; i++) if (v[i] > m) m = v[i]; return m; }

// ---- the cases ----------------------------------------------------------------------------

typedef struct Case Case;
typedef void (*CaseFn)(const Case *c, Res *r, int reps);
struct Case {
    const char *id;
    const char *what;
    const char *unit;       // "ms/..." (less is better) or "M paths/s" (more is better)
    int quick;              // in the quick set too
    CaseFn fn;
    const char *args;       // the scene, as options of bin/render (or bin/view, for -sim on)
    int a, b;               // the case's own numbers (threads, bodies, ...)
};

static int full;            // -full
static int csv_out;         // -csv: the CSV lines on the output too
static Sim SIM;             // the physics demo of a case (big: not on the stack)
static int quiet;

// ---- helpers ----------------------------------------------------------------------------

static void set_env(const char *name, const char *value){
#ifdef _WIN32
    char buf[256];
    snprintf(buf, sizeof(buf), "%s=%s", name, value ? value : "");
    _putenv(buf);
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static int max_threads(void){
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}
static void set_threads(int n){
#ifdef _OPENMP
    omp_set_num_threads(n > 0 ? n : omp_get_num_procs());
#else
    (void)n;
#endif
}

// The options a case gives, as if on the command line of bin/render (or bin/view if they
// start with -sim on). The scene they describe is made into *s (and *sim for the physics
// demos). 0 on success.
static int setup(const char *args, Options *o, Scene *s, Sim *sim){
    char buf[512];
    char *argv[64];
    int argc = 0;
    snprintf(buf, sizeof(buf), "%s", args);
    argv[argc++] = "bench";
    for (char *t = strtok(buf, " "); t && argc < 63; t = strtok(NULL, " ")) argv[argc++] = t;
    argv[argc] = NULL;
    int is_sim = strstr(args, "-sim on") != NULL;
    memset(s, 0, sizeof(*s));
    options_parse(o, is_sim ? PROG_VIEW : PROG_RENDER, argc, argv);
    if (is_sim){
        memset(sim, 0, sizeof(*sim));
        if (sim_setup(sim, o, s) != 0) return 1;
        sim_step(sim, s, 0);                     // the bodies that start at t = 0
        scene_set_accel(s, ACCEL_BVH, &o->ap);
        return 0;
    }
    return options_setup_scene(o, s);
}
static void teardown(Options *o, Scene *s, Sim *sim){
    if (o->sim.on) sim_free(sim);
    scene_free(s);
}

static int file_there(const char *path){ FILE *f = fopen(path, "rb"); if (f) fclose(f); return f != NULL; }
static int models_there(const char *args, Res *r){
    const char *need[] = {DRAGON, DRAGON2, SKY, HELMET};
    for (int i = 0; i < 4; i++)
        if (strstr(args, need[i]) && !file_there(need[i])){ char why[160]; snprintf(why, sizeof(why), "no %s", need[i]); skip(r, why); return 0; }
    return 1;
}

// until a new GPU is ready (or 20 s)
static void gpu_ready_wait(Gpu *g){
    double t0 = clock_s();
    while (g && gpu_state(g) == GPU_PENDING && clock_s() - t0 < 20) gpu_poll(g);
}

// the GPU, made once (hwrt = 0: our layer without its hardware ray queries, a second one)
static Gpu *gpus[2];
static Gpu *gpu_get(int hwrt_off){
    if (!gpu_available()) return NULL;
    if (gpus[hwrt_off]) return gpu_state(gpus[hwrt_off]) == GPU_READY ? gpus[hwrt_off] : NULL;
    gpu_set_timing(1);
    if (hwrt_off) set_env("LR_GPU_HWRT", "0");
    Gpu *g = gpu_create(NULL, 64, 64);
    gpu_ready_wait(g);
    if (hwrt_off) set_env("LR_GPU_HWRT", NULL);
    gpus[hwrt_off] = g;
    return g && gpu_state(g) == GPU_READY ? g : NULL;
}

static Camera scene_camera(const Scene *s, int w, int h){
    return camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)w / h);
}
static RenderParams scene_params(const Options *o, const Scene *s){
    RenderParams p = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, o->width, o->height);
    p.mode = o->shading;
    return p;
}

// passes of the GPU path tracer until `upto` are done (gpu_view_trace_limit set before);
// 0 if they never will (the GPU lost, or the path tracer's pipelines failed)
static int trace_until(Gpu *g, GpuView *v, const Scene *s, const RenderParams *rp, const MarchParams *mp, int upto){
    while (gpu_view_trace_passes(v) < upto && gpu_state(g) == GPU_READY){
        if (!gpu_view_trace_more(v, s, rp, mp)) return 0;
        gpu_wait(g);
    }
    return gpu_view_trace_passes(v) >= upto;
}

// ---- A: the GPU path tracer ----------------------------------------------------------------
// ms per pass (1 sample per pixel of the whole image): the passes after the warm-up, from
// the CPU's clock with the GPU waited for (the denoiser off, caustics path traced); the GPU
// column: its timers, the kernel alone.
static void case_trace(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    Gpu *g = gpu_get(c->a);
    if (!g){ skip(r, c->a ? "no second GPU" : "no GPU"); return; }
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    scene_set_accel(&s, ACCEL_BVH, &o.ap);
    GpuView *v = gpu_view_create(g, o.width, o.height);
    if (!v){ teardown(&o, &s, &SIM); skip(r, "no view"); return; }
    gpu_view_trace_denoise(v, 0);
    RenderParams rp = scene_params(&o, &s);
    GpuTraceParams tp = {scene_camera(&s, o.width, o.height), o.max_depth, o.mis, (real)o.clamp, !o.sampler, 0, 0};
    const int warm = 4, passes = full ? 48 : 24;
    for (int k = 0; k < reps; k++){
        if (!gpu_view_trace_start(v, &s, &tp)){ skip(r, "the GPU cannot hold the scene"); break; }
        gpu_view_trace_limit(v, warm + passes);
        if (!trace_until(g, v, &s, &rp, &o.march, warm)){ skip(r, "the GPU path tracer failed"); break; }
        gpu_view_gpu_ms_reset(v);
        int p0 = gpu_view_trace_passes(v);
        double t0 = clock_s();
        if (!trace_until(g, v, &s, &rp, &o.march, warm + passes)){ skip(r, "the GPU path tracer failed"); break; }
        int p1 = gpu_view_trace_passes(v);
        if (p1 > p0) add(r, (clock_s() - t0) * 1000.0 / (p1 - p0));
        add_gpu(r, gpu_view_gpu_ms(v, 0));
    }
    snprintf(r->note, sizeof(r->note), "%dx%d, %d instances", o.width, o.height, s.ninsts);
    gpu_view_destroy(v);
    teardown(&o, &s, &SIM);
}

// ---- B: the CPU path tracer --------------------------------------------------------------
// M paths per second (more is better), 320x320, 2 samples per pixel, c->a threads (0: all)
static void case_cpu_trace(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    const int W = o.width, H = o.height, spp = 2;
    Camera cam = scene_camera(&s, W, H);
    TraceParams tp = trace_default_params();
    tp.max_depth = o.max_depth;
    tp.mis = o.mis;
    tp.sampler = o.sampler;
    Vec3 *img = malloc((size_t)W * H * sizeof(Vec3));
    if (!img){ teardown(&o, &s, &SIM); skip(r, "out of memory"); return; }
    set_threads(c->a);
    for (int k = 0; k < reps; k++){
        double t0 = clock_s();
        trace_image(&s, &cam, &tp, W, H, spp, 0, 0, img, NULL, NULL);   // (render's loop)
        add(r, (double)W * H * spp / (clock_s() - t0) / 1e6);
    }
    set_threads(0);
    free(img);
    snprintf(r->note, sizeof(r->note), "%dx%d, %d spp, %d thread(s)", W, H, spp, c->a ? c->a : max_threads());
    teardown(&o, &s, &SIM);
}

// ---- C: the rasterizers ------------------------------------------------------------------
// ms per frame: frames drawn one after another, the GPU waited for at the end (its work
// overlaps as in the viewer)
static void case_gpu_raster(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    Gpu *g = gpu_get(0);
    if (!g){ skip(r, "no GPU"); return; }
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    GpuView *v = gpu_view_create(g, o.width, o.height);
    RenderParams rp = scene_params(&o, &s);
    const int warm = 5, frames = full ? 120 : 60;
    for (int k = 0; k < reps && v; k++){
        for (int i = 0; i < warm; i++) gpu_view_present_scene(v, &s, &rp, &o.march, 0);
        gpu_wait(g);
        double t0 = clock_s();
        int ok = 1;
        for (int i = 0; i < frames && ok; i++){ ok = gpu_view_present_scene(v, &s, &rp, &o.march, 0); gpu_poll(g); }
        gpu_wait(g);
        if (!ok){ skip(r, "the GPU cannot draw the scene"); break; }
        add(r, (clock_s() - t0) * 1000.0 / frames);
    }
    snprintf(r->note, sizeof(r->note), "%dx%d, %d instances", o.width, o.height, s.ninsts);
    gpu_view_destroy(v);
    teardown(&o, &s, &SIM);
}

static void case_cpu_raster(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    Framebuffer *fb = fb_create(o.width, o.height);
    RenderParams rp = scene_params(&o, &s);
    const int frames = full ? 10 : 5;
    fb_clear(fb, rgb8(0, 0, 0));
    render_scene(fb, &s, &rp);                   // (warm-up)
    for (int k = 0; k < reps && fb; k++){
        double t0 = clock_s();
        for (int i = 0; i < frames; i++){ fb_clear(fb, rgb8(0, 0, 0)); render_scene(fb, &s, &rp); }
        add(r, (clock_s() - t0) * 1000.0 / frames);
    }
    snprintf(r->note, sizeof(r->note), "%dx%d", o.width, o.height);
    fb_destroy(fb);
    teardown(&o, &s, &SIM);
}

// ---- D: the denoiser ---------------------------------------------------------------------
static void case_denoise(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    Gpu *g = gpu_get(0);
    if (!g){ skip(r, "no GPU"); return; }
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    scene_set_accel(&s, ACCEL_BVH, &o.ap);
    GpuView *v = gpu_view_create(g, o.width, o.height);
    RenderParams rp = scene_params(&o, &s);
    GpuTraceParams tp = {scene_camera(&s, o.width, o.height), o.max_depth, o.mis, (real)o.clamp, !o.sampler, 0, 0};
    if (!v || !gpu_view_trace_denoise(v, 1) || !gpu_view_trace_start(v, &s, &tp)) skip(r, "no denoiser here");
    else if (gpu_view_trace_limit(v, 4), !trace_until(g, v, &s, &rp, &o.march, 4)) skip(r, "the GPU path tracer failed");
    else {
        gpu_view_bench_denoise(v, 3);            // (warm-up)
        for (int k = 0; k < reps; k++){
            double ms = gpu_view_bench_denoise(v, full ? 40 : 20);
            if (ms < 0){ skip(r, "no denoiser here"); break; }
            add(r, ms);
        }
        snprintf(r->note, sizeof(r->note), "%dx%d", o.width, o.height);
    }
    gpu_view_destroy(v);
    teardown(&o, &s, &SIM);
}

// ---- E: the trees and the scene's updates ------------------------------------------------
// every instance moved a little (in place: what moves is the cost, not where to)
static void jiggle(Scene *s, int first, int count, int round){
    for (int i = first; i < first + count && i < s->ninsts; i++){
        Xform x = s->insts[i].xf;
        x.m[1][3] += (round & 1) ? 0.01f : -0.01f;
        scene_move_instance(s, i, x);
    }
}

// the CPU's top level over all the instances, built again after all of them moved
static void case_cpu_tlas(const Case *c, Res *r, int reps){
    Options o; static Scene s;
    if (!models_there(c->args, r)) return;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    scene_update_instances(&s, &o.ap);
    const int rounds = 5;
    for (int k = 0; k < reps; k++){
        double ms = 0;
        for (int i = 0; i < rounds; i++){
            jiggle(&s, 0, s.ninsts, k * rounds + i);
            double t0 = clock_s();
            scene_update_instances(&s, &o.ap);
            ms += (clock_s() - t0) * 1000.0;
        }
        add(r, ms / rounds);
    }
    snprintf(r->note, sizeof(r->note), "%d instances", s.ninsts);
    teardown(&o, &s, &SIM);
}

// a frame of the GPU path tracer where c->a instances moved: the CPU's top level made if the
// GPU reads it, the records sent, the GPU's top level built (GPU column: its builds), a pass
static void case_moving_trace(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    Gpu *g = gpu_get(0);
    if (!g){ skip(r, "no GPU"); return; }
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    GpuView *v = gpu_view_create(g, o.width, o.height);
    gpu_view_trace_denoise(v, 0);
    RenderParams rp = scene_params(&o, &s);
    GpuTraceParams tp = {scene_camera(&s, o.width, o.height), o.max_depth, o.mis, (real)o.clamp, !o.sampler, 0, 0};
    const int moved = c->a < s.ninsts ? c->a : s.ninsts, frames = full ? 40 : 20;
    if (!v || !gpu_view_trace_start(v, &s, &tp)) skip(r, "the GPU cannot hold the scene");
    else {
        int round = 0;
        for (int k = 0; k < reps; k++){
            double t0 = 0;
            for (int f = -3; f < frames; f++){   // (3 frames of warm-up)
                if (f == 0){ gpu_wait(g); gpu_view_gpu_ms_reset(v); t0 = clock_s(); }
                jiggle(&s, s.ninsts - moved, moved, round++);
                if (gpu_view_trace_reads_tlas(v, &s)) scene_update_instances(&s, &o.ap);
                gpu_view_trace_move(v, &s, &tp, 0);
                gpu_view_trace_limit(v, gpu_view_trace_passes(v) + 1);
                if (!gpu_view_trace_more(v, &s, &rp, &o.march)){ skip(r, "the GPU path tracer failed"); k = reps; break; }
                gpu_wait(g);
                if (f == frames - 1) add(r, (clock_s() - t0) * 1000.0 / frames);
            }
            add_gpu(r, gpu_view_gpu_ms(v, 1));
        }
        snprintf(r->note, sizeof(r->note), "%dx%d, %d of %d instances moved a frame%s", o.width, o.height, moved, s.ninsts,
                 r->ngpu ? "; GPU column: the tree builds" : "");
    }
    gpu_view_destroy(v);
    teardown(&o, &s, &SIM);
}

// ---- F: the rigid bodies -----------------------------------------------------------------
// A pile: n bodies (balls, boxes, torus hulls in turn) dropped into a room `half` wide,
// stepped at 1/60 s (the world's substeps); ms per step over steps [from, to). c->a: n,
// c->b: half; args: "cpu" with threads, or "gpu".
static void pile_world(PhysWorld *pw, Scene *sc, int n, real half){
    phys_init(pw);
    pw->air_density = 0;
    rigid_set_room(&pw->rigid, (Aabb){vec3(-half, 0, -half), vec3(half, 20000, half)});
    memset(sc, 0, sizeof(*sc));
    int geom = -1;
    int mat = scene_add_material(sc, (Material){0});
    int mesh = scene_load_mesh_object(sc, "models/torus.obj", mat, 40);
    if (mesh >= 0) geom = rigid_geom_hull(&pw->rigid, sc->meshes[mesh].tris, sc->meshes[mesh].ntris, 0);
    Rng rr;
    rng_seed(&rr, 7, 1);
    for (int i = 0; i < n; i++){
        Vec3 p = vec3((half - 30) * (2 * rng_real(&rr) - 1), 30 + i * 6.0 * (300 / half) * (300 / half),
                      (half - 30) * (2 * rng_real(&rr) - 1));
        int k = i % 3;
        if (k == 0) rigid_add_sphere(&pw->rigid, p, 12 + 6 * rng_real(&rr), 1);
        else if (k == 1 || geom < 0) rigid_add_box(&pw->rigid, p, vec3(10 + 6 * rng_real(&rr), 10, 12), quat_axis_angle(vec3(0, 1, 0), rng_real(&rr) * 3), 1);
        else rigid_add_hull_body(&pw->rigid, geom, xform_make(p, quat_to_mat3(quat_axis_angle(vec3(1, 0, 0), rng_real(&rr) * 3)).m, 1), 1);
    }
}

static void case_pile(const Case *c, Res *r, int reps){
    const int gpu = c->args && !strcmp(c->args, "gpu");
    int threads = 0, n = c->a, from, to;
    real half = (real)c->b;
    if (c->args && !strncmp(c->args, "cpu1", 4)) threads = 1;
    // the window measured: the falling (1..4 s), or a settled tall pile (6..9 s)
    if (half >= 200){ from = 60; to = 240; } else { from = 360; to = 540; }
    if (!file_there("models/torus.obj")){ skip(r, "no models/torus.obj"); return; }
    Gpu *g = gpu ? gpu_get(0) : NULL;
    if (gpu && !g){ skip(r, "no GPU"); return; }
    set_threads(threads);
    int last_contacts = 0, asleep = 0;
    for (int k = 0; k < reps; k++){
        PhysWorld pw;
        Scene sc;
        pile_world(&pw, &sc, n, half);
        GpuRigid *gr = gpu ? gpu_rigid_create(g, &pw) : NULL;
        if (gpu && !gr){ skip(r, "no GPU bodies"); phys_free(&pw); scene_free(&sc); break; }
        if (gr) gpu_rigid_set_wait(gr, 1);
        double t0 = 0;
        for (int st = 0; st < to; st++){
            if (st == from){ if (gr){ gpu_wait(g); gpu_rigid_gpu_ms_reset(gr); } t0 = clock_s(); }
            phys_step(&pw, 1.0 / 60);
        }
        if (gr) gpu_rigid_sync(gr);
        add(r, (clock_s() - t0) * 1000.0 / (to - from));
        if (gr) add_gpu(r, gpu_rigid_gpu_ms(gr));
        last_contacts = pw.rigid.ncontacts;
        asleep = rigid_sleeping(&pw.rigid);
        gpu_rigid_destroy(gr);
        phys_free(&pw);
        scene_free(&sc);
    }
    set_threads(0);
    snprintf(r->note, sizeof(r->note), "%d bodies, room %.0f wide, steps %d..%d at 1/60 s: %d contacts, %d asleep at the end; %s",
             n, 2 * (double)half, from, to, last_contacts, asleep,
             gpu ? "GPU" : threads == 1 ? "1 thread" : "all threads");
}

// ---- G: the air and the cloud ------------------------------------------------------------
// ms per step of the blast demo with no bodies (the air and its cloud alone), 1/60 s
static void case_air(const Case *c, Res *r, int reps){
    const int gpu = c->a;
    Gpu *g = gpu ? gpu_get(0) : NULL;
    if (gpu && !g){ skip(r, "no GPU"); return; }
    for (int k = 0; k < reps; k++){
        Options o; static Scene s;
        if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
        GpuFluid *gf = gpu ? gpu_fluid_create(g, &SIM.pw) : NULL;
        if (gpu && !gf){ skip(r, "no GPU air"); teardown(&o, &s, &SIM); return; }
        if (gf) gpu_fluid_set_wait(gf, 1);
        const int warm = 20, steps = full ? 120 : 60;
        double t0 = 0;
        for (int st = 0; st < warm + steps; st++){
            if (st == warm){ if (gf){ gpu_wait(g); gpu_fluid_gpu_ms_reset(gf); } t0 = clock_s(); }
            phys_step(&SIM.pw, 1.0 / 60);
        }
        if (g) gpu_wait(g);
        add(r, (clock_s() - t0) * 1000.0 / steps);
        if (gf) add_gpu(r, gpu_fluid_gpu_ms(gf));
        gpu_fluid_destroy(gf);
        snprintf(r->note, sizeof(r->note), "air %d^3, cloud %d^3, %s", o.sim.fluid_res, o.cloud.resolution, gpu ? "GPU" : "CPU");
        teardown(&o, &s, &SIM);
    }
}

// ---- H: the start --------------------------------------------------------------------------
static void case_gpu_ready(const Case *c, Res *r, int reps){
    (void)c;
    if (!gpu_available()){ skip(r, "no GPU"); return; }
    for (int k = 0; k < reps; k++){
        double t0 = clock_s();
        Gpu *g = gpu_create(NULL, 64, 64);
        gpu_ready_wait(g);
        int ok = g && gpu_state(g) == GPU_READY;
        double ms = (clock_s() - t0) * 1000.0;
        gpu_destroy(g);
        if (!ok){ skip(r, "no GPU"); return; }
        add(r, ms);
    }
}

// a new GPU to the first path traced pass and the first rasterized frame of a small scene:
// the device, the pipelines made, the scene sent
static void case_first_frame(const Case *c, Res *r, int reps){
    if (!gpu_available()){ skip(r, "no GPU"); return; }
    for (int k = 0; k < reps; k++){
        Options o; static Scene s;
        if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
        scene_set_accel(&s, ACCEL_BVH, &o.ap);
        double t0 = clock_s();
        Gpu *g = gpu_create(NULL, o.width, o.height);
        gpu_ready_wait(g);
        GpuView *v = g && gpu_state(g) == GPU_READY ? gpu_view_create(g, o.width, o.height) : NULL;
        RenderParams rp = scene_params(&o, &s);
        GpuTraceParams tp = {scene_camera(&s, o.width, o.height), o.max_depth, o.mis, (real)o.clamp, !o.sampler, 0, 0};
        int ok = v && gpu_view_present_scene(v, &s, &rp, &o.march, 0) && gpu_view_trace_start(v, &s, &tp);
        if (ok){ gpu_view_trace_limit(v, 1); ok = trace_until(g, v, &s, &rp, &o.march, 1); gpu_wait(g); }
        double ms = (clock_s() - t0) * 1000.0;
        gpu_view_destroy(v);
        gpu_destroy(g);
        teardown(&o, &s, &SIM);
        if (!ok){ skip(r, "no GPU"); return; }
        add(r, ms);
        // the first one compiles the pipelines; the others find them in the driver's (or the
        // browser's) cache
        if (k == 0) snprintf(r->note, sizeof(r->note), "the first %.0f ms (its pipelines compiled)", ms);
    }
}

// A viewer's start: a new GPU, the path tracer's kernels asked for (gpu_view_trace_prepare),
// then frames rasterized one after another, as the viewer does, until the first path traced
// pass is done: the longest frame (ms). In a browser the kernels compile for seconds the
// first time; made at once (LR_GPU_ASYNC=0) they hold up every frame behind them. Once: the
// browser keeps what it compiled (the later runs of a page find it).
static void case_compile_frames(const Case *c, Res *r, int reps){
    (void)reps;
    if (!gpu_available()){ skip(r, "no GPU"); return; }
    Options o; static Scene s;
    if (setup(c->args, &o, &s, &SIM)){ skip(r, "scene not made"); return; }
    scene_set_accel(&s, ACCEL_BVH, &o.ap);
    Gpu *g = gpu_create(NULL, o.width, o.height);
    gpu_ready_wait(g);
    GpuView *v = g && gpu_state(g) == GPU_READY ? gpu_view_create(g, o.width, o.height) : NULL;
    RenderParams rp = scene_params(&o, &s);
    GpuTraceParams tp = {scene_camera(&s, o.width, o.height), o.max_depth, o.mis, (real)o.clamp, !o.sampler, 0, 0};
    double t0 = clock_s(), worst = 0;
    int frames = 0;
    if (v){
        gpu_view_trace_prepare(v);
        gpu_view_trace_start(v, &s, &tp);
        gpu_view_trace_limit(v, 1);
        while (gpu_view_trace_passes(v) < 1 && gpu_state(g) == GPU_READY && clock_s() - t0 < 60){
            double f0 = clock_s();
            gpu_view_present_scene(v, &s, &rp, &o.march, 0);
            if (!gpu_view_trace_more(v, &s, &rp, &o.march)) break;   // (its pipelines failed)
            gpu_wait(g);
            double ms = (clock_s() - f0) * 1000.0;
            if (ms > worst) worst = ms;
            frames++;
        }
    }
    int ok = v && gpu_view_trace_passes(v) >= 1;
    double total = (clock_s() - t0) * 1000.0;
    gpu_view_destroy(v);
    gpu_destroy(g);
    teardown(&o, &s, &SIM);
    if (!ok){ skip(r, "no GPU"); return; }
    add(r, worst);
    snprintf(r->note, sizeof(r->note), "%d frames rasterized in the %.0f ms until the first path traced pass", frames, total);
}

// a scene made: loading its models, its trees, the sky demo's props and light probe
static void case_setup(const Case *c, Res *r, int reps){
    if (!models_there(c->args, r)) return;
    for (int k = 0; k < reps; k++){
        Options o; static Scene s;
        double t0 = clock_s();
        int err = setup(c->args, &o, &s, &SIM);
        double ms = (clock_s() - t0) * 1000.0;
        if (err){ skip(r, "scene not made"); return; }
        if (!k) snprintf(r->note, sizeof(r->note), "%d triangles, %d instances", s.ntris, s.ninsts);
        teardown(&o, &s, &SIM);
        add(r, ms);
    }
}

// ---- I: the viewer, uncapped ---------------------------------------------------------------
// bin/view itself, in its window with -vsync off (frames shown as soon as they are ready: the
// rate the machine can do), for 6 s; its frames per second and path tracing passes per
// second from its own counters (-stats: the first 2 s left out). ms per frame.
static char view_exe[512];          // bin/view, beside this program
static void case_viewer(const Case *c, Res *r, int reps){
#ifdef __EMSCRIPTEN__
    (void)c; (void)reps;
    skip(r, "natively only (a page is paced by the browser)");
#else
    if (!view_exe[0]){ skip(r, "no bin/view beside bin/bench"); return; }
    const char *stats = "bench_view_stats.txt";
    double pps = 0;
    for (int k = 0; k < reps; k++){
        char cmd[1024];
        remove(stats);
        snprintf(cmd, sizeof(cmd), "\"%s\" %s -vsync off -autoquit 6 -stats %s -o out/bench_view > %s 2>&1",
                 view_exe, c->args, stats,
#ifdef _WIN32
                 "NUL"
#else
                 "/dev/null"
#endif
                 );
        if (app_run_command(cmd)){}             // (its answer: the stats file says how it went)
        FILE *f = fopen(stats, "r");
        if (!f){ skip(r, "bin/view gave no -stats"); return; }
        // line by line, "key value": a line whose value is not a number ("mode rt") is passed
        // over, not the end of the file
        char line[160], key[64];
        double val, ms = 0;
        while (fgets(line, sizeof(line), f)){
            if (sscanf(line, "%63s %lf", key, &val) != 2) continue;
            if (!strcmp(key, "ms_per_frame")) ms = val;
            else if (!strcmp(key, "passes_per_s")) pps = val;
        }
        fclose(f);
        remove(stats);
        if (ms > 0) add(r, ms);
    }
    double med = median(r->v, r->n);
    snprintf(r->note, sizeof(r->note), "%.0f frames/s, %.0f passes/s; window, -vsync off", med > 0 ? 1000 / med : 0, pps);
#endif
}

// ---- the list ----------------------------------------------------------------------------

#define R640 "-width 640 -height 640"
#define R720 "-width 1280 -height 720"
#define SKYDEMO(props) "-sim on -demo sky -bodies 0 -props " #props " -sky " SKY " " R720

static const Case CASES[] = {
    {"A1", "GPU path tracer: boxes",                    "ms/pass", 1, case_trace, "-scene boxes " R640, 0, 0},
    {"A2", "GPU path tracer: spheres (mirror, glass)",  "ms/pass", 0, case_trace, "-scene spheres " R640, 0, 0},
    {"A3", "GPU path tracer: dragon (871k triangles)",  "ms/pass", 1, case_trace, "-scene mesh -obj " DRAGON " -rot -35 " R640, 0, 0},
    {"A4", "GPU path tracer: cloud",                    "ms/pass", 1, case_trace, "-scene cloud " R640, 0, 0},
    {"A5", "GPU path tracer: spheres in cloud and fog", "ms/pass", 0, case_trace, "-scene spheres -cloud on -fog 0.0015 " R640, 0, 0},
    {"A6", "GPU path tracer: 40 instanced dragons",     "ms/pass", 0, case_trace, "-scene spheres -obj " DRAGON2 " -instances 40 " R640, 0, 0},
    {"A7", "GPU path tracer: helmet (glTF, textures)",  "ms/pass", 0, case_trace, "-gltf " HELMET " " R640, 0, 0},
    {"A8", "GPU path tracer: sky demo, 3k props",       "ms/pass", 1, case_trace, SKYDEMO(3000), 0, 0},
    {"A9", "GPU path tracer: sky demo, 48k props",      "ms/pass", 0, case_trace, SKYDEMO(48000), 0, 0},
#ifdef LR_VK
    {"A1c", "GPU path tracer, no hardware rays: boxes",  "ms/pass", 0, case_trace, "-scene boxes " R640, 1, 0},
    {"A3c", "GPU path tracer, no hardware rays: dragon", "ms/pass", 0, case_trace, "-scene mesh -obj " DRAGON " -rot -35 " R640, 1, 0},
    {"A8c", "GPU path tracer, no hardware rays: sky 3k", "ms/pass", 0, case_trace, SKYDEMO(3000), 1, 0},
#endif
    {"B1", "CPU path tracer: boxes, all threads",       "M paths/s", 1, case_cpu_trace, "-scene boxes -width 320 -height 320", 0, 0},
    {"B2", "CPU path tracer: boxes, 1 thread",          "M paths/s", 0, case_cpu_trace, "-scene boxes -width 320 -height 320", 1, 0},
    {"B3", "CPU path tracer: dragon, all threads",      "M paths/s", 1, case_cpu_trace, "-scene mesh -obj " DRAGON " -rot -35 -width 320 -height 320", 0, 0},
    {"B4", "CPU path tracer: dragon, 1 thread",         "M paths/s", 0, case_cpu_trace, "-scene mesh -obj " DRAGON " -rot -35 -width 320 -height 320", 1, 0},
    {"C1", "GPU rasterizer: sky demo, 3k props",        "ms/frame", 1, case_gpu_raster, SKYDEMO(3000), 0, 0},
    {"C2", "GPU rasterizer: sky demo, 18k props",       "ms/frame", 0, case_gpu_raster, SKYDEMO(18000), 0, 0},
    {"C3", "GPU rasterizer: sky demo, 48k props",       "ms/frame", 1, case_gpu_raster, SKYDEMO(48000), 0, 0},
    {"C4", "GPU rasterizer: dragon (871k triangles)",   "ms/frame", 0, case_gpu_raster, "-scene mesh -obj " DRAGON " -rot -35 -backend raster " R640, 0, 0},
    {"C5", "CPU rasterizer: boxes",                     "ms/frame", 1, case_cpu_raster, "-scene boxes -backend raster " R640, 0, 0},
    {"C6", "CPU rasterizer: dragon",                    "ms/frame", 0, case_cpu_raster, "-scene mesh -obj " DRAGON " -rot -35 -backend raster " R640, 0, 0},
    {"D1", "denoiser: 1280x720",                        "ms/run", 1, case_denoise, "-scene boxes " R720, 0, 0},
    {"D2", "denoiser: 640x640",                         "ms/run", 0, case_denoise, "-scene boxes " R640, 0, 0},
    {"E1", "CPU top level (SAH), all moved: 48k",       "ms/build", 1, case_cpu_tlas, SKYDEMO(48000), 0, 0},
    {"E2", "CPU top level (SAH), all moved: 18k",       "ms/build", 0, case_cpu_tlas, SKYDEMO(18000), 0, 0},
    {"E3", "path traced frame, 2000 moving: 48k",       "ms/frame", 1, case_moving_trace, SKYDEMO(48000), 2000, 0},
    {"E4", "path traced frame, 2000 moving: 18k",       "ms/frame", 0, case_moving_trace, SKYDEMO(18000), 2000, 0},
    {"E5", "path traced frame, 2000 moving: 3k",        "ms/frame", 0, case_moving_trace, SKYDEMO(3000), 2000, 0},
    {"F1", "bodies on the CPU: 1000 falling, all threads", "ms/step", 1, case_pile, "cpu", 1000, 300},
    {"F2", "bodies on the CPU: 1000 falling, 1 thread",    "ms/step", 0, case_pile, "cpu1", 1000, 300},
    {"F3", "bodies on the GPU: 1000 falling",              "ms/step", 1, case_pile, "gpu", 1000, 300},
    {"F4", "bodies on the CPU: tall pile of 400",          "ms/step", 0, case_pile, "cpu", 400, 90},
    {"F5", "bodies on the GPU: tall pile of 400",          "ms/step", 0, case_pile, "gpu", 400, 90},
    {"F6", "bodies on the GPU: 2000 falling",              "ms/step", 0, case_pile, "gpu", 2000, 400},
    {"G1", "air and cloud on the GPU: 32^3, cloud 96^3",   "ms/step", 1, case_air, "-sim on -demo blast -balls 0 -boxes 0 -balloons 0 -fluidres 32 -cloudres 96", 1, 0},
    {"G2", "air and cloud on the GPU: 32^3, cloud 192^3",  "ms/step", 0, case_air, "-sim on -demo blast -balls 0 -boxes 0 -balloons 0 -fluidres 32 -cloudres 192", 1, 0},
    {"G3", "air and cloud on the CPU: 32^3, cloud 96^3",   "ms/step", 1, case_air, "-sim on -demo blast -balls 0 -boxes 0 -balloons 0 -fluidres 32 -cloudres 96", 0, 0},
    {"H1", "start: a GPU ready",                           "ms", 1, case_gpu_ready, NULL, 0, 0},
    {"H2", "start: a new GPU to the first frames",         "ms", 1, case_first_frame, "-scene boxes -width 256 -height 256", 0, 0},
    {"H3", "start: the sky demo made (props, light probe)", "ms", 1, case_setup, SKYDEMO(3000), 0, 0},
    {"H4", "start: the dragon loaded, its tree built",     "ms", 0, case_setup, "-scene mesh -obj " DRAGON " -rot -35 " R640, 0, 0},
    {"H5", "start: the helmet loaded (glTF)",              "ms", 0, case_setup, "-gltf " HELMET " " R640, 0, 0},
    {"H6", "start: the longest frame until path traced",   "ms", 0, case_compile_frames, "-scene boxes -width 640 -height 640", 0, 0},
    {"I1", "viewer: rasterized, still",                    "ms/frame", 1, case_viewer, "-backend raster " R720, 0, 0},
    {"I2", "viewer: rasterized, moving",                   "ms/frame", 0, case_viewer, "-backend raster -orbit 15 " R720, 0, 0},
    {"I3", "viewer: path traced, still",                   "ms/frame", 1, case_viewer, "-backend rt " R720, 0, 0},
    {"I4", "viewer: path traced, moving",                  "ms/frame", 0, case_viewer, "-backend rt -orbit 15 " R720, 0, 0},
    {"I5", "viewer: hybrid, still",                        "ms/frame", 0, case_viewer, "-backend hybrid " R720, 0, 0},
    {"I6", "viewer: sky demo, bodies falling, path traced", "ms/frame", 0, case_viewer, "-sim on -demo sky -sky " SKY " -backend rt " R720, 0, 0},
};
#define NCASES ((int)(sizeof(CASES) / sizeof(CASES[0])))

// ---- running them ------------------------------------------------------------------------

static int selected(const Case *c, const char *only){
    if (!only) return full || c->quick;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", only);
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ","))
        if (!strcmp(t, c->id) || (strlen(t) == 1 && t[0] == c->id[0] && (full || c->quick))) return 1;
    return 0;
}

static void csv_escape(FILE *f, const char *s){
    fputc('"', f);
    for (; s && *s; s++){ if (*s == '"') fputc('"', f); fputc(*s, f); }
    fputc('"', f);
}

// ---- -compare ----------------------------------------------------------------------------

typedef struct { char id[16], backend[16], unit[24], what[96], tag[32]; double med, gpu; } Row;

static int read_csv(const char *path, Row *rows, int max){
    FILE *f = fopen(path, "r");
    if (!f){ fprintf(stderr, "cannot read %s\n", path); return -1; }
    char line[2048];
    int n = 0;
    if (!fgets(line, sizeof(line), f)){ fclose(f); return 0; }        // (the header)
    while (n < max && fgets(line, sizeof(line), f)){
        // tag,backend,gpu,set,case,what,unit,median,min,max,reps,gpu_ms,note
        char *fld[16];
        int nf = 0;
        char *p = line;
        while (*p && nf < 16){
            if (*p == '"'){
                char *o = ++p, *w = p;
                while (*p && !(*p == '"' && p[1] != '"')){ if (*p == '"') p++; *w++ = *p++; }
                if (*p) p++;             // (past the closing quote, before the field ends where it was)
                *w = 0;
                fld[nf++] = o;
                if (*p == ',') p++;
            } else {
                fld[nf++] = p;
                while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
                if (*p){ *p++ = 0; }
            }
        }
        if (nf < 8) continue;
        Row *r = &rows[n++];
        snprintf(r->tag, sizeof(r->tag), "%s", fld[0]);
        snprintf(r->backend, sizeof(r->backend), "%s", fld[1]);
        r->gpu = nf > 11 ? atof(fld[11]) : 0;
        snprintf(r->id, sizeof(r->id), "%s", fld[4]);
        snprintf(r->what, sizeof(r->what), "%s", fld[5]);
        snprintf(r->unit, sizeof(r->unit), "%s", fld[6]);
        r->med = atof(fld[7]);
    }
    fclose(f);
    return n;
}

static int compare(const char *a, const char *b){
    static Row ra[512], rb[512];
    int na = read_csv(a, ra, 512), nb = read_csv(b, rb, 512);
    if (na < 0 || nb < 0) return 1;
    printf("%-5s %-5s %-46s %12s %12s %9s\n", "case", "on", "what", "before", "after", "change");
    for (int j = 0; j < nb; j++){
        const Row *n = &rb[j], *o = NULL;
        for (int i = na - 1; i >= 0 && !o; i--)          // (the last of that case in the old file)
            if (!strcmp(ra[i].id, n->id) && !strcmp(ra[i].backend, n->backend)) o = &ra[i];
        if (!o || o->med <= 0 || n->med <= 0){ printf("%-5s %-5s %-46.46s %12s %12.3f %9s\n", n->id, n->backend, n->what, "-", n->med, ""); continue; }
        size_t ul = strlen(n->unit);
        int more_better = ul >= 2 && !strcmp(n->unit + ul - 2, "/s");   // (M paths/s; not ms/step)
        double gain = more_better ? n->med / o->med : o->med / n->med;   // > 1: faster
        printf("%-5s %-5s %-46.46s %12.3f %12.3f %+8.1f%%%s\n", n->id, n->backend, n->what, o->med, n->med,
               (gain - 1) * 100, gain < 0.95 ? "  slower" : gain > 1.05 ? "  faster" : "");
    }
    return 0;
}

// a number for a table: 3 significant digits, no exponent
static void num(char *buf, size_t n, double v){
    if (v >= 100) snprintf(buf, n, "%.0f", v);
    else if (v >= 10) snprintf(buf, n, "%.1f", v);
    else snprintf(buf, n, "%.3g", v);
}

static void table_row(Row rows[][512], const int *n, int nfiles, const char *id, const char *what, const char *unit){
    printf("| %s | %s | %s |", id, what, unit);
    for (int f = 0; f < nfiles; f++){
        const Row *r = NULL;
        for (int j = 0; j < n[f]; j++) if (!strcmp(rows[f][j].id, id)) r = &rows[f][j];
        char a[32], b[32];
        if (!r){ printf(" - |"); continue; }
        num(a, sizeof(a), r->med);
        if (r->gpu > 0){ num(b, sizeof(b), r->gpu); printf(" %s (GPU %s) |", a, b); }
        else printf(" %s |", a);
    }
    printf("\n");
}

// -table: the cases of the files side by side, the last result of each case in each file
static int table(char **files, int nfiles){
    static Row rows[8][512];
    int n[8];
    if (nfiles > 8) nfiles = 8;
    for (int f = 0; f < nfiles; f++) if ((n[f] = read_csv(files[f], rows[f], 512)) < 0) return 1;
    printf("| case | what | unit |");
    for (int f = 0; f < nfiles; f++) printf(" %s |", n[f] ? rows[f][0].backend : files[f]);
    printf("\n|---|---|---|");
    for (int f = 0; f < nfiles; f++) printf("---|");
    printf("\n");
    // the cases in the order of CASES (then those only in the files)
    for (int i = 0; i < NCASES; i++){
        int any = 0;
        for (int f = 0; f < nfiles; f++) for (int j = 0; j < n[f]; j++) if (!strcmp(rows[f][j].id, CASES[i].id)) any = 1;
        if (!any) continue;
        table_row(rows, n, nfiles, CASES[i].id, CASES[i].what, CASES[i].unit);
    }
    // and the cases this build does not have (another build's own, e.g. A1c natively), once each
    for (int f = 0; f < nfiles; f++)
        for (int j = 0; j < n[f]; j++){
            int known = 0;
            for (int i = 0; i < NCASES && !known; i++) known = !strcmp(CASES[i].id, rows[f][j].id);
            for (int g = 0; g < f && !known; g++) for (int k = 0; k < n[g] && !known; k++) known = !strcmp(rows[g][k].id, rows[f][j].id);
            for (int k = 0; k < j && !known; k++) known = !strcmp(rows[f][k].id, rows[f][j].id);
            if (!known) table_row(rows, n, nfiles, rows[f][j].id, rows[f][j].what, rows[f][j].unit);
        }
    return 0;
}

// ---- -shots: the same images natively and in a page, to compare (-imgdiff) ----------------
// Each image is made on the GPU with fixed inputs and saved as <dir>/<name>_<backend>.png;
// in a page (which has no files) its PNG goes to the console as a line "PNG <file> <base64>",
// which tools/webbench.mjs -shots <dir> writes out. With them, "SHOT <name> <mean r g b>".
static void shot_put(const char *dir, const char *name, int w, int h, const uint32_t *px){
    unsigned char *rgb = malloc((size_t)w * h * 3);
    if (!rgb) return;
    double m[3] = {0, 0, 0};
    for (int i = 0; i < w * h; i++){
        rgb[3 * i] = (unsigned char)(px[i] >> 16); rgb[3 * i + 1] = (unsigned char)(px[i] >> 8); rgb[3 * i + 2] = (unsigned char)px[i];
        for (int c = 0; c < 3; c++) m[c] += rgb[3 * i + c];
    }
    printf("SHOT %s %.2f %.2f %.2f\n", name, m[0] / (w * h), m[1] / (w * h), m[2] / (w * h));
    char file[256];
    snprintf(file, sizeof(file), "%s_%s.png", name, BACKEND);
#ifdef __EMSCRIPTEN__
    (void)dir;
    size_t n = 0;
    unsigned char *png = png_encode(w, h, 3, rgb, &n);
    if (png){
        static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        char *s = malloc((n + 2) / 3 * 4 + 1);
        size_t k = 0;
        for (size_t i = 0; s && i < n; i += 3){
            uint32_t v = (uint32_t)png[i] << 16 | (i + 1 < n ? (uint32_t)png[i + 1] << 8 : 0) | (i + 2 < n ? png[i + 2] : 0);
            s[k++] = B64[v >> 18 & 63]; s[k++] = B64[v >> 12 & 63];
            s[k++] = i + 1 < n ? B64[v >> 6 & 63] : '='; s[k++] = i + 2 < n ? B64[v & 63] : '=';
        }
        if (s){ s[k] = 0; printf("PNG %s %s\n", file, s); free(s); }
        free(png);
    }
#else
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    if (png_write(path, w, h, 3, rgb)) fprintf(stderr, "cannot write %s\n", path);
#endif
    free(rgb);
}

static void shot_trace(const char *dir, const char *name, const char *args, int spp, int denoise){
    Gpu *g = gpu_get(0);
    Options o; static Scene s;
    if (!g || !models_there(args, &(Res){0}) || setup(args, &o, &s, &SIM)){ printf("SHOT %s skipped\n", name); return; }
    scene_set_accel(&s, ACCEL_BVH, &o.ap);
    GpuView *v = gpu_view_create(g, o.width, o.height);
    Image img = image_create(o.width, o.height);
    RenderParams rp = scene_params(&o, &s);
    Camera cam = scene_camera(&s, o.width, o.height);
    if (v) gpu_view_trace_denoise(v, denoise);
    if (v && gpu_frame_trace(g, v, &s, &cam, &rp, &o.march, spp, o.max_depth, o.mis, (real)o.clamp, o.sampler, 0, &img, 0) == 0){
        uint32_t *px = malloc((size_t)o.width * o.height * 4);
        for (int i = 0; px && i < o.width * o.height; i++) px[i] = image_pack_srgb(img.px[i]);
        if (px) shot_put(dir, name, o.width, o.height, px);
        free(px);
    } else printf("SHOT %s skipped\n", name);
    image_free(&img);
    gpu_view_destroy(v);
    teardown(&o, &s, &SIM);
}

static void shot_raster(const char *dir, const char *name, const char *args){
    Gpu *g = gpu_get(0);
    Options o; static Scene s;
    if (!g || !models_there(args, &(Res){0}) || setup(args, &o, &s, &SIM)){ printf("SHOT %s skipped\n", name); return; }
    GpuView *v = gpu_view_create(g, o.width, o.height);
    Framebuffer *fb = fb_create(o.width, o.height);
    RenderParams rp = scene_params(&o, &s);
    if (v && fb && gpu_frame_raster(g, v, &s, &rp, &o.march, 1, fb) == 0) shot_put(dir, name, o.width, o.height, fb->color);
    else printf("SHOT %s skipped\n", name);
    fb_destroy(fb);
    gpu_view_destroy(v);
    teardown(&o, &s, &SIM);
}

// the CPU's path tracer (all threads; each pixel's samples seeded by the pixel, so the image
// does not depend on how many threads ran) and the CPU's rasterizer
static void shot_cpu(const char *dir, const char *name, const char *args, int spp){
    Options o; static Scene s;
    if (!models_there(args, &(Res){0}) || setup(args, &o, &s, &SIM)){ printf("SHOT %s skipped\n", name); return; }
    const int W = o.width, H = o.height;
    uint32_t *px = malloc((size_t)W * H * 4);
    if (!px){ teardown(&o, &s, &SIM); return; }
    if (o.backend == BACKEND_RASTER){
        Framebuffer *fb = fb_create(W, H);
        RenderParams rp = scene_params(&o, &s);
        if (fb){ fb_clear(fb, rgb8(0, 0, 0)); render_scene(fb, &s, &rp); memcpy(px, fb->color, (size_t)W * H * 4); }
        fb_destroy(fb);
    } else {
        Camera cam = scene_camera(&s, W, H);
        TraceParams tp = trace_default_params();
        tp.max_depth = o.max_depth;
        tp.mis = o.mis;
        tp.sampler = o.sampler;
        Vec3 *img = malloc((size_t)W * H * sizeof(Vec3));
        if (img){
            trace_image(&s, &cam, &tp, W, H, spp, 0, 0, img, NULL, NULL);
            for (int i = 0; i < W * H; i++) px[i] = image_pack_srgb(img[i]);
            free(img);
        }
    }
    shot_put(dir, name, W, H, px);
    free(px);
    teardown(&o, &s, &SIM);
}

static int shots(const char *dir){
    printf("bench: shots, %s (%s)\n", BACKEND, gpu_get(0) ? gpu_message(gpu_get(0)) : "no GPU");
    double t0 = clock_s();
    shot_cpu(dir, "cpu_boxes_pt", "-scene boxes -width 160 -height 160", 16);
    shot_cpu(dir, "cpu_boxes_raster", "-scene boxes -backend raster -width 256 -height 256", 0);
    shot_trace(dir, "boxes_pt", "-scene boxes -width 256 -height 256", 64, 0);
    shot_trace(dir, "spheres_media_pt", "-scene spheres -cloud on -fog 0.0015 -width 256 -height 256", 64, 0);
    shot_trace(dir, "helmet_pt", "-gltf " HELMET " -width 256 -height 256", 32, 0);
    shot_trace(dir, "sky_pt", "-sim on -demo sky -bodies 0 -props 3000 -sky " SKY " -width 320 -height 180", 32, 0);
    shot_trace(dir, "boxes_denoised", "-scene boxes -width 256 -height 256", 8, 1);
    shot_trace(dir, "sky_denoised", "-sim on -demo sky -bodies 0 -props 3000 -sky " SKY " -width 320 -height 180", 8, 1);
    shot_raster(dir, "boxes_raster", "-scene boxes -backend raster -width 256 -height 256");
    shot_raster(dir, "sky_raster", "-sim on -demo sky -bodies 0 -props 3000 -sky " SKY " -width 320 -height 180");
    printf("bench: done in %.0f s\n", clock_s() - t0);
    return 0;
}

// -imgdiff a.png b.png: how far two images are apart (8-bit levels): RMSE, mean difference
// per channel, the share of pixels more than 8 levels apart
static int imgdiff(const char *a, const char *b){
    int wa, ha, ca, wb, hb, cb;
    unsigned char *pa = png_read(a, &wa, &ha, &ca, 3), *pb = png_read(b, &wb, &hb, &cb, 3);
    if (!pa || !pb || wa != wb || ha != hb){ fprintf(stderr, "cannot compare %s and %s\n", a, b); free(pa); free(pb); return 1; }
    double se = 0, d[3] = {0, 0, 0};
    long far = 0;
    for (int i = 0; i < wa * ha; i++){
        int big = 0;
        for (int c = 0; c < 3; c++){
            int e = (int)pa[3 * i + c] - (int)pb[3 * i + c];
            se += (double)e * e;
            d[c] += e;
            if (e > 8 || e < -8) big = 1;
        }
        far += big;
    }
    int n = wa * ha;
    printf("%s vs %s: RMSE %.2f levels, mean difference %+.2f %+.2f %+.2f, %.2f%% of pixels > 8 levels apart\n",
           a, b, sqrt(se / (3.0 * n)), d[0] / n, d[1] / n, d[2] / n, 100.0 * far / n);
    free(pa);
    free(pb);
    return 0;
}

int main(int argc, char **argv){
#ifdef __EMSCRIPTEN__
    // the page's address: bench.html?full&only=A,F3&tag=x -> -full -only A,F3 -tag x -csv
    static char qs[512];
    static char *wargv[32];
    if (argc <= 1){
        const char *s = emscripten_run_script_string("location.search.substring(1)");
        snprintf(qs, sizeof(qs), "%s", s ? s : "");
        int n = 0, k = 0;
        static char opt[16][64];
        wargv[n++] = "bench";
        for (char *t = strtok(qs, "&"); t && n < 29 && k < 16; t = strtok(NULL, "&")){
            char *eq = strchr(t, '=');
            if (eq) *eq = 0;
            snprintf(opt[k], sizeof(opt[k]), "-%s", t);
            wargv[n++] = opt[k++];
            if (eq) wargv[n++] = eq + 1;
        }
        wargv[n++] = "-csv";
        wargv[n] = NULL;
        argc = n;
        argv = wargv;
    }
#endif
    const char *only = NULL, *out = NULL, *tag = "", *cmp_a = NULL, *cmp_b = NULL, *shots_dir = NULL;
    int list = 0, reps = 0;
    for (int i = 1; i < argc; i++){
        if (!strcmp(argv[i], "-full")) full = 1;
        else if (!strcmp(argv[i], "-quick")) full = 0;
        else if (!strcmp(argv[i], "-list")) list = 1;
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else if (!strcmp(argv[i], "-csv")) csv_out = 1;
        else if (!strcmp(argv[i], "-only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "-tag") && i + 1 < argc) tag = argv[++i];
        else if (!strcmp(argv[i], "-reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-env") && i + 1 < argc){     // NAME=value: an environment variable (a page has no other way)
            static char kv[8][128];
            static int nkv;
            char *eq = strchr(argv[++i], '=');
            if (eq && nkv < 8){
                snprintf(kv[nkv], sizeof(kv[nkv]), "%.*s", (int)(eq - argv[i]), argv[i]);
                set_env(kv[nkv++], eq + 1);
            }
        }
        else if (!strcmp(argv[i], "-compare") && i + 2 < argc){ cmp_a = argv[i + 1]; cmp_b = argv[i + 2]; i += 2; }
        else if (!strcmp(argv[i], "-table") && i + 1 < argc) return table(argv + i + 1, argc - i - 1);
        else if (!strcmp(argv[i], "-shots") && i + 1 < argc) shots_dir = argv[++i];
        else if (!strcmp(argv[i], "-imgdiff") && i + 2 < argc) return imgdiff(argv[i + 1], argv[i + 2]);
        else {
            fprintf(stderr, "usage: %s [-quick|-full] [-only A,F3] [-reps n] [-o file.csv] [-tag label] [-list]\n"
                            "       %s -compare old.csv new.csv\n", argv[0], argv[0]);
            return 1;
        }
    }
    if (cmp_a) return compare(cmp_a, cmp_b);
    if (shots_dir) return shots(shots_dir);
    if (list){
        for (int i = 0; i < NCASES; i++)
            printf("%-4s %-5s %-50s %-10s %s\n", CASES[i].id, CASES[i].quick ? "quick" : "full", CASES[i].what, CASES[i].unit,
                   CASES[i].args ? CASES[i].args : "");
        return 0;
    }
    if (reps <= 0) reps = full ? 5 : 3;
    {   // bin/view beside this program (group I)
        snprintf(view_exe, sizeof(view_exe), "%s", argv[0]);
        char *slash = strrchr(view_exe, '/'), *bs = strrchr(view_exe, '\\');
        if (bs && (!slash || bs > slash)) slash = bs;
        size_t at = slash ? (size_t)(slash - view_exe + 1) : 0;
#ifdef _WIN32
        snprintf(view_exe + at, sizeof(view_exe) - at, "view.exe");
#else
        snprintf(view_exe + at, sizeof(view_exe) - at, "view");
#endif
        if (!file_there(view_exe)) view_exe[0] = 0;
    }
    if (reps > MAXREP) reps = MAXREP;

    FILE *csv = NULL;
    if (out){
        FILE *t = fopen(out, "r");
        int fresh = !t;
        if (t) fclose(t);
        csv = fopen(out, "a");
        if (!csv){ fprintf(stderr, "cannot write %s (does the folder exist?)\n", out); return 1; }
        if (fresh) fprintf(csv, "tag,backend,gpu,set,case,what,unit,median,min,max,reps,gpu_ms,note\n");
    }
    // (the scenes' own messages go to stdout as they are made: the results are on stderr
    // too with -q, and in the CSV)
    Gpu *g = gpu_get(0);
    const char *gname = g ? gpu_message(g) : "no GPU";
    printf("bench: %s set, %s (%s), %d repetitions, %d threads\n", full ? "full" : "quick", BACKEND, gname, reps, max_threads());
    double t_all = clock_s();
    for (int i = 0; i < NCASES; i++){
        const Case *c = &CASES[i];
        if (!selected(c, only)) continue;
        Res r;
        memset(&r, 0, sizeof(r));
        int nreps = c->fn == case_pile && !full ? 2 : c->fn == case_pile ? 3 : c->fn == case_viewer ? 2 : c->fn == case_compile_frames ? 1 : reps;
        double t0 = clock_s();
        c->fn(c, &r, nreps);
        double secs = clock_s() - t0;
        double med = median(r.v, r.n), gmed = median(r.gpu, r.ngpu);
        char line[512], gcol[32] = "";
        if (r.ngpu) snprintf(gcol, sizeof(gcol), "  GPU %.3f ms", gmed);
        if (r.skipped && r.n == 0) snprintf(line, sizeof(line), "%-4s %-50s skipped: %s", c->id, c->what, r.note);
        else snprintf(line, sizeof(line), "%-4s %-50s %10.3f %-9s (%.3f..%.3f)%s  [%s%s%.0f s]",
                      c->id, c->what, med, c->unit, vmin(r.v, r.n), vmax(r.v, r.n), gcol, r.note, r.note[0] ? "; " : "", secs);
        printf("RESULT %s\n", line);
        if (quiet) fprintf(stderr, "%s\n", line);
        fflush(stdout);
        for (int w = 0; w < 2 && r.n > 0; w++){
            FILE *f = w == 0 ? csv : (csv_out ? stdout : NULL);
            if (!f) continue;
            if (w == 1) fprintf(f, "CSV ");
            fprintf(f, "%s,%s,", tag, BACKEND);
            csv_escape(f, gname);
            fprintf(f, ",%s,%s,", full ? "full" : "quick", c->id);
            csv_escape(f, c->what);
            fprintf(f, ",%s,%.4f,%.4f,%.4f,%d,%.4f,", c->unit, med, vmin(r.v, r.n), vmax(r.v, r.n), r.n, r.ngpu ? gmed : 0.0);
            csv_escape(f, r.note);
            fprintf(f, "\n");
            fflush(f);
        }
    }
    printf("bench: done in %.0f s\n", clock_s() - t_all);
    if (csv) fclose(csv);
    for (int k = 0; k < 2; k++) if (gpus[k]) gpu_destroy(gpus[k]);
    return 0;
}
