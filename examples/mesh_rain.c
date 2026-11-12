// mesh_rain.c - a scene that never stops filling up: meshes (a dragon, a torus, a cube, a
// sphere) appear under the ceiling of the room at random, turned at random, and fall as
// convex hulls onto each other. The frame rate is in the window's title.
//
// Everything here goes through the engine's public API (nothing in engine/ is special for
// it): the room from scene_cornell_box, each model loaded once in its own space
// (scene_load_mesh_object) with its convex hull (rigid_geom_hull), every falling copy a
// hull body drawn as an instance (phys_add_mesh_to_scene). With a GPU (make deps) the
// bodies are simulated there (gpu/rigid_gpu.h) and the scene is rasterized there
// (gpu_view_present_scene); without one the CPU does both.
//
// When -max bodies are in the room, the oldest one is taken back up and dropped again, so
// the rain goes on forever at a steady cost.
//
//   make bin/mesh_rain && bin/mesh_rain
//   bin/mesh_rain -rate 30 -max 1000 -size 50       more, smaller, faster
//   bin/mesh_rain -obj models/torus.obj             only one model
//   bin/mesh_rain -gpu off                          everything on the CPU
//
// Keys: Space pause / resume the rain, + / - rain faster / slower, Esc quit.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lr.h"
#include "gpu/gpu.h"
#include "gpu/view_gpu.h"
#include "gpu/rigid_gpu.h"

#define ROOM 555.0
#define MAX_MODELS 8
#define STEP (1.0 / 60.0)        // physics steps of at most 1/60 s (4 substeps each)

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct { int mesh, geom; const char *name; } Model;

static Rng rng;
static real rnd(real lo, real hi){ return lo + (hi - lo) * rng_real(&rng); }

static Quat random_rot(void){
    Vec3 axis = v_norm(vec3(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)));
    return quat_axis_angle(axis, rnd(0, (real)PI));
}

// a spot under the ceiling, away from the walls and in front of the lamp
static Vec3 drop_spot(real size){
    real m = size * 0.6 + 10;
    return vec3(rnd(m, ROOM - m), rnd(ROOM - 1.6 * size, ROOM - 0.7 * size), rnd(m + 40, ROOM - m));
}

static void usage(void){
    printf("usage: mesh_rain [-n width] [-h height] [-rate meshes/s] [-max bodies] [-size units]\n"
           "                 [-obj model.obj|.ply] [-gpu on|off] [-seed n] [-autoquit seconds]\n");
}

int main(int argc, char **argv){
    int W = 800, H = 800, max_bodies = -1, use_gpu = 1, seed = 1;
    double rate = 8, autoquit = 0;
    real size = 70;
    const char *only = NULL;
    for (int i = 1; i < argc; i++){
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-help") || !strcmp(a, "--help")){ usage(); return 0; }
        if (!v){ usage(); return 1; }
        if (!strcmp(a, "-n")) W = atoi(v);
        else if (!strcmp(a, "-h")) H = atoi(v);
        else if (!strcmp(a, "-rate")) rate = atof(v);
        else if (!strcmp(a, "-max")) max_bodies = atoi(v);
        else if (!strcmp(a, "-size")) size = (real)atof(v);
        else if (!strcmp(a, "-obj")) only = v;
        else if (!strcmp(a, "-gpu")) use_gpu = strcmp(v, "off") != 0;
        else if (!strcmp(a, "-seed")) seed = atoi(v);
        else if (!strcmp(a, "-autoquit")) autoquit = atof(v);
        else { fprintf(stderr, "unknown option %s\n", a); usage(); return 1; }
        i++;
    }
    if (W < 64) W = 64;
    if (H < 64) H = 64;
    if (rate < 0) rate = 0;
    if (size < 10) size = 10;
    rng_seed(&rng, 0x5EEDULL + (uint64_t)seed * 0x9E3779B97F4A7C15ULL, 3);
    srgb_lut_init();

    // ---- the room and the models
    static Scene scene;
    SceneOptions so = {.kind = SCENE_EMPTY};
    scene_cornell_box(&scene, &so);

    PhysWorld pw;
    phys_init(&pw);                                   // no air: pw.fluid stays NULL
    Aabb room = {vec3(0, 0, 0), vec3(ROOM, ROOM, ROOM)};
    rigid_set_room(&pw.rigid, room);

    static const Vec3 colors[] = {{0.80, 0.55, 0.25}, {0.25, 0.55, 0.85}, {0.85, 0.30, 0.30},
                                  {0.35, 0.75, 0.40}, {0.90, 0.85, 0.30}, {0.70, 0.45, 0.80}};
    int ncolors = 0, mats[6];
    for (int k = 0; k < 6; k++){
        Material m = {MAT_DIFFUSE, colors[k], vec3(0, 0, 0), 1};
        int id = scene_add_material(&scene, m);
        if (id >= 0) mats[ncolors++] = id;
    }
    if (ncolors == 0){ fprintf(stderr, "no room for materials\n"); return 1; }

    const char *paths[] = {"models/dragon_recon/dragon_vrip_res4.ply", "models/torus.obj",
                           "models/cube.obj", "models/sphere.obj"};
    Model models[MAX_MODELS];
    int nmodels = 0;
    for (int k = 0; k < (only ? 1 : 4) && nmodels < MAX_MODELS; k++){
        const char *path = only ? only : paths[k];
        FILE *f = fopen(path, "rb");
        if (!f){ printf("(%s not found: skipped)\n", path); continue; }
        fclose(f);
        int mesh = scene_load_mesh_object(&scene, path, mats[0], size);
        if (mesh < 0) continue;
        int geom = rigid_geom_hull(&pw.rigid, scene.meshes[mesh].tris, scene.meshes[mesh].ntris, 0);
        if (geom < 0){ fprintf(stderr, "could not make a convex hull of %s\n", path); continue; }
        models[nmodels++] = (Model){mesh, geom, path};
        printf("model %s: %d triangles, hull of %d corners\n", path, scene.meshes[mesh].ntris,
               pw.rigid.geoms[geom]->hull.nv);
    }
    if (nmodels == 0){ fprintf(stderr, "no model to drop (run from the repository's folder; make models)\n"); return 1; }

    // ---- the window, and the GPU in it if there is one
    LrWindow *win = window_open(W, H, "mesh rain");
    if (!win) return 1;
    Framebuffer *fb = fb_create(W, H);
    if (!fb){ fprintf(stderr, "out of memory\n"); return 1; }
    Gpu *gpu = NULL;
    GpuView *gview = NULL;
    GpuRigid *grigid = NULL;
    if (use_gpu && gpu_available()){
        void *dpy;
        uintptr_t wh;
        gpu = window_native(win, &dpy, &wh) ? gpu_create_window(dpy, wh, W, H) : NULL;
        double t0 = now_s();
        while (gpu && gpu_state(gpu) == GPU_PENDING && now_s() - t0 < 20) gpu_poll(gpu);
        if (gpu && gpu_state(gpu) == GPU_READY) gview = gpu_view_create(gpu, W, H);
        if (gview){
            grigid = gpu_rigid_create(gpu, &pw);
            printf("GPU: %s draws the room and simulates the bodies%s\n", gpu_message(gpu), grigid ? "" : " (no: the CPU does)");
        } else {
            printf("GPU not used (%s): the CPU does everything\n", gpu ? gpu_message(gpu) : "none");
            if (gpu){ gpu_destroy(gpu); gpu = NULL; }
        }
    } else printf("no GPU in this build (make deps) or -gpu off: the CPU does everything\n");
    // a full room's pile: on an RTX 3070, 500 hulls keep the screen's 165 frames/s; past ~700
    // the hulls' contact tests (48-corner SAT, LR_GPU_TIMING=1) become the limit. The CPU's
    // solver and rasterizer slow down past a hundred or so.
    if (max_bodies < 0) max_bodies = grigid ? 500 : 120;
    if (max_bodies < 1) max_bodies = 1;

    int *ring = malloc((size_t)max_bodies * sizeof(int));     // the falling bodies, oldest first
    if (!ring){ fprintf(stderr, "out of memory\n"); return 1; }
    int nring = 0, oldest = 0;

    RenderParams rp = render_params_camera(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, W, H);
    MarchParams mp = march_default_params();

    printf("\nkeys: Space pause the rain, + / - faster / slower, Esc quit\n"
           "raining %.0f meshes per second, at most %d in the room (then the oldest is dropped again)\n\n", rate, max_bodies);

    double t_start = now_s(), t_last = t_start, t_title = t_start, spawn_debt = 0;
    double phys_ms = 0, draw_ms = 0, last_phys = 0, last_draw = 0;
    long frames = 0, title_frames = 0, total_frames = 0;
    int paused = 0, running = 1;
    double fps = 0;
    while (running){
        LrEvent ev;
        while (window_poll(win, &ev)){
            if (ev.type == LR_CLOSE) running = 0;
            if (ev.type != LR_KEYDOWN) continue;
            if (ev.key == LR_KEY_ESCAPE) running = 0;
            else if (ev.key == ' ') paused = !paused;
            else if (ev.key == '+' || ev.key == '=') rate = rate < 1 ? 1 : rate * 1.5;
            else if (ev.key == '-') rate /= 1.5;
        }
        double t = now_s(), dt = t - t_last;
        t_last = t;
        if (dt > 1.0 / 20) dt = 1.0 / 20;              // a slow frame shows slow motion, no jump

        // ---- the rain: new bodies until -max, then the oldest one taken back up
        if (!paused){
            spawn_debt += rate * dt;
            while (spawn_debt >= 1){
                spawn_debt -= 1;
                const Model *m = &models[(int)(rng_real(&rng) * nmodels) % nmodels];
                Vec3 at = drop_spot(size);
                Quat q = random_rot();
                Vec3 vel = vec3(rnd(-80, 80), rnd(-150, 0), rnd(-80, 80));
                Vec3 spin = vec3(rnd(-3, 3), rnd(-3, 3), rnd(-3, 3));
                int b;
                if (nring < max_bodies){
                    b = rigid_add_hull_body(&pw.rigid, m->geom, xform_make(at, quat_to_mat3(q).m, 1), 0.8f);
                    if (b < 0) continue;
                    phys_add_mesh_to_scene(&pw, b, &scene, m->mesh, mats[(int)(rng_real(&rng) * ncolors) % ncolors]);
                    ring[nring++] = b;
                } else {
                    b = ring[oldest];
                    oldest = (oldest + 1) % max_bodies;
                    Body *body = &pw.rigid.bodies[b];
                    body->pos = at;
                    body->rot = q;
                }
                Body *body = &pw.rigid.bodies[b];
                body->vel = vel;
                body->angvel = spin;
                body->dirty = 1;
            }
        }

        // ---- physics (steps of at most 1/60 s), then the instances follow the bodies
        double p0 = now_s();
        int n = (int)(dt / STEP + 0.999);
        if (n < 1) n = 1;
        for (int k = 0; k < n; k++) phys_step(&pw, (real)(dt / n));
        phys_sync_scene(&pw, &scene);
        double p1 = now_s();

        // ---- draw
        if (gview){
            if (!gpu_view_present_scene(gview, &scene, &rp, &mp, 0)){
                fprintf(stderr, "this GPU cannot draw the scene: the CPU rasterizes from now on\n");
                gpu_view_destroy(gview);
                gview = NULL;
            }
            if (gpu && gpu_state(gpu) == GPU_FAILED){
                fprintf(stderr, "GPU lost (%s): the CPU carries on\n", gpu_message(gpu));
                gpu_rigid_destroy(grigid);
                grigid = NULL;
                if (gview) gpu_view_destroy(gview);
                gview = NULL;
            }
        }
        if (!gview){
            fb_clear(fb, rgb8(0, 0, 0));
            render_scene(fb, &scene, &rp);
            window_present(win, fb);
        }
        double p2 = now_s();
        phys_ms += (p1 - p0) * 1000;
        draw_ms += (p2 - p1) * 1000;
        frames++;
        title_frames++;
        total_frames++;

        // ---- the frame rate, twice a second
        if (t - t_title >= 0.5){
            fps = title_frames / (t - t_title);
            char title[256];
            snprintf(title, sizeof(title), "mesh rain | %.0f fps (%.1f ms) | %d meshes, %.0f/s%s | physics %.2f ms%s, draw %.2f ms (CPU time) | %s",
                     fps, 1000.0 / (fps > 0 ? fps : 1), nring, rate, paused ? " (paused)" : "",
                     phys_ms / frames, grigid ? " (GPU)" : "", draw_ms / frames,
                     gview ? gpu_message(gpu) : "CPU raster");
            window_set_title(win, title);
            last_phys = phys_ms / frames;
            last_draw = draw_ms / frames;
            t_title = t;
            title_frames = 0;
            phys_ms = draw_ms = 0;
            frames = 0;
        }
        if (autoquit > 0 && t - t_start > autoquit){
            printf("autoquit after %.1f s: %.0f frames/s now (%.0f on average), %d meshes, physics on the %s (%.2f ms of CPU), "
                   "drawn on the %s (%.2f ms of CPU)\n", t - t_start, fps, total_frames / (t - t_start), nring,
                   grigid ? "GPU" : "CPU", last_phys, gview ? "GPU" : "CPU", last_draw);
            running = 0;
        }
    }

    gpu_rigid_destroy(grigid);                         // the bodies back to the CPU first
    if (gview) gpu_view_destroy(gview);
    if (gpu) gpu_destroy(gpu);
    free(ring);
    phys_free(&pw);
    scene_free(&scene);
    fb_destroy(fb);
    window_close(win);
    return 0;
}
