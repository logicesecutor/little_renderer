// test_gpu_fluid.c - the GPU air (gpu/fluid_gpu.c) against the CPU air (phys/fluid.c).
//
// Two identical worlds: a room of air, a cloud, a fixed ball that pushes the air upwards
// (an obstacle with a velocity), wind, buoyancy and vorticity, and an explosion at step 10.
// One is stepped on the CPU, the other on the GPU, with the same time steps; then the GPU's
// cloud is read back and compared voxel by voxel, and the air around the ball (the drag
// samples) is compared too. The two do not round alike (multigrid sweeps in parallel, fused
// multiply-adds...), so they agree closely but not to the last bit.
//
// Then (natively) the cloud as the GPU views draw it: a view that takes the densities from
// the fluid's own buffers (gpu_fluid_attach_view) must show the same frame, to the last
// bit, as one given the cloud read back to the CPU, after an odd and an even number of
// steps (the fluid's two buffers take turns); and the view and the fluid can be destroyed
// in either order, and the world made again (the viewer's restart), without the view
// showing anything else than the cloud there is.
//
// Part of make test (without a GPU it says so and passes). In a browser: make webtest, then
// open web/build/test_gpu_fluid.html; the result is printed, and kept in window.lrTest
// ("PASS ..." or "FAIL ...").
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#include "lr.h"
#include "app/gpu_frame.h"
#include "gpu/fluid_gpu.h"

#define STEPS 40
#define DT (1.0f / 30)

static PhysWorld wc, wg;      // CPU, GPU
static Gpu *g;
static GpuFluid *gf;
static int stage, step;
static uint32_t v_before;
static int failed;
static float *start;          // the cloud before the steps

// a room of air with a cloud in it (the caller's: phys_free does not free it)
static Volume *build(PhysWorld *pw){
    Aabb room = {vec3(0, 0, 0), vec3(555, 555, 555)};
    Volume *v = volume_create(64, 64, 64, room);
    if (!v){ fprintf(stderr, "out of memory\n"); exit(1); }
    CloudParams cp = cloud_default_params();
    cp.resolution = 64;
    volume_add_cloud(v, (Aabb){vec3(150, 230, 150), vec3(420, 430, 420)}, &cp);
    phys_init(pw);
    rigid_set_room(&pw->rigid, room);
    pw->fluid = fluid_create(room, 555.0f / 32, v);
    if (!pw->fluid){ fprintf(stderr, "out of memory\n"); exit(1); }
    pw->fluid->buoyancy = 40;
    pw->fluid->wind = vec3(25, 0, 0);
    pw->fluid->vorticity = 0.3f;
    pw->fluid->open = FLUID_OPEN_ZLO;
    int i = rigid_add_sphere(&pw->rigid, vec3(278, 160, 278), 70, 1.0f);
    Body *b = &pw->rigid.bodies[i];
    b->inv_mass = 0;              // fixed: it stays, but its surface moves the air
    b->vel = vec3(0, 150, 0);
    b->angvel = vec3(0, 0, 1.5f);
    return v;
}

static void check(const char *what, double got, double want, double tol){
    int ok = fabs(got - want) <= tol;
    if (!ok) failed = 1;
    printf("  %-60s %10.5f  (want %.4f +- %.4f)  %s\n", what, got, want, tol, ok ? "ok" : "FAILED");
}

static void compare(void){
    const Volume *a = wc.fluid->density, *b = wg.fluid->density;
    size_t n = (size_t)a->nx * a->ny * a->nz;
    double sa = 0, sb = 0, diff = 0, mx = 0, ca[3] = {0}, cb[3] = {0};
    for (size_t i = 0; i < n; i++){
        double x = a->density[i], y = b->density[i];
        sa += x; sb += y;
        diff += fabs(x - y);
        if (fabs(x - y) > mx) mx = fabs(x - y);
        int vx = (int)(i % a->nx), vy = (int)(i / a->nx % a->ny), vz = (int)(i / ((size_t)a->nx * a->ny));
        ca[0] += x * vx; ca[1] += x * vy; ca[2] += x * vz;
        cb[0] += y * vx; cb[1] += y * vy; cb[2] += y * vz;
    }
    printf("after %d steps of %.4f s (64^3 cloud, 32^3 air, a moving obstacle, an explosion):\n", STEPS, (double)DT);
    check("cloud amount, GPU / CPU", sb / sa, 1, 0.01);
    check("voxel difference: sum |GPU - CPU| / sum CPU", diff / sa, 0, 0.03);
    check("largest voxel difference / CPU max density", mx / a->max_density, 0, 0.15);
    double shift = 0;
    for (int k = 0; k < 3; k++){ double d = ca[k] / sa - cb[k] / sb; shift += d * d; }
    check("cloud center, GPU - CPU (voxels)", sqrt(shift), 0, 0.05);
    // and the test means something: the cloud did move
    double moved = 0, s0 = 0;
    for (size_t i = 0; i < n; i++){ moved += fabs(a->density[i] - start[i]); s0 += start[i]; }
    check("the CPU cloud changed: sum |end - start| / sum start (> 0.3)", moved / s0 > 0.3 ? 1 : 0, 1, 0);
    Vec3 va = phys_air_velocity_around(&wc, 0), vb = phys_air_velocity_around(&wg, 0);
    printf("  air around the ball: CPU (%.2f %.2f %.2f)  GPU (%.2f %.2f %.2f)\n", (double)va.x, (double)va.y, (double)va.z,
           (double)vb.x, (double)vb.y, (double)vb.z);
    check("air around the ball: |GPU - CPU| / |CPU|", v_len(v_sub(va, vb)) / v_len(va), 0, 0.03);
}

// one turn of the comparison (the browser calls it once per frame; natively the GPU is
// waited for in between): 1 when it is over
static int advance(void){
    switch (stage){
    case 0:                                                    // the GPU starts
        if (gpu_state(g) == GPU_PENDING) return 0;
        if (gpu_state(g) == GPU_FAILED || !(gf = gpu_fluid_create(g, &wg))){
            printf("FAIL: no GPU (%s)\n", gpu_message(g));
            failed = 2;
            return 1;
        }
        printf("GPU: %s\n", gpu_message(g));
        stage = 1;
        return 0;
    case 1:                                                    // one step each, in lockstep
        if (gpu_fluid_pending(gf)) return 0;
        if (step == STEPS){
            v_before = wg.fluid->density->version;
            gpu_fluid_sync_cloud(gf);
            stage = 2;
            return 0;
        }
        if (step == 10){
            phys_blast(&wc, vec3(300, 330, 260), 220, 600, 0);
            phys_blast(&wg, vec3(300, 330, 260), 220, 600, 0);
        }
        phys_step(&wc, DT);
        phys_step(&wg, DT);
        step++;
        return 0;
    case 2:                                                    // the cloud comes back
        if (wg.fluid->density->version == v_before || gpu_fluid_pending(gf)) return 0;
        compare();
        stage = 3;
        return 1;
    }
    return 1;
}

#ifdef __EMSCRIPTEN__
static void frame(void){
    if (!advance()) return;
    printf("%s: GPU air test\n", failed ? "FAIL" : "PASS");
    if (failed == 2) EM_ASM({ window.lrTest = 'FAIL no GPU'; });
    else EM_ASM({ window.lrTest = UTF8ToString($0); }, failed ? "FAIL" : "PASS");
    emscripten_cancel_main_loop();
}
#else

// ---- the cloud as the views draw it -----------------------------------------------------------
#define W 96
#define H 96
static Scene scene;           // the room, and the cloud of the world `w`
static PhysWorld w;
static RenderParams rp;
static MarchParams mp;
static Framebuffer *fa, *fb;

static void yes(const char *what, int ok){
    if (!ok) failed = 1;
    printf("  %-60s %s\n", what, ok ? "ok" : "FAILED");
}

static void world_make(void){
    memset(&scene, 0, sizeof(scene));
    SceneOptions so = {.kind = SCENE_BOXES};
    scene_cornell_box(&scene, &so);
    scene_add_volume(&scene, build(&w));       // (the scene's from here on: scene_free frees it)
    rp = render_params_camera(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, W, H);
}
static void world_free(void){
    phys_free(&w);
    scene_free(&scene);
}

// n steps of the air (each waited for), then the GPU's cloud in the CPU's Volume too
static void steps_and_sync(GpuFluid *f, int n){
    for (int k = 0; k < n; k++){
        phys_step(&w, DT);
        gpu_wait(g);
    }
    Volume *v = w.fluid->density;
    uint32_t v0 = v->version;
    gpu_fluid_sync_cloud(f);
    for (int k = 0; k < 400 && v->version == v0; k++) gpu_wait(g);
    if (v->version == v0) yes("the cloud read back", 0);
}

// the frame of `a` (it may draw the fluid's buffer) and of `b` (never attached: the CPU's cloud)
static int same_frames(GpuView *a, GpuView *b){
    if (gpu_frame_raster(g, a, &scene, &rp, &mp, 1, fa) || gpu_frame_raster(g, b, &scene, &rp, &mp, 1, fb)) return 0;
    return !memcmp(fa->color, fb->color, (size_t)W * H * sizeof(uint32_t));
}

static long pixels_differing(const Framebuffer *x, const Framebuffer *y){
    long n = 0;
    for (int i = 0; i < W * H; i++) n += x->color[i] != y->color[i];
    return n;
}

static void views(void){
    printf("the cloud drawn from the fluid's buffers (a %dx%d frame, rasterized, the cloud marched):\n", W, H);
    mp = march_default_params();
    fa = fb_create(W, H);
    fb = fb_create(W, H);
    Framebuffer *f0 = fb_create(W, H);
    world_make();
    GpuView *cpu = gpu_view_create(g, W, H), *v1 = gpu_view_create(g, W, H);
    GpuFluid *f = gpu_fluid_create(g, &w);
    if (!fa || !fb || !f0 || !cpu || !v1 || !f){ yes("a fluid and two views", 0); return; }
    gpu_fluid_set_wait(f, 1);
    gpu_fluid_attach_view(f, v1);
    yes("before any step: the same frame", same_frames(v1, cpu));
    memcpy(f0->color, fa->color, (size_t)W * H * sizeof(uint32_t));
    steps_and_sync(f, 3);
    yes("after 3 steps (the cloud in its other buffer)", same_frames(v1, cpu));
    long moved = pixels_differing(f0, fa);
    char what[96];
    snprintf(what, sizeof(what), "and the cloud moved (%ld pixels changed)", moved);
    yes(what, moved > 0);
    steps_and_sync(f, 1);
    yes("after 4 steps (back in the first)", same_frames(v1, cpu));
    // the view goes first: the fluid steps on, and another view can take its cloud
    gpu_view_destroy(v1);
    steps_and_sync(f, 3);
    GpuView *v2 = gpu_view_create(g, W, H);
    if (!v2){ yes("a third view", 0); return; }
    gpu_fluid_attach_view(f, v2);
    yes("the view destroyed, 3 more steps, a new view", same_frames(v2, cpu));
    // the fluid goes first: the view draws the CPU's cloud, as it is and as the CPU moves it
    gpu_fluid_destroy(f);
    yes("the fluid destroyed: the CPU's cloud", same_frames(v2, cpu));
    memcpy(f0->color, fa->color, (size_t)W * H * sizeof(uint32_t));
    for (int k = 0; k < 3; k++) phys_step(&w, DT);
    yes("3 steps on the CPU", same_frames(v2, cpu) && pixels_differing(f0, fa) > 0);
    // the viewer's restart, a few times: the world freed and made again (its cloud may get
    // the old one's address), a new fluid, the same view
    int ok = 1;
    f = NULL;
    for (int r = 0; r < 6 && ok; r++){
        world_free();
        world_make();
        if (!(f = gpu_fluid_create(g, &w))){ ok = 0; break; }
        gpu_fluid_set_wait(f, 1);
        gpu_fluid_attach_view(f, v2);
        ok = same_frames(v2, cpu);
        memcpy(f0->color, fa->color, (size_t)W * H * sizeof(uint32_t));
        steps_and_sync(f, 2 + r % 2);
        ok = ok && same_frames(v2, cpu) && pixels_differing(f0, fa) > 0;
        if (r < 5){ gpu_fluid_destroy(f); f = NULL; }   // (the last one outlives the view, below)
    }
    yes("6 restarts: a new world and fluid, the same view", ok);
    gpu_view_destroy(v2);
    gpu_view_destroy(cpu);
    if (f){
        steps_and_sync(f, 1);                 // nothing draws it any more
        gpu_fluid_destroy(f);
    }
    world_free();
    fb_destroy(fa); fb_destroy(fb); fb_destroy(f0);
}
#endif

int main(void){
#ifndef __EMSCRIPTEN__
    printf("test_gpu_fluid\n");
    g = gpu_create(NULL, 16, 16);
    for (int k = 0; k < 20000 && g && gpu_state(g) == GPU_PENDING; k++) gpu_poll(g);
    if (!g || gpu_state(g) != GPU_READY){
        printf("  no GPU (%s): skipped\nall GPU air tests passed\n", g ? gpu_message(g) : "none");
        gpu_destroy(g);
        return 0;
    }
#endif
    Volume *vc = build(&wc), *vg = build(&wg);
    const Volume *v = wc.fluid->density;
    size_t n = (size_t)v->nx * v->ny * v->nz;
    start = malloc(n * sizeof(float));
    if (!start) return 1;
    memcpy(start, v->density, n * sizeof(float));
#ifdef __EMSCRIPTEN__
    (void)vc; (void)vg;
    g = gpu_create(NULL, 16, 16);
    emscripten_set_main_loop(frame, 0, 0);
    return 0;
#else
    for (long k = 0; k < 100000 && !advance(); k++)
        if (gpu_state(g) != GPU_READY){ printf("  the GPU was lost (%s)\n", gpu_message(g)); failed = 1; break; }
        else gpu_wait(g);
    if (stage != 3) failed = 1;
    gpu_fluid_destroy(gf);
    phys_free(&wc); phys_free(&wg);
    volume_free(vc); volume_free(vg);
    free(start);
    if (!failed) views();
    gpu_destroy(g);
    printf(failed ? "test_gpu_fluid: FAILED\n" : "all GPU air tests passed\n");
    return failed ? 1 : 0;
#endif
}
