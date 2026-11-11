// test_gpu_fluid.c - the GPU air (gpu/fluid_gpu.c) against the CPU air (phys/fluid.c).
//
// Two identical worlds: a room of air, a cloud, a fixed ball that pushes the air upwards
// (an obstacle with a velocity), wind, buoyancy and vorticity, and an explosion at step 10.
// One is stepped on the CPU, the other on the GPU, with the same time steps; then the GPU's
// cloud is read back and compared voxel by voxel, and the air around the ball (the drag
// samples) is compared too. The two do not round alike (multigrid sweeps in parallel, fused
// multiply-adds...), so they agree closely but not to the last bit.
//
// Web only (it needs WebGPU): make webtest, then open web/build/test_gpu_fluid.html; the
// result is printed, and kept in window.lrTest ("PASS ..." or "FAIL ...").
#include <stdio.h>

#ifndef __EMSCRIPTEN__
int main(void){
    printf("test_gpu_fluid needs a GPU build: make webtest, then open web/build/test_gpu_fluid.html\n");
    return 0;
}
#else
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten.h>
#include "core/volume.h"
#include "gpu/gpu.h"
#include "gpu/fluid_gpu.h"
#include "phys/world.h"

#define STEPS 40
#define DT (1.0f / 30)

static PhysWorld wc, wg;      // CPU, GPU
static Gpu *g;
static GpuFluid *gf;
static int stage, step;
static uint32_t v_before;
static int failed;
static float *start;          // the cloud before the steps

static void build(PhysWorld *pw){
    Aabb room = {vec3(0, 0, 0), vec3(555, 555, 555)};
    Volume *v = volume_create(64, 64, 64, room);
    CloudParams cp = cloud_default_params();
    cp.resolution = 64;
    volume_add_cloud(v, (Aabb){vec3(150, 230, 150), vec3(420, 430, 420)}, &cp);
    phys_init(pw);
    rigid_set_room(&pw->rigid, room);
    pw->fluid = fluid_create(room, 555.0f / 32, v);
    pw->fluid->buoyancy = 40;
    pw->fluid->wind = vec3(25, 0, 0);
    pw->fluid->vorticity = 0.3f;
    pw->fluid->open = FLUID_OPEN_ZLO;
    int i = rigid_add_sphere(&pw->rigid, vec3(278, 160, 278), 70, 1.0f);
    Body *b = &pw->rigid.bodies[i];
    b->inv_mass = 0;              // fixed: it stays, but its surface moves the air
    b->vel = vec3(0, 150, 0);
    b->angvel = vec3(0, 0, 1.5f);
}

static void check(const char *what, double got, double want, double tol){
    int ok = fabs(got - want) <= tol;
    if (!ok) failed = 1;
    printf("  %-52s %10.5f  (want %.4f +- %.4f)  %s\n", what, got, want, tol, ok ? "ok" : "FAILED");
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
    const char *verdict = failed ? "FAIL" : "PASS";
    printf("%s: GPU air test\n", verdict);
    EM_ASM({ window.lrTest = UTF8ToString($0); }, failed ? "FAIL" : "PASS");
}

static void frame(void){
    switch (stage){
    case 0:                                                    // the GPU starts
        if (gpu_state(g) == GPU_PENDING) return;
        if (gpu_state(g) == GPU_FAILED || !(gf = gpu_fluid_create(g, &wg))){
            printf("FAIL: no GPU (%s)\n", gpu_message(g));
            EM_ASM({ window.lrTest = 'FAIL no GPU'; });
            emscripten_cancel_main_loop();
            return;
        }
        printf("GPU: %s\n", gpu_message(g));
        stage = 1;
        return;
    case 1:                                                    // one step each, in lockstep
        if (gpu_fluid_pending(gf)) return;
        if (step == STEPS){
            v_before = wg.fluid->density->version;
            gpu_fluid_sync_cloud(gf);
            stage = 2;
            return;
        }
        if (step == 10){
            phys_blast(&wc, vec3(300, 330, 260), 220, 600, 0);
            phys_blast(&wg, vec3(300, 330, 260), 220, 600, 0);
        }
        phys_step(&wc, DT);
        phys_step(&wg, DT);
        step++;
        return;
    case 2:                                                    // the cloud comes back
        if (wg.fluid->density->version == v_before || gpu_fluid_pending(gf)) return;
        compare();
        stage = 3;
        emscripten_cancel_main_loop();
        return;
    }
}

int main(void){
    build(&wc);
    build(&wg);
    const Volume *v = wc.fluid->density;
    size_t n = (size_t)v->nx * v->ny * v->nz;
    start = malloc(n * sizeof(float));
    if (!start) return 1;
    memcpy(start, v->density, n * sizeof(float));
    g = gpu_create(NULL, 16, 16);
    emscripten_set_main_loop(frame, 0, 0);
    return 0;
}
#endif
