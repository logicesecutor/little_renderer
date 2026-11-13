// test_gpu_rigid.c - the GPU's rigid bodies (gpu/rigid_gpu.c) against known answers and
// against the CPU's (phys/rigid.c).
//
// The same checks as test_physics.c (free fall, bounce, momentum, a resting stack, a tilted
// box landing on a face), each stepped on the GPU; then a pile of balls and boxes dropped in
// the room, on both, compared by what must hold whatever the solver's order: everyone inside
// the room, nobody sunk into anybody, all at rest; and the time per step, CPU vs GPU, from a
// few tens to a few thousand bodies.
//
// Needs a GPU build (make deps): bin/test_gpu_rigid. Without a GPU it says so and passes.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lr.h"
#include "gpu/gpu.h"
#include "gpu/rigid_gpu.h"

static int failures = 0;
static Gpu *g;

static void check(const char *what, double got, double want, double tol){
    double err = got - want;
    if (err < 0) err = -err;
    int ok = err <= tol;
    printf("  %-52s %12.4f  (want %.4f +- %.4f)  %s\n", what, got, want, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static double now_ms(void){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static double max_speed(const RigidWorld *w){
    double m = 0;
    for (int i = 0; i < w->nbodies; i++){
        const Body *b = &w->bodies[i];
        double s = v_len(b->vel) + v_len(b->angvel) * v_len(b->half);
        if (s > m) m = s;
    }
    return m;
}

// a world (no air) whose bodies the GPU steps, or the CPU (gpu = 0)
typedef struct { PhysWorld pw; GpuRigid *gr; } World;

static void world_init(World *w, int gpu){
    phys_init(&w->pw);
    w->pw.air_density = 0;        // no air: the answers above are for a vacuum
    w->gr = NULL;
    (void)gpu;
}
static void world_start(World *w, int gpu){
    if (!gpu) return;
    w->gr = gpu_rigid_create(g, &w->pw);
    gpu_rigid_set_wait(w->gr, 1);
}
// t seconds in steps of 1/240 s, 4 substeps each (1/960 s, as test_physics.c)
static void world_run(World *w, double t){
    int n = (int)(t * 240 + 0.5);
    for (int k = 0; k < n; k++) phys_step(&w->pw, 1.0f / 240);
    gpu_rigid_sync(w->gr);
}
static void world_free(World *w){
    gpu_rigid_destroy(w->gr);
    phys_free(&w->pw);
}

static void test_basics(void){
    printf("on the GPU: free fall, bounce, momentum, a resting stack, a box landing flat\n");
    World w;
    world_init(&w, 1);
    w.pw.rigid.angular_damping = 0;
    rigid_add_sphere(&w.pw.rigid, vec3(0, 1000, 0), 10, 1);
    world_start(&w, 1);
    world_run(&w, 1.0);
    check("free fall: velocity after 1 s", w.pw.rigid.bodies[0].vel.y, -981, 0.5);
    check("free fall: height after 1 s", w.pw.rigid.bodies[0].pos.y, 1000 - 0.5 * 981, 1.5);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(-500, 0, -500), vec3(500, 1000, 500)});
    w.pw.rigid.wall_restitution = 0.5f;
    int b = rigid_add_sphere(&w.pw.rigid, vec3(0, 410, 0), 10, 1);
    w.pw.rigid.bodies[b].restitution = 0.5f;
    w.pw.rigid.angular_damping = 0;
    world_start(&w, 1);
    double top = 0;
    int bounced = 0;
    for (int k = 0; k < 240 * 2; k++){
        phys_step(&w.pw, 1.0f / 240);
        gpu_rigid_sync(w.gr);
        const Body *s = &w.pw.rigid.bodies[b];
        if (s->vel.y > 0) bounced = 1;
        if (bounced && s->pos.y > top) top = s->pos.y;
    }
    check("bounce: peak after the bounce (center)", top, 10 + 0.25 * 400, 4);
    world_free(&w);

    world_init(&w, 1);
    w.pw.rigid.gravity = vec3(0, 0, 0);
    w.pw.rigid.angular_damping = 0;
    int a = rigid_add_sphere(&w.pw.rigid, vec3(0, 0, 0), 20, 1);
    rigid_add_sphere(&w.pw.rigid, vec3(100, 8, 0), 25, 2);
    rigid_add_box(&w.pw.rigid, vec3(0, 120, 3), vec3(20, 15, 25), quat_axis_angle(v_norm(vec3(1, 2, 0.5)), 0.7), 1.5);
    rigid_add_box(&w.pw.rigid, vec3(110, 125, -6), vec3(18, 22, 15), quat_axis_angle(v_norm(vec3(0.3, 1, 1)), 1.2), 0.8);
    w.pw.rigid.bodies[0].vel = vec3(200, 0, 0);
    w.pw.rigid.bodies[1].vel = vec3(-150, 0, 0);
    w.pw.rigid.bodies[2].vel = vec3(180, 0, 0);
    w.pw.rigid.bodies[3].vel = vec3(-160, 0, 0);
    Vec3 p0 = rigid_momentum(&w.pw.rigid);
    real scale = v_len(w.pw.rigid.bodies[0].vel) * w.pw.rigid.bodies[0].mass * 4;
    world_start(&w, 1);
    world_run(&w, 1.0);
    Vec3 p1 = rigid_momentum(&w.pw.rigid);
    check("momentum: change / total", v_len(v_sub(p1, p0)) / scale, 0, 1e-3);
    check("momentum: sphere a turned around (vx < 0)", w.pw.rigid.bodies[a].vel.x < 0, 1, 0);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    for (int k = 0; k < 3; k++) rigid_add_box(&w.pw.rigid, vec3(278, 30 + 60 * k, 278), vec3(40, 30, 40), quat_identity(), 1);
    int s = rigid_add_sphere(&w.pw.rigid, vec3(278, 180 + 25, 278), 25, 1);
    world_start(&w, 1);
    world_run(&w, 3.0);
    check("stack: top box height", w.pw.rigid.bodies[2].pos.y, 150, 1.5);
    check("stack: top box drift sideways", r_fabs(w.pw.rigid.bodies[2].pos.x - 278) + r_fabs(w.pw.rigid.bodies[2].pos.z - 278), 0, 1);
    check("stack: ball height", w.pw.rigid.bodies[s].pos.y, 205, 1.5);
    check("stack: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 1);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    b = rigid_add_box(&w.pw.rigid, vec3(278, 300, 278), vec3(50, 20, 35), quat_axis_angle(vec3(1, 0.3, 0.5), 0.9), 1);
    world_start(&w, 1);
    world_run(&w, 4.0);
    real y = w.pw.rigid.bodies[b].pos.y;
    check("tilted box: center height - nearest half size", r_fmin(r_fabs(y - 50), r_fmin(r_fabs(y - 20), r_fabs(y - 35))), 0, 1.5);
    check("tilted box: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 1);
    world_free(&w);
}

// the 12 triangles of a box (half sizes h), turned by r and moved to c
static int box_tris(Triangle *t, Vec3 h, Mat3 r, Vec3 c){
    Vec3 p[8];
    for (int i = 0; i < 8; i++)
        p[i] = v_add(c, mat3_mul_vec(r, vec3((i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z)));
    static const int q[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    int n = 0;
    for (int f = 0; f < 6; f++){
        Vec3 a = p[q[f][0]], b = p[q[f][1]], cc = p[q[f][2]], d = p[q[f][3]];
        t[n].q = a; t[n].u = v_sub(b, a); t[n].v = v_sub(cc, a); t[n].n[0] = -1; t[n].material = 0; n++;
        t[n].q = a; t[n].u = v_sub(cc, a); t[n].v = v_sub(d, a); t[n].n[0] = -1; t[n].material = 0; n++;
    }
    return n;
}

// a torus around the y axis (radii R, r), nu x nv quads
static int torus_tris(Triangle *t, real R, real r, int nu, int nv){
    int n = 0;
    for (int i = 0; i < nu; i++)
        for (int j = 0; j < nv; j++){
            Vec3 p[4];
            for (int k = 0; k < 4; k++){
                real u = (i + (k == 1 || k == 2)) * 2 * (real)PI / nu, v = (j + (k >= 2)) * 2 * (real)PI / nv;
                p[k] = vec3((R + r * r_cos(v)) * r_cos(u), r * r_sin(v), (R + r * r_cos(v)) * r_sin(u));
            }
            t[n].q = p[0]; t[n].u = v_sub(p[1], p[0]); t[n].v = v_sub(p[2], p[0]); t[n].n[0] = -1; t[n].material = 0; n++;
            t[n].q = p[0]; t[n].u = v_sub(p[2], p[0]); t[n].v = v_sub(p[3], p[0]); t[n].n[0] = -1; t[n].material = 0; n++;
        }
    return n;
}

static void test_meshes(void){
    printf("on the GPU: hulls landing and stacking, a ball, a hull and a box on a fixed mesh\n");
    static Triangle tri[4096];
    Mat3 turn = quat_to_mat3(quat_axis_angle(v_norm(vec3(0.2, 0.4, 1)), 0.6));
    World w;
    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    int n = box_tris(tri, vec3(50, 20, 35), turn, vec3(10, 30, -5));
    int g = rigid_geom_hull(&w.pw.rigid, tri, n, 0);
    Xform place = xform_make(vec3(278, 300, 278), quat_to_mat3(quat_axis_angle(vec3(1, 0.3, 0.5), 0.9)).m, 1);
    int b = rigid_add_hull_body(&w.pw.rigid, g, place, 1);
    world_start(&w, 1);
    world_run(&w, 3.0);
    real y = w.pw.rigid.bodies[b].pos.y;
    check("hull (box mesh): centre height - nearest half size", r_fmin(r_fabs(y - 50), r_fmin(r_fabs(y - 20), r_fabs(y - 35))), 0, 1.5);
    Vec3 top = w.pw.rigid.bodies[b].pos;
    Xform on = body_mesh_xform(&w.pw.rigid.bodies[b]);
    on.m[1][3] += 2 * y + 1;
    int b2 = rigid_add_hull_body(&w.pw.rigid, g, on, 1);
    world_run(&w, 3.0);
    check("second hull on the first: height difference", w.pw.rigid.bodies[b2].pos.y - w.pw.rigid.bodies[b].pos.y, 2 * y, 1.5);
    check("second hull drift sideways", r_fabs(w.pw.rigid.bodies[b2].pos.x - top.x) + r_fabs(w.pw.rigid.bodies[b2].pos.z - top.z), 0, 2);
    check("hulls: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 2);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    n = torus_tris(tri, 60, 20, 32, 16);
    g = rigid_geom_hull(&w.pw.rigid, tri, n, 0);
    b = rigid_add_hull_body(&w.pw.rigid, g, xform_make(vec3(278, 200, 278), quat_to_mat3(quat_axis_angle(vec3(1, 0, 0.3), 0.5)).m, 1), 1);
    world_start(&w, 1);
    world_run(&w, 4.0);
    check("torus hull lands flat: centre height", w.pw.rigid.bodies[b].pos.y, 20, 1.5);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    n = box_tris(tri, vec3(120, 50, 120), mat3_identity(), vec3(0, 50, 0));
    int gm = rigid_geom_mesh(&w.pw.rigid, tri, n);
    rigid_add_mesh_collider(&w.pw.rigid, gm, xform_make(vec3(278, 0, 278), mat3_identity().m, 1));
    int s = rigid_add_sphere(&w.pw.rigid, vec3(220, 300, 278), 25, 1);
    n = box_tris(tri, vec3(30, 20, 30), mat3_identity(), vec3(0, 0, 0));
    g = rigid_geom_hull(&w.pw.rigid, tri, n, 0);
    int hb = rigid_add_hull_body(&w.pw.rigid, g, xform_make(vec3(330, 250, 300), quat_to_mat3(quat_axis_angle(vec3(0, 1, 0), 0.4)).m, 1), 1);
    int bx = rigid_add_box(&w.pw.rigid, vec3(278, 400, 220), vec3(20, 20, 20), quat_identity(), 1);
    world_start(&w, 1);
    world_run(&w, 4.0);
    check("ball on the mesh: centre height", w.pw.rigid.bodies[s].pos.y, 125, 1.5);
    check("hull on the mesh: centre height", w.pw.rigid.bodies[hb].pos.y, 120, 1.5);
    check("box on the mesh: centre height", w.pw.rigid.bodies[bx].pos.y, 120, 1.5);
    check("on the mesh: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 2);
    world_free(&w);
}

// n bodies (2/3 balls, 1/3 boxes) dropped in a room sized for them
static void pile(World *w, int n, int gpu, real *room_side){
    world_init(w, gpu);
    real side = 555 * r_fmax(1, r_pow(n / 60.0f, 1.0f / 2.0f) * 0.8f);
    *room_side = side;
    rigid_set_room(&w->pw.rigid, (Aabb){vec3(0, 0, 0), vec3(side, 4 * side, side)});
    Rng rng;
    rng_seed(&rng, 1234, 5);
    int per_row = (int)(side / 75);
    for (int i = 0; i < n; i++){
        Vec3 at = vec3(45 + 75 * (i % per_row), 60 + 75 * (i / (per_row * per_row)), 45 + 75 * ((i / per_row) % per_row));
        at = v_add(at, vec3(rng_real(&rng) * 8 - 4, 0, rng_real(&rng) * 8 - 4));
        if (i % 3 == 2){
            Vec3 h = vec3(15 + 15 * rng_real(&rng), 15 + 15 * rng_real(&rng), 15 + 15 * rng_real(&rng));
            Vec3 ax = v_norm(vec3(rng_real(&rng) - 0.5f, rng_real(&rng) - 0.5f, rng_real(&rng) - 0.5f));
            rigid_add_box(&w->pw.rigid, at, h, quat_axis_angle(ax, 3 * rng_real(&rng)), 0.7f);
        } else rigid_add_sphere(&w->pw.rigid, at, 18 + 14 * rng_real(&rng), 1 + 6 * rng_real(&rng));
    }
    world_start(w, gpu);
}

// what a settled pile must satisfy, whoever stepped it
static void pile_checks(const char *who, const RigidWorld *rw, real side){
    int outside = 0;
    double worst = 0;
    for (int i = 0; i < rw->nbodies; i++){
        const Body *a = &rw->bodies[i];
        real r = a->shape == BODY_SPHERE ? a->half.x : r_fmin(a->half.x, r_fmin(a->half.y, a->half.z));
        if (a->pos.x < r - 2 || a->pos.z < r - 2 || a->pos.x > side - r + 2 || a->pos.z > side - r + 2 || a->pos.y < r - 2) outside++;
        if (a->shape != BODY_SPHERE) continue;
        for (int j = i + 1; j < rw->nbodies; j++){
            const Body *b = &rw->bodies[j];
            if (b->shape != BODY_SPHERE) continue;
            double over = a->half.x + b->half.x - v_len(v_sub(a->pos, b->pos));
            if (over > worst) worst = over;
        }
    }
    char what[120];
    snprintf(what, sizeof(what), "pile (%s): bodies outside the room", who);
    check(what, outside, 0, 0);
    snprintf(what, sizeof(what), "pile (%s): deepest ball-ball overlap, units", who);
    check(what, worst, 0, 2.5);
    // settled: nobody falling or bouncing any more (balls may still roll: no rolling resistance)
    double vy = 0;
    for (int i = 0; i < rw->nbodies; i++) vy = fmax(vy, fabs(rw->bodies[i].vel.y));
    snprintf(what, sizeof(what), "pile (%s): fastest vertical speed after 8 s, units / s", who);
    check(what, vy, 0, 10);
}

static void test_pile(void){
    printf("a pile of 60 balls and boxes, CPU and GPU\n");
    for (int gpu = 0; gpu < 2; gpu++){
        World w;
        real side;
        pile(&w, 60, gpu, &side);
        world_run(&w, 8.0);    // (on the CPU a ball rolls off a box at 5 s and lands at 6.5)
        pile_checks(gpu ? "GPU" : "CPU", &w.pw.rigid, side);
        world_free(&w);
    }
}

static void timings(void){
    printf("time per 1/60 s step (4 substeps, 16 iterations), CPU vs GPU, while the pile falls and settles:\n");
    int ns[] = {64, 256, 1024, 2048};
    for (size_t k = 0; k < sizeof(ns) / sizeof(ns[0]); k++){
        double ms[2];
        int contacts[2];
        for (int gpu = 0; gpu < 2; gpu++){
            World w;
            real side;
            pile(&w, ns[k], gpu, &side);
            gpu_rigid_set_wait(w.gr, 0);
            if (!gpu && ns[k] > 1024){ ms[0] = -1; contacts[0] = 0; world_free(&w); continue; }
            int steps = 60;
            for (int s = 0; s < 30; s++) phys_step(&w.pw, 1.0f / 60);   // into the thick of it
            double t0 = now_ms();
            for (int s = 0; s < steps; s++) phys_step(&w.pw, 1.0f / 60);
            gpu_rigid_sync(w.gr);
            if (gpu) gpu_wait(g);
            ms[gpu] = (now_ms() - t0) / steps;
            contacts[gpu] = w.pw.rigid.ncontacts;
            world_free(&w);
        }
        if (ms[0] < 0) printf("  %5d bodies: CPU (skipped, minutes), GPU %7.2f ms (%d contacts)\n", ns[k], ms[1], contacts[1]);
        else printf("  %5d bodies: CPU %7.2f ms (%d contacts), GPU %7.2f ms (%d contacts)\n", ns[k], ms[0], contacts[0], ms[1], contacts[1]);
    }
}

int main(void){
    if (!gpu_available()){ printf("test_gpu_rigid: this build has no GPU (make deps): skipped\n"); return 0; }
    g = gpu_create(NULL, 64, 64);
    for (int k = 0; k < 20000 && g && gpu_state(g) == GPU_PENDING; k++) gpu_poll(g);
    if (!g || gpu_state(g) != GPU_READY){ printf("test_gpu_rigid: no GPU (%s): skipped\n", g ? gpu_message(g) : "none"); return 0; }
    printf("GPU: %s\n", gpu_message(g));
    test_basics();
    test_meshes();
    test_pile();
    timings();
    gpu_destroy(g);
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all GPU rigid body tests passed\n");
    return 0;
}
