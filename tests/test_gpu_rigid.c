// test_gpu_rigid.c - the GPU's rigid bodies (gpu/rigid_gpu.c) against known answers and
// against the CPU's (phys/rigid.c).
//
// The same checks as test_physics.c (free fall, bounce, momentum, a resting stack, a tilted
// box landing on a face), each stepped on the GPU; then a pile of balls and boxes dropped in
// the room, on both, compared by what must hold whatever the solver's order: everyone inside
// the room, nobody sunk into anybody, all at rest; and the time per step, CPU vs GPU, from a
// few tens to a few thousand bodies.
//
// Needs a GPU build (the default one): bin/test_gpu_rigid. Without a GPU it says so and passes.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lr.h"
#include "gpu/gpu.h"
#include "gpu/rigid_gpu.h"
#include "test_util.h"
#include "test_phys_util.h"

static Gpu *g;

// a world (no air) whose bodies the GPU steps, or the CPU (gpu = 0)
typedef struct { PhysWorld pw; GpuRigid *gr; } World;

static void world_init(World *w, int gpu){
    phys_init(&w->pw);
    w->pw.air_density = 0;        // no air: the answers above are for a vacuum
    w->pw.rigid.sleep = 0;        // the solver alone (sleeping: test_sleep, settled)
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
    check_near("free fall: velocity after 1 s", w.pw.rigid.bodies[0].vel.y, -981, 0.5);
    check_near("free fall: height after 1 s", w.pw.rigid.bodies[0].pos.y, 1000 - 0.5 * 981, 1.5);
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
    check_near("bounce: peak after the bounce (center)", top, 10 + 0.25 * 400, 4);
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
    check_near("momentum: change / total", v_len(v_sub(p1, p0)) / scale, 0, 1e-3);
    check_near("momentum: sphere a turned around (vx < 0)", w.pw.rigid.bodies[a].vel.x < 0, 1, 0);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    for (int k = 0; k < 3; k++) rigid_add_box(&w.pw.rigid, vec3(278, 30 + 60 * k, 278), vec3(40, 30, 40), quat_identity(), 1);
    int s = rigid_add_sphere(&w.pw.rigid, vec3(278, 180 + 25, 278), 25, 1);
    world_start(&w, 1);
    world_run(&w, 3.0);
    check_near("stack: top box height", w.pw.rigid.bodies[2].pos.y, 150, 1.5);
    check_near("stack: top box drift sideways", r_fabs(w.pw.rigid.bodies[2].pos.x - 278) + r_fabs(w.pw.rigid.bodies[2].pos.z - 278), 0, 1);
    check_near("stack: ball height", w.pw.rigid.bodies[s].pos.y, 205, 1.5);
    check_near("stack: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 1);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    b = rigid_add_box(&w.pw.rigid, vec3(278, 300, 278), vec3(50, 20, 35), quat_axis_angle(vec3(1, 0.3, 0.5), 0.9), 1);
    world_start(&w, 1);
    world_run(&w, 4.0);
    real y = w.pw.rigid.bodies[b].pos.y;
    check_near("tilted box: center height - nearest half size", r_fmin(r_fabs(y - 50), r_fmin(r_fabs(y - 20), r_fabs(y - 35))), 0, 1.5);
    check_near("tilted box: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 1);
    world_free(&w);
}

// as test_physics.c's test_fast_rolling: a ball thrown level along the floor, and one rolling
// on a fixed box's top, roll on (the sweep stopped them at what they touched at once)
static void test_fast_rolling(void){
    printf("on the GPU: fast and touching (a ball thrown along the floor, one rolling on a box)\n");
    real thrown = 0, box_run = 0;
    for (int k = 0; k < 2; k++){
        World w;
        world_init(&w, 1);
        rigid_set_room(&w.pw.rigid, (Aabb){vec3(-5000, 0, -5000), vec3(5000, 1e6f, 5000)});
        Vec3 at = vec3(0, 136 + 15, -600), v = vec3(0, 0, 1500), spin = vec3(0, 0, 0);
        if (k == 1){
            int top = rigid_add_box(&w.pw.rigid, vec3(0, 50, 0), vec3(100, 50, 2000), quat_identity(), 1);
            w.pw.rigid.bodies[top].inv_mass = 0;
            w.pw.rigid.bodies[top].inv_inertia = vec3(0, 0, 0);
            at = vec3(0, 115.2f, -1000);
            v = vec3(0, 0, 1000);
            spin = vec3(1000.0f / 15, 0, 0);
        }
        int b = rigid_add_sphere(&w.pw.rigid, at, 15, 1);
        w.pw.rigid.bodies[b].vel = v;
        w.pw.rigid.bodies[b].angvel = spin;
        w.pw.substeps = 4;
        world_start(&w, 1);
        // the editor's steps: 1/120 s
        for (int s = 0; s < (k ? 60 : 240); s++) phys_step(&w.pw, 1.0f / 120);
        gpu_rigid_sync(w.gr);
        if (k == 0) thrown = w.pw.rigid.bodies[b].pos.z - at.z;
        else box_run = w.pw.rigid.bodies[b].pos.z - at.z;
        world_free(&w);
    }
    check_near("the thrown ball after 2 s: farther than 20 m (units)", thrown > 2000 ? 1 : 0, 1, 0);
    check_near("rolled on the box in 0.5 s (units)", box_run, 500, 40);
    printf("  (thrown %.0f)\n", thrown);
}

static void test_meshes(void){
    printf("on the GPU: hulls landing and stacking, a ball, a hull and a box on a fixed mesh\n");
    static Triangle tri[4096];
    Mat3 turn = quat_to_mat3(quat_axis_angle(v_norm(vec3(0.2, 0.4, 1)), 0.6));
    World w;
    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    int n = box_tris(tri, vec3(50, 20, 35), turn, vec3(10, 30, -5));
    int geom = rigid_geom_hull(&w.pw.rigid, tri, n, 0);
    Xform place = xform_make(vec3(278, 300, 278), quat_to_mat3(quat_axis_angle(vec3(1, 0.3, 0.5), 0.9)).m, 1);
    int b = rigid_add_hull_body(&w.pw.rigid, geom, place, 1);
    world_start(&w, 1);
    world_run(&w, 3.0);
    real y = w.pw.rigid.bodies[b].pos.y;
    check_near("hull (box mesh): centre height - nearest half size", r_fmin(r_fabs(y - 50), r_fmin(r_fabs(y - 20), r_fabs(y - 35))), 0, 1.5);
    Vec3 top = w.pw.rigid.bodies[b].pos;
    Xform on = body_mesh_xform(&w.pw.rigid.bodies[b]);
    on.m[1][3] += 2 * y + 1;
    int b2 = rigid_add_hull_body(&w.pw.rigid, geom, on, 1);
    world_run(&w, 3.0);
    check_near("second hull on the first: height difference", w.pw.rigid.bodies[b2].pos.y - w.pw.rigid.bodies[b].pos.y, 2 * y, 1.5);
    check_near("second hull drift sideways", r_fabs(w.pw.rigid.bodies[b2].pos.x - top.x) + r_fabs(w.pw.rigid.bodies[b2].pos.z - top.z), 0, 2);
    check_near("hulls: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 2);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    n = torus_tris(tri, 60, 20, 32, 16);
    geom = rigid_geom_hull(&w.pw.rigid, tri, n, 0);
    b = rigid_add_hull_body(&w.pw.rigid, geom, xform_make(vec3(278, 200, 278), quat_to_mat3(quat_axis_angle(vec3(1, 0, 0.3), 0.5)).m, 1), 1);
    world_start(&w, 1);
    world_run(&w, 4.0);
    check_near("torus hull lands flat: centre height", w.pw.rigid.bodies[b].pos.y, 20, 1.5);
    world_free(&w);

    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    n = box_tris(tri, vec3(120, 50, 120), mat3_identity(), vec3(0, 50, 0));
    int gm = rigid_geom_mesh(&w.pw.rigid, tri, n);
    rigid_add_mesh_collider(&w.pw.rigid, gm, xform_make(vec3(278, 0, 278), mat3_identity().m, 1));
    int s = rigid_add_sphere(&w.pw.rigid, vec3(220, 300, 278), 25, 1);
    n = box_tris(tri, vec3(30, 20, 30), mat3_identity(), vec3(0, 0, 0));
    geom = rigid_geom_hull(&w.pw.rigid, tri, n, 0);
    int hb = rigid_add_hull_body(&w.pw.rigid, geom, xform_make(vec3(330, 250, 300), quat_to_mat3(quat_axis_angle(vec3(0, 1, 0), 0.4)).m, 1), 1);
    int bx = rigid_add_box(&w.pw.rigid, vec3(278, 400, 220), vec3(20, 20, 20), quat_identity(), 1);
    world_start(&w, 1);
    world_run(&w, 4.0);
    check_near("ball on the mesh: centre height", w.pw.rigid.bodies[s].pos.y, 125, 1.5);
    check_near("hull on the mesh: centre height", w.pw.rigid.bodies[hb].pos.y, 120, 1.5);
    check_near("box on the mesh: centre height", w.pw.rigid.bodies[bx].pos.y, 120, 1.5);
    check_near("on the mesh: fastest point (at rest), units / s", max_speed(&w.pw.rigid), 0, 2);
    world_free(&w);

    // a mesh brought into a world already running on the GPU (its pairs' buffers are made then)
    world_init(&w, 1);
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    s = rigid_add_sphere(&w.pw.rigid, vec3(220, 300, 278), 25, 1);
    world_start(&w, 1);
    world_run(&w, 0.1);
    n = box_tris(tri, vec3(120, 50, 120), mat3_identity(), vec3(0, 50, 0));
    gm = rigid_geom_mesh(&w.pw.rigid, tri, n);
    rigid_add_mesh_collider(&w.pw.rigid, gm, xform_make(vec3(278, 0, 278), mat3_identity().m, 1));
    world_run(&w, 4.0);
    check_near("ball on a mesh added later: centre height", w.pw.rigid.bodies[s].pos.y, 125, 1.5);
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
static double fastest_vy(const RigidWorld *rw){
    double vy = 0;
    for (int i = 0; i < rw->nbodies; i++) vy = fmax(vy, fabs(rw->bodies[i].vel.y));
    return vy;
}
// calm: the least of the fastest vertical speeds seen at 10, 11 and 12 s
static void pile_checks(const char *who, const RigidWorld *rw, real side, double calm){
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
    check_near(what, outside, 0, 0);
    snprintf(what, sizeof(what), "pile (%s): deepest ball-ball overlap, units", who);
    check_near(what, worst, 0, 2.5);
    // settled: nobody falling or bouncing any more (balls may still roll: no rolling resistance,
    // and one may roll off a box and drop: at one of three moments a second apart, none does)
    snprintf(what, sizeof(what), "pile (%s): fastest vertical speed, 10-12 s, units / s", who);
    check_near(what, calm, 0, 10);
}

static void test_pile(void){
    printf("a pile of 60 balls and boxes, CPU and GPU\n");
    for (int gpu = 0; gpu < 2; gpu++){
        World w;
        real side;
        pile(&w, 60, gpu, &side);
        world_run(&w, 10.0);   // (on the CPU a ball rolls off a box at 6.5 s and lands at 8.5)
        double calm = fastest_vy(&w.pw.rigid);
        for (int k = 0; k < 2; k++){ world_run(&w, 1.0); calm = fmin(calm, fastest_vy(&w.pw.rigid)); }
        pile_checks(gpu ? "GPU" : "CPU", &w.pw.rigid, side, calm);
        world_free(&w);
    }
}

// as test_physics.c's: a stack falls asleep, a box dropped on it wakes it, all settle again
static void test_sleep(void){
    printf("on the GPU: sleeping\n");
    World w;
    world_init(&w, 1);
    w.pw.rigid.sleep = 1;
    rigid_set_room(&w.pw.rigid, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    for (int k = 0; k < 3; k++) rigid_add_box(&w.pw.rigid, vec3(278, 30 + 60 * k, 278), vec3(40, 30, 40), quat_identity(), 1);
    world_start(&w, 1);
    world_run(&w, 2.0);
    check_near("asleep after 2 s (of 3)", rigid_sleeping(&w.pw.rigid), 3, 0);
    // all asleep, nothing changes: the steps are not made (P10), and nothing moves
    int skipped0 = gpu_rigid_skipped(w.gr);
    Vec3 top0 = w.pw.rigid.bodies[2].pos;
    world_run(&w, 0.5);
    check_near("all asleep: steps not made (of 120)", gpu_rigid_skipped(w.gr) - skipped0, 120, 2);
    check_near("all asleep: the top box moved", v_len(v_sub(w.pw.rigid.bodies[2].pos, top0)), 0, 0);
    int d = rigid_add_box(&w.pw.rigid, vec3(278, 400, 278), vec3(40, 30, 40), quat_identity(), 1);
    int fewest = 3;
    for (int k = 0; k < 240; k++){
        world_run(&w, 1.0 / 240);
        if (rigid_sleeping(&w.pw.rigid) < fewest) fewest = rigid_sleeping(&w.pw.rigid);
    }
    check_near("the stack woken by the falling box (fewest asleep, of 3)", fewest, 1, 1);
    world_run(&w, 3.0);
    check_near("asleep again 4 s later (of 4)", rigid_sleeping(&w.pw.rigid), 4, 0);
    check_near("the dropped box on top: height", w.pw.rigid.bodies[d].pos.y, 210, 1.5);
    check_near("the stack's top box: height", w.pw.rigid.bodies[2].pos.y, 150, 1.5);
    body_apply_impulse(&w.pw.rigid.bodies[0], vec3(1e5, 0, 0), w.pw.rigid.bodies[0].pos);
    world_run(&w, 1.0 / 240);
    check_near("an impulse wakes a body", w.pw.rigid.bodies[0].sleeping, 0, 0);
    world_free(&w);
}

// a settled pile (8 s), with and without sleeping
static void settled(void){
    printf("time per 1/60 s step of a settled pile of 1024 bodies (after 8 s), sleeping off / on:\n");
    for (int gpu = 0; gpu < 2; gpu++){
        double ms[2];
        int asleep = 0;
        for (int sl = 0; sl < 2; sl++){
            World w;
            real side;
            pile(&w, 1024, gpu, &side);
            w.pw.rigid.sleep = sl;
            for (int s = 0; s < 8 * 60; s++) phys_step(&w.pw, 1.0f / 60);
            gpu_rigid_sync(w.gr);
            if (gpu) gpu_wait(g);
            double t0 = clock_ms();
            for (int s = 0; s < 30; s++) phys_step(&w.pw, 1.0f / 60);
            gpu_rigid_sync(w.gr);
            if (gpu) gpu_wait(g);
            ms[sl] = (clock_ms() - t0) / 30;
            if (sl) asleep = rigid_sleeping(&w.pw.rigid);
            world_free(&w);
        }
        printf("  %s: %7.2f ms / %7.2f ms (%d of 1024 asleep)\n", gpu ? "GPU" : "CPU", ms[0], ms[1], asleep);
    }
}

static void timings(void){
    printf("time per 1/60 s step (4 substeps, 10 iterations), CPU vs GPU, while the pile falls and settles:\n");
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
            double t0 = clock_ms();
            for (int s = 0; s < steps; s++) phys_step(&w.pw, 1.0f / 60);
            gpu_rigid_sync(w.gr);
            if (gpu) gpu_wait(g);
            ms[gpu] = (clock_ms() - t0) / steps;
            contacts[gpu] = w.pw.rigid.ncontacts;
            world_free(&w);
        }
        if (ms[0] < 0) printf("  %5d bodies: CPU (skipped, minutes), GPU %7.2f ms (%d contacts)\n", ns[k], ms[1], contacts[1]);
        else printf("  %5d bodies: CPU %7.2f ms (%d contacts), GPU %7.2f ms (%d contacts)\n", ns[k], ms[0], contacts[0], ms[1], contacts[1]);
    }
}

int main(void){
    if (!gpu_available()){ printf("test_gpu_rigid: this build has no GPU (NOGPU=1): skipped\n"); return 0; }
    g = gpu_create(NULL, 64, 64);
    for (int k = 0; k < 20000 && g && gpu_state(g) == GPU_PENDING; k++) gpu_poll(g);
    if (!g || gpu_state(g) != GPU_READY){ printf("test_gpu_rigid: no GPU (%s): skipped\n", g ? gpu_message(g) : "none"); return 0; }
    printf("GPU: %s\n", gpu_message(g));
    test_basics();
    test_fast_rolling();
    test_meshes();
    test_pile();
    test_sleep();
    timings();
    settled();
    gpu_destroy(g);
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all GPU rigid body tests passed\n");
    return 0;
}
