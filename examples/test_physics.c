// test_physics.c - checks the physics engine against answers known in advance:
//   free fall (y = y0 - g t^2 / 2), bounce height (h' = e^2 h), momentum conservation in
//   collisions, a resting stack that stays put, a box landing flat, a divergence-free flow
//   after the projection, and a cloud that is carried by the wind without losing mass.
//
//   make test_physics && ./bin/test_physics
#include <stdio.h>
#include <stdlib.h>
#include "lr.h"

static int failures = 0;

static void check(const char *what, double got, double want, double tol){
    double err = got - want;
    if (err < 0) err = -err;
    int ok = err <= tol;
    printf("  %-48s %12.4f  (want %.4f +- %.4f)  %s\n", what, got, want, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

// the fastest point of any body (center speed + spin * size), units / s
static double max_speed(const RigidWorld *w){
    double m = 0;
    for (int i = 0; i < w->nbodies; i++){
        const Body *b = &w->bodies[i];
        double s = v_len(b->vel) + v_len(b->angvel) * v_len(b->half);
        if (s > m) m = s;
    }
    return m;
}

static void test_free_fall(void){
    printf("free fall\n");
    RigidWorld w;
    rigid_init(&w);
    w.angular_damping = 0;
    rigid_add_sphere(&w, vec3(0, 1000, 0), 10, 1);
    const real dt = 1.0 / 240;
    for (int i = 0; i < 240; i++) rigid_step(&w, dt);
    // semi-implicit Euler is exact for the velocity and off by g dt t / 2 for the position
    check("velocity after 1 s", w.bodies[0].vel.y, -981, 0.05);
    check("height after 1 s", w.bodies[0].pos.y, 1000 - 0.5 * 981, 981 * dt * 0.5 + 0.1);
    rigid_free(&w);
}

static void test_bounce(void){
    printf("bounce (restitution 0.5: the ball comes back to 1/4 of its height)\n");
    RigidWorld w;
    rigid_init(&w);
    w.angular_damping = 0;
    rigid_set_room(&w, (Aabb){vec3(-1000, 0, -1000), vec3(1000, 5000, 1000)});
    w.wall_restitution = 0.5;
    int i = rigid_add_sphere(&w, vec3(0, 410, 0), 10, 1);
    w.bodies[i].restitution = 0.5;
    const real dt = 1.0 / 1000;
    real top = 0;
    int bounced = 0;
    for (int k = 0; k < 3000; k++){
        real vy = w.bodies[i].vel.y;
        rigid_step(&w, dt);
        if (vy < 0 && w.bodies[i].vel.y > 0) bounced = 1;
        if (bounced && w.bodies[i].pos.y > top) top = w.bodies[i].pos.y;
        if (bounced && w.bodies[i].vel.y < 0) break;
    }
    // drop height 400 (center 410, radius 10); comes back to 100 above the floor
    check("peak after the bounce (center)", top, 10 + 0.25 * 400, 4);
    rigid_free(&w);
}

static void test_momentum(void){
    printf("momentum (no gravity, spheres and boxes colliding)\n");
    RigidWorld w;
    rigid_init(&w);
    w.gravity = vec3(0, 0, 0);
    w.angular_damping = 0;
    int a = rigid_add_sphere(&w, vec3(-100, 0, 0), 20, 1);
    int b = rigid_add_sphere(&w, vec3(100, 5, 0), 30, 1);
    int c = rigid_add_box(&w, vec3(0, -200, 3), vec3(20, 30, 25), quat_axis_angle(vec3(1, 1, 0), 0.6), 2);
    int d = rigid_add_box(&w, vec3(0, 200, -4), vec3(25, 15, 20), quat_axis_angle(vec3(0, 1, 1), -0.4), 0.5);
    w.bodies[a].vel = vec3(200, 0, 0);
    w.bodies[b].vel = vec3(-150, 0, 0);
    w.bodies[c].vel = vec3(0, 300, 0);
    w.bodies[d].vel = vec3(0, -250, 0);
    w.bodies[a].restitution = w.bodies[b].restitution = 1;
    Vec3 p0 = rigid_momentum(&w);
    for (int k = 0; k < 600; k++) rigid_step(&w, 1.0 / 600);
    Vec3 p1 = rigid_momentum(&w);
    real scale = v_len(p0) + 1e5;
    check("momentum x change / total", (p1.x - p0.x) / scale, 0, 1e-4);
    check("momentum y change / total", (p1.y - p0.y) / scale, 0, 1e-4);
    check("momentum z change / total", (p1.z - p0.z) / scale, 0, 1e-4);
    // collisions happened: the first two spheres turned around
    check("sphere a turned around (vx < 0)", w.bodies[a].vel.x < 0, 1, 0);
    rigid_free(&w);
}

static void test_stack(void){
    printf("resting stack (3 boxes and a ball on top stay where they are)\n");
    RigidWorld w;
    rigid_init(&w);
    rigid_set_room(&w, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    for (int k = 0; k < 3; k++) rigid_add_box(&w, vec3(278, 30 + 60 * k, 278), vec3(40, 30, 40), quat_identity(), 1);
    int s = rigid_add_sphere(&w, vec3(278, 180 + 25, 278), 25, 1);
    for (int k = 0; k < 4 * 240 * 3; k++) rigid_step(&w, 1.0 / 960);
    check("top box height", w.bodies[2].pos.y, 150, 1.5);
    check("top box drift sideways", r_fabs(w.bodies[2].pos.x - 278) + r_fabs(w.bodies[2].pos.z - 278), 0, 1);
    check("ball height", w.bodies[s].pos.y, 205, 1.5);
    check("fastest point (at rest), units / s", max_speed(&w), 0, 1);
    rigid_free(&w);
}

static void test_box_lands_flat(void){
    printf("a tilted box falls and settles on a face\n");
    RigidWorld w;
    rigid_init(&w);
    rigid_set_room(&w, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    int b = rigid_add_box(&w, vec3(278, 300, 278), vec3(50, 20, 35), quat_axis_angle(vec3(1, 0.3, 0.5), 0.9), 1);
    for (int k = 0; k < 4 * 240 * 4; k++) rigid_step(&w, 1.0 / 960);
    // resting on a face: the center is one of the half sizes above the floor
    real y = w.bodies[b].pos.y;
    real best = r_fmin(r_fabs(y - 50), r_fmin(r_fabs(y - 20), r_fabs(y - 35)));
    check("center height - nearest half size", best, 0, 1.5);
    check("fastest point (at rest), units / s", max_speed(&w), 0, 1);
    rigid_free(&w);
}

static void test_fluid(void){
    printf("fluid\n");
    Aabb room = {vec3(0, 0, 0), vec3(555, 555, 555)};
    Volume *v = volume_create(48, 48, 48, room);
    // a ball of cloud on the left
    for (int z = 0; z < 48; z++)
        for (int y = 0; y < 48; y++)
            for (int x = 0; x < 48; x++){
                Vec3 c = volume_voxel_center(v, x, y, z);
                v->density[((size_t)z * 48 + y) * 48 + x] = v_len(v_sub(c, vec3(150, 278, 278))) < 80 ? 1 : 0;
            }
    volume_update(v);
    Fluid *f = fluid_create(room, 555.0 / 24, v);
    f->open = 0;
    f->pressure_cycles = 20;
    f->vorticity = 0;
    // a swirl: a strongly divergent kick, then the projection must remove the divergence
    fluid_add_impulse(f, vec3(278, 278, 278), 200, vec3(300, 0, 0), 1);
    real m0 = fluid_total_density(f);
    fluid_step(f, 1.0 / 60);
    // relative to the velocity scale / cell size, the divergence left must be small
    check("max |div u| * h / |u|max after projection", f->max_divergence * f->h / 300, 0, 0.02);
    // a steady wind carries the cloud right without changing its amount much
    Fluid *g = fluid_create(room, 555.0 / 24, v);
    g->open = FLUID_OPEN_XLO | FLUID_OPEN_XHI;
    g->vorticity = 0;
    g->wind = vec3(100, 0, 0);
    g->wind_rate = 1000;   // the air takes the wind's speed at once
    for (int k = 0; k < 60; k++) fluid_step(g, 1.0 / 60);
    // center of mass of the cloud
    double sx = 0, sm = 0;
    for (int z = 0; z < 48; z++)
        for (int y = 0; y < 48; y++)
            for (int x = 0; x < 48; x++){
                double d = v->density[((size_t)z * 48 + y) * 48 + x];
                sx += d * volume_voxel_center(v, x, y, z).x;
                sm += d;
            }
    check("cloud center x after 1 s of 100/s wind", sx / sm, 250, 15);
    check("cloud amount kept (ratio)", fluid_total_density(g) / m0, 1, 0.1);
    fluid_free(f);
    fluid_free(g);
    volume_free(v);

    // an expansion (explosion) in open air: the flow leaving a sphere of radius R whose
    // divergence is s (1 - (r/R)^2) is Q = s 4 pi R^3 (2/15), so at distance d the air
    // moves outwards at Q / (4 pi d^2)
    Fluid *e = fluid_create(room, 555.0 / 48, NULL);
    e->open = 63;                 // every face open
    e->vorticity = 0;
    e->wind_rate = 0;
    e->pressure_cycles = 10;
    fluid_add_expansion(e, vec3(278, 278, 278), 100, 30, 1);
    fluid_step(e, 1.0 / 60);
    check("multigrid: divergence left / rate (10 V-cycles)", e->max_divergence / 30, 0, 0.01);
    double want = 30.0 * 100 * 100 * 100 * (2.0 / 15) / (180.0 * 180.0);
    double up = fluid_velocity(e, vec3(278, 278 + 180, 278)).y, side = fluid_velocity(e, vec3(278 - 180, 278, 278)).x;
    check("expansion: outward speed at 180 (up) / theory", up / want, 1, 0.2);
    check("expansion: outward speed at 180 (side) / theory", -side / want, 1, 0.2);
    fluid_free(e);
}

int main(void){
    test_free_fall();
    test_bounce();
    test_momentum();
    test_stack();
    test_box_lands_flat();
    test_fluid();
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all physics tests passed\n");
    return 0;
}
