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

// ---- mesh bodies (phys/convex.h) ----

// the 12 triangles of a box (half sizes h), turned by r and moved to c, in a growing array
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

static void test_hulls(void){
    printf("mesh bodies: hulls (mass, landing, stacking), a mesh collider, the air's view of them\n");
    static Triangle tri[4096];
    // a box mesh, turned and moved in its own frame: the hull's mass and axes must undo that
    Mat3 turn = quat_to_mat3(quat_axis_angle(v_norm(vec3(0.2, 0.4, 1)), 0.6));
    Vec3 h = vec3(50, 20, 35);
    int n = box_tris(tri, h, turn, vec3(10, 30, -5));
    RigidWorld w;
    rigid_init(&w);
    rigid_set_room(&w, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    int g = rigid_geom_hull(&w, tri, n, 0);
    check("hull made", g >= 0, 1, 0);
    if (g < 0){ rigid_free(&w); return; }
    const RigidGeom *G = w.geoms[g];
    check("hull volume / box volume", G->volume / (8 * h.x * h.y * h.z), 1, 1e-3);
    check("hull centre of mass (moved by 10,30,-5)", v_len(v_sub(G->com, vec3(10, 30, -5))), 0, 1e-2);
    // principal moments of a box with density 1: m/3 (b^2 + c^2)
    real m = 8 * h.x * h.y * h.z, mo[3] = {m / 3 * (h.y * h.y + h.z * h.z), m / 3 * (h.x * h.x + h.z * h.z), m / 3 * (h.x * h.x + h.y * h.y)};
    real got[3] = {G->inertia.x, G->inertia.y, G->inertia.z}, err = 0;
    for (int i = 0; i < 3; i++){
        real best = 1e30f;
        for (int k = 0; k < 3; k++) best = r_fmin(best, r_fabs(got[i] - mo[k]) / mo[k]);
        err = r_fmax(err, best);
    }
    check("principal moments vs the box's (worst rel. error)", err, 0, 2e-3);
    // it falls tilted and lands on a face, like a box would
    Xform place = xform_make(vec3(278, 300, 278), quat_to_mat3(quat_axis_angle(vec3(1, 0.3, 0.5), 0.9)).m, 1);
    int b = rigid_add_hull_body(&w, g, place, 1);
    for (int k = 0; k < 4 * 240 * 3; k++) rigid_step(&w, 1.0 / 960);
    real y = w.bodies[b].pos.y;
    real best = r_fmin(r_fabs(y - 50), r_fmin(r_fabs(y - 20), r_fabs(y - 35)));
    check("hull at rest: centre height - nearest half size", best, 0, 1.5);
    // a second one, dropped flat on top of it
    Vec3 top = w.bodies[b].pos;
    Mat3 rb = quat_to_mat3(w.bodies[b].rot);
    Xform on = body_mesh_xform(&w.bodies[b]);
    on.m[1][3] += 2 * y + 1;
    int b2 = rigid_add_hull_body(&w, g, on, 1);
    (void)rb;
    for (int k = 0; k < 4 * 240 * 3; k++) rigid_step(&w, 1.0 / 960);
    check("second hull rests on the first: height difference", w.bodies[b2].pos.y - w.bodies[b].pos.y, 2 * y, 1.5);
    check("second hull drift sideways", r_fabs(w.bodies[b2].pos.x - top.x) + r_fabs(w.bodies[b2].pos.z - top.z), 0, 2);
    check("fastest point (at rest), units / s", max_speed(&w), 0, 2);
    // the air: the hull's cells inside it
    int inside = 0, tot = 0;
    for (int k = 0; k < 2000; k++){
        Vec3 p = vec3((k * 37 % 101) / 100.0f - 0.5f, (k * 53 % 103) / 102.0f - 0.5f, (k * 71 % 107) / 106.0f - 0.5f);
        Vec3 lp = vec3(p.x * 2 * h.x * 0.9f, p.y * 2 * h.y * 0.9f, p.z * 2 * h.z * 0.9f);   // inside the box, a bit in
        Vec3 wp = xform_point(&place, v_add(mat3_mul_vec(turn, lp), vec3(10, 30, -5)));
        (void)wp;
        Body probe = w.bodies[b];
        Xform xf = body_mesh_xform(&probe);
        inside += body_contains(&probe, xform_point(&xf, v_add(mat3_mul_vec(turn, lp), vec3(10, 30, -5))));
        tot++;
    }
    check("points of the mesh inside the body (air cells)", (double)inside / tot, 1, 0.02);
    rigid_free(&w);

    // a torus: its hull is a puck that lands flat (centre at the tube radius)
    rigid_init(&w);
    rigid_set_room(&w, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    n = torus_tris(tri, 60, 20, 32, 16);
    g = rigid_geom_hull(&w, tri, n, 0);
    b = rigid_add_hull_body(&w, g, xform_make(vec3(278, 200, 278), quat_to_mat3(quat_axis_angle(vec3(1, 0, 0.3), 0.5)).m, 1), 1);
    for (int k = 0; k < 4 * 240 * 4; k++) rigid_step(&w, 1.0 / 960);
    check("torus hull lands flat: centre height", w.bodies[b].pos.y, 20, 1.5);
    check("fastest point (at rest), units / s", max_speed(&w), 0, 2);
    rigid_free(&w);

    // a fixed mesh (a block of 12 triangles, top at y = 100) with a ball and a hull on it,
    // and a torus mesh: its inside as the air sees it has the torus's volume
    rigid_init(&w);
    rigid_set_room(&w, (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)});
    n = box_tris(tri, vec3(120, 50, 120), mat3_identity(), vec3(0, 50, 0));
    int gm = rigid_geom_mesh(&w, tri, n);
    rigid_add_mesh_collider(&w, gm, xform_make(vec3(278, 0, 278), mat3_identity().m, 1));
    int s = rigid_add_sphere(&w, vec3(220, 300, 278), 25, 1);
    n = box_tris(tri, vec3(30, 20, 30), mat3_identity(), vec3(0, 0, 0));
    g = rigid_geom_hull(&w, tri, n, 0);
    int hb = rigid_add_hull_body(&w, g, xform_make(vec3(330, 250, 300), quat_to_mat3(quat_axis_angle(vec3(0, 1, 0), 0.4)).m, 1), 1);
    int bx = rigid_add_box(&w, vec3(278, 400, 220), vec3(20, 20, 20), quat_identity(), 1);
    for (int k = 0; k < 4 * 240 * 4; k++) rigid_step(&w, 1.0 / 960);
    check("ball on the mesh: centre height", w.bodies[s].pos.y, 125, 1.5);
    check("hull on the mesh: centre height", w.bodies[hb].pos.y, 120, 1.5);
    check("box on the mesh: centre height", w.bodies[bx].pos.y, 120, 1.5);
    check("fastest point (at rest), units / s", max_speed(&w), 0, 2);
    rigid_free(&w);

    rigid_init(&w);
    n = torus_tris(tri, 60, 20, 48, 24);
    gm = rigid_geom_mesh(&w, tri, n);
    const RigidGeom *T = w.geoms[gm];
    int cells = 0;
    for (int i = 0; i < OCC_WORDS; i++) cells += __builtin_popcount(T->occ[i]);
    real cell = 8 * T->half.x * T->half.y * T->half.z / (OCC_RES * OCC_RES * OCC_RES);
    check("torus inside / 2 pi^2 R r^2", cells * cell / (2 * PI * PI * 60 * 400), 1, 0.05);
    rigid_free(&w);

    // two hulls collide in empty space: momentum is kept
    rigid_init(&w);
    w.gravity = vec3(0, 0, 0);
    w.angular_damping = 0;
    n = box_tris(tri, vec3(30, 20, 25), turn, vec3(0, 0, 0));
    g = rigid_geom_hull(&w, tri, n, 0);
    int a1 = rigid_add_hull_body(&w, g, xform_make(vec3(0, 0, 0), mat3_identity().m, 1), 1);
    int a2 = rigid_add_hull_body(&w, g, xform_make(vec3(120, 15, 5), turn.m, 1), 2);
    w.bodies[a1].vel = vec3(200, 0, 0);
    w.bodies[a2].vel = vec3(-100, 0, 0);
    Vec3 p0 = rigid_momentum(&w);
    for (int k = 0; k < 960; k++) rigid_step(&w, 1.0 / 960);
    Vec3 p1 = rigid_momentum(&w);
    real scale = v_len(p0) + 1e-6f;
    check("hulls: momentum change / total", v_len(v_sub(p1, p0)) / scale, 0, 1e-3);
    check("hulls: they met (first one slowed down)", w.bodies[a1].vel.x < 150, 1, 0);
    rigid_free(&w);
}

int main(void){
    test_free_fall();
    test_bounce();
    test_momentum();
    test_stack();
    test_box_lands_flat();
    test_hulls();
    test_fluid();
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all physics tests passed\n");
    return 0;
}
