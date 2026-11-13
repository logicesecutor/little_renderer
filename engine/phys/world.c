#define _POSIX_C_SOURCE 200809L
#include "phys/world.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

void phys_init(PhysWorld *pw){
    memset(pw, 0, sizeof(*pw));
    rigid_init(&pw->rigid);
    pw->air_density = 0.0012f;
    pw->drag_sphere = 0.47f;
    pw->drag_box = 1.05f;
    pw->substeps = 4;
}

void phys_free(PhysWorld *pw){
    rigid_free(&pw->rigid);
    fluid_free(pw->fluid);
    pw->fluid = NULL;
}

// radius of a sphere around the body
static real bound_radius(const Body *b){
    return b->shape == BODY_SPHERE ? b->half.x : v_len(b->half);
}

Vec3 phys_air_velocity_around(const PhysWorld *pw, int i){
    if (!pw->fluid) return vec3(0, 0, 0);
    if (pw->air_step) return i < pw->nair_around ? pw->air_around[i] : vec3(0, 0, 0);
    const Body *b = &pw->rigid.bodies[i];
    // just outside the cells the body covers: there the air is free
    real r = (b->shape == BODY_SPHERE ? b->half.x : (b->half.x + b->half.y + b->half.z) / 3) * 1.15f + pw->fluid->h;
    static const real k = 0.57735027f;   // 1 / sqrt(3)
    const Vec3 dirs[14] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
                           {k, k, k}, {k, k, -k}, {k, -k, k}, {k, -k, -k}, {-k, k, k}, {-k, k, -k}, {-k, -k, k}, {-k, -k, -k}};
    Vec3 sum = vec3(0, 0, 0);
    for (int d = 0; d < 14; d++) sum = v_add(sum, fluid_velocity(pw->fluid, v_add(b->pos, v_scale(dirs[d], r))));
    return v_scale(sum, 1.0f / 14);
}

// the area the body shows to a flow along unit direction d
static real frontal_area(const Body *b, Vec3 d){
    if (b->shape == BODY_SPHERE) return (real)PI * b->half.x * b->half.x;
    // a box's shadow: each face pair contributes its area times |cos| of its normal with d
    Mat3 r = quat_to_mat3(b->rot);
    return 4 * (b->half.y * b->half.z * r_fabs(v_dot(mat3_col(r, 0), d)) +
                b->half.x * b->half.z * r_fabs(v_dot(mat3_col(r, 1), d)) +
                b->half.x * b->half.y * r_fabs(v_dot(mat3_col(r, 2), d)));
}

// drag only (the buoyancy is constant)
static Vec3 drag_force(const PhysWorld *pw, int i, real dt){
    const Body *b = &pw->rigid.bodies[i];
    if (b->inv_mass == 0) return vec3(0, 0, 0);
    Vec3 du = v_sub(phys_air_velocity_around(pw, i), b->vel);
    real s = v_len(du);
    if (s < 1e-4f) return vec3(0, 0, 0);
    Vec3 d = v_scale(du, 1.0f / s);
    real cd = b->shape == BODY_SPHERE ? pw->drag_sphere : pw->drag_box;
    real f = 0.5f * pw->air_density * cd * frontal_area(b, d) * s * s;
    // never more than what brings the body to the air's speed in one step (the explicit
    // force would overshoot for very light bodies in a strong wind)
    real fmax = s * b->mass / dt;
    if (f > fmax) f = fmax;
    return v_scale(d, f);
}

static Vec3 buoyancy_force(const PhysWorld *pw, int i){
    const Body *b = &pw->rigid.bodies[i];
    return v_scale(pw->rigid.gravity, -pw->air_density * b->volume);
}

Vec3 phys_air_force(const PhysWorld *pw, int i){
    return v_add(drag_force(pw, i, 1.0f / 60), buoyancy_force(pw, i));
}

// the cells inside each body become obstacles moving with it
static void mark_solids(PhysWorld *pw){
    Fluid *f = pw->fluid;
    fluid_clear_solids(f);
    for (int i = 0; i < pw->rigid.nbodies; i++){
        const Body *b = &pw->rigid.bodies[i];
        real r = bound_radius(b);
        int x0 = (int)r_floor((b->pos.x - r - f->bounds.lo.x) / f->h), x1 = (int)r_floor((b->pos.x + r - f->bounds.lo.x) / f->h);
        int y0 = (int)r_floor((b->pos.y - r - f->bounds.lo.y) / f->h), y1 = (int)r_floor((b->pos.y + r - f->bounds.lo.y) / f->h);
        int z0 = (int)r_floor((b->pos.z - r - f->bounds.lo.z) / f->h), z1 = (int)r_floor((b->pos.z + r - f->bounds.lo.z) / f->h);
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (z0 < 0) z0 = 0;
        if (x1 >= f->nx) x1 = f->nx - 1;
        if (y1 >= f->ny) y1 = f->ny - 1;
        if (z1 >= f->nz) z1 = f->nz - 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++){
                    Vec3 c = fluid_cell_center(f, x, y, z);
                    if (!body_contains(b, c)) continue;
                    size_t k = fluid_idx(f, x, y, z);
                    Vec3 v = body_point_velocity(b, c);
                    f->solid[k] = 1;
                    f->su[k] = (float)v.x;
                    f->sv[k] = (float)v.y;
                    f->sw[k] = (float)v.z;
                }
    }
}

void phys_step(PhysWorld *pw, real dt){
    if (dt <= 0) return;
    RigidWorld *w = &pw->rigid;
    double t0 = now_ms();
    // the air's forces, from the air as it is now, held for the substeps
    int n = w->nbodies;
    Vec3 forces[256];
    Vec3 *fs = n <= 256 ? forces : malloc((size_t)n * sizeof(Vec3));
    if (!fs){ perror("malloc"); return; }
    int steps = pw->substeps > 0 ? pw->substeps : 1;
    real h = dt / steps;
    for (int i = 0; i < n; i++) fs[i] = v_add(drag_force(pw, i, h), buoyancy_force(pw, i));
    if (!pw->rigid_step || !pw->rigid_step(pw->rigid_user, pw, dt, steps, fs)) rigid_steps(w, dt, steps, fs);
    if (fs != forces) free(fs);
    double t1 = now_ms();
    if (pw->fluid && pw->air_step) pw->air_step(pw->air_user, pw, dt);
    else if (pw->fluid){
        mark_solids(pw);
        fluid_step(pw->fluid, dt);
    }
    pw->ms_rigid = t1 - t0;
    pw->ms_fluid = now_ms() - t1;
    pw->time += dt;
}

void phys_blast(PhysWorld *pw, Vec3 center, real radius, real speed, real impulse){
    // the gas of the explosion: the air expands from the inner half of the sphere for
    // 0.15 s. A divergence s (1 - (r/R)^2) pushes the air out at s R (2/15) at the core's
    // edge (see test_physics.c): s is chosen so that this is `speed`.
    if (pw->fluid){
        real core = radius * 0.5f;
        fluid_add_expansion(pw->fluid, center, core, speed * 7.5f / core, 0.15f);
    }
    for (int i = 0; i < pw->rigid.nbodies; i++){
        Body *b = &pw->rigid.bodies[i];
        if (b->inv_mass == 0) continue;
        Vec3 d = v_sub(b->pos, center);
        real l = v_len(d);
        if (l >= radius) continue;
        Vec3 dir = l > 1e-3f ? v_scale(d, 1.0f / l) : vec3(0, 1, 0);
        real fall = 1 - l / radius;
        real dv = impulse * frontal_area(b, dir) * fall * b->inv_mass;
        if (dv > speed * fall) dv = speed * fall;
        b->vel = v_add(b->vel, v_scale(dir, dv));
        b->dirty = 1;
        // a little spin, as the push is never exactly through the center of mass
        Vec3 off = v_cross(dir, vec3(0.3f, 1, 0.2f));
        b->angvel = v_add(b->angvel, v_scale(off, dv * 0.5f / (bound_radius(b) + 1)));
    }
}

// ---- the scene ------------------------------------------------------------------------------

// the 6 faces of a box as quads with u x v pointing outwards
static void box_quads(const Body *b, int material, Primitive *out){
    Mat3 r = quat_to_mat3(b->rot);
    int m = 0;
    for (int k = 0; k < 3; k++){
        int i = (k + 1) % 3, j = (k + 2) % 3;
        real hk = k == 0 ? b->half.x : (k == 1 ? b->half.y : b->half.z);
        real hi = i == 0 ? b->half.x : (i == 1 ? b->half.y : b->half.z);
        real hj = j == 0 ? b->half.x : (j == 1 ? b->half.y : b->half.z);
        Vec3 ek = mat3_col(r, k), u = v_scale(mat3_col(r, i), 2 * hi), v = v_scale(mat3_col(r, j), 2 * hj);
        for (int side = -1; side <= 1; side += 2){
            Vec3 c = v_add(b->pos, v_scale(ek, side * hk));
            Vec3 q = v_sub(c, v_scale(v_add(u, v), 0.5f));
            // e_i x e_j = e_k: (u, v) faces +e_k; swapped, -e_k
            out[m++] = side > 0 ? make_quad(q, u, v, material) : make_quad(q, v, u, material);
        }
    }
}

int phys_add_mesh_to_scene(PhysWorld *pw, int i, Scene *s, int mesh, int material){
    Body *b = &pw->rigid.bodies[i];
    if (!b->g || mesh < 0 || mesh >= s->nmeshes) return -1;
    b->material = material;
    b->inst = scene_add_instance(s, mesh, body_mesh_xform(b), material);
    return b->inst;
}

// where a ball or a box puts its unit shape (scene.h: INST_SPHERE, INST_BOX)
static Xform shape_xform(const Body *b){
    return xform_axes(quat_to_mat3(b->rot), b->half, b->pos);
}

int phys_add_to_scene(PhysWorld *pw, int i, Scene *s, int material){
    Body *b = &pw->rigid.bodies[i];
    b->material = material;
    if (b->g) return -1;                // a mesh body: phys_add_mesh_to_scene
    b->inst = scene_add_instance(s, b->shape == BODY_SPHERE ? INST_SPHERE : INST_BOX, shape_xform(b), material);
    return b->inst;
}

int phys_add_to_scene_prims(PhysWorld *pw, int i, Scene *s, int material){
    Body *b = &pw->rigid.bodies[i];
    b->material = material;
    if (b->g) return -1;
    if (b->shape == BODY_SPHERE){
        b->prim = scene_add_prim(s, make_sphere(b->pos, b->half.x, material));
    } else {
        Primitive q[6];
        box_quads(b, material, q);
        b->prim = scene_add_prim(s, q[0]);
        for (int k = 1; k < 6; k++) scene_add_prim(s, q[k]);
    }
    return b->prim;
}

void phys_sync_scene(const PhysWorld *pw, Scene *s){
    for (int i = 0; i < pw->rigid.nbodies; i++){
        const Body *b = &pw->rigid.bodies[i];
        if (b->inst >= 0 && b->inv_mass > 0 && b->inst < s->ninsts)
            scene_move_instance(s, b->inst, b->g ? body_mesh_xform(b) : shape_xform(b));
        if (b->prim < 0) continue;
        if (b->shape == BODY_SPHERE){
            s->prims[b->prim] = make_sphere(b->pos, b->half.x, b->material);
        } else {
            Primitive q[6];
            box_quads(b, b->material, q);
            for (int k = 0; k < 6; k++) s->prims[b->prim + k] = q[k];
        }
    }
}
