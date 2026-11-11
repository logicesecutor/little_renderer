#include "app/simdemo.h"

#include <stdio.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/volume.h"

#define ROOM 555.0
#define MAX_PHYS_DT (1.0 / 60.0)

static int add_mat(Scene *s, MatType t, Vec3 albedo, real ior){
    Material m = {t, albedo, vec3(0, 0, 0), ior};
    return scene_add_material(s, m);
}

static real rnd(Sim *sim, real lo, real hi){ return lo + (hi - lo) * rng_real(&sim->rng); }

static Spawn *new_spawn(Sim *sim, double t, BodyShape shape, Vec3 pos, Vec3 half, real density, int material){
    if (sim->nspawns >= SIM_MAX_SPAWNS) return NULL;
    Spawn *p = &sim->spawns[sim->nspawns++];
    memset(p, 0, sizeof(*p));
    p->t = t; p->shape = shape; p->pos = pos; p->half = half; p->density = density; p->material = material;
    p->rot = quat_identity();
    p->restitution = 0.4f;
    p->friction = 0.5f;
    return p;
}

static void sort_spawns(Sim *sim){
    for (int i = 1; i < sim->nspawns; i++)
        for (int j = i; j > 0 && sim->spawns[j].t < sim->spawns[j - 1].t; j--){
            Spawn t = sim->spawns[j]; sim->spawns[j] = sim->spawns[j - 1]; sim->spawns[j - 1] = t;
        }
}

// a random spot over the floor, away from the walls
static Vec3 spot(Sim *sim, real margin, real y){
    return vec3(rnd(sim, margin, ROOM - margin), y, rnd(sim, margin + 40, ROOM - margin));
}

static Quat random_rot(Sim *sim){
    Vec3 axis = v_norm(vec3(rnd(sim, -1, 1), rnd(sim, -1, 1), rnd(sim, -1, 1)));
    return quat_axis_angle(axis, rnd(sim, 0, (real)PI));
}

int sim_setup(Sim *sim, const Options *o, Scene *s){
    memset(sim, 0, sizeof(*sim));
#ifdef _OPENMP
    if (o->threads > 0) omp_set_num_threads(o->threads);
#endif
    rng_seed(&sim->rng, 0x51AB1E5EEDULL + (uint64_t)o->sim.seed * 0x9E3779B97F4A7C15ULL, 7);
    sim->blast_t = -1;
    sim->max_dt = MAX_PHYS_DT;

    // the room, nothing in it (the fog, if asked, stays a separate uniform medium)
    SceneOptions so = o->scene;
    so.kind = SCENE_EMPTY;
    so.cloud = NULL;
    if (scene_cornell_box(s, &so) != 0) return 1;
    if (o->set_cam) s->cam_from = o->cam;
    if (o->set_look) s->cam_at = o->look;
    if (o->set_fov) s->cam_vfov_deg = o->fov;

    // materials
    int glass = add_mat(s, MAT_GLASS, vec3(1, 1, 1), 1.5);
    int mirror = add_mat(s, MAT_MIRROR, vec3(0.95, 0.93, 0.90), 1);
    int ball_cols[3] = {add_mat(s, MAT_DIFFUSE, vec3(0.85, 0.42, 0.12), 1),     // orange
                        add_mat(s, MAT_DIFFUSE, vec3(0.12, 0.30, 0.75), 1),     // blue
                        add_mat(s, MAT_DIFFUSE, vec3(0.80, 0.70, 0.15), 1)};    // yellow
    int box_cols[2] = {add_mat(s, MAT_DIFFUSE, vec3(0.73, 0.73, 0.73), 1),      // white
                       add_mat(s, MAT_DIFFUSE, vec3(0.55, 0.35, 0.20), 1)};     // wood
    int balloon_cols[3] = {add_mat(s, MAT_DIFFUSE, vec3(0.90, 0.15, 0.35), 1), // pink-red
                           add_mat(s, MAT_DIFFUSE, vec3(0.10, 0.65, 0.60), 1), // teal
                           add_mat(s, MAT_DIFFUSE, vec3(0.55, 0.25, 0.80), 1)};// purple
    sim->mat_drop = mirror;

    // the cloud: one grid over the whole room, so the air can carry it anywhere
    CloudParams cp = o->cloud;
    int res = cp.resolution > 0 ? cp.resolution : 128;
    Aabb room = {vec3(0, 0, 0), vec3(ROOM, ROOM, ROOM)};
    Volume *v = volume_create(res, res, res, room);
    if (!v){ fprintf(stderr, "out of memory for the cloud\n"); return 1; }
    Aabb cbox = o->sim.demo == SIM_BLAST ? (Aabb){vec3(80, 150, 110), vec3(475, 400, 450)}
                                         : (Aabb){vec3(80, 220, 110), vec3(475, 440, 450)};
    volume_add_cloud(v, cbox, &cp);
    if (scene_add_volume(s, v) < 0) return 1;
    sim->cloud = v;

    // physics
    PhysWorld *pw = &sim->pw;
    phys_init(pw);
    rigid_set_room(&pw->rigid, room);
    pw->fluid = fluid_create(room, ROOM / o->sim.fluid_res, v);
    if (!pw->fluid){ fprintf(stderr, "out of memory for the air\n"); return 1; }
    pw->fluid->wind = o->sim.wind;
    pw->fluid->buoyancy = o->sim.buoyancy;
    pw->fluid->vorticity = o->sim.vorticity;
    pw->fluid->open = FLUID_OPEN_ZLO;   // the Cornell box has no front wall

    // the bodies
    const SimOptions *so2 = &o->sim;
    if (o->sim.demo == SIM_DROP){
        // heavy things fall from under the ceiling, one after another, through the cloud
        // interleaved: ball, ball, box, ball, ball, box ...
        int n = so2->balls + so2->boxes;
        int nb = 0, nx = 0;
        for (int k = 0; k < n; k++){
            double t = 0.25 + 0.4 * k;
            int want_box = (k % 3 == 2 && nx < so2->boxes) || nb >= so2->balls;
            if (want_box){
                Vec3 half = vec3(rnd(sim, 22, 38), rnd(sim, 22, 38), rnd(sim, 22, 38));
                Spawn *p = new_spawn(sim, t, BODY_BOX, spot(sim, 90, 470), half, 0.7f, box_cols[nx % 2]);
                if (p){ p->rot = random_rot(sim); p->vel = vec3(rnd(sim, -60, 60), 0, rnd(sim, -60, 60)); }
                nx++;
            } else {
                real r = rnd(sim, 26, 42);
                int kind = nb % 5;   // glass, steel (mirror), orange, blue, yellow
                int mat = kind == 0 ? glass : (kind == 1 ? mirror : ball_cols[(kind - 2) % 3]);
                real dens = kind == 0 ? 2.5f : (kind == 1 ? 7.8f : 1.0f);
                Spawn *p = new_spawn(sim, t, BODY_SPHERE, spot(sim, 80, 480), vec3(r, r, r), dens, mat);
                if (p){ p->vel = vec3(rnd(sim, -60, 60), 0, rnd(sim, -60, 60)); p->restitution = kind == 0 ? 0.6f : 0.45f; }
                nb++;
            }
        }
        // balloons (a bit lighter than air) start on the floor and rise through the cloud
        // (a little lighter than air, so they rise slowly: about 1 m/s). At the back, away
        // from under the lamp, so that they do not shade the room once at the ceiling.
        for (int k = 0; k < so2->balloons; k++){
            real r = rnd(sim, 28, 36);
            Vec3 at = vec3(k % 2 ? rnd(sim, 70, 180) : rnd(sim, 375, 485), r + 1, rnd(sim, 330, 480));
            Spawn *p = new_spawn(sim, 0, BODY_SPHERE, at, vec3(r, r, r), 0.00105f, balloon_cols[k % 3]);
            if (p){ p->restitution = 0.7f; p->friction = 0.3f; }
        }
    } else {
        // everything starts at rest on the floor, under the cloud; the explosion at 1 s
        int nb = 0;
        for (int k = 0; k < so2->balls; k++){
            real r = rnd(sim, 26, 40);
            int kind = nb++ % 5;
            int mat = kind == 0 ? glass : (kind == 1 ? mirror : ball_cols[(kind - 2) % 3]);
            real dens = kind == 0 ? 2.5f : (kind == 1 ? 7.8f : 1.0f);
            new_spawn(sim, 0, BODY_SPHERE, spot(sim, 70, r + 0.5f), vec3(r, r, r), dens, mat);
        }
        for (int k = 0; k < so2->boxes; k++){
            Vec3 half = vec3(rnd(sim, 22, 36), rnd(sim, 22, 36), rnd(sim, 22, 36));
            Spawn *p = new_spawn(sim, 0, BODY_BOX, spot(sim, 80, half.y + 0.5f), half, 0.5f, box_cols[k % 2]);
            if (p) p->rot = quat_axis_angle(vec3(0, 1, 0), rnd(sim, 0, (real)PI));
        }
        for (int k = 0; k < so2->balloons; k++){
            real r = rnd(sim, 26, 34);
            // slightly heavier than air: they rest on the floor until the blast
            Spawn *p = new_spawn(sim, 0, BODY_SPHERE, spot(sim, 70, r + 0.5f), vec3(r, r, r), 0.0016f, balloon_cols[k % 3]);
            if (p){ p->restitution = 0.7f; p->friction = 0.3f; }
        }
        // in the lower part of the cloud: it bursts, and the shock reaches the floor
        sim->blast_t = 1.0;
        sim->blast_at = vec3(278, 190, 290);
        sim->blast_radius = 400;
        sim->blast_speed = 1100;
        sim->blast_impulse = 50000;
    }
    sort_spawns(sim);
    // things placed at t = 0 must not overlap: push later ones up until they are clear
    for (int i = 0; i < sim->nspawns; i++){
        Spawn *a = &sim->spawns[i];
        if (a->t > 0) break;
        for (int tries = 0; tries < 20; tries++){
            int clear = 1;
            for (int j = 0; j < i; j++){
                Spawn *b = &sim->spawns[j];
                real ra = a->shape == BODY_SPHERE ? a->half.x : v_len(a->half);
                real rb = b->shape == BODY_SPHERE ? b->half.x : v_len(b->half);
                if (v_len(v_sub(a->pos, b->pos)) < ra + rb + 2){ clear = 0; break; }
            }
            if (clear) break;
            a->pos = spot(sim, 70, a->pos.y);
        }
    }
    printf("physics: %d bodies scheduled, air %dx%dx%d cells (%.1f units), cloud %dx%dx%d voxels, wind %g,%g,%g\n",
           sim->nspawns, pw->fluid->nx, pw->fluid->ny, pw->fluid->nz, (double)pw->fluid->h, v->nx, v->ny, v->nz,
           (double)o->sim.wind.x, (double)o->sim.wind.y, (double)o->sim.wind.z);
    return 0;
}

static void spawn_due(Sim *sim, Scene *s){
    PhysWorld *pw = &sim->pw;
    while (sim->next_spawn < sim->nspawns && sim->spawns[sim->next_spawn].t <= pw->time + 1e-9){
        const Spawn *p = &sim->spawns[sim->next_spawn++];
        int i = p->shape == BODY_SPHERE ? rigid_add_sphere(&pw->rigid, p->pos, p->half.x, p->density)
                                        : rigid_add_box(&pw->rigid, p->pos, p->half, p->rot, p->density);
        Body *b = &pw->rigid.bodies[i];
        b->vel = p->vel;
        b->restitution = p->restitution;
        b->friction = p->friction;
        phys_add_to_scene(pw, i, s, p->material);
    }
}

void sim_step(Sim *sim, Scene *s, double dt){
    PhysWorld *pw = &sim->pw;
    double mx = sim->max_dt > 0 ? sim->max_dt : MAX_PHYS_DT;
    int n = (int)(dt / mx + 0.999);
    if (n < 1) n = 1;
    real h = (real)(dt / n);
    for (int k = 0; k < n; k++){
        spawn_due(sim, s);
        if (sim->blast_t >= 0 && pw->time + 1e-9 >= sim->blast_t){
            sim_blast_now(sim);
            sim->blast_t = -1;
        }
        phys_step(pw, h);
        sim->steps++;
    }
    phys_sync_scene(pw, s);
}

void sim_drop_ball(Sim *sim, Scene *s){
    PhysWorld *pw = &sim->pw;
    real r = rnd(sim, 25, 40);
    Vec3 p = spot(sim, 80, 480);
    int i = rigid_add_sphere(&pw->rigid, p, r, 7.8f);
    pw->rigid.bodies[i].vel = vec3(rnd(sim, -80, 80), 0, rnd(sim, -80, 80));
    phys_add_to_scene(pw, i, s, sim->mat_drop);
}

void sim_blast_now(Sim *sim){
    Vec3 c = sim->blast_radius > 0 ? sim->blast_at : vec3(278, 190, 290);
    real r = sim->blast_radius > 0 ? sim->blast_radius : 400;
    real sp = sim->blast_speed > 0 ? sim->blast_speed : 1100;
    real im = sim->blast_impulse > 0 ? sim->blast_impulse : 50000;
    phys_blast(&sim->pw, c, r, sp, im);
}

void sim_free(Sim *sim){
    phys_free(&sim->pw);   // the cloud volume belongs to the scene
}

void sim_status(const Sim *sim, char *buf, int len){
    const PhysWorld *pw = &sim->pw;
    snprintf(buf, (size_t)len, "t %.2f s | %d bodies, %d contacts | rigid %.1f ms, air %.1f ms%s",
             (double)pw->time, pw->rigid.nbodies, pw->rigid.ncontacts, pw->ms_rigid, pw->ms_fluid,
             pw->air_step ? " (on the GPU)" : "");
}
