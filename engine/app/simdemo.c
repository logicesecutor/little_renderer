#include "app/simdemo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/volume.h"

#define ROOM 555.0
#define MAX_PHYS_DT (1.0 / 60.0)

static int add_mat(Scene *s, MatType t, Vec3 albedo, real ior){
    Material m = t == MAT_GLASS ? material_glass(albedo, ior) : t == MAT_MIRROR ? material_mirror(albedo) : material_diffuse(albedo);
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

// ---- SIM_SKY: a floor under the sky, a rain of bodies onto its middle among still props ----

#define FIELD_X 278.0
#define FIELD_Z 300.0
#define FIELD_RAIN 380.0        // the bodies appear over a disc of this radius ...
#define FIELD_WALLS 900.0       // ... inside invisible walls this far from its centre
#define FIELD_PROPS_IN 1050.0   // the props stand in a ring around them
#define FIELD_PROPS_OUT 6000.0

// a material of every kind: metals polished and rough, glass clear and frosted, plastics,
// a ceramic, rubber; and how dense a body of it is
static const struct { float r, g, b, metal, rough, trans, dens; } FIELD_MATS[] = {
    {1.00f, 0.78f, 0.34f, 1, 0.15f, 0, 7.8f},    // gold
    {0.95f, 0.64f, 0.54f, 1, 0.35f, 0, 7.8f},    // copper
    {0.95f, 0.95f, 0.95f, 1, 0.02f, 0, 7.8f},    // chrome
    {0.91f, 0.92f, 0.92f, 1, 0.45f, 0, 2.7f},    // brushed aluminium
    {0.55f, 0.25f, 0.80f, 1, 0.30f, 0, 2.7f},    // anodized purple
    {1.00f, 1.00f, 1.00f, 0, 0.00f, 1, 2.5f},    // clear glass
    {0.70f, 0.90f, 1.00f, 0, 0.30f, 1, 2.5f},    // frosted blue glass
    {0.80f, 0.08f, 0.06f, 0, 0.30f, 0, 1.0f},    // red plastic
    {0.08f, 0.22f, 0.75f, 0, 0.25f, 0, 1.0f},    // blue plastic
    {0.15f, 0.55f, 0.25f, 0, 0.10f, 0, 2.3f},    // green ceramic
    {0.90f, 0.70f, 0.10f, 0, 0.50f, 0, 1.0f},    // yellow plastic
    {0.85f, 0.85f, 0.85f, 0, 0.90f, 0, 1.5f},    // white matte
    {0.03f, 0.03f, 0.03f, 0, 0.08f, 0, 1.2f},    // black gloss
    {0.95f, 0.40f, 0.08f, 0, 0.70f, 0, 1.1f},    // orange rubber
};
#define FIELD_NMATS ((int)(sizeof(FIELD_MATS) / sizeof(FIELD_MATS[0])))

static int field_setup(Sim *sim, const Options *o, Scene *s){
    SceneOptions so = o->scene;
    so.kind = SCENE_SKY;
    so.gltf_path = NULL;
    so.bare = 1;
    so.cloud = NULL;
    so.instances = 0;
    if (scene_cornell_box(s, &so) != 0) return 1;
    s->cam_from = vec3(FIELD_X + 760, 560, FIELD_Z - 1550);
    s->cam_at = vec3(FIELD_X - 60, 90, FIELD_Z + 120);
    s->cam_vfov_deg = 42;
    options_bake_probe(o, s);                  // the floor and the sky: the rest comes after
    if (o->set_cam) s->cam_from = o->cam;
    if (o->set_look) s->cam_at = o->look;
    if (o->set_fov) s->cam_vfov_deg = o->fov;
    int mats[FIELD_NMATS];
    for (int i = 0; i < FIELD_NMATS; i++)
        mats[i] = scene_add_material(s, material_pbr(vec3(FIELD_MATS[i].r, FIELD_MATS[i].g, FIELD_MATS[i].b),
                                                     FIELD_MATS[i].metal, FIELD_MATS[i].rough, 1.5, FIELD_MATS[i].trans));

    // physics: no air, the floor and walls far enough not to be seen
    PhysWorld *pw = &sim->pw;
    phys_init(pw);
    rigid_set_room(&pw->rigid, (Aabb){vec3(FIELD_X - FIELD_WALLS, 0, FIELD_Z - FIELD_WALLS),
                                      vec3(FIELD_X + FIELD_WALLS, 8000, FIELD_Z + FIELD_WALLS)});
    // the models: dragons and tori at two sizes each, as convex hulls for the bodies (their
    // meshes drawn), any size for the props
    static const struct { const char *path; real size; } M[4] = {
        {"models/dragon_recon/dragon_vrip_res4.ply", 95}, {"models/torus.obj", 70},
        {"models/dragon_recon/dragon_vrip_res4.ply", 55}, {"models/torus.obj", 42}};
    int mesh[4], geom[4], nm = 0;
    for (int k = 0; k < 4; k++){
        int m = scene_load_mesh_object(s, M[k].path, mats[0], M[k].size);
        int g = m >= 0 ? rigid_geom_hull(&pw->rigid, s->meshes[m].tris, s->meshes[m].ntris, 0) : -1;
        if (g < 0){ fprintf(stderr, "(%s: not found or no hull, skipped: make models)\n", M[k].path); continue; }
        mesh[nm] = m; geom[nm] = g; nm++;
    }

    // the bodies: spheres, boxes and hulls, from over the disc, within 15 s
    int nb = o->sim.bodies;
    double span = nb > 100 ? 15.0 : 0.15 * nb;
    for (int k = 0; k < nb; k++){
        double t = 0.1 + span * k / (nb > 0 ? nb : 1);
        real a = rnd(sim, 0, 2 * (real)PI), r = FIELD_RAIN * r_sqrt(rnd(sim, 0, 1));
        Vec3 at = vec3(FIELD_X + r * r_cos(a), rnd(sim, 700, 1400), FIELD_Z + r * r_sin(a));
        int mi = (int)rnd(sim, 0, FIELD_NMATS) % FIELD_NMATS;
        real kind = rnd(sim, 0, 1), dens = FIELD_MATS[mi].dens;
        Spawn *p;
        if (kind < 0.35f || (kind >= 0.65f && nm == 0)){
            real rr = rnd(sim, 12, 32);
            p = new_spawn(sim, t, BODY_SPHERE, at, vec3(rr, rr, rr), dens, mats[mi]);
        } else if (kind < 0.65f){
            p = new_spawn(sim, t, BODY_BOX, at, vec3(rnd(sim, 10, 26), rnd(sim, 10, 26), rnd(sim, 10, 26)), dens, mats[mi]);
        } else {
            int j = (int)rnd(sim, 0, nm) % nm;
            p = new_spawn(sim, t, BODY_HULL, at, pw->rigid.geoms[geom[j]]->half, dens, mats[mi]);
            if (p){ p->geom = geom[j]; p->mesh = mesh[j]; }
        }
        if (!p) break;
        p->rot = random_rot(sim);
        p->vel = vec3(rnd(sim, -40, 40), rnd(sim, -200, 0), rnd(sim, -40, 40));
        p->restitution = FIELD_MATS[mi].trans > 0 ? 0.5f : 0.3f;
    }

    // the props: instances only (nothing collides with them), on the floor in the ring, not
    // overlapping and out of the camera's way (each tried at up to 60 spots), a few big chrome balls among them
    int np = o->sim.props, placed = 0;
    Vec3 *at = malloc((size_t)(np > 0 ? np : 1) * sizeof(Vec3));
    real *rad = malloc((size_t)(np > 0 ? np : 1) * sizeof(real));
    if (!at || !rad){ free(at); free(rad); fprintf(stderr, "out of memory for the props\n"); return 1; }
    real ex = s->cam_from.x - (real)FIELD_X, ez = s->cam_from.z - (real)FIELD_Z;
    for (int k = 0; k < np; k++){
        real kind = rnd(sim, 0, 1), big = rnd(sim, 0, 1);
        int mi = big < 0.01f ? 2 : (int)rnd(sim, 0, FIELD_NMATS) % FIELD_NMATS;
        Vec3 half = vec3(rnd(sim, 12, 40), rnd(sim, 12, 60), rnd(sim, 12, 40));
        real sr = big < 0.01f ? rnd(sim, 120, 200) : rnd(sim, 12, 50);
        int j = nm ? (int)rnd(sim, 0, nm) % nm : 0;
        real sc = rnd(sim, 0.5f, 1.8f);
        int shape = big < 0.01f || kind < 0.3f || nm == 0 ? 0 : kind < 0.6f ? 1 : 2;
        real R = shape == 0 ? sr : shape == 1 ? v_len(half) : M[0].size * 0.6f * sc;
        Vec3 pos = vec3(0, 0, 0);
        int ok = 0;
        for (int tries = 0; tries < 60 && !ok; tries++){
            real a = rnd(sim, 0, 2 * (real)PI);
            real r = r_sqrt(rnd(sim, (real)(FIELD_PROPS_IN * FIELD_PROPS_IN), (real)(FIELD_PROPS_OUT * FIELD_PROPS_OUT)));
            pos = vec3(FIELD_X + r * r_cos(a), 0, FIELD_Z + r * r_sin(a));
            // (and not between the camera and the rain)
            real u = ((pos.x - (real)FIELD_X) * ex + (pos.z - (real)FIELD_Z) * ez) / (ex * ex + ez * ez + 1e-6f);
            u = u < 0 ? 0 : u > 1.2f ? 1.2f : u;
            real cx = pos.x - (real)FIELD_X - u * ex, cz = pos.z - (real)FIELD_Z - u * ez;
            ok = cx * cx + cz * cz >= (R + 200) * (R + 200);
            for (int q = 0; q < placed && ok; q++){
                real dx = pos.x - at[q].x, dz = pos.z - at[q].z, d = R + rad[q] + 5;
                ok = dx * dx + dz * dz >= d * d;
            }
        }
        if (!ok) continue;
        at[placed] = pos;
        rad[placed++] = R;
        Mat3 yaw = quat_to_mat3(quat_axis_angle(vec3(0, 1, 0), rnd(sim, 0, 2 * (real)PI)));
        if (shape == 0) scene_add_instance(s, INST_SPHERE, xform_axes(mat3_identity(), vec3(sr, sr, sr), vec3(pos.x, sr, pos.z)), mats[mi]);
        else if (shape == 1) scene_add_instance(s, INST_BOX, xform_axes(yaw, half, vec3(pos.x, half.y, pos.z)), mats[mi]);
        else scene_add_instance(s, mesh[j], xform_make(pos, yaw.m, sc), mats[mi]);
    }
    free(at);
    free(rad);

    // the viewer's K: a blast in the middle of the pile
    sim->blast_at = vec3(FIELD_X, 20, FIELD_Z);
    sim->blast_radius = 600;
    sim->blast_speed = 1500;
    sim->blast_impulse = 60000;
    sort_spawns(sim);
    printf("physics: %d bodies falling within %.0f s onto the floor under the sky, among %d props (%d instances)\n",
           sim->nspawns, span, placed, s->ninsts);
    return 0;
}

int sim_setup(Sim *sim, const Options *o, Scene *s){
    memset(sim, 0, sizeof(*sim));
#ifdef _OPENMP
    if (o->threads > 0) omp_set_num_threads(o->threads);
#endif
    rng_seed(&sim->rng, 0x51AB1E5EEDULL + (uint64_t)o->sim.seed * 0x9E3779B97F4A7C15ULL, 7);
    sim->blast_t = -1;
    sim->max_dt = MAX_PHYS_DT;
    if (o->sim.demo == SIM_SKY) return field_setup(sim, o, s);

    // the room, nothing in it (the fog, if asked, stays a separate uniform medium)
    SceneOptions so = o->scene;
    so.kind = SCENE_EMPTY;
    so.cloud = NULL;
    if (scene_cornell_box(s, &so) != 0) return 1;
    options_bake_probe(o, s);                  // the empty room's: the bodies move
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
            double t = 0.25 + (n > 50 ? 20.0 / n : 0.4) * k;   // many: all of them within 20 s
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
    // meshes: the model standing in the room (fixed, its triangles collide, the air flows
    // around it), and copies of it falling as convex hulls
    if ((so2->collider || so2->meshes > 0) && o->scene.obj_path){
        int mat = scene_add_material(s, scene_mesh_material_tex(s, &o->scene));
        if (so2->collider){
            int mesh = scene_load_mesh_object(s, o->scene.obj_path, mat, 260);
            int g = mesh >= 0 ? rigid_geom_mesh(&pw->rigid, s->meshes[mesh].tris, s->meshes[mesh].ntris) : -1;
            if (g < 0){ fprintf(stderr, "could not make a collider of %s\n", o->scene.obj_path); return 1; }
            int b = rigid_add_mesh_collider(&pw->rigid, g, xform_rot_y(vec3(278, 0, 250), o->scene.rot_deg, 1));
            phys_add_mesh_to_scene(pw, b, s, mesh, -1);
        }
        if (so2->meshes > 0){
            int mesh = scene_load_mesh_object(s, o->scene.obj_path, mat, 120);
            int g = mesh >= 0 ? rigid_geom_hull(&pw->rigid, s->meshes[mesh].tris, s->meshes[mesh].ntris, 0) : -1;
            if (g < 0){ fprintf(stderr, "could not make a convex hull of %s\n", o->scene.obj_path); return 1; }
            for (int k = 0; k < so2->meshes; k++){
                int drop = o->sim.demo == SIM_DROP;
                Spawn *p = new_spawn(sim, drop ? 0.45 + 0.4 * k : 0, BODY_HULL, spot(sim, 90, drop ? 420 : 0.5f),
                                     pw->rigid.geoms[g]->half, 0.8f, -1);
                if (!p) break;
                p->geom = g;
                p->mesh = mesh;
                p->rot = drop ? random_rot(sim) : quat_axis_angle(vec3(0, 1, 0), rnd(sim, 0, (real)PI));
                if (drop) p->vel = vec3(rnd(sim, -60, 60), 0, rnd(sim, -60, 60));
            }
        }
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
        int i;
        if (p->shape == BODY_HULL){
            i = rigid_add_hull_body(&pw->rigid, p->geom, xform_make(p->pos, quat_to_mat3(p->rot).m, 1), p->density);
            if (i < 0) continue;
        } else i = p->shape == BODY_SPHERE ? rigid_add_sphere(&pw->rigid, p->pos, p->half.x, p->density)
                                           : rigid_add_box(&pw->rigid, p->pos, p->half, p->rot, p->density);
        Body *b = &pw->rigid.bodies[i];
        b->vel = p->vel;
        b->restitution = p->restitution;
        b->friction = p->friction;
        if (p->shape == BODY_HULL) phys_add_mesh_to_scene(pw, i, s, p->mesh, p->material);
        else phys_add_to_scene(pw, i, s, p->material);
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
    snprintf(buf, (size_t)len, "t %.2f s | %d bodies, %d contacts | rigid %.1f ms%s, air %.1f ms%s",
             (double)pw->time, pw->rigid.nbodies, pw->rigid.ncontacts, pw->ms_rigid, pw->rigid_step ? " (GPU)" : "",
             pw->ms_fluid, pw->air_step ? " (on the GPU)" : "");
}
