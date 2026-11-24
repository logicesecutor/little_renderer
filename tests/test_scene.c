// test_scene.c - the scene's instances (scene/scene.h) added, moved and removed, and what the
// path tracer's trees (rt/accel.h) see of them afterwards.
//
//   make bin/test_scene && bin/test_scene      (part of make test)
//
// 1. Removing an instance: the last takes its place with a new version, the count drops,
//    the scene's version changes; removing the last moves nothing.
// 2. A copy that follows the versions (as the GPU's do) ends up with the scene's instances.
// 3. The top level after a removal: a ray at a removed instance misses it, one at the moved
//    instance (now at the removed one's index) hits it there.
// 4. The light probe baked in steps (rt/probe.h, probe_job_*), its levels made in steps
//    too (scene/env.h, env_prepare_start / _step), is the probe and the grid baked at once,
//    to the bit.
// 5. Glowing shapes sampled as lights (Scene.emitters, rt/tracer.c): a floor of albedo 0.5
//    under a glowing ball of radiance 10, radius 10, 50 above it: the floor right below sends
//    0.5 * 10 * (10 / 50)^2 = 0.2. With next event estimation (MIS on and off) and found by
//    the paths alone: the same mean, NEE's far less noisy. A glowing box: NEE and the paths
//    alone agree. A stretched ball is not a light; a lamp-less scene with none is unchanged.
// 6. Meshes, materials and textures removed (scene_remove_mesh, _material, _texmap): the last
//    one takes the place of the one removed, and what the scene has of it follows (the
//    instances' meshes, the triangles' and instances' materials, the materials' textures).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "scene/scene.h"
#include "rt/accel.h"
#include "rt/intersect.h"
#include "rt/probe.h"
#include "scene/env.h"
#include "rt/tracer.h"
#include "base/rng.h"
#include "test_util.h"

static Xform at(real x, real r){ return xform_axes(mat3_identity(), vec3(r, r, r), vec3(x, 0, 0)); }

static int hit_inst(Scene *s, real x){
    scene_update_instances(s, NULL);
    Ray ray = {vec3(x, 0, -100), vec3(0, 0, 1)};
    Hit h;
    return scene_intersect(s, &ray, 1e-3f, 1e9f, &h) ? h.inst : -2;
}

int main(void){
    Scene s;
    memset(&s, 0, sizeof(s));
    int mat = scene_add_material(&s, material_diffuse(vec3(0.5f, 0.5f, 0.5f)));
    for (int i = 0; i < 4; i++) scene_add_instance(&s, i & 1 ? INST_BOX : INST_SPHERE, at(100 * i, 10), mat);
    scene_set_accel(&s, ACCEL_BVH, NULL);

    printf("1. removing instances\n");
    check("four instances, the ray at x = 100 hits instance 1", s.ninsts == 4 && hit_inst(&s, 100) == 1);
    unsigned v0 = s.inst_version;
    int moved = scene_remove_instance(&s, 1);
    check("removing 1: the last (3) moved into its place", moved == 3 && s.ninsts == 3);
    check("... the box from x = 300, with a new version", s.insts[1].mesh == INST_BOX &&
          s.insts[1].xf.m[0][3] == 300 && s.insts[1].version > v0 && s.inst_version == s.insts[1].version);
    check("... the others as they were", s.insts[0].version <= v0 && s.insts[2].version <= v0 && s.insts[2].mesh == INST_SPHERE);
    unsigned v1 = s.inst_version;
    check("removing the last (2): nothing moved", scene_remove_instance(&s, 2) == -1 && s.ninsts == 2 && s.inst_version > v1);
    check("an index out of range: nothing", scene_remove_instance(&s, 5) == -1 && s.ninsts == 2);

    printf("2. a copy that follows the versions\n");
    // (as the GPU's: a copy made after the removals; then one added and one moved: it takes
    // only those above its version, not the one that stayed)
    unsigned copy = s.inst_version;
    scene_add_instance(&s, INST_BOX, at(500, 10), mat);
    scene_move_instance(&s, 0, at(-100, 10));
    int runs = 0, first = -1, last = -1;
    for (int i = 0, a, b; scene_changed_run(&s, copy, 0, &i, &a, &b);){ runs++; if (first < 0) first = a; last = b; }
    check("after a removal, an add and a move: the two changed, not the one kept", runs == 2 && first == 0 && last == 2);

    printf("3. the top level\n");
    check("the ray at the removed instance (x = 100) misses", hit_inst(&s, 100) == -2);
    check("the moved one (x = 300) is hit as instance 1", hit_inst(&s, 300) == 1);
    check("the moved and the added: instances 0 (x = -100) and 2 (x = 500)", hit_inst(&s, -100) == 0 && hit_inst(&s, 500) == 2);
    check("the removed last one (x = 200) misses", hit_inst(&s, 200) == -2);

    scene_free(&s);

    printf("4. the light probe in steps\n");
    memset(&s, 0, sizeof(s));
    mat = scene_add_material(&s, material_diffuse(vec3(0.6f, 0.5f, 0.4f)));
    scene_add_prim(&s, make_quad(vec3(-500, 0, -500), vec3(0, 0, 1000), vec3(1000, 0, 0), mat));
    scene_add_instance(&s, INST_SPHERE, xform_axes(mat3_identity(), vec3(40, 40, 40), vec3(80, 40, 0)), mat);
    scene_add_instance(&s, INST_BOX, xform_axes(mat3_identity(), vec3(30, 30, 30), vec3(-60, 30, 40)), mat);
    s.env = env_procedural_sky(64, 30, 120);
    s.light = -1;
    s.probe_set = 1;
    s.probe_at = vec3(0, 100, 0);
    s.probe_box = (Aabb){vec3(-500, 0, -500), vec3(500, 400, 500)};
    scene_set_accel(&s, ACCEL_BVH, NULL);
    Env *p1 = probe_bake_scene(&s, 16, 4);
    IrrGrid *g1 = probe_grid_bake_scene(&s, 32);
    ProbeJob *job = probe_job_start(&s, 16, 4, 32);
    int steps = 0, early = job ? probe_job_finish(job, &(Env *){0}, &(IrrGrid *){0}) : 0;
    double was = 0, done = 0;
    int rising = 1;
    while (job && done < 1 && steps < 100000){ done = probe_job_step(job, &s, 0); rising &= done >= was; was = done; steps++; }
    Env *p2 = NULL;
    IrrGrid *g2 = NULL;
    int fin = job ? probe_job_finish(job, &p2, &g2) : 2;
    check("a job: not finished before its steps; many steps, done rising to 1", job && early == 1 && steps > 8 && rising && done == 1);
    check("finished: a probe and a grid", fin == 0 && p1 && p2 && g1 && g2);
    if (p1 && p2 && g1 && g2){
        check("the probe: its texels the same, to the bit", p1->w == p2->w && p1->h == p2->h &&
              !memcmp(p1->rgb, p2->rgb, sizeof(float) * 3 * (size_t)p1->w * p1->h) && p2->boxed && p2->center.y == p1->center.y);
        int lv = 1;
        for (int k = 0; k < ENV_LEVELS; k++)
            lv &= p1->lw[k] == p2->lw[k] && !memcmp(p1->level[k], p2->level[k], sizeof(float) * 3 * (size_t)p1->lw[k] * p1->lh[k]);
        check("its levels (made in steps) and harmonics the same, to the bit", lv && !memcmp(p1->sh, p2->sh, sizeof(p1->sh)) &&
              !memcmp(p1->cond, p2->cond, sizeof(float) * (size_t)p1->h * (p1->w + 1)) && p2->id != p1->id);
        size_t ng = (size_t)g1->n[0] * g1->n[1] * g1->n[2];
        check("the grid: the same probes, to the bit", g2->n[0] == g1->n[0] && g2->n[1] == g1->n[1] && !memcmp(g1->sh, g2->sh, ng * IRR_FLOATS * sizeof(float)));
    }
    env_free(p1); env_free(p2); irr_grid_free(g1); irr_grid_free(g2);
    scene_free(&s);
    printf("5. glowing shapes as lights\n");
    {
        Scene g;
        memset(&g, 0, sizeof(g));
        int floor = scene_add_material(&g, material_diffuse(vec3(0.5f, 0.5f, 0.5f)));
        Material glow = material_pbr(vec3(0, 0, 0), 0, 1, 1.5f, 0);
        glow.emission = vec3(10, 10, 10);
        int mg = scene_add_material(&g, glow);
        scene_add_prim(&g, make_quad(vec3(-5000, 0, -5000), vec3(0, 0, 10000), vec3(10000, 0, 0), floor));
        int ball = scene_add_instance(&g, INST_SPHERE, xform_axes(mat3_identity(), vec3(10, 10, 10), vec3(0, 50, 0)), mg);
        g.light = -1;
        scene_set_accel(&g, ACCEL_BVH, NULL);
        check("the ball is the scene's one glowing shape", g.nemitters == 1 && g.emitters[0] == ball);
        // the floor point below the ball, seen from the side
        Ray r = {vec3(-200, 20, 0), v_norm(vec3(200, -20, 0))};
        const int N = 40000;
        double mean[3], var[3];
        for (int mode = 0; mode < 3; mode++){            // 0: NEE + MIS, 1: NEE without MIS, 2: the paths alone
            TraceParams tp = trace_default_params();
            tp.mis = mode != 1;
            tp.sampler = 0;
            g.nemitters = mode == 2 ? 0 : 1;
            Rng rng;
            rng_seed(&rng, 1234, 5);
            double sum = 0, sum2 = 0;
            for (int k = 0; k < N; k++){ double v = trace_path_ex(&g, r, &rng, &tp).x; sum += v; sum2 += v * v; }
            mean[mode] = sum / N;
            var[mode] = sum2 / N - mean[mode] * mean[mode];
        }
        g.nemitters = 1;
        printf("   means %.5f %.5f %.5f, standard deviations %.4f %.4f %.4f\n", mean[0], mean[1], mean[2], sqrt(var[0]), sqrt(var[1]), sqrt(var[2]));
        // (within 2%: the diffuse BSDF is not quite Lambert's, seen this low)
        check_near("with NEE and MIS: 0.5 * 10 * (10 / 50)^2", mean[0], 0.2, 0.004);
        check_near("NEE without MIS", mean[1], 0.2, 0.004);
        check_near("the paths alone (noisier)", mean[2], 0.2, 0.012);
        check("NEE's noise a fraction of the paths' alone (under a fifth)", sqrt(var[0]) < 0.2 * sqrt(var[2]) && sqrt(var[1]) < 0.2 * sqrt(var[2]));
        // a glowing box instead, turned: NEE and the paths alone agree
        Mat3 turn = quat_to_mat3(quat_axis_angle(v_norm(vec3(1, 2, 0.5f)), 0.7f));
        scene_move_instance(&g, ball, xform_axes(turn, vec3(12, 8, 10), vec3(0, 50, 0)));
        g.insts[ball].mesh = INST_BOX;
        scene_update_instances(&g, NULL);
        check("a box is one too", g.nemitters == 1);
        double mb[2], vb[2];
        for (int mode = 0; mode < 2; mode++){
            TraceParams tp = trace_default_params();
            tp.sampler = 0;
            g.nemitters = mode ? 0 : 1;
            Rng rng;
            rng_seed(&rng, 99, 7);
            double sum = 0, sum2 = 0;
            for (int k = 0; k < N; k++){ double v = trace_path_ex(&g, r, &rng, &tp).x; sum += v; sum2 += v * v; }
            mb[mode] = sum / N;
            vb[mode] = sum2 / N - mb[mode] * mb[mode];
        }
        printf("   box: means %.5f %.5f, standard deviations %.4f %.4f\n", mb[0], mb[1], sqrt(vb[0]), sqrt(vb[1]));
        check_near("the box: NEE's mean, the paths' alone", mb[0] - mb[1], 0, 3 * sqrt(vb[1] / N) + 3 * sqrt(vb[0] / N));
        check("... NEE's noise under half (a face by its area: half of them turned away)", sqrt(vb[0]) < sqrt(vb[1]) / 2);
        // stretched, a ball is not sampled
        scene_move_instance(&g, ball, xform_axes(mat3_identity(), vec3(10, 20, 10), vec3(0, 50, 0)));
        g.insts[ball].mesh = INST_SPHERE;
        scene_update_instances(&g, NULL);
        check("a stretched ball: not a light", g.nemitters == 0);
        scene_free(&g);
    }
    printf("6. removing meshes, materials and textures\n");
    {
        // three materials, three meshes of one triangle each (of material 0, 1, 2), instances
        // of the first and the last
        Scene r;
        memset(&r, 0, sizeof(r));
        int m[3], mesh[3];
        for (int k = 0; k < 3; k++){
            m[k] = scene_add_material(&r, material_diffuse(vec3(0.2f + 0.3f * k, 0.5f, 0.5f)));
            Triangle *t = calloc(1, sizeof(Triangle));
            t->q = vec3(-20, -20, 0); t->u = vec3(40, 0, 0); t->v = vec3(0, 40, 0);
            for (int c = 0; c < 3; c++){ t->n[c] = -1; t->t[c] = -1; }
            t->material = m[k];
            mesh[k] = scene_add_mesh_arrays(&r, t, 1, NULL, 0, NULL, NULL, 0);
        }
        scene_add_instance(&r, mesh[0], at(0, 1), -1);
        int ic = scene_add_instance(&r, mesh[2], at(200, 1), -1);
        scene_set_accel(&r, ACCEL_BVH, NULL);
        check("three meshes, the ray at x = 200 hits the last one's instance", r.nmeshes == 3 && hit_inst(&r, 200) == ic);
        check("a mesh with an instance is not removed", scene_remove_mesh(&r, mesh[0]) == -2 && r.nmeshes == 3);
        unsigned id2 = r.meshes[2].id;
        moved = scene_remove_mesh(&r, mesh[1]);
        check("removing the middle one: the last moved into its place, its instance with it",
              moved == 2 && r.nmeshes == 2 && r.meshes[1].id == id2 && r.insts[ic].mesh == 1);
        scene_set_accel(&r, ACCEL_BVH, NULL);
        check("... still hit at x = 200, of its material", hit_inst(&r, 200) == ic && scene_instance_material(&r, ic, 0) == m[2]);
        // material 1 has no user now: the last (2) takes its number, in the mesh's triangles too
        unsigned ver = r.meshes[1].version;
        moved = scene_remove_material(&r, m[1]);
        check("removing material 1: the last (2) is 1 now, and the triangles of it say so",
              moved == 2 && r.nmats == 2 && r.meshes[1].tris[0].material == 1 && r.mats[1].albedo.x > 0.7f);
        check("... the mesh has a new version, and its tree is still good", r.meshes[1].version != ver && scene_mesh_bvh(&r, 1) != NULL);
        check("... the first mesh was not touched", r.meshes[0].tris[0].material == 0 && scene_mesh_bvh(&r, 0) != NULL);
        check("removing the last material: nothing moved", scene_remove_material(&r, 1) == -1 && r.nmats == 1);
        // three textures; material 0 has the first as its colour and the third as its normal map
        unsigned char px[4 * 4 * 4];
        int tex[3];
        for (int k = 0; k < 3; k++){
            TexMap tm;
            memset(px, 40 + 60 * k, sizeof(px));
            texmap_make(&tm, px, 4, 4, 0);
            tex[k] = scene_add_texmap(&r, &tm);
        }
        r.mats[0].tex_base = tex[0];
        r.mats[0].tex_normal = tex[2];
        unsigned tv = r.tex_version;
        moved = scene_remove_texmap(&r, tex[1]);
        check("removing the middle texture: the last takes its number, the material follows",
              moved == 3 && r.ntexmaps == 2 && r.mats[0].tex_normal == 2 && r.mats[0].tex_base == 1 && r.tex_version != tv);
        check("... and shows what it showed", texmap_texel(&r.texmaps[1], 0, 1, 1).x > 0.55f);
        moved = scene_remove_texmap(&r, 1);
        check("removing a texture a material still has: it has none there", moved == 2 && r.ntexmaps == 1 && r.mats[0].tex_base == 0 && r.mats[0].tex_normal == 1);
        check("a number out of range: nothing", scene_remove_texmap(&r, 5) == 0 && scene_remove_mesh(&r, 7) == -1 && scene_remove_material(&r, 9) == -1);
        scene_free(&r);
    }
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all passed\n");
    return 0;
}
