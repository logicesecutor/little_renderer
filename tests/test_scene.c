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
#include <stdio.h>
#include <string.h>
#include "scene/scene.h"
#include "rt/accel.h"
#include "rt/intersect.h"
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
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all passed\n");
    return 0;
}
