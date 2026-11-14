// hello_physics.c - the physics engine in a few lines: a steel ball falls through a cloud.
// The ball pushes the air, the air carries the cloud: a hole opens and a wake follows.
//
//   make bin/hello_physics && bin/hello_physics     -> out/hello_physics/frame_NN.png
#include <stdio.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir((p), 0755)
#endif
#include "lr.h"
#include "app/frame.h"   // frame_render_rt: one path traced image

int main(void){
    static Scene scene;
    SceneOptions so = {.kind = SCENE_EMPTY};
    scene_cornell_box(&scene, &so);                       // the room and the lamp

    // the cloud: one grid over the whole room, so the air can carry it anywhere
    Aabb room = {vec3(0, 0, 0), vec3(555, 555, 555)};
    Volume *cloud = volume_create(96, 96, 96, room);
    CloudParams cp = cloud_default_params();
    volume_add_cloud(cloud, (Aabb){vec3(80, 220, 110), vec3(475, 440, 450)}, &cp);
    scene_add_volume(&scene, cloud);

    // the physics: the room's walls, 40^3 cells of air carrying the cloud, a steel ball
    PhysWorld pw;
    phys_init(&pw);
    rigid_set_room(&pw.rigid, room);
    pw.fluid = fluid_create(room, 555.0 / 40, cloud);
    int ball = rigid_add_sphere(&pw.rigid, vec3(278, 500, 278), 40, 7.8);
    Material steel = material_mirror(vec3(0.95, 0.93, 0.9));
    phys_add_to_scene(&pw, ball, &scene, scene_add_material(&scene, steel));

    Camera cam = camera_make(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, 1);
    TraceParams tp = trace_default_params();
    Image img = image_create(200, 200);
    make_dir("out");
    make_dir("out/hello_physics");
    for (int f = 0; f < 30; f++){
        for (int k = 0; k < 2; k++) phys_step(&pw, 1.0 / 60);   // 30 frames per second
        phys_sync_scene(&pw, &scene);                           // move the ball's instance
        scene_update_instances(&scene, NULL);                   // the path tracer follows it
        frame_render_rt(&scene, &cam, &img, 16, &tp, 0, 0, (uint64_t)f);
        char path[64];
        snprintf(path, sizeof(path), "out/hello_physics/frame_%02d.png", f);
        image_write_png(&img, path);
        printf("\rframe %2d: ball at y = %5.1f, cloud mass %.0f   ", f, (double)pw.rigid.bodies[ball].pos.y,
               (double)fluid_total_density(pw.fluid));
        fflush(stdout);
    }
    printf("\nwrote out/hello_physics/frame_00..29.png\n");
    image_free(&img);
    phys_free(&pw);        // frees the air, not the cloud: the scene owns it
    scene_free(&scene);
    return 0;
}
