// hello_rt.c - the smallest program that PATH TRACES a scene with the engine (liblr.a).
// Builds the Cornell box, adds a BVH, traces 16 paths per pixel, writes a PPM.
// hello_raster.c rasterizes the very same scene.
//
//   make bin/hello_rt && bin/hello_rt        -> out/hello_rt.ppm
#include <stdio.h>
#include "lr.h"

int main(void){
    static Scene scene;                                   // must start zeroed
    SceneOptions so = {.kind = SCENE_BOXES};             // everything else zero: no model, no cloud, no fog
    if (scene_cornell_box(&scene, &so) != 0) return 1;

    AccelParams ap = accel_default_params();              // octree depth, BVH leaf, bins, split
    scene_set_accel(&scene, ACCEL_BVH, &ap);

    const int W = 200, H = 200, SPP = 16, BOUNCES = 50;
    Camera cam = camera_make(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, (real)W / H);
    Image img = image_create(W, H);

    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            Rng rng;
            rng_seed(&rng, (uint64_t)(y * W + x), 1);
            Vec3 sum = vec3(0, 0, 0);
            for (int s = 0; s < SPP; s++){
                Ray r = camera_ray(&cam, (x + rng_real(&rng)) / W, (y + rng_real(&rng)) / H);
                sum = v_add(sum, trace_path(&scene, r, &rng, BOUNCES));
            }
            img.px[y * W + x] = v_scale(sum, 1.0 / SPP);
        }
    }

    int err = image_write_ppm(&img, "out/hello_rt.ppm");
    printf(err ? "could not write out/hello_rt.ppm\n" : "wrote out/hello_rt.ppm\n");
    image_free(&img);
    scene_free(&scene);
    return err;
}
