// hello_raster.c - the smallest program that RASTERIZES a scene with the engine (liblr.a).
// Builds the Cornell box, draws it with the rasterizer in a few milliseconds, writes a PPM.
// hello_rt.c path traces the very same scene.
//
//   make bin/hello_raster && bin/hello_raster   -> out/hello_raster.ppm
#include <stdio.h>
#include "lr.h"

int main(void){
    static Scene scene;                                   // must start zeroed
    SceneOptions so = {.kind = SCENE_BOXES};             // everything else zero: no model, no cloud, no fog
    if (scene_cornell_box(&scene, &so) != 0) return 1;

    const int W = 200, H = 200;
    Framebuffer *fb = fb_create(W, H);
    fb_clear(fb, rgb8(0, 0, 0));

    // the scene's camera as view + projection matrices
    RenderParams p = render_params_camera(scene.cam_from, scene.cam_at, scene.cam_up,
                                          scene.cam_vfov_deg, W, H);
    p.mode = MODE_PHONG;                                  // or MODE_GOURAUD, MODE_FLAT, MODE_WIREFRAME
    render_scene(fb, &scene, &p);

    int err = fb_save_ppm(fb, "out/hello_raster.ppm");
    printf(err ? "could not write out/hello_raster.ppm\n" : "wrote out/hello_raster.ppm\n");
    fb_destroy(fb);
    scene_free(&scene);
    return err;
}
