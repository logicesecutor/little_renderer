// custom_scene.c - build your own scene instead of the Cornell box: a floor, a lamp, a
// glass sphere and a mirror sphere. Path traced (multithreaded with OpenMP) AND rasterized,
// because both backends of the engine draw the same Scene.
//
//   make bin/custom_scene && bin/custom_scene   -> out/custom_scene.ppm, out/custom_scene_raster.ppm
#include <stdio.h>
#include "lr.h"

int main(void){
    static Scene s;   // a Scene must start zeroed (static, or memset to 0)

    Material white  = {MAT_DIFFUSE, vec3(0.75, 0.75, 0.75), vec3(0, 0, 0), 1.0};
    Material orange = {MAT_DIFFUSE, vec3(0.80, 0.45, 0.15), vec3(0, 0, 0), 1.0};
    Material lamp   = {MAT_LIGHT,   vec3(0, 0, 0),          vec3(12, 12, 12), 1.0};
    Material glass  = {MAT_GLASS,   vec3(1, 1, 1),          vec3(0, 0, 0), 1.5};
    Material mirror = {MAT_MIRROR,  vec3(0.9, 0.9, 0.9),    vec3(0, 0, 0), 1.0};
    int m_white = scene_add_material(&s, white), m_orange = scene_add_material(&s, orange);
    int m_lamp = scene_add_material(&s, lamp), m_glass = scene_add_material(&s, glass);
    int m_mirror = scene_add_material(&s, mirror);

    // a quad is corner + two edges; for the lamp, u x v must point to where it shines (down)
    scene_add_prim(&s, make_quad(vec3(-600, 0, -600), vec3(0, 0, 1200), vec3(1200, 0, 0), m_white));   // floor
    scene_add_prim(&s, make_quad(vec3(-600, 0, 400), vec3(1200, 0, 0), vec3(0, 800, 0), m_orange));    // back wall
    s.light = scene_add_prim(&s, make_quad(vec3(-150, 500, -150), vec3(300, 0, 0), vec3(0, 0, 300), m_lamp));
    scene_add_prim(&s, make_sphere(vec3(-130, 100, 0), 100, m_glass));
    scene_add_prim(&s, make_sphere(vec3(130, 100, 80), 100, m_mirror));

    s.cam_from = vec3(0, 220, -650);
    s.cam_at = vec3(0, 110, 0);
    s.cam_up = vec3(0, 1, 0);
    s.cam_vfov_deg = 40;

    AccelParams ap = accel_default_params();
    scene_set_accel(&s, ACCEL_NONE, &ap);   // 5 primitives: no tree needed

    const int W = 320, H = 240, SPP = 64, BOUNCES = 50;
    Camera cam = camera_make(s.cam_from, s.cam_at, s.cam_up, s.cam_vfov_deg, (real)W / H);
    Image img = image_create(W, H);

    // trace_path only reads the scene, so rows can run in parallel; each pixel has its own Rng
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            Rng rng;
            rng_seed(&rng, (uint64_t)(y * W + x), 7);
            Vec3 sum = vec3(0, 0, 0);
            for (int k = 0; k < SPP; k++){
                Ray r = camera_ray(&cam, (x + rng_real(&rng)) / W, (y + rng_real(&rng)) / H);
                Vec3 c = trace_path(&s, r, &rng, BOUNCES);
                if (v_max(c) > 8) c = v_scale(c, 8 / v_max(c));   // firefly clamp (glass + small lamp)
                sum = v_add(sum, c);
            }
            img.px[y * W + x] = v_scale(sum, 1.0 / SPP);
        }
    }

    int err = image_write_ppm(&img, "out/custom_scene.ppm");
    printf(err ? "could not write out/custom_scene.ppm\n" : "wrote out/custom_scene.ppm\n");
    image_free(&img);

    // the same scene through the rasterizer: no shadows or reflections, but instant
    Framebuffer *fb = fb_create(W, H);
    fb_clear(fb, rgb8(0, 0, 0));
    RenderParams p = render_params_camera(s.cam_from, s.cam_at, s.cam_up, s.cam_vfov_deg, W, H);
    render_scene(fb, &s, &p);
    err |= fb_save_ppm(fb, "out/custom_scene_raster.ppm");
    printf("wrote out/custom_scene_raster.ppm\n");
    fb_destroy(fb);
    scene_free(&s);
    return err;
}
