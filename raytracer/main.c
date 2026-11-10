// main.c - `raytracer`: renders the Cornell box into a PPM file. Run with -h for options.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "camera.h"
#include "image.h"
#include "options.h"
#include "scene.h"
#include "tracer.h"

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv){
    Options opt;
    options_parse(&opt, PROG_RAYTRACER, argc, argv);

    static Scene scene;   // static: zero-initialised, as scene_cornell_box expects
    if (options_setup_scene(&opt, &scene) != 0) return 1;

    const int W = opt.width, H = opt.height, spp = opt.spp;
    Camera cam = camera_make(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, (double)W / H);
    Image img = image_create(W, H);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("rendering %dx%d, %d samples/pixel, max %d bounces, %d thread(s)%s\n",
           W, H, spp, opt.max_depth, threads, opt.clamp > 0 ? ", clamped" : "");

    double t0 = now_s();
    int rows_done = 0;

    // Every row is independent, so rows are handed out to threads one at a time.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            uint64_t pixel = (uint64_t)y * W + x;
            Rng rng;
            rng_seed(&rng, 0x853c49e6748fea9bULL ^ (pixel * 0x9E3779B97F4A7C15ULL), pixel);

            Vec3 sum = vec3(0, 0, 0);
            for (int s = 0; s < spp; s++){
                // a random point inside the pixel: averaging them gives anti-aliasing for free
                double u = (x + rng_double(&rng)) / W;
                double v = (y + rng_double(&rng)) / H;
                Vec3 c = trace_path(&scene, camera_ray(&cam, u, v), &rng, opt.max_depth);
                if (!(c.x == c.x && c.y == c.y && c.z == c.z)) continue;   // skip NaNs
                // optional firefly clamp: caps one sample's brightness. Removes speckles
                // but also removes some real energy (biased), so it is off by default.
                if (opt.clamp > 0 && v_max(c) > opt.clamp) c = v_scale(c, opt.clamp / v_max(c));
                sum = v_add(sum, c);
            }
            img.px[pixel] = v_scale(sum, 1.0 / spp);
        }

        int done;
#ifdef _OPENMP
#pragma omp atomic capture
#endif
        done = ++rows_done;
        if (done % 8 == 0 || done == H){
            fprintf(stderr, "\r  %3d%%", done * 100 / H);
            fflush(stderr);
        }
    }

    double secs = now_s() - t0;
    fprintf(stderr, "\n");
    int err = image_write_ppm(&img, opt.out);
    if (err) fprintf(stderr, "could not write %s (does the folder exist?)\n", opt.out);
    else printf("wrote %s in %.1f s (%.2f M paths/s)\n", opt.out, secs, (double)W * H * spp / secs / 1e6);
    image_free(&img);
    scene_free(&scene);
    return err;
}
