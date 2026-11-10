// render.c - offline renderer: one image of the Cornell box into a PPM file, made by the
// path tracer (-backend rt, default) or the rasterizer (-backend raster). Run with -h.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "app/options.h"
#include "lr.h"

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;   // (casts: see Makefile, PRECFLAGS)
}

static int render_raster(const Options *opt, const Scene *scene){
    Framebuffer *fb = fb_create(opt->width, opt->height);
    if (!fb){ fprintf(stderr, "out of memory\n"); return 1; }
    RenderParams p = render_params_camera(scene->cam_from, scene->cam_at, scene->cam_up,
                                          scene->cam_vfov_deg, opt->width, opt->height);
    p.mode = opt->shading;
    printf("rasterizing %dx%d, %s shading\n", opt->width, opt->height, render_mode_name(p.mode));

    double t0 = now_s();
    fb_clear(fb, rgb8(0, 0, 0));
    render_scene(fb, scene, &p);
    double ms = (now_s() - t0) * 1000.0;
    if (scene->nvolumes > 0 && opt->raster_clouds){
        // the media, raymarched over the opaque frame (raster/volume_march.h)
        RasterVolumes *rv = raster_volumes_create();
        raster_draw_volumes(rv, fb, scene, &p, &opt->march);
        double ml, mm;
        raster_volumes_timing(rv, &ml, &mm);
        printf("raymarched %d medium(s): light grid %.1f ms, march %.1f ms\n", scene->nvolumes, ml, mm);
        ms += ml + mm;
        raster_volumes_free(rv);
    }

    int err = fb_save_ppm(fb, opt->out);
    if (err) fprintf(stderr, "could not write %s (does the folder exist?)\n", opt->out);
    else printf("wrote %s in %.1f ms\n", opt->out, ms);
    fb_destroy(fb);
    return err;
}

static int render_rt(const Options *opt, const Scene *scene){
    const int W = opt->width, H = opt->height, spp = opt->spp;
    Camera cam = camera_make(scene->cam_from, scene->cam_at, scene->cam_up, scene->cam_vfov_deg, (real)W / H);
    Image img = image_create(W, H);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    TraceParams tp = trace_default_params();
    tp.max_depth = opt->max_depth;
    tp.mis = opt->mis;
    tp.skip_caustics = opt->caustics_lt;     // light tracing adds them below
    printf("path tracing %dx%d, %d samples/pixel, max %d bounces, %d thread(s)%s%s%s\n",
           W, H, spp, opt->max_depth, threads, opt->clamp > 0 ? ", clamped" : "",
           opt->mis ? ", MIS" : "", opt->caustics_lt ? ", caustics by light tracing" : "");

    double t0 = now_s();
    int rows_done = 0;
    const real clamp = (real)opt->clamp;

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
                real u = (x + rng_real(&rng)) / W;
                real v = (y + rng_real(&rng)) / H;
                Vec3 c = trace_path_ex(scene, camera_ray(&cam, u, v), &rng, &tp);
                if (!(c.x == c.x && c.y == c.y && c.z == c.z)) continue;   // skip NaNs
                // optional firefly clamp: caps one sample's brightness. Removes speckles
                // but also removes some real energy (biased), so it is off by default.
                if (clamp > 0 && v_max(c) > clamp) c = v_scale(c, clamp / v_max(c));
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

    fprintf(stderr, "\n");
    if (opt->caustics_lt){
        // the caustic paths the path tracer left out, traced from the lamp
        long long photons = (long long)((double)W * H * spp * opt->photons);
        double l0 = now_s();
        light_trace_caustics(scene, &cam, W, H, photons, &tp, 0x5EED0F11A57ULL, img.px);
        printf("light tracing: %lld light paths in %.1f s\n", photons, now_s() - l0);
    }
    double secs = now_s() - t0;
    int err = image_write_ppm(&img, opt->out);
    if (err) fprintf(stderr, "could not write %s (does the folder exist?)\n", opt->out);
    else printf("wrote %s in %.1f s (%.2f M paths/s)\n", opt->out, secs, (double)W * H * spp / secs / 1e6);
    image_free(&img);
    return err;
}

int main(int argc, char **argv){
    Options opt;
    options_parse(&opt, PROG_RENDER, argc, argv);

    static Scene scene;   // static: zero-initialised, as scene_cornell_box expects
    if (options_setup_scene(&opt, &scene) != 0) return 1;

    int err = opt.backend == BACKEND_RASTER ? render_raster(&opt, &scene) : render_rt(&opt, &scene);
    scene_free(&scene);
    return err;
}
