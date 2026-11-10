// sim.c - the physics engine at work: a simulation in the Cornell box (rigid bodies, the air
// and the cloud it carries), rendered frame by frame with the path tracer.
//
//   make && bin/sim                     drop demo: 120 frames into out/sim/, then out/sim.gif
//   bin/sim -demo blast -frames 90      an explosion in the cloud
//   bin/sim -backend raster             the same in seconds: rasterized, cloud raymarched
//   bin/sim -h                          every option
//
// Each frame: advance the physics by 1/fps seconds (steps of at most 1/60 s, each with 4
// rigid-body substeps), move the scene's primitives to the bodies, rebuild the path
// tracer's acceleration structure, render, write frame_NNNN.ppm.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "app/frame.h"
#include "app/options.h"
#include "app/simdemo.h"
#include "lr.h"

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(int argc, char **argv){
    Options o;
    options_parse(&o, PROG_SIM, argc, argv);

    static Scene scene;
    static Sim sim;
    if (sim_setup(&sim, &o, &scene) != 0) return 1;
    {   // the frames folder (and its parents)
        char tmp[512];
        snprintf(tmp, sizeof(tmp), "%s", o.out);
        for (char *p = tmp + 1; *p; p++) if (*p == '/'){ *p = 0; mkdir(tmp, 0755); *p = '/'; }
        mkdir(tmp, 0755);
    }

    const int W = o.width, H = o.height;
    Camera cam = camera_make(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, (real)W / H);
    TraceParams tp = trace_default_params();
    tp.max_depth = o.max_depth;
    tp.mis = o.mis;
    tp.skip_caustics = o.caustics_lt;
    Image img = image_create(W, H);
    Framebuffer *fb = o.backend == BACKEND_RASTER ? fb_create(W, H) : NULL;
    RasterVolumes *rvol = raster_volumes_create();   // the cloud, raymarched (raster backend)
    RenderParams rp = render_params_camera(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, W, H);
    rp.mode = o.shading;
    srgb_lut_init();

    printf("rendering %d frames at %g fps, %dx%d, %s\n", o.sim.frames, o.sim.fps, W, H,
           o.backend == BACKEND_RASTER ? (o.raster_clouds ? "rasterized, cloud raymarched" : "rasterized, no cloud") : "path traced");
    if (o.backend != BACKEND_RASTER) printf("  %d samples/pixel, max %d bounces%s\n", o.spp, o.max_depth,
                                            o.caustics_lt ? ", caustics by light tracing" : "");
    double t0 = now_s(), phys_s = 0, render_s = 0;
    for (int f = 0; f < o.sim.frames; f++){
        double a = now_s();
        if (f > 0) sim_step(&sim, &scene, 1.0 / o.sim.fps);
        else sim_step(&sim, &scene, 0);   // frame 0: the starting state (bodies due at t = 0)
        double b = now_s();
        phys_s += b - a;

        char path[600];
        snprintf(path, sizeof(path), "%s/frame_%04d.ppm", o.out, f);
        int err;
        if (o.backend == BACKEND_RASTER){
            fb_clear(fb, rgb8(0, 0, 0));
            render_scene(fb, &scene, &rp);
            if (o.raster_clouds) raster_draw_volumes(rvol, fb, &scene, &rp, &o.march);
            err = fb_save_ppm(fb, path);
        } else {
            scene_set_accel(&scene, options_resolve_accel(&o, &scene), &o.ap);
            frame_render_rt(&scene, &cam, &img, o.spp, &tp, (real)o.clamp, o.caustics_lt ? o.photons : 0, (uint64_t)f + 1);
            err = image_write_ppm(&img, path);
        }
        if (err){ fprintf(stderr, "could not write %s\n", path); return 1; }
        double c = now_s();
        render_s += c - b;
        char st[256];
        sim_status(&sim, st, sizeof(st));
        double eta = (c - t0) / (f + 1) * (o.sim.frames - f - 1);
        printf("\rframe %4d/%d | %s | physics %.0f ms, render %.1f s | eta %.0f s   ",
               f + 1, o.sim.frames, st, (b - a) * 1000, c - b, eta);
        fflush(stdout);
    }
    printf("\n%d frames in %s/ in %.0f s (physics %.1f s, rendering %.0f s)\n", o.sim.frames, o.out,
           now_s() - t0, phys_s, render_s);

    if (o.sim.gif){
        char cmd[1400];
        snprintf(cmd, sizeof(cmd), "python3 tools/frames_to_gif.py %s %s.gif %g", o.out, o.out, o.sim.fps);
        printf("%s\n", cmd);
        if (system(cmd) != 0) fprintf(stderr, "could not make the GIF (needs python3 with Pillow); the frames are there\n");
    }

    image_free(&img);
    if (fb) fb_destroy(fb);
    raster_volumes_free(rvol);
    sim_free(&sim);
    scene_free(&scene);
    return 0;
}
