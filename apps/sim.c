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
// tracer's acceleration structure, render, write frame_NNNN.ppm. At the end the frames
// become a GIF with tools/frames_to_gif.py, run by uv (which fetches Pillow by itself) or,
// without uv, by python3 (with Pillow installed).
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir((p), 0755)
#endif
#include <time.h>

#include "app/frame.h"
#include "app/gpu_frame.h"
#include "app/options.h"
#include "gpu/fluid_gpu.h"
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
        for (char *p = tmp + 1; *p; p++) if (*p == '/'){ *p = 0; make_dir(tmp); *p = '/'; }
        make_dir(tmp);
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

    // the GPU (-gpu on, the default when the build has one): the air, the cloud and the frames
    Gpu *gpu = NULL;
    GpuView *gview = NULL;
    GpuFluid *gfluid = NULL;
    if (o.gpu && gpu_frame_open(W, H, &gpu, &gview) == 0){
        printf("GPU: %s: the frames%s are made there\n", gpu_message(gpu), o.gpu_fluid ? ", the air and the cloud" : "");
        if (o.gpu_fluid && (gfluid = gpu_fluid_create(gpu, &sim.pw))){
            gpu_fluid_attach_view(gfluid, gview);
            gpu_fluid_set_wait(gfluid, 1);       // every step runs, however slow the GPU: a reproducible film
        }
    }

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
        int err = -1;
        if (gview){
            // on the GPU; if it cannot draw this scene, the CPU takes over from here on
            if (o.backend == BACKEND_RASTER){
                if (gpu_frame_raster(gpu, gview, &scene, &rp, &o.march, o.raster_clouds, fb) == 0) err = fb_save_ppm(fb, path);
            } else {
                scene_set_accel(&scene, ACCEL_BVH, &o.ap);
                if (gpu_frame_trace(gpu, gview, &scene, &cam, &rp, &o.march, o.spp, o.max_depth, o.mis, (real)o.clamp, &img, 0) == 0)
                    err = image_write_ppm(&img, path);
            }
            if (err < 0){
                fprintf(stderr, "\nthe GPU cannot draw this scene: the CPU renders the frames\n");
                gpu_view_destroy(gview);
                gview = NULL;
            }
        }
        if (err >= 0){
            // drawn on the GPU
        } else if (gfluid){
            // the air is on the GPU: the CPU renderers need its cloud back first
            Volume *cv = sim.pw.fluid->density;
            uint32_t v0 = cv ? cv->version : 0;
            gpu_fluid_sync_cloud(gfluid);
            for (int k = 0; cv && k < 200 && cv->version == v0; k++) gpu_wait(gpu);
        }
        if (err >= 0){
        } else if (o.backend == BACKEND_RASTER){
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
        // uv reads the script's header (PEP 723) and runs it with Pillow; python3 is the
        // fallback for a machine without uv
        // fallback for a machine without uv. On Windows uv usually lives in
        // %USERPROFILE%\.local\bin, which an MSYS2 shell does not have in its PATH.
        static const char *runners[] = {
            "uv run --quiet",
#ifdef _WIN32
            "\"%USERPROFILE%\\.local\\bin\\uv.exe\" run --quiet",
#endif
            "python3",
        };
        char args[1300], cmd[1500];
        snprintf(args, sizeof(args), "tools/frames_to_gif.py \"%s\" \"%s.gif\" %g", o.out, o.out, o.sim.fps);
        int done = 0;
        for (size_t k = 0; k < sizeof(runners) / sizeof(runners[0]) && !done; k++){
#ifdef _WIN32
            // cmd.exe drops the first and last quote of the line: wrap it in one more pair
            snprintf(cmd, sizeof(cmd), "\"%s %s\"", runners[k], args);
#else
            snprintf(cmd, sizeof(cmd), "%s %s", runners[k], args);
#endif
            printf("%s\n", cmd);
            fflush(stdout);
            done = system(cmd) == 0;
        }
        if (!done) fprintf(stderr, "could not make the GIF (needs uv, or python3 with Pillow); the frames are there\n");
    }

    gpu_fluid_destroy(gfluid);
    gpu_view_destroy(gview);
    gpu_destroy(gpu);
    image_free(&img);
    if (fb) fb_destroy(fb);
    raster_volumes_free(rvol);
    sim_free(&sim);
    scene_free(&scene);
    return 0;
}
