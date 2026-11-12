// view.c - real-time viewer: fly through the scene with both backends of the engine.
//
//   make && bin/view [options]           (run with -h for the list)
//
// Three modes (key M cycles them, -backend picks the first one):
//   hybrid  the rasterizer draws every frame while the camera moves (fast, local lighting
//           only); when the camera stops, the path tracer takes over and refines the image
//           progressively, painting over the raster preview pass after pass.
//   rt      path tracer only; while moving it renders 1 sample per pixel at 1/preview
//           resolution so the image follows the input.
//   raster  rasterizer only.
//
// Clouds and fog: the rasterizer draws them by raymarching over its frame
// (raster/volume_march.h, half resolution by default), so they show while moving too.
//
// With -sim on the physics demo runs live (app/simdemo.h): every frame the simulation
// advances by the time that passed (at most 1/20 s, so a slow machine sees slow motion)
// and the scene follows the bodies. In hybrid mode (the default) the rasterizer and the
// raymarched cloud draw every frame while it runs; pause (Space) and the path tracer takes
// over and refines the frozen moment. In rt mode the path tracer renders one pass per
// frame (at 1/preview resolution, -movespp samples) and the window shows a running
// average of the last few passes (each new one weighs 1/3): a little motion blur for much
// less noise.
//
// Progressive path tracing: every pixel keeps a running sum of its samples and a count;
// when the camera is still, one more sample per pixel is added per pass, a band of rows
// at a time, so input is checked every ~20 ms. Any camera change starts over.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "app/options.h"
#include "app/simdemo.h"
#include "lr.h"
#include "gpu/gpu.h"
#include "gpu/view_gpu.h"
#include "gpu/fluid_gpu.h"
#include "gpu/rigid_gpu.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

static const char *CONTROLS =
    "controls\n"
    "  W A S D          move forward / left / back / right\n"
    "  E / Q            move up / down\n"
    "  mouse drag       look around (left button)\n"
    "  arrow keys       look around\n"
    "  mouse wheel      zoom (field of view)\n"
    "  Shift (hold)     move 4x faster\n"
    "  + / -            movement speed x1.5 / /1.5\n"
    "  M                mode: hybrid -> rt -> raster\n"
    "  V                raster shading: phong -> gouraud -> flat -> wireframe\n"
    "  [ / ]            fewer / more bounces per path\n"
    "  B                cycle acceleration: none -> octree -> bvh\n"
    "  F                rt mode: low-resolution preview while moving on/off\n"
    "  1 2 3 4          rt mode: preview resolution 1/1, 1/2, 1/4, 1/8\n"
    "  R                back to the starting camera\n"
    "  X                GPU path tracing: denoiser on / off\n"
    "  P                save the current image as PPM\n"
    "  C                print a render command for this exact view\n"
    "  H                print these controls\n"
    "  Esc              quit\n";

static const char *SIM_CONTROLS =
    "physics (-sim on)\n"
    "  Space            pause / resume the simulation\n"
    "  K                an explosion in the cloud\n"
    "  N                drop a steel ball\n"
    "  T                restart the demo\n";

#define HYBRID_SETTLE_S 0.12   // hybrid: start path tracing after the camera is still this long

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;   // (casts: see Makefile, PRECFLAGS)
}

// ---- camera as position + two angles ------------------------------------------------

typedef struct {
    Vec3 pos;
    double yaw, pitch;   // radians; yaw 0 / pitch 0 = looking down +z
    double fov;          // vertical field of view in degrees
} View;

static Vec3 view_forward(const View *v){
    return vec3(cos(v->pitch) * sin(v->yaw), sin(v->pitch), cos(v->pitch) * cos(v->yaw));
}
static Vec3 view_right(const View *v){ return v_norm(v_cross(view_forward(v), vec3(0, 1, 0))); }

static View view_from_scene(const Scene *s){
    View v;
    Vec3 f = v_norm(v_sub(s->cam_at, s->cam_from));
    v.pos = s->cam_from;
    v.pitch = asin(fmax(-1.0, fmin(1.0, f.y)));
    v.yaw = atan2(f.x, f.z);
    v.fov = s->cam_vfov_deg;
    return v;
}

// ---- progressive accumulation buffer (path tracer) ----------------------------------

typedef struct {
    int W, H;          // window size
    int scale;         // current internal resolution = window / scale
    int rw, rh;
    Vec3 *sum;         // per internal pixel: sum of all samples so far
    int *count;        // per internal pixel: how many samples
    int passes;        // completed passes = samples per pixel
    int band_y;        // next row to render in the current pass
} Accum;

static void accum_reset(Accum *a, int scale){
    a->scale = scale;
    a->rw = (a->W + scale - 1) / scale;
    a->rh = (a->H + scale - 1) / scale;
    memset(a->sum, 0, (size_t)a->rw * a->rh * sizeof(Vec3));
    memset(a->count, 0, (size_t)a->rw * a->rh * sizeof(int));
    a->passes = 0;
    a->band_y = 0;
}

// one new sample for every pixel in rows [y0, y1) of the internal image
static void render_rows(const Scene *s, const Camera *cam, Accum *a, int y0, int y1, int max_depth, int mis, real clamp,
                        int sampler){
    TraceParams tp = trace_default_params();
    tp.max_depth = max_depth;
    tp.mis = mis;
    tp.sampler = sampler;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = y0; y < y1; y++){
        for (int x = 0; x < a->rw; x++){
            int k = y * a->rw + x;
            // seed from (pixel, sample number): every sample of every pixel is different
            Rng rng;
            rng_seed(&rng, ((uint64_t)k * 0x9E3779B97F4A7C15ULL) ^ ((uint64_t)a->count[k] * 0xD1B54A32D192ED03ULL), (uint64_t)k);
            Sampler sm = sampler_make(&rng, x, y, (uint32_t)a->count[k], 0, tp.sampler);
            real jx, jy;
            sampler_2d(&sm, SAMPLE_CAMERA, &jx, &jy);
            real u = (x + jx) / a->rw;
            real v = (y + jy) / a->rh;
            Vec3 c = trace_path_sampled(s, camera_ray(cam, u, v), &sm, &tp);
            if (!(c.x == c.x && c.y == c.y && c.z == c.z)) c = vec3(0, 0, 0);
            if (clamp > 0 && v_max(c) > clamp) c = v_scale(c, clamp / v_max(c));
            a->sum[k] = v_add(a->sum[k], c);
            a->count[k]++;
        }
    }
}

// sim mode: blend the pass just rendered into the running average and show that
typedef struct { Vec3 *px; int n, scale; } Ema;

static void ema_update(Ema *e, const Accum *a, Framebuffer *fb, int restart){
    if (restart || e->scale != a->scale) e->n = 0;
    e->scale = a->scale;
    real w = e->n < 2 ? 1.0f / (e->n + 1) : 1.0f / 3;
    e->n++;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < a->rw * a->rh; k++){
        Vec3 c = a->count[k] ? v_scale(a->sum[k], 1.0f / a->count[k]) : vec3(0, 0, 0);
        e->px[k] = v_add(v_scale(e->px[k], 1 - w), v_scale(c, w));
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < a->H; y++)
        for (int x = 0; x < a->W; x++)
            fb->color[y * a->W + x] = image_pack_srgb(e->px[(y / a->scale) * a->rw + x / a->scale]);
}

// average -> sRGB -> framebuffer. Pixels without samples yet keep their old color, so a new
// pass "paints over" the previous image (or the raster preview) instead of flashing black.
static void accum_to_fb(const Accum *a, Framebuffer *fb){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < a->H; y++){
        int iy = y / a->scale;
        for (int x = 0; x < a->W; x++){
            int k = iy * a->rw + x / a->scale;
            int n = a->count[k];
            if (n == 0) continue;
            fb->color[y * a->W + x] = image_pack_srgb(v_scale(a->sum[k], 1.0 / n));
        }
    }
}

static void save_accum(const Accum *a, const char *path){
    Image img = image_create(a->rw, a->rh);
    for (int k = 0; k < a->rw * a->rh; k++)
        img.px[k] = a->count[k] ? v_scale(a->sum[k], 1.0 / a->count[k]) : vec3(0, 0, 0);
    if (image_write_ppm(&img, path) == 0)
        printf("saved %s (%dx%d, path traced, %d samples/pixel)\n", path, a->rw, a->rh, a->passes);
    else
        fprintf(stderr, "could not write %s (does the folder exist?)\n", path);
    image_free(&img);
}

static void save_fb(const Framebuffer *fb, const char *path, const char *what){
    if (fb_save_ppm(fb, path) == 0) printf("saved %s (%dx%d, %s)\n", path, fb->w, fb->h, what);
    else fprintf(stderr, "could not write %s (does the folder exist?)\n", path);
}

// Saves what the window shows: the path traced image at full precision once it has at least
// one full pass, otherwise the window's pixels (a raster frame, or a preview being refined).
static void save_view(const Accum *a, const Framebuffer *fb, int showing_raster, RenderMode shading,
                      const char *prefix, int *shot_no){
    char path[512], what[64];
    snprintf(path, sizeof(path), "%s_%03d.ppm", prefix, ++*shot_no);
    if (!showing_raster && a->passes > 0 && a->scale == 1){ save_accum(a, path); return; }
    if (showing_raster) snprintf(what, sizeof(what), "rasterized, %s", render_mode_name(shading));
    else snprintf(what, sizeof(what), "path tracing preview, first pass not finished");
    save_fb(fb, path, what);
}

// ---- the viewer's state ---------------------------------------------------------------
//
// One iteration of the loop is frame(). Natively main calls it until it returns 0; in a web
// browser (Emscripten) the browser calls it once per display refresh instead, since a page
// must not run an endless loop. So everything the loop keeps between iterations lives here.

typedef struct { int fwd, back, left, right, up, down, tl, tr, tu, td, fast, drag, mx, my; } Input;

static Options o;
static Scene scene;
static Sim sim;
static int W, H;
static LrWindow *win;
static Framebuffer *fb;
static Accum acc;
static Ema ema;
static RasterVolumes *rvol;            // the rasterizer's clouds (light grids)
static View view, home;
static Backend mode;
static RenderMode shading;
static int max_depth, preview, preview_on;
static double speed;
static AccelKind accel;
static Input in;
static int running = 1, shot_no = 0, band = 4, force_present = 1;
static int reset = 1;                  // path tracer: start over at full resolution
static int raster_dirty = 1;           // rasterizer: draw a new frame
static int showing_raster = 1;         // what the window shows right now
static double t_start, t_prev, last_present, last_title, last_change;
static long frames;          // frames shown (for -autoquit's frame rate)

// where a frame's time goes on the CPU (printed by -autoquit), after the first second
enum { PROF_SIM, PROF_ACCEL, PROF_MOVE, PROF_MORE, PROF_WAIT, PROF_RASTER, PROF_PRESENT, PROF_FRAME, PROF_N };
static double prof[PROF_N];
static long prof_frames;
static int prof_on;
#define PROF(k, stmt) do { double p0_ = now_s(); stmt; if (prof_on) prof[k] += now_s() - p0_; } while (0)
static void prof_print(void){
    if (prof_frames <= 0) return;
    double s = 1000.0 / prof_frames, known = 0;
    for (int k = 0; k < PROF_FRAME; k++) known += prof[k];
    printf("per frame, CPU ms: physics %.2f | scene trees %.2f | GPU scene update %.2f | trace submit %.2f | "
           "waiting for the GPU %.2f | raster %.2f | present %.2f | other %.2f | total %.2f (%ld frames)\n",
           prof[PROF_SIM] * s, prof[PROF_ACCEL] * s, prof[PROF_MOVE] * s, prof[PROF_MORE] * s, prof[PROF_WAIT] * s,
           prof[PROF_RASTER] * s, prof[PROF_PRESENT] * s, (prof[PROF_FRAME] - known) * s, prof[PROF_FRAME] * s, prof_frames);
}
static double pass_t0, pass_ms, raster_ms;
static int sim_paused = 0, sim_moved = 0;
static double sim_last;
// the GPU (-gpu on, the default in a browser): frames are shown and clouds raymarched there
static Gpu *gpu;
static GpuView *gview;
static GpuFluid *gfluid;     // -sim on: the air and the cloud stepped on the GPU
static GpuRigid *grigid;     // -sim on: the balls and boxes stepped on the GPU

// -sim on with the GPU: the air and the bodies go there (again, after a restart)
static void gpu_sim_start(void){
    if (!gview || !o.sim.on) return;
    if (o.gpu_fluid && !gfluid && (gfluid = gpu_fluid_create(gpu, &sim.pw)))
        gpu_fluid_attach_view(gfluid, gview);
    if (o.gpu_rigid && !grigid) grigid = gpu_rigid_create(gpu, &sim.pw);
}
static void gpu_sim_stop(void){
    gpu_fluid_destroy(gfluid);
    gfluid = NULL;
    gpu_rigid_destroy(grigid);
    grigid = NULL;
}
static int gpu_scene;        // the raster frame being shown is drawn by the GPU rasterizer (fb is not)
static int gpu_tracing;      // the path traced image being made is the GPU's (acc is not used)
static Camera gpu_cam;       // and its camera, and bounces
static int gpu_depth;
static int gpu_scene_moved;  // the physics moved the scene since that image began
static int gpu_shown_passes; // its samples per pixel when it was last shown
static Framebuffer *gfb;     // -gpu offscreen: the GPU's frame, back on the CPU to be shown
static int gpu_frame_owed;   // offscreen: a frame was drawn but could not be read back yet
#ifdef __EMSCRIPTEN__
static int want_shot;        // a screenshot was asked for (1), to download too (2)
#endif
static RenderParams last_rp;           // the camera of the last raster frame, for the GPU march
static int frame_volumes;              // the frame on screen still needs its media drawn

#ifdef __EMSCRIPTEN__
// A screenshot of what the GPU shows (the canvas cannot be read back directly): the pixels
// become a PNG, kept in window.lrCapture, and downloaded when asked for with the P key.
EM_JS(void, js_capture_done, (const uint32_t *px, int w, int h, int download), {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(w, h);
    const src = HEAPU32.subarray(px >> 2, (px >> 2) + w * h);
    for (let i = 0; i < w * h; i++) {
        const v = src[i];
        img.data[4 * i] = (v >> 16) & 255; img.data[4 * i + 1] = (v >> 8) & 255;
        img.data[4 * i + 2] = v & 255; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    window.lrCapture = c.toDataURL('image/png');
    if (download) {
        const a = document.createElement('a');
        a.href = window.lrCapture; a.download = 'little_renderer.png'; a.click();
    }
});
static void do_capture(int download);
// callable from JavaScript: Module._lr_capture() (the result lands in window.lrCapture)
EMSCRIPTEN_KEEPALIVE void lr_capture(void);
#endif

static void on_gpu_frame(void *user, const uint32_t *px, int w, int h);

#ifndef __EMSCRIPTEN__
// Natively, a screenshot of what the GPU shows (P, -autoquit): the CPU's framebuffer does not
// have it. The path traced image is read at full precision; anything else is the next frame.
static char shot_path[512];
static int shot_waiting;      // one is on its way
static int shot_started;      // -autoquit: it was asked for

static void save_traced(void *user, const float *sc, int w, int h){
    (void)user;
    Image img = image_create(w, h);
    for (int i = 0; i < w * h; i++){
        float c = sc[4 * i + 3];
        img.px[i] = c > 0 ? vec3(sc[4 * i] / c, sc[4 * i + 1] / c, sc[4 * i + 2] / c) : vec3(0, 0, 0);
    }
    if (image_write_ppm(&img, shot_path) == 0)
        printf("saved %s (%dx%d, path traced on the GPU, %d samples/pixel%s)\n", shot_path, w, h,
               gpu_view_trace_passes(gview), o.denoise ? ", denoised" : "");
    else fprintf(stderr, "could not write %s (does the folder exist?)\n", shot_path);
    image_free(&img);
    shot_waiting = 0;
}

static void gpu_screenshot(void){
    snprintf(shot_path, sizeof(shot_path), "%s_%03d.ppm", o.out, ++shot_no);
    shot_waiting = 1;
    if (!showing_raster && gpu_tracing && gpu_view_trace_passes(gview) > 0){
        if (!gpu_view_trace_read(gview, save_traced, NULL)) shot_waiting = 0;
        return;
    }
    if (!gpu_offscreen(gpu)) gpu_view_capture(gview, on_gpu_frame, NULL);   // offscreen: every frame comes back
    force_present = 1;
}
#endif

// a frame read back from the GPU: shown (offscreen), and kept as a screenshot if one was asked
static void on_gpu_frame(void *user, const uint32_t *px, int w, int h){
    (void)user;
#ifndef __EMSCRIPTEN__
    if (shot_waiting && shot_path[0]){
        Framebuffer shot = {w, h, (uint32_t *)px, NULL};
        if (fb_save_ppm(&shot, shot_path) == 0) printf("saved %s (%dx%d, drawn by the GPU)\n", shot_path, w, h);
        else fprintf(stderr, "could not write %s (does the folder exist?)\n", shot_path);
        shot_path[0] = 0;
        shot_waiting = 0;
    }
#endif
    if (gpu_offscreen(gpu) && win){
        if (!gfb) gfb = fb_create(w, h);
        if (gfb && gfb->w == w && gfb->h == h){
            memcpy(gfb->color, px, (size_t)w * h * sizeof(uint32_t));
            window_present(win, gfb);
        }
    }
#ifdef __EMSCRIPTEN__
    if (want_shot){
        js_capture_done(px, w, h, want_shot == 2);
        printf("screenshot: %dx%d\n", w, h);
        want_shot = 0;
    }
#endif
}

// wait a little when there is nothing to do (natively; the browser paces the frames itself)
static void idle(int ms){
#ifdef __EMSCRIPTEN__
    (void)ms;
#else
    struct timespec ts = {0, (long)ms * 1000 * 1000};
    nanosleep(&ts, NULL);
#endif
}

// A new path traced image on the GPU, for the camera as it is now. 0 = this GPU cannot hold
// the scene: the CPU path traces from now on.
static int gpu_trace_begin(const Camera *cam, Vec3 at, int max_depth){
    GpuTraceParams tp = {*cam, max_depth, o.mis, (real)o.clamp, !o.sampler, o.caustics_lt ? (real)o.photons : 0, o.photon_cache};
    if (!gpu_view_trace_start(gview, &scene, &tp)){
        o.gpu_trace = 0;
        gpu_tracing = 0;
        return 0;
    }
    // what shows through until every pixel has a sample: the rasterized scene, same camera
    last_rp = render_params_camera(view.pos, at, vec3(0, 1, 0), view.fov, W, H);
    last_rp.mode = shading;
    gpu_tracing = 1;
    gpu_cam = *cam;
    gpu_depth = max_depth;
    gpu_shown_passes = -1;
    force_present = 1;
    return 1;
}

// Samples per pixel the GPU path tracer keeps from one frame to the next while things move
// (gpu_view_trace_move): the camera alone changes nothing on the surfaces, so they keep
// many (-history, 32); with the physics running the light changes too (shadows, the cloud),
// so fewer, and a change shows within a few frames.
#define GPU_HISTORY_PHYSICS 12

// rt mode with the GPU: path traced every frame, whatever moves. A new camera or the
// physics' new state carry the image over (each pixel keeps the samples of its surface,
// then the denoiser smooths what is new); other changes (bounces, a reset) start over.
// 0 = this GPU cannot hold the scene: the CPU path traces from now on.
static int gpu_trace_frame(const Camera *cam, Vec3 at, int max_depth){
    gpu_scene_moved |= sim_moved;
    if (!gpu_tracing || reset || max_depth != gpu_depth){
        if (!gpu_trace_begin(cam, at, max_depth)) return 0;
        gpu_scene_moved = 0;
    } else if ((gpu_scene_moved || memcmp(cam, &gpu_cam, sizeof(*cam)) != 0) && !gpu_view_trace_busy(gview)){
        // (a slow GPU still busy with a band: the move waits for it, and the frame shows the
        // last image as it is, with the raster of its own camera, not a newer one)
        GpuTraceParams tp = {*cam, max_depth, o.mis, (real)o.clamp, !o.sampler, o.caustics_lt ? (real)o.photons : 0, o.photon_cache};
        int keep = gpu_scene_moved && o.history > GPU_HISTORY_PHYSICS ? GPU_HISTORY_PHYSICS : o.history;
        gpu_scene_moved = 0;
        int moved;
        PROF(PROF_MOVE, moved = gpu_view_trace_move(gview, &scene, &tp, keep));
        if (!moved){
            o.gpu_trace = 0;
            gpu_tracing = 0;
            return 0;
        }
        // under the pixels that were hidden until now (no sample yet): the scene rasterized
        last_rp = render_params_camera(view.pos, at, vec3(0, 1, 0), view.fov, W, H);
        last_rp.mode = shading;
        gpu_cam = *cam;
        force_present = 1;
    }
    reset = 0;
    if (gpu_view_trace_passes(gview) < o.spp) PROF(PROF_MORE, gpu_view_trace_more(gview, &scene, &last_rp, &o.march));
    else idle(10);
    // natively: sleep until the band is done (bands are sized to take < ~45 ms). A timed
    // nap would not do: on Windows the shortest one lasts ~15 ms, longer than a band.
    if (gpu_view_trace_busy(gview)) PROF(PROF_WAIT, gpu_wait(gpu));
    pass_ms = gpu_view_trace_pass_ms(gview);
    return 1;
}

// one iteration: input, camera, physics, render, show. 0 = quit.
static int frame(void){
    const double frame_t0 = now_s();
    // ---- 0. the GPU starts asynchronously: wait for it (or fall back to the CPU)
    if (gpu && !gview){
        GpuState gs = gpu_state(gpu);
        if (gs == GPU_PENDING) return 1;
        if (gs == GPU_READY) gview = gpu_view_create(gpu, W, H);
        if (!gview){
            fprintf(stderr, "GPU not used (%s): the CPU draws everything\n", gpu_message(gpu));
            gpu_destroy(gpu);
            gpu = NULL;
        } else {
            printf("GPU: frames shown and clouds raymarched on %s%s\n", gpu_message(gpu),
                   gpu_offscreen(gpu) ? " (offscreen, read back to the canvas)" : "");
            gpu_view_trace_denoise(gview, o.denoise);
            gpu_view_trace_wavefront(gview, o.wavefront);
            if (o.gpu_trace < 0){
                o.gpu_trace = !gpu_software(gpu);
                if (!o.gpu_trace) printf("GPU: a software GPU: the CPU path traces (faster here; -gputrace on to try it anyway)\n");
            }
            gpu_sim_start();
            if (gfluid) printf("GPU: the air (%dx%dx%d cells) and the cloud are simulated there\n",
                               sim.pw.fluid->nx, sim.pw.fluid->ny, sim.pw.fluid->nz);
            if (grigid) printf("GPU: the balls and boxes are simulated there\n");
        }
        force_present = raster_dirty = 1;
    } else if (gview && gpu_state(gpu) == GPU_FAILED){
        // the browser took the GPU away: carry on with the CPU
#ifdef __EMSCRIPTEN__
        fprintf(stderr, "GPU lost (%s): the CPU draws everything from now on%s\n", gpu_message(gpu),
                gpu_offscreen(gpu) ? "" : " (if the image no longer changes, reload with ?gpu=off or ?gpu=offscreen)");
#else
        fprintf(stderr, "GPU lost (%s): the CPU draws everything from now on\n", gpu_message(gpu));
#endif
        gpu_sim_stop();              // the air and the bodies go back to the CPU
        gpu_scene = gpu_tracing = 0;
        reset = 1;
        gpu_view_destroy(gview);
        gview = NULL;
        gpu_destroy(gpu);
        gpu = NULL;
        gpu_frame_owed = 0;
        force_present = raster_dirty = 1;
    }

    // ---- 1. input
    int changed = 0;
    LrEvent ev;
    while (window_poll(win, &ev)){
        if (ev.type == LR_KEYDOWN || ev.type == LR_KEYUP){
            int down = ev.type == LR_KEYDOWN;
            switch (ev.key){
            case 'w': in.fwd = down; break;
            case 's': in.back = down; break;
            case 'a': in.left = down; break;
            case 'd': in.right = down; break;
            case 'e': in.up = down; break;
            case 'q': in.down = down; break;
            case LR_KEY_LEFT: in.tl = down; break;
            case LR_KEY_RIGHT: in.tr = down; break;
            case LR_KEY_UP: in.tu = down; break;
            case LR_KEY_DOWN: in.td = down; break;
            case LR_KEY_SHIFT: in.fast = down; break;
            default: break;
            }
            if (!down) continue;
            switch (ev.key){
            case LR_KEY_ESCAPE: running = 0; break;
            case 'r': view = home; changed = 1; break;
            case 'p':
#ifdef __EMSCRIPTEN__
                if (gview){ do_capture(1); break; }   // a PNG download
#else
                if (gview){ if (!shot_waiting) gpu_screenshot(); break; }
#endif
                save_view(&acc, fb, showing_raster, shading, o.out, &shot_no);
                break;
            case 'c': {
                Options cur = o;
                cur.max_depth = max_depth;
                printf("this view: ");
                options_print_command(&cur, view.pos, v_add(view.pos, v_scale(view_forward(&view), 500)), view.fov, stdout);
                fflush(stdout);
                break;
            }
            case 'm':
                mode = mode == BACKEND_HYBRID ? BACKEND_RT : (mode == BACKEND_RT ? BACKEND_RASTER : BACKEND_HYBRID);
                printf("mode: %s\n", backend_name(mode));
                fflush(stdout);
                changed = 1;
                break;
            case 'v':
                shading = shading == MODE_PHONG ? MODE_GOURAUD : (shading == MODE_GOURAUD ? MODE_FLAT :
                          (shading == MODE_FLAT ? MODE_WIREFRAME : MODE_PHONG));
                printf("raster shading: %s\n", render_mode_name(shading));
                fflush(stdout);
                raster_dirty = 1;
                if (mode == BACKEND_HYBRID) changed = 1;   // show the new shading now
                break;
            case 'b': {
                accel = (AccelKind)((accel + 1) % 3);
                printf("building %s...\n", accel_name(accel));
                fflush(stdout);
                scene_set_accel(&scene, accel, &o.ap);
                char buf[256];
                scene_accel_stats(&scene, buf, sizeof(buf));
                printf("%s\n", buf);
                if (mode != BACKEND_RASTER) changed = 1;
                break;
            }
            case 'x':
                if (!gview) break;
                o.denoise = !o.denoise;
                printf("denoiser: %s\n", gpu_view_trace_denoise(gview, o.denoise) ? "on" : "off");
                fflush(stdout);
                gpu_shown_passes = -1;
                force_present = 1;
                break;
            case 'f': preview_on = !preview_on; break;
            case '1': preview = 1; preview_on = 0; break;
            case '2': preview = 2; preview_on = 1; break;
            case '3': preview = 4; preview_on = 1; break;
            case '4': preview = 8; preview_on = 1; break;
            case '[': if (max_depth > 1){ max_depth--; changed = 1; } break;
            case ']': max_depth++; changed = 1; break;
            case '+': case '=': speed *= 1.5; break;
            case '-': speed /= 1.5; break;
            case 'h': printf("\n%s\n", CONTROLS); if (o.sim.on) printf("%s\n", SIM_CONTROLS); break;
            case ' ':
                if (o.sim.on){ sim_paused = !sim_paused; printf(sim_paused ? "paused\n" : "running\n"); fflush(stdout); }
                break;
            case 'k': if (o.sim.on){ sim_blast_now(&sim); printf("boom\n"); fflush(stdout); } break;
            case 'n': if (o.sim.on) sim_drop_ball(&sim, &scene); break;
            case 't':
                if (o.sim.on){
                    gpu_sim_stop();
                    sim_free(&sim);
                    scene_free(&scene);
                    memset(&scene, 0, sizeof(scene));
                    if (sim_setup(&sim, &o, &scene) != 0) return 0;
                    sim.max_dt = 1.0 / 30;
                    gpu_sim_start();
                    sim_step(&sim, &scene, 0);
                    scene_set_accel(&scene, accel, &o.ap);
                    changed = 1;
                }
                break;
            default: break;
            }
        } else if (ev.type == LR_BUTTONDOWN){
            if (ev.button == 1){ in.drag = 1; in.mx = ev.x; in.my = ev.y; }
        } else if (ev.type == LR_BUTTONUP){
            if (ev.button == 1) in.drag = 0;
        } else if (ev.type == LR_WHEEL){
            if (ev.wheel > 0 && view.fov > 5){ view.fov -= 2; changed = 1; }     // wheel up: zoom in
            if (ev.wheel < 0 && view.fov < 120){ view.fov += 2; changed = 1; }   // wheel down: zoom out
        } else if (ev.type == LR_MOTION && in.drag){
            int dx = ev.x - in.mx, dy = ev.y - in.my;
            in.mx = ev.x;
            in.my = ev.y;
            if (dx || dy){
                view.yaw -= dx * 0.004;      // drag right -> turn right
                view.pitch -= dy * 0.004;    // drag up -> look up
                changed = 1;
            }
        } else if (ev.type == LR_CLOSE){
            running = 0;
        } else if (ev.type == LR_EXPOSE){
            force_present = 1;
        }
    }

    // ---- 2. move the camera with the held keys (frame-rate independent: * dt)
    double t = now_s(), dt = fmin(t - t_prev, 0.1);
    t_prev = t;
    Vec3 f = view_forward(&view), r = view_right(&view), mv = vec3(0, 0, 0);
    if (in.fwd) mv = v_add(mv, f);
    if (in.back) mv = v_sub(mv, f);
    if (in.right) mv = v_add(mv, r);
    if (in.left) mv = v_sub(mv, r);
    if (in.up) mv = v_add(mv, vec3(0, 1, 0));
    if (in.down) mv = v_sub(mv, vec3(0, 1, 0));
    if (v_dot(mv, mv) > 0){
        view.pos = v_add(view.pos, v_scale(v_norm(mv), speed * (in.fast ? 4 : 1) * dt));
        changed = 1;
    }
    double turn = 1.5 * dt;
    if (in.tl){ view.yaw += turn; changed = 1; }
    if (in.tr){ view.yaw -= turn; changed = 1; }
    if (in.tu){ view.pitch += turn; changed = 1; }
    if (in.td){ view.pitch -= turn; changed = 1; }
    view.pitch = fmax(-1.55, fmin(1.55, view.pitch));
    if (o.orbit != 0){
        // -orbit: swing to and fro around the point the start camera looks at, at a constant speed
        static double phase;
        phase += o.orbit * dt;
        double p = fmod(phase, 100.0), deg = p < 50 ? p - 25 : 75 - p;
        double a = deg * PI / 180, ca = cos(a), sa = sin(a);
        Vec3 c = scene.cam_at, d = v_sub(home.pos, c);
        view.pos = v_add(c, vec3(d.x * ca + d.z * sa, d.y, -d.x * sa + d.z * ca));
        Vec3 fw = v_norm(v_sub(c, view.pos));
        view.yaw = atan2(fw.x, fw.z);
        view.pitch = asin(fmax(-1.0, fmin(1.0, fw.y)));
        changed = 1;
    }
    if (changed){ last_change = t; raster_dirty = 1; }

    // ---- 2b. physics: advance by the time that passed, then the image starts over
    if (o.sim.on){
        double sdt = fmin(t - sim_last, 0.05);
        sim_last = t;
        sim_moved = 0;
        if (!sim_paused && sdt > 0){
            PROF(PROF_SIM, sim_step(&sim, &scene, sdt));
            PROF(PROF_ACCEL, scene_update_instances(&scene, &o.ap));   // the bodies are instances: the world stays
            sim_moved = 1;
            raster_dirty = 1;
        }
    }

    f = view_forward(&view);
    Vec3 at = v_add(view.pos, f);
    Camera cam = camera_make(view.pos, at, vec3(0, 1, 0), view.fov, (real)W / H);

    // ---- 3. render
    // hybrid: rasterize while the camera moves, path trace once it has settled
    int use_raster = mode == BACKEND_RASTER ||
                     (mode == BACKEND_HYBRID && (changed || t - last_change < HYBRID_SETTLE_S ||
                                                 (o.sim.on && !sim_paused)));
    // the CPU path tracer reads the cloud on the CPU: bring the GPU's back (it lands a frame later)
    // (the GPU path tracer takes it straight from the GPU)
    if (gfluid && !use_raster && !(gview && o.gpu_trace > 0)) gpu_fluid_sync_cloud(gfluid);
    if (use_raster){
        if (raster_dirty){
            double r0 = now_s();
            RenderParams p = render_params_camera(view.pos, at, vec3(0, 1, 0), view.fov, W, H);
            p.mode = shading;
            // the surfaces: by the GPU rasterizer when it is there (at presentation), otherwise here
            gpu_scene = gview && o.gpu_raster;
            if (!gpu_scene){
                fb_clear(fb, rgb8(0, 0, 0));
                render_scene(fb, &scene, &p);
            }
            // the media: on the GPU when it is there (at presentation), otherwise here
            if (o.raster_clouds && !gview) raster_draw_volumes(rvol, fb, &scene, &p, &o.march);
            last_rp = p;
            frame_volumes = o.raster_clouds && gview && scene.nvolumes > 0;
            raster_ms = (now_s() - r0) * 1000.0;
            if (prof_on) prof[PROF_RASTER] += raster_ms / 1000.0;
            raster_dirty = 0;
            force_present = 1;
        } else {
            idle(5);                                     // nothing new to draw
        }
        // the path tracer starts over when it takes back over
        if (acc.passes || acc.band_y || acc.scale != 1) accum_reset(&acc, 1);
        gpu_tracing = 0;
        reset = 0;
        pass_t0 = now_s();
    } else if (gview && o.gpu_trace > 0 && gpu_trace_frame(&cam, at, max_depth)){
        // done: the GPU path traces (every frame, the camera or the bodies moving or not)
    } else if (sim_moved){
        // the simulation moved: one pass of the new state, blended with the last ones
        int sc = preview_on ? preview : 1;
        accum_reset(&acc, sc);
        pass_t0 = now_s();
        for (int i = 0; i < o.move_spp; i++) render_rows(&scene, &cam, &acc, 0, acc.rh, max_depth, o.mis, (real)o.clamp, o.sampler);
        acc.passes = o.move_spp;
        pass_ms = (now_s() - pass_t0) * 1000.0;
        ema_update(&ema, &acc, fb, changed);
        force_present = 1;
        gpu_tracing = 0;
        reset = 1;   // once paused: start refining at full resolution
    } else if (changed || reset){
        // rt mode: start over. If the camera moved, render one whole pass right away
        // (cheap at 1/preview resolution) so the image follows the input.
        int sc = changed && preview_on && mode == BACKEND_RT ? preview : 1;
        accum_reset(&acc, sc);
        reset = 0;
        pass_t0 = now_s();
        if (changed){
            int n = sc > 1 ? o.move_spp : 1;          // -movespp: less grain, slower
            for (int i = 0; i < n; i++) render_rows(&scene, &cam, &acc, 0, acc.rh, max_depth, o.mis, (real)o.clamp, o.sampler);
            acc.passes = n;
            pass_ms = (now_s() - pass_t0) * 1000.0;
            pass_t0 = now_s();
            accum_to_fb(&acc, fb);
            force_present = 1;
            if (sc > 1) reset = 1;  // next calm frame: back to full resolution
        }
    } else if (acc.passes < o.spp){
        // calm: refine at full resolution, a band of rows at a time (~20 ms each)
        double b0 = now_s();
        int y1 = acc.band_y + band < acc.rh ? acc.band_y + band : acc.rh;
        render_rows(&scene, &cam, &acc, acc.band_y, y1, max_depth, o.mis, (real)o.clamp, o.sampler);
        double bt = now_s() - b0;
        if (bt < 0.010 && band < acc.rh) band *= 2;
        else if (bt > 0.040 && band > 1) band /= 2;
        acc.band_y = y1;
        if (acc.band_y >= acc.rh){
            acc.band_y = 0;
            acc.passes++;
            pass_ms = (now_s() - pass_t0) * 1000.0;
            pass_t0 = now_s();
            accum_to_fb(&acc, fb);
            force_present = 1;
        }
    } else {
        idle(10);                                     // done refining
    }

    // ---- 4. show it
    double tn = now_s();
    const int gpu_traced = !use_raster && gpu_tracing;
    const int gpu_passes = gpu_traced ? gpu_view_trace_passes(gview) : 0;
    int refining = !use_raster && !sim_moved &&     // bands land on screen ~30x per second
                   (gpu_traced ? gpu_passes < o.spp || gpu_passes != gpu_shown_passes : acc.passes < o.spp);
    double present0 = now_s();
    if (force_present || (refining && tn - last_present > 0.033)){
        if (!use_raster && !sim_moved && !gpu_traced) accum_to_fb(&acc, fb);
        if (gview){
            // offscreen the frame must come back to be seen: ask for it. If the previous one
            // is still on its way, the GPU is behind: draw nothing now (no queue of frames
            // waiting for it), and try again next time so the last image is not lost.
            gpu_frame_owed = gpu_offscreen(gpu) && !gpu_view_capture(gview, on_gpu_frame, NULL);
            if (gpu_frame_owed){
                // (nothing drawn)
            } else if (gpu_traced){
                gpu_view_present_traced(gview, &scene, &last_rp, &o.march, o.raster_clouds && scene.nvolumes > 0);
                gpu_shown_passes = gpu_passes;
            } else if (use_raster && gpu_scene){
                if (!gpu_view_present_scene(gview, &scene, &last_rp, &o.march, frame_volumes)){
                    // this GPU cannot draw the scene: the CPU rasterizes from now on
                    o.gpu_raster = 0;
                    gpu_scene = 0;
                    raster_dirty = 1;
                    fb_clear(fb, rgb8(0, 0, 0));
                    render_scene(fb, &scene, &last_rp);
                    gpu_view_present(gview, fb, &scene, &last_rp, &o.march, frame_volumes);
                }
            } else gpu_view_present(gview, fb, &scene, &last_rp, &o.march, use_raster && frame_volumes);
        } else window_present(win, fb);
        showing_raster = use_raster;
        last_present = tn;
        frames++;
        force_present = gpu_frame_owed;
    }
    if (prof_on){
        prof[PROF_PRESENT] += now_s() - present0;
        prof[PROF_FRAME] += now_s() - frame_t0;
        prof_frames++;
    }
    prof_on = now_s() - t_start > 1.0;
    if (tn - last_title > 0.5){
        char title[512];
        if (use_raster)
        {
            double ml = 0, mm = 0;
            raster_volumes_timing(rvol, &ml, &mm);
            if (scene.nvolumes && o.raster_clouds)
                snprintf(title, sizeof(title), "view | %s | raster %s %.1f ms/frame (cloud: light %.0f + march %.0f ms) | speed %.0f",
                         backend_name(mode), render_mode_name(shading), raster_ms, ml, mm, speed);
            else
                snprintf(title, sizeof(title), "view | %s | raster %s %.1f ms/frame | speed %.0f",
                         backend_name(mode), render_mode_name(shading), raster_ms, speed);
            if (gpu_scene){
                size_t l = strlen(title);
                snprintf(title + l, sizeof(title) - l, " | triangles on the GPU");
            }
        }
        else if (gpu_tracing)
            snprintf(title, sizeof(title), "view | %s | path tracing on the GPU%s | %d/%d spp | %.0f ms/pass | %d bounces | speed %.0f",
                     backend_name(mode), o.denoise ? ", denoised" : "", gpu_view_trace_passes(gview), o.spp, pass_ms,
                     max_depth, speed);
        else
            snprintf(title, sizeof(title), "view | %s | path tracing %s | %d/%d spp | %.0f ms/pass | 1/%d res | %d bounces | speed %.0f",
                     backend_name(mode), accel_name(accel), acc.passes, o.spp, pass_ms, acc.scale, max_depth, speed);
        if (o.sim.on){
            char st[200];
            sim_status(&sim, st, sizeof(st));
            size_t l = strlen(title);
            snprintf(title + l, sizeof(title) - l, " | %s%s", st, sim_paused ? " | paused" : "");
        }
        if (gview){
            size_t l = strlen(title);
            snprintf(title + l, sizeof(title) - l, " | GPU%s %s", gpu_offscreen(gpu) ? " (offscreen)" : "", gpu_message(gpu));
        }
        window_set_title(win, title);
        last_title = tn;
    }
#ifndef __EMSCRIPTEN__
    if (o.autoquit > 0 && tn - t_start > o.autoquit && gview){
        // the GPU's image: ask for it, and quit once it is saved
        if (!shot_started){
            shot_started = 1;
            printf("autoquit after %.1f s (%s, %.0f frames/s): ", tn - t_start, backend_name(mode), frames / (tn - t_start));
            if (o.sim.on && !sim_paused){ char st[200]; sim_status(&sim, st, sizeof(st)); printf("%s\n", st); }
            else if (showing_raster) printf("raster frame on the GPU\n");
            else printf("%d samples/pixel on the GPU, last pass %.1f ms\n", gpu_view_trace_passes(gview), pass_ms);
            prof_print();
            if (o.orbit != 0){      // the camera of the screenshot, to render it again
                Options cur = o;
                cur.max_depth = max_depth;
                printf("this view: ");
                options_print_command(&cur, view.pos, v_add(view.pos, v_scale(view_forward(&view), 500)), view.fov, stdout);
            }
            gpu_screenshot();
        }
        if (!shot_waiting) running = 0;
        return running;
    }
#endif
    if (o.autoquit > 0 && tn - t_start > o.autoquit){
        printf("autoquit after %.1f s (%s, %.0f frames/s): ", tn - t_start, backend_name(mode), frames / (tn - t_start));
        if (o.sim.on && !sim_paused){ char st[200]; sim_status(&sim, st, sizeof(st)); printf("%s, last pass %.0f ms\n", st, pass_ms); }
        else if (showing_raster) printf("last raster frame %.1f ms\n", raster_ms);
        else printf("%d samples/pixel, last pass %.0f ms\n", acc.passes, pass_ms);
        prof_print();
        save_view(&acc, fb, showing_raster, shading, o.out, &shot_no);
        running = 0;
    }
    return running;
}

#ifdef __EMSCRIPTEN__
static void do_capture(int download){
    if (!gview) return;
    want_shot = download ? 2 : 1;
    if (!gpu_offscreen(gpu)) gpu_view_capture(gview, on_gpu_frame, NULL);   // offscreen every frame comes back
    force_present = 1;
}
void lr_capture(void){ do_capture(0); }
#endif

static void shutdown_viewer(void){
    window_close(win);
    fb_destroy(fb);
    free(acc.sum);
    free(acc.count);
    free(ema.px);
    raster_volumes_free(rvol);
    gpu_sim_stop();
    gpu_view_destroy(gview);
    gpu_destroy(gpu);
    fb_destroy(gfb);
    if (o.sim.on) sim_free(&sim);
    scene_free(&scene);
}

#ifdef __EMSCRIPTEN__
static void frame_web(void){
    if (!frame()){ shutdown_viewer(); emscripten_cancel_main_loop(); }
}
#endif

// ---- main ---------------------------------------------------------------------------

int main(int argc, char **argv){
    options_parse(&o, PROG_VIEW, argc, argv);
    if (o.sim.on){
        if (sim_setup(&sim, &o, &scene) != 0) return 1;
        sim.max_dt = 1.0 / 30;                     // one physics step per frame
        sim_step(&sim, &scene, 0);                 // the bodies that start at t = 0
        scene_set_accel(&scene, options_resolve_accel(&o, &scene), &o.ap);
    } else if (options_setup_scene(&o, &scene) != 0) return 1;
    srgb_lut_init();

    W = o.width;
    H = o.height;
    win = window_open(W, H, "view");
    if (o.gpu && win){
#ifdef __EMSCRIPTEN__
        gpu = gpu_create(o.gpu == 2 ? NULL : "#canvas", W, H);
#else
        // natively the GPU draws in the window itself (or, -gpu offscreen, hands the frames back)
        void *dpy;
        uintptr_t wh;
        gpu = o.gpu != 2 && window_native(win, &dpy, &wh) ? gpu_create_window(dpy, wh, W, H) : gpu_create(NULL, W, H);
#endif
    }
    if (!win) return 1;
    fb = fb_create(W, H);
    acc = (Accum){W, H, 1, W, H, calloc((size_t)W * H, sizeof(Vec3)), calloc((size_t)W * H, sizeof(int)), 0, 0};
    if (!fb || !acc.sum || !acc.count){ fprintf(stderr, "out of memory\n"); return 1; }

    printf("\n%s\n", CONTROLS);
    if (o.sim.on) printf("%s\n", SIM_CONTROLS);

    view = view_from_scene(&scene);
    home = view;
    mode = o.backend;
    shading = o.shading;
    max_depth = o.max_depth;
    preview = o.preview;
    preview_on = o.preview > 1;
    speed = o.speed;
    accel = scene_accel_kind(&scene);
    memset(&in, 0, sizeof(in));
    t_start = t_prev = last_change = pass_t0 = sim_last = now_s();
    rvol = raster_volumes_create();
    if (!rvol){ fprintf(stderr, "out of memory\n"); return 1; }
    ema = (Ema){calloc((size_t)W * H, sizeof(Vec3)), 0, 0};
    if (!ema.px){ fprintf(stderr, "out of memory\n"); return 1; }
    if (o.sim.on && o.preview == 4){ preview = 2; }   // -sim: 1/2 resolution by default

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(frame_web, 0, 1);      // the browser calls frame_web; never returns
#else
    while (frame()){}
    shutdown_viewer();
#endif
    return 0;
}
