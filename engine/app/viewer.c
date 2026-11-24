// viewer.c - the real-time viewer (app/viewer.h): fly through a scene with both backends of the
// engine, a frame at a time. bin/view is its program (apps/view.c); the web editor drives it
// through the same functions.
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

#include "app/viewer.h"
#include "app/simdemo.h"
#include "app/hud.h"
#include "lr.h"
#include "gpu/gpu.h"
#include "gpu/view_gpu.h"
#include "gpu/fluid_gpu.h"
#include "gpu/rigid_gpu.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#if defined(__EMSCRIPTEN_PTHREADS__) && defined(_OPENMP)
#include <omp.h>
#endif
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
    "  L                rasterizer: make the light probe again (things moved)\n"
    "  P                save the current image as PNG\n"
    "  I                the frame rate on the image: on / off\n"
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

// ---- camera as position + two angles ------------------------------------------------

Vec3 view_forward(const View *v){
    return vec3(cos(v->pitch) * sin(v->yaw), sin(v->pitch), cos(v->pitch) * cos(v->yaw));
}
Vec3 view_right(const View *v){ return v_norm(v_cross(view_forward(v), vec3(0, 1, 0))); }

View view_from_scene(const Scene *s){
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
                        int sampler, real iclamp){
    TraceParams tp = trace_default_params();
    tp.indirect_clamp = iclamp;
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
    if (image_write_png(&img, path) == 0)
        printf("saved %s (%dx%d, path traced, %d samples/pixel)\n", path, a->rw, a->rh, a->passes);
    else
        fprintf(stderr, "could not write %s (does the folder exist?)\n", path);
    image_free(&img);
}

static void save_fb(const Framebuffer *fb, const char *path, const char *what){
    if (fb_save_png(fb, path) == 0) printf("saved %s (%dx%d, %s)\n", path, fb->w, fb->h, what);
    else fprintf(stderr, "could not write %s (does the folder exist?)\n", path);
}

// Saves what the window shows: the path traced image at full precision once it has at least
// one full pass, otherwise the window's pixels (a raster frame, or a preview being refined).
static void save_view(const Accum *a, const Framebuffer *fb, int showing_raster, RenderMode shading,
                      const char *prefix, int *shot_no){
    char path[512], what[64];
    snprintf(path, sizeof(path), "%s_%03d.png", prefix, ++*shot_no);
    if (!showing_raster && a->passes > 0 && a->scale == 1){ save_accum(a, path); return; }
    if (showing_raster) snprintf(what, sizeof(what), "rasterized, %s", render_mode_name(shading));
    else snprintf(what, sizeof(what), "path tracing preview, first pass not finished");
    save_fb(fb, path, what);
}

// ---- the viewer's state ---------------------------------------------------------------
//
// One iteration of the loop is viewer_frame(). Natively viewer_run calls it until it returns 0;
// in a web browser (Emscripten) the browser calls it once per display refresh instead, since a page
// must not run an endless loop. So everything the loop keeps between iterations lives in one
// Viewer, V; a frame is its steps in order (viewer_gpu, viewer_input, viewer_update,
// viewer_render, viewer_present, viewer_report), each reading and changing V.

typedef struct { int fwd, back, left, right, up, down, tl, tr, tu, td, fast, drag, mx, my; } Input;

// where a frame's time goes on the CPU (printed by -autoquit), after the first second
enum { PROF_SIM, PROF_ACCEL, PROF_MOVE, PROF_MORE, PROF_WAIT, PROF_RASTER, PROF_PRESENT, PROF_FRAME, PROF_N };

typedef struct {
    Options o;
    Scene scene;
    Sim sim;                   // -sim on: the physics
    int W, H;
    LrWindow *win;
    Framebuffer *fb;           // the CPU's frame
    Accum acc;                 // the CPU path tracer's image
    Ema ema;                   // ... and its running average while the physics moves
    RasterVolumes *rvol;       // the rasterizer's clouds (light grids)

    // what is shown, and how (the keys change these)
    View view, home;
    Backend mode;
    RenderMode shading;
    int max_depth, preview, preview_on;
    double speed;
    AccelKind accel;
    Input in;
    int hud_on;
    char hud_last[64];         // the text of hud_px
    uint32_t *hud_px;          // the frame rate drawn over the image
    int hud_w, hud_h;

    // the loop
    int running, shot_no;
    // what a program driving the viewer changes (viewer.h; the defaults are bin/view's)
    int input_off;             // the window's keys and mouse are not the viewer's (the program's own UI)
    int title_off;             // the window's title is not written
    double settle;             // hybrid: path tracing once still this long (s); 0 = HYBRID_SETTLE_S
    int keep_trace;            // hybrid: the path traced image carried over a move, not started over
    int ext_changed;           // the program moved the camera or changed the scene since the last frame
    double fps;                // frames shown per second (at the last title)
    int band;                  // CPU path tracer: rows per step while refining
    int force_present;         // show a frame now
    int reset;                 // path tracer: start over at full resolution
    int raster_dirty;          // rasterizer: draw a new frame
    int showing_raster;        // what the window shows right now
    double t_start, t_prev, last_present, last_title, last_change;
    double orbit_phase;        // -orbit: where the swing is
    long frames;               // frames shown (for -autoquit's frame rate)
    long title_frames;         // frames shown when the title was last written (its frame rate)
    // path tracing passes made (1 sample per pixel of the image, all images: the work done, as
    // frames/s is what is shown), and at the title and the -stats window's start
    double passes_total, cpu_passes, title_passes, stats_passes;
    long passes_seen, stats_frames;
    double stats_t0;           // -stats: frames counted from here (the start left out)
    int stats_written;
    double prof[PROF_N];       // where a frame's time goes, seconds (after the first second)
    long prof_frames;
    int prof_on;
    double pass_t0, pass_ms, raster_ms;
    int sim_paused, sim_moved;
    double sim_acc, sim_pause_at;  // -simfixed on: time not stepped yet; pause once the simulation reaches this (0: no)
    double sim_last;

    // the GPU (-gpu on, the default in a browser): frames are shown and clouds raymarched there
    Gpu *gpu;
    GpuView *gview;
    GpuFluid *gfluid;          // -sim on: the air and the cloud stepped on the GPU
    GpuRigid *grigid;          // -sim on: the balls and boxes stepped on the GPU
    int gpu_scene;             // the raster frame being shown is drawn by the GPU rasterizer (fb is not)
    int gpu_tracing;           // the path traced image being made is the GPU's (acc is not used)
    Camera gpu_cam;            // and its camera, and bounces
    int gpu_depth;
    int gpu_scene_moved;       // the physics moved the scene since that image began
    int gpu_scene_edited;      // an edit (viewer_scene_edited) since the last move
    int edit_pending;          // edits came, the image is to start over once they stop
    double edit_t;             // (when the last one came)
    int gpu_shown_passes;      // its samples per pixel when it was last shown
    long trace_starts, trace_moves;   // (viewer_status: images begun, and carried over a move)
    Framebuffer *gfb;          // -gpu offscreen: the GPU's frame, back on the CPU to be shown
    int gpu_frame_owed;        // offscreen: a frame was drawn but could not be read back yet
    RenderParams last_rp;      // the camera of the last raster frame, for the GPU march
    int frame_volumes;         // the frame on screen still needs its media drawn
    int want_shot;             // web: a screenshot was asked for (1), to download too (2)
    char shot_path[512];       // native: the screenshot of the GPU's image
    int shot_waiting;          // one is on its way
    int shot_started;          // -autoquit: it was asked for
} Viewer;

static Viewer V = {.running = 1, .band = 4, .force_present = 1, .reset = 1, .raster_dirty = 1, .showing_raster = 1,
                   .hud_on = 1};

#define PROF(k, stmt) do { double p0_ = clock_s(); stmt; if (V.prof_on) V.prof[k] += clock_s() - p0_; } while (0)
static void prof_print(void){
    if (V.prof_frames <= 0) return;
    double s = 1000.0 / V.prof_frames, known = 0;
    for (int k = 0; k < PROF_FRAME; k++) known += V.prof[k];
    printf("per frame, CPU ms: physics %.2f | scene trees %.2f | GPU scene update %.2f | trace submit %.2f | "
           "waiting for the GPU %.2f | raster %.2f | present %.2f | other %.2f | total %.2f (%ld frames)\n",
           V.prof[PROF_SIM] * s, V.prof[PROF_ACCEL] * s, V.prof[PROF_MOVE] * s, V.prof[PROF_MORE] * s, V.prof[PROF_WAIT] * s,
           V.prof[PROF_RASTER] * s, V.prof[PROF_PRESENT] * s, (V.prof[PROF_FRAME] - known) * s, V.prof[PROF_FRAME] * s, V.prof_frames);
}

// the frame rate drawn over the image (app/hud.c): the text's pixels, made again with the
// title (twice a second); on the GPU they are blended over each frame by gpu_view_hud
#define HUD_X 8
#define HUD_Y 8

static void hud_update(double fps){
    char text[64];
    snprintf(text, sizeof(text), "%.0f fps  %.1f ms", fps, fps > 0 ? 1000.0 / fps : 0.0);
    if (V.hud_px && !strcmp(text, V.hud_last)) return;
    snprintf(V.hud_last, sizeof(V.hud_last), "%s", text);
    free(V.hud_px);
    int scale = V.fb && V.fb->w >= 1280 ? 3 : 2;
    V.hud_px = hud_text(text, scale, &V.hud_w, &V.hud_h);
    if (V.gview) gpu_view_hud(V.gview, V.hud_on ? V.hud_px : NULL, V.hud_w, V.hud_h, HUD_X, HUD_Y);
    if (V.hud_on) V.force_present = 1;            // shown even when nothing else changes
}

// a CPU frame shown with the text over it (put on just for the window, then taken off again)
static void present_with_hud(Framebuffer *f){
    uint32_t *saved = NULL;
    int on = V.hud_on && V.hud_px && f->w >= HUD_X + V.hud_w && f->h >= HUD_Y + V.hud_h;
    if (on) hud_draw(f, V.hud_px, V.hud_w, V.hud_h, HUD_X, HUD_Y, &saved);
    window_present(V.win, f);
    if (on) hud_restore(f, V.hud_w, V.hud_h, HUD_X, HUD_Y, saved);
}
// The CPU's trees over what the physics moves (the top level over the instances, 1.3 ms for
// the 5000 of -demo sky) are made when something is about to read them, not every simulated
// frame: the CPU path tracer and the probe's bake trace through them, and the GPU path tracer
// copies the top level unless it builds its own (gpu_view_trace_reads_tlas). The rasterizers
// and the ray queries of our Vulkan layer never do. (Nothing to do when they are up to date:
// the scene counts its instances' moves, Scene.inst_version.)
static void scene_trees(void){ PROF(PROF_ACCEL, scene_update_instances(&V.scene, &V.o.ap)); }

// -sim on with the GPU: the air and the bodies go there (again, after a restart)
static void gpu_sim_start(void){
    if (!V.gview || !V.o.sim.on) return;
    if (V.o.gpu_fluid && !V.gfluid && (V.gfluid = gpu_fluid_create(V.gpu, &V.sim.pw)))
        gpu_fluid_attach_view(V.gfluid, V.gview);
    if (V.o.gpu_rigid && !V.grigid) V.grigid = gpu_rigid_create(V.gpu, &V.sim.pw);
}

// -simfixed on: a step only once the GPU's last one is done and back on the CPU. They come
// back a step or two late, as fast as the GPU is, and what the CPU does with them (the air's
// drag, the bodies it adds) then depended on the timing: two runs parted. (The browser
// cannot wait for the GPU: the frame goes on, and the step waits for a later frame.)
static int gpu_sim_pending(void){
    return (V.grigid && gpu_rigid_pending(V.grigid)) || (V.gfluid && gpu_fluid_busy(V.gfluid));
}
static void gpu_sim_stop(void){
    gpu_fluid_destroy(V.gfluid);
    V.gfluid = NULL;
    gpu_rigid_destroy(V.grigid);
    V.grigid = NULL;
}

// the demo again from t = 0: the scene made again as sim_setup makes it. 0 = done
static int sim_restart(void){
    gpu_sim_stop();
    sim_free(&V.sim);
    scene_free(&V.scene);
    memset(&V.scene, 0, sizeof(V.scene));
    if (sim_setup(&V.sim, &V.o, &V.scene) != 0) return 1;
    V.sim.max_dt = 1.0 / 30;
    gpu_sim_start();
    sim_step(&V.sim, &V.scene, 0);
    scene_set_accel(&V.scene, V.accel, &V.o.ap);
    V.sim_acc = 0;
    return 0;
}

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
static void save_traced(void *user, const float *sc, int w, int h){
    (void)user;
    if (!sc){                    // the read failed: no screenshot, and nothing waits for one
        fprintf(stderr, "no screenshot: the GPU's image could not be read back (%s not written)\n", V.shot_path);
        V.shot_waiting = 0;
        return;
    }
    Image img = image_create(w, h);
    for (int i = 0; i < w * h; i++){
        float c = sc[4 * i + 3];
        img.px[i] = c > 0 ? vec3(sc[4 * i] / c, sc[4 * i + 1] / c, sc[4 * i + 2] / c) : vec3(0, 0, 0);
    }
    if (image_write_png(&img, V.shot_path) == 0)
        printf("saved %s (%dx%d, path traced on the GPU, %d samples/pixel%s)\n", V.shot_path, w, h,
               gpu_view_trace_passes(V.gview), V.o.denoise ? ", denoised" : "");
    else fprintf(stderr, "could not write %s (does the folder exist?)\n", V.shot_path);
    image_free(&img);
    V.shot_waiting = 0;
}

static void gpu_screenshot(void){
    snprintf(V.shot_path, sizeof(V.shot_path), "%s_%03d.png", V.o.out, ++V.shot_no);
    V.shot_waiting = 1;
    if (!V.showing_raster && V.gpu_tracing && gpu_view_trace_passes(V.gview) > 0){
        if (!gpu_view_trace_read(V.gview, save_traced, NULL)) V.shot_waiting = 0;
        return;
    }
    if (!gpu_offscreen(V.gpu)) gpu_view_capture(V.gview, on_gpu_frame, NULL);   // offscreen: every frame comes back
    V.force_present = 1;
}
#endif

// a frame read back from the GPU: shown (offscreen), and kept as a screenshot if one was asked
static void on_gpu_frame(void *user, const uint32_t *px, int w, int h){
    (void)user;
    if (!px){                    // the read failed: this frame is not shown, a screenshot not made
#ifndef __EMSCRIPTEN__
        if (V.shot_waiting && V.shot_path[0]){
            fprintf(stderr, "no screenshot: the GPU's frame could not be read back (%s not written)\n", V.shot_path);
            V.shot_path[0] = 0;
            V.shot_waiting = 0;
        }
#else
        V.want_shot = 0;
#endif
        return;
    }
#ifndef __EMSCRIPTEN__
    if (V.shot_waiting && V.shot_path[0]){
        Framebuffer shot = {w, h, (uint32_t *)px, NULL};
        if (fb_save_png(&shot, V.shot_path) == 0) printf("saved %s (%dx%d, drawn by the GPU)\n", V.shot_path, w, h);
        else fprintf(stderr, "could not write %s (does the folder exist?)\n", V.shot_path);
        V.shot_path[0] = 0;
        V.shot_waiting = 0;
    }
#endif
    if (gpu_offscreen(V.gpu) && V.win){
        if (!V.gfb) V.gfb = fb_create(w, h);
        if (V.gfb && V.gfb->w == w && V.gfb->h == h){
            memcpy(V.gfb->color, px, (size_t)w * h * sizeof(uint32_t));
            present_with_hud(V.gfb);
        }
    }
#ifdef __EMSCRIPTEN__
    if (V.want_shot){
        js_capture_done(px, w, h, V.want_shot == 2);
        printf("screenshot: %dx%d\n", w, h);
        V.want_shot = 0;
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

// -maxfps on, a window in step with the display: the path tracer's bands run while the frame
// waits for the display, not before it (gpu_trace_frame)
static int trace_overlaps(void){ return V.o.maxfps && V.o.vsync && V.gpu && !gpu_offscreen(V.gpu); }

// "this view: " and the command that renders what the window shows (key C, autoquit with -orbit)
static void print_view_command(void){
    Options cur = V.o;
    cur.max_depth = V.max_depth;
    printf("this view: ");
    options_print_command(&cur, V.view.pos, v_add(V.view.pos, v_scale(view_forward(&V.view), 500)), V.view.fov, stdout);
    fflush(stdout);
}

// the GPU path tracer's parameters for this camera, from the options
static GpuTraceParams trace_params(const Camera *cam, int depth){
    GpuTraceParams tp = {*cam, depth, V.o.mis, (real)V.o.clamp, !V.o.sampler, V.o.caustics_lt ? (real)V.o.photons : 0, V.o.photon_cache,
                         (real)V.o.iclamp};
    return tp;
}

// the rasterizer's view of the camera as it is now (looking at `at`), in the current shading
static RenderParams raster_params(Vec3 at){
    RenderParams p = render_params_camera(V.view.pos, at, vec3(0, 1, 0), V.view.fov, V.W, V.H);
    p.mode = V.shading;
    return p;
}

// A new path traced image on the GPU, for the camera as it is now. 0 = this GPU cannot hold
// the scene: the CPU path traces from now on.
static int gpu_trace_begin(const Camera *cam, Vec3 at, int depth){
    GpuTraceParams tp = trace_params(cam, depth);
    scene_trees();               // (the meshes' own trees are copied when the scene is new)
    if (!gpu_view_trace_start(V.gview, &V.scene, &tp)){
        V.o.gpu_trace = 0;
        V.gpu_tracing = 0;
        return 0;
    }
    // what shows through until every pixel has a sample: the rasterized scene, same camera
    V.last_rp = raster_params(at);
    V.gpu_tracing = 1;
    V.trace_starts++;
    V.gpu_cam = *cam;
    V.gpu_depth = depth;
    V.gpu_shown_passes = -1;
    V.force_present = 1;
    return 1;
}

// Samples per pixel the GPU path tracer keeps from one frame to the next while things move
// (gpu_view_trace_move): the camera alone changes nothing on the surfaces, so they keep
// many (-history, 32); with the physics running the light changes too (shadows, the cloud),
// so fewer, and a change shows within a few frames.
#define GPU_HISTORY_PHYSICS 12
// An edit (viewer_scene_edited) changes light on surfaces that did not move, which the samples
// carried over cannot tell: a pixel keeps one at most while edits come, and the image starts
// over once none came for EDIT_SETTLE seconds (the old shadow gone at once, not fading out).
#define GPU_HISTORY_EDIT 1
#define EDIT_SETTLE 0.12

// rt mode with the GPU: path traced every frame, whatever moves. A new camera or the
// physics' new state carry the image over (each pixel keeps the samples of its surface,
// then the denoiser smooths what is new); other changes (bounces, a reset) start over.
// 0 = this GPU cannot hold the scene: the CPU path traces from now on.
static int gpu_trace_frame(const Camera *cam, Vec3 at, int depth){
    V.gpu_scene_moved |= V.sim_moved;
    if (V.edit_pending && clock_s() - V.edit_t >= EDIT_SETTLE){ V.edit_pending = 0; V.reset = 1; }
    if (!V.gpu_tracing || V.reset || depth != V.gpu_depth){
        if (!gpu_trace_begin(cam, at, depth)) return 0;
        V.gpu_scene_moved = V.gpu_scene_edited = 0;
    } else if ((V.gpu_scene_moved || memcmp(cam, &V.gpu_cam, sizeof(*cam)) != 0) && !gpu_view_trace_busy(V.gview)){
        // (a slow GPU still busy with a band: the move waits for it, and the frame shows the
        // last image as it is, with the raster of its own camera, not a newer one)
        GpuTraceParams tp = trace_params(cam, depth);
        int keep = V.gpu_scene_moved && V.o.history > GPU_HISTORY_PHYSICS ? GPU_HISTORY_PHYSICS : V.o.history;
        if (V.gpu_scene_edited && keep > GPU_HISTORY_EDIT) keep = GPU_HISTORY_EDIT;
        V.gpu_scene_moved = V.gpu_scene_edited = 0;
        int moved;
        if (gpu_view_trace_reads_tlas(V.gview, &V.scene)) scene_trees();
        PROF(PROF_MOVE, moved = gpu_view_trace_move(V.gview, &V.scene, &tp, keep));
        V.trace_moves++;
        if (!moved){
            V.o.gpu_trace = 0;
            V.gpu_tracing = 0;
            return 0;
        }
        // under the pixels that were hidden until now (no sample yet): the scene rasterized
        V.last_rp = raster_params(at);
        V.gpu_cam = *cam;
        V.force_present = 1;
    }
    V.reset = 0;
    int more = 1;
    if (gpu_view_trace_passes(V.gview) < V.o.spp) PROF(PROF_MORE, more = gpu_view_trace_more(V.gview, &V.scene, &V.last_rp, &V.o.march));
    else if (!V.o.maxfps) idle(10);             // (-maxfps on: the image shown again, as many times as can be)
    if (!more){                               // its pipelines failed: the CPU path traces
        V.o.gpu_trace = 0;
        V.gpu_tracing = 0;
        return 0;
    }
    // natively: sleep until the band is done (bands are sized to take < ~45 ms). A timed
    // nap would not do: on Windows the shortest one lasts ~15 ms, longer than a band. With
    // -maxfps on in a window in step with the display, not: the band runs while the frame
    // waits for the display (its image shows what the band added, being after it on the
    // queue), so the GPU is not idle between the two
    if (gpu_view_trace_busy(V.gview) && !trace_overlaps()) PROF(PROF_WAIT, gpu_wait(V.gpu));
    else gpu_poll(V.gpu);                       // (the bands that finished: their callbacks)
    V.pass_ms = gpu_view_trace_pass_ms(V.gview);
    return 1;
}

// -stats file: the frame rate and the path tracer's passes per second since stats_t0, for
// bin/bench (group I), written once
static void write_stats(double tn){
    if (V.stats_written || !V.o.stats || V.stats_t0 <= 0 || tn <= V.stats_t0) return;
    V.stats_written = 1;
    FILE *f = fopen(V.o.stats, "w");
    if (!f){ fprintf(stderr, "cannot write %s\n", V.o.stats); return; }
    double fps = (V.frames - V.stats_frames) / (tn - V.stats_t0);
    fprintf(f, "fps %.3f\nms_per_frame %.4f\npasses_per_s %.3f\nmode %s\nseconds %.2f\n", fps, fps > 0 ? 1000.0 / fps : 0.0,
            (V.passes_total - V.stats_passes) / (tn - V.stats_t0), backend_name(V.mode), tn - V.stats_t0);
    fclose(f);
}

// what -autoquit prints at the end: how long, the mode, the frame rate, then what was showing
// (on_gpu: the GPU's image, about to be saved)
static void autoquit_report(double tn, int on_gpu){
    printf("autoquit after %.1f s (%s, %.0f frames/s): ", tn - V.t_start, backend_name(V.mode), V.frames / (tn - V.t_start));
    if (V.o.sim.on && !V.sim_paused){
        char st[200];
        sim_status(&V.sim, st, sizeof(st));
        if (on_gpu) printf("%s\n", st); else printf("%s, last pass %.0f ms\n", st, V.pass_ms);
    } else if (V.showing_raster){
        if (on_gpu) printf("raster frame on the GPU\n"); else printf("last raster frame %.1f ms\n", V.raster_ms);
    } else if (on_gpu) printf("%d samples/pixel on the GPU, last pass %.1f ms\n", gpu_view_trace_passes(V.gview), V.pass_ms);
    else printf("%d samples/pixel, last pass %.0f ms\n", V.acc.passes, V.pass_ms);
    prof_print();
}

// what a frame's steps hand on to the next ones
typedef struct {
    double t0;         // when it began
    int changed;       // the camera moved or the image must start over (input, held keys)
    double t;          // the time of its camera and physics
    Vec3 at;           // its camera, looking at this point
    Camera cam;
    int use_raster;    // rasterized (else path traced)
    double tn;         // when it was shown
} Frame;

// 0. The GPU starts asynchronously: wait for it (1: not yet), or fall back to the CPU; and
// fall back too when the GPU is lost.
static int viewer_gpu(void){
    if (V.gpu && !V.gview){
        GpuState gs = gpu_state(V.gpu);
        if (gs == GPU_PENDING) return 1;
        if (gs == GPU_READY) V.gview = gpu_view_create(V.gpu, V.W, V.H);
        // -maxfps on: the path tracer's work per frame within 60% of the display's period (or
        // -band ms), so frames keep coming while it refines; off: the bands of before
        // (85% when the bands run while the frame waits for the display: trace_overlaps)
        if (V.gview) gpu_view_trace_budget(V.gview, V.o.band_ms > 0 ? V.o.band_ms : !V.o.maxfps ? 0 :
                                                (trace_overlaps() ? 0.85 : 0.6) * 1000.0 / window_refresh_hz(V.win));
        // frames not waiting for the display (-vsync off, offscreen): at least a whole pass
        // each, up to 33 ms (30 frames/s), so the path tracer is most of the GPU's work, not
        // the denoiser and the composite every frame carries
        if (V.gview) gpu_view_trace_budget_pass(V.gview, V.o.maxfps && V.o.band_ms <= 0 && !trace_overlaps() ? 33.0 : 0);
        if (!V.gview){
            fprintf(stderr, "GPU not used (%s): the CPU draws everything\n", gpu_message(V.gpu));
            gpu_destroy(V.gpu);
            V.gpu = NULL;
        } else {
            printf("GPU: frames shown and clouds raymarched on %s%s\n", gpu_message(V.gpu),
                   gpu_offscreen(V.gpu) ? " (offscreen, read back to the canvas)" : "");
            gpu_view_trace_denoise(V.gview, V.o.denoise);
            gpu_view_trace_denoise_moving(V.gview, V.o.denoise_move);
            gpu_view_aa(V.gview, V.o.aa);
            gpu_view_trace_wavefront(V.gview, V.o.wavefront);
            gpu_view_trace_limit(V.gview, V.o.spp);
            if (V.o.gpu_trace < 0){
                V.o.gpu_trace = !gpu_software(V.gpu);
                if (!V.o.gpu_trace) printf("GPU: a software GPU: the CPU path traces (faster here; -gputrace on to try it anyway)\n");
            }
            // the path tracer's kernels asked for now: in a browser they compile in the
            // background (seconds, the first time) while the rasterized scene is shown
            if (V.o.gpu_trace && V.mode != BACKEND_RASTER) gpu_view_trace_prepare(V.gview);
            gpu_sim_start();
            if (V.gfluid) printf("GPU: the air (%dx%dx%d cells) and the cloud are simulated there\n",
                               V.sim.pw.fluid->nx, V.sim.pw.fluid->ny, V.sim.pw.fluid->nz);
            if (V.grigid) printf("GPU: the balls and boxes are simulated there\n");
        }
        V.force_present = V.raster_dirty = 1;
    } else if (V.gview && gpu_state(V.gpu) == GPU_FAILED){
        // the browser took the GPU away: carry on with the CPU
#ifdef __EMSCRIPTEN__
        fprintf(stderr, "GPU lost (%s): the CPU draws everything from now on%s\n", gpu_message(V.gpu),
                gpu_offscreen(V.gpu) ? "" : " (if the image no longer changes, reload with ?gpu=off or ?gpu=offscreen)");
#else
        fprintf(stderr, "GPU lost (%s): the CPU draws everything from now on\n", gpu_message(V.gpu));
#endif
        gpu_sim_stop();              // the air and the bodies go back to the CPU
        V.gpu_scene = V.gpu_tracing = 0;
        V.reset = 1;
        gpu_view_destroy(V.gview);
        V.gview = NULL;
        gpu_destroy(V.gpu);
        V.gpu = NULL;
        V.gpu_frame_owed = 0;
        V.force_present = V.raster_dirty = 1;
    }
    return 0;
}

// 1. The window's events: the keys held for the camera (V.in), the keys pressed, the mouse.
// 0: quit at once (the physics could not start again).
static int viewer_input(Frame *fr){
    LrEvent ev;
    if (V.ext_changed){ fr->changed = 1; V.ext_changed = 0; }
    while (window_poll(V.win, &ev)){
        if (V.input_off && ev.type != LR_CLOSE && ev.type != LR_EXPOSE) continue;   // (the program's)
        if (ev.type == LR_KEYDOWN || ev.type == LR_KEYUP){
            int down = ev.type == LR_KEYDOWN;
            switch (ev.key){
            case 'w': V.in.fwd = down; break;
            case 's': V.in.back = down; break;
            case 'a': V.in.left = down; break;
            case 'd': V.in.right = down; break;
            case 'e': V.in.up = down; break;
            case 'q': V.in.down = down; break;
            case LR_KEY_LEFT: V.in.tl = down; break;
            case LR_KEY_RIGHT: V.in.tr = down; break;
            case LR_KEY_UP: V.in.tu = down; break;
            case LR_KEY_DOWN: V.in.td = down; break;
            case LR_KEY_SHIFT: V.in.fast = down; break;
            default: break;
            }
            if (!down) continue;
            switch (ev.key){
            case LR_KEY_ESCAPE: V.running = 0; break;
            case 'r': V.view = V.home; fr->changed = 1; break;
            case 'i':
                V.hud_on = !V.hud_on;
                if (V.gview) gpu_view_hud(V.gview, V.hud_on ? V.hud_px : NULL, V.hud_w, V.hud_h, HUD_X, HUD_Y);
                V.raster_dirty = 1;
                break;
            case 'p':
#ifdef __EMSCRIPTEN__
                if (V.gview){ do_capture(1); break; }   // a PNG download
#else
                if (V.gview){ if (!V.shot_waiting) gpu_screenshot(); break; }
#endif
                save_view(&V.acc, V.fb, V.showing_raster, V.shading, V.o.out, &V.shot_no);
                break;
            case 'c':
                print_view_command();
                break;
            case 'm':
                V.mode = V.mode == BACKEND_HYBRID ? BACKEND_RT : (V.mode == BACKEND_RT ? BACKEND_RASTER : BACKEND_HYBRID);
                printf("mode: %s\n", backend_name(V.mode));
                fflush(stdout);
                fr->changed = 1;
                break;
            case 'v':
                V.shading = V.shading == MODE_PHONG ? MODE_GOURAUD : (V.shading == MODE_GOURAUD ? MODE_FLAT :
                          (V.shading == MODE_FLAT ? MODE_WIREFRAME : MODE_PHONG));
                printf("raster shading: %s\n", render_mode_name(V.shading));
                fflush(stdout);
                V.raster_dirty = 1;
                if (V.mode == BACKEND_HYBRID) fr->changed = 1;   // show the new shading now
                break;
            case 'b': {
                V.accel = (AccelKind)((V.accel + 1) % 3);
                printf("building %s...\n", accel_name(V.accel));
                fflush(stdout);
                scene_set_accel(&V.scene, V.accel, &V.o.ap);
                char buf[256];
                scene_accel_stats(&V.scene, buf, sizeof(buf));
                printf("%s\n", buf);
                if (V.mode != BACKEND_RASTER) fr->changed = 1;
                break;
            }
            case 'x':
                if (!V.gview) break;
                V.o.denoise = !V.o.denoise;
                printf("denoiser: %s\n", gpu_view_trace_denoise(V.gview, V.o.denoise) ? "on" : "off");
                fflush(stdout);
                V.gpu_shown_passes = -1;
                V.force_present = 1;
                break;
            case 'l':
                scene_trees();                     // the bake traces the scene
                options_bake_probe(&V.o, &V.scene);
                fflush(stdout);
                V.raster_dirty = 1;
                if (V.mode == BACKEND_HYBRID) fr->changed = 1;
                break;
            case 'f': V.preview_on = !V.preview_on; break;
            case '1': V.preview = 1; V.preview_on = 0; break;
            case '2': V.preview = 2; V.preview_on = 1; break;
            case '3': V.preview = 4; V.preview_on = 1; break;
            case '4': V.preview = 8; V.preview_on = 1; break;
            case '[': if (V.max_depth > 1){ V.max_depth--; fr->changed = 1; } break;
            case ']': V.max_depth++; fr->changed = 1; break;
            case '+': case '=': V.speed *= 1.5; break;
            case '-': V.speed /= 1.5; break;
            case 'h': printf("\n%s\n", CONTROLS); if (V.o.sim.on) printf("%s\n", SIM_CONTROLS); break;
            case ' ':
                if (V.o.sim.on){ V.sim_paused = !V.sim_paused; printf(V.sim_paused ? "paused\n" : "running\n"); fflush(stdout); }
                break;
            case 'k': if (V.o.sim.on){ sim_blast_now(&V.sim); printf("boom\n"); fflush(stdout); } break;
            case 'n': if (V.o.sim.on) sim_drop_ball(&V.sim, &V.scene); break;
            case 't':
                if (V.o.sim.on){
                    if (sim_restart() != 0) return 0;
                    fr->changed = 1;
                }
                break;
            default: break;
            }
        } else if (ev.type == LR_BUTTONDOWN){
            if (ev.button == 1){ V.in.drag = 1; V.in.mx = ev.x; V.in.my = ev.y; }
        } else if (ev.type == LR_BUTTONUP){
            if (ev.button == 1) V.in.drag = 0;
        } else if (ev.type == LR_WHEEL){
            if (ev.wheel > 0 && V.view.fov > 5){ V.view.fov -= 2; fr->changed = 1; }     // wheel up: zoom in
            if (ev.wheel < 0 && V.view.fov < 120){ V.view.fov += 2; fr->changed = 1; }   // wheel down: zoom out
        } else if (ev.type == LR_MOTION && V.in.drag){
            int dx = ev.x - V.in.mx, dy = ev.y - V.in.my;
            V.in.mx = ev.x;
            V.in.my = ev.y;
            if (dx || dy){
                V.view.yaw -= dx * 0.004;      // drag right -> turn right
                V.view.pitch -= dy * 0.004;    // drag up -> look up
                fr->changed = 1;
            }
        } else if (ev.type == LR_CLOSE){
            V.running = 0;
        } else if (ev.type == LR_EXPOSE){
            V.force_present = 1;
        }
    }
    return 1;
}

// 2. The camera moved by the held keys (frame-rate independent: * dt) or -orbit, and the
// physics advanced; the frame's camera.
static void viewer_update(Frame *fr){
    fr->t = clock_s();
    double dt = fmin(fr->t - V.t_prev, 0.1);
    V.t_prev = fr->t;
    Vec3 f = view_forward(&V.view), r = view_right(&V.view), mv = vec3(0, 0, 0);
    if (V.in.fwd) mv = v_add(mv, f);
    if (V.in.back) mv = v_sub(mv, f);
    if (V.in.right) mv = v_add(mv, r);
    if (V.in.left) mv = v_sub(mv, r);
    if (V.in.up) mv = v_add(mv, vec3(0, 1, 0));
    if (V.in.down) mv = v_sub(mv, vec3(0, 1, 0));
    if (v_dot(mv, mv) > 0){
        V.view.pos = v_add(V.view.pos, v_scale(v_norm(mv), V.speed * (V.in.fast ? 4 : 1) * dt));
        fr->changed = 1;
    }
    double turn = 1.5 * dt;
    if (V.in.tl){ V.view.yaw += turn; fr->changed = 1; }
    if (V.in.tr){ V.view.yaw -= turn; fr->changed = 1; }
    if (V.in.tu){ V.view.pitch += turn; fr->changed = 1; }
    if (V.in.td){ V.view.pitch -= turn; fr->changed = 1; }
    V.view.pitch = fmax(-1.55, fmin(1.55, V.view.pitch));
    if (V.o.orbit != 0){
        // -orbit: swing to and fro around the point the start camera looks at, at a constant speed
        V.orbit_phase += V.o.orbit * dt;
        double p = fmod(V.orbit_phase, 100.0), deg = p < 50 ? p - 25 : 75 - p;
        double a = deg * PI / 180, ca = cos(a), sa = sin(a);
        Vec3 c = V.scene.cam_at, d = v_sub(V.home.pos, c);
        V.view.pos = v_add(c, vec3(d.x * ca + d.z * sa, d.y, -d.x * sa + d.z * ca));
        Vec3 fw = v_norm(v_sub(c, V.view.pos));
        V.view.yaw = atan2(fw.x, fw.z);
        V.view.pitch = asin(fmax(-1.0, fmin(1.0, fw.y)));
        fr->changed = 1;
    }
    if (fr->changed){ V.last_change = fr->t; V.raster_dirty = 1; }

    // ---- 2b. physics: advance by the time that passed, then the image starts over. At most
    // -simhz steps a second (60): faster frames step every few, and those between refine the
    // same image (no new guide, reprojection, top level or bodies' step on the GPU). Each
    // step takes all the time since the last one, so slower frames step every frame, as before
    if (V.o.sim.on){
        double sdt = fmin(fr->t - V.sim_last, 0.05);
        V.sim_moved = 0;
        if (V.o.sim_fixed){
            // -simfixed on: steps of exactly 1 / simhz, as many as the time asks (at most 4 a
            // frame: a slow frame slows the simulation down rather than piling steps up), so the
            // same seed gives the same motion whatever the frame rate
            const double h = 1.0 / V.o.sim_hz;
            V.sim_acc = V.sim_paused ? 0 : V.sim_acc + fmin(fr->t - V.sim_last, 0.1);
            V.sim_last = fr->t;
            int n = 0;
            while (!V.sim_paused && V.sim_acc >= h && n < 4){
                if (gpu_sim_pending()){ if (V.sim_acc > h) V.sim_acc = h; break; }   // (slower, not piled up)
                PROF(PROF_SIM, sim_step(&V.sim, &V.scene, h));
                V.sim_acc -= h;
                n++;
                // (within half a step: float time after n steps of 1/60 falls a hair short of n / 60)
                if (V.sim_pause_at > 0 && V.sim.pw.time >= V.sim_pause_at - 0.5 * h){ V.sim_paused = 1; V.sim_pause_at = 0; }
            }
            if (n == 4) V.sim_acc = 0;
            if (n){ V.sim_moved = 1; V.raster_dirty = 1; }
        } else {
            if (V.sim_paused || sdt <= 0 || V.o.sim_hz <= 0 || sdt >= 1.0 / V.o.sim_hz) V.sim_last = fr->t;
            else sdt = 0;                                   // (not yet: the time goes on adding up)
            if (!V.sim_paused && sdt > 0){
                PROF(PROF_SIM, sim_step(&V.sim, &V.scene, sdt));   // the bodies are instances: the world stays
                V.sim_moved = 1;                                  // (their trees: scene_trees, when read)
                V.raster_dirty = 1;
            }
        }
    }

    f = view_forward(&V.view);
    fr->at = v_add(V.view.pos, f);
    fr->cam = camera_make(V.view.pos, fr->at, vec3(0, 1, 0), V.view.fov, (real)V.W / V.H);
}

// 3. The image drawn: rasterized, path traced on the GPU, or a step of the CPU path tracer.
static void viewer_render(Frame *fr){
    // hybrid: rasterize while the camera moves, path trace once it has settled
    fr->use_raster = V.mode == BACKEND_RASTER ||
                     (V.mode == BACKEND_HYBRID && (fr->changed || fr->t - V.last_change < (V.settle > 0 ? V.settle : HYBRID_SETTLE_S) ||
                                                 (V.o.sim.on && !V.sim_paused)));
    // the CPU path tracer reads the cloud on the CPU: bring the GPU's back (it lands a frame later)
    // (the GPU path tracer takes it straight from the GPU)
    if (V.gfluid && !fr->use_raster && !(V.gview && V.o.gpu_trace > 0)) gpu_fluid_sync_cloud(V.gfluid);
    if (fr->use_raster){
        if (V.raster_dirty || V.o.maxfps){            // (-maxfps on: drawn again, changed or not)
            double r0 = clock_s();
            RenderParams p = raster_params(fr->at);
            // the surfaces: by the GPU rasterizer when it is there (at presentation), otherwise here
            // (media too big for this GPU's buffers: both here, the surfaces under them)
            int gvol = V.gview && gpu_view_volumes_fit(V.gview, &V.scene, &V.o.march);
            V.gpu_scene = V.gview && V.o.gpu_raster && gvol;
            if (!V.gpu_scene){
                fb_clear(V.fb, rgb8(0, 0, 0));
                render_scene(V.fb, &V.scene, &p);
            }
            // the media: on the GPU when it is there (at presentation), otherwise here
            if (V.o.raster_clouds && !gvol) raster_draw_volumes(V.rvol, V.fb, &V.scene, &p, &V.o.march);
            V.last_rp = p;
            V.frame_volumes = V.o.raster_clouds && gvol && V.scene.nvolumes > 0;
            V.raster_ms = (clock_s() - r0) * 1000.0;
            if (V.prof_on) V.prof[PROF_RASTER] += V.raster_ms / 1000.0;
            V.raster_dirty = 0;
            V.force_present = 1;
        } else {
            idle(5);                                     // nothing new to draw
        }
        // the path tracer starts over when it takes back over (the GPU's, with keep_trace:
        // carries its image over the move, each pixel keeping the samples of its surface)
        if (V.acc.passes || V.acc.band_y || V.acc.scale != 1) accum_reset(&V.acc, 1);
        if (!V.keep_trace){
            V.gpu_tracing = 0;
            V.reset = 0;
        }
        V.pass_t0 = clock_s();
    } else if (V.gview && V.o.gpu_trace > 0 && gpu_trace_frame(&fr->cam, fr->at, V.max_depth)){
        // done: the GPU path traces (every frame, the camera or the bodies moving or not)
    } else if (V.sim_moved){
        // the simulation moved: one pass of the new state, blended with the last ones
        int sc = V.preview_on ? V.preview : 1;
        accum_reset(&V.acc, sc);
        scene_trees();
        V.pass_t0 = clock_s();
        for (int i = 0; i < V.o.move_spp; i++) render_rows(&V.scene, &fr->cam, &V.acc, 0, V.acc.rh, V.max_depth, V.o.mis, (real)V.o.clamp, V.o.sampler, (real)V.o.iclamp);
        V.acc.passes = V.o.move_spp;
        V.pass_ms = (clock_s() - V.pass_t0) * 1000.0;
        ema_update(&V.ema, &V.acc, V.fb, fr->changed);
        V.force_present = 1;
        V.gpu_tracing = 0;
        V.reset = 1;   // once paused: start refining at full resolution
    } else if (fr->changed || V.reset){
        // rt mode: start over. If the camera moved, render one whole pass right away
        // (cheap at 1/preview resolution) so the image follows the input.
        int sc = fr->changed && V.preview_on && V.mode == BACKEND_RT ? V.preview : 1;
        accum_reset(&V.acc, sc);
        V.reset = 0;
        scene_trees();
        V.pass_t0 = clock_s();
        if (fr->changed){
            int n = sc > 1 ? V.o.move_spp : 1;          // -movespp: less grain, slower
            for (int i = 0; i < n; i++) render_rows(&V.scene, &fr->cam, &V.acc, 0, V.acc.rh, V.max_depth, V.o.mis, (real)V.o.clamp, V.o.sampler, (real)V.o.iclamp);
            V.acc.passes = n;
            V.pass_ms = (clock_s() - V.pass_t0) * 1000.0;
            V.pass_t0 = clock_s();
            accum_to_fb(&V.acc, V.fb);
            V.force_present = 1;
            if (sc > 1) V.reset = 1;  // next calm frame: back to full resolution
        }
    } else if (V.acc.passes < V.o.spp){
        // calm: refine at full resolution, a band of rows at a time (~20 ms each)
        scene_trees();               // (after a pause or a change of mode: made once)
        double b0 = clock_s();
        int y1 = V.acc.band_y + V.band < V.acc.rh ? V.acc.band_y + V.band : V.acc.rh;
        render_rows(&V.scene, &fr->cam, &V.acc, V.acc.band_y, y1, V.max_depth, V.o.mis, (real)V.o.clamp, V.o.sampler, (real)V.o.iclamp);
        double bt = clock_s() - b0;
        if (bt < 0.010 && V.band < V.acc.rh) V.band *= 2;
        else if (bt > 0.040 && V.band > 1) V.band /= 2;
        V.acc.band_y = y1;
        if (V.acc.band_y >= V.acc.rh){
            V.acc.band_y = 0;
            V.acc.passes++;
            V.pass_ms = (clock_s() - V.pass_t0) * 1000.0;
            V.pass_t0 = clock_s();
            accum_to_fb(&V.acc, V.fb);
            V.force_present = 1;
        }
    } else if (!V.o.maxfps){
        idle(10);                                     // done refining
    }
}

// 4. Shown: in the window (the GPU's frame or the CPU's), when something new is there to see.
static void viewer_present(Frame *fr){
    fr->tn = clock_s();
    const int gpu_traced = !fr->use_raster && V.gpu_tracing;
    const int gpu_passes = gpu_traced ? gpu_view_trace_passes(V.gview) : 0;
    int refining = !fr->use_raster && !V.sim_moved &&     // bands land on screen ~30x per second
                   (gpu_traced ? gpu_passes < V.o.spp || gpu_passes != V.gpu_shown_passes : V.acc.passes < V.o.spp);
    double present0 = clock_s();
    // the passes made since the last frame (an image started over: all of its own)
    if (!gpu_traced && !fr->use_raster){          // (the CPU's: an image started over counts all of its own)
        long now = V.acc.passes;
        V.cpu_passes += now >= V.passes_seen ? now - V.passes_seen : now;
        V.passes_seen = now;
    }
    V.passes_total = (V.gview ? gpu_view_trace_work(V.gview) : 0) + V.cpu_passes;
    // -maxfps on: every time round; off: when something changed, or the path tracer's image
    // ~30 times a second while it refines
    if (V.o.maxfps || V.force_present || (refining && fr->tn - V.last_present > 0.033)){
        if (!fr->use_raster && !V.sim_moved && !gpu_traced) accum_to_fb(&V.acc, V.fb);
        if (V.gview){
            // offscreen the frame must come back to be seen: ask for it. If the previous one
            // is still on its way, the GPU is behind: draw nothing now (no queue of frames
            // waiting for it), and try again next time so the last image is not lost.
            V.gpu_frame_owed = gpu_offscreen(V.gpu) && !gpu_view_capture(V.gview, on_gpu_frame, NULL);
            if (V.gpu_frame_owed){
                // (nothing drawn)
            } else if (gpu_traced){
                gpu_view_present_traced(V.gview, &V.scene, &V.last_rp, &V.o.march, V.o.raster_clouds && V.scene.nvolumes > 0);
                V.gpu_shown_passes = gpu_passes;
            } else if (fr->use_raster && V.gpu_scene){
                if (!gpu_view_present_scene(V.gview, &V.scene, &V.last_rp, &V.o.march, V.frame_volumes)){
                    // this GPU cannot draw the scene: the CPU rasterizes from now on
                    V.o.gpu_raster = 0;
                    V.gpu_scene = 0;
                    V.raster_dirty = 1;
                    fb_clear(V.fb, rgb8(0, 0, 0));
                    render_scene(V.fb, &V.scene, &V.last_rp);
                    gpu_view_present(V.gview, V.fb, &V.scene, &V.last_rp, &V.o.march, V.frame_volumes);
                }
            } else gpu_view_present(V.gview, V.fb, &V.scene, &V.last_rp, &V.o.march, fr->use_raster && V.frame_volumes);
        } else present_with_hud(V.fb);
        V.showing_raster = fr->use_raster;
        if (V.gpu_frame_owed){
            // nothing was shown: not a frame (it was counted once: "318998 frames/s"). Let
            // the frame on its way arrive (its callback) and wait a little, instead of asking
            // again at once and keeping a core busy
            gpu_poll(V.gpu);
            idle(1);
        } else {
            V.last_present = fr->tn;
            V.frames++;
        }
        V.force_present = V.gpu_frame_owed;
    }
    if (V.prof_on){
        V.prof[PROF_PRESENT] += clock_s() - present0;
        V.prof[PROF_FRAME] += clock_s() - fr->t0;
        V.prof_frames++;
    }
    V.prof_on = clock_s() - V.t_start > 1.0;
}

// 5. The title (twice a second), -stats, and -autoquit's screenshot and end. 0 = quit.
static int viewer_report(const Frame *fr){
    if (fr->tn - V.last_title > 0.5){
        char title[512];
        if (fr->use_raster)
        {
            double ml = 0, mm = 0;
            raster_volumes_timing(V.rvol, &ml, &mm);
            if (V.scene.nvolumes && V.o.raster_clouds)
                snprintf(title, sizeof(title), "view | %s | raster %s %.1f ms/frame (cloud: light %.0f + march %.0f ms) | speed %.0f",
                         backend_name(V.mode), render_mode_name(V.shading), V.raster_ms, ml, mm, V.speed);
            else
                snprintf(title, sizeof(title), "view | %s | raster %s %.1f ms/frame | speed %.0f",
                         backend_name(V.mode), render_mode_name(V.shading), V.raster_ms, V.speed);
            if (V.gpu_scene){
                size_t l = strlen(title);
                snprintf(title + l, sizeof(title) - l, " | triangles on the GPU");
            }
        }
        else if (V.gpu_tracing)
            snprintf(title, sizeof(title), "view | %s | path tracing on the GPU%s | %d/%d spp | %.0f ms/pass | %d bounces | speed %.0f",
                     backend_name(V.mode), V.o.denoise ? ", denoised" : "", gpu_view_trace_passes(V.gview), V.o.spp, V.pass_ms,
                     V.max_depth, V.speed);
        else
            snprintf(title, sizeof(title), "view | %s | path tracing %s | %d/%d spp | %.0f ms/pass | 1/%d res | %d bounces | speed %.0f",
                     backend_name(V.mode), accel_name(V.accel), V.acc.passes, V.o.spp, V.pass_ms, V.acc.scale, V.max_depth, V.speed);
        if (V.o.sim.on){
            char st[200];
            sim_status(&V.sim, st, sizeof(st));
            size_t l = strlen(title);
            snprintf(title + l, sizeof(title) - l, " | %s%s", st, V.sim_paused ? " | paused" : "");
        }
        if (V.gview){
            size_t l = strlen(title);
            snprintf(title + l, sizeof(title) - l, " | GPU%s %s", gpu_offscreen(V.gpu) ? " (offscreen)" : "", gpu_message(V.gpu));
        }
        // the frame rate first, always: frames shown since the last title, every mode
        double fps = (V.frames - V.title_frames) / (fr->tn - V.last_title);
        double pps = (V.passes_total - V.title_passes) / (fr->tn - V.last_title);
        char shown[700], work[48] = "";
        if (!fr->use_raster) snprintf(work, sizeof(work), ", %.0f passes/s", pps);   // (the path tracer's work)
        snprintf(shown, sizeof(shown), "view | %.0f fps (%.1f ms%s)%s", fps, fps > 0 ? 1000.0 / fps : 0.0, work,
                 strncmp(title, "view", 4) == 0 ? title + 4 : title);
        if (!V.title_off) window_set_title(V.win, shown);
        V.fps = fps;
        hud_update(fps);
        V.title_frames = V.frames;
        V.title_passes = V.passes_total;
        V.last_title = fr->tn;
    }
    if (V.o.stats && V.stats_t0 == 0 && fr->tn - V.t_start > (V.o.autoquit / 3 < 2 ? V.o.autoquit / 3 : 2)){
        V.stats_t0 = fr->tn;
        V.stats_frames = V.frames;
        V.stats_passes = V.passes_total;
    }
    if (V.o.stats && V.o.autoquit > 0 && fr->tn - V.t_start > V.o.autoquit) write_stats(fr->tn);
#ifndef __EMSCRIPTEN__
    if (V.o.autoquit > 0 && fr->tn - V.t_start > V.o.autoquit && V.gview){
        // the GPU's image: ask for it, and quit once it is saved
        if (!V.shot_started){
            V.shot_started = 1;
            autoquit_report(fr->tn, 1);
            if (V.o.orbit != 0) print_view_command();   // the camera of the screenshot, to render it again
            gpu_screenshot();
        }
        if (!V.shot_waiting) V.running = 0;
        return V.running;
    }
#endif
    if (V.o.autoquit > 0 && fr->tn - V.t_start > V.o.autoquit){
        autoquit_report(fr->tn, 0);
        save_view(&V.acc, V.fb, V.showing_raster, V.shading, V.o.out, &V.shot_no);
        V.running = 0;
    }
    return V.running;
}

// one iteration: the steps above in order. 0 = quit.
int viewer_frame(void){
    Frame fr = {.t0 = clock_s()};
    if (viewer_gpu()) return 1;                  // (the GPU is not ready yet)
    if (!viewer_input(&fr)) return 0;
    viewer_update(&fr);
    viewer_render(&fr);
    viewer_present(&fr);
    return viewer_report(&fr);
}

#ifdef __EMSCRIPTEN__
static void do_capture(int download){
    if (!V.gview) return;
    V.want_shot = download ? 2 : 1;
    if (!gpu_offscreen(V.gpu)) gpu_view_capture(V.gview, on_gpu_frame, NULL);   // offscreen every frame comes back
    V.force_present = 1;
}
void lr_capture(void){ do_capture(0); }
// callable from JavaScript: Module._lr_prof() prints where the frames' time went on the CPU
// since the last call (or the first second), as -autoquit does natively, and starts over
EMSCRIPTEN_KEEPALIVE void lr_prof(void){
    prof_print();
    memset(V.prof, 0, sizeof(V.prof));
    V.prof_frames = 0;
}
#endif

void viewer_close(void){
    window_close(V.win);
    V.win = NULL;
    fb_destroy(V.fb);
    free(V.acc.sum);
    free(V.acc.count);
    free(V.ema.px);
    raster_volumes_free(V.rvol);
    gpu_sim_stop();
    gpu_view_destroy(V.gview);
    gpu_destroy(V.gpu);
    fb_destroy(V.gfb);
    if (V.o.sim.on) sim_free(&V.sim);
    scene_free(&V.scene);
}

Options *viewer_options(void){ return &V.o; }
Scene *viewer_scene(void){ return &V.scene; }

void viewer_print_controls(void){
    printf("\n%s\n", CONTROLS);
    if (V.o.sim.on) printf("%s\n", SIM_CONTROLS);
}

int viewer_setup_scene(void){
    if (V.o.sim.on){
        if (sim_setup(&V.sim, &V.o, &V.scene) != 0) return 1;
        V.sim.max_dt = 1.0 / 30;                     // one physics step per frame
        sim_step(&V.sim, &V.scene, 0);                 // the bodies that start at t = 0
        scene_set_accel(&V.scene, options_resolve_accel(&V.o, &V.scene), &V.o.ap);
    } else if (options_setup_scene(&V.o, &V.scene) != 0) return 1;
    return 0;
}

int viewer_open(const char *title){
    srgb_lut_init();
    V.W = V.o.width;
    V.H = V.o.height;
    V.win = window_open(V.W, V.H, title);
    if (V.o.gpu && V.win){
#ifdef __EMSCRIPTEN__
        V.gpu = gpu_create(V.o.gpu == 2 ? NULL : "#canvas", V.W, V.H);
#else
        // natively the GPU draws in the window itself (or, -gpu offscreen, hands the frames back)
        void *dpy;
        uintptr_t wh;
        gpu_set_vsync(V.o.vsync);
        V.gpu = V.o.gpu != 2 && window_native(V.win, &dpy, &wh) ? gpu_create_window(dpy, wh, V.W, V.H) : gpu_create(NULL, V.W, V.H);
#endif
    }
    if (!V.win) return 1;
    V.fb = fb_create(V.W, V.H);
    V.acc = (Accum){V.W, V.H, 1, V.W, V.H, calloc((size_t)V.W * V.H, sizeof(Vec3)), calloc((size_t)V.W * V.H, sizeof(int)), 0, 0};
    if (!V.fb || !V.acc.sum || !V.acc.count){ fprintf(stderr, "out of memory\n"); return 1; }

    V.view = view_from_scene(&V.scene);
    V.home = V.view;
    V.mode = V.o.backend;
    V.shading = V.o.shading;
    V.max_depth = V.o.max_depth;
    V.preview = V.o.preview;
    V.preview_on = V.o.preview > 1;
    V.speed = V.o.speed;
    V.accel = scene_accel_kind(&V.scene);
    memset(&V.in, 0, sizeof(V.in));
    V.t_start = V.t_prev = V.last_change = V.pass_t0 = V.sim_last = clock_s();
    V.rvol = raster_volumes_create();
    if (!V.rvol){ fprintf(stderr, "out of memory\n"); return 1; }
    V.ema = (Ema){calloc((size_t)V.W * V.H, sizeof(Vec3)), 0, 0};
    if (!V.ema.px){ fprintf(stderr, "out of memory\n"); return 1; }
    return 0;
}

#ifdef __EMSCRIPTEN__
static void frame_web(void){
    if (!viewer_frame()){ viewer_close(); emscripten_cancel_main_loop(); }
}
#endif

void viewer_run(void){
#ifdef __EMSCRIPTEN__
#if defined(__EMSCRIPTEN_PTHREADS__) && defined(_OPENMP)
    printf("CPU: %d threads (the threaded page: cross-origin isolated)\n", omp_get_max_threads());
#else
    printf("CPU: 1 thread (the single-threaded page)\n");
#endif
    emscripten_set_main_loop(frame_web, 0, 1);      // the browser calls frame_web; never returns
#else
    while (viewer_frame()){}
    viewer_close();
#endif
}

// ---- driven by a program (viewer.h) ---------------------------------------------------------

void viewer_set_input(int on){
    V.input_off = !on;
    memset(&V.in, 0, sizeof(V.in));
    if (V.win) window_set_input(V.win, on);
}
void viewer_set_title(int on){ V.title_off = !on; }
void viewer_set_hud(int on){
    V.hud_on = on != 0;
    if (V.gview) gpu_view_hud(V.gview, V.hud_on ? V.hud_px : NULL, V.hud_w, V.hud_h, HUD_X, HUD_Y);
    V.raster_dirty = V.force_present = 1;
}
void viewer_set_settle(double seconds){ V.settle = seconds; }
void viewer_set_keep_trace(int on){ V.keep_trace = on != 0; }

void viewer_set_backend(Backend b){
    if (b == V.mode) return;
    V.mode = b;
    if (b != BACKEND_RASTER) V.reset = 1;      // the path traced image starts over
    V.ext_changed = 1;
}
Backend viewer_backend(void){ return V.mode; }

View viewer_view(void){ return V.view; }
void viewer_set_view(const View *v){
    V.view = *v;
    V.view.pitch = fmax(-1.55, fmin(1.55, V.view.pitch));
    V.ext_changed = 1;
}
void viewer_set_home(const View *v){ V.home = *v; }

void viewer_scene_edited(void){
    V.gpu_scene_edited = V.edit_pending = 1;
    V.edit_t = clock_s();
    viewer_scene_changed(0);
}
void viewer_scene_changed(int restart){
    if (restart) V.reset = 1;                  // the path traced image starts over
    else V.gpu_scene_moved = 1;                // the GPU's carries its samples where nothing changed
    V.raster_dirty = 1;
    V.ext_changed = 1;
}

int viewer_sim_on(void){ return V.o.sim.on; }
void viewer_sim_pause(int paused){ if (V.o.sim.on) V.sim_paused = paused != 0; }
int viewer_sim_paused(void){ return V.o.sim.on && V.sim_paused; }
double viewer_sim_time(void){ return V.o.sim.on ? (double)V.sim.pw.time : 0; }
int viewer_sim_bodies(void){ return V.o.sim.on ? V.sim.pw.rigid.nbodies : 0; }
void viewer_sim_line(char *buf, int len){
    if (len <= 0) return;
    if (V.o.sim.on) sim_status(&V.sim, buf, len);
    else buf[0] = 0;
}
void viewer_sim_pause_at(double t){ V.sim_pause_at = V.o.sim.on && t > 0 ? t : 0; }
unsigned long long viewer_sim_digest(void){
    unsigned long long h = 1469598103934665603ULL;     // FNV-1a over every instance's place and turn
    for (int i = 0; i < V.scene.ninsts; i++){
        const unsigned char *b = (const unsigned char *)&V.scene.insts[i].xf;
        for (size_t k = 0; k < sizeof(Xform); k++){ h ^= b[k]; h *= 1099511628211ULL; }
    }
    return h;
}
int viewer_sim_restart(void){
    if (!V.o.sim.on) return 1;
    if (sim_restart() != 0) return 1;
    viewer_scene_changed(1);
    return 0;
}

void viewer_relight(void){
    scene_trees();                             // the bake traces the scene
    options_bake_probe(&V.o, &V.scene);
    V.raster_dirty = V.force_present = 1;
}

void viewer_redraw(void){ V.raster_dirty = V.force_present = 1; }

int viewer_resize(int w, int h){
    if (w < 1 || h < 1) return 1;
    if (w == V.W && h == V.H) return 0;
    Framebuffer *fb = fb_create(w, h);
    Vec3 *sum = calloc((size_t)w * h, sizeof(Vec3)), *ema = calloc((size_t)w * h, sizeof(Vec3));
    int *count = calloc((size_t)w * h, sizeof(int));
    if (!fb || !sum || !ema || !count || window_resize(V.win, w, h) != 0){
        fb_destroy(fb);
        free(sum);
        free(ema);
        free(count);
        return 1;
    }
    fb_destroy(V.fb);
    V.fb = fb;
    free(V.acc.sum);
    free(V.acc.count);
    V.acc = (Accum){w, h, 1, w, h, sum, count, 0, 0};
    free(V.ema.px);
    V.ema = (Ema){ema, 0, 0};
    V.W = V.o.width = w;
    V.H = V.o.height = h;
    // the GPU: its surface at the new size, and its view made again (viewer_gpu, next frame)
    if (V.gview){
        gpu_view_destroy(V.gview);
        V.gview = NULL;
    }
    if (V.gpu) gpu_resize(V.gpu, w, h);
    fb_destroy(V.gfb);
    V.gfb = NULL;
    V.gpu_tracing = V.gpu_scene = V.gpu_frame_owed = 0;
    V.reset = V.raster_dirty = V.force_present = 1;
    V.ext_changed = 1;
    return 0;
}

void viewer_status(ViewerStatus *st){
    memset(st, 0, sizeof(*st));
    st->tracing = !V.showing_raster;
    st->on_gpu = V.gview != NULL;
    st->gpu_starting = V.gpu && !V.gview;
    st->gpu_tracing = V.gpu_tracing && st->tracing;
    st->spp = st->gpu_tracing ? gpu_view_trace_passes(V.gview) : V.acc.passes;
    st->spp_target = V.o.spp;
    st->fps = V.fps;
    st->width = V.W;
    st->height = V.H;
    st->trace_starts = V.trace_starts;
    st->trace_moves = V.trace_moves;
    st->trace_pass_ms = V.gview ? gpu_view_trace_pass_ms(V.gview) : V.pass_ms;
    st->trace_work = V.gview ? gpu_view_trace_work(V.gview) : 0;
    if (V.gview) gpu_view_trace_band(V.gview, &st->trace_rows, &st->trace_reps, &st->trace_budget_ms);
}
