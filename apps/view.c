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
static void render_rows(const Scene *s, const Camera *cam, Accum *a, int y0, int y1, int max_depth, int mis, real clamp){
    TraceParams tp = trace_default_params();
    tp.max_depth = max_depth;
    tp.mis = mis;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = y0; y < y1; y++){
        for (int x = 0; x < a->rw; x++){
            int k = y * a->rw + x;
            // seed from (pixel, sample number): every sample of every pixel is different
            Rng rng;
            rng_seed(&rng, ((uint64_t)k * 0x9E3779B97F4A7C15ULL) ^ ((uint64_t)a->count[k] * 0xD1B54A32D192ED03ULL), (uint64_t)k);
            real u = (x + rng_real(&rng)) / a->rw;
            real v = (y + rng_real(&rng)) / a->rh;
            Vec3 c = trace_path_ex(s, camera_ray(cam, u, v), &rng, &tp);
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

// ---- main ---------------------------------------------------------------------------

int main(int argc, char **argv){
    Options o;
    options_parse(&o, PROG_VIEW, argc, argv);
    static Scene scene;
    static Sim sim;
    if (o.sim.on){
        if (sim_setup(&sim, &o, &scene) != 0) return 1;
        sim.max_dt = 1.0 / 30;                     // one physics step per frame
        sim_step(&sim, &scene, 0);                 // the bodies that start at t = 0
        scene_set_accel(&scene, options_resolve_accel(&o, &scene), &o.ap);
    } else if (options_setup_scene(&o, &scene) != 0) return 1;
    srgb_lut_init();

    const int W = o.width, H = o.height;
    LrWindow *win = window_open(W, H, "view");
    if (!win) return 1;
    Framebuffer *fb = fb_create(W, H);
    Accum acc = {W, H, 1, W, H, calloc((size_t)W * H, sizeof(Vec3)), calloc((size_t)W * H, sizeof(int)), 0, 0};
    if (!fb || !acc.sum || !acc.count){ fprintf(stderr, "out of memory\n"); return 1; }

    printf("\n%s\n", CONTROLS);
    if (o.sim.on) printf("%s\n", SIM_CONTROLS);

    View view = view_from_scene(&scene), home = view;
    Backend mode = o.backend;
    RenderMode shading = o.shading;
    int max_depth = o.max_depth, preview = o.preview, preview_on = o.preview > 1;
    double speed = o.speed;
    AccelKind accel = scene_accel_kind(&scene);
    struct { int fwd, back, left, right, up, down, tl, tr, tu, td, fast, drag, mx, my; } in;
    memset(&in, 0, sizeof(in));

    int running = 1, shot_no = 0, band = 4, force_present = 1;
    int reset = 1;          // path tracer: start over at full resolution
    int raster_dirty = 1;   // rasterizer: draw a new frame
    int showing_raster = 1; // what the window shows right now
    double t_start = now_s(), t_prev = t_start, last_present = 0, last_title = 0, last_change = t_start;
    double pass_t0 = t_start, pass_ms = 0, raster_ms = 0;
    RasterVolumes *rvol = raster_volumes_create();   // the rasterizer's clouds (light grids)
    if (!rvol){ fprintf(stderr, "out of memory\n"); return 1; }
    int sim_paused = 0, sim_moved = 0;
    double sim_last = t_start;
    Ema ema = {calloc((size_t)W * H, sizeof(Vec3)), 0, 0};
    if (!ema.px){ fprintf(stderr, "out of memory\n"); return 1; }
    if (o.sim.on && o.preview == 4){ preview = 2; }   // -sim: 1/2 resolution by default

    while (running){
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
                case 'p': save_view(&acc, fb, showing_raster, shading, o.out, &shot_no); break;
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
                        sim_free(&sim);
                        scene_free(&scene);
                        memset(&scene, 0, sizeof(scene));
                        if (sim_setup(&sim, &o, &scene) != 0) return 1;
                        sim.max_dt = 1.0 / 30;
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
        if (changed){ last_change = t; raster_dirty = 1; }

        // ---- 2b. physics: advance by the time that passed, then the image starts over
        if (o.sim.on){
            double sdt = fmin(t - sim_last, 0.05);
            sim_last = t;
            sim_moved = 0;
            if (!sim_paused && sdt > 0){
                sim_step(&sim, &scene, sdt);
                scene_set_accel(&scene, accel, &o.ap);
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
        if (use_raster){
            if (raster_dirty){
                double r0 = now_s();
                RenderParams p = render_params_camera(view.pos, at, vec3(0, 1, 0), view.fov, W, H);
                p.mode = shading;
                fb_clear(fb, rgb8(0, 0, 0));
                render_scene(fb, &scene, &p);
                if (o.raster_clouds) raster_draw_volumes(rvol, fb, &scene, &p, &o.march);
                raster_ms = (now_s() - r0) * 1000.0;
                raster_dirty = 0;
                force_present = 1;
            } else {
                struct timespec ts = {0, 5 * 1000 * 1000};   // nothing new to draw
                nanosleep(&ts, NULL);
            }
            // the path tracer starts over when it takes back over
            if (acc.passes || acc.band_y || acc.scale != 1) accum_reset(&acc, 1);
            reset = 0;
            pass_t0 = now_s();
        } else if (sim_moved){
            // the simulation moved: one pass of the new state, blended with the last ones
            int sc = preview_on ? preview : 1;
            accum_reset(&acc, sc);
            pass_t0 = now_s();
            for (int i = 0; i < o.move_spp; i++) render_rows(&scene, &cam, &acc, 0, acc.rh, max_depth, o.mis, (real)o.clamp);
            acc.passes = o.move_spp;
            pass_ms = (now_s() - pass_t0) * 1000.0;
            ema_update(&ema, &acc, fb, changed);
            force_present = 1;
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
                for (int i = 0; i < n; i++) render_rows(&scene, &cam, &acc, 0, acc.rh, max_depth, o.mis, (real)o.clamp);
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
            render_rows(&scene, &cam, &acc, acc.band_y, y1, max_depth, o.mis, (real)o.clamp);
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
            struct timespec ts = {0, 10 * 1000 * 1000};   // done refining: idle
            nanosleep(&ts, NULL);
        }

        // ---- 4. show it
        double tn = now_s();
        int refining = !use_raster && !sim_moved && acc.passes < o.spp;   // bands land on screen ~30x per second
        if (force_present || (refining && tn - last_present > 0.033)){
            if (!use_raster && !sim_moved) accum_to_fb(&acc, fb);
            window_present(win, fb);
            showing_raster = use_raster;
            last_present = tn;
            force_present = 0;
        }
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
            }
            else
                snprintf(title, sizeof(title), "view | %s | path tracing %s | %d/%d spp | %.0f ms/pass | 1/%d res | %d bounces | speed %.0f",
                         backend_name(mode), accel_name(accel), acc.passes, o.spp, pass_ms, acc.scale, max_depth, speed);
            if (o.sim.on){
                char st[200];
                sim_status(&sim, st, sizeof(st));
                size_t l = strlen(title);
                snprintf(title + l, sizeof(title) - l, " | %s%s", st, sim_paused ? " | paused" : "");
            }
            window_set_title(win, title);
            last_title = tn;
        }
        if (o.autoquit > 0 && tn - t_start > o.autoquit){
            printf("autoquit after %.1f s (%s): ", tn - t_start, backend_name(mode));
            if (o.sim.on && !sim_paused){ char st[200]; sim_status(&sim, st, sizeof(st)); printf("%s, last pass %.0f ms\n", st, pass_ms); }
            else if (showing_raster) printf("last raster frame %.1f ms\n", raster_ms);
            else printf("%d samples/pixel, last pass %.0f ms\n", acc.passes, pass_ms);
            save_view(&acc, fb, showing_raster, shading, o.out, &shot_no);
            running = 0;
        }
    }

    window_close(win);
    fb_destroy(fb);
    free(acc.sum);
    free(acc.count);
    free(ema.px);
    raster_volumes_free(rvol);
    if (o.sim.on) sim_free(&sim);
    scene_free(&scene);
    return 0;
}
