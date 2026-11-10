// rtview.c - real-time viewer: fly through the scene while OUR path tracer (the same
// trace_path() as `raytracer`) refines the picture. Plain Xlib, no other library.
//
//   make rtview && ./rtview [same options as raytracer] [-preview n] [-speed u]
//
// How it stays interactive (progressive rendering):
//   * every pixel keeps a running sum of its samples and a sample count;
//   * while the camera moves, each frame renders ONE sample per pixel at 1/preview
//     resolution (cheap), so the image follows the mouse;
//   * when the camera stops, it switches to full resolution and keeps adding one sample
//     per pixel per pass, a few rows at a time, so input is checked every ~20 ms;
//   * any camera change throws the sums away and starts again.
#define _POSIX_C_SOURCE 200809L
#include <X11/XKBlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "camera.h"
#include "image.h"
#include "options.h"
#include "scene.h"
#include "tracer.h"

static const char *CONTROLS =
    "controls\n"
    "  W A S D          move forward / left / back / right\n"
    "  E / Q            move up / down\n"
    "  mouse drag       look around (left button)\n"
    "  arrow keys       look around\n"
    "  mouse wheel      zoom (field of view)\n"
    "  Shift (hold)     move 4x faster\n"
    "  + / -            movement speed x1.5 / /1.5\n"
    "  [ / ]            fewer / more bounces per path\n"
    "  B                cycle acceleration: none -> octree -> bvh\n"
    "  F                low-resolution preview while moving on/off\n"
    "  1 2 3 4          preview resolution 1/1, 1/2, 1/4, 1/8\n"
    "  R                back to the starting camera\n"
    "  P                save the current image as PPM\n"
    "  C                print a raytracer command for this exact view\n"
    "  H                print these controls\n"
    "  Esc              quit\n";

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

// ---- progressive accumulation buffer ------------------------------------------------

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
static void render_rows(const Scene *s, const Camera *cam, Accum *a, int y0, int y1, int max_depth, double clamp){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = y0; y < y1; y++){
        for (int x = 0; x < a->rw; x++){
            int k = y * a->rw + x;
            // seed from (pixel, sample number): every sample of every pixel is different
            Rng rng;
            rng_seed(&rng, ((uint64_t)k * 0x9E3779B97F4A7C15ULL) ^ ((uint64_t)a->count[k] * 0xD1B54A32D192ED03ULL), (uint64_t)k);
            double u = (x + rng_double(&rng)) / a->rw;
            double v = (y + rng_double(&rng)) / a->rh;
            Vec3 c = trace_path(s, camera_ray(cam, u, v), &rng, max_depth);
            if (!(c.x == c.x && c.y == c.y && c.z == c.z)) c = vec3(0, 0, 0);
            if (clamp > 0 && v_max(c) > clamp) c = v_scale(c, clamp / v_max(c));
            a->sum[k] = v_add(a->sum[k], c);
            a->count[k]++;
        }
    }
}

// ---- display ------------------------------------------------------------------------

static unsigned char srgb_lut[4096];   // linear [0,1] -> sRGB byte, pow() is too slow per pixel
static int rshift, gshift, bshift;     // where R, G, B go in an X11 pixel

static int mask_shift(unsigned long m){
    int s = 0;
    if (!m) return 0;
    while (!(m & 1)){ m >>= 1; s++; }
    return s;
}

static inline int lut_index(double c){
    if (!(c > 0)) return 0;
    if (c >= 1) return 4095;
    return (int)(c * 4095.0 + 0.5);
}

// average -> sRGB -> window pixels. Pixels without samples yet keep their old color, so
// a new pass "paints over" the previous image instead of flashing black.
static void present(const Accum *a, uint32_t *pix){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < a->H; y++){
        int iy = y / a->scale;
        for (int x = 0; x < a->W; x++){
            int k = iy * a->rw + x / a->scale;
            int n = a->count[k];
            if (n == 0) continue;
            Vec3 c = v_scale(a->sum[k], 1.0 / n);
            pix[y * a->W + x] = ((uint32_t)srgb_lut[lut_index(c.x)] << rshift) |
                                ((uint32_t)srgb_lut[lut_index(c.y)] << gshift) |
                                ((uint32_t)srgb_lut[lut_index(c.z)] << bshift);
        }
    }
}

static void save_screenshot(const Accum *a, const char *prefix, int *shot_no){
    Image img = image_create(a->rw, a->rh);
    for (int k = 0; k < a->rw * a->rh; k++)
        img.px[k] = a->count[k] ? v_scale(a->sum[k], 1.0 / a->count[k]) : vec3(0, 0, 0);
    char path[512];
    snprintf(path, sizeof(path), "%s_%03d.ppm", prefix, ++*shot_no);
    if (image_write_ppm(&img, path) == 0)
        printf("saved %s (%dx%d, %d samples/pixel)\n", path, a->rw, a->rh, a->passes);
    else
        fprintf(stderr, "could not write %s (does the folder exist?)\n", path);
    image_free(&img);
}

// ---- main ---------------------------------------------------------------------------

int main(int argc, char **argv){
    Options o;
    options_parse(&o, PROG_RTVIEW, argc, argv);
    static Scene scene;
    if (options_setup_scene(&o, &scene) != 0) return 1;

    const int W = o.width, H = o.height;
    Display *dpy = XOpenDisplay(NULL);
    if (!dpy){ fprintf(stderr, "cannot open X display (is DISPLAY set? run it from your desktop)\n"); return 1; }
    int scr = DefaultScreen(dpy);
    Visual *vis = DefaultVisual(dpy, scr);
    if (DefaultDepth(dpy, scr) < 24){ fprintf(stderr, "need a 24/32-bit display\n"); return 1; }
    rshift = mask_shift(vis->red_mask);
    gshift = mask_shift(vis->green_mask);
    bshift = mask_shift(vis->blue_mask);
    for (int i = 0; i < 4096; i++) srgb_lut[i] = image_srgb8(i / 4095.0);

    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), 0, 0, W, H, 0,
                                     BlackPixel(dpy, scr), BlackPixel(dpy, scr));
    XSizeHints *hints = XAllocSizeHints();             // fixed size: no resizing
    hints->flags = PMinSize | PMaxSize;
    hints->min_width = hints->max_width = W;
    hints->min_height = hints->max_height = H;
    XSetWMNormalHints(dpy, win, hints);
    XFree(hints);
    XStoreName(dpy, win, "rtview");
    XSelectInput(dpy, win, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                           PointerMotionMask | ExposureMask | StructureNotifyMask);
    Atom wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wm_delete, 1);
    XkbSetDetectableAutoRepeat(dpy, True, NULL);        // held keys: no fake release events
    XMapWindow(dpy, win);
    GC gc = XCreateGC(dpy, win, 0, NULL);

    uint32_t *pix = calloc((size_t)W * H, sizeof(uint32_t));
    XImage *ximg = XCreateImage(dpy, vis, DefaultDepth(dpy, scr), ZPixmap, 0, (char *)pix, W, H, 32, 0);
    Accum acc = {W, H, 1, W, H, calloc((size_t)W * H, sizeof(Vec3)), calloc((size_t)W * H, sizeof(int)), 0, 0};
    if (!pix || !ximg || !acc.sum || !acc.count){ fprintf(stderr, "out of memory\n"); return 1; }

    printf("\n%s\n", CONTROLS);

    View view = view_from_scene(&scene), home = view;
    int max_depth = o.max_depth, preview = o.preview, preview_on = o.preview > 1;
    double speed = o.speed;
    AccelKind accel = scene.accel;
    struct { int fwd, back, left, right, up, down, tl, tr, tu, td, fast, drag, mx, my; } in;
    memset(&in, 0, sizeof(in));

    int running = 1, shot_no = 0, band = 4, force_present = 1, reset = 1;
    double t_start = now_s(), t_prev = t_start, last_present = 0, last_title = 0;
    double pass_t0 = t_start, pass_ms = 0;

    while (running){
        // ---- 1. input
        int changed = 0;
        while (XPending(dpy)){
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == KeyPress || ev.type == KeyRelease){
                int down = ev.type == KeyPress;
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                switch (k){
                case XK_w: in.fwd = down; break;
                case XK_s: in.back = down; break;
                case XK_a: in.left = down; break;
                case XK_d: in.right = down; break;
                case XK_e: in.up = down; break;
                case XK_q: in.down = down; break;
                case XK_Left: in.tl = down; break;
                case XK_Right: in.tr = down; break;
                case XK_Up: in.tu = down; break;
                case XK_Down: in.td = down; break;
                case XK_Shift_L: case XK_Shift_R: in.fast = down; break;
                default: break;
                }
                if (!down) continue;
                switch (k){
                case XK_Escape: running = 0; break;
                case XK_r: view = home; changed = 1; break;
                case XK_p: save_screenshot(&acc, o.out, &shot_no); break;
                case XK_c: {
                    Options cur = o;
                    cur.max_depth = max_depth;
                    printf("this view: ");
                    options_print_command(&cur, view.pos, v_add(view.pos, v_scale(view_forward(&view), 500)), view.fov, stdout);
                    fflush(stdout);
                    break;
                }
                case XK_b: {
                    accel = (AccelKind)((accel + 1) % 3);
                    printf("building %s...\n", accel_name(accel));
                    fflush(stdout);
                    scene_set_accel(&scene, accel, &o.ap);
                    char buf[256];
                    scene_accel_stats(&scene, buf, sizeof(buf));
                    printf("%s\n", buf);
                    changed = 1;
                    break;
                }
                case XK_f: preview_on = !preview_on; break;
                case XK_1: preview = 1; preview_on = 0; break;
                case XK_2: preview = 2; preview_on = 1; break;
                case XK_3: preview = 4; preview_on = 1; break;
                case XK_4: preview = 8; preview_on = 1; break;
                case XK_bracketleft: if (max_depth > 1){ max_depth--; changed = 1; } break;
                case XK_bracketright: max_depth++; changed = 1; break;
                case XK_plus: case XK_equal: case XK_KP_Add: speed *= 1.5; break;
                case XK_minus: case XK_KP_Subtract: speed /= 1.5; break;
                case XK_h: printf("\n%s\n", CONTROLS); break;
                default: break;
                }
            } else if (ev.type == ButtonPress){
                if (ev.xbutton.button == Button1){ in.drag = 1; in.mx = ev.xbutton.x; in.my = ev.xbutton.y; }
                if (ev.xbutton.button == Button4 && view.fov > 5){ view.fov -= 2; changed = 1; }    // wheel up
                if (ev.xbutton.button == Button5 && view.fov < 120){ view.fov += 2; changed = 1; }  // wheel down
            } else if (ev.type == ButtonRelease){
                if (ev.xbutton.button == Button1) in.drag = 0;
            } else if (ev.type == MotionNotify && in.drag){
                int dx = ev.xmotion.x - in.mx, dy = ev.xmotion.y - in.my;
                in.mx = ev.xmotion.x;
                in.my = ev.xmotion.y;
                if (dx || dy){
                    view.yaw -= dx * 0.004;      // drag right -> turn right
                    view.pitch -= dy * 0.004;    // drag up -> look up
                    changed = 1;
                }
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete){
                running = 0;
            } else if (ev.type == Expose){
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

        f = view_forward(&view);
        Camera cam = camera_make(view.pos, v_add(view.pos, f), vec3(0, 1, 0), view.fov, (double)W / H);

        // ---- 3. render
        if (changed || reset){
            // Start over. If the camera moved, render one whole pass right away (cheap at
            // 1/preview resolution) so the image follows the input.
            int sc = changed && preview_on ? preview : 1;
            accum_reset(&acc, sc);
            reset = 0;
            pass_t0 = now_s();
            if (changed){
                int n = sc > 1 ? o.move_spp : 1;          // -movespp: less grain, slower
                for (int i = 0; i < n; i++) render_rows(&scene, &cam, &acc, 0, acc.rh, max_depth, o.clamp);
                acc.passes = n;
                pass_ms = (now_s() - pass_t0) * 1000.0;
                pass_t0 = now_s();
                force_present = 1;
                if (sc > 1) reset = 1;  // next calm frame: back to full resolution
            }
        } else if (acc.passes < o.spp){
            // calm: refine at full resolution, a band of rows at a time (~20 ms each)
            double b0 = now_s();
            int y1 = acc.band_y + band < acc.rh ? acc.band_y + band : acc.rh;
            render_rows(&scene, &cam, &acc, acc.band_y, y1, max_depth, o.clamp);
            double bt = now_s() - b0;
            if (bt < 0.010 && band < acc.rh) band *= 2;
            else if (bt > 0.040 && band > 1) band /= 2;
            acc.band_y = y1;
            if (acc.band_y >= acc.rh){
                acc.band_y = 0;
                acc.passes++;
                pass_ms = (now_s() - pass_t0) * 1000.0;
                pass_t0 = now_s();
                force_present = 1;
            }
        } else {
            struct timespec ts = {0, 10 * 1000 * 1000};   // done refining: idle
            nanosleep(&ts, NULL);
        }

        // ---- 4. show it (at most ~30 times per second)
        double tn = now_s();
        if (force_present || tn - last_present > 0.033){
            present(&acc, pix);
            XPutImage(dpy, win, gc, ximg, 0, 0, 0, 0, W, H);
            XFlush(dpy);
            last_present = tn;
            force_present = 0;
        }
        if (tn - last_title > 0.5){
            char title[256];
            snprintf(title, sizeof(title), "rtview | %s | %d/%d spp | %.0f ms/pass | 1/%d res | %d bounces | speed %.0f",
                     accel_name(accel), acc.passes, o.spp, pass_ms, acc.scale, max_depth, speed);
            XStoreName(dpy, win, title);
            last_title = tn;
        }
        if (o.autoquit > 0 && tn - t_start > o.autoquit){
            printf("autoquit after %.1f s: %d samples/pixel, last pass %.0f ms\n", tn - t_start, acc.passes, pass_ms);
            save_screenshot(&acc, o.out, &shot_no);
            running = 0;
        }
    }

    ximg->data = NULL;   // the pixels are ours, not Xlib's
    XDestroyImage(ximg);
    XFreeGC(dpy, gc);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    free(pix);
    free(acc.sum);
    free(acc.count);
    scene_free(&scene);
    return 0;
}
