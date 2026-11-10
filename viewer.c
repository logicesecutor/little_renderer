// viewer.c - real-time window using plain Xlib (Linux / X11).
//
//   make viewer && ./viewer [model.obj] [texture.ppm]
//
// Keys: 1-5 shading mode, space pause/resume, c toggle back-face culling, Esc / q quit.
//
// The only "window" code is: open a window, wrap our pixel array in an XImage, and call
// XPutImage once per frame. The renderer itself does not know any of this exists.
#define _POSIX_C_SOURCE 200809L
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "framebuffer.h"
#include "mesh.h"
#include "renderer.h"
#include "texture.h"

#define W 800
#define H 600

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv){
    const char *model_path = argc > 1 ? argv[1] : "models/torus.obj";
    const char *tex_path = argc > 2 ? argv[2] : NULL;

    Mesh *mesh = mesh_load_obj(model_path);
    if (!mesh){ fprintf(stderr, "could not load %s (run: make models)\n", model_path); return 1; }
    mesh_normalize(mesh);
    if (mesh->nvn == 0) mesh_compute_normals(mesh);

    Texture *tex = tex_path ? texture_load_ppm(tex_path) : NULL;
    if (!tex) tex = texture_checker(256, 16);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy){ fprintf(stderr, "cannot open X display (is DISPLAY set?)\n"); return 1; }
    int scr = DefaultScreen(dpy);
    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), 0, 0, W, H, 0,
                                     BlackPixel(dpy, scr), BlackPixel(dpy, scr));
    XStoreName(dpy, win, "little_renderer");
    XSelectInput(dpy, win, KeyPressMask | ExposureMask | StructureNotifyMask);
    Atom wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wm_delete, 1);
    XMapWindow(dpy, win);
    GC gc = XCreateGC(dpy, win, 0, NULL);

    Framebuffer *fb = fb_create(W, H);
    // The XImage points straight at our pixel array (0x00RRGGBB, 32 bits per pixel).
    XImage *img = XCreateImage(dpy, DefaultVisual(dpy, scr), DefaultDepth(dpy, scr), ZPixmap, 0,
                               (char *)fb->color, W, H, 32, 0);

    RenderParams p = render_default_params(W, H);
    p.texture = tex;
    p.mode = MODE_PHONG;

    float angle = 0.0f;
    int paused = 0, running = 1;
    double last = now_ms(), fps_t0 = last, render_acc = 0;
    int frames = 0;

    while (running){
        while (XPending(dpy)){
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == KeyPress){
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_Escape || k == XK_q) running = 0;
                else if (k == XK_space) paused = !paused;
                else if (k == XK_c) p.cull_backfaces = !p.cull_backfaces;
                else if (k >= XK_1 && k < XK_1 + MODE_COUNT) p.mode = (RenderMode)(k - XK_1);
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete){
                running = 0;
            }
        }

        double t = now_ms();
        if (!paused) angle += (float)(t - last) * 0.001f;   // 1 radian per second
        last = t;
        p.model = mat4_mul(mat4_rotate_x(0.4f), mat4_rotate_y(angle));

        fb_clear(fb, rgb8(24, 26, 32));
        double r0 = now_ms();
        render_mesh(fb, mesh, &p);
        render_acc += now_ms() - r0;

        XPutImage(dpy, win, gc, img, 0, 0, 0, 0, W, H);
        XFlush(dpy);

        frames++;
        if (t - fps_t0 >= 1000.0){
            char title[128];
            snprintf(title, sizeof(title), "little_renderer - %s - %d fps - %.1f ms/frame render",
                     render_mode_name(p.mode), frames, render_acc / frames);
            XStoreName(dpy, win, title);
            printf("%s\n", title);
            frames = 0; render_acc = 0; fps_t0 = t;
        }

        // cap at ~60 fps so we do not spin a core at 100% for nothing
        double spent = now_ms() - t;
        if (spent < 16.0){
            struct timespec ts = {0, (long)((16.0 - spent) * 1e6)};
            nanosleep(&ts, NULL);
        }
    }

    img->data = NULL;            // the pixels belong to the framebuffer, not to Xlib
    XDestroyImage(img);
    XFreeGC(dpy, gc);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    fb_destroy(fb);
    texture_destroy(tex);
    mesh_destroy(mesh);
    return 0;
}
