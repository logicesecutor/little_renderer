// window.c - the window on Linux (and other X11 systems): plain Xlib. On Windows,
// window_win32.c is compiled instead, in a browser window_web.c (this file is then empty).
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include "platform/window_common.h"

#include <X11/XKBlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

struct LrWindow {
    Display *dpy;
    Window win;
    GC gc;
    XImage *ximg;
    uint32_t *pix;           // what X shows, in the display's own pixel layout
    int w, h;
    int rshift, gshift, bshift;
    Atom wm_delete;
    EventQueue later;        // events made here, given before X's (the releases after a FocusOut)
};

// position of the lowest set bit: where a color channel goes in an X pixel
static int mask_shift(unsigned long m){
    int s = 0;
    if (!m) return 0;
    while (!(m & 1)){ m >>= 1; s++; }
    return s;
}

LrWindow *window_open(int width, int height, const char *title){
    Display *dpy = XOpenDisplay(NULL);
    if (!dpy){ fprintf(stderr, "cannot open X display (is DISPLAY set? run it from your desktop)\n"); return NULL; }
    int scr = DefaultScreen(dpy);
    Visual *vis = DefaultVisual(dpy, scr);
    if (DefaultDepth(dpy, scr) < 24){ fprintf(stderr, "need a 24/32-bit display\n"); XCloseDisplay(dpy); return NULL; }

    LrWindow *w = calloc(1, sizeof(*w));
    if (!w){ XCloseDisplay(dpy); return NULL; }
    w->dpy = dpy;
    w->w = width;
    w->h = height;
    w->rshift = mask_shift(vis->red_mask);
    w->gshift = mask_shift(vis->green_mask);
    w->bshift = mask_shift(vis->blue_mask);

    w->win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), 0, 0, width, height, 0,
                                 BlackPixel(dpy, scr), BlackPixel(dpy, scr));
    XSizeHints *hints = XAllocSizeHints();             // fixed size: no resizing
    hints->flags = PMinSize | PMaxSize;
    hints->min_width = hints->max_width = width;
    hints->min_height = hints->max_height = height;
    XSetWMNormalHints(dpy, w->win, hints);
    XFree(hints);
    XStoreName(dpy, w->win, title);
    XSelectInput(dpy, w->win, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                              PointerMotionMask | ExposureMask | StructureNotifyMask | FocusChangeMask);
    w->wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, w->win, &w->wm_delete, 1);
    XkbSetDetectableAutoRepeat(dpy, True, NULL);        // held keys: no fake release events
    XMapWindow(dpy, w->win);
    w->gc = XCreateGC(dpy, w->win, 0, NULL);

    w->pix = calloc((size_t)width * height, sizeof(uint32_t));
    if (w->pix)
        w->ximg = XCreateImage(dpy, vis, DefaultDepth(dpy, scr), ZPixmap, 0, (char *)w->pix, width, height, 32, 0);
    if (!w->pix || !w->ximg){ fprintf(stderr, "out of memory\n"); window_close(w); return NULL; }
    return w;
}

static int translate_key(KeySym k){
    switch (k){
    case XK_Escape: return LR_KEY_ESCAPE;
    case XK_Tab: return LR_KEY_TAB;
    case XK_Left: return LR_KEY_LEFT;
    case XK_Right: return LR_KEY_RIGHT;
    case XK_Up: return LR_KEY_UP;
    case XK_Down: return LR_KEY_DOWN;
    case XK_Shift_L: case XK_Shift_R: return LR_KEY_SHIFT;
    case XK_KP_Add: return '+';
    case XK_KP_Subtract: return '-';
    default: break;
    }
    if (k >= XK_A && k <= XK_Z) return (int)(k - XK_A + 'a');
    if (k >= 32 && k < 127) return (int)k;   // Latin-1 keysyms are the ASCII codes
    return 0;
}

int window_poll(LrWindow *w, LrEvent *ev){
    if (queue_pop(&w->later, ev)) return 1;
    while (XPending(w->dpy)){
        XEvent xe;
        XNextEvent(w->dpy, &xe);
        ev->type = LR_NONE;
        ev->key = ev->button = ev->x = ev->y = ev->wheel = 0;
        switch (xe.type){
        case KeyPress:
        case KeyRelease:
            ev->key = translate_key(XLookupKeysym(&xe.xkey, 0));
            if (!ev->key) continue;
            ev->type = xe.type == KeyPress ? LR_KEYDOWN : LR_KEYUP;
            return 1;
        case ButtonPress:
        case ButtonRelease:
            ev->x = xe.xbutton.x;
            ev->y = xe.xbutton.y;
            if (xe.xbutton.button == Button4 || xe.xbutton.button == Button5){
                if (xe.type == ButtonRelease) continue;   // a wheel "click" is press + release
                ev->type = LR_WHEEL;
                ev->wheel = xe.xbutton.button == Button4 ? 1 : -1;
                return 1;
            }
            ev->type = xe.type == ButtonPress ? LR_BUTTONDOWN : LR_BUTTONUP;
            ev->button = (int)xe.xbutton.button;
            return 1;
        case MotionNotify:
            ev->type = LR_MOTION;
            ev->x = xe.xmotion.x;
            ev->y = xe.xmotion.y;
            return 1;
        case ClientMessage:
            if ((Atom)xe.xclient.data.l[0] != w->wm_delete) continue;
            ev->type = LR_CLOSE;
            return 1;
        case Expose:
            ev->type = LR_EXPOSE;
            return 1;
        case FocusOut:
            queue_release_keys(&w->later);
            return queue_pop(&w->later, ev);
        default:
            continue;
        }
    }
    return 0;
}

void window_present(LrWindow *w, const Framebuffer *fb){
    int n = w->w * w->h;
    if (fb->w != w->w || fb->h != w->h) return;
    if (w->rshift == 16 && w->gshift == 8 && w->bshift == 0){
        for (int i = 0; i < n; i++) w->pix[i] = fb->color[i];       // the usual layout: copy
    } else {
        for (int i = 0; i < n; i++){
            uint32_t c = fb->color[i];
            w->pix[i] = (((c >> 16) & 255) << w->rshift) | (((c >> 8) & 255) << w->gshift) | ((c & 255) << w->bshift);
        }
    }
    XPutImage(w->dpy, w->win, w->gc, w->ximg, 0, 0, 0, 0, w->w, w->h);
    XFlush(w->dpy);
}

// The display's refresh rate from XRandR (1.2: the mode of the monitor under the window's
// centre), loaded when there (libXrandr.so.2) so that nothing needs it to build or run. Its
// few types are declared here as Xrandr.h has them (they have not changed since 2007).
typedef XID LrRRCrtc, LrRRMode, LrRROutput;
typedef struct {
    LrRRMode id;
    unsigned int width, height;
    unsigned long dotClock;
    unsigned int hSyncStart, hSyncEnd, hTotal, hSkew;
    unsigned int vSyncStart, vSyncEnd, vTotal;
    char *name;
    unsigned int nameLength;
    unsigned long modeFlags;
} LrRRModeInfo;
typedef struct {
    Time timestamp, configTimestamp;
    int ncrtc;
    LrRRCrtc *crtcs;
    int noutput;
    LrRROutput *outputs;
    int nmode;
    LrRRModeInfo *modes;
} LrRRScreenResources;
typedef struct {
    Time timestamp;
    int x, y;
    unsigned int width, height;
    LrRRMode mode;
    unsigned short rotation;
    int noutput;
    LrRROutput *outputs;
    unsigned short rotations;
    int npossible;
    LrRROutput *possible;
} LrRRCrtcInfo;
enum { LR_RR_INTERLACE = 0x10, LR_RR_DOUBLESCAN = 0x20 };

double window_refresh_hz(LrWindow *w){
    static void *lib;
    static int tried;
    static LrRRScreenResources *(*get_res)(Display *, Window);
    static void (*free_res)(LrRRScreenResources *);
    static LrRRCrtcInfo *(*get_crtc)(Display *, LrRRScreenResources *, LrRRCrtc);
    static void (*free_crtc)(LrRRCrtcInfo *);
    if (!tried){
        tried = 1;
        lib = dlopen("libXrandr.so.2", RTLD_LAZY | RTLD_LOCAL);
        if (lib){
            *(void **)&get_res = dlsym(lib, "XRRGetScreenResourcesCurrent");
            *(void **)&free_res = dlsym(lib, "XRRFreeScreenResources");
            *(void **)&get_crtc = dlsym(lib, "XRRGetCrtcInfo");
            *(void **)&free_crtc = dlsym(lib, "XRRFreeCrtcInfo");
        }
    }
    double hz = 0;
    if (w && get_res && free_res && get_crtc && free_crtc){
        int cx = 0, cy = 0;
        Window child;
        Window root = DefaultRootWindow(w->dpy);
        XTranslateCoordinates(w->dpy, w->win, root, w->w / 2, w->h / 2, &cx, &cy, &child);
        LrRRScreenResources *res = get_res(w->dpy, root);
        for (int i = 0; res && i < res->ncrtc && hz <= 0; i++){
            LrRRCrtcInfo *c = get_crtc(w->dpy, res, res->crtcs[i]);
            if (!c) continue;
            if (c->mode && cx >= c->x && cx < c->x + (int)c->width && cy >= c->y && cy < c->y + (int)c->height)
                for (int m = 0; m < res->nmode; m++){
                    const LrRRModeInfo *mi = &res->modes[m];
                    if (mi->id != c->mode || !mi->hTotal || !mi->vTotal) continue;
                    hz = (double)mi->dotClock / ((double)mi->hTotal * mi->vTotal);
                    if (mi->modeFlags & LR_RR_INTERLACE) hz *= 2;
                    if (mi->modeFlags & LR_RR_DOUBLESCAN) hz /= 2;
                }
            free_crtc(c);
        }
        if (res) free_res(res);
    }
    return hz >= 20 && hz <= 1000 ? hz : 60.0;   // (no XRandR, no monitor there, or nonsense)
}

int window_native(LrWindow *w, void **display, uintptr_t *window){
    if (!w) return 0;
    *display = w->dpy;
    *window = (uintptr_t)w->win;
    return 1;
}

void window_set_title(LrWindow *w, const char *title){
    XStoreName(w->dpy, w->win, title);
}

void window_close(LrWindow *w){
    if (!w) return;
    if (w->ximg){ w->ximg->data = NULL; XDestroyImage(w->ximg); }   // the pixels are ours
    free(w->pix);
    if (w->gc) XFreeGC(w->dpy, w->gc);
    XDestroyWindow(w->dpy, w->win);
    XCloseDisplay(w->dpy);
    free(w);
}

#endif   // !_WIN32 && !__EMSCRIPTEN__
