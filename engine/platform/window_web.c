// window_web.c - the window in a web browser (Emscripten): an HTML <canvas id="canvas">.
// The same API as window.c (platform/window.h). The browser delivers keyboard and mouse
// events to callbacks (emscripten/html5.h); they are queued, and window_poll hands them out,
// so the program sees the same events as on Linux and Windows. Frames are copied into the
// canvas with putImageData (the CPU backends; the GPU backend draws on the canvas itself).
#ifdef __EMSCRIPTEN__
#include "platform/window.h"

#include <emscripten.h>
#include <emscripten/html5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QUEUE 512
#define CANVAS "#canvas"

struct LrWindow {
    int w, h;
    uint8_t *rgba;           // the frame as the canvas wants it: R, G, B, A bytes
    LrEvent q[QUEUE];
    int head, tail;
};

static void push(LrWindow *w, LrEventType type, int key, int button, int x, int y, int wheel){
    int next = (w->tail + 1) % QUEUE;
    if (next == w->head) return;
    LrEvent *e = &w->q[w->tail];
    e->type = type; e->key = key; e->button = button; e->x = x; e->y = y; e->wheel = wheel;
    w->tail = next;
}

// KeyboardEvent.key -> the engine's key code (printable keys: their lowercase character)
static int translate_key(const EmscriptenKeyboardEvent *e){
    const char *k = e->key;
    if (!strcmp(k, "Escape")) return LR_KEY_ESCAPE;
    if (!strcmp(k, "Tab")) return LR_KEY_TAB;
    if (!strcmp(k, "ArrowLeft")) return LR_KEY_LEFT;
    if (!strcmp(k, "ArrowRight")) return LR_KEY_RIGHT;
    if (!strcmp(k, "ArrowUp")) return LR_KEY_UP;
    if (!strcmp(k, "ArrowDown")) return LR_KEY_DOWN;
    if (!strcmp(k, "Shift")) return LR_KEY_SHIFT;
    if (k[0] && !k[1]){                           // one character
        int c = (unsigned char)k[0];
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        if (c >= 32 && c < 127) return c;
    }
    return 0;
}

static EM_BOOL on_key(int type, const EmscriptenKeyboardEvent *e, void *user){
    LrWindow *w = user;
    int k = translate_key(e);
    if (!k) return EM_FALSE;
    if (e->ctrlKey || e->metaKey || e->altKey) return EM_FALSE;   // leave browser shortcuts alone
    if (type == EMSCRIPTEN_EVENT_KEYDOWN){ if (!e->repeat) push(w, LR_KEYDOWN, k, 0, 0, 0, 0); }
    else push(w, LR_KEYUP, k, 0, 0, 0, 0);
    return EM_TRUE;                               // handled: no page scrolling on arrows / space
}

static EM_BOOL on_mouse(int type, const EmscriptenMouseEvent *e, void *user){
    LrWindow *w = user;
    int x = (int)e->targetX, y = (int)e->targetY;
    int b = e->button == 0 ? 1 : (e->button == 1 ? 2 : 3);
    if (type == EMSCRIPTEN_EVENT_MOUSEDOWN) push(w, LR_BUTTONDOWN, 0, b, x, y, 0);
    else if (type == EMSCRIPTEN_EVENT_MOUSEUP) push(w, LR_BUTTONUP, 0, b, x, y, 0);
    else push(w, LR_MOTION, 0, 0, x, y, 0);
    return EM_TRUE;
}

static EM_BOOL on_wheel(int type, const EmscriptenWheelEvent *e, void *user){
    (void)type;
    LrWindow *w = user;
    if (e->deltaY != 0) push(w, LR_WHEEL, 0, 0, 0, 0, e->deltaY < 0 ? 1 : -1);
    return EM_TRUE;                               // do not scroll the page
}

static EM_BOOL on_blur(int type, const EmscriptenFocusEvent *e, void *user){
    (void)type; (void)e;
    LrWindow *w = user;                           // keys released elsewhere never arrive: release all
    push(w, LR_KEYUP, LR_KEY_SHIFT, 0, 0, 0, 0);
    for (const char *k = "wasdeq"; *k; k++) push(w, LR_KEYUP, *k, 0, 0, 0, 0);
    push(w, LR_KEYUP, LR_KEY_LEFT, 0, 0, 0, 0);
    push(w, LR_KEYUP, LR_KEY_RIGHT, 0, 0, 0, 0);
    push(w, LR_KEYUP, LR_KEY_UP, 0, 0, 0, 0);
    push(w, LR_KEYUP, LR_KEY_DOWN, 0, 0, 0, 0);
    return EM_FALSE;
}

EM_JS(void, js_present, (const uint8_t *rgba, int w, int h), {
    const c = document.querySelector('#canvas');
    if (!c) return;
    if (!Module.lrCtx) Module.lrCtx = c.getContext('2d');
    if (!Module.lrCtx) {   // the canvas already shows WebGPU: it cannot take 2D drawings too
        if (!Module.lrNo2d) { Module.lrNo2d = 1; err('the canvas is taken by WebGPU: reload with ?gpu=off or ?gpu=offscreen'); }
        return;
    }
    // a copy: the ImageData must not point into the WebAssembly memory, which can grow
    const px = new Uint8ClampedArray(HEAPU8.subarray(rgba, rgba + w * h * 4));
    Module.lrCtx.putImageData(new ImageData(px, w, h), 0, 0);
});

EM_JS(void, js_set_title, (const char *t), {
    const s = UTF8ToString(t);
    const el = document.getElementById('title');
    if (el) el.textContent = s; else document.title = s;
});

LrWindow *window_open(int width, int height, const char *title){
    LrWindow *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->w = width;
    w->h = height;
    w->rgba = malloc((size_t)width * height * 4);
    if (!w->rgba){ free(w); return NULL; }
    emscripten_set_canvas_element_size(CANVAS, width, height);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, w, 1, on_key);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, w, 1, on_key);
    emscripten_set_mousedown_callback(CANVAS, w, 1, on_mouse);
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, w, 1, on_mouse);   // also outside
    emscripten_set_mousemove_callback(CANVAS, w, 1, on_mouse);
    emscripten_set_wheel_callback(CANVAS, w, 1, on_wheel);
    emscripten_set_blur_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, w, 1, on_blur);
    js_set_title(title);
    return w;
}

int window_poll(LrWindow *w, LrEvent *ev){
    if (w->head == w->tail) return 0;
    *ev = w->q[w->head];
    w->head = (w->head + 1) % QUEUE;
    return 1;
}

void window_present(LrWindow *w, const Framebuffer *fb){
    if (fb->w != w->w || fb->h != w->h) return;
    size_t n = (size_t)w->w * w->h;
    for (size_t i = 0; i < n; i++){
        uint32_t c = fb->color[i];
        w->rgba[4 * i + 0] = (uint8_t)(c >> 16);
        w->rgba[4 * i + 1] = (uint8_t)(c >> 8);
        w->rgba[4 * i + 2] = (uint8_t)c;
        w->rgba[4 * i + 3] = 255;
    }
    js_present(w->rgba, w->w, w->h);
}

int window_native(LrWindow *w, void **display, uintptr_t *window){
    (void)w;
    *display = NULL;
    *window = 0;
    return 0;                                   // the GPU draws on the canvas itself
}

void window_set_title(LrWindow *w, const char *title){
    (void)w;
    js_set_title(title);
}

void window_close(LrWindow *w){
    if (!w) return;
    free(w->rgba);
    free(w);
}

#endif   // __EMSCRIPTEN__
