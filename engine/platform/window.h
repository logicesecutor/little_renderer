// window.h - the only part that talks to the operating system's windowing: X11 / Xlib in
// window.c (Linux), Win32 in window_win32.c (Windows), an HTML canvas in window_web.c (web
// browsers, Emscripten); each compiles to nothing on the others.
// A fixed-size window that shows a Framebuffer and reports keyboard and mouse events.
// Porting the engine to another platform (Cocoa, SDL...) means writing one more such file.
#ifndef LR_WINDOW_H
#define LR_WINDOW_H

#include <stdint.h>
#include "core/image.h"

typedef struct LrWindow LrWindow;   // opaque

typedef enum {
    LR_NONE,
    LR_KEYDOWN, LR_KEYUP,             // key
    LR_BUTTONDOWN, LR_BUTTONUP,       // button (1 = left, 2 = middle, 3 = right), x, y
    LR_MOTION,                        // x, y (mouse position in pixels)
    LR_WHEEL,                         // wheel = +1 (up / away) or -1 (down / towards you)
    LR_CLOSE,                         // the window's close button
    LR_EXPOSE                         // the window needs to be redrawn
} LrEventType;

// keys: printable keys are their lowercase ASCII character ('w', '1', '[', '+', ...)
enum {
    LR_KEY_TAB = 9, LR_KEY_ESCAPE = 27,
    LR_KEY_LEFT = 256, LR_KEY_RIGHT, LR_KEY_UP, LR_KEY_DOWN, LR_KEY_SHIFT
};

typedef struct {
    LrEventType type;
    int key;
    int button;
    int x, y;
    int wheel;
} LrEvent;

// NULL if there is no display (e.g. DISPLAY not set); prints why.
LrWindow *window_open(int width, int height, const char *title);
// Next pending event; returns 0 when there are none (never blocks).
int window_poll(LrWindow *win, LrEvent *ev);
// Shows the framebuffer (its size must match the window).
void window_present(LrWindow *win, const Framebuffer *fb);
void window_set_title(LrWindow *win, const char *title);
// The operating system's handles of the window, for a GPU to draw in it (gpu_create_window):
// X11: the Display * and the Window; Win32: the HINSTANCE and the HWND. 0 in a browser
// (the GPU draws on the canvas there).
int window_native(LrWindow *win, void **display, uintptr_t *window);
void window_close(LrWindow *win);

#endif
