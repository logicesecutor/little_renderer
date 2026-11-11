// window_win32.c - the window on Windows: plain Win32 (user32 + gdi32), no other library.
// The same API as window.c (platform/window.h). Messages from Windows arrive in a window
// procedure (a callback); it turns them into LrEvents in a small queue that window_poll
// empties, so the program sees the same events as on Linux.
#ifdef _WIN32
#include "platform/window.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <stdlib.h>

#define QUEUE 512

struct LrWindow {
    HWND hwnd;
    int w, h;
    BITMAPINFO bmi;          // describes fb->color to GDI: 32-bit, top-down
    LrEvent q[QUEUE];        // events waiting for window_poll (a ring buffer)
    int head, tail;
    int buttons;             // mouse buttons held (for SetCapture)
};

static void push(LrWindow *w, LrEventType type, int key, int button, int x, int y, int wheel){
    int next = (w->tail + 1) % QUEUE;
    if (next == w->head) return;                  // full: drop the event
    LrEvent *e = &w->q[w->tail];
    e->type = type; e->key = key; e->button = button; e->x = x; e->y = y; e->wheel = wheel;
    w->tail = next;
}

// virtual key -> the engine's key code: printable keys are their lowercase character,
// as on X11 (where a key reports its unshifted symbol)
static int translate_key(WPARAM vk){
    switch (vk){
    case VK_ESCAPE: return LR_KEY_ESCAPE;
    case VK_TAB: return LR_KEY_TAB;
    case VK_LEFT: return LR_KEY_LEFT;
    case VK_RIGHT: return LR_KEY_RIGHT;
    case VK_UP: return LR_KEY_UP;
    case VK_DOWN: return LR_KEY_DOWN;
    case VK_SHIFT: case VK_LSHIFT: case VK_RSHIFT: return LR_KEY_SHIFT;
    case VK_SPACE: return ' ';
    case VK_ADD: return '+';
    case VK_SUBTRACT: return '-';
    case VK_OEM_PLUS: return '=';                 // the "= +" key (the viewer takes both)
    case VK_OEM_MINUS: return '-';
    case VK_OEM_4: return '[';
    case VK_OEM_6: return ']';
    case VK_OEM_COMMA: return ',';
    case VK_OEM_PERIOD: return '.';
    default: break;
    }
    if (vk >= 'A' && vk <= 'Z') return (int)(vk - 'A' + 'a');
    if (vk >= '0' && vk <= '9') return (int)vk;
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) return (int)(vk - VK_NUMPAD0 + '0');
    return 0;
}

static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp){
    if (msg == WM_NCCREATE){
        CREATESTRUCTA *cs = (CREATESTRUCTA *)lp;
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
        return DefWindowProcA(hwnd, msg, wp, lp);
    }
    LrWindow *w = (LrWindow *)GetWindowLongPtrA(hwnd, GWLP_USERDATA);
    if (!w) return DefWindowProcA(hwnd, msg, wp, lp);
    int x = GET_X_LPARAM(lp), y = GET_Y_LPARAM(lp);
    switch (msg){
    case WM_CLOSE:
        push(w, LR_CLOSE, 0, 0, 0, 0, 0);         // the program decides; do not destroy yet
        return 0;
    case WM_KEYDOWN: case WM_SYSKEYDOWN: {
        int k = translate_key(wp);
        if (k && !(lp & (1 << 30))) push(w, LR_KEYDOWN, k, 0, 0, 0, 0);   // bit 30: auto-repeat
        if (msg == WM_SYSKEYDOWN) break;          // let Alt+F4 & co. through
        return 0;
    }
    case WM_KEYUP: case WM_SYSKEYUP: {
        int k = translate_key(wp);
        if (k) push(w, LR_KEYUP, k, 0, 0, 0, 0);
        if (msg == WM_SYSKEYUP) break;
        return 0;
    }
    case WM_LBUTTONDOWN: case WM_MBUTTONDOWN: case WM_RBUTTONDOWN: {
        int b = msg == WM_LBUTTONDOWN ? 1 : (msg == WM_MBUTTONDOWN ? 2 : 3);
        if (!w->buttons++) SetCapture(hwnd);      // keep getting the mouse while dragging out
        push(w, LR_BUTTONDOWN, 0, b, x, y, 0);
        return 0;
    }
    case WM_LBUTTONUP: case WM_MBUTTONUP: case WM_RBUTTONUP: {
        int b = msg == WM_LBUTTONUP ? 1 : (msg == WM_MBUTTONUP ? 2 : 3);
        if (w->buttons > 0 && !--w->buttons) ReleaseCapture();
        push(w, LR_BUTTONUP, 0, b, x, y, 0);
        return 0;
    }
    case WM_MOUSEMOVE:
        push(w, LR_MOTION, 0, 0, x, y, 0);
        return 0;
    case WM_MOUSEWHEEL:                           // its coordinates are on the screen: unused
        push(w, LR_WHEEL, 0, 0, 0, 0, GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 1 : -1);
        return 0;
    case WM_KILLFOCUS:
        // keys released while another window has the focus never arrive here: release all
        push(w, LR_KEYUP, LR_KEY_SHIFT, 0, 0, 0, 0);
        for (const char *k = "wasdeq"; *k; k++) push(w, LR_KEYUP, *k, 0, 0, 0, 0);
        push(w, LR_KEYUP, LR_KEY_LEFT, 0, 0, 0, 0);
        push(w, LR_KEYUP, LR_KEY_RIGHT, 0, 0, 0, 0);
        push(w, LR_KEYUP, LR_KEY_UP, 0, 0, 0, 0);
        push(w, LR_KEYUP, LR_KEY_DOWN, 0, 0, 0, 0);
        return 0;
    case WM_ERASEBKGND:
        return 1;                                 // we paint every pixel: no flicker
    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);
        EndPaint(hwnd, &ps);
        push(w, LR_EXPOSE, 0, 0, 0, 0, 0);
        return 0;
    }
    default:
        break;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

LrWindow *window_open(int width, int height, const char *title){
    static int registered = 0;
    HINSTANCE inst = GetModuleHandleA(NULL);
    // one image pixel = one screen pixel, also on scaled (high-DPI) displays
    SetProcessDPIAware();
    if (!registered){
        WNDCLASSA wc;
        ZeroMemory(&wc, sizeof(wc));
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = window_proc;
        wc.hInstance = inst;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wc.lpszClassName = "little_renderer";
        if (!RegisterClassA(&wc)){ fprintf(stderr, "RegisterClass failed (%lu)\n", GetLastError()); return NULL; }
        registered = 1;
    }
    LrWindow *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->w = width;
    w->h = height;
    BITMAPINFOHEADER *bh = &w->bmi.bmiHeader;
    bh->biSize = sizeof(*bh);
    bh->biWidth = width;
    bh->biHeight = -height;                       // negative: row 0 at the top, like ours
    bh->biPlanes = 1;
    bh->biBitCount = 32;                          // 0x00RRGGBB in memory = B, G, R, 0: what GDI expects
    bh->biCompression = BI_RGB;

    // a fixed-size window whose inside (client area) is exactly width x height
    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT r = {0, 0, width, height};
    AdjustWindowRect(&r, style, FALSE);
    w->hwnd = CreateWindowExA(0, "little_renderer", title, style, CW_USEDEFAULT, CW_USEDEFAULT,
                              r.right - r.left, r.bottom - r.top, NULL, NULL, inst, w);
    if (!w->hwnd){ fprintf(stderr, "CreateWindow failed (%lu)\n", GetLastError()); free(w); return NULL; }
    ShowWindow(w->hwnd, SW_SHOWNORMAL);
    UpdateWindow(w->hwnd);
    SetForegroundWindow(w->hwnd);
    return w;
}

int window_poll(LrWindow *w, LrEvent *ev){
    MSG msg;
    // let Windows deliver what it has (into the queue) until there is an event for us
    while (w->head == w->tail && PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)){
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    if (w->head == w->tail) return 0;
    *ev = w->q[w->head];
    w->head = (w->head + 1) % QUEUE;
    return 1;
}

void window_present(LrWindow *w, const Framebuffer *fb){
    if (fb->w != w->w || fb->h != w->h) return;
    HDC dc = GetDC(w->hwnd);
    SetDIBitsToDevice(dc, 0, 0, (DWORD)w->w, (DWORD)w->h, 0, 0, 0, (UINT)w->h, fb->color, &w->bmi, DIB_RGB_COLORS);
    ReleaseDC(w->hwnd, dc);
}

int window_native(LrWindow *w, void **display, uintptr_t *window){
    if (!w) return 0;
    *display = (void *)GetModuleHandleA(NULL);
    *window = (uintptr_t)w->hwnd;
    return 1;
}

void window_set_title(LrWindow *w, const char *title){
    SetWindowTextA(w->hwnd, title);
}

void window_close(LrWindow *w){
    if (!w) return;
    SetWindowLongPtrA(w->hwnd, GWLP_USERDATA, 0);
    DestroyWindow(w->hwnd);
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) DispatchMessageA(&msg);
    free(w);
}

#endif   // _WIN32
