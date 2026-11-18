// window_common.h - what the three windows (window.c, window_win32.c, window_web.c) share,
// private to platform/: the queue of events waiting for window_poll, and the keys released
// when the window loses the focus.
#ifndef LR_WINDOW_COMMON_H
#define LR_WINDOW_COMMON_H

#include "platform/window.h"

#define EVENT_QUEUE 512

typedef struct {
    LrEvent q[EVENT_QUEUE];  // a ring buffer
    int head, tail;
} EventQueue;

static inline int queue_empty(const EventQueue *q){ return q->head == q->tail; }

static inline void queue_push(EventQueue *q, LrEventType type, int key, int button, int x, int y, int wheel){
    int next = (q->tail + 1) % EVENT_QUEUE;
    if (next == q->head) return;                  // full: the event dropped
    LrEvent *e = &q->q[q->tail];
    e->type = type; e->key = key; e->button = button; e->x = x; e->y = y; e->wheel = wheel;
    q->tail = next;
}

static inline int queue_pop(EventQueue *q, LrEvent *ev){
    if (queue_empty(q)) return 0;
    *ev = q->q[q->head];
    q->head = (q->head + 1) % EVENT_QUEUE;
    return 1;
}

// The window lost the focus: the keys released whose real releases go to the window that has
// it now (Win32's WM_KILLFOCUS, the page's blur, X11's FocusOut), so that a movement key held
// then does not stay down
static inline void queue_release_keys(EventQueue *q){
    static const int keys[] = {LR_KEY_SHIFT, 'w', 'a', 's', 'd', 'e', 'q', LR_KEY_LEFT, LR_KEY_RIGHT, LR_KEY_UP, LR_KEY_DOWN};
    for (int i = 0; i < (int)(sizeof(keys) / sizeof(keys[0])); i++) queue_push(q, LR_KEYUP, keys[i], 0, 0, 0, 0);
}

#endif
