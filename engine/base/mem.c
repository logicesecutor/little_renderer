// mem.c - running out of memory, and arrays that grow (base/mem.h).
#include "base/mem.h"

#include <stdio.h>
#include <stdlib.h>

static void (*oom_hook)(const char *where);

void lr_set_oom(void (*hook)(const char *where)){ oom_hook = hook; }

_Noreturn void lr_oom(const char *where){
    fprintf(stderr, "out of memory (%s)\n", where);
    if (oom_hook) oom_hook(where);
    exit(1);
}

void *lr_grow_to(void *p, size_t elem, int *cap, int need, int first){
    if (need <= *cap) return p;
    int c = *cap ? *cap : (first > 0 ? first : need);
    while (c < need) c *= 2;
    p = realloc(p, (size_t)c * elem);
    if (!p) lr_oom("growing an array");
    *cap = c;
    return p;
}
