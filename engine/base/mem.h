// mem.h - running out of memory, one way for the whole engine; arrays that grow.
#ifndef LR_MEM_H
#define LR_MEM_H

#include <stddef.h>

// Memory ran out where the engine cannot go on without it (a scene, a tree, the solver's
// arrays): says so ("out of memory (where)") and calls the program's hook (lr_set_oom); with
// none, or if it returns, the program ends (exit 1). (Loading a file - formats/, glTF, the
// GPU - fails with an error instead: a file can be too big, the program goes on.)
_Noreturn void lr_oom(const char *where);
void lr_set_oom(void (*hook)(const char *where));   // e.g. to save what can be saved first

// p, an array of *cap elements of elem bytes, with room for at least need: its size doubled
// (from first, or from need when first is 0, the first time) until they fit. lr_oom if not.
// (The test inline: it is on paths that add one element at a time.)
void *lr_grow_to(void *p, size_t elem, int *cap, int need, int first);
static inline void *lr_grow(void *p, size_t elem, int *cap, int need, int first){
    return need <= *cap ? p : lr_grow_to(p, elem, cap, need, first);
}

#endif
