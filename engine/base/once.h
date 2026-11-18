// once.h - a table made the first time it is wanted, once, whichever threads ask for it at
// the same moment (the engine's lookup tables: sRGB, the PNG checksum and Huffman codes, the
// blue noise). The first to come makes it; the others wait until it is made.
//
//     static LrOnce made;
//     if (lr_once_begin(&made)){ ... fill the table ...; lr_once_done(&made); }
//     ... the table is filled, for every thread ...
#ifndef LR_ONCE_H
#define LR_ONCE_H

#include <stdatomic.h>
#include "base/spin.h"   // lr_cpu_relax

typedef atomic_int LrOnce;   // 0 not yet, 1 being made, 2 made (a static starts at 0)

// 1 for the caller that is to make it (it then calls lr_once_done); 0 once it is made
static inline int lr_once_begin(LrOnce *o){
    if (atomic_load_explicit(o, memory_order_acquire) == 2) return 0;
    int z = 0;
    if (atomic_compare_exchange_strong_explicit(o, &z, 1, memory_order_acquire, memory_order_acquire)) return 1;
    while (atomic_load_explicit(o, memory_order_acquire) != 2){ lr_cpu_relax(); sched_yield(); }   // another makes it
    return 0;
}
static inline void lr_once_done(LrOnce *o){ atomic_store_explicit(o, 2, memory_order_release); }
static inline int lr_once_made(LrOnce *o){ return atomic_load_explicit(o, memory_order_acquire) == 2; }

#endif
