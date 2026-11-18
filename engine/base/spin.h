// spin.h - a barrier for the threads of one OpenMP parallel region that spins instead of
// sleeping (docs/PERF_REVIEW.md, N1).
//
// Why: natively on Windows libgomp's threads sleep in the kernel between regions and in its
// barriers (OMP_WAIT_POLICY=active changes nothing there), so each region or barrier wakes
// every thread again: 50-100 us, which a small loop (the air's 32^3 grid) cannot pay back.
// A step made of many short phases runs them in one region instead, its threads meeting
// here between the phases: they spin a while (pause), then give their core away (yield)
// while they wait, but never sleep. Only for waits that are short: every thread of the
// region spins at once.
//
//     LrBarrier bar; lr_barrier_init(&bar);
//     #pragma omp parallel
//     { ... phase 1 ...; lr_barrier_wait(&bar, omp_get_num_threads()); ... phase 2 ... }
#ifndef LR_SPIN_H
#define LR_SPIN_H

#include <stdatomic.h>
#include <sched.h>

typedef struct {
    atomic_int arrived;   // threads at the barrier in this round
    atomic_int round;     // rounds completed: the waiting threads watch it change
} LrBarrier;

static inline void lr_barrier_init(LrBarrier *b){
    atomic_init(&b->arrived, 0);
    atomic_init(&b->round, 0);
}

static inline void lr_cpu_relax(void){
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// Waits until all n threads of the region have called it (n = omp_get_num_threads(), the
// same for all). What each thread wrote before it is seen by every thread after it.
static inline void lr_barrier_wait(LrBarrier *b, int n){
    if (n <= 1) return;
    // the round cannot end before this thread arrives, so this is the round it waits in
    int r = atomic_load_explicit(&b->round, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&b->arrived, 1, memory_order_acq_rel) == n - 1){
        atomic_store_explicit(&b->arrived, 0, memory_order_relaxed);
        atomic_store_explicit(&b->round, r + 1, memory_order_release);   // (publishes the reset too)
        return;
    }
    for (unsigned s = 0; atomic_load_explicit(&b->round, memory_order_acquire) == r; s++){
        if (s < 4096) lr_cpu_relax();   // ~0.1-0.2 ms of spinning
        else sched_yield();             // then the core goes to whoever needs it
    }
}

#endif
