// omp_web.c - the part of an OpenMP runtime that the engine's `#pragma omp` loops need, on
// pthreads, for the threaded web build (make web THREADS=1: Emscripten has no libomp). The
// compiler (clang -fopenmp) turns each pragma into calls of the __kmpc_* functions below (the
// ABI of LLVM's libomp); natively the real libgomp / libomp is used and this file is empty.
//
// What it does: a team of threads made once (navigator.hardwareConcurrency of them, the
// workers Emscripten has ready: -sPTHREAD_POOL_SIZE), parallel regions (`parallel`, with
// `if`), loops with schedule(static) and schedule(dynamic, n), reductions, `critical`,
// `single`, barriers, tasks with taskwait (the BVH builder's), `atomic` (the compiler's own),
// and omp_get_num_procs / omp_get_max_threads / omp_set_num_threads / omp_get_thread_num /
// omp_get_num_threads. Nested regions run on the thread that meets them (as libgomp does by
// default). Not done (the engine does not use them): ordered, sections, locks, guided and
// runtime schedules (run as dynamic), cancellation, task dependences.
#if defined(__EMSCRIPTEN__) && defined(_OPENMP) && defined(__EMSCRIPTEN_PTHREADS__)
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/threading.h>

typedef int32_t kmp_int32;
typedef uint32_t kmp_uint32;
typedef int64_t kmp_int64;
typedef uint64_t kmp_uint64;
typedef struct ident ident_t;
typedef void (*kmpc_micro)(kmp_int32 *gtid, kmp_int32 *btid, ...);
typedef kmp_int32 (*kmp_routine)(kmp_int32, void *);
typedef kmp_int32 kmp_critical_name[8];

#define MAX_THREADS 64
#define MAX_ARGS 16

// ---- the team ----
static struct {
    int made, size;                   // threads made (with the main one), the team's size now
    int want;                         // omp_set_num_threads (0: all)
    pthread_t tid[MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t go;
    // the region being run: its microtask and arguments, its size; job counts the regions
    kmpc_micro fn;
    void *args[MAX_ARGS];
    int argc, nth;
    _Atomic unsigned job;
    _Atomic int done;                 // threads finished with the region
    // barrier
    _Atomic int bar_count;
    _Atomic unsigned bar_gen;
    // the loop being dispatched (dynamic schedules): its iterations 0..trip-1 handed out
    _Atomic unsigned disp_seq, disp_ready;
    _Atomic int64_t disp_next;
    int64_t disp_lb, disp_trip, disp_inc, disp_chunk;
    _Atomic unsigned single_seq;
    // tasks: a stack of those not yet run, the number not yet finished
    pthread_mutex_t task_mu;
    struct Task *tasks;
    _Atomic int outstanding;
    pthread_mutex_t red_mu;           // reductions
} T = {.mu = PTHREAD_MUTEX_INITIALIZER, .go = PTHREAD_COND_INITIALIZER,
       .task_mu = PTHREAD_MUTEX_INITIALIZER, .red_mu = PTHREAD_MUTEX_INITIALIZER};
static pthread_mutex_t crit_mu = PTHREAD_MUTEX_INITIALIZER;

static _Thread_local int my_id;              // in the team
static _Thread_local int in_region;          // inside a parallel region (nested ones: alone)
static _Thread_local int serial_depth;       // inside a region run alone (if(0), nested)
static _Thread_local unsigned my_disp, my_single;
static _Thread_local struct Task *my_task;   // the task this thread runs (its children counted there)

struct Task {
    struct Task *next, *parent;
    _Atomic int children;
    // the compiler's kmp_task_t follows (shareds, routine, part_id, ...), then the shareds
};
#define TASK_OF(t) ((void *)((char *)(t) + sizeof(struct Task)))
#define HEAD_OF(k) ((struct Task *)((char *)(k) - sizeof(struct Task)))
typedef struct { void *shareds; kmp_routine routine; kmp_int32 part_id; } kmp_task_t;

// waiting: a worker gives its turn up; the browser's main thread (which may not block) does what
// the workers asked of it meanwhile (Emscripten proxies some of their calls to it)
static void relax(void){
    if (emscripten_is_main_browser_thread()) emscripten_current_thread_process_queued_calls();
    else sched_yield();
}
static int procs(void){ int n = emscripten_num_logical_cores(); return n < 1 ? 1 : n > MAX_THREADS ? MAX_THREADS : n; }
static int alone(void){ return serial_depth > 0 || !in_region; }   // (outside a region: a team of one)
static int team_size(void){ return alone() ? 1 : T.nth; }

// The microtask takes exactly 2 + n arguments (it is not variadic): wasm checks a call's
// signature, so each count of arguments has its own call through its own type.
#define P void *
static void call(kmpc_micro fn, kmp_int32 *g, kmp_int32 *b, void **a, int n){
    switch (n){
    case 0: ((void (*)(kmp_int32 *, kmp_int32 *))fn)(g, b); break;
    case 1: ((void (*)(kmp_int32 *, kmp_int32 *, P))fn)(g, b, a[0]); break;
    case 2: ((void (*)(kmp_int32 *, kmp_int32 *, P, P))fn)(g, b, a[0], a[1]); break;
    case 3: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P))fn)(g, b, a[0], a[1], a[2]); break;
    case 4: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3]); break;
    case 5: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4]); break;
    case 6: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5]); break;
    case 7: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
    case 8: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
    case 9: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); break;
    case 10: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); break;
    case 11: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]); break;
    case 12: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]); break;
    case 13: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]); break;
    case 14: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]); break;
    case 15: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]); break;
    default: ((void (*)(kmp_int32 *, kmp_int32 *, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(g, b, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]); break;
    }
}
#undef P

// ---- tasks ----
static void run_task(struct Task *t){
    struct Task *was = my_task;
    my_task = t;
    kmp_task_t *k = TASK_OF(t);
    k->routine(my_id, k);
    my_task = was;
    if (t->parent) atomic_fetch_sub(&t->parent->children, 1);
    atomic_fetch_sub(&T.outstanding, 1);
    free(t);
}
static int run_one(void){
    pthread_mutex_lock(&T.task_mu);
    struct Task *t = T.tasks;
    if (t) T.tasks = t->next;
    pthread_mutex_unlock(&T.task_mu);
    if (!t) return 0;
    run_task(t);
    return 1;
}

// ---- barrier: every thread of the team, and every task (threads that wait run them) ----
static void barrier(void){
    if (team_size() == 1){ while (run_one()){} return; }
    unsigned gen = atomic_load(&T.bar_gen);
    if (atomic_fetch_add(&T.bar_count, 1) == T.nth - 1){
        while (atomic_load(&T.outstanding) > 0) if (!run_one()) relax();
        atomic_store(&T.bar_count, 0);
        atomic_fetch_add(&T.bar_gen, 1);
    } else {
        while (atomic_load(&T.bar_gen) == gen) if (!run_one()) relax();
    }
}

// ---- the threads ----
static void *worker(void *arg){
    my_id = (int)(intptr_t)arg;
    unsigned seen = 0;
    for (;;){
        pthread_mutex_lock(&T.mu);
        while (atomic_load(&T.job) == seen) pthread_cond_wait(&T.go, &T.mu);
        seen = atomic_load(&T.job);
        int mine = my_id < T.nth;
        pthread_mutex_unlock(&T.mu);
        if (mine){
            kmp_int32 g = my_id, b = my_id;
            in_region = 1;
            my_disp = my_single = 0;          // (the region's loops and singles, counted from 0)
            call(T.fn, &g, &b, T.args, T.argc);
            barrier();                        // (the region's end: its tasks done)
            in_region = 0;
            atomic_fetch_add(&T.done, 1);
        }
    }
    return NULL;
}

static void make_team(void){
    if (T.made) return;
    T.made = 1;
    int n = procs();
    for (int i = 1; i < n; i++)
        if (pthread_create(&T.tid[i], NULL, worker, (void *)(intptr_t)i) == 0) T.made = i + 1;
    T.size = T.made;
}

static void fork_region(kmpc_micro fn, int argc, void **args){
    if (in_region || serial_depth){            // nested: alone, on this thread
        kmp_int32 g = my_id, b = 0;
        serial_depth++;
        call(fn, &g, &b, args, argc);
        serial_depth--;
        return;
    }
    make_team();
    int nth = T.want > 0 && T.want < T.made ? T.want : T.made;
    if (nth <= 1){
        kmp_int32 g = 0, b = 0;
        serial_depth++;
        call(fn, &g, &b, args, argc);
        serial_depth--;
        return;
    }
    pthread_mutex_lock(&T.mu);
    T.fn = fn;
    T.argc = argc > MAX_ARGS ? MAX_ARGS : argc;
    memcpy(T.args, args, sizeof(void *) * (size_t)T.argc);
    T.nth = nth;
    atomic_store(&T.done, 0);
    atomic_store(&T.bar_count, 0);
    atomic_store(&T.disp_seq, 0);
    atomic_store(&T.disp_ready, 0);
    atomic_store(&T.single_seq, 0);
    atomic_fetch_add(&T.job, 1);
    pthread_cond_broadcast(&T.go);
    pthread_mutex_unlock(&T.mu);
    my_id = 0;
    in_region = 1;
    my_disp = my_single = 0;
    kmp_int32 g = 0, b = 0;
    call(fn, &g, &b, T.args, T.argc);
    barrier();
    in_region = 0;
    while (atomic_load(&T.done) < nth - 1) relax();   // (the main thread may not block in a browser)
}

void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro fn, ...){
    (void)loc;
    void *args[MAX_ARGS];
    va_list ap;
    va_start(ap, fn);
    for (int i = 0; i < argc && i < MAX_ARGS; i++) args[i] = va_arg(ap, void *);
    va_end(ap);
    fork_region(fn, argc, args);
}
void __kmpc_fork_call_if(ident_t *loc, kmp_int32 argc, kmpc_micro fn, kmp_int32 cond, void *args){
    (void)loc;
    if (cond){ fork_region(fn, argc, (void **)args); return; }
    kmp_int32 g = my_id, b = 0;
    serial_depth++;
    call(fn, &g, &b, (void **)args, argc);
    serial_depth--;
}
void __kmpc_serialized_parallel(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; serial_depth++; }
void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; serial_depth--; }
void __kmpc_push_num_threads(ident_t *loc, kmp_int32 gtid, kmp_int32 n){ (void)loc; (void)gtid; (void)n; }
kmp_int32 __kmpc_global_thread_num(ident_t *loc){ (void)loc; return my_id; }
kmp_int32 __kmpc_ok_to_fork(ident_t *loc){ (void)loc; return 1; }
void __kmpc_barrier(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; barrier(); }
void __kmpc_flush(ident_t *loc){ (void)loc; atomic_thread_fence(memory_order_seq_cst); }

// ---- loops: schedule(static) ----
// iterations of [*lb, *ub] (inclusive) step incr: this thread's share in *lb, *ub (and with a
// chunk, *stride to the next of its chunks)
#define STATIC_INIT(NAME, TYPE, UTYPE) \
void NAME(ident_t *loc, kmp_int32 gtid, kmp_int32 sched, kmp_int32 *plast, TYPE *plb, TYPE *pub, TYPE *pst, TYPE incr, TYPE chunk){ \
    (void)loc; (void)gtid; \
    int nth = team_size(), id = alone() ? 0 : my_id; \
    TYPE lb = *plb, ub = *pub; \
    UTYPE trip = incr > 0 ? (ub < lb ? 0 : (UTYPE)(ub - lb) / (UTYPE)incr + 1) : (lb < ub ? 0 : (UTYPE)(lb - ub) / (UTYPE)(-incr) + 1); \
    if (plast) *plast = 0; \
    if (trip == 0){ *pst = incr; return; } \
    if (nth == 1){ if (plast) *plast = 1; *pst = (TYPE)(trip * (UTYPE)incr); return; } \
    if ((sched & 0xff) == 33 && chunk > 0){            /* static, chunked */ \
        *plb = lb + (TYPE)((UTYPE)id * (UTYPE)chunk * (UTYPE)incr); \
        *pub = *plb + (TYPE)(((UTYPE)chunk - 1) * (UTYPE)incr); \
        *pst = (TYPE)((UTYPE)nth * (UTYPE)chunk * (UTYPE)incr); \
        if (plast) *plast = (UTYPE)id == ((trip - 1) / (UTYPE)chunk) % (UTYPE)nth; \
        return; \
    } \
    UTYPE per = trip / (UTYPE)nth, extra = trip % (UTYPE)nth; \
    UTYPE first = (UTYPE)id * per + ((UTYPE)id < extra ? (UTYPE)id : extra); \
    UTYPE count = per + ((UTYPE)id < extra ? 1 : 0); \
    if (count == 0){ *plb = ub + incr; *pub = ub; } \
    else { *plb = lb + (TYPE)(first * (UTYPE)incr); *pub = *plb + (TYPE)((count - 1) * (UTYPE)incr); } \
    if (plast) *plast = count > 0 && first + count == trip; \
    *pst = (TYPE)(trip * (UTYPE)incr); \
}
STATIC_INIT(__kmpc_for_static_init_4, kmp_int32, kmp_uint32)
STATIC_INIT(__kmpc_for_static_init_4u, kmp_uint32, kmp_uint32)
STATIC_INIT(__kmpc_for_static_init_8, kmp_int64, kmp_uint64)
STATIC_INIT(__kmpc_for_static_init_8u, kmp_uint64, kmp_uint64)
void __kmpc_for_static_fini(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; }

// ---- loops: schedule(dynamic, chunk) (and guided, runtime: as dynamic) ----
// The first thread to reach a loop sets it up, the others wait for that; then each takes the
// next chunk of iterations until there are none (every thread meets the same loops in order)
static _Thread_local struct { int64_t lb, trip, inc, chunk, next; } my_loop;   // (a loop run alone)
#define DISPATCH(INIT, NEXT, TYPE) \
void INIT(ident_t *loc, kmp_int32 gtid, kmp_int32 sched, TYPE lb, TYPE ub, TYPE st, TYPE chunk){ \
    (void)loc; (void)gtid; (void)sched; \
    int64_t inc = (int64_t)st, trip = inc > 0 ? ((int64_t)ub < (int64_t)lb ? 0 : ((int64_t)ub - (int64_t)lb) / inc + 1) \
                                              : ((int64_t)lb < (int64_t)ub ? 0 : ((int64_t)lb - (int64_t)ub) / -inc + 1); \
    if (alone()){ my_loop.lb = (int64_t)lb; my_loop.trip = trip; my_loop.inc = inc; my_loop.chunk = chunk > 0 ? chunk : 1; \
                  my_loop.next = 0; return; } \
    unsigned seq = ++my_disp, prev = seq - 1; \
    if (atomic_compare_exchange_strong(&T.disp_seq, &prev, seq)){ \
        T.disp_lb = (int64_t)lb; T.disp_trip = trip; T.disp_inc = inc; T.disp_chunk = chunk > 0 ? chunk : 1; \
        atomic_store(&T.disp_next, 0); \
        atomic_store(&T.disp_ready, seq); \
    } else while (atomic_load(&T.disp_ready) != seq) relax(); \
} \
int NEXT(ident_t *loc, kmp_int32 gtid, kmp_int32 *plast, TYPE *plb, TYPE *pub, TYPE *pst){ \
    (void)loc; (void)gtid; \
    int64_t c, i, trip, lb0, inc; \
    if (alone()){ c = my_loop.chunk; i = my_loop.next; my_loop.next += c; trip = my_loop.trip; lb0 = my_loop.lb; inc = my_loop.inc; } \
    else { c = T.disp_chunk; i = atomic_fetch_add(&T.disp_next, c); trip = T.disp_trip; lb0 = T.disp_lb; inc = T.disp_inc; } \
    if (i >= trip) return 0; \
    int64_t e = i + c - 1 < trip - 1 ? i + c - 1 : trip - 1; \
    *plb = (TYPE)(lb0 + i * inc); \
    *pub = (TYPE)(lb0 + e * inc); \
    *pst = (TYPE)inc; \
    if (plast) *plast = e == trip - 1; \
    return 1; \
}
DISPATCH(__kmpc_dispatch_init_4, __kmpc_dispatch_next_4, kmp_int32)
DISPATCH(__kmpc_dispatch_init_4u, __kmpc_dispatch_next_4u, kmp_uint32)
DISPATCH(__kmpc_dispatch_init_8, __kmpc_dispatch_next_8, kmp_int64)
DISPATCH(__kmpc_dispatch_init_8u, __kmpc_dispatch_next_8u, kmp_uint64)
void __kmpc_dispatch_fini_4(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; }
void __kmpc_dispatch_fini_8(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; }
void __kmpc_dispatch_deinit(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; }

// ---- reductions (each thread's part added under a lock), critical, single, master ----
kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32 gtid, kmp_int32 n, size_t size, void *data,
                               void (*fn)(void *, void *), kmp_critical_name *lck){
    (void)loc; (void)gtid; (void)n; (void)size; (void)data; (void)fn; (void)lck;
    pthread_mutex_lock(&T.red_mu);
    return 1;
}
void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 gtid, kmp_critical_name *lck){ (void)loc; (void)gtid; (void)lck; pthread_mutex_unlock(&T.red_mu); }
kmp_int32 __kmpc_reduce(ident_t *loc, kmp_int32 gtid, kmp_int32 n, size_t size, void *data,
                        void (*fn)(void *, void *), kmp_critical_name *lck){
    return __kmpc_reduce_nowait(loc, gtid, n, size, data, fn, lck);
}
void __kmpc_end_reduce(ident_t *loc, kmp_int32 gtid, kmp_critical_name *lck){ __kmpc_end_reduce_nowait(loc, gtid, lck); barrier(); }
void __kmpc_critical(ident_t *loc, kmp_int32 gtid, kmp_critical_name *c){ (void)loc; (void)gtid; (void)c; pthread_mutex_lock(&crit_mu); }
void __kmpc_critical_with_hint(ident_t *loc, kmp_int32 gtid, kmp_critical_name *c, uint32_t hint){ (void)hint; __kmpc_critical(loc, gtid, c); }
void __kmpc_end_critical(ident_t *loc, kmp_int32 gtid, kmp_critical_name *c){ (void)loc; (void)gtid; (void)c; pthread_mutex_unlock(&crit_mu); }
kmp_int32 __kmpc_single(ident_t *loc, kmp_int32 gtid){
    (void)loc; (void)gtid;
    if (alone()) return 1;
    unsigned seq = ++my_single, prev = seq - 1;
    return atomic_compare_exchange_strong(&T.single_seq, &prev, seq);
}
void __kmpc_end_single(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; }
kmp_int32 __kmpc_master(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; return alone() || my_id == 0; }
void __kmpc_end_master(ident_t *loc, kmp_int32 gtid){ (void)loc; (void)gtid; }

// ---- tasks: queued for the team (run by whoever waits: taskwait, barriers), or at once ----
void *__kmpc_omp_task_alloc(ident_t *loc, kmp_int32 gtid, kmp_int32 flags, size_t size_task, size_t size_shareds, kmp_routine fn){
    (void)loc; (void)gtid; (void)flags;
    size_t at = (sizeof(struct Task) + size_task + 15) & ~(size_t)15;
    struct Task *t = calloc(1, at + size_shareds + 16);
    if (!t) abort();
    kmp_task_t *k = TASK_OF(t);
    k->shareds = size_shareds ? (char *)t + at : NULL;
    k->routine = fn;
    return k;
}
kmp_int32 __kmpc_omp_task(ident_t *loc, kmp_int32 gtid, void *task){
    (void)loc; (void)gtid;
    struct Task *t = HEAD_OF(task);
    t->parent = my_task;
    atomic_fetch_add(&T.outstanding, 1);
    if (t->parent) atomic_fetch_add(&t->parent->children, 1);
    if (alone()){ run_task(t); return 0; }
    pthread_mutex_lock(&T.task_mu);
    t->next = T.tasks;
    T.tasks = t;
    pthread_mutex_unlock(&T.task_mu);
    return 0;
}
// (if(0) tasks: run at once by the compiler, between these two)
static _Thread_local struct Task *if0_was;
void __kmpc_omp_task_begin_if0(ident_t *loc, kmp_int32 gtid, void *task){ (void)loc; (void)gtid; (void)task; }
void __kmpc_omp_task_complete_if0(ident_t *loc, kmp_int32 gtid, void *task){ (void)loc; (void)gtid; free(HEAD_OF(task)); (void)if0_was; }
kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32 gtid){
    (void)loc; (void)gtid;
    if (!my_task){                                   // (outside any task: all of them)
        while (atomic_load(&T.outstanding) > 0) if (!run_one()) relax();
        return 0;
    }
    while (atomic_load(&my_task->children) > 0) if (!run_one()) relax();
    return 0;
}

// ---- the omp_* functions the engine calls ----
int omp_get_num_procs(void){ return procs(); }
int omp_get_max_threads(void){ int n = procs(); return T.want > 0 && T.want < n ? T.want : n; }
void omp_set_num_threads(int n){ T.want = n > 0 ? n : 0; }
int omp_get_thread_num(void){ return alone() ? 0 : my_id; }
int omp_get_num_threads(void){ return team_size(); }
int omp_in_parallel(void){ return !alone(); }
double omp_get_wtime(void){ return emscripten_get_now() / 1000.0; }

#else
typedef int lr_omp_web_unused;   // (natively, and in the single-threaded web build: nothing)
#endif
