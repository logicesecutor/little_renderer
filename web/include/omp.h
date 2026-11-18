// omp.h for the threaded web build (make web THREADS=1): Emscripten has no OpenMP runtime;
// engine/base/omp_web.c is the part of one the engine uses (on pthreads). Only the functions
// the engine calls; natively the compiler's own omp.h is used.
#ifndef LR_WEB_OMP_H
#define LR_WEB_OMP_H
#ifdef __cplusplus
extern "C" {
#endif
int omp_get_num_procs(void);
int omp_get_max_threads(void);
void omp_set_num_threads(int n);
int omp_get_thread_num(void);
int omp_get_num_threads(void);
int omp_in_parallel(void);
double omp_get_wtime(void);
#ifdef __cplusplus
}
#endif
#endif
