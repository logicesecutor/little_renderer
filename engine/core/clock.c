#define _POSIX_C_SOURCE 200809L   // clock_gettime
#include "core/clock.h"

#include <time.h>

double clock_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double clock_ms(void){ return 1000.0 * clock_s(); }
