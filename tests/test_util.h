// test_util.h - what the unit tests share: a check that prints its line and counts failures.
// Each test includes it once; main() ends with "if (failures) ...".
#ifndef LR_TEST_UTIL_H
#define LR_TEST_UTIL_H

#include <math.h>
#include <stdio.h>

static int failures;

// a condition: "ok" or "FAIL"
static inline void check(const char *what, int ok){
    printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

// a value within tol of what it should be
static inline void check_near(const char *what, double got, double want, double tol){
    int ok = fabs(got - want) <= tol;
    printf("  %-58s %12.5f  (want %.5f +- %.5f)  %s\n", what, got, want, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

#endif
