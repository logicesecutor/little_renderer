// poisson.h - physics: the pressure equation of the fluid, solved with multigrid.
//
// On a grid of cubic cells of size h, find p with
//     (sum of the neighbors' p  -  k * p) / h^2 = rhs        in every fluid cell
// where the neighbors are the fluid cells around it, and k counts them plus the open faces
// of the box next to it (beyond an open face p = 0: "Dirichlet"). Solid cells and closed
// walls are left out of the sum (no pressure acts across them: "Neumann").
//
// Gauss-Seidel sweeps remove the error quickly at the scale of a cell but very slowly at
// the scale of the grid (a 48^3 grid needs thousands of sweeps). Multigrid fixes that: the
// residual left after a few sweeps is smooth, so it is solved on a grid half as fine (where
// it is rough again, and sweeps work again), recursively, and the correction is brought
// back. One "V-cycle" cuts the error by a fixed factor whatever the grid size.
//
// Every sweep is red-black (a 3D checkerboard: one color is updated from the other), so it
// runs in parallel and gives the same result for any number of threads.
#ifndef LR_POISSON_H
#define LR_POISSON_H

#include <stdint.h>
#include "core/lrmath.h"

#define POISSON_MAX_LEVELS 8

typedef struct {
    int nx, ny, nz;
    real h;
    float *p, *rhs, *res;    // level 0: p and rhs are the caller's arrays
    uint16_t *links;         // bits 0..5: fluid neighbors -x +x -y +y -z +z; bits 6..8: open faces
    uint8_t *solid;          // level 0: the caller's mask
} PoissonLevel;

typedef struct {
    PoissonLevel lv[POISSON_MAX_LEVELS];
    int nlevels;
    int pre, post;           // smoothing sweeps before / after the coarse correction (3, 3)
} Poisson;

// allocates the hierarchy for an nx * ny * nz grid of cell size h; 0 on success
int poisson_init(Poisson *ps, int nx, int ny, int nz, real h);
void poisson_free(Poisson *ps);
// Prepares the links of all levels from the solid mask (1 = solid, nx*ny*nz, x fastest) and
// the open faces (bits: -x +x -y +y -z +z, as FLUID_OPEN_*). Call when either changed.
void poisson_setup(Poisson *ps, uint8_t *solid, unsigned open);
// `cycles` V-cycles on L p = rhs, starting from the p given (warm start)
void poisson_solve(Poisson *ps, float *p, float *rhs, int cycles);
// the largest |rhs - L p| over the fluid cells
real poisson_residual(Poisson *ps, float *p, float *rhs);

#endif
