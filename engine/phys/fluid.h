// fluid.h - physics: the air in the room, and the cloud it carries ("Stable Fluids").
//
// The air is a velocity field on a "MAC" grid of cubic cells (Harlow & Welch 1965): the
// pressure lives at the cell centers, and each velocity component on the faces it crosses
// (u on the x faces, v on the y faces, w on the z faces). This staggering is what makes the
// projection exact: the divergence of a cell is the flow through its 6 faces, and removing
// the pressure gradient leaves none (a grid with everything at the centers cannot see
// checkerboard patterns, and keeps some divergence). Every step (Stam, "Stable Fluids", SIGGRAPH 1999; "Real-Time Fluid Dynamics for Games", 2003):
//   1. forces       wind, buoyancy of the cloud, vorticity confinement (keeps the swirls)
//   2. advection    the velocity is carried by itself: semi-Lagrangian, u(x) <- u(x - dt u)
//   3. projection   air is (nearly) incompressible: solve a Poisson equation for the pressure
//                   and subtract its gradient, so that the flow has no divergence. This is
//                   what makes air flow AROUND a moving ball instead of through it.
//   4. the cloud's density (a Volume, at its own, finer resolution) is carried by the flow
//      (MacCormack advection: sharper than plain semi-Lagrangian)
// Moving rigid bodies are obstacles: cells inside a body take the body's velocity (set by
// phys/world.c each step), so the projection pushes the air, and the cloud, out of the way.
#ifndef LR_FLUID_H
#define LR_FLUID_H

#include <stdint.h>
#include "core/lrmath.h"
#include "core/volume.h"
#include "phys/poisson.h"

// a region where the air expands (an explosion, a burst of hot gas): see fluid_add_expansion
#define FLUID_MAX_SOURCES 8
typedef struct { Vec3 center; real radius, rate, time_left; } FluidSource;

typedef struct {
    int nx, ny, nz;            // cells per axis
    Aabb bounds;               // the box the grid fills
    real h, inv_h;             // cell size (cubic cells), and 1 / h
    float *u, *v, *w;          // velocity (units / s) on the faces: u has (nx+1) ny nz values,
                               // v nx (ny+1) nz, w nx ny (nz+1); see fluid_iu/iv/iw
    float *tu, *tv, *tw;       // scratch, same sizes
    float *p, *div;            // pressure and divergence, one per cell
    float *scratch;            // 4 values per cell (vorticity)
    uint8_t *solid;            // per cell: 1 = inside a rigid body (set every step by the coupling)
    float *su, *sv, *sw;       // per cell: the body's velocity in those cells
    Volume *density;           // the cloud (not owned: the scene owns it), or NULL
    float *d0, *d1;            // scratch for the density advection

    // parameters (fluid_create sets the defaults)
    Vec3 wind;                 // the air drifts towards this velocity...
    real wind_rate;            // ...at this rate, per second
    real buoyancy;             // upward acceleration per unit of cloud density (units / s^2)
    real vorticity;            // vorticity confinement strength (0 = off)
    real dissipation;          // fraction of the cloud density lost per second
    int pressure_cycles;       // multigrid V-cycles for the pressure (phys/poisson.h)
    Poisson poisson;
    unsigned open;             // FLUID_OPEN_* bits: faces the air can flow through
    real max_divergence;       // diagnostics: largest |div u| after the last projection
                               // (not counting the expansion sources)
    FluidSource sources[FLUID_MAX_SOURCES];
    int nsources;
} Fluid;

// Faces of the grid's box. A closed face is a wall (no flow across it); an open face lets
// air in and out (outside, the pressure is 0). The Cornell box is open at the front (z = lo,
// the camera side); an outdoor sky is open everywhere except the ground.
enum {
    FLUID_OPEN_XLO = 1, FLUID_OPEN_XHI = 2, FLUID_OPEN_YLO = 4, FLUID_OPEN_YHI = 8,
    FLUID_OPEN_ZLO = 16, FLUID_OPEN_ZHI = 32,
    FLUID_OPEN_SKY = 1 | 2 | 8 | 16 | 32
};

// A grid filling `bounds` with cells of about `cell` units (rounded so they are cubic).
// density may be NULL (air only). NULL if out of memory.
Fluid *fluid_create(Aabb bounds, real cell, Volume *density);
void fluid_free(Fluid *f);

// index of a cell, and of the u / v / w faces: fluid_iu(f, x, y, z) is the x face between
// cells x-1 and x (x = 0 .. nx)
static inline size_t fluid_idx(const Fluid *f, int x, int y, int z){ return ((size_t)z * f->ny + y) * f->nx + x; }
static inline size_t fluid_iu(const Fluid *f, int x, int y, int z){ return ((size_t)z * f->ny + y) * (f->nx + 1) + x; }
static inline size_t fluid_iv(const Fluid *f, int x, int y, int z){ return ((size_t)z * (f->ny + 1) + y) * f->nx + x; }
static inline size_t fluid_iw(const Fluid *f, int x, int y, int z){ return ((size_t)z * f->ny + y) * f->nx + x; }
Vec3 fluid_cell_center(const Fluid *f, int x, int y, int z);
// air velocity at a world point (trilinear, clamped to the grid)
Vec3 fluid_velocity(const Fluid *f, Vec3 p);

// clear the obstacle mask (before the bodies are marked again)
void fluid_clear_solids(Fluid *f);
// add a velocity change to the air inside a sphere, fading to 0 at its edge. With radial = 1
// `dv.x` is the outward speed (a blast); otherwise dv is a direction (a gust).
void fluid_add_impulse(Fluid *f, Vec3 center, real radius, Vec3 dv, int radial);

// Air appearing inside a sphere: for `duration` seconds the projection makes the flow's
// divergence there `rate` (per second, fading to 0 at the edge) instead of 0, so the air
// rushes outwards at about rate * radius / 3 at the edge. A plain outward kick would not do:
// the projection removes any flow that compresses or expands the (incompressible) air.
void fluid_add_expansion(Fluid *f, Vec3 center, real radius, real rate, real duration);

void fluid_step(Fluid *f, real dt);

// diagnostics
real fluid_kinetic_energy(const Fluid *f);   // sum of |u|^2 / 2 per cell volume (arbitrary units)
real fluid_total_density(const Fluid *f);     // sum of the cloud's voxels

#endif
