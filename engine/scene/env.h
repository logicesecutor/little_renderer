// env.h - the environment: light arriving from infinitely far away, as an equirectangular
// map (longitude across, latitude down), for the sky of an open scene (a Radiance .hdr file
// or a procedural sky) and for the rasterizers' light probe of a room (rt/probe.h).
//
// What each pipeline takes from it (docs/PBR.md, PBR4):
//   - the path tracers: the radiance along a ray that leaves the scene (env_eval), and
//     directions drawn by the map's luminance (env_sample, env_pdf: next event estimation
//     and MIS, as for the lamp);
//   - the rasterizers, split-sum (Karis 2013): the map prefiltered with GGX for 6
//     roughnesses (env_level; each level half the previous one's size down to
//     ENV_LEVEL_MIN wide: smaller ones interpolate the wide lobes visibly wrong near the
//     poles), and the irradiance as 9 spherical harmonics
//     (env_irradiance). A probe of a room also knows the room's box: a lookup from a point
//     follows the direction to the box's wall and looks at that wall point from the probe's
//     centre (env_box_dir: "parallax-corrected" reflections).
//
// Directions: y up. u = 0.5 + atan2(x, z) / 2 pi (the image's middle looks along +z, where
// the default cameras look), v = acos(y) / pi (the top row is the zenith); `rot` turns the
// map about y.
#ifndef LR_ENV_H
#define LR_ENV_H

#include "base/lrmath.h"

#define ENV_LEVELS 6            // prefiltered levels: roughness k / (ENV_LEVELS - 1)
#define ENV_LEVEL0_MAX 512      // level 0's width at most (narrower maps keep their size)
#define ENV_LEVEL_MIN 64        // no level narrower than this

typedef struct Env {
    int w, h;                   // the map (w = 2 h)
    float *rgb;                 // w * h * 3 linear radiance, rows top to bottom
    real scale;                 // every radiance times this (the sky's strength)
    real rot;                   // turned about y by this many radians (env_set_rot)
    real rot_c, rot_s;          // its cosine and sine
    // importance sampling: pixel (x, y) is drawn with probability lum * sin(theta) / sum
    float *marg;                // h + 1: the rows' cumulative distribution (0 ... 1)
    float *cond;                // h * (w + 1): each row's own, over its pixels
    double sum;                 // the sum of lum * sin(theta) over the pixels
    float *wmax;                // w * h: the largest luminance of each pixel's 3 x 3 (its weight is that times
    real *row_sin;              // h: its row's sin(theta)); made once, read by every sample and density
    // the rasterizers
    int lw[ENV_LEVELS], lh[ENV_LEVELS];   // each level's size (lw[k] = 2 lh[k])
    float *level[ENV_LEVELS];   // level k: lw[k] x lh[k] rgb
    float sh[9][3];             // the irradiance's spherical harmonics, as E / pi (radiance units)
    Vec3 mean;                  // the average radiance (over the sphere)
    // a probe: the room's box and the probe's place in it (boxed = 0: a sky, at infinity)
    int boxed;
    Vec3 center, bmin, bmax;
    unsigned id;                // a new number every env_prepare (the GPU uploads it again)
} Env;

// An environment from a map (it takes rgb, which must come from malloc), prepared; NULL if
// out of memory (rgb is then freed)
Env *env_from_rgb(float *rgb, int w, int h);
// a Radiance .hdr file (formats/hdr.h), resampled to a 2:1 map if it is not one
Env *env_load_hdr(const char *path);
// A clear sky made here: blue above, haze at the horizon, a sun (width w, w / 2 high);
// sun_elev / sun_azim in degrees
Env *env_procedural_sky(int w, real sun_elev, real sun_azim);
// the same radiance from everywhere (tests: the white furnace)
Env *env_constant(Vec3 c);
// turns the map about y (radians)
void env_set_rot(Env *e, real rot);
// (Re)makes the sampling tables, the levels and the harmonics from rgb
int env_prepare(Env *e);
void env_free(Env *e);

// map coordinates (0..1) of a direction, and back
void env_uv(const Env *e, Vec3 d, real *u, real *v);
Vec3 env_dir(const Env *e, real u, real v);

// the radiance arriving along -d (coming from direction d), bilinear, times scale
Vec3 env_eval(const Env *e, Vec3 d);
// the same, the direction's map coordinates by fast approximations (the rasterizers)
Vec3 env_eval_fast(const Env *e, Vec3 d);
// a direction drawn by the map's luminance: returns its radiance, *d and *pdf (solid angle)
Vec3 env_sample(const Env *e, real u1, real u2, Vec3 *d, real *pdf);
// the density env_sample draws d with
real env_pdf(const Env *e, Vec3 d);
// the probability env_sample draws pixel (x, y) with (the GPU's copy of the tables)
real env_pixel_prob(const Env *e, int x, int y);

// the map prefiltered with GGX of this roughness (trilinear between the levels), times scale
Vec3 env_level(const Env *e, Vec3 d, real rough);
// the irradiance from the hemisphere around n, divided by pi, times scale
Vec3 env_irradiance(const Env *e, Vec3 n);
// a probe: the direction from its centre to where the ray from p along d leaves its box
// (a sky: d itself)
Vec3 env_box_dir(const Env *e, Vec3 p, Vec3 d);

// the 9 spherical harmonics basis functions at unit direction d (the order env_irradiance's
// coefficients use)
void env_sh_basis(Vec3 d, real *y);

// An irradiance grid: n[0] x n[1] x n[2] small probes at the centres of a box's cells, each
// only the 9 harmonics of the irradiance there (as Env.sh: E / pi; rt/probe.c bakes them).
// The diffuse light of a point is blended from the 8 probes around it - those that ended up
// inside something (valid = 0) left out - so a wall near a bright object gets its light and
// one far from it does not, where one probe would give every point the same.
typedef struct IrrGrid {
    int n[3];
    Vec3 lo, hi;                // the box
    float *sh;                  // per probe (x fastest, then y, then z): 27 harmonics (9 rgb), valid
    unsigned id;                // a new number when baked (the GPU uploads it again)
} IrrGrid;
#define IRR_FLOATS 28
// the blended irradiance (E / pi) at p for normal n, times scale; 0 if no valid probe is near
// (then *ok = 0: use another source)
Vec3 irr_grid_irradiance(const IrrGrid *g, Vec3 p, Vec3 n, int *ok);
// the same in two steps: the probes around p blended (27 numbers: 9 harmonics, rgb; returns 0
// if no valid probe is near), then evaluated for a normal (as often as wanted)
int irr_grid_blend(const IrrGrid *g, Vec3 p, float c[27]);
Vec3 irr_sh_eval(const float c[27], Vec3 n);
void irr_grid_free(IrrGrid *g);

#endif
