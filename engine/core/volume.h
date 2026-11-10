// volume.h - a participating medium (cloud, smoke, fog) stored as a voxel grid of density.
//
// The grid is the common representation for every backend and every way of making a cloud:
//   - the path tracer samples it (rt/medium.c),
//   - a real-time raymarcher can sample the same grid,
//   - volume_make_cloud fills it with procedural noise (animated by a time parameter),
//   - a fluid simulation (forces, wind, buoyancy) would move density through the same grid.
//
// Density is unitless, 0 = empty and about 1 = dense. The optical properties turn it into
// physics: a ray travelling a short distance ds through density d is absorbed or scattered
// with probability sigma_t * d * ds (sigma_t = extinction coefficient, per scene unit).
#ifndef LR_VOLUME_H
#define LR_VOLUME_H

#include <stddef.h>
#include <stdint.h>
#include "core/lrmath.h"
#include "core/noise.h"
#include "core/scene.h"   // Aabb

typedef struct Volume {
    int nx, ny, nz;          // voxels per axis
    float *density;          // nx * ny * nz values, x fastest: index (z * ny + y) * nx + x
    Aabb bounds;             // the world-space box the grid fills (voxel centers are inside)

    // optical properties (the same everywhere; only the density varies)
    real sigma_t;            // extinction at density 1, per scene unit
    Vec3 albedo;             // single-scattering albedo sigma_s / sigma_t per channel (clouds ~0.99)
    real g;                  // Henyey-Greenstein anisotropy in (-1, 1): 0 = uniform, > 0 = forward

    // derived, kept up to date by volume_update
    real max_density;        // largest density in the grid (the "majorant" for delta tracking)
    Aabb active;             // smallest box around all voxels with density > 0
    // Majorant grid: the grid cut into bricks of VOLUME_BRICK^3 voxels; brick_max holds the
    // largest density that volume_density can return anywhere inside each brick (so it also
    // covers the voxels around the brick that the interpolation reaches into). Renderers use
    // it to take long steps through thin regions and to skip empty ones.
    int bnx, bny, bnz;       // bricks per axis
    float *brick_max;        // bnx * bny * bnz values, x fastest
    // incremented by every volume_update: caches built from the density (the rasterizer's
    // light grid) compare it to know when to rebuild
    uint32_t version;
} Volume;

#ifndef VOLUME_BRICK
#define VOLUME_BRICK 8   // voxels per brick side (try -DVOLUME_BRICK=16)
#endif

// a grid of zeros; NULL if out of memory. Optical defaults: a white cloud (see volume.c).
Volume *volume_create(int nx, int ny, int nz, Aabb bounds);
void volume_free(Volume *v);
// Recompute max_density, active and the majorant grid after the density values changed.
// Call it after every edit (baking, simulation step); renderers rely on all three.
void volume_update(Volume *v);

// world position of the center of voxel (x, y, z)
Vec3 volume_voxel_center(const Volume *v, int x, int y, int z);

// Density at a world position: trilinear interpolation between the 8 nearest voxel centers,
// 0 outside the grid.
static inline real volume_density(const Volume *v, Vec3 p){
    // position in voxel units, voxel centers at integer + 0.5
    real gx = (p.x - v->bounds.lo.x) / (v->bounds.hi.x - v->bounds.lo.x) * v->nx - 0.5f;
    real gy = (p.y - v->bounds.lo.y) / (v->bounds.hi.y - v->bounds.lo.y) * v->ny - 0.5f;
    real gz = (p.z - v->bounds.lo.z) / (v->bounds.hi.z - v->bounds.lo.z) * v->nz - 0.5f;
    if (!(gx > -1 && gy > -1 && gz > -1 && gx < v->nx && gy < v->ny && gz < v->nz)) return 0;
    int x0 = gx < 0 ? -1 : (int)gx, y0 = gy < 0 ? -1 : (int)gy, z0 = gz < 0 ? -1 : (int)gz;
    real fx = gx - x0, fy = gy - y0, fz = gz - z0;
    real c[8];
    for (int k = 0; k < 8; k++){
        int x = x0 + (k & 1), y = y0 + ((k >> 1) & 1), z = z0 + (k >> 2);
        int inside = x >= 0 && y >= 0 && z >= 0 && x < v->nx && y < v->ny && z < v->nz;
        c[k] = inside ? v->density[((size_t)z * v->ny + y) * v->nx + x] : 0;   // outside the grid = empty
    }
    real c00 = c[0] + (c[1] - c[0]) * fx, c10 = c[2] + (c[3] - c[2]) * fx;
    real c01 = c[4] + (c[5] - c[4]) * fx, c11 = c[6] + (c[7] - c[6]) * fx;
    real c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
    return c0 + (c1 - c0) * fz;
}

// Henyey-Greenstein phase function: the probability density (per steradian) that light
// travelling in direction a continues in direction b after scattering, with cos_theta =
// dot(a, b). g > 0 favours small turns (forward scattering, like the water droplets of a
// cloud), g < 0 backward, g = 0 every direction alike (1 / 4pi).
static inline real phase_hg(real cos_theta, real g){
    real d = 1.0f + g * g - 2.0f * g * cos_theta;
    return (1.0f - g * g) / (4.0f * (real)PI * d * r_sqrt(d));
}

// ---- procedural clouds ---------------------------------------------------------------

#define CLOUD_MAX_PUFFS 16

typedef struct CloudParams {
    // shape
    uint32_t seed;     // which cloud: placement of the puffs and the noise
    int puffs;         // number of round "cauliflower" lobes (<= 16)
    real detail;       // 0 = smooth blobs .. 1 = strongly eroded, wispy billows
    real time;         // seconds, for animation: the detail noise drifts with `wind`
    Vec3 wind;         // drift of the detail, in cloud sizes per second
    // grid
    int resolution;    // voxels along the longest side of the box (the others in proportion)
    // material (copied into the Volume)
    real sigma_t;      // extinction at density 1, per scene unit
    Vec3 albedo;       // single-scattering albedo
    real g;            // Henyey-Greenstein anisotropy
} CloudParams;

CloudParams cloud_default_params(void);

// A cloud as a function: everything cloud_density needs, prepared once from the params.
// It lives in the cloud's own coordinates: the center of its box is 0 and the box spans
// [-aspect, aspect] on each axis, where aspect = box size / largest side (so the longest
// axis spans [-1, 1] and the puffs stay round whatever the box's proportions).
typedef struct {
    CloudParams p;
    Noise noise;
    Vec3 aspect;
    int npuffs;
    Vec3 puff_c[CLOUD_MAX_PUFFS];   // puff centers and radii, in cloud coordinates
    real puff_r[CLOUD_MAX_PUFFS];
} Cloud;

void cloud_init(Cloud *c, const CloudParams *cp, Vec3 aspect);
// Density at u (cloud coordinates): 0 outside the shape, up to about 1 in the core.
// Thread-safe; a real-time renderer can call it directly instead of using a grid.
real cloud_density(const Cloud *c, Vec3 u);

// Fills the whole grid with the cloud, fitted to the grid's bounds, copies the material
// from cp, then calls volume_update. Runs in parallel (OpenMP). cp->resolution is ignored.
void volume_make_cloud(Volume *v, const CloudParams *cp);

// Adds a cloud fitted to `box` into an existing grid (which may be much bigger, e.g. the
// whole room, so that a simulation can carry the cloud anywhere): each voxel inside the box
// becomes max(its density, the cloud's). Copies the material from cp, calls volume_update.
void volume_add_cloud(Volume *v, Aabb box, const CloudParams *cp);

// volume_create with cp->resolution voxels along the longest side of bounds, then
// volume_make_cloud. NULL if out of memory.
Volume *volume_new_cloud(Aabb bounds, const CloudParams *cp);

#endif
