#include "phys/fluid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#define PFOR _Pragma("omp parallel for schedule(dynamic, 1)")
#else
#define PFOR
#endif

// ---- creation -----------------------------------------------------------------------------

static float *falloc(size_t n){
    float *p = calloc(n ? n : 1, sizeof(float));
    if (!p){ perror("calloc"); exit(1); }
    return p;
}

static size_t ncells(const Fluid *f){ return (size_t)f->nx * f->ny * f->nz; }
static size_t nu(const Fluid *f){ return (size_t)(f->nx + 1) * f->ny * f->nz; }
static size_t nv(const Fluid *f){ return (size_t)f->nx * (f->ny + 1) * f->nz; }
static size_t nw(const Fluid *f){ return (size_t)f->nx * f->ny * (f->nz + 1); }

Fluid *fluid_create(Aabb bounds, real cell, Volume *density){
    Fluid *f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    Vec3 size = v_sub(bounds.hi, bounds.lo);
    f->nx = (int)(size.x / cell + 0.5f); if (f->nx < 4) f->nx = 4;
    f->ny = (int)(size.y / cell + 0.5f); if (f->ny < 4) f->ny = 4;
    f->nz = (int)(size.z / cell + 0.5f); if (f->nz < 4) f->nz = 4;
    // cubic cells: use the x spacing and stretch the other sides slightly if needed
    f->h = size.x / f->nx;
    f->inv_h = 1.0f / f->h;
    f->bounds = bounds;
    f->bounds.hi = vec3(bounds.lo.x + f->h * f->nx, bounds.lo.y + f->h * f->ny, bounds.lo.z + f->h * f->nz);
    size_t n = ncells(f);
    f->u = falloc(nu(f)); f->v = falloc(nv(f)); f->w = falloc(nw(f));
    f->tu = falloc(nu(f)); f->tv = falloc(nv(f)); f->tw = falloc(nw(f));
    f->p = falloc(n); f->div = falloc(n); f->scratch = falloc(4 * n);
    if (poisson_init(&f->poisson, f->nx, f->ny, f->nz, f->h) != 0){ perror("calloc"); exit(1); }
    f->su = falloc(n); f->sv = falloc(n); f->sw = falloc(n);
    f->solid = calloc(n, 1);
    if (!f->solid){ perror("calloc"); exit(1); }
    f->density = density;
    if (density){
        size_t nd = (size_t)density->nx * density->ny * density->nz;
        f->d0 = falloc(nd);
        f->d1 = falloc(nd);
    }
    f->wind = vec3(0, 0, 0);
    f->wind_rate = 0.3f;
    f->buoyancy = 0;
    f->vorticity = 0.25f;
    f->dissipation = 0.0f;
    f->pressure_cycles = 5;
    f->open = FLUID_OPEN_ZLO;
    return f;
}

void fluid_free(Fluid *f){
    if (!f) return;
    free(f->u); free(f->v); free(f->w); free(f->tu); free(f->tv); free(f->tw);
    free(f->p); free(f->div); free(f->scratch); poisson_free(&f->poisson); free(f->su); free(f->sv); free(f->sw); free(f->solid);
    free(f->d0); free(f->d1);
    free(f);
}

Vec3 fluid_cell_center(const Fluid *f, int x, int y, int z){
    return vec3(f->bounds.lo.x + (x + 0.5f) * f->h, f->bounds.lo.y + (y + 0.5f) * f->h, f->bounds.lo.z + (z + 0.5f) * f->h);
}

void fluid_clear_solids(Fluid *f){
    memset(f->solid, 0, ncells(f));
}

// ---- sampling -------------------------------------------------------------------------------

// Trilinear sample of a grid of sx * sy * sz values (x fastest) at grid coordinates g
// (value k sits at coordinate k), clamped to the grid: values at the border continue
// outwards. Optionally returns the min and max of the 8 values used.
static float sample3(const float *a, int sx, int sy, int sz, real gx, real gy, real gz, float *mn, float *mx){
    gx = gx < 0 ? 0 : (gx > sx - 1 ? sx - 1 : gx);
    gy = gy < 0 ? 0 : (gy > sy - 1 ? sy - 1 : gy);
    gz = gz < 0 ? 0 : (gz > sz - 1 ? sz - 1 : gz);
    int x0 = (int)gx, y0 = (int)gy, z0 = (int)gz;
    if (x0 > sx - 2) x0 = sx > 1 ? sx - 2 : 0;
    if (y0 > sy - 2) y0 = sy > 1 ? sy - 2 : 0;
    if (z0 > sz - 2) z0 = sz > 1 ? sz - 2 : 0;
    int x1 = x0 + 1 < sx ? x0 + 1 : x0, y1 = y0 + 1 < sy ? y0 + 1 : y0, z1 = z0 + 1 < sz ? z0 + 1 : z0;
    real fx = gx - x0, fy = gy - y0, fz = gz - z0;
#define A(x, y, z) a[((size_t)(z) * sy + (y)) * sx + (x)]
    float c000 = A(x0, y0, z0), c100 = A(x1, y0, z0), c010 = A(x0, y1, z0), c110 = A(x1, y1, z0);
    float c001 = A(x0, y0, z1), c101 = A(x1, y0, z1), c011 = A(x0, y1, z1), c111 = A(x1, y1, z1);
#undef A
    if (mn){
        float lo = c000, hi = c000;
        float cs[7] = {c100, c010, c110, c001, c101, c011, c111};
        for (int i = 0; i < 7; i++){ if (cs[i] < lo) lo = cs[i]; if (cs[i] > hi) hi = cs[i]; }
        *mn = lo; *mx = hi;
    }
    real c00 = c000 + (c100 - c000) * fx, c10 = c010 + (c110 - c010) * fx;
    real c01 = c001 + (c101 - c001) * fx, c11 = c011 + (c111 - c011) * fx;
    real c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
    return (float)(c0 + (c1 - c0) * fz);
}

// one velocity component at a world point. u sits at (i, j+1/2, k+1/2) h, and so on.
static float sample_u(const Fluid *f, const float *u, Vec3 p){
    const real s = f->inv_h;
    return sample3(u, f->nx + 1, f->ny, f->nz, (p.x - f->bounds.lo.x) * s, (p.y - f->bounds.lo.y) * s - 0.5f,
                   (p.z - f->bounds.lo.z) * s - 0.5f, NULL, NULL);
}
static float sample_v(const Fluid *f, const float *v, Vec3 p){
    const real s = f->inv_h;
    return sample3(v, f->nx, f->ny + 1, f->nz, (p.x - f->bounds.lo.x) * s - 0.5f, (p.y - f->bounds.lo.y) * s,
                   (p.z - f->bounds.lo.z) * s - 0.5f, NULL, NULL);
}
static float sample_w(const Fluid *f, const float *w, Vec3 p){
    const real s = f->inv_h;
    return sample3(w, f->nx, f->ny, f->nz + 1, (p.x - f->bounds.lo.x) * s - 0.5f, (p.y - f->bounds.lo.y) * s - 0.5f,
                   (p.z - f->bounds.lo.z) * s, NULL, NULL);
}

Vec3 fluid_velocity(const Fluid *f, Vec3 p){
    return vec3(sample_u(f, f->u, p), sample_v(f, f->v, p), sample_w(f, f->w, p));
}

// A cell-centered grid of the Volume's resolution: world <-> voxel coordinates, prepared
// once per step (no divisions in the inner loops).
typedef struct { int nx, ny, nz; Vec3 lo, scale, vox; } VoxGrid;

static VoxGrid voxgrid(const Volume *v){
    VoxGrid g;
    Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
    g.nx = v->nx; g.ny = v->ny; g.nz = v->nz;
    g.lo = v->bounds.lo;
    g.scale = vec3(v->nx / size.x, v->ny / size.y, v->nz / size.z);
    g.vox = vec3(size.x / v->nx, size.y / v->ny, size.z / v->nz);
    return g;
}

static inline Vec3 voxgrid_center(const VoxGrid *g, int x, int y, int z){
    return vec3(g->lo.x + (x + 0.5f) * g->vox.x, g->lo.y + (y + 0.5f) * g->vox.y, g->lo.z + (z + 0.5f) * g->vox.z);
}

static inline float voxgrid_sample(const VoxGrid *g, const float *a, Vec3 p, float *mn, float *mx){
    return sample3(a, g->nx, g->ny, g->nz, (p.x - g->lo.x) * g->scale.x - 0.5f, (p.y - g->lo.y) * g->scale.y - 0.5f,
                   (p.z - g->lo.z) * g->scale.z - 0.5f, mn, mx);
}

// world positions of the faces
static Vec3 pos_u(const Fluid *f, int x, int y, int z){
    return vec3(f->bounds.lo.x + x * f->h, f->bounds.lo.y + (y + 0.5f) * f->h, f->bounds.lo.z + (z + 0.5f) * f->h);
}
static Vec3 pos_v(const Fluid *f, int x, int y, int z){
    return vec3(f->bounds.lo.x + (x + 0.5f) * f->h, f->bounds.lo.y + y * f->h, f->bounds.lo.z + (z + 0.5f) * f->h);
}
static Vec3 pos_w(const Fluid *f, int x, int y, int z){
    return vec3(f->bounds.lo.x + (x + 0.5f) * f->h, f->bounds.lo.y + (y + 0.5f) * f->h, f->bounds.lo.z + z * f->h);
}

// ---- boundaries -------------------------------------------------------------------------------
//
// A face of the box that is closed is a wall: no air crosses it, its velocity is 0. An open
// face keeps whatever the flow does. A face touching a cell inside a body moves with the
// body (the mean of the solid cells' velocities on either side).
static int solid_at(const Fluid *f, int x, int y, int z){
    return x >= 0 && y >= 0 && z >= 0 && x < f->nx && y < f->ny && z < f->nz && f->solid[fluid_idx(f, x, y, z)];
}

// 1 if the face's value is fixed by a wall or a body (and then sets *val)
static int fixed_u(const Fluid *f, int x, int y, int z, float *val){
    if (x == 0 && !(f->open & FLUID_OPEN_XLO)){ *val = 0; return 1; }
    if (x == f->nx && !(f->open & FLUID_OPEN_XHI)){ *val = 0; return 1; }
    int a = solid_at(f, x - 1, y, z), b = solid_at(f, x, y, z);
    if (!a && !b) return 0;
    *val = ((a ? f->su[fluid_idx(f, x - 1, y, z)] : 0) + (b ? f->su[fluid_idx(f, x, y, z)] : 0)) / (a + b);
    return 1;
}
static int fixed_v(const Fluid *f, int x, int y, int z, float *val){
    if (y == 0 && !(f->open & FLUID_OPEN_YLO)){ *val = 0; return 1; }
    if (y == f->ny && !(f->open & FLUID_OPEN_YHI)){ *val = 0; return 1; }
    int a = solid_at(f, x, y - 1, z), b = solid_at(f, x, y, z);
    if (!a && !b) return 0;
    *val = ((a ? f->sv[fluid_idx(f, x, y - 1, z)] : 0) + (b ? f->sv[fluid_idx(f, x, y, z)] : 0)) / (a + b);
    return 1;
}
static int fixed_w(const Fluid *f, int x, int y, int z, float *val){
    if (z == 0 && !(f->open & FLUID_OPEN_ZLO)){ *val = 0; return 1; }
    if (z == f->nz && !(f->open & FLUID_OPEN_ZHI)){ *val = 0; return 1; }
    int a = solid_at(f, x, y, z - 1), b = solid_at(f, x, y, z);
    if (!a && !b) return 0;
    *val = ((a ? f->sw[fluid_idx(f, x, y, z - 1)] : 0) + (b ? f->sw[fluid_idx(f, x, y, z)] : 0)) / (a + b);
    return 1;
}

static void enforce_boundaries(Fluid *f){
    const int nx = f->nx, ny = f->ny, nz = f->nz;
    PFOR
    for (int z = 0; z <= nz; z++)
        for (int y = 0; y <= ny; y++)
            for (int x = 0; x <= nx; x++){
                float val;
                if (y < ny && z < nz && fixed_u(f, x, y, z, &val)) f->u[fluid_iu(f, x, y, z)] = val;
                if (x < nx && z < nz && fixed_v(f, x, y, z, &val)) f->v[fluid_iv(f, x, y, z)] = val;
                if (x < nx && y < ny && fixed_w(f, x, y, z, &val)) f->w[fluid_iw(f, x, y, z)] = val;
            }
}

// ---- 1. forces --------------------------------------------------------------------------------

static void add_forces(Fluid *f, real dt){
    const int nx = f->nx, ny = f->ny, nz = f->nz;
    const real a = f->wind_rate * dt > 1 ? 1 : f->wind_rate * dt;
    const int buoy = f->buoyancy != 0 && f->density;
    PFOR
    for (int z = 0; z <= nz; z++)
        for (int y = 0; y <= ny; y++)
            for (int x = 0; x <= nx; x++){
                // wind: the air relaxes towards the wind velocity
                if (y < ny && z < nz){ float *q = &f->u[fluid_iu(f, x, y, z)]; *q += (float)((f->wind.x - *q) * a); }
                if (x < nx && z < nz){
                    float *q = &f->v[fluid_iv(f, x, y, z)];
                    *q += (float)((f->wind.y - *q) * a);
                    // buoyancy: the cloud's (warm, moist) air rises
                    if (buoy) *q += (float)(f->buoyancy * volume_density(f->density, pos_v(f, x, y, z)) * dt);
                }
                if (x < nx && y < ny){ float *q = &f->w[fluid_iw(f, x, y, z)]; *q += (float)((f->wind.z - *q) * a); }
            }
}

// Vorticity confinement (Fedkiw, Stam & Jensen 2001). The numerical blur of advection
// smooths small swirls away. Find where the flow rotates (vorticity omega = curl u), and
// push the air around those spots, along N x omega, where N points towards stronger
// rotation: the swirls are kept alive at the scale of the grid. Computed at the cell
// centers, then spread to the faces.
static void vorticity_confinement(Fluid *f, real dt){
    if (f->vorticity <= 0) return;
    const int nx = f->nx, ny = f->ny, nz = f->nz;
    const size_t n = ncells(f);
    const real inv2h = 0.5f / f->h;
    float *cu = f->scratch, *cv = f->scratch + n, *cw = f->scratch + 2 * n, *mag = f->scratch + 3 * n;
    // cell-center velocity
    PFOR
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = fluid_idx(f, x, y, z);
                cu[i] = 0.5f * (f->u[fluid_iu(f, x, y, z)] + f->u[fluid_iu(f, x + 1, y, z)]);
                cv[i] = 0.5f * (f->v[fluid_iv(f, x, y, z)] + f->v[fluid_iv(f, x, y + 1, z)]);
                cw[i] = 0.5f * (f->w[fluid_iw(f, x, y, z)] + f->w[fluid_iw(f, x, y, z + 1)]);
            }
#define AT(g, x, y, z) g[fluid_idx(f, (x) < 0 ? 0 : ((x) >= nx ? nx - 1 : (x)), (y) < 0 ? 0 : ((y) >= ny ? ny - 1 : (y)), \
                                     (z) < 0 ? 0 : ((z) >= nz ? nz - 1 : (z)))]
    // vorticity: stored in tu / tv / tw (as cell values, they are big enough)
    float *ox = f->tu, *oy = f->tv, *oz = f->tw;
    PFOR
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = fluid_idx(f, x, y, z);
                real dwdy = (AT(cw, x, y + 1, z) - AT(cw, x, y - 1, z)) * inv2h;
                real dvdz = (AT(cv, x, y, z + 1) - AT(cv, x, y, z - 1)) * inv2h;
                real dudz = (AT(cu, x, y, z + 1) - AT(cu, x, y, z - 1)) * inv2h;
                real dwdx = (AT(cw, x + 1, y, z) - AT(cw, x - 1, y, z)) * inv2h;
                real dvdx = (AT(cv, x + 1, y, z) - AT(cv, x - 1, y, z)) * inv2h;
                real dudy = (AT(cu, x, y + 1, z) - AT(cu, x, y - 1, z)) * inv2h;
                ox[i] = (float)(dwdy - dvdz);
                oy[i] = (float)(dudz - dwdx);
                oz[i] = (float)(dvdx - dudy);
                mag[i] = (float)r_sqrt(ox[i] * ox[i] + oy[i] * oy[i] + oz[i] * oz[i]);
            }
    // the force at the cell centers, into cu / cv / cw
    const real eps = f->vorticity * f->h;
    PFOR
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = fluid_idx(f, x, y, z);
                Vec3 g = vec3((AT(mag, x + 1, y, z) - AT(mag, x - 1, y, z)) * inv2h,
                              (AT(mag, x, y + 1, z) - AT(mag, x, y - 1, z)) * inv2h,
                              (AT(mag, x, y, z + 1) - AT(mag, x, y, z - 1)) * inv2h);
                real gl = v_len(g);
                Vec3 force = vec3(0, 0, 0);
                if (gl > 1e-6f && !f->solid[i]) force = v_scale(v_cross(v_scale(g, 1.0f / gl), vec3(ox[i], oy[i], oz[i])), eps);
                cu[i] = (float)force.x; cv[i] = (float)force.y; cw[i] = (float)force.z;
            }
#undef AT
    // to the inner faces: the mean of the two cells
    PFOR
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = fluid_idx(f, x, y, z);
                if (x > 0) f->u[fluid_iu(f, x, y, z)] += (float)(0.5f * (cu[i] + cu[fluid_idx(f, x - 1, y, z)]) * dt);
                if (y > 0) f->v[fluid_iv(f, x, y, z)] += (float)(0.5f * (cv[i] + cv[fluid_idx(f, x, y - 1, z)]) * dt);
                if (z > 0) f->w[fluid_iw(f, x, y, z)] += (float)(0.5f * (cw[i] + cw[fluid_idx(f, x, y, z - 1)]) * dt);
            }
}

// ---- 2. advection of the velocity (semi-Lagrangian) ---------------------------------------------
//
// The new velocity at x is the old velocity at the point the air came from: x - dt * u(x).
// Unconditionally stable: whatever dt, the result is an interpolation of existing values.
static void advect_velocity(Fluid *f, real dt){
    const int nx = f->nx, ny = f->ny, nz = f->nz;
    PFOR
    for (int z = 0; z <= nz; z++)
        for (int y = 0; y <= ny; y++)
            for (int x = 0; x <= nx; x++){
                if (y < ny && z < nz){
                    Vec3 c = pos_u(f, x, y, z);
                    f->tu[fluid_iu(f, x, y, z)] = sample_u(f, f->u, v_sub(c, v_scale(fluid_velocity(f, c), dt)));
                }
                if (x < nx && z < nz){
                    Vec3 c = pos_v(f, x, y, z);
                    f->tv[fluid_iv(f, x, y, z)] = sample_v(f, f->v, v_sub(c, v_scale(fluid_velocity(f, c), dt)));
                }
                if (x < nx && y < ny){
                    Vec3 c = pos_w(f, x, y, z);
                    f->tw[fluid_iw(f, x, y, z)] = sample_w(f, f->w, v_sub(c, v_scale(fluid_velocity(f, c), dt)));
                }
            }
    float *t;
    t = f->u; f->u = f->tu; f->tu = t;
    t = f->v; f->v = f->tv; f->tv = t;
    t = f->w; f->w = f->tw; f->tw = t;
}

// ---- 3. projection ---------------------------------------------------------------------------------
//
// Helmholtz: any velocity field = a divergence-free part + the gradient of a scalar. Solve
//   laplacian(p) = div(u)
// and subtract grad(p): what is left has no divergence. Walls and bodies are "Neumann"
// boundaries (their faces' velocity is given, so the pressure does not act across them),
// open faces are "Dirichlet" (pressure 0 outside). The Poisson equation is solved by
// multigrid (phys/poisson.c).

static void compute_divergence(Fluid *f){
    const int nx = f->nx, ny = f->ny, nz = f->nz;
    const real inv_h = 1.0f / f->h;
    PFOR
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = fluid_idx(f, x, y, z);
                if (f->solid[i]){ f->div[i] = 0; continue; }
                real d = (f->u[fluid_iu(f, x + 1, y, z)] - f->u[fluid_iu(f, x, y, z)]) +
                         (f->v[fluid_iv(f, x, y + 1, z)] - f->v[fluid_iv(f, x, y, z)]) +
                         (f->w[fluid_iw(f, x, y, z + 1)] - f->w[fluid_iw(f, x, y, z)]);
                f->div[i] = (float)(d * inv_h);
            }
}

// add sign * the expansion sources to the divergence
static void apply_sources(Fluid *f, real sign){
    for (int s = 0; s < f->nsources; s++){
        const FluidSource *src = &f->sources[s];
        PFOR
        for (int z = 0; z < f->nz; z++)
            for (int y = 0; y < f->ny; y++)
                for (int x = 0; x < f->nx; x++){
                    size_t i = fluid_idx(f, x, y, z);
                    if (f->solid[i]) continue;
                    real l = v_len(v_sub(fluid_cell_center(f, x, y, z), src->center)) / src->radius;
                    if (l < 1) f->div[i] += (float)(sign * src->rate * (1 - l * l));
                }
    }
}

// the divergence the sources impose at a point
static real source_divergence(const Fluid *f, Vec3 p){
    real d = 0;
    for (int s = 0; s < f->nsources; s++){
        const FluidSource *src = &f->sources[s];
        real l = v_len(v_sub(p, src->center)) / src->radius;
        if (l < 1) d += src->rate * (1 - l * l);
    }
    return d;
}

void fluid_add_expansion(Fluid *f, Vec3 center, real radius, real rate, real duration){
    if (f->nsources >= FLUID_MAX_SOURCES || radius <= 0 || duration <= 0) return;
    FluidSource *s = &f->sources[f->nsources++];
    s->center = center;
    s->radius = radius;
    s->rate = rate;
    s->time_left = duration;
}

static void project(Fluid *f, real *max_div_out){
    const int nx = f->nx, ny = f->ny, nz = f->nz;
    compute_divergence(f);
    apply_sources(f, -1);   // solve for div u = rate inside the sources, 0 elsewhere
    // p is kept from the last step: a good first guess, the flow changes little per step
    poisson_setup(&f->poisson, f->solid, f->open);
    poisson_solve(&f->poisson, f->p, f->div, f->pressure_cycles);
    // u -= grad p on every face the pressure acts across
    const real inv_h = 1.0f / f->h;
#define PC(x, y, z) (((x) < 0 || (y) < 0 || (z) < 0 || (x) >= nx || (y) >= ny || (z) >= nz) ? 0 : f->p[fluid_idx(f, x, y, z)])
    PFOR
    for (int z = 0; z <= nz; z++)
        for (int y = 0; y <= ny; y++)
            for (int x = 0; x <= nx; x++){
                float val;
                if (y < ny && z < nz && !fixed_u(f, x, y, z, &val))
                    f->u[fluid_iu(f, x, y, z)] -= (float)((PC(x, y, z) - PC(x - 1, y, z)) * inv_h);
                if (x < nx && z < nz && !fixed_v(f, x, y, z, &val))
                    f->v[fluid_iv(f, x, y, z)] -= (float)((PC(x, y, z) - PC(x, y - 1, z)) * inv_h);
                if (x < nx && y < ny && !fixed_w(f, x, y, z, &val))
                    f->w[fluid_iw(f, x, y, z)] -= (float)((PC(x, y, z) - PC(x, y, z - 1)) * inv_h);
            }
#undef PC
    compute_divergence(f);
    apply_sources(f, -1);
    real mx = 0;
    for (size_t i = 0; i < ncells(f); i++) if (r_fabs(f->div[i]) > mx) mx = r_fabs(f->div[i]);
    *max_div_out = mx;
}

// ---- 4. the cloud: MacCormack advection ------------------------------------------------------------
//
// Plain semi-Lagrangian advection blurs the cloud a little every step. MacCormack (Selle et
// al. 2008) estimates that error: advect backwards (phi_hat), then advect phi_hat forwards
// again and compare with the start; half the difference is the error, and subtracting it
// makes the result second-order accurate. The result is clamped to the values it was
// interpolated from, so it never creates new extremes (no overshoot, no negative density).
// the voxel range [lo, hi] of the Volume covering the world box b grown by `grow`
static void voxel_range(const Volume *v, Aabb b, real grow, int lo[3], int hi[3]){
    Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
    real s[3] = {v->nx / size.x, v->ny / size.y, v->nz / size.z};
    real bl[3] = {b.lo.x - grow - v->bounds.lo.x, b.lo.y - grow - v->bounds.lo.y, b.lo.z - grow - v->bounds.lo.z};
    real bh[3] = {b.hi.x + grow - v->bounds.lo.x, b.hi.y + grow - v->bounds.lo.y, b.hi.z + grow - v->bounds.lo.z};
    int n[3] = {v->nx, v->ny, v->nz};
    for (int k = 0; k < 3; k++){
        lo[k] = (int)r_floor(bl[k] * s[k] - 0.5f);
        hi[k] = (int)r_ceil(bh[k] * s[k] - 0.5f);
        if (lo[k] < 0) lo[k] = 0;
        if (hi[k] > n[k] - 1) hi[k] = n[k] - 1;
    }
}

static float max_abs(const float *a, size_t n){
    float m = 0;
    for (size_t i = 0; i < n; i++){ float x = a[i] < 0 ? -a[i] : a[i]; if (x > m) m = x; }
    return m;
}

static void advect_density(Fluid *f, real dt){
    Volume *v = f->density;
    if (!v) return;
    const int nx = v->nx, ny = v->ny;
    const size_t n = (size_t)nx * ny * v->nz;
    if (v->active.hi.x < v->active.lo.x) return;          // no cloud left
    memcpy(f->d0, v->density, n * sizeof(float));
    const float keep = (float)(f->dissipation * dt >= 1 ? 0 : 1 - f->dissipation * dt);
    const VoxGrid vg = voxgrid(v);
    // Only the voxels the cloud can reach this step: its box, grown by the largest distance
    // the air moves in dt (plus the interpolation's reach). The first pass must cover what
    // the second reads, one more displacement further.
    Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
    real vox = r_fmax(size.x / nx, r_fmax(size.y / ny, size.z / v->nz));
    real umax = r_fmax(max_abs(f->u, nu(f)), r_fmax(max_abs(f->v, nv(f)), max_abs(f->w, nw(f))));
    real disp = umax * r_sqrt(3.0f) * dt;
    int lo1[3], hi1[3], lo2[3], hi2[3];
    voxel_range(v, v->active, disp + 2 * vox, lo1, hi1);
    voxel_range(v, v->active, 2 * disp + 3 * vox, lo2, hi2);
    // phi_hat: one step backwards
    PFOR
    for (int z = lo2[2]; z <= hi2[2]; z++)
        for (int y = lo2[1]; y <= hi2[1]; y++)
            for (int x = lo2[0]; x <= hi2[0]; x++){
                Vec3 c = voxgrid_center(&vg, x, y, z);
                Vec3 from = v_sub(c, v_scale(fluid_velocity(f, c), dt));
                f->d1[((size_t)z * ny + y) * nx + x] = voxgrid_sample(&vg, f->d0, from, NULL, NULL);
            }
    // correction, clamp, obstacles
    PFOR
    for (int z = lo1[2]; z <= hi1[2]; z++)
        for (int y = lo1[1]; y <= hi1[1]; y++)
            for (int x = lo1[0]; x <= hi1[0]; x++){
                size_t i = ((size_t)z * ny + y) * nx + x;
                Vec3 c = voxgrid_center(&vg, x, y, z);
                // no cloud inside a body
                int fx = (int)r_floor((c.x - f->bounds.lo.x) * f->inv_h), fy = (int)r_floor((c.y - f->bounds.lo.y) * f->inv_h),
                    fz = (int)r_floor((c.z - f->bounds.lo.z) * f->inv_h);
                if (solid_at(f, fx, fy, fz)){ v->density[i] = 0; continue; }
                Vec3 vel = fluid_velocity(f, c);
                float mn, mx;
                voxgrid_sample(&vg, f->d0, v_sub(c, v_scale(vel, dt)), &mn, &mx);
                float back = voxgrid_sample(&vg, f->d1, v_add(c, v_scale(vel, dt)), NULL, NULL);
                float r = f->d1[i] + 0.5f * (f->d0[i] - back);
                if (r < mn) r = mn;
                if (r > mx) r = mx;
                r *= keep;
                // Where the air expands (an explosion) the same cloud fills more space: the
                // density drops as d(rho)/dt = -rho div u. (Everywhere else div u = 0.)
                if (f->nsources) r *= (float)r_exp(-source_divergence(f, c) * dt);
                // drop what is too thin to see, so the cloud's box does not grow forever
                v->density[i] = r < 1e-4f ? 0 : r;
            }
    volume_update(v);
}

// ---- the step -------------------------------------------------------------------------------------

void fluid_step(Fluid *f, real dt){
    if (dt <= 0) return;
    enforce_boundaries(f);
    add_forces(f, dt);
    vorticity_confinement(f, dt);
    enforce_boundaries(f);
    advect_velocity(f, dt);
    enforce_boundaries(f);
    project(f, &f->max_divergence);
    advect_density(f, dt);
    // the expansion sources run out
    int k = 0;
    for (int s = 0; s < f->nsources; s++){
        f->sources[s].time_left -= dt;
        if (f->sources[s].time_left > 1e-6f) f->sources[k++] = f->sources[s];
    }
    f->nsources = k;
}

void fluid_add_impulse(Fluid *f, Vec3 center, real radius, Vec3 dv, int radial){
    const int nx = f->nx, ny = f->ny, nz = f->nz;
#define KICK(arr, idx, pos, comp) do { \
        Vec3 d_ = v_sub(pos, center); real l_ = v_len(d_); \
        if (l_ < radius){ real fall_ = 1 - l_ / radius; \
            real a_ = radial ? (l_ > 1e-6f ? d_.comp * dv.x * fall_ / l_ : 0) : dv.comp * fall_; \
            arr[idx] += (float)a_; } } while (0)
    PFOR
    for (int z = 0; z <= nz; z++)
        for (int y = 0; y <= ny; y++)
            for (int x = 0; x <= nx; x++){
                if (y < ny && z < nz) KICK(f->u, fluid_iu(f, x, y, z), pos_u(f, x, y, z), x);
                if (x < nx && z < nz) KICK(f->v, fluid_iv(f, x, y, z), pos_v(f, x, y, z), y);
                if (x < nx && y < ny) KICK(f->w, fluid_iw(f, x, y, z), pos_w(f, x, y, z), z);
            }
#undef KICK
    enforce_boundaries(f);
}

real fluid_kinetic_energy(const Fluid *f){
    double e = 0;
    for (size_t i = 0; i < nu(f); i++) e += 0.5 * (double)f->u[i] * f->u[i];
    for (size_t i = 0; i < nv(f); i++) e += 0.5 * (double)f->v[i] * f->v[i];
    for (size_t i = 0; i < nw(f); i++) e += 0.5 * (double)f->w[i] * f->w[i];
    return (real)(e * f->h * f->h * f->h);
}

real fluid_total_density(const Fluid *f){
    if (!f->density) return 0;
    double s = 0;
    const Volume *v = f->density;
    for (size_t i = 0; i < (size_t)v->nx * v->ny * v->nz; i++) s += v->density[i];
    return (real)s;
}
