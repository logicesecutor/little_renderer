#include "core/volume.h"

#include <stdio.h>
#include <stdlib.h>
#include "core/rng.h"

// ---- the grid -----------------------------------------------------------------------

Volume *volume_create(int nx, int ny, int nz, Aabb bounds){
    if (nx < 1 || ny < 1 || nz < 1) return NULL;
    Volume *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->density = calloc((size_t)nx * ny * nz, sizeof(float));
    if (!v->density){ free(v); return NULL; }
    v->nx = nx; v->ny = ny; v->nz = nz;
    v->bounds = bounds;
    v->bnx = (nx + VOLUME_BRICK - 1) / VOLUME_BRICK;
    v->bny = (ny + VOLUME_BRICK - 1) / VOLUME_BRICK;
    v->bnz = (nz + VOLUME_BRICK - 1) / VOLUME_BRICK;
    v->brick_max = calloc((size_t)v->bnx * v->bny * v->bnz, sizeof(float));
    if (!v->brick_max){ free(v->density); free(v); return NULL; }
    // a white cloud: see docs/VOLUMES.md for what these numbers do
    CloudParams d = cloud_default_params();
    v->sigma_t = d.sigma_t;
    v->albedo = d.albedo;
    v->g = d.g;
    volume_update(v);
    return v;
}

void volume_free(Volume *v){
    if (!v) return;
    free(v->density);
    free(v->brick_max);
    free(v);
}

Vec3 volume_voxel_center(const Volume *v, int x, int y, int z){
    Vec3 lo = v->bounds.lo, size = v_sub(v->bounds.hi, v->bounds.lo);
    return vec3(lo.x + (x + 0.5f) * size.x / v->nx,
                lo.y + (y + 0.5f) * size.y / v->ny,
                lo.z + (z + 0.5f) * size.z / v->nz);
}

// Inside brick i (voxels i*B .. i*B+B-1 along an axis) the interpolation reads voxels
// i*B-1 .. i*B+B, so each brick's majorant is the max over that range, one voxel wider.
static void update_bricks(Volume *v){
    const int B = VOLUME_BRICK;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int bz = 0; bz < v->bnz; bz++)
        for (int by = 0; by < v->bny; by++)
            for (int bx = 0; bx < v->bnx; bx++){
                int x0 = bx * B - 1, x1 = bx * B + B, y0 = by * B - 1, y1 = by * B + B;
                int z0 = bz * B - 1, z1 = bz * B + B;
                if (x0 < 0) x0 = 0;
                if (y0 < 0) y0 = 0;
                if (z0 < 0) z0 = 0;
                if (x1 > v->nx - 1) x1 = v->nx - 1;
                if (y1 > v->ny - 1) y1 = v->ny - 1;
                if (z1 > v->nz - 1) z1 = v->nz - 1;
                float m = 0;
                for (int z = z0; z <= z1; z++)
                    for (int y = y0; y <= y1; y++)
                        for (int x = x0; x <= x1; x++){
                            float d = v->density[((size_t)z * v->ny + y) * v->nx + x];
                            if (d > m) m = d;
                        }
                v->brick_max[((size_t)bz * v->bny + by) * v->bnx + bx] = m;
            }
}

void volume_update(Volume *v){
    v->version++;
    update_bricks(v);
    int x0 = v->nx, y0 = v->ny, z0 = v->nz, x1 = -1, y1 = -1, z1 = -1;
    float mx = 0;
    for (int z = 0; z < v->nz; z++)
        for (int y = 0; y < v->ny; y++)
            for (int x = 0; x < v->nx; x++){
                float d = v->density[((size_t)z * v->ny + y) * v->nx + x];
                if (!(d > 0)) continue;
                if (d > mx) mx = d;
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
                if (z < z0) z0 = z;
                if (z > z1) z1 = z;
            }
    v->max_density = mx;
    if (x1 < 0){                                     // empty: a box that nothing can enter
        v->active.lo = vec3(1, 1, 1);
        v->active.hi = vec3(-1, -1, -1);
        return;
    }
    // trilinear interpolation spreads a voxel's density up to one voxel around its center
    Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
    Vec3 vox = vec3(size.x / v->nx, size.y / v->ny, size.z / v->nz);
    Vec3 lo = v_sub(volume_voxel_center(v, x0, y0, z0), vox);
    Vec3 hi = v_add(volume_voxel_center(v, x1, y1, z1), vox);
    Aabb b = v->bounds;
    v->active.lo = vec3(r_fmax(lo.x, b.lo.x), r_fmax(lo.y, b.lo.y), r_fmax(lo.z, b.lo.z));
    v->active.hi = vec3(r_fmin(hi.x, b.hi.x), r_fmin(hi.y, b.hi.y), r_fmin(hi.z, b.hi.z));
}

// ---- procedural cumulus clouds --------------------------------------------------------

CloudParams cloud_default_params(void){
    CloudParams p;
    p.seed = 1;
    p.puffs = 12;
    p.detail = 0.6f;
    p.time = 0;
    p.wind = vec3(0.05f, 0.0f, 0.02f);
    p.lean = vec3(0, 0, 0);
    p.resolution = 128;
    p.sigma_t = 0.08f;
    p.albedo = vec3(0.99f, 0.99f, 0.99f);
    p.g = 0.3f;
    return p;
}

static inline real smoothstep(real a, real b, real x){
    real t = (x - a) / (b - a);
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return t * t * (3 - 2 * t);
}

// A cumulus is a heap of round lobes ("puffs") sitting on a flat base: the air rises,
// cools, and the water condenses at the same height everywhere, so the bottom is flat;
// above it the rising bubbles make the cauliflower top. The puffs are placed at random
// (from the seed), bigger and higher near the middle.
void cloud_init(Cloud *c, const CloudParams *cp, Vec3 aspect){
    c->p = *cp;
    c->aspect = aspect;
    noise_init(&c->noise, cp->seed);
    Rng r;
    rng_seed(&r, 0xC10D5EEDULL ^ cp->seed, 3);
    int n = cp->puffs < 1 ? 1 : (cp->puffs > CLOUD_MAX_PUFFS ? CLOUD_MAX_PUFFS : cp->puffs);
    real base = -0.55f * aspect.y;                   // height of the flat base
    for (int i = 0; i < n; i++){
        // the first puff is the central tower, the others spread around it
        real px = i == 0 ? 0 : (2 * rng_real(&r) - 1) * 0.6f * aspect.x;
        real pz = i == 0 ? 0 : (2 * rng_real(&r) - 1) * 0.55f * aspect.z;
        real off = r_sqrt((px * px) / (aspect.x * aspect.x) + (pz * pz) / (aspect.z * aspect.z));   // 0 middle .. ~1 edge
        real rad = (0.50f - 0.20f * off + 0.08f * rng_real(&r)) * (aspect.y > 0.5f ? 1.0f : 2.0f * aspect.y);
        real top = aspect.y - rad;                   // keep every puff inside the box
        real py = base + rad * (0.55f + (1.0f - off) * (0.9f + 0.4f * rng_real(&r)));
        if (py > top) py = top;
        c->puff_c[i] = vec3(px, py, pz);
        c->puff_r[i] = rad;
    }
    c->npuffs = n;
}

// a point in the noise's space, stretched along a wind (dir, unit; s = how strongly): the
// features come out longer that way, streaks
static Vec3 along_wind(Vec3 q, Vec3 dir, real s){
    return s > 0 ? v_sub(q, v_scale(dir, v_dot(q, dir) * (s / (1 + s)))) : q;
}

real cloud_density(const Cloud *c, Vec3 u){
    const Noise *n = &c->noise;
    // animation: the detail noise drifts with the wind (2 = one cloud size in these units)
    Vec3 drift = v_scale(c->p.wind, 2.0f * c->p.time);
    // blown by a wind: each height is shifted downwind in proportion (the top pushed over the
    // base, as a cumulus sheared by the wind above it), the detail drawn out along the wind
    Vec3 uo = u;                                     // (the box walls stay where they are)
    Vec3 lean = vec3(c->p.lean.x, 0, c->p.lean.z);
    real ls = v_len(lean);
    Vec3 ld = ls > 0 ? v_scale(lean, 1.0f / ls) : vec3(0, 0, 0);
    if (ls > 0) u = v_sub(u, v_scale(lean, u.y + 0.55f * c->aspect.y));

    // 1. domain warping: push the sample point around with smooth noise, so the puffs get
    //    irregular outlines instead of perfect spheres
    Vec3 q = along_wind(v_add(v_scale(u, 1.6f), drift), ld, 2 * ls);
    Vec3 w = vec3(noise_fbm(n, q, 3), noise_fbm(n, v_add(q, vec3(31.4f, 0, 0)), 3),
                  noise_fbm(n, v_add(q, vec3(0, 0, 47.2f)), 3));
    Vec3 uw = v_add(u, v_scale(w, 0.16f));

    // 2. the shape: the strongest of the puffs (1 at a center, 0 at its surface), cut by
    //    the flat base and faded towards the box walls
    real shape = 0;
    for (int i = 0; i < c->npuffs; i++){
        real s = 1.0f - v_len(v_sub(uw, c->puff_c[i])) / c->puff_r[i];
        if (s > shape) shape = s;
    }
    if (shape <= 0) return 0;
    real base = -0.55f * c->aspect.y;
    shape *= smoothstep(base - 0.02f, base + 0.08f, uw.y);
    shape *= smoothstep(c->aspect.x, c->aspect.x - 0.1f, r_fabs(uo.x)) *
             smoothstep(c->aspect.y, c->aspect.y - 0.1f, r_fabs(uo.y)) *
             smoothstep(c->aspect.z, c->aspect.z - 0.1f, r_fabs(uo.z));
    if (shape <= 0) return 0;

    // 3. erosion: billows (inverted Worley) and fine Perlin detail eat into the edges. The
    //    core, where shape is large, survives; thin edges break into wisps.
    Vec3 qd = along_wind(v_add(v_scale(uw, 3.2f), v_scale(drift, 1.6f)), ld, 2 * ls);
    real bill = noise_billow(n, qd, 3);
    real fine = 0.5f + 0.5f * noise_fbm(n, v_add(v_scale(qd, 2.3f), vec3(5.2f, 1.3f, 7.7f)), 3);
    real detail = 0.75f * bill + 0.25f * fine;       // 0 .. 1, 1 = keep
    real d = shape - c->p.detail * (1.0f - detail) * 0.45f;
    d *= 5.0f;                                        // dense core, short soft edge
    return d <= 0 ? 0 : (d > 1 ? 1 : d);
}

void volume_make_cloud(Volume *v, const CloudParams *cp){
    Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
    real big = r_fmax(size.x, r_fmax(size.y, size.z));
    Vec3 aspect = v_scale(size, 1.0f / big);
    Vec3 mid = v_scale(v_add(v->bounds.lo, v->bounds.hi), 0.5f);
    Cloud c;
    cloud_init(&c, cp, aspect);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int z = 0; z < v->nz; z++)
        for (int y = 0; y < v->ny; y++)
            for (int x = 0; x < v->nx; x++){
                // world position -> cloud coordinates (largest half-size = 1)
                Vec3 u = v_scale(v_sub(volume_voxel_center(v, x, y, z), mid), 2.0f / big);
                v->density[((size_t)z * v->ny + y) * v->nx + x] = (float)cloud_density(&c, u);
            }
    v->sigma_t = cp->sigma_t;
    v->albedo = cp->albedo;
    v->g = cp->g;
    volume_update(v);
}

void volume_add_cloud(Volume *v, Aabb box, const CloudParams *cp){
    Vec3 size = v_sub(box.hi, box.lo);
    real big = r_fmax(size.x, r_fmax(size.y, size.z));
    Vec3 aspect = v_scale(size, 1.0f / big);
    Vec3 mid = v_scale(v_add(box.lo, box.hi), 0.5f);
    Cloud c;
    cloud_init(&c, cp, aspect);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int z = 0; z < v->nz; z++)
        for (int y = 0; y < v->ny; y++)
            for (int x = 0; x < v->nx; x++){
                Vec3 p = volume_voxel_center(v, x, y, z);
                if (p.x < box.lo.x || p.y < box.lo.y || p.z < box.lo.z || p.x > box.hi.x || p.y > box.hi.y || p.z > box.hi.z) continue;
                float d = (float)cloud_density(&c, v_scale(v_sub(p, mid), 2.0f / big));
                float *o = &v->density[((size_t)z * v->ny + y) * v->nx + x];
                if (d > *o) *o = d;
            }
    v->sigma_t = cp->sigma_t;
    v->albedo = cp->albedo;
    v->g = cp->g;
    volume_update(v);
}

Volume *volume_new_cloud(Aabb bounds, const CloudParams *cp){
    Vec3 size = v_sub(bounds.hi, bounds.lo);
    real big = r_fmax(size.x, r_fmax(size.y, size.z));
    int res = cp->resolution > 0 ? cp->resolution : 128;
    int nx = (int)(res * size.x / big + 0.5f), ny = (int)(res * size.y / big + 0.5f), nz = (int)(res * size.z / big + 0.5f);
    Volume *v = volume_create(nx < 1 ? 1 : nx, ny < 1 ? 1 : ny, nz < 1 ? 1 : nz, bounds);
    if (v) volume_make_cloud(v, cp);
    return v;
}
