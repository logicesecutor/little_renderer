#define _POSIX_C_SOURCE 200809L
#include "raster/volume_march.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/volume.h"

// Multiple-scattering octaves (Wrenninge, "Art-Directable Multiple Volumetric Scattering",
// 2015): octave i adds the lamp light again with weight OCT_WEIGHT^i, optical depth
// OCT_DEPTH^i and anisotropy 0.5^i. Light that has bounced many times inside a cloud has
// gone through less of it on its way (it took the easy paths) and has forgotten its
// direction: less extinction, more uniform. The constants were tuned against the path
// traced cloud (bin/render -scene cloud).
#define OCT_WEIGHT 0.7f
#define OCT_DEPTH 0.3f

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

MarchParams march_default_params(void){
    MarchParams p = {128, 1.0f, 48, -1.0f, 3, 2.0f, 1};
    return p;
}

// ---- the lamp ----------------------------------------------------------------------------

typedef struct {
    Vec3 pos, normal;    // center, and the side it emits towards (0 = all sides)
    Vec3 radiance;       // emitted radiance
    real area;
} Lamp;

static Lamp scene_lamp(const Scene *s){
    Lamp l = {vec3(278, 554, 278), vec3(0, -1, 0), vec3(0, 0, 0), 1};
    if (s->light < 0 || s->light >= s->nprims) return l;
    const Primitive *p = &s->prims[s->light];
    l.radiance = s->mats[p->material].emission;
    l.pos = scene_light_center(s);
    if (p->type == PRIM_QUAD){ l.normal = p->normal; l.area = p->area; }
    else { l.normal = vec3(0, 0, 0); l.area = (real)PI * p->radius * p->radius; }
    return l;
}

// Irradiance from the lamp at x (unoccluded). For a disk of area A seen face-on from d away
// it is exactly L A / (d^2 + A / pi): close to the lamp it stays finite, far away it is
// the point-light L A / d^2. The cosine of the lamp's own emitting side is included.
static Vec3 lamp_irradiance(const Lamp *l, Vec3 x, Vec3 *dir_from_lamp){
    Vec3 d = v_sub(x, l->pos);
    real d2 = v_dot(d, d);
    real len = r_sqrt(d2);
    Vec3 w = len > 1e-6f ? v_scale(d, 1.0f / len) : vec3(0, -1, 0);
    *dir_from_lamp = w;
    real c = v_dot(l->normal, l->normal) > 0 ? v_dot(l->normal, w) : 1;
    if (c <= 0) return vec3(0, 0, 0);
    return v_scale(l->radiance, l->area * c / (d2 + l->area / (real)PI));
}

// Light reaching a medium from the room's surfaces, as one uniform radiance: the lamp's
// power spread over the walls, times the interreflection of walls with mean albedo a
// (a / (1 - a): one bounce, two, ...), back out as radiance (/ pi). The color is the
// walls' mean albedo color, so a cloud between a red and a green wall picks up both.
static Vec3 ambient_estimate(const Scene *s, const Lamp *l){
    real area = 0;
    Vec3 alb = vec3(0, 0, 0);
    for (int i = 0; i < s->nprims; i++){
        const Primitive *p = &s->prims[i];
        if (i == s->light || p->type != PRIM_QUAD) continue;
        const Material *m = &s->mats[p->material];
        if (m->type == MAT_LIGHT) continue;
        Vec3 a = material_kind(m) == MAT_DIFFUSE ? m->albedo : v_scale(m->albedo, 0.5f);
        alb = v_add(alb, v_scale(a, p->area));
        area += p->area;
    }
    if (area <= 0) return vec3(0, 0, 0);
    alb = v_scale(alb, 1.0f / area);
    real power = (real)PI * l->area * (l->radiance.x + l->radiance.y + l->radiance.z) / 3;
    real e = power / area;                                    // first-bounce irradiance
    Vec3 r;
    r.x = alb.x / (1 - r_fmin(alb.x, 0.95f)) * e / (real)PI;
    r.y = alb.y / (1 - r_fmin(alb.y, 0.95f)) * e / (real)PI;
    r.z = alb.z / (1 - r_fmin(alb.z, 0.95f)) * e / (real)PI;
    return r;
}

// ---- shared with the GPU raymarcher (gpu/view_gpu.c) ----------------------------------------

void march_light_dims(const Volume *v, int light_res, int dims[3]){
    // no finer than the medium itself
    Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
    real big = r_fmax(size.x, r_fmax(size.y, size.z));
    int n[3] = {(int)(light_res * size.x / big + 0.5f), (int)(light_res * size.y / big + 0.5f),
                (int)(light_res * size.z / big + 0.5f)};
    int vn[3] = {v->nx, v->ny, v->nz};
    for (int k = 0; k < 3; k++){
        if (n[k] > vn[k]) n[k] = vn[k];
        if (n[k] < 2) n[k] = 2;
        dims[k] = n[k];
    }
}

MarchLamp march_lamp(const Scene *s){
    Lamp l = scene_lamp(s);
    MarchLamp m = {l.pos, l.normal, l.radiance, l.area};
    return m;
}

Vec3 march_ambient(const Scene *s, const MarchParams *mp){
    if (mp && mp->ambient >= 0) return vec3(mp->ambient, mp->ambient, mp->ambient);
    Lamp l = scene_lamp(s);
    return ambient_estimate(s, &l);
}

// ---- ray against a box ---------------------------------------------------------------------

static int ray_box(Vec3 o, Vec3 inv_d, Aabb b, real *t0, real *t1){
    real tx0 = (b.lo.x - o.x) * inv_d.x, tx1 = (b.hi.x - o.x) * inv_d.x;
    real ty0 = (b.lo.y - o.y) * inv_d.y, ty1 = (b.hi.y - o.y) * inv_d.y;
    real tz0 = (b.lo.z - o.z) * inv_d.z, tz1 = (b.hi.z - o.z) * inv_d.z;
    real a = r_fmax(r_fmax(r_fmin(tx0, tx1), r_fmin(ty0, ty1)), r_fmin(tz0, tz1));
    real c = r_fmin(r_fmin(r_fmax(tx0, tx1), r_fmax(ty0, ty1)), r_fmax(tz0, tz1));
    *t0 = a; *t1 = c;
    return c >= a && c > 0;
}

static Vec3 safe_inv(Vec3 d){
    return vec3(r_fabs(d.x) > 1e-12f ? 1.0f / d.x : 1e30f, r_fabs(d.y) > 1e-12f ? 1.0f / d.y : 1e30f,
                r_fabs(d.z) > 1e-12f ? 1.0f / d.z : 1e30f);
}

static real voxel_size(const Volume *v){
    Vec3 s = v_sub(v->bounds.hi, v->bounds.lo);
    return r_fmin(s.x / v->nx, r_fmin(s.y / v->ny, s.z / v->nz));
}

// optical depth of all media along the segment a -> b (midpoint rule, about a voxel per step)
static real optical_depth(const Scene *s, Vec3 a, Vec3 b){
    Vec3 d = v_sub(b, a);
    real len = v_len(d);
    if (len < 1e-6f) return 0;
    Vec3 dir = v_scale(d, 1.0f / len), inv = safe_inv(dir);
    real tau = 0;
    for (int k = 0; k < s->nvolumes; k++){
        const Volume *v = s->volumes[k];
        if (v->max_density <= 0) continue;
        real t0, t1;
        if (!ray_box(a, inv, v->active, &t0, &t1)) continue;
        t0 = r_fmax(t0, 0);
        t1 = r_fmin(t1, len);
        if (t1 <= t0) continue;
        real ds = voxel_size(v);
        int n = (int)r_ceil((t1 - t0) / ds);
        if (n > 256) n = 256;
        if (n < 1) n = 1;
        ds = (t1 - t0) / n;
        real sum = 0;
        for (int i = 0; i < n; i++) sum += volume_density(v, v_add(a, v_scale(dir, t0 + (i + 0.5f) * ds)));
        tau += v->sigma_t * sum * ds;
    }
    return tau;
}

// ---- light grids -------------------------------------------------------------------------------

// For each medium, a coarse grid over its bounds holding the optical depth from that point
// to the lamp through ALL media (so the fog dims the light reaching the cloud, and the
// cloud shades the fog below it).
typedef struct {
    int nx, ny, nz;
    Aabb bounds;
    float *tau;
} LightGrid;

typedef struct { Vec3 L; real T, tend; } LowSample;   // one ray of the half-resolution pass

struct RasterVolumes {
    LowSample *low;
    size_t low_n;
    LightGrid grid[MAX_VOLUMES];
    const Volume *vols[MAX_VOLUMES];   // what they were built for
    uint32_t versions[MAX_VOLUMES];
    int nvolumes, light_res;
    Vec3 lamp_pos;
    double ms_light, ms_march;
};

RasterVolumes *raster_volumes_create(void){
    return calloc(1, sizeof(RasterVolumes));
}

void raster_volumes_free(RasterVolumes *rv){
    if (!rv) return;
    for (int i = 0; i < MAX_VOLUMES; i++) free(rv->grid[i].tau);
    free(rv->low);
    free(rv);
}

void raster_volumes_timing(const RasterVolumes *rv, double *ms_light, double *ms_march){
    if (ms_light) *ms_light = rv->ms_light;
    if (ms_march) *ms_march = rv->ms_march;
}

static int grids_valid(const RasterVolumes *rv, const Scene *s, Vec3 lamp, int light_res){
    if (rv->nvolumes != s->nvolumes || rv->light_res != light_res) return 0;
    if (v_len(v_sub(rv->lamp_pos, lamp)) > 1e-3f) return 0;
    for (int i = 0; i < s->nvolumes; i++)
        if (rv->vols[i] != s->volumes[i] || rv->versions[i] != s->volumes[i]->version) return 0;
    return 1;
}

static void build_grids(RasterVolumes *rv, const Scene *s, const Lamp *l, int light_res){
    for (int k = 0; k < s->nvolumes; k++){
        const Volume *v = s->volumes[k];
        LightGrid *g = &rv->grid[k];
        int dims[3];
        march_light_dims(v, light_res, dims);
        int nx = dims[0], ny = dims[1], nz = dims[2];
        Vec3 size = v_sub(v->bounds.hi, v->bounds.lo);
        if (g->nx * g->ny * g->nz != nx * ny * nz || !g->tau){
            free(g->tau);
            g->tau = calloc((size_t)nx * ny * nz, sizeof(float));
            if (!g->tau){ perror("calloc"); exit(1); }
        }
        g->nx = nx; g->ny = ny; g->nz = nz;
        g->bounds = v->bounds;
        Vec3 cell = vec3(size.x / nx, size.y / ny, size.z / nz);
        // only where the medium is (grown by a cell): elsewhere it is never looked up
        Aabb act = v->active;
        int empty = act.hi.x < act.lo.x;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
                for (int x = 0; x < nx; x++){
                    Vec3 c = vec3(v->bounds.lo.x + (x + 0.5f) * cell.x, v->bounds.lo.y + (y + 0.5f) * cell.y,
                                  v->bounds.lo.z + (z + 0.5f) * cell.z);
                    float *t = &g->tau[((size_t)z * ny + y) * nx + x];
                    if (empty || c.x < act.lo.x - cell.x || c.y < act.lo.y - cell.y || c.z < act.lo.z - cell.z ||
                        c.x > act.hi.x + cell.x || c.y > act.hi.y + cell.y || c.z > act.hi.z + cell.z){ *t = 0; continue; }
                    *t = (float)optical_depth(s, c, l->pos);
                }
        rv->vols[k] = v;
        rv->versions[k] = v->version;
    }
    rv->nvolumes = s->nvolumes;
    rv->light_res = light_res;
    rv->lamp_pos = l->pos;
}

// trilinear lookup of the light grid (clamped at its border)
static real grid_tau(const LightGrid *g, Vec3 p){
    real gx = (p.x - g->bounds.lo.x) / (g->bounds.hi.x - g->bounds.lo.x) * g->nx - 0.5f;
    real gy = (p.y - g->bounds.lo.y) / (g->bounds.hi.y - g->bounds.lo.y) * g->ny - 0.5f;
    real gz = (p.z - g->bounds.lo.z) / (g->bounds.hi.z - g->bounds.lo.z) * g->nz - 0.5f;
    gx = r_fmin(r_fmax(gx, 0), g->nx - 1.001f);
    gy = r_fmin(r_fmax(gy, 0), g->ny - 1.001f);
    gz = r_fmin(r_fmax(gz, 0), g->nz - 1.001f);
    int x = (int)gx, y = (int)gy, z = (int)gz;
    real fx = gx - x, fy = gy - y, fz = gz - z;
    const size_t sy = (size_t)g->nx, sz = (size_t)g->nx * g->ny;
    const float *c = g->tau + (size_t)z * sz + (size_t)y * sy + x;
    real c00 = c[0] + (c[1] - c[0]) * fx, c10 = c[sy] + (c[sy + 1] - c[sy]) * fx;
    real c01 = c[sz] + (c[sz + 1] - c[sz]) * fx, c11 = c[sz + sy] + (c[sz + sy + 1] - c[sz + sy]) * fx;
    real c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
    return c0 + (c1 - c0) * fz;
}

// largest density the brick around p can have (majorant grid), for skipping empty space
static real brick_majorant(const Volume *v, Vec3 p){
    real gx = (p.x - v->bounds.lo.x) / (v->bounds.hi.x - v->bounds.lo.x) * v->nx;
    real gy = (p.y - v->bounds.lo.y) / (v->bounds.hi.y - v->bounds.lo.y) * v->ny;
    real gz = (p.z - v->bounds.lo.z) / (v->bounds.hi.z - v->bounds.lo.z) * v->nz;
    int bx = (int)(gx / VOLUME_BRICK), by = (int)(gy / VOLUME_BRICK), bz = (int)(gz / VOLUME_BRICK);
    if (gx < 0 || gy < 0 || gz < 0 || bx >= v->bnx || by >= v->bny || bz >= v->bnz) return 0;
    return v->brick_max[((size_t)bz * v->bny + by) * v->bnx + bx];
}

// ---- the march ---------------------------------------------------------------------------------

static float srgb_to_linear_lut[256];

static void init_lut(void){
    static int done = 0;
    if (done) return;
    for (int i = 0; i < 256; i++){
        double c = i / 255.0;
        srgb_to_linear_lut[i] = (float)(c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4));
    }
    done = 1;
}

// a fixed, per-pixel offset of the first step in [0, 1): breaks up the banding of regular
// steps into fine noise (the same every frame, so it does not flicker)
static real pixel_jitter(int x, int y){
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)y * 0xd8163841u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (h >> 8) * (1.0f / 16777216.0f);
}

// everything a ray needs, the same for every pixel of a frame
typedef struct {
    const RasterVolumes *rv;
    const Scene *s;
    const MarchParams *mp;
    Lamp lamp;
    Vec3 amb, eye;
    real vstep[MAX_VOLUMES];
    int octaves;
} MarchCtx;

// One ray from the eye along unit dir, stopped at tmax: the light it collects (L) and the
// fraction of what is behind that gets through (T).
static void march_ray(const MarchCtx *mc, Vec3 dir, real tmax, real jit, Vec3 *Lout, real *Tout){
    const Scene *s = mc->s;
    const MarchParams *mp = mc->mp;
    const Vec3 eye = mc->eye, inv = safe_inv(dir);
    Vec3 L = vec3(0, 0, 0);
    real T = 1;
    *Lout = L; *Tout = T;
    // the part of the ray inside each medium
    real ta = 1e30f, tb = 0;
    real seg0[MAX_VOLUMES], seg1[MAX_VOLUMES];
    for (int k = 0; k < s->nvolumes; k++){
        const Volume *v = s->volumes[k];
        seg0[k] = 1; seg1[k] = 0;
        if (v->max_density <= 0) continue;
        real a, b;
        if (!ray_box(eye, inv, v->active, &a, &b)) continue;
        a = r_fmax(a, 0);
        b = r_fmin(b, tmax);
        if (b <= a) continue;
        seg0[k] = a; seg1[k] = b;
        ta = r_fmin(ta, a);
        tb = r_fmax(tb, b);
    }
    if (tb <= ta) return;
    // steps: about one voxel of the finest medium the ray is in, but never more than
    // max_steps over the whole segment; 4x longer through empty bricks
    const real ds_floor = (tb - ta) / mp->max_steps;
    real t = ta;
    int first = 1, steps = 0;
    while (t < tb && T > 0.003f && steps < 4 * mp->max_steps){
        steps++;
        real ds = 1e30f;
        for (int k = 0; k < s->nvolumes; k++)
            if (t >= seg0[k] && t <= seg1[k]) ds = r_fmin(ds, mc->vstep[k]);
        if (ds >= 1e30f) ds = ds_floor > 0 ? ds_floor : 1;
        ds = r_fmax(ds, ds_floor);
        if (first){ t += ds * jit; first = 0; if (t >= tb) break; }
        Vec3 pos = v_add(eye, v_scale(dir, t));
        real sig = 0;
        Vec3 src = vec3(0, 0, 0);       // sum over media of sigma_t * albedo * S
        int any = 0, lamp_done = 0;
        Vec3 E = vec3(0, 0, 0), wl = vec3(0, -1, 0);
        real cosv = 0;
        for (int k = 0; k < s->nvolumes; k++){
            if (t < seg0[k] || t > seg1[k]) continue;
            const Volume *v = s->volumes[k];
            if (brick_majorant(v, pos) <= 0) continue;
            any = 1;
            real rho = volume_density(v, pos);
            if (rho <= 0) continue;
            real sg = v->sigma_t * rho;
            // light from the lamp: dimmed by the media on the way (light grid), turned
            // towards the eye by the phase function; plus the octaves
            if (!lamp_done){ E = lamp_irradiance(&mc->lamp, pos, &wl); cosv = -v_dot(wl, dir); lamp_done = 1; }
            real tau = grid_tau(&mc->rv->grid[k], pos);
            real lamp_term = 0, a = 1, b = 1, c = 1;
            for (int o = 0; o < mc->octaves; o++){
                lamp_term += a * r_exp(-b * tau) * phase_hg(cosv, v->g * c);
                a *= OCT_WEIGHT; b *= OCT_DEPTH; c *= 0.5f;
            }
            // the walls' light reaches the inside of a dense cloud less
            real amb_t = 0.35f + 0.65f * r_exp(-0.25f * tau);
            Vec3 S = v_add(v_scale(E, lamp_term * mp->brightness), v_scale(mc->amb, amb_t));
            src = v_add(src, v_scale(v_mul(v->albedo, S), sg));
            sig += sg;
        }
        real step = any ? ds : ds * 4;   // nothing here: hurry through
        // do not step over the start of a medium with finer voxels
        for (int k = 0; k < s->nvolumes; k++)
            if (seg0[k] > t && seg0[k] < t + step && mc->vstep[k] < step) step = seg0[k] - t + 1e-3f;
        if (sig > 0){
            real st = r_exp(-sig * step);
            // energy-conserving step (Hillaire 2015): the light scattered in this step,
            // integrated exactly for a density constant over the step
            L = v_add(L, v_scale(src, T * (1 - st) / sig));
            T *= st;
        }
        t += step;
    }
    *Lout = L; *Tout = T;
}

static void composite(Framebuffer *fb, size_t idx, Vec3 L, real T){
    if (T >= 1) return;
    uint32_t c = fb->color[idx];
    Vec3 bg = vec3(srgb_to_linear_lut[(c >> 16) & 255], srgb_to_linear_lut[(c >> 8) & 255], srgb_to_linear_lut[c & 255]);
    fb->color[idx] = image_pack_srgb(v_add(L, v_scale(bg, T)));
}

void raster_draw_volumes(RasterVolumes *rv, Framebuffer *fb, const Scene *s, const RenderParams *p,
                         const MarchParams *mp){
    rv->ms_light = rv->ms_march = 0;
    if (s->nvolumes == 0) return;
    MarchParams def = march_default_params();
    if (!mp) mp = &def;
    init_lut();
    srgb_lut_init();
    MarchCtx mc;
    mc.rv = rv; mc.s = s; mc.mp = mp;
    mc.lamp = scene_lamp(s);
    mc.eye = p->eye;

    double t0 = now_ms();
    if (!grids_valid(rv, s, mc.lamp.pos, mp->light_res)) build_grids(rv, s, &mc.lamp, mp->light_res);
    double t1 = now_ms();
    rv->ms_light = t1 - t0;

    mc.amb = mp->ambient >= 0 ? vec3(mp->ambient, mp->ambient, mp->ambient) : ambient_estimate(s, &mc.lamp);
    // each medium's step: about one of its voxels (a 16^3 fog takes long steps, a 128^3
    // cloud short ones); a ray takes the shortest of the media it is in at the moment
    int nonempty = 0;
    for (int k = 0; k < s->nvolumes; k++){
        mc.vstep[k] = voxel_size(s->volumes[k]) * (mp->step_voxels > 0 ? mp->step_voxels : 1);
        if (s->volumes[k]->max_density > 0) nonempty = 1;
    }
    if (!nonempty) return;
    mc.octaves = mp->octaves < 1 ? 1 : (mp->octaves > 4 ? 4 : mp->octaves);

    // camera: rows of the view matrix are the camera axes; projection gives the lens
    const Vec3 right = vec3(p->view.m[0][0], p->view.m[0][1], p->view.m[0][2]);
    const Vec3 up = vec3(p->view.m[1][0], p->view.m[1][1], p->view.m[1][2]);
    const Vec3 back = vec3(p->view.m[2][0], p->view.m[2][1], p->view.m[2][2]);
    const real sx = 1.0f / p->proj.m[0][0], sy = 1.0f / p->proj.m[1][1];
    const real A = p->proj.m[2][2], B = p->proj.m[2][3];   // NDC z = -A - B / z_eye
    const int W = fb->w, H = fb->h;
    // the (unnormalized) direction through pixel coordinates (px, py): at view depth D
    // the ray is at eye + D * du
#define PIXEL_DIR(px, py) v_sub(v_add(v_scale(right, (2 * (px) / W - 1) * sx), v_scale(up, (1 - 2 * (py) / H) * sy)), back)
    // distance along the unit direction to the opaque surface of pixel idx (1e30 if none)
#define SURFACE(idx, dl) (fb->depth[idx] < 1 && B / (A + 2 * fb->depth[idx] - 1) > 0 ? B / (A + 2 * fb->depth[idx] - 1) * (dl) : 1e30f)

    const int sc = mp->scale >= 2 ? 2 : 1;
    if (sc == 1){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++){
                Vec3 du = PIXEL_DIR(x + 0.5f, y + 0.5f);
                real dl = v_len(du);
                size_t idx = (size_t)y * W + x;
                Vec3 L;
                real T;
                march_ray(&mc, v_scale(du, 1.0f / dl), SURFACE(idx, dl), pixel_jitter(x, y), &L, &T);
                composite(fb, idx, L, T);
            }
    } else {
        // Half resolution: one ray per 2x2 block (a quarter of the work), through the
        // block's center, stopped at the farthest surface of the block. Media are smooth,
        // so a bilinear upsampling is enough, EXCEPT across depth edges: a block half on
        // a near sphere and half on the far wall. There each full-resolution pixel only
        // takes the blocks whose ray stopped about where its own does ("bilateral").
        const int w2 = (W + 1) / 2, h2 = (H + 1) / 2;
        size_t n2 = (size_t)w2 * h2;
        if (rv->low_n < n2){
            free(rv->low);
            rv->low = malloc(n2 * sizeof(*rv->low));
            if (!rv->low){ perror("malloc"); exit(1); }
            rv->low_n = n2;
        }
        LowSample *low = rv->low;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
        for (int j = 0; j < h2; j++)
            for (int i = 0; i < w2; i++){
                Vec3 du = PIXEL_DIR(2 * i + 1.0f, 2 * j + 1.0f);
                real dl = v_len(du);
                real tend = 0;
                for (int q = 0; q < 4; q++){
                    int x = 2 * i + (q & 1), y = 2 * j + (q >> 1);
                    if (x >= W || y >= H) continue;
                    tend = r_fmax(tend, SURFACE((size_t)y * W + x, dl));
                }
                LowSample *ls = &low[(size_t)j * w2 + i];
                march_ray(&mc, v_scale(du, 1.0f / dl), tend, pixel_jitter(i, j), &ls->L, &ls->T);
                ls->tend = tend;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++){
                size_t idx = (size_t)y * W + x;
                Vec3 du = PIXEL_DIR(x + 0.5f, y + 0.5f);
                real dl = v_len(du);
                real own = SURFACE(idx, dl);
                // the 4 nearest low-resolution samples (centers at 2i + 1)
                real fx = (x + 0.5f) / 2 - 1, fy = (y + 0.5f) / 2 - 1;
                int i0 = (int)r_floor(fx), j0 = (int)r_floor(fy);
                real ax = fx - i0, ay = fy - j0;
                Vec3 L = vec3(0, 0, 0);
                real T = 0, wsum = 0;
                for (int q = 0; q < 4; q++){
                    int i = i0 + (q & 1), j = j0 + (q >> 1);
                    if (i < 0) i = 0;
                    if (j < 0) j = 0;
                    if (i >= w2) i = w2 - 1;
                    if (j >= h2) j = h2 - 1;
                    const LowSample *ls = &low[(size_t)j * w2 + i];
                    real wb = ((q & 1) ? ax : 1 - ax) * ((q >> 1) ? ay : 1 - ay);
                    real a = r_fmin(own, 1e6f), b = r_fmin(ls->tend, 1e6f);
                    real rel = r_fabs(a - b) / r_fmax(r_fmin(a, b), 1);
                    real w = (wb + 1e-4f) / (0.02f + rel);
                    L = v_add(L, v_scale(ls->L, w));
                    T += ls->T * w;
                    wsum += w;
                }
                composite(fb, idx, v_scale(L, 1.0f / wsum), T / wsum);
            }
    }
#undef PIXEL_DIR
#undef SURFACE
    rv->ms_march = now_ms() - t1;
}
