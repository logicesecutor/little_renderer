// env.c - the environment map (see env.h)
#include "scene/env.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "formats/hdr.h"

static inline Vec3 v_lerp_(Vec3 a, Vec3 b, real f){ return v_add(a, v_scale(v_sub(b, a), f)); }
static inline real bsdf_clamp01_(real x){ return x < 0 ? 0 : x > 1 ? 1 : x; }
static inline real lum3(const float *c){ return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }


// atan2 and acos by polynomials (errors 1e-5 and 7e-5 radians, a 512-wide map's texel is
// 1.2e-2): the library's are most of a rasterizer's lookup's time. (Not for the path
// tracers' densities: those must find the very pixel env_sample drew from.)
static inline real fast_atan2(real y, real x){
    real ax = r_fabs(x), ay = r_fabs(y);
    real mx = ax > ay ? ax : ay, mn = ax > ay ? ay : ax;
    if (mx <= 0) return 0;
    real a = mn / mx, s = a * a;
    real r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (ay > ax) r = (real)(0.5 * PI) - r;
    if (x < 0) r = (real)PI - r;
    return y < 0 ? -r : r;
}
static inline real fast_acos(real x){
    real ax = r_fabs(x);
    real r = r_sqrt(r_fmax(1 - ax, 0)) * (1.5707288f + ax * (-0.2121144f + ax * (0.0742610f - 0.0187293f * ax)));
    return x < 0 ? (real)PI - r : r;
}

// the map's own coordinates (no rot): exact, and fast (for the rasterizers' lookups)
static void map_uv(Vec3 d, real *u, real *v){
    real y = d.y < -1 ? -1 : d.y > 1 ? 1 : d.y;
    real uu = 0.5f + r_atan2(d.x, d.z) * (real)(0.5 / PI);
    *u = uu - r_floor(uu);
    *v = r_acos(y) * (real)(1.0 / PI);
}
static void map_uv_fast(Vec3 d, real *u, real *v){
    real y = d.y < -1 ? -1 : d.y > 1 ? 1 : d.y;
    real uu = 0.5f + fast_atan2(d.x, d.z) * (real)(0.5 / PI);
    *u = uu - r_floor(uu);
    *v = fast_acos(y) * (real)(1.0 / PI);
}
static Vec3 map_dir(real u, real v){
    real ph = (u - 0.5f) * (real)(2 * PI), th = v * (real)PI;
    real st = r_sin(th);
    return vec3(st * r_sin(ph), r_cos(th), st * r_cos(ph));
}

void env_uv(const Env *e, Vec3 d, real *u, real *v){
    if (e->rot != 0) d = vec3(d.x * e->rot_c - d.z * e->rot_s, d.y, d.x * e->rot_s + d.z * e->rot_c);
    map_uv(d, u, v);
}

void env_set_rot(Env *e, real rot){
    e->rot = rot;
    e->rot_c = r_cos(rot);
    e->rot_s = r_sin(rot);
}
Vec3 env_dir(const Env *e, real u, real v){
    Vec3 d = map_dir(u, v);
    return e->rot != 0 ? vec3(d.x * e->rot_c + d.z * e->rot_s, d.y, -d.x * e->rot_s + d.z * e->rot_c) : d;
}

// bilinear in an rgb map of w x h: across wraps around, down stops at the poles
static Vec3 bilerp(const float *img, int w, int h, real u, real v){
    real x = u * w - 0.5f, y = v * h - 0.5f;
    int x0 = (int)r_floor(x), y0 = (int)r_floor(y);
    real fx = x - x0, fy = y - y0;
    int y1 = y0 + 1;
    if (y0 < 0) y0 = 0;
    if (y1 > h - 1) y1 = h - 1;
    if (y0 > h - 1) y0 = h - 1;
    int xa = x0 < 0 ? x0 + w : (x0 >= w ? x0 - w : x0);       // (u in 0..1: x0 in -1 .. w)
    if (xa < 0 || xa >= w) xa = ((xa % w) + w) % w;
    int xb = xa + 1 == w ? 0 : xa + 1;
    const float *a = img + 3 * ((size_t)y0 * w + xa), *b = img + 3 * ((size_t)y0 * w + xb);
    const float *c = img + 3 * ((size_t)y1 * w + xa), *d = img + 3 * ((size_t)y1 * w + xb);
    Vec3 r;
    r.x = (1 - fy) * ((1 - fx) * a[0] + fx * b[0]) + fy * ((1 - fx) * c[0] + fx * d[0]);
    r.y = (1 - fy) * ((1 - fx) * a[1] + fx * b[1]) + fy * ((1 - fx) * c[1] + fx * d[1]);
    r.z = (1 - fy) * ((1 - fx) * a[2] + fx * b[2]) + fy * ((1 - fx) * c[2] + fx * d[2]);
    return r;
}

Vec3 env_eval_fast(const Env *e, Vec3 d){
    real u, v;
    if (e->rot != 0) d = vec3(d.x * e->rot_c - d.z * e->rot_s, d.y, d.x * e->rot_s + d.z * e->rot_c);
    map_uv_fast(d, &u, &v);
    return v_scale(bilerp(e->rgb, e->w, e->h, u, v), e->scale);
}

Vec3 env_eval(const Env *e, Vec3 d){
    real u, v;
    env_uv(e, d, &u, &v);
    return v_scale(bilerp(e->rgb, e->w, e->h, u, v), e->scale);
}

// ---- importance sampling ----

static int upper_bound(const float *c, int n, real x){   // the last i with c[i] <= x, in 0..n-1
    int lo = 0, hi = n;                                   // c[0] = 0 <= x < c[n] (= 1)
    while (hi - lo > 1){
        int mid = (lo + hi) / 2;
        if (c[mid] <= x) lo = mid; else hi = mid;
    }
    return lo;
}

// a pixel's weight for sampling: the largest luminance around it (3 x 3), so that every
// pixel the bilinear lookup reaches from a bright one is drawn too
static real pixel_weight(const Env *e, int x, int y){
    return (real)e->wmax[(size_t)y * e->w + x] * e->row_sin[y];
}

// The table behind pixel_weight: each pixel's luminance once, then the largest of 3 across
// (the map wraps around) and of 3 down (it stops at the poles), on all cores. The same
// numbers as the 9 luminances and their maximum taken at every call, which every sample of
// the sky and every density asked for (and env_prepare and the GPU's copy for every pixel:
// 90 ms each for a 2048 x 1024 sky on one thread).
static int make_weights(Env *e){
    const int w = e->w, h = e->h;
    free(e->wmax); free(e->row_sin);
    e->wmax = malloc(sizeof(float) * (size_t)w * h);
    e->row_sin = malloc(sizeof(real) * (size_t)h);
    float *lum = malloc(sizeof(float) * (size_t)w * h);
    if (!e->wmax || !e->row_sin || !lum){ free(lum); return 1; }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; y++){
        const float *p = e->rgb + 3 * (size_t)y * w;
        float *l = lum + (size_t)y * w, *o = e->wmax + (size_t)y * w;
        for (int x = 0; x < w; x++) l[x] = (float)lum3(p + 3 * x);
        for (int x = 0; x < w; x++){
            float a = l[x == 0 ? w - 1 : x - 1], b = l[x], c = l[x + 1 == w ? 0 : x + 1];
            float m = a > b ? a : b;
            o[x] = m > c ? m : c;
        }
        e->row_sin[y] = r_sin((y + 0.5f) * (real)PI / h);
    }
    // down: lum is free now, the rows' maxima go through it
    memcpy(lum, e->wmax, sizeof(float) * (size_t)w * h);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; y++){
        const float *a = lum + (size_t)(y > 0 ? y - 1 : y) * w, *b = lum + (size_t)y * w, *c = lum + (size_t)(y + 1 < h ? y + 1 : y) * w;
        float *o = e->wmax + (size_t)y * w;
        for (int x = 0; x < w; x++){
            float m = a[x] > b[x] ? a[x] : b[x];
            m = m > c[x] ? m : c[x];
            o[x] = m > 0 ? m : 0;                   // (as fmax(0, ...): a NaN or negative pixel weighs nothing)
        }
    }
    free(lum);
    return 0;
}

static real pixel_prob(const Env *e, int x, int y){
    if (!(e->sum > 0)) return 0;
    return (real)(pixel_weight(e, x, y) / e->sum);
}

real env_pixel_prob(const Env *e, int x, int y){ return pixel_prob(e, x, y); }

Vec3 env_sample(const Env *e, real u1, real u2, Vec3 *d, real *pdf){
    *pdf = 0;
    if (!(e->sum > 0)) return vec3(0, 0, 0);
    int y = upper_bound(e->marg, e->h, u2);
    real m0 = e->marg[y], m1 = e->marg[y + 1];
    real dv = m1 > m0 ? (u2 - m0) / (m1 - m0) : 0.5f;
    const float *c = e->cond + (size_t)y * (e->w + 1);
    int x = upper_bound(c, e->w, u1);
    real c0 = c[x], c1 = c[x + 1];
    real du = c1 > c0 ? (u1 - c0) / (c1 - c0) : 0.5f;
    if (dv > 0.9999f) dv = 0.9999f;
    if (du > 0.9999f) du = 0.9999f;
    real u = (x + du) / e->w, v = (y + dv) / e->h;
    real st = r_sin(v * (real)PI);
    if (st < 1e-6f) return vec3(0, 0, 0);
    *d = env_dir(e, u, v);
    *pdf = pixel_prob(e, x, y) * e->w * e->h / ((real)(2 * PI * PI) * st);
    return env_eval(e, *d);
}

real env_pdf(const Env *e, Vec3 d){
    if (!(e->sum > 0)) return 0;
    real u, v;
    env_uv(e, d, &u, &v);
    real st = r_sin(v * (real)PI);
    if (st < 1e-6f) return 0;
    int x = (int)(u * e->w), y = (int)(v * e->h);
    if (x > e->w - 1) x = e->w - 1;
    if (y > e->h - 1) y = e->h - 1;
    return pixel_prob(e, x, y) * e->w * e->h / ((real)(2 * PI * PI) * st);
}

// ---- the rasterizers: levels and harmonics ----

static inline real radical_inverse(unsigned i){
    i = (i << 16) | (i >> 16);
    i = ((i & 0x55555555u) << 1) | ((i & 0xAAAAAAAAu) >> 1);
    i = ((i & 0x33333333u) << 2) | ((i & 0xCCCCCCCCu) >> 2);
    i = ((i & 0x0F0F0F0Fu) << 4) | ((i & 0xF0F0F0F0u) >> 4);
    i = ((i & 0x00FF00FFu) << 8) | ((i & 0xFF00FF00u) >> 8);
    return (real)(i * 2.3283064365386963e-10);
}

// trilinear in the chain of box-filtered copies of level 0 (src[k]: (lw >> k) x (lh >> k))
static Vec3 chain_lookup(float *const *src, int nsrc, int lw, int lh, real u, real v, real lod){
    if (lod < 0) lod = 0;
    if (lod > nsrc - 1) lod = (real)(nsrc - 1);
    int k = (int)lod;
    real f = lod - k;
    Vec3 a = bilerp(src[k], lw >> k, lh >> k, u, v);
    if (f <= 0 || k + 1 >= nsrc) return a;
    return v_lerp_(a, bilerp(src[k + 1], lw >> (k + 1), lh >> (k + 1), u, v), f);
}

#define PREFILTER_SAMPLES 128

// level k: for every texel's direction n (= the view = the reflected ray, as split-sum
// assumes), the average of L(l) (n.l) over GGX of roughness k / (ENV_LEVELS - 1), by
// importance sampling the half vectors; each sample reads the chain at the level whose
// texels are as large as the solid angle it stands for (filtered importance sampling,
// Krivanek & Colbert 2008), so 128 samples are enough
static void prefilter_level(Env *e, float *const *src, int nsrc, int k){
    int w = e->lw[k], h = e->lh[k];
    float *out = e->level[k];
    real r = (real)k / (ENV_LEVELS - 1);
    real a = r_fmax(r * r, 1e-3f), a2 = a * a;
    real texel = (real)(2 * PI * PI) / ((real)e->lw[0] * e->lh[0]);   // a level-0 texel's solid angle (at the equator)
    // What a sample's half vector, weight and level are depends on the level alone, not on
    // the texel: made once here (the same values as made per texel: the same expressions),
    // so a texel's 128 samples call no sin, cos or log2 (docs/PERF_REVIEW.md, N4)
    real s_ch[PREFILTER_SAMPLES], s_tc[PREFILTER_SAMPLES], s_bs[PREFILTER_SAMPLES], s_lod[PREFILTER_SAMPLES];
    for (int i = 0; i < PREFILTER_SAMPLES; i++){
        real e1 = (i + 0.5f) / PREFILTER_SAMPLES, e2 = radical_inverse((unsigned)i);
        real ch = r_sqrt((1 - e1) / (1 + (a2 - 1) * e1)), sh = r_sqrt(r_fmax(1 - ch * ch, 0));
        real ph = (real)(2 * PI) * e2;
        real dd = ch * ch * (a2 - 1) + 1;
        real D = a2 / ((real)PI * dd * dd);
        real pdf = D * 0.25f;                       // D (n.h) / (4 v.h), v = n
        s_ch[i] = ch;
        s_tc[i] = sh * r_cos(ph);
        s_bs[i] = sh * r_sin(ph);
        s_lod[i] = 0.5f * r_log2(1 / (PREFILTER_SAMPLES * pdf * texel));
    }
    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            Vec3 n = map_dir((x + 0.5f) / w, (y + 0.5f) / h);
            Vec3 t = r_fabs(n.y) < 0.999f ? v_norm(v_cross(vec3(0, 1, 0), n)) : vec3(1, 0, 0);
            Vec3 b = v_cross(n, t);
            Vec3 sum = vec3(0, 0, 0);
            real wsum = 0;
            for (int i = 0; i < PREFILTER_SAMPLES; i++){
                const real ch = s_ch[i];
                Vec3 hv = v_add(v_scale(n, ch), v_add(v_scale(t, s_tc[i]), v_scale(b, s_bs[i])));
                Vec3 l = v_sub(v_scale(hv, 2 * ch), n);
                real nl = v_dot(n, l);
                if (nl <= 0) continue;
                real lu, lv;
                map_uv(l, &lu, &lv);       // (map_uv_fast: 1% faster still, the images no longer the same)
                sum = v_add(sum, v_scale(chain_lookup(src, nsrc, e->lw[0], e->lh[0], lu, lv, s_lod[i]), nl));
                wsum += nl;
            }
            if (wsum > 0) sum = v_scale(sum, 1 / wsum);
            float *o = out + 3 * ((size_t)y * w + x);
            o[0] = (float)sum.x; o[1] = (float)sum.y; o[2] = (float)sum.z;
        }
    }
}

void env_sh_basis(Vec3 d, real *y){
    y[0] = 0.282095f;
    y[1] = 0.488603f * d.y;
    y[2] = 0.488603f * d.z;
    y[3] = 0.488603f * d.x;
    y[4] = 1.092548f * d.x * d.y;
    y[5] = 1.092548f * d.y * d.z;
    y[6] = 0.315392f * (3 * d.z * d.z - 1);
    y[7] = 1.092548f * d.x * d.z;
    y[8] = 0.546274f * (d.x * d.x - d.y * d.y);
}

// the harmonics of level 0, and the irradiance's (Ramamoorthi & Hanrahan 2001: the cosine
// lobe keeps band l times pi, 2 pi / 3, pi / 4), stored divided by pi
static void make_sh(Env *e){
    double c[9][3];
    memset(c, 0, sizeof(c));
    double mean[3] = {0, 0, 0}, area = 0;
    int w = e->lw[0], h = e->lh[0];
    for (int y = 0; y < h; y++){
        real v = (y + 0.5f) / h;
        real dw = (real)(2 * PI * PI) / ((real)w * h) * r_sin(v * (real)PI);
        for (int x = 0; x < w; x++){
            Vec3 d = map_dir((x + 0.5f) / w, v);
            const float *p = e->level[0] + 3 * ((size_t)y * w + x);
            real yb[9];
            env_sh_basis(d, yb);
            for (int i = 0; i < 9; i++)
                for (int k = 0; k < 3; k++) c[i][k] += p[k] * yb[i] * dw;
            for (int k = 0; k < 3; k++) mean[k] += p[k] * dw;
            area += dw;
        }
    }
    static const double band[9] = {1, 2.0 / 3, 2.0 / 3, 2.0 / 3, 0.25, 0.25, 0.25, 0.25, 0.25};   // A_l / pi
    for (int i = 0; i < 9; i++)
        for (int k = 0; k < 3; k++) e->sh[i][k] = (float)(c[i][k] * band[i]);
    e->mean = vec3((real)(mean[0] / area), (real)(mean[1] / area), (real)(mean[2] / area));
}

int env_prepare(Env *e){
    static unsigned next_id;
    e->id = ++next_id;
    int w = e->w, h = e->h;
    // sampling tables
    free(e->marg); free(e->cond);
    e->marg = malloc(sizeof(float) * (size_t)(h + 1));
    e->cond = malloc(sizeof(float) * (size_t)h * (w + 1));
    double *rows = malloc(sizeof(double) * (size_t)h);
    if (!e->marg || !e->cond || !rows || make_weights(e)){ free(rows); return 1; }
    e->sum = 0;
    for (int y = 0; y < h; y++){
        float *c = e->cond + (size_t)y * (w + 1);
        double acc = 0;
        c[0] = 0;
        for (int x = 0; x < w; x++){
            acc += pixel_weight(e, x, y);
            c[x + 1] = (float)acc;
        }
        rows[y] = acc;
        for (int x = 1; x <= w; x++) c[x] = acc > 0 ? (float)(c[x] / acc) : (float)x / w;
        c[w] = 1;
        e->sum += acc;
    }
    double acc = 0;
    e->marg[0] = 0;
    for (int y = 0; y < h; y++){ acc += rows[y]; e->marg[y + 1] = e->sum > 0 ? (float)(acc / e->sum) : (float)(y + 1) / h; }
    e->marg[h] = 1;
    free(rows);

    // level 0: the map at most ENV_LEVEL0_MAX wide (and at least ENV_LEVEL_MIN), box-filtered;
    // the others half the previous one, down to ENV_LEVEL_MIN
    int l0 = w < ENV_LEVEL0_MAX ? w : ENV_LEVEL0_MAX;
    l0 &= ~3;
    if (l0 < ENV_LEVEL_MIN) l0 = ENV_LEVEL_MIN;
    for (int k = 0; k < ENV_LEVELS; k++){
        int lw = k ? e->lw[k - 1] / 2 : l0;
        if (lw < ENV_LEVEL_MIN) lw = k ? e->lw[k - 1] : ENV_LEVEL_MIN;
        e->lw[k] = lw;
        e->lh[k] = lw / 2;
        free(e->level[k]);
        e->level[k] = malloc(sizeof(float) * 3 * (size_t)e->lw[k] * e->lh[k]);
        if (!e->level[k]) return 1;
    }
    const int LW = e->lw[0], LH = e->lh[0];
    int ns = (w + LW - 1) / LW;                       // samples per side of a level-0 texel
    if (ns < 1) ns = 1;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < LH; y++)
        for (int x = 0; x < LW; x++){
            Vec3 s = vec3(0, 0, 0);
            for (int j = 0; j < ns; j++)
                for (int i = 0; i < ns; i++)
                    s = v_add(s, bilerp(e->rgb, w, h, (x + (i + 0.5f) / ns) / LW, (y + (j + 0.5f) / ns) / LH));
            float *o = e->level[0] + 3 * ((size_t)y * LW + x);
            o[0] = (float)(s.x / (ns * ns)); o[1] = (float)(s.y / (ns * ns)); o[2] = (float)(s.z / (ns * ns));
        }
    // the chain of 2x2 averages under level 0, for the prefiltering's lookups
    float *src[16];
    int nsrc = 1;
    src[0] = e->level[0];
    while (nsrc < 16 && (LW >> nsrc) >= 4){
        int sw = LW >> nsrc, sh = LH >> nsrc, pw = sw * 2;
        float *s = malloc(sizeof(float) * 3 * (size_t)sw * sh), *p = src[nsrc - 1];
        if (!s) break;
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < sw; x++)
                for (int k = 0; k < 3; k++)
                    s[3 * (y * sw + x) + k] = 0.25f * (p[3 * ((2 * y) * pw + 2 * x) + k] + p[3 * ((2 * y) * pw + 2 * x + 1) + k] +
                                                       p[3 * ((2 * y + 1) * pw + 2 * x) + k] + p[3 * ((2 * y + 1) * pw + 2 * x + 1) + k]);
        src[nsrc++] = s;
    }
    for (int k = 1; k < ENV_LEVELS; k++) prefilter_level(e, src, nsrc, k);
    for (int k = 1; k < nsrc; k++) free(src[k]);
    make_sh(e);
    return 0;
}

Env *env_from_rgb(float *rgb, int w, int h){
    Env *e = calloc(1, sizeof(*e));
    if (!e){ free(rgb); return NULL; }
    e->w = w; e->h = h; e->rgb = rgb; e->scale = 1;
    env_set_rot(e, 0);
    if (env_prepare(e)){ env_free(e); return NULL; }
    return e;
}

void env_free(Env *e){
    if (!e) return;
    free(e->rgb); free(e->marg); free(e->cond); free(e->wmax); free(e->row_sin);
    for (int k = 0; k < ENV_LEVELS; k++) free(e->level[k]);
    free(e);
}

Env *env_load_hdr(const char *path){
    int w, h;
    float *img = hdr_load(path, &w, &h);
    if (!img) return NULL;
    if (w != 2 * h){                                  // not 2:1: resampled to 2 h x h
        int nw = 2 * h;
        float *r = malloc(sizeof(float) * 3 * (size_t)nw * h);
        if (!r){ free(img); return NULL; }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < nw; x++){
                Vec3 c = bilerp(img, w, h, (x + 0.5f) / nw, (y + 0.5f) / h);
                float *o = r + 3 * ((size_t)y * nw + x);
                o[0] = (float)c.x; o[1] = (float)c.y; o[2] = (float)c.z;
            }
        free(img);
        img = r;
        w = nw;
    }
    return env_from_rgb(img, w, h);
}

Env *env_procedural_sky(int w, real sun_elev, real sun_azim){
    int h = w / 2;
    float *img = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (!img) return NULL;
    real el = sun_elev * (real)PI / 180, az = sun_azim * (real)PI / 180;
    Vec3 sun = vec3(r_cos(el) * r_sin(az), r_sin(el), r_cos(el) * r_cos(az));
    const real sun_cos = r_cos(1.0f * (real)PI / 180);          // a sun 1 degree in radius
    const Vec3 sun_le = vec3(2600, 2400, 2100);                  // ~2.5x the sky's irradiance
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            Vec3 d = map_dir((x + 0.5f) / w, (y + 0.5f) / h);
            Vec3 c;
            if (d.y >= 0){
                real t = 1 - d.y, t4 = t * t * t * t;
                Vec3 zenith = vec3(0.18f, 0.32f, 0.75f), horizon = vec3(0.75f, 0.8f, 0.85f);
                c = v_add(v_scale(zenith, 1 - t4), v_scale(horizon, t4));
                real cs = v_dot(d, sun);
                c = v_add(c, v_scale(vec3(1.0f, 0.9f, 0.7f), 0.6f * r_pow(r_fmax(cs, 0), 32)));   // the glow around the sun
                if (cs > sun_cos) c = v_add(c, sun_le);
            } else {
                real t = r_fmin(-d.y * 4, 1);
                c = v_add(v_scale(vec3(0.7f, 0.72f, 0.75f), 1 - t), v_scale(vec3(0.12f, 0.11f, 0.1f), t));   // the ground, far away
            }
            float *o = img + 3 * ((size_t)y * w + x);
            o[0] = (float)c.x; o[1] = (float)c.y; o[2] = (float)c.z;
        }
    return env_from_rgb(img, w, h);
}

Env *env_constant(Vec3 c){
    int w = 64, h = 32;
    float *img = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (!img) return NULL;
    for (int i = 0; i < w * h; i++){ img[3 * i] = (float)c.x; img[3 * i + 1] = (float)c.y; img[3 * i + 2] = (float)c.z; }
    return env_from_rgb(img, w, h);
}

Vec3 env_level(const Env *e, Vec3 d, real rough){
    real u, v;
    if (e->rot != 0) d = vec3(d.x * e->rot_c - d.z * e->rot_s, d.y, d.x * e->rot_s + d.z * e->rot_c);
    map_uv_fast(d, &u, &v);
    real lod = bsdf_clamp01_(rough) * (ENV_LEVELS - 1);
    int k = (int)lod;
    if (k > ENV_LEVELS - 2) k = ENV_LEVELS - 2;
    real f = lod - k;
    if (f <= 0) return v_scale(bilerp(e->level[k], e->lw[k], e->lh[k], u, v), e->scale);
    if (f >= 1) return v_scale(bilerp(e->level[k + 1], e->lw[k + 1], e->lh[k + 1], u, v), e->scale);
    Vec3 a = bilerp(e->level[k], e->lw[k], e->lh[k], u, v);
    Vec3 b = bilerp(e->level[k + 1], e->lw[k + 1], e->lh[k + 1], u, v);
    return v_scale(v_lerp_(a, b, f), e->scale);
}

Vec3 env_irradiance(const Env *e, Vec3 n){
    if (e->rot != 0) n = vec3(n.x * e->rot_c - n.z * e->rot_s, n.y, n.x * e->rot_s + n.z * e->rot_c);
    real y[9];
    env_sh_basis(n, y);
    real c[3] = {0, 0, 0};
    for (int i = 0; i < 9; i++)
        for (int k = 0; k < 3; k++) c[k] += e->sh[i][k] * y[i];
    return v_scale(vec3(r_fmax(c[0], 0), r_fmax(c[1], 0), r_fmax(c[2], 0)), e->scale);
}

Vec3 env_box_dir(const Env *e, Vec3 p, Vec3 d){
    if (!e->boxed) return d;
    real t = 1e30f;
    real pc[3] = {p.x, p.y, p.z}, dc[3] = {d.x, d.y, d.z};
    real lo[3] = {e->bmin.x, e->bmin.y, e->bmin.z}, hi[3] = {e->bmax.x, e->bmax.y, e->bmax.z};
    for (int k = 0; k < 3; k++){
        if (dc[k] > 1e-8f) t = r_fmin(t, (hi[k] - pc[k]) / dc[k]);
        else if (dc[k] < -1e-8f) t = r_fmin(t, (lo[k] - pc[k]) / dc[k]);
    }
    if (!(t > 0) || t >= 1e30f) return d;
    Vec3 q = v_sub(v_add(p, v_scale(d, t)), e->center);
    real l = v_len(q);
    return l > 1e-6f ? v_scale(q, 1 / l) : d;
}

// ---- the irradiance grid ----

void irr_grid_free(IrrGrid *g){
    if (!g) return;
    free(g->sh);
    free(g);
}

int irr_grid_blend(const IrrGrid *g, Vec3 p, float out[27]){
    // p in cell-centre units: probe i at (i + 0.5) / n of the box. Its 8 probes (where a
    // side has one probe, the same one twice with weight 0) blended first, valid ones only
    // (the 28th float: 1 or 0), then evaluated once for n. No branches: the points of a
    // frame come in any order
    real lo[3] = {g->lo.x, g->lo.y, g->lo.z}, hi[3] = {g->hi.x, g->hi.y, g->hi.z};
    real pc[3] = {p.x, p.y, p.z};
    float wk[3][2];
    size_t at = 0, stride[3], st = IRR_FLOATS;
    for (int k = 0; k < 3; k++){
        int nk = g->n[k];
        real x = (pc[k] - lo[k]) / (hi[k] - lo[k]) * nk - 0.5f;
        x = r_fmin(r_fmax(x, 0), (real)(nk - 1));
        int i0 = (int)x;
        i0 = i0 > nk - 2 ? (nk - 2 < 0 ? 0 : nk - 2) : i0;
        real f = x - i0;
        int two = i0 + 1 < nk;
        wk[k][0] = (float)(1 - f); wk[k][1] = two ? (float)f : 0;
        at += st * (size_t)i0;
        stride[k] = two ? st : 0;
        st *= (size_t)nk;
    }
    const float *base = g->sh + at;
    float acc[IRR_FLOATS];
#if defined(__GNUC__)
    // 4 at a time (GCC's and clang's vector types: SSE on x86, NEON on ARM, WASM's SIMD or
    // plain code otherwise; f4u read where a float may start)
    typedef float f4 __attribute__((vector_size(16)));
    typedef float f4u __attribute__((vector_size(16), aligned(4)));
    f4 a4[IRR_FLOATS / 4];
    for (int i = 0; i < IRR_FLOATS / 4; i++) a4[i] = (f4){0, 0, 0, 0};
    for (int c = 0; c < 8; c++){
        const float *q = base + (c & 1) * stride[0] + ((c >> 1) & 1) * stride[1] + (c >> 2) * stride[2];
        float w = wk[0][c & 1] * wk[1][(c >> 1) & 1] * wk[2][c >> 2] * q[27];
        f4 w4 = {w, w, w, w};
        const f4u *q4 = (const f4u *)q;
        for (int i = 0; i < IRR_FLOATS / 4; i++) a4[i] += w4 * q4[i];
    }
    memcpy(acc, a4, sizeof(acc));
#else
    for (int i = 0; i < IRR_FLOATS; i++) acc[i] = 0;
    for (int c = 0; c < 8; c++){
        const float *q = base + (c & 1) * stride[0] + ((c >> 1) & 1) * stride[1] + (c >> 2) * stride[2];
        float w = wk[0][c & 1] * wk[1][(c >> 1) & 1] * wk[2][c >> 2] * q[27];
        for (int i = 0; i < IRR_FLOATS; i++) acc[i] += w * q[i];
    }
#endif
    real wsum = acc[27];
    if (!(wsum > 1e-4f)) return 0;
    float iw = (float)(1 / wsum);
    for (int i = 0; i < 27; i++) out[i] = acc[i] * iw;
    return 1;
}

Vec3 irr_sh_eval(const float c[27], Vec3 n){
    real y[9];
    env_sh_basis(n, y);
    real r[3] = {0, 0, 0};
    for (int i = 0; i < 9; i++)
        for (int k = 0; k < 3; k++) r[k] += c[3 * i + k] * y[i];
    return vec3(r_fmax(r[0], 0), r_fmax(r[1], 0), r_fmax(r[2], 0));
}

Vec3 irr_grid_irradiance(const IrrGrid *g, Vec3 p, Vec3 n, int *ok){
    float c[27];
    *ok = irr_grid_blend(g, p, c);
    return *ok ? irr_sh_eval(c, n) : vec3(0, 0, 0);
}
