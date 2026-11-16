#include "core/texture.h"

#include "core/image.h"   // rgb8
#include "core/png.h"
#include "core/jpeg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Texture *texture_load_png(const char *path){
    int w, h, ch;
    unsigned char *px = png_read(path, &w, &h, &ch, 3);
    if (!px) return NULL;
    Texture *t = malloc(sizeof(*t));
    if (t) t->px = malloc((size_t)w * h * sizeof(uint32_t));
    if (!t || !t->px){ free(t); free(px); return NULL; }
    t->w = w;
    t->h = h;
    for (size_t i = 0; i < (size_t)w * h; i++) t->px[i] = rgb8(px[3 * i], px[3 * i + 1], px[3 * i + 2]);
    free(px);
    return t;
}

Texture *texture_checker(int size, int cells){
    Texture *t = malloc(sizeof(*t));
    if (t) t->px = malloc((size_t)size * size * sizeof(uint32_t));
    if (!t || !t->px){ free(t); return NULL; }
    t->w = t->h = size;
    for (int y = 0; y < size; y++){
        for (int x = 0; x < size; x++){
            int on = ((x * cells / size) + (y * cells / size)) & 1;
            t->px[y * size + x] = on ? rgb8(235, 235, 225) : rgb8(200, 60, 50);
        }
    }
    return t;
}

void texture_destroy(Texture *t){
    if (!t) return;
    free(t->px);
    free(t);
}

Vec3 texture_sample(const Texture *t, real u, real v){
    u -= r_floor(u);                       // wrap into [0, 1)
    v -= r_floor(v);
    int x = (int)(u * t->w);
    int y = (int)((1.0f - v) * t->h);     // image row 0 is the top, v = 0 is the bottom
    if (x >= t->w) x = t->w - 1;
    if (y >= t->h) y = t->h - 1;
    if (y < 0) y = 0;
    uint32_t c = t->px[y * t->w + x];
    return vec3(((c >> 16) & 255) / 255.0f, ((c >> 8) & 255) / 255.0f, (c & 255) / 255.0f);
}

// ---- the materials' textures ------------------------------------------------------------

static float SRGB[256];
static int srgb_ready;

const float *texmap_srgb_table(void){
    if (!srgb_ready)
#ifdef _OPENMP
#pragma omp critical(texmap_srgb)
#endif
    if (!srgb_ready){
        for (int i = 0; i < 256; i++){
            double c = i / 255.0;
            SRGB[i] = (float)(c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4));
        }
#ifdef _OPENMP
#pragma omp flush
#endif
        srgb_ready = 1;
    }
    return SRGB;
}

static unsigned char to_srgb8(double c){
    c = c < 0 ? 0 : c > 1 ? 1 : c;
    c = c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055;
    return (unsigned char)(c * 255 + 0.5);
}

// the linear values where the sRGB byte steps up: byte(c) = how many of them c reaches (the
// same bytes as to_srgb8 without a pow per texel: making the helmet's mips spent most of its
// time there)
static double SRGB_UP[255];
static int srgb_up_ready;

static void srgb_up_init(void){
    if (srgb_up_ready) return;
#ifdef _OPENMP
#pragma omp critical(texmap_srgb_up)
#endif
    if (!srgb_up_ready){
        for (int i = 0; i < 255; i++){
            double e = (i + 0.5) / 255;
            SRGB_UP[i] = e <= 0.0031308 * 12.92 ? e / 12.92 : pow((e + 0.055) / 1.055, 2.4);
        }
#ifdef _OPENMP
#pragma omp flush
#endif
        srgb_up_ready = 1;
    }
}

static unsigned srgb8_fast(double c){
    unsigned lo = 0, hi = 255;          // the answer is in [lo, hi]
    while (lo < hi){
        unsigned mid = (lo + hi) / 2;   // byte > mid when c reaches SRGB_UP[mid]
        if (c >= SRGB_UP[mid]) lo = mid + 1; else hi = mid;
    }
    return lo;
}

int texmap_make(TexMap *t, const unsigned char *rgba, int w, int h, int srgb){
    memset(t, 0, sizeof(*t));
    if (w <= 0 || h <= 0) return 1;
    const float *lut = texmap_srgb_table();
    srgb_up_init();
    t->w = w; t->h = h; t->srgb = srgb;
    size_t total = 0;
    int lw = w, lh = h, n = 0;
    while (n < TEXMAP_LEVELS){
        t->lw[n] = lw; t->lh[n] = lh; t->at[n] = total;
        total += (size_t)lw * lh;
        n++;
        if (lw == 1 && lh == 1) break;
        lw = lw > 1 ? lw / 2 : 1;
        lh = lh > 1 ? lh / 2 : 1;
    }
    t->levels = n;
    t->px = malloc(total * 4);
    if (!t->px) return 1;
    for (size_t i = 0; i < (size_t)w * h; i++)
        t->px[i] = (uint32_t)rgba[4 * i] | (uint32_t)rgba[4 * i + 1] << 8 | (uint32_t)rgba[4 * i + 2] << 16 | (uint32_t)rgba[4 * i + 3] << 24;
    // each level: 2 x 2 averages of the one above (a side of 1, or odd: its last texel twice)
    for (int k = 1; k < n; k++){
        const uint32_t *src = t->px + t->at[k - 1];
        uint32_t *dst = t->px + t->at[k];
        int sw = t->lw[k - 1], sh = t->lh[k - 1];
        for (int y = 0; y < t->lh[k]; y++)
            for (int x = 0; x < t->lw[k]; x++){
                int x0 = 2 * x < sw ? 2 * x : sw - 1, x1 = 2 * x + 1 < sw ? 2 * x + 1 : sw - 1;
                int y0 = 2 * y < sh ? 2 * y : sh - 1, y1 = 2 * y + 1 < sh ? 2 * y + 1 : sh - 1;
                uint32_t c[4] = {src[y0 * sw + x0], src[y0 * sw + x1], src[y1 * sw + x0], src[y1 * sw + x1]};
                uint32_t out = 0;
                for (int ch = 0; ch < 4; ch++){
                    unsigned v;
                    if (srgb && ch < 3){
                        double s = 0;
                        for (int i = 0; i < 4; i++) s += lut[(c[i] >> (8 * ch)) & 255];
                        v = srgb8_fast(s * 0.25);
                    } else {                    // linear: the integer average, rounded
                        unsigned s = 0;
                        for (int i = 0; i < 4; i++) s += (c[i] >> (8 * ch)) & 255;
                        v = (s + 2) >> 2;
                    }
                    out |= (uint32_t)v << (8 * ch);
                }
                dst[y * t->lw[k] + x] = out;
            }
    }
    return 0;
}

int texmap_load_png(TexMap *t, const char *path, int srgb){
    int w, h, ch;
    unsigned char *px = png_read(path, &w, &h, &ch, 4);
    if (!px){ fprintf(stderr, "texture: cannot read %s\n", path); return 1; }
    int r = texmap_make(t, px, w, h, srgb);
    free(px);
    if (!r) snprintf(t->name, sizeof(t->name), "%s", path);
    return r;
}

int texmap_decode(TexMap *t, const unsigned char *data, size_t size, int srgb){
    int w, h, ch;
    unsigned char *px = NULL;
    if (size >= 8 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') px = png_decode(data, size, &w, &h, &ch, 4);
    else if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8) px = jpeg_decode(data, size, &w, &h, &ch, 4);
    else fprintf(stderr, "texture: neither a PNG nor a JPEG image\n");
    if (!px) return 1;
    int r = texmap_make(t, px, w, h, srgb);
    free(px);
    return r;
}

int texmap_load(TexMap *t, const char *path, int srgb){
    FILE *f = fopen(path, "rb");
    if (!f){ fprintf(stderr, "texture: cannot open %s\n", path); return 1; }
    size_t cap = 1 << 16, n = 0, k;
    unsigned char *buf = malloc(cap);
    while (buf && (k = fread(buf + n, 1, cap - n, f)) > 0){
        n += k;
        if (n == cap){
            unsigned char *q = realloc(buf, cap * 2);
            if (!q){ free(buf); buf = NULL; break; }
            buf = q; cap *= 2;
        }
    }
    fclose(f);
    if (!buf) return 1;
    int r = texmap_decode(t, buf, n, srgb);
    free(buf);
    if (r) fprintf(stderr, "texture: cannot read %s\n", path);
    else snprintf(t->name, sizeof(t->name), "%s", path);
    return r;
}

void texmap_free(TexMap *t){
    free(t->px);
    t->px = NULL;
}

static inline int wrap_i(int x, int n){
    x %= n;
    return x < 0 ? x + n : x;
}

Vec4 texmap_texel(const TexMap *t, int level, int x, int y){
    int w = t->lw[level], h = t->lh[level];
    uint32_t c = t->px[t->at[level] + (size_t)wrap_i(y, h) * w + wrap_i(x, w)];
    Vec4 r;
    if (t->srgb){
        const float *lut = srgb_ready ? SRGB : texmap_srgb_table();
        r.x = lut[c & 255]; r.y = lut[(c >> 8) & 255]; r.z = lut[(c >> 16) & 255];
    } else {
        r.x = (c & 255) * (1.0f / 255); r.y = ((c >> 8) & 255) * (1.0f / 255); r.z = ((c >> 16) & 255) * (1.0f / 255);
    }
    r.w = (c >> 24) * (1.0f / 255);
    return r;
}

static inline Vec4 v4_lerp(Vec4 a, Vec4 b, real f){
    return (Vec4){a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f, a.w + (b.w - a.w) * f};
}

static inline Vec4 decode(uint32_t c, const float *lut){
    Vec4 r;
    if (lut){ r.x = lut[c & 255]; r.y = lut[(c >> 8) & 255]; r.z = lut[(c >> 16) & 255]; }
    else { r.x = (c & 255) * (1.0f / 255); r.y = ((c >> 8) & 255) * (1.0f / 255); r.z = ((c >> 16) & 255) * (1.0f / 255); }
    r.w = (c >> 24) * (1.0f / 255);
    return r;
}

// a coordinate into 0..1 by the wrap mode (repeat: the fraction; clamp; mirror: every other
// period turned over)
static inline real fold(real u, int mode){
    if (mode == TEXMAP_CLAMP) return u < 0 ? 0 : u > 1 ? 1 : u;
    if (mode == TEXMAP_MIRROR){
        real m = u - 2 * r_floor(u * 0.5f);
        return m > 1 ? 2 - m : m;
    }
    return u >= 0 && u < 1 ? u : u - r_floor(u);
}
// texel x (-1 .. n) of a row of n: around for repeat, else held at the edge
static inline int edge(int x, int n, int mode){
    if (mode == TEXMAP_REPEAT) return x < 0 ? x + n : x >= n ? x - n : x;
    return x < 0 ? 0 : x >= n ? n - 1 : x;
}

// (u, v folded into 0..1 first: the 4 texels' rows and columns found without a division)
Vec4 texmap_bilinear(const TexMap *t, int level, real u, real v){
    u = fold(u, t->wrap_u);
    v = fold(v, t->wrap_v);
    int w = t->lw[level], h = t->lh[level];
    real x = u * w - 0.5f, y = v * h - 0.5f;
    real fx0 = r_floor(x), fy0 = r_floor(y);
    int x0 = (int)fx0, y0 = (int)fy0;                  // -1 .. w (u = 1 when clamped)
    real fx = x - fx0, fy = y - fy0;
    if (x0 < -1) x0 = -1;                              // (rounding, NaN)
    if (x0 > w - 1) x0 = w - 1;
    if (y0 < -1) y0 = -1;
    if (y0 > h - 1) y0 = h - 1;
    int xa = edge(x0, w, t->wrap_u), xb = edge(x0 + 1, w, t->wrap_u);
    int ya = edge(y0, h, t->wrap_v), yb = edge(y0 + 1, h, t->wrap_v);
    const uint32_t *px = t->px + t->at[level];
    const float *lut = t->srgb ? (srgb_ready ? SRGB : texmap_srgb_table()) : NULL;
    Vec4 a = decode(px[(size_t)ya * w + xa], lut), b = decode(px[(size_t)ya * w + xb], lut);
    Vec4 c = decode(px[(size_t)yb * w + xa], lut), d = decode(px[(size_t)yb * w + xb], lut);
    return v4_lerp(v4_lerp(a, b, fx), v4_lerp(c, d, fx), fy);
}

Vec4 texmap_sample(const TexMap *t, real u, real v, real lod){
    if (!(u > -1e6f && u < 1e6f && v > -1e6f && v < 1e6f)) u = v = 0;    // (NaN, inf: far too big)
    if (!(lod > 0)) return texmap_bilinear(t, 0, u, v);
    if (lod >= t->levels - 1) return texmap_bilinear(t, t->levels - 1, u, v);
    int k = (int)lod;
    real f = lod - k;
    Vec4 a = texmap_bilinear(t, k, u, v);
    return f > 0 ? v4_lerp(a, texmap_bilinear(t, k + 1, u, v), f) : a;
}

real texmap_lod(const TexMap *t, real dudx, real dvdx, real dudy, real dvdy){
    real ax = dudx * t->w, ay = dvdx * t->h, bx = dudy * t->w, by = dvdy * t->h;
    real m = r_fmax(ax * ax + ay * ay, bx * bx + by * by);
    return m > 1 ? 0.5f * r_log2(m) : 0;
}

// ---- procedural textures ----

static uint32_t hash2(int x, int y){
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)y * 0xd8163841u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
static double hash01(int x, int y){ return (hash2(x, y) & 0xffffff) / 16777216.0; }
// smooth value noise in [0, 1] at (x, y) (period p, so the texture repeats)
static double vnoise(double x, double y, int p){
    int x0 = (int)floor(x), y0 = (int)floor(y);
    double fx = x - x0, fy = y - y0;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    double a = hash01(wrap_i(x0, p), wrap_i(y0, p)), b = hash01(wrap_i(x0 + 1, p), wrap_i(y0, p));
    double c = hash01(wrap_i(x0, p), wrap_i(y0 + 1, p)), d = hash01(wrap_i(x0 + 1, p), wrap_i(y0 + 1, p));
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}
static double fbm(double x, double y, int p){
    double s = 0, amp = 0.5;
    for (int o = 0; o < 4; o++){ s += amp * vnoise(x, y, p); x *= 2; y *= 2; p *= 2; amp *= 0.5; }
    return s;
}

// The bricks: 8 rows of 4 bricks (every other row shifted by half), mortar 1/32 of a row
// wide. height(): 1 on a brick (its edges bevelled), 0 in the mortar; brick(): which one.
#define BR_ROWS 8
#define BR_COLS 4
static double brick_height(double u, double v, int *bx, int *by){
    double fy = v * BR_ROWS;
    int row = (int)floor(fy);
    double fx = u * BR_COLS + (row & 1 ? 0.5 : 0);
    int col = (int)floor(fx);
    *bx = wrap_i(col, BR_COLS); *by = wrap_i(row, BR_ROWS);
    double dx = fx - col, dy = fy - row;
    double e = fmin(fmin(dx, 1 - dx) * BR_COLS / BR_ROWS * 2, fmin(dy, 1 - dy));   // distance to the edge, in rows
    double m = 1.0 / 32, bevel = 1.0 / 16;
    return e < m ? 0 : (e < m + bevel ? (e - m) / bevel : 1);
}

int texmap_procedural(TexMap *t, const char *name){
    const int N = 512;
    unsigned char *px = malloc((size_t)N * N * 4);
    if (!px) return 1;
    // colour (sRGB) or data; the rows in parallel (each pixel its own)
    int srgb = strcmp(name, "bricks_normal") && strcmp(name, "bricks_mr") && strcmp(name, "plate_mr");
    int known = !srgb || !strcmp(name, "checker") || !strcmp(name, "bricks");
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++){
            double u = (x + 0.5) / N, v = (y + 0.5) / N;
            double r = 0, g = 0, b = 0;                         // linear (colour) or as stored
            unsigned char *o = px + 4 * ((size_t)y * N + x);
            int bx, by;
            if (!strcmp(name, "checker")){
                int on = ((int)(u * 8) + (int)(v * 8)) & 1;
                r = on ? 0.82 : 0.6; g = on ? 0.82 : 0.07; b = on ? 0.75 : 0.05;
            } else if (!strcmp(name, "bricks")){
                double h = brick_height(u, v, &bx, &by);
                double tone = 0.75 + 0.5 * hash01(bx, by), grain = 0.8 + 0.4 * fbm(u * 32, v * 32, 32);
                double br = 0.45 * tone * grain, bg = 0.13 * tone * grain, bb = 0.07 * tone * grain;
                double mo = 0.45 + 0.1 * fbm(u * 64, v * 64, 64);
                double k = h > 0 ? 1 : 0;
                r = k * br + (1 - k) * mo; g = k * bg + (1 - k) * mo; b = k * bb + (1 - k) * mo * 0.95;
            } else if (!strcmp(name, "bricks_normal")){
                // the slope of the height (bricks 6 units above the mortar's floor, a little rough),
                // by central differences; x along u, y up (towards -v), z out of the surface
                double eps = 1.0 / N, scale = 0.012;
                double hx = (brick_height(u + eps, v, &bx, &by) - brick_height(u - eps, v, &bx, &by)) / (2 * eps);
                double hy = (brick_height(u, v - eps, &bx, &by) - brick_height(u, v + eps, &bx, &by)) / (2 * eps);
                hx += 4 * (fbm((u + eps) * 32, v * 32, 32) - fbm((u - eps) * 32, v * 32, 32)) / (2 * eps) * 0.02;
                hy += 4 * (fbm(u * 32, (v - eps) * 32, 32) - fbm(u * 32, (v + eps) * 32, 32)) / (2 * eps) * 0.02;
                double nx = -hx * scale, ny = -hy * scale, nz = 1, l = sqrt(nx * nx + ny * ny + nz * nz);
                r = 0.5 + 0.5 * nx / l; g = 0.5 + 0.5 * ny / l; b = 0.5 + 0.5 * nz / l;
            } else if (!strcmp(name, "bricks_mr")){
                double h = brick_height(u, v, &bx, &by);
                g = h > 0 ? 0.55 + 0.25 * fbm(u * 32, v * 32, 32) : 0.95;    // roughness
                b = 0;                                                     // metallic
            } else if (!strcmp(name, "plate_mr")){
                double wear = fbm(u * 8, v * 8, 8);
                double scratch = fbm(u * 64, v * 4, 64);
                g = 0.15 + 0.5 * wear * wear + 0.15 * scratch;               // roughness
                b = wear > 0.55 ? 0.15 : 1.0;                                 // metal, but where it wore off
            }
            if (srgb){ o[0] = to_srgb8(r); o[1] = to_srgb8(g); o[2] = to_srgb8(b); }
            else { o[0] = (unsigned char)(r * 255 + 0.5); o[1] = (unsigned char)(g * 255 + 0.5); o[2] = (unsigned char)(b * 255 + 0.5); }
            o[3] = 255;
        }
    int res = known ? texmap_make(t, px, N, N, srgb) : 1;
    free(px);
    if (!known) fprintf(stderr, "texture: no procedural texture called %s\n", name);
    if (!res) snprintf(t->name, sizeof(t->name), "%s", name);
    return res;
}
