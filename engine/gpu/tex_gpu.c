// tex_gpu.c - the materials' textures on the GPU (docs/PBR.md, PBR5).
//
// Every TexMap's mip levels go into one RGBA8 texture, the atlas: each level a rectangle,
// the rectangles placed in rows (the tallest first). A table says where they are. The WGSL
// (GPU_WGSL_TEX) reads texels with textureLoad and filters by hand as core/texture.c does:
// bilinear inside a level's rectangle (wrapping around inside it, so nothing bleeds in from
// a neighbour), trilinear between levels, sRGB decoded per texel. One texture binding, so the
// path tracer (whose 8 storage buffers are taken) can read it as the rasterizer does.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const GPU_WGSL_TEX = WGSL(
/* The includer declares the atlas, var TEXA: texture_2d<f32>, and fn tex_entry(k: u32) -> vec4u,
   the table: entry t (t < 32) is texture t's first rectangle, its levels, srgb, its size
   (w | h << 16); from entry 32 on the rectangles, (x, y, w, h). */
fn tex_srgb(c: f32) -> f32 { return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045); }
/* how coordinates outside 0..1 read (TexMap.wrap_u, wrap_v): 0 repeat, 1 clamp, 2 mirror; a
   mirrored or clamped coordinate is folded into 0..1 first, its texels then clamped */
fn tex_fold(u: f32, mode: u32) -> f32 {
    if (mode == 1u) { return clamp(u, 0.0, 1.0); }
    if (mode == 2u) { let m = u - 2.0 * floor(u * 0.5); return select(m, 2.0 - m, m > 1.0); }
    return u - floor(u);
}
fn tex_index(x: i32, n: i32, mode: u32) -> i32 {
    if (mode == 0u) { return select(select(x, x - n, x >= n), x + n, x < 0); }
    return clamp(x, 0, n - 1);
}
fn tex_texel(t: u32, level: u32, x: i32, y: i32) -> vec4f {
    let e = tex_entry(t);
    let r = tex_entry(e.x + level);
    var c = textureLoad(TEXA, vec2i(i32(r.x) + x, i32(r.y) + y), 0);
    if ((e.z & 1u) != 0u) { c = vec4f(tex_srgb(c.x), tex_srgb(c.y), tex_srgb(c.z), c.w); }
    return c;
}
fn tex_bilinear(t: u32, level: u32, uv: vec2f) -> vec4f {
    let e = tex_entry(t);
    let r = tex_entry(e.x + level);
    let mu = (e.z >> 1u) & 3u;
    let mv = (e.z >> 3u) & 3u;
    let w = i32(r.z);
    let h = i32(r.w);
    let x = tex_fold(uv.x, mu) * f32(w) - 0.5;
    let y = tex_fold(uv.y, mv) * f32(h) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let xa = tex_index(i32(x0), w, mu);
    let xb = tex_index(i32(x0) + 1, w, mu);
    let ya = tex_index(i32(y0), h, mv);
    let yb = tex_index(i32(y0) + 1, h, mv);
    let a = tex_texel(t, level, xa, ya);
    let b = tex_texel(t, level, xb, ya);
    let c = tex_texel(t, level, xa, yb);
    let d = tex_texel(t, level, xb, yb);
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}
/* texture t at uv, level lod (fractional; <= 0: level 0, bilinear) */
fn tex_sample(t: u32, uv0: vec2f, lod: f32) -> vec4f {
    var uv = uv0;
    if (!(abs(uv.x) < 1e6 && abs(uv.y) < 1e6)) { uv = vec2f(0.0); }
    let levels = tex_entry(t).y;
    if (!(lod > 0.0)) { return tex_bilinear(t, 0u, uv); }
    if (lod >= f32(levels - 1u)) { return tex_bilinear(t, levels - 1u, uv); }
    let k = u32(lod);
    let f = lod - f32(k);
    let a = tex_bilinear(t, k, uv);
    if (f <= 0.0) { return a; }
    return mix(a, tex_bilinear(t, k + 1u, uv), f);
}
/* scene.c tex_at: the level for a footprint of 2^uv_lod texture coordinates (scaled by k) */
fn tex_at(t: u32, uv: vec2f, uv_lod: f32, k: f32) -> vec4f {
    var lod = -1000.0;
    if (uv_lod > -500.0) {
        let e = tex_entry(t).w;
        lod = uv_lod + log2(k * f32(max(e & 0xffffu, e >> 16u)));
    }
    return tex_sample(t, uv, lod);
}
/* The material's textures (scene.c material_at, material_normal, material_occlusion), packed
   (gpu_tex_pack): x = base | emissive << 8, y = metal-roughness | occlusion << 8 | its strength
   (0..255) << 16, z = normal | its scale (1/4096) << 8, each 1 + index (0 none); w: the uv
   scale's bits */
struct TexMat { base: vec3f, metallic: f32, rough: f32, emission: vec3f };
fn tex_scale(tex: vec4u) -> f32 { return select(1.0, bitcast<f32>(tex.w), bitcast<f32>(tex.w) > 0.0); }
fn tex_material(tex: vec4u, base: vec3f, metallic: f32, rough: f32, emission: vec3f, uv0: vec2f, uv_lod: f32) -> TexMat {
    let k = tex_scale(tex);
    let uv = uv0 * k;
    var r = TexMat(base, metallic, rough, emission);
    let tb = tex.x & 255u;
    let te = (tex.x >> 8u) & 255u;
    let tm = tex.y & 255u;
    if (tb > 0u) { r.base *= tex_at(tb - 1u, uv, uv_lod, k).rgb; }
    if (tm > 0u) {
        let c = tex_at(tm - 1u, uv, uv_lod, k);
        r.rough *= c.y;
        r.metallic *= c.z;
    }
    if (te > 0u) { r.emission *= tex_at(te - 1u, uv, uv_lod, k).rgb; }
    return r;
}
fn tex_occlusion(tex: vec4u, uv0: vec2f, uv_lod: f32) -> f32 {
    let to = (tex.y >> 8u) & 255u;
    if (to == 0u) { return 1.0; }
    let k = tex_scale(tex);
    let s = f32((tex.y >> 16u) & 255u) / 255.0;
    return 1.0 + s * (tex_at(to - 1u, uv0 * k, uv_lod, k).x - 1.0);
}
fn tex_normal(tex: vec4u, uv0: vec2f, uv_lod: f32, n: vec3f, t: vec4f) -> vec3f {
    let tn = tex.z & 255u;
    if (tn == 0u) { return n; }
    let k = tex_scale(tex);
    let ns = f32(tex.z >> 8u) / 4096.0;
    let c = tex_at(tn - 1u, uv0 * k, uv_lod, k);
    var tt = t.xyz - n * dot(n, t.xyz);
    let l = length(tt);
    if (!(l > 1e-6)) { return n; }
    tt /= l;
    let b = cross(n, tt) * t.w;
    let o = tt * (ns * (2.0 * c.x - 1.0)) + b * (ns * (2.0 * c.y - 1.0)) + n * (2.0 * c.z - 1.0);
    let lo = length(o);
    return select(n, o / lo, lo > 1e-6);
}
/* a direction as 2 numbers (octahedral) and back: the rasterizer's tangents per vertex */
fn oct_decode(e: vec2f) -> vec3f {
    var v = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        let s = vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        let xy = (vec2f(1.0) - abs(v.yx)) * s;
        v = vec3f(xy.x, xy.y, v.z);
    }
    return normalize(v);
}
);

void gpu_tex_pack(const Material *m, int ntex, uint32_t out[4]){
    #define TX(i) ((i) > 0 && (i) <= ntex ? (uint32_t)(i) : 0u)
    real os = m->occlusion > 0 ? m->occlusion : 1, ns = m->normal_scale > 0 ? m->normal_scale : 1;
    uint32_t osq = (uint32_t)(r_fmin(os, 1) * 255 + 0.5f), nsq = (uint32_t)(r_fmin(ns, 4000) * 4096 + 0.5f);
    out[0] = TX(m->tex_base) | TX(m->tex_emissive) << 8;
    out[1] = TX(m->tex_mr) | TX(m->tex_occlusion) << 8 | (TX(m->tex_occlusion) ? osq << 16 : 0u);
    out[2] = TX(m->tex_normal) ? TX(m->tex_normal) | nsq << 8 : 0u;
    float sc = (float)m->uv_scale;
    memcpy(&out[3], &sc, 4);
    #undef TX
}

void gpu_oct_encode(Vec3 d, float *out){
    real s = r_fabs(d.x) + r_fabs(d.y) + r_fabs(d.z);
    if (!(s > 0)){ out[0] = out[1] = 0; return; }
    real x = d.x / s, y = d.y / s;
    if (d.z < 0){
        real ox = (1 - r_fabs(y)) * (x >= 0 ? 1 : -1), oy = (1 - r_fabs(x)) * (y >= 0 ? 1 : -1);
        x = ox; y = oy;
    }
    out[0] = (float)x; out[1] = (float)y;
}

typedef struct { int tex, level, w, h, x, y; } Rect;
static int by_height(const void *a, const void *b){
    const Rect *p = a, *q = b;
    return q->h != p->h ? q->h - p->h : q->w - p->w;
}

static void textures_release(GpuTextures *gt){
    if (gt->view) wgpuTextureViewRelease(gt->view);
    if (gt->tex){ wgpuTextureDestroy(gt->tex); wgpuTextureRelease(gt->tex); }
    gt->view = NULL;
    gt->tex = NULL;
}

static void make_atlas(Gpu *g, GpuTextures *gt, int w, int h, const unsigned char *px){
    WGPUTextureDescriptor td;
    memset(&td, 0, sizeof(td));
    td.label = gpu_str("material textures");
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    td.dimension = WGPUTextureDimension_2D;
    td.size = (WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1};
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    gt->tex = wgpuDeviceCreateTexture(g->device, &td);
    WGPUTexelCopyTextureInfo dst;
    memset(&dst, 0, sizeof(dst));
    dst.texture = gt->tex;
    dst.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyBufferLayout lay;
    memset(&lay, 0, sizeof(lay));
    lay.bytesPerRow = (uint32_t)w * 4;
    lay.rowsPerImage = (uint32_t)h;
    WGPUExtent3D size = {(uint32_t)w, (uint32_t)h, 1};
    wgpuQueueWriteTexture(g->queue, &dst, px, (size_t)w * h * 4, &lay, &size);
    gt->view = wgpuTextureCreateView(gt->tex, NULL);
}

// rows, left to right, a new row when one does not fit: the height they take in width aw
static int pack_rows(Rect *r, int nr, int aw){
    int x = 0, y = 0, row_h = 0;
    for (int i = 0; i < nr; i++){
        if (x + r[i].w > aw){ y += row_h; x = 0; row_h = 0; }
        r[i].x = x; r[i].y = y;
        x += r[i].w;
        if (r[i].h > row_h) row_h = r[i].h;
    }
    return y + row_h;
}

int gpu_textures_sync(Gpu *g, GpuTextures *gt, const Scene *s){
    const void *key = s->ntexmaps ? (const void *)s->texmaps[0].px : NULL;
    if (gt->tex && gt->nsrc == s->ntexmaps && gt->src == (const void *)s->texmaps && gt->key == key) return 0;
    textures_release(gt);
    memset(gt->table, 0, sizeof(gt->table));
    gt->src = s->texmaps; gt->nsrc = s->ntexmaps; gt->key = key;
    gt->ntex = 0;
    int n = s->ntexmaps < GPU_TEX_MAX ? s->ntexmaps : GPU_TEX_MAX;
    if (s->ntexmaps > GPU_TEX_MAX) fprintf(stderr, "GPU: %d textures, the GPU takes the first %d\n", s->ntexmaps, GPU_TEX_MAX);
    int nr = 0;
    for (int t = 0; t < n; t++) nr += s->texmaps[t].levels;
    if (nr > GPU_TEX_TABLE - GPU_TEX_MAX){
        fprintf(stderr, "GPU: too many texture levels (%d): no textures on the GPU\n", nr);
        n = nr = 0;
    }
    Rect *r = malloc(sizeof(Rect) * (size_t)(nr ? nr : 1));
    if (!r) n = nr = 0;
    // the atlas: as narrow as fits under 8192 rows (widths 1024 .. 8192); if nothing fits, the
    // largest texture's top level is left out (the GPU reads it from the next one down), and
    // again, until it does
    int skip[GPU_TEX_MAX] = {0}, aw = 0, ah = 0;
    for (int tries = 0; nr && tries < 64; tries++){
        int k = 0, widest = 1;
        for (int t = 0; t < n; t++)
            for (int l = skip[t]; l < s->texmaps[t].levels; l++){
                r[k++] = (Rect){t, l, s->texmaps[t].lw[l], s->texmaps[t].lh[l], 0, 0};
                if (s->texmaps[t].lw[l] > widest) widest = s->texmaps[t].lw[l];
            }
        nr = k;
        qsort(r, (size_t)nr, sizeof(Rect), by_height);
        aw = 0;
        for (int w = 1024; w <= 8192 && !aw; w *= 2){
            if (w < widest) continue;
            ah = pack_rows(r, nr, w);
            if (ah <= 8192) aw = w;
        }
        if (aw) break;
        int big = 0;                                       // the texture with the largest top level
        for (int t = 1; t < n; t++)
            if ((long)s->texmaps[t].lw[skip[t]] * s->texmaps[t].lh[skip[t]] > (long)s->texmaps[big].lw[skip[big]] * s->texmaps[big].lh[skip[big]]) big = t;
        if (skip[big] + 1 >= s->texmaps[big].levels) break;
        skip[big]++;
    }
    if (nr && !aw){
        fprintf(stderr, "GPU: the textures do not fit in an atlas of 8192 x 8192: no textures on the GPU\n");
        n = nr = 0;
    }
    for (int t = 0; t < n; t++)
        if (skip[t]) fprintf(stderr, "GPU: texture %s from %d x %d (its first %d level(s) left out: the atlas is full)\n",
                             s->texmaps[t].name, s->texmaps[t].lw[skip[t]], s->texmaps[t].lh[skip[t]], skip[t]);
    if (!nr){
        unsigned char none[4] = {255, 255, 255, 255};
        make_atlas(g, gt, 1, 1, none);
        free(r);
        return 1;
    }
    unsigned char *px = calloc((size_t)aw * ah, 4);
    if (!px){ free(r); unsigned char none[4] = {255, 255, 255, 255}; make_atlas(g, gt, 1, 1, none); return 1; }
    // the table: per texture its first rectangle (in level order), then the rectangles
    int next = GPU_TEX_MAX;
    for (int t = 0; t < n; t++){
        const TexMap *tm = &s->texmaps[t];
        gt->table[t][0] = (uint32_t)next;
        gt->table[t][1] = (uint32_t)(tm->levels - skip[t]);
        gt->table[t][2] = (tm->srgb ? 1u : 0u) | (uint32_t)(tm->wrap_u & 3) << 1 | (uint32_t)(tm->wrap_v & 3) << 3;
        gt->table[t][3] = (uint32_t)tm->lw[skip[t]] | (uint32_t)tm->lh[skip[t]] << 16;
        next += tm->levels - skip[t];
    }
    for (int i = 0; i < nr; i++){
        const TexMap *tm = &s->texmaps[r[i].tex];
        uint32_t *e = gt->table[gt->table[r[i].tex][0] + (uint32_t)(r[i].level - skip[r[i].tex])];
        e[0] = (uint32_t)r[i].x; e[1] = (uint32_t)r[i].y; e[2] = (uint32_t)r[i].w; e[3] = (uint32_t)r[i].h;
        const uint32_t *src = tm->px + tm->at[r[i].level];
        for (int yy = 0; yy < r[i].h; yy++)
            memcpy(px + 4 * ((size_t)(r[i].y + yy) * aw + r[i].x), src + (size_t)yy * r[i].w, (size_t)r[i].w * 4);
    }
    make_atlas(g, gt, aw, ah, px);
    gt->ntex = n;
    free(px);
    free(r);
    return 1;
}

void gpu_textures_free(GpuTextures *gt){
    textures_release(gt);
}

#else
typedef int tex_gpu_unused;
#endif
