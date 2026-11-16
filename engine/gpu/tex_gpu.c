// tex_gpu.c - the materials' textures on the GPU (docs/PBR.md, PBR5).
//
// Every TexMap's mip levels go into one RGBA8 texture, the atlas: each level a rectangle,
// the rectangles placed in rows (the tallest first). A table says where they are. The WGSL
// (GPU_WGSL_TEX) reads texels with textureLoad and filters by hand as scene/texture.c does:
// bilinear inside a level's rectangle (wrapping around inside it, so nothing bleeds in from
// a neighbour), trilinear between levels, sRGB decoded per texel. One texture binding, so the
// path tracer (whose 8 storage buffers are taken) can read it as the rasterizer does.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GPU_WGSL_EXPORT(GPU_WGSL_TEX, tex);

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

// a reference let go (the texture goes when its last user lets it go: no wgpuTextureDestroy,
// another user's bind groups may still hold it until their next sync)
static void textures_release(GpuTextures *gt){
    if (gt->view) wgpuTextureViewRelease(gt->view);
    if (gt->tex) wgpuTextureRelease(gt->tex);
    gt->view = NULL;
    gt->tex = NULL;
}

// an empty atlas of w x h (WebGPU starts textures at zero)
static void new_atlas(Gpu *g, GpuTextures *gt, int w, int h){
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
    gt->view = wgpuTextureCreateView(gt->tex, NULL);
}

// w x h RGBA8 texels written at (x, y) of the atlas, straight from where they are (a texture
// write needs no row alignment, unlike a buffer copy: no copy of the whole atlas on the CPU)
static void put_rect(Gpu *g, GpuTextures *gt, int x, int y, int w, int h, const void *px){
    WGPUTexelCopyTextureInfo dst;
    memset(&dst, 0, sizeof(dst));
    dst.texture = gt->tex;
    dst.aspect = WGPUTextureAspect_All;
    dst.origin = (WGPUOrigin3D){(uint32_t)x, (uint32_t)y, 0};
    WGPUTexelCopyBufferLayout lay;
    memset(&lay, 0, sizeof(lay));
    lay.bytesPerRow = (uint32_t)w * 4;
    lay.rowsPerImage = (uint32_t)h;
    WGPUExtent3D size = {(uint32_t)w, (uint32_t)h, 1};
    wgpuQueueWriteTexture(g->queue, &dst, px, (size_t)w * h * 4, &lay, &size);
}

static void white_atlas(Gpu *g, GpuTextures *gt){
    static const unsigned char none[4] = {255, 255, 255, 255};
    new_atlas(g, gt, 1, 1);
    put_rect(g, gt, 0, 0, 1, 1, none);
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

// the scene's atlas, built into gt (the Gpu's shared one)
static void build_atlas(Gpu *g, GpuTextures *gt, const Scene *s){
    textures_release(gt);
    memset(gt->table, 0, sizeof(gt->table));
    gt->src = s->texmaps; gt->nsrc = s->ntexmaps; gt->version = s->tex_version;
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
        white_atlas(g, gt);
        free(r);
        return;
    }
    new_atlas(g, gt, aw, ah);
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
        put_rect(g, gt, r[i].x, r[i].y, r[i].w, r[i].h, tm->px + tm->at[r[i].level]);
    }
    gt->ntex = n;
    free(r);
}

// The atlas is the Gpu's, one for every user (the rasterizer and the path tracer each built
// their own, through a full-size copy on the CPU: the helmet's 120 MB twice). Each user keeps
// a reference to its texture and view and a copy of the table: changed (1) when the scene's
// textures did.
int gpu_textures_sync(Gpu *g, GpuTextures *gt, const Scene *s){
    if (!g->atlas){
        g->atlas = calloc(1, sizeof(GpuTextures));
        if (!g->atlas) return 0;
    }
    GpuTextures *a = g->atlas;
    if (!a->tex || a->nsrc != s->ntexmaps || a->src != (const void *)s->texmaps || a->version != s->tex_version)
        build_atlas(g, a, s);
    if (gt->tex == a->tex && gt->view == a->view) return 0;
    textures_release(gt);
    memcpy(gt, a, sizeof(*gt));
    gpu_ref_texture(gt->tex);
    gpu_ref_view(gt->view);
    return 1;
}

void gpu_textures_free(GpuTextures *gt){
    textures_release(gt);
}

void gpu_atlas_free(Gpu *g){
    if (!g || !g->atlas) return;
    textures_release(g->atlas);
    free(g->atlas);
    g->atlas = NULL;
}

#else
typedef int tex_gpu_unused;
#endif
