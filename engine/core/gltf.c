// gltf.c - see gltf.h. The file's JSON (core/json.c) is read once into a tree; buffers are
// kept as they are (or decoded from base64) and accessors read straight out of them.
#include "core/gltf.h"
#include "core/json.h"
#include "core/texture.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p; size_t n; uint8_t *own; } Buf;

typedef struct {
    const char *path;
    char dir[1024];             // the file's folder (with its '/'), for relative URIs
    Json *j;
    uint8_t *file;              // the whole file (a .glb's BIN chunk points into it)
    size_t file_n;
    const uint8_t *bin;         // the .glb's BIN chunk
    size_t bin_n;
    Buf *bufs; int nbufs;
    Scene *s;
    int (*tex)[2]; int ntex;    // the scene's texture for (glTF texture, sRGB): 0 not yet, -1 failed
    int *mats; int nmats;       // the scene's material for each glTF material
    int def_mat;                // ... for primitives without one (glTF's default material)
    int *meshes; int nmeshes;   // the scene's mesh for each glTF mesh (-2 not yet, -1 none)
    int warned_alpha, warned_tc, warned_sparse, warned_mats, warned_mode, warned_mirror;
    long visits;                // nodes visited (a graph that shares children grows fast)
} G;

static int fail(const char *path, const char *why){
    fprintf(stderr, "gltf: %s: %s\n", path, why);
    return -1;
}

static uint8_t *read_file(const char *path, size_t *n){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, k;
    uint8_t *b = malloc(cap);
    *n = 0;
    while (b && (k = fread(b + *n, 1, cap - *n, f)) > 0){
        *n += k;
        if (*n == cap){
            uint8_t *q = realloc(b, cap * 2);
            if (!q){ free(b); b = NULL; break; }
            b = q; cap *= 2;
        }
    }
    fclose(f);
    return b;
}

// a size or an offset from the JSON (absent: def): a whole number from 0 up; anything else
// (negative, fractional, huge, not a number) is refused, 1, so that it never wraps a check
static int json_size(const Json *v, size_t def, size_t *out){
    if (!v){ *out = def; return 0; }
    if (v->type != JSON_NUMBER || !(v->num >= 0) || v->num > (double)(SIZE_MAX / 4) || v->num != floor(v->num)) return 1;
    *out = (size_t)v->num;
    return 0;
}

static uint32_t le32(const uint8_t *p){ return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

// ---- URIs ------------------------------------------------------------------------------------

static int b64(int c){
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static uint8_t *b64_decode(const char *s, size_t *n){
    size_t len = strlen(s);
    uint8_t *out = malloc(len / 4 * 3 + 3), *o = out;
    if (!out) return NULL;
    unsigned acc = 0;
    int bits = 0;
    for (size_t i = 0; i < len; i++){
        int v = b64((unsigned char)s[i]);
        if (v < 0) continue;                    // '=', white space
        acc = acc << 6 | (unsigned)v;
        bits += 6;
        if (bits >= 8){ bits -= 8; *o++ = (uint8_t)(acc >> bits); }
    }
    *n = (size_t)(o - out);
    return out;
}

// a URI's bytes: a data: URI (base64), or a file beside the model (%xx decoded). malloc'ed.
static uint8_t *uri_bytes(G *g, const char *uri, size_t *n){
    if (!strncmp(uri, "data:", 5)){
        const char *c = strstr(uri, ";base64,");
        if (!c){ fprintf(stderr, "gltf: a data: URI not in base64\n"); return NULL; }
        return b64_decode(c + 8, n);
    }
    char path[2048];
    size_t k = (size_t)snprintf(path, sizeof(path), "%s", g->dir);
    for (const char *p = uri; *p && k + 1 < sizeof(path); p++){
        if (*p == '%' && p[1] && p[2]){
            char hx[3] = {p[1], p[2], 0};
            path[k++] = (char)strtol(hx, NULL, 16);
            p += 2;
        } else path[k++] = *p;
    }
    path[k] = 0;
    uint8_t *b = read_file(path, n);
    if (!b) fprintf(stderr, "gltf: cannot read %s\n", path);
    return b;
}

static int load_buffers(G *g){
    const Json *bl = json_get(g->j, "buffers");
    g->nbufs = json_len(bl);
    g->bufs = calloc((size_t)(g->nbufs ? g->nbufs : 1), sizeof(Buf));
    if (!g->bufs) return 1;
    for (int i = 0; i < g->nbufs; i++){
        const Json *b = json_at(bl, i);
        const char *uri = json_str(json_get(b, "uri"), NULL);
        size_t want;
        if (json_size(json_get(b, "byteLength"), 0, &want)){ fprintf(stderr, "gltf: buffer %d: bad byteLength\n", i); return 1; }
        Buf *d = &g->bufs[i];
        if (uri){
            d->own = uri_bytes(g, uri, &d->n);
            if (!d->own) return 1;
            d->p = d->own;
        } else if (i == 0 && g->bin){
            d->p = g->bin;
            d->n = g->bin_n;
        } else { fprintf(stderr, "gltf: buffer %d has no data\n", i); return 1; }
        if (d->n < want){ fprintf(stderr, "gltf: buffer %d is shorter than its byteLength\n", i); return 1; }
    }
    return 0;
}

// ---- accessors -------------------------------------------------------------------------------

typedef struct {
    const uint8_t *p;           // the first element (NULL: all zeros)
    int count, comps, ctype, norm;
    size_t stride, esize;
} Acc;

static int comp_size(int ct){
    switch (ct){ case 5120: case 5121: return 1; case 5122: case 5123: return 2; case 5125: case 5126: return 4; }
    return 0;
}
static int type_comps(const char *t){
    if (!t) return 0;
    if (!strcmp(t, "SCALAR")) return 1;
    if (!strcmp(t, "VEC2")) return 2;
    if (!strcmp(t, "VEC3")) return 3;
    if (!strcmp(t, "VEC4") || !strcmp(t, "MAT2")) return 4;
    if (!strcmp(t, "MAT3")) return 9;
    if (!strcmp(t, "MAT4")) return 16;
    return 0;
}

// accessor `index` checked against its buffer; 0 on success
static int acc_get(G *g, int index, Acc *a){
    const Json *ac = json_at(json_get(g->j, "accessors"), index);
    memset(a, 0, sizeof(*a));
    if (!ac){ fprintf(stderr, "gltf: no accessor %d\n", index); return 1; }
    if (json_get(ac, "sparse") && !g->warned_sparse++)
        fprintf(stderr, "gltf: sparse accessors are not supported (their base values are used)\n");
    a->count = json_int(json_get(ac, "count"), 0);
    a->ctype = json_int(json_get(ac, "componentType"), 0);
    a->comps = type_comps(json_str(json_get(ac, "type"), NULL));
    a->norm = json_get(ac, "normalized") && json_get(ac, "normalized")->type == JSON_TRUE;
    int cs = comp_size(a->ctype);
    if (a->count < 0 || !cs || !a->comps){ fprintf(stderr, "gltf: accessor %d: bad type\n", index); return 1; }
    a->esize = (size_t)cs * a->comps;
    int bv = json_int(json_get(ac, "bufferView"), -1);
    if (bv < 0) return 0;                       // zeros
    const Json *v = json_at(json_get(g->j, "bufferViews"), bv);
    int bi = json_int(json_get(v, "buffer"), -1);
    if (!v || bi < 0 || bi >= g->nbufs){ fprintf(stderr, "gltf: accessor %d: bad buffer view\n", index); return 1; }
    size_t voff, vlen, aoff;
    if (json_size(json_get(v, "byteOffset"), 0, &voff) || json_size(json_get(v, "byteLength"), 0, &vlen) ||
        json_size(json_get(ac, "byteOffset"), 0, &aoff) || json_size(json_get(v, "byteStride"), 0, &a->stride)){
        fprintf(stderr, "gltf: accessor %d: a bad offset, length or stride\n", index);
        return 1;
    }
    if (!a->stride) a->stride = a->esize;
    // inside the view, the view inside its buffer: by subtraction, so that nothing wraps
    const Buf *b = &g->bufs[bi];
    int past = voff > b->n || vlen > b->n - voff;
    if (!past && a->count > 0)
        past = aoff > vlen || a->esize > vlen - aoff || (size_t)(a->count - 1) > (vlen - aoff - a->esize) / a->stride;
    if (past){
        fprintf(stderr, "gltf: accessor %d runs past its buffer\n", index);
        return 1;
    }
    a->p = b->p + voff + aoff;
    return 0;
}

static double comp_read(const uint8_t *p, int ct, int norm){
    switch (ct){
    case 5120: { int8_t v = (int8_t)p[0]; return norm ? fmax(v / 127.0, -1.0) : v; }
    case 5121: return norm ? p[0] / 255.0 : p[0];
    case 5122: { int16_t v = (int16_t)(p[0] | p[1] << 8); return norm ? fmax(v / 32767.0, -1.0) : v; }
    case 5123: { unsigned v = (unsigned)(p[0] | p[1] << 8); return norm ? v / 65535.0 : v; }
    case 5125: return (double)le32(p);
    case 5126: { uint32_t u = le32(p); float f; memcpy(&f, &u, 4); return f; }
    }
    return 0;
}

// element i's first n components (missing ones 0)
static void acc_floats(const Acc *a, int i, double *out, int n){
    for (int k = 0; k < n; k++) out[k] = 0;
    if (!a->p) return;
    const uint8_t *e = a->p + a->stride * (size_t)i;
    int cs = comp_size(a->ctype);
    for (int k = 0; k < n && k < a->comps; k++) out[k] = comp_read(e + (size_t)k * cs, a->ctype, a->norm);
}
static uint32_t acc_index(const Acc *a, int i){
    if (!a->p) return 0;
    const uint8_t *e = a->p + a->stride * (size_t)i;
    return a->ctype == 5121 ? e[0] : a->ctype == 5123 ? (uint32_t)(e[0] | e[1] << 8) : le32(e);
}

// ---- textures and materials ------------------------------------------------------------------

static int wrap_mode(int gl){ return gl == 33071 ? TEXMAP_CLAMP : gl == 33648 ? TEXMAP_MIRROR : TEXMAP_REPEAT; }

// glTF texture ti decoded (its image and sampler) as sRGB or not; 0 on success. Reads the
// JSON and the buffers only: several run at once
static int decode_texture(G *g, int ti, int srgb, TexMap *tm){
    const Json *t = json_at(json_get(g->j, "textures"), ti);
    int src = json_int(json_get(t, "source"), -1);
    const Json *im = json_at(json_get(g->j, "images"), src);
    if (!im){ fprintf(stderr, "gltf: texture %d has no image we can read (KTX2 and WebP are not supported)\n", ti); return 1; }
    const char *uri = json_str(json_get(im, "uri"), NULL);
    uint8_t *own = NULL;
    const uint8_t *data = NULL;
    size_t n = 0;
    if (uri){
        own = uri_bytes(g, uri, &n);
        data = own;
    } else {
        const Json *v = json_at(json_get(g->j, "bufferViews"), json_int(json_get(im, "bufferView"), -1));
        int bi = json_int(json_get(v, "buffer"), -1);
        size_t off;
        if (v && bi >= 0 && bi < g->nbufs && !json_size(json_get(v, "byteOffset"), 0, &off) && !json_size(json_get(v, "byteLength"), 0, &n)
            && off <= g->bufs[bi].n && n <= g->bufs[bi].n - off) data = g->bufs[bi].p + off;
    }
    if (!data){ fprintf(stderr, "gltf: image %d: no data\n", src); free(own); return 1; }
    int err = texmap_decode(tm, data, n, srgb);
    free(own);
    if (err){ fprintf(stderr, "gltf: image %d cannot be read\n", src); return 1; }
    const Json *sm = json_at(json_get(g->j, "samplers"), json_int(json_get(t, "sampler"), -1));
    tm->wrap_u = wrap_mode(json_int(json_get(sm, "wrapS"), 10497));
    tm->wrap_v = wrap_mode(json_int(json_get(sm, "wrapT"), 10497));
    const char *base = strrchr(g->path, '/');
    base = base ? base + 1 : g->path;
    snprintf(tm->name, sizeof(tm->name), "%.40s#%d%s", base, src, srgb ? "" : "l");
    return 0;
}

// the scene's texture for a textureInfo (0 = none); made here if decode_textures did not
static int texture_of(G *g, const Json *info, int srgb){
    if (!info) return 0;
    int ti = json_int(json_get(info, "index"), -1);
    if (ti < 0 || ti >= g->ntex) return 0;
    if (json_int(json_get(info, "texCoord"), 0) != 0 && !g->warned_tc++)
        fprintf(stderr, "gltf: texture coordinates other than 0 are not supported (0 used)\n");
    if (g->tex[ti][srgb]) return g->tex[ti][srgb] > 0 ? g->tex[ti][srgb] : 0;
    TexMap tm;
    g->tex[ti][srgb] = decode_texture(g, ti, srgb, &tm) ? -1 : scene_add_texmap(g->s, &tm);
    return g->tex[ti][srgb] > 0 ? g->tex[ti][srgb] : 0;
}

// every texture the materials use, decoded at once (each image's JPEG or PNG and its mip
// levels on its own thread), then given to the scene in order
static void decode_textures(G *g){
    typedef struct { int ti, srgb, err; TexMap tm; } Job;
    Job *jobs = malloc(sizeof(Job) * (size_t)(2 * g->ntex + 1));
    if (!jobs) return;
    int nj = 0;
    const Json *mats = json_get(g->j, "materials");
    for (int i = 0; i < json_len(mats); i++){
        const Json *m = json_at(mats, i), *pbr = json_get(m, "pbrMetallicRoughness");
        const Json *ef = json_get(m, "emissiveFactor");
        int glows = json_num(json_at(ef, 0), 0) > 0 || json_num(json_at(ef, 1), 0) > 0 || json_num(json_at(ef, 2), 0) > 0;
        const Json *info[5] = {json_get(pbr, "baseColorTexture"), json_get(pbr, "metallicRoughnessTexture"),
                               json_get(m, "normalTexture"), json_get(m, "occlusionTexture"), glows ? json_get(m, "emissiveTexture") : NULL};
        const int srgb[5] = {1, 0, 0, 0, 1};
        for (int k = 0; k < 5; k++){
            int ti = json_int(json_get(info[k], "index"), -1);
            if (ti < 0 || ti >= g->ntex || g->tex[ti][srgb[k]]) continue;
            g->tex[ti][srgb[k]] = -2;                    // (queued)
            jobs[nj].ti = ti; jobs[nj].srgb = srgb[k];
            nj++;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < nj; i++) jobs[i].err = decode_texture(g, jobs[i].ti, jobs[i].srgb, &jobs[i].tm);
    for (int i = 0; i < nj; i++)
        g->tex[jobs[i].ti][jobs[i].srgb] = jobs[i].err ? -1 : scene_add_texmap(g->s, &jobs[i].tm);
    free(jobs);
}

static Material material_of(G *g, const Json *m){
    Material r = material_pbr(vec3(1, 1, 1), 1, 1, 1.5f, 0);     // glTF's default material
    if (!m) return r;
    const Json *pbr = json_get(m, "pbrMetallicRoughness");
    const Json *bc = json_get(pbr, "baseColorFactor");
    if (json_len(bc) >= 3) r.albedo = vec3((real)json_num(json_at(bc, 0), 1), (real)json_num(json_at(bc, 1), 1), (real)json_num(json_at(bc, 2), 1));
    r.metallic = (real)json_num(json_get(pbr, "metallicFactor"), 1);
    r.roughness = (real)json_num(json_get(pbr, "roughnessFactor"), 1);
    r.tex_base = texture_of(g, json_get(pbr, "baseColorTexture"), 1);
    r.tex_mr = texture_of(g, json_get(pbr, "metallicRoughnessTexture"), 0);
    const Json *nt = json_get(m, "normalTexture");
    real ns = (real)json_num(json_get(nt, "scale"), 1);
    if (nt && ns != 0){ r.tex_normal = texture_of(g, nt, 0); r.normal_scale = r_fabs(ns); }
    const Json *ot = json_get(m, "occlusionTexture");
    real os = (real)json_num(json_get(ot, "strength"), 1);
    if (ot && os > 0){ r.tex_occlusion = texture_of(g, ot, 0); r.occlusion = os; }
    const Json *ext = json_get(m, "extensions");
    real es = (real)json_num(json_get(json_get(ext, "KHR_materials_emissive_strength"), "emissiveStrength"), 1);
    const Json *ef = json_get(m, "emissiveFactor");
    if (json_len(ef) >= 3)
        r.emission = v_scale(vec3((real)json_num(json_at(ef, 0), 0), (real)json_num(json_at(ef, 1), 0), (real)json_num(json_at(ef, 2), 0)), es);
    if (material_glows(&r)) r.tex_emissive = texture_of(g, json_get(m, "emissiveTexture"), 1);
    r.ior = (real)json_num(json_get(json_get(ext, "KHR_materials_ior"), "ior"), 1.5);
    if (!(r.ior >= 1)) r.ior = 1.5f;
    r.transmission = (real)json_num(json_get(json_get(ext, "KHR_materials_transmission"), "transmissionFactor"), 0);
    const char *am = json_str(json_get(m, "alphaMode"), "OPAQUE");
    if (strcmp(am, "OPAQUE") && !g->warned_alpha++)
        fprintf(stderr, "gltf: alpha modes MASK and BLEND are not supported (drawn opaque)\n");
    return r;
}

static int scene_material(G *g, int mi){
    if (mi >= 0 && mi < g->nmats) return g->mats[mi];
    return g->def_mat;
}

static int add_material(G *g, Material m){
    if (g->s->nmats >= MAX_MATS){
        if (!g->warned_mats++) fprintf(stderr, "gltf: more than %d materials in the scene: the rest share the last\n", MAX_MATS);
        return g->s->nmats - 1;
    }
    return scene_add_material(g->s, m);
}

// ---- meshes ----------------------------------------------------------------------------------

typedef struct { void *p; int n, cap; size_t elem; } Vec;
static int vec_push(Vec *v, const void *e){
    if (v->n == v->cap){
        int nc = v->cap ? v->cap * 2 : 64;
        void *q = realloc(v->p, (size_t)nc * v->elem);
        if (!q) return 1;
        v->p = q; v->cap = nc;
    }
    memcpy((char *)v->p + (size_t)v->n * v->elem, e, v->elem);
    v->n++;
    return 0;
}

// glTF mesh mi as a scene mesh (all its primitives), or -1 if it has no triangles
static int mesh_of(G *g, int mi){
    if (mi < 0 || mi >= g->nmeshes) return -1;
    if (g->meshes[mi] != -2) return g->meshes[mi];
    g->meshes[mi] = -1;
    const Json *prims = json_get(json_at(json_get(g->j, "meshes"), mi), "primitives");
    Vec tris = {0, 0, 0, sizeof(Triangle)}, nrm = {0, 0, 0, sizeof(Vec3)}, uvs = {0, 0, 0, sizeof(Vec2)}, tan = {0, 0, 0, sizeof(Vec4)};
    Vec todo = {0, 0, 0, sizeof(int[2])};                 // uv ranges whose tangents are computed
    int bad = 0;
    for (int pi = 0; pi < json_len(prims) && !bad; pi++){
        const Json *pr = json_at(prims, pi);
        int mode = json_int(json_get(pr, "mode"), 4);
        if (mode != 4 && mode != 5 && mode != 6){
            if (!g->warned_mode++) fprintf(stderr, "gltf: points and lines are not drawn\n");
            continue;
        }
        const Json *at = json_get(pr, "attributes");
        Acc pos, an, auv, atan, idx;
        if (acc_get(g, json_int(json_get(at, "POSITION"), -1), &pos) || pos.comps < 3){ bad = 1; break; }
        int has_n = json_get(at, "NORMAL") && !acc_get(g, json_int(json_get(at, "NORMAL"), -1), &an) && an.count >= pos.count;
        int has_uv = json_get(at, "TEXCOORD_0") && !acc_get(g, json_int(json_get(at, "TEXCOORD_0"), -1), &auv) && auv.count >= pos.count;
        int has_t = has_uv && json_get(at, "TANGENT") && !acc_get(g, json_int(json_get(at, "TANGENT"), -1), &atan) && atan.count >= pos.count && atan.comps >= 4;
        int has_i = json_get(pr, "indices") != NULL;
        if (has_i && acc_get(g, json_int(json_get(pr, "indices"), -1), &idx)){ bad = 1; break; }
        int nb = nrm.n, ub = uvs.n;
        for (int i = 0; i < pos.count && !bad; i++){
            double d[4];
            if (has_n){
                acc_floats(&an, i, d, 3);
                Vec3 v = vec3((real)d[0], (real)d[1], (real)d[2]);
                real l = v_len(v);
                v = l > 1e-12f ? v_scale(v, 1 / l) : vec3(0, 1, 0);
                bad |= vec_push(&nrm, &v);
            }
            if (has_uv){
                acc_floats(&auv, i, d, 2);
                Vec2 v = vec2((real)d[0], (real)d[1]);
                bad |= vec_push(&uvs, &v);
                Vec4 t = {1, 0, 0, 1};
                if (has_t){
                    acc_floats(&atan, i, d, 4);
                    Vec3 x = vec3((real)d[0], (real)d[1], (real)d[2]);
                    real l = v_len(x);
                    if (l > 1e-12f) x = v_scale(x, 1 / l);
                    t = (Vec4){x.x, x.y, x.z, d[3] < 0 ? -1 : 1};
                }
                bad |= vec_push(&tan, &t);
            }
        }
        if (has_uv && !has_t){ int r[2] = {ub, uvs.n - ub}; bad |= vec_push(&todo, r); }
        int n = has_i ? idx.count : pos.count;
        int ntri = mode == 4 ? n / 3 : n - 2;
        int material = scene_material(g, json_int(json_get(pr, "material"), -1));
        for (int k = 0; k < ntri && !bad; k++){
            int c[3];
            if (mode == 4){ c[0] = 3 * k; c[1] = 3 * k + 1; c[2] = 3 * k + 2; }
            else if (mode == 5){ c[0] = k; c[1] = k % 2 ? k + 2 : k + 1; c[2] = k % 2 ? k + 1 : k + 2; }
            else { c[0] = 0; c[1] = k + 1; c[2] = k + 2; }
            uint32_t v[3];
            int ok = 1;
            for (int q = 0; q < 3; q++){
                v[q] = has_i ? acc_index(&idx, c[q]) : (uint32_t)c[q];
                ok &= v[q] < (uint32_t)pos.count;
            }
            if (!ok){ fprintf(stderr, "gltf: mesh %d: an index past its vertices\n", mi); bad = 1; break; }
            double a[3], b[3], e[3];
            acc_floats(&pos, (int)v[0], a, 3);
            acc_floats(&pos, (int)v[1], b, 3);
            acc_floats(&pos, (int)v[2], e, 3);
            Triangle t;
            t.q = vec3((real)a[0], (real)a[1], (real)a[2]);
            t.u = vec3((real)(b[0] - a[0]), (real)(b[1] - a[1]), (real)(b[2] - a[2]));
            t.v = vec3((real)(e[0] - a[0]), (real)(e[1] - a[1]), (real)(e[2] - a[2]));
            for (int q = 0; q < 3; q++){
                t.n[q] = has_n ? nb + (int)v[q] : -1;
                t.t[q] = has_uv ? ub + (int)v[q] : -1;
            }
            t.material = material;
            bad |= vec_push(&tris, &t);
        }
    }
    if (bad || tris.n == 0){
        if (bad) fprintf(stderr, "gltf: mesh %d left out\n", mi);
        free(tris.p); free(nrm.p); free(uvs.p); free(tan.p); free(todo.p);
        return -1;
    }
    for (int i = 0; i < todo.n; i++){
        const int *r = (const int *)todo.p + 2 * i;
        scene_compute_tangents(tris.p, tris.n, uvs.p, tan.p, r[0], r[1]);
    }
    free(todo.p);
    g->meshes[mi] = scene_add_mesh_arrays(g->s, tris.p, tris.n, nrm.p, nrm.n, uvs.p, tan.p, uvs.n);
    return g->meshes[mi];
}

// ---- the node tree ---------------------------------------------------------------------------
// 4 x 4 matrices as glTF stores them: column by column

static void mat_mul(const double *a, const double *b, double *o){
    double r[16];
    for (int c = 0; c < 4; c++)
        for (int k = 0; k < 4; k++){
            double s = 0;
            for (int i = 0; i < 4; i++) s += a[i * 4 + k] * b[c * 4 + i];
            r[c * 4 + k] = s;
        }
    memcpy(o, r, sizeof(r));
}

static void node_local(const Json *nd, double *m){
    const Json *mx = json_get(nd, "matrix");
    if (json_len(mx) == 16){
        for (int i = 0; i < 16; i++) m[i] = json_num(json_at(mx, i), i % 5 == 0);
        return;
    }
    const Json *tj = json_get(nd, "translation"), *rj = json_get(nd, "rotation"), *sj = json_get(nd, "scale");
    double t[3], q[4], s[3];
    for (int i = 0; i < 3; i++){ t[i] = json_num(json_at(tj, i), 0); s[i] = json_num(json_at(sj, i), 1); }
    for (int i = 0; i < 4; i++) q[i] = json_num(json_at(rj, i), i == 3);
    double l = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (l > 0) for (int i = 0; i < 4; i++) q[i] /= l;
    double x = q[0], y = q[1], z = q[2], w = q[3];
    memset(m, 0, 16 * sizeof(double));
    m[0] = (1 - 2 * (y * y + z * z)) * s[0]; m[1] = 2 * (x * y + w * z) * s[0];       m[2] = 2 * (x * z - w * y) * s[0];
    m[4] = 2 * (x * y - w * z) * s[1];       m[5] = (1 - 2 * (x * x + z * z)) * s[1]; m[6] = 2 * (y * z + w * x) * s[1];
    m[8] = 2 * (x * z + w * y) * s[2];       m[9] = 2 * (y * z - w * x) * s[2];       m[10] = (1 - 2 * (x * x + y * y)) * s[2];
    m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
}

typedef struct { int mesh; double m[16]; } Placed;

static void visit(G *g, int ni, const double *parent, int depth, Vec *out){
    const Json *nd = json_at(json_get(g->j, "nodes"), ni);
    // (a cycle stops at depth 64; a node listing the next one twice, and so on, would be
    // visited 2^64 times: at most a million visits)
    if (!nd || depth > 64) return;
    if (++g->visits > 1000000){
        if (g->visits == 1000001) fprintf(stderr, "gltf: %s: more than a million nodes placed: the rest left out\n", g->path);
        return;
    }
    double local[16], world[16];
    node_local(nd, local);
    mat_mul(parent, local, world);
    if (json_get(nd, "mesh")){
        int sm = mesh_of(g, json_int(json_get(nd, "mesh"), -1));
        if (sm >= 0){
            Placed p;
            p.mesh = sm;
            memcpy(p.m, world, sizeof(world));
            vec_push(out, &p);
        }
    }
    const Json *ch = json_get(nd, "children");
    for (int i = 0; i < json_len(ch); i++) visit(g, json_int(json_at(ch, i), -1), world, depth + 1, out);
}

static void g_free(G *g){
    for (int i = 0; i < g->nbufs; i++) free(g->bufs[i].own);
    free(g->bufs);
    free(g->tex);
    free(g->mats);
    free(g->meshes);
    json_free(g->j);
    free(g->file);
}

int scene_load_gltf(Scene *s, const char *path, real size, Vec3 base, real rot_y_deg){
    G gg, *g = &gg;
    memset(g, 0, sizeof(*g));
    g->path = path;
    g->s = s;
    const char *slash = strrchr(path, '/'), *bs = strrchr(path, '\\');
    if (bs && (!slash || bs > slash)) slash = bs;
    if (slash && (size_t)(slash - path) + 2 < sizeof(g->dir)){ memcpy(g->dir, path, (size_t)(slash - path) + 1); g->dir[slash - path + 1] = 0; }
    g->file = read_file(path, &g->file_n);
    if (!g->file) return fail(path, "cannot read the file");
    // a .glb: a 12-byte header, then chunks (JSON first, then BIN)
    if (g->file_n >= 12 && le32(g->file) == 0x46546C67u){
        if (le32(g->file + 4) != 2){ g_free(g); return fail(path, "not glTF 2.0"); }
        size_t at = 12;
        const uint8_t *js = NULL;
        size_t js_n = 0;
        while (at + 8 <= g->file_n){
            size_t n = le32(g->file + at);
            uint32_t type = le32(g->file + at + 4);
            if (n > g->file_n - at - 8) break;              // (by subtraction: no wrap)
            if (type == 0x4E4F534Au && !js){ js = g->file + at + 8; js_n = n; }
            else if (type == 0x004E4942u && !g->bin){ g->bin = g->file + at + 8; g->bin_n = n; }
            at += 8 + ((n + 3) & ~(size_t)3);
        }
        if (!js){ g_free(g); return fail(path, "no JSON chunk"); }
        g->j = json_parse((const char *)js, js_n);
    } else g->j = json_parse((const char *)g->file, g->file_n);
    if (!g->j){ g_free(g); return fail(path, "bad JSON"); }
    const char *ver = json_str(json_get(json_get(g->j, "asset"), "version"), "");
    if (strncmp(ver, "2.", 2)){ g_free(g); return fail(path, "not glTF 2.0"); }
    const Json *req = json_get(g->j, "extensionsRequired");
    for (int i = 0; i < json_len(req); i++){
        const char *e = json_str(json_at(req, i), "");
        if (!strcmp(e, "KHR_draco_mesh_compression") || !strcmp(e, "EXT_meshopt_compression")){
            fprintf(stderr, "gltf: %s: %s is not supported\n", path, e);
            g_free(g);
            return -1;
        }
        if (strcmp(e, "KHR_mesh_quantization") && strcmp(e, "KHR_materials_emissive_strength"))
            fprintf(stderr, "gltf: %s: required extension %s is not supported (drawn without it)\n", path, e);
    }
    if (load_buffers(g)){ g_free(g); return fail(path, "cannot read its buffers"); }
    g->ntex = json_len(json_get(g->j, "textures"));
    g->tex = calloc((size_t)(g->ntex ? g->ntex : 1), sizeof(int[2]));
    g->nmats = json_len(json_get(g->j, "materials"));
    g->mats = calloc((size_t)(g->nmats ? g->nmats : 1), sizeof(int));
    g->nmeshes = json_len(json_get(g->j, "meshes"));
    g->meshes = malloc((size_t)(g->nmeshes ? g->nmeshes : 1) * sizeof(int));
    if (!g->tex || !g->mats || !g->meshes){ g_free(g); return fail(path, "out of memory"); }
    for (int i = 0; i < g->nmeshes; i++) g->meshes[i] = -2;
    // the materials (the default one only if a primitive has none), their textures first
    decode_textures(g);
    for (int i = 0; i < g->nmats; i++) g->mats[i] = add_material(g, material_of(g, json_at(json_get(g->j, "materials"), i)));
    g->def_mat = -1;
    for (int mi = 0; mi < g->nmeshes && g->def_mat < 0; mi++){
        const Json *prims = json_get(json_at(json_get(g->j, "meshes"), mi), "primitives");
        for (int pi = 0; pi < json_len(prims); pi++){
            int m = json_int(json_get(json_at(prims, pi), "material"), -1);
            if (m < 0 || m >= g->nmats){ g->def_mat = add_material(g, material_of(g, NULL)); break; }
        }
    }
    // the default scene's nodes (or, without scenes, the nodes no node has as a child)
    Vec placed = {0, 0, 0, sizeof(Placed)};
    double id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const Json *scenes = json_get(g->j, "scenes");
    if (json_len(scenes) > 0){
        const Json *roots = json_get(json_at(scenes, json_int(json_get(g->j, "scene"), 0)), "nodes");
        if (!roots) roots = json_get(json_at(scenes, 0), "nodes");
        for (int i = 0; i < json_len(roots); i++) visit(g, json_int(json_at(roots, i), -1), id, 0, &placed);
    } else {
        int nn = json_len(json_get(g->j, "nodes"));
        char *child = calloc((size_t)(nn ? nn : 1), 1);
        for (int i = 0; child && i < nn; i++){
            const Json *ch = json_get(json_at(json_get(g->j, "nodes"), i), "children");
            for (int k = 0; k < json_len(ch); k++){ int c = json_int(json_at(ch, k), -1); if (c >= 0 && c < nn) child[c] = 1; }
        }
        for (int i = 0; child && i < nn; i++) if (!child[i]) visit(g, i, id, 0, &placed);
        free(child);
    }
    if (placed.n == 0){ free(placed.p); g_free(g); return fail(path, "no mesh to draw"); }
    // where it goes: its world box over every instance, fitted into `size`
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < placed.n; i++){
        const Placed *p = (const Placed *)placed.p + i;
        Aabb b = s->meshes[p->mesh].box;
        for (int c = 0; c < 8; c++){
            double x = c & 1 ? b.hi.x : b.lo.x, y = c & 2 ? b.hi.y : b.lo.y, z = c & 4 ? b.hi.z : b.lo.z;
            for (int k = 0; k < 3; k++){
                double w = p->m[k] * x + p->m[4 + k] * y + p->m[8 + k] * z + p->m[12 + k];
                if (w < lo[k]) lo[k] = w;
                if (w > hi[k]) hi[k] = w;
            }
        }
    }
    double ext = fmax(hi[0] - lo[0], fmax(hi[1] - lo[1], hi[2] - lo[2]));
    double k = ext > 0 ? size / ext : 1, a = rot_y_deg * PI / 180, ca = cos(a), sa = sin(a);
    double cx = (lo[0] + hi[0]) / 2, cz = (lo[2] + hi[2]) / 2;
    // root = move to base * turn about y * scale * move (cx, lo.y, cz) to the origin
    double root[16] = {k * ca, 0, -k * sa, 0,  0, k, 0, 0,  k * sa, 0, k * ca, 0,  0, 0, 0, 1};
    root[12] = base.x - k * (ca * cx + sa * cz);
    root[13] = base.y - k * lo[1];
    root[14] = base.z - k * (-sa * cx + ca * cz);
    int added = 0;
    for (int i = 0; i < placed.n; i++){
        const Placed *p = (const Placed *)placed.p + i;
        double m[16];
        mat_mul(root, p->m, m);
        double det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
        if (det < 0 && !g->warned_mirror++)
            fprintf(stderr, "gltf: %s: a mirroring transform: its normal maps may look turned inside out\n", path);
        Xform xf;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++) xf.m[r][c] = (real)m[c * 4 + r];
        scene_add_instance(s, p->mesh, xf, -1);
        added++;
    }
    free(placed.p);
    g_free(g);
    return added;
}
