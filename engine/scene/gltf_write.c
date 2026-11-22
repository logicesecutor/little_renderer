// gltf_write.c - a scene written as glTF 2.0 binary (see gltf_write.h)
#include "scene/gltf_write.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "base/mem.h"
#include "formats/png.h"
#include "scene/texture.h"

// ---- a growing text, and a growing binary -------------------------------------------------

typedef struct { char *s; size_t n, cap; int items; } Text;

static void tx_put(Text *t, const char *fmt, ...){
    for (;;){
        va_list ap;
        va_start(ap, fmt);
        int k = vsnprintf(t->s ? t->s + t->n : NULL, t->s ? t->cap - t->n : 0, fmt, ap);
        va_end(ap);
        if (k < 0) return;
        if (t->s && t->n + (size_t)k < t->cap){ t->n += (size_t)k; return; }
        size_t want = t->cap ? t->cap : 1024;
        while (want <= t->n + (size_t)k) want *= 2;
        char *b = realloc(t->s, want);
        if (!b) lr_oom("gltf_write");
        t->s = b;
        t->cap = want;
    }
}
// the next item of a JSON array (a comma before all but the first); returns its index
static int tx_item(Text *t){
    if (t->items) tx_put(t, ",");
    return t->items++;
}
static void tx_string(Text *t, const char *s){
    tx_put(t, "\"");
    for (; *s; s++){
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') tx_put(t, "\\%c", c);
        else if (c < 32) tx_put(t, "\\u%04x", c);
        else tx_put(t, "%c", c);
    }
    tx_put(t, "\"");
}

typedef struct { unsigned char *d; size_t n, cap; } Bin;

// room for n more bytes (the new ones zero)
static void bin_room(Bin *b, size_t n){
    if (b->n + n <= b->cap) return;
    size_t want = b->cap ? b->cap : 65536;
    while (want < b->n + n) want *= 2;
    unsigned char *p = realloc(b->d, want);
    if (!p) lr_oom("gltf_write");
    memset(p + b->cap, 0, want - b->cap);
    b->d = p;
    b->cap = want;
}
// zeros to a multiple of 4 bytes (every accessor's data starts there; the chunk ends there)
static void bin_pad(Bin *b){
    size_t pad = (4 - b->n % 4) % 4;
    bin_room(b, pad);
    b->n += pad;
}
static size_t bin_add(Bin *b, const void *d, size_t n){
    bin_pad(b);
    bin_room(b, n);
    size_t at = b->n;
    memcpy(b->d + at, d, n);
    b->n += n;
    return at;
}

// ---- the writer ------------------------------------------------------------------------------

typedef struct { int material, pos, nrm, uv; } GPrim;    // material: the scene's (-1: a shape's, none)
typedef struct { GPrim *p; int n; } Geometry;

struct GltfWriter {
    const Scene *s;
    real unit;
    Bin bin;
    Text views, accessors, meshes, materials, textures, images, samplers, nodes, root_ext;
    Geometry *geoms;
    int ngeoms, cap_geoms;
    unsigned char *child;            // per node: 1 if it is a group's (not in the scene itself)
    int cap_child;
    int *mat_map, *tex_map;          // the scene's materials and textures written: their glTF index (-1: not yet)
    char used[16][64];
    int nused, required[16];
    int use_transmission, use_ior, use_strength;
};

GltfWriter *gltf_writer_new(const Scene *s, real unit){
    GltfWriter *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->s = s;
    w->unit = unit;
    w->mat_map = malloc(sizeof(int) * (size_t)(s->nmats > 0 ? s->nmats : 1));
    w->tex_map = malloc(sizeof(int) * (size_t)(s->ntexmaps > 0 ? s->ntexmaps : 1));
    if (!w->mat_map || !w->tex_map){ gltf_writer_free(w); return NULL; }
    for (int i = 0; i < s->nmats; i++) w->mat_map[i] = -1;
    for (int i = 0; i < s->ntexmaps; i++) w->tex_map[i] = -1;
    return w;
}

void gltf_writer_free(GltfWriter *w){
    if (!w) return;
    free(w->bin.d);
    Text *t[] = {&w->views, &w->accessors, &w->meshes, &w->materials, &w->textures, &w->images, &w->samplers, &w->nodes, &w->root_ext};
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) free(t[i]->s);
    for (int i = 0; i < w->ngeoms; i++) free(w->geoms[i].p);
    free(w->geoms);
    free(w->child);
    free(w->mat_map);
    free(w->tex_map);
    free(w);
}

// data in the binary, as a buffer view (target: 34962 vertices, 0 none); its index
static int add_view(GltfWriter *w, const void *d, size_t n, int target){
    size_t at = bin_add(&w->bin, d, n);
    int i = tx_item(&w->views);
    tx_put(&w->views, "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu", at, n);
    if (target) tx_put(&w->views, ",\"target\":%d", target);
    tx_put(&w->views, "}");
    return i;
}

// count float vectors of `comps` (2, 3) as an accessor; bounds: POSITION's min and max
static int add_floats(GltfWriter *w, const float *v, int count, int comps, int bounds){
    int view = add_view(w, v, sizeof(float) * (size_t)count * comps, 34962);
    int i = tx_item(&w->accessors);
    tx_put(&w->accessors, "{\"bufferView\":%d,\"componentType\":5126,\"count\":%d,\"type\":\"VEC%d\"", view, count, comps);
    if (bounds && count > 0){
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (int k = 0; k < count; k++)
            for (int c = 0; c < comps; c++){
                float x = v[k * comps + c];
                if (x < lo[c]) lo[c] = x;
                if (x > hi[c]) hi[c] = x;
            }
        tx_put(&w->accessors, ",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    }
    tx_put(&w->accessors, "}");
    return i;
}

// a texture of the scene (1 + an index into texmaps), as a PNG image; its glTF index (-1: none)
static int texture_of(GltfWriter *w, int tex){
    const Scene *s = w->s;
    if (tex <= 0 || tex > s->ntexmaps) return -1;
    if (w->tex_map[tex - 1] >= 0) return w->tex_map[tex - 1];
    const TexMap *t = &s->texmaps[tex - 1];
    unsigned char *rgba = malloc((size_t)t->w * t->h * 4);
    if (!rgba) return -1;
    for (size_t i = 0; i < (size_t)t->w * t->h; i++){
        uint32_t p = t->px[i];
        rgba[4 * i] = p & 255; rgba[4 * i + 1] = p >> 8 & 255; rgba[4 * i + 2] = p >> 16 & 255; rgba[4 * i + 3] = p >> 24 & 255;
    }
    size_t n = 0;
    unsigned char *png = png_encode(t->w, t->h, 4, rgba, &n);
    free(rgba);
    if (!png) return -1;
    int view = add_view(w, png, n, 0);
    free(png);
    int img = tx_item(&w->images);
    tx_put(&w->images, "{\"bufferView\":%d,\"mimeType\":\"image/png\"}", view);
    static const int WRAP[3] = {10497, 33071, 33648};       // TEXMAP_REPEAT, _CLAMP, _MIRROR
    int smp = tx_item(&w->samplers);
    tx_put(&w->samplers, "{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":%d,\"wrapT\":%d}",
           WRAP[t->wrap_u >= 0 && t->wrap_u < 3 ? t->wrap_u : 0], WRAP[t->wrap_v >= 0 && t->wrap_v < 3 ? t->wrap_v : 0]);
    int i = tx_item(&w->textures);
    tx_put(&w->textures, "{\"sampler\":%d,\"source\":%d}", smp, img);
    w->tex_map[tex - 1] = i;
    return i;
}

// a material of the scene, as glTF's; its index
static int material_of(GltfWriter *w, int mat){
    const Scene *s = w->s;
    if (mat < 0 || mat >= s->nmats) mat = 0;
    if (w->mat_map[mat] >= 0) return w->mat_map[mat];
    const Material *m = &s->mats[mat];
    Vec3 base = m->albedo, em = m->emission;
    real metal = m->metallic, rough = m->roughness, trans = m->transmission, ior = m->ior;
    switch (m->type){                                          // (the presets, in metallic-roughness)
    case MAT_DIFFUSE: metal = 0; rough = 1; trans = 0; break;
    case MAT_MIRROR: metal = 1; rough = 0; trans = 0; break;
    case MAT_GLASS: metal = 0; rough = 0; trans = 1; break;
    case MAT_LIGHT: base = vec3(0, 0, 0); metal = 0; rough = 1; trans = 0; break;
    default: break;
    }
    int tb = texture_of(w, m->tex_base), tm = texture_of(w, m->tex_mr), tn = texture_of(w, m->tex_normal);
    int te = texture_of(w, m->tex_emissive), to = texture_of(w, m->tex_occlusion);
    int i = tx_item(&w->materials);
    Text *t = &w->materials;
    tx_put(t, "{\"name\":\"material %d\",\"pbrMetallicRoughness\":{\"baseColorFactor\":[%.6g,%.6g,%.6g,1],\"metallicFactor\":%.6g,\"roughnessFactor\":%.6g",
           mat, (double)base.x, (double)base.y, (double)base.z, (double)metal, (double)rough);
    if (tb >= 0) tx_put(t, ",\"baseColorTexture\":{\"index\":%d}", tb);
    if (tm >= 0) tx_put(t, ",\"metallicRoughnessTexture\":{\"index\":%d}", tm);
    tx_put(t, "}");
    if (tn >= 0) tx_put(t, ",\"normalTexture\":{\"index\":%d,\"scale\":%.6g}", tn, (double)(m->normal_scale > 0 ? m->normal_scale : 1));
    if (to >= 0) tx_put(t, ",\"occlusionTexture\":{\"index\":%d,\"strength\":%.6g}", to, (double)(m->occlusion > 0 ? m->occlusion : 1));
    // the emission: glTF's factor is at most 1; more is KHR_materials_emissive_strength
    real most = r_fmax(em.x, r_fmax(em.y, em.z)), strength = most > 1 ? most : 1;
    if (most > 0) tx_put(t, ",\"emissiveFactor\":[%.6g,%.6g,%.6g]", (double)(em.x / strength), (double)(em.y / strength), (double)(em.z / strength));
    if (te >= 0) tx_put(t, ",\"emissiveTexture\":{\"index\":%d}", te);
    int ext = 0;
    if (trans > 0 || (ior != 1.5f && ior > 0) || strength > 1){
        tx_put(t, ",\"extensions\":{");
        if (trans > 0){ tx_put(t, "\"KHR_materials_transmission\":{\"transmissionFactor\":%.6g}", (double)trans); ext = 1; w->use_transmission = 1; }
        if (ior != 1.5f && ior > 0){ tx_put(t, "%s\"KHR_materials_ior\":{\"ior\":%.6g}", ext ? "," : "", (double)ior); ext = 1; w->use_ior = 1; }
        if (strength > 1){ tx_put(t, "%s\"KHR_materials_emissive_strength\":{\"emissiveStrength\":%.6g}", ext ? "," : "", (double)strength); w->use_strength = 1; }
        tx_put(t, "}");
    }
    tx_put(t, "}");
    w->mat_map[mat] = i;
    return i;
}

// one primitive's triangles (3 vertices each) into the binary
static GPrim prim_data(GltfWriter *w, const float *pos, const float *nrm, const float *uv, int nv, int material){
    GPrim p = {material, add_floats(w, pos, nv, 3, 1), add_floats(w, nrm, nv, 3, 0), uv ? add_floats(w, uv, nv, 2, 0) : -1};
    return p;
}

static Geometry *new_geometry(GltfWriter *w){
    w->geoms = lr_grow(w->geoms, sizeof(Geometry), &w->cap_geoms, w->ngeoms + 1, 8);
    Geometry *g = &w->geoms[w->ngeoms];
    memset(g, 0, sizeof(*g));
    return g;
}

// the unit shapes as triangles: the box [-1, 1]^3 (a face's normal, 0..1 across each face)
// and the unit sphere (64 x 32, its normals its points)
static int shape_geometry(GltfWriter *w, int shape){
    int nv = shape == INST_BOX ? 36 : 64 * 32 * 6;
    float *pos = malloc(sizeof(float) * 3 * nv), *nrm = malloc(sizeof(float) * 3 * nv), *uv = malloc(sizeof(float) * 2 * nv);
    if (!pos || !nrm || !uv){ free(pos); free(nrm); free(uv); return -1; }
    int k = 0;
#define VTX(P, N, U, V) do { pos[3*k] = (float)(P).x; pos[3*k+1] = (float)(P).y; pos[3*k+2] = (float)(P).z; \
                             nrm[3*k] = (float)(N).x; nrm[3*k+1] = (float)(N).y; nrm[3*k+2] = (float)(N).z; \
                             uv[2*k] = (float)(U); uv[2*k+1] = (float)(V); k++; } while (0)
    if (shape == INST_BOX){
        for (int f = 0; f < 6; f++){
            Primitive q = unit_box_face(f, 0);
            Vec3 n = v_norm(v_cross(q.u, q.v));
            Vec3 a = q.q, b = v_add(q.q, q.u), c = v_add(v_add(q.q, q.u), q.v), d = v_add(q.q, q.v);
            VTX(a, n, 0, 1); VTX(b, n, 1, 1); VTX(c, n, 1, 0);
            VTX(a, n, 0, 1); VTX(c, n, 1, 0); VTX(d, n, 0, 0);
        }
    } else {
        const int NU = 64, NV = 32;
        for (int j = 0; j < NV; j++)
            for (int i = 0; i < NU; i++){
                Vec3 p[4];
                real us[4] = {(real)i / NU, (real)(i + 1) / NU, (real)(i + 1) / NU, (real)i / NU};
                real vs[4] = {(real)j / NV, (real)j / NV, (real)(j + 1) / NV, (real)(j + 1) / NV};
                for (int c = 0; c < 4; c++){
                    real th = vs[c] * (real)PI, ph = us[c] * (real)(2 * PI);
                    p[c] = vec3(r_sin(th) * r_cos(ph), r_cos(th), -r_sin(th) * r_sin(ph));
                }
                VTX(p[0], p[0], us[0], vs[0]); VTX(p[2], p[2], us[2], vs[2]); VTX(p[1], p[1], us[1], vs[1]);
                VTX(p[0], p[0], us[0], vs[0]); VTX(p[3], p[3], us[3], vs[3]); VTX(p[2], p[2], us[2], vs[2]);
            }
    }
#undef VTX
    Geometry *g = new_geometry(w);
    g->p = malloc(sizeof(GPrim));
    if (!g->p){ free(pos); free(nrm); free(uv); return -1; }
    g->p[0] = prim_data(w, pos, nrm, uv, k, -1);
    g->n = 1;
    free(pos); free(nrm); free(uv);
    return w->ngeoms++;
}

int gltf_add_geometry(GltfWriter *w, int mesh){
    if (mesh == INST_BOX || mesh == INST_SPHERE) return shape_geometry(w, mesh);
    const Scene *s = w->s;
    if (mesh < 0 || mesh >= s->nmeshes) return -1;
    const SceneMesh *m = &s->meshes[mesh];
    // its triangles, a primitive per material (in the order they first appear)
    int *mats = malloc(sizeof(int) * (size_t)(m->ntris > 0 ? m->ntris : 1)), nm = 0;
    if (!mats) return -1;
    for (int t = 0; t < m->ntris; t++){
        int seen = 0;
        for (int k = 0; k < nm && !seen; k++) seen = mats[k] == m->tris[t].material;
        if (!seen) mats[nm++] = m->tris[t].material;
    }
    Geometry *g = new_geometry(w);
    g->p = malloc(sizeof(GPrim) * (size_t)(nm > 0 ? nm : 1));
    float *pos = malloc(sizeof(float) * 9 * (size_t)(m->ntris > 0 ? m->ntris : 1));
    float *nrm = malloc(sizeof(float) * 9 * (size_t)(m->ntris > 0 ? m->ntris : 1));
    float *uv = m->nuvs > 0 ? malloc(sizeof(float) * 6 * (size_t)m->ntris) : NULL;
    if (!g->p || !pos || !nrm || (m->nuvs > 0 && !uv)){ free(mats); free(g->p); g->p = NULL; free(pos); free(nrm); free(uv); return -1; }
    for (int k = 0; k < nm; k++){
        int nv = 0;
        const Material *mt = mats[k] >= 0 && mats[k] < s->nmats ? &s->mats[mats[k]] : NULL;
        real us = mt && mt->uv_scale > 0 ? mt->uv_scale : 1;        // (no KHR_texture_transform: the scale is in the coordinates)
        for (int t = 0; t < m->ntris; t++){
            const Triangle *tr = &m->tris[t];
            if (tr->material != mats[k]) continue;
            Vec3 p[3] = {tr->q, v_add(tr->q, tr->u), v_add(tr->q, tr->v)};
            Vec3 flat = v_norm(v_cross(tr->u, tr->v));
            for (int c = 0; c < 3; c++, nv++){
                Vec3 n = tr->n[c] >= 0 && tr->n[c] < m->nnormals ? v_norm(m->normals[tr->n[c]]) : flat;
                pos[3 * nv] = (float)p[c].x; pos[3 * nv + 1] = (float)p[c].y; pos[3 * nv + 2] = (float)p[c].z;
                nrm[3 * nv] = (float)n.x; nrm[3 * nv + 1] = (float)n.y; nrm[3 * nv + 2] = (float)n.z;
                if (uv){
                    Vec2 q = tr->t[c] >= 0 && tr->t[c] < m->nuvs ? m->uvs[tr->t[c]] : vec2(0, 0);
                    uv[2 * nv] = (float)(q.x * us); uv[2 * nv + 1] = (float)(q.y * us);
                }
            }
        }
        g->p[k] = prim_data(w, pos, nrm, uv, nv, mats[k]);
    }
    g->n = nm;
    free(mats); free(pos); free(nrm); free(uv);
    return w->ngeoms++;
}

int gltf_add_mesh(GltfWriter *w, int geometry, int material){
    if (geometry < 0 || geometry >= w->ngeoms) return -1;
    const Geometry *g = &w->geoms[geometry];
    int i = tx_item(&w->meshes);
    tx_put(&w->meshes, "{\"primitives\":[");
    for (int k = 0; k < g->n; k++){
        const GPrim *p = &g->p[k];
        int mat = material_of(w, material >= 0 ? material : p->material);
        tx_put(&w->meshes, "%s{\"attributes\":{\"POSITION\":%d,\"NORMAL\":%d", k ? "," : "", p->pos, p->nrm);
        if (p->uv >= 0) tx_put(&w->meshes, ",\"TEXCOORD_0\":%d", p->uv);
        tx_put(&w->meshes, "},\"material\":%d,\"mode\":4}", mat);
    }
    tx_put(&w->meshes, "]}");
    return i;
}

int gltf_add_node(GltfWriter *w, const char *name, int mesh, const Xform *xf, const char *ext, const char *extras){
    int i = tx_item(&w->nodes);
    Text *t = &w->nodes;
    tx_put(t, "{");
    int first = 1;
    if (name){ tx_put(t, "\"name\":"); tx_string(t, name); first = 0; }
    if (mesh >= 0){ tx_put(t, "%s\"mesh\":%d", first ? "" : ",", mesh); first = 0; }
    if (xf){
        // column-major, everything times unit (the rotation and scale too: the geometry is in
        // the scene's units)
        const real k = w->unit;
        tx_put(t, "%s\"matrix\":[%.9g,%.9g,%.9g,0,%.9g,%.9g,%.9g,0,%.9g,%.9g,%.9g,0,%.9g,%.9g,%.9g,1]", first ? "" : ",",
               (double)(k * xf->m[0][0]), (double)(k * xf->m[1][0]), (double)(k * xf->m[2][0]),
               (double)(k * xf->m[0][1]), (double)(k * xf->m[1][1]), (double)(k * xf->m[2][1]),
               (double)(k * xf->m[0][2]), (double)(k * xf->m[1][2]), (double)(k * xf->m[2][2]),
               (double)(k * xf->m[0][3]), (double)(k * xf->m[1][3]), (double)(k * xf->m[2][3]));
        first = 0;
    }
    if (ext){ tx_put(t, "%s\"extensions\":{%s}", first ? "" : ",", ext); first = 0; }
    if (extras) tx_put(t, "%s\"extras\":{%s}", first ? "" : ",", extras);
    tx_put(t, "}");
    return i;
}

int gltf_add_group(GltfWriter *w, const char *name, const int *children, int n, const char *ext, const char *extras){
    int i = gltf_add_node(w, name, -1, NULL, ext, extras);
    // (the node's text ends with '}': its children go in before it)
    w->nodes.n--;
    tx_put(&w->nodes, "%s\"children\":[", w->nodes.s[w->nodes.n - 1] == '{' ? "" : ",");
    if (w->nodes.items > w->cap_child){                   // (the flags of nodes added since: 0)
        int was = w->cap_child;
        w->child = lr_grow(w->child, 1, &w->cap_child, w->nodes.items, 64);
        memset(w->child + was, 0, (size_t)(w->cap_child - was));
    }
    for (int k = 0; k < n; k++){
        tx_put(&w->nodes, "%s%d", k ? "," : "", children[k]);
        if (children[k] >= 0 && children[k] < i) w->child[children[k]] = 1;
    }
    tx_put(&w->nodes, "]}");
    return i;
}

void gltf_use_extension(GltfWriter *w, const char *name, const char *json, int required){
    int k = 0;
    while (k < w->nused && strcmp(w->used[k], name)) k++;
    if (k == w->nused && w->nused < 16) snprintf(w->used[w->nused++], sizeof(w->used[0]), "%s", name);
    if (k < 16 && required) w->required[k] = 1;
    if (json){
        if (w->root_ext.items++) tx_put(&w->root_ext, ",");
        tx_string(&w->root_ext, name);
        tx_put(&w->root_ext, ":%s", json);
    }
}

int gltf_write_glb(GltfWriter *w, const char *path){
    if (w->use_transmission) gltf_use_extension(w, "KHR_materials_transmission", NULL, 0);
    if (w->use_ior) gltf_use_extension(w, "KHR_materials_ior", NULL, 0);
    if (w->use_strength) gltf_use_extension(w, "KHR_materials_emissive_strength", NULL, 0);
    Text j = {0};
    tx_put(&j, "{\"asset\":{\"version\":\"2.0\",\"generator\":\"little_renderer\"},\"scene\":0,\"scenes\":[{\"nodes\":[");
    for (int i = 0, k = 0; i < w->nodes.items; i++)
        if (!(w->child && i < w->cap_child && w->child[i])) tx_put(&j, "%s%d", k++ ? "," : "", i);
    tx_put(&j, "]}]");
    if (w->nused){
        tx_put(&j, ",\"extensionsUsed\":[");
        for (int i = 0; i < w->nused; i++){ if (i) tx_put(&j, ","); tx_string(&j, w->used[i]); }
        tx_put(&j, "]");
        int nreq = 0;
        for (int i = 0; i < w->nused; i++) nreq += w->required[i];
        if (nreq){
            tx_put(&j, ",\"extensionsRequired\":[");
            for (int i = 0, n = 0; i < w->nused; i++) if (w->required[i]){ if (n++) tx_put(&j, ","); tx_string(&j, w->used[i]); }
            tx_put(&j, "]");
        }
    }
    if (w->root_ext.items) tx_put(&j, ",\"extensions\":{%s}", w->root_ext.s);
    struct { const char *key; Text *t; } arrays[] = {{"nodes", &w->nodes}, {"meshes", &w->meshes}, {"materials", &w->materials},
        {"textures", &w->textures}, {"images", &w->images}, {"samplers", &w->samplers}, {"accessors", &w->accessors}, {"bufferViews", &w->views}};
    for (size_t i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++)
        if (arrays[i].t->items) tx_put(&j, ",\"%s\":[%s]", arrays[i].key, arrays[i].t->s);
    bin_pad(&w->bin);
    if (w->bin.n) tx_put(&j, ",\"buffers\":[{\"byteLength\":%zu}]", w->bin.n);
    tx_put(&j, "}");
    while (j.n % 4) tx_put(&j, " ");
    // the file: a header, the JSON chunk, the binary chunk
    FILE *f = fopen(path, "wb");
    if (!f){ free(j.s); return 1; }
    uint32_t total = (uint32_t)(12 + 8 + j.n + (w->bin.n ? 8 + w->bin.n : 0));
    uint32_t head[3] = {0x46546C67u, 2, total};                // "glTF", version 2, length
    uint32_t jc[2] = {(uint32_t)j.n, 0x4E4F534Au};             // "JSON"
    uint32_t bc[2] = {(uint32_t)w->bin.n, 0x004E4942u};        // "BIN\0"
    int ok = fwrite(head, 4, 3, f) == 3 && fwrite(jc, 4, 2, f) == 2 && fwrite(j.s, 1, j.n, f) == j.n;
    if (ok && w->bin.n) ok = fwrite(bc, 4, 2, f) == 2 && fwrite(w->bin.d, 1, w->bin.n, f) == w->bin.n;
    free(j.s);
    return fclose(f) || !ok;
}
