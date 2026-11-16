// test_gltf.c - scene/gltf.c (and formats/json.c) on models made here.
//
//   make bin/test_gltf && bin/test_gltf      (part of make test)
//
// A small model is written out as .gltf (its buffer and a PNG texture in data: URIs, then as
// files beside it) and as .glb, and loaded into a scene: an indexed quad (interleaved
// positions and normals, a byte stride) used by two nodes (translation / rotation / scale,
// and a child with a matrix), a triangle strip with byte indices, a material with every
// factor and extension read, a texture with a clamping sampler.
// 1. JSON: values, escapes, errors.
// 2. What arrives: instances, triangles, where they stand (the nodes' transforms and the
//    placement), normals, texture coordinates (not turned over), tangents, the material,
//    the texture's pixels and wrap modes - the same from the three files.
// 3. Broken files are refused, not crashed on.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "formats/json.h"
#include "scene/gltf.h"
#include "formats/png.h"

static int failures;

static void check(const char *what, int ok){
    printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}
static int near(double a, double b, double tol){ return fabs(a - b) <= tol; }

static void test_json(void){
    printf("1. JSON\n");
    const char *t = "{\"a\": [1, -2.5e3, true, false, null], \"s\": \"x\\u00e9\\ud83d\\ude00\\n\\\"\", \"o\": {\"k\": {}}}";
    Json *j = json_parse(t, strlen(t));
    check("numbers, literals, nesting", j && json_len(json_get(j, "a")) == 5 && json_num(json_at(json_get(j, "a"), 1), 0) == -2500
          && json_at(json_get(j, "a"), 2)->type == JSON_TRUE && json_get(json_get(j, "o"), "k")->type == JSON_OBJECT);
    check("escapes, \\u and a surrogate pair to UTF-8", j && !strcmp(json_str(json_get(j, "s"), ""), "x\xc3\xa9\xf0\x9f\x98\x80\n\""));
    check("missing keys and wrong types give the defaults", j && json_num(json_get(j, "zz"), 7) == 7 && json_str(json_get(j, "a"), "d")[0] == 'd'
          && json_get(NULL, "a") == NULL && json_len(json_get(j, "s")) == 0);
    json_free(j);
    const char *bad[] = {"{\"a\":}", "[1,2", "{\"a\" 1}", "\"abc", "[1] x", "{\"a\":[1,{\"b\":tru}]}", "", "[\"\\q\"]"};
    int refused = 0;
    for (int i = 0; i < 8; i++){ Json *b = json_parse(bad[i], strlen(bad[i])); refused += !b; json_free(b); }
    check("8 broken texts refused (messages on stderr)", refused == 8);
}

// ---- the model ---------------------------------------------------------------------------

static char *b64(const unsigned char *d, size_t n){
    static const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *o = malloc(n / 3 * 4 + 8), *p = o;
    for (size_t i = 0; i < n; i += 3){
        unsigned v = d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        *p++ = A[v >> 18 & 63]; *p++ = A[v >> 12 & 63];
        *p++ = i + 1 < n ? A[v >> 6 & 63] : '=';
        *p++ = i + 2 < n ? A[v & 63] : '=';
    }
    *p = 0;
    return o;
}

// the buffer: the quad's interleaved positions + normals (24 bytes a vertex), its uvs, its
// ushort indices, the strip's positions and byte indices
static unsigned char buf[512];
static size_t buf_n;
static size_t put(const void *d, size_t n){ size_t at = buf_n; memcpy(buf + at, d, n); buf_n += (n + 3) & ~(size_t)3; return at; }

static unsigned char tex_px[4 * 4 * 4];
static unsigned char *png;
static size_t png_n;

static char *model_json(const char *buffer_uri, const char *image_uri, int glb){
    float pn[4][6] = {{0, 0, 0, 0, 0, 2}, {2, 0, 0, 0, 0, 2}, {2, 1, 0, 0, 0, 2}, {0, 1, 0, 0, 0, 2}};   // normals not unit
    float uv[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    unsigned short ix[6] = {0, 1, 2, 0, 2, 3};
    float sp[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    unsigned char sx[4] = {0, 1, 2, 3};
    buf_n = 0;
    size_t o_pn = put(pn, sizeof(pn)), o_uv = put(uv, sizeof(uv)), o_ix = put(ix, sizeof(ix));
    size_t o_sp = put(sp, sizeof(sp)), o_sx = put(sx, sizeof(sx));
    size_t o_png = glb ? put(png, png_n) : 0;
    char *j = malloc(1 << 16);
    char imgs[8192];
    if (glb) snprintf(imgs, sizeof(imgs), "{\"bufferView\": 6, \"mimeType\": \"image/png\"}");
    else snprintf(imgs, sizeof(imgs), "{\"uri\": \"%s\"}", image_uri);
    char bufs[4096];
    if (glb) snprintf(bufs, sizeof(bufs), "{\"byteLength\": %zu}", buf_n);
    else snprintf(bufs, sizeof(bufs), "{\"uri\": \"%s\", \"byteLength\": %zu}", buffer_uri, buf_n);
    snprintf(j, 1 << 16,
        "{\"asset\": {\"version\": \"2.0\"}, \"scene\": 0, \"scenes\": [{\"nodes\": [0, 3]}],\n"
        " \"nodes\": [{\"translation\": [1, 0, 0], \"rotation\": [0, 0.7071068, 0, 0.7071068], \"scale\": [2, 2, 2], \"mesh\": 0, \"children\": [1]},\n"
        "   {\"matrix\": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,3,0,1], \"mesh\": 0, \"children\": [2]},\n"
        "   {\"name\": \"empty\"}, {\"mesh\": 1, \"translation\": [0, 0, -4]}],\n"
        " \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0, \"NORMAL\": 1, \"TEXCOORD_0\": 2}, \"indices\": 3, \"material\": 0}]},\n"
        "   {\"primitives\": [{\"attributes\": {\"POSITION\": 4}, \"indices\": 5, \"mode\": 5}]}],\n"
        " \"accessors\": [{\"bufferView\": 0, \"componentType\": 5126, \"count\": 4, \"type\": \"VEC3\", \"min\": [0,0,0], \"max\": [2,1,0]},\n"
        "   {\"bufferView\": 0, \"byteOffset\": 12, \"componentType\": 5126, \"count\": 4, \"type\": \"VEC3\"},\n"
        "   {\"bufferView\": 1, \"componentType\": 5126, \"count\": 4, \"type\": \"VEC2\"},\n"
        "   {\"bufferView\": 2, \"componentType\": 5123, \"count\": 6, \"type\": \"SCALAR\"},\n"
        "   {\"bufferView\": 3, \"componentType\": 5126, \"count\": 4, \"type\": \"VEC3\"},\n"
        "   {\"bufferView\": 4, \"componentType\": 5121, \"count\": 4, \"type\": \"SCALAR\"}],\n"
        " \"bufferViews\": [{\"buffer\": 0, \"byteOffset\": %zu, \"byteLength\": 96, \"byteStride\": 24},\n"
        "   {\"buffer\": 0, \"byteOffset\": %zu, \"byteLength\": 32}, {\"buffer\": 0, \"byteOffset\": %zu, \"byteLength\": 12},\n"
        "   {\"buffer\": 0, \"byteOffset\": %zu, \"byteLength\": 48}, {\"buffer\": 0, \"byteOffset\": %zu, \"byteLength\": 4},\n"
        "   {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 4}, {\"buffer\": 0, \"byteOffset\": %zu, \"byteLength\": %zu}],\n"
        " \"buffers\": [%s],\n"
        " \"materials\": [{\"pbrMetallicRoughness\": {\"baseColorFactor\": [0.5, 0.25, 1, 1], \"metallicFactor\": 0.25, \"roughnessFactor\": 0.75,\n"
        "     \"baseColorTexture\": {\"index\": 0}}, \"normalTexture\": {\"index\": 0, \"scale\": 0.5}, \"occlusionTexture\": {\"index\": 0, \"strength\": 0.3},\n"
        "     \"emissiveFactor\": [1, 0.5, 0], \"extensions\": {\"KHR_materials_emissive_strength\": {\"emissiveStrength\": 4},\n"
        "     \"KHR_materials_ior\": {\"ior\": 1.33}, \"KHR_materials_transmission\": {\"transmissionFactor\": 0.6}}}],\n"
        " \"textures\": [{\"source\": 0, \"sampler\": 0}], \"samplers\": [{\"wrapS\": 33071, \"wrapT\": 33648}],\n"
        " \"images\": [%s]}",
        o_pn, o_uv, o_ix, o_sp, o_sx, o_png, glb ? png_n : (size_t)0, bufs, imgs);
    return j;
}

static void write_file(const char *path, const void *d, size_t n){
    FILE *f = fopen(path, "wb");
    if (f){ fwrite(d, 1, n, f); fclose(f); }
}

static int load(Scene *s, const char *path){
    memset(s, 0, sizeof(*s));
    scene_add_material(s, material_diffuse(vec3(1, 1, 1)));     // (a scene's own material first)
    return scene_load_gltf(s, path, 330, vec3(278, 0, 278), 0);
}

// the model as loaded: everything the test knows about it
static void check_model(const char *what, const char *path){
    printf("2. %s\n", what);
    Scene s;
    int n = load(&s, path);
    check("3 instances (one mesh in two nodes, the strip in a third)", n == 3 && s.ninsts == 3 && s.nmeshes == 2);
    if (n != 3){ scene_free(&s); return; }
    SceneMesh *q = &s.meshes[s.insts[0].mesh], *st = &s.meshes[s.insts[2].mesh];
    check("the quad: 2 triangles; the strip of 4 vertices: 2", q->ntris == 2 && st->ntris == 2 && s.insts[1].mesh == s.insts[0].mesh);
    check("normals: unit, along +z", q->nnormals == 4 && near(q->normals[0].z, 1, 1e-6) && near(v_len(q->normals[2]), 1, 1e-6));
    check("texture coordinates as stored (v down, not turned over)", q->nuvs == 4 && near(q->uvs[0].y, 1, 0) && near(q->uvs[3].y, 0, 0));
    // tangent along +u = +x; up the image (-v) = +y, n = +z: w (n x t) = +y, w = 1
    check("tangents computed: +u, handedness 1", near(q->tangents[0].x, 1, 1e-5) && q->tangents[0].w == 1);
    // placement: the world box of all three, 330 on its longest side, standing at y = 0
    Aabb b = s.insts[0].box;
    for (int i = 1; i < 3; i++){
        Aabb c = s.insts[i].box;
        b.lo = vec3(r_fmin(b.lo.x, c.lo.x), r_fmin(b.lo.y, c.lo.y), r_fmin(b.lo.z, c.lo.z));
        b.hi = vec3(r_fmax(b.hi.x, c.hi.x), r_fmax(b.hi.y, c.hi.y), r_fmax(b.hi.z, c.hi.z));
    }
    real ext = r_fmax(b.hi.x - b.lo.x, r_fmax(b.hi.y - b.lo.y, b.hi.z - b.lo.z));
    check("placed: 330 at its longest, on the floor, centred", near(ext, 330, 0.5) && near(b.lo.y, 0, 0.5)
          && near((b.lo.x + b.hi.x) / 2, 278, 0.5) && near((b.lo.z + b.hi.z) / 2, 278, 0.5));
    // the model's own units: node 0 = T(1,0,0) R_y(90) S(2): its +x goes to -z, 2 long; node 1 is
    // 3 units up in node 0's space (6 in the world); the strip 4 units back. The scale k comes
    // from the model's extent: x 0..1 (strip) .. node 0 (x 1..1, z -4..0) -> z -4..0, y 0..8
    real k = 330.0f / 8;
    Vec3 a0 = xform_point(&s.insts[0].xf, vec3(0, 0, 0)), a1 = xform_point(&s.insts[0].xf, vec3(1, 0, 0));
    Vec3 c0 = xform_point(&s.insts[1].xf, vec3(0, 0, 0));
    check("node 0: translation, rotation, scale", near(a1.x - a0.x, 0, 1e-3) && near(a1.z - a0.z, -2 * k, 1e-2) && near(a1.y - a0.y, 0, 1e-3));
    check("node 1: its matrix, under node 0", near(c0.y - a0.y, 6 * k, 1e-2) && near(c0.x - a0.x, 0, 1e-3));
    Vec3 s0 = xform_point(&s.insts[2].xf, vec3(0, 0, 0));
    check("node 3: the strip 4 units back, beside node 0", near(s0.z - a0.z, -4 * k, 1e-2) && near(s0.x - a0.x, -k, 1e-2));
    // the material
    const Material *m = &s.mats[q->tris[0].material];
    check("material: base colour, metallic, roughness", m->type == MAT_PBR && near(m->albedo.x, 0.5, 1e-6) && near(m->albedo.y, 0.25, 1e-6)
          && near(m->metallic, 0.25, 1e-6) && near(m->roughness, 0.75, 1e-6));
    check("emission x KHR_materials_emissive_strength; ior; transmission", near(m->emission.x, 4, 1e-5) && near(m->emission.y, 2, 1e-5)
          && near(m->ior, 1.33, 1e-6) && near(m->transmission, 0.6, 1e-6));
    check("normal scale, occlusion strength", near(m->normal_scale, 0.5, 1e-6) && near(m->occlusion, 0.3, 1e-6));
    const Material *ms = &s.mats[st->tris[0].material];
    check("a primitive without a material: glTF's default (white, metallic 1, rough 1)",
          ms->type == MAT_PBR && ms->albedo.x == 1 && ms->metallic == 1 && ms->roughness == 1 && ms != m);
    // the texture: decoded once as colour (sRGB) and once as data (normal, occlusion)
    int ok = m->tex_base > 0 && m->tex_normal > 0 && m->tex_occlusion == m->tex_normal && m->tex_base != m->tex_normal && s.ntexmaps == 2;
    check("texture: one sRGB, one linear (shared by the normal and occlusion)", ok);
    if (ok){
        const TexMap *t = &s.texmaps[m->tex_normal - 1];
        check("its pixels, its sampler (clamp across, mirror down)", t->w == 4 && t->h == 4 && !memcmp(t->px, tex_px, sizeof(tex_px))
              && t->wrap_u == TEXMAP_CLAMP && t->wrap_v == TEXMAP_MIRROR && s.texmaps[m->tex_base - 1].srgb == 1 && t->srgb == 0);
    }
    scene_free(&s);
}

int main(void){
    test_json();
    for (int i = 0; i < 16; i++){
        tex_px[4 * i] = (unsigned char)(i * 16); tex_px[4 * i + 1] = (unsigned char)(255 - i * 9);
        tex_px[4 * i + 2] = (unsigned char)(i * 5); tex_px[4 * i + 3] = 255;
    }
    png = png_encode(4, 4, 4, tex_px, &png_n);
    // .gltf, everything in data: URIs
    char *j0 = model_json("", "", 0);                     // (fills buf: its base64 next)
    free(j0);
    char *bu = b64(buf, buf_n), *iu = b64(png, png_n);
    char *duri = malloc(strlen(bu) + 64), *iuri = malloc(strlen(iu) + 64);
    sprintf(duri, "data:application/octet-stream;base64,%s", bu);
    sprintf(iuri, "data:image/png;base64,%s", iu);
    char *j = model_json(duri, iuri, 0);
    write_file("out/test_gltf_embedded.gltf", j, strlen(j));
    free(j);
    check_model(".gltf with data: URIs", "out/test_gltf_embedded.gltf");
    // .gltf with its buffer and image in files beside it (a space in a name: %20)
    j = model_json("test%20gltf.bin", "test_gltf.png", 0);
    write_file("out/test_gltf_files.gltf", j, strlen(j));
    write_file("out/test gltf.bin", buf, buf_n);
    write_file("out/test_gltf.png", png, png_n);
    free(j);
    check_model(".gltf with files", "out/test_gltf_files.gltf");
    // .glb: the JSON chunk (padded with spaces) and the BIN chunk (the image inside)
    j = model_json(NULL, NULL, 1);
    size_t jn = (strlen(j) + 3) & ~(size_t)3, bn = (buf_n + 3) & ~(size_t)3, total = 12 + 8 + jn + 8 + bn;
    unsigned char *g = calloc(total, 1);
    uint32_t hdr[5] = {0x46546C67u, 2, (uint32_t)total, (uint32_t)jn, 0x4E4F534Au};
    memcpy(g, hdr, 20);
    memset(g + 20, ' ', jn);
    memcpy(g + 20, j, strlen(j));
    uint32_t bh[2] = {(uint32_t)bn, 0x004E4942u};
    memcpy(g + 20 + jn, bh, 8);
    memcpy(g + 28 + jn, buf, buf_n);
    write_file("out/test_gltf.glb", g, total);
    check_model(".glb", "out/test_gltf.glb");
    // 3. broken
    printf("3. broken files (messages on stderr expected)\n");
    int refused = 0, tries = 0;
    Scene s;
    for (size_t cut = 0; cut < total; cut += total / 23 + 1){                 // truncated
        write_file("out/test_gltf_bad.glb", g, cut);
        int n = load(&s, "out/test_gltf_bad.glb");
        refused += n < 0;
        tries++;
        scene_free(&s);
    }
    const char *bad_json[] = {
        "{\"asset\": {\"version\": \"1.0\"}}",
        "{\"asset\": {\"version\": \"2.0\"}, \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0}}]}], \"nodes\": [{\"mesh\": 0}],"
        " \"accessors\": [{\"bufferView\": 0, \"componentType\": 5126, \"count\": 9, \"type\": \"VEC3\"}],"
        " \"bufferViews\": [{\"buffer\": 0, \"byteLength\": 12}], \"buffers\": [{\"uri\": \"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA\", \"byteLength\": 12}]}",
        "{\"asset\": {\"version\": \"2.0\"}, \"extensionsRequired\": [\"KHR_draco_mesh_compression\"]}",
        "{\"asset\": {\"version\": \"2.0\"}, \"nodes\": [{\"children\": [0]}]}",
        // offsets that once wrapped the bounds check: negative, fractional, a huge stride
        "{\"asset\": {\"version\": \"2.0\"}, \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0}}]}], \"nodes\": [{\"mesh\": 0}],"
        " \"accessors\": [{\"bufferView\": 0, \"componentType\": 5126, \"count\": 1, \"type\": \"VEC3\"}],"
        " \"bufferViews\": [{\"buffer\": 0, \"byteOffset\": -4096, \"byteLength\": 4108}], \"buffers\": [{\"uri\": \"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA\", \"byteLength\": 12}]}",
        "{\"asset\": {\"version\": \"2.0\"}, \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0}}]}], \"nodes\": [{\"mesh\": 0}],"
        " \"accessors\": [{\"bufferView\": 0, \"componentType\": 5126, \"count\": 1, \"type\": \"VEC3\", \"byteOffset\": 0.5}],"
        " \"bufferViews\": [{\"buffer\": 0, \"byteLength\": 12}], \"buffers\": [{\"uri\": \"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA\", \"byteLength\": 12}]}",
        "{\"asset\": {\"version\": \"2.0\"}, \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0}}]}], \"nodes\": [{\"mesh\": 0}],"
        " \"accessors\": [{\"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"}],"
        " \"bufferViews\": [{\"buffer\": 0, \"byteLength\": 12, \"byteStride\": 1e19}], \"buffers\": [{\"uri\": \"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA\", \"byteLength\": 12}]}",
    };
    for (int i = 0; i < 7; i++){
        write_file("out/test_gltf_bad.gltf", bad_json[i], strlen(bad_json[i]));
        int n = load(&s, "out/test_gltf_bad.gltf");
        refused += n < 0;
        tries++;
        scene_free(&s);
    }
    char d[96];
    snprintf(d, sizeof(d), "truncated, version, past the buffer, Draco, loop, bad offsets: %d of %d refused", refused, tries);
    check(d, refused == tries);
    free(g); free(j); free(bu); free(iu); free(duri); free(iuri); free(png);
    if (failures){ printf("test_gltf: %d FAILED\n", failures); return 1; }
    printf("all glTF tests passed\n");
    return 0;
}
