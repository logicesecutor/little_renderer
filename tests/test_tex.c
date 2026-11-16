// test_tex.c - the materials' textures (core/texture.h, core/scene.h; docs/PBR.md, PBR5)
// checked on their own.
//
//   make bin/test_tex && bin/test_tex      (part of make test)
//
// 1. Mip levels: each half the one above, down to 1 x 1 (odd sides too); the last level is
//    the image's mean - in linear light for colour, so a black and white checker averages to
//    0.5 of the light (sRGB 188), not to sRGB 128 (0.22 of it).
// 2. Lookups: bilinear filtering is exact at texel centres and halfway between them; the
//    coordinates repeat (u + 1, v - 2 are the same point); trilinear's level follows the
//    footprint (texmap_lod: one texel per pixel -> 0, 4 -> 2; the longer side counts).
// 3. Tangents (scene_compute_tangents): along +u, and their handedness w such that
//    w (n x t) points up the image - for mirrored texture coordinates too.
// 4. Normal maps (material_normal): a map tilted up the image (green > 0.5) bends the normal
//    up the image, mirrored or not; a flat one leaves it alone. The procedural bricks' map:
//    the top of each brick leans up the image, the bottom down.
// 5. Materials: material_at applies the base colour (times the factor), metallic and
//    roughness (green and blue, as glTF) and uv_scale; scene_texture loads a name once.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "core/texture.h"
#include "core/scene.h"
#include "core/rng.h"

static int failures;

static void check(const char *what, double got, double want, double tol){
    double err = fabs(got - want);
    int ok = err <= tol;
    printf("  %-58s %10.5f  (want %.5f +- %.5f)  %s\n", what, got, want, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static void test_mips(void){
    printf("1. mip levels\n");
    int w = 300, h = 37;
    unsigned char *px = malloc((size_t)w * h * 4);
    Rng rng;
    rng_seed(&rng, 7, 1);
    double sum[4] = {0};
    const float *lut = texmap_srgb_table();
    for (int i = 0; i < w * h * 4; i++){
        px[i] = (unsigned char)(rng_real(&rng) * 256);
        sum[i % 4] += i % 4 < 3 ? lut[px[i]] : px[i] / 255.0;
    }
    TexMap t;
    if (texmap_make(&t, px, w, h, 1)){ printf("  texmap_make failed  FAIL\n"); failures++; return; }
    int halving = 1;
    for (int k = 1; k < t.levels; k++)
        halving &= t.lw[k] == (t.lw[k - 1] > 1 ? t.lw[k - 1] / 2 : 1) && t.lh[k] == (t.lh[k - 1] > 1 ? t.lh[k - 1] / 2 : 1);
    check("levels of 300 x 37 (300 150 75 37 18 9 4 2 1)", t.levels, 9, 0);
    check("each level half the one above", halving, 1, 0);
    check("the last level is 1 x 1", t.lw[t.levels - 1] * t.lh[t.levels - 1], 1, 0);
    // the odd sides repeat their last texel, so the mean is that of a slightly different
    // image: compare with a power of two too
    free(px);
    w = 64; h = 32;
    px = malloc((size_t)w * h * 4);
    memset(sum, 0, sizeof(sum));
    for (int i = 0; i < w * h * 4; i++){
        px[i] = (unsigned char)(rng_real(&rng) * 256);
        sum[i % 4] += i % 4 < 3 ? lut[px[i]] : px[i] / 255.0;
    }
    texmap_free(&t);
    texmap_make(&t, px, w, h, 1);
    Vec4 last = texmap_texel(&t, t.levels - 1, 0, 0);
    check("64 x 32 sRGB: last level red = the mean (linear)", last.x, sum[0] / (w * h), 0.01);
    check("64 x 32 sRGB: last level alpha = the mean", last.w, sum[3] / (w * h), 0.01);
    texmap_free(&t);
    // a 2 x 2 black and white checker
    unsigned char ck[16];
    for (int i = 0; i < 4; i++){
        unsigned char c = (i == 0 || i == 3) ? 255 : 0;
        ck[4 * i] = ck[4 * i + 1] = ck[4 * i + 2] = c; ck[4 * i + 3] = 255;
    }
    texmap_make(&t, ck, 2, 2, 1);
    check("black / white checker, sRGB: level 1 = half the light", texmap_texel(&t, 1, 0, 0).x, 0.5, 0.005);
    check("... stored as sRGB 188", t.px[t.at[1]] & 255, 188, 1);
    texmap_free(&t);
    texmap_make(&t, ck, 2, 2, 0);
    check("the same as data (not sRGB): level 1 = 0.5", texmap_texel(&t, 1, 0, 0).x, 0.5, 0.005);
    texmap_free(&t);
    free(px);
}

static void test_lookups(void){
    printf("2. lookups\n");
    int w = 8, h = 4;
    unsigned char px[8 * 4 * 4];
    for (int i = 0; i < w * h; i++){
        px[4 * i] = (unsigned char)(i * 7 % 256); px[4 * i + 1] = (unsigned char)(i * 31 % 256);
        px[4 * i + 2] = (unsigned char)(255 - i * 5); px[4 * i + 3] = 255;
    }
    TexMap t;
    texmap_make(&t, px, w, h, 0);
    double e_centre = 0, e_half = 0, e_wrap = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            Vec4 a = texmap_texel(&t, 0, x, y);
            Vec4 c = texmap_bilinear(&t, 0, (x + 0.5f) / w, (y + 0.5f) / h);
            e_centre = fmax(e_centre, fabs(a.x - c.x) + fabs(a.y - c.y) + fabs(a.z - c.z));
            Vec4 b = texmap_texel(&t, 0, x + 1, y);           // wraps at the right edge
            Vec4 m = texmap_bilinear(&t, 0, (x + 1.0f) / w, (y + 0.5f) / h);
            e_half = fmax(e_half, fabs((a.y + b.y) / 2 - m.y));
        }
    Rng rng;
    rng_seed(&rng, 3, 1);
    for (int i = 0; i < 1000; i++){
        real u = rng_real(&rng), v = rng_real(&rng), lod = 3 * rng_real(&rng);
        Vec4 a = texmap_sample(&t, u, v, lod), b = texmap_sample(&t, u + 1, v - 2, lod), c = texmap_sample(&t, u - 3, v + 5, lod);
        e_wrap = fmax(e_wrap, fmax(fabs(a.y - b.y), fabs(a.y - c.y)));
    }
    check("bilinear at texel centres = the texels", e_centre, 0, 1e-5);
    check("bilinear halfway = the mean of the two (and wraps)", e_half, 0, 1e-5);
    check("u + 1, v - 2 (and u - 3, v + 5) read the same", e_wrap, 0, 2e-3);
    check("lod: 1 texel per pixel", texmap_lod(&t, 1.0f / w, 0, 0, 1.0f / h), 0, 1e-5);
    check("lod: 4 texels per pixel across", texmap_lod(&t, 4.0f / w, 0, 0, 1.0f / h), 2, 1e-4);
    check("lod: the longer side (8 down, 2 across)", texmap_lod(&t, 2.0f / w, 0, 0, 8.0f / h), 3, 1e-4);
    check("lod: under a texel -> level 0", texmap_lod(&t, 0.1f / w, 0, 0, 0.1f / h), 0, 0);
    Vec4 top = texmap_texel(&t, t.levels - 1, 0, 0);
    check("trilinear past the last level = the last level", texmap_sample(&t, 0.3f, 0.7f, 20).y, top.y, 1e-5);
    texmap_free(&t);
}

// a 1 x 1 quad in the plane z = 0 (normal +z) as two triangles, its texture coordinates
// with u along +x (or -x: mirrored) and v down the image along -y
static void quad(Triangle *tr, Vec2 *uv, int mirror){
    Vec3 p[4] = {vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 1, 0), vec3(0, 1, 0)};
    for (int i = 0; i < 4; i++) uv[i] = vec2(mirror ? 1 - p[i].x : p[i].x, 1 - p[i].y);
    int ix[2][3] = {{0, 1, 2}, {0, 2, 3}};
    for (int k = 0; k < 2; k++){
        tr[k].q = p[ix[k][0]];
        tr[k].u = v_sub(p[ix[k][1]], p[ix[k][0]]);
        tr[k].v = v_sub(p[ix[k][2]], p[ix[k][0]]);
        for (int j = 0; j < 3; j++){ tr[k].n[j] = -1; tr[k].t[j] = ix[k][j]; }
        tr[k].material = 0;
    }
}

static void test_tangents_normals(void){
    printf("3. tangents\n");
    Triangle tr[2];
    Vec2 uv[4];
    Vec4 tg[4];
    for (int mirror = 0; mirror < 2; mirror++){
        quad(tr, uv, mirror);
        scene_compute_tangents(tr, 2, uv, tg, 0, 4);
        double ex = 0, ew = 0;
        for (int i = 0; i < 4; i++){
            ex = fmax(ex, fabs(tg[i].x - (mirror ? -1 : 1)) + fabs(tg[i].y) + fabs(tg[i].z));
            Vec3 b = v_scale(v_cross(vec3(0, 0, 1), vec3(tg[i].x, tg[i].y, tg[i].z)), tg[i].w);
            ew = fmax(ew, fabs(b.y - 1));                  // up the image = +y
        }
        check(mirror ? "mirrored: tangent along +u (= -x)" : "tangent along +u (= +x)", ex, 0, 1e-5);
        check(mirror ? "mirrored: w (n x t) up the image" : "w (n x t) up the image", ew, 0, 1e-5);
    }

    printf("4. normal maps\n");
    // a constant normal map tilted up the image, and a flat one
    unsigned char up[4] = {128, 204, 220, 255}, flat[4] = {128, 128, 255, 255};
    TexMap maps[2];
    texmap_make(&maps[0], up, 1, 1, 0);
    texmap_make(&maps[1], flat, 1, 1, 0);
    Scene s;
    memset(&s, 0, sizeof(s));
    s.texmaps = maps; s.ntexmaps = 2;
    Material m = material_pbr(vec3(1, 1, 1), 0, 1, 1.5f, 0);
    for (int mirror = 0; mirror < 2; mirror++){
        quad(tr, uv, mirror);
        scene_compute_tangents(tr, 2, uv, tg, 0, 4);
        m.tex_normal = 1;
        Vec3 n = material_normal(&s, &m, vec2(0.5f, 0.5f), MATERIAL_LOD0, vec3(0, 0, 1), tg[0]);
        check(mirror ? "mirrored: tilted map bends the normal up (+y)" : "tilted map bends the normal up (+y)", n.y > 0.3 && fabs(n.x) < 0.01, 1, 0);
        m.tex_normal = 2;
        n = material_normal(&s, &m, vec2(0.5f, 0.5f), MATERIAL_LOD0, vec3(0, 0, 1), tg[0]);
        check(mirror ? "mirrored: flat map leaves it" : "flat map leaves it", fabs(n.x) + fabs(n.y) + fabs(n.z - 1), 0, 2e-2);
    }
    texmap_free(&maps[0]); texmap_free(&maps[1]);
    // the bricks: rows of 64 texels; in row 1, at x = 128 (between its joints), the green
    // channel over the row's top quarter leans up, over its bottom quarter down
    TexMap b;
    if (texmap_procedural(&b, "bricks_normal")){ printf("  no bricks_normal  FAIL\n"); failures++; return; }
    double topg = 0, botg = 0;
    int rh = b.h / 8;
    for (int y = 0; y < rh / 4; y++){
        topg += texmap_texel(&b, 0, 128, rh + y).y - 0.5;
        botg += texmap_texel(&b, 0, 128, 2 * rh - 1 - y).y - 0.5;
    }
    check("bricks: the top of a brick leans up the image", topg > 0.5, 1, 0);
    check("bricks: its bottom leans down", botg < -0.5, 1, 0);
    texmap_free(&b);
}

static void test_materials(void){
    printf("5. materials\n");
    Scene *s = calloc(1, sizeof(Scene));
    int c = scene_texture(s, "checker", 1);
    int c2 = scene_texture(s, "checker", 1);
    int mr = scene_texture(s, "plate_mr", 0);
    check("scene_texture: a name is loaded once", c == c2 && c > 0 && mr > 0 && s->ntexmaps == 2, 1, 0);
    check("scene_texture: an unknown name -> 0", scene_texture(s, "no such texture", 1), 0, 0);
    Material m = material_pbr(vec3(0.5f, 1, 1), 1, 1, 1.5f, 0);
    m.tex_base = c; m.tex_mr = mr;
    TexMap *tc = &s->texmaps[c - 1], *tm = &s->texmaps[mr - 1];
    Vec2 at = vec2(0.37f, 0.61f);
    Material a = material_at(s, &m, at, MATERIAL_LOD0);
    Vec4 tex = texmap_sample(tc, at.x, at.y, 0), mrv = texmap_sample(tm, at.x, at.y, 0);
    check("base = texture x factor", a.albedo.x, 0.5 * tex.x, 1e-4);
    check("roughness = green", a.roughness, mrv.y, 1e-4);
    check("metallic = blue", a.metallic, mrv.z, 1e-4);
    m.uv_scale = 4;
    a = material_at(s, &m, vec2(at.x / 4, at.y / 4), MATERIAL_LOD0);
    check("uv_scale: (u, v) / 4 with uv_scale 4 reads (u, v)", a.albedo.x, 0.5 * tex.x, 1e-4);
    scene_free(s);
    free(s);
}

int main(void){
    test_mips();
    test_lookups();
    test_tangents_normals();
    test_materials();
    if (failures){ printf("test_tex: %d FAILED\n", failures); return 1; }
    printf("test_tex: all passed\n");
    return 0;
}
