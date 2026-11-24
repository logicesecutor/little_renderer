// test_tex.c - the materials' textures (scene/texture.h, scene/scene.h; docs/PBR.md, PBR5)
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
// 6. Artists' maps made into glTF's (scene/pbr_maps.h, PBR7): a height ramp rising to the
//    right gives normals leaning left by atan(strength / 2) per texel step (a ramp up the
//    image: leaning down), a flat height a flat normal, the same tilt from a wider map over
//    the same part of the image; normal maps combined (a flat detail leaves the base, a flat
//    base gives the detail, two tilts add up); green turned over; roughness and metallic
//    packed into green and blue (the larger size, the missing one white); and through
//    material_normal the baked bump bends a quad's normal the way the height rises.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "scene/texture.h"
#include "scene/scene.h"
#include "scene/pbr_maps.h"
#include "base/rng.h"
#include "formats/png.h"
#include "test_util.h"

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
    check_near("levels of 300 x 37 (300 150 75 37 18 9 4 2 1)", t.levels, 9, 0);
    check_near("each level half the one above", halving, 1, 0);
    check_near("the last level is 1 x 1", t.lw[t.levels - 1] * t.lh[t.levels - 1], 1, 0);
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
    check_near("64 x 32 sRGB: last level red = the mean (linear)", last.x, sum[0] / (w * h), 0.01);
    check_near("64 x 32 sRGB: last level alpha = the mean", last.w, sum[3] / (w * h), 0.01);
    texmap_free(&t);
    // a 2 x 2 black and white checker
    unsigned char ck[16];
    for (int i = 0; i < 4; i++){
        unsigned char c = (i == 0 || i == 3) ? 255 : 0;
        ck[4 * i] = ck[4 * i + 1] = ck[4 * i + 2] = c; ck[4 * i + 3] = 255;
    }
    texmap_make(&t, ck, 2, 2, 1);
    check_near("black / white checker, sRGB: level 1 = half the light", texmap_texel(&t, 1, 0, 0).x, 0.5, 0.005);
    check_near("... stored as sRGB 188", t.px[t.at[1]] & 255, 188, 1);
    texmap_free(&t);
    texmap_make(&t, ck, 2, 2, 0);
    check_near("the same as data (not sRGB): level 1 = 0.5", texmap_texel(&t, 1, 0, 0).x, 0.5, 0.005);
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
    check_near("bilinear at texel centres = the texels", e_centre, 0, 1e-5);
    check_near("bilinear halfway = the mean of the two (and wraps)", e_half, 0, 1e-5);
    check_near("u + 1, v - 2 (and u - 3, v + 5) read the same", e_wrap, 0, 2e-3);
    check_near("lod: 1 texel per pixel", texmap_lod(&t, 1.0f / w, 0, 0, 1.0f / h), 0, 1e-5);
    check_near("lod: 4 texels per pixel across", texmap_lod(&t, 4.0f / w, 0, 0, 1.0f / h), 2, 1e-4);
    check_near("lod: the longer side (8 down, 2 across)", texmap_lod(&t, 2.0f / w, 0, 0, 8.0f / h), 3, 1e-4);
    check_near("lod: under a texel -> level 0", texmap_lod(&t, 0.1f / w, 0, 0, 0.1f / h), 0, 0);
    Vec4 top = texmap_texel(&t, t.levels - 1, 0, 0);
    check_near("trilinear past the last level = the last level", texmap_sample(&t, 0.3f, 0.7f, 20).y, top.y, 1e-5);
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
        check_near(mirror ? "mirrored: tangent along +u (= -x)" : "tangent along +u (= +x)", ex, 0, 1e-5);
        check_near(mirror ? "mirrored: w (n x t) up the image" : "w (n x t) up the image", ew, 0, 1e-5);
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
        check_near(mirror ? "mirrored: tilted map bends the normal up (+y)" : "tilted map bends the normal up (+y)", n.y > 0.3 && fabs(n.x) < 0.01, 1, 0);
        m.tex_normal = 2;
        n = material_normal(&s, &m, vec2(0.5f, 0.5f), MATERIAL_LOD0, vec3(0, 0, 1), tg[0]);
        check_near(mirror ? "mirrored: flat map leaves it" : "flat map leaves it", fabs(n.x) + fabs(n.y) + fabs(n.z - 1), 0, 2e-2);
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
    check_near("bricks: the top of a brick leans up the image", topg > 0.5, 1, 0);
    check_near("bricks: its bottom leans down", botg < -0.5, 1, 0);
    texmap_free(&b);
}

static void test_materials(void){
    printf("5. materials\n");
    Scene *s = calloc(1, sizeof(Scene));
    int c = scene_texture(s, "checker", 1);
    int c2 = scene_texture(s, "checker", 1);
    int mr = scene_texture(s, "plate_mr", 0);
    check_near("scene_texture: a name is loaded once", c == c2 && c > 0 && mr > 0 && s->ntexmaps == 2, 1, 0);
    check_near("scene_texture: an unknown name -> 0", scene_texture(s, "no such texture", 1), 0, 0);
    {
        // a file is known by what it shows, not by its name: the same picture is held once
        // (whatever the file is called), and a file of the same name with another picture is
        // another texture (an image edited and dropped again in the editor was ignored)
        Scene *fs = calloc(1, sizeof(Scene));
        unsigned char red[4 * 4 * 3], green[4 * 4 * 3];
        for (int i = 0; i < 16; i++){ red[3 * i] = 200; red[3 * i + 1] = 30; red[3 * i + 2] = 20; green[3 * i] = 20; green[3 * i + 1] = 200; green[3 * i + 2] = 30; }
        int wrote = png_write("out/test_tex_a.png", 4, 4, 3, red) == 0 && png_write("out/test_tex_b.png", 4, 4, 3, red) == 0;
        int a = scene_texture(fs, "out/test_tex_a.png", 1), a2 = scene_texture(fs, "out/test_tex_a.png", 1);
        int b = scene_texture(fs, "out/test_tex_b.png", 1);
        int data = scene_texture(fs, "out/test_tex_a.png", 0);
        check("scene_texture: a file asked for twice, and the same picture under another name: one texture",
              wrote && a > 0 && a2 == a && b == a && fs->ntexmaps == 2);
        check("... as colour and as data: two", data > 0 && data != a);
        wrote = png_write("out/test_tex_a.png", 4, 4, 3, green) == 0;
        int g = scene_texture(fs, "out/test_tex_a.png", 1);
        check("... the file written again with another picture: another texture", wrote && g > 0 && g != a && fs->ntexmaps == 3);
        check("... which shows the new picture", g > 0 && texmap_texel(&fs->texmaps[g - 1], 0, 1, 1).y > 0.5f && texmap_texel(&fs->texmaps[a - 1], 0, 1, 1).x > 0.5f);
        remove("out/test_tex_a.png"); remove("out/test_tex_b.png");
        scene_free(fs);
        free(fs);
    }
    Material m = material_pbr(vec3(0.5f, 1, 1), 1, 1, 1.5f, 0);
    m.tex_base = c; m.tex_mr = mr;
    TexMap *tc = &s->texmaps[c - 1], *tm = &s->texmaps[mr - 1];
    Vec2 at = vec2(0.37f, 0.61f);
    Material a = material_at(s, &m, at, MATERIAL_LOD0);
    Vec4 tex = texmap_sample(tc, at.x, at.y, 0), mrv = texmap_sample(tm, at.x, at.y, 0);
    check_near("base = texture x factor", a.albedo.x, 0.5 * tex.x, 1e-4);
    check_near("roughness = green", a.roughness, mrv.y, 1e-4);
    check_near("metallic = blue", a.metallic, mrv.z, 1e-4);
    m.uv_scale = 4;
    a = material_at(s, &m, vec2(at.x / 4, at.y / 4), MATERIAL_LOD0);
    check_near("uv_scale: (u, v) / 4 with uv_scale 4 reads (u, v)", a.albedo.x, 0.5 * tex.x, 1e-4);
    scene_free(s);
    free(s);
}

// a w x h grey map, its value f(x, y) 0..1
static void grey_map(TexMap *t, int w, int h, real (*f)(int, int, int, int)){
    unsigned char *p = malloc((size_t)w * h * 4);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            real v = f(x, y, w, h);
            int c = (int)(v * 255 + 0.5f);
            unsigned char *o = p + 4 * ((size_t)y * w + x);
            o[0] = o[1] = o[2] = (unsigned char)(c < 0 ? 0 : c > 255 ? 255 : c);
            o[3] = 255;
        }
    texmap_make(t, p, w, h, 0);
    free(p);
}
static real ramp_x(int x, int y, int w, int h){ (void)y; (void)h; return (real)x / (w - 1); }       // (wraps at the edge: read inside)
// (the same ramp on a map twice as wide, over the same part of the image)
static real ramp_up(int x, int y, int w, int h){ (void)x; (void)w; return 1 - (real)y / (h - 1); }
static real flat_half(int x, int y, int w, int h){ (void)x; (void)y; (void)w; (void)h; return 0.5f; }
static real tri_x(int x, int y, int w, int h){ (void)y; (void)w; (void)h; return (real)(x % 32) / 32; }
// texel (x, y) of a normal map, decoded to -1..1
static Vec3 nrm_at(const TexMap *t, int x, int y){
    Vec4 c = texmap_texel(t, 0, x, y);
    return vec3(2 * c.x - 1, 2 * c.y - 1, 2 * c.z - 1);
}
static void test_pbr_maps(void){
    printf("6. artists' maps (pbr_maps.h)\n");
    TexMap h, n, f, c, g, ro, me, mr, h2, n3;
    // a ramp to the right on a 256-wide map: a step of 1/255 per texel
    grey_map(&h, 256, 64, ramp_x);
    check("a height map made into a normal map", pbr_bump_to_normal(&n, &h, 32) == 0 && n.w == 256 && n.h == 64);
    Vec3 a = nrm_at(&n, 100, 30);
    // slope 32 * (1/255) (a step of 1/255 per texel; strength 32, 256 wide): tilt atan(32 / 255)
    real want = (real)atan(32.0 / 255);
    check_near("a ramp rising to the right: leans left (-x) by atan(32 / 255)", (real)atan2(-a.x, a.z), want, 0.006);
    check_near("... and not up or down", a.y, 0, 0.01);
    grey_map(&h2, 512, 64, ramp_x);                 // the same ramp over the same part of the image, twice as wide
    pbr_bump_to_normal(&n3, &h2, 32);
    Vec3 a2 = nrm_at(&n3, 200, 30);
    check_near("a map twice as wide, the same ramp: the same tilt", (real)atan2(-a2.x, a2.z), (real)atan2(-a.x, a.z), 0.006);
    texmap_free(&h2); texmap_free(&n3);
    TexMap hu, nu;
    grey_map(&hu, 64, 256, ramp_up);
    pbr_bump_to_normal(&nu, &hu, 32);
    Vec3 b = nrm_at(&nu, 30, 100);
    check_near("a ramp rising up the image (64 x 256): leans down (-y) as much", (real)atan2(-b.y, b.z), want, 0.006);
    check("... and not sideways", fabs(b.x) < 0.01);
    texmap_free(&hu); texmap_free(&nu);
    grey_map(&f, 32, 32, flat_half);
    pbr_bump_to_normal(&c, &f, 4);
    Vec3 fl = nrm_at(&c, 10, 10);
    check("a flat height: a flat normal", fabs(fl.x) < 0.01 && fabs(fl.y) < 0.01 && fl.z > 0.99);
    // combined: a flat detail leaves the base; a flat base gives the detail; tilts add up
    check("normal maps combined", pbr_combine_normals(&g, &n, &c) == 0 && g.w == 256);
    Vec3 g1 = nrm_at(&g, 100, 30);
    check("a flat detail on top: the base as it was", fabs(g1.x - a.x) < 0.012 && fabs(g1.y - a.y) < 0.012);
    TexMap g2;
    pbr_combine_normals(&g2, &c, &n);
    Vec3 g2v = nrm_at(&g2, 100, 30);
    check("a flat base: the detail as it was", fabs(g2v.x - a.x) < 0.012 && fabs(g2v.y - a.y) < 0.012);
    TexMap steep, ns, both;
    grey_map(&steep, 256, 64, tri_x);                // a saw: 1/32 per texel
    pbr_bump_to_normal(&ns, &steep, 4);
    pbr_combine_normals(&both, &ns, &ns);
    Vec3 s1 = nrm_at(&ns, 40, 30), s2 = nrm_at(&both, 40, 30);
    real t1 = (real)atan2(-s1.x, s1.z), t2 = (real)atan2(-s2.x, s2.z);
    check_near("the same tilt twice: twice the tilt", t2, 2 * t1, 0.02);
    texmap_free(&g2); texmap_free(&steep); texmap_free(&ns); texmap_free(&both);
    // green turned over
    TexMap fg;
    pbr_flip_green(&fg, &n);
    Vec4 o = texmap_texel(&n, 0, 7, 9), q = texmap_texel(&fg, 0, 7, 9);
    check("flip green: green 1 - green, red and blue kept", fabs(q.y - (1 - o.y)) < 0.003 && q.x == o.x && q.z == o.z);
    texmap_free(&fg);
    // roughness and metallic packed
    grey_map(&ro, 128, 128, ramp_x);
    grey_map(&me, 64, 64, flat_half);
    check("roughness and metallic packed (the larger size)", pbr_pack_mr(&mr, &ro, &me) == 0 && mr.w == 128 && mr.h == 128);
    Vec4 m1 = texmap_texel(&mr, 0, 64, 64), r1 = texmap_texel(&ro, 0, 64, 64);
    check_near("roughness in green", m1.y, r1.x, 0.003);
    check_near("metallic in blue", m1.z, 0.5f, 0.004);
    check("red white", m1.x == 1);
    TexMap mr2;
    pbr_pack_mr(&mr2, NULL, &me);
    check("a metallic map alone: roughness white (the factor alone)", texmap_texel(&mr2, 0, 3, 3).y == 1 && mr2.w == 64);
    texmap_free(&mr);                     // (the next call makes mr again: it would lose this one)
    check("neither: refused", pbr_pack_mr(&mr, NULL, NULL) != 0);
    texmap_free(&mr2);
    // through material_normal: a quad's normal bends away from where the height rises
    Scene *sc = calloc(1, sizeof(Scene));
    TexMap nn;
    pbr_bump_to_normal(&nn, &h, 8);
    Material m = material_pbr(vec3(1, 1, 1), 0, 0.5f, 1.5f, 0);
    m.tex_normal = scene_add_texmap(sc, &nn);
    Vec3 bent = material_normal(sc, &m, vec2(0.4f, 0.5f), MATERIAL_LOD0, vec3(0, 0, 1), (Vec4){1, 0, 0, 1});
    check("a quad facing +z, u along +x, height rising to +x: its normal leans to -x", bent.x < -0.01f && bent.z > 0.9f);
    scene_free(sc);
    free(sc);
    texmap_free(&h); texmap_free(&n); texmap_free(&f); texmap_free(&c); texmap_free(&g); texmap_free(&ro); texmap_free(&me);
}

int main(void){
    test_mips();
    test_lookups();
    test_tangents_normals();
    test_materials();
    test_pbr_maps();
    if (failures){ printf("test_tex: %d FAILED\n", failures); return 1; }
    printf("test_tex: all passed\n");
    return 0;
}
