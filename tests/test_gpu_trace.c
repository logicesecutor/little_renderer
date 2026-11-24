// test_gpu_trace.c - the GPU path tracer's offline image (app/gpu_frame.h): exactly the samples
// asked for, and the same image every time. Each scene is traced twice, on a GPU and a view of
// its own each time (as two runs of bin/render would), and the two images must be the same to
// the last bit: nothing may depend on how fast the GPU happened to be (how many passes a
// submission holds) or on where the allocator put a buffer. A model's triangles in the box
// make the tracer's buffers grow after the layout was first written: it must be written again.
//
// A glowing ball over a floor (its light sampled: Scene.emitters): the GPU's image, by the one
// kernel and by the wavefront's, has the CPU's mean (within 2%), and little noise where the
// ball lights the floor.
//
// A floor of every map (PBR7: base colour, a bump on top of a normal map, roughness and
// metallic apart, occlusion, a glow) under a lamp: the GPU's image (both kernels) has the CPU's
// mean within 2%, and over the floor the CPU's pattern (the images correlate).
//
// The guide (each pixel's first surface, for the denoiser and the reprojection) drawn by the
// rasterizer is the traced one: in a room of spheres (a mirror, glass), on a textured floor
// under a sky, and with instances (a mesh's, shapes'), what each pixel sees (the kind and the
// object), its normal (within 10 degrees), distance (1%) and albedo (0.02) agree in all but a
// few pixels (the spheres' outlines: the rasterizer's spheres are triangles).
//
// Temporal gradients (A-SVGF): a box under the lamp is moved, and one pass later the floor
// where its shadow was is as bright as it now is (its pixels kept none of their samples: the
// gradient samples saw the light change), while the walls far from it kept most of theirs;
// without the gradients the old shadow stays (the same history everywhere).
//
// Views come and go (a window resized makes its view again): 300 made and destroyed on one
// GPU, some with their path tracer, leave nothing in the list of the objects alive (the path
// tracer's timer was left there: after 250 no view could be made), and the last one traces.
//
// Part of make test. Without a GPU it says so and passes.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "lr.h"
#include "test_util.h"
#include "app/gpu_frame.h"
#include "rt/tracer.h"
#include "scene/pbr_maps.h"
#ifdef LR_GPU
#include "gpu/gpu_internal.h"     // gpu_live_count
#endif

#define W 96
#define H 96

// the scene traced with spp samples per pixel into img, on a new GPU and view; the samples
// per pixel the view counted, -1 if there is no GPU or it cannot hold the scene
static int trace_once(Scene *s, int spp, Image *img){
    Gpu *g;
    GpuView *v;
    if (gpu_frame_open(W, H, &g, &v)) return -1;
    RenderParams rp = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, W, H);
    MarchParams mp = march_default_params();
    Camera cam = camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)W / H);
    int passes = -1;
    if (gpu_frame_trace(g, v, s, &cam, &rp, &mp, spp, 8, 1, 0, 1, 0, img, 0) == 0) passes = gpu_view_trace_passes(v);
    gpu_frame_close(g, v);
    return passes;
}

static void same_twice(const char *name, const SceneOptions *so, int spp){
    static Scene scene;
    memset(&scene, 0, sizeof(scene));
    char what[96];
    if (scene_cornell_box(&scene, so) != 0){
        snprintf(what, sizeof(what), "%s: the scene", name);
        check(what, 0);
        return;
    }
    AccelParams ap = accel_default_params();
    scene_set_accel(&scene, ACCEL_BVH, &ap);
    Image a = image_create(W, H), b = image_create(W, H);
    int pa = trace_once(&scene, spp, &a), pb = trace_once(&scene, spp, &b);
    snprintf(what, sizeof(what), "%s: %d samples per pixel, no more (%d, %d)", name, spp, pa, pb);
    check(what, pa == spp && pb == spp);
    double sum = 0;
    for (int i = 0; i < W * H; i++) sum += a.px[i].x + a.px[i].y + a.px[i].z;
    snprintf(what, sizeof(what), "%s: an image (mean %.3f)", name, sum / (3.0 * W * H));
    check(what, pa == spp && sum > 0);
    snprintf(what, sizeof(what), "%s: traced again, the same to the last bit", name);
    check(what, pa == spp && pb == spp && !memcmp(a.px, b.px, (size_t)W * H * sizeof(a.px[0])));
    image_free(&a);
    image_free(&b);
    scene_free(&scene);
}

// the image's mean and, over the floor's square under the ball, its pixels' spread around their
// own neighbours' (noise): on the GPU (wavefront: its kernels) or the CPU
static void emit_stats(Scene *s, int where, int spp, double *mean, double *noise){
    Image img = image_create(W, H);
    Camera cam = camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)W / H);
    int ok = 1;
    if (where < 2){
        Gpu *g;
        GpuView *v;
        ok = gpu_frame_open(W, H, &g, &v) == 0;
        if (ok){
            gpu_view_trace_wavefront(v, where == 1);
            RenderParams rp = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, W, H);
            MarchParams mp = march_default_params();
            ok = gpu_frame_trace(g, v, s, &cam, &rp, &mp, spp, 8, 1, 0, 0, 0, &img, 0) == 0;
            gpu_frame_close(g, v);
        }
    } else {
        TraceParams tp = trace_default_params();
        Vec3 *px = malloc(sizeof(Vec3) * W * H);
        trace_image(s, &cam, &tp, W, H, spp, 1, 0, px, NULL, NULL);
        for (int i = 0; i < W * H; i++) img.px[i] = px[i];
        free(px);
    }
    double sum = 0, d2 = 0;
    int n = 0;
    for (int i = 0; i < W * H; i++) sum += img.px[i].x;
    for (int y = H / 2; y < H - 8; y++)
        for (int x = W / 4; x < 3 * W / 4; x++){
            double c = img.px[y * W + x].x, nb = (img.px[y * W + x - 1].x + img.px[y * W + x + 1].x) / 2;
            d2 += (c - nb) * (c - nb);
            n++;
        }
    *mean = ok ? sum / (W * H) : -1;
    *noise = ok ? sqrt(d2 / n) : -1;
    image_free(&img);
}

static void glowing_ball(void){
    static Scene s;
    memset(&s, 0, sizeof(s));
    int floor = scene_add_material(&s, material_diffuse(vec3(0.5f, 0.5f, 0.5f)));
    Material glow = material_pbr(vec3(0, 0, 0), 0, 1, 1.5f, 0);
    glow.emission = vec3(10, 10, 10);
    int mg = scene_add_material(&s, glow);
    scene_add_prim(&s, make_quad(vec3(-2000, 0, -2000), vec3(0, 0, 4000), vec3(4000, 0, 0), floor));
    scene_add_instance(&s, INST_SPHERE, xform_axes(mat3_identity(), vec3(10, 10, 10), vec3(0, 50, 0)), mg);
    s.light = -1;
    s.cam_from = vec3(0, 160, -220);
    s.cam_at = vec3(0, 10, 0);
    s.cam_up = vec3(0, 1, 0);
    s.cam_vfov_deg = 40;
    scene_set_accel(&s, ACCEL_BVH, NULL);
    double m[3], nz[3];
    for (int k = 0; k < 3; k++) emit_stats(&s, k, 64, &m[k], &nz[k]);
    printf("  glowing ball: means %.4f %.4f %.4f (GPU, wavefront, CPU), noise %.4f %.4f %.4f\n", m[0], m[1], m[2], nz[0], nz[1], nz[2]);
    check("a glowing ball sampled as a light: the GPU's mean is the CPU's (2%)", m[0] > 0 && fabs(m[0] - m[2]) < 0.02 * m[2]);
    check("... the wavefront's too", m[1] > 0 && fabs(m[1] - m[2]) < 0.02 * m[2]);
    check("... with as little noise as the CPU's (under twice)", nz[0] < 2 * nz[2] && nz[1] < 2.5 * nz[2]);
    scene_free(&s);
}

// a w x h map of f's colour (each channel 0..1)
static int map_of(Scene *s, int w, int h, int srgb, Vec3 (*f)(int, int)){
    unsigned char *p = malloc((size_t)w * h * 4);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            Vec3 c = f(x, y);
            unsigned char *o = p + 4 * ((size_t)y * w + x);
            o[0] = (unsigned char)(c.x * 255 + 0.5f); o[1] = (unsigned char)(c.y * 255 + 0.5f); o[2] = (unsigned char)(c.z * 255 + 0.5f); o[3] = 255;
        }
    TexMap t;
    texmap_make(&t, p, w, h, srgb);
    free(p);
    return scene_add_texmap(s, &t);
}
static int tile(int x, int y){ return ((x / 32) + (y / 32)) & 1; }
static real edge(int x, int y){ real u = (x % 32) / 31.0f, v = (y % 32) / 31.0f; real e = r_fmin(r_fmin(u, 1 - u), r_fmin(v, 1 - v)); return r_fmin(1, e * 8); }
static Vec3 f_base(int x, int y){ return tile(x, y) ? vec3(0.75f, 0.45f, 0.3f) : vec3(0.85f, 0.8f, 0.7f); }
static Vec3 f_height(int x, int y){ real e = edge(x, y); return vec3(e, e, e); }
static Vec3 f_normal(int x, int y){ real u = ((x % 16) - 8) / 8.0f; (void)y; return vec3(0.5f + 0.3f * u, 0.5f, 0.9f); }
static Vec3 f_rough(int x, int y){ real r = tile(x, y) ? 0.85f : 0.2f; return vec3(r, r, r); }
static Vec3 f_metal(int x, int y){ real m = (x / 32) % 4 == 0 ? 1.0f : 0.0f; (void)y; return vec3(m, m, m); }
static Vec3 f_ao(int x, int y){ real a = 0.4f + 0.6f * edge(x, y); return vec3(a, a, a); }
static Vec3 f_glow(int x, int y){ int on = (x % 32) > 13 && (x % 32) < 18 && (y % 32) > 13 && (y % 32) < 18 && tile(x, y); return on ? vec3(1, 0.6f, 0.2f) : vec3(0, 0, 0); }

static void pbr_plate(void){
    static Scene s;
    memset(&s, 0, sizeof(s));
    Material m = material_pbr(vec3(1, 1, 1), 1, 1, 1.5f, 0);
    m.emission = vec3(2, 2, 2);
    PbrMapSet ms = {.base = map_of(&s, 128, 128, 1, f_base), .normal = map_of(&s, 128, 128, 0, f_normal),
                    .bump = map_of(&s, 128, 128, 0, f_height), .rough = map_of(&s, 128, 128, 0, f_rough),
                    .metal = map_of(&s, 64, 64, 0, f_metal), .ao = map_of(&s, 128, 128, 0, f_ao),
                    .emissive = map_of(&s, 128, 128, 1, f_glow), .bump_strength = 6};
    check("every map made (pbr_apply)", pbr_apply(&s, &m, &ms, NULL, NULL) == 0 && m.tex_normal && m.tex_mr && m.tex_occlusion && m.tex_emissive);
    m.uv_scale = 2;
    int fm = scene_add_material(&s, m);
    Primitive fl = make_quad(vec3(-100, 0, -100), vec3(0, 0, 200), vec3(200, 0, 0), fm);
    scene_add_prim(&s, fl);
    int lm = scene_add_material(&s, material_light(vec3(12, 12, 12)));
    s.light = scene_add_prim(&s, make_quad(vec3(-30, 120, -30), vec3(60, 0, 0), vec3(0, 0, 60), lm));
    s.cam_from = vec3(0, 150, -160);
    s.cam_at = vec3(0, 0, 0);
    s.cam_up = vec3(0, 1, 0);
    s.cam_vfov_deg = 45;
    scene_set_accel(&s, ACCEL_BVH, NULL);
    Image img[3];
    double mean[3];
    Camera cam = camera_make(s.cam_from, s.cam_at, s.cam_up, s.cam_vfov_deg, (real)W / H);
    for (int k = 0; k < 3; k++){
        img[k] = image_create(W, H);
        mean[k] = -1;
        if (k < 2){
            Gpu *g;
            GpuView *v;
            if (gpu_frame_open(W, H, &g, &v)) continue;
            gpu_view_trace_wavefront(v, k == 1);
            RenderParams rp = render_params_camera(s.cam_from, s.cam_at, s.cam_up, s.cam_vfov_deg, W, H);
            MarchParams mp = march_default_params();
            int ok = gpu_frame_trace(g, v, &s, &cam, &rp, &mp, 256, 8, 1, 0, 0, 0, &img[k], 0) == 0;
            gpu_frame_close(g, v);
            if (!ok) continue;
        } else {
            TraceParams tp = trace_default_params();
            Vec3 *px = malloc(sizeof(Vec3) * W * H);
            trace_image(&s, &cam, &tp, W, H, 256, 1, 0, px, NULL, NULL);
            for (int i = 0; i < W * H; i++) img[k].px[i] = px[i];
            free(px);
        }
        double sum = 0;
        for (int i = 0; i < W * H; i++) sum += (img[k].px[i].x + img[k].px[i].y + img[k].px[i].z) / 3;
        mean[k] = sum / (W * H);
    }
    // how alike the GPU's and the CPU's pictures are: the correlation of their pixels (4 x 4 blocks, the noise averaged)
    double corr[2] = {0, 0};
    for (int k = 0; k < 2; k++){
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        int n = 0;
        for (int by = 0; by < H / 4; by++)
            for (int bx = 0; bx < W / 4; bx++){
                double a = 0, b = 0;
                for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++){
                    Vec3 p = img[k].px[(4 * by + j) * W + 4 * bx + i], q = img[2].px[(4 * by + j) * W + 4 * bx + i];
                    a += p.x + p.y + p.z; b += q.x + q.y + q.z;
                }
                sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; n++;
            }
        double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n), cv = sab / n - (sa / n) * (sb / n);
        corr[k] = va > 0 && vb > 0 ? cv / sqrt(va * vb) : 0;
    }
    printf("  every map: means %.4f %.4f %.4f (GPU, wavefront, CPU), correlation with the CPU %.4f %.4f\n", mean[0], mean[1], mean[2], corr[0], corr[1]);
    check("a floor of every map: the GPU's mean is the CPU's (2%)", mean[0] > 0 && fabs(mean[0] - mean[2]) < 0.02 * mean[2]);
    check("... the wavefront's too", mean[1] > 0 && fabs(mean[1] - mean[2]) < 0.02 * mean[2]);
    check("... and the same picture (correlation over 0.97)", corr[0] > 0.97 && corr[1] > 0.97);
    for (int k = 0; k < 3; k++) image_free(&img[k]);
    scene_free(&s);
}

// one new image's guide, rasterized (raster = 1) or traced, into out (3 x W x H vec4)
static int guide_of(Scene *s, int raster, float *out){
    Gpu *g;
    GpuView *v;
    if (gpu_frame_open(W, H, &g, &v)) return 0;
    Camera cam = camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)W / H);
    RenderParams rp = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, W, H);
    MarchParams mp = march_default_params();
    GpuTraceParams tp = {cam, 8, 1, 0, 0, 0, 0, 4, 1};
    gpu_view_trace_raster_guide(v, raster ? 2 : 0);
    gpu_view_trace_prepare(v);
    int ok = gpu_view_trace_start(v, s, &tp);
    for (int k = 0; ok && k < 1000 && gpu_view_trace_passes(v) < 1; k++){
        ok = gpu_view_trace_more(v, s, &rp, &mp);
        gpu_wait(g);
    }
    ok = ok && gpu_view_trace_planes_read(v, 2, 3, out);
    gpu_frame_close(g, v);
    return ok;
}

static void guide_agrees(const char *name, Scene *s){
    float *a = malloc(sizeof(float) * 12 * W * H), *b = malloc(sizeof(float) * 12 * W * H);
    int ok = a && b && guide_of(s, 0, a) && guide_of(s, 1, b);
    char what[160];
    int differ = 0, surfaces = 0, objects = 0;
    for (int i = 0; ok && i < W * H; i++){
        const float *na = a + 4 * i, *ca = a + 4 * (W * H + i), *nb = b + 4 * i, *cb = b + 4 * (W * H + i);
        int ka = (int)(ca[3] - 4 * floorf(ca[3] / 4)), kb = (int)(cb[3] - 4 * floorf(cb[3] / 4));
        int oa = (int)floorf(ca[3] / 4), ob = (int)floorf(cb[3] / 4);
        int same = ka == kb && oa == ob;
        if (same && ka == 0){
            surfaces++;
            objects += oa > 0;
            float c = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
            same = c > 0.985f && fabsf(na[3] - nb[3]) <= 0.01f * na[3] &&
                   fabsf(ca[0] - cb[0]) < 0.02f && fabsf(ca[1] - cb[1]) < 0.02f && fabsf(ca[2] - cb[2]) < 0.02f;
        }
        differ += !same;
    }
    snprintf(what, sizeof(what), "%s: the rasterized guide is the traced one but in %d of %d pixels (%d surfaces, %d of instances): under 4%%",
             name, differ, W * H, surfaces, objects);
    check(what, ok && differ * 25 < W * H && surfaces > W * H / 4);
    free(a);
    free(b);
}

// passes more passes (moving: one image each) into the view
static int passes_more(Gpu *g, GpuView *v, const Scene *s, int passes){
    RenderParams rp = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, W, H);
    MarchParams mp = march_default_params();
    int want = gpu_view_trace_passes(v) + passes, ok = 1;
    for (int k = 0; ok && k < 4000 && gpu_view_trace_passes(v) < want; k++){
        ok = gpu_view_trace_more(v, s, &rp, &mp);
        gpu_wait(g);
    }
    for (int k = 0; k < 100000 && gpu_view_trace_busy(v); k++) gpu_wait(g);
    return ok;
}
// the box at x (its shadow under the lamp), and the image: 48 passes (fresh = 1), else one
// pass after moving it there from where it was (grads: with temporal gradients); per pixel the
// mean, the samples kept and what it sees (the guide's plane 3 w: the kind and the object)
static int shadow_image(Scene *s, int box, real x, int fresh, int grads, Gpu **gp, GpuView **vp, float *mean, float *count, float *seen){
    AccelParams ap = accel_default_params();
    Xform xf = s->insts[box].xf;
    xf.m[0][3] = x;
    scene_move_instance(s, box, xf);
    scene_update_instances(s, &ap);
    Camera cam = camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)W / H);
    GpuTraceParams tp = {cam, 8, 1, 0, 0, 0, 0, 4, 1};
    int ok = 1;
    if (fresh){
        if (gpu_frame_open(W, H, gp, vp)) return 0;
        gpu_view_trace_prepare(*vp);
        ok = gpu_view_trace_start(*vp, s, &tp) && passes_more(*gp, *vp, s, 48);
    } else {
        gpu_view_trace_gradients(*vp, grads ? 32 : 0);
        ok = gpu_view_trace_move(*vp, s, &tp, 32) && passes_more(*gp, *vp, s, 1);
    }
    float *p = malloc(sizeof(float) * 16 * W * H);
    ok = ok && p && gpu_view_trace_planes_read(*vp, 0, 4, p);
    for (int i = 0; ok && i < W * H; i++){
        float n = p[4 * i + 3];
        mean[i] = n > 0 ? (p[4 * i] + p[4 * i + 1] + p[4 * i + 2]) / (3 * n) : -1;
        count[i] = n;
        seen[i] = p[4 * (3 * W * H + i) + 3];
    }
    free(p);
    return ok;
}

static void gradients(void){
    static Scene s;
    memset(&s, 0, sizeof(s));
    SceneOptions empty = {.kind = SCENE_EMPTY};
    if (scene_cornell_box(&s, &empty) != 0){ check("gradients: the scene", 0); return; }
    Vec3 lc = scene_light_center(&s);
    int m = scene_add_material(&s, material_diffuse(vec3(0.7f, 0.7f, 0.7f)));
    int box = scene_add_instance(&s, INST_BOX, xform_rot_y(vec3(lc.x, 90, lc.z), 0, 70), m);
    AccelParams ap = accel_default_params();
    scene_set_accel(&s, ACCEL_BVH, &ap);
    const real x0 = lc.x, x1 = lc.x + 190;
    float *a = malloc(sizeof(float) * W * H * 9), *ca = a + W * H, *b = ca + W * H, *cb = b + W * H, *c = cb + W * H, *cc = c + W * H;
    float *sa = cc + W * H, *sb_ = sa + W * H, *sc_ = sb_ + W * H;
    double moved[2] = {1, 1}, kept[2] = {0, 0};
    int ok = a != NULL;
    for (int grads = 0; ok && grads < 2; grads++){
        Gpu *g, *g2;
        GpuView *v, *v2;
        ok = shadow_image(&s, box, x0, 1, 0, &g, &v, a, ca, sa) && shadow_image(&s, box, x1, 0, grads, &g, &v, b, cb, sb_);
        if (ok) gpu_frame_close(g, v);
        ok = ok && shadow_image(&s, box, x1, 1, 0, &g2, &v2, c, cc, sc_);   // the truth: the box there from the start
        if (ok) gpu_frame_close(g2, v2);
        // where the old shadow was (the same surface as before, now twice as bright and more),
        // and where nothing changed
        double sb = 0, sc = 0, nk = 0, nu = 0;
        int n1 = 0;
        for (int i = 0; ok && i < W * H; i++){
            if (a[i] < 0 || b[i] < 0 || c[i] <= 0) continue;
            if (sa[i] == sc_[i] && c[i] > 2 * a[i] && c[i] > 0.05f){ sb += b[i]; sc += c[i]; n1++; }
            if (fabsf(c[i] - a[i]) < 0.05f * c[i] && (i % W < W / 6 || i % W > 5 * W / 6)){ nk += cb[i]; nu += 1; }
        }
        moved[grads] = n1 > 0 ? fabs(sb - sc) / sc : 1;
        kept[grads] = nu > 0 ? nk / nu : 0;
        (void)ca; (void)cc;
        if (n1 < 20) ok = 0;
        // reset the box for the next run
        Xform xf = s.insts[box].xf;
        xf.m[0][3] = x0;
        scene_move_instance(&s, box, xf);
    }
    char what[200];
    snprintf(what, sizeof(what), "gradients: where the shadow was, one pass after the box moved, off the truth by %.0f%% (%.0f%% without them): under a third of it",
             100 * moved[1], 100 * moved[0]);
    check(what, ok && moved[1] < moved[0] / 3 && moved[1] < 0.15);
    snprintf(what, sizeof(what), "... and the walls far from it kept %.1f samples a pixel (%.1f without them): over 16", kept[1], kept[0]);
    check(what, ok && kept[1] > 16);
    free(a);
    scene_free(&s);
}

// An edit's move (gpu_view_trace_edited): the box's colour changed, grey to red, its light the
// same over its albedo (what the gradients compare): its pixels must not keep their grey
// samples, as they would with the physics' move; the walls keep theirs.
static void edit_colour(void){
    static Scene s;
    memset(&s, 0, sizeof(s));
    SceneOptions empty = {.kind = SCENE_EMPTY};
    if (scene_cornell_box(&s, &empty) != 0){ check("an edit's colour: the scene", 0); return; }
    Vec3 lc = scene_light_center(&s);
    int m = scene_add_material(&s, material_diffuse(vec3(0.7f, 0.7f, 0.7f)));
    int box = scene_add_instance(&s, INST_BOX, xform_rot_y(vec3(lc.x, 90, lc.z), 0, 70), m);
    AccelParams ap = accel_default_params();
    scene_set_accel(&s, ACCEL_BVH, &ap);
    Camera cam = camera_make(s.cam_from, s.cam_at, s.cam_up, s.cam_vfov_deg, (real)W / H);
    GpuTraceParams tp = {cam, 8, 1, 0, 0, 0, 0, 4, 1};
    float *p = malloc(sizeof(float) * 16 * W * H);
    double red[2] = {0, 0}, wall[2] = {0, 0};
    int ok = p != NULL;
    for (int edit = 0; ok && edit < 2; edit++){
        Gpu *g;
        GpuView *v;
        s.mats[m] = material_diffuse(vec3(0.7f, 0.7f, 0.7f));
        if (gpu_frame_open(W, H, &g, &v)){ ok = 0; break; }
        gpu_view_trace_prepare(v);
        ok = gpu_view_trace_start(v, &s, &tp) && passes_more(g, v, &s, 48);
        s.mats[m] = material_diffuse(vec3(0.7f, 0.1f, 0.1f));
        gpu_view_trace_limit(v, 1);
        gpu_view_trace_gradients(v, 32);
        gpu_view_trace_edited(v, edit);
        ok = ok && gpu_view_trace_move(v, &s, &tp, 1) && passes_more(g, v, &s, 1);
        ok = ok && gpu_view_trace_planes_read(v, 0, 4, p);
        // the box's pixels: how red (red over green, 7 for the new colour, 1 for the old); the
        // walls': the samples they kept
        double r = 0, gr = 0, nk = 0;
        int nw = 0;
        for (int i = 0; ok && i < W * H; i++){
            const float *a = p + 4 * i, *gd = p + 4 * (3 * W * H + i);
            if (gd[3] == 4.0f * (box + 1)){ r += a[0]; gr += a[1]; }
            else if (gd[3] == 0.0f && (i % W < W / 6 || i % W > 5 * W / 6)){ nk += a[3]; nw++; }
        }
        red[edit] = gr > 0 ? r / gr : 0;
        wall[edit] = nw ? nk / nw : 0;
        gpu_frame_close(g, v);
    }
    char what[200];
    snprintf(what, sizeof(what), "an edit's colour: the box red over green %.1f one pass after it turned red (%.1f as the physics' move): over 4", red[1], red[0]);
    check(what, ok && red[1] > 4);
    snprintf(what, sizeof(what), "... and the walls kept %.1f samples a pixel: over 16", wall[1]);
    check(what, ok && wall[1] > 16);
    free(p);
    scene_free(&s);
}

static void raster_guide(void){
    static Scene s;
    AccelParams ap = accel_default_params();
    SceneOptions sph = {.kind = SCENE_SPHERES};
    memset(&s, 0, sizeof(s));
    if (scene_cornell_box(&s, &sph) == 0){
        scene_set_accel(&s, ACCEL_BVH, &ap);
        guide_agrees("spheres, a mirror and glass", &s);
        scene_free(&s);
    } else check("spheres: the scene", 0);
    SceneOptions sky = {.kind = SCENE_SKY};
    memset(&s, 0, sizeof(s));
    if (scene_cornell_box(&s, &sky) == 0){
        scene_set_accel(&s, ACCEL_BVH, &ap);
        guide_agrees("a textured floor under the sky", &s);
        scene_free(&s);
    } else check("sky: the scene", 0);
    SceneOptions tor = {.kind = SCENE_MESH, .obj_path = "models/torus.obj", .instances = 3, .mesh_tex = "bricks"};
    memset(&s, 0, sizeof(s));
    if (scene_cornell_box(&s, &tor) == 0){
        Vec3 c = v_scale(v_add(s.cam_at, s.cam_from), 0.5);
        int m = scene_add_material(&s, s.mats[0]);
        scene_add_instance(&s, INST_SPHERE, xform_rot_y(v_add(c, vec3(-60, -40, 0)), 0, 40), m);
        scene_add_instance(&s, INST_BOX, xform_rot_y(v_add(c, vec3(60, -40, 40)), 30, 35), m);
        scene_set_accel(&s, ACCEL_BVH, &ap);
        guide_agrees("instances (a mesh's, a ball, a box), bricks", &s);
        scene_free(&s);
    } else check("instances: the scene", 0);
}

// views made and destroyed, every 30th with its path tracer: the objects alive are as many
// after as before, and a view made then still traces
static void views_come_and_go(void){
    Gpu *g;
    GpuView *v;
    if (gpu_frame_open(W, H, &g, &v)){ check("views come and go: the GPU", 0); return; }
#ifdef LR_GPU
    const int n0 = gpu_live_count();
#endif
    int made = 0;
    for (int k = 0; k < 300; k++){
        GpuView *w = gpu_view_create(g, W, H);
        if (!w) break;
        if (k % 30 == 0) gpu_view_trace_prepare(w);
        gpu_view_destroy(w);
        made++;
    }
    check("300 views made and destroyed on one GPU", made == 300);
#ifdef LR_GPU
    check("... leave no object in the list of those alive", gpu_live_count() == n0);
#endif
    static Scene s;
    memset(&s, 0, sizeof(s));
    SceneOptions boxes = {.kind = SCENE_BOXES};
    int ok = scene_cornell_box(&s, &boxes) == 0;
    GpuView *w = ok ? gpu_view_create(g, W, H) : NULL;
    if (w){
        AccelParams ap = accel_default_params();
        scene_set_accel(&s, ACCEL_BVH, &ap);
        Camera cam = camera_make(s.cam_from, s.cam_at, s.cam_up, s.cam_vfov_deg, (real)W / H);
        GpuTraceParams tp = {cam, 8, 1, 0, 0, 0, 0, 4, 1};
        ok = gpu_view_trace_start(w, &s, &tp) && passes_more(g, w, &s, 2) && gpu_view_trace_passes(w) >= 2;
        gpu_view_destroy(w);
    }
    check("... and the next one traces", ok && w != NULL);
    if (ok) scene_free(&s);
    gpu_frame_close(g, v);
}

int main(void){
    printf("test_gpu_trace\n");
    Gpu *g;
    GpuView *v;
    if (gpu_frame_open(W, H, &g, &v)){
        printf("  no GPU: skipped\nall GPU path tracer tests passed\n");
        return 0;
    }
    gpu_frame_close(g, v);
    SceneOptions boxes = {.kind = SCENE_BOXES};
    same_twice("boxes", &boxes, 20);          // (20: not a number of passes a fast GPU's submissions add up to)
    SceneOptions mesh = {.kind = SCENE_MESH, .obj_path = "models/torus.obj"};
    for (int k = 0; k < 3; k++) same_twice("a model in the box", &mesh, 5);
    glowing_ball();
    pbr_plate();
    raster_guide();
    gradients();
    edit_colour();
    views_come_and_go();
    printf(failures ? "test_gpu_trace: %d FAILED\n" : "all GPU path tracer tests passed\n", failures);
    return failures ? 1 : 0;
}
