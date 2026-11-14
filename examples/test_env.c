// test_env.c - the environment (core/env.h, core/hdr.h; docs/PBR.md, PBR4) checked on its own.
//
//   make bin/test_env && bin/test_env      (part of make test)
//
// 1. .hdr files: written and read back, every pixel within RGBE's precision (8 bits of
//    mantissa against the pixel's largest channel), run-length encoded rows included.
// 2. Importance sampling, on a sky with a sun: the density integrates to 1 over the sphere;
//    env_sample states the density env_pdf gives for its direction; the radiance's integral
//    by importance sampling equals the integral by quadrature.
// 3. The rasterizers' lookups: a constant environment stays constant at every roughness and
//    in the irradiance; the irradiance from the harmonics equals the cosine integral for an
//    environment of degree 2 (which 9 harmonics hold exactly); a prefiltered level equals
//    the GGX-weighted integral by quadrature.
// 4. If models/skies/ has the downloaded skies, they load and sample consistently too.
// 5. The path tracer under the sky: the white furnace. A white sphere (diffuse, plastic, a
//    rough metal, glass) under a uniform white sky looks exactly as bright as the sky, with
//    MIS and without (next event estimation only, and bounces only for perfect lobes).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "core/env.h"
#include "core/hdr.h"
#include "core/rng.h"
#include "core/scene.h"
#include "rt/tracer.h"

static int failures;

static void check(const char *what, double got, double want, double tol){
    double err = fabs(got - want);
    int ok = err <= tol;
    printf("  %-58s %10.5f  (want %.5f +- %.5f)  %s\n", what, got, want, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static double lum(Vec3 c){ return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

// the integral over the sphere of f(d) by the midpoint rule on a (2 n) x n grid of (phi, theta)
typedef double (*SphereFn)(Vec3 d, void *ctx);
static double sphere_integral(SphereFn f, void *ctx, int n){
    double sum = 0;
    for (int j = 0; j < n; j++){
        double th = (j + 0.5) * PI / n;
        double dw = (PI / n) * (2 * PI / (2 * n)) * sin(th);
        for (int i = 0; i < 2 * n; i++){
            double ph = (i + 0.5) * 2 * PI / (2 * n);
            Vec3 d = vec3((real)(sin(th) * sin(ph)), (real)cos(th), (real)(sin(th) * cos(ph)));
            sum += f(d, ctx) * dw;
        }
    }
    return sum;
}

static double fn_pdf(Vec3 d, void *e){ return env_pdf(e, d); }
static double fn_lum(Vec3 d, void *e){ return lum(env_eval(e, d)); }

static void test_hdr(void){
    printf("1. .hdr files\n");
    int w = 41, h = 23;
    float *img = malloc(sizeof(float) * 3 * w * h);
    Rng r;
    rng_seed(&r, 7, 1);
    for (int i = 0; i < w * h; i++)
        for (int k = 0; k < 3; k++){
            int y = i / w;
            double v = y < 5 ? 0.37 + y : exp((rng_real(&r) - 0.5) * 16);   // runs in the first rows
            img[3 * i + k] = (y == 7 && k == 1) ? 0 : (float)v;
        }
    if (hdr_save("out/test_env.hdr", img, w, h)){ printf("  cannot write out/test_env.hdr  FAIL\n"); failures++; free(img); return; }
    int w2, h2;
    float *back = hdr_load("out/test_env.hdr", &w2, &h2);
    if (!back || w2 != w || h2 != h){ printf("  read back  FAIL\n"); failures++; free(img); free(back); return; }
    double worst = 0;
    for (int i = 0; i < w * h; i++){
        double m = fmax(img[3 * i], fmax(img[3 * i + 1], img[3 * i + 2]));
        for (int k = 0; k < 3; k++) if (m > 0) worst = fmax(worst, fabs(back[3 * i + k] - img[3 * i + k]) / m);
    }
    check("worst error / the pixel's largest channel", worst, 0, 1.0 / 128);
    free(img);
    free(back);
}

static void test_sampling(const char *name, Env *e){
    char what[96];
    int grid = 2 * e->h < 1024 ? 2 * e->h : 1024;   // quadrature rows
    double total = sphere_integral(fn_pdf, e, grid);
    snprintf(what, sizeof(what), "%s: the density's integral", name);
    check(what, total, 1, 0.01);
    Rng r;
    rng_seed(&r, 11, 3);
    double worst = 0, est = 0;
    int n = 400000, bad = 0, off = 0;
    for (int i = 0; i < n; i++){
        Vec3 d;
        real pdf;
        Vec3 L = env_sample(e, rng_real(&r), rng_real(&r), &d, &pdf);
        if (!(pdf > 0)){ bad++; continue; }
        double p2 = env_pdf(e, d);
        double rel = fabs(p2 - pdf) / pdf;
        if (rel > 2e-3){ off++; if (getenv("LR_DEBUG") && off < 5){ real u, v; env_uv(e, d, &u, &v); printf("    off: pdf %g env_pdf %g at u %.7f v %.7f (x %.4f y %.4f)\n", pdf, p2, u, v, u * e->w, v * e->h); } }
        else worst = fmax(worst, rel);
        est += lum(L) / pdf;
    }
    est /= n;
    snprintf(what, sizeof(what), "%s: worst |env_pdf - stated pdf| / pdf (the others)", name);
    check(what, worst, 0, 2e-3);
    snprintf(what, sizeof(what), "%s: samples across a pixel border from theirs, %%", name);
    check(what, 100.0 * off / n, 0, 0.05);
    snprintf(what, sizeof(what), "%s: samples without a density (of %d)", name, n);
    check(what, bad, 0, 0);
    double quad = sphere_integral(fn_lum, e, grid);
    snprintf(what, sizeof(what), "%s: radiance integral, sampled / quadrature", name);
    check(what, est / quad, 1, 0.01);
}

static double poly_env(Vec3 d){ return 2 + 0.5 * d.x + 0.8 * d.y + 0.6 * d.z * d.z - 0.4 * d.x * d.y; }
typedef struct { Vec3 n; } IrrCtx;
static double fn_irr(Vec3 l, void *c){
    IrrCtx *x = c;
    double nl = v_dot(x->n, l);
    return nl > 0 ? poly_env(l) * nl / PI : 0;
}
// the prefiltering's definition: L(l) (n.l) weighted by the density of l (D / 4, v = n),
// over that weight's integral; the quadrature only where D matters (n.h > cutoff)
static double prefiltered(Env *e, Vec3 n, real a2, int grid){
    double num = 0, den = 0;
    for (int j = 0; j < grid; j++){
        double th = (j + 0.5) * PI / grid;
        double dw = (PI / grid) * (2 * PI / (2 * grid)) * sin(th);
        for (int i = 0; i < 2 * grid; i++){
            double ph = (i + 0.5) * 2 * PI / (2 * grid);
            Vec3 l = vec3((real)(sin(th) * sin(ph)), (real)cos(th), (real)(sin(th) * cos(ph)));
            double nl = v_dot(n, l);
            if (nl <= 0) continue;
            Vec3 hv = v_norm(v_add(n, l));
            double ch = v_dot(n, hv), dd = ch * ch * (a2 - 1) + 1, D = a2 / (PI * dd * dd);
            double wgt = nl * D / 4 * dw;
            if (wgt < 1e-9 * dw) continue;
            num += lum(env_eval(e, l)) * wgt;
            den += wgt;
        }
    }
    return num / den;
}

static void test_lookups(void){
    printf("3. the rasterizers' lookups\n");
    Env *c = env_constant(vec3(0.3f, 0.6f, 0.9f));
    double worst = 0;
    Vec3 dirs[5] = {{0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {0.6f, 0.0f, -0.8f}, {0.48f, 0.6f, 0.64f}};
    for (int k = 0; k < 5; k++){
        for (int j = 0; j <= 10; j++){
            Vec3 a = env_level(c, dirs[k], j / 10.0f);
            worst = fmax(worst, fabs(a.y - 0.6) / 0.6);
        }
        Vec3 irr = env_irradiance(c, dirs[k]);
        worst = fmax(worst, fabs(irr.z - 0.9) / 0.9);
    }
    check("constant: worst relative error (levels, irradiance)", worst, 0, 2e-3);
    env_free(c);

    int w = 256, h = 128;
    float *img = malloc(sizeof(float) * 3 * w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            double ph = ((x + 0.5) / w - 0.5) * 2 * PI, th = (y + 0.5) / h * PI;
            Vec3 d = vec3((real)(sin(th) * sin(ph)), (real)cos(th), (real)(sin(th) * cos(ph)));
            for (int k = 0; k < 3; k++) img[3 * (y * w + x) + k] = (float)poly_env(d);
        }
    Env *e = env_from_rgb(img, w, h);
    worst = 0;
    double worst_pre = 0;
    for (int k = 0; k < 5; k++){
        IrrCtx x = {v_norm(dirs[k])};
        double want = sphere_integral(fn_irr, &x, 256);
        worst = fmax(worst, fabs(env_irradiance(e, x.n).x - want) / want);
        for (int lv = 1; lv < ENV_LEVELS; lv++){
            real r = (real)lv / (ENV_LEVELS - 1);
            double want_pre = prefiltered(e, x.n, r * r * r * r, lv == 1 ? 512 : 192);
            double got = lum(env_level(e, x.n, r));
            if (getenv("LR_DEBUG")) printf("    dir %d level %d: %.5f want %.5f\n", k, lv, got, want_pre);
            worst_pre = fmax(worst_pre, fabs(got - want_pre) / want_pre);
        }
    }
    check("degree 2: irradiance, harmonics against quadrature", worst, 0, 0.01);
    check("degree 2: prefiltered levels against quadrature", worst_pre, 0, 0.02);
    env_free(e);
}

static void test_furnace(void){
    printf("5. the path tracer under a uniform white sky\n");
    static const struct { const char *name; Material m; double tol; } mats[5] = {
        {"white diffuse", {.type = MAT_PBR, .albedo = {1, 1, 1}, .ior = 1.5f, .roughness = 1}, 0.01},
        {"white plastic, roughness 0.3", {.type = MAT_PBR, .albedo = {1, 1, 1}, .ior = 1.5f, .roughness = 0.3f}, 0.01},
        {"white metal, roughness 0.6", {.type = MAT_PBR, .albedo = {1, 1, 1}, .ior = 1.5f, .metallic = 1, .roughness = 0.6f}, 0.01},
        {"smooth glass", {.type = MAT_PBR, .albedo = {1, 1, 1}, .ior = 1.5f, .transmission = 1}, 0.01},
        {"white mirror", {.type = MAT_PBR, .albedo = {1, 1, 1}, .ior = 1.5f, .metallic = 1}, 0.001},
    };
    for (int k = 0; k < 5; k++)
        for (int mis = 1; mis >= 0; mis--){
            static Scene sc;
            memset(&sc, 0, sizeof(sc));
            sc.light = -1;
            int m = scene_add_material(&sc, mats[k].m);
            scene_add_prim(&sc, make_sphere(vec3(0, 0, 0), 1, m));
            sc.env = env_constant(vec3(1, 1, 1));
            TraceParams tp = trace_default_params();
            tp.mis = mis;
            Rng r;
            rng_seed(&r, 5 + k, 2);
            double sum = 0;
            int n = 20000;
            for (int i = 0; i < n; i++){
                // rays at the sphere from everywhere, hitting it anywhere
                Vec3 dir = v_norm(vec3(rng_real(&r) - 0.5f, rng_real(&r) - 0.5f, rng_real(&r) - 0.5f));
                Vec3 o = v_scale(dir, -5);
                Vec3 t = v_norm(v_cross(dir, vec3(0.3f, 1, 0.1f))), b = v_cross(dir, t);
                real a = rng_real(&r) * (real)(2 * PI), rr = 0.98f * r_sqrt(rng_real(&r));
                o = v_add(o, v_add(v_scale(t, rr * r_cos(a)), v_scale(b, rr * r_sin(a))));
                Ray ray = {o, dir};
                sum += lum(trace_path_ex(&sc, ray, &r, &tp));
            }
            char what[96];
            snprintf(what, sizeof(what), "%s, MIS %s", mats[k].name, mis ? "on" : "off");
            check(what, sum / n, 1, mats[k].tol);
            scene_free(&sc);
        }
}

int main(void){
    test_hdr();
    printf("2. importance sampling\n");
    Env *sky = env_procedural_sky(512, 35, -30);
    test_sampling("procedural sky", sky);
    env_set_rot(sky, 1.1f);
    test_sampling("procedural sky, turned", sky);
    env_free(sky);
    test_lookups();
    const char *files[2] = {"models/skies/kloofendal_48d_partly_cloudy_2k.hdr", "models/skies/studio_small_09_2k.hdr"};
    int printed = 0;
    for (int i = 0; i < 2; i++){
        FILE *f = fopen(files[i], "rb");
        if (!f) continue;
        fclose(f);
        if (!printed++) printf("4. the downloaded skies\n");
        Env *e = env_load_hdr(files[i]);
        if (!e){ printf("  %s does not load  FAIL\n", files[i]); failures++; continue; }
        printf("  %s: %d x %d, mean radiance %.3f %.3f %.3f\n", files[i], e->w, e->h, e->mean.x, e->mean.y, e->mean.z);
        test_sampling(i ? "studio" : "kloofendal", e);
        env_free(e);
    }
    if (!printed) printf("4. (no skies in models/skies/: skipped)\n");
    test_furnace();
    if (failures){ printf("%d environment tests FAILED\n", failures); return 1; }
    printf("all environment tests passed\n");
    return 0;
}
