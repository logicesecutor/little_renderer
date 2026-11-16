// test_brdf.c - the BSDF of core/brdf.h (docs/PBR.md, PBR1) checked on its own.
//
//   make bin/test_brdf && bin/test_brdf      (part of make test)
//
// 1. Reciprocity: f(v, l) = f(l, v) for reflection; through glass f(v, l) / eta_l^2 =
//    f(l, v) / eta_v^2 (the radiance BTDF).
// 2. Sampling against evaluation: for materials of every kind and views from straight on
//    to grazing, bsdf_sample's mean weight (with its perfect lobes) equals the integral of
//    f |cos| over the sphere by quadrature (bsdf_eval, plus the perfect lobes' reflectance),
//    and its density integrates to the share of samples it draws from the rough lobes: the
//    directions are drawn with the density it states.
// 3. The white furnace: white materials keep the energy they receive, no more, and (with
//    the multiple-scattering compensation and the diffuse layer under the specular one) no
//    less, but for rough glass, whose light bouncing between its microfacets is lost.
#include <stdio.h>
#include <math.h>
#include "core/brdf.h"
#include "core/rng.h"

static int failures;

static void check(const char *what, double got, double want, double tol){
    double err = fabs(got - want);
    int ok = err <= tol;
    printf("  %-58s %9.4f  (want %.4f +- %.4f)  %s\n", what, got, want, tol, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}
static void check_le(const char *what, double got, double most){
    int ok = got <= most;
    printf("  %-58s %9.4f  (want <= %.4f)  %s\n", what, got, most, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static Vec3 dir_at(real cos_t, real phi){
    real s = r_sqrt(r_fmax(0, 1 - cos_t * cos_t));
    return vec3(s * r_cos(phi), s * r_sin(phi), cos_t);
}

// the integral of f |cos| over the sphere (luminance), the integral of the density, by a
// midpoint rule in (theta, phi): fine enough for roughness >= 0.2
static void quadrature(Bsdf *b, Vec3 n, Vec3 v, double *energy, double *pdf_total){
    const int NT = 3000, NP = 1000;
    double e = 0, p = 0;
    for (int i = 0; i < NT; i++){
        double th = PI * (i + 0.5) / NT, ct = cos(th), w = sin(th);
        for (int j = 0; j < NP; j++){
            Vec3 l = dir_at((real)ct, (real)(2 * PI * (j + 0.5) / NP));
            real pdf;
            Vec3 f = bsdf_eval(b, n, v, l, &pdf);
            e += bsdf_lum(f) * fabs(ct) * w;
            p += pdf * w;
        }
    }
    double dw = PI / NT * 2 * PI / NP;
    *energy = e * dw;
    *pdf_total = p * dw;
}

// the mean weight of bsdf_sample (luminance), the share of samples from the rough lobes
static void sampled(Bsdf *b, Vec3 n, Vec3 v, int count, double *energy, double *rough_share, double *err){
    Rng r;
    rng_seed(&r, 7, 3);
    double s = 0, s2 = 0;
    int rough = 0;
    for (int k = 0; k < count; k++){
        BsdfSample bs;
        real u0 = rng_real(&r), u1 = rng_real(&r), u2 = rng_real(&r);
        double w = 0;
        if (bsdf_sample(b, n, v, u0, u1, u2, &bs)){
            w = bsdf_lum(bs.w);
            if (!bs.delta) rough++;
        }
        s += w;
        s2 += w * w;
    }
    *energy = s / count;
    *rough_share = (double)rough / count;
    double var = s2 / count - *energy * *energy;
    *err = sqrt(var > 0 ? var / count : 0);
}

int main(void){
    const Vec3 n = vec3(0, 0, 1);
    printf("test_brdf: the BSDF (core/brdf.h)\n");

    // ---- 1. reciprocity
    printf("reciprocity\n");
    {
        Rng r;
        rng_seed(&r, 1, 1);
        double worst = 0;
        Material mats[3] = {material_pbr(vec3(0.8f, 0.5f, 0.2f), 0, 0.5f, 1.5f, 0), material_pbr(vec3(0.9f, 0.6f, 0.3f), 1, 0.3f, 1.5f, 0),
                            material_pbr(vec3(0.5f, 0.5f, 0.5f), 0.4f, 0.8f, 1.4f, 0)};
        for (int m = 0; m < 3; m++){
            Bsdf b = bsdf_make(&mats[m], 1);
            for (int k = 0; k < 2000; k++){
                Vec3 v = dir_at((real)(0.05 + 0.95 * rng_real(&r)), (real)(2 * PI * rng_real(&r)));
                Vec3 l = dir_at((real)(0.05 + 0.95 * rng_real(&r)), (real)(2 * PI * rng_real(&r)));
                real p1, p2;
                double a = bsdf_lum(bsdf_eval(&b, n, v, l, &p1)), c = bsdf_lum(bsdf_eval(&b, n, l, v, &p2));
                double rel = fabs(a - c) / fmax(fmax(a, c), 1e-6);
                if (rel > worst) worst = rel;
            }
        }
        check("reflection: worst |f(v,l) - f(l,v)| / f", worst, 0, 2e-3);
        // rough glass, ior 1.5: v outside, l inside
        Material g = material_pbr(vec3(1, 1, 1), 0, 0.4f, 1.5f, 1);
        Bsdf out = bsdf_make(&g, 1), in = bsdf_make(&g, 0);
        worst = 0;
        int tested = 0;
        for (int k = 0; k < 4000; k++){
            Vec3 v = dir_at((real)(0.05 + 0.95 * rng_real(&r)), (real)(2 * PI * rng_real(&r)));
            Vec3 l = dir_at((real)-(0.05 + 0.95 * rng_real(&r)), (real)(2 * PI * rng_real(&r)));
            real p1, p2;
            double a = bsdf_lum(bsdf_eval(&out, n, v, l, &p1));            // eta_l = 1.5
            double c = bsdf_lum(bsdf_eval(&in, v_neg(n), l, v, &p2));      // eta_v = 1.5
            if (a < 1e-3 && c < 1e-3) continue;
            tested++;
            double rel = fabs(a / (1.5 * 1.5) - c) / fmax(fmax(a / 2.25, c), 1e-6);
            if (rel > worst) worst = rel;
        }
        printf("  (%d pairs through the glass)\n", tested);
        check("refraction: worst |f(v,l)/eta_l^2 - f(l,v)/eta_v^2| / f", worst, 0, 2e-3);
    }

    // ---- 2. sampling against evaluation
    printf("sampling against evaluation (luminance; views at cos 1, 0.5, 0.1)\n");
    {
        struct { const char *name; Material m; int outside; } cases[] = {
            {"diffuse preset", material_diffuse(vec3(0.73f, 0.73f, 0.73f)), 1},
            {"plastic, roughness 0.3", material_pbr(vec3(0.2f, 0.4f, 0.8f), 0, 0.3f, 1.5f, 0), 1},
            {"gold, roughness 0.5", material_pbr(vec3(1.0f, 0.78f, 0.34f), 1, 0.5f, 1.5f, 0), 1},
            {"half metal, roughness 0.8", material_pbr(vec3(0.6f, 0.6f, 0.6f), 0.5f, 0.8f, 1.5f, 0), 1},
            {"rough glass 0.4, entering", material_pbr(vec3(1, 1, 1), 0, 0.4f, 1.5f, 1), 1},
            {"rough glass 0.4, leaving", material_pbr(vec3(1, 1, 1), 0, 0.4f, 1.5f, 1), 0},
            {"frosted half-clear 0.6", material_pbr(vec3(0.9f, 0.9f, 0.7f), 0, 0.6f, 1.5f, 0.5f), 1},
            {"smooth plastic", material_pbr(vec3(0.8f, 0.3f, 0.3f), 0, 0, 1.5f, 0), 1},
        };
        const real cosv[3] = {1, 0.5f, 0.1f};
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++){
            Bsdf b = bsdf_make(&cases[c].m, cases[c].outside);
            for (int k = 0; k < 3; k++){
                Vec3 v = dir_at(cosv[k] > 0.999f ? (real)0.999 : cosv[k], (real)0.3);
                double eq, pq, es, rough, err;
                quadrature(&b, n, v, &eq, &pq);
                sampled(&b, n, v, 2000000, &es, &rough, &err);
                // the perfect lobes' share of the energy: what sampling finds beyond the quadrature
                if (b.smooth){
                    real cv = v_dot(v, n), fd = bsdf_fresnel_dielectric(cv, b.eta);
                    Vec3 fm = bsdf_fresnel_metal(b.base, cv);
                    eq += (1 - b.metallic) * fd + b.metallic * bsdf_lum(fm);
                    if (fd < 1) eq += (1 - b.metallic) * b.transmission * (1 - fd) * bsdf_lum(b.base);
                }
                char what[96];
                snprintf(what, sizeof(what), "%s, cos %.1f: sampled energy", cases[c].name, (double)cosv[k]);
                check(what, es, eq, 4 * err + 2e-3);
                snprintf(what, sizeof(what), "%s, cos %.1f: density's integral", cases[c].name, (double)cosv[k]);
                check(what, pq, rough, 3e-3 + 0.002);
            }
        }
    }

    // ---- 3. the white furnace (the energy kept, by sampling)
    printf("white furnace: the energy kept (luminance) at cos 1 / 0.5 / 0.1\n");
    {
        struct { const char *name; Material m; double lo; } cases[] = {
            {"smooth mirror", material_mirror(vec3(1, 1, 1)), 0.999},
            {"smooth glass", material_glass(vec3(1, 1, 1), 1.5f), 0.999},
            {"white metal, roughness 0.2", material_pbr(vec3(1, 1, 1), 1, 0.2f, 1.5f, 0), 0.99},
            {"white metal, roughness 0.5", material_pbr(vec3(1, 1, 1), 1, 0.5f, 1.5f, 0), 0.99},
            {"white metal, roughness 1", material_pbr(vec3(1, 1, 1), 1, 1, 1.5f, 0), 0.99},
            {"white diffuse preset", material_diffuse(vec3(1, 1, 1)), 0.99},
            {"white plastic, roughness 0.3", material_pbr(vec3(1, 1, 1), 0, 0.3f, 1.5f, 0), 0.99},
            {"white rough glass 0.3", material_pbr(vec3(1, 1, 1), 0, 0.3f, 1.5f, 1), 0.90},
        };
        const real cosv[3] = {1, 0.5f, 0.1f};
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++){
            double e[3], err[3], rough;
            Bsdf b = bsdf_make(&cases[c].m, 1);
            for (int k = 0; k < 3; k++){
                Vec3 v = dir_at(cosv[k] > 0.999f ? (real)0.999 : cosv[k], 0);
                sampled(&b, n, v, 1000000, &e[k], &rough, &err[k]);
            }
            char what[96];
            snprintf(what, sizeof(what), "%s: %.3f / %.3f / %.3f, most", cases[c].name, e[0], e[1], e[2]);
            check_le(what, fmax(e[0], fmax(e[1], e[2])), 1.0 + 4 * fmax(err[0], fmax(err[1], err[2])) + 1e-3);
            double least = fmin(e[0], fmin(e[1], e[2]));
            snprintf(what, sizeof(what), "%s: least", cases[c].name);
            check(what, least >= cases[c].lo ? cases[c].lo : least, cases[c].lo, 0);
        }
    }

    if (failures){ printf("%d BSDF test(s) FAILED\n", failures); return 1; }
    printf("all BSDF tests passed\n");
    return 0;
}
