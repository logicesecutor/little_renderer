// test_gpu_brdf.c - the BSDF in WGSL (gpu/brdf_gpu.c) against the C (core/brdf.h), on the
// same inputs: materials of every kind, views and lights in every direction, the sampling's
// numbers. Both must give the same evaluation and density, and draw the same direction with
// the same weight (to float precision: a case on the edge of a decision, say a lobe's
// chance, may go either way, so a few in ten thousand may differ).
//
// Needs a GPU build (make deps): bin/test_gpu_brdf (part of make test). Without a GPU it says
// so and passes.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/brdf.h"
#include "core/rng.h"
#include "gpu/brdf_gpu.h"
#include "gpu/gpu.h"

#define N 65536

static double rel(double a, double b){ return fabs(a - b) / fmax(1e-3, fmax(fabs(a), fabs(b))); }

int main(void){
    Gpu *g = gpu_create(NULL, 64, 64);
    for (int k = 0; k < 20000 && g && gpu_state(g) == GPU_PENDING; k++) gpu_poll(g);
    if (!g || gpu_state(g) != GPU_READY){ printf("test_gpu_brdf: no GPU (%s): skipped\n", g ? gpu_message(g) : "none"); return 0; }
    float *in = malloc((size_t)N * GPU_BSDF_IN * sizeof(float)), *out = malloc((size_t)N * GPU_BSDF_OUT * sizeof(float));
    if (!in || !out) return 1;
    Rng r;
    rng_seed(&r, 11, 5);
    for (int i = 0; i < N; i++){
        float *c = in + (size_t)i * GPU_BSDF_IN;
        int kind = i % 6;     // rough dielectric, metal, glass, mixed, smooth plastic, smooth glass/metal
        c[0] = (float)rng_real(&r); c[1] = (float)rng_real(&r); c[2] = (float)rng_real(&r);
        c[3] = kind == 1 ? 1.0f : kind == 3 ? (float)rng_real(&r) : 0.0f;
        c[4] = kind >= 4 ? (rng_real(&r) < 0.5 ? 0.0f : 0.01f) : (float)(0.03 + 0.97 * rng_real(&r));
        c[5] = (float)(1.2 + 0.8 * rng_real(&r));
        c[6] = kind == 2 ? 1.0f : kind == 3 ? (float)rng_real(&r) : kind == 5 ? 1.0f : 0.0f;
        if (kind == 5 && i % 12 == 5){ c[3] = 1; c[6] = 0; }
        c[7] = rng_real(&r) < 0.7 ? 1.0f : 0.0f;
        Vec3 v = sample_cosine_hemisphere(vec3(0, 0, 1), &r);
        Vec3 l = v_norm(vec3(rng_real(&r) * 2 - 1, rng_real(&r) * 2 - 1, rng_real(&r) * 2 - 1));
        c[8] = (float)v.x; c[9] = (float)v.y; c[10] = (float)v.z;
        c[11] = (float)l.x; c[12] = (float)l.y; c[13] = (float)l.z;
        c[14] = (float)rng_real(&r); c[15] = (float)rng_real(&r); c[16] = (float)rng_real(&r);
    }
    printf("test_gpu_brdf: %d cases on %s\n", N, gpu_message(g));
    if (!gpu_bsdf_run(g, in, N, out)){ printf("the GPU did not run the check: FAIL\n"); return 1; }
    int bad_eval = 0, bad_pdf = 0, bad_flags = 0, bad_dir = 0, bad_w = 0, samples = 0;
    double worst_f = 0, worst_w = 0;
    for (int i = 0; i < N; i++){
        const float *c = in + (size_t)i * GPU_BSDF_IN, *o = out + (size_t)i * GPU_BSDF_OUT;
        Material m = material_pbr(vec3(c[0], c[1], c[2]), c[3], c[4], c[5], c[6]);
        Bsdf b = bsdf_make(&m, c[7] > 0.5f);
        Vec3 n = vec3(0, 0, 1), v = vec3(c[8], c[9], c[10]), l = vec3(c[11], c[12], c[13]);
        real pdf;
        Vec3 f = bsdf_eval(&b, n, v, l, &pdf);
        double ef = fmax(rel(f.x, o[0]), fmax(rel(f.y, o[1]), rel(f.z, o[2])));
        if (ef > 2e-3) bad_eval++;
        if (ef > worst_f && ef < 1) worst_f = ef;
        if (rel(pdf, o[3]) > 2e-3) bad_pdf++;
        Bsdf b2 = bsdf_make(&m, c[7] > 0.5f);
        BsdfSample s;
        int ok = bsdf_sample(&b2, n, v, c[14], c[15], c[16], &s);
        if (ok != (o[4] > 0.5f) || (ok && (s.delta != (o[5] > 0.5f) || s.through != (o[6] > 0.5f)))){ bad_flags++; continue; }
        if (!ok) continue;
        samples++;
        if (fabs(s.l.x - o[8]) + fabs(s.l.y - o[9]) + fabs(s.l.z - o[10]) > 3e-3){ bad_dir++; continue; }
        double ew = fmax(rel(s.w.x, o[11]), fmax(rel(s.w.y, o[12]), rel(s.w.z, o[13])));
        if (ew > 2e-3 || rel(s.pdf, o[7]) > 2e-3){
            if (getenv("LR_DEBUG") && bad_w < 8)
                printf("  case %d (kind %d, rough %g, metal %g, trans %g, out %g, delta %d): w %g %g %g / %g %g %g, pdf %g / %g\n", i, i % 6,
                       c[4], c[3], c[6], c[7], s.delta, (double)s.w.x, (double)s.w.y, (double)s.w.z, o[11], o[12], o[13], (double)s.pdf, o[7]);
            bad_w++;
        }
        if (ew > worst_w && ew < 1) worst_w = ew;
    }
    int bad = bad_eval + bad_pdf + bad_flags + bad_dir + bad_w;
    printf("  evaluation: %d differ (worst kept %.1e), densities: %d differ\n", bad_eval, worst_f, bad_pdf);
    printf("  sampling (%d drawn): %d decided otherwise, %d other directions, %d other weights (worst kept %.1e)\n",
           samples, bad_flags, bad_dir, bad_w, worst_w);
    int pass = bad <= N / 2000;
    printf("%s\n", pass ? "the WGSL BSDF matches the C" : "the WGSL BSDF differs from the C: FAIL");
    free(in);
    free(out);
    gpu_destroy(g);
    return pass ? 0 : 1;
}
