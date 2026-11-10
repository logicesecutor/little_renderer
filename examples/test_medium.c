// test_medium.c - checks the participating-medium sampling against exact answers.
//
//   make bin/test_medium && bin/test_medium
//
// 1. Homogeneous medium: the transmittance over a distance L is exactly exp(-sigma * L).
//    Ratio tracking (average of its estimates) and delta tracking (fraction of rays that get
//    through) must both converge to it.
// 2. Non-uniform medium (density rising along x): the exact transmittance is exp(-integral),
//    computed here by fine numerical integration of volume_density.
//    The same along 3 rays through a blob in a big grid (the majorant-grid walk).
//    Two overlapping media: transmittances multiply; collisions split by extinction.
// 3. Henyey-Greenstein sampling: the average cosine of the scattering angle must equal g.
// Each check prints the estimate, the exact value and the error in standard errors.
#include <math.h>
#include <stdio.h>
#include "lr.h"

static int report(const char *what, double est, double exact, double stderr_){
    double z = stderr_ > 0 ? fabs(est - exact) / stderr_ : 0;
    int ok = z < 4.0;                    // a correct estimator is within 4 standard errors
    printf("  %-44s %.5f  exact %.5f  (%.1f sigma) %s\n", what, est, exact, z, ok ? "ok" : "FAIL");
    return ok;
}

int main(void){
    const int N = 400000;
    int ok = 1;
    Rng rng;
    rng_seed(&rng, 12345, 1);

    // ---- 1. homogeneous cube 0..100 filled with density 1
    Aabb box = {vec3(0, 0, 0), vec3(100, 100, 100)};
    Volume *v = volume_create(8, 8, 8, box);
    for (int i = 0; i < 8 * 8 * 8; i++) v->density[i] = 1;
    v->sigma_t = 0.02f;
    volume_update(v);
    // a ray through the middle along x, from x = 30 to x = 80: tau = 0.02 * 50 = 1
    // (well inside: near the faces the trilinear interpolation fades to 0 over one voxel)
    Ray r = {vec3(0, 50, 50), vec3(1, 0, 0)};
    double exact = exp(-0.02 * 50.0);
    double sum = 0, sum2 = 0;
    int through = 0;
    for (int i = 0; i < N; i++){
        double t = medium_transmittance(v, &r, 30, 80, &rng);
        sum += t; sum2 += t * t;
        real th;
        if (!medium_sample_distance(v, &r, 30, 80, &rng, &th)) through++;
    }
    double m = sum / N, var = sum2 / N - m * m;
    printf("homogeneous medium, optical depth 1\n");
    ok &= report("ratio tracking transmittance", m, exact, sqrt(var / N));
    double p = (double)through / N;
    ok &= report("delta tracking: fraction that gets through", p, exact, sqrt(p * (1 - p) / N));

    // ---- 2. density rising linearly along x (0 at x = 0, 1 at x = 100)
    for (int z = 0; z < 8; z++) for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++)
        v->density[(z * 8 + y) * 8 + x] = (float)volume_voxel_center(v, x, 0, 0).x / 100.0f;
    v->sigma_t = 0.04f;
    volume_update(v);
    double tau = 0;
    const int STEPS = 200000;
    for (int i = 0; i < STEPS; i++){
        double x = 10 + 80.0 * (i + 0.5) / STEPS;
        tau += v->sigma_t * volume_density(v, vec3(x, 50, 50)) * 80.0 / STEPS;
    }
    exact = exp(-tau);
    sum = sum2 = 0; through = 0;
    for (int i = 0; i < N; i++){
        double t = medium_transmittance(v, &r, 10, 90, &rng);
        sum += t; sum2 += t * t;
        real th;
        if (!medium_sample_distance(v, &r, 10, 90, &rng, &th)) through++;
    }
    m = sum / N; var = sum2 / N - m * m;
    printf("density rising along x, optical depth %.3f\n", tau);
    ok &= report("ratio tracking transmittance", m, exact, sqrt(var / N));
    p = (double)through / N;
    ok &= report("delta tracking: fraction that gets through", p, exact, sqrt(p * (1 - p) / N));

    // ---- 2b. a blob in a big grid: most bricks of the majorant grid are empty or thin, so
    // the walk takes many differently sized steps (the brick-by-brick DDA is what is tested)
    volume_free(v);
    v = volume_create(64, 48, 40, box);
    for (int z = 0; z < 40; z++) for (int y = 0; y < 48; y++) for (int x = 0; x < 64; x++){
        Vec3 c = volume_voxel_center(v, x, y, z);
        double d2 = (c.x - 55) * (c.x - 55) + (c.y - 45) * (c.y - 45) + (c.z - 50) * (c.z - 50);
        v->density[((size_t)z * 48 + y) * 64 + x] = (float)(d2 < 30 * 30 ? exp(-d2 / 300.0) : 0);
    }
    v->sigma_t = 0.08f;
    volume_update(v);
    printf("blob in a 64x48x40 grid (%dx%dx%d bricks)\n", v->bnx, v->bny, v->bnz);
    Ray rays[3] = {{vec3(-10, 47, 51), v_norm(vec3(1, 0.05, -0.02))},
                   {vec3(55, -10, 30), v_norm(vec3(0.03, 1, 0.3))},
                   {vec3(120, 120, 120), v_norm(vec3(-1, -1.1, -1.05))}};
    for (int k = 0; k < 3; k++){
        tau = 0;
        for (int i = 0; i < STEPS; i++){
            double t = 200.0 * (i + 0.5) / STEPS;
            tau += v->sigma_t * volume_density(v, v_add(rays[k].o, v_scale(rays[k].d, t))) * 200.0 / STEPS;
        }
        exact = exp(-tau);
        sum = sum2 = 0; through = 0;
        for (int i = 0; i < N; i++){
            double t = medium_transmittance(v, &rays[k], 0, 200, &rng);
            sum += t; sum2 += t * t;
            real th;
            if (!medium_sample_distance(v, &rays[k], 0, 200, &rng, &th)) through++;
        }
        m = sum / N; var = sum2 / N - m * m;
        char what[64];
        snprintf(what, sizeof(what), "ray %d (tau %.3f): ratio tracking", k + 1, tau);
        ok &= report(what, m, exact, sqrt(var / N));
        p = (double)through / N;
        snprintf(what, sizeof(what), "ray %d: delta tracking gets through", k + 1);
        ok &= report(what, p, exact, sqrt(p * (1 - p) / N));
    }

    // ---- 2c. two overlapping uniform media: transmittances multiply, and a collision belongs
    // to medium i with probability sigma_i / (sigma_1 + sigma_2)
    {
        static Scene sc;
        Volume *a = volume_create(8, 8, 8, box), *b2 = volume_create(8, 8, 8, box);
        for (int i = 0; i < 512; i++){ a->density[i] = 1; b2->density[i] = 1; }
        a->sigma_t = 0.01f; b2->sigma_t = 0.03f;
        volume_update(a); volume_update(b2);
        scene_add_volume(&sc, a);
        scene_add_volume(&sc, b2);
        exact = exp(-(0.01 + 0.03) * 50.0);
        sum = sum2 = 0; through = 0;
        int first = 0, hits = 0;
        for (int i = 0; i < N; i++){
            double t = scene_medium_transmittance(&sc, &r, 30, 80, &rng);
            sum += t; sum2 += t * t;
            real th;
            int which;
            if (!scene_medium_collision(&sc, &r, 30, 80, &rng, &th, &which)) through++;
            else { hits++; if (which == 0) first++; }
        }
        m = sum / N; var = sum2 / N - m * m;
        printf("two overlapping media, sigma 0.01 + 0.03\n");
        ok &= report("transmittance of both", m, exact, sqrt(var / N));
        p = (double)through / N;
        ok &= report("first collision: none before the end", p, exact, sqrt(p * (1 - p) / N));
        double q = (double)first / hits;
        ok &= report("collision belongs to medium 1", q, 0.25, sqrt(q * (1 - q) / hits));
        scene_free(&sc);
    }

    // ---- 3. Henyey-Greenstein: E[cos theta] = g
    printf("Henyey-Greenstein sampling, mean cosine\n");
    double gs[] = {-0.5, 0.0, 0.3, 0.85};
    for (int k = 0; k < 4; k++){
        Vec3 d = v_norm(vec3(0.3, -0.5, 0.8));
        double s1 = 0, s2 = 0;
        for (int i = 0; i < N; i++){
            double c = v_dot(d, medium_sample_phase(d, (real)gs[k], &rng));
            s1 += c; s2 += c * c;
        }
        double mc = s1 / N, vc = s2 / N - mc * mc;
        char what[64];
        snprintf(what, sizeof(what), "g = %.2f", gs[k]);
        ok &= report(what, mc, gs[k], sqrt(vc / N));
    }
    // and the phase function integrates to 1 over the sphere (numerically, in cos theta)
    double integ = 0;
    for (int i = 0; i < 100000; i++){
        double c = -1 + 2.0 * (i + 0.5) / 100000;
        integ += 2 * PI * phase_hg((real)c, 0.85f) * 2.0 / 100000;
    }
    ok &= report("phase_hg(g = 0.85) integrates to", integ, 1.0, 1e-3);

    // ---- 4. the rasterizer's raymarching (raster/volume_march.c): a medium in an unlit
    // scene only absorbs, so a pixel behind a uniform slab of extinction s and thickness d
    // keeps exactly exp(-s d) of its (linear) light; the z-buffer must stop the ray
    printf("raster raymarching, transmittance of a uniform block (in sRGB steps)\n");
    {
        static Scene sc;                       // no lamp, no walls: no light to scatter
        Volume *u = volume_create(16, 16, 16, (Aabb){vec3(-50, -50, -50), vec3(50, 50, 50)});
        for (int i = 0; i < 16 * 16 * 16; i++) u->density[i] = 1;
        u->sigma_t = 0.01f;
        volume_update(u);
        scene_add_volume(&sc, u);
        sc.light = -1;
        const int W = 33;
        RenderParams rp = render_params_camera(vec3(0, 0, -200), vec3(0, 0, 0), vec3(0, 1, 0), 10, W, W);
        Framebuffer *fb = fb_create(W, W);
        RasterVolumes *rv = raster_volumes_create();
        const unsigned char bg = 200;
        double lin = pow((bg / 255.0 + 0.055) / 1.055, 2.4);
        for (int pass = 0; pass < 4; pass++){
            int half = pass & 1, wall = pass >> 1;
            fb_clear(fb, rgb8(bg, bg, bg));
            // a "surface" in the middle of the block (distance 200 on the axis): its depth
            double dist = 200;
            if (wall){
                double A = rp.proj.m[2][2], B = rp.proj.m[2][3];
                double zn = -A - B / (-dist);   // NDC z of view depth dist
                for (int i = 0; i < W * W; i++) fb->depth[i] = (real)(zn * 0.5 + 0.5);
            }
            MarchParams mp = march_default_params();
            mp.scale = half ? 2 : 1;
            raster_draw_volumes(rv, fb, &sc, &rp, &mp);
            double d = wall ? 50 : 100;     // the path through the block before the surface
            unsigned char want = image_srgb8(lin * exp(-0.01 * d));
            unsigned char got = (unsigned char)(fb->color[(W / 2) * W + W / 2] & 255);
            char what[96];
            snprintf(what, sizeof(what), "center pixel, %s, %s", wall ? "surface inside" : "nothing behind",
                     half ? "half resolution" : "full resolution");
            ok &= report(what, got, want, 1.0);
        }
        raster_volumes_free(rv);
        fb_destroy(fb);
        scene_free(&sc);
    }

    volume_free(v);
    printf(ok ? "all checks passed\n" : "SOME CHECKS FAILED\n");
    return ok ? 0 : 1;
}
