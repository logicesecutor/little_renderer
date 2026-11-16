// probe.c - light probes (see probe.h)
#include "rt/probe.h"

#include <stdlib.h>
#include <string.h>
#include "base/rng.h"
#include "base/sampler.h"
#include "rt/tracer.h"
#include "rt/accel.h"

Env *probe_bake(const Scene *s, Vec3 center, const Aabb *box, int w, int spp){
    if (w < 8) w = 8;
    w &= ~1;
    int h = w / 2;
    if (spp < 1) spp = 1;
    float *img = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (!img) return NULL;
    TraceParams tp = trace_default_params();
    tp.max_depth = PROBE_DEPTH;
    tp.sampler = 0;
    tp.no_lamp_seen = 1;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++){
        Rng rng;
        rng_seed(&rng, 0x9e3779b97f4a7c15ull ^ (uint64_t)y, 977);
        for (int x = 0; x < w; x++){
            Vec3 sum = vec3(0, 0, 0);
            for (int i = 0; i < spp; i++){
                // uniform over the texel's solid angle: u uniform, cos(theta) uniform in the row
                real u = (x + rng_real(&rng)) / w;
                real c0 = r_cos(y * (real)PI / h), c1 = r_cos((y + 1) * (real)PI / h);
                real c = c0 + (c1 - c0) * rng_real(&rng);
                real v = r_acos(c < -1 ? -1 : c > 1 ? 1 : c) * (real)(1.0 / PI);
                Env tmp = {0};                          // (env_dir of an unturned map)
                Ray ray = {center, env_dir(&tmp, u, v)};
                Sampler sm = sampler_make(&rng, 0, 0, 0, 0, 0);
                Vec3 L = trace_path_sampled(s, ray, &sm, &tp);
                if (L.x == L.x && L.y == L.y && L.z == L.z) sum = v_add(sum, L);   // (no NaN)
            }
            float *o = img + 3 * ((size_t)y * w + x);
            o[0] = (float)(sum.x / spp); o[1] = (float)(sum.y / spp); o[2] = (float)(sum.z / spp);
        }
    }
    Env *e = env_from_rgb(img, w, h);
    if (e && box){
        e->boxed = 1;
        e->center = center;
        e->bmin = box->lo;
        e->bmax = box->hi;
    }
    return e;
}

// How open the space around c is: the harmonic mean of how far 64 rays in all directions
// go before they hit something (most of them hitting a shape from its back: c is inside
// something, 0; a few: c is behind an open surface, as a glTF model's thin panels); rays
// that hit nothing count as `far`
static real openness_ex(const Scene *s, Vec3 c, real far, int *escaped){
    real inv = 0;
    int backs = 0;
    *escaped = 0;
    for (int i = 0; i < 64; i++){
        real z = 1 - 2 * (i + 0.5f) / 64, r = r_sqrt(r_fmax(0, 1 - z * z)), ph = 2.399963f * i;   // a Fibonacci sphere
        Ray ray = {c, vec3(r * r_cos(ph), z, r * r_sin(ph))};
        Hit h;
        if (scene_intersect(s, &ray, 1e-3f, far, &h)){
            // the back of a closed shape (a mesh, a sphere, an instance; not a quad: walls
            // are seen from either side)
            int closed = h.inst >= 0 || h.prim >= s->nprims || s->prims[h.prim].type == PRIM_SPHERE;
            if (!h.front_face && closed && s->mats[hit_material(s, &h)].type != MAT_LIGHT) backs++;
            inv += 1 / r_fmax(h.t, 1e-3f);
        } else { inv += 1 / far; (*escaped)++; }
    }
    return backs > 32 ? 0 : 64 / inv;
}
static real openness(const Scene *s, Vec3 c, real far){
    int e;
    return openness_ex(s, c, far, &e);
}

// Where the probe goes: the place asked for, unless a point of a 3 x 3 x 3 grid over the box
// (25%, 50%, 75% of each side; an open scene's: around the place, up to 300 units away) is
// more open (openness): a model standing in the middle of the room would hide the room
// from a probe inside it or just above it
static Vec3 probe_place(const Scene *s, Vec3 at, const Aabb *box){
    Vec3 lo = box->lo, hi = box->hi;
    real far = v_len(v_sub(hi, lo));
    if (far > 4000){                                  // an open scene: around the place
        lo = v_sub(at, vec3(300, 0, 300));
        hi = v_add(at, vec3(300, 300, 300));
        far = 4000;
    }
    Vec3 best = at;
    real best_o = openness(s, at, far) * 1.05f;      // (the place asked for wins a near tie)
    for (int i = 1; i <= 3; i++)
        for (int j = 1; j <= 3; j++)
            for (int k = 1; k <= 3; k++){
                Vec3 c = vec3(lo.x + (hi.x - lo.x) * i / 4, lo.y + (hi.y - lo.y) * j / 4, lo.z + (hi.z - lo.z) * k / 4);
                real o = openness(s, c, far);
                if (o > best_o){ best_o = o; best = c; }
            }
    return best;
}

IrrGrid *probe_grid_bake_scene(const Scene *s, int paths){
    if (s->light < 0 && !s->env) return NULL;
    Vec3 lo, hi;
    int n[3] = {5, 5, 5};
    if (s->probe_set && v_len(v_sub(s->probe_box.hi, s->probe_box.lo)) < 4000){
        lo = s->probe_box.lo; hi = s->probe_box.hi;
    } else if (s->probe_set){                         // open: around the probe, from the ground up
        lo = vec3(s->probe_at.x - 700, s->probe_box.lo.y, s->probe_at.z - 700);
        hi = vec3(s->probe_at.x + 700, s->probe_box.lo.y + 400, s->probe_at.z + 700);
        n[1] = 3;
    } else return NULL;
    IrrGrid *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    size_t np = (size_t)n[0] * n[1] * n[2];
    g->sh = calloc(np * IRR_FLOATS, sizeof(float));
    if (!g->sh){ free(g); return NULL; }
    for (int k = 0; k < 3; k++) g->n[k] = n[k];
    g->lo = lo; g->hi = hi;
    static unsigned next_id;
    g->id = ++next_id;
    if (paths < 16) paths = 16;
    TraceParams tp = trace_default_params();
    tp.max_depth = PROBE_DEPTH;
    tp.sampler = 0;
    tp.no_lamp_seen = 1;
    tp.no_sky_seen = 1;
    real far = v_len(v_sub(hi, lo)) * 4;
    // a probe inside a box of quads (they are seen from either side: no back to hit) is
    // known by none of its rays leaving the scene, where they do from the probe's own place
    int ref_escaped = 0;
    if (s->probe_set) openness_ex(s, s->probe_at, far, &ref_escaped);
    static const double band[9] = {1, 2.0 / 3, 2.0 / 3, 2.0 / 3, 0.25, 0.25, 0.25, 0.25, 0.25};
    #pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < (int)np; idx++){
        int ix = idx % n[0], iy = (idx / n[0]) % n[1], iz = idx / (n[0] * n[1]);
        Vec3 c = vec3(lo.x + (hi.x - lo.x) * (ix + 0.5f) / n[0], lo.y + (hi.y - lo.y) * (iy + 0.5f) / n[1],
                      lo.z + (hi.z - lo.z) * (iz + 0.5f) / n[2]);
        float *q = g->sh + IRR_FLOATS * (size_t)idx;
        int escaped;
        if (openness_ex(s, c, far, &escaped) <= 0 || (ref_escaped > 0 && escaped == 0)) continue;   // inside something: unused
        Rng rng;
        rng_seed(&rng, 0x51ed27u + (uint64_t)idx, 31);
        double acc[9][3];
        memset(acc, 0, sizeof(acc));
        real turn = rng_real(&rng) * (real)(2 * PI);
        for (int i = 0; i < paths; i++){
            // a Fibonacci sphere, turned at random per probe
            real z = 1 - 2 * (i + rng_real(&rng)) / paths, r = r_sqrt(r_fmax(0, 1 - z * z)), ph = 2.399963f * i + turn;
            Vec3 d = vec3(r * r_cos(ph), z, r * r_sin(ph));
            Ray ray = {c, d};
            Sampler sm = sampler_make(&rng, 0, 0, 0, 0, 0);
            Vec3 L = trace_path_sampled(s, ray, &sm, &tp);
            if (!(L.x == L.x && L.y == L.y && L.z == L.z)) continue;
            // a rare, very bright sample (the lamp seen through glass) would leave a blotch:
            // clamped (the diffuse light is a little low where caustics fall). (The sky seen
            // directly is not in these: sampled below, by its luminance.)
            real mx = v_max(L);
            if (mx > PROBE_GRID_CLAMP) L = v_scale(L, PROBE_GRID_CLAMP / mx);
            real y[9];
            env_sh_basis(d, y);
            for (int b = 0; b < 9; b++){ acc[b][0] += L.x * y[b]; acc[b][1] += L.y * y[b]; acc[b][2] += L.z * y[b]; }
        }
        // the sky seen directly: directions drawn by its luminance (a sun is found), each
        // counted if nothing is in the way
        double sky[9][3];
        memset(sky, 0, sizeof(sky));
        if (s->env){
            for (int i = 0; i < paths; i++){
                Vec3 d;
                real pdf;
                Vec3 L = env_sample(s->env, (i + rng_real(&rng)) / paths, rng_real(&rng), &d, &pdf);
                if (!(pdf > 0)) continue;
                Ray ray = {c, d};
                if (scene_occluded(s, &ray, 1e-3f, 1e30f)) continue;
                real y[9];
                env_sh_basis(d, y);
                for (int b = 0; b < 9; b++){ sky[b][0] += L.x * y[b] / pdf; sky[b][1] += L.y * y[b] / pdf; sky[b][2] += L.z * y[b] / pdf; }
            }
        }
        for (int b = 0; b < 9; b++)
            for (int k = 0; k < 3; k++) q[3 * b + k] = (float)((acc[b][k] * 4 * PI + sky[b][k]) / paths * band[b]);
        q[27] = 1;
    }
    return g;
}

Env *probe_bake_scene(const Scene *s, int w, int spp){
    if (s->light < 0 && !s->env) return NULL;
    if (w <= 0) w = s->light >= 0 ? PROBE_RES : PROBE_RES_OPEN;
    if (s->probe_set) return probe_bake(s, probe_place(s, s->probe_at, &s->probe_box), &s->probe_box, w, spp);
    if (s->light < 0) return NULL;
    Aabb b = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};   // lo, hi
    int any = 0;
    for (int i = 0; i < s->nprims; i++){
        const Primitive *p = &s->prims[i];
        if (i == s->light || p->type != PRIM_QUAD) continue;
        Vec3 c[4] = {p->q, v_add(p->q, p->u), v_add(p->q, p->v), v_add(v_add(p->q, p->u), p->v)};
        for (int k = 0; k < 4; k++){
            b.lo = vec3(r_fmin(b.lo.x, c[k].x), r_fmin(b.lo.y, c[k].y), r_fmin(b.lo.z, c[k].z));
            b.hi = vec3(r_fmax(b.hi.x, c[k].x), r_fmax(b.hi.y, c[k].y), r_fmax(b.hi.z, c[k].z));
        }
        any = 1;
    }
    if (!any) return NULL;
    Vec3 center = v_scale(v_add(b.lo, b.hi), 0.5f);
    return probe_bake(s, probe_place(s, center, &b), &b, w, spp);
}
