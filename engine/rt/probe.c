// probe.c - light probes (see probe.h)
#include "rt/probe.h"

#include <stdlib.h>
#include "core/rng.h"
#include "core/sampler.h"
#include "rt/tracer.h"

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

Env *probe_bake_scene(const Scene *s, int w, int spp){
    if (s->light < 0 && !s->env) return NULL;
    if (w <= 0) w = s->light >= 0 ? PROBE_RES : PROBE_RES_OPEN;
    if (s->probe_set) return probe_bake(s, s->probe_at, &s->probe_box, w, spp);
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
    return probe_bake(s, center, &b, w, spp);
}
