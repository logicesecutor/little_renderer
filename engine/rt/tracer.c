#include "rt/tracer.h"

#include <stdlib.h>

#include "scene/brdf.h"
#include "scene/env.h"
#include "scene/volume.h"
#include "rt/accel.h"
#include "rt/medium.h"
#include "base/sampler.h"

// Ray offsets that keep a new ray from re-hitting the surface it starts on ("acne").
// float has ~7 significant digits: at 555 units one step of float is ~6e-5, so the float
// build needs a larger offset than the double build. Override with -DRT_EPS=... if needed.
#ifndef RT_EPS
#ifdef LR_DOUBLE
#define RT_EPS 1e-4           // minimum ray distance
#define RT_SHADOW_MARGIN 1e-3 // shadow rays stop this far before the light
#else
#define RT_EPS 2e-3
#define RT_SHADOW_MARGIN 2e-2
#endif
#endif
#define RR_START 3            // bounces before Russian roulette may stop a path

// The power heuristic (Veach 1995): when two sampling techniques can produce the same
// path, with densities pa and pb (per unit solid angle), weight the one that produced it by
// pa^2 / (pa^2 + pb^2). The weights of the two add up to 1 for every path, so the sum stays
// unbiased, and each path is counted mostly by the technique that samples it best.
static inline real power_heuristic(real pa, real pb){
    real a = pa * pa, b = pb * pb;
    return a + b > 0 ? a / (a + b) : 0;
}

// Density per unit solid angle with which next event estimation picks direction wi: a point
// is picked with density 1 / area on the lamp, and an area dA at distance d, tilted by
// theta_light, covers the solid angle cos(theta_light) dA / d^2.
static inline real light_pdf(const Primitive *light, real dist2, real cos_light){
    return dist2 / (cos_light * light->area);
}

// Next event estimation (direct light sampling) at a surface.
// Pick one random point on the light, check it is visible, and return the light it sends
// towards the viewer, already multiplied by the BSDF (scene/brdf.h) and the cosines:
//
//   L_direct = f(v, wi) * Le * |cos(theta_surface)| * cos(theta_light) * area / dist^2
//
// The area / dist^2 * cos(theta_light) factor converts "pick a point with probability
// 1 / area" into "probability per unit solid angle", as the rendering equation needs.
// With MIS the result is weighted against the chance that the BSDF's sampling (bsdf_sample)
// would have found the same point. Only the BSDF's rough part: a perfect mirror or
// refraction cannot be aimed at a point.
// (ua, ub: where on the lamp, in [0, 1)^2. The callers pass the pair's second number as ua:
// the code before the sampler drew both inside one call's arguments, which gcc evaluates
// last to first.)
static Vec3 sample_direct_light(const Scene *s, const Hit *h, Bsdf *b, Vec3 v, int mis, real ua, real ub, Rng *rng){
    const Primitive *light = &s->prims[s->light];
    Vec3 le = s->mats[light->material].emission;

    Vec3 xl = v_add(light->q, v_add(v_scale(light->u, ua), v_scale(light->v, ub)));
    Vec3 to_light = v_sub(xl, h->p);
    real dist2 = v_dot(to_light, to_light);
    real dist = r_sqrt(dist2);
    Vec3 wi = v_scale(to_light, 1.0 / dist);

    real cos_surf = v_dot(h->n, wi);
    real cos_light = -v_dot(light->normal, wi);   // the light only emits on its normal side
    if (cos_light <= 0) return vec3(0, 0, 0);
    real pdf;
    Vec3 f = bsdf_eval(b, h->n, v, wi, &pdf);
    if (!(v_max(f) > 0)) return vec3(0, 0, 0);

    // shadow ray: anything between the point and the light blocks it
    Ray shadow = {h->p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, dist - RT_SHADOW_MARGIN)) return vec3(0, 0, 0);

    real g = r_fabs(cos_surf) * cos_light * light->area / dist2;
    if (s->nvolumes) g *= scene_medium_transmittance(s, &shadow, 0, dist, rng);   // a cloud in the way
    if (mis) g *= power_heuristic(light_pdf(light, dist2, cos_light), pdf);
    return v_scale(v_mul(f, le), g);
}

// Next event estimation towards the sky: a direction drawn by its luminance (env_sample),
// the light arriving along it if nothing is in the way:
//   L_direct = f(v, wi) * L_sky(wi) * |cos| / pdf_sky(wi)
// with MIS against the BSDF's own sampling of the same direction.
static Vec3 sample_direct_env(const Scene *s, const Hit *h, Bsdf *b, Vec3 v, int mis, real ua, real ub, Rng *rng){
    Vec3 wi;
    real pdf;
    Vec3 le = env_sample(s->env, ua, ub, &wi, &pdf);
    if (!(pdf > 0) || !(v_max(le) > 0)) return vec3(0, 0, 0);
    real bpdf;
    Vec3 f = bsdf_eval(b, h->n, v, wi, &bpdf);
    if (!(v_max(f) > 0)) return vec3(0, 0, 0);
    Ray shadow = {h->p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, 1e30f)) return vec3(0, 0, 0);
    real g = r_fabs(v_dot(h->n, wi)) / pdf;
    if (s->nvolumes) g *= scene_medium_transmittance(s, &shadow, 0, 1e6f, rng);
    if (mis) g *= power_heuristic(pdf, bpdf);
    return v_scale(v_mul(f, le), g);
}

// the same from a point in a medium (the phase function for the BSDF)
static Vec3 sample_env_in_medium(const Scene *s, Vec3 p, Vec3 d, real g, int mis, real ua, real ub, Rng *rng){
    Vec3 wi;
    real pdf;
    Vec3 le = env_sample(s->env, ua, ub, &wi, &pdf);
    if (!(pdf > 0) || !(v_max(le) > 0)) return vec3(0, 0, 0);
    Ray shadow = {p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, 1e30f)) return vec3(0, 0, 0);
    real tr = scene_medium_transmittance(s, &shadow, 0, 1e6f, rng);
    if (tr <= 0) return vec3(0, 0, 0);
    real ph = phase_hg(v_dot(d, wi), g);
    real w = ph * tr / pdf;
    if (mis) w *= power_heuristic(pdf, ph);
    return v_scale(le, w);
}

// ---- the glowing shapes as lights (Scene.emitters) ---------------------------------------
//
// A small, bright glowing ball (a lamp's bulb) lit the scene only through the paths that
// happened to hit it: a few found it, most did not, and the floor around it was full of
// fireflies. Next event estimation aims at it, as at the lamp: one of the glowing shapes,
// picked uniformly (1 / n), then
//   a ball: a direction in the cone it fills seen from p (Shirley et al. 1996), density
//           1 / (2 pi (1 - cos_max)) per unit solid angle, cos_max = sqrt(1 - r^2 / d^2);
//   a box:  a point on its surface, a face by its area: density d^2 / (cos_light * area).
// With MIS a path that hits one after a rough bounce is weighted against this (emit_pdf),
// without it not counted (NEE has it). The numbers come from the plain random stream, and
// only in a scene with such shapes: other scenes' images stay as they were, to the bit.

static int emitter_slot(const Scene *s, int inst){
    for (int k = 0; k < s->nemitters; k++) if (s->emitters[k] == inst) return k;
    return -1;
}

// 1 - cos_max of a ball of radius r whose centre is dist2 away (squared), without the
// cancellation of 1 - sqrt(1 - x) for a small, far ball; 0 from inside
static real ball_cone(real r, real dist2){
    real sin2 = r * r / dist2;
    if (!(sin2 < 1)) return 0;
    return sin2 / (1 + r_sqrt(1 - sin2));
}

// the density (per unit solid angle, the pick included) of reaching the shape's point at
// distance t along unit d from p, its outward normal n there; 0: NEE could not have
static real emit_pdf(const Scene *s, int k, Vec3 p, Vec3 d, real t, Vec3 n){
    Emitter e = scene_emitter(s, s->emitters[k]);
    if (e.ball){
        Vec3 to = v_sub(e.c, p);
        real om = ball_cone(e.r, v_dot(to, to));
        return om > 0 ? 1 / (2 * (real)PI * om * s->nemitters) : 0;
    }
    real cl = -v_dot(n, d);
    return cl > 0 ? t * t / (cl * e.area * s->nemitters) : 0;
}

// a point of a glowing shape seen from p: the direction, distance and density (per unit solid
// angle, the pick included), and what it sends; 0 if none (p inside it, or the face turned away)
static int emit_sample(const Scene *s, Vec3 p, real u0, real ua, real ub, Vec3 *wi, real *dist, real *pdf, Vec3 *le){
    int k = (int)(u0 * s->nemitters);
    if (k >= s->nemitters) k = s->nemitters - 1;
    Emitter e = scene_emitter(s, s->emitters[k]);
    *le = e.le;
    if (e.ball){
        Vec3 to = v_sub(e.c, p);
        real d2 = v_dot(to, to), om = ball_cone(e.r, d2);
        if (!(om > 0)) return 0;
        real dc = r_sqrt(d2), cos_t = 1 - ua * om, sin_t = r_sqrt(r_fmax(0, 1 - cos_t * cos_t)), ph = 2 * (real)PI * ub;
        Vec3 w = v_scale(to, 1 / dc), t, b;
        onb_from_normal(w, &t, &b);
        *wi = v_add(v_scale(w, cos_t), v_add(v_scale(t, sin_t * r_cos(ph)), v_scale(b, sin_t * r_sin(ph))));
        real disc = e.r * e.r - d2 * sin_t * sin_t;           // (the near side of the ball along wi)
        *dist = dc * cos_t - r_sqrt(r_fmax(0, disc));
        *pdf = 1 / (2 * (real)PI * om * s->nemitters);
        return *dist > 0;
    }
    // a box: a face by its area (two of each axis i, 2 a_j by 2 a_k), a point on it
    real fa[3];
    for (int i = 0; i < 3; i++) fa[i] = 4 * v_len(v_cross(e.a[(i + 1) % 3], e.a[(i + 2) % 3]));
    real x = ua * e.area;
    int f = 0;
    while (f < 5 && x >= fa[f >> 1]){ x -= fa[f >> 1]; f++; }
    int i = f >> 1;
    real sg = f & 1 ? -1 : 1, uu = r_fmin(x / fa[i], 1) * 2 - 1, vv = ub * 2 - 1;
    Vec3 aj = e.a[(i + 1) % 3], ak = e.a[(i + 2) % 3];
    Vec3 q = v_add(e.c, v_add(v_scale(e.a[i], sg), v_add(v_scale(aj, uu), v_scale(ak, vv))));
    Vec3 n = v_scale(v_norm(v_cross(aj, ak)), sg);
    if (v_dot(n, e.a[i]) * sg < 0) n = v_neg(n);              // (outward, whatever the axes' handedness)
    Vec3 to = v_sub(q, p);
    real d2 = v_dot(to, to);
    *dist = r_sqrt(d2);
    *wi = v_scale(to, 1 / *dist);
    real cl = -v_dot(n, *wi);
    if (!(cl > 0)) return 0;
    *pdf = d2 / (cl * e.area * s->nemitters);
    return 1;
}

// next event estimation of the glowing shapes at a surface (as sample_direct_light)
static Vec3 sample_direct_emit(const Scene *s, const Hit *h, Bsdf *b, Vec3 v, int mis, Rng *rng){
    real u0 = rng_real(rng), ua = rng_real(rng), ub = rng_real(rng);
    Vec3 wi, le;
    real dist, pdf;
    if (!emit_sample(s, h->p, u0, ua, ub, &wi, &dist, &pdf, &le)) return vec3(0, 0, 0);
    real bpdf;
    Vec3 f = bsdf_eval(b, h->n, v, wi, &bpdf);
    if (!(v_max(f) > 0)) return vec3(0, 0, 0);
    Ray shadow = {h->p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, dist - RT_SHADOW_MARGIN)) return vec3(0, 0, 0);
    real g = r_fabs(v_dot(h->n, wi)) / pdf;
    if (s->nvolumes) g *= scene_medium_transmittance(s, &shadow, 0, dist, rng);
    if (mis) g *= power_heuristic(pdf, bpdf);
    return v_scale(v_mul(f, le), g);
}
// ... and from a point in a medium (the phase function for the BSDF)
static Vec3 sample_emit_in_medium(const Scene *s, Vec3 p, Vec3 d, real g, int mis, Rng *rng){
    real u0 = rng_real(rng), ua = rng_real(rng), ub = rng_real(rng);
    Vec3 wi, le;
    real dist, pdf;
    if (!emit_sample(s, p, u0, ua, ub, &wi, &dist, &pdf, &le)) return vec3(0, 0, 0);
    Ray shadow = {p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, dist - RT_SHADOW_MARGIN)) return vec3(0, 0, 0);
    real tr = scene_medium_transmittance(s, &shadow, 0, dist, rng);
    real ph = phase_hg(v_dot(d, wi), g);
    real w = ph * tr / pdf;
    if (mis) w *= power_heuristic(pdf, ph);
    return v_scale(le, w);
}

// the numbers for the sky's sample: the lamp's pair when there is no lamp, else plain ones
static void env_numbers(const Scene *s, Sampler *sm, int depth, real *ua, real *ub){
    if (s->light < 0) sampler_2d(sm, SAMPLE_LIGHT(depth), ua, ub);
    else { *ua = rng_real(sm->rng); *ub = rng_real(sm->rng); }
}

// The material at a hit with its textures applied (PBR5; into *tmp when it has any), and
// the shading normal bent by its normal map. A bent normal that would turn away from v (the
// direction the ray came from) is tilted back towards it, just past the horizon: the
// light that gets there must still reach the viewer.
static const Material *hit_shading(const Scene *s, Hit *h, Vec3 v, Material *tmp){
    const Material *m = &s->mats[hit_material(s, h)];
    if (!material_textured(m) || m->type == MAT_LIGHT) return m;
    Vec2 uv;
    Vec4 tg;
    hit_texcoords(s, h, &uv, &tg);
    *tmp = material_at(s, m, uv, MATERIAL_LOD0);
    if (m->tex_normal){
        Vec3 n = material_normal(s, m, uv, MATERIAL_LOD0, h->n, tg);
        real c = v_dot(n, v);
        if (c < 1e-3f) n = v_norm(v_add(n, v_scale(v, 1e-3f - c)));
        h->n = n;
    }
    return tmp;
}

// the BSDF has a part next event estimation can find (not only perfect mirror / refraction)
static inline int bsdf_rough_part(const Bsdf *b){
    return !b->smooth || (1 - b->metallic) * (1 - b->transmission) > 0;
}

// The same for a point p inside a medium, reached by a ray travelling along d. There is no
// surface, so no cos(theta_surface): the phase function takes the place of the BRDF, and
// the light is dimmed by the medium between p and the lamp (the cloud's self-shadowing):
//
//   L_direct = phase(d, wi) * Le * T(p <-> light) * cos(theta_light) * area / dist^2
static Vec3 sample_light_in_medium(const Scene *s, Vec3 p, Vec3 d, real g, int mis, real ua, real ub, Rng *rng){
    const Primitive *light = &s->prims[s->light];
    Vec3 le = s->mats[light->material].emission;

    Vec3 xl = v_add(light->q, v_add(v_scale(light->u, ua), v_scale(light->v, ub)));
    Vec3 to_light = v_sub(xl, p);
    real dist2 = v_dot(to_light, to_light);
    real dist = r_sqrt(dist2);
    Vec3 wi = v_scale(to_light, 1.0 / dist);
    real cos_light = -v_dot(light->normal, wi);
    if (cos_light <= 0) return vec3(0, 0, 0);

    Ray shadow = {p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, dist - RT_SHADOW_MARGIN)) return vec3(0, 0, 0);
    real tr = scene_medium_transmittance(s, &shadow, 0, dist, rng);
    if (tr <= 0) return vec3(0, 0, 0);

    // light arrives travelling along -wi and leaves along -d: the scattering angle has
    // cos = dot(-wi, -d) = dot(wi, d)
    real ph = phase_hg(v_dot(d, wi), g);
    real w = ph * tr * cos_light * light->area / dist2;
    if (mis) w *= power_heuristic(light_pdf(light, dist2, cos_light), ph);
    return v_scale(le, w);
}

// Russian roulette: after a few bounces, stop dim paths at random and boost the survivors
// by 1/p. On average the result is unchanged (unbiased) but much cheaper. Returns 0 to stop.
// u: the number that decides (SAMPLE_MISC's first).
static int russian_roulette_u(Vec3 *throughput, int depth, real u){
    if (depth < RR_START) return 1;
    real p = r_fmin(v_max(*throughput), 0.95);
    if (u >= p) return 0;
    *throughput = v_scale(*throughput, 1.0 / p);
    return 1;
}
static int russian_roulette(Vec3 *throughput, int depth, Sampler *sm){
    return depth < RR_START || russian_roulette_u(throughput, depth, sampler_1d(sm, SAMPLE_MISC(depth), 0));
}

TraceParams trace_default_params(void){
    TraceParams p = {.max_depth = 50, .mis = 1, .skip_caustics = 0, .sampler = 1, .no_lamp_seen = 0, .no_sky_seen = 0};
    return p;
}

Vec3 trace_path(const Scene *s, Ray ray, Rng *rng, int max_depth){
    TraceParams p = trace_default_params();
    p.max_depth = max_depth;
    return trace_path_ex(s, ray, rng, &p);
}

// What a lamp hit means after the previous vertex:
enum {
    EMIT_FULL,   // camera rays, perfect mirror and refraction bounces: nothing else could find the lamp
    EMIT_NONE,   // after a rough / medium bounce without MIS: already counted by NEE
    EMIT_MIS     // after a rough / medium bounce with MIS: weight against NEE
};

Vec3 trace_path_ex(const Scene *s, Ray ray, Rng *rng, const TraceParams *tp){
    Sampler sm = sampler_make(rng, 0, 0, 0, 0, 0);      // no pixel, no sequence: all from rng
    return trace_path_sampled(s, ray, &sm, tp);
}

void trace_image(const Scene *s, const Camera *cam, const TraceParams *tp, int W, int H, int spp, uint64_t seed,
                 real clamp, Vec3 *px, void (*row_done)(void *user), void *user){
    // rows are independent: handed out to the threads one at a time
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            uint64_t pixel = (uint64_t)y * W + x;
            Rng rng;
            rng_seed(&rng, 0x853c49e6748fea9bULL ^ (pixel * 0x9E3779B97F4A7C15ULL) ^ (seed * 0xD1B54A32D192ED03ULL), pixel);
            Vec3 sum = vec3(0, 0, 0);
            for (int k = 0; k < spp; k++){
                // a random point inside the pixel: averaging them gives anti-aliasing for free
                Sampler sm = sampler_make(&rng, x, y, (uint32_t)k, (uint32_t)seed * 0x9e3779b9u, tp->sampler);
                real jx, jy;
                sampler_2d(&sm, SAMPLE_CAMERA, &jx, &jy);
                real u = (x + jx) / W, v = (y + jy) / H;
                Vec3 c = trace_path_sampled(s, camera_ray(cam, u, v), &sm, tp);
                if (!(c.x == c.x && c.y == c.y && c.z == c.z)) continue;   // skip NaNs
                if (clamp > 0 && v_max(c) > clamp) c = v_scale(c, clamp / v_max(c));
                sum = v_add(sum, c);
            }
            px[pixel] = v_scale(sum, (real)1 / spp);
        }
        if (row_done) row_done(user);
    }
}

Vec3 trace_path_sampled(const Scene *s, Ray ray, Sampler *sm, const TraceParams *tp){
    Rng *rng = sm->rng;
    const int max_depth = tp->max_depth, mis = tp->mis;
    Vec3 radiance = vec3(0, 0, 0);    // what we have collected so far
    Vec3 throughput = vec3(1, 1, 1);  // how much of the light found from here on reaches the camera
    int emit = EMIT_FULL;             // how to count the lamp if the next hit is the lamp
    real bounce_pdf = 0;              // EMIT_MIS: solid-angle density of the last bounce direction
    // skip_caustics: the camera saw a diffuse surface first (caustic_chain) and the path is
    // in no medium; caustic_spec counts the mirror / glass vertices since the last diffuse
    // one. If such a path reaches the lamp right after mirror / glass (camera -> D ... S -> lamp),
    // light tracing has it covered.
    int caustic_chain = 0, caustic_spec = 0;
    const int media = s->nvolumes > 0;
    if (media || mis) ray.d = v_norm(ray.d); // distances and pdfs below need a unit direction

    for (int depth = 0; ; depth++){
        Hit h;
        int found = scene_intersect(s, &ray, RT_EPS, 1e30, &h);

        // In a medium the ray may scatter before it reaches the surface: delta tracking picks
        // the point. If it does, that point replaces the surface hit for this bounce.
        real ts;
        int which;
        if (media && scene_medium_collision(s, &ray, 0, found ? h.t : 1e30, rng, &ts, &which)){
            const Volume *vol = s->volumes[which];
            if (depth >= max_depth) break;
            Vec3 p = v_add(ray.o, v_scale(ray.d, ts));
            // a real collision scatters with probability sigma_s / sigma_t = albedo (else absorbed)
            throughput = v_mul(throughput, vol->albedo);
            // 1. light arriving straight from the lamp (and the sky), through the cloud
            real la, lb;
            if (s->light >= 0){
                sampler_2d(sm, SAMPLE_LIGHT(depth), &la, &lb);
                radiance = v_add(radiance, v_mul(throughput, sample_light_in_medium(s, p, ray.d, vol->g, mis, lb, la, rng)));
            }
            if (s->env){
                env_numbers(s, sm, depth, &la, &lb);
                radiance = v_add(radiance, v_mul(throughput, sample_env_in_medium(s, p, ray.d, vol->g, mis, la, lb, rng)));
            }
            if (s->nemitters) radiance = v_add(radiance, v_mul(throughput, sample_emit_in_medium(s, p, ray.d, vol->g, mis, rng)));
            // 2. continue in a direction drawn from the phase function. Its pdf equals the
            //    phase function, so the weight phase / pdf is 1: throughput is unchanged.
            real da, db;
            sampler_2d(sm, SAMPLE_DIR(depth), &da, &db);
            Vec3 nd = medium_sample_phase_u(ray.d, vol->g, da, db);
            bounce_pdf = phase_hg(v_dot(ray.d, nd), vol->g);
            ray.o = p;
            ray.d = nd;
            emit = mis ? EMIT_MIS : EMIT_NONE;
            caustic_chain = 0;
            if (!russian_roulette(&throughput, depth, sm)) break;
            continue;
        }
        if (!found){                          // escaped: the sky (black without one)
            if (s->env && emit != EMIT_NONE && !(depth == 0 && tp->no_sky_seen)){
                Vec3 le = env_eval(s->env, ray.d);
                if (emit == EMIT_MIS) le = v_scale(le, power_heuristic(bounce_pdf, env_pdf(s->env, ray.d)));
                radiance = v_add(radiance, v_mul(throughput, le));
            }
            break;
        }

        Material mtex;
        const Material *m = hit_shading(s, &h, v_neg(v_norm(ray.d)), &mtex);

        if (m->type == MAT_LIGHT){
            if (h.front_face && !(caustic_chain && caustic_spec) && !(depth == 0 && tp->no_lamp_seen)){
                if (emit == EMIT_FULL){
                    radiance = v_add(radiance, v_mul(throughput, m->emission));
                } else if (emit == EMIT_MIS && h.inst < 0 && h.prim == s->light){
                    // the bounce found the lamp: weight against next event estimation
                    const Primitive *l = &s->prims[s->light];
                    real cos_light = -v_dot(l->normal, ray.d);
                    if (cos_light > 0){
                        real w = power_heuristic(bounce_pdf, light_pdf(l, h.t * h.t, cos_light));
                        radiance = v_add(radiance, v_mul(throughput, v_scale(m->emission, w)));
                    }
                }
                // EMIT_NONE: the light was already counted by next event estimation; adding
                // it again here would count it twice. (A second, unsampled MAT_LIGHT object
                // seen after a diffuse bounce is not counted either way.)
            }
            break;
        }
        // a glowing surface (glTF's emissive): found by the paths that hit it; a glowing shape
        // after a rough bounce (or a medium's) was sampled as a light: weighted against that,
        // or (without MIS) counted there already. (From inside it, or one NEE could not reach
        // from the last vertex: counted here, whole.)
        if (material_glows(m)){
            int k = s->nemitters && emit != EMIT_FULL && h.inst >= 0 ? emitter_slot(s, h.inst) : -1;
            real pe = k >= 0 && h.front_face ? emit_pdf(s, k, ray.o, v_norm(ray.d), h.t * v_len(ray.d), h.ng) : 0;
            if (!(pe > 0)) radiance = v_add(radiance, v_mul(throughput, m->emission));
            else if (emit == EMIT_MIS) radiance = v_add(radiance, v_mul(throughput, v_scale(m->emission, power_heuristic(bounce_pdf, pe))));
        }
        if (depth >= max_depth) break;

        // the surface's BSDF (scene/brdf.h), seen from v (back along the ray)
        Vec3 v = v_neg(v_norm(ray.d));
        Bsdf b = bsdf_make(m, h.front_face);
        // 1. light arriving straight from the lamp and the sky (the BSDF's rough part)
        if (bsdf_rough_part(&b)){
            real la, lb;
            if (s->light >= 0){
                sampler_2d(sm, SAMPLE_LIGHT(depth), &la, &lb);
                radiance = v_add(radiance, v_mul(throughput, sample_direct_light(s, &h, &b, v, mis, lb, la, rng)));
            }
            if (s->env){
                env_numbers(s, sm, depth, &la, &lb);
                radiance = v_add(radiance, v_mul(throughput, sample_direct_env(s, &h, &b, v, mis, la, lb, rng)));
            }
            if (s->nemitters) radiance = v_add(radiance, v_mul(throughput, sample_direct_emit(s, &h, &b, v, mis, rng)));
        }
        // 2. continue the path in a direction drawn from the BSDF: its weight f cos / pdf.
        //    (The numbers only when they are used: SAMPLE_MISC's pair for Russian roulette
        //    and the lobe, SAMPLE_DIR's for a direction that is not fixed.)
        bsdf_view(&b, v_dot(v, h.n));
        real ur = 0, ul = 0, da = 0, db = 0;
        if (depth >= RR_START || bsdf_nlobes(&b) > 1) sampler_2d(sm, SAMPLE_MISC(depth), &ur, &ul);
        int lobe = bsdf_pick(&b, ul);
        if (lobe >= 0 && !bsdf_lobe_delta(&b, lobe)) sampler_2d(sm, SAMPLE_DIR(depth), &da, &db);
        BsdfSample bs;
        if (!bsdf_sample_lobe(&b, lobe, h.n, v, da, db, &bs)) break;
        Vec3 next = bs.l;
        throughput = v_mul(throughput, bs.w);
        if (bs.delta){                        // a perfect mirror or refraction
            emit = EMIT_FULL;
            caustic_spec = 1;
        } else {
            bounce_pdf = bs.pdf;
            emit = mis ? EMIT_MIS : EMIT_NONE;
            if (depth == 0) caustic_chain = tp->skip_caustics;
            caustic_spec = 0;
        }

        if (!russian_roulette_u(&throughput, depth, ur)) break;

        ray.o = h.p;
        ray.d = (media || mis) ? v_norm(next) : next;
    }
    return radiance;
}

// ---- light tracing for caustics -------------------------------------------------------
//
// A caustic on the floor (lamp light focused by a glass sphere) is the path
//   camera <- floor (diffuse) <- glass ... <- lamp,
// seen after more bounces: camera <- wall <- floor <- glass <- lamp, or in a mirror:
// camera <- wall <- mirror <- floor <- glass <- lamp.
// The path tracer can only find it by chance: from the floor it bounces at random, and must
// happen to go through the glass and hit the small lamp. Next event estimation cannot help,
// because the glass blocks the shadow ray. Traced the other way it is easy: start at the
// lamp, go through the glass (mirror and glass have only one or two possible directions),
// land on the floor, and from there connect to the camera with a shadow ray. Every light
// path that gets there contributes, so the caustic is sharp after a few samples. The light
// path then keeps bouncing (diffuse, mirror, glass), and every diffuse vertex it reaches is
// connected to the camera too.
// So light tracing covers exactly the paths  lamp -> S -> (anything) -> D -> camera: the
// first vertex after the lamp is a mirror or glass, the last one before the camera diffuse,
// no medium. The path tracer leaves out exactly these (skip_caustics: it saw a diffuse
// surface first and reaches the lamp right after a mirror / glass), so every path is counted
// once.
//
// Contribution of a light path landing at x (Veach 1997, "light tracing"):
//   beta    = Le * area * pi * (product of the mirror / glass albedos)   [emitted power per path]
//   pixel  += beta * (albedo / pi) * cos(theta_x) / (dist^2 * pixel_area * cos(theta_c)^3) / N
// cos(theta_x): between the surface normal and the direction to the camera; dist: from x to
// the camera; cos(theta_c)^3 / dist^2 turns area on the surface into area on the virtual
// screen; N: number of light paths. With smooth (interpolated) normals the light arriving at
// x is weighted by |cos_shading| / |cos_geometric|, which is what the path tracer's BRDF
// does (the "adjoint BSDF" correction).
//
// Pixels are summed as 32.32 fixed-point integers with atomic adds: integer addition does
// not depend on the order, so the image is the same for any number of threads.

#define LT_FIXED 4294967296.0     // 2^32
#define LT_BATCH 4096             // light paths per random-number stream

// (a roughness texture may make any texel of a material smooth: such a material counts)
static int scene_has_specular(const Scene *s){
    for (int i = 0; i < s->nmats; i++)
        if (material_has_delta(&s->mats[i]) || (s->mats[i].tex_mr && s->mats[i].type != MAT_LIGHT)) return 1;
    return 0;
}

void light_trace_caustics(const Scene *s, const Camera *cam, int w, int h, long long photons,
                          const TraceParams *tp, uint64_t seed, Vec3 *out){
    if (photons <= 0 || w <= 0 || h <= 0 || s->light < 0 || s->light >= s->nprims) return;
    if (!scene_has_specular(s)) return;                     // no caustics possible
    const Primitive *light = &s->prims[s->light];
    const Vec3 le = s->mats[light->material].emission;
    const real pix_area = camera_screen_area(cam) / ((real)w * h);
    const double scale = LT_FIXED / (double)photons;
    const int media = s->nvolumes > 0;
    uint64_t *acc = calloc((size_t)w * h * 3, sizeof(uint64_t));
    if (!acc) return;
    const long long nbatch = (photons + LT_BATCH - 1) / LT_BATCH;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (long long b = 0; b < nbatch; b++){
        Rng rng;
        rng_seed(&rng, seed ^ ((uint64_t)b * 0x9E3779B97F4A7C15ULL), (uint64_t)b * 2 + 1);
        long long end = (b + 1) * LT_BATCH < photons ? (b + 1) * LT_BATCH : photons;
        for (long long k = b * LT_BATCH; k < end; k++){
            // emit: a uniform point on the lamp, a cosine-distributed direction
            Vec3 xl = v_add(light->q, v_add(v_scale(light->u, rng_real(&rng)), v_scale(light->v, rng_real(&rng))));
            Ray ray = {xl, sample_cosine_hemisphere(light->normal, &rng)};
            Vec3 beta = v_scale(le, light->area * PI);       // Le cos / (pdf_area * pdf_dir)
            Vec3 thr = vec3(1, 1, 1);                         // for Russian roulette

            for (int depth = 0; depth < tp->max_depth; depth++){
                Hit hit;
                int found = scene_intersect(s, &ray, RT_EPS, 1e30, &hit);
                real ts;
                int which;
                // a scattering event in a medium: that path is the path tracer's job
                if (media && scene_medium_collision(s, &ray, 0, found ? hit.t : 1e30, &rng, &ts, &which)) break;
                if (!found) break;
                Material mtex;
                const Material *m = hit_shading(s, &hit, v_neg(ray.d), &mtex);
                if (m->type == MAT_LIGHT) break;

                Bsdf bsdf = bsdf_make(m, hit.front_face);
                Vec3 wi = v_neg(ray.d);                   // towards where the light came from
                real cg = r_fabs(v_dot(hit.ng, wi)), cs = r_fabs(v_dot(hit.n, wi));
                real corr = cg > 0 ? cs / cg : 0;         // smooth normals (adjoint BSDF)
                // 1. connect the BSDF's rough part to the camera (not at the first vertex:
                //    lamp -> rough surface is the path tracer's, by NEE)
                if (depth > 0 && bsdf_rough_part(&bsdf)){
                    Vec3 to_cam = v_sub(cam->origin, hit.p);
                    real dist2 = v_dot(to_cam, to_cam), dist = r_sqrt(dist2);
                    Vec3 wo = v_scale(to_cam, 1.0 / dist);
                    real cos_o = v_dot(hit.ng, wo);           // > 0: the camera on the light's side
                    // the BSDF seen from the camera (v = wo, l = wi): its side's normal and index
                    Bsdf bc = cos_o > 0 ? bsdf : bsdf_make(m, !hit.front_face);
                    Vec3 nc = cos_o > 0 ? hit.n : v_neg(hit.n);
                    real pdf;
                    Vec3 f = cos_o != 0 && v_dot(nc, wo) > 0 ? bsdf_eval(&bc, nc, wo, wi, &pdf) : vec3(0, 0, 0);
                    real sx, sy, cos_c;
                    if (v_max(f) > 0 && camera_project(cam, hit.p, &sx, &sy, &cos_c) &&
                        sx >= 0 && sy >= 0 && sx < 1 && sy < 1){
                        int px = (int)(sx * w), py = (int)(sy * h);
                        Ray sh = {hit.p, wo};
                        if (px < w && py < h && !scene_occluded(s, &sh, RT_EPS, dist * (1.0 - 1e-4))){
                            real tr = media ? scene_medium_transmittance(s, &sh, 0, dist, &rng) : 1;
                            real wgt = corr * r_fabs(cos_o) * tr / (dist2 * pix_area * cos_c * cos_c * cos_c);
                            Vec3 c = v_scale(v_mul(beta, f), wgt);
                            double cc[3] = {c.x * scale, c.y * scale, c.z * scale};
                            size_t idx = ((size_t)py * w + px) * 3;
                            for (int ch = 0; ch < 3; ch++){
                                if (!(cc[ch] > 0) || cc[ch] > 1.8e19) continue;   // NaN, 0 or absurd
                                uint64_t v = (uint64_t)(cc[ch] + 0.5);
#ifdef _OPENMP
#pragma omp atomic
#endif
                                acc[idx + ch] += v;
                            }
                        }
                    }
                }
                // 2. keep bouncing, in a direction drawn from the BSDF (seen from the light's
                //    side): the following rough vertices are connected to the camera as well.
                //    Rough lobes: weight * |wi.ns| / |wi.ng| * |wo.ng| / |wo.ns| (smooth
                //    normals); through the surface: * (eta_from / eta_to)^2, the adjoint of the
                //    radiance's (eta_to / eta_from)^2.
                BsdfSample bs;
                real ul = rng_real(&rng), u1 = rng_real(&rng), u2 = rng_real(&rng);
                if (!bsdf_sample(&bsdf, hit.n, wi, ul, u1, u2, &bs)) break;
                if (depth == 0 && !bs.delta) break;           // lamp -> rough: the path tracer's
                real og = v_dot(hit.ng, bs.l), os = v_dot(hit.n, bs.l);
                if (!bs.delta && (bs.through ? og >= 0 : og <= 0)) break;   // (only with smooth normals)
                Vec3 f = bs.w;
                if (!bs.delta) f = v_scale(f, os != 0 ? corr * r_fabs(og / os) : 0);
                if (bs.through) f = v_scale(f, bsdf.eta * bsdf.eta);
                beta = v_mul(beta, f);
                thr = v_mul(thr, f);
                if (depth >= RR_START){
                    real p = r_fmin(v_max(thr), 0.95);
                    if (rng_real(&rng) >= p) break;
                    beta = v_scale(beta, 1.0 / p);
                    thr = v_scale(thr, 1.0 / p);
                }
                Vec3 next = bs.l;
                ray.o = hit.p;
                ray.d = v_norm(next);
            }
        }
    }

    for (size_t i = 0; i < (size_t)w * h; i++){
        out[i].x += (real)(acc[3 * i] / LT_FIXED);
        out[i].y += (real)(acc[3 * i + 1] / LT_FIXED);
        out[i].z += (real)(acc[3 * i + 2] / LT_FIXED);
    }
    free(acc);
}
