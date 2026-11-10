#include "rt/tracer.h"

#include <stdlib.h>

#include "core/volume.h"
#include "rt/accel.h"
#include "rt/medium.h"

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

// Next event estimation (direct light sampling) for a diffuse surface.
// Pick one random point on the light, check it is visible, and return the light it sends
// towards the hit point, already multiplied by the Lambert BRDF (albedo / pi) and cosines:
//
//   L_direct = (albedo / pi) * Le * cos(theta_surface) * cos(theta_light) * area / dist^2
//
// The area / dist^2 * cos(theta_light) factor converts "pick a point with probability
// 1 / area" into "probability per unit solid angle", as the rendering equation needs.
// With MIS the result is weighted against the chance that the cosine-distributed bounce
// (pdf cos(theta_surface) / pi) would have found the same point.
static Vec3 sample_direct_light(const Scene *s, const Hit *h, Vec3 albedo, int mis, Rng *rng){
    const Primitive *light = &s->prims[s->light];
    Vec3 le = s->mats[light->material].emission;

    Vec3 xl = v_add(light->q, v_add(v_scale(light->u, rng_real(rng)),
                                    v_scale(light->v, rng_real(rng))));
    Vec3 to_light = v_sub(xl, h->p);
    real dist2 = v_dot(to_light, to_light);
    real dist = r_sqrt(dist2);
    Vec3 wi = v_scale(to_light, 1.0 / dist);

    real cos_surf = v_dot(h->n, wi);
    real cos_light = -v_dot(light->normal, wi);   // the light only emits on its normal side
    if (cos_surf <= 0 || cos_light <= 0) return vec3(0, 0, 0);

    // shadow ray: anything between the point and the light blocks it
    Ray shadow = {h->p, wi};
    if (scene_occluded(s, &shadow, RT_EPS, dist - RT_SHADOW_MARGIN)) return vec3(0, 0, 0);

    real g = cos_surf * cos_light * light->area / dist2;
    if (s->nvolumes) g *= scene_medium_transmittance(s, &shadow, 0, dist, rng);   // a cloud in the way
    if (mis) g *= power_heuristic(light_pdf(light, dist2, cos_light), cos_surf / PI);
    return v_scale(v_mul(albedo, le), g / PI);
}

// The same for a point p inside a medium, reached by a ray travelling along d. There is no
// surface, so no cos(theta_surface): the phase function takes the place of the BRDF, and
// the light is dimmed by the medium between p and the lamp (the cloud's self-shadowing):
//
//   L_direct = phase(d, wi) * Le * T(p <-> light) * cos(theta_light) * area / dist^2
static Vec3 sample_light_in_medium(const Scene *s, Vec3 p, Vec3 d, real g, int mis, Rng *rng){
    const Primitive *light = &s->prims[s->light];
    Vec3 le = s->mats[light->material].emission;

    Vec3 xl = v_add(light->q, v_add(v_scale(light->u, rng_real(rng)),
                                    v_scale(light->v, rng_real(rng))));
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

// Schlick's approximation of the Fresnel reflectance
static real schlick(real cos_i, real eta){
    real r0 = (1.0 - eta) / (1.0 + eta);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * r_pow(1.0 - cos_i, 5.0);
}

// Russian roulette: after a few bounces, stop dim paths at random and boost the survivors
// by 1/p. On average the result is unchanged (unbiased) but much cheaper. Returns 0 to stop.
static int russian_roulette(Vec3 *throughput, int depth, Rng *rng){
    if (depth < RR_START) return 1;
    real p = r_fmin(v_max(*throughput), 0.95);
    if (rng_real(rng) >= p) return 0;
    *throughput = v_scale(*throughput, 1.0 / p);
    return 1;
}

TraceParams trace_default_params(void){
    TraceParams p = {50, 1, 0};
    return p;
}

Vec3 trace_path(const Scene *s, Ray ray, Rng *rng, int max_depth){
    TraceParams p = trace_default_params();
    p.max_depth = max_depth;
    return trace_path_ex(s, ray, rng, &p);
}

// What a lamp hit means after the previous vertex:
enum {
    EMIT_FULL,   // camera rays, mirror and glass bounces: nothing else could find the lamp
    EMIT_NONE,   // after a diffuse / medium bounce without MIS: already counted by NEE
    EMIT_MIS     // after a diffuse / medium bounce with MIS: weight against NEE
};

Vec3 trace_path_ex(const Scene *s, Ray ray, Rng *rng, const TraceParams *tp){
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
            // 1. light arriving straight from the lamp, through the cloud
            radiance = v_add(radiance, v_mul(throughput, sample_light_in_medium(s, p, ray.d, vol->g, mis, rng)));
            // 2. continue in a direction drawn from the phase function. Its pdf equals the
            //    phase function, so the weight phase / pdf is 1: throughput is unchanged.
            Vec3 nd = medium_sample_phase(ray.d, vol->g, rng);
            bounce_pdf = phase_hg(v_dot(ray.d, nd), vol->g);
            ray.o = p;
            ray.d = nd;
            emit = mis ? EMIT_MIS : EMIT_NONE;
            caustic_chain = 0;
            if (!russian_roulette(&throughput, depth, rng)) break;
            continue;
        }
        if (!found) break;                    // escaped: the background is black

        const Material *m = &s->mats[scene_shape_material(s, h.prim)];

        if (m->type == MAT_LIGHT){
            if (h.front_face && !(caustic_chain && caustic_spec)){
                if (emit == EMIT_FULL){
                    radiance = v_add(radiance, v_mul(throughput, m->emission));
                } else if (emit == EMIT_MIS && h.prim == s->light){
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
        if (depth >= max_depth) break;

        Vec3 d = v_norm(ray.d);
        Vec3 next;

        switch (m->type){
        case MAT_DIFFUSE:
            // 1. light arriving straight from the lamp
            radiance = v_add(radiance, v_mul(throughput, sample_direct_light(s, &h, m->albedo, mis, rng)));
            // 2. continue the path in a cosine-distributed direction. With pdf = cos/pi and
            //    BRDF = albedo/pi, the weight  BRDF * cos / pdf  is simply the albedo.
            next = sample_cosine_hemisphere(h.n, rng);
            bounce_pdf = v_dot(h.n, next) / PI;
            throughput = v_mul(throughput, m->albedo);
            emit = mis ? EMIT_MIS : EMIT_NONE;
            if (depth == 0) caustic_chain = tp->skip_caustics;
            caustic_spec = 0;
            break;

        case MAT_MIRROR:
            next = v_reflect(d, h.n);
            throughput = v_mul(throughput, m->albedo);
            emit = EMIT_FULL;
            caustic_spec = 1;
            break;

        case MAT_GLASS: {
            real eta = h.front_face ? 1.0 / m->ior : m->ior;     // entering or leaving the glass
            real cos_i = r_fmin(v_dot(v_neg(d), h.n), 1.0);
            real sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
            // total internal reflection, or reflect with probability = Fresnel reflectance
            if (sin_t2 > 1.0 || rng_real(rng) < schlick(cos_i, eta)) next = v_reflect(d, h.n);
            else next = v_refract(d, h.n, eta);
            throughput = v_mul(throughput, m->albedo);
            emit = EMIT_FULL;
            caustic_spec = 1;
            break;
        }

        default:
            next = d;   // not reached
        }

        if (!russian_roulette(&throughput, depth, rng)) break;

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

static int scene_has_specular(const Scene *s){
    for (int i = 0; i < s->nmats; i++)
        if (s->mats[i].type == MAT_MIRROR || s->mats[i].type == MAT_GLASS) return 1;
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
            int nspec = 0, ndiff = 0;

            for (int depth = 0; depth < tp->max_depth; depth++){
                Hit hit;
                int found = scene_intersect(s, &ray, RT_EPS, 1e30, &hit);
                real ts;
                int which;
                // a scattering event in a medium: that path is the path tracer's job
                if (media && scene_medium_collision(s, &ray, 0, found ? hit.t : 1e30, &rng, &ts, &which)) break;
                if (!found) break;
                const Material *m = &s->mats[scene_shape_material(s, hit.prim)];
                if (m->type == MAT_LIGHT) break;

                if (m->type == MAT_DIFFUSE){
                    if (nspec == 0 && ndiff == 0) break;      // lamp -> D: the path tracer's job (NEE)
                    Vec3 wi = v_neg(ray.d);                   // towards where the light came from
                    real cg = r_fabs(v_dot(hit.ng, wi)), cs = r_fabs(v_dot(hit.n, wi));
                    real corr = cg > 0 ? cs / cg : 0;         // smooth normals (adjoint BSDF)
                    // 1. connect to the camera
                    Vec3 to_cam = v_sub(cam->origin, hit.p);
                    real dist2 = v_dot(to_cam, to_cam), dist = r_sqrt(dist2);
                    Vec3 wo = v_scale(to_cam, 1.0 / dist);
                    real cos_o = v_dot(hit.ng, wo);           // the camera must see the lit side
                    real sx, sy, cos_c;
                    if (cos_o > 0 && camera_project(cam, hit.p, &sx, &sy, &cos_c) &&
                        sx >= 0 && sy >= 0 && sx < 1 && sy < 1){
                        int px = (int)(sx * w), py = (int)(sy * h);
                        Ray sh = {hit.p, wo};
                        if (px < w && py < h && !scene_occluded(s, &sh, RT_EPS, dist * (1.0 - 1e-4))){
                            real tr = media ? scene_medium_transmittance(s, &sh, 0, dist, &rng) : 1;
                            real wgt = corr * cos_o * tr / ((real)PI * dist2 * pix_area * cos_c * cos_c * cos_c);
                            Vec3 c = v_scale(v_mul(beta, m->albedo), wgt);
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
                    // 2. keep bouncing (cosine-distributed around the shading normal): the
                    //    following diffuse vertices are connected to the camera as well.
                    //    Weight = albedo * |wi.ns| / |wi.ng| * |wo.ng| / |wo.ns|.
                    Vec3 nd = sample_cosine_hemisphere(hit.n, &rng);
                    real og = v_dot(hit.ng, nd), os = v_dot(hit.n, nd);
                    if (og <= 0 || os <= 0) break;            // (only with smooth normals)
                    Vec3 f = v_scale(m->albedo, corr * og / os);
                    beta = v_mul(beta, f);
                    thr = v_mul(thr, f);
                    if (depth >= RR_START){
                        real p = r_fmin(v_max(thr), 0.95);
                        if (rng_real(&rng) >= p) break;
                        beta = v_scale(beta, 1.0 / p);
                        thr = v_scale(thr, 1.0 / p);
                    }
                    ray.o = hit.p;
                    ray.d = nd;
                    ndiff++;
                    continue;
                }
                // mirror or glass: exactly as in the path tracer
                Vec3 d = ray.d, next;
                if (m->type == MAT_MIRROR){
                    next = v_reflect(d, hit.n);
                } else {
                    real eta = hit.front_face ? 1.0 / m->ior : m->ior;
                    real cos_i = r_fmin(v_dot(v_neg(d), hit.n), 1.0);
                    real sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
                    if (sin_t2 > 1.0 || rng_real(&rng) < schlick(cos_i, eta)) next = v_reflect(d, hit.n);
                    else next = v_refract(d, hit.n, eta);
                }
                beta = v_mul(beta, m->albedo);
                thr = v_mul(thr, m->albedo);
                nspec++;
                if (depth >= RR_START){
                    real p = r_fmin(v_max(thr), 0.95);
                    if (rng_real(&rng) >= p) break;
                    beta = v_scale(beta, 1.0 / p);
                    thr = v_scale(thr, 1.0 / p);
                }
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
