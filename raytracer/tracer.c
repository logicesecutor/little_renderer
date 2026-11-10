#include "tracer.h"

#define EPS 1e-4         // minimum ray distance: stops a ray from re-hitting its own surface
#define RR_START 3       // bounces before Russian roulette may stop a path

// Next event estimation (direct light sampling) for a diffuse surface.
// Pick one random point on the light, check it is visible, and return the light it sends
// towards the hit point, already multiplied by the Lambert BRDF (albedo / pi) and cosines:
//
//   L_direct = (albedo / pi) * Le * cos(theta_surface) * cos(theta_light) * area / dist^2
//
// The area / dist^2 * cos(theta_light) factor converts "pick a point with probability
// 1 / area" into "probability per unit solid angle", as the rendering equation needs.
static Vec3 sample_direct_light(const Scene *s, const Hit *h, Vec3 albedo, Rng *rng){
    const Primitive *light = &s->prims[s->light];
    Vec3 le = s->mats[light->material].emission;

    Vec3 xl = v_add(light->q, v_add(v_scale(light->u, rng_double(rng)),
                                    v_scale(light->v, rng_double(rng))));
    Vec3 to_light = v_sub(xl, h->p);
    double dist2 = v_dot(to_light, to_light);
    double dist = sqrt(dist2);
    Vec3 wi = v_scale(to_light, 1.0 / dist);

    double cos_surf = v_dot(h->n, wi);
    double cos_light = -v_dot(light->normal, wi);   // the light only emits on its normal side
    if (cos_surf <= 0 || cos_light <= 0) return vec3(0, 0, 0);

    // shadow ray: anything between the point and the light blocks it
    Ray shadow = {h->p, wi};
    if (scene_occluded(s, &shadow, EPS, dist - 1e-3)) return vec3(0, 0, 0);

    double g = cos_surf * cos_light * light->area / dist2;
    return v_scale(v_mul(albedo, le), g / PI);
}

// Schlick's approximation of the Fresnel reflectance
static double schlick(double cos_i, double eta){
    double r0 = (1.0 - eta) / (1.0 + eta);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cos_i, 5.0);
}

Vec3 trace_path(const Scene *s, Ray ray, Rng *rng, int max_depth){
    Vec3 radiance = vec3(0, 0, 0);    // what we have collected so far
    Vec3 throughput = vec3(1, 1, 1);  // how much of the light found from here on reaches the camera
    int count_emission = 1;           // camera rays and specular bounces "see" lights directly

    for (int depth = 0; ; depth++){
        Hit h;
        if (!scene_intersect(s, &ray, EPS, 1e30, &h)) break;   // escaped: the background is black

        const Material *m = &s->mats[s->prims[h.prim].material];

        if (m->type == MAT_LIGHT){
            // After a diffuse bounce the light was already counted by sample_direct_light;
            // adding it again here would count it twice.
            if (count_emission && h.front_face) radiance = v_add(radiance, v_mul(throughput, m->emission));
            break;
        }
        if (depth >= max_depth) break;

        Vec3 d = v_norm(ray.d);
        Vec3 next;

        switch (m->type){
        case MAT_DIFFUSE:
            // 1. light arriving straight from the lamp
            radiance = v_add(radiance, v_mul(throughput, sample_direct_light(s, &h, m->albedo, rng)));
            // 2. continue the path in a cosine-distributed direction. With pdf = cos/pi and
            //    BRDF = albedo/pi, the weight  BRDF * cos / pdf  is simply the albedo.
            next = sample_cosine_hemisphere(h.n, rng);
            throughput = v_mul(throughput, m->albedo);
            count_emission = 0;
            break;

        case MAT_MIRROR:
            next = v_reflect(d, h.n);
            throughput = v_mul(throughput, m->albedo);
            count_emission = 1;
            break;

        case MAT_GLASS: {
            double eta = h.front_face ? 1.0 / m->ior : m->ior;   // entering or leaving the glass
            double cos_i = fmin(v_dot(v_neg(d), h.n), 1.0);
            double sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
            // total internal reflection, or reflect with probability = Fresnel reflectance
            if (sin_t2 > 1.0 || rng_double(rng) < schlick(cos_i, eta)) next = v_reflect(d, h.n);
            else next = v_refract(d, h.n, eta);
            throughput = v_mul(throughput, m->albedo);
            count_emission = 1;
            break;
        }

        default:
            next = d;   // not reached
        }

        // Russian roulette: after a few bounces, stop dim paths at random and boost the
        // survivors by 1/p. On average the result is unchanged (unbiased) but much cheaper.
        if (depth >= RR_START){
            double p = fmin(v_max(throughput), 0.95);
            if (rng_double(rng) >= p) break;
            throughput = v_scale(throughput, 1.0 / p);
        }

        ray.o = h.p;
        ray.d = next;
    }
    return radiance;
}
