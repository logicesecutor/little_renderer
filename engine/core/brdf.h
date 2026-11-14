// brdf.h - how a surface scatters light: glTF 2.0's metallic-roughness model with
// transmission (docs/PBR.md), for every pipeline that shades with it. Inline, on `real`;
// gpu/ has the same functions in WGSL.
//
// A material (core/scene.h; its presets through material_resolved): base colour, metallic,
// roughness, index of refraction, transmission. Its BSDF is four lobes:
//   specular:     GGX microfacets (alpha = roughness^2), the height-correlated Smith
//                 masking-shadowing, Schlick's Fresnel: F0 from the ior for the dielectric
//                 part, the base colour for the metallic part
//   multiple scattering (metals): what single-scattering microfacets lose (light bouncing
//                 between them is masked, not reflected), given back as a diffuse-like lobe
//                 (Kulla & Conty 2017): (1 - E(v)) (1 - E(l)) / (pi (1 - E_avg)) F_ms, E the
//                 white lobe's albedo (core/brdf_tables.h); a white rough metal keeps all
//   diffuse:      Lambert with the base colour, dielectric and opaque only ((1 - metallic)
//                 (1 - transmission)), times what the specular layer lets through both ways:
//                 (1 - Es(v)) (1 - Es(l)) / (1 - Es_avg), Es the specular lobe's albedo with
//                 its Fresnel (F0 (E - B) + B from core/brdf_tables.h; smooth: F itself).
//                 glTF's appendix B has 1 - F(v.h) instead, which gives up to 30% more light
//                 than arrives at grazing angles; this keeps what arrives, and stays reciprocal
//   transmission: the rough dielectric BTDF (Walter et al. 2007) tinted by the base colour,
//                 (1 - metallic) transmission (1 - F)
// with real refraction (KHR_materials_volume's way, not thin walls).
// Roughness BSDF_SMOOTH or less is perfectly smooth: the specular and transmission lobes
// become a mirror reflection and a refraction (delta: no density, only sampled), as the old
// mirror and glass.
//
// Directions point away from the surface: v towards the viewer (the camera's side of the
// path), l towards the light. n is the shading normal on v's side. f(v, l) is the radiance
// BSDF: the transmission lobe carries (eta_l / eta_v)^2, so light entering glass gets
// denser per solid angle and leaving it thinner (it cancels for a closed object).
// bsdf_sample draws l: the lobe by its share of the Fresnel-weighted energy at v, the
// specular and transmission lobes' microfacet normal from the GGX visible normals (Dupuy &
// Benyoub 2023, the spherical-cap form of Heitz 2018), the diffuse one by the cosine. Its
// density is that of the lobes together (one-sample MIS between them), as bsdf_eval gives it.
#ifndef LR_BRDF_H
#define LR_BRDF_H

#include "core/lrmath.h"
#include "core/rng.h"
#include "core/scene.h"
#include "core/brdf_tables.h"

#define BSDF_SMOOTH 0.02   // roughness up to this: perfectly smooth

enum { BSDF_SPEC, BSDF_DIFF, BSDF_TRANS, BSDF_MS, BSDF_LOBES };

typedef struct {
    Vec3 base;
    real metallic, transmission;
    real rough, alpha;   // roughness, and GGX's alpha = roughness^2
    real eta;            // index of refraction on v's side / on the other side
    real e_avg;          // the white GGX lobe's mean albedo at this roughness (BRDF_E_AVG)
    real es_avg;         // the dielectric specular lobe's (with its Fresnel), for the diffuse
    int smooth;          // the specular and transmission lobes are perfect (delta)
    Vec3 ms;             // the multiple-scattering lobe's colour (0: none)
    real kd_norm;        // 1 / (1 - Es_avg): the diffuse layer's normalization
    real kms_norm;       // 1 / (pi (1 - E_avg))
    // seen from v at cosine cv (bsdf_view; kept while the calls see v from the same cosine):
    // the lobes' chances, the specular lobe's albedo (with its Fresnel), the white one's
    real cv, p[BSDF_LOBES], es_v, e_v;
} Bsdf;

typedef struct {
    Vec3 l;          // the direction drawn
    Vec3 w;          // f cos / pdf (a delta lobe: its reflectance / the chance it was picked)
    real pdf;        // density per solid angle (0 for a delta lobe)
    int delta;       // a perfect reflection or refraction
    int through;     // l is on the other side of the surface (refracted)
} BsdfSample;

static inline real bsdf_clamp01(real x){ return x < 0 ? 0 : x > 1 ? 1 : x; }
static inline real bsdf_lum(Vec3 c){ return (real)0.2126 * c.x + (real)0.7152 * c.y + (real)0.0722 * c.z; }

// a table of core/brdf_tables.h at this roughness and cosine mu (bilinear)
static inline real bsdf_table(const float t[BRDF_TABLE_N][BRDF_TABLE_N], real rough, real mu){
    real x = rough * (BRDF_TABLE_N - 1), y = bsdf_clamp01(mu) * (BRDF_TABLE_N - 1);
    int j = (int)x, i = (int)y;
    if (j > BRDF_TABLE_N - 2) j = BRDF_TABLE_N - 2;
    if (i > BRDF_TABLE_N - 2) i = BRDF_TABLE_N - 2;
    real fx = x - j, fy = y - i;
    real e0 = t[j][i] + fy * (t[j][i + 1] - t[j][i]);
    real e1 = t[j + 1][i] + fy * (t[j + 1][i + 1] - t[j + 1][i]);
    return e0 + fx * (e1 - e0);
}
// the white GGX lobe's albedo at cosine mu
static inline real bsdf_albedo(const Bsdf *b, real mu){ return bsdf_table(BRDF_E, b->rough, mu); }
// and the same with Schlick's term inside
static inline real bsdf_albedo_b(const Bsdf *b, real mu){ return bsdf_table(BRDF_B, b->rough, mu); }

// Fresnel reflectance of a dielectric (Schlick), light arriving at cos_i on the side of
// index eta_i, eta = eta_i / eta_t: the cosine on the thinner side, 1 past the critical angle
static inline real bsdf_fresnel_dielectric(real cos_i, real eta){
    cos_i = bsdf_clamp01(cos_i);
    real s2 = eta * eta * (1 - cos_i * cos_i);
    if (s2 >= 1) return 1;
    real c = eta > 1 ? r_sqrt(1 - s2) : cos_i;
    real f0 = (1 - eta) / (1 + eta);
    f0 *= f0;
    real x = 1 - c, x2 = x * x;
    return f0 + (1 - f0) * x2 * x2 * x;
}

// the mean of bsdf_fresnel_dielectric over the hemisphere (cosine-weighted): F0 + (1 - F0) / 21
// from the thinner side; from the denser one past the critical angle everything reflects,
// 1 - (1 - F_avg) / eta^2 (what crosses one way crosses back)
static inline real bsdf_fresnel_dielectric_avg(real eta){
    real e = eta > 1 ? 1 / eta : eta;
    real f0 = (1 - e) / (1 + e);
    f0 *= f0;
    real avg = f0 + (1 - f0) / 21;
    return eta > 1 ? 1 - (1 - avg) / (eta * eta) : avg;
}

// the share of light the dielectric specular lobe reflects at cosine mu (rough: its albedo
// with its Fresnel, F0 (E - B) + B; smooth: Fresnel itself), and the mean over the
// hemisphere: what the diffuse layer below gets is 1 minus it, both ways
static inline real bsdf_spec_albedo(const Bsdf *b, real mu){
    if (b->smooth || b->eta > 1) return bsdf_fresnel_dielectric(mu, b->eta);
    real f0 = (1 - b->eta) / (1 + b->eta), e = bsdf_albedo(b, mu), bb = bsdf_albedo_b(b, mu);
    f0 *= f0;
    return f0 * (e - bb) + bb;
}
static inline real bsdf_spec_albedo_avg(const Bsdf *b){
    return b->smooth || b->eta > 1 ? bsdf_fresnel_dielectric_avg(b->eta) : b->es_avg;
}

// a metal's (Schlick, F0 = its colour)
static inline Vec3 bsdf_fresnel_metal(Vec3 f0, real c){
    real x = 1 - bsdf_clamp01(c), x2 = x * x, x5 = x2 * x2 * x;
    return vec3(f0.x + (1 - f0.x) * x5, f0.y + (1 - f0.y) * x5, f0.z + (1 - f0.z) * x5);
}

// GGX: the distribution of microfacet normals (nh = n.h)
static inline real ggx_d(real nh, real a){
    real a2 = a * a, d = nh * nh * (a2 - 1) + 1;
    return a2 / ((real)PI * d * d);
}
// Smith masking of one direction at cosine c
static inline real ggx_g1(real c, real a){
    c = r_fabs(c);
    return 2 * c / (c + r_sqrt(a * a + (1 - a * a) * c * c));
}
// height-correlated masking-shadowing G2 / (4 |cv| |cl|)
static inline real ggx_vis(real cv, real cl, real a){
    cv = r_fabs(cv); cl = r_fabs(cl);
    real a2 = a * a;
    real d = cl * r_sqrt(cv * cv * (1 - a2) + a2) + cv * r_sqrt(cl * cl * (1 - a2) + a2);
    return d > 0 ? (real)0.5 / d : 0;
}

// a microfacet normal seen from v, with density G1(v) max(0, v.h) D(h) / (n.v)
static inline Vec3 ggx_sample_vndf(Vec3 n, Vec3 v, real a, real u1, real u2){
    Vec3 t, bt;
    onb_from_normal(n, &t, &bt);
    Vec3 vh = v_norm(vec3(a * v_dot(v, t), a * v_dot(v, bt), v_dot(v, n)));   // the view, stretched
    real phi = 2 * (real)PI * u1;
    real z = (1 - u2) * (1 + vh.z) - vh.z;                                     // a spherical cap
    real st = r_sqrt(r_fmax(0, 1 - z * z));
    Vec3 hs = vec3(st * r_cos(phi) + vh.x, st * r_sin(phi) + vh.y, z + vh.z);
    Vec3 hl = v_norm(vec3(a * hs.x, a * hs.y, r_fmax(hs.z, 0)));              // unstretched
    return v_add(v_add(v_scale(t, hl.x), v_scale(bt, hl.y)), v_scale(n, hl.z));
}

// the multiple-scattering lobe's colour (Kulla & Conty): F_avg^2 E_avg / (1 - F_avg (1 - E_avg))
// with the metal's mean Fresnel F_avg = F0 + (1 - F0) / 21
static inline Vec3 bsdf_ms_color(const Bsdf *b){
    real e = b->e_avg;
    Vec3 fa = vec3(b->base.x + (1 - b->base.x) / 21, b->base.y + (1 - b->base.y) / 21, b->base.z + (1 - b->base.z) / 21);
    return vec3(fa.x * fa.x * e / (1 - fa.x * (1 - e)), fa.y * fa.y * e / (1 - fa.y * (1 - e)), fa.z * fa.z * e / (1 - fa.z * (1 - e)));
}

// outside: v is on the side the geometric normal points to (the outside of a closed
// object). An opaque material sees air on both sides.
static inline Bsdf bsdf_make(const Material *mat, int outside){
    Material r0 = material_resolved(mat);
    const Material *m = &r0;
    Bsdf b;
    b.base = m->albedo;
    b.metallic = bsdf_clamp01(m->metallic);
    b.transmission = bsdf_clamp01(m->transmission);
    real r = bsdf_clamp01(m->roughness);
    b.smooth = r <= (real)BSDF_SMOOTH;
    b.rough = r;
    b.alpha = r_fmax(r * r, (real)1e-4);
    real ior = m->ior > 0 ? m->ior : (real)1.5;
    b.eta = b.transmission > 0 && !outside ? ior : 1 / ior;
    real x = r * (BRDF_TABLE_N - 1);
    int j = (int)x;
    if (j > BRDF_TABLE_N - 2) j = BRDF_TABLE_N - 2;
    b.e_avg = BRDF_E_AVG[j] + (x - j) * (BRDF_E_AVG[j + 1] - BRDF_E_AVG[j]);
    real b_avg = BRDF_B_AVG[j] + (x - j) * (BRDF_B_AVG[j + 1] - BRDF_B_AVG[j]);
    real f0 = (1 - b.eta) / (1 + b.eta);
    f0 *= f0;
    b.es_avg = f0 * (b.e_avg - b_avg) + b_avg;
    b.kd_norm = 1 / (1 - bsdf_spec_albedo_avg(&b));
    b.kms_norm = 1 / ((real)PI * (1 - b.e_avg));
    b.ms = b.smooth || b.metallic <= 0 ? vec3(0, 0, 0) : bsdf_ms_color(&b);
    b.cv = -1;
    return b;
}

// What the BSDF needs of v alone: the chance bsdf_sample picks each lobe (each by about the
// light it reflects; rough transmission keeps a share past the critical angle: tilted
// microfacets still let light through), and the lobes' albedos at v.
static inline void bsdf_view(Bsdf *b, real cv){
    if (b->cv == cv) return;
    b->cv = cv;
    real fd = bsdf_fresnel_dielectric(cv, b->eta);
    real fs = b->smooth ? fd : r_fmin(fd, (real)0.9);
    real m = b->metallic, t = b->transmission, lb = bsdf_lum(b->base);
    b->es_v = (1 - m) * (1 - t) > 0 ? bsdf_spec_albedo(b, cv) : 0;
    b->e_v = b->smooth || m <= 0 ? 1 : bsdf_albedo(b, cv);
    real w[BSDF_LOBES];
    w[BSDF_SPEC] = (1 - m) * fs + m * bsdf_lum(bsdf_fresnel_metal(b->base, cv));
    w[BSDF_DIFF] = (1 - m) * (1 - t) * (1 - b->es_v) * lb;
    w[BSDF_TRANS] = (1 - m) * t * (1 - fs) * lb;
    w[BSDF_MS] = m * bsdf_lum(b->ms) * (1 - b->e_v);
    real sum = w[0] + w[1] + w[2] + w[3];
    for (int k = 0; k < BSDF_LOBES; k++) b->p[k] = sum > 0 ? w[k] / sum : 0;
}

// The BSDF for light arriving from l and leaving towards v, without its perfect (delta)
// lobes; *pdf: the density with which bsdf_sample draws l from v.
static inline Vec3 bsdf_eval(Bsdf *b, Vec3 n, Vec3 v, Vec3 l, real *pdf){
    Vec3 f = vec3(0, 0, 0);
    *pdf = 0;
    real cv = v_dot(v, n), cl = v_dot(l, n);
    if (cv <= 0) return f;
    bsdf_view(b, cv);
    const real *p = b->p;
    const real m = b->metallic, t = b->transmission, a = b->alpha;
    if (cl > 0){
        Vec3 h = v_norm(v_add(v, l));
        real vh = bsdf_clamp01(v_dot(v, h)), nh = v_dot(n, h);
        real fd = bsdf_fresnel_dielectric(vh, b->eta);
        real kd = (1 - m) * (1 - t);
        if (kd > 0){
            kd *= (1 - b->es_v) * (1 - bsdf_spec_albedo(b, cl)) * b->kd_norm;
            f = v_scale(b->base, kd / (real)PI);
            *pdf += p[BSDF_DIFF] * cl / (real)PI;
        }
        if (!b->smooth && m > 0){
            real k = m * (1 - b->e_v) * (1 - bsdf_albedo(b, cl)) * b->kms_norm;
            f = v_add(f, v_scale(b->ms, k));
            *pdf += p[BSDF_MS] * cl / (real)PI;
        }
        if (!b->smooth && nh > 0){
            real d = ggx_d(nh, a);
            Vec3 fm = bsdf_fresnel_metal(b->base, vh);
            real s = d * ggx_vis(cv, cl, a);
            f = v_add(f, vec3(((1 - m) * fd + m * fm.x) * s, ((1 - m) * fd + m * fm.y) * s, ((1 - m) * fd + m * fm.z) * s));
            *pdf += p[BSDF_SPEC] * ggx_g1(cv, a) * d / (4 * cv);
        }
    } else if (cl < 0 && !b->smooth && t > 0 && m < 1){
        const real eta = b->eta;
        Vec3 h = v_norm(v_neg(v_add(v_scale(v, eta), l)));     // the refraction's half vector
        if (v_dot(h, n) < 0) h = v_neg(h);
        real vh = v_dot(v, h), lh = v_dot(l, h), nh = v_dot(n, h);
        real den = eta * vh + lh;
        den *= den;
        if (vh > 0 && lh < 0 && nh > 0 && den > (real)1e-12){
            real fd = bsdf_fresnel_dielectric(vh, eta);
            real d = ggx_d(nh, a);
            real g2 = ggx_vis(cv, cl, a) * 4 * cv * -cl;
            real ft = vh * -lh * (1 - fd) * d * g2 / (cv * -cl * den);
            f = v_scale(b->base, (1 - m) * t * ft);
            *pdf = p[BSDF_TRANS] * ggx_g1(cv, a) * vh * d / cv * -lh / den;
        }
    }
    return f;
}

// The lobe bsdf_sample draws from for the number ul (after bsdf_view), or -1. A lobe
// whose direction is fixed (perfect mirror or refraction) needs no numbers for it
// (bsdf_lobe_delta): a caller may save drawing them.
static inline int bsdf_pick(const Bsdf *b, real ul){
    const real *p = b->p;
    int lobe = 0;
    for (real acc = p[0]; lobe < BSDF_LOBES - 1 && (ul >= acc || p[lobe] <= 0); acc += p[++lobe]);
    return p[lobe] > 0 ? lobe : -1;
}
static inline int bsdf_lobe_delta(const Bsdf *b, int lobe){ return b->smooth && (lobe == BSDF_SPEC || lobe == BSDF_TRANS); }
// how many lobes bsdf_sample may pick (after bsdf_view): with one, ul does not matter
static inline int bsdf_nlobes(const Bsdf *b){ return (b->p[0] > 0) + (b->p[1] > 0) + (b->p[2] > 0) + (b->p[3] > 0); }

// Draws the direction l light comes from, for v, from lobe `lobe` (bsdf_pick): u1, u2 the
// direction. 0 if none (absorbed, or a microfacet sample that leaves the hemisphere).
static inline int bsdf_sample_lobe(Bsdf *b, int lobe, Vec3 n, Vec3 v, real u1, real u2, BsdfSample *s){
    real cv = v_dot(v, n);
    if (cv <= 0 || lobe < 0) return 0;
    bsdf_view(b, cv);
    const real *p = b->p;
    if (p[lobe] <= 0) return 0;
    const real m = b->metallic, t = b->transmission;
    s->delta = 0;
    s->through = 0;
    if (b->smooth && (lobe == BSDF_SPEC || lobe == BSDF_TRANS)){
        real fd = bsdf_fresnel_dielectric(cv, b->eta);
        if (lobe == BSDF_SPEC){
            Vec3 fm = bsdf_fresnel_metal(b->base, cv);
            real k = 1 / p[lobe];
            s->l = v_reflect(v_neg(v), n);
            s->w = vec3(((1 - m) * fd + m * fm.x) * k, ((1 - m) * fd + m * fm.y) * k, ((1 - m) * fd + m * fm.z) * k);
        } else {
            if (fd >= 1) return 0;
            s->l = v_refract(v_neg(v), n, b->eta);
            s->w = v_scale(b->base, (1 - m) * t * (1 - fd) / p[lobe]);
            s->through = 1;
        }
        s->pdf = 0;
        s->delta = 1;
        return 1;
    }
    Vec3 l;
    if (lobe == BSDF_DIFF || lobe == BSDF_MS) l = sample_cosine_hemisphere_u(n, u1, u2);
    else {
        Vec3 h = ggx_sample_vndf(n, v, b->alpha, u1, u2);
        real vh = v_dot(v, h);
        if (vh <= 0) return 0;
        if (lobe == BSDF_SPEC) l = v_reflect(v_neg(v), h);
        else {
            if (b->eta * b->eta * (1 - vh * vh) >= 1) return 0;       // no refraction at this microfacet
            l = v_refract(v_neg(v), h, b->eta);
        }
    }
    // a reflection below the surface, or a refraction above it: lost (masked)
    if ((lobe == BSDF_TRANS) != (v_dot(l, n) < 0)) return 0;
    real pdf;
    Vec3 f = bsdf_eval(b, n, v, l, &pdf);
    if (!(pdf > 0)) return 0;
    s->l = l;
    s->w = v_scale(f, r_fabs(v_dot(l, n)) / pdf);
    s->pdf = pdf;
    s->through = v_dot(l, n) < 0;
    return 1;
}

// The same, ul picking the lobe.
static inline int bsdf_sample(Bsdf *b, Vec3 n, Vec3 v, real ul, real u1, real u2, BsdfSample *s){
    real cv = v_dot(v, n);
    if (cv <= 0) return 0;
    bsdf_view(b, cv);
    return bsdf_sample_lobe(b, bsdf_pick(b, ul), n, v, u1, u2, s);
}

// the material has perfect (delta) lobes: mirror reflection or refraction
static inline int material_has_delta(const Material *m){
    Material r = material_resolved(m);
    return m->type != MAT_LIGHT && bsdf_clamp01(r.roughness) <= (real)BSDF_SMOOTH;
}

#endif
