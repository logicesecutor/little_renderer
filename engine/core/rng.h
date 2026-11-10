// rng.h - random numbers (PCG32) and the sampling helpers built on them. Header only.
// Give every pixel (or thread) its own generator: results then do not depend on how many
// threads run.
#ifndef LR_RNG_H
#define LR_RNG_H

#include <stdint.h>
#include "core/lrmath.h"

typedef struct { uint64_t state, inc; } Rng;

// PCG32 (pcg-random.org): a 64-bit LCG whose output is scrambled by a xorshift + rotation
static inline uint32_t rng_u32(Rng *r){
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static inline void rng_seed(Rng *r, uint64_t seed, uint64_t stream){
    r->state = 0;
    r->inc = (stream << 1u) | 1u;   // must be odd
    rng_u32(r);
    r->state += seed;
    rng_u32(r);
}

// uniform in [0, 1)
static inline real rng_real(Rng *r){
#ifdef LR_DOUBLE
    return rng_u32(r) * (1.0 / 4294967296.0);
#else
    // a float has 24 bits of mantissa: use 24 random bits so the result can never round up to 1
    return (real)(rng_u32(r) >> 8) * (1.0f / 16777216.0f);
#endif
}

// Builds two tangents t, b so that (t, b, n) is an orthonormal basis.
// Branchless method from Duff et al. 2017, "Building an Orthonormal Basis, Revisited".
static inline void onb_from_normal(Vec3 n, Vec3 *t, Vec3 *b){
    real s = r_copysign(1.0, n.z);
    real a = -1.0 / (s + n.z);
    real c = n.x * n.y * a;
    *t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    *b = vec3(c, s + n.y * n.y * a, -n.y);
}

// Direction on the hemisphere around n with probability density cos(theta) / pi.
// (Malley's method: pick a uniform point on the unit disk, lift it onto the hemisphere.)
static inline Vec3 sample_cosine_hemisphere(Vec3 n, Rng *r){
    real u1 = rng_real(r), u2 = rng_real(r);
    real phi = 2.0 * PI * u1;
    real rad = r_sqrt(u2);
    real x = rad * r_cos(phi), y = rad * r_sin(phi), z = r_sqrt(r_fmax(0.0, 1.0 - u2));
    Vec3 t, b;
    onb_from_normal(n, &t, &b);
    return v_add(v_add(v_scale(t, x), v_scale(b, y)), v_scale(n, z));
}

#endif
