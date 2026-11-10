// noise.h - procedural noise, the raw material of clouds (and of any natural-looking pattern).
//
//   Perlin (gradient) noise: smooth random hills and valleys, values in about [-1, 1].
//   Worley (cellular) noise: distance to the nearest of randomly scattered points; inverted
//                            it gives round "cauliflower" bumps, the billows of cumulus clouds.
//   fBm (fractal Brownian motion): a sum of octaves, each at twice the frequency and half the
//                            amplitude of the previous, so there is detail at every scale.
//
// All functions are deterministic for a given seed and safe to call from many threads.
#ifndef LR_NOISE_H
#define LR_NOISE_H

#include <stdint.h>
#include "core/lrmath.h"

typedef struct {
    uint8_t perm[512];   // Perlin's permutation table, stored twice to avoid a wrap
    uint32_t seed;       // for the Worley feature points
} Noise;

void noise_init(Noise *n, uint32_t seed);

real noise_perlin(const Noise *n, Vec3 p);                // about [-1, 1], 0 at integer points
real noise_worley(const Noise *n, Vec3 p);                // distance to the nearest feature point, [0, ~1]
real noise_fbm(const Noise *n, Vec3 p, int octaves);      // Perlin fBm, about [-1, 1]
real noise_billow(const Noise *n, Vec3 p, int octaves);   // inverted Worley fBm, [0, 1], 1 = bump top

#endif
