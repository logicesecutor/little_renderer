// sampler.h - the random numbers of a path, from a low-discrepancy sequence.
//
// A path needs a few numbers per bounce (where on the lamp, which direction, whether to go
// on). Taken from a plain random generator they clump and leave gaps, and the noise falls
// like 1 / sqrt(samples). Taken from a well spread sequence the noise falls faster, and at a
// handful of samples per pixel looks finer:
//
//   - the 2D Sobol sequence ((0,2)-sequence in base 2: every 2^k consecutive points, k even,
//     put one point in each of 2^k equal squares), Owen-scrambled with Burley's hash
//     (JCGT 2020, "Practical Hash-based Owen Scrambling"): random but still stratified;
//     each pair of dimensions (camera jitter; per bounce: the lamp point, the direction, the
//     roulette and glass choices) has its own scramble and its own shuffle of the order;
//   - pixels share the sequence within 64 x 64 tiles, each shifted (Cranley-Patterson) by a
//     blue-noise value: neighbouring pixels then get very different samples, and the error
//     left in the image is fine-grained ("blue") rather than clumpy, which a denoiser
//     removes far better (Georgiev and Fajardo 2016). The blue-noise mask is made here
//     (void-and-cluster, Ulichney 1993), on first use.
//
// Dimensions past SAMPLER_PAIRS (deep bounces), and what a medium consumes while tracking
// (a variable number of numbers), come from the Rng. With on = 0 everything comes from the
// Rng, in the same order as before the sampler existed (the old images, bit for bit).
// The GPU path tracer (gpu/trace_gpu.c) has the same functions in WGSL.
#ifndef LR_SAMPLER_H
#define LR_SAMPLER_H

#include <stdint.h>
#include "core/rng.h"

#define SAMPLER_PAIRS 13                    // camera + 4 bounces x 3 pairs
#define SAMPLE_CAMERA 0
#define SAMPLE_LIGHT(depth) (1 + 3 * (depth))
#define SAMPLE_DIR(depth) (2 + 3 * (depth))
#define SAMPLE_MISC(depth) (3 + 3 * (depth))   // x: Russian roulette, y: reflect or refract
#define BLUE_NOISE_SIZE 64

typedef struct {
    Rng *rng;          // what the sequence does not cover
    uint32_t index;    // which sample of the pixel this is
    uint32_t seed;     // the image's scramble
    int x, y;          // the pixel
    int on;            // 0 = everything from rng
} Sampler;

static inline Sampler sampler_make(Rng *rng, int x, int y, uint32_t index, uint32_t seed, int on){
    Sampler s = {rng, index, seed, x, y, on};
    return s;
}

// two numbers in [0, 1) of dimension pair `dim`, or one of them (comp 0 or 1)
void sampler_2d(Sampler *s, int dim, real *a, real *b);
real sampler_1d(Sampler *s, int dim, int comp);

// the blue-noise mask: BLUE_NOISE_SIZE^2 ranks 0..4095, each once (made on first call)
const uint16_t *blue_noise_mask(void);

// the pieces, shared with the GPU's WGSL (same results)
uint32_t sampler_hash(uint32_t x);
uint32_t sampler_owen(uint32_t x, uint32_t seed);
uint32_t sampler_sobol1(uint32_t i);

#endif
