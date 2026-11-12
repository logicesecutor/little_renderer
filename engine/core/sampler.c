#include "core/sampler.h"

#include <math.h>
#include <string.h>

uint32_t sampler_hash(uint32_t x){
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint32_t reverse_bits(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

// Owen scrambling: each bit is flipped depending on (a hash of) all the bits above it, which
// keeps the points stratified. Burley's hash runs on the reversed bits (it mixes upward).
uint32_t sampler_owen(uint32_t x, uint32_t seed){
    x = reverse_bits(x);
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return reverse_bits(x);
}

// the second Sobol dimension (the first is reverse_bits(i), van der Corput)
uint32_t sampler_sobol1(uint32_t i){
    uint32_t r = 0, v = 1u << 31;
    for (; i; i >>= 1, v ^= v >> 1) if (i & 1) r ^= v;
    return r;
}

// ---- the blue-noise mask: void-and-cluster ---------------------------------------------
// Points are added one at a time where the pattern has its largest gap ("void"), measured
// by a Gaussian energy that wraps around the tile; a point's rank is when it was added. The
// first tenth is made by relaxing a random start (moving the most crowded point to the
// largest gap until nothing moves), and ranked by removing the most crowded first.
#define BN BLUE_NOISE_SIZE
#define BN_N (BN * BN)
static uint16_t bn_rank[BN_N];
static volatile int bn_ready;

static void bn_update(float *energy, const float *kern, int p, float sign){
    int px = p % BN, py = p / BN;
    for (int y = 0; y < BN; y++){
        const float *k = kern + ((y - py) & (BN - 1)) * BN;
        float *e = energy + y * BN;
        for (int x = 0; x < BN; x++) e[x] += sign * k[(x - px) & (BN - 1)];
    }
}

static int bn_extreme(const float *energy, const unsigned char *on, int want_on, int want_max){
    int best = -1;
    for (int i = 0; i < BN_N; i++){
        if (on[i] != want_on) continue;
        if (best < 0 || (want_max ? energy[i] > energy[best] : energy[i] < energy[best])) best = i;
    }
    return best;
}

static void bn_make(void){
    static float kern[BN_N], energy[BN_N], e0[BN_N];
    static unsigned char on[BN_N], on0[BN_N];
    const double sigma = 1.9;
    for (int y = 0; y < BN; y++)
        for (int x = 0; x < BN; x++){
            int dx = x < BN / 2 ? x : x - BN, dy = y < BN / 2 ? y : y - BN;
            kern[y * BN + x] = (float)exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
    memset(on, 0, sizeof(on));
    memset(energy, 0, sizeof(energy));
    Rng rng;
    rng_seed(&rng, 0x5eedb10eULL, 7);
    const int ones = BN_N / 10;
    for (int n = 0; n < ones; ){
        int p = (int)(rng_u32(&rng) % BN_N);
        if (on[p]) continue;
        on[p] = 1;
        bn_update(energy, kern, p, 1);
        n++;
    }
    for (int it = 0; it < BN_N; it++){                   // relax
        int c = bn_extreme(energy, on, 1, 1);
        on[c] = 0;
        bn_update(energy, kern, c, -1);
        int v = bn_extreme(energy, on, 0, 0);
        on[v] = 1;
        bn_update(energy, kern, v, 1);
        if (v == c) break;
    }
    memcpy(on0, on, sizeof(on));
    memcpy(e0, energy, sizeof(energy));
    for (int r = ones - 1; r >= 0; r--){                  // the start, most crowded removed first
        int c = bn_extreme(energy, on, 1, 1);
        on[c] = 0;
        bn_update(energy, kern, c, -1);
        bn_rank[c] = (uint16_t)r;
    }
    memcpy(on, on0, sizeof(on));
    memcpy(energy, e0, sizeof(energy));
    for (int r = ones; r < BN_N; r++){                    // the rest, each in the largest gap
        int v = bn_extreme(energy, on, 0, 0);
        on[v] = 1;
        bn_update(energy, kern, v, 1);
        bn_rank[v] = (uint16_t)r;
    }
}

const uint16_t *blue_noise_mask(void){
    if (!bn_ready){
#ifdef _OPENMP
#pragma omp critical (lr_blue_noise)
#endif
        {
            if (!bn_ready){ bn_make(); bn_ready = 1; }
        }
    }
    return bn_rank;
}

static inline real bn_value(const uint16_t *m, int x, int y){
    return ((real)m[(y & (BN - 1)) * BN + (x & (BN - 1))] + (real)0.5) * (real)(1.0 / BN_N);
}

static inline real unit(uint32_t v){
#ifdef LR_DOUBLE
    return v * (1.0 / 4294967296.0);
#else
    return (real)(v >> 8) * (1.0f / 16777216.0f);
#endif
}

static inline real wrap01(real t){ return t >= 1 ? t - 1 : t; }

void sampler_2d(Sampler *s, int dim, real *a, real *b){
    if (!s->on || dim >= SAMPLER_PAIRS){
        *a = rng_real(s->rng);
        *b = rng_real(s->rng);
        return;
    }
    const uint16_t *m = blue_noise_mask();
    uint32_t tile = sampler_hash(((uint32_t)(s->x >> 6) * 73856093u) ^ ((uint32_t)(s->y >> 6) * 19349663u));
    uint32_t seed = sampler_hash(s->seed ^ sampler_hash(((uint32_t)dim * 0x9e3779b9u) ^ tile));
    uint32_t i = sampler_owen(s->index, seed);           // the order, shuffled (stratified prefixes)
    uint32_t xa = sampler_owen(reverse_bits(i), sampler_hash(seed ^ 0xa511e9b3u));
    uint32_t xb = sampler_owen(sampler_sobol1(i), sampler_hash(seed ^ 0x63d83595u));
    int ox = s->x + 23 * dim, oy = s->y + 41 * dim;
    *a = wrap01(unit(xa) + bn_value(m, ox, oy));
    *b = wrap01(unit(xb) + bn_value(m, ox + 32, oy + 17));
}

real sampler_1d(Sampler *s, int dim, int comp){
    if (!s->on || dim >= SAMPLER_PAIRS) return rng_real(s->rng);
    real a, b;
    sampler_2d(s, dim, &a, &b);
    return comp ? b : a;
}
