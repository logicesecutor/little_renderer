#include "core/noise.h"

#include "core/rng.h"

void noise_init(Noise *n, uint32_t seed){
    // a random permutation of 0..255 (Fisher-Yates shuffle with our PCG32)
    Rng r;
    rng_seed(&r, 0x9E3779B97F4A7C15ULL ^ seed, 7);
    for (int i = 0; i < 256; i++) n->perm[i] = (uint8_t)i;
    for (int i = 255; i > 0; i--){
        int j = (int)(rng_u32(&r) % (uint32_t)(i + 1));
        uint8_t t = n->perm[i]; n->perm[i] = n->perm[j]; n->perm[j] = t;
    }
    for (int i = 0; i < 256; i++) n->perm[256 + i] = n->perm[i];
    n->seed = seed;
}

// ---- Perlin noise (Ken Perlin, "Improving Noise", SIGGRAPH 2002) -------------------------
//
// Every integer lattice point gets a pseudo-random gradient (one of 12 cube edge directions,
// picked by hashing the point through the permutation table). Inside a cell, each of the 8
// corners contributes dot(gradient, offset to the point); the 8 values are blended with the
// smooth fade curve 6t^5 - 15t^4 + 10t^3, whose first and second derivatives are 0 at the
// corners, so the result has no visible grid.

static inline real fade(real t){ return t * t * t * (t * (t * 6 - 15) + 10); }
static inline real lerp(real a, real b, real t){ return a + t * (b - a); }

static inline real grad(int hash, real x, real y, real z){
    int h = hash & 15;                       // 12 edge directions (+4 repeated)
    real u = h < 8 ? x : y;
    real v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

real noise_perlin(const Noise *n, Vec3 p){
    real fx = r_floor(p.x), fy = r_floor(p.y), fz = r_floor(p.z);
    int X = (int)((long long)fx & 255), Y = (int)((long long)fy & 255), Z = (int)((long long)fz & 255);
    real x = p.x - fx, y = p.y - fy, z = p.z - fz;
    real u = fade(x), v = fade(y), w = fade(z);
    const uint8_t *P = n->perm;
    int A = P[X] + Y, AA = P[A] + Z, AB = P[A + 1] + Z;
    int B = P[X + 1] + Y, BA = P[B] + Z, BB = P[B + 1] + Z;
    return lerp(lerp(lerp(grad(P[AA], x, y, z),         grad(P[BA], x - 1, y, z), u),
                     lerp(grad(P[AB], x, y - 1, z),     grad(P[BB], x - 1, y - 1, z), u), v),
                lerp(lerp(grad(P[AA + 1], x, y, z - 1), grad(P[BA + 1], x - 1, y, z - 1), u),
                     lerp(grad(P[AB + 1], x, y - 1, z - 1), grad(P[BB + 1], x - 1, y - 1, z - 1), u), v), w);
}

real noise_fbm(const Noise *n, Vec3 p, int octaves){
    real sum = 0, amp = 0.5f, norm = 0;
    for (int o = 0; o < octaves; o++){
        sum += amp * noise_perlin(n, p);
        norm += amp;
        amp *= 0.5f;
        // x2 frequency, plus an odd offset so the octaves' lattices do not line up
        p = vec3(p.x * 2.0f + 17.13f, p.y * 2.0f + 5.71f, p.z * 2.0f + 11.37f);
    }
    return sum / norm;
}

// ---- Worley noise (Steven Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996) ----
//
// Space is cut into unit cells with one random feature point each. The value at p is the
// distance to the nearest feature point; that point is in p's cell or one of its 26
// neighbours.

static inline uint32_t hash3(int x, int y, int z, uint32_t seed){
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)y * 0xd8163841u ^ (uint32_t)z * 0xcb1ab31fu ^ seed * 0x9e3779b9u;
    h ^= h >> 16; h *= 0x7feb352du;          // integer hash finalizer (good bit mixing)
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

real noise_worley(const Noise *n, Vec3 p){
    real fx = r_floor(p.x), fy = r_floor(p.y), fz = r_floor(p.z);
    int cx = (int)fx, cy = (int)fy, cz = (int)fz;
    real best = 10;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++){
        uint32_t h = hash3(cx + dx, cy + dy, cz + dz, n->seed);
        // three 10-bit coordinates of the feature point inside its cell
        real px = (real)(cx + dx) + (h & 1023) * (1.0f / 1024.0f);
        real py = (real)(cy + dy) + ((h >> 10) & 1023) * (1.0f / 1024.0f);
        real pz = (real)(cz + dz) + ((h >> 20) & 1023) * (1.0f / 1024.0f);
        real ex = px - p.x, ey = py - p.y, ez = pz - p.z;
        real d2 = ex * ex + ey * ey + ez * ez;
        if (d2 < best) best = d2;
    }
    return r_sqrt(best);
}

real noise_billow(const Noise *n, Vec3 p, int octaves){
    real sum = 0, amp = 0.5f, norm = 0;
    for (int o = 0; o < octaves; o++){
        real w = 1.0f - noise_worley(n, p);   // 1 at a feature point, falling off around it
        if (w < 0) w = 0;
        sum += amp * w;
        norm += amp;
        amp *= 0.5f;
        p = vec3(p.x * 2.0f + 3.31f, p.y * 2.0f + 7.77f, p.z * 2.0f + 1.13f);
    }
    return sum / norm;
}
