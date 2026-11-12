#include "rt/medium.h"

// The part [t0, t1] of the ray inside the box (slab test), clipped to [tmin, tmax].
static int clip_to_box(const Aabb *b, const Ray *r, real tmin, real tmax, real *t0, real *t1){
    real lo = tmin, hi = tmax;
    const real o[3] = {r->o.x, r->o.y, r->o.z}, d[3] = {r->d.x, r->d.y, r->d.z};
    const real bl[3] = {b->lo.x, b->lo.y, b->lo.z}, bh[3] = {b->hi.x, b->hi.y, b->hi.z};
    for (int a = 0; a < 3; a++){
        if (d[a] == 0){                             // parallel to this pair of planes
            if (o[a] < bl[a] || o[a] > bh[a]) return 0;
            continue;
        }
        real inv = 1.0f / d[a];
        real ta = (bl[a] - o[a]) * inv, tb = (bh[a] - o[a]) * inv;
        if (ta > tb){ real t = ta; ta = tb; tb = t; }
        if (ta > lo) lo = ta;
        if (tb < hi) hi = tb;
        if (lo >= hi) return 0;
    }
    *t0 = lo;
    *t1 = hi;
    return 1;
}

// ---- walking the majorant grid (3D DDA) --------------------------------------------------
//
// The ray crosses the bricks of the majorant grid one after the other (Amanatides & Woo,
// "A Fast Voxel Traversal Algorithm", 1987): along each axis the distance between two brick
// boundaries is constant, so the next boundary is always the nearest of three precomputed
// values. Each step returns one segment [ta, tb] and the majorant of the brick it crosses.

// If even the global majorant gives an optical depth below this over the whole segment,
// tracking takes only a few steps anyway: one segment with the global majorant is then
// cheaper than walking the bricks (measured: thin media up to ~20% faster).
#define MEDIUM_DDA_MIN_TAU 4.0f

typedef struct {
    const Volume *v;
    int global;              // 1 = a single segment with the global majorant
    real t, t1;              // where the walk is, and where it stops
    int i[3];                // current brick
    int step[3];             // +1 / -1 per axis
    real next[3];            // t of the next brick boundary on each axis
    real delta[3];           // t between two boundaries on each axis
    int n[3];                // bricks per axis
} BrickWalk;

static int walk_init(BrickWalk *w, const Volume *v, const Ray *r, real tmin, real tmax){
    real t0, t1;
    if (!clip_to_box(&v->active, r, tmin, tmax, &t0, &t1)) return 0;
    w->v = v;
    w->t = t0;
    w->t1 = t1;
    w->global = v->sigma_t * v->max_density * (t1 - t0) < MEDIUM_DDA_MIN_TAU;
#ifdef MEDIUM_NO_BRICKS
    w->global = 1;            // for comparison: make CFLAGS="-O2 -DMEDIUM_NO_BRICKS"
#endif
    if (w->global) return 1;
    const real o[3] = {r->o.x, r->o.y, r->o.z}, d[3] = {r->d.x, r->d.y, r->d.z};
    const real lo[3] = {v->bounds.lo.x, v->bounds.lo.y, v->bounds.lo.z};
    const real hi[3] = {v->bounds.hi.x, v->bounds.hi.y, v->bounds.hi.z};
    const int nvox[3] = {v->nx, v->ny, v->nz};
    w->n[0] = v->bnx; w->n[1] = v->bny; w->n[2] = v->bnz;
    // start in the middle of the first segment's first bit, to stay clear of boundaries
    for (int a = 0; a < 3; a++){
        real bs = (hi[a] - lo[a]) / nvox[a] * VOLUME_BRICK;      // brick size on this axis
        real p = o[a] + d[a] * t0;
        int i = (int)r_floor((p - lo[a]) / bs);
        if (i < 0) i = 0;
        if (i > w->n[a] - 1) i = w->n[a] - 1;
        w->i[a] = i;
        if (d[a] > 0){
            w->step[a] = 1;
            w->next[a] = (lo[a] + (i + 1) * bs - o[a]) / d[a];
            w->delta[a] = bs / d[a];
        } else if (d[a] < 0){
            w->step[a] = -1;
            w->next[a] = (lo[a] + i * bs - o[a]) / d[a];
            w->delta[a] = -bs / d[a];
        } else {
            w->step[a] = 0;
            w->next[a] = REAL_MAX;
            w->delta[a] = REAL_MAX;
        }
    }
    return 1;
}

// next segment [ta, tb] and the extinction majorant sigma_t * brick_max over it; 0 = done
static int walk_next(BrickWalk *w, real *ta, real *tb, real *smax){
    if (w->t >= w->t1) return 0;
    if (w->global){
        *ta = w->t; *tb = w->t1; *smax = w->v->sigma_t * w->v->max_density;
        w->t = w->t1;
        return 1;
    }
    for (int a = 0; a < 3; a++)
        if (w->i[a] < 0 || w->i[a] >= w->n[a]) return 0;   // walked out of the grid
    int ax = w->next[0] < w->next[1] ? (w->next[0] < w->next[2] ? 0 : 2) : (w->next[1] < w->next[2] ? 1 : 2);
    real end = w->next[ax] < w->t1 ? w->next[ax] : w->t1;
    if (end < w->t) end = w->t;                          // (a boundary just behind us: rounding)
    const Volume *v = w->v;
    *ta = w->t;
    *tb = end;
    *smax = v->sigma_t * v->brick_max[((size_t)w->i[2] * v->bny + w->i[1]) * v->bnx + w->i[0]];
    w->t = end;
    w->i[ax] += w->step[ax];
    w->next[ax] += w->delta[ax];
    return 1;
}

// Delta tracking (Woodcock tracking, 1965), brick by brick. Pretend each brick is uniformly
// as dense as its densest point, sigma_max = sigma_t * brick_max. In a uniform medium the
// distance to the next collision is exponential: t += -ln(1 - xi) / sigma_max. At that point
// the collision is real with probability sigma(x) / sigma_max; otherwise it is a "null"
// collision with imaginary matter that changes nothing, and the walk continues. At a brick
// border the exponential simply starts again with the next brick's majorant (it has no
// memory), and empty bricks are skipped at no cost. The real collisions come out exactly as
// in the true, non-uniform medium.
int medium_sample_distance(const Volume *v, const Ray *ray, real tmin, real tmax, Rng *rng, real *t_hit){
    if (v->max_density <= 0 || v->sigma_t <= 0) return 0;
    BrickWalk w;
    if (!walk_init(&w, v, ray, tmin, tmax)) return 0;
    real ta, tb, smax;
    while (walk_next(&w, &ta, &tb, &smax)){
        if (smax <= 0) continue;                           // empty brick
        real t = ta;
        for (;;){
            t -= r_log(1.0f - rng_real(rng)) / smax;
            if (t >= tb) break;                            // on to the next brick
            real s = v->sigma_t * volume_density(v, v_add(ray->o, v_scale(ray->d, t)));
            if (rng_real(rng) * smax < s){ *t_hit = t; return 1; }
        }
    }
    return 0;
}

// Ratio tracking (Novák et al. 2014). Same tentative collisions as delta tracking, but
// instead of deciding at random whether each is real, multiply the transmittance by the
// probability that it is NOT: T *= 1 - sigma(x) / sigma_max. This has much less noise than
// "1 if delta tracking gets through, else 0". When T gets small, Russian roulette ends it.
real medium_transmittance(const Volume *v, const Ray *ray, real tmin, real tmax, Rng *rng){
    if (v->max_density <= 0 || v->sigma_t <= 0) return 1;
    BrickWalk w;
    if (!walk_init(&w, v, ray, tmin, tmax)) return 1;
    real tr = 1, ta, tb, smax;
    while (walk_next(&w, &ta, &tb, &smax)){
        if (smax <= 0) continue;
        real t = ta;
        for (;;){
            t -= r_log(1.0f - rng_real(rng)) / smax;
            if (t >= tb) break;
            real s = v->sigma_t * volume_density(v, v_add(ray->o, v_scale(ray->d, t)));
            if (s > smax) s = smax;                        // (rounding exactly at a brick border)
            tr *= 1.0f - s / smax;
            if (tr < 0.1f){                                // unbiased early exit
                if (rng_real(rng) < 0.5f) return 0;
                tr *= 2;
            }
        }
    }
    return tr;
}

// ---- several media --------------------------------------------------------------------
//
// Where media overlap their extinctions add up. Each medium's collisions form an independent
// random process, and the first collision of the sum is simply the earliest of each
// medium's own first collision (the minimum of independent exponential "clocks"). So every
// medium is tracked on its own, each one only up to the best collision found so far, and the
// winner decides how the light scatters (its albedo and phase function). Transmittances
// multiply.

int scene_medium_collision(const Scene *s, const Ray *ray, real tmin, real tmax, Rng *rng, real *t, int *which){
    int w = -1;
    real best = tmax;
    for (int i = 0; i < s->nvolumes; i++){
        real ti;
        if (medium_sample_distance(s->volumes[i], ray, tmin, best, rng, &ti)){ best = ti; w = i; }
    }
    if (w < 0) return 0;
    *t = best;
    *which = w;
    return 1;
}

real scene_medium_transmittance(const Scene *s, const Ray *ray, real tmin, real tmax, Rng *rng){
    real tr = 1;
    for (int i = 0; i < s->nvolumes && tr > 0; i++) tr *= medium_transmittance(s->volumes[i], ray, tmin, tmax, rng);
    return tr;
}

// Henyey-Greenstein sampling: invert its cumulative distribution in cos(theta), then pick
// the azimuth uniformly around d.
Vec3 medium_sample_phase(Vec3 d, real g, Rng *rng){
    real xi = rng_real(rng);
    return medium_sample_phase_u(d, g, xi, rng_real(rng));
}

Vec3 medium_sample_phase_u(Vec3 d, real g, real xi, real u_phi){
    real cos_t;
    if (r_fabs(g) < 1e-3f){
        cos_t = 1.0f - 2.0f * xi;                    // isotropic
    } else {
        real sq = (1.0f - g * g) / (1.0f + g - 2.0f * g * xi);
        cos_t = (1.0f + g * g - sq * sq) / (2.0f * g);
    }
    cos_t = cos_t < -1 ? -1 : (cos_t > 1 ? 1 : cos_t);
    real sin_t = r_sqrt(1.0f - cos_t * cos_t);
    real phi = 2.0f * (real)PI * u_phi;
    Vec3 t, b;
    onb_from_normal(d, &t, &b);
    return v_add(v_add(v_scale(t, sin_t * r_cos(phi)), v_scale(b, sin_t * r_sin(phi))), v_scale(d, cos_t));
}
