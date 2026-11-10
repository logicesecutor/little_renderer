#include "phys/poisson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIG_GRID 32768   // fewer cells than this: one thread (thread start-up would dominate)

static size_t cells(const PoissonLevel *l){ return (size_t)l->nx * l->ny * l->nz; }

int poisson_init(Poisson *ps, int nx, int ny, int nz, real h){
    memset(ps, 0, sizeof(*ps));
    ps->pre = ps->post = 3;
    for (int k = 0; k < POISSON_MAX_LEVELS; k++){
        PoissonLevel *l = &ps->lv[k];
        l->nx = nx; l->ny = ny; l->nz = nz; l->h = h;
        size_t n = cells(l);
        l->res = calloc(n, sizeof(float));
        l->links = calloc(n, sizeof(uint16_t));
        if (k > 0){
            l->p = calloc(n, sizeof(float));
            l->rhs = calloc(n, sizeof(float));
            l->solid = calloc(n, 1);
            if (!l->p || !l->rhs || !l->solid) return 1;
        }
        if (!l->res || !l->links) return 1;
        ps->nlevels = k + 1;
        // the next level: half as many cells per axis, while every side keeps >= 4 cells
        if (nx < 8 || ny < 8 || nz < 8) break;
        nx = (nx + 1) / 2; ny = (ny + 1) / 2; nz = (nz + 1) / 2; h *= 2;
    }
    return 0;
}

void poisson_free(Poisson *ps){
    for (int k = 0; k < ps->nlevels; k++){
        PoissonLevel *l = &ps->lv[k];
        free(l->res); free(l->links);
        if (k > 0){ free(l->p); free(l->rhs); free(l->solid); }
    }
    memset(ps, 0, sizeof(*ps));
}

// links of one level from its solid mask
static void make_links(PoissonLevel *l, unsigned open){
    const int nx = l->nx, ny = l->ny, nz = l->nz;
    int big = cells(l) >= BIG_GRID;
    (void)big;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (big)
#endif
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = ((size_t)z * ny + y) * nx + x;
                uint16_t m = 0, o = 0;
                if (!l->solid[i]){
                    const int nb[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z}, {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
                    for (int q = 0; q < 6; q++){
                        int a = nb[q][0], b = nb[q][1], c = nb[q][2];
                        if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz){
                            if (open >> q & 1) o++;            // the face numbering matches q
                            continue;
                        }
                        if (!l->solid[((size_t)c * ny + b) * nx + a]) m |= (uint16_t)(1u << q);
                    }
                }
                l->links[i] = (uint16_t)(m | o << 6);
            }
}

void poisson_setup(Poisson *ps, uint8_t *solid, unsigned open){
    ps->lv[0].solid = solid;
    make_links(&ps->lv[0], open);
    for (int k = 1; k < ps->nlevels; k++){
        PoissonLevel *c = &ps->lv[k], *f = &ps->lv[k - 1];
        // a coarse cell is solid only if all its (up to 8) fine cells are
        for (int z = 0; z < c->nz; z++)
            for (int y = 0; y < c->ny; y++)
                for (int x = 0; x < c->nx; x++){
                    int all = 1;
                    for (int q = 0; q < 8 && all; q++){
                        int a = 2 * x + (q & 1), b = 2 * y + (q >> 1 & 1), d = 2 * z + (q >> 2);
                        if (a >= f->nx || b >= f->ny || d >= f->nz) continue;
                        if (!f->solid[((size_t)d * f->ny + b) * f->nx + a]) all = 0;
                    }
                    c->solid[((size_t)z * c->ny + y) * c->nx + x] = (uint8_t)all;
                }
        make_links(c, open);
    }
}

// sum of the linked neighbors' p and the count k
static inline real neighbors(const PoissonLevel *l, const float *p, size_t i, unsigned m, int *k){
    const size_t sy = (size_t)l->nx, sz = (size_t)l->nx * l->ny;
    real s = 0;
    if (m & 1u)  s += p[i - 1];
    if (m & 2u)  s += p[i + 1];
    if (m & 4u)  s += p[i - sy];
    if (m & 8u)  s += p[i + sy];
    if (m & 16u) s += p[i - sz];
    if (m & 32u) s += p[i + sz];
    *k = __builtin_popcount(m & 63u) + (int)(m >> 6);
    return s;
}

// red-black Gauss-Seidel sweeps
static void smooth(PoissonLevel *l, int sweeps){
    const int nx = l->nx, ny = l->ny, nz = l->nz;
    const real h2 = l->h * l->h;
    int big = cells(l) >= BIG_GRID;
    (void)big;
    for (int it = 0; it < sweeps; it++)
        for (int color = 0; color < 2; color++){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (big)
#endif
            for (int z = 0; z < nz; z++)
                for (int y = 0; y < ny; y++)
                    for (int x = (y + z + color) & 1; x < nx; x += 2){
                        size_t i = ((size_t)z * ny + y) * nx + x;
                        unsigned m = l->links[i];
                        if (!m) continue;
                        int k;
                        real s = neighbors(l, l->p, i, m, &k);
                        l->p[i] = (float)((s - h2 * l->rhs[i]) / k);
                    }
        }
}

// res = rhs - L p
static void residual(PoissonLevel *l){
    const real inv_h2 = 1.0f / (l->h * l->h);
    const size_t n = cells(l);
    int big = n >= BIG_GRID;
    (void)big;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (big)
#endif
    for (size_t i = 0; i < n; i++){
        unsigned m = l->links[i];
        if (!m){ l->res[i] = 0; continue; }
        int k;
        real s = neighbors(l, l->p, i, m, &k);
        l->res[i] = (float)(l->rhs[i] - (s - k * l->p[i]) * inv_h2);
    }
}

static void vcycle(Poisson *ps, int k){
    PoissonLevel *l = &ps->lv[k];
    if (k == ps->nlevels - 1){          // the coarsest grid: a few cells, just sweep
        smooth(l, 40);
        return;
    }
    smooth(l, ps->pre);
    residual(l);
    PoissonLevel *c = &ps->lv[k + 1];
    // restrict: the coarse right-hand side is the mean residual of the fine fluid cells
    for (int z = 0; z < c->nz; z++)
        for (int y = 0; y < c->ny; y++)
            for (int x = 0; x < c->nx; x++){
                size_t ci = ((size_t)z * c->ny + y) * c->nx + x;
                real s = 0;
                int n = 0;
                for (int q = 0; q < 8; q++){
                    int a = 2 * x + (q & 1), b = 2 * y + (q >> 1 & 1), d = 2 * z + (q >> 2);
                    if (a >= l->nx || b >= l->ny || d >= l->nz) continue;
                    size_t fi = ((size_t)d * l->ny + b) * l->nx + a;
                    if (!l->links[fi]) continue;
                    s += l->res[fi];
                    n++;
                }
                c->rhs[ci] = n ? (float)(s / n) : 0;
                c->p[ci] = 0;
            }
    vcycle(ps, k + 1);
    // prolong: every fine cell takes its coarse cell's correction
    const int nx = l->nx, ny = l->ny, nz = l->nz;
    int big = cells(l) >= BIG_GRID;
    (void)big;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (big)
#endif
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++){
                size_t i = ((size_t)z * ny + y) * nx + x;
                if (!l->links[i]) continue;
                l->p[i] += c->p[((size_t)(z / 2) * c->ny + y / 2) * c->nx + x / 2];
            }
    smooth(l, ps->post);
}

void poisson_solve(Poisson *ps, float *p, float *rhs, int cycles){
    ps->lv[0].p = p;
    ps->lv[0].rhs = rhs;
    for (int c = 0; c < cycles; c++) vcycle(ps, 0);
}

real poisson_residual(Poisson *ps, float *p, float *rhs){
    PoissonLevel *l = &ps->lv[0];
    l->p = p;
    l->rhs = rhs;
    residual(l);
    real m = 0;
    for (size_t i = 0; i < cells(l); i++) if (r_fabs(l->res[i]) > m) m = r_fabs(l->res[i]);
    return m;
}
