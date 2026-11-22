// probe.c - light probes (see probe.h)
#include "rt/probe.h"

#include <stdlib.h>
#include <string.h>
#include "base/rng.h"
#include "base/sampler.h"
#include "rt/tracer.h"
#include "rt/accel.h"
#include "base/clock.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static TraceParams probe_trace_params(void){
    TraceParams tp = trace_default_params();
    tp.max_depth = PROBE_DEPTH;
    tp.sampler = 0;
    tp.no_lamp_seen = 1;
    return tp;
}

// the generator of row y (its texels draw from it in turn)
static void row_rng(Rng *rng, int y){ rng_seed(rng, 0x9e3779b97f4a7c15ull ^ (uint64_t)y, 977); }

// texels x0..x1-1 of row y of a w x h probe at center into img, drawing from the row's rng
static void bake_texels(const Scene *s, Vec3 center, int w, int h, int spp, const TraceParams *tp, int y, int x0, int x1,
                        Rng *rng, float *img){
    for (int x = x0; x < x1; x++){
        Vec3 sum = vec3(0, 0, 0);
        for (int i = 0; i < spp; i++){
            // uniform over the texel's solid angle: u uniform, cos(theta) uniform in the row
            real u = (x + rng_real(rng)) / w;
            real c0 = r_cos(y * (real)PI / h), c1 = r_cos((y + 1) * (real)PI / h);
            real c = c0 + (c1 - c0) * rng_real(rng);
            real v = r_acos(c < -1 ? -1 : c > 1 ? 1 : c) * (real)(1.0 / PI);
            Env tmp = {0};                          // (env_dir of an unturned map)
            Ray ray = {center, env_dir(&tmp, u, v)};
            Sampler sm = sampler_make(rng, 0, 0, 0, 0, 0);
            Vec3 L = trace_path_sampled(s, ray, &sm, tp);
            if (L.x == L.x && L.y == L.y && L.z == L.z) sum = v_add(sum, L);   // (no NaN)
        }
        float *o = img + 3 * ((size_t)y * w + x);
        o[0] = (float)(sum.x / spp); o[1] = (float)(sum.y / spp); o[2] = (float)(sum.z / spp);
    }
}

// row y of a w x h probe at center: its texels into img
static void bake_row(const Scene *s, Vec3 center, int w, int h, int spp, const TraceParams *tp, int y, float *img){
    Rng rng;
    row_rng(&rng, y);
    bake_texels(s, center, w, h, spp, tp, y, 0, w, &rng, img);
}

// the baked texels as the probe (img taken)
static Env *probe_env(float *img, int w, int h, Vec3 center, const Aabb *box){
    Env *e = env_from_rgb(img, w, h);
    if (e && box){
        e->boxed = 1;
        e->center = center;
        e->bmin = box->lo;
        e->bmax = box->hi;
    }
    return e;
}

Env *probe_bake(const Scene *s, Vec3 center, const Aabb *box, int w, int spp){
    if (w < 8) w = 8;
    w &= ~1;
    int h = w / 2;
    if (spp < 1) spp = 1;
    float *img = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (!img) return NULL;
    TraceParams tp = probe_trace_params();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++) bake_row(s, center, w, h, spp, &tp, y, img);
    return probe_env(img, w, h, center, box);
}

// How open the space around c is: the harmonic mean of how far 64 rays in all directions
// go before they hit something (most of them hitting a shape from its back: c is inside
// something, 0; a few: c is behind an open surface, as a glTF model's thin panels); rays
// that hit nothing count as `far`
static real openness_ex(const Scene *s, Vec3 c, real far, int *escaped){
    real inv = 0;
    int backs = 0;
    *escaped = 0;
    for (int i = 0; i < 64; i++){
        real z = 1 - 2 * (i + 0.5f) / 64, r = r_sqrt(r_fmax(0, 1 - z * z)), ph = 2.399963f * i;   // a Fibonacci sphere
        Ray ray = {c, vec3(r * r_cos(ph), z, r * r_sin(ph))};
        Hit h;
        if (scene_intersect(s, &ray, 1e-3f, far, &h)){
            // the back of a closed shape (a mesh, a sphere, an instance; not a quad: walls
            // are seen from either side)
            int closed = h.inst >= 0 || h.prim >= s->nprims || s->prims[h.prim].type == PRIM_SPHERE;
            if (!h.front_face && closed && s->mats[hit_material(s, &h)].type != MAT_LIGHT) backs++;
            inv += 1 / r_fmax(h.t, 1e-3f);
        } else { inv += 1 / far; (*escaped)++; }
    }
    return backs > 32 ? 0 : 64 / inv;
}
static real openness(const Scene *s, Vec3 c, real far){
    int e;
    return openness_ex(s, c, far, &e);
}

// Where the probe goes: the place asked for, unless a point of a 3 x 3 x 3 grid over the box
// (25%, 50%, 75% of each side; an open scene's: around the place, up to 300 units away) is
// more open (openness): a model standing in the middle of the room would hide the room
// from a probe inside it or just above it
static Vec3 probe_place(const Scene *s, Vec3 at, const Aabb *box){
    Vec3 lo = box->lo, hi = box->hi;
    real far = v_len(v_sub(hi, lo));
    if (far > 4000){                                  // an open scene: around the place
        lo = v_sub(at, vec3(300, 0, 300));
        hi = v_add(at, vec3(300, 300, 300));
        far = 4000;
    }
    Vec3 best = at;
    real best_o = openness(s, at, far) * 1.05f;      // (the place asked for wins a near tie)
    for (int i = 1; i <= 3; i++)
        for (int j = 1; j <= 3; j++)
            for (int k = 1; k <= 3; k++){
                Vec3 c = vec3(lo.x + (hi.x - lo.x) * i / 4, lo.y + (hi.y - lo.y) * j / 4, lo.z + (hi.z - lo.z) * k / 4);
                real o = openness(s, c, far);
                if (o > best_o){ best_o = o; best = c; }
            }
    return best;
}

// a grid being baked: the grid and what each of its probes needs
typedef struct {
    IrrGrid *g;
    int n[3], paths, ref_escaped;
    Vec3 lo, hi;
    real far;
    TraceParams tp;
} GridBake;

// the grid made (its probes zero, unused); 0 = none for this scene, or out of memory
static int grid_begin(const Scene *s, int paths, GridBake *gb){
    memset(gb, 0, sizeof(*gb));
    if (s->light < 0 && !s->env) return 0;
    Vec3 lo, hi;
    int n[3] = {5, 5, 5};
    if (s->probe_set && v_len(v_sub(s->probe_box.hi, s->probe_box.lo)) < 4000){
        lo = s->probe_box.lo; hi = s->probe_box.hi;
    } else if (s->probe_set){                         // open: around the probe, from the ground up
        lo = vec3(s->probe_at.x - 700, s->probe_box.lo.y, s->probe_at.z - 700);
        hi = vec3(s->probe_at.x + 700, s->probe_box.lo.y + 400, s->probe_at.z + 700);
        n[1] = 3;
    } else return 0;
    IrrGrid *g = calloc(1, sizeof(*g));
    if (!g) return 0;
    size_t np = (size_t)n[0] * n[1] * n[2];
    g->sh = calloc(np * IRR_FLOATS, sizeof(float));
    if (!g->sh){ free(g); return 0; }
    for (int k = 0; k < 3; k++){ g->n[k] = n[k]; gb->n[k] = n[k]; }
    g->lo = lo; g->hi = hi;
    static unsigned next_id;
    g->id = ++next_id;
    gb->g = g;
    gb->lo = lo; gb->hi = hi;
    gb->paths = paths < 16 ? 16 : paths;
    gb->tp = probe_trace_params();
    gb->tp.no_sky_seen = 1;
    gb->far = v_len(v_sub(hi, lo)) * 4;
    // a probe inside a box of quads (they are seen from either side: no back to hit) is
    // known by none of its rays leaving the scene, where they do from the probe's own place
    if (s->probe_set) openness_ex(s, s->probe_at, gb->far, &gb->ref_escaped);
    return 1;
}
static int grid_count(const GridBake *gb){ return gb->n[0] * gb->n[1] * gb->n[2]; }

// one probe of the grid being baked: where it is, its generator, its sums so far, and how
// many of its samples are done (its paths, then as many of the sky)
typedef struct {
    int idx, i;
    Vec3 c;
    Rng rng;
    real turn;
    double acc[9][3], sky[9][3];
} GridProbe;

// probe idx begun; 0: it is inside something (unused: nothing to do)
static int grid_probe_begin(const Scene *s, const GridBake *gb, int idx, GridProbe *p){
    const int *n = gb->n;
    const Vec3 lo = gb->lo, hi = gb->hi;
    int ix = idx % n[0], iy = (idx / n[0]) % n[1], iz = idx / (n[0] * n[1]);
    memset(p, 0, sizeof(*p));
    p->idx = idx;
    p->c = vec3(lo.x + (hi.x - lo.x) * (ix + 0.5f) / n[0], lo.y + (hi.y - lo.y) * (iy + 0.5f) / n[1],
                lo.z + (hi.z - lo.z) * (iz + 0.5f) / n[2]);
    int escaped;
    if (openness_ex(s, p->c, gb->far, &escaped) <= 0 || (gb->ref_escaped > 0 && escaped == 0)) return 0;
    rng_seed(&p->rng, 0x51ed27u + (uint64_t)idx, 31);
    p->turn = rng_real(&p->rng) * (real)(2 * PI);
    return 1;
}

// up to `most` more of its samples; 1 when all are done
static int grid_probe_work(const Scene *s, const GridBake *gb, GridProbe *p, int most){
    const int paths = gb->paths;
    for (int k = 0; k < most && p->i < 2 * paths; k++, p->i++){
        if (p->i < paths){
            int i = p->i;
            // a Fibonacci sphere, turned at random per probe
            real z = 1 - 2 * (i + rng_real(&p->rng)) / paths, r = r_sqrt(r_fmax(0, 1 - z * z)), ph = 2.399963f * i + p->turn;
            Vec3 d = vec3(r * r_cos(ph), z, r * r_sin(ph));
            Ray ray = {p->c, d};
            Sampler sm = sampler_make(&p->rng, 0, 0, 0, 0, 0);
            Vec3 L = trace_path_sampled(s, ray, &sm, &gb->tp);
            if (!(L.x == L.x && L.y == L.y && L.z == L.z)) continue;
            // a rare, very bright sample (the lamp seen through glass) would leave a blotch:
            // clamped (the diffuse light is a little low where caustics fall). (The sky seen
            // directly is not in these: sampled below, by its luminance.)
            real mx = v_max(L);
            if (mx > PROBE_GRID_CLAMP) L = v_scale(L, PROBE_GRID_CLAMP / mx);
            real y[9];
            env_sh_basis(d, y);
            for (int b = 0; b < 9; b++){ p->acc[b][0] += L.x * y[b]; p->acc[b][1] += L.y * y[b]; p->acc[b][2] += L.z * y[b]; }
        } else if (s->env){
            // the sky seen directly: directions drawn by its luminance (a sun is found), each
            // counted if nothing is in the way
            int i = p->i - paths;
            Vec3 d;
            real pdf;
            Vec3 L = env_sample(s->env, (i + rng_real(&p->rng)) / paths, rng_real(&p->rng), &d, &pdf);
            if (!(pdf > 0)) continue;
            Ray ray = {p->c, d};
            if (scene_occluded(s, &ray, 1e-3f, 1e30f)) continue;
            real y[9];
            env_sh_basis(d, y);
            for (int b = 0; b < 9; b++){ p->sky[b][0] += L.x * y[b] / pdf; p->sky[b][1] += L.y * y[b] / pdf; p->sky[b][2] += L.z * y[b] / pdf; }
        }
    }
    return p->i >= 2 * paths;
}

// its sums into the grid
static void grid_probe_end(const GridBake *gb, const GridProbe *p){
    static const double band[9] = {1, 2.0 / 3, 2.0 / 3, 2.0 / 3, 0.25, 0.25, 0.25, 0.25, 0.25};
    float *q = gb->g->sh + IRR_FLOATS * (size_t)p->idx;
    for (int b = 0; b < 9; b++)
        for (int k = 0; k < 3; k++) q[3 * b + k] = (float)((p->acc[b][k] * 4 * PI + p->sky[b][k]) / gb->paths * band[b]);
    q[27] = 1;
}

// probe idx of the grid, whole
static void grid_probe(const Scene *s, const GridBake *gb, int idx){
    GridProbe p;
    if (!grid_probe_begin(s, gb, idx, &p)) return;              // inside something: unused
    grid_probe_work(s, gb, &p, 2 * gb->paths);
    grid_probe_end(gb, &p);
}

IrrGrid *probe_grid_bake_scene(const Scene *s, int paths){
    GridBake gb;
    if (!grid_begin(s, paths, &gb)) return NULL;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < grid_count(&gb); idx++) grid_probe(s, &gb, idx);
    return gb.g;
}

// where the scene's probe goes, and the box its lookups are corrected by; 0: none
static int probe_where(const Scene *s, int *w, Vec3 *center, Aabb *box){
    if (s->light < 0 && !s->env) return 0;
    if (*w <= 0) *w = s->light >= 0 ? PROBE_RES : PROBE_RES_OPEN;
    if (s->probe_set){
        *box = s->probe_box;
        *center = probe_place(s, s->probe_at, &s->probe_box);
        return 1;
    }
    if (s->light < 0) return 0;
    Aabb b = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};   // lo, hi
    int any = 0;
    for (int i = 0; i < s->nprims; i++){
        const Primitive *p = &s->prims[i];
        if (i == s->light || p->type != PRIM_QUAD) continue;
        Vec3 c[4] = {p->q, v_add(p->q, p->u), v_add(p->q, p->v), v_add(v_add(p->q, p->u), p->v)};
        for (int k = 0; k < 4; k++){
            b.lo = vec3(r_fmin(b.lo.x, c[k].x), r_fmin(b.lo.y, c[k].y), r_fmin(b.lo.z, c[k].z));
            b.hi = vec3(r_fmax(b.hi.x, c[k].x), r_fmax(b.hi.y, c[k].y), r_fmax(b.hi.z, c[k].z));
        }
        any = 1;
    }
    if (!any) return 0;
    *box = b;
    *center = probe_place(s, v_scale(v_add(b.lo, b.hi), 0.5f), &b);
    return 1;
}

Env *probe_bake_scene(const Scene *s, int w, int spp){
    Vec3 c;
    Aabb box;
    if (!probe_where(s, &w, &c, &box)) return NULL;
    return probe_bake(s, c, &box, w, spp);
}

// ---- the same bake in steps (probe.h) ----

struct ProbeJob {
    int w, h, spp, row, x;          // the probe: its size, the rows done and the texels of the next
    Rng rng;                        // (that row's generator)
    Vec3 center;
    Aabb box;
    float *img;
    TraceParams tp;
    GridBake gb;                    // the grid (gb.g NULL: none), and its probes done
    int grid, idx;
    GridProbe gp;                   // the one being baked (begun: gp_on)
    int gp_on;
    double probe_work, grid_work;   // (paths: how much of the whole each part is)
    Env *env;                       // then the probe made of the texels, its levels made in steps
    EnvPrep *prep;
    double prep_done;
    int failed;                     // (out of memory)
};
#define PREP_SHARE 0.15             // (of the whole: the levels' part)

ProbeJob *probe_job_start(const Scene *s, int w, int spp, int grid_paths){
    ProbeJob *j = calloc(1, sizeof(*j));
    if (!j) return NULL;
    if (!probe_where(s, &w, &j->center, &j->box)){ free(j); return NULL; }
    if (w < 8) w = 8;
    w &= ~1;
    j->w = w;
    j->h = w / 2;
    j->spp = spp < 1 ? 1 : spp;
    j->img = malloc(sizeof(float) * 3 * (size_t)j->w * j->h);
    if (!j->img){ free(j); return NULL; }
    j->tp = probe_trace_params();
    j->grid = grid_begin(s, grid_paths, &j->gb);
    j->probe_work = (double)j->w * j->h * j->spp;
    j->grid_work = j->grid ? 2.0 * grid_count(&j->gb) * j->gb.paths : 0;
    return j;
}

static int traced(const ProbeJob *j){ return j->row >= j->h && (!j->grid || j->idx >= grid_count(&j->gb)); }

static double job_done(const ProbeJob *j){
    if (j->env) return j->prep ? 1 - PREP_SHARE + PREP_SHARE * j->prep_done : 1;
    double d = j->probe_work * (j->row + (double)j->x / j->w) / j->h + (j->grid ? j->grid_work * j->idx / grid_count(&j->gb) : 0);
    return (1 - PREP_SHARE) * d / (j->probe_work + j->grid_work);
}

double probe_job_step(ProbeJob *j, const Scene *s, double ms){
    double t0 = clock_ms();
    if (j->env){                                       // the levels (a third of a second at once in a page)
        if (j->prep && (j->prep_done = env_prepare_step(j->prep, ms)) >= 1){ env_prepare_free(j->prep); j->prep = NULL; }
        return job_done(j);
    }
    while (!traced(j)){
        if (j->row < j->h){
#ifdef _OPENMP
            if (j->x == 0){                            // (natively a few whole rows at once, on the threads)
                int n = omp_get_max_threads();
                if (n > j->h - j->row) n = j->h - j->row;
                const int y0 = j->row;
                #pragma omp parallel for schedule(dynamic, 1)
                for (int y = y0; y < y0 + n; y++) bake_row(s, j->center, j->w, j->h, j->spp, &j->tp, y, j->img);
                j->row += n;
                if (clock_ms() - t0 >= ms) break;
                continue;
            }
#endif
            // (in a page a few texels at a time: a row of 512 takes longer than a step should)
            if (j->x == 0) row_rng(&j->rng, j->row);
            bake_texels(s, j->center, j->w, j->h, j->spp, &j->tp, j->row, j->x, j->x + 1, &j->rng, j->img);
            if (++j->x == j->w){ j->x = 0; j->row++; }
        } else {
            if (!j->gp_on){
                if (!grid_probe_begin(s, &j->gb, j->idx, &j->gp)){ j->idx++; continue; }   // (unused)
                j->gp_on = 1;
            }
            if (grid_probe_work(s, &j->gb, &j->gp, 32)){ grid_probe_end(&j->gb, &j->gp); j->gp_on = 0; j->idx++; }
        }
        if (clock_ms() - t0 >= ms) break;
    }
    if (traced(j)){                                    // all traced: the probe, its levels next
        j->env = env_wrap_rgb(j->img, j->w, j->h);
        j->img = NULL;
        if (j->env){
            j->env->boxed = 1;
            j->env->center = j->center;
            j->env->bmin = j->box.lo;
            j->env->bmax = j->box.hi;
            j->prep = env_prepare_start(j->env);
        }
        if (!j->env || !j->prep){ env_free(j->env); j->env = NULL; j->failed = 1; }
    }
    return j->failed ? 1 : job_done(j);
}

int probe_job_finish(ProbeJob *j, Env **probe, IrrGrid **irr){
    if (!j->failed && (!j->env || j->prep)) return 1;
    *probe = j->failed ? NULL : j->env;
    j->env = NULL;
    *irr = j->failed ? NULL : j->gb.g;
    if (!j->failed) j->gb.g = NULL;
    probe_job_free(j);
    return *probe ? 0 : -1;
}

void probe_job_free(ProbeJob *j){
    if (!j) return;
    env_prepare_free(j->prep);
    env_free(j->env);
    free(j->img);
    irr_grid_free(j->gb.g);
    free(j);
}
