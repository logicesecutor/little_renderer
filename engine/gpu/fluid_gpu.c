#include "gpu/fluid_gpu.h"

#include <stdlib.h>

#ifndef LR_GPU
// ---- built without webgpu.h: the air stays on the CPU ------------------------------------
GpuFluid *gpu_fluid_create(Gpu *g, PhysWorld *pw){ (void)g; (void)pw; return NULL; }
void gpu_fluid_attach_view(GpuFluid *gf, GpuView *v){ (void)gf; (void)v; }
void gpu_fluid_sync_cloud(GpuFluid *gf){ (void)gf; }
int gpu_fluid_pending(const GpuFluid *gf){ (void)gf; return 0; }
void gpu_fluid_set_wait(GpuFluid *gf, int wait){ (void)gf; (void)wait; }
void gpu_fluid_destroy(GpuFluid *gf){ (void)gf; }
double gpu_fluid_gpu_ms(GpuFluid *gf){ (void)gf; return 0; }
void gpu_fluid_gpu_ms_reset(GpuFluid *gf){ (void)gf; }
#else

#include <stdio.h>
#include <string.h>
#include "scene/volume.h"
#include "gpu/gpu_internal.h"
#include "phys/fluid.h"
#include "phys/rigid.h"

// ---- the kernels ------------------------------------------------------------------------------
//
// Every kernel is one @compute entry "main" of 64 threads per workgroup, over cells (nx ny nz),
// over grid points ((nx+1)(ny+1)(nz+1): the u, v and w faces at once, as the CPU loops do), over
// voxels of the cloud, or over bodies. The comments of phys/fluid.c explain the method.

// the step's parameters (struct GFP on the C side)
GPU_WGSL(FHEAD, fluid_head);

// trilinear sample of array QA (dims S, x fastest), clamped to the grid (sample3 in fluid.c);
// QN names the function. QM: the same, also returning the min and max of the 8 values.
GPU_WGSL(SAMPLE, fluid_sample);
GPU_WGSL(SAMPLE_MM, fluid_sample_mm);

// the air velocity at a world point (needs u, v, w and their samplers)
GPU_WGSL(VEL, fluid_vel);

// obstacles (needs solid)
GPU_WGSL(SOLID, fluid_solid);

// 1 if the face's value is fixed by a wall or a body, and the value (needs solid, svel)
GPU_WGSL(FIXED, fluid_fixed);

/* the cells inside each body become obstacles moving with it (mark_solids in world.c);
   a body is 6 vec4: position (w: 0 sphere, 1 box, 2 + k a mesh body whose cells are bits
   k.. of occ: 48^3 over [-half, half] of its own frame, phys/convex.h), its 3 axes (w: half
   sizes), velocity (w: bounding radius), angular velocity (w: radius of the drag samples) */
GPU_WGSL(K_MARK, fluid_mark);

/* walls and bodies fix the faces they touch (enforce_boundaries) */
GPU_WGSL(K_BOUND, fluid_bound);

/* wind and buoyancy (add_forces) */
GPU_WGSL(K_FORCES, fluid_forces);

/* vorticity confinement in 4 passes: velocity at the cell centers; vorticity and its length;
   the force N x omega; spread to the faces */
GPU_WGSL(K_VA, fluid_va);
GPU_WGSL(K_VB, fluid_vb);
GPU_WGSL(K_VC, fluid_vc);
GPU_WGSL(K_VD, fluid_vd);

/* semi-Lagrangian advection of the velocity into tu, tv, tw */
GPU_WGSL(K_ADV, fluid_adv);

/* the divergence, minus what the expansion sources ask for */
GPU_WGSL(K_DIV, fluid_div);

/* u -= grad p across the faces the pressure acts on */
GPU_WGSL(K_GRAD, fluid_grad);

/* the cloud, MacCormack: one step backwards into d1, then the correction */
GPU_WGSL(K_D1, fluid_d1);
GPU_WGSL(K_D2, fluid_d2);

/* the air around each body: the mean over 14 points on a shell just outside it */
GPU_WGSL(K_SAMP, fluid_samp);

// ---- the multigrid kernels (phys/poisson.c), on the levels' own parameters ----------------------

GPU_WGSL(LHEAD, fluid_level_head);

// neighbors of a cell through its links (needs p, links)
GPU_WGSL(NSUM, fluid_nsum);

GPU_WGSL(K_LINKS, fluid_links);
/* a coarse cell is solid only if all its fine cells are (L: the fine level) */
GPU_WGSL(K_CSOLID, fluid_csolid);
GPU_WGSL(K_SMOOTH, fluid_smooth);
GPU_WGSL(K_RESID, fluid_resid);
/* the coarse right-hand side: the mean residual of the fine fluid cells (L: the fine level) */
GPU_WGSL(K_RESTRICT, fluid_restrict);
GPU_WGSL(K_PROLONG, fluid_prolong);
/* the coarsest grid: 40 sweeps by one workgroup, which can wait for itself */
GPU_WGSL(K_COARSEST, fluid_coarsest);

// ---- the C side ---------------------------------------------------------------------------------

typedef struct {
    uint32_t n[4];
    float lo[4], a[4], wind[4];
    uint32_t dn[4];
    float dlo[4], dhi[4];
    uint32_t m[4];
    float src[8][4];
    float rate[8];
} GFP;

typedef struct { uint32_t d[4], c[4]; float h[4]; } GLU;

enum { K_MARK_, K_BOUND_, K_FORCES_, K_VA_, K_VB_, K_VC_, K_VD_, K_ADV_, K_DIV_, K_GRAD_, K_D1_, K_D2_, K_SAMP_,
       K_LINKS_, K_CSOLID_, K_SMOOTH_, K_RESID_, K_RESTRICT_, K_PROLONG_, K_COARSEST_, NK };

// what each kernel is made of: its bindings, its body, and the pieces it needs
enum { NEED_VEL = 1, NEED_SOLID = 2, NEED_FIXED = 4, NEED_D0 = 8, NEED_D0MM = 16, NEED_D1 = 32, NEED_NSUM = 64, LEVEL = 128 };
static const struct { const char *name, *types; const char **body; int needs; } KERNELS[NK] = {
    {"air: obstacles",    "URWWR",    &K_MARK, 0},
    {"air: walls",        "UWWWRR",   &K_BOUND, NEED_SOLID | NEED_FIXED},
    {"air: forces",       "UWWWR",    &K_FORCES, 0},
    {"air: vorticity 1",  "URRRW",    &K_VA, 0},
    {"air: vorticity 2",  "URW",      &K_VB, 0},
    {"air: vorticity 3",  "URRW",     &K_VC, 0},
    {"air: vorticity 4",  "URWWW",    &K_VD, 0},
    {"air: advection",    "URRRWWW",  &K_ADV, NEED_VEL},
    {"air: divergence",   "URRRRW",   &K_DIV, 0},
    {"air: gradient",     "UWWWRRR",  &K_GRAD, NEED_SOLID | NEED_FIXED},
    {"cloud: advection",  "URRRRW",   &K_D1, NEED_VEL | NEED_D0},
    {"cloud: correction", "URRRRRWR", &K_D2, NEED_VEL | NEED_SOLID | NEED_D0MM | NEED_D1},
    {"air: around bodies","URRRRW",   &K_SAMP, NEED_VEL},
    {"pressure: links",   "URW",      &K_LINKS, LEVEL},
    {"pressure: coarse",  "URW",      &K_CSOLID, LEVEL},
    {"pressure: sweep",   "URWR",     &K_SMOOTH, LEVEL | NEED_NSUM},
    {"pressure: residual","URRRW",    &K_RESID, LEVEL | NEED_NSUM},
    {"pressure: restrict","URRWW",    &K_RESTRICT, LEVEL},
    {"pressure: prolong", "URWR",     &K_PROLONG, LEVEL},
    {"pressure: coarsest","URWR",     &K_COARSEST, LEVEL | NEED_NSUM},
};

typedef struct {
    int nx, ny, nz;
    WGPUBuffer links, res, solid, p, rhs;   // level 0: solid, p and rhs are the fluid's
    WGPUBuffer lu[2];                       // parameters, for the two colors of the sweeps
    WGPUBindGroup bg_links, bg_csolid, bg_smooth[2], bg_resid, bg_restrict, bg_prolong, bg_coarsest;
} GLevel;

struct GpuFluid {
    Gpu *g;
    PhysWorld *pw;
    Fluid *f;
    Volume *cloud;                // f->density, or NULL
    WGPUBindGroupLayout bgl[NK];
    WGPUComputePipeline pipe[NK];
    WGPUBuffer fp;                // GFP
    // the velocity is in u, v, w or in tu, tv, tw, taking turns: the advection writes the set
    // it does not read, which is then the velocity (no copy back): `cur` = 1 when it is tu, tv, tw
    WGPUBuffer u, v, w, tu, tv, tw, p, div, solid, svel, cvel, omg, frc;
    int cur;
    // the cloud's two buffers take turns with them: it is in dens[cur], and its advection
    // reads that one and writes the other, where it is when the step ends (no copy either);
    // d1: MacCormack's first step. `link`: where the views that draw it find it (NULL: no cloud)
    WGPUBuffer dens[2], d1;
    GpuDensity *link;
    WGPUBuffer bodies, around;    // per body: 6 vec4 in, 1 vec4 out
    size_t bodies_cap, around_cap;
    WGPUBuffer occ;               // the mesh bodies' shapes: which cells they fill
    size_t occ_cap;
    int nocc;                     // shapes uploaded
    WGPUBindGroup bg[2][NK];      // the fluid kernels' bind groups, for each set the velocity can be in
                                  // (MARK and SAMP follow the bodies)
    GLevel lv[POISSON_MAX_LEVELS];
    int nlevels;
    float *body_data;
    // readbacks: the air around the bodies (every step), the cloud (when asked)
    WGPUBuffer around_rb, dens_rb;
    size_t around_rb_cap;
    int around_pending, around_n;
    int dens_pending, dens_wanted;
    uint64_t steps, dens_step;
    Vec3 *around_cpu;
    int around_cpu_cap;
    // back-pressure: steps submitted and not finished yet. A slow GPU must not fall behind
    // without limit (the frames would queue up behind the air): beyond MAX_INFLIGHT the step
    // waits, and its time is added to the next one (semi-Lagrangian steps take any dt).
    int inflight;
    real dt_owed;
    int wait;                     // offline: wait for the GPU instead of skipping
    GpuTimer *timer;              // LR_GPU_TIMING=1: the GPU time of the step's passes
};
// the step's passes, as the timer counts them
enum { FT_AIR, FT_PRESSURE, FT_CLOUD, FT_AROUND, NFT };
static const char *const FT_NAMES[NFT] = {"obstacles, forces, advection", "pressure (multigrid)", "the cloud's advection",
                                          "the air around the bodies"};

#define MAX_INFLIGHT 2
#define MAX_OWED 0.1f

// the GPU fluids alive: a "work done" callback can arrive after gpu_fluid_destroy
// (creation is refused when the list is full: one left out would have its callbacks ignored)
#define MAX_LIVE_FLUID 64
static GpuFluid *live[MAX_LIVE_FLUID];
static int is_live(const GpuFluid *gf){
    for (int i = 0; i < MAX_LIVE_FLUID; i++) if (gf && live[i] == gf) return 1;
    return 0;
}
static int live_slot(void){
    for (int i = 0; i < MAX_LIVE_FLUID; i++) if (!live[i]) return i;
    fprintf(stderr, "GPU: %d fluids already: no more\n", MAX_LIVE_FLUID);
    return -1;
}

static void on_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg; (void)u2;
    GpuFluid *gf = u1;
    if (is_live(gf) && gf->inflight > 0) gf->inflight--;
}

static size_t nu(const Fluid *f){ return (size_t)(f->nx + 1) * f->ny * f->nz; }
static size_t nv(const Fluid *f){ return (size_t)f->nx * (f->ny + 1) * f->nz; }
static size_t nw(const Fluid *f){ return (size_t)f->nx * f->ny * (f->nz + 1); }
static size_t ncells(const Fluid *f){ return (size_t)f->nx * f->ny * f->nz; }
static size_t npoints(const Fluid *f){ return (size_t)(f->nx + 1) * (f->ny + 1) * (f->nz + 1); }

// the WGSL of kernel k: the header, the samplers and helpers it needs, its body
static char *kernel_source(int k){
    const int nd = KERNELS[k].needs;
    // room for every part it may take (six samplers, each at most SAMPLE's length plus names)
    size_t sl = strlen(SAMPLE) > strlen(SAMPLE_MM) ? strlen(SAMPLE) : strlen(SAMPLE_MM);
    size_t cap = strlen(LHEAD) + strlen(FHEAD) + strlen(*KERNELS[k].body) + strlen(NSUM) + strlen(SOLID) + strlen(FIXED) +
                 strlen(VEL) + 6 * (2 * sl + 2) + 16;
    char *s = malloc(cap);
    if (!s) return NULL;
    s[0] = 0;
#define ADD(t) do { strcat(s, t); strcat(s, "\n"); } while (0)
    ADD(nd & LEVEL ? LHEAD : FHEAD);
    ADD(*KERNELS[k].body);
    if (nd & NEED_NSUM) ADD(NSUM);
    if (nd & NEED_SOLID) ADD(SOLID);
    if (nd & NEED_FIXED) ADD(FIXED);
    // samplers: copies of SAMPLE with QN / QA replaced
    const char *names[6] = {"u", "v", "w", "d0", "d0", "d1"};
    const char *tmpl[6] = {SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE_MM, SAMPLE};
    int want[6] = {nd & NEED_VEL, nd & NEED_VEL, nd & NEED_VEL, (nd & (NEED_D0 | NEED_D0MM)) != 0, nd & NEED_D0MM, nd & NEED_D1};
    for (int i = 0; i < 6; i++){
        if (!want[i]) continue;
        const char *t = tmpl[i];
        char *o = s + strlen(s);
        for (; *t; t++){
            if (t[0] == 'Q' && (t[1] == 'N' || t[1] == 'A')){ strcpy(o, names[i]); o += strlen(names[i]); t++; }
            else *o++ = *t;
        }
        *o++ = '\n';
        *o = 0;
    }
    if (nd & NEED_VEL) ADD(VEL);
#undef ADD
    return s;
}

static WGPUBuffer storage(Gpu *g, const char *label, size_t bytes){
    return gpu_buffer(g, label, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
}

static WGPUBuffer uniform(Gpu *g, const char *label, const void *data, size_t bytes){
    WGPUBuffer b = gpu_buffer(g, label, bytes, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    if (data) wgpuQueueWriteBuffer(g->queue, b, 0, data, bytes);
    return b;
}

#define BG(k, ...) do { WGPUBuffer b_[] = {__VA_ARGS__}; gf->bg[a][k] = gpu_bind(g, gf->bgl[k], b_, (int)(sizeof(b_) / sizeof(b_[0]))); } while (0)
#define LBG(dst, k, ...) do { WGPUBuffer b_[] = {__VA_ARGS__}; dst = gpu_bind(g, gf->bgl[k], b_, (int)(sizeof(b_) / sizeof(b_[0]))); } while (0)

// the bind groups that do not depend on the bodies
static void make_bind_groups(GpuFluid *gf){
    Gpu *g = gf->g;
    for (int a = 0; a < 2; a++){      // a: the set the velocity is in (the advection writes the other)
        WGPUBuffer u = a ? gf->tu : gf->u, v = a ? gf->tv : gf->v, w = a ? gf->tw : gf->w;
        WGPUBuffer tu = a ? gf->u : gf->tu, tv = a ? gf->v : gf->tv, tw = a ? gf->w : gf->tw;
        BG(K_BOUND_, gf->fp, u, v, w, gf->solid, gf->svel);
        BG(K_FORCES_, gf->fp, u, v, w, gf->dens[a]);
        BG(K_VA_, gf->fp, u, v, w, gf->cvel);
        BG(K_VB_, gf->fp, gf->cvel, gf->omg);
        BG(K_VC_, gf->fp, gf->omg, gf->solid, gf->frc);
        BG(K_VD_, gf->fp, gf->frc, u, v, w);
        BG(K_ADV_, gf->fp, u, v, w, tu, tv, tw);
        BG(K_DIV_, gf->fp, u, v, w, gf->solid, gf->div);
        BG(K_GRAD_, gf->fp, u, v, w, gf->p, gf->solid, gf->svel);
        // (the cloud's advection runs after the velocity's, which made this set the velocity:
        // the cloud is still in the other set's buffer, and goes into this one's)
        BG(K_D1_, gf->fp, u, v, w, gf->dens[1 - a], gf->d1);
        BG(K_D2_, gf->fp, u, v, w, gf->dens[1 - a], gf->d1, gf->dens[a], gf->solid);
    }
    for (int k = 0; k < gf->nlevels; k++){
        GLevel *l = &gf->lv[k];
        LBG(l->bg_links, K_LINKS_, l->lu[0], l->solid, l->links);
        if (k > 0) LBG(l->bg_csolid, K_CSOLID_, gf->lv[k - 1].lu[0], gf->lv[k - 1].solid, l->solid);
        for (int c = 0; c < 2; c++) LBG(l->bg_smooth[c], K_SMOOTH_, l->lu[c], l->links, l->p, l->rhs);
        LBG(l->bg_resid, K_RESID_, l->lu[0], l->links, l->p, l->rhs, l->res);
        if (k + 1 < gf->nlevels){
            GLevel *c = &gf->lv[k + 1];
            LBG(l->bg_restrict, K_RESTRICT_, l->lu[0], l->links, l->res, c->rhs, c->p);
            LBG(l->bg_prolong, K_PROLONG_, l->lu[0], l->links, l->p, c->p);
        }
        LBG(l->bg_coarsest, K_COARSEST_, l->lu[0], l->links, l->p, l->rhs);
    }
}

// the bodies' buffers can grow: their bind groups follow
_Static_assert(OCC_RES == 48 && OCC_WORDS % 32 == 0, "K_MARK's OCC, and its offsets in units of 32 words");

static void reserve_bodies(GpuFluid *gf, const RigidWorld *rw){
    Gpu *g = gf->g;
    int n = rw->nbodies;
    int grown = gpu_buffer_reserve(g, &gf->bodies, &gf->bodies_cap, (size_t)(n > 0 ? n : 1) * 96,
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "bodies");
    grown |= gpu_buffer_reserve(g, &gf->around, &gf->around_cap, (size_t)(n > 0 ? n : 1) * 16,
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "air around bodies");
    grown |= gpu_buffer_reserve(g, &gf->occ, &gf->occ_cap, (size_t)(rw->ngeoms > 0 ? rw->ngeoms : 1) * OCC_WORDS * 4,
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "mesh bodies' cells");
    if (grown) gf->nocc = 0;
    for (; gf->nocc < rw->ngeoms; gf->nocc++)          // shapes never change once made
        wgpuQueueWriteBuffer(g->queue, gf->occ, (uint64_t)gf->nocc * OCC_WORDS * 4, rw->geoms[gf->nocc]->occ, OCC_WORDS * 4);
    if (!grown && gf->bg[0][K_MARK_]) return;
    for (int a = 0; a < 2; a++){
        if (gf->bg[a][K_MARK_]) wgpuBindGroupRelease(gf->bg[a][K_MARK_]);
        if (gf->bg[a][K_SAMP_]) wgpuBindGroupRelease(gf->bg[a][K_SAMP_]);
        BG(K_MARK_, gf->fp, gf->bodies, gf->solid, gf->svel, gf->occ);
        BG(K_SAMP_, gf->fp, a ? gf->tu : gf->u, a ? gf->tv : gf->v, a ? gf->tw : gf->w, gf->bodies, gf->around);
    }
}

static void level_params(GpuFluid *gf, int k){
    const Fluid *f = gf->f;
    const PoissonLevel *pl = &f->poisson.lv[k];
    GLevel *l = &gf->lv[k];
    for (int c = 0; c < 2; c++){
        GLU u;
        memset(&u, 0, sizeof(u));
        u.d[0] = (uint32_t)pl->nx; u.d[1] = (uint32_t)pl->ny; u.d[2] = (uint32_t)pl->nz; u.d[3] = f->open;
        if (k + 1 < f->poisson.nlevels){
            const PoissonLevel *cl = &f->poisson.lv[k + 1];
            u.c[0] = (uint32_t)cl->nx; u.c[1] = (uint32_t)cl->ny; u.c[2] = (uint32_t)cl->nz;
        }
        u.c[3] = (uint32_t)c;
        u.h[0] = (float)(pl->h * pl->h);
        if (!l->lu[c]) l->lu[c] = uniform(gf->g, "pressure level", &u, sizeof(u));
        else wgpuQueueWriteBuffer(gf->g->queue, l->lu[c], 0, &u, sizeof(u));
    }
}

static void free_all(GpuFluid *gf){
    WGPUBuffer bufs[] = {gf->fp, gf->u, gf->v, gf->w, gf->tu, gf->tv, gf->tw, gf->p, gf->div, gf->solid, gf->svel, gf->cvel,
                         gf->omg, gf->frc, gf->dens[0], gf->dens[1], gf->d1, gf->bodies, gf->around, gf->around_rb, gf->dens_rb, gf->occ};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    for (int k = 0; k < gf->nlevels; k++){
        GLevel *l = &gf->lv[k];
        WGPUBuffer lb[] = {k ? l->solid : NULL, l->links, l->res, k ? l->p : NULL, k ? l->rhs : NULL, l->lu[0], l->lu[1]};
        for (size_t i = 0; i < sizeof(lb) / sizeof(lb[0]); i++) if (lb[i]){ wgpuBufferDestroy(lb[i]); wgpuBufferRelease(lb[i]); }
        WGPUBindGroup bgs[] = {l->bg_links, l->bg_csolid, l->bg_smooth[0], l->bg_smooth[1], l->bg_resid, l->bg_restrict,
                               l->bg_prolong, l->bg_coarsest};
        for (size_t i = 0; i < sizeof(bgs) / sizeof(bgs[0]); i++) if (bgs[i]) wgpuBindGroupRelease(bgs[i]);
    }
    for (int k = 0; k < NK; k++){
        for (int a = 0; a < 2; a++) if (gf->bg[a][k]) wgpuBindGroupRelease(gf->bg[a][k]);
        if (gf->pipe[k]) wgpuComputePipelineRelease(gf->pipe[k]);
        if (gf->bgl[k]) wgpuBindGroupLayoutRelease(gf->bgl[k]);
    }
    if (gf->link){
        gf->link->buf = NULL;                 // (the views that still hold it see the cloud gone)
        gpu_density_release(gf->link);
    }
    free(gf->body_data);
    free(gf->around_cpu);
    free(gf);
}

static void air_step(void *user, PhysWorld *pw, real dt);

GpuFluid *gpu_fluid_create(Gpu *g, PhysWorld *pw){
    if (!g || gpu_state(g) != GPU_READY || !pw || !pw->fluid || live_slot() < 0) return NULL;
    Fluid *f = pw->fluid;
    GpuFluid *gf = calloc(1, sizeof(*gf));
    if (!gf) return NULL;
    gf->g = g;
    gf->pw = pw;
    gf->f = f;
    gf->cloud = f->density;
    for (int k = 0; k < NK; k++){
        char *src = kernel_source(k);
        if (!src){ free_all(gf); return NULL; }
        gf->bgl[k] = gpu_layout(g, WGPUShaderStage_Compute, KERNELS[k].types);
        gf->pipe[k] = gpu_compute(g, KERNELS[k].name, src, "main", gf->bgl[k]);
        free(src);
    }
    const size_t n = ncells(f);
    gf->fp = uniform(g, "air parameters", NULL, sizeof(GFP));
    gf->u = storage(g, "air u", nu(f) * 4); gf->v = storage(g, "air v", nv(f) * 4); gf->w = storage(g, "air w", nw(f) * 4);
    gf->tu = storage(g, "air tu", nu(f) * 4); gf->tv = storage(g, "air tv", nv(f) * 4); gf->tw = storage(g, "air tw", nw(f) * 4);
    gf->p = storage(g, "pressure", n * 4);
    gf->div = storage(g, "divergence", n * 4);
    gf->solid = storage(g, "obstacles", n * 4);
    gf->svel = storage(g, "obstacle velocity", n * 16);
    gf->cvel = storage(g, "cell velocity", n * 16);
    gf->omg = storage(g, "vorticity", n * 16);
    gf->frc = storage(g, "vorticity force", n * 16);
    size_t nd = gf->cloud ? (size_t)gf->cloud->nx * gf->cloud->ny * gf->cloud->nz : 1;
    gf->dens[0] = storage(g, "cloud", nd * 4);
    gf->dens[1] = storage(g, "cloud (the other turn)", nd * 4);
    gf->d1 = storage(g, "cloud d1", nd * 4);
    if (gf->cloud){
        if (!(gf->link = calloc(1, sizeof(*gf->link)))){ free_all(gf); return NULL; }
        gf->link->volume = gf->cloud->id;
        gf->link->buf = gf->dens[0];
        gf->link->refs = 1;
    }
    wgpuQueueWriteBuffer(g->queue, gf->u, 0, f->u, nu(f) * 4);
    wgpuQueueWriteBuffer(g->queue, gf->v, 0, f->v, nv(f) * 4);
    wgpuQueueWriteBuffer(g->queue, gf->w, 0, f->w, nw(f) * 4);
    wgpuQueueWriteBuffer(g->queue, gf->p, 0, f->p, n * 4);
    if (gf->cloud) wgpuQueueWriteBuffer(g->queue, gf->dens[0], 0, gf->cloud->density, nd * 4);
    // the pressure levels: the same sizes as the CPU's multigrid
    gf->nlevels = f->poisson.nlevels;
    for (int k = 0; k < gf->nlevels; k++){
        const PoissonLevel *pl = &f->poisson.lv[k];
        GLevel *l = &gf->lv[k];
        size_t ln = (size_t)pl->nx * pl->ny * pl->nz;
        l->nx = pl->nx; l->ny = pl->ny; l->nz = pl->nz;
        l->links = storage(g, "pressure links", ln * 4);
        l->res = storage(g, "pressure residual", ln * 4);
        l->solid = k ? storage(g, "pressure obstacles", ln * 4) : gf->solid;
        l->p = k ? storage(g, "pressure correction", ln * 4) : gf->p;
        l->rhs = k ? storage(g, "pressure rhs", ln * 4) : gf->div;
        level_params(gf, k);
    }
    make_bind_groups(gf);
    reserve_bodies(gf, &pw->rigid);
    gf->timer = gpu_timing_report(g) ? gpu_timer_create(g, 8, NFT) : NULL;   // (only when its times are asked for)
    live[live_slot()] = gf;
    pw->air_step = air_step;
    pw->air_user = gf;
    pw->air_around = NULL;
    pw->nair_around = 0;
    return gf;
}

void gpu_fluid_attach_view(GpuFluid *gf, GpuView *v){
    if (gf && gf->link) gpu_view_volume_on_gpu(v, gf->link);
}

// ---- readbacks ----------------------------------------------------------------------------------

static void on_around(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuFluid *gf = u1;
    if (!is_live(gf)) return;                           // cancelled as it was destroyed: gf is gone
    gf->around_pending = 0;                             // (also after a failure: the next step asks again)
    if (status != WGPUMapAsyncStatus_Success) return;   // nothing is mapped
    const float *a = wgpuBufferGetConstMappedRange(gf->around_rb, 0, (size_t)gf->around_n * 16);
    if (a){
        if (gf->around_cpu_cap < gf->around_n){
            Vec3 *n = realloc(gf->around_cpu, (size_t)gf->around_n * sizeof(Vec3));
            if (n){ gf->around_cpu = n; gf->around_cpu_cap = gf->around_n; }
        }
        if (gf->around_cpu_cap >= gf->around_n){
            for (int i = 0; i < gf->around_n; i++) gf->around_cpu[i] = vec3(a[4 * i], a[4 * i + 1], a[4 * i + 2]);
            gf->pw->air_around = gf->around_cpu;
            gf->pw->nair_around = gf->around_n;
        }
    }
    wgpuBufferUnmap(gf->around_rb);
}

static void on_cloud(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuFluid *gf = u1;
    if (!is_live(gf)) return;
    gf->dens_pending = 0;
    if (status != WGPUMapAsyncStatus_Success) return;
    Volume *v = gf->cloud;
    size_t n = (size_t)v->nx * v->ny * v->nz;
    const float *d = wgpuBufferGetConstMappedRange(gf->dens_rb, 0, n * 4);
    if (d){
        memcpy(v->density, d, n * 4);
        volume_update(v);
    }
    wgpuBufferUnmap(gf->dens_rb);
}

static void map_read(WGPUBuffer b, size_t bytes, WGPUBufferMapCallback cb, void *user){
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = cb;
    mi.userdata1 = user;
    wgpuBufferMapAsync(b, WGPUMapMode_Read, 0, bytes, mi);
}

void gpu_fluid_sync_cloud(GpuFluid *gf){
    if (!gf || !gf->cloud || gf->dens_pending || gf->dens_step == gf->steps) return;
    Gpu *g = gf->g;
    size_t bytes = (size_t)gf->cloud->nx * gf->cloud->ny * gf->cloud->nz * 4;
    if (!gf->dens_rb) gf->dens_rb = gpu_buffer(g, "cloud readback", bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gf->dens[gf->cur], 0, gf->dens_rb, 0, bytes);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    gf->dens_pending = 1;
    gf->dens_step = gf->steps;
    map_read(gf->dens_rb, bytes, on_cloud, gf);
}

// ---- the step -----------------------------------------------------------------------------------

// a compute pass of the step, timed under `stage` when timing is on
static WGPUComputePassEncoder step_pass(GpuFluid *gf, WGPUCommandEncoder enc, int stage){
    WGPUComputePassDescriptor pd;
    memset(&pd, 0, sizeof(pd));
    pd.timestampWrites = gpu_timer_next(gf->timer, stage);
    return wgpuCommandEncoderBeginComputePass(enc, &pd);
}

static void run(WGPUComputePassEncoder pass, GpuFluid *gf, int k, WGPUBindGroup bg, size_t threads){
    wgpuComputePassEncoderSetPipeline(pass, gf->pipe[k]);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    if (k == K_COARSEST_) wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    else gpu_dispatch(pass, threads);
}

static size_t lcells(const GLevel *l){ return (size_t)l->nx * l->ny * l->nz; }

static void vcycle(WGPUComputePassEncoder pass, GpuFluid *gf, int k){
    GLevel *l = &gf->lv[k];
    const Poisson *ps = &gf->f->poisson;
    if (k == gf->nlevels - 1){
        run(pass, gf, K_COARSEST_, l->bg_coarsest, 0);
        return;
    }
    for (int s = 0; s < ps->pre; s++)
        for (int c = 0; c < 2; c++) run(pass, gf, K_SMOOTH_, l->bg_smooth[c], lcells(l));
    run(pass, gf, K_RESID_, l->bg_resid, lcells(l));
    run(pass, gf, K_RESTRICT_, l->bg_restrict, lcells(&gf->lv[k + 1]));
    vcycle(pass, gf, k + 1);
    run(pass, gf, K_PROLONG_, l->bg_prolong, lcells(l));
    for (int s = 0; s < ps->post; s++)
        for (int c = 0; c < 2; c++) run(pass, gf, K_SMOOTH_, l->bg_smooth[c], lcells(l));
}

static void v4(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

static void air_step(void *user, PhysWorld *pw, real dt){
    GpuFluid *gf = user;
    Fluid *f = pw->fluid;
    Gpu *g = gf->g;
    if (dt <= 0 || gpu_state(g) != GPU_READY) return;
    if (gf->inflight >= MAX_INFLIGHT && gf->wait) gpu_wait(g);
    if (gf->inflight >= MAX_INFLIGHT){
        gf->dt_owed = gf->dt_owed + dt > MAX_OWED ? MAX_OWED : gf->dt_owed + dt;
        return;
    }
    dt += gf->dt_owed;
    gf->dt_owed = 0;
    const int nb = pw->rigid.nbodies;

    // the bodies
    reserve_bodies(gf, &pw->rigid);
    if (nb > 0){
        float *bd = realloc(gf->body_data, (size_t)nb * 96);
        if (!bd) return;
        gf->body_data = bd;
        for (int i = 0; i < nb; i++){
            const Body *b = &pw->rigid.bodies[i];
            float *o = bd + 24 * i;
            Mat3 r = quat_to_mat3(b->rot);
            int sphere = b->shape == BODY_SPHERE;
            real bound = sphere ? b->half.x : v_len(b->half);
            real shell = (sphere ? b->half.x : (b->half.x + b->half.y + b->half.z) / 3) * 1.15f + f->h;
            float kind = sphere ? 0.0f : 1.0f;
            for (int k = 0; b->g && k < pw->rigid.ngeoms; k++)
                if (pw->rigid.geoms[k] == b->g) kind = 2.0f + (float)k * (OCC_WORDS / 32);   // in 32-word units
            v4(o, b->pos, kind);
            v4(o + 4, sphere ? vec3(1, 0, 0) : mat3_col(r, 0), (float)b->half.x);
            v4(o + 8, sphere ? vec3(0, 1, 0) : mat3_col(r, 1), (float)b->half.y);
            v4(o + 12, sphere ? vec3(0, 0, 1) : mat3_col(r, 2), (float)b->half.z);
            v4(o + 16, b->vel, (float)bound);
            v4(o + 20, b->angvel, (float)shell);
        }
        wgpuQueueWriteBuffer(g->queue, gf->bodies, 0, bd, (size_t)nb * 96);
    }

    // the parameters
    GFP P;
    memset(&P, 0, sizeof(P));
    P.n[0] = (uint32_t)f->nx; P.n[1] = (uint32_t)f->ny; P.n[2] = (uint32_t)f->nz; P.n[3] = f->open;
    v4(P.lo, f->bounds.lo, (float)f->h);
    P.a[0] = (float)f->inv_h;
    P.a[1] = (float)dt;
    P.a[2] = (float)(f->wind_rate * dt > 1 ? 1 : f->wind_rate * dt);
    P.a[3] = (float)(gf->cloud ? f->buoyancy * dt : 0);
    v4(P.wind, f->wind, (float)(f->vorticity * f->h));
    const Volume *cv = gf->cloud;
    if (cv){
        P.dn[0] = (uint32_t)cv->nx; P.dn[1] = (uint32_t)cv->ny; P.dn[2] = (uint32_t)cv->nz; P.dn[3] = 1;
        v4(P.dlo, cv->bounds.lo, (float)(f->dissipation * dt >= 1 ? 0 : 1 - f->dissipation * dt));
        v4(P.dhi, cv->bounds.hi, (float)f->nsources);
    } else {
        P.dn[0] = P.dn[1] = P.dn[2] = 1;
        v4(P.dlo, vec3(0, 0, 0), 1);
        v4(P.dhi, vec3(1, 1, 1), (float)f->nsources);
    }
    P.m[0] = (uint32_t)nb;
    for (int s = 0; s < f->nsources && s < 8; s++){
        v4(P.src[s], f->sources[s].center, (float)f->sources[s].radius);
        P.rate[s] = (float)f->sources[s].rate;
    }
    wgpuQueueWriteBuffer(g->queue, gf->fp, 0, &P, sizeof(P));
    // the open faces may have changed (the levels' links use them)
    for (int k = 0; k < gf->nlevels; k++) level_params(gf, k);

    const size_t nc = ncells(f), np = npoints(f);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    // 1. obstacles, forces, vorticity, advection (into the other set, the velocity from then on)
    int a = gf->cur;
    WGPUComputePassEncoder pass = step_pass(gf, enc, FT_AIR);
    run(pass, gf, K_MARK_, gf->bg[a][K_MARK_], nc);
    run(pass, gf, K_BOUND_, gf->bg[a][K_BOUND_], np);
    run(pass, gf, K_FORCES_, gf->bg[a][K_FORCES_], np);
    if (f->vorticity > 0){
        run(pass, gf, K_VA_, gf->bg[a][K_VA_], nc);
        run(pass, gf, K_VB_, gf->bg[a][K_VB_], nc);
        run(pass, gf, K_VC_, gf->bg[a][K_VC_], nc);
        run(pass, gf, K_VD_, gf->bg[a][K_VD_], nc);
    }
    run(pass, gf, K_BOUND_, gf->bg[a][K_BOUND_], np);
    run(pass, gf, K_ADV_, gf->bg[a][K_ADV_], np);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    a = gf->cur = 1 - a;
    // 2. projection: divergence, multigrid (the levels' obstacles and links first), gradient
    pass = step_pass(gf, enc, FT_PRESSURE);
    run(pass, gf, K_BOUND_, gf->bg[a][K_BOUND_], np);
    run(pass, gf, K_DIV_, gf->bg[a][K_DIV_], nc);
    for (int k = 0; k < gf->nlevels; k++){
        if (k > 0) run(pass, gf, K_CSOLID_, gf->lv[k].bg_csolid, lcells(&gf->lv[k]));
        run(pass, gf, K_LINKS_, gf->lv[k].bg_links, lcells(&gf->lv[k]));
    }
    for (int c = 0; c < f->pressure_cycles; c++) vcycle(pass, gf, 0);
    run(pass, gf, K_GRAD_, gf->bg[a][K_GRAD_], np);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    // 3. the cloud, from the other set's buffer into this one's
    if (cv){
        size_t nd = (size_t)cv->nx * cv->ny * cv->nz;
        gf->link->buf = gf->dens[a];
        pass = step_pass(gf, enc, FT_CLOUD);
        run(pass, gf, K_D1_, gf->bg[a][K_D1_], nd);
        run(pass, gf, K_D2_, gf->bg[a][K_D2_], nd);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    // 4. the air around the bodies, for the drag of the next step
    int sample = nb > 0 && !gf->around_pending;
    if (sample){
        pass = step_pass(gf, enc, FT_AROUND);
        run(pass, gf, K_SAMP_, gf->bg[a][K_SAMP_], (size_t)nb);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        gpu_buffer_reserve(g, &gf->around_rb, &gf->around_rb_cap, (size_t)nb * 16,
                           WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "air around readback");
        wgpuCommandEncoderCopyBufferToBuffer(enc, gf->around, 0, gf->around_rb, 0, (size_t)nb * 16);
    }
    gpu_timer_finish(gf->timer, enc);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    gpu_timer_submitted(gf->timer);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    WGPUQueueWorkDoneCallbackInfo wd;
    memset(&wd, 0, sizeof(wd));
    wd.mode = WGPUCallbackMode_AllowSpontaneous;
    wd.callback = on_done;
    wd.userdata1 = gf;
    gf->inflight++;
    wgpuQueueOnSubmittedWorkDone(g->queue, wd);
    if (sample){
        gf->around_pending = 1;
        gf->around_n = nb;
        map_read(gf->around_rb, (size_t)nb * 16, on_around, gf);
    }

    // as fluid_step: the expansion sources run out
    int k = 0;
    for (int s = 0; s < f->nsources; s++){
        f->sources[s].time_left -= dt;
        if (f->sources[s].time_left > 1e-6f) f->sources[k++] = f->sources[s];
    }
    f->nsources = k;
    gf->steps++;
    if (gf->cloud) gf->cloud->version++;   // the GPU's cloud changed (the CPU's copy did not)
}

int gpu_fluid_pending(const GpuFluid *gf){ return gf ? gf->inflight : 0; }
void gpu_fluid_set_wait(GpuFluid *gf, int wait){
#if !defined(__EMSCRIPTEN__) || defined(LR_WEB_WAIT)      // (a page can wait only if built to: bin/bench's)
    if (gf) gf->wait = wait;
#else
    (void)gf; (void)wait;
#endif
}

double gpu_fluid_gpu_ms(GpuFluid *gf){ return gf ? gpu_timer_ms_per_round(gf->timer) : 0; }
void gpu_fluid_gpu_ms_reset(GpuFluid *gf){ if (gf) gpu_timer_reset(gf->timer); }

void gpu_fluid_destroy(GpuFluid *gf){
    if (!gf) return;
    PhysWorld *pw = gf->pw;
    if (pw->air_user == gf){
        pw->air_step = NULL;
        pw->air_user = NULL;
        pw->air_around = NULL;
        pw->nair_around = 0;
    }
    if (gf->timer){
        char title[160];
        const Volume *cv = gf->cloud;
        snprintf(title, sizeof(title), "the air, per step (%dx%dx%d cells, a cloud of %dx%dx%d voxels)", gf->f->nx, gf->f->ny, gf->f->nz,
                 cv ? cv->nx : 0, cv ? cv->ny : 0, cv ? cv->nz : 0);
        gpu_timer_print(gf->timer, title, FT_NAMES);
        gpu_timer_destroy(gf->timer);
    }
    for (int i = 0; i < MAX_LIVE_FLUID; i++) if (live[i] == gf) live[i] = NULL;
    free_all(gf);
}

#endif   // LR_GPU
