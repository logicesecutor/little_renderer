#include "gpu/rigid_gpu.h"
#include "base/mem.h"

#include <stdlib.h>

#ifndef LR_GPU
// ---- built without webgpu.h: the bodies stay on the CPU ------------------------------------
GpuRigid *gpu_rigid_create(Gpu *g, PhysWorld *pw){ (void)g; (void)pw; return NULL; }
void gpu_rigid_set_wait(GpuRigid *gr, int wait){ (void)gr; (void)wait; }
void gpu_rigid_sync(GpuRigid *gr){ (void)gr; }
void gpu_rigid_destroy(GpuRigid *gr){ (void)gr; }
int gpu_rigid_skipped(const GpuRigid *gr){ (void)gr; return 0; }
int gpu_rigid_pending(const GpuRigid *gr){ (void)gr; return 0; }
double gpu_rigid_gpu_ms(GpuRigid *gr){ (void)gr; return 0; }
void gpu_rigid_gpu_ms_reset(GpuRigid *gr){ (void)gr; }
#else

#include <stdio.h>
#include <string.h>
#include "gpu/gpu_internal.h"
#include "phys/rigid.h"

// ---- the kernels ---------------------------------------------------------------------------
//
// Bodies (struct Body, 8 vec4): pos (w: 0 sphere, 1 box, 2 hull, 3 fixed mesh), rot (quaternion x y z w), vel (w:
// 1 / mass), ang (angular velocity; w: restitution), half (w: friction), iner (1 / inertia in
// the body's frame), f (sleeping, rigid.c's: x = how long it has been still, y = 1 asleep, z =
// its 1 / mass while asleep, when vel.w is 0: to the others it is fixed) and t (x: a hull's or
// mesh's shape, its index in G; yzw: the air's force on it when it fell asleep). Contacts (struct Contact, 10 vec4), as rigid.c's: the
// point (w: depth), the normal from b to a (w: a), ra (w: b, NONE = a wall), rb (w: bias), the
// tangents (w: the normal's and the first tangent's effective mass), j = the accumulated
// impulses (w: the second tangent's mass), imp = this iteration's impulse (w: friction; its sign bit set: a bounce, K_RELAX), la =
// the point in a's frame (w: the depth when found), lb = in b's (a wall: the world point;
// w: the approach speed it is to bounce from, 0 once it has: rigid.c's vn0).
// Contacts are found once per physics step, with a margin (speculative contacts, as rigid.c),
// and followed through the substeps by la and lb (K_PREP).
// cnt[0] counts the contacts, cnt[1 + i] those of body i; off[] (n + 1) are where each body's
// contacts start in L[], as 2 c (it is a) or 2 c + 1 (it is b).
GPU_WGSL(RHEAD, rigid_head);

/* velocities from the forces (gravity, the air's, held for the substeps); the contact counts cleared */
GPU_WGSL(K_VEL, rigid_vel);

/* contacts, in three steps: every pair of bodies by their bounding spheres (n i + j for the
   pair i < j, then the n walls), into two lists: balls and boxes (these tests), hulls and
   meshes (K_CONVEX); then each list, one thread per entry, launched as many as the lists
   hold (indirect dispatch). */
GPU_WGSL(K_NARROW, rigid_narrow);


/* the bodies by where they are (K_BROAD finds each one's neighbours there): three passes */
GPU_WGSL(K_GRID, rigid_grid);
static const char *K_GCOUNT = "const GRID_STEP = 0u;", *K_GSCAN = "const GRID_STEP = 1u;", *K_GPLACE = "const GRID_STEP = 2u;";

/* the end of a step: how long each body has been still, or slow and going nowhere; those
   still or going nowhere long enough fall asleep (unless a body they touch is still moving):
   velocity 0, and vel.w = 0 (to the others a fixed body) with the mass kept in f.z. Two
   passes: each body's state, then the decisions (which read their neighbours') */
GPU_WGSL(K_SLEEP, rigid_sleep);
static const char *K_SLEEP0 = "const SLEEP_STEP = 0u;", *K_SLEEP1 = "const SLEEP_STEP = 1u;";

/* the end of a step: the fast bodies swept along their paths (continuous collision); two
   passes, where each goes, then there */
GPU_WGSL(K_CCD, rigid_ccd);
static const char *K_CCD0 = "const CCD_STEP = 0u;", *K_CCD1 = "const CCD_STEP = 1u;";

/* after the bodies moved in a substep: one more solver pass, the push out of overlaps taken
   back (each contact's target, then K_SOLVE, then the velocities alone back: K_VELS) */
GPU_WGSL(K_RELAX, rigid_relax);

/* the pairs whose bounding spheres meet (and the walls), sorted into the three lists; or
   (WAKE, sleeping, rigid.c's) which sleeping bodies wake up this step: a body faster than the
   wake speed near one, or the air's force on it changed by a quarter of its weight */
GPU_WGSL(K_BROAD, rigid_broad);
static const char *K_PAIRS = "const WAKE = false;", *K_WAKES = "const WAKE = true;";

/* the three lists' lengths as counts of 64-thread workgroups (indirect dispatch: x, 1, 1 at args[0], [4], [8]) */
GPU_WGSL(K_ARGS, rigid_args);

/* the triangles' list's length (after K_MESHQ), at args[12] */
GPU_WGSL(K_ARGS2, rigid_args2);

/* the balls' and boxes' list */
GPU_WGSL(K_SIMPLE, rigid_simple);

/* the hulls' list: one workgroup per pair, its 64 threads sharing the separating axis
   test (the faces of each, the pairs of edges), the best axis found by a reduction, and the
   feature to make contacts from (convex.c's order of preference, so the same choices) */
GPU_WGSL(K_HEAVY, rigid_heavy);

/* what the hull pairs' kernels share: the shapes, the memo, the features and K_HEAVY's queue */
GPU_WGSL(K_MEMO, rigid_memo);

/* before K_HEAVY, one thread per pair: the walls, balls, and the pairs the memo settles; the
   rest queued for K_HEAVY (its workgroup count at args[20], K_ARGS3) */
GPU_WGSL(K_HULLPRE, rigid_hullpre);
GPU_WGSL(K_ARGS3, rigid_args3);

/* then the hull pairs' contacts from the features chosen: one thread per pair (the clipping
   is one thread's work) */
GPU_WGSL(K_HULLC, rigid_hullc);

/* hulls and fixed meshes (phys/convex.c, the same tests): G holds, per shape, two header
   rows (h0: corners, faces, edges, kind 2; or tree nodes, triangles, 0, kind 3. h1: where
   they start), then the corners, the faces (n, d; first loop index, count), the loops (one
   corner index per row), the edges (a, b, fa, fb); a mesh: its tree's nodes (lo, left; hi,
   right; first, count) and triangles (3 corners each, in the leaves' order). Boxes use the
   unit box (P.m.x) scaled by their half sizes. */
GPU_WGSL(K_CONVEX, rigid_convex);


/* body - fixed mesh pairs, in parallel: each pair walks its mesh's tree and lists the
   triangles near the body (one entry per triangle); each entry is tested by its own thread
   and its contacts gathered per pair (at most 64); then each pair keeps 4 of them */
GPU_WGSL(K_MESHQ, rigid_meshq);

GPU_WGSL(K_MESHTRI, rigid_meshtri);

GPU_WGSL(K_MESHFLUSH, rigid_meshflush);


/* where each body's contacts go: off[i] = the contacts of the bodies before i (an exclusive
   prefix sum, P8). One workgroup of 256 threads: each adds up its run of n / 256 bodies,
   the 256 sums are scanned together (Hillis-Steele, 8 rounds), then each thread writes its
   run's offsets from its sum's start. A few thousand bodies are a run of 4-16 each. */
GPU_WGSL(K_SCAN, rigid_scan);

/* each contact into its bodies' lists */
GPU_WGSL(K_FILL, rigid_fill);

/* each body's list in an order of its own (the other body, then the point), not the order
   the contacts were found in: the pairs, the contacts and the list entries all come from
   atomic counters, whose order changes from run to run, and a body adds up its contacts'
   impulses in its list's order (floating point: another order, another sum), and K_PREP
   takes the first match in last step's list. With K_MESHFLUSH's candidates sorted too, the
   same start gives the same bodies, to the bit, run after run (-simfixed) */
GPU_WGSL(K_SORT, rigid_sort);

/* each contact before solving, every substep: where its points are now (la, lb moved with
   their bodies) and so its depth, arms, tangents, effective masses with each body's mass
   split among its contacts, the bias (bounce, push-out, or for a gap how fast it may close),
   and the impulse to start from (warm start), applied first: in the step's first substep
   the same contact's in the step before, then its own of the substep before */
GPU_WGSL(K_PREP, rigid_prep);

/* Each body's contact list, read 8 entries at a time by its 8 threads (a thread per body
   waited on one contact after the other); the 8 results added up by the first thread in the
   list's order (summed in another order, piles came to rest less often: 600 hulls asleep in
   40 s 3 times out of 6, against 6 of 6). LANES(...) declares the workgroup's memory;
   chunks() is how many rounds of 8 the longest list in the workgroup needs. */
GPU_WGSL(K_LANES, rigid_lanes);

/* the warm start: each body's velocities with its contacts' starting impulses (each with the
   whole mass: the shares average back to it), into V (the iterations' velocities) */
GPU_WGSL(K_APPLY, rigid_apply);

/* One iteration, one pass (P10; it was two: every contact's impulse, then every body's
   sum): each body goes through its contacts, works out each one's impulse from the
   velocities before the iteration (Vi, and the impulses so far Ji) and adds it to its own
   velocities (Vo). A contact is worked out by both of its bodies, the same way, from the
   same numbers; its first body keeps the result (Jo, and C.j for the next substep). A
   fixed or sleeping body does only its first-body contacts (their impulses) and keeps its
   velocities; the fixed mesh, never a first body, nothing. 8 threads per body (K_LANES). */
GPU_WGSL(K_SOLVE, rigid_solve);

/* velocities -> positions, orientations (the slow bodies resting on something damped first);
   or (K_VELS) the velocities alone */
GPU_WGSL(K_POS, rigid_pos);
static const char *K_MOVE = "const MOVE = true;", *K_VELS = "const MOVE = false;";

// ---- the C side ----------------------------------------------------------------------------

typedef struct {
    uint32_t n[4];
    float h[4], g[4], lo[4], hi[4], d[4], h2[4];
    uint32_t m[4];
    float q[4], s[4], c[4];
} GRP;

enum { R_GCOUNT, R_GSCAN, R_GPLACE, R_VEL, R_BROAD, R_ARGS, R_SIMPLE, R_HULLPRE, R_ARGS3, R_HEAVY, R_HULLC, R_MESHQ, R_ARGS2, R_MESHTRI, R_MESHFLUSH,
       R_SCAN, R_FILL, R_SORT, R_PREP, R_SOLVE, R_APPLY, R_POS, R_WAKE, R_SLEEP, R_RELAX, R_VELS, R_SLEEP0, R_CCD0, R_CCD1, NR };
static const struct { const char *name, *types; const char **part[4]; } RK[NR] = {
    {"bodies: grid (count)", "URW",  {&K_GCOUNT, &K_GRID}},
    {"bodies: grid (scan)", "URW",   {&K_GSCAN, &K_GRID}},
    {"bodies: grid (place)", "URW",  {&K_GPLACE, &K_GRID}},
    {"bodies: forces",   "UWWRWWW",  {&K_VEL}},
    {"bodies: pairs",    "URWWWRR",  {&K_PAIRS, &K_BROAD}},
    {"bodies: pair counts", "UWW",   {&K_ARGS}},
    {"bodies: contacts", "URWWRW",   {&K_NARROW, &K_SIMPLE}},
    {"bodies: contacts (hulls, the memo)", "URWWRWRWW", {&K_NARROW, &K_CONVEX, &K_MEMO, &K_HULLPRE}},
    {"bodies: hulls to test", "UWW", {&K_ARGS3}},
    {"bodies: contacts (hulls)", "URWWRWRWW", {&K_NARROW, &K_CONVEX, &K_MEMO, &K_HEAVY}},
    {"bodies: contacts (hulls' faces)", "URWWRWRWW", {&K_NARROW, &K_CONVEX, &K_MEMO, &K_HULLC}},
    {"bodies: near triangles", "UR--RWRW", {&K_NARROW, &K_CONVEX, &K_MESHQ}},
    {"bodies: triangle counts", "UWW", {&K_ARGS2}},
    {"bodies: triangles", "UR--RWRRWW", {&K_NARROW, &K_CONVEX, &K_MESHTRI}},
    {"bodies: contacts (meshes)", "URWWRW--RW", {&K_NARROW, &K_CONVEX, &K_MESHFLUSH}},
    {"bodies: lists 1",  "UWWW",     {&K_SCAN}},
    {"bodies: lists 2",  "URWRWW",   {&K_FILL}},
    {"bodies: lists 3",  "URRRW",    {&K_SORT}},
    {"bodies: prepare",  "URWWRRRRW", {&K_PREP}},
    {"bodies: solve",    "URWRRRWRW", {&K_LANES, &K_SOLVE}},
    {"bodies: warm start", "URRRRW", {&K_LANES, &K_APPLY}},
    {"bodies: move",     "UWRRRR",   {&K_MOVE, &K_POS}},
    {"bodies: wake",     "URWWWRR",  {&K_WAKES, &K_BROAD}},
    {"bodies: sleep",    "UWRWRRR",  {&K_SLEEP1, &K_SLEEP}},
    {"bodies: relax",    "URWR",     {&K_RELAX}},
    {"bodies: velocities", "UWRRRR", {&K_VELS, &K_POS}},
    {"bodies: quiet",    "UWRWRRR",  {&K_SLEEP0, &K_SLEEP}},
    {"bodies: swept",    "UWRWR",    {&K_CCD0, &K_CCD}},
    {"bodies: swept (moved)", "UWRWR", {&K_CCD1, &K_CCD}},
};

#define BODY_BYTES 128
#define CONTACT_BYTES 160
#define MAX_INFLIGHT 2

struct GpuRigid {
    Gpu *g;
    PhysWorld *pw;
    WGPUBindGroupLayout bgl[NR];
    WGPUComputePipeline pipe[NR];
    WGPUBuffer par, bodies, forces, cnt, cursor, wake;
    WGPUBuffer pairs, pc, args;                  // the candidate pairs' lists, their counts, dispatch sizes
    WGPUBuffer memo;                             // hull pairs: what decided them last step (K_HEAVY, P3)
    WGPUBuffer feat;                             // hull pairs: the feature to make contacts from (K_HULLC)
    WGPUBuffer grid;                             // the bodies by where they are (K_GRID)
    uint32_t grid_h;                             // its slots (a power of 2)
    int memo_rd;                                 // its table read this step (the other is written)
    WGPUBuffer tris, store, scount;              // body - mesh: the triangles to test, their contacts per pair
    int meshes;                                  // (at full size once the world has a fixed mesh: 2.5 KB a pair)
    size_t cap_pairs;                            // per list
    WGPUBuffer shapes;                           // the hulls' and meshes' shapes (G), and the unit box
    size_t shapes_cap;
    int nshapes;                                 // RigidWorld.geoms in it
    uint32_t unit_box;                           // its shape index in G
    WGPUBuffer contacts[2], lists[2], offs[2];   // this step's and the one before's
    WGPUBuffer vel[2], imps[2];                  // the iterations' velocities and impulses, in turns (K_SOLVE)
    size_t cap_bodies, cap_contacts;             // what the buffers hold
    int nb;                                      // bodies on the GPU
    int prev_nb;                                 // bodies in the last substep (its lists)
    int parity;
    WGPUBindGroup bg[NR][2];
    WGPUBindGroup bg_solve[2][2], bg_pos[2][2];  // by the contacts' parity, and the iteration's turn
    WGPUBindGroup bg_vels[2][2];
    WGPUBuffer start, ccd, anchor;               // where the bodies began the step, where K_CCD puts
                                                 // them; the anchors (K_SLEEP)
    void *zeros;                                 // (an anchor's for bodies sent again)
    int turn;                                    // the iterations' buffers read next (vel[turn], imps[turn])
    float *staging;
    unsigned char *par_cpu;                      // the substeps' parameters of a step, packed (kept)
    size_t par_cpu_cap;
    // the bodies back: one readback at a time
    WGPUBuffer rb;
    size_t rb_cap;
    int rb_pending, rb_n;
    int inflight, wait;
    uint64_t subs[MAX_INFLIGHT];                 // the last steps' submissions (gpu_wait_for)
    unsigned nsub;                               // steps submitted
    WGPUBuffer par_all;                          // every substep's parameters, 256 bytes apart:
    size_t par_all_cap;                          // copied into par before its substep's passes
    GpuTimer *timer;                             // LR_GPU_TIMING=1: GPU time per kernel
    // a world all asleep, which nothing has changed since, is not stepped (P10): it would
    // stay as it is. Steps are numbered; the bodies read back say whether all slept after
    // theirs, and the last change (bodies, forces) the step it went in with
    unsigned step_no, changed_at, rb_req_step, rb_step;
    int rb_valid, rb_all_asleep;
    Vec3 *last_f;                                // the forces of the last step
    int last_n, last_null;
    int skipped;                                 // steps not stepped
};

static const char *const RK_NAMES[NR] = {"grid: count", "grid: scan", "grid: place", "forces", "pairs (bounding spheres)", "pair counts", "contacts: balls, boxes",
    "contacts: hulls, the memo", "hulls to test", "contacts: hulls", "contacts: hulls' faces", "mesh: near triangles", "mesh: triangle counts", "mesh: triangles", "mesh: 4 contacts per pair",
    "contact lists 1", "contact lists 2", "prepare", "solve and apply (an iteration)", "warm start", "move", "wake", "sleep",
    "relax: targets", "relax: velocities", "sleep: quiet", "swept", "swept: moved"};

// (a "work done" callback can arrive after the destroy: the list of the objects alive)
static int is_live(const GpuRigid *gr){ return gpu_live_has(GPU_LIVE_RIGID, gr); }

static void release(WGPUBuffer *b){ if (*b){ wgpuBufferDestroy(*b); wgpuBufferRelease(*b); *b = NULL; } }

static void release_groups(GpuRigid *gr){
    for (int k = 0; k < NR; k++)
        for (int p = 0; p < 2; p++) if (gr->bg[k][p]){ wgpuBindGroupRelease(gr->bg[k][p]); gr->bg[k][p] = NULL; }
    for (int p = 0; p < 2; p++){
        for (int t = 0; t < 2; t++) if (gr->bg_solve[p][t]){ wgpuBindGroupRelease(gr->bg_solve[p][t]); gr->bg_solve[p][t] = NULL; }
        for (int t = 0; t < 2; t++){
            if (gr->bg_pos[p][t]){ wgpuBindGroupRelease(gr->bg_pos[p][t]); gr->bg_pos[p][t] = NULL; }
            if (gr->bg_vels[p][t]){ wgpuBindGroupRelease(gr->bg_vels[p][t]); gr->bg_vels[p][t] = NULL; }
        }
    }
}

static void make_groups(GpuRigid *gr){
    Gpu *g = gr->g;
    release_groups(gr);
    for (int p = 0; p < 2; p++){
        WGPUBuffer C = gr->contacts[p], L = gr->lists[p], O = gr->offs[p];
        WGPUBuffer Cp = gr->contacts[!p], Lp = gr->lists[!p], Op = gr->offs[!p];
#define BG(k, ...) do { WGPUBuffer b_[] = {__VA_ARGS__}; gr->bg[k][p] = gpu_bind(g, gr->bgl[k], b_, (int)(sizeof(b_) / sizeof(b_[0]))); } while (0)
        BG(R_GCOUNT, gr->par, gr->bodies, gr->grid);
        BG(R_GSCAN, gr->par, gr->bodies, gr->grid);
        BG(R_GPLACE, gr->par, gr->bodies, gr->grid);
        BG(R_VEL, gr->par, gr->bodies, gr->cnt, gr->forces, gr->pc, gr->wake, gr->start);
        BG(R_BROAD, gr->par, gr->bodies, gr->pairs, gr->pc, gr->wake, gr->forces, gr->grid);
        BG(R_ARGS, gr->par, gr->pc, gr->args);
        BG(R_SIMPLE, gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc);
        BG(R_HULLPRE, gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc, gr->shapes, gr->memo, gr->feat);
        BG(R_ARGS3, gr->par, gr->pc, gr->args);
        BG(R_HEAVY, gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc, gr->shapes, gr->memo, gr->feat);
        BG(R_HULLC, gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc, gr->shapes, gr->memo, gr->feat);
        { WGPUBuffer b_[] = {gr->par, gr->bodies, NULL, NULL, gr->pairs, gr->pc, gr->shapes, gr->tris};
          gr->bg[R_MESHQ][p] = gpu_bind(g, gr->bgl[R_MESHQ], b_, 8); }
        BG(R_ARGS2, gr->par, gr->pc, gr->args);
        { WGPUBuffer b_[] = {gr->par, gr->bodies, NULL, NULL, gr->pairs, gr->pc, gr->shapes, gr->tris, gr->store, gr->scount};
          gr->bg[R_MESHTRI][p] = gpu_bind(g, gr->bgl[R_MESHTRI], b_, 10); }
        { WGPUBuffer b_[] = {gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc, NULL, NULL, gr->store, gr->scount};
          gr->bg[R_MESHFLUSH][p] = gpu_bind(g, gr->bgl[R_MESHFLUSH], b_, 10); }
        BG(R_SCAN, gr->par, gr->cnt, O, gr->cursor);
        BG(R_FILL, gr->par, C, gr->cnt, O, gr->cursor, L);
        BG(R_SORT, gr->par, gr->bodies, C, O, L);
        BG(R_PREP, gr->par, gr->bodies, C, gr->cnt, O, Cp, Lp, Op, gr->imps[0]);
        BG(R_APPLY, gr->par, gr->bodies, C, O, L, gr->vel[0]);
        for (int t = 0; t < 2; t++){
            WGPUBuffer b_[] = {gr->par, gr->bodies, C, O, L, gr->vel[t], gr->vel[!t], gr->imps[t], gr->imps[!t]};
            gr->bg_solve[p][t] = gpu_bind(g, gr->bgl[R_SOLVE], b_, 9);
        }
        for (int t = 0; t < 2; t++){
            WGPUBuffer b_[] = {gr->par, gr->bodies, gr->vel[t], O, L, C};
            gr->bg_pos[p][t] = gpu_bind(g, gr->bgl[R_POS], b_, 6);
            gr->bg_vels[p][t] = gpu_bind(g, gr->bgl[R_VELS], b_, 6);
        }
        BG(R_WAKE, gr->par, gr->bodies, gr->pairs, gr->pc, gr->wake, gr->forces, gr->grid);
        BG(R_SLEEP, gr->par, gr->bodies, gr->forces, gr->anchor, C, O, L);
        BG(R_SLEEP0, gr->par, gr->bodies, gr->forces, gr->anchor, C, O, L);
        BG(R_RELAX, gr->par, gr->bodies, C, gr->cnt);
        BG(R_CCD0, gr->par, gr->bodies, gr->start, gr->ccd, gr->shapes);
        BG(R_CCD1, gr->par, gr->bodies, gr->start, gr->ccd, gr->shapes);
#undef BG
    }
}

// the buffers sized by the capacity (reserve makes them, again when it grows): one list for
// reserve and gpu_rigid_destroy, so that neither forgets one (six once leaked on destroy)
static void release_sized(GpuRigid *gr){
    WGPUBuffer *all[] = {&gr->bodies, &gr->forces, &gr->cnt, &gr->cursor, &gr->wake, &gr->contacts[0], &gr->contacts[1],
                         &gr->lists[0], &gr->lists[1], &gr->offs[0], &gr->offs[1], &gr->pairs, &gr->pc, &gr->args,
                         &gr->tris, &gr->store, &gr->scount, &gr->memo, &gr->feat, &gr->grid, &gr->vel[0], &gr->vel[1], &gr->imps[0], &gr->imps[1],
                         &gr->start, &gr->ccd, &gr->anchor};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) release(all[i]);
}

// the body - mesh pairs' buffers: a world without a fixed mesh has no such pair, and one entry
static void reserve_mesh_pairs(GpuRigid *gr, int meshes){
    Gpu *g = gr->g;
    size_t pairs = meshes ? gr->cap_pairs : 1;
    release(&gr->tris); release(&gr->store); release(&gr->scount);
    gr->tris = gpu_storage(g, "bodies: triangles to test", 64 * pairs * 8);
    gr->store = gpu_storage(g, "bodies: mesh contacts", 64 * pairs * 32);
    gr->scount = gpu_storage(g, "bodies: mesh contact counts", pairs * 4);
    gr->meshes = meshes;
}

static int has_meshes(const RigidWorld *w){
    for (int i = 0; i < w->ngeoms; i++) if (w->geoms[i]->kind == BODY_MESH) return 1;
    return 0;
}

// room for nb bodies (and their contacts); 1 = the buffers were made again (their bodies lost)
static int reserve(GpuRigid *gr, int nb, int meshes){
    size_t want = (size_t)(nb > 0 ? nb : 1);
    if (want <= gr->cap_bodies && gr->bodies){
        if (meshes && !gr->meshes){ reserve_mesh_pairs(gr, 1); make_groups(gr); }   // the world's first mesh
        return 0;
    }
    size_t cap = 64;
    while (cap < want) cap *= 2;
    Gpu *g = gr->g;
    release_sized(gr);
    gr->cap_bodies = cap;
    gr->cap_contacts = 32 * cap;
    gr->bodies = gpu_storage(g, "bodies", cap * BODY_BYTES);
    gr->forces = gpu_storage(g, "bodies: forces", cap * 32);
    gr->cnt = gpu_storage(g, "bodies: contact counts", (cap + 1) * 4);
    gr->cursor = gpu_storage(g, "bodies: list cursors", cap * 4);
    gr->wake = gpu_storage(g, "bodies: to wake", cap * 4);
    gr->start = gpu_storage(g, "bodies: where the step began", cap * 16);
    gr->ccd = gpu_storage(g, "bodies: where the sweep puts them", cap * 16);
    gr->anchor = gpu_storage(g, "bodies: anchors", cap * 32);       // (zeroed: none)
    free(gr->zeros);
    gr->zeros = calloc(cap, 32);
    if (!gr->zeros) lr_oom("rigid bodies on the GPU");
    { uint32_t *z = calloc(cap, 4); if (z){ wgpuQueueWriteBuffer(g->compute, gr->wake, 0, z, cap * 4); free(z); } }
    gr->cap_pairs = 32 * cap;
    gr->grid_h = 64;
    while (gr->grid_h < 2 * cap) gr->grid_h *= 2;
    gr->grid = gpu_storage(g, "bodies: grid", ((size_t)gr->grid_h + 2 + 4 * cap) * 4);
    gr->pairs = gpu_storage(g, "bodies: candidate pairs", 3 * gr->cap_pairs * 8);
    gr->pc = gpu_storage(g, "bodies: pair counts", 32);
    gr->args = gpu_buffer(g, "bodies: pair dispatches", 96, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst);
    reserve_mesh_pairs(gr, meshes);
    gr->memo = gpu_storage(g, "bodies: hull pairs remembered", 2 * 13 * (2 * gr->cap_pairs) * 4);   // (zeroed: empty)
    gr->memo_rd = 0;
    gr->feat = gpu_storage(g, "bodies: hull pairs' features, K_HEAVY's queue", 2 * gr->cap_pairs * 16);
    for (int p = 0; p < 2; p++){
        gr->contacts[p] = gpu_storage(g, "contacts", gr->cap_contacts * CONTACT_BYTES);
        gr->lists[p] = gpu_storage(g, "contact lists", gr->cap_contacts * 2 * 4);
        gr->offs[p] = gpu_storage(g, "contact list starts", (cap + 1) * 4);
        gr->vel[p] = gpu_storage(g, "bodies: velocities while solving", cap * 32);
        gr->imps[p] = gpu_storage(g, "contacts: impulses while solving", gr->cap_contacts * 16);
    }
    free(gr->staging);
    gr->staging = malloc(cap * BODY_BYTES);
    if (!gr->staging) lr_oom("rigid bodies on the GPU");
    make_groups(gr);
    gr->prev_nb = 0;          // no lists of the last substep: no warm start this time
    return 1;
}

static void put4(float *d, Vec3 v, float w){ d[0] = (float)v.x; d[1] = (float)v.y; d[2] = (float)v.z; d[3] = w; }

// ---- the shapes (G): see K_CONVEX ----
typedef struct { float *f; size_t n, cap; } Rows;   // vec4 rows
static float *row(Rows *r){
    if (r->n >= r->cap){
        size_t cap = r->cap ? r->cap * 2 : 1024;
        float *nf = realloc(r->f, cap * 16);
        if (!nf) return NULL;
        r->f = nf;
        r->cap = cap;
    }
    float *o = r->f + 4 * r->n++;
    memset(o, 0, 16);
    return o;
}
static void urow(Rows *r, uint32_t a, uint32_t b, uint32_t c, uint32_t d){
    float *o = row(r);
    if (!o) return;
    uint32_t u[4] = {a, b, c, d};
    memcpy(o, u, 16);
}
static void put_hull(Rows *r, const Convex *c, size_t hdr){
    uint32_t v0 = (uint32_t)r->n;
    for (int i = 0; i < c->nv; i++){ float *o = row(r); if (o) put4(o, c->v[i], 0); }
    uint32_t f0 = (uint32_t)r->n;
    for (int f = 0; f < c->nf; f++){
        float *o = row(r);
        if (o) put4(o, c->f[f].n, (float)c->f[f].d);
        urow(r, (uint32_t)c->f[f].first, (uint32_t)c->f[f].count, 0, 0);
    }
    uint32_t l0 = (uint32_t)r->n;
    int nl = 0;
    for (int f = 0; f < c->nf; f++) nl += c->f[f].count;
    for (int k = 0; k < nl; k++) urow(r, (uint32_t)c->fv[k], 0, 0, 0);
    uint32_t e0 = (uint32_t)r->n;
    for (int e = 0; e < c->ne; e++) urow(r, (uint32_t)c->e[e].a, (uint32_t)c->e[e].b, (uint32_t)c->e[e].fa, (uint32_t)c->e[e].fb);
    uint32_t h0[4] = {(uint32_t)c->nv, (uint32_t)c->nf, (uint32_t)c->ne, 2}, h1[4] = {v0, f0, l0, e0};
    memcpy(r->f + 4 * hdr, h0, 16);
    memcpy(r->f + 4 * (hdr + 1), h1, 16);
}
static void put_mesh(Rows *r, const MeshTree *t, size_t hdr){
    uint32_t node0 = (uint32_t)r->n;
    for (int i = 0; i < t->nnodes; i++){
        const MeshNode *nd = &t->nodes[i];
        float *o = row(r);
        if (o){ put4(o, nd->box.lo, 0); uint32_t u = (uint32_t)nd->left; memcpy(o + 3, &u, 4); }
        o = row(r);
        if (o){ put4(o, nd->box.hi, 0); uint32_t u = (uint32_t)nd->right; memcpy(o + 3, &u, 4); }
        urow(r, (uint32_t)nd->first, (uint32_t)nd->count, 0, 0);
    }
    uint32_t tri0 = (uint32_t)r->n;
    for (int q = 0; q < t->ntris; q++){              // in the leaves' order
        const Vec3 *c = t->tri + 3 * (size_t)t->order[q];
        for (int k = 0; k < 3; k++){ float *o = row(r); if (o) put4(o, c[k], 0); }
    }
    uint32_t h0[4] = {(uint32_t)t->nnodes, (uint32_t)t->ntris, 0, 3}, h1[4] = {node0, tri0, 0, 0};
    memcpy(r->f + 4 * hdr, h0, 16);
    memcpy(r->f + 4 * (hdr + 1), h1, 16);
}

static void make_groups(GpuRigid *gr);

// every shape of the world (they never change once made) and the unit box, when new ones came
static void upload_shapes(GpuRigid *gr, const RigidWorld *w){
    if (gr->shapes && gr->nshapes == w->ngeoms) return;
    static Convex unit;
    static int made;
    if (!made){ convex_box(vec3(1, 1, 1), &unit); made = 1; }
    Rows r = {NULL, 0, 0};
    int n = w->ngeoms + 1;
    for (int i = 0; i < 2 * n; i++) row(&r);          // the headers
    for (int i = 0; i < w->ngeoms; i++){
        const RigidGeom *g = w->geoms[i];
        if (g->kind == BODY_HULL) put_hull(&r, &g->hull, 2 * (size_t)i);
        else put_mesh(&r, &g->tree, 2 * (size_t)i);
    }
    put_hull(&r, &unit, 2 * (size_t)w->ngeoms);
    gr->unit_box = (uint32_t)w->ngeoms;
    if (!r.f) return;
    if (gpu_buffer_reserve(gr->g, &gr->shapes, &gr->shapes_cap, r.n * 16,
                           WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "bodies: shapes") && gr->bodies) make_groups(gr);
    wgpuQueueWriteBuffer(gr->g->compute, gr->shapes, 0, r.f, r.n * 16);
    free(r.f);
    gr->nshapes = w->ngeoms;
}

static uint32_t shape_index(const RigidWorld *w, const RigidGeom *g){
    for (int k = 0; k < w->ngeoms; k++) if (w->geoms[k] == g) return (uint32_t)k;
    return 0;
}

static void pack_body(const RigidWorld *w, const Body *b, float *o){
    memset(o, 0, BODY_BYTES);
    float kind = b->shape == BODY_SPHERE ? 0.0f : b->shape == BODY_BOX ? 1.0f : b->shape == BODY_HULL ? 2.0f : 3.0f;
    put4(o, b->pos, kind);
    if (b->g){ uint32_t gi = shape_index(w, b->g); memcpy(o + 28, &gi, 4); }
    o[4] = (float)b->rot.x; o[5] = (float)b->rot.y; o[6] = (float)b->rot.z; o[7] = (float)b->rot.w;
    put4(o + 8, b->vel, (float)b->inv_mass);
    put4(o + 12, b->angvel, (float)b->restitution);
    put4(o + 16, b->half, (float)b->friction);
    put4(o + 20, b->inv_inertia, 0);
}

static void on_bodies(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuRigid *gr = u1;
    if (!is_live(gr)) return;                           // cancelled as it was destroyed: gr is gone
    gr->rb_pending = 0;                                 // (also after a failure: the next step asks again)
    if (status != WGPUMapAsyncStatus_Success) return;   // nothing is mapped
    RigidWorld *w = &gr->pw->rigid;
    const float *d = wgpuBufferGetConstMappedRange(gr->rb, 0, (size_t)gr->rb_n * BODY_BYTES + 16);
    if (d){
        for (int i = 0; i < gr->rb_n && i < w->nbodies; i++){
            Body *b = &w->bodies[i];
            if (b->dirty) continue;            // changed on the CPU since: the GPU takes it again
            const float *o = d + 32 * (size_t)i;
            b->pos = vec3(o[0], o[1], o[2]);
            b->rot = quat(o[7], o[4], o[5], o[6]);
            b->vel = vec3(o[8], o[9], o[10]);
            b->angvel = vec3(o[12], o[13], o[14]);
            b->sleep_time = o[24];
            b->sleeping = o[25] > 0.5f;
        }
        // all asleep after that step: no body moves (vel.w is its inverse mass while it is
        // awake, 0 asleep or fixed); a body changed on the CPU since counts as awake
        int awake = 0;
        for (int i = 0; i < gr->rb_n && !awake; i++)
            awake = d[32 * (size_t)i + 11] != 0 || (i < w->nbodies && w->bodies[i].dirty);
        gr->rb_all_asleep = !awake;
        gr->rb_step = gr->rb_req_step;
        gr->rb_valid = 1;
        uint32_t nc;
        memcpy(&nc, d + 32 * (size_t)gr->rb_n, 4);
        w->ncontacts = (int)(nc < gr->cap_contacts ? nc : gr->cap_contacts);
    }
    wgpuBufferUnmap(gr->rb);
}

static void on_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg; (void)u2;
    GpuRigid *gr = u1;
    if (is_live(gr) && gr->inflight > 0) gr->inflight--;
}

static void dispatch1(WGPUComputePassEncoder pass, GpuRigid *gr, int k, size_t threads){
    wgpuComputePassEncoderSetPipeline(pass, gr->pipe[k]);
    WGPUBindGroup bg = k == R_SOLVE ? gr->bg_solve[gr->parity][gr->turn] : k == R_POS ? gr->bg_pos[gr->parity][gr->turn]
                     : k == R_VELS ? gr->bg_vels[gr->parity][gr->turn] : gr->bg[k][gr->parity];
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    if (k == R_SCAN || k == R_ARGS || k == R_ARGS2 || k == R_ARGS3 || k == R_GSCAN) wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    else if (k == R_SIMPLE) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 0);
    else if (k == R_HEAVY) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 80);
    else if (k == R_HULLPRE || k == R_HULLC) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 16);
    else if (k == R_MESHQ) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 64);
    else if (k == R_MESHFLUSH) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 32);
    else if (k == R_MESHTRI) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 48);
    else gpu_dispatch(pass, threads);
}

// kernel k in the substep's pass, or (timed) in a pass of its own
static void dispatch(WGPUCommandEncoder enc, WGPUComputePassEncoder *pass, GpuRigid *gr, int k, size_t threads){
    const WGPUPassTimestampWrites *tw = gpu_timer_next(gr->timer, k);
    if (tw){
        if (*pass){ wgpuComputePassEncoderEnd(*pass); wgpuComputePassEncoderRelease(*pass); *pass = NULL; }
        WGPUComputePassDescriptor d;
        memset(&d, 0, sizeof(d));
        d.timestampWrites = tw;
        WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &d);
        dispatch1(p, gr, k, threads);
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        return;
    }
    if (!*pass) *pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    dispatch1(*pass, gr, k, threads);
}

// as the kernels' body_radius (rigid_head.wgsl)
static real body_radius_of(const Body *b){ return b->shape == BODY_SPHERE ? b->half.x : v_len(b->half); }

// the forces the same as the last step's (and remembered for the next)
static int forces_same(GpuRigid *gr, const Vec3 *forces, int nb){
    int same = gr->last_n == nb && gr->last_null == (forces == NULL) &&
               (!forces || !nb || !memcmp(gr->last_f, forces, (size_t)nb * sizeof(Vec3)));
    if (same) return 1;
    Vec3 *f = forces && nb ? realloc(gr->last_f, (size_t)nb * sizeof(Vec3)) : gr->last_f;
    if (forces && nb && !f){ gr->last_n = -1; return 0; }
    gr->last_f = f;
    if (forces && nb) memcpy(f, forces, (size_t)nb * sizeof(Vec3));
    gr->last_n = nb;
    gr->last_null = forces == NULL;
    return 0;
}

static int rigid_step_gpu(void *user, PhysWorld *pw, real dt, int substeps, const Vec3 *forces){
    GpuRigid *gr = user;
    Gpu *g = gr->g;
    RigidWorld *w = &pw->rigid;
    if (dt <= 0 || gpu_state(g) != GPU_READY) return 0;
    if (gr->wait) gpu_wait(g);
    // MAX_INFLIGHT steps on their way: wait for the oldest one alone (not for the frame being
    // drawn after it, as waiting for the whole queue did: the CPU prepares the next frame
    // meanwhile)
    for (int tries = 0; gr->inflight >= MAX_INFLIGHT && tries < 4 && gpu_state(g) == GPU_READY; tries++){
        unsigned oldest = gr->nsub - (unsigned)gr->inflight;
        if (tries < 3) gpu_wait_for(g, gr->subs[oldest % MAX_INFLIGHT]);
        else gpu_wait(g);
    }
    const int nb = w->nbodies;
    // Everything asleep after a step the bodies came back from, and nothing changed since
    // (no body new or changed on the CPU, the same forces): the step would change nothing,
    // so it is not made. (Sleeping bodies wake on the GPU only near a moving body, or when
    // the force on them changes.)
    int changed = nb != gr->nb || (size_t)nb > gr->cap_bodies || !gr->bodies;
    for (int i = 0; i < nb && !changed; i++) changed = w->bodies[i].dirty;
    if (!forces_same(gr, forces, nb)) changed = 1;
    if (changed) gr->changed_at = gr->step_no;
    else if (w->sleep && gr->rb_valid && gr->rb_all_asleep && gr->rb_step >= gr->changed_at){
        gr->skipped++;
        return 1;
    }
    // the bodies the CPU changed (or all of them, into new buffers), in runs of neighbours:
    // one write per run, not per body; and the forces
    int fresh = reserve(gr, nb, has_meshes(w));
    upload_shapes(gr, w);
    for (int i = 0; i < nb;){
        Body *b = &w->bodies[i];
        if (!fresh && !b->dirty && i < gr->nb){ i++; continue; }
        int first = i;
        for (; i < nb && (fresh || w->bodies[i].dirty || i >= gr->nb); i++){
            Body *bi = &w->bodies[i];
            pack_body(w, bi, gr->staging + (size_t)(i - first) * (BODY_BYTES / 4));
            bi->dirty = 0;
            bi->force = bi->torque = vec3(0, 0, 0);
        }
        wgpuQueueWriteBuffer(g->compute, gr->bodies, (uint64_t)first * BODY_BYTES, gr->staging, (size_t)(i - first) * BODY_BYTES);
        wgpuQueueWriteBuffer(g->compute, gr->anchor, (uint64_t)first * 32, gr->zeros, (size_t)(i - first) * 32);   // (no anchor yet)
    }
    gr->nb = nb;
    if (nb == 0) return 1;
    float *f = gr->staging;
    for (int i = 0; i < nb; i++){
        put4(f + 8 * i, forces ? forces[i] : vec3(0, 0, 0), 0);
        put4(f + 8 * i + 4, vec3(0, 0, 0), 0);
    }
    wgpuQueueWriteBuffer(g->compute, gr->forces, 0, f, (size_t)nb * 32);

    int steps = substeps > 0 ? substeps : 1;
    real h = dt / steps;
    // the substeps' parameters, all written at once: the whole step is one command buffer,
    // each substep copying its own into par first (they used to be a submit each, for the
    // write in between)
    gpu_buffer_reserve(g, &gr->par_all, &gr->par_all_cap, (size_t)steps * 256,
                       WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, "bodies: parameters per substep");
    if (gr->par_cpu_cap < (size_t)steps * 256){
        free(gr->par_cpu);
        gr->par_cpu = malloc((size_t)steps * 256);
        gr->par_cpu_cap = gr->par_cpu ? (size_t)steps * 256 : 0;
    }
    unsigned char *par_cpu = gr->par_cpu;
    if (!par_cpu) return 0;
    memset(par_cpu, 0, (size_t)steps * 256);
    WGPUCommandEncoder enc = gpu_encoder(g, g->compute);
    {   // the memo's table written this step starts empty (its keys)
        uint64_t cap = 2 * (uint64_t)gr->cap_pairs, wr = (uint64_t)(gr->memo_rd ^ 1);
        wgpuCommandEncoderClearBuffer(enc, gr->memo, wr * 13 * cap * 4, cap * 4);
    }
    wgpuCommandEncoderClearBuffer(enc, gr->grid, 0, ((uint64_t)gr->grid_h + 2) * 4);   // (its counts)
    // the grid's cells: three times the radius of the bodies up to twice the average (the
    // bigger ones, and the fixed meshes, are on a list every body looks through)
    real rsum = 0, rbig, rmax = 0;
    int nr = 0;
    for (int i = 0; i < nb; i++) if (w->bodies[i].shape != BODY_MESH){ rsum += body_radius_of(&w->bodies[i]); nr++; }
    rbig = nr ? 2 * rsum / nr : 1;
    for (int i = 0; i < nb; i++){ real r = body_radius_of(&w->bodies[i]); if (r <= rbig && r > rmax) rmax = r; }
    for (int s = 0; s < steps; s++){
        GRP P;
        memset(&P, 0, sizeof(P));
        P.n[0] = (uint32_t)nb;
        P.n[1] = (uint32_t)gr->cap_contacts;
        P.n[2] = (uint32_t)gr->prev_nb;
        P.n[3] = (uint32_t)w->has_room;
        P.h[0] = (float)h; P.h[1] = 0.5f; P.h[2] = 0.2f; P.h[3] = 60.0f;   // rigid.c's CONTACT_SLOP, BAUMGARTE, BOUNCE_THRESHOLD
        put4(P.g, w->gravity, (float)w->linear_damping);
        put4(P.lo, w->room.lo, (float)w->wall_restitution);
        put4(P.hi, w->room.hi, (float)w->wall_friction);
        P.d[0] = (float)w->angular_damping;
        P.d[1] = (float)dt;
        P.d[2] = 0.1f;                           // rigid.c's SPECULATIVE_MIN
        P.d[3] = 300.0f;                         // rigid.c's MAX_PUSH
        P.h2[0] = 5.0f; P.h2[1] = 0.5f;          // rigid.c's SLEEP_SPEED, SLEEP_TIME, WAKE_SPEED
        P.h2[2] = 20.0f;
        P.h2[3] = (float)gr->memo_rd;            // K_HEAVY reads this table of the memo, writes the other
        P.m[0] = gr->unit_box;
        P.m[1] = (uint32_t)w->ngeoms;
        P.m[2] = (uint32_t)gr->cap_pairs;
        P.m[3] = (uint32_t)s;
        P.q[0] = (float)(3 * (rmax > 0 ? rmax : 1)); P.q[1] = (float)rbig; P.q[2] = (float)gr->grid_h;
        P.s[0] = 60.0f; P.s[1] = 2.5f; P.s[2] = 1.5f; P.s[3] = 2.0f;   // rigid.c's SLOW_SPEED, ANCHOR_MOVE, ANCHOR_TIME, SETTLE_DAMPING
        P.c[0] = 0.5f;                           // rigid.c's CCD_FRACTION
        P.c[1] = s == steps - 1 ? 1.0f : 0.0f;
        memcpy(par_cpu + (size_t)s * 256, &P, sizeof(P));
    }
    wgpuQueueWriteBuffer(g->compute, gr->par_all, 0, par_cpu, (size_t)steps * 256);
    for (int s = 0; s < steps; s++){
        wgpuCommandEncoderCopyBufferToBuffer(enc, gr->par_all, (uint64_t)s * 256, gr->par, 0, sizeof(GRP));
        WGPUComputePassEncoder pass = NULL;
        if (s == 0){
            dispatch(enc, &pass, gr, R_GCOUNT, (size_t)nb);
            dispatch(enc, &pass, gr, R_GSCAN, 1);
            dispatch(enc, &pass, gr, R_GPLACE, (size_t)nb);
        }
        if (s == 0 && w->sleep) dispatch(enc, &pass, gr, R_WAKE, 27 * (size_t)nb);
        dispatch(enc, &pass, gr, R_VEL, (size_t)nb);
        if (s == 0){
            // the contacts, once per step (with margins: they are followed through the substeps)
            dispatch(enc, &pass, gr, R_BROAD, 27 * (size_t)nb);
            dispatch(enc, &pass, gr, R_ARGS, 1);
            dispatch(enc, &pass, gr, R_SIMPLE, 0);
            dispatch(enc, &pass, gr, R_HULLPRE, 0);
            dispatch(enc, &pass, gr, R_ARGS3, 1);
            dispatch(enc, &pass, gr, R_HEAVY, 0);
            dispatch(enc, &pass, gr, R_HULLC, 0);
            dispatch(enc, &pass, gr, R_MESHQ, 0);
            dispatch(enc, &pass, gr, R_ARGS2, 1);
            dispatch(enc, &pass, gr, R_MESHTRI, 0);
            dispatch(enc, &pass, gr, R_MESHFLUSH, 0);
            dispatch(enc, &pass, gr, R_SCAN, 1);
            dispatch(enc, &pass, gr, R_FILL, gr->cap_contacts);
            dispatch(enc, &pass, gr, R_SORT, (size_t)nb);
        }
        dispatch(enc, &pass, gr, R_PREP, gr->cap_contacts);
        dispatch(enc, &pass, gr, R_APPLY, 8 * (size_t)nb);              // the warm start, into vel[0]; 8 threads a body
        gr->turn = 0;
        for (int it = 0; it < w->iterations; it++){
            dispatch(enc, &pass, gr, R_SOLVE, 8 * (size_t)nb);          // vel, imps [turn] -> [!turn]; 8 threads a body
            gr->turn ^= 1;
        }
        dispatch(enc, &pass, gr, R_POS, (size_t)nb);                    // vel[turn] back into the bodies, moved
        // relax: the push out of overlaps taken back (rigid.c, step 5)
        dispatch(enc, &pass, gr, R_RELAX, gr->cap_contacts);
        dispatch(enc, &pass, gr, R_SOLVE, 8 * (size_t)nb);
        gr->turn ^= 1;
        dispatch(enc, &pass, gr, R_VELS, (size_t)nb);                   // vel[turn] back, not moved
        if (s == steps - 1){
            // the end of the step: the fast bodies swept, then who falls asleep
            dispatch(enc, &pass, gr, R_CCD0, (size_t)nb);
            dispatch(enc, &pass, gr, R_CCD1, (size_t)nb);
            if (w->sleep){
                dispatch(enc, &pass, gr, R_SLEEP0, (size_t)nb);
                dispatch(enc, &pass, gr, R_SLEEP, (size_t)nb);
            }
        }
        if (pass){ wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); }
    }
    gr->prev_nb = nb;          // (the next step's contacts start from this one's)
    gr->parity ^= 1;
    gr->memo_rd ^= 1;          // (what this step wrote is read by the next)
    // the bodies back to the CPU (if the last readback has landed). Decided once: the last
    // one's callback can come in the middle of this (a spontaneous callback, during the
    // submit), and a map asked for without the copy here would read the copy before, a step
    // older and without the bodies new since (zeros: the next time the buffers grow they
    // were sent back as such, and became NaN)
    const int readback = !gr->rb_pending;
    if (readback){
        size_t bytes = (size_t)nb * BODY_BYTES + 16;
        gpu_buffer_reserve(g, &gr->rb, &gr->rb_cap, bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "bodies back");
        wgpuCommandEncoderCopyBufferToBuffer(enc, gr->bodies, 0, gr->rb, 0, (size_t)nb * BODY_BYTES);
        // the contact count (cnt[0] of the last substep's buffers)
        wgpuCommandEncoderCopyBufferToBuffer(enc, gr->cnt, 0, gr->rb, (size_t)nb * BODY_BYTES, 4);
    }
    gpu_timer_finish(gr->timer, enc);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    gr->subs[gr->nsub % MAX_INFLIGHT] = gpu_submit_to(g, g->compute, cb);
    gr->nsub++;
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    gpu_timer_submitted(gr->timer);
    if (readback){
        gr->rb_pending = 1;
        gr->rb_n = nb;
        gr->rb_req_step = gr->step_no;
        WGPUBufferMapCallbackInfo mi;
        memset(&mi, 0, sizeof(mi));
        mi.mode = WGPUCallbackMode_AllowSpontaneous;
        mi.callback = on_bodies;
        mi.userdata1 = gr;
        wgpuBufferMapAsync(gr->rb, WGPUMapMode_Read, 0, (size_t)nb * BODY_BYTES + 16, mi);
    }
    WGPUQueueWorkDoneCallbackInfo wi;
    memset(&wi, 0, sizeof(wi));
    wi.mode = WGPUCallbackMode_AllowSpontaneous;
    wi.callback = on_done;
    wi.userdata1 = gr;
    gr->inflight++;
    gr->step_no++;
    wgpuQueueOnSubmittedWorkDone(g->compute, wi);
    if (gr->wait) gpu_rigid_sync(gr);
    return 1;
}

GpuRigid *gpu_rigid_create(Gpu *g, PhysWorld *pw){
    if (!g || gpu_state(g) != GPU_READY || !pw) return NULL;
    GpuRigid *gr = calloc(1, sizeof(*gr));
    if (!gr) return NULL;
    if (!gpu_live_add(GPU_LIVE_RIGID, gr)){ free(gr); return NULL; }
    gr->g = g;
    gr->pw = pw;
    for (int k = 0; k < NR; k++){
        const char *parts[6] = {GPU_WGSL_UTIL, RHEAD};
        int np = 2;
        for (int q = 0; q < 4 && RK[k].part[q]; q++) parts[np++] = *RK[k].part[q];
        char *src = gpu_wgsl_join(parts, np);
        if (!src){ gpu_rigid_destroy(gr); return NULL; }
        gr->bgl[k] = gpu_layout(g, WGPUShaderStage_Compute, RK[k].types);
        gr->pipe[k] = gpu_compute(g, RK[k].name, src, "main", gr->bgl[k]);
        free(src);
    }
    gr->par = gpu_buffer(g, "bodies: parameters", sizeof(GRP), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    upload_shapes(gr, &pw->rigid);
    reserve(gr, pw->rigid.nbodies, has_meshes(&pw->rigid));
    for (int i = 0; i < pw->rigid.nbodies; i++) pw->rigid.bodies[i].dirty = 1;
    pw->rigid_step = rigid_step_gpu;
    pw->rigid_user = gr;
    gr->timer = gpu_timing_report(g) ? gpu_timer_create(g, 512, NR) : NULL;   // (only when its times are asked for)
    return gr;
}

void gpu_rigid_set_wait(GpuRigid *gr, int wait){ if (gr) gr->wait = wait; }

int gpu_rigid_skipped(const GpuRigid *gr){ return gr ? gr->skipped : 0; }
int gpu_rigid_pending(const GpuRigid *gr){ return gr && (gr->rb_pending || gr->inflight > 0); }
double gpu_rigid_gpu_ms(GpuRigid *gr){ return gr ? gpu_timer_ms_per_round(gr->timer) : 0; }
void gpu_rigid_gpu_ms_reset(GpuRigid *gr){ if (gr) gpu_timer_reset(gr->timer); }

void gpu_rigid_sync(GpuRigid *gr){
    if (!gr) return;
    for (int k = 0; k < 2000 && gr->rb_pending; k++) gpu_wait(gr->g);
}

void gpu_rigid_destroy(GpuRigid *gr){
    if (!gr) return;
    gpu_rigid_sync(gr);
    if (gr->timer){
        char title[64];
        snprintf(title, sizeof(title), "rigid bodies, per physics step (%d bodies)", gr->nb);
        gpu_timer_print(gr->timer, title, RK_NAMES);
        gpu_timer_destroy(gr->timer);
    }
    gpu_live_remove(GPU_LIVE_RIGID, gr);
    if (gr->pw && gr->pw->rigid_user == gr){
        gr->pw->rigid_step = NULL;
        gr->pw->rigid_user = NULL;
        // back to the CPU's solver: the count shown was the GPU's, the CPU's list holds none of
        // those contacts (it read that many of last step's to warm start from: past its end)
        gr->pw->rigid.ncontacts = 0;
    }
    release_groups(gr);
    release_sized(gr);
    WGPUBuffer *own[] = {&gr->par, &gr->par_all, &gr->shapes, &gr->rb};
    for (size_t i = 0; i < sizeof(own) / sizeof(own[0]); i++) release(own[i]);
    for (int k = 0; k < NR; k++){
        if (gr->pipe[k]) wgpuComputePipelineRelease(gr->pipe[k]);
        if (gr->bgl[k]) wgpuBindGroupLayoutRelease(gr->bgl[k]);
    }
    free(gr->staging);
    free(gr->zeros);
    free(gr->par_cpu);
    free(gr->last_f);
    free(gr);
}

#endif
