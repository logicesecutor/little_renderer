// trace_gpu.c - the path tracer of rt/tracer.c (trace_path_ex), on the GPU.
//
// One compute kernel, one thread per pixel, one sample per thread: the whole path is traced
// in a loop inside the shader (a "megakernel"), and the sample is added to an accumulation
// buffer (sum of the samples, and their count, per pixel) that the view shows. The same light
// transport as the CPU: diffuse / mirror / glass / lamp, next event estimation with multiple
// importance sampling, Russian roulette, and the media (delta tracking for the collisions,
// ratio tracking for the shadow rays, both walking the majorant bricks with a 3D DDA).
//
// The scene is the CPU's, flattened into buffers:
//   nodes   the BVH (rt/bvh.h), 32 bytes per node: box, and children or a range of `aux`
//   aux     the BVH's shape ids (leaves are ranges of it), then the vertex normals
//   prims   spheres and quads, with the precomputed data of scene/scene.h
//   tris    mesh triangles, 48 bytes: corner, two edges, 3 normal indices, material
// What differs from the CPU: 32-bit floats whatever the build, and another random number
// generator (PCG with a 32-bit state: WGSL has no 64-bit integers). So the two images are
// not equal sample by sample; they converge to the same picture.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scene/brdf.h"
#include "scene/env.h"
#include "base/sampler.h"
#include "scene/volume.h"
#include "rt/accel.h"
#include "rt/bvh.h"

GPU_WGSL(TRACE, trace);

// Reprojection: the new image's pixels take the samples of the previous image where their
// surface was (both images' planes, the new one's guide already made by gbuf). The guide's
// point (along the pixel's ray, at the distance of its first diffuse surface: through a
// mirror, the point behind it, which moves as the reflection does) is projected into the old
// camera; of the 4 old pixels around it (bilinear weights), those that saw the same thing
// count: same kind, a normal within ~25 degrees, a distance from the old eye within 2% (25%
// in a medium: its distance is a noisy estimate, and its light smooth) plus how much it changes from that old pixel to
// the next (on a wall seen edgewise it changes a lot within a pixel; the smaller of the two
// sides, so that an edge next to the pixel does not count). A pixel that matches none (it was hidden, or
// off screen) starts with no samples. The count is capped at `history`: older samples weigh
// less, so that a change of light fades in, and the blur of blending again every frame (and
// the streak it drags a bright sample into, along the motion) stays short. (Taking the
// nearest old pixel instead does not blur, but duplicates pixels under subpixel motion: the
// noise lines up in streaks, worse.)
GPU_WGSL(WAVE, trace_wave);

GPU_WGSL(REPROJECT, trace_reproject);

// The image to show or read back: the path tracer's (its samples, or denoised) plus the
// caustics the path tracer left out: the light tracer's (exact, averaged over its passes),
// blended with the photon cache's while the light tracer has few passes (weight P / (P + 8)).
// The cache is read where the camera sees a diffuse surface directly (the 5th plane: the
// point and the transmittance up to it), bilinearly between the cells along the surface.
GPU_WGSL(RESOLVE, trace_resolve);

// the photon cache fades: every sum times a factor
GPU_WGSL(DECAY, trace_decay);

// ---- the BVH builder: a linear BVH (Karras 2012, "Maximizing Parallelism in the
// Construction of BVHs, Octrees, and k-d Trees"), entirely on the GPU ----
// morton:  each item's box centre, as a 30-bit Morton code (10 bits per axis inside the
//          bounds), with the item's index: (code, index), padded with the largest key
// bsort:   a bitonic sort of the keys, one dispatch per (k, j) step
// hier:    internal node i (of n - 1) finds the range of keys it covers and where it splits,
//          from the length of the codes' common prefix (equal codes: of their positions)
// leaves:  leaf j: item keys[j].y's box, its shape id in aux
// up:      the boxes from the leaves up, in passes: a node is done in pass p when both its
//          children were done before (flags: the pass a node was done in + 1; leaves 1);
//          one dispatch's threads do not see each other's writes, the next dispatch does
// The nodes are this path tracer's: internal i at node0 + i, leaf j at node0 + n - 1 + j,
// the root at node0.
GPU_WGSL(BUILD, trace_build);

// ---- the BVH builder, PLOC (Meister & Bittner 2018, "Parallel Locally-Ordered Clustering
// for Bounding Volume Hierarchy Construction"): the leaves in Morton order (morton, bsort,
// leaves above) are the first clusters; then, in rounds until one is left, each cluster
// finds its nearest among the R clusters on each side (the least surface area of the two
// boxes joined; ties: the lower pair of positions), and mutual nearest pairs merge (the
// lower one becomes their node, the higher one leaves the list). The list keeps its order.
// A cluster is a node of the tree being made: internal k (0 .. n - 2) or leaf j (n - 1 + j);
// a merge makes internal (n - 2) - (the merges before it), so the last one is 0, the root.
// The nodes are laid out as they are made: internal k's two children side by side at
// node0 + 1 + 2 k and + 2 k (one cache line for both boxes, as the LBVH's), the root at
// node0; the clusters' boxes meanwhile in pl. Deterministic: no atomics.
// Many clusters: rounds of four dispatches (U.m.x: the round)
//   nn:      the nearest of each cluster
//   mark:    who stays and who merges; their counts summed in each block of 256
//   top:     the blocks' sums (one workgroup), the next round's count (and the sizes of its
//            dispatches, when they are indirect)
//   scatter: the merges write their nodes; the clusters left move, in order, to the other list
// Few clusters (U.m.w or less): small, one workgroup doing every round (barriers between
// the steps): the whole build (U.m.y = 0) or the rest after the rounds (U.m.y = 1).
// pl: [0], [1]: the count of the lists 0 and 1, [2] the list in use, [3] its count | the two
// lists (2 n) | the nearest (n) | the marks' sums in their block (n) | the blocks' sums (2 per
// 256) | the clusters' boxes (8 words each: 2 n - 1)
GPU_WGSL(PLOC, trace_ploc);

#ifdef LR_VK
// ---- hardware ray tracing (our Vulkan layer, webgpu/vk/webgpu_rt.h) ----
// traverse() with ray queries, in place of the BVH's (renamed traverse_bvh, unused): the
// same Found. The top level's instances: the world's triangles (custom data HW_WTRI), its
// spheres and quads as boxes (HW_WPRIM: primitive i is prims[i], hit here by the CPU's
// tests), and the scene's instances (custom data: the instance; a mesh's triangles, its
// first triangle in the SBT offset, which ray queries leave unused; a ball or a box: the
// unit box, hit here in its own space). Triangles are opaque: the hardware keeps the
// nearest; a box is a candidate the loop below tests against the nearest so far.
GPU_WGSL(HW_TRAVERSE, trace_hw_traverse);
#define HW_WTRI 0xffffffu
#define HW_WPRIM 0xfffffeu
#define HW_BINDING 20
#endif

#define RT_EPS_F 2e-3f           // rt/tracer.c, float build: minimum ray distance
#define RT_SHADOW_MARGIN_F 2e-2f // shadow rays stop this far before the lamp
#define CHUNK_TRIS 32768

typedef struct {
    float origin[4], upper_left[4], horizontal[4], vertical[4];
    uint32_t dims[4], opt[4], cnt[4];
    float f[4];
    uint32_t ext[4];
    float lt[4];
    uint32_t ins[4];
    float sky[4];
    uint32_t txl[4];
    uint32_t ad[4];                    // adaptive sampling: the error a tile stops at (float bits), samples first, (AS2), (AS3)
    float mats[MAX_MATS][16];          // albedo, emission, pbr, tex (uint32 bits)
} GTP;
// one copy of the parameters per pass of a submission (each its own sample index), 256-byte
// aligned (WebGPU's minimum offset alignment for uniform bindings)
#define MAX_REPS 16
#define GTP_STRIDE ((sizeof(GTP) + 255) / 256 * 256)

// where an instanced mesh is on the GPU (and what it was made from)
typedef struct {
    unsigned id;                        // SceneMesh.id: which mesh (not its arrays' addresses: a
    int ntris, nnormals;                // later mesh can be given a freed one's)
    int dynamic;
    unsigned version;
    int built;                          // (LR_GPU_BLAS=gpu) its BVH was built on the GPU
    uint32_t node0, ref0, tri0, nrm0;   // its BVH's root, its leaves' shapes, triangles, normals
    uint32_t uv0;                       // its texture coordinates (with textures)
    uint32_t root;                      // the root as the kernel's stack takes it (WIDE | node0 if 4-wide)
} GpuMeshCopy;

// a BVH for the GPU to build (gpu_trace_build): n boxes, into nodes[node0 ..], aux[ref0 ..]
typedef struct {
    int n;
    uint32_t node0, ref0, ref_add;      // a leaf's shape id: its item + ref_add
    size_t box_at, par_at;              // its boxes and parameters, in the arena
    int ploc;                           // built by PLOC (else LBVH)
} GpuBuild;

struct GpuTrace {
    Gpu *g;
    int w, h;
    WGPUBindGroupLayout bgl, rp_bgl;
    WGPUComputePipeline pipe, gbuf_pipe, rp_pipe, adapt_pipe;
    float adapt_thr;                   // gpu_trace_set_adaptive: the error a tile stops at (0: off)
    int adapt_min;                     // ... and the samples each pixel takes first
    int adapt_sky;                     // ... in a tile of sky or lamp only (AS2; 0: as the others)
    WGPUBuffer params, rp_params, blue;   // parameters (MAX_REPS copies), reprojection's, blue noise
    WGPUBuffer acc[2];       // the image's samples, and (once reprojected) the previous image's
    int cur;                 // which one is the image's
    int planes;              // 5 with the guide's planes (denoiser, caustics), 1 if this GPU's buffers are too small
    Camera cam;              // the image's camera
    int has_image;
    // light-traced caustics: the 64-bit sums (ltrace), their passes, and the resolved image
    WGPUBindGroupLayout lt_bgl, rs_bgl;
    WGPUComputePipeline lt_pipe, rs_pipe;
    WGPUBuffer caus, rs_params, final, lt_params;   // (its own parameters: written in the same submit)
    int caus_passes, specular;
    // the photon cache (the viewer): grid, its fading, its light paths' parameters
    WGPUBindGroupLayout dc_bgl;
    // the bind groups of every frame, kept (gpu_bind_cached): the tracer's (0), the light
    // tracer's (1) and the photon cache's (2); the resolve, the reprojection, the cache's fade
    GpuBindCache bc_trace[3], bc_resolve, bc_reproject, bc_decay;
    // the BVH builds' (one per build of a frame, and PLOC's two) and the wavefront kernels':
    // each pass's parameters are chosen by a dynamic offset, so a build or a kernel has one
    // bind group for all its dispatches, kept from frame to frame
    GpuBindCache bc_build[8], bc_ploc[8], bc_wf[6];
    WGPUComputePipeline dc_pipe;
    WGPUBuffer grid, dc_params, pc_params;
    double pc_paths;         // light paths in the grid, faded as the grid is
    float cell;              // its cells' size
    WGPUBuffer nodes, aux, prims, tris;
    // the sky (bindings 16, 17): its radiance and probabilities, its distributions; a 1 x 1
    // pair when the scene has none. sky_id: the Env's id they hold (0: none)
    WGPUTexture sky_tex, skyc_tex;
    WGPUTextureView sky_view, skyc_view;
    unsigned sky_id;
    // the materials' textures (bindings 18, 19: the atlas, its table), and the triangles'
    // texture coordinates in aux (with any texture in the scene)
    GpuTextures gt;
    WGPUBuffer txt;
    uint32_t uvtri_at, uv_at;
    int textured;
    // the kernels compiled with the sky (SKY_ON), made when a scene first has one
    WGPUComputePipeline pipe_sky, wf_shade_sky;
    int pipe_sky_failed, wf_sky_failed;
    size_t nodes_cap, aux_cap, prims_cap, tris_cap;
    uint32_t nnodes, normals_at;   // (nnodes: the world's, in slots)
    int world_wide;
    int wide_ok;             // LR_GPU_WIDE=1: the CPU's 4-wide trees as such (else binary: faster here)
    int tlas_mode;           // the top level: the CPU's (SAH) when it is current, or built here
                             // (LBVH): 2 by the instances' count (tlas_cpu), LR_GPU_TLAS=cpu 1, =gpu 0
    int builder;             // how the GPU builds its BVHs (LR_GPU_BVH=ploc|lbvh): 1 PLOC, 0 LBVH,
                             // -1 PLOC for meshes (better trees), LBVH for the top level (as good there, faster)
    int blas_gpu;            // LR_GPU_BLAS=gpu: every mesh's BVH built on the GPU once (to measure the builders)
    // the layout (see "the scene on the GPU"): instanced meshes, instances, the top level
    GpuMeshCopy *mc;
    int nmc, laid_out;
    int inst_cap, ninst;
    uint32_t tlas_node0, tlas_ref0, world_node0, world_ref0, layout_nodes, layout_aux;
    // the BVH builder (BUILD): pipelines, scratch (keys, flags), this frame's builds and their
    // boxes and parameters (an arena: written once, then the builds run in order)
    WGPUBindGroupLayout bd_bgl;
    WGPUComputePipeline bd_morton, bd_sort, bd_hier, bd_leaves, bd_up;
    WGPUBuffer bd_keys, bd_flags, bd_boxes, bd_par;
    size_t bd_keys_cap, bd_flags_cap, bd_boxes_cap, bd_par_cap;
    // PLOC: layouts without and with the dispatch sizes, pipelines, its lists, the sizes
    WGPUBindGroupLayout pl_bgl, pl_bgl_args;
    WGPUComputePipeline pl_init, pl_nn, pl_mark, pl_top, pl_scatter, pl_small;
    int ploc_small;          // this many clusters or fewer: PLOC in one workgroup (LR_GPU_PLOC_SMALL)
    WGPUBuffer pl_buf, pl_args;
    size_t pl_cap;
    // LR_GPU_BVH_CHECK=1: the builds read back, checked (one tree over all the items) and
    // their cost (bvh_cost) and depth summed, by their order in the frame
    int check, ck_count, ck_pending, ck_mapped, ck_n[8], ck_depth[8];
    uint32_t ck_node0[8];
    size_t ck_at[8], ck_cap;
    WGPUBuffer ck_buf;
    double ck_cost[8];
    long ck_builds[8], ck_bad;
    GpuBuild builds[8];
    int nbuild;
    unsigned char *arena_boxes, *arena_par;
    size_t arena_nboxes, arena_npar, arena_boxes_cap, arena_par_cap;
    // what is on the GPU now
    Primitive *prim_copy;
    int nprim_copy;
    // the instances as they were uploaded (upload_instances): nothing is packed or sent again
    // while the scene's are the same (Scene.inst_version); memory for packing them, kept
    int inst_valid, inst_count, inst_prim0, inst_tl;
    unsigned inst_version, insts_id;
    void *sc_rec, *sc_box, *sc_nd, *sc_ax, *sc_hw;
    size_t sc_rec_cap, sc_box_cap, sc_nd_cap, sc_ax_cap, sc_hw_cap;
    unsigned mesh_version;   // the world's triangles and normals it holds: Scene.tris_version
    int mesh_ntris, mesh_nnormals;
    int valid;
    Bvh own;                 // built here when the scene has no BVH of its own
    int own_built;
    GpuTimer *timer;         // LR_GPU_TIMING=1 (natively): the GPU time of the passes, by stage
    // the wavefront tracer (WAVE; gpu_trace_set_wavefront): its kernels and their layouts,
    // the paths' state (WF), the counters (WC), the indirect dispatches' sizes (WA), and the
    // parameters of each wave (a band of more paths than WF holds goes in several waves)
    int wavefront, wf_made, wf_failed;
    WGPUBindGroupLayout wf_bgl[6];
    WGPUComputePipeline wf_pipe[6];
    WGPUBuffer wf, wfc, wfa, wf_params;
    int wf_rows;             // rows per wave
    const char *types_wgsl, *media_wgsl;
#ifdef LR_VK
    // hardware ray tracing (g->rt; LR_GPU_HWRT=0 natively: the BVH, as everywhere else): the
    // bottom levels of the world's triangles, its spheres and quads (boxes), each mesh, the
    // unit box (balls and boxes); the top level over them and the instances, built every
    // frame from records written by upload_instances (HW_TRAVERSE)
    int hw;
    WGPUBuffer hw_vtx, hw_boxes, hw_rec, hw_unit_buf;   // vertices (3 per triangle, as `tris`), boxes, records
    size_t hw_vtx_cap, hw_boxes_cap, hw_rec_cap;
    LRBlas hw_tri, hw_prim, hw_unit, *hw_mesh;
    unsigned char *hw_mesh_dirty;
    int hw_nmesh, hw_tri_dirty, hw_prim_dirty, hw_unit_dirty;
    LRTlas hw_tlas;
    uint32_t hw_tlas_cap, hw_ninst;
    int hw_tlas_dirty;       // the top level's records are newer than it is
    int hw_ident;            // record base + i is instance i's (every instance has its bottom level)
    int hw_tlas_update;      // ... and only some moved: the top level updated, not built (hw_build)
    int hw_updates;          // updates since its last build
#endif
};
// (the guide: an image's first hits for the denoiser and the reprojection, once per image;
// timed apart, or every band that begins an image counts it as path tracing: while things
// move, a pass seemed 2.7x cheaper than it was, and the bands were sized from that)
enum { TS_TRACE, TS_LIGHT, TS_BUILD, TS_GUIDE, TS_WF0, NTS = TS_WF0 + 6 };
static const char *const TS_NAMES[NTS] = {"path tracer", "light tracer (caustics)", "BVH builds", "guide",
    "  wavefront: camera rays", "  wavefront: extend", "  wavefront: shade", "  wavefront: sizes", "  wavefront: shadow rays",
    "  wavefront: into the image"};

// a compute pass, timed under `stage` when timing is on; `units`: the work it does
static WGPUComputePassEncoder timed_pass(GpuTrace *t, WGPUCommandEncoder enc, int stage, double units){
    WGPUComputePassDescriptor pd;
    memset(&pd, 0, sizeof(pd));
    pd.timestampWrites = gpu_timer_next(t->timer, stage);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pd);
    if (pd.timestampWrites) gpu_timer_units(t->timer, units);
    return pass;
}

void gpu_trace_before_submit(GpuTrace *t, WGPUCommandEncoder enc){ if (t) gpu_timer_finish(t->timer, enc); }
static void check_builds(GpuTrace *t);
void gpu_trace_after_submit(GpuTrace *t){
    if (!t) return;
    gpu_timer_submitted(t->timer);
    check_builds(t);
}

// the path tracer's module: types, media, the BSDF, TRACE (with the sky's code or without),
// and the wavefront's kernels (wave); free() it
static char *trace_source(GpuTrace *t, int sky, int wave){
    const char *parts[] = {sky ? "const SKY_ON: bool = true;\n" : "const SKY_ON: bool = false;\n",
                           t->types_wgsl, "\n", t->media_wgsl, "\n", GPU_WGSL_BSDF, "\n", GPU_WGSL_TEX, "\n", TRACE, "\n",
                           wave ? WAVE : ""};
    const int n = (int)(sizeof(parts) / sizeof(parts[0]));
    size_t len = 1;
    for (int i = 0; i < n; i++) len += strlen(parts[i]);
#ifdef LR_VK
    if (t->hw) len += strlen(HW_TRAVERSE) + 8;
#endif
    char *src = malloc(len);
    if (!src) return NULL;
    src[0] = 0;
    for (int i = 0; i < n; i++) strcat(src, parts[i]);
#ifdef LR_VK
    if (t->hw){
        char *f = strstr(src, "fn traverse(");
        if (f){
            memmove(f + 15, f + 11, strlen(f + 11) + 1);   // (room made by the 4 bytes of len above)
            memcpy(f + 11, "_bvh", 4);
        }
        strcat(src, "\n");
        strcat(src, HW_TRAVERSE);
    }
#endif
    return src;
}

// a layout's types (gpu_layout) with the acceleration structure at HW_BINDING when the
// kernels trace with ray queries and use the scene (binding 5, the BVH, is there)
static const char *layout_types(const GpuTrace *t, const char *types, char *buf, size_t n){
#ifdef LR_VK
    size_t k = strlen(types);
    if (t->hw && k > 5 && types[5] == 'R' && n > HW_BINDING + 1){
        memset(buf, '-', HW_BINDING);
        memcpy(buf, types, k < HW_BINDING ? k : HW_BINDING);
        buf[HW_BINDING] = 'A';
        buf[HW_BINDING + 1] = 0;
        return buf;
    }
#else
    (void)t; (void)buf; (void)n;
#endif
    return types;
}

#ifdef LR_VK
// the bind group entry of the top level (HW_BINDING), chained to `te`
static void hw_entry(GpuTrace *t, WGPUBindGroupEntry *e, LRBindGroupEntryTlas *te){
    memset(te, 0, sizeof(*te));
    te->chain.sType = LRSType_BindGroupEntryTlas;
    te->tlas = t->hw_tlas;
    memset(e, 0, sizeof(*e));
    e->binding = HW_BINDING;
    e->nextInChain = &te->chain;
}
#endif

GpuTrace *gpu_trace_create(Gpu *g, int width, int height, const char *types_wgsl, const char *media_wgsl){
    if (!g || g->state != GPU_READY) return NULL;
    GpuTrace *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->g = g;
    t->w = width;
    t->h = height;
    t->types_wgsl = types_wgsl;
    t->media_wgsl = media_wgsl;
#ifdef LR_VK
    t->hw = g->rt;
#endif
    char *src = trace_source(t, 0, 0);
    char lt[32];
    if (!src){ free(t); return NULL; }
    // frame, densities, bricks, media boxes, parameters, nodes, aux, prims, tris, accumulation,
    // blue noise
    t->bgl = gpu_layout(g, WGPUShaderStage_Compute, layout_types(t, "URRRDRRRRWU-----TTTU", lt, sizeof(lt)));
    t->timer = gpu_timer_create(g, 1024, NTS);
    const char *wide_env = getenv("LR_GPU_WIDE");
    t->wide_ok = wide_env && atoi(wide_env) > 0;
    const char *tlas_env = getenv("LR_GPU_TLAS");
    t->tlas_mode = !tlas_env ? 2 : !strcmp(tlas_env, "gpu") || !strcmp(tlas_env, "lbvh") ? 0 : !strcmp(tlas_env, "cpu") ? 1 : 2;
    const char *bvh_env = getenv("LR_GPU_BVH");
    t->builder = !bvh_env ? -1 : !strcmp(bvh_env, "lbvh") ? 0 : !strcmp(bvh_env, "ploc") ? 1 : -1;
    const char *blas_env = getenv("LR_GPU_BLAS");
    t->blas_gpu = blas_env && !strcmp(blas_env, "gpu");
    const char *check_env = getenv("LR_GPU_BVH_CHECK");
    t->check = check_env && atoi(check_env) > 0;
    t->pipe = gpu_compute(g, "path tracer", src, "main", t->bgl);
    t->planes = (uint64_t)width * height * 80 <= g->max_storage ? 5 : 1;
    if (t->planes == 5){
        t->gbuf_pipe = gpu_compute(g, "path tracer: guide", src, "gbuf", t->bgl);
        t->adapt_pipe = gpu_compute(g, "path tracer: adaptive sampling", src, "adapt", t->bgl);
        t->rp_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URW");
        t->rp_pipe = gpu_compute(g, "path tracer: reprojection", REPROJECT, "main", t->rp_bgl);
        t->rp_params = gpu_buffer(g, "reprojection parameters", 9 * 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    }
    // the BVH builder
    t->bd_bgl = gpu_layout(g, WGPUShaderStage_Compute, "DRWWWW");   // (D: a pass's parameters, at its offset)
    t->bd_morton = gpu_compute(g, "BVH builder: morton", BUILD, "morton", t->bd_bgl);
    t->bd_sort = gpu_compute(g, "BVH builder: sort", BUILD, "bsort", t->bd_bgl);
    t->bd_hier = gpu_compute(g, "BVH builder: hierarchy", BUILD, "hier", t->bd_bgl);
    t->bd_leaves = gpu_compute(g, "BVH builder: leaves", BUILD, "leaves", t->bd_bgl);
    t->bd_up = gpu_compute(g, "BVH builder: boxes", BUILD, "up", t->bd_bgl);
    t->pl_bgl = gpu_layout(g, WGPUShaderStage_Compute, "DWW");
    t->pl_bgl_args = gpu_layout(g, WGPUShaderStage_Compute, "DWWW");
    t->pl_init = gpu_compute(g, "BVH builder (PLOC): init", PLOC, "init", t->pl_bgl_args);
    t->pl_nn = gpu_compute(g, "BVH builder (PLOC): nearest", PLOC, "nn", t->pl_bgl);
    t->pl_mark = gpu_compute(g, "BVH builder (PLOC): mark", PLOC, "mark", t->pl_bgl);
    t->pl_top = gpu_compute(g, "BVH builder (PLOC): sums", PLOC, "top", t->pl_bgl_args);
    t->pl_scatter = gpu_compute(g, "BVH builder (PLOC): merge", PLOC, "scatter", t->pl_bgl);
    t->pl_small = gpu_compute(g, "BVH builder (PLOC): one workgroup", PLOC, "small", t->pl_bgl);
    const char *small_env = getenv("LR_GPU_PLOC_SMALL");
    t->ploc_small = small_env && atoi(small_env) > 0 ? atoi(small_env) : 1024;
    t->pl_args = gpu_buffer(g, "BVH builder (PLOC): dispatch sizes", 32, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect);
    // the light tracer: the same bindings, its sums (11) instead of the samples (9)
    t->lt_bgl = gpu_layout(g, WGPUShaderStage_Compute, layout_types(t, "URRRDRRRR-UW------TU", lt, sizeof(lt)));
    t->lt_pipe = gpu_compute(g, "light tracer", src, "ltrace", t->lt_bgl);
    t->rs_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRWRR");
    t->dc_bgl = gpu_layout(g, WGPUShaderStage_Compute, "UW");
    t->dc_pipe = gpu_compute(g, "photon cache: fade", DECAY, "main", t->dc_bgl);
    t->dc_params = gpu_buffer(g, "photon cache: fade parameters", 32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->pc_params = gpu_buffer(g, "photon cache: light paths' parameters", GTP_STRIDE * MAX_REPS,
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->rs_pipe = gpu_compute(g, "caustics: resolve", RESOLVE, "main", t->rs_bgl);
    t->rs_params = gpu_buffer(g, "caustics: resolve parameters", 32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->lt_params = gpu_buffer(g, "light tracer parameters", GTP_STRIDE * MAX_REPS, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->caus = gpu_buffer(g, "caustics", (size_t)width * height * 24, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    t->final = gpu_buffer(g, "path traced image with caustics", (size_t)width * height * 16,
                          WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    free(src);
    t->params = gpu_buffer(g, "path tracer parameters", GTP_STRIDE * MAX_REPS, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    {
        // the blue-noise mask (base/sampler.c), 2 ranks per 32 bits, then the BSDF's tables
        const uint16_t *m = blue_noise_mask();
        uint32_t packed[BLUE_NOISE_SIZE * BLUE_NOISE_SIZE / 2];
        for (int i = 0; i < BLUE_NOISE_SIZE * BLUE_NOISE_SIZE / 2; i++) packed[i] = (uint32_t)m[2 * i] | ((uint32_t)m[2 * i + 1] << 16);
        float tab[GPU_BSDF_TABLE_FLOATS];
        gpu_bsdf_tables(tab);
        t->blue = gpu_buffer(g, "blue noise and BSDF tables", sizeof(packed) + sizeof(tab), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        wgpuQueueWriteBuffer(g->queue, t->blue, 0, packed, sizeof(packed));
        wgpuQueueWriteBuffer(g->queue, t->blue, sizeof(packed), tab, sizeof(tab));
    }
    t->acc[0] = gpu_buffer(g, "path tracer samples", (size_t)width * height * 16 * t->planes,
                           WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    return t;
}

WGPUBuffer gpu_trace_accum(GpuTrace *t){ return t->acc[t->cur]; }
int gpu_trace_guided(const GpuTrace *t){ return t->planes == 5; }

static void put(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

// ---- the BVH builder's C side --------------------------------------------------------------
typedef struct { uint32_t n[4], m[4]; float lo[4], ext[4]; } BuildParams;
#define BP_STRIDE 256

// memory kept from one frame to the next: at least `bytes` at *p (NULL: none to be had)
static void *scratch(void **p, size_t *cap, size_t bytes){
    if (*cap >= bytes && *p) return *p;
    size_t want = bytes + bytes / 4 + 256;
    free(*p);
    *p = malloc(want);
    *cap = *p ? want : 0;
    return *p;
}

static uint32_t pow2_at_least(uint32_t n){ uint32_t p = 2; while (p < n) p <<= 1; return p; }
static int up_passes(int n){ int k = 0; while ((1 << k) < n) k++; return n > 1 ? 34 + k : 0; }
// PLOC's rounds of four dispatches from n clusters down to `small` (then one workgroup does
// the rest): a round leaves about 5/6 of them (measured: 0.71-0.83), two more to be safe
static int ploc_rounds(int n, int small){ int r = 0; for (int64_t m = n; m > small; m = m * 5 / 6) r++; return r ? r + 2 : 0; }
#define PLOC_DIRECT_MAX 262144   // up to this many items the rounds' dispatches are sized by n,
                                 // else by the round's count (indirect: slower per dispatch)
static size_t ploc_words(int n){ return 16 + 4 * (size_t)n + 2 * (((size_t)n + 255) / 256) + 8 * (2 * (size_t)n - 1); }
static int sort_passes(uint32_t p){ int c = 0; for (uint32_t k = 2; k <= p; k <<= 1) for (uint32_t j = k >> 1; j > 0; j >>= 1) c++; return c; }

static void *arena_put(unsigned char **buf, size_t *len, size_t *cap, const void *data, size_t bytes){
    size_t at = (*len + 255) / 256 * 256;
    if (at + bytes > *cap){
        size_t want = (at + bytes) * 2;
        unsigned char *nb = realloc(*buf, want);
        if (!nb) return NULL;
        *buf = nb;
        *cap = want;
    }
    if (data) memcpy(*buf + at, data, bytes);
    *len = at + bytes;
    return *buf + at;
}

// a BVH over n boxes (2 x vec4 each: lo, hi) to build on the GPU at the next gpu_trace_build
// (`mesh`: a mesh's triangles, where the tree's quality pays: PLOC unless told otherwise)
static void build_queue(GpuTrace *t, const float *boxes, int n, uint32_t node0, uint32_t ref0, uint32_t ref_add, Aabb bounds, int mesh){
    if (n <= 0 || t->nbuild >= (int)(sizeof(t->builds) / sizeof(t->builds[0]))) return;
    if (t->nbuild == 0) t->arena_nboxes = t->arena_npar = 0;
    GpuBuild *b = &t->builds[t->nbuild];
    b->n = n; b->node0 = node0; b->ref0 = ref0; b->ref_add = ref_add;
    unsigned char *at = arena_put(&t->arena_boxes, &t->arena_nboxes, &t->arena_boxes_cap, boxes, (size_t)n * 32);
    if (!at) return;
    b->box_at = (size_t)(at - t->arena_boxes);
    uint32_t p = pow2_at_least((uint32_t)n);
    BuildParams base;
    memset(&base, 0, sizeof(base));
    base.n[0] = (uint32_t)n; base.n[1] = p; base.n[2] = node0; base.n[3] = ref0;
    base.m[2] = ref_add;
    Vec3 ext = v_sub(bounds.hi, bounds.lo);
    put(base.lo, bounds.lo, 0);
    base.ext[0] = ext.x > 0 ? (float)(1.0 / ext.x) : 0.0f;
    base.ext[1] = ext.y > 0 ? (float)(1.0 / ext.y) : 0.0f;
    base.ext[2] = ext.z > 0 ? (float)(1.0 / ext.z) : 0.0f;
    // parameters: the base one, one per sort step, then one per pass up (LBVH) or (PLOC, when
    // more than `small` items) one per round (m.x) and one for the rest (m.y = 1)
    int ploc = t->builder == 1 || (t->builder < 0 && mesh);
    b->ploc = ploc && n > 2 ? (n <= t->ploc_small ? 2 : 1) : 0;
    if (b->ploc) base.m[3] = (uint32_t)t->ploc_small;
    int nlast = b->ploc == 2 ? 0 : b->ploc ? ploc_rounds(n, t->ploc_small) + 1 : up_passes(n);
    int npar = 1 + sort_passes(p) + nlast;
    size_t first = (t->arena_npar + 255) / 256 * 256;
    unsigned char *par = arena_put(&t->arena_par, &t->arena_npar, &t->arena_par_cap, NULL, (size_t)npar * BP_STRIDE);
    if (!par) return;
    memset(par, 0, (size_t)npar * BP_STRIDE);
    int k = 0;
    memcpy(par + (size_t)k++ * BP_STRIDE, &base, sizeof(base));
    for (uint32_t kk = 2; kk <= p; kk <<= 1)
        for (uint32_t j = kk >> 1; j > 0; j >>= 1){
            BuildParams q = base;
            q.m[0] = j; q.m[1] = kk;
            memcpy(par + (size_t)k++ * BP_STRIDE, &q, sizeof(q));
        }
    for (int u = 1; u <= nlast; u++){
        BuildParams q = base;
        if (!b->ploc) q.m[1] = (uint32_t)u;
        else if (u < nlast) q.m[0] = (uint32_t)(u - 1);
        else q.m[1] = 1;
        memcpy(par + (size_t)k++ * BP_STRIDE, &q, sizeof(q));
    }
    b->par_at = first;
    t->nbuild++;
}

static void build_dispatch(GpuTrace *t, WGPUComputePassEncoder pass, WGPUComputePipeline pipe, const GpuBuild *b,
                           int par, size_t threads, size_t keys_bytes, size_t flags_bytes){
    WGPUBindGroupEntry e[6];
    memset(e, 0, sizeof(e));
    WGPUBuffer bufs[6] = {t->bd_par, t->bd_boxes, t->bd_keys, t->nodes, t->aux, t->bd_flags};
    for (int i = 0; i < 6; i++){ e[i].binding = (uint32_t)i; e[i].buffer = bufs[i]; e[i].size = WGPU_WHOLE_SIZE; }
    e[0].size = sizeof(BuildParams);          // (at the dynamic offset: this pass's)
    e[1].offset = b->box_at; e[1].size = (size_t)b->n * 32;
    e[2].size = keys_bytes;
    e[5].size = flags_bytes;
    // one bind group for all the build's passes, the same next frame while the build is (it
    // was one per pass, made every frame: 114 of them for a top level over 1500 instances)
    WGPUBindGroup bg = gpu_bind_cached(t->g, &t->bc_build[b - t->builds], t->bd_bgl, e, 6, NULL);
    uint32_t off = (uint32_t)(b->par_at + (size_t)par * BP_STRIDE);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 1, &off);
    gpu_dispatch(pass, threads);
}

// A PLOC dispatch of build b: the nodes, its lists (and with_args the dispatch sizes), and
// the parameters `par`, by their offset: the build's two bind groups (without and with the
// sizes) are kept, as build_dispatch's
static void ploc_dispatch(GpuTrace *t, WGPUComputePassEncoder pass, WGPUComputePipeline pipe, const GpuBuild *b, int par,
                          int with_args, size_t threads, WGPUBuffer args, uint64_t at){
    WGPUBindGroupEntry e[4];
    memset(e, 0, sizeof(e));
    WGPUBuffer bufs[4] = {t->bd_par, t->nodes, t->pl_buf, t->pl_args};
    for (int i = 0; i < 4; i++){ e[i].binding = (uint32_t)i; e[i].buffer = bufs[i]; e[i].size = WGPU_WHOLE_SIZE; }
    e[0].size = sizeof(BuildParams);
    e[2].size = ploc_words(b->n) * 4;
    WGPUBindGroup bg = gpu_bind_cached(t->g, &t->bc_ploc[b - t->builds], with_args ? t->pl_bgl_args : t->pl_bgl, e,
                                       with_args ? 4 : 3, NULL);
    uint32_t off = (uint32_t)(b->par_at + (size_t)par * BP_STRIDE);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 1, &off);
    if (args){ wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args, at); return; }
    size_t groups = (threads + 255) / 256;
    uint32_t gx = groups < 65535 ? (uint32_t)groups : 65535u;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, (uint32_t)((groups + gx - 1) / gx), 1);
}

// PLOC over a build's leaves (in Morton order: after morton, bsort and leaves)
static void ploc_build(GpuTrace *t, WGPUComputePassEncoder pass, const GpuBuild *b, int par){
    if (b->ploc == 2){                  // in one workgroup
        ploc_dispatch(t, pass, t->pl_small, b, 0, 0, 1, NULL, 0);
        return;
    }
    int rounds = ploc_rounds(b->n, t->ploc_small);
    WGPUBuffer ia = b->n <= PLOC_DIRECT_MAX ? NULL : t->pl_args;
    ploc_dispatch(t, pass, t->pl_init, b, par, 1, (size_t)b->n, NULL, 0);
    for (int r = 0; r < rounds; r++){
        uint64_t at = (uint64_t)(r & 1) * 16;
        ploc_dispatch(t, pass, t->pl_nn, b, par + r, 0, (size_t)b->n, ia, at);
        ploc_dispatch(t, pass, t->pl_mark, b, par + r, 0, (size_t)b->n, ia, at);
        ploc_dispatch(t, pass, t->pl_top, b, par + r, 1, 1, NULL, 0);
        ploc_dispatch(t, pass, t->pl_scatter, b, par + r, 0, (size_t)b->n, ia, at);
    }
    ploc_dispatch(t, pass, t->pl_small, b, par + rounds, 0, 1, NULL, 0);   // the rest, in one workgroup
}

// the BVHs queued since the last call (the top level over the instances, dynamic meshes')
#ifdef LR_VK
static void hw_build(GpuTrace *t, WGPUCommandEncoder enc);
#endif
static void gpu_trace_build(GpuTrace *t, WGPUCommandEncoder enc){
#ifdef LR_VK
    hw_build(t, enc);
#endif
    if (t->nbuild == 0) return;
    Gpu *g = t->g;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    size_t keys_bytes = 8, flags_bytes = 8, pl_bytes = 0;
    for (int i = 0; i < t->nbuild; i++){
        size_t kb = (size_t)pow2_at_least((uint32_t)t->builds[i].n) * 8, fb = (size_t)t->builds[i].n * 8;
        if (kb > keys_bytes) keys_bytes = kb;
        if (fb > flags_bytes) flags_bytes = fb;
        if (t->builds[i].ploc && ploc_words(t->builds[i].n) * 4 > pl_bytes) pl_bytes = ploc_words(t->builds[i].n) * 4;
    }
    if (pl_bytes) gpu_buffer_reserve(g, &t->pl_buf, &t->pl_cap, pl_bytes, st, "BVH builder (PLOC): lists");
    gpu_buffer_reserve(g, &t->bd_keys, &t->bd_keys_cap, keys_bytes, st, "BVH builder: keys");
    gpu_buffer_reserve(g, &t->bd_flags, &t->bd_flags_cap, flags_bytes, st, "BVH builder: flags");
    gpu_buffer_reserve(g, &t->bd_boxes, &t->bd_boxes_cap, t->arena_nboxes, st, "BVH builder: boxes");
    gpu_buffer_reserve(g, &t->bd_par, &t->bd_par_cap, t->arena_npar, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                       "BVH builder: parameters");
    wgpuQueueWriteBuffer(g->queue, t->bd_boxes, 0, t->arena_boxes, (t->arena_nboxes + 3) / 4 * 4);
    wgpuQueueWriteBuffer(g->queue, t->bd_par, 0, t->arena_par, (t->arena_npar + 3) / 4 * 4);
    WGPUComputePassEncoder pass = timed_pass(t, enc, TS_BUILD, 1);
    for (int i = 0; i < t->nbuild; i++){
        const GpuBuild *b = &t->builds[i];
        uint32_t p = pow2_at_least((uint32_t)b->n);
        int par = 0;
        build_dispatch(t, pass, t->bd_morton, b, par++, p, keys_bytes, flags_bytes);
        for (uint32_t kk = 2; kk <= p; kk <<= 1)
            for (uint32_t j = kk >> 1; j > 0; j >>= 1) build_dispatch(t, pass, t->bd_sort, b, par++, p, keys_bytes, flags_bytes);
        if (b->ploc){
            build_dispatch(t, pass, t->bd_leaves, b, 0, (size_t)b->n, keys_bytes, flags_bytes);
            ploc_build(t, pass, b, par);
            continue;
        }
        if (b->n > 1) build_dispatch(t, pass, t->bd_hier, b, 0, (size_t)b->n - 1, keys_bytes, flags_bytes);
        build_dispatch(t, pass, t->bd_leaves, b, 0, (size_t)b->n, keys_bytes, flags_bytes);
        for (int u = 0; u < up_passes(b->n); u++) build_dispatch(t, pass, t->bd_up, b, par++, (size_t)b->n - 1, keys_bytes, flags_bytes);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    if (t->check && !t->ck_pending){
        size_t total = 0;
        for (int i = 0; i < t->nbuild; i++){
            t->ck_at[i] = total;
            t->ck_n[i] = t->builds[i].n;
            t->ck_node0[i] = t->builds[i].node0;
            total += (2 * (size_t)t->builds[i].n - 1) * 32;
        }
        gpu_buffer_reserve(g, &t->ck_buf, &t->ck_cap, total, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "BVH check: readback");
        for (int i = 0; i < t->nbuild; i++)
            wgpuCommandEncoderCopyBufferToBuffer(enc, t->nodes, (uint64_t)t->ck_node0[i] * 32, t->ck_buf, t->ck_at[i],
                                                 (2 * (size_t)t->ck_n[i] - 1) * 32);
        t->ck_count = t->nbuild;
        t->ck_pending = 1;
    }
    t->nbuild = 0;
}

// (LR_GPU_BVH_CHECK) one built tree: n items, nodes as the tracer's (8 words), children
// absolute (from node0); its cost (bvh_cost) and depth; 0 if it is not one tree over them all
static int check_tree(const uint32_t *w, int n, uint32_t node0, double *cost, int *depth){
    size_t nn = 2 * (size_t)n - 1;
    unsigned char *seen = calloc(nn, 1);
    uint32_t *stack = malloc(nn * 2 * sizeof(uint32_t));
    if (!seen || !stack){ free(seen); free(stack); return 0; }
    float f[8];
    double sum = 0, root = 0;
    int leaves = 0, ok = 1, top = 0, deep = 0;
    stack[top++] = 0; stack[top++] = 1;
    seen[0] = 1;
    while (top > 0 && ok){
        uint32_t d = stack[--top], i = stack[--top];
        if ((int)d > deep) deep = (int)d;
        memcpy(f, w + 8 * (size_t)i, 32);
        double ex = (double)f[4] - f[0], ey = (double)f[5] - f[1], ez = (double)f[6] - f[2];
        double area = 2 * (ex * ey + ey * ez + ez * ex);
        if (!(area >= 0)) ok = 0;
        sum += area;
        if (i == 0) root = area;
        uint32_t a = w[8 * (size_t)i + 3], b = w[8 * (size_t)i + 7];
        if (b & 0x80000000u){ leaves++; continue; }
        uint32_t c[2] = {a - node0, b - node0};
        for (int k = 0; k < 2; k++){
            if (c[k] >= nn || seen[c[k]]){ ok = 0; break; }
            seen[c[k]] = 1;
            float g[8];
            memcpy(g, w + 8 * (size_t)c[k], 32);
            for (int x = 0; x < 3; x++)
                if (g[x] < f[x] - 1e-3f * (1 + fabsf(f[x])) || g[4 + x] > f[4 + x] + 1e-3f * (1 + fabsf(f[4 + x]))) ok = 0;
            stack[top++] = c[k]; stack[top++] = d + 1;
        }
    }
    free(seen);
    free(stack);
    *cost = root > 0 ? sum / root : 0;
    *depth = deep;
    return ok && leaves == n;
}

static void on_check(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    ((GpuTrace *)u1)->ck_mapped = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

static void check_builds(GpuTrace *t){
    if (t->ck_pending != 1) return;
    t->ck_pending = 0;
    if (t->ck_count == 0) return;
    size_t total = t->ck_at[t->ck_count - 1] + (2 * (size_t)t->ck_n[t->ck_count - 1] - 1) * 32;
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = on_check;
    mi.userdata1 = t;
    t->ck_mapped = 0;
    wgpuBufferMapAsync(t->ck_buf, WGPUMapMode_Read, 0, total, mi);
    for (int k = 0; k < 4000 && t->ck_mapped == 0; k++) gpu_wait(t->g);
    if (t->ck_mapped != 1) return;
    const uint32_t *w = wgpuBufferGetConstMappedRange(t->ck_buf, 0, total);
    for (int i = 0; w && i < t->ck_count; i++){
        double cost = 0;
        int depth = 0;
        if (!check_tree(w + t->ck_at[i] / 4, t->ck_n[i], t->ck_node0[i], &cost, &depth)){ t->ck_bad++; continue; }
        t->ck_cost[i] += cost;
        t->ck_builds[i]++;
        if (depth > t->ck_depth[i]) t->ck_depth[i] = depth;
    }
    wgpuBufferUnmap(t->ck_buf);
}

// ---- the scene on the GPU ----------------------------------------------------------------
// nodes: each instanced mesh's BVH | the top-level BVH over the instances | the world's BVH
// aux:   the meshes' leaves' shapes | the top level's instances | the vertex normals (the
//        world's, then each mesh's) | the world's leaves' shapes
// tris:  the world's triangles | each mesh's (its normal indices moved past the ones before)
// prims: the world's primitives | 2 records per instance (transforms; mesh BVH root, material,
//        world box)
// The world's BVH comes last, so that a new one (the physics moved a body) is uploaded alone.
// Moving instances only rewrites their records, and the top level is rebuilt on the GPU
// (gpu_trace_build, BUILD). A dynamic mesh (its triangles change) gets its triangles
// rewritten and its BVH rebuilt on the GPU too, in place.

static int meshes_same(const GpuTrace *t, const Scene *s){
    if (t->nmc != s->nmeshes) return 0;
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        const GpuMeshCopy *c = &t->mc[m];
        if (c->id != sm->id || c->ntris != sm->ntris || c->nnormals != sm->nnormals || c->dynamic != sm->dynamic) return 0;
        if (!sm->dynamic && c->version != sm->version) return 0;
    }
    return 1;
}

// all but the world's BVH and primitives stay where they are
static int layout_same(const GpuTrace *t, const Scene *s){
    return t->laid_out && t->mesh_version == s->tris_version && t->mesh_ntris == s->ntris &&
           t->mesh_nnormals == s->nnormals && meshes_same(t, s) && s->ninsts <= t->inst_cap;
}

static int geometry_same(const GpuTrace *t, const Scene *s){
    return t->valid && layout_same(t, s) && t->nprim_copy == s->nprims &&
           (s->nprims == 0 || memcmp(t->prim_copy, s->prims, (size_t)s->nprims * sizeof(Primitive)) == 0);
}

// a BVH's nodes as the GPU's, children and leaves moved by node0 / ref0
#define GPU_WIDE 0x80000000u
// how many 32-byte slots a BVH takes on the GPU: its 4-wide nodes (4 slots each), or its nodes
static uint32_t gpu_slots(const Bvh *b, int wide){ return wide ? 4u * (uint32_t)b->nwide : (uint32_t)b->nnodes; }

// a 4-wide BVH: each node 4 slots, one per child: its box; inner: a = its first slot, b = 0;
// leaf: a = its first shape, b = 0x80000000 | count (a leaf node of the binary layout); empty:
// b = 0xffffffff (and a box that no ray meets)
static void put_wide(uint32_t *nd, const Bvh *b, uint32_t node0, uint32_t ref0){
    for (int i = 0; i < b->nwide; i++){
        const BvhWideNode *n = &b->wide[i];
        for (int k = 0; k < 4; k++){
            uint32_t *o = nd + 8 * (4 * (size_t)i + k);
            float lo[3] = {(float)n->b[0][k], (float)n->b[1][k], (float)n->b[2][k]};
            float hi[3] = {(float)n->b[3][k], (float)n->b[4][k], (float)n->b[5][k]};
            if (n->count[k] < 0){ lo[0] = lo[1] = lo[2] = hi[0] = hi[1] = hi[2] = 1e30f; }
            memcpy(o, lo, 12);
            memcpy(o + 4, hi, 12);
            o[3] = n->count[k] > 0 ? ref0 + (uint32_t)n->child[k] : n->count[k] == 0 ? node0 + 4u * (uint32_t)n->child[k] : 0;
            o[7] = n->count[k] > 0 ? (0x80000000u | (uint32_t)n->count[k]) : n->count[k] == 0 ? 0 : 0xffffffffu;
        }
    }
}

static void put_nodes(uint32_t *nd, const Bvh *b, uint32_t node0, uint32_t ref0){
    for (int i = 0; i < b->nnodes; i++){
        const BvhNode *n = &b->nodes[i];
        float lo[4], hi[4];
        put(lo, n->box.lo, 0);
        put(hi, n->box.hi, 0);
        uint32_t *o = nd + 8 * (size_t)i;
        memcpy(o, lo, 12);
        memcpy(o + 4, hi, 12);
        o[3] = n->count > 0 ? ref0 + (uint32_t)n->start : node0 + (uint32_t)n->left;
        o[7] = n->count > 0 ? (0x80000000u | (uint32_t)n->count) : node0 + (uint32_t)n->right;
    }
}

// triangles as the GPU's (48 bytes), normal indices moved by nrm0
static void put_tris(uint32_t *tr, const Triangle *src, int n, uint32_t nrm0){
    for (int i = 0; i < n; i++){
        const Triangle *q = &src[i];
        uint32_t *o = tr + 12 * (size_t)i;
        float a[4];
        put(a, q->q, 0); memcpy(o, a, 12);
        put(a, q->u, 0); memcpy(o + 4, a, 12);
        put(a, q->v, 0); memcpy(o + 8, a, 12);
        int smooth_n = q->n[0] >= 0;
        o[3] = smooth_n ? nrm0 + (uint32_t)q->n[0] : 0xffffffffu;
        o[7] = smooth_n ? nrm0 + (uint32_t)q->n[1] : 0;
        o[11] = ((uint32_t)(q->material & 15) << 28) | (smooth_n ? ((nrm0 + (uint32_t)q->n[2]) & 0x0fffffffu) : 0);
    }
}

static int upload_tris(GpuTrace *t, const Triangle *src, int n, uint32_t first, uint32_t nrm0){
    uint32_t *tr = malloc((size_t)CHUNK_TRIS * 48);
    if (!tr) return 0;
    for (int f = 0; f < n; f += CHUNK_TRIS){
        int k = n - f < CHUNK_TRIS ? n - f : CHUNK_TRIS;
        put_tris(tr, src + f, k, nrm0);
        wgpuQueueWriteBuffer(t->g->queue, t->tris, (uint64_t)(first + (uint32_t)f) * 48, tr, (size_t)k * 48);
#ifdef LR_VK
        if (t->hw && t->hw_vtx){            // its corners, for the bottom levels (36 bytes: q, q + u, q + v)
            float *v = (float *)tr;
            for (int i = 0; i < k; i++){
                const Triangle *q = &src[f + i];
                Vec3 c[3] = {q->q, v_add(q->q, q->u), v_add(q->q, q->v)};
                for (int j = 0; j < 3; j++){
                    v[9 * i + 3 * j] = (float)c[j].x; v[9 * i + 3 * j + 1] = (float)c[j].y; v[9 * i + 3 * j + 2] = (float)c[j].z;
                }
            }
            wgpuQueueWriteBuffer(t->g->queue, t->hw_vtx, (uint64_t)(first + (uint32_t)f) * 36, v, (size_t)k * 36);
        }
#endif
    }
    free(tr);
    return 1;
}

static int too_big(Gpu *g, size_t bytes){
    if (bytes <= g->max_storage) return 0;
    fprintf(stderr, "GPU: the scene needs a %.0f MB buffer, this GPU allows %.0f MB: the CPU path traces\n",
            (double)bytes / 1048576, (double)g->max_storage / 1048576);
    return 1;
}

// everything but the world's BVH and primitives: the meshes, the room for the instances and
// the top level, the normals, the world's triangles
static int upload_layout(GpuTrace *t, const Scene *s){
    t->inst_valid = 0;               // (the meshes' places are in the instances' records)
    Gpu *g = t->g;
    t->laid_out = 0;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    GpuMeshCopy *mc = realloc(t->mc, (size_t)(s->nmeshes ? s->nmeshes : 1) * sizeof(GpuMeshCopy));
    if (!mc) return 0;
    t->mc = mc;
    uint32_t nodes = 0, refs = 0, tris = (uint32_t)s->ntris, nrm = (uint32_t)s->nnormals, nuv = (uint32_t)s->nuvs;
    t->textured = s->ntexmaps > 0;
    size_t most_nodes = 1, most_refs = 1;
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        const Bvh *mb = scene_mesh_bvh(s, m);
        if (!mb){
            fprintf(stderr, "GPU path tracer: mesh %d has no BVH (scene_set_accel builds them): the CPU path traces\n", m);
            return 0;
        }
        GpuMeshCopy *c = &mc[m];
        c->id = sm->id; c->ntris = sm->ntris; c->nnormals = sm->nnormals;
        c->dynamic = sm->dynamic; c->version = sm->version; c->built = 0;
        c->node0 = nodes; c->ref0 = refs; c->tri0 = tris; c->nrm0 = nrm; c->uv0 = nuv;
        // a dynamic mesh: room for the GPU's BVH of it too (2n - 1 nodes, n leaves of one
        // triangle), binary as the GPU builds it; the others 4-wide when the CPU made them so
        int on_gpu = sm->dynamic || t->blas_gpu;
        int wide = t->wide_ok && mb->wide && !on_gpu;
        uint32_t nn = gpu_slots(mb, wide), nr = (uint32_t)mb->nrefs;
        c->root = wide ? (GPU_WIDE | c->node0) : c->node0;
        if (on_gpu && (uint32_t)(2 * sm->ntris - 1) > nn) nn = (uint32_t)(2 * sm->ntris - 1);
        if (on_gpu && (uint32_t)sm->ntris > nr) nr = (uint32_t)sm->ntris;
        nodes += nn; refs += nr; tris += (uint32_t)sm->ntris; nrm += (uint32_t)sm->nnormals; nuv += (uint32_t)sm->nuvs;
        if ((size_t)gpu_slots(mb, wide) > most_nodes) most_nodes = (size_t)gpu_slots(mb, wide);
        if ((size_t)mb->nrefs > most_refs) most_refs = (size_t)mb->nrefs;
    }
    t->nmc = s->nmeshes;
    // room for more (bodies keep appearing): doubling, so the layout is rarely redone
    t->inst_cap = 0;
    if (s->ninsts > 0){ t->inst_cap = 64; while (t->inst_cap < s->ninsts) t->inst_cap *= 2; }
    t->tlas_node0 = nodes;
    t->tlas_ref0 = refs;
    if (t->inst_cap){ nodes += (uint32_t)(2 * t->inst_cap - 1); refs += (uint32_t)t->inst_cap; }
    t->normals_at = refs;
    t->world_node0 = nodes;
    t->uvtri_at = refs + 3 * nrm;
    t->uv_at = t->uvtri_at + (t->textured ? 3 * tris : 0);
    t->world_ref0 = t->uv_at + (t->textured ? 6 * nuv : 0);
    size_t bytes_tris = (size_t)(tris ? tris : 1) * 48;
    if (too_big(g, bytes_tris)) return 0;
    gpu_buffer_reserve(g, &t->tris, &t->tris_cap, bytes_tris, st, "triangles");
#ifdef LR_VK
    if (t->hw){
        // the corners (made again: the bottom levels read them where they are), every
        // bottom level of triangles to be made again
        gpu_buffer_reserve(g, &t->hw_vtx, &t->hw_vtx_cap, (size_t)(tris ? tris : 1) * 36, st, "triangles' corners");
        for (int m = 0; m < t->hw_nmesh; m++) lrBlasRelease(t->hw_mesh[m]);
        free(t->hw_mesh); free(t->hw_mesh_dirty);
        t->hw_nmesh = s->nmeshes;
        t->hw_mesh = calloc((size_t)(s->nmeshes ? s->nmeshes : 1), sizeof(LRBlas));
        t->hw_mesh_dirty = calloc((size_t)(s->nmeshes ? s->nmeshes : 1), 1);
        if (!t->hw_mesh || !t->hw_mesh_dirty){ t->hw_nmesh = 0; return 0; }
        lrBlasRelease(t->hw_tri);
        t->hw_tri = NULL;
        if (tris > 0xffffffu)                // (a mesh's first triangle goes in 24 bits)
            fprintf(stderr, "GPU path tracer: more than 16M triangles: some meshes will be hit wrongly\n");
    }
#endif
    // (the node and shape buffers: sized with the world's part, by upload_world)
    t->layout_nodes = nodes;
    t->layout_aux = t->world_ref0;
    gpu_buffer_reserve(g, &t->nodes, &t->nodes_cap, ((size_t)nodes + t->nnodes + 1) * 32, st | WGPUBufferUsage_CopySrc, "BVH nodes");
    gpu_buffer_reserve(g, &t->aux, &t->aux_cap, ((size_t)t->world_ref0 + 2 * (size_t)t->nnodes + 1) * 4, st, "BVH shapes and normals");

    uint32_t *nd = malloc(most_nodes * 32);
    uint32_t *ax = malloc(most_refs * 4);
    if (!nd || !ax){ free(nd); free(ax); return 0; }
    for (int m = 0; m < s->nmeshes; m++){
        const Bvh *mb = scene_mesh_bvh(s, m);
        const GpuMeshCopy *c = &mc[m];
        int wide = (c->root & GPU_WIDE) != 0;
        if (wide) put_wide(nd, mb, c->node0, c->ref0); else put_nodes(nd, mb, c->node0, c->ref0);
        wgpuQueueWriteBuffer(g->queue, t->nodes, (uint64_t)c->node0 * 32, nd, (size_t)gpu_slots(mb, wide) * 32);
        for (int i = 0; i < mb->nrefs; i++) ax[i] = c->tri0 + (uint32_t)mb->refs[i];   // + the primitives, when walked
        wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)c->ref0 * 4, ax, (size_t)mb->nrefs * 4);
    }
    free(nd);
    free(ax);
    // the normals, the world's then each mesh's
    size_t most_n = (size_t)s->nnormals + 1;
    for (int m = 0; m < s->nmeshes; m++) if ((size_t)s->meshes[m].nnormals > most_n) most_n = (size_t)s->meshes[m].nnormals;
    float *nf = malloc(most_n * 12);
    if (!nf) return 0;
    for (int i = 0; i < s->nnormals; i++){
        nf[3 * i] = (float)s->normals[i].x; nf[3 * i + 1] = (float)s->normals[i].y; nf[3 * i + 2] = (float)s->normals[i].z;
    }
    if (s->nnormals) wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)t->normals_at * 4, nf, (size_t)s->nnormals * 12);
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        for (int i = 0; i < sm->nnormals; i++){
            nf[3 * i] = (float)sm->normals[i].x; nf[3 * i + 1] = (float)sm->normals[i].y; nf[3 * i + 2] = (float)sm->normals[i].z;
        }
        if (sm->nnormals)
            wgpuQueueWriteBuffer(g->queue, t->aux, ((uint64_t)t->normals_at + 3 * (uint64_t)mc[m].nrm0) * 4, nf, (size_t)sm->nnormals * 12);
    }
    free(nf);
    // the texture coordinates: each triangle's 3 indices (NONE: it has none), then the
    // coordinates and tangents, 6 floats each; the world's, then each mesh's
    if (t->textured){
        size_t most_t = (size_t)s->ntris + 1, most_u = (size_t)s->nuvs + 1;
        for (int m = 0; m < s->nmeshes; m++){
            if ((size_t)s->meshes[m].ntris > most_t) most_t = (size_t)s->meshes[m].ntris;
            if ((size_t)s->meshes[m].nuvs > most_u) most_u = (size_t)s->meshes[m].nuvs;
        }
        uint32_t *ti = malloc(most_t * 12);
        float *uf = malloc(most_u * 24);
        if (!ti || !uf){ free(ti); free(uf); return 0; }
        for (int m = -1; m < s->nmeshes; m++){
            const Triangle *tr = m < 0 ? s->tris : s->meshes[m].tris;
            int nt = m < 0 ? s->ntris : s->meshes[m].ntris, nu = m < 0 ? s->nuvs : s->meshes[m].nuvs;
            const Vec2 *uv = m < 0 ? s->uvs : s->meshes[m].uvs;
            const Vec4 *tg = m < 0 ? s->tangents : s->meshes[m].tangents;
            uint32_t t0 = m < 0 ? 0 : mc[m].tri0, u0 = m < 0 ? 0 : mc[m].uv0;
            for (int i = 0; i < nt; i++){
                int has = tr[i].t[0] >= 0 && tr[i].t[1] >= 0 && tr[i].t[2] >= 0 &&
                          tr[i].t[0] < nu && tr[i].t[1] < nu && tr[i].t[2] < nu;
                for (int k = 0; k < 3; k++) ti[3 * i + k] = has ? u0 + (uint32_t)tr[i].t[k] : 0xffffffffu;
            }
            for (int i = 0; i < nu; i++){
                float *o = uf + 6 * i;
                o[0] = (float)uv[i].x; o[1] = (float)uv[i].y;
                o[2] = (float)tg[i].x; o[3] = (float)tg[i].y; o[4] = (float)tg[i].z; o[5] = (float)tg[i].w;
            }
            if (nt) wgpuQueueWriteBuffer(g->queue, t->aux, ((uint64_t)t->uvtri_at + 3 * (uint64_t)t0) * 4, ti, (size_t)nt * 12);
            if (nu) wgpuQueueWriteBuffer(g->queue, t->aux, ((uint64_t)t->uv_at + 6 * (uint64_t)u0) * 4, uf, (size_t)nu * 24);
        }
        free(ti);
        free(uf);
    }
    // the triangles: the world's (they are many: only now), then each mesh's
    if (!upload_tris(t, s->tris, s->ntris, 0, 0)) return 0;
    for (int m = 0; m < s->nmeshes; m++)
        if (!upload_tris(t, s->meshes[m].tris, s->meshes[m].ntris, mc[m].tri0, mc[m].nrm0)) return 0;
    t->mesh_version = s->tris_version;
    t->mesh_ntris = s->ntris;
    t->mesh_nnormals = s->nnormals;
    t->laid_out = 1;
    return 1;
}

// the world's BVH and primitives (after the layout; a new layout if the buffers had to grow)
static int upload_world(GpuTrace *t, const Scene *s){
    Gpu *g = t->g;
    t->valid = 0;
    t->inst_valid = 0;               // (the instances' records come after the primitives)
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    // the BVH: the scene's own if it has one (it must be up to date: scene_set_accel after
    // the geometry changed), otherwise one is built here
    const Bvh *b = scene_bvh(s);
    if (!b){
        if (t->own_built) bvh_free(&t->own);
        AccelParams ap = accel_default_params();
        bvh_build(&t->own, s, ap.bvh_leaf, ap.bvh_bins, ap.bvh_split);
        t->own_built = 1;
        b = &t->own;
    }
    if (b->depth > 36) fprintf(stderr, "GPU path tracer: the BVH is %d levels deep, its stack holds 40: some hits may be missed\n", b->depth);
    const int wide = t->wide_ok && b->wide != NULL;
    const uint32_t slots = gpu_slots(b, wide);
    size_t bytes_nodes = ((size_t)t->layout_nodes + (size_t)slots + 1) * 32;
    size_t bytes_aux = ((size_t)t->layout_aux + (size_t)b->nrefs + 1) * 4;
    size_t bytes_prims = ((size_t)s->nprims + 2 * (size_t)t->inst_cap + 1) * 96;
    if (too_big(g, bytes_nodes) || too_big(g, bytes_aux)) return 0;
    int grew = gpu_buffer_reserve(g, &t->nodes, &t->nodes_cap, bytes_nodes, st | WGPUBufferUsage_CopySrc, "BVH nodes");
    grew |= gpu_buffer_reserve(g, &t->aux, &t->aux_cap, bytes_aux, st, "BVH shapes and normals");
    gpu_buffer_reserve(g, &t->prims, &t->prims_cap, bytes_prims, st, "primitives and instances");
    if (grew){
        // they grew (a new buffer, empty): everything again. (Told by gpu_buffer_reserve, not
        // by the handle: a new buffer can get the address of the one just released, and the
        // layout, the normals among it, was then never written to it.)
        if (!upload_layout(t, s)) return 0;
        t->inst_valid = 0;
    }
    uint32_t *nd = malloc((size_t)(slots ? slots : 1) * 32);
    uint32_t *ax = malloc((size_t)(b->nrefs ? b->nrefs : 1) * 4);
    if (!nd || !ax){ free(nd); free(ax); return 0; }
    if (wide) put_wide(nd, b, t->world_node0, t->world_ref0); else put_nodes(nd, b, t->world_node0, t->world_ref0);
    if (slots) wgpuQueueWriteBuffer(g->queue, t->nodes, (uint64_t)t->world_node0 * 32, nd, (size_t)slots * 32);
    for (int i = 0; i < b->nrefs; i++) ax[i] = (uint32_t)b->refs[i];
    if (b->nrefs) wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)t->world_ref0 * 4, ax, (size_t)b->nrefs * 4);
    free(nd);
    free(ax);
    t->nnodes = slots;
    t->world_wide = wide;
    // spheres and quads
    float *pr = malloc((size_t)(s->nprims ? s->nprims : 1) * 96);
    if (!pr) return 0;
    for (int i = 0; i < s->nprims; i++){
        const Primitive *p = &s->prims[i];
        float *o = pr + 24 * (size_t)i;
        put(o, p->center, (float)p->radius);
        put(o + 4, p->q, p->type == PRIM_QUAD ? 1.0f : 0.0f);
        put(o + 8, p->u, (float)p->material);
        put(o + 12, p->v, (float)p->area);
        put(o + 16, p->normal, (float)p->plane_d);
        put(o + 20, p->w, 0);
    }
    if (s->nprims) wgpuQueueWriteBuffer(g->queue, t->prims, 0, pr, (size_t)s->nprims * 96);
#ifdef LR_VK
    if (t->hw){
        // their boxes (a little larger: a flat quad's box is not flat), a bottom level of them
        float *bx = pr;
        for (int i = 0; i < s->nprims; i++){
            const Primitive *p = &s->prims[i];
            Aabb b;
            if (p->type == PRIM_QUAD){
                b.lo = b.hi = p->q;
                Vec3 c[3] = {v_add(p->q, p->u), v_add(p->q, p->v), v_add(v_add(p->q, p->u), p->v)};
                for (int k = 0; k < 3; k++){
                    b.lo = vec3(r_fmin(b.lo.x, c[k].x), r_fmin(b.lo.y, c[k].y), r_fmin(b.lo.z, c[k].z));
                    b.hi = vec3(r_fmax(b.hi.x, c[k].x), r_fmax(b.hi.y, c[k].y), r_fmax(b.hi.z, c[k].z));
                }
            } else {
                Vec3 r = vec3(p->radius, p->radius, p->radius);
                b.lo = v_sub(p->center, r); b.hi = v_add(p->center, r);
            }
            Vec3 d = v_sub(b.hi, b.lo);
            real pad = 1e-4 * (d.x + d.y + d.z) + 1e-4;
            Vec3 pv = vec3(pad, pad, pad);
            b.lo = v_sub(b.lo, pv); b.hi = v_add(b.hi, pv);
            float *o = bx + 6 * (size_t)i;
            o[0] = (float)b.lo.x; o[1] = (float)b.lo.y; o[2] = (float)b.lo.z;
            o[3] = (float)b.hi.x; o[4] = (float)b.hi.y; o[5] = (float)b.hi.z;
        }
        gpu_buffer_reserve(g, &t->hw_boxes, &t->hw_boxes_cap, (size_t)(s->nprims + 1) * 24, st, "spheres' and quads' boxes");
        if (s->nprims) wgpuQueueWriteBuffer(g->queue, t->hw_boxes, 0, bx, (size_t)s->nprims * 24);
        lrBlasRelease(t->hw_prim);
        t->hw_prim = NULL;
    }
#endif
    free(pr);
    Primitive *c = realloc(t->prim_copy, (size_t)(s->nprims ? s->nprims : 1) * sizeof(Primitive));
    if (!c) return 0;
    t->prim_copy = c;
    if (s->nprims) memcpy(c, s->prims, (size_t)s->nprims * sizeof(Primitive));
    t->nprim_copy = s->nprims;
    // the photon cache's cells: from the size of everything
    Aabb all = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
    if (b->nnodes > 0) all = b->nodes[0].box;
    for (int i = 0; i < s->ninsts; i++){
        all.lo = vec3(r_fmin(all.lo.x, s->insts[i].box.lo.x), r_fmin(all.lo.y, s->insts[i].box.lo.y), r_fmin(all.lo.z, s->insts[i].box.lo.z));
        all.hi = vec3(r_fmax(all.hi.x, s->insts[i].box.hi.x), r_fmax(all.hi.y, s->insts[i].box.hi.y), r_fmax(all.hi.z, s->insts[i].box.hi.z));
    }
    Vec3 ext = v_sub(all.hi, all.lo);
    real e = ext.x > ext.y ? (ext.x > ext.z ? ext.x : ext.z) : (ext.y > ext.z ? ext.y : ext.z);
    t->cell = (float)(e / 256.0);
    if (!(t->cell > 0) || t->cell > 1e20f) t->cell = 1.0f;
    t->specular = 0;
    // (a roughness texture may make any texel smooth: such a material counts, as rt/tracer.c)
    for (int m = 0; m < s->nmats; m++)
        if (material_has_delta(&s->mats[m]) || (s->mats[m].tex_mr && s->mats[m].type != MAT_LIGHT)) t->specular = 1;
    if (s->light < 0) t->specular = 0;      // no lamp: no light tracing (the sky's caustics are path traced)
    t->valid = 1;
    return 1;
}

static int hw_on(const GpuTrace *t){
#ifdef LR_VK
    return t->hw;
#else
    (void)t;
    return 0;
#endif
}

// Changed records closer than this are sent in one write with those between
#define REC_GAP 32

#ifdef LR_VK
static LRBlas hw_blas(GpuTrace *t, const char *label, LRGeometryType type, WGPUBuffer buf, uint64_t offset, uint32_t count){
    LRGeometry g;
    memset(&g, 0, sizeof(g));
    g.type = type;
    g.buffer = buf;
    g.offset = offset;
    g.stride = type == LRGeometry_Triangles ? 12 : 24;
    g.count = count;
    g.opaque = type == LRGeometry_Triangles;
    LRBlasDescriptor d;
    memset(&d, 0, sizeof(d));
    d.label = gpu_str(label);
    d.geometryCount = 1;
    d.geometries = &g;
    return lrDeviceCreateBlas(t->g->device, &d);
}

static void hw_record(LRTlasInstance *r, const Xform *xf, uint32_t custom, uint32_t sbt, LRBlas b){
    memset(r, 0, sizeof(*r));
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 4; k++) r->transform[i][k] = xf ? (float)xf->m[i][k] : (i == k ? 1.0f : 0.0f);
    r->custom_mask = custom | 0xffu << 24;
    r->sbt_flags = sbt;
    r->blas = lrBlasGetAddress(b);
}

// The top level is updated in place (lrCommandEncoderUpdateTlas) while at most a quarter
// of the instances moved since the last time, up to HW_UPDATES in a row; else built again
#define HW_UPDATES 32

// the bottom levels missing made (built at the next gpu_trace_build), the top level's records
// (some: the same scene's instances, prev of them before; only those changed since `since`
// are new)
static int hw_records(GpuTrace *t, const Scene *s, int some, unsigned since, int prev){
    Gpu *g = t->g;
    const LRBlas made[2] = {t->hw_tri, t->hw_prim};
    const int had_unit = t->hw_unit != NULL;
    int new_mesh = 0;
    if (!t->hw_tri && s->ntris > 0){
        t->hw_tri = hw_blas(t, "the world's triangles", LRGeometry_Triangles, t->hw_vtx, 0, (uint32_t)s->ntris);
        t->hw_tri_dirty = 1;
    }
    if (!t->hw_prim && s->nprims > 0){
        t->hw_prim = hw_blas(t, "spheres and quads", LRGeometry_Boxes, t->hw_boxes, 0, (uint32_t)s->nprims);
        t->hw_prim_dirty = 1;
    }
    if (!t->hw_unit){
        static const float unit[6] = {-1, -1, -1, 1, 1, 1};
        if (!t->hw_unit_buf) t->hw_unit_buf = gpu_buffer(g, "the unit box", 24, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        wgpuQueueWriteBuffer(g->queue, t->hw_unit_buf, 0, unit, 24);
        t->hw_unit = hw_blas(t, "the unit box", LRGeometry_Boxes, t->hw_unit_buf, 0, 1);
        t->hw_unit_dirty = 1;
    }
    for (int m = 0; m < t->hw_nmesh && m < s->nmeshes; m++)
        if (!t->hw_mesh[m] && s->meshes[m].ntris > 0){
            t->hw_mesh[m] = hw_blas(t, "a mesh", LRGeometry_Triangles, t->hw_vtx, (uint64_t)t->mc[m].tri0 * 36, (uint32_t)s->meshes[m].ntris);
            t->hw_mesh_dirty[m] = 1;
            new_mesh = 1;
        }
    const uint32_t base = (t->hw_tri ? 1u : 0u) + (t->hw_prim ? 1u : 0u);
    const uint32_t now = base + (uint32_t)s->ninsts;
    for (int i = prev; i < s->ninsts && some; i++){          // (the new ones each need a bottom level)
        const Instance *in = &s->insts[i];
        some = in->mesh < 0 || (in->mesh < t->hw_nmesh && t->hw_mesh[in->mesh]);
    }
    if (some && t->hw_ident && t->hw_tlas && !new_mesh && had_unit && made[0] == t->hw_tri && made[1] == t->hw_prim &&
        t->hw_ninst == base + (uint32_t)prev && now <= t->hw_tlas_cap && t->hw_rec_cap >= now * sizeof(LRTlasInstance) &&
        t->sc_hw && t->sc_hw_cap >= now * sizeof(LRTlasInstance)){
        // the records of the instances moved since, in runs (gaps under REC_GAP joined in)
        LRTlasInstance *r = t->sc_hw;
        int moved = 0;
        for (int i = 0; i < s->ninsts;){
            if (s->insts[i].version <= since){ i++; continue; }
            int first = i, last = i;
            for (int j = i; j < s->ninsts && j <= last + REC_GAP; j++){
                const Instance *in = &s->insts[j];
                if (in->version <= since) continue;
                LRBlas b = in->mesh >= 0 ? t->hw_mesh[in->mesh] : t->hw_unit;
                hw_record(&r[base + j], &in->xf, (uint32_t)j, in->mesh >= 0 ? t->mc[in->mesh].tri0 : 0, b);
                last = j;
                moved++;
            }
            wgpuQueueWriteBuffer(g->queue, t->hw_rec, (base + (uint64_t)first) * sizeof(*r), &r[base + first],
                                 (size_t)(last + 1 - first) * sizeof(*r));
            i = last + 1;
        }
        static int upd_env = -1;          // LR_GPU_TLAS_UPDATE=0: always built (to measure)
        if (upd_env < 0){ const char *e = getenv("LR_GPU_TLAS_UPDATE"); upd_env = !e || atoi(e) != 0; }
        t->hw_tlas_update = upd_env && moved <= s->ninsts / 4 && t->hw_updates < HW_UPDATES && now == t->hw_ninst;
        t->hw_ninst = now;
        t->hw_tlas_dirty = 1;
        return 1;
    }
    // the records: the world's triangles, its spheres and quads, the instances
    size_t most = (size_t)s->ninsts + 2;
    LRTlasInstance *r = scratch(&t->sc_hw, &t->sc_hw_cap, most * sizeof(*r));
    if (!r) return 0;
    uint32_t n = 0;
    if (t->hw_tri) hw_record(&r[n++], NULL, HW_WTRI, 0, t->hw_tri);
    if (t->hw_prim) hw_record(&r[n++], NULL, HW_WPRIM, 0, t->hw_prim);
    for (int i = 0; i < s->ninsts; i++){
        const Instance *in = &s->insts[i];
        LRBlas b = in->mesh >= 0 ? (in->mesh < t->hw_nmesh ? t->hw_mesh[in->mesh] : NULL) : t->hw_unit;
        if (!b) continue;
        hw_record(&r[n++], &in->xf, (uint32_t)i, in->mesh >= 0 ? t->mc[in->mesh].tri0 : 0, b);
    }
    gpu_buffer_reserve(g, &t->hw_rec, &t->hw_rec_cap, most * sizeof(*r), WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                       "top level: instances");
    if (n) wgpuQueueWriteBuffer(g->queue, t->hw_rec, 0, r, (size_t)n * sizeof(*r));
    if (!t->hw_tlas || n > t->hw_tlas_cap){
        uint32_t cap = 64;
        while (cap < n) cap *= 2;
        for (int k = 0; k < 3; k++) gpu_bind_cache_clear(&t->bc_trace[k]);   // (they bind the top level)
        for (int k = 0; k < 6; k++) gpu_bind_cache_clear(&t->bc_wf[k]);
        lrTlasRelease(t->hw_tlas);
        t->hw_tlas = lrDeviceCreateTlas(g->device, cap, gpu_str("the scene's top level"));
        t->hw_tlas_cap = t->hw_tlas ? cap : 0;
        if (!t->hw_tlas) return 0;
    }
    t->hw_ninst = n;
    t->hw_tlas_dirty = 1;
    t->hw_ident = n == base + (uint32_t)s->ninsts;
    t->hw_tlas_update = 0;
    return 1;
}

// the bottom levels that changed, then the top level if they did or its records are new
// (hw_records: whenever the instances moved)
static void hw_build(GpuTrace *t, WGPUCommandEncoder enc){
    if (!t->hw || !t->hw_tlas) return;
    int blas = (t->hw_tri_dirty && t->hw_tri) || (t->hw_prim_dirty && t->hw_prim) || (t->hw_unit_dirty && t->hw_unit);
    if (t->hw_tri_dirty && t->hw_tri) lrCommandEncoderBuildBlas(enc, t->hw_tri);
    if (t->hw_prim_dirty && t->hw_prim) lrCommandEncoderBuildBlas(enc, t->hw_prim);
    if (t->hw_unit_dirty && t->hw_unit) lrCommandEncoderBuildBlas(enc, t->hw_unit);
    t->hw_tri_dirty = t->hw_prim_dirty = t->hw_unit_dirty = 0;
    for (int m = 0; m < t->hw_nmesh; m++)
        if (t->hw_mesh_dirty[m] && t->hw_mesh[m]){ lrCommandEncoderBuildBlas(enc, t->hw_mesh[m]); t->hw_mesh_dirty[m] = 0; blas = 1; }
    if (!blas && !t->hw_tlas_dirty) return;
    if (t->hw_tlas_update && !blas){
        lrCommandEncoderUpdateTlas(enc, t->hw_tlas, t->hw_rec, 0, t->hw_ninst);
        t->hw_updates++;
    } else {
        lrCommandEncoderBuildTlas(enc, t->hw_tlas, t->hw_rec, 0, t->hw_ninst);
        t->hw_updates = 0;
    }
    t->hw_tlas_dirty = t->hw_tlas_update = 0;
}
#endif

// Whether the top level is the CPU's: its SAH tree traces faster (docs/STATUS.md,
// 2026-11-13; 12.5 against 15.6 ms a pass with 48k instances), but it is built on the CPU
// each frame the instances moved (4.9 ms for 18k, 12 for 48k), the GPU's LBVH in under
// half a ms; so the CPU's up to TLAS_GPU_FROM instances (18k: as fast either way; 48k: 23 ->
// 27 frames/s with the GPU's).
#define TLAS_GPU_FROM 32768
static int tlas_cpu(const GpuTrace *t, int ninsts){
    return t->tlas_mode == 1 || (t->tlas_mode == 2 && ninsts < TLAS_GPU_FROM);
}

// the instances' records, the dynamic meshes' triangles; their BVHs to build (gpu_trace_build)
static int upload_instances(GpuTrace *t, const Scene *s){
    Gpu *g = t->g;
    t->ninst = s->ninsts;
    // with ray queries the top level is the hardware's (hw_records): no BVH of the instances
    const int bvh = !hw_on(t);
    const Bvh *tl = bvh && tlas_cpu(t, s->ninsts) ? scene_tlas(s) : NULL;
    if (tl && !((uint32_t)tl->nnodes <= (uint32_t)(2 * t->inst_cap - 1) && tl->nrefs <= t->inst_cap)) tl = NULL;
    // the same instances as last time (the camera moved, not they): nothing to send. (Not if
    // a tree asked for then was never built, or the top level would now come from elsewhere.)
    const int same = t->inst_valid && t->nbuild == 0 && t->inst_version == s->inst_version && t->inst_count == s->ninsts &&
                     t->inst_prim0 == s->nprims && t->inst_tl == (tl != NULL);
    // the same scene's instances, as many or more (the layout had room: inst_valid), the rest
    // as before: only the records of those moved or added since are sent (Instance.version),
    // the boxes kept are the others' still
    const size_t ni = (size_t)(s->ninsts > 0 ? s->ninsts : 0);
    const int prev_count = t->inst_count;
    const int some = !same && t->inst_valid && t->nbuild == 0 && t->insts_id == s->insts_id && t->inst_count <= s->ninsts &&
                     t->inst_prim0 == s->nprims && t->inst_tl == (tl != NULL) && t->sc_rec && t->sc_rec_cap >= ni * 192 &&
                     (!bvh || (t->sc_box && t->sc_box_cap >= ni * 32));
    t->nbuild = 0;
    if (s->ninsts > 0 && !same){
        float *rec = some ? t->sc_rec : scratch(&t->sc_rec, &t->sc_rec_cap, ni * 192);
        float *box = !bvh ? NULL : some ? t->sc_box : scratch(&t->sc_box, &t->sc_box_cap, ni * 32);
        if (!rec || (bvh && !box)) return 0;
        Aabb all = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
        for (int i = 0; i < s->ninsts; i++){
            const Instance *in = &s->insts[i];
            if (some && in->version <= t->inst_version){
                all.lo = vec3(r_fmin(all.lo.x, in->box.lo.x), r_fmin(all.lo.y, in->box.lo.y), r_fmin(all.lo.z, in->box.lo.z));
                all.hi = vec3(r_fmax(all.hi.x, in->box.hi.x), r_fmax(all.hi.y, in->box.hi.y), r_fmax(all.hi.z, in->box.hi.z));
                continue;
            }
            float *o = rec + 48 * (size_t)i;
            for (int r = 0; r < 3; r++)
                for (int k = 0; k < 4; k++){ o[4 * r + k] = (float)in->inv.m[r][k]; o[12 + 4 * r + k] = (float)in->xf.m[r][k]; }
            // a mesh: its BVH's root, its index; a shape (scene.h INST_*): the kind the kernel tests
            uint32_t kind = in->mesh >= 0 ? (uint32_t)in->mesh : (in->mesh == INST_SPHERE ? 0xfffffff0u : 0xfffffff1u);
            uint32_t meta[4] = {in->mesh >= 0 ? t->mc[in->mesh].root : 0, in->material >= 0 ? (uint32_t)in->material : 0xffffffffu, kind, 0};
            memcpy(o + 24, meta, 16);
            put(o + 28, in->box.lo, 0);
            put(o + 32, in->box.hi, 0);
            memset(o + 36, 0, 48);
            if (!bvh) continue;
            put(box + 8 * (size_t)i, in->box.lo, 0);
            put(box + 8 * (size_t)i + 4, in->box.hi, 0);
            all.lo = vec3(r_fmin(all.lo.x, in->box.lo.x), r_fmin(all.lo.y, in->box.lo.y), r_fmin(all.lo.z, in->box.lo.z));
            all.hi = vec3(r_fmax(all.hi.x, in->box.hi.x), r_fmax(all.hi.y, in->box.hi.y), r_fmax(all.hi.z, in->box.hi.z));
        }
        if (some){
            // runs of the records changed (gaps of fewer than REC_GAP joined in)
            for (int i = 0; i < s->ninsts;){
                if (s->insts[i].version <= t->inst_version){ i++; continue; }
                int first = i, last = i;
                for (int j = i; j < s->ninsts && j <= last + REC_GAP; j++) if (s->insts[j].version > t->inst_version) last = j;
                wgpuQueueWriteBuffer(g->queue, t->prims, (uint64_t)s->nprims * 96 + (uint64_t)first * 192, rec + 48 * (size_t)first,
                                     (size_t)(last + 1 - first) * 192);
                i = last + 1;
            }
        } else wgpuQueueWriteBuffer(g->queue, t->prims, (uint64_t)s->nprims * 96, rec, ni * 192);
        if (tl){
            // the CPU's (SAH) top level, as it is
            uint32_t *nd = scratch(&t->sc_nd, &t->sc_nd_cap, (size_t)tl->nnodes * 32);
            uint32_t *ax = scratch(&t->sc_ax, &t->sc_ax_cap, (size_t)tl->nrefs * 4);
            if (nd && ax){
                put_nodes(nd, tl, t->tlas_node0, t->tlas_ref0);
                for (int i = 0; i < tl->nrefs; i++) ax[i] = (uint32_t)tl->refs[i];
                wgpuQueueWriteBuffer(g->queue, t->nodes, (uint64_t)t->tlas_node0 * 32, nd, (size_t)tl->nnodes * 32);
                wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)t->tlas_ref0 * 4, ax, (size_t)tl->nrefs * 4);
            } else tl = NULL;
        }
        if (!tl && bvh) build_queue(t, box, s->ninsts, t->tlas_node0, t->tlas_ref0, 0, all, 0);   // the top level: instance ids
    }
    const unsigned since = t->inst_version;     // (hw_records)
    (void)since; (void)prev_count;
    t->inst_valid = 1;
    t->inst_version = s->inst_version;
    t->insts_id = s->insts_id;
    t->inst_count = s->ninsts;
    t->inst_prim0 = s->nprims;
    t->inst_tl = tl != NULL;
    // dynamic meshes: their new triangles, and a BVH of them built on the GPU in their place
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        GpuMeshCopy *c = &t->mc[m];
        if (sm->dynamic ? c->version == sm->version : !t->blas_gpu || c->built) continue;
        if (sm->dynamic && !upload_tris(t, sm->tris, sm->ntris, c->tri0, c->nrm0)) return 0;
#ifdef LR_VK
        if (hw_on(t)){                       // its bottom level built again instead
            if (m < t->hw_nmesh) t->hw_mesh_dirty[m] = 1;
            c->version = sm->version;
            c->built = 1;
            continue;
        }
#endif
        float *box = malloc((size_t)(sm->ntris ? sm->ntris : 1) * 32);
        if (!box) return 0;
        for (int i = 0; i < sm->ntris; i++){
            Aabb bb = triangle_bounds(&sm->tris[i]);
            put(box + 8 * (size_t)i, bb.lo, 0);
            put(box + 8 * (size_t)i + 4, bb.hi, 0);
        }
        build_queue(t, box, sm->ntris, c->node0, c->ref0, c->tri0, sm->box, 1);
        free(box);
        c->version = sm->version;
        c->built = 1;
    }
#ifdef LR_VK
    if (hw_on(t) && !(same && t->hw_tlas) && !hw_records(t, s, some, since, prev_count)){ t->inst_valid = 0; return 0; }
#endif
    return 1;
}

// the sky's textures, again when the scene's sky changed (a 1 x 1 pair without one)
static void upload_sky(GpuTrace *t, const Scene *s){
    const Env *e = s->env;
    unsigned id = e ? e->id : 0;
    if (t->sky_tex && id == t->sky_id) return;
    if (t->sky_view) wgpuTextureViewRelease(t->sky_view);
    if (t->skyc_view) wgpuTextureViewRelease(t->skyc_view);
    if (t->sky_tex){ wgpuTextureDestroy(t->sky_tex); wgpuTextureRelease(t->sky_tex); }
    if (t->skyc_tex){ wgpuTextureDestroy(t->skyc_tex); wgpuTextureRelease(t->skyc_tex); }
    float none[4] = {0, 0, 0, 0};
    float *rgba = NULL, *cdf = NULL;
    int w = 1, h = 1;
    if (e && (rgba = malloc(sizeof(float) * 4 * (size_t)e->w * e->h)) && (cdf = calloc((size_t)e->w * (e->h + 1), sizeof(float)))){
        w = e->w; h = e->h;
        for (int y = 0; y < h; y++){
            for (int x = 0; x < w; x++){
                const float *c = e->rgb + 3 * ((size_t)y * w + x);
                float *o = rgba + 4 * ((size_t)y * w + x);
                o[0] = c[0]; o[1] = c[1]; o[2] = c[2];
                o[3] = (float)env_pixel_prob(e, x, y);
                cdf[(size_t)y * w + x] = e->cond[(size_t)y * (w + 1) + x + 1];
            }
        }
        for (int y = 0; y < h; y++) cdf[(size_t)h * w + y] = e->marg[y + 1];
    } else {
        free(rgba);
        rgba = NULL;
        if (e) fprintf(stderr, "GPU: no memory for the sky: the GPU path tracer leaves it out\n");
        id = 0;
    }
    t->sky_tex = gpu_texture_f32(t->g, "sky", w, h, 4, rgba ? rgba : none);
    t->skyc_tex = gpu_texture_f32(t->g, "sky: distributions", w, rgba ? h + 1 : 1, 1, rgba ? cdf : none);
    t->sky_view = wgpuTextureCreateView(t->sky_tex, NULL);
    t->skyc_view = wgpuTextureCreateView(t->skyc_tex, NULL);
    t->sky_id = id;
    free(rgba);
    free(cdf);
}

int gpu_trace_reads_tlas(const GpuTrace *t, const Scene *s){ return tlas_cpu(t, s->ninsts) && !hw_on(t); }

int gpu_trace_sync(GpuTrace *t, const Scene *s){
    upload_sky(t, s);
    if (!t->txt) t->txt = gpu_buffer(t->g, "path tracer: the textures' table", sizeof(t->gt.table), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    if (gpu_textures_sync(t->g, &t->gt, s)) wgpuQueueWriteBuffer(t->g->queue, t->txt, 0, t->gt.table, sizeof(t->gt.table));
    if (t->laid_out && t->textured != (s->ntexmaps > 0)) t->laid_out = 0;   // the layout without its texture coordinates
    if (!geometry_same(t, s)){
        if (!layout_same(t, s) && !upload_layout(t, s)) return 0;
        if (!upload_world(t, s)) return 0;
    }
    return upload_instances(t, s);
}

void gpu_trace_clear(GpuTrace *t, WGPUCommandEncoder enc){
    wgpuCommandEncoderClearBuffer(enc, t->acc[t->cur], 0, (uint64_t)t->w * t->h * 16 * t->planes);
}

int gpu_trace_caustics_on(const GpuTrace *t, const GpuTraceParams *tp){ return tp->photons > 0 && t->specular; }

// light paths per pass: photons per pixel
static uint32_t light_paths(const GpuTrace *t, const GpuTraceParams *tp){
    if (!gpu_trace_caustics_on(t, tp)) return 0;
    double n = tp->photons * t->w * t->h;
    return n < 1 ? 1u : (n > 1.6e7 ? 16000000u : (uint32_t)n);
}

// the parameters of `reps` passes, sample indices index, index + 1, ...
static void fill_params(GpuTrace *t, GTP *Pp, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                        int nvolumes, float lt_mode){
    GTP P;
    memset(&P, 0, sizeof(P));
    put(P.origin, tp->cam.origin, 0);
    put(P.upper_left, tp->cam.upper_left, 0);
    put(P.horizontal, tp->cam.horizontal, 0);
    put(P.vertical, tp->cam.vertical, 0);
    P.dims[0] = (uint32_t)t->w; P.dims[1] = (uint32_t)t->h; P.dims[2] = (uint32_t)y0; P.dims[3] = (uint32_t)rows;
    P.opt[0] = (uint32_t)(tp->max_depth > 0 ? tp->max_depth : 0);
    P.opt[1] = tp->mis ? 1u : 0u;
    P.opt[2] = (uint32_t)s->nprims;
    P.opt[3] = s->light >= 0 && s->light < s->nprims ? (uint32_t)s->light : 0xffffffffu;
    P.cnt[0] = t->nnodes | (t->world_wide && t->nnodes ? GPU_WIDE : 0u);
    P.cnt[1] = t->normals_at;
    P.cnt[2] = 0;
    P.cnt[3] = (uint32_t)nvolumes;
    P.f[0] = (float)tp->clamp;
    P.f[1] = RT_EPS_F;
    P.f[2] = RT_SHADOW_MARGIN_F;
    P.f[3] = t->planes == 5 ? 1.0f : 0.0f;
    P.ext[0] = tp->random ? 0u : 1u;
    P.ext[1] = 0x2545f491u;
    P.ext[2] = gpu_trace_caustics_on(t, tp) ? 1u : 0u;
    P.ext[3] = light_paths(t, tp);
    P.lt[0] = P.ext[3] ? (float)(4294967296.0 / P.ext[3]) : 0.0f;
    P.lt[1] = (float)(camera_screen_area(&tp->cam) / ((double)t->w * t->h));
    P.lt[2] = t->cell;
    P.lt[3] = lt_mode;              // 1 = the light paths fill the photon cache
    if (s->env && t->sky_id == s->env->id){
        P.sky[0] = (float)s->env->scale;
        P.sky[1] = (float)r_cos(s->env->rot);
        P.sky[2] = (float)r_sin(s->env->rot);
        P.sky[3] = 1;
    } else P.sky[1] = 1;
    P.ins[0] = (uint32_t)t->ninst;
    P.ins[1] = (uint32_t)s->nprims;
    P.ins[2] = t->tlas_node0;
    P.ins[3] = t->world_node0;
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        Material r = material_resolved(&s->mats[m]);
        put(P.mats[m], r.albedo, (float)r.type);
        put(P.mats[m] + 4, r.emission, (float)r.ior);
        put(P.mats[m] + 8, vec3(r.metallic, r.roughness, r.transmission), 0);
        // its textures the GPU has (as 1 + index), the scale's bits (none without the layout's coordinates)
        const Material *mm = &s->mats[m];
        int ok = t->textured && mm->type != MAT_LIGHT;
        uint32_t tx[4];
        gpu_tex_pack(mm, ok ? t->gt.ntex : 0, tx);
        memcpy(&P.mats[m][12], tx, 16);
    }
    P.txl[0] = t->uvtri_at; P.txl[1] = t->uv_at; P.txl[2] = t->textured ? 1u : 0u;
    if (t->adapt_thr > 0 && t->planes == 5){
        memcpy(&P.ad[0], &t->adapt_thr, 4);
        P.ad[1] = (uint32_t)(t->adapt_min > 1 ? t->adapt_min : 1);
        P.ad[2] = (uint32_t)(t->adapt_sky > 0 ? t->adapt_sky : 0);
    }
    *Pp = P;
}

static void write_params(GpuTrace *t, WGPUBuffer dst, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                         int nvolumes, uint32_t index, int reps, float lt_mode){
    Gpu *g = t->g;
    GTP P;
    fill_params(t, &P, s, tp, y0, rows, nvolumes, lt_mode);
    if (reps < 1) reps = 1;
    if (reps > MAX_REPS) reps = MAX_REPS;
    unsigned char buf[GTP_STRIDE * MAX_REPS];
    for (int r = 0; r < reps; r++){
        P.cnt[2] = index + (uint32_t)r;
        memcpy(buf + (size_t)r * GTP_STRIDE, &P, sizeof(P));
    }
    wgpuQueueWriteBuffer(g->queue, dst, 0, buf, (size_t)(reps - 1) * GTP_STRIDE + sizeof(P));
}

// `repeats` dispatches, the r-th with the r-th copy of the parameters (light = 1: the light
// tracer, whose binding 11 takes the place of 9; 3: the path tracer's bindings, timed as the
// guide; 4: its bindings, not timed): one bind group, the copy chosen by its dynamic offset
// (it was a bind group per dispatch, each several us of the CPU's time)
static void dispatch_ex(GpuTrace *t, WGPUCommandEncoder enc, WGPUComputePipeline pipe, size_t threads, int repeats,
                        WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat, int light){
    const int guide = light == 3, untimed = light == 4;
    if (guide || untimed) light = 0;
    WGPUBuffer b[11] = {frame_u, dens, bricks, vstat, light == 2 ? t->pc_params : light ? t->lt_params : t->params,
                        t->nodes, t->aux, t->prims, t->tris,
                        light == 2 ? t->grid : light ? t->caus : t->acc[t->cur], t->blue};
    WGPUBindGroupEntry e[16];
    memset(e, 0, sizeof(e));
    for (int i = 0; i < 11; i++){
        e[i].binding = (uint32_t)(light && i == 9 ? 11 : i);
        e[i].buffer = b[i];
        e[i].size = i == 4 ? sizeof(GTP) : WGPU_WHOLE_SIZE;   // (4: at the dynamic offset)
    }
    int nent = 11;
    if (!light){                    // the path tracer: the sky too (the light tracer has none)
        e[11].binding = 16; e[11].textureView = t->sky_view;
        e[12].binding = 17; e[12].textureView = t->skyc_view;
        nent = 13;
    }
    e[nent].binding = 18; e[nent].textureView = t->gt.view;           // the textures
    e[nent + 1].binding = 19; e[nent + 1].buffer = t->txt; e[nent + 1].size = WGPU_WHOLE_SIZE;
    nent += 2;
#ifdef LR_VK
    LRBindGroupEntryTlas te;
    if (t->hw) hw_entry(t, &e[nent++], &te);
#endif
    if (repeats < 1) repeats = 1;
    if (repeats > MAX_REPS) repeats = MAX_REPS;
    // timed: the tracer per full-image pass (1 sample per pixel), the light tracer per pass,
    // the guide per image
    WGPUComputePassEncoder pass = light == 2 || untimed ? wgpuCommandEncoderBeginComputePass(enc, NULL)
                                : guide ? timed_pass(t, enc, TS_GUIDE, 1)
                                : timed_pass(t, enc, light ? TS_LIGHT : TS_TRACE,
                                             light ? repeats : (double)threads * repeats / ((double)t->w * t->h));
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    const void *chain = NULL;
#ifdef LR_VK
    if (t->hw) chain = t->hw_tlas;
#endif
    WGPUBindGroup bg = gpu_bind_cached(t->g, &t->bc_trace[light], light ? t->lt_bgl : t->bgl, e, nent, chain);
    for (int r = 0; r < repeats; r++){
        uint32_t off = (uint32_t)(r * GTP_STRIDE);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 1, &off);
        gpu_dispatch(pass, threads);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

static void dispatch(GpuTrace *t, WGPUCommandEncoder enc, WGPUComputePipeline pipe, size_t threads, int repeats,
                     WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    dispatch_ex(t, enc, pipe, threads, repeats, frame_u, dens, bricks, vstat, 0);
}

// ---- the wavefront tracer --------------------------------------------------------------

enum { WF_GEN, WF_EXT, WF_SHADE, WF_ARGS, WF_SHADOW, WF_RESOLVE, NWF };
static const char *const WF_ENTRY[NWF] = {"wf_gen", "wf_ext", "wf_shade", "wf_args", "wf_shadow", "wf_resolve"};
// the bindings each kernel uses (index = binding; as gpu_layout): within 8 storage buffers each.
// (4, the parameters: D, the wave's copy chosen by a dynamic offset, as the single kernel's)
static const char *const WF_LAYOUT[NWF] = {
    "----D----WU-W",      // gen: parameters, samples (their count seeds the path), blue noise, WF
    "----DRRRR---W",      // ext: the BVH and shapes, WF
    "URRRD-RRR-U-WW--TTTU", // shade: media, shapes and normals, blue noise, WF, WC, the sky, the textures
    "------------WWW",    // args: WF, WC, WA
    "URRRDRRRR---W",      // shadow: media, the BVH and shapes, WF
    "----D----W--W"};     // resolve: samples, WF
_Static_assert(NWF == 6, "GpuTrace.wf_bgl, wf_pipe, bc_wf");
#define WF_PATH_BYTES 140          // per path: 8 vec4 of state, hit and shadow ray, 3 queue entries
#define WF_BYTES_MAX (96u << 20)   // WF at most 96 MB (browsers' default limit per buffer: 128 MB)
#define MAX_WAVES 64

void gpu_trace_set_wavefront(GpuTrace *t, int on){ if (t) t->wavefront = on; }

static WGPUComputePipeline pipeline_of(Gpu *g, const char *label, WGPUShaderModule m, const char *entry, WGPUBindGroupLayout bgl){
    WGPUPipelineLayoutDescriptor ld;
    memset(&ld, 0, sizeof(ld));
    ld.bindGroupLayoutCount = 1;
    ld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g->device, &ld);
    WGPUComputePipelineDescriptor cd;
    memset(&cd, 0, sizeof(cd));
    cd.label = gpu_str(label);
    cd.layout = pl;
    cd.compute.module = m;
    cd.compute.entryPoint = gpu_str(entry);
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(g->device, &cd);
    wgpuPipelineLayoutRelease(pl);
    return p;
}

// made the first time it is used: one shader module, six pipelines, the buffers
static int wf_ready(GpuTrace *t){
    if (t->wf_made) return 1;
    if (t->wf_failed) return 0;
    Gpu *g = t->g;
    size_t bytes = WF_BYTES_MAX < g->max_storage ? WF_BYTES_MAX : (size_t)g->max_storage;
    size_t paths = bytes / WF_PATH_BYTES;
    t->wf_rows = (int)(paths / (size_t)t->w);
    if (t->wf_rows < 1){ t->wf_failed = 1; return 0; }
    if (t->wf_rows > t->h) t->wf_rows = t->h;
    if ((t->h + t->wf_rows - 1) / t->wf_rows > MAX_WAVES) t->wf_rows = (t->h + MAX_WAVES - 1) / MAX_WAVES;
    size_t n = (size_t)t->wf_rows * t->w;
    size_t wf_bytes = (1 + 8 * n + 3 * ((n + 3) / 4)) * 16;
    char *src = trace_source(t, 0, 1);
    if (!src){ t->wf_failed = 1; return 0; }
    WGPUShaderModule m = gpu_shader(g, "path tracer: wavefront", src);
    free(src);
    for (int k = 0; k < NWF; k++){
        char label[64];
        snprintf(label, sizeof(label), "path tracer: %s", WF_ENTRY[k]);
        char lt[32];
        t->wf_bgl[k] = gpu_layout(g, WGPUShaderStage_Compute, layout_types(t, WF_LAYOUT[k], lt, sizeof(lt)));
        t->wf_pipe[k] = pipeline_of(g, label, m, WF_ENTRY[k], t->wf_bgl[k]);
    }
    wgpuShaderModuleRelease(m);
    t->wf = gpu_buffer(g, "wavefront: paths", wf_bytes, WGPUBufferUsage_Storage);
    t->wfc = gpu_buffer(g, "wavefront: counters", 16, WGPUBufferUsage_Storage);
    t->wfa = gpu_buffer(g, "wavefront: dispatch sizes", 32, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect);
    t->wf_params = gpu_buffer(g, "wavefront: parameters", GTP_STRIDE * MAX_REPS * MAX_WAVES,
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->wf_made = t->wf && t->wfc && t->wfa && t->wf_params;
    if (!t->wf_made) t->wf_failed = 1;
    return t->wf_made;
}

// kernel k's bind group, kept (gpu_bind_cached): the same for every wave and pass, whose
// parameters are chosen by the dynamic offset of binding 4 (it was a group per kernel, wave
// and pass, made every frame)
static WGPUBindGroup wf_bind(GpuTrace *t, int k, WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    WGPUBuffer b[20] = {frame_u, dens, bricks, vstat, t->wf_params, t->nodes, t->aux, t->prims, t->tris,
                        t->acc[t->cur], t->blue, NULL, t->wf, t->wfc, t->wfa, NULL, NULL, NULL, NULL, t->txt};
    WGPUBindGroupEntry e[21];
    memset(e, 0, sizeof(e));
    int n = 0;
    const void *chain = NULL;
#ifdef LR_VK
    LRBindGroupEntryTlas te;
    if (t->hw && WF_LAYOUT[k][5] == 'R'){ hw_entry(t, &e[n++], &te); chain = t->hw_tlas; }
#endif
    for (int i = 0; WF_LAYOUT[k][i]; i++){
        if (WF_LAYOUT[k][i] == '-') continue;
        e[n].binding = (uint32_t)i;
        if (WF_LAYOUT[k][i] == 'T') e[n].textureView = i == 16 ? t->sky_view : (i == 17 ? t->skyc_view : t->gt.view);
        else {
            e[n].buffer = b[i];
            e[n].size = i == 4 ? sizeof(GTP) : WGPU_WHOLE_SIZE;   // (4: at the dynamic offset)
        }
        n++;
    }
    return gpu_bind_cached(t->g, &t->bc_wf[k], t->wf_bgl[k], e, n, chain);
}

static void write_params(GpuTrace *t, WGPUBuffer dst, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                         int nvolumes, uint32_t index, int reps, float lt_mode);

// the scene has a sky (uploaded): the kernels with its code, made now if not yet (the single
// kernel; the wavefront's shade step if the wavefront is made). 0: the scene has none, or
// they cannot be made (then the sky is left out)
static int sky_kernels(GpuTrace *t, const Scene *s){
    if (!s->env || t->sky_id != s->env->id) return 0;
    if (!t->pipe_sky && !t->pipe_sky_failed){
        char *src = trace_source(t, 1, 0);
        if (src) t->pipe_sky = gpu_compute(t->g, "path tracer (sky)", src, "main", t->bgl);
        free(src);
        if (!t->pipe_sky){ t->pipe_sky_failed = 1; fprintf(stderr, "GPU: the path tracer with a sky could not be made\n"); }
    }
    if (t->wf_made && !t->wf_shade_sky && !t->wf_sky_failed){
        char *src = trace_source(t, 1, 1);
        if (src){
            WGPUShaderModule m = gpu_shader(t->g, "path tracer: wavefront (sky)", src);
            t->wf_shade_sky = pipeline_of(t->g, "path tracer: wf_shade (sky)", m, "wf_shade", t->wf_bgl[WF_SHADE]);
            wgpuShaderModuleRelease(m);
        }
        free(src);
        if (!t->wf_shade_sky) t->wf_sky_failed = 1;
    }
    return t->pipe_sky != NULL && (!t->wf_made || t->wf_shade_sky != NULL);
}

static void fill_params(GpuTrace *t, GTP *P, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                        int nvolumes, float lt_mode);

// a band: per pass and wave, the camera rays, then per bounce extend, shade, sizes, shadow
// rays, then the image; all in one compute pass
static void wf_encode(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                      int y0, int rows, int repeats, int nvolumes, uint32_t index,
                      WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    const int waves = (rows + t->wf_rows - 1) / t->wf_rows;
    if (repeats < 1) repeats = 1;
    if (repeats > MAX_REPS) repeats = MAX_REPS;
    GTP P;
    fill_params(t, &P, s, tp, y0, rows, nvolumes, 0);
    size_t used = (size_t)(repeats * waves - 1) * GTP_STRIDE + sizeof(GTP);
    unsigned char *buf = calloc(1, (size_t)repeats * waves * GTP_STRIDE);
    if (!buf) return;
    for (int r = 0; r < repeats; r++)
        for (int w = 0; w < waves; w++){
            P.dims[2] = (uint32_t)(y0 + w * t->wf_rows);
            P.dims[3] = (uint32_t)(rows - w * t->wf_rows < t->wf_rows ? rows - w * t->wf_rows : t->wf_rows);
            P.cnt[2] = index + (uint32_t)r;
            memcpy(buf + ((size_t)r * waves + w) * GTP_STRIDE, &P, sizeof(P));
        }
    wgpuQueueWriteBuffer(t->g->queue, t->wf_params, 0, buf, used);
    free(buf);
    const int iters = (tp->max_depth > 0 ? (tp->max_depth < 4096 ? tp->max_depth : 4096) : 0) + 1;
    WGPUComputePipeline shade_sky = sky_kernels(t, s) ? t->wf_shade_sky : NULL;
    static int prof = -1;
    if (prof < 0){ const char *e = getenv("LR_WF_PROFILE"); prof = e && atoi(e) > 0; }
    const double units = (double)t->w * rows * repeats / ((double)t->w * t->h);
    WGPUComputePassEncoder pass = prof ? NULL : timed_pass(t, enc, TS_TRACE, units);
    // (LR_WF_PROFILE=1: every dispatch in a pass of its own, timed by kernel)
#define WF_STEP(k, call) do { \
        if (prof) pass = timed_pass(t, enc, TS_WF0 + (k), (k) == WF_GEN || (k) == WF_RESOLVE ? units / waves : units / waves / iters); \
        wgpuComputePassEncoderSetPipeline(pass, (k) == WF_SHADE && shade_sky ? shade_sky : t->wf_pipe[k]); \
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg[k], WF_LAYOUT[k][4] == 'D', &toff); \
        call; \
        if (prof){ wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); } \
    } while (0)
    WGPUBindGroup bg[NWF];
    for (int k = 0; k < NWF; k++) bg[k] = wf_bind(t, k, frame_u, dens, bricks, vstat);
    for (int r = 0; r < repeats; r++)
        for (int w = 0; w < waves; w++){
            uint32_t toff = (uint32_t)(((size_t)r * waves + w) * GTP_STRIDE);
            size_t n = (size_t)t->w * (rows - w * t->wf_rows < t->wf_rows ? rows - w * t->wf_rows : t->wf_rows);
            WF_STEP(WF_GEN, gpu_dispatch(pass, n));
            for (int it = 0; it < iters; it++){
                WF_STEP(WF_EXT, if (it == 0) gpu_dispatch(pass, n); else wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, t->wfa, 0));
                WF_STEP(WF_SHADE, if (it == 0) gpu_dispatch(pass, n); else wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, t->wfa, 0));
                WF_STEP(WF_ARGS, wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1));
                WF_STEP(WF_SHADOW, wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, t->wfa, 16));
            }
            WF_STEP(WF_RESOLVE, gpu_dispatch(pass, n));
        }
#undef WF_STEP
    if (!prof){
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
}

void gpu_trace_encode(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                      int y0, int rows, int repeats, int nvolumes, uint32_t index,
                      WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    gpu_trace_build(t, enc);
    if (t->wavefront && wf_ready(t)){
        wf_encode(t, enc, s, tp, y0, rows, repeats, nvolumes, index, frame_u, dens, bricks, vstat);
        return;
    }
    write_params(t, t->params, s, tp, y0, rows, nvolumes, index, repeats, 0);
    dispatch(t, enc, sky_kernels(t, s) ? t->pipe_sky : t->pipe, (size_t)t->w * rows, repeats, frame_u, dens, bricks, vstat);
}

void gpu_trace_set_adaptive(GpuTrace *t, double threshold, int min_samples, int min_sky){
    if (!t) return;
    t->adapt_thr = threshold > 0 ? (float)threshold : 0;
    t->adapt_min = min_samples;
    t->adapt_sky = min_sky;
}

int gpu_trace_adapt(GpuTrace *t, WGPUCommandEncoder enc, WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    // (after gpu_trace_encode in the same submission: its parameters, the first copy, are the tiles')
    if (!t || t->adapt_thr <= 0 || t->planes != 5 || !t->adapt_pipe || (t->wavefront && wf_ready(t))) return 0;
    size_t tiles = (size_t)((t->w + 7) / 8) * ((t->h + 7) / 8);
    dispatch_ex(t, enc, t->adapt_pipe, tiles, 1, frame_u, dens, bricks, vstat, 4);
    return 1;
}

void gpu_trace_light(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                     int passes, int nvolumes, uint32_t index,
                     WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    if (!gpu_trace_caustics_on(t, tp) || passes < 1) return;
    write_params(t, t->lt_params, s, tp, 0, t->h, nvolumes, index, passes, 0);
    dispatch_ex(t, enc, t->lt_pipe, light_paths(t, tp), passes, frame_u, dens, bricks, vstat, 1);
    t->caus_passes += passes;
}

void gpu_trace_caustics_clear(GpuTrace *t, WGPUCommandEncoder enc){
    wgpuCommandEncoderClearBuffer(enc, t->caus, 0, (uint64_t)t->w * t->h * 24);
    t->caus_passes = 0;
}

int gpu_trace_caustic_passes(const GpuTrace *t){ return t->caus_passes; }

#define PC_CELLS (1u << 20)
void gpu_trace_photons(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp, int nvolumes,
                       uint32_t index, double fade, WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks,
                       WGPUBuffer vstat){
    if (!gpu_trace_caustics_on(t, tp) || t->planes != 5) return;
    Gpu *g = t->g;
    if (!t->grid){
        t->grid = gpu_buffer(g, "photon cache", (size_t)PC_CELLS * 32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        wgpuCommandEncoderClearBuffer(enc, t->grid, 0, (uint64_t)PC_CELLS * 32);
        t->pc_paths = 0;
    }
    if (fade < 1 && t->pc_paths > 0){
        struct { float k[4]; uint32_t n[4]; } D = {{(float)fade, 0, 0, 0}, {PC_CELLS * 4, 0, 0, 0}};
        wgpuQueueWriteBuffer(g->queue, t->dc_params, 0, &D, sizeof(D));
        WGPUBuffer b[2] = {t->dc_params, t->grid};
        WGPUBindGroup bg = gpu_bind_cached_buffers(g, &t->bc_decay, t->dc_bgl, b, 2);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        wgpuComputePassEncoderSetPipeline(pass, t->dc_pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        gpu_dispatch(pass, (size_t)PC_CELLS * 4);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        t->pc_paths *= fade;
    }
    write_params(t, t->pc_params, s, tp, 0, t->h, nvolumes, index ^ 0x5bd1e995u, 1, 1.0f);
    dispatch_ex(t, enc, t->lt_pipe, light_paths(t, tp), 1, frame_u, dens, bricks, vstat, 2);
    t->pc_paths += light_paths(t, tp);
}

void gpu_trace_photons_clear(GpuTrace *t, WGPUCommandEncoder enc){
    if (t->grid) wgpuCommandEncoderClearBuffer(enc, t->grid, 0, (uint64_t)PC_CELLS * 32);
    t->pc_paths = 0;
}

double gpu_trace_photon_paths(const GpuTrace *t){ return t->pc_paths; }

WGPUBuffer gpu_trace_resolve(GpuTrace *t, WGPUCommandEncoder enc, WGPUBuffer base, int cache){
    int use_cache = cache && t->grid && t->pc_paths > 0;
    double we = t->caus_passes ? (use_cache ? t->caus_passes / (t->caus_passes + 8.0) : 1.0) : 0.0;
    struct { uint32_t dims[4]; float k[4]; } R = {{(uint32_t)t->w, (uint32_t)t->h, use_cache ? 1u : 0u, 0},
        {t->caus_passes ? (float)(1.0 / (4294967296.0 * t->caus_passes)) : 0.0f, (float)we,
         use_cache ? (float)(1.0 / (t->pc_paths * 256.0)) : 0.0f, t->cell}};
    wgpuQueueWriteBuffer(t->g->queue, t->rs_params, 0, &R, sizeof(R));
    WGPUBuffer b[6] = {t->rs_params, base, t->caus, t->final, t->grid ? t->grid : t->caus, t->acc[t->cur]};
    WGPUBindGroup bg = gpu_bind_cached_buffers(t->g, &t->bc_resolve, t->rs_bgl, b, 6);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, t->rs_pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    gpu_dispatch(pass, (size_t)t->w * t->h);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    return t->final;
}

static void put_cam(float *d, const Camera *c){
    put(d, c->origin, 0); put(d + 4, c->upper_left, 0); put(d + 8, c->horizontal, 0); put(d + 12, c->vertical, 0);
}

void gpu_trace_begin_image(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                           int history, int nvolumes,
                           WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    Gpu *g = t->g;
    gpu_trace_build(t, enc);         // the top level over the instances (they may have moved)
    if (t->planes != 5){             // no guide on this GPU: no reprojection either
        gpu_trace_clear(t, enc);
        t->cam = tp->cam;
        t->has_image = 1;
        return;
    }
    int reproject = history > 0 && t->has_image;
    if (reproject && !t->acc[1 - t->cur])
        t->acc[1 - t->cur] = gpu_buffer(g, "path tracer samples (previous image)", (size_t)t->w * t->h * 16 * t->planes,
                                        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    if (reproject) t->cur = 1 - t->cur;
    write_params(t, t->params, s, tp, 0, t->h, nvolumes, 0, 1, 0);
    dispatch_ex(t, enc, t->gbuf_pipe, (size_t)t->w * t->h, 1, frame_u, dens, bricks, vstat, 3);
    if (reproject){
        struct { float cams[8][4]; uint32_t dims[4]; } R;
        memset(&R, 0, sizeof(R));
        put_cam(R.cams[0], &t->cam);
        put_cam(R.cams[4], &tp->cam);
        R.dims[0] = (uint32_t)t->w; R.dims[1] = (uint32_t)t->h; R.dims[2] = (uint32_t)history;
        wgpuQueueWriteBuffer(g->queue, t->rp_params, 0, &R, sizeof(R));
        WGPUBuffer b[3] = {t->rp_params, t->acc[1 - t->cur], t->acc[t->cur]};
        WGPUBindGroup bg = gpu_bind_cached_buffers(g, &t->bc_reproject, t->rp_bgl, b, 3);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        wgpuComputePassEncoderSetPipeline(pass, t->rp_pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        gpu_dispatch(pass, (size_t)t->w * t->h);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    t->cam = tp->cam;
    t->has_image = 1;
}

GpuTimer *gpu_trace_timer(GpuTrace *t){ return t ? t->timer : NULL; }

void gpu_trace_destroy(GpuTrace *t){
    if (!t) return;
    if (t->check){
        for (int i = 0; i < 8; i++)
            if (t->ck_builds[i]) printf("GPU BVH check, build %d: %d items, cost %.3f (mean of %ld), depth %d\n", i, t->ck_n[i], t->ck_cost[i] / t->ck_builds[i], t->ck_builds[i], t->ck_depth[i]);
        printf("GPU BVH check: %ld builds were not one tree over their items\n", t->ck_bad);
    }
    if (t->timer && gpu_timing_report(t->g)){
        for (int k = 0; k < NTS; k++){
            double ms = gpu_timer_ms_per_unit(t->timer, k);
            if (ms > 0) printf("GPU time, %s: %.3f ms per %s\n", TS_NAMES[k], ms,
                               k == TS_TRACE || k >= TS_WF0 ? "pass (1 sample per pixel, the whole image)" : k == TS_LIGHT ? "pass" :
                               k == TS_GUIDE ? "image" : "frame with builds");
        }
        gpu_timer_destroy(t->timer);
    }
    for (int k = 0; k < NWF; k++){
        if (t->wf_pipe[k]) wgpuComputePipelineRelease(t->wf_pipe[k]);
        if (t->wf_bgl[k]) wgpuBindGroupLayoutRelease(t->wf_bgl[k]);
    }
    for (int k = 0; k < 3; k++) gpu_bind_cache_clear(&t->bc_trace[k]);
    for (int k = 0; k < 8; k++){ gpu_bind_cache_clear(&t->bc_build[k]); gpu_bind_cache_clear(&t->bc_ploc[k]); }
    for (int k = 0; k < 6; k++) gpu_bind_cache_clear(&t->bc_wf[k]);
    gpu_bind_cache_clear(&t->bc_resolve); gpu_bind_cache_clear(&t->bc_reproject); gpu_bind_cache_clear(&t->bc_decay);
    WGPUBuffer wfb[] = {t->wf, t->wfc, t->wfa, t->wf_params};
    for (int i = 0; i < 4; i++) if (wfb[i]){ wgpuBufferDestroy(wfb[i]); wgpuBufferRelease(wfb[i]); }
    if (t->ck_buf){ wgpuBufferDestroy(t->ck_buf); wgpuBufferRelease(t->ck_buf); }
    WGPUBuffer bufs[] = {t->params, t->rp_params, t->blue, t->acc[0], t->acc[1], t->nodes, t->aux, t->prims, t->tris,
                         t->caus, t->rs_params, t->final, t->lt_params, t->grid, t->dc_params, t->pc_params,
                         t->bd_keys, t->bd_flags, t->bd_boxes, t->bd_par, t->pl_buf, t->pl_args};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    WGPUComputePipeline pipes[] = {t->pipe, t->pipe_sky, t->wf_shade_sky, t->gbuf_pipe, t->adapt_pipe, t->rp_pipe, t->lt_pipe, t->rs_pipe, t->dc_pipe,
                                   t->bd_morton, t->bd_sort, t->bd_hier, t->bd_leaves, t->bd_up,
                                   t->pl_init, t->pl_nn, t->pl_mark, t->pl_top, t->pl_scatter, t->pl_small};
    for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); i++) if (pipes[i]) wgpuComputePipelineRelease(pipes[i]);
    WGPUBindGroupLayout bgls[] = {t->bgl, t->rp_bgl, t->lt_bgl, t->rs_bgl, t->dc_bgl, t->bd_bgl, t->pl_bgl, t->pl_bgl_args};
    for (size_t i = 0; i < sizeof(bgls) / sizeof(bgls[0]); i++) if (bgls[i]) wgpuBindGroupLayoutRelease(bgls[i]);
    if (t->sky_view) wgpuTextureViewRelease(t->sky_view);
    if (t->skyc_view) wgpuTextureViewRelease(t->skyc_view);
    if (t->sky_tex){ wgpuTextureDestroy(t->sky_tex); wgpuTextureRelease(t->sky_tex); }
    if (t->skyc_tex){ wgpuTextureDestroy(t->skyc_tex); wgpuTextureRelease(t->skyc_tex); }
    gpu_textures_free(&t->gt);
    if (t->txt){ wgpuBufferDestroy(t->txt); wgpuBufferRelease(t->txt); }
#ifdef LR_VK
    for (int m = 0; m < t->hw_nmesh; m++) lrBlasRelease(t->hw_mesh[m]);
    free(t->hw_mesh); free(t->hw_mesh_dirty);
    lrBlasRelease(t->hw_tri); lrBlasRelease(t->hw_prim); lrBlasRelease(t->hw_unit); lrTlasRelease(t->hw_tlas);
    WGPUBuffer hb[] = {t->hw_vtx, t->hw_boxes, t->hw_rec, t->hw_unit_buf};
    for (int i = 0; i < 4; i++) if (hb[i]){ wgpuBufferDestroy(hb[i]); wgpuBufferRelease(hb[i]); }
#endif
    free(t->mc);
    free(t->sc_rec); free(t->sc_box); free(t->sc_nd); free(t->sc_ax); free(t->sc_hw);
    free(t->arena_boxes);
    free(t->arena_par);
    if (t->own_built) bvh_free(&t->own);
    free(t->prim_copy);
    free(t);
}

#else
typedef int trace_gpu_unused;
#endif
