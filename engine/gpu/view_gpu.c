#include "gpu/view_gpu.h"

#include <stdlib.h>

#ifndef LR_GPU
struct GpuView { int unused; };
GpuView *gpu_view_create(Gpu *g, int width, int height){ (void)g; (void)width; (void)height; return NULL; }
void gpu_view_present(GpuView *v, const Framebuffer *fb, const Scene *s, const RenderParams *p,
                      const MarchParams *mp, int volumes){ (void)v; (void)fb; (void)s; (void)p; (void)mp; (void)volumes; }
void gpu_view_destroy(GpuView *v){ (void)v; }
int gpu_view_ready(GpuView *v){ (void)v; return -1; }
int gpu_view_trace_compiling(const GpuView *v){ (void)v; return 0; }
void gpu_view_gpu_totals(GpuView *v, double ms[8]){ (void)v; for (int k = 0; k < 8; k++) ms[k] = 0; }
void gpu_view_hud(GpuView *v, const uint32_t *argb, int w, int h, int x, int y){ (void)v; (void)argb; (void)w; (void)h; (void)x; (void)y; }
int gpu_view_capture(GpuView *v, GpuCaptureFn fn, void *user){ (void)v; (void)fn; (void)user; return 0; }
int gpu_view_present_scene(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    (void)v; (void)s; (void)p; (void)mp; (void)volumes;
    return 0;
}
int gpu_view_volumes_fit(GpuView *v, const Scene *s, const MarchParams *mp){ (void)v; (void)s; (void)mp; return 0; }
int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp){ (void)v; (void)s; (void)tp; return 0; }
int gpu_view_trace_failed(const GpuView *v){ (void)v; return 1; }
void gpu_view_trace_prepare(GpuView *v){ (void)v; }
int gpu_view_trace_move(GpuView *v, const Scene *s, const GpuTraceParams *tp, int history){ (void)v; (void)s; (void)tp; (void)history; return 0; }
int gpu_view_trace_busy(const GpuView *v){ (void)v; return 0; }
int gpu_view_trace_reads_tlas(const GpuView *v, const Scene *s){ (void)v; (void)s; return 0; }
void gpu_view_trace_limit(GpuView *v, int spp){ (void)v; (void)spp; }
void gpu_view_trace_budget(GpuView *v, double ms){ (void)v; (void)ms; }
void gpu_view_trace_budget_live(GpuView *v, double ms){ (void)v; (void)ms; }
void gpu_view_trace_budget_pass(GpuView *v, double max_ms){ (void)v; (void)max_ms; }
int gpu_view_trace_read(GpuView *v, GpuTraceReadFn fn, void *user){ (void)v; (void)fn; (void)user; return 0; }
int gpu_view_trace_more(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp){ (void)v; (void)s; (void)p; (void)mp; return 0; }
int gpu_view_trace_passes(const GpuView *v){ (void)v; return 0; }
double gpu_view_trace_work(const GpuView *v){ (void)v; return 0; }
double gpu_view_trace_pass_ms(const GpuView *v){ (void)v; return 0; }
void gpu_view_trace_band(const GpuView *v, int *rows, int *reps, double *budget_ms){ (void)v; *rows = *reps = 0; *budget_ms = 0; }
int gpu_view_trace_denoise(GpuView *v, int on){ (void)v; (void)on; return 0; }
void gpu_view_trace_denoise_moving(GpuView *v, int passes){ (void)v; (void)passes; }
void gpu_view_aa(GpuView *v, int on){ (void)v; (void)on; }
void gpu_view_trace_wavefront(GpuView *v, int on){ (void)v; (void)on; }
void gpu_view_trace_raster_guide(GpuView *v, int mode){ (void)v; (void)mode; }
void gpu_view_trace_gradients(GpuView *v, int history_max){ (void)v; (void)history_max; }
void gpu_view_trace_edited(GpuView *v, int on){ (void)v; (void)on; }
int gpu_view_trace_has_gradients(const GpuView *v){ (void)v; return 0; }
int gpu_view_trace_raster_guided(const GpuView *v){ (void)v; return 0; }
int gpu_view_trace_planes_read(GpuView *v, int first, int n, float *out){ (void)v; (void)first; (void)n; (void)out; return 0; }
double gpu_view_gpu_ms(GpuView *v, int what){ (void)v; (void)what; return 0; }
void gpu_view_gpu_ms_reset(GpuView *v){ (void)v; }
double gpu_view_bench_denoise(GpuView *v, int n){ (void)v; (void)n; return -1; }
int gpu_view_present_traced(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    (void)v; (void)s; (void)p; (void)mp; (void)volumes;
    return 0;
}
#else

#include <stdio.h>
#include <string.h>
#include "base/clock.h"
#include "base/envvar.h"
#include "scene/volume.h"
#include "scene/env.h"
#include "gpu/gpu_internal.h"

// ---- the shaders -----------------------------------------------------------------------------
//
// The same model as raster/volume_march.c, line for line: density by trilinear interpolation
// (0 outside the grid), the majorant bricks to hurry through empty space, the light grid of
// optical depths to the lamp, the lamp's irradiance L A cos / (d^2 + A/pi), 3 octaves of
// multiple scattering, the walls' ambient light, energy-conserving steps.

#define TYPES GPU_WGSL_TYPES
#define COMMON GPU_WGSL_MEDIA
GPU_WGSL_EXPORT(GPU_WGSL_TYPES, types);

GPU_WGSL_EXPORT(GPU_WGSL_MEDIA, media);

GPU_WGSL(LIGHT, view_light);

// the frame the media are drawn over: a framebuffer of the CPU renderers (two buffers)...
GPU_WGSL(FRAME_CPU, view_frame_cpu);
// ...or what the GPU rasterizer drew (two textures)
GPU_WGSL(FRAME_GPU, view_frame_gpu);

GPU_WGSL(COMPOSITE, view_composite);

/* The text over the image (the frame rate; app/hud.c draws it into argb pixels): a rectangle
   of the canvas, its pixels from the buffer, blended over what the composite pass drew. */
GPU_WGSL(HUD, view_hud);

/* Anti-aliasing of rasterized frames (gpu_view_aa): FXAA over what the composite drew. */
GPU_WGSL(FXAA, view_fxaa);

/* What volume_update does on the CPU, for media whose densities live on the GPU (and, the
   same way, for all the others): the majorant of every brick (with the voxel of interpolation
   around it), and, with atomics, the box of the voxels with density > 0 and their maximum. */
GPU_WGSL(STATS, view_stats);

GPU_WGSL(FINISH, view_finish);

// ---- the CPU side ---------------------------------------------------------------------------

// the uniform block, laid out as WGSL lays out struct Frame (16-byte rows)
typedef struct {
    float lo[4], hi[4], alo[4], ahi[4], albedo[4];
    uint32_t dims[4], bdims[4], ldims[4];
    float p[4];
} GVol;
typedef struct {
    float right[4], up[4], back[4], eye[4], proj[4], size[4];
    float lamp_pos[4], lamp_n[4], lamp_L[4], amb[4];
    GVol vols[GPU_MAX_VOLUMES];
} GFrame;

struct GpuView {
    Gpu *g;
    int w, h;
    uintptr_t live;          // its ticket in the list of the objects alive (its callbacks' userdata2)
    int wavefront;           // gpu_view_trace_wavefront
    WGPUBuffer frame_u, sel_u[GPU_MAX_VOLUMES];
    WGPUBuffer dens, bricks, light, color, depth;
    size_t dens_cap, bricks_cap, light_cap;
    WGPUBuffer vstat, acc;      // per medium: active box and max (VStat), and the atomics that make it
    WGPUComputePipeline light_pipe, stats_pipe, finish_pipe;
    GpuPipeReq *preq[7];        // the frame's pipelines when made asynchronously (gpu_set_async_views)
    int pst[7];                 // and where each is: 0 being made, 1 made, -1 failed
    WGPURenderPipeline comp_pipe, comp_gpu_pipe;   // over a CPU framebuffer / over the GPU rasterizer's frame
    // the text over the image (gpu_view_hud): its pixels, where, and the pass that blends it
    WGPURenderPipeline hud_pipe;
    WGPUBindGroupLayout hud_bgl;
    WGPUBuffer hud_u, hud_px;
    // anti-aliasing (gpu_view_aa): the composite draws into aa_tex, FXAA from it onto the canvas
    int aa;
    WGPURenderPipeline fxaa_pipe;
    WGPUBindGroupLayout fxaa_bgl;
    WGPUTexture aa_tex;
    WGPUTextureView aa_view;
    // the bind groups of every frame, kept (gpu_bind_cached)
    GpuBindCache bc_comp, bc_comp_gpu, bc_hud, bc_fxaa, bc_stats[GPU_MAX_VOLUMES], bc_finish[GPU_MAX_VOLUMES], bc_light[GPU_MAX_VOLUMES];
    size_t hud_cap;
    int hud_w;                  // its width (0: none)
    // explicit: "auto" would drop the bindings a shader does not read
    WGPUBindGroupLayout light_bgl, comp_bgl, comp_gpu_bgl, stats_bgl, finish_bgl;
    GpuRaster *raster;          // the scene's surfaces drawn on the GPU (created when first used)
    int raster_failed;          // it cannot be made on this GPU (its pipelines): for good
    // ... or it could not draw this scene (a mesh too big for a buffer): not asked again until
    // the scene is another (scene_sig); the same for the path tracer (t_unfit)
    int r_unfit, t_unfit;
    uint64_t r_unfit_sig, t_unfit_sig;
    int vol_warned;             // gpu_view_volumes_fit said the media do not fit
    // the path tracer (created when first used) and the image it is making
    GpuTrace *trace;
    int trace_failed, show_traced;   // (trace_failed: its kernels cannot be made on this GPU: for good)
    GpuTraceParams tparams;
    int t_prep;                 // the image is still to begin: 1 from nothing, 2 from the previous one (reprojected)
    int t_history;              // with 2: samples kept per pixel at most
    int t_hmax, grad_next;      // ... and up to how many the temporal gradients may keep (0: none; gpu_view_trace_gradients)
    int t_hedit, grad_edit;     // ... and whether the move is an edit's (gpu_view_trace_edited)
    int t_hist;                 // a whole pass done since the image started: every later one
                                // begins from it (reprojected), the raster under it not needed
    int t_moved, t_live;        // moved since the last band / the band running follows a move
    int t_sub_rows, t_sub_reps; // the band running
    int t_sub_prep;             // and it begins an image (its time includes the whole image's guide)
    double t_gpu_pass_ms;       // a whole pass, as the last band took
    int t_y, t_rows, t_reps;    // next row; rows per band and passes per submit (both adapt to the GPU)
    int t_limit;                // gpu_view_trace_limit: the most passes of an image (0: no limit)
    double t_budget_ms;         // gpu_view_trace_budget: GPU time per frame (0: bands of 12-45 ms)
    double t_pass_max;          // gpu_view_trace_budget_pass: at least a whole pass a frame, up to this (0: no)
    double t_live_ms;           // gpu_view_trace_budget_live: the budget while things move (0: t_budget_ms)
    int t_ready_seen;           // its pipelines were ready once (before: still compiling, in a browser)
    int rguide;                 // a new image's guide: 0 traced, 1 the cheaper of the two, 2 rasterized where it can
    int t_rguided;              // ... and the last one's was rasterized
    // rguide 1: each way's GPU time per image (its timers), measured in turns; the way chosen
    int rg_way, rg_phase, rg_images;
    int rg_idle;                // images in a row whose guide was traced (RG_IDLE: the rasterizer's targets let go)
    double rg_sum[2];
    int rg_n[2];
    int t_gpu_timed;            // t_gpu_pass_ms comes from the GPU's timer
    int t_passes;               // samples per pixel submitted
    double t_work;              // passes' worth of work done, all images (gpu_view_trace_work)
    int t_done, t_done_next;    // finished; and what that becomes when the running band ends
    int t_inflight;             // a band is running
    int t_bands;                // bands finished since the image started
    uint32_t t_index;           // the sample (of the sequence: base/sampler.h) the next pass takes
    // the denoiser (created when first used), and the band count of the image it holds
    GpuDenoise *den;
    int den_failed, denoise, den_bands;
    int den_move;               // its a-trous passes while things move (0: all; gpu_view_trace_denoise_moving)
    double t_submit_ms, t_pass_t0, t_pass_ms;
    // reading the samples back
    WGPUBuffer t_rb;
    int t_rb_pending;
    GpuTraceReadFn t_read_fn;
    void *t_read_user;
    // media whose densities are already on the GPU (a GPU fluid's cloud): copied, not uploaded
    GpuDensity *src[GPU_MAX_SOURCES];
    int nsrc;
    // what is on the GPU now
    uint32_t vol_ids[GPU_MAX_VOLUMES];     // (Volume.id: not its address, which a later one can be given)
    uint32_t versions[GPU_MAX_VOLUMES];
    int nvols, light_res;
    float lamp[3];
    GVol gv[GPU_MAX_VOLUMES];
    int light_valid;
    float *depth_f;     // the double build: the z-buffer as 32-bit floats (it stores doubles)
    // screenshot: the next frame is also drawn into tex, copied into readback, mapped
    GpuCaptureFn cap_fn;
    void *cap_user;
    int cap_pending;    // 1 = draw it next frame, 2 = waiting for the map
    WGPUTexture cap_tex;
    WGPUBuffer cap_buf;
    uint32_t cap_row;   // bytes per row in cap_buf (a multiple of 256)
};

// a "work done" callback can arrive after gpu_view_destroy (creation is refused when the list
// of the objects alive is full: one left out would have its callbacks ignored)
// (u2: the ticket the callback was given with the view; the address alone could be a later
// view's, made where this one was: a window resized)
static int view_is_live(const GpuView *v, void *u2){ return gpu_live_has(GPU_LIVE_VIEW, v, (uintptr_t)u2); }

// What a scene is, for "the scene that did not fit": a number that changes when its geometry,
// instances, textures, media or sky do (their versions and counts; not the camera, not what the
// physics moves without adding anything)
static uint64_t scene_sig(const Scene *s){
    uint64_t h = 1469598103934665603ull;
#define SIG(x) h = (h ^ (uint64_t)(x)) * 1099511628211ull
    SIG(s->tris_version); SIG(s->ntris); SIG(s->nnormals); SIG(s->nprims);
    SIG(s->inst_version); SIG(s->ninsts); SIG(s->tex_version); SIG(s->ntexmaps);
    SIG(s->nmeshes); SIG(s->nvolumes); SIG(s->env ? s->env->id : 0);
    for (int m = 0; m < s->nmeshes; m++){ SIG(s->meshes[m].id); SIG(s->meshes[m].version); SIG(s->meshes[m].ntris); }
    for (int k = 0; k < s->nvolumes; k++){ SIG(s->volumes[k]->id); SIG(s->volumes[k]->nx); SIG(s->volumes[k]->ny); SIG(s->volumes[k]->nz); }
#undef SIG
    return h;
}

static void v4(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

static WGPUPipelineLayout make_pipeline_layout(Gpu *g, WGPUBindGroupLayout bgl){
    WGPUPipelineLayoutDescriptor d;
    memset(&d, 0, sizeof(d));
    d.bindGroupLayoutCount = 1;
    d.bindGroupLayouts = &bgl;
    return wgpuDeviceCreatePipelineLayout(g->device, &d);
}

GpuView *gpu_view_create(Gpu *g, int width, int height){
    if (!g || g->state != GPU_READY || (!g->surface && !g->offscreen)) return NULL;
    GpuView *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    if (!(v->live = gpu_live_add(GPU_LIVE_VIEW, v))){ free(v); return NULL; }
    v->g = g;
    v->w = width;
    v->h = height;
    v->rguide = lr_env_int("LR_GPU_RGUIDE", 1);
    v->rg_way = 1;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    v->frame_u = gpu_buffer(g, "frame", sizeof(GFrame), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    for (int k = 0; k < GPU_MAX_VOLUMES; k++){
        v->sel_u[k] = gpu_buffer(g, "light volume", 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        uint32_t sel[4] = {(uint32_t)k, 0, 0, 0};
        wgpuQueueWriteBuffer(g->queue, v->sel_u[k], 0, sel, sizeof(sel));
    }
    v->color = gpu_buffer(g, "color", (size_t)width * height * 4, st);
    v->depth = gpu_buffer(g, "depth", (size_t)width * height * 4, st);
    gpu_buffer_reserve(g, &v->dens, &v->dens_cap, 1024, st, "density");
    gpu_buffer_reserve(g, &v->bricks, &v->bricks_cap, 1024, st, "bricks");
    gpu_buffer_reserve(g, &v->light, &v->light_cap, 1024, WGPUBufferUsage_Storage, "light grid");
    v->vstat = gpu_buffer(g, "media boxes", GPU_MAX_VOLUMES * 48, WGPUBufferUsage_Storage);
    v->acc = gpu_buffer(g, "media box atomics", GPU_MAX_VOLUMES * 32, st);
    if (sizeof(real) != sizeof(float) && !(v->depth_f = malloc((size_t)width * height * sizeof(float)))){
        gpu_view_destroy(v);
        return NULL;
    }

    const char *const light_src[] = {TYPES, COMMON, LIGHT}, *const comp_src[] = {GPU_WGSL_UTIL, TYPES, COMMON, FRAME_CPU, COMPOSITE},
                      *const compg_src[] = {GPU_WGSL_UTIL, TYPES, COMMON, FRAME_GPU, COMPOSITE}, *const stats_src[] = {TYPES, STATS},
                      *const finish_src[] = {TYPES, FINISH};
    const struct { const char *label; const char *const *parts; int n; } mods[5] = {
        {"light grid", light_src, 3}, {"composite", comp_src, 5}, {"composite (GPU frame)", compg_src, 5},
        {"media stats", stats_src, 2}, {"media boxes", finish_src, 2}};
    WGPUShaderModule mm[5];
    for (int k = 0; k < 5; k++){
        char *src = gpu_wgsl_join(mods[k].parts, mods[k].n);
        mm[k] = src ? gpu_shader(g, mods[k].label, src) : NULL;
        free(src);
    }
    WGPUShaderModule ml = mm[0], mc = mm[1], mcg = mm[2], ms = mm[3], mf = mm[4];

    v->light_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRWUR");
    v->comp_bgl = gpu_layout(g, WGPUShaderStage_Fragment, "URRRRRR");
    {   // the same, with bindings 4 and 5 as the rasterizer's color and depth textures
        WGPUBindGroupLayoutEntry e[8];
        memset(e, 0, sizeof(e));
        for (int i = 0; i < 8; i++){
            e[i].binding = (uint32_t)i;
            e[i].visibility = WGPUShaderStage_Fragment;
            if (i == 4 || i == 5){
                e[i].texture.sampleType = i == 4 ? WGPUTextureSampleType_Float : WGPUTextureSampleType_Depth;
                e[i].texture.viewDimension = WGPUTextureViewDimension_2D;
            } else e[i].buffer.type = i == 0 ? WGPUBufferBindingType_Uniform   // (as comp_bgl; 7: the path tracer's samples)
                                             : WGPUBufferBindingType_ReadOnlyStorage;
        }
        WGPUBindGroupLayoutDescriptor d;
        memset(&d, 0, sizeof(d));
        d.entryCount = 8;
        d.entries = e;
        v->comp_gpu_bgl = wgpuDeviceCreateBindGroupLayout(g->device, &d);
    }
    WGPUPipelineLayout comp_gpu_pl = make_pipeline_layout(g, v->comp_gpu_bgl);
    v->stats_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URWWU");
    v->finish_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URWU");
    WGPUPipelineLayout light_pl = make_pipeline_layout(g, v->light_bgl);
    WGPUPipelineLayout comp_pl = make_pipeline_layout(g, v->comp_bgl);
    WGPUPipelineLayout stats_pl = make_pipeline_layout(g, v->stats_bgl);
    WGPUPipelineLayout finish_pl = make_pipeline_layout(g, v->finish_bgl);

    WGPUComputePipelineDescriptor cd;
    memset(&cd, 0, sizeof(cd));
    cd.label = gpu_str("light grid");
    cd.layout = light_pl;
    cd.compute.module = ml;
    cd.compute.entryPoint = gpu_str("light_grid");
    v->preq[0] = gpu_compute_desc_async(g, &cd, &v->light_pipe, &v->pst[0]);
    cd.label = gpu_str("media stats");
    cd.layout = stats_pl;
    cd.compute.module = ms;
    cd.compute.entryPoint = gpu_str("brick_stats");
    v->preq[1] = gpu_compute_desc_async(g, &cd, &v->stats_pipe, &v->pst[1]);
    cd.label = gpu_str("media boxes");
    cd.layout = finish_pl;
    cd.compute.module = mf;
    cd.compute.entryPoint = gpu_str("vol_finish");
    v->preq[2] = gpu_compute_desc_async(g, &cd, &v->finish_pipe, &v->pst[2]);

    WGPUColorTargetState target;
    memset(&target, 0, sizeof(target));
    target.format = g->surface_format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs;
    memset(&fs, 0, sizeof(fs));
    fs.module = mc;
    fs.entryPoint = gpu_str("fs");
    fs.targetCount = 1;
    fs.targets = &target;
    WGPURenderPipelineDescriptor rd;
    memset(&rd, 0, sizeof(rd));
    rd.label = gpu_str("composite");
    rd.layout = comp_pl;
    rd.vertex.module = mc;
    rd.vertex.entryPoint = gpu_str("vs");
    rd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rd.multisample.count = 1;
    rd.multisample.mask = 0xFFFFFFFFu;
    rd.fragment = &fs;
    v->preq[3] = gpu_render_async(g, &rd, &v->comp_pipe, &v->pst[3]);
    rd.label = gpu_str("composite (GPU frame)");
    rd.layout = comp_gpu_pl;
    rd.vertex.module = mcg;
    fs.module = mcg;
    v->preq[4] = gpu_render_async(g, &rd, &v->comp_gpu_pipe, &v->pst[4]);
    {   // the text over the image: blended by its alpha
        WGPUShaderModule mh = gpu_shader(g, "text over the image", HUD);
        v->hud_bgl = gpu_layout(g, WGPUShaderStage_Vertex | WGPUShaderStage_Fragment, "UR");
        WGPUPipelineLayout hud_pl = make_pipeline_layout(g, v->hud_bgl);
        WGPUBlendState blend;
        memset(&blend, 0, sizeof(blend));
        blend.color.operation = blend.alpha.operation = WGPUBlendOperation_Add;
        blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
        blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend.alpha.srcFactor = WGPUBlendFactor_One;
        blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        target.blend = &blend;
        rd.label = gpu_str("text over the image");
        rd.layout = hud_pl;
        rd.vertex.module = mh;
        fs.module = mh;
        v->preq[5] = gpu_render_async(g, &rd, &v->hud_pipe, &v->pst[5]);
        target.blend = NULL;
        v->hud_u = gpu_buffer(g, "text over the image: where", 32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        wgpuShaderModuleRelease(mh);
        wgpuPipelineLayoutRelease(hud_pl);
    }
    {   // FXAA: the composite's image (a texture) onto the canvas
        WGPUShaderModule mx = gpu_shader(g, "anti-aliasing", FXAA);
        v->fxaa_bgl = gpu_layout(g, WGPUShaderStage_Fragment, "T");
        WGPUPipelineLayout fxaa_pl = make_pipeline_layout(g, v->fxaa_bgl);
        rd.label = gpu_str("anti-aliasing");
        rd.layout = fxaa_pl;
        rd.vertex.module = mx;
        fs.module = mx;
        v->preq[6] = gpu_render_async(g, &rd, &v->fxaa_pipe, &v->pst[6]);
        wgpuShaderModuleRelease(mx);
        wgpuPipelineLayoutRelease(fxaa_pl);
    }
    wgpuShaderModuleRelease(mcg);
    wgpuPipelineLayoutRelease(comp_gpu_pl);
    wgpuShaderModuleRelease(ml);
    wgpuShaderModuleRelease(mc);
    wgpuShaderModuleRelease(ms);
    wgpuShaderModuleRelease(mf);
    wgpuPipelineLayoutRelease(light_pl);
    wgpuPipelineLayoutRelease(comp_pl);
    wgpuPipelineLayoutRelease(stats_pl);
    wgpuPipelineLayoutRelease(finish_pl);
    // made asynchronously: the GPU rasterizer too, now, so that gpu_view_ready waits for it
    if (gpu_async_views() && !(v->raster = gpu_raster_create(g, v->w, v->h))) v->raster_failed = 1;
    return v;
}

int gpu_view_ready(GpuView *v){
    if (!v) return -1;
    for (int k = 0; k < 7; k++){
        if (v->pst[k] < 0) return -1;
        if (v->pst[k] == 0) return 0;
    }
    if (v->raster){
        int r = gpu_raster_ready(v->raster);
        if (r == 0) return 0;
        if (r < 0){ gpu_raster_destroy(v->raster); v->raster = NULL; v->raster_failed = 1; }   // (the CPU rasterizes)
    }
    return 1;
}

static WGPUBindGroupEntry entry(uint32_t binding, WGPUBuffer b){
    WGPUBindGroupEntry e;
    memset(&e, 0, sizeof(e));
    e.binding = binding;
    e.buffer = b;
    e.size = WGPU_WHOLE_SIZE;
    return e;
}

// densities, bricks and light grid layout of the scene's media; uploads what changed.
// Returns 1 if the light grids must be recomputed.
static WGPUBuffer gpu_source(const GpuView *v, const Volume *vol){
    for (int i = 0; i < v->nsrc; i++) if (v->src[i]->volume == vol->id) return v->src[i]->buf;
    return NULL;
}

void gpu_density_release(GpuDensity *d){
    if (d && --d->refs <= 0) free(d);
}

// the fluids that are gone leave the list; 1 if any did (their media are the CPU's again)
static int sources_prune(GpuView *v){
    int gone = 0;
    for (int i = 0; i < v->nsrc; ){
        if (v->src[i]->buf){ i++; continue; }
        gpu_density_release(v->src[i]);
        v->src[i] = v->src[--v->nsrc];
        gone = 1;
    }
    return gone;
}

// the cells of the first n media: densities, bricks and (with mp) light grids
static void media_cells(const Scene *s, int n, const MarchParams *mp, size_t *nd, size_t *nb, size_t *nl){
    *nd = *nb = *nl = 0;
    for (int k = 0; k < n; k++){
        const Volume *vol = s->volumes[k];
        *nd += (size_t)vol->nx * vol->ny * vol->nz;
        *nb += (size_t)vol->bnx * vol->bny * vol->bnz;
        if (mp){
            int ld[3];
            march_light_dims(vol, mp->light_res, ld);
            *nl += (size_t)ld[0] * ld[1] * ld[2];
        }
    }
}

int gpu_view_volumes_fit(GpuView *v, const Scene *s, const MarchParams *mp){
    if (!v || !s) return 0;
    const int n = s->nvolumes < GPU_MAX_VOLUMES ? s->nvolumes : GPU_MAX_VOLUMES;
    size_t nd, nb, nl;
    media_cells(s, n, mp, &nd, &nb, &nl);
    size_t most = nd > nb ? nd : nb;
    if (nl > most) most = nl;
    if ((uint64_t)most * 4 <= v->g->max_storage) return 1;
    if (!v->vol_warned++)
        fprintf(stderr, "gpu: the media need a buffer of %.0f MB, more than this GPU's %.0f MB: marched on the CPU\n",
                (double)most * 4 / 1048576, (double)v->g->max_storage / 1048576);
    return 0;
}

static int upload_volumes(GpuView *v, const Scene *s, const MarchParams *mp, MarchLamp *lamp, WGPUCommandEncoder enc){
    Gpu *g = v->g;
    int n = s->nvolumes < GPU_MAX_VOLUMES ? s->nvolumes : GPU_MAX_VOLUMES;
    if (sources_prune(v)) v->nvols = -1;      // copy everything again
    int changed = n != v->nvols || mp->light_res != v->light_res ||
                  lamp->pos.x != v->lamp[0] || lamp->pos.y != v->lamp[1] || lamp->pos.z != v->lamp[2];
    for (int k = 0; k < n && !changed; k++)
        if (v->vol_ids[k] != s->volumes[k]->id || v->versions[k] != s->volumes[k]->version) changed = 1;
    if (!changed) return !v->light_valid;
    size_t nd, nb, nl;
    media_cells(s, n, mp, &nd, &nb, &nl);
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    int fresh = gpu_buffer_reserve(g, &v->dens, &v->dens_cap, nd * 4, st, "density");
    gpu_buffer_reserve(g, &v->bricks, &v->bricks_cap, nb * 4, st, "bricks");
    gpu_buffer_reserve(g, &v->light, &v->light_cap, nl * 4, WGPUBufferUsage_Storage, "light grid");
    uint32_t od = 0, ob = 0, ol = 0;
    for (int k = 0; k < n; k++){
        const Volume *vol = s->volumes[k];
        GVol *gv = &v->gv[k];
        int ld[3];
        march_light_dims(vol, mp->light_res, ld);
        Vec3 size = v_sub(vol->bounds.hi, vol->bounds.lo);
        real voxel = r_fmin(size.x / vol->nx, r_fmin(size.y / vol->ny, size.z / vol->nz));
        v4(gv->lo, vol->bounds.lo, 0);
        v4(gv->hi, vol->bounds.hi, 0);
        v4(gv->alo, vol->active.lo, 0);
        v4(gv->ahi, vol->active.hi, 0);
        v4(gv->albedo, vol->albedo, (float)vol->sigma_t);
        gv->dims[0] = (uint32_t)vol->nx; gv->dims[1] = (uint32_t)vol->ny; gv->dims[2] = (uint32_t)vol->nz; gv->dims[3] = od;
        gv->bdims[0] = (uint32_t)vol->bnx; gv->bdims[1] = (uint32_t)vol->bny; gv->bdims[2] = (uint32_t)vol->bnz; gv->bdims[3] = ob;
        gv->ldims[0] = (uint32_t)ld[0]; gv->ldims[1] = (uint32_t)ld[1]; gv->ldims[2] = (uint32_t)ld[2]; gv->ldims[3] = ol;
        int empty = vol->active.hi.x < vol->active.lo.x || vol->max_density <= 0;
        gv->p[0] = (float)vol->g;
        gv->p[1] = (float)(voxel * (mp->step_voxels > 0 ? mp->step_voxels : 1));
        gv->p[2] = empty ? 0.0f : (float)vol->max_density;
        gv->p[3] = (float)voxel;
        size_t cnt = (size_t)vol->nx * vol->ny * vol->nz;
        if (fresh || v->vol_ids[k] != vol->id || v->versions[k] != vol->version || n != v->nvols){
            WGPUBuffer from = gpu_source(v, vol);
            if (from) wgpuCommandEncoderCopyBufferToBuffer(enc, from, 0, v->dens, (uint64_t)od * 4, cnt * 4);
            else wgpuQueueWriteBuffer(g->queue, v->dens, (uint64_t)od * 4, vol->density, cnt * 4);
        }
        // (the bricks, the active box and the maximum are computed on the GPU: STATS)
        od += (uint32_t)cnt;
        ob += (uint32_t)(vol->bnx * vol->bny * vol->bnz);
        ol += (uint32_t)(ld[0] * ld[1] * ld[2]);
        v->vol_ids[k] = vol->id;
        v->versions[k] = vol->version;
    }
    v->nvols = n;
    v->light_res = mp->light_res;
    v->lamp[0] = (float)lamp->pos.x; v->lamp[1] = (float)lamp->pos.y; v->lamp[2] = (float)lamp->pos.z;
    v->light_valid = 0;
    return 1;
}

static void ensure_capture(GpuView *v){
    if (v->cap_tex) return;
    Gpu *g = v->g;
    WGPUTextureDescriptor td;
    memset(&td, 0, sizeof(td));
    td.label = gpu_str("screenshot");
    td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    td.dimension = WGPUTextureDimension_2D;
    td.size = (WGPUExtent3D){(uint32_t)v->w, (uint32_t)v->h, 1};
    td.format = g->surface_format;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    v->cap_tex = wgpuDeviceCreateTexture(g->device, &td);
    v->cap_row = ((uint32_t)v->w * 4 + 255) & ~255u;
    v->cap_buf = gpu_buffer(g, "screenshot readback", (size_t)v->cap_row * v->h, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
}

// the composite's image when FXAA follows (canvas-sized, the canvas's format)
static int ensure_aa(GpuView *v){
    if (v->aa_view) return 1;
    Gpu *g = v->g;
    WGPUTextureDescriptor td;
    memset(&td, 0, sizeof(td));
    td.label = gpu_str("before anti-aliasing");
    td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
    td.dimension = WGPUTextureDimension_2D;
    td.size = (WGPUExtent3D){(uint32_t)v->w, (uint32_t)v->h, 1};
    td.format = g->surface_format;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    if (!(v->aa_tex = wgpuDeviceCreateTexture(g->device, &td))) return 0;
    v->aa_view = wgpuTextureCreateView(v->aa_tex, NULL);
    return v->aa_view != NULL;
}

static void on_capture_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    GpuView *v = u1;
    if (!view_is_live(v, u2)) return;   // cancelled as the view was destroyed: v is gone
    v->cap_pending = 0;                 // (also after a failure: the next frame can be asked for)
    if (status != WGPUMapAsyncStatus_Success){          // nothing is mapped
        if (status == WGPUMapAsyncStatus_Error)
            fprintf(stderr, "GPU: reading the frame back failed: %.*s\n",
                    gpu_sv_len(msg), msg.data ? msg.data : "");
        // whoever waits for the frame is told (the device is still ready: it would wait for ever)
        if (v->cap_fn) v->cap_fn(v->cap_user, NULL, v->w, v->h);
        return;
    }
    const uint8_t *src = wgpuBufferGetConstMappedRange(v->cap_buf, 0, (size_t)v->cap_row * v->h);
    uint32_t *px = malloc((size_t)v->w * v->h * sizeof(uint32_t));
    if (!(src && px) && v->cap_fn) v->cap_fn(v->cap_user, NULL, v->w, v->h);
    if (src && px){
        int bgra = v->g->surface_format == WGPUTextureFormat_BGRA8Unorm;
        for (int y = 0; y < v->h; y++)
            for (int x = 0; x < v->w; x++){
                const uint8_t *q = src + (size_t)y * v->cap_row + 4 * x;
                uint32_t r = bgra ? q[2] : q[0], gg = q[1], b = bgra ? q[0] : q[2];
                px[(size_t)y * v->w + x] = (r << 16) | (gg << 8) | b;
            }
        if (v->cap_fn) v->cap_fn(v->cap_user, px, v->w, v->h);
    }
    free(px);
    wgpuBufferUnmap(v->cap_buf);
}

void gpu_view_volume_on_gpu(GpuView *v, GpuDensity *d){
    if (!v || !d) return;
    sources_prune(v);
    v->nvols = -1;   // copy everything again
    for (int i = 0; i < v->nsrc; i++){
        if (v->src[i] == d) return;
        if (v->src[i]->volume == d->volume){         // another fluid has the medium now
            gpu_density_release(v->src[i]);
            v->src[i] = d;
            d->refs++;
            return;
        }
    }
    if (v->nsrc == GPU_MAX_SOURCES){
        fprintf(stderr, "GPU: a view draws at most %d media from the GPU's own densities\n", GPU_MAX_SOURCES);
        return;
    }
    v->src[v->nsrc++] = d;
    d->refs++;
}

int gpu_view_capture(GpuView *v, GpuCaptureFn fn, void *user){
    if (!v || v->cap_pending) return 0;
    v->cap_fn = fn;
    v->cap_user = user;
    v->cap_pending = 1;
    return 1;
}

// The first half of a frame: the camera and the lamp into the uniforms, and the media made
// ready (densities, bricks and boxes, light grids), for the composite pass that follows.
// Returns how many media it made ready (0 when they do not fit this GPU: what reads them
// after must be given that count, not the scene's).
static int frame_begin(GpuView *v, WGPUCommandEncoder enc, const Scene *s, const RenderParams *p,
                        const MarchParams *mp, int volumes){
    Gpu *g = v->g;
    MarchParams def = march_default_params();
    if (!mp) mp = &def;
    int n = volumes ? (s->nvolumes < GPU_MAX_VOLUMES ? s->nvolumes : GPU_MAX_VOLUMES) : 0;
    if (n && !gpu_view_volumes_fit(v, s, mp)) n = 0;   // (too big for this GPU: none drawn here)
    MarchLamp lamp = march_lamp(s);

    int relight = n ? upload_volumes(v, s, mp, &lamp, enc) : 0;

    GFrame F;
    memset(&F, 0, sizeof(F));
    v4(F.right, vec3(p->view.m[0][0], p->view.m[0][1], p->view.m[0][2]), (float)v->show_traced);
    v4(F.up, vec3(p->view.m[1][0], p->view.m[1][1], p->view.m[1][2]), 0);
    v4(F.back, vec3(p->view.m[2][0], p->view.m[2][1], p->view.m[2][2]), 0);
    v4(F.eye, p->eye, 0);
    F.proj[0] = (float)(1.0 / p->proj.m[0][0]);
    F.proj[1] = (float)(1.0 / p->proj.m[1][1]);
    F.proj[2] = (float)p->proj.m[2][2];
    F.proj[3] = (float)p->proj.m[2][3];
    F.size[0] = (float)v->w; F.size[1] = (float)v->h; F.size[2] = (float)mp->max_steps; F.size[3] = (float)n;
    v4(F.lamp_pos, lamp.pos, (float)lamp.area);
    v4(F.lamp_n, lamp.normal, v_dot(lamp.normal, lamp.normal) > 0 ? 1.0f : 0.0f);
    v4(F.lamp_L, lamp.radiance, (float)mp->brightness);
    v4(F.amb, march_ambient(s, mp), (float)(mp->octaves < 1 ? 1 : (mp->octaves > 4 ? 4 : mp->octaves)));
    for (int k = 0; k < n; k++) F.vols[k] = v->gv[k];
    wgpuQueueWriteBuffer(g->queue, v->frame_u, 0, &F, sizeof(F));

    if (relight){
        // the media's bricks, boxes and maxima, from the densities now on the GPU
        uint32_t init[GPU_MAX_VOLUMES * 8];
        for (int k = 0; k < GPU_MAX_VOLUMES; k++){
            uint32_t *a = init + 8 * k;
            a[0] = a[1] = a[2] = 0xFFFFFFFFu;
            a[3] = a[4] = a[5] = a[6] = a[7] = 0;
        }
        wgpuQueueWriteBuffer(g->queue, v->acc, 0, init, sizeof(init));
        WGPUComputePassEncoder sp = wgpuCommandEncoderBeginComputePass(enc, NULL);
        for (int k = 0; k < n; k++){
            WGPUBindGroupEntry e[5] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks),
                                       entry(3, v->acc), entry(4, v->sel_u[k])};
            WGPUBindGroup bg = gpu_bind_cached(g, &v->bc_stats[k], v->stats_bgl, e, 5, NULL);
            uint32_t groups = (v->gv[k].bdims[0] * v->gv[k].bdims[1] * v->gv[k].bdims[2] + 63) / 64;
            uint32_t gx = groups < 65535 ? groups : 65535;
            wgpuComputePassEncoderSetPipeline(sp, v->stats_pipe);
            wgpuComputePassEncoderSetBindGroup(sp, 0, bg, 0, NULL);
            wgpuComputePassEncoderDispatchWorkgroups(sp, gx, (groups + gx - 1) / gx, 1);
            WGPUBindGroupEntry f[4] = {entry(0, v->frame_u), entry(1, v->acc), entry(2, v->vstat), entry(3, v->sel_u[k])};
            bg = gpu_bind_cached(g, &v->bc_finish[k], v->finish_bgl, f, 4, NULL);
            wgpuComputePassEncoderSetPipeline(sp, v->finish_pipe);
            wgpuComputePassEncoderSetBindGroup(sp, 0, bg, 0, NULL);
            wgpuComputePassEncoderDispatchWorkgroups(sp, 1, 1, 1);
        }
        wgpuComputePassEncoderEnd(sp);
        wgpuComputePassEncoderRelease(sp);
        // one compute pass per medium: every light grid cell marches to the lamp
        for (int k = 0; k < n; k++){
            WGPUBindGroupEntry e[6] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks),
                                       entry(3, v->light), entry(4, v->sel_u[k]), entry(5, v->vstat)};
            WGPUBindGroup bg = gpu_bind_cached(g, &v->bc_light[k], v->light_bgl, e, 6, NULL);
            WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, NULL);
            wgpuComputePassEncoderSetPipeline(cp, v->light_pipe);
            wgpuComputePassEncoderSetBindGroup(cp, 0, bg, 0, NULL);
            wgpuComputePassEncoderDispatchWorkgroups(cp, (v->gv[k].ldims[0] + 3) / 4, (v->gv[k].ldims[1] + 3) / 4,
                                                     (v->gv[k].ldims[2] + 3) / 4);
            wgpuComputePassEncoderEnd(cp);
            wgpuComputePassEncoderRelease(cp);
        }
        v->light_valid = 1;
    }
    return n;
}

// The second half: the composite pass (pipe, with the frame's inputs in bg, a kept one:
// gpu_bind_cached) onto the canvas (unless offscreen) and the readback texture if a capture
// was asked for; then everything is submitted. With aa (and gpu_view_aa on), the composite
// draws into a texture once, and FXAA from it onto those.
static void frame_end(GpuView *v, WGPUCommandEncoder enc, WGPURenderPipeline pipe, WGPUBindGroup bg, int aa){
    Gpu *g = v->g;
    int capture = v->cap_pending == 1;
    WGPUSurfaceTexture st;
    memset(&st, 0, sizeof(st));
    WGPUTextureView view = NULL;
    if (g->surface){
        wgpuSurfaceGetCurrentTexture(g->surface, &st);
        if (st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
            st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
            view = wgpuTextureCreateView(st.texture, NULL);
        else {
            // no frame this time (it is skipped); Outdated or Lost: the surface configured
            // again, or the window would stay frozen. A texture given with the failure is ours
            // to release.
            static int told = 0;
            if (!told++) fprintf(stderr, "GPU: no canvas texture to draw on (status %d)\n", (int)st.status);
            if (st.texture){ wgpuTextureRelease(st.texture); st.texture = NULL; }
            if ((st.status == WGPUSurfaceGetCurrentTextureStatus_Outdated || st.status == WGPUSurfaceGetCurrentTextureStatus_Lost) &&
                gpu_state(g) == GPU_READY)
                gpu_surface_reconfigure(g);
        }
    }
    WGPURenderPassColorAttachment ca;
    memset(&ca, 0, sizeof(ca));
    ca.view = view;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    WGPURenderPassDescriptor rp;
    memset(&rp, 0, sizeof(rp));
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    aa = aa && v->aa && v->fxaa_pipe && (view || capture) && ensure_aa(v);
    if (aa){
        ca.view = v->aa_view;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
        wgpuRenderPassEncoderSetPipeline(pass, pipe);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
        WGPUBindGroupEntry fe;
        memset(&fe, 0, sizeof(fe));
        fe.binding = 0;
        fe.textureView = v->aa_view;
        pipe = v->fxaa_pipe;
        bg = gpu_bind_cached(g, &v->bc_fxaa, v->fxaa_bgl, &fe, 1, NULL);
    }
    for (int target = view ? 0 : 1; target < 1 + capture; target++){
        // target 0: the canvas; target 1: the screenshot texture (same pass, same pixels)
        WGPUTextureView tv = view;
        if (target == 1){
            ensure_capture(v);
            tv = wgpuTextureCreateView(v->cap_tex, NULL);
        }
        ca.view = tv;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
        wgpuRenderPassEncoderSetPipeline(pass, pipe);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        if (target == 0 && v->hud_w > 0 && v->hud_pipe){
            // the text, on the canvas only (screenshots stay clean)
            WGPUBuffer hb[2] = {v->hud_u, v->hud_px};
            WGPUBindGroup hg = gpu_bind_cached_buffers(g, &v->bc_hud, v->hud_bgl, hb, 2);
            wgpuRenderPassEncoderSetPipeline(pass, v->hud_pipe);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, hg, 0, NULL);
            wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        }
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
        if (target == 1){
            wgpuTextureViewRelease(tv);
            WGPUTexelCopyTextureInfo src;
            memset(&src, 0, sizeof(src));
            src.texture = v->cap_tex;
            WGPUTexelCopyBufferInfo dst;
            memset(&dst, 0, sizeof(dst));
            dst.buffer = v->cap_buf;
            dst.layout.bytesPerRow = v->cap_row;
            dst.layout.rowsPerImage = (uint32_t)v->h;
            WGPUExtent3D ext = {(uint32_t)v->w, (uint32_t)v->h, 1};
            wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ext);
        }
    }
    gpu_denoise_before_submit(v->den, enc);     // (its pass's time, if it ran in this frame)
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    gpu_denoise_after_submit(v->den);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    if (capture){
        v->cap_pending = 2;
        WGPUBufferMapCallbackInfo mi;
        memset(&mi, 0, sizeof(mi));
        mi.mode = WGPUCallbackMode_AllowSpontaneous;
        mi.callback = on_capture_mapped;
        mi.userdata1 = v;
        mi.userdata2 = gpu_live_ticket(v->live);
        wgpuBufferMapAsync(v->cap_buf, WGPUMapMode_Read, 0, (size_t)v->cap_row * v->h, mi);
    }
    if (view){
        // present first, then let go of the texture: natively, a surface texture released
        // before it is presented is thrown away, and the window's images never come back
#ifndef __EMSCRIPTEN__
        wgpuSurfacePresent(g->surface);   // in a browser the canvas shows the frame by itself
#endif
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);
    }
}

void gpu_view_present(GpuView *v, const Framebuffer *fb, const Scene *s, const RenderParams *p,
                      const MarchParams *mp, int volumes){
    if (!v || fb->w != v->w || fb->h != v->h) return;
    Gpu *g = v->g;
    // the frame: colors and depths of the CPU renderer
    const size_t np = (size_t)v->w * v->h;
    wgpuQueueWriteBuffer(g->queue, v->color, 0, fb->color, np * 4);
    for (size_t i = 0; v->depth_f && i < np; i++) v->depth_f[i] = (float)fb->depth[i];
    wgpuQueueWriteBuffer(g->queue, v->depth, 0, v->depth_f ? (const void *)v->depth_f : fb->depth, np * 4);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    frame_begin(v, enc, s, p, mp, volumes);
    WGPUBindGroupEntry e[7] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks),
                               entry(3, v->light), entry(4, v->color), entry(5, v->depth), entry(6, v->vstat)};
    frame_end(v, enc, v->comp_pipe, gpu_bind_cached(g, &v->bc_comp, v->comp_bgl, e, 7, NULL), 1);
}

// The path tracer's image to show or read back: the samples, or with the denoiser on, the
// denoised image (made again only when a band finished since).
static WGPUBuffer traced_image(GpuView *v, WGPUCommandEncoder enc){
    WGPUBuffer samples = gpu_trace_accum(v->trace), base = samples;
    if (v->denoise && !v->den_failed && gpu_trace_guided(v->trace)){
        if (!v->den && !(v->den = gpu_denoise_create(v->g, v->w, v->h))) v->den_failed = 1;
        else {
            if (v->den_bands != v->t_bands){
                // while things move, fewer a-trous passes (gpu_view_trace_denoise_moving): the
                // image lasts a frame, carries up to 12-32 samples per pixel over, and the
                // denoiser was a quarter of the GPU's time (4.6 ms a frame at 1080p)
                gpu_denoise_encode(v->den, enc, samples, v->t_live ? v->den_move : 0);
                v->den_bands = v->t_bands;
            }
            base = gpu_denoise_output(v->den);
        }
    }
    // the caustics on top (after the denoiser: they stay sharp): light traced, and while that
    // has few passes (or things move) the photon cache's
    if (gpu_trace_caustics_on(v->trace, &v->tparams) &&
        (gpu_trace_caustic_passes(v->trace) > 0 || (v->tparams.photon_cache && gpu_trace_photon_paths(v->trace) > 0)))
        return gpu_trace_resolve(v->trace, enc, base, v->tparams.photon_cache);
    return base;
}

// the GPU rasterizer's frame (and, with traced = 1, the path tracer's samples over it), composited
static int present_gpu_frame(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes,
                             int traced, int draw_scene){
    if (!v || v->raster_failed) return 0;
    if (volumes && s->nvolumes > 0 && !gpu_view_volumes_fit(v, s, mp)) return 0;   // the CPU marches them
    Gpu *g = v->g;
    if (!v->raster && !(v->raster = gpu_raster_create(g, v->w, v->h))){ v->raster_failed = 1; return 0; }
    // a scene the rasterizer could not draw (a mesh too big for this GPU): the CPU draws it, and
    // the GPU is asked again when the scene has changed (it was given up for good: one model too
    // big, dropped and removed again, left the CPU drawing until the page was loaded again)
    const uint64_t sig = draw_scene ? scene_sig(s) : 0;
    if (draw_scene && v->r_unfit && v->r_unfit_sig == sig) return 0;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    if (draw_scene && !gpu_raster_draw(v->raster, enc, s, p)){
        wgpuCommandEncoderRelease(enc);
        v->r_unfit = 1;
        v->r_unfit_sig = sig;
        return 0;
    }
    if (draw_scene) v->r_unfit = 0;
    if (traced && v->t_prep == 1) gpu_trace_clear(v->trace, enc);   // nothing to show yet
    v->show_traced = traced ? (draw_scene ? 1 : 2) : 0;   // (2: holes filled from neighbours)
    frame_begin(v, enc, s, p, mp, volumes);
    v->show_traced = 0;
    WGPUBindGroupEntry e[8] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks), entry(3, v->light),
                               entry(4, NULL), entry(5, NULL), entry(6, v->vstat),
                               entry(7, traced ? traced_image(v, enc) : v->trace ? gpu_trace_accum(v->trace) : v->color)};
    e[4].size = e[5].size = 0;
    e[4].textureView = gpu_raster_color(v->raster);
    e[5].textureView = gpu_raster_depth(v->raster);
    // (the path tracer's image needs none: its samples are spread over each pixel)
    frame_end(v, enc, v->comp_gpu_pipe, gpu_bind_cached(g, &v->bc_comp_gpu, v->comp_gpu_bgl, e, 8, NULL), !traced);
    return 1;
}

int gpu_view_present_scene(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    return present_gpu_frame(v, s, p, mp, volumes, 0, 1);
}

// ---- path tracing ----------------------------------------------------------------------------

void gpu_view_trace_prepare(GpuView *v){
    if (!v || v->trace || v->trace_failed) return;
    // the rasterizer's pipelines first: in Chrome one made while the path tracer's compile
    // would wait for them, and with it the frames that show the scene meanwhile
    if (!v->raster && !v->raster_failed && !(v->raster = gpu_raster_create(v->g, v->w, v->h))) v->raster_failed = 1;
    if (!(v->trace = gpu_trace_create(v->g, v->w, v->h, GPU_WGSL_TYPES, GPU_WGSL_MEDIA))) v->trace_failed = 1;
}

// The scene onto the GPU for the path tracer; 0 = it does not fit (its media or a buffer too
// big for this GPU, a mesh without its tree, no memory). That is the scene's, not the GPU's:
// remembered with what the scene was (scene_sig), so that the same one is not tried again
// (and its message said again) every frame, and another one is.
static int trace_sync(GpuView *v, const Scene *s){
    const uint64_t sig = scene_sig(s);
    if (v->t_unfit && v->t_unfit_sig == sig) return 0;
    v->t_unfit = 0;
    if (gpu_view_volumes_fit(v, s, NULL) && gpu_trace_sync(v->trace, s)) return 1;
    v->t_unfit = 1;
    v->t_unfit_sig = sig;
    return 0;
}

int gpu_view_trace_failed(const GpuView *v){ return !v || v->trace_failed; }

int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp){
    if (!v || v->trace_failed) return 0;
    if (!v->trace && !(v->trace = gpu_trace_create(v->g, v->w, v->h, GPU_WGSL_TYPES, GPU_WGSL_MEDIA))){
        v->trace_failed = 1;
        return 0;
    }
    gpu_trace_set_wavefront(v->trace, v->wavefront);
    if (!trace_sync(v, s)) return 0;
    v->tparams = *tp;
    v->t_prep = 1;
    v->t_hist = 0;
    v->t_y = 0;
    // a new image starts its samples at a multiple of 2^16: the first n of them are then as
    // evenly spread as the sequence allows (while things move, the count just goes on, so
    // the samples a pixel keeps from frame to frame are never the same twice)
    v->t_index = (v->t_index + 0xffffu) & ~0xffffu;
    v->t_passes = v->t_done = v->t_done_next = 0;
    v->t_bands = 0;
    v->den_bands = -1;
    v->t_moved = 0;
    if (v->t_rows < 1) v->t_rows = 16;
    if (v->t_reps < 1) v->t_reps = 1;
    v->t_pass_t0 = clock_ms();
    return 1;
}

int gpu_view_trace_move(GpuView *v, const Scene *s, const GpuTraceParams *tp, int history){
    if (!v || !v->trace || history < 1) return gpu_view_trace_start(v, s, tp);
    if (v->trace_failed || !trace_sync(v, s)) return 0;
    v->tparams = *tp;
    if (v->t_prep != 1){ v->t_prep = 2; v->t_history = history; v->t_hmax = v->grad_next; v->t_hedit = v->grad_edit; }   // (an image not begun yet has nothing to keep)
    // the bands go on where they were: on a slow GPU (a band per frame) every row gets its turn
    v->t_passes = v->t_done = v->t_done_next = 0;
    v->t_bands = 0;
    v->den_bands = -1;
    v->t_moved = 1;
    v->t_pass_t0 = clock_ms();
    return 1;
}

// The GPU time the path tracer's band may take this frame: the budget, or with t_pass_max at
// least a whole pass (a tenth over, so a band is never a pass less a sliver that then costs a
// frame of its own), up to t_pass_max. Frames that do not wait for the display carry a fixed
// cost (the denoiser, the composite, the bodies, the guide: ~9 ms at 1080p in the sky demo);
// a band of 60% of a 165 Hz period beside it left the path tracer 39% of the GPU's time.
static double frame_budget(const GpuView *v){
    double b = v->t_live && v->t_live_ms > 0 ? v->t_live_ms : v->t_budget_ms;
    if (b > 0 && v->t_pass_max > 0 && v->t_gpu_pass_ms > 0){
        double p = 1.1 * v->t_gpu_pass_ms;
        if (p > v->t_pass_max) p = v->t_pass_max;
        if (p > b) b = p;
    }
    return b;
}

// A band finished: how long it took decides the size of the next ones. Small bands keep a
// slow GPU responsive (and under the browser's watchdog); a fast GPU gets the whole image,
// then several passes, per submit.
static void on_band_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg;
    GpuView *v = u1;
    if (!view_is_live(v, u2)) return;
    v->t_inflight = 0;
    v->t_done = v->t_done_next;
    if (v->t_done > 0) v->t_hist = 1;
    v->t_bands++;
    if (v->t_sub_rows > 0 && v->t_sub_reps > 0) v->t_work += (double)v->t_sub_rows * v->t_sub_reps / v->h;
    double ms = clock_ms() - v->t_submit_ms;
    // the time of a pass: from bands that did not begin an image (their time includes the guide
    // of the whole image), or a smaller one than known (the first includes making the pipelines).
    // A larger one from a band that began an image moves it a tenth of the way: while the
    // camera moves every band begins one, and a lucky low value would stay, piling up passes
    // (the helmet orbiting: 4 passes per frame, 19 ms frames, against the 10 ms meant)
    double per_pass = v->t_sub_rows > 0 && v->t_sub_reps > 0 ? ms * v->h / ((double)v->t_sub_rows * v->t_sub_reps) : 0;
    // the GPU's own time of the passes since (its timer, when the GPU has one): the wall time
    // above also counts what waited on the queue before the band (a frame waiting for the
    // display, a readback), which made bands ever smaller once frames came every few ms
    double gpu_pass = gpu_timer_recent_ms_per_unit(gpu_trace_timer(v->trace), 0);
    if (gpu_pass > 0){
        v->t_gpu_timed = 1;
        v->t_gpu_pass_ms = v->t_gpu_pass_ms > 0 ? v->t_gpu_pass_ms + 0.3 * (gpu_pass - v->t_gpu_pass_ms) : gpu_pass;
        per_pass = 0;
    }
    if (v->t_gpu_timed) per_pass = 0;
    if (per_pass > 0){
        if (!v->t_sub_prep || v->t_gpu_pass_ms <= 0 || per_pass < v->t_gpu_pass_ms) v->t_gpu_pass_ms = per_pass;
        else v->t_gpu_pass_ms += 0.1 * (per_pass - v->t_gpu_pass_ms);
    }
    // a budget per frame and the GPU's own time of a pass: the band that fits it, at once
    if (v->t_budget_ms > 0 && v->t_gpu_timed && v->t_gpu_pass_ms > 0){
        double k = frame_budget(v) / v->t_gpu_pass_ms;
        if (k >= 1){ v->t_rows = v->h; v->t_reps = k >= 16 ? 16 : (int)(k + 0.25); }   // (a little over: the budget has room)
        else { v->t_reps = 1; v->t_rows = (int)(v->h * k); if (v->t_rows < 4) v->t_rows = 4; }
        return;
    }
    // (with a budget per frame, without a timer: within 0.6 and 1.5 times it)
    const double lo = v->t_budget_ms > 0 ? 0.6 * v->t_budget_ms : 12, hi = v->t_budget_ms > 0 ? 1.5 * v->t_budget_ms : 45;
    if (ms < lo){
        if (v->t_rows < v->h) v->t_rows = v->t_rows * 2 < v->h ? v->t_rows * 2 : v->h;
        else if (v->t_reps < 16) v->t_reps *= 2;
    } else if (ms > hi){
        if (v->t_reps > 1) v->t_reps /= 2;
        else if (v->t_rows > 4) v->t_rows /= 2;
    }
}

// Which guide a new image gets (1 rasterized, 0 traced): with rguide 1, the one that has cost
// less GPU time per image, as measured: rasterized is the cheaper with a software walk of the
// trees (natively 2.2 -> 1.0 ms at 2560 x 1440), traced often with the hardware's ray queries
// and in a browser's simple scenes (its walk there reaches a pixel's first hit about as fast
// as the copy of the G-buffer). Each way for RG_IMAGES images in turn (after RG_SETTLE: the
// timers read back late), then the cheaper for RG_HOLD images, then both again (the scene
// changes). Without timers: rasterized unless the hardware's ray queries trace.
#define RG_SETTLE 6
#define RG_IMAGES 24
#define RG_HOLD 900
#define RG_IDLE 64
static int rguide_way(GpuView *v){
    if (v->rguide != 1) return v->rguide == 2;
    GpuTimer *t = gpu_trace_timer(v->trace);
    if (!t) return !gpu_trace_hw(v->trace);
    double traced = gpu_timer_recent_ms_per_unit(t, gpu_trace_traced_guide_stage());
    double rast = gpu_timer_recent_ms_per_unit(t, gpu_trace_guide_stage());
    v->rg_images++;
    if (v->rg_phase < 2 && v->rg_images > RG_SETTLE && traced > 0 && (rast > 0 || !v->rg_way)){
        v->rg_sum[v->rg_way] += traced + (v->rg_way ? rast : 0);
        v->rg_n[v->rg_way]++;
    }
    if (v->rg_phase < 2 && v->rg_n[v->rg_way] >= RG_IMAGES){
        v->rg_phase++;
        v->rg_images = 0;
        if (v->rg_phase == 1) v->rg_way = !v->rg_way;           // now the other one
        else {
            v->rg_way = v->rg_sum[1] / v->rg_n[1] <= v->rg_sum[0] / v->rg_n[0];
            if (gpu_timing_report(v->g))
                printf("GPU: the guide %s (%.3f ms an image rasterized, %.3f traced)\n", v->rg_way ? "rasterized" : "traced",
                       v->rg_sum[1] / v->rg_n[1], v->rg_sum[0] / v->rg_n[0]);
        }
    } else if (v->rg_phase == 2 && v->rg_images >= RG_HOLD){
        v->rg_phase = 0;                                         // measured again, from the one chosen
        v->rg_images = 0;
        v->rg_sum[0] = v->rg_sum[1] = 0;
        v->rg_n[0] = v->rg_n[1] = 0;
    }
    return v->rg_way;
}

int gpu_view_trace_more(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp){
    // one band in flight at most (also for a new image's start: on a slow GPU, frames go on
    // showing the last image until the band ends, rather than piling work up on the queue)
    if (!v || v->trace_failed) return 0;
    if (!v->trace || v->t_inflight) return 1;
    if (v->t_limit > 0 && v->t_passes >= v->t_limit) return 1;   // the image has all its samples
    int ready = gpu_trace_ready(v->trace, s);                    // (in a browser: still compiling?)
    if (ready < 0){ v->trace_failed = 1; return 0; }
    if (!ready) return 1;
    v->t_ready_seen = 1;
    Gpu *g = v->g;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    // the media: densities, bricks and boxes up to date; as many as frame_begin made ready (none
    // when they do not fit with their light grids: the kernels were given the scene's count,
    // and read media that were never uploaded)
    const int nvol = frame_begin(v, enc, s, p, mp, 1);
    const GpuMedia media = {v->frame_u, v->dens, v->bricks, v->vstat};   // (as frame_begin left them)
    v->t_sub_prep = v->t_prep != 0;
    if (v->t_prep){
        // the guide (each pixel's first surface) rasterized, where there are no media: it was a
        // fifth to a third of the GPU's time traced, and is made for every image, every frame
        // while things move (the pixels behind a mirror or glass are traced still)
        WGPUTextureView rg[3];
        const WGPUTextureView *rgp = NULL;
        if (nvol == 0 && rguide_way(v) && v->raster && !v->raster_failed && !(v->r_unfit && v->r_unfit_sig == scene_sig(s)) &&
            gpu_raster_guide(v->raster, enc, s, &v->tparams.cam, gpu_trace_timer(v->trace), gpu_trace_guide_stage())){
            gpu_raster_guide_views(v->raster, rg);
            rgp = rg;
        }
        v->t_rguided = rgp != NULL;
        // the guide traced for RG_IDLE images in a row (the traced way chosen, media in the
        // scene, a view told to trace it): the rasterizer's four full-size targets are let go
        // (they stayed for good, 44 bytes a pixel); made again when the guide is rasterized again
        if (rgp) v->rg_idle = 0;
        else if (++v->rg_idle == RG_IDLE) gpu_raster_guide_release(v->raster);
        gpu_trace_begin_image(v->trace, enc, s, &v->tparams, v->t_prep == 2 ? v->t_history : 0,
                              v->t_prep == 2 ? v->t_hmax : 0, v->t_prep == 2 && v->t_hedit, v->t_index, nvol,
                              &media, rgp);
        gpu_trace_caustics_clear(v->trace, enc);
        if (v->t_prep == 1) gpu_trace_photons_clear(v->trace, enc);   // a new scene or view: the cache too
        v->t_prep = 0;
    }
    // a band of rows, or (the band being the whole image) several passes
    int whole = v->t_y == 0 && v->t_rows >= v->h;
    int rows = v->t_rows < v->h - v->t_y ? v->t_rows : v->h - v->t_y;
    int reps = whole ? v->t_reps : 1;
    if (v->t_limit > 0 && reps > v->t_limit - v->t_passes) reps = v->t_limit - v->t_passes;   // (>= 1: checked above)
    // something is moving: a frame each, so no more passes than fit in ~10 ms; or the budget
    // per frame (gpu_view_trace_budget), moving or not
    v->t_live = v->t_moved;
    v->t_moved = 0;
    if ((v->t_live || v->t_budget_ms > 0) && reps > 1 && v->t_gpu_pass_ms > 0){
        const double budget = frame_budget(v);
        double ms = budget > 0 && (budget < 10 || !v->t_live) ? budget : 10.0;
        int fit = (int)(ms / v->t_gpu_pass_ms);
        reps = fit < 1 ? 1 : (fit < reps ? fit : reps);
    }
    v->t_sub_rows = rows;
    v->t_sub_reps = reps;
    // while things move, the lower indirect clamp: in the editor's Play, turning, the red ball's
    // light bounced off the floor made fireflies the denoiser spread into blotches all over it
    // (763 red pixels at 1440p; 44 at 1, the still image 0.1% darker at 1 than at 4)
    GpuTraceParams ptp = v->tparams;
    if (v->t_live && ptp.indirect_clamp_move > 0 && (ptp.indirect_clamp <= 0 || ptp.indirect_clamp_move < ptp.indirect_clamp))
        ptp.indirect_clamp = ptp.indirect_clamp_move;
    gpu_trace_encode(v->trace, enc, s, &ptp, v->t_y, rows, reps, nvol, v->t_index, &media);
    // the caustics: while things move, one pass of light paths into the photon cache, which
    // fades a little each time (what moved fades out); still, a pass ending with this band
    // brings as many passes of the (exact) light tracer, and while that has few the cache
    // goes on filling. While things move, a quarter of the light paths: those caustics last a
    // frame (or fade out of the cache), and all of them cost as much as the path tracer's pass
    // in a room of moving bodies; all of them again once the view is still.
    GpuTraceParams ltp = v->tparams;
    if (v->t_live) ltp.photons *= 0.25;
    if (gpu_trace_caustics_on(v->trace, &ltp)){
        if (v->t_live && ltp.photon_cache){
            gpu_trace_photons(v->trace, enc, s, &ltp, nvol, v->t_index, 0.9, &media);
        } else if (v->t_y + rows >= v->h){
            gpu_trace_light(v->trace, enc, s, &ltp, reps, nvol, v->t_index, &media);
            if (v->tparams.photon_cache && gpu_trace_caustic_passes(v->trace) < 32)
                gpu_trace_photons(v->trace, enc, s, &v->tparams, nvol, v->t_index, 1.0, &media);
        }
    }
    gpu_trace_before_submit(v->trace, enc);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    gpu_trace_after_submit(v->trace);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    v->t_inflight = 1;
    v->t_submit_ms = clock_ms();
    WGPUQueueWorkDoneCallbackInfo wd;
    memset(&wd, 0, sizeof(wd));
    wd.mode = WGPUCallbackMode_AllowSpontaneous;
    wd.callback = on_band_done;
    wd.userdata1 = v;
    wd.userdata2 = gpu_live_ticket(v->live);
    wgpuQueueOnSubmittedWorkDone(g->queue, wd);
    v->t_y += rows;
    if (v->t_y >= v->h){
        v->t_y = 0;
        v->t_passes += reps;
        v->t_index += (uint32_t)reps;
        double now = clock_ms();
        v->t_pass_ms = (now - v->t_pass_t0) / reps;
        v->t_pass_t0 = now;
    }
    v->t_done_next = v->t_passes;
    return 1;
}

void gpu_view_trace_limit(GpuView *v, int spp){ if (v) v->t_limit = spp > 0 ? spp : 0; }
void gpu_view_trace_budget(GpuView *v, double ms){ if (v) v->t_budget_ms = ms > 0 ? ms : 0; }
void gpu_view_trace_budget_live(GpuView *v, double ms){ if (v) v->t_live_ms = ms > 0 ? ms : 0; }
void gpu_view_trace_budget_pass(GpuView *v, double max_ms){ if (v) v->t_pass_max = max_ms > 0 ? max_ms : 0; }
int gpu_view_trace_passes(const GpuView *v){ return v ? v->t_done : 0; }
double gpu_view_trace_work(const GpuView *v){ return v ? v->t_work : 0; }
int gpu_view_trace_busy(const GpuView *v){ return v && v->t_inflight; }
int gpu_view_trace_compiling(const GpuView *v){ return v && v->trace && !v->trace_failed && !v->t_ready_seen; }
void gpu_view_gpu_totals(GpuView *v, double ms[8]){
    gpu_trace_gpu_totals(v ? v->trace : NULL, ms);
    ms[5] = v ? gpu_denoise_total_ms(v->den) : 0;
}
int gpu_view_trace_reads_tlas(const GpuView *v, const Scene *s){ return !v || !v->trace || gpu_trace_reads_tlas(v->trace, s); }

static void on_trace_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    GpuView *v = u1;
    if (!view_is_live(v, u2)) return;                   // cancelled as it was destroyed: v is gone
    v->t_rb_pending = 0;                                // (also after a failure: it can be asked for again)
    if (status != WGPUMapAsyncStatus_Success){          // nothing is mapped
        if (status == WGPUMapAsyncStatus_Error)
            fprintf(stderr, "GPU: reading the path traced image back failed: %.*s\n",
                    gpu_sv_len(msg), msg.data ? msg.data : "");
        // whoever waits for the image is told (the device is still ready: it would wait for ever)
        if (v->t_read_fn) v->t_read_fn(v->t_read_user, NULL, v->w, v->h);
        return;
    }
    size_t bytes = (size_t)v->w * v->h * 16;
    const float *p = wgpuBufferGetConstMappedRange(v->t_rb, 0, bytes);
    if (v->t_read_fn) v->t_read_fn(v->t_read_user, p, v->w, v->h);     // (p NULL: failed too)
    wgpuBufferUnmap(v->t_rb);
}

int gpu_view_trace_read(GpuView *v, GpuTraceReadFn fn, void *user){
    if (!v || !v->trace || v->t_rb_pending) return 0;
    Gpu *g = v->g;
    size_t bytes = (size_t)v->w * v->h * 16;
    if (!v->t_rb) v->t_rb = gpu_buffer(g, "path tracer readback", bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (!v->t_rb) return 0;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    if (v->t_prep == 1) gpu_trace_clear(v->trace, enc);
    wgpuCommandEncoderCopyBufferToBuffer(enc, traced_image(v, enc), 0, v->t_rb, 0, bytes);
    gpu_denoise_before_submit(v->den, enc);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    gpu_denoise_after_submit(v->den);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    v->t_read_fn = fn;
    v->t_read_user = user;
    v->t_rb_pending = 1;
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = on_trace_mapped;
    mi.userdata1 = v;
    mi.userdata2 = gpu_live_ticket(v->live);
    wgpuBufferMapAsync(v->t_rb, WGPUMapMode_Read, 0, bytes, mi);
    return 1;
}
// a whole pass, as the GPU took for the last band (submission to completion, scaled to the
// whole image); before any band, the time between passes
double gpu_view_trace_pass_ms(const GpuView *v){ return !v ? 0 : v->t_gpu_pass_ms > 0 ? v->t_gpu_pass_ms : v->t_pass_ms; }
void gpu_view_trace_band(const GpuView *v, int *rows, int *reps, double *budget_ms){
    *rows = v ? v->t_rows : 0;
    *reps = v ? v->t_reps : 0;
    *budget_ms = v ? frame_budget(v) : 0;
}

void gpu_view_trace_raster_guide(GpuView *v, int mode){ if (v) v->rguide = mode < 0 ? 0 : mode > 2 ? 2 : mode; }
void gpu_view_trace_gradients(GpuView *v, int history_max){ if (v) v->grad_next = history_max > 0 ? history_max : 0; }
void gpu_view_trace_edited(GpuView *v, int on){ if (v) v->grad_edit = on != 0; }
int gpu_view_trace_has_gradients(const GpuView *v){ return v && v->trace && gpu_trace_gradients_ok(v->trace); }
int gpu_view_trace_raster_guided(const GpuView *v){ return v && v->t_rguided; }

int gpu_view_trace_planes_read(GpuView *v, int first, int planes, float *out){
    if (!v || !v->trace || !gpu_trace_guided(v->trace) || first < 0 || planes < 1 || first + planes > 5) return 0;
    Gpu *g = v->g;
    for (int k = 0; k < 100000 && v->t_inflight; k++) gpu_wait(g);
    const size_t n = (size_t)v->w * v->h * 16 * (size_t)planes;
    WGPUBuffer rb = gpu_buffer(g, "planes readback", n, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (!rb) return 0;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gpu_trace_accum(v->trace), (uint64_t)v->w * v->h * 16 * (uint64_t)first, rb, 0, n);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    int ok = gpu_map_wait(g, rb, n, 100000);
    if (ok){
        const void *p = wgpuBufferGetConstMappedRange(rb, 0, n);
        ok = p != NULL;
        if (p) memcpy(out, p, n);
        wgpuBufferUnmap(rb);
    }
    wgpuBufferDestroy(rb);
    wgpuBufferRelease(rb);
    return ok;
}
void gpu_view_trace_wavefront(GpuView *v, int on){
    if (!v) return;
    v->wavefront = on;
    gpu_trace_set_wavefront(v->trace, on);
}

double gpu_view_gpu_ms(GpuView *v, int what){
    GpuTimer *t = v ? gpu_trace_timer(v->trace) : NULL;
    return t ? gpu_timer_ms_per_unit(t, what == 1 ? 2 : 0) : 0;
}
void gpu_view_gpu_ms_reset(GpuView *v){ if (v) gpu_timer_reset(gpu_trace_timer(v->trace)); }

double gpu_view_bench_denoise(GpuView *v, int n){
    if (!v || !v->trace || n <= 0 || !gpu_trace_guided(v->trace) || v->den_failed) return -1;
    if (!v->den && !(v->den = gpu_denoise_create(v->g, v->w, v->h))){ v->den_failed = 1; return -1; }
    gpu_wait(v->g);
    double t0 = clock_s();
    WGPUCommandEncoder enc = gpu_encoder(v->g, v->g->queue);
    for (int i = 0; i < n; i++) gpu_denoise_encode(v->den, enc, gpu_trace_accum(v->trace), 0);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    gpu_submit(v->g, cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    gpu_wait(v->g);
    v->den_bands = -1;                 // (made again for the image shown)
    return (clock_s() - t0) * 1000.0 / n;
}

void gpu_view_trace_denoise_moving(GpuView *v, int passes){ if (v) v->den_move = passes > 0 ? passes : 0; }
void gpu_view_aa(GpuView *v, int on){ if (v) v->aa = on != 0; }

int gpu_view_trace_denoise(GpuView *v, int on){
    if (!v) return 0;
    if (v->denoise != on) v->den_bands = -1;
    v->denoise = on;
    return on && !v->den_failed && (!v->trace || gpu_trace_guided(v->trace));
}

int gpu_view_present_traced(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    if (!v || !v->trace) return 0;
    // the rasterized scene shows through where there is no sample yet: only during the first
    // pass of an image begun from nothing. An image carried over (reprojected) has samples
    // wherever the last one had; the few pixels it lacks (uncovered by what moved, in rows not
    // yet traced) the composite fills from their traced neighbours. (Drawn under every image
    // until its first pass, the raster was ~3 ms of each frame while the bodies moved: with
    // a band of less than a pass per frame, that pass never came.)
    return present_gpu_frame(v, s, p, mp, volumes, 1, v->t_done == 0 && !v->t_hist);
}

void gpu_view_hud(GpuView *v, const uint32_t *argb, int w, int h, int x, int y){
    if (!v) return;
    if (!argb || w <= 0 || h <= 0 || x + w > v->w || y + h > v->h || x < 0 || y < 0){ v->hud_w = 0; return; }
    Gpu *g = v->g;
    size_t bytes = (size_t)w * h * 4;
    gpu_buffer_reserve(g, &v->hud_px, &v->hud_cap, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "text over the image");
    wgpuQueueWriteBuffer(g->queue, v->hud_px, 0, argb, bytes);
    float u[8] = {(float)x, (float)y, (float)w, (float)h, (float)v->w, (float)v->h, 0, 0};
    wgpuQueueWriteBuffer(g->queue, v->hud_u, 0, u, sizeof(u));
    v->hud_w = w;
}

void gpu_view_destroy(GpuView *v){
    if (!v) return;
    for (int k = 0; k < 7; k++) gpu_compute_async_cancel(v->preq[k], &v->pst[k]);
    GpuBindCache *bcs[] = {&v->bc_comp, &v->bc_comp_gpu, &v->bc_hud, &v->bc_fxaa};
    for (size_t i = 0; i < sizeof(bcs) / sizeof(bcs[0]); i++) gpu_bind_cache_clear(bcs[i]);
    for (int k = 0; k < GPU_MAX_VOLUMES; k++){
        gpu_bind_cache_clear(&v->bc_stats[k]); gpu_bind_cache_clear(&v->bc_finish[k]); gpu_bind_cache_clear(&v->bc_light[k]);
    }
    WGPUBuffer bufs[] = {v->frame_u, v->dens, v->bricks, v->light, v->color, v->depth,
                         v->vstat, v->acc, v->hud_u, v->hud_px};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    for (int k = 0; k < GPU_MAX_VOLUMES; k++) if (v->sel_u[k]){ wgpuBufferDestroy(v->sel_u[k]); wgpuBufferRelease(v->sel_u[k]); }
    if (v->light_pipe) wgpuComputePipelineRelease(v->light_pipe);
    if (v->comp_pipe) wgpuRenderPipelineRelease(v->comp_pipe);
    if (v->comp_gpu_pipe) wgpuRenderPipelineRelease(v->comp_gpu_pipe);
    if (v->hud_pipe) wgpuRenderPipelineRelease(v->hud_pipe);
    if (v->hud_bgl) wgpuBindGroupLayoutRelease(v->hud_bgl);
    if (v->fxaa_pipe) wgpuRenderPipelineRelease(v->fxaa_pipe);
    if (v->fxaa_bgl) wgpuBindGroupLayoutRelease(v->fxaa_bgl);
    if (v->aa_view) wgpuTextureViewRelease(v->aa_view);
    if (v->aa_tex){ wgpuTextureDestroy(v->aa_tex); wgpuTextureRelease(v->aa_tex); }
    gpu_raster_destroy(v->raster);
    gpu_trace_destroy(v->trace);
    gpu_denoise_destroy(v->den);
    gpu_live_remove(GPU_LIVE_VIEW, v);
    if (v->stats_pipe) wgpuComputePipelineRelease(v->stats_pipe);
    if (v->finish_pipe) wgpuComputePipelineRelease(v->finish_pipe);
    WGPUBindGroupLayout bgls[] = {v->light_bgl, v->comp_bgl, v->comp_gpu_bgl, v->stats_bgl, v->finish_bgl};
    for (size_t i = 0; i < sizeof(bgls) / sizeof(bgls[0]); i++) if (bgls[i]) wgpuBindGroupLayoutRelease(bgls[i]);
    if (v->cap_buf){ wgpuBufferDestroy(v->cap_buf); wgpuBufferRelease(v->cap_buf); }
    if (v->t_rb){ wgpuBufferDestroy(v->t_rb); wgpuBufferRelease(v->t_rb); }
    if (v->cap_tex){ wgpuTextureDestroy(v->cap_tex); wgpuTextureRelease(v->cap_tex); }
    for (int i = 0; i < v->nsrc; i++) gpu_density_release(v->src[i]);
    free(v->depth_f);
    free(v);
}

#endif   // LR_GPU
