// test_gpu_rt.c - hardware ray tracing in our Vulkan layer (webgpu/vk/webgpu_rt.h): bottom levels
// of triangles and of boxes, a top level of moved and scaled instances, rays cast from a
// compute kernel with ray queries (our WGSL compiler), each hit checked against the answer
// worked out here; then one instance moved, the top level built again, and checked again.
//
// Part of make test. Without our Vulkan layer (make VK=1) or a GPU with ray queries it says
// so and passes.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu/gpu.h"
#ifdef LR_GPU
#include "gpu/gpu_internal.h"
#endif

#if defined(LR_GPU) && defined(LR_VK)
static int failures;

static void check(const char *what, int ok){
    printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

// Each ray: from (x, y, -10) along +z. Out: t (-1 none), kind, custom data, primitive;
// the instance, the barycentrics, front facing.
GPU_WGSL(KERNEL, test_rt);

#define NX 32
#define NY 32

typedef struct { float s, x, y, z; uint32_t custom; int sphere; } Inst;   // scale, then move

static void origin(int i, float *x, float *y){
    *x = -9.0f + 18.0f * ((float)(i % NX) + 0.37f) / (float)NX;
    *y = -3.0f + 6.0f * ((float)(i / NX) + 0.61f) / (float)NY;
}

// the nearest hit of ray i, worked out here; 0 if too near an edge to call
typedef struct { float t; int kind; uint32_t custom; int prim, inst; float b1, b2; } Want;
static int expect(const Inst *in, int n, int i, Want *w){
    float x, y;
    origin(i, &x, &y);
    memset(w, 0, sizeof(*w));
    w->t = -1;
    for (int k = 0; k < n; k++){
        float u = (x - in[k].x) / in[k].s, v = (y - in[k].y) / in[k].s, t;
        if (in[k].sphere){
            float r2 = u * u + v * v;
            if (fabsf(r2 - 1) < 1e-3f) return 0;
            if (r2 > 1) continue;
            t = in[k].z - in[k].s * sqrtf(1 - r2) + 10;
        } else {
            if (fabsf(fabsf(u) - 1) < 1e-3f || fabsf(fabsf(v) - 1) < 1e-3f || fabsf(u - v) < 1e-3f) return 0;
            if (fabsf(u) > 1 || fabsf(v) > 1) continue;
            t = in[k].z + 10;
        }
        if (w->t >= 0 && t >= w->t) continue;
        w->t = t;
        w->kind = in[k].sphere ? 2 : 1;
        w->custom = in[k].custom;
        w->inst = k;
        if (!in[k].sphere){
            // triangle 0: (-1,-1) (1,-1) (1,1); 1: (-1,-1) (1,1) (-1,1)
            if (u > v){ w->prim = 0; w->b2 = (v + 1) / 2; w->b1 = (u + 1) / 2 - w->b2; }
            else { w->prim = 1; w->b1 = (u + 1) / 2; w->b2 = (v - u) / 2; }
        }
    }
    return 1;
}

static void on_map(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    *(int *)u1 = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

static int read_back(Gpu *g, WGPUBuffer src, size_t n, void *out){
    WGPUBuffer rb = gpu_buffer(g, "test: readback", n, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, rb, 0, n);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    int done = 0;
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowProcessEvents;
    mi.callback = on_map;
    mi.userdata1 = &done;
    wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, n, mi);
    for (int k = 0; k < 100000 && !done; k++) gpu_wait(g);
    int ok = done == 1;
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

static void records(const Inst *in, int n, LRBlas tri, LRBlas box, LRTlasInstance *r){
    memset(r, 0, (size_t)n * sizeof(*r));
    for (int k = 0; k < n; k++){
        r[k].transform[0][0] = r[k].transform[1][1] = r[k].transform[2][2] = in[k].s;
        r[k].transform[0][3] = in[k].x; r[k].transform[1][3] = in[k].y; r[k].transform[2][3] = in[k].z;
        r[k].custom_mask = in[k].custom | 0xffu << 24;
        r[k].blas = lrBlasGetAddress(in[k].sphere ? box : tri);
    }
}

// cast the rays, compare with what they should hit; the number of rays checked
static int cast(Gpu *g, const Inst *in, int n, WGPUComputePipeline pipe, WGPUBindGroup bg, WGPUBuffer out,
                LRTlas tlas, WGPUBuffer inst_buf, const char *what){
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    lrCommandEncoderBuildTlas(enc, tlas, inst_buf, 0, (uint32_t)n);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    gpu_dispatch(pass, NX * NY);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    float *o = malloc(NX * NY * 8 * sizeof(float));
    int ok = o && read_back(g, out, NX * NY * 8 * sizeof(float), o), checked = 0, hits = 0, bad = 0;
    for (int i = 0; ok && i < NX * NY; i++){
        Want w;
        if (!expect(in, n, i, &w)) continue;
        checked++;
        const float *a = o + 8 * i;
        int good = fabsf(a[0] - w.t) < 1e-3f * (1 + fabsf(w.t));
        if (w.t >= 0){
            hits++;
            good = good && (int)a[1] == w.kind && (uint32_t)a[2] == w.custom && (int)a[4] == w.inst;
            if (w.kind == 1) good = good && (int)a[3] == w.prim && fabsf(a[5] - w.b1) < 1e-3f && fabsf(a[6] - w.b2) < 1e-3f;
        } else good = good && (int)a[1] == 0;
        if (!good && bad++ < 5){
            float x, y; origin(i, &x, &y);
            printf("    ray %d (%.3f, %.3f): got t %.4f kind %d custom %d prim %d inst %d b %.4f %.4f; want t %.4f kind %d custom %u prim %d inst %d b %.4f %.4f\n",
                   i, x, y, a[0], (int)a[1], (int)a[2], (int)a[3], (int)a[4], a[5], a[6], w.t, w.kind, w.custom, w.prim, w.inst, w.b1, w.b2);
        }
    }
    char line[160];
    snprintf(line, sizeof(line), "%s: %d rays (%d hits) as expected", what, checked - bad, hits);
    check(line, ok && !bad && checked > NX * NY / 2 && hits > 50);
    free(o);
    return checked;
}
#endif

int main(void){
    Gpu *g = gpu_create(NULL, 64, 64);
    for (int k = 0; k < 20000 && g && gpu_state(g) == GPU_PENDING; k++) gpu_poll(g);
    if (!g || gpu_state(g) != GPU_READY){
        printf("test_gpu_rt: no GPU (%s): skipped\n", g ? gpu_message(g) : "none");
        if (g) gpu_destroy(g);
        return 0;
    }
#if defined(LR_GPU) && defined(LR_VK)
    if (!g->rt){
        printf("test_gpu_rt: the GPU has no ray queries: skipped\n");
        gpu_destroy(g);
        return 0;
    }
    printf("hardware ray tracing: acceleration structures and ray queries\n");
    // a square of two triangles, and a box, in their own space
    static const float tri[18] = {-1, -1, 0, 1, -1, 0, 1, 1, 0,   -1, -1, 0, 1, 1, 0, -1, 1, 0};
    static const float box[6] = {-1, -1, -1, 1, 1, 1};
    WGPUBufferUsage use = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    WGPUBuffer tb = gpu_buffer(g, "test: triangles", sizeof(tri), use), bb = gpu_buffer(g, "test: boxes", sizeof(box), use);
    wgpuQueueWriteBuffer(g->queue, tb, 0, tri, sizeof(tri));
    wgpuQueueWriteBuffer(g->queue, bb, 0, box, sizeof(box));
    LRGeometry gt = {LRGeometry_Triangles, tb, 0, 12, 2, 1}, gb = {LRGeometry_Boxes, bb, 0, 24, 1, 0};
    LRBlasDescriptor dt = {gpu_str("test: squares"), 1, &gt, 0}, db = {gpu_str("test: spheres"), 1, &gb, 0};
    LRBlas bt = lrDeviceCreateBlas(g->device, &dt), bx = lrDeviceCreateBlas(g->device, &db);
    LRTlas tlas = lrDeviceCreateTlas(g->device, 8, gpu_str("test: instances"));
    check("acceleration structures made", bt && bx && tlas);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    lrCommandEncoderBuildBlas(enc, bt);
    lrCommandEncoderBuildBlas(enc, bx);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    Inst in[3] = {{1, 0, 0, 5, 7, 0}, {2, 5, 0, 10, 9, 1}, {3, -5, 0, 20, 11, 0}};
    LRTlasInstance rec[3];
    records(in, 3, bt, bx, rec);
    WGPUBuffer ib = gpu_buffer(g, "test: instance records", sizeof(rec), use);
    wgpuQueueWriteBuffer(g->queue, ib, 0, rec, sizeof(rec));
    WGPUBuffer out = gpu_buffer(g, "test: hits", NX * NY * 8 * sizeof(float), use);

    WGPUBindGroupLayout bgl = gpu_layout(g, WGPUShaderStage_Compute, "AW");
    WGPUComputePipeline pipe = gpu_compute(g, "test: rays", KERNEL, "main", bgl);
    check("a kernel with ray queries compiled", pipe != NULL);
    LRBindGroupEntryTlas te;
    memset(&te, 0, sizeof(te));
    te.chain.sType = LRSType_BindGroupEntryTlas;
    te.tlas = tlas;
    WGPUBindGroupEntry e[2];
    memset(e, 0, sizeof(e));
    e[0].binding = 0;
    e[0].nextInChain = &te.chain;
    e[1].binding = 1;
    e[1].buffer = out;
    e[1].size = WGPU_WHOLE_SIZE;
    WGPUBindGroupDescriptor bd;
    memset(&bd, 0, sizeof(bd));
    bd.layout = bgl;
    bd.entryCount = 2;
    bd.entries = e;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g->device, &bd);
    if (pipe && bg && bt && bx && tlas){
        cast(g, in, 3, pipe, bg, out, tlas, ib, "squares and a sphere, moved and scaled");
        // move the first square behind the sphere's plane: the top level built again
        in[0].z = 8;
        in[0].x = 4;
        records(in, 3, bt, bx, rec);
        wgpuQueueWriteBuffer(g->queue, ib, 0, rec, sizeof(rec));
        cast(g, in, 3, pipe, bg, out, tlas, ib, "one moved, the top level built again");
    }
    if (bg) wgpuBindGroupRelease(bg);
    if (pipe) wgpuComputePipelineRelease(pipe);
    wgpuBindGroupLayoutRelease(bgl);
    lrTlasRelease(tlas); lrBlasRelease(bt); lrBlasRelease(bx);
    wgpuBufferRelease(tb); wgpuBufferRelease(bb); wgpuBufferRelease(ib); wgpuBufferRelease(out);
    gpu_destroy(g);
    printf(failures ? "test_gpu_rt: %d FAILED\n" : "all hardware ray tracing tests passed\n", failures);
    return failures ? 1 : 0;
#else
    printf("test_gpu_rt: hardware ray tracing needs our Vulkan layer (make VK=1): skipped\n");
    gpu_destroy(g);
    return 0;
#endif
}
