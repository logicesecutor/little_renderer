// brdf_gpu.c - the BSDF of core/brdf.h in WGSL (GPU_WGSL_BSDF), for the GPU's path tracer
// (and rasterizer), and a check of it against the C (gpu_bsdf_run, examples/test_gpu_brdf.c).
//
// The WGSL is the C function by function, in f32; core/brdf.h explains the model. A module
// that includes it defines bsdf_tab(k): float k of the tables gpu_bsdf_tables packs
// (core/brdf_tables.h: BRDF_E, BRDF_B, BRDF_E_AVG, BRDF_B_AVG, one after the other).
#include "gpu/brdf_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/brdf.h"
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

GPU_WGSL_EXPORT(GPU_WGSL_BSDF, bsdf);

void gpu_bsdf_tables(float *out){
    memcpy(out, BRDF_E, sizeof(BRDF_E));
    memcpy(out + 1024, BRDF_B, sizeof(BRDF_B));
    memcpy(out + 2048, BRDF_E_AVG, sizeof(BRDF_E_AVG));
    memcpy(out + 2080, BRDF_B_AVG, sizeof(BRDF_B_AVG));
}

// ---- the check: the WGSL on cases, against the C ----
GPU_WGSL(CHECK, brdf_check);

static void on_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    *(int *)u1 = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

int gpu_bsdf_run(Gpu *g, const float *in, int n, float *out){
    if (!g || g->state != GPU_READY || n <= 0) return 0;
    char *src = malloc(strlen(GPU_WGSL_BSDF) + strlen(CHECK) + 64);
    if (!src) return 0;
    sprintf(src, "const GPU_BSDF_TAB_VEC4 = %du;\n", GPU_BSDF_TABLE_FLOATS / 4);   // (the tables' size)
    strcat(src, GPU_WGSL_BSDF);
    strcat(src, "\n");
    strcat(src, CHECK);
    WGPUBindGroupLayout bgl = gpu_layout(g, WGPUShaderStage_Compute, "URW");
    WGPUComputePipeline pipe = gpu_compute(g, "BSDF check", src, "main", bgl);
    free(src);
    float tab[GPU_BSDF_TABLE_FLOATS];
    gpu_bsdf_tables(tab);
    size_t bin = (size_t)n * GPU_BSDF_IN * 4, bout = (size_t)n * GPU_BSDF_OUT * 4;
    WGPUBuffer bt = gpu_buffer(g, "BSDF check: tables", sizeof(tab), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bi = gpu_buffer(g, "BSDF check: cases", bin, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bo = gpu_buffer(g, "BSDF check: answers", bout, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer rb = gpu_buffer(g, "BSDF check: readback", bout, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(g->queue, bt, 0, tab, sizeof(tab));
    wgpuQueueWriteBuffer(g->queue, bi, 0, in, bin);
    WGPUBuffer bufs[3] = {bt, bi, bo};
    WGPUBindGroup bg = gpu_bind(g, bgl, bufs, 3);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)((n + 63) / 64), 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(enc, bo, 0, rb, 0, bout);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    int mapped = 0;
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = on_mapped;
    mi.userdata1 = &mapped;
    wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, bout, mi);
    for (int k = 0; k < 4000 && mapped == 0; k++) gpu_wait(g);
    int ok = mapped == 1;
    if (ok){
        const float *r = wgpuBufferGetConstMappedRange(rb, 0, bout);
        if (r) memcpy(out, r, bout); else ok = 0;
        wgpuBufferUnmap(rb);
    }
    wgpuBindGroupRelease(bg);
    WGPUBuffer all[4] = {bt, bi, bo, rb};
    for (int i = 0; i < 4; i++){ wgpuBufferDestroy(all[i]); wgpuBufferRelease(all[i]); }
    wgpuComputePipelineRelease(pipe);
    wgpuBindGroupLayoutRelease(bgl);
    return ok;
}

#else
int gpu_bsdf_run(Gpu *g, const float *in, int n, float *out){ (void)g; (void)in; (void)n; (void)out; return 0; }
#endif
