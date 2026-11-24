// denoise_gpu.c - a denoiser for the GPU path tracer's image, on the GPU.
//
// An edge-avoiding a-trous wavelet filter (Dammertz et al. 2010) steered by the variance of
// each pixel, as in SVGF (Schied et al. 2017) without its temporal part:
//   prep     the mean of each pixel's samples over its albedo (the irradiance, which is
//            smooth where the colour is not), and the variance of that mean: from the pixel's
//            own samples once it has 4, before that from its neighbours (a 7 x 7 window)
//   a-trous  5 passes of a 5 x 5 B3 spline with holes (1, 2, 4, 8, 16 pixels apart: a
//            65 x 65 footprint), each neighbour weighted by how alike it is: luminance
//            (relative to the noise expected there), normal and distance of the first
//            diffuse surface; the variance is filtered alongside (with the squared weights)
//   last     times the albedo again
// The guide (normal, distance, albedo) is that of the ray through the pixel's centre, through
// mirrors and glass to the first diffuse surface (trace_gpu.c). Lamps and the sky are not
// filtered; pixels seen through a medium are filtered on luminance only. As the samples add
// up the variance falls, the luminance weights tighten, and the filter fades out by itself.
//
// What the passes read is packed, because they are memory bound (25 taps, 5 times): per
// pixel 8 bytes of image (irradiance and variance, half floats) and 8 bytes of guide (the
// normal octahedron-encoded in 2 x 16 bits, the distance as a float whose sign tells the
// kind), against 48 bytes unpacked: about 3 times less to read.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdlib.h>
#include <string.h>

GPU_WGSL(DENOISE, denoise);

#define PASSES 5

struct GpuDenoise {
    Gpu *g;
    int w, h;
    WGPUBindGroupLayout bgl;
    WGPUComputePipeline prep, atrous;
    WGPUBuffer u[PASSES + 1];      // the parameters of each pass
    WGPUBuffer ul[PASSES];         // ... of pass i (1..PASSES-1) as the last one (fewer passes)
    WGPUBuffer a, b;               // ping-pong (irradiance and variance, packed)
    WGPUBuffer guide, out;         // the guide, packed; the image
    GpuBindCache bc[PASSES + 1];   // each pass's bind group, kept
    GpuBindCache bcl[PASSES];      // ... and as the last one
    GpuTimer *timer;               // its pass's GPU time (NULL: no timestamps)
};

void gpu_denoise_before_submit(GpuDenoise *d, WGPUCommandEncoder enc){ if (d) gpu_timer_finish(d->timer, enc); }
void gpu_denoise_after_submit(GpuDenoise *d){ if (d) gpu_timer_submitted(d->timer); }
double gpu_denoise_total_ms(const GpuDenoise *d){ return d ? gpu_timer_total_ms(d->timer, 0) : 0; }

GpuDenoise *gpu_denoise_create(Gpu *g, int width, int height){
    if (!g || g->state != GPU_READY) return NULL;
    GpuDenoise *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->g = g;
    d->w = width;
    d->h = height;
    d->bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRWWW");
    const char *const parts[] = {GPU_WGSL_UTIL, DENOISE};
    char *src = gpu_wgsl_join(parts, 2);
    d->prep = gpu_compute(g, "denoiser: prep", src ? src : DENOISE, "prep", d->bgl);
    d->atrous = gpu_compute(g, "denoiser: a-trous", src ? src : DENOISE, "atrous", d->bgl);
    free(src);
    size_t n = (size_t)width * height;
    d->a = gpu_buffer(g, "denoiser A", n * 8, WGPUBufferUsage_Storage);
    d->b = gpu_buffer(g, "denoiser B", n * 8, WGPUBufferUsage_Storage);
    d->guide = gpu_buffer(g, "denoiser guide", n * 8, WGPUBufferUsage_Storage);
    d->out = gpu_buffer(g, "denoised image", n * 16, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    d->timer = gpu_timer_create(g, 4, 1);
    int ok = 1;
    for (int i = 0; i <= PASSES; i++){
        for (int last = 0; last < 2; last++){
            if (last ? i == 0 || i == PASSES : 0) continue;      // (u[PASSES] is the last already)
            struct { uint32_t size[4]; float s[4]; } P = {
                {(uint32_t)width, (uint32_t)height, i ? 1u << (i - 1) : 0u, i == PASSES || last ? 1u : 0u},
                {4.0f, 0.0f, 1.0f, 0.0f}};
            WGPUBuffer *u = last ? &d->ul[i] : &d->u[i];
            *u = gpu_buffer(g, "denoiser parameters", sizeof(P), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
            if (*u) wgpuQueueWriteBuffer(g->queue, *u, 0, &P, sizeof(P));
            else ok = 0;
        }
    }
    if (!ok || !d->prep || !d->atrous || !d->a || !d->b || !d->guide || !d->out){ gpu_denoise_destroy(d); return NULL; }
    return d;
}

void gpu_denoise_encode(GpuDenoise *d, WGPUCommandEncoder enc, WGPUBuffer pixels, int passes){
    Gpu *g = d->g;
    if (passes < 1 || passes > PASSES) passes = PASSES;
    WGPUComputePassDescriptor pd;
    memset(&pd, 0, sizeof(pd));
    pd.timestampWrites = gpu_timer_next(d->timer, 0);
    if (pd.timestampWrites) gpu_timer_units(d->timer, 1);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pd);
    // prep: pixels -> a (and the guide); then a -> b, b -> a, ...; the last pass -> out
    for (int i = 0; i <= passes; i++){
        WGPUBuffer in = i == 0 ? d->b : (i % 2 ? d->a : d->b);
        WGPUBuffer to = i == 0 ? d->a : (i % 2 ? d->b : d->a);
        const int early_last = i == passes && passes < PASSES;
        WGPUBuffer bufs[6] = {early_last ? d->ul[i] : d->u[i], pixels, in, to, d->guide, d->out};
        WGPUBindGroup bg = gpu_bind_cached_buffers(g, early_last ? &d->bcl[i] : &d->bc[i], d->bgl, bufs, 6);   // (kept: pixels is one of two or three)
        wgpuComputePassEncoderSetPipeline(pass, i == 0 ? d->prep : d->atrous);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        gpu_dispatch(pass, (size_t)d->w * d->h);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

WGPUBuffer gpu_denoise_output(GpuDenoise *d){ return d->out; }

void gpu_denoise_destroy(GpuDenoise *d){
    if (!d) return;
    for (int i = 0; i <= PASSES; i++) gpu_bind_cache_clear(&d->bc[i]);
    for (int i = 0; i < PASSES; i++) gpu_bind_cache_clear(&d->bcl[i]);
    WGPUBuffer bufs[2 * PASSES + 5] = {d->a, d->b, d->guide, d->out};
    for (int i = 0; i <= PASSES; i++) bufs[4 + i] = d->u[i];
    for (int i = 0; i < PASSES; i++) bufs[5 + PASSES + i] = d->ul[i];
    for (int i = 0; i < 2 * PASSES + 5; i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    if (d->prep) wgpuComputePipelineRelease(d->prep);
    if (d->atrous) wgpuComputePipelineRelease(d->atrous);
    if (d->bgl) wgpuBindGroupLayoutRelease(d->bgl);
    gpu_timer_destroy(d->timer);
    free(d);
}

#else
typedef int denoise_gpu_unused;
#endif
