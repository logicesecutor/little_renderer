// test_gpu_basic.c - the GPU layer's basics, whatever implements webgpu.h (wgpu-native, or our
// Vulkan layer: make VK=1): buffer writes small and large, copies, clears, read back by
// mapping, several submissions in flight, waiting for one of them. No shaders.
//
// Part of make test. Without a GPU it says so and passes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu/gpu.h"
#ifdef LR_GPU
#include "gpu/gpu_internal.h"
#endif

static int failures;

static void check(const char *what, int ok){
    printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

#ifdef LR_GPU
static void on_map(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    *(int *)u1 = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

// the buffer's first n bytes, read back
static int read_back(Gpu *g, WGPUBuffer src, size_t n, unsigned char *out){
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
#endif

int main(void){
    Gpu *g = gpu_create(NULL, 64, 64);
    for (int k = 0; k < 20000 && g && gpu_state(g) == GPU_PENDING; k++) gpu_poll(g);
    if (!g || gpu_state(g) != GPU_READY){
        printf("test_gpu_basic: no GPU (%s): skipped\n", g ? gpu_message(g) : "none");
        if (g) gpu_destroy(g);
        return 0;
    }
#ifdef LR_GPU
    printf("the GPU layer's basics: buffers, copies, clears, readback\n");
    const size_t small = 1024, big = 3u << 20;          // (written in the command buffer / through staging memory)
    unsigned char *a = malloc(big), *b = malloc(big);
    for (size_t i = 0; i < big; i++) a[i] = (unsigned char)(i * 7 + (i >> 11));
    WGPUBufferUsage use = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer s = gpu_buffer(g, "test: small", small, use), l = gpu_buffer(g, "test: big", big, use), c = gpu_buffer(g, "test: copy", big, use);
    wgpuQueueWriteBuffer(g->queue, s, 0, a, small);
    check("a small write, read back", read_back(g, s, small, b) && !memcmp(a, b, small));
    wgpuQueueWriteBuffer(g->queue, l, 0, a, big);
    check("a 3 MB write, read back", read_back(g, l, big, b) && !memcmp(a, b, big));
    // two writes to the same place before a submit: the second wins
    wgpuQueueWriteBuffer(g->queue, s, 16, a + 100, 64);
    wgpuQueueWriteBuffer(g->queue, s, 16, a + 200, 64);
    check("two writes in a row, the last one wins", read_back(g, s, small, b) && !memcmp(b + 16, a + 200, 64) && !memcmp(b, a, 16));
    // copy, then clear a part, in one command buffer
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, l, 0, c, 0, big);
    wgpuCommandEncoderClearBuffer(enc, c, 4096, 8192);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    uint64_t idx = gpu_submit(g, cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    gpu_wait_for(g, idx);
    int ok = read_back(g, c, big, b);
    for (size_t i = 0; ok && i < big; i++) if (b[i] != (i >= 4096 && i < 4096 + 8192 ? 0 : a[i])) ok = 0;
    check("a copy, then a clear of part of it", ok);
    // many submissions in flight, then the last one's result
    for (int k = 0; k < 50; k++){
        unsigned char v[256];
        memset(v, k, sizeof(v));
        wgpuQueueWriteBuffer(g->queue, s, 0, v, sizeof(v));
        enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, s, 0, c, (uint64_t)k * 256, 256);
        cb = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
    }
    ok = read_back(g, c, 50 * 256, b);
    for (int k = 0; ok && k < 50 * 256; k++) if (b[k] != (unsigned char)(k / 256)) ok = 0;
    check("50 submissions, each copying what was written before it", ok);
    wgpuBufferDestroy(s); wgpuBufferRelease(s);
    wgpuBufferDestroy(l); wgpuBufferRelease(l);
    wgpuBufferDestroy(c); wgpuBufferRelease(c);
    free(a); free(b);
#endif
    gpu_destroy(g);
    printf(failures ? "test_gpu_basic: %d FAILED\n" : "all GPU basics passed\n", failures);
    return failures ? 1 : 0;
}
