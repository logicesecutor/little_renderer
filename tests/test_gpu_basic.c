// test_gpu_basic.c - the GPU layer's basics, whatever implements webgpu.h (natively our
// Vulkan layer, in a page the browser's): buffer writes small and large, copies, clears, read back by
// mapping, several submissions in flight, waiting for one of them. No shaders. And the list of
// the objects alive (gpu_internal.h): an object is known by its ticket, not by its address.
//
// Part of make test. Without a GPU it says so and passes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu/gpu.h"
#include "test_util.h"
#ifdef LR_GPU
#include "gpu/gpu_internal.h"
#include "test_gpu_util.h"
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
    // the list of the objects alive: one made where another was destroyed is not that one (a
    // late callback of the first, given its ticket, finds nothing)
    {
        static int place;
        const int n0 = gpu_live_count();
        uintptr_t t1 = gpu_live_add(GPU_LIVE_VIEW, &place);
        int had = gpu_live_has(GPU_LIVE_VIEW, &place, t1);
        gpu_live_remove(GPU_LIVE_VIEW, &place);
        int gone = !gpu_live_has(GPU_LIVE_VIEW, &place, t1);
        uintptr_t t2 = gpu_live_add(GPU_LIVE_VIEW, &place);
        check("an object alive is found by its ticket, not once destroyed", t1 && had && gone);
        check("another made at its address is not it: the old ticket finds nothing", t2 && t2 != t1 &&
              !gpu_live_has(GPU_LIVE_VIEW, &place, t1) && gpu_live_has(GPU_LIVE_VIEW, &place, t2) &&
              !gpu_live_has(GPU_LIVE_TIMER, &place, t2) && !gpu_live_has(GPU_LIVE_VIEW, &place, 0));
        gpu_live_remove(GPU_LIVE_VIEW, &place);
        check("... and the list is as it was", gpu_live_count() == n0);
    }
#endif
    gpu_destroy(g);
    printf(failures ? "test_gpu_basic: %d FAILED\n" : "all GPU basics passed\n", failures);
    return failures ? 1 : 0;
}
