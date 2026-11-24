// test_gpu_util.h - what the GPU tests share: a buffer read back to the CPU.
#ifndef LR_TEST_GPU_UTIL_H
#define LR_TEST_GPU_UTIL_H

#include <string.h>
#include "gpu/gpu_internal.h"

// the buffer's first n bytes, read back
static inline int read_back(Gpu *g, WGPUBuffer src, size_t n, void *out){
    WGPUBuffer rb = gpu_buffer(g, "test: readback", n, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, rb, 0, n);
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

#endif
