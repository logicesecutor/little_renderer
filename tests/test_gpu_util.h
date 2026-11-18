// test_gpu_util.h - what the GPU tests share: a buffer read back to the CPU.
#ifndef LR_TEST_GPU_UTIL_H
#define LR_TEST_GPU_UTIL_H

#include <string.h>
#include "gpu/gpu_internal.h"

static inline void on_map(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    *(int *)u1 = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

// the buffer's first n bytes, read back
static inline int read_back(Gpu *g, WGPUBuffer src, size_t n, void *out){
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
