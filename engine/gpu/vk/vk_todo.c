// vk_todo.c - the part of webgpu.h the Vulkan layer does not do yet: textures, render passes
// and the window's surface. Each says so once, and gives nothing: the engine then runs what
// does not need them (compute: the path tracer, physics; offscreen frames read back).
#include "gpu/vk/vk_impl.h"

#include <stdio.h>

static void todo(const char *what){
    static const char *said[32];
    for (int i = 0; i < 32; i++){
        if (said[i] == what) return;
        if (!said[i]){ said[i] = what; break; }
    }
    fprintf(stderr, "Vulkan layer: %s not implemented yet\n", what);
}

WGPUTexture wgpuDeviceCreateTexture(WGPUDevice d, WGPUTextureDescriptor const *t){ (void)d; (void)t; todo("textures"); return NULL; }
WGPUTextureView wgpuTextureCreateView(WGPUTexture t, WGPUTextureViewDescriptor const *d){ (void)t; (void)d; return NULL; }
void wgpuTextureAddRef(WGPUTexture t){ if (t) obj_ref(&t->o); }
void wgpuTextureDestroy(WGPUTexture t){ (void)t; }
void wgpuTextureRelease(WGPUTexture t){ if (t) obj_unref(&t->o); }
void wgpuTextureViewAddRef(WGPUTextureView v){ if (v) obj_ref(&v->o); }
void wgpuTextureViewRelease(WGPUTextureView v){ if (v) obj_unref(&v->o); }
void wgpuQueueWriteTexture(WGPUQueue q, WGPUTexelCopyTextureInfo const *dst, void const *data, size_t size,
                           WGPUTexelCopyBufferLayout const *layout, WGPUExtent3D const *ext){
    (void)q; (void)dst; (void)data; (void)size; (void)layout; (void)ext; todo("texture writes");
}
void wgpuCommandEncoderCopyTextureToBuffer(WGPUCommandEncoder e, WGPUTexelCopyTextureInfo const *src,
                                           WGPUTexelCopyBufferInfo const *dst, WGPUExtent3D const *size){
    (void)e; (void)src; (void)dst; (void)size; todo("texture copies");
}

WGPURenderPipeline wgpuDeviceCreateRenderPipeline(WGPUDevice d, WGPURenderPipelineDescriptor const *p){ (void)d; (void)p; todo("render pipelines"); return NULL; }
void wgpuRenderPipelineRelease(WGPURenderPipeline p){ (void)p; }
WGPURenderPassEncoder wgpuCommandEncoderBeginRenderPass(WGPUCommandEncoder e, WGPURenderPassDescriptor const *d){ (void)e; (void)d; todo("render passes"); return NULL; }
void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder p, WGPURenderPipeline r){ (void)p; (void)r; }
void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder p, uint32_t i, WGPUBindGroup g, size_t n, uint32_t const *o){ (void)p; (void)i; (void)g; (void)n; (void)o; }
void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder p, uint32_t a, uint32_t b, uint32_t c, uint32_t d){ (void)p; (void)a; (void)b; (void)c; (void)d; }
void wgpuRenderPassEncoderEnd(WGPURenderPassEncoder p){ (void)p; }
void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder p){ (void)p; }

WGPUSurface wgpuInstanceCreateSurface(WGPUInstance i, WGPUSurfaceDescriptor const *d){ (void)i; (void)d; todo("window surfaces"); return NULL; }
WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface s, WGPUAdapter a, WGPUSurfaceCapabilities *c){ (void)s; (void)a; (void)c; return WGPUStatus_Error; }
void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities c){ (void)c; }
void wgpuSurfaceConfigure(WGPUSurface s, WGPUSurfaceConfiguration const *c){ (void)s; (void)c; }
void wgpuSurfaceGetCurrentTexture(WGPUSurface s, WGPUSurfaceTexture *t){ (void)s; (void)t; }
WGPUStatus wgpuSurfacePresent(WGPUSurface s){ (void)s; return WGPUStatus_Error; }
void wgpuSurfaceRelease(WGPUSurface s){ (void)s; }
