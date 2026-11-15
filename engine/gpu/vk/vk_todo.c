// vk_todo.c - the part of webgpu.h the Vulkan layer does not do yet: the window's surface.
// It says so once, and gives nothing: the engine then draws offscreen.
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

WGPUSurface wgpuInstanceCreateSurface(WGPUInstance i, WGPUSurfaceDescriptor const *d){ (void)i; (void)d; todo("window surfaces"); return NULL; }
WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface s, WGPUAdapter a, WGPUSurfaceCapabilities *c){ (void)s; (void)a; (void)c; return WGPUStatus_Error; }
void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities c){ (void)c; }
void wgpuSurfaceConfigure(WGPUSurface s, WGPUSurfaceConfiguration const *c){ (void)s; (void)c; }
void wgpuSurfaceGetCurrentTexture(WGPUSurface s, WGPUSurfaceTexture *t){ (void)s; (void)t; }
WGPUStatus wgpuSurfacePresent(WGPUSurface s){ (void)s; return WGPUStatus_Error; }
void wgpuSurfaceRelease(WGPUSurface s){ (void)s; }
