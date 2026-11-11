#define _POSIX_C_SOURCE 200809L   // clock_gettime
#include "gpu/gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LR_GPU
// ---- built without webgpu.h: no GPU, the CPU backends do everything ------------------------

struct Gpu { int unused; };

int gpu_available(void){ return 0; }
Gpu *gpu_create(const char *canvas, int width, int height){
    (void)canvas; (void)width; (void)height;
    return calloc(1, sizeof(Gpu));
}
Gpu *gpu_create_window(void *display, uintptr_t window, int width, int height){
    (void)display; (void)window; (void)width; (void)height;
    return calloc(1, sizeof(Gpu));
}
GpuState gpu_state(const Gpu *g){ (void)g; return GPU_FAILED; }
void gpu_poll(Gpu *g){ (void)g; }
void gpu_wait(Gpu *g){ (void)g; }
const char *gpu_message(const Gpu *g){ (void)g; return "this build has no GPU backend (make deps, then make; or make web)"; }
int gpu_offscreen(const Gpu *g){ (void)g; return 0; }
int gpu_software(const Gpu *g){ (void)g; return 0; }
void gpu_destroy(Gpu *g){ free(g); }

#else
#include "gpu/gpu_internal.h"
#ifdef LR_WGPU_NATIVE
#include <webgpu/wgpu.h>          // wgpuDevicePoll: wgpu-native's way to wait for the GPU
#endif

int gpu_available(void){ return 1; }

static void fail(Gpu *g, const char *what, WGPUStringView msg){
    g->state = GPU_FAILED;
    snprintf(g->message, sizeof(g->message), "%s: %.*s", what, msg.data ? (int)(msg.length == WGPU_STRLEN ? strlen(msg.data) : msg.length) : 0,
             msg.data ? msg.data : "");
    fprintf(stderr, "GPU: %s\n", g->message);
}

static void on_error(WGPUDevice const *device, WGPUErrorType type, WGPUStringView msg, void *u1, void *u2){
    (void)device; (void)u1; (void)u2;
    fprintf(stderr, "GPU error %d: %.*s\n", (int)type, (int)(msg.length == WGPU_STRLEN ? strlen(msg.data) : msg.length), msg.data);
}

// the GPUs alive: a device-lost callback can arrive after gpu_destroy, and must not touch it
#define MAX_LIVE 4
static Gpu *live[MAX_LIVE];
static int is_live(const Gpu *g){
    for (int i = 0; i < MAX_LIVE; i++) if (g && live[i] == g) return 1;
    return 0;
}

static void on_lost(WGPUDevice const *device, WGPUDeviceLostReason reason, WGPUStringView msg, void *u1, void *u2){
    (void)device; (void)u2;
    Gpu *g = u1;
    if (!is_live(g)) return;   // our own gpu_destroy
    // "destroyed" while we still use it: the browser took the device away (its GPU process
    // was reset, the tab was moved to another GPU...)
    fail(g, reason == WGPUDeviceLostReason_Destroyed ? "device destroyed by the browser" : "device lost", msg);
}

static void configure_surface(Gpu *g){
    if (g->offscreen){
        g->surface_format = WGPUTextureFormat_RGBA8Unorm;
        return;
    }
#ifdef __EMSCRIPTEN__
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector src;
    memset(&src, 0, sizeof(src));
    src.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    src.selector = gpu_str(g->canvas);
    WGPUSurfaceDescriptor sd;
    memset(&sd, 0, sizeof(sd));
    sd.nextInChain = &src.chain;
    g->surface = wgpuInstanceCreateSurface(g->instance, &sd);
#endif
    if (!g->surface) return;   // no window: compute and offscreen only
    WGPUSurfaceCapabilities caps;
    memset(&caps, 0, sizeof(caps));
    wgpuSurfaceGetCapabilities(g->surface, g->adapter, &caps);
    g->surface_format = caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
    // a non-sRGB format: the frames are already sRGB encoded, as on the CPU path
    for (size_t i = 0; i < caps.formatCount; i++)
        if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm || caps.formats[i] == WGPUTextureFormat_RGBA8Unorm){
            g->surface_format = caps.formats[i];
            break;
        }
    // opaque if the platform has it (a browser canvas does; some window systems only "auto")
    WGPUCompositeAlphaMode alpha = caps.alphaModeCount ? caps.alphaModes[0] : WGPUCompositeAlphaMode_Auto;
    for (size_t i = 0; i < caps.alphaModeCount; i++)
        if (caps.alphaModes[i] == WGPUCompositeAlphaMode_Opaque) alpha = WGPUCompositeAlphaMode_Opaque;
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    WGPUSurfaceConfiguration cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.device = g->device;
    cfg.format = g->surface_format;
    cfg.usage = WGPUTextureUsage_RenderAttachment;
    cfg.width = (uint32_t)g->width;
    cfg.height = (uint32_t)g->height;
    cfg.alphaMode = alpha;
    cfg.presentMode = WGPUPresentMode_Fifo;      // in step with the display: no tearing
    wgpuSurfaceConfigure(g->surface, &cfg);
}

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView msg, void *u1, void *u2){
    (void)u2;
    Gpu *g = u1;
    if (status != WGPURequestDeviceStatus_Success){ fail(g, "no device", msg); return; }
    g->device = device;
    g->queue = wgpuDeviceGetQueue(device);
    configure_surface(g);
    WGPUAdapterInfo info;
    memset(&info, 0, sizeof(info));
    wgpuAdapterGetInfo(g->adapter, &info);
    // vendor, architecture, device, driver: whichever the implementation fills in (a browser
    // tells little, a native one names the card and its driver)
    {
        const WGPUStringView parts[4] = {info.vendor, info.architecture, info.device, info.description};
        size_t at = 0;
        g->message[0] = 0;
        for (int k = 0; k < 4; k++){
            WGPUStringView p = parts[k];
            size_t n = p.data ? (p.length == WGPU_STRLEN ? strlen(p.data) : p.length) : 0;
            while (n && p.data[n - 1] == ' ') n--;
            if (!n || at + n + 2 >= sizeof(g->message)) continue;
            if (at){ memcpy(g->message + at, k == 3 ? ", " : " ", k == 3 ? 2 : 1); at += k == 3 ? 2 : 1; }
            memcpy(g->message + at, p.data, n);
            at += n;
            g->message[at] = 0;
        }
    }
    // a software adapter (SwiftShader, llvmpipe, WARP): WebGPU emulated on the CPU
    g->software = info.adapterType == WGPUAdapterType_CPU;
    {
        char low[sizeof(g->message)];
        size_t i = 0;
        for (; g->message[i]; i++) low[i] = (char)(g->message[i] >= 'A' && g->message[i] <= 'Z' ? g->message[i] + 32 : g->message[i]);
        low[i] = 0;
        if (strstr(low, "swiftshader") || strstr(low, "llvmpipe") || strstr(low, "software") || strstr(low, "basic render"))
            g->software = 1;
    }
    wgpuAdapterInfoFreeMembers(info);
    g->state = GPU_READY;
    printf("GPU ready: %s\n", g->message);
}

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView msg, void *u1, void *u2){
    (void)u2;
    Gpu *g = u1;
    if (status != WGPURequestAdapterStatus_Success){ fail(g, "no GPU adapter (no WebGPU in this browser, or no GPU driver)", msg); return; }
    g->adapter = adapter;
    WGPULimits lim;
    memset(&lim, 0, sizeof(lim));
    wgpuAdapterGetLimits(adapter, &lim);
    WGPUDeviceDescriptor dd;
    memset(&dd, 0, sizeof(dd));
    dd.label = gpu_str("little_renderer");
    // ask for the largest storage buffers the adapter allows: the cloud's grids are big
    WGPULimits want;
    memset(&want, 0, sizeof(want));
    want = lim;
    dd.requiredLimits = &want;
    g->max_storage = lim.maxStorageBufferBindingSize < lim.maxBufferSize ? lim.maxStorageBufferBindingSize : lim.maxBufferSize;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    dd.uncapturedErrorCallbackInfo.userdata1 = g;
    dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    dd.deviceLostCallbackInfo.callback = on_lost;
    dd.deviceLostCallbackInfo.userdata1 = g;
    WGPURequestDeviceCallbackInfo cb;
    memset(&cb, 0, sizeof(cb));
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_device;
    cb.userdata1 = g;
    wgpuAdapterRequestDevice(adapter, &dd, cb);
}

// natively: the window's surface, made before the adapter is chosen (it must be able to show it)
static void native_surface(Gpu *g){
#if !defined(__EMSCRIPTEN__)
    if (!g->native_window) return;
    WGPUSurfaceDescriptor sd;
    memset(&sd, 0, sizeof(sd));
#ifdef _WIN32
    WGPUSurfaceSourceWindowsHWND src;
    memset(&src, 0, sizeof(src));
    src.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
    src.hinstance = g->native_display;
    src.hwnd = (void *)(uintptr_t)g->native_window;
#else
    WGPUSurfaceSourceXlibWindow src;
    memset(&src, 0, sizeof(src));
    src.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
    src.display = g->native_display;
    src.window = g->native_window;
#endif
    sd.nextInChain = &src.chain;
    g->surface = wgpuInstanceCreateSurface(g->instance, &sd);
    if (!g->surface) fprintf(stderr, "GPU: no surface for the window: drawing offscreen\n");
    g->offscreen = g->surface == NULL;
#else
    (void)g;
#endif
}

static Gpu *create(const char *canvas, void *display, uint64_t window, int width, int height){
    Gpu *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->state = GPU_PENDING;
    g->width = width;
    g->height = height;
    g->offscreen = canvas == NULL && window == 0;
    g->native_display = display;
    g->native_window = window;
    snprintf(g->canvas, sizeof(g->canvas), "%s", canvas ? canvas : "");
    for (int i = 0; i < MAX_LIVE; i++) if (!live[i]){ live[i] = g; break; }
    snprintf(g->message, sizeof(g->message), "starting");
    g->instance = wgpuCreateInstance(NULL);
    if (!g->instance){ snprintf(g->message, sizeof(g->message), "no WebGPU instance"); g->state = GPU_FAILED; return g; }
    native_surface(g);
    WGPURequestAdapterOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.powerPreference = WGPUPowerPreference_HighPerformance;
    opt.compatibleSurface = g->surface;
    WGPURequestAdapterCallbackInfo cb;
    memset(&cb, 0, sizeof(cb));
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_adapter;
    cb.userdata1 = g;
    wgpuInstanceRequestAdapter(g->instance, &opt, cb);
    return g;
}

Gpu *gpu_create(const char *canvas, int width, int height){ return create(canvas, NULL, 0, width, height); }
Gpu *gpu_create_window(void *display, uintptr_t window, int width, int height){
    return create(NULL, display, (uint64_t)window, width, height);
}

void gpu_poll(Gpu *g){
#ifndef __EMSCRIPTEN__
    if (!g || !g->instance) return;
    wgpuInstanceProcessEvents(g->instance);
#ifdef LR_WGPU_NATIVE
    if (g->device) wgpuDevicePoll(g->device, 0, NULL);
#endif
#else
    (void)g;
#endif
}

#ifndef __EMSCRIPTEN__
static void on_idle(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg; (void)u2;
    *(int *)u1 = 1;
}
#endif

void gpu_wait(Gpu *g){
#ifndef __EMSCRIPTEN__
    if (!g || g->state != GPU_READY) return;
    int done = 0;
    WGPUQueueWorkDoneCallbackInfo wd;
    memset(&wd, 0, sizeof(wd));
    wd.mode = WGPUCallbackMode_AllowProcessEvents;
    wd.callback = on_idle;
    wd.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(g->queue, wd);
    while (!done && g->state == GPU_READY){
#ifdef LR_WGPU_NATIVE
        wgpuDevicePoll(g->device, 1, NULL);   // sleeps until the GPU is done
#endif
        gpu_poll(g);
    }
    gpu_poll(g);                              // and the callbacks that this completed
#else
    (void)g;
#endif
}

GpuState gpu_state(const Gpu *g){
    if (!g) return GPU_FAILED;
    gpu_poll((Gpu *)g);                       // natively: deliver what is ready
    return g->state;
}
const char *gpu_message(const Gpu *g){ return g ? g->message : "no GPU"; }
int gpu_offscreen(const Gpu *g){ return g && g->offscreen; }
int gpu_software(const Gpu *g){ return g && g->software; }

void gpu_destroy(Gpu *g){
    if (!g) return;
    for (int i = 0; i < MAX_LIVE; i++) if (live[i] == g) live[i] = NULL;
    if (g->surface) wgpuSurfaceRelease(g->surface);
    if (g->queue) wgpuQueueRelease(g->queue);
    if (g->device) wgpuDeviceRelease(g->device);
    if (g->adapter) wgpuAdapterRelease(g->adapter);
    if (g->instance) wgpuInstanceRelease(g->instance);
    free(g);
}

// ---- helpers -------------------------------------------------------------------------------

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
double gpu_now_ms(void){ return emscripten_get_now(); }
#else
#include <time.h>
double gpu_now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

WGPUShaderModule gpu_shader(Gpu *g, const char *label, const char *wgsl){
    WGPUShaderSourceWGSL src;
    memset(&src, 0, sizeof(src));
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = gpu_str(wgsl);
    WGPUShaderModuleDescriptor d;
    memset(&d, 0, sizeof(d));
    d.nextInChain = &src.chain;
    d.label = gpu_str(label);
    return wgpuDeviceCreateShaderModule(g->device, &d);
}

WGPUBuffer gpu_buffer(Gpu *g, const char *label, size_t size, WGPUBufferUsage usage){
    WGPUBufferDescriptor d;
    memset(&d, 0, sizeof(d));
    d.label = gpu_str(label);
    d.size = (size + 3) & ~(size_t)3;
    if (d.size < 16) d.size = 16;
    d.usage = usage;
    return wgpuDeviceCreateBuffer(g->device, &d);
}

int gpu_buffer_reserve(Gpu *g, WGPUBuffer *buf, size_t *cap, size_t size, WGPUBufferUsage usage, const char *label){
    if (*buf && *cap >= size) return 0;
    if (*buf){ wgpuBufferDestroy(*buf); wgpuBufferRelease(*buf); }
    size_t want = size + size / 4 + 1024;   // a little room to grow without recreating
    *buf = gpu_buffer(g, label, want, usage);
    *cap = want;
    return 1;
}

WGPUBindGroupLayout gpu_layout(Gpu *g, WGPUShaderStage stage, const char *types){
    WGPUBindGroupLayoutEntry e[16];
    memset(e, 0, sizeof(e));
    int n = 0;
    for (; types[n] && n < 16; n++){
        e[n].binding = (uint32_t)n;
        e[n].visibility = stage;
        e[n].buffer.type = types[n] == 'U' ? WGPUBufferBindingType_Uniform
                         : types[n] == 'R' ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
    }
    WGPUBindGroupLayoutDescriptor d;
    memset(&d, 0, sizeof(d));
    d.entryCount = (size_t)n;
    d.entries = e;
    return wgpuDeviceCreateBindGroupLayout(g->device, &d);
}

WGPUComputePipeline gpu_compute(Gpu *g, const char *label, const char *wgsl, const char *entry, WGPUBindGroupLayout bgl){
    WGPUShaderModule m = gpu_shader(g, label, wgsl);
    WGPUPipelineLayoutDescriptor ld;
    memset(&ld, 0, sizeof(ld));
    ld.bindGroupLayoutCount = 1;
    ld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g->device, &ld);
    WGPUComputePipelineDescriptor cd;
    memset(&cd, 0, sizeof(cd));
    cd.label = gpu_str(label);
    cd.layout = pl;
    cd.compute.module = m;
    cd.compute.entryPoint = gpu_str(entry);
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(g->device, &cd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(m);
    return p;
}

WGPUBindGroup gpu_bind(Gpu *g, WGPUBindGroupLayout bgl, const WGPUBuffer *bufs, int n){
    WGPUBindGroupEntry e[16];
    memset(e, 0, sizeof(e));
    for (int i = 0; i < n && i < 16; i++){
        e[i].binding = (uint32_t)i;
        e[i].buffer = bufs[i];
        e[i].size = WGPU_WHOLE_SIZE;
    }
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = bgl;
    d.entryCount = (size_t)(n < 16 ? n : 16);
    d.entries = e;
    return wgpuDeviceCreateBindGroup(g->device, &d);
}

void gpu_dispatch(WGPUComputePassEncoder pass, size_t threads){
    if (!threads) return;
    size_t groups = (threads + 63) / 64;
    uint32_t gx = groups < 65535 ? (uint32_t)groups : 65535u;
    uint32_t gy = (uint32_t)((groups + gx - 1) / gx);
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
}

#endif   // LR_GPU
