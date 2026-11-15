// vk_core.c - instance, adapter, device, queue, buffers, uploads, submissions (vk_impl.h)
#include "gpu/vk/vk_impl.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

VkFns vk;

// ---- objects ----
void obj_init(Obj *o, void (*free)(Obj *)){ o->refs = 1; o->free = free; }
void obj_ref(Obj *o){ if (o) o->refs++; }
void obj_unref(Obj *o){ if (o && --o->refs == 0 && o->free) o->free(o); }
void refs_add(Refs *r, Obj *o){
    if (!o) return;
    if (r->n == r->cap){
        int cap = r->cap ? 2 * r->cap : 16;
        Obj **v = realloc(r->v, (size_t)cap * sizeof(*v));
        if (!v) return;
        r->v = v; r->cap = cap;
    }
    obj_ref(o);
    r->v[r->n++] = o;
}
void refs_release(Refs *r){
    for (int i = 0; i < r->n; i++) obj_unref(r->v[i]);
    free(r->v);
    r->v = NULL; r->n = r->cap = 0;
}

static WGPUStringView sv(const char *s){ WGPUStringView v = {s, s ? strlen(s) : 0}; return v; }

void vk_error(WGPUDevice dev, WGPUErrorType type, const char *fmt, ...){
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (dev && dev->on_error.callback){
        WGPUDevice d = dev;
        dev->on_error.callback(&d, type, sv(msg), dev->on_error.userdata1, dev->on_error.userdata2);
    } else fprintf(stderr, "Vulkan layer: %s\n", msg);
}

// ---- the loader ----
static int load_loader(WGPUInstance in){
#ifdef _WIN32
    HMODULE h = LoadLibraryA("vulkan-1.dll");
    if (!h) return 0;
    vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void (*)(void))GetProcAddress(h, "vkGetInstanceProcAddr");
#else
    void *h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
    *(void **)&vk.vkGetInstanceProcAddr = dlsym(h, "vkGetInstanceProcAddr");
#endif
    in->lib = (void *)h;
    if (!vk.vkGetInstanceProcAddr) return 0;
#define X(f) vk.f = (PFN_##f)vk.vkGetInstanceProcAddr(NULL, #f); if (!vk.f) return 0;
    VK_GLOBAL_FNS(X)
#undef X
    return 1;
}

static int has_layer(const char *name){
    uint32_t n = 0;
    if (vk.vkEnumerateInstanceLayerProperties(&n, NULL) != VK_SUCCESS || !n) return 0;
    VkLayerProperties *p = malloc(n * sizeof(*p));
    if (!p) return 0;
    int found = 0;
    if (vk.vkEnumerateInstanceLayerProperties(&n, p) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++) if (!strcmp(p[i].layerName, name)) found = 1;
    free(p);
    return found;
}

static int has_instance_ext(const char *name){
    uint32_t n = 0;
    if (vk.vkEnumerateInstanceExtensionProperties(NULL, &n, NULL) != VK_SUCCESS || !n) return 0;
    VkExtensionProperties *p = malloc(n * sizeof(*p));
    if (!p) return 0;
    int found = 0;
    if (vk.vkEnumerateInstanceExtensionProperties(NULL, &n, p) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++) if (!strcmp(p[i].extensionName, name)) found = 1;
    free(p);
    return found;
}

static void instance_free(Obj *o){
    WGPUInstance in = (WGPUInstance)o;
    if (in->vk && vk.vkDestroyInstance) vk.vkDestroyInstance(in->vk, NULL);
    free(in);
}

WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const *descriptor){
    (void)descriptor;
    WGPUInstance in = calloc(1, sizeof(*in));
    if (!in) return NULL;
    obj_init(&in->o, instance_free);
    if (!load_loader(in)){ fprintf(stderr, "Vulkan layer: no Vulkan loader (vulkan-1.dll / libvulkan.so.1)\n"); free(in); return NULL; }
    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "little_renderer";
    app.pEngineName = "little_renderer";
    app.apiVersion = VK_API_VERSION_1_3;
    const char *layers[1];
    uint32_t nlayers = 0;
    // LR_VK_VALIDATE=1: the Khronos validation layer (development), if installed
    const char *val = getenv("LR_VK_VALIDATE");
    if (val && *val && *val != '0'){
        if (has_layer("VK_LAYER_KHRONOS_validation")){ layers[nlayers++] = "VK_LAYER_KHRONOS_validation"; in->validation = 1; }
        else fprintf(stderr, "Vulkan layer: LR_VK_VALIDATE set, but no validation layer installed\n");
    }
    const char *exts[4];
    uint32_t nexts = 0;
    if (has_instance_ext("VK_KHR_surface")){
        exts[nexts++] = "VK_KHR_surface";
#ifdef _WIN32
        if (has_instance_ext("VK_KHR_win32_surface")) exts[nexts++] = "VK_KHR_win32_surface";
#else
        if (has_instance_ext("VK_KHR_xlib_surface")) exts[nexts++] = "VK_KHR_xlib_surface";
#endif
    }
    VkInstanceCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = nlayers;
    ci.ppEnabledLayerNames = layers;
    ci.enabledExtensionCount = nexts;
    ci.ppEnabledExtensionNames = exts;
    VkResult r = vk.vkCreateInstance(&ci, NULL, &in->vk);
    if (r != VK_SUCCESS){ fprintf(stderr, "Vulkan layer: vkCreateInstance failed (%d)\n", r); free(in); return NULL; }
#define X(f) vk.f = (PFN_##f)vk.vkGetInstanceProcAddr(in->vk, #f); \
             if (!vk.f){ fprintf(stderr, "Vulkan layer: no %s\n", #f); instance_free(&in->o); return NULL; }
    VK_INSTANCE_FNS(X)
#undef X
    return in;
}

void wgpuInstanceRelease(WGPUInstance instance){ if (instance) obj_unref(&instance->o); }

// ---- adapter ----
static void adapter_free(Obj *o){ free(o); }

WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance instance, WGPURequestAdapterOptions const *options,
                                      WGPURequestAdapterCallbackInfo callbackInfo){
    (void)options;
    WGPUFuture fut = {1};
    uint32_t n = 0;
    VkPhysicalDevice pds[16];
    if (vk.vkEnumeratePhysicalDevices(instance->vk, &n, NULL) != VK_SUCCESS || !n){
        callbackInfo.callback(WGPURequestAdapterStatus_Unavailable, NULL, sv("no Vulkan device"), callbackInfo.userdata1, callbackInfo.userdata2);
        return fut;
    }
    if (n > 16) n = 16;
    vk.vkEnumeratePhysicalDevices(instance->vk, &n, pds);
    // a discrete GPU first, then integrated, then anything with Vulkan 1.3
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < n; i++){
        VkPhysicalDeviceProperties p;
        vk.vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (score > best_score){ best = (int)i; best_score = score; }
    }
    if (best < 0){
        callbackInfo.callback(WGPURequestAdapterStatus_Unavailable, NULL, sv("no Vulkan 1.3 device"), callbackInfo.userdata1, callbackInfo.userdata2);
        return fut;
    }
    WGPUAdapter ad = calloc(1, sizeof(*ad));
    if (!ad) return fut;
    obj_init(&ad->o, adapter_free);
    ad->inst = instance;
    ad->pd = pds[best];
    vk.vkGetPhysicalDeviceProperties(ad->pd, &ad->props);
    vk.vkGetPhysicalDeviceMemoryProperties(ad->pd, &ad->mem);
    uint32_t nf = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(ad->pd, &nf, NULL);
    VkQueueFamilyProperties qf[32];
    if (nf > 32) nf = 32;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(ad->pd, &nf, qf);
    ad->family = VK_QUEUE_FAMILY_IGNORED;
    for (uint32_t i = 0; i < nf; i++)
        if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)){
            ad->family = i;
            ad->ts_bits = qf[i].timestampValidBits;
            break;
        }
    if (ad->family == VK_QUEUE_FAMILY_IGNORED){
        free(ad);
        callbackInfo.callback(WGPURequestAdapterStatus_Unavailable, NULL, sv("no graphics and compute queue"), callbackInfo.userdata1, callbackInfo.userdata2);
        return fut;
    }
    callbackInfo.callback(WGPURequestAdapterStatus_Success, ad, sv(""), callbackInfo.userdata1, callbackInfo.userdata2);
    return fut;
}

void wgpuAdapterRelease(WGPUAdapter adapter){ if (adapter) obj_unref(&adapter->o); }

WGPUStatus wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits *l){
    const VkPhysicalDeviceLimits *v = &adapter->props.limits;
    WGPUChainedStruct *next = l->nextInChain;
    memset(l, 0, sizeof(*l));
    l->nextInChain = next;
    l->maxTextureDimension1D = v->maxImageDimension1D;
    l->maxTextureDimension2D = v->maxImageDimension2D;
    l->maxTextureDimension3D = v->maxImageDimension3D;
    l->maxTextureArrayLayers = v->maxImageArrayLayers;
    l->maxBindGroups = v->maxBoundDescriptorSets < 4 ? v->maxBoundDescriptorSets : 4;
    l->maxBindGroupsPlusVertexBuffers = 24;
    l->maxBindingsPerBindGroup = 1000;
    l->maxDynamicUniformBuffersPerPipelineLayout = v->maxDescriptorSetUniformBuffersDynamic;
    l->maxDynamicStorageBuffersPerPipelineLayout = v->maxDescriptorSetStorageBuffersDynamic;
    l->maxSampledTexturesPerShaderStage = v->maxPerStageDescriptorSampledImages;
    l->maxSamplersPerShaderStage = v->maxPerStageDescriptorSamplers;
    l->maxStorageBuffersPerShaderStage = v->maxPerStageDescriptorStorageBuffers;
    l->maxStorageTexturesPerShaderStage = v->maxPerStageDescriptorStorageImages;
    l->maxUniformBuffersPerShaderStage = v->maxPerStageDescriptorUniformBuffers;
    l->maxUniformBufferBindingSize = v->maxUniformBufferRange;
    l->maxStorageBufferBindingSize = v->maxStorageBufferRange;
    l->minUniformBufferOffsetAlignment = (uint32_t)(v->minUniformBufferOffsetAlignment > 256 ? v->minUniformBufferOffsetAlignment : 256);
    l->minStorageBufferOffsetAlignment = (uint32_t)(v->minStorageBufferOffsetAlignment > 256 ? v->minStorageBufferOffsetAlignment : 256);
    l->maxVertexBuffers = v->maxVertexInputBindings < 16 ? v->maxVertexInputBindings : 16;
    // the largest buffer: the largest device-local heap, at most 4 GB - 4 (a binding's range)
    uint64_t heap = 0;
    for (uint32_t i = 0; i < adapter->mem.memoryHeapCount; i++)
        if (adapter->mem.memoryHeaps[i].size > heap) heap = adapter->mem.memoryHeaps[i].size;
    l->maxBufferSize = heap < 0xfffffffcull ? heap : 0xfffffffcull;
    l->maxVertexAttributes = v->maxVertexInputAttributes;
    l->maxVertexBufferArrayStride = v->maxVertexInputBindingStride;
    l->maxInterStageShaderVariables = v->maxVertexOutputComponents / 4;
    l->maxColorAttachments = v->maxColorAttachments;
    l->maxColorAttachmentBytesPerSample = 32;
    l->maxComputeWorkgroupStorageSize = v->maxComputeSharedMemorySize;
    l->maxComputeInvocationsPerWorkgroup = v->maxComputeWorkGroupInvocations;
    l->maxComputeWorkgroupSizeX = v->maxComputeWorkGroupSize[0];
    l->maxComputeWorkgroupSizeY = v->maxComputeWorkGroupSize[1];
    l->maxComputeWorkgroupSizeZ = v->maxComputeWorkGroupSize[2];
    l->maxComputeWorkgroupsPerDimension = v->maxComputeWorkGroupCount[0] < 65535 ? v->maxComputeWorkGroupCount[0] : 65535;
    l->maxImmediateSize = v->maxPushConstantsSize;
    return WGPUStatus_Success;
}

WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature){
    if (feature == WGPUFeatureName_TimestampQuery) return adapter->ts_bits > 0 && adapter->props.limits.timestampComputeAndGraphics;
    return 0;
}

static WGPUStringView dup_sv(const char *s){
    size_t n = strlen(s);
    char *c = malloc(n + 1);
    if (!c) return sv(NULL);
    memcpy(c, s, n + 1);
    WGPUStringView v = {c, n};
    return v;
}

WGPUStatus wgpuAdapterGetInfo(WGPUAdapter adapter, WGPUAdapterInfo *info){
    const VkPhysicalDeviceProperties *p = &adapter->props;
    const char *vendor = p->vendorID == 0x10de ? "NVIDIA" : p->vendorID == 0x1002 ? "AMD" : p->vendorID == 0x8086 ? "Intel" :
                         p->vendorID == 0x10005 ? "Mesa" : "";
    char desc[96];
    snprintf(desc, sizeof(desc), "Vulkan %u.%u.%u (our layer)", p->apiVersion >> 22, (p->apiVersion >> 12) & 0x3ff, p->apiVersion & 0xfff);
    WGPUChainedStruct *next = info->nextInChain;
    memset(info, 0, sizeof(*info));
    info->nextInChain = next;
    info->vendor = dup_sv(vendor);
    info->architecture = dup_sv("");
    info->device = dup_sv(p->deviceName);
    info->description = dup_sv(desc);
    info->backendType = WGPUBackendType_Vulkan;
    info->adapterType = p->deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? WGPUAdapterType_DiscreteGPU :
                        p->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? WGPUAdapterType_IntegratedGPU :
                        p->deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? WGPUAdapterType_CPU : WGPUAdapterType_Unknown;
    info->vendorID = p->vendorID;
    info->deviceID = p->deviceID;
    return WGPUStatus_Success;
}

void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo info){
    free((void *)info.vendor.data);
    free((void *)info.architecture.data);
    free((void *)info.device.data);
    free((void *)info.description.data);
}

int vk_memory_type(WGPUAdapter ad, uint32_t bits, VkFlags want, VkFlags avoid){
    for (uint32_t i = 0; i < ad->mem.memoryTypeCount; i++){
        VkFlags f = ad->mem.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return (int)i;
    }
    for (uint32_t i = 0; i < ad->mem.memoryTypeCount; i++){
        VkFlags f = ad->mem.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want) return (int)i;
    }
    return -1;
}

// ---- device ----
static void device_free(Obj *o){
    WGPUDevice d = (WGPUDevice)o;
    if (d->vk){
        vk.vkDeviceWaitIdle(d->vk);
        wgpuDevicePoll(d, 1, NULL);
        refs_release(&d->up_refs);
        if (d->up_open){ vk.vkEndCommandBuffer(d->up); d->up_open = 0; }
        if (d->stage_cur){ obj_unref(&d->stage_cur->o); d->stage_cur = NULL; }   // (onto the free list)
        for (Staging *s = d->stage_free; s;){
            Staging *n = s->next;
            vk.vkDestroyBuffer(d->vk, s->buf, NULL);
            vk.vkFreeMemory(d->vk, s->mem, NULL);
            free(s);
            s = n;
        }
        for (int i = 0; i < d->ndpools; i++) vk.vkDestroyDescriptorPool(d->vk, d->dpools[i], NULL);
        if (d->pool) vk.vkDestroyCommandPool(d->vk, d->pool, NULL);
        if (d->tl) vk.vkDestroySemaphore(d->vk, d->tl, NULL);
        vk.vkDestroyDevice(d->vk, NULL);
    }
    if (d->ad && d->ad->inst && d->ad->inst->device == d) d->ad->inst->device = NULL;
    free(d->free_cbs); free(d->dpools); free(d->subs); free(d->pend);
    free(d);
}

WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter adapter, WGPUDeviceDescriptor const *descriptor,
                                    WGPURequestDeviceCallbackInfo callbackInfo){
    WGPUFuture fut = {2};
    WGPUDevice d = calloc(1, sizeof(*d));
    if (!d) return fut;
    obj_init(&d->o, device_free);
    d->ad = adapter;
    if (descriptor){
        d->on_error = descriptor->uncapturedErrorCallbackInfo;
        for (size_t i = 0; i < descriptor->requiredFeatureCount; i++)
            if (descriptor->requiredFeatures[i] == WGPUFeatureName_TimestampQuery) d->timestamps = 1;
    }
    // Vulkan 1.2 (timeline semaphores, buffer addresses) and 1.3 (dynamic rendering) features
    VkPhysicalDeviceVulkan13Features f13;
    memset(&f13, 0, sizeof(f13));
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features f12;
    memset(&f12, 0, sizeof(f12));
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 have;
    memset(&have, 0, sizeof(have));
    have.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    have.pNext = &f12;
    vk.vkGetPhysicalDeviceFeatures2(adapter->pd, &have);
    if (!f12.timelineSemaphore || !f13.dynamicRendering){
        free(d);
        callbackInfo.callback(WGPURequestDeviceStatus_Error, NULL, sv("the Vulkan device lacks timeline semaphores or dynamic rendering"),
                              callbackInfo.userdata1, callbackInfo.userdata2);
        return fut;
    }
    VkPhysicalDeviceVulkan13Features e13;
    memset(&e13, 0, sizeof(e13));
    e13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    e13.dynamicRendering = VK_TRUE;
    e13.synchronization2 = f13.synchronization2;
    e13.maintenance4 = f13.maintenance4;
    VkPhysicalDeviceVulkan12Features e12;
    memset(&e12, 0, sizeof(e12));
    e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    e12.pNext = &e13;
    e12.timelineSemaphore = VK_TRUE;
    e12.bufferDeviceAddress = f12.bufferDeviceAddress;
    e12.vulkanMemoryModel = f12.vulkanMemoryModel;
    e12.vulkanMemoryModelDeviceScope = f12.vulkanMemoryModelDeviceScope;
    e12.scalarBlockLayout = f12.scalarBlockLayout;
    e12.uniformBufferStandardLayout = f12.uniformBufferStandardLayout;
    e12.shaderFloat16 = 0;
    VkPhysicalDeviceFeatures2 want;
    memset(&want, 0, sizeof(want));
    want.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    want.pNext = &e12;
    want.features.shaderInt64 = have.features.shaderInt64;
    want.features.fragmentStoresAndAtomics = have.features.fragmentStoresAndAtomics;
    want.features.vertexPipelineStoresAndAtomics = have.features.vertexPipelineStoresAndAtomics;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi;
    memset(&qi, 0, sizeof(qi));
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = adapter->family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    const char *exts[4];
    uint32_t nexts = 0;
    exts[nexts++] = "VK_KHR_swapchain";
    {   // (only if the device has it: an offscreen-only device may not)
        uint32_t n = 0;
        vk.vkEnumerateDeviceExtensionProperties(adapter->pd, NULL, &n, NULL);
        VkExtensionProperties *p = malloc((n ? n : 1) * sizeof(*p));
        int found = 0;
        if (p && vk.vkEnumerateDeviceExtensionProperties(adapter->pd, NULL, &n, p) == VK_SUCCESS)
            for (uint32_t i = 0; i < n; i++) if (!strcmp(p[i].extensionName, "VK_KHR_swapchain")) found = 1;
        free(p);
        if (!found) nexts = 0;
    }
    VkDeviceCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &want;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = nexts;
    ci.ppEnabledExtensionNames = exts;
    VkResult r = vk.vkCreateDevice(adapter->pd, &ci, NULL, &d->vk);
    if (r != VK_SUCCESS){
        free(d);
        callbackInfo.callback(WGPURequestDeviceStatus_Error, NULL, sv("vkCreateDevice failed"), callbackInfo.userdata1, callbackInfo.userdata2);
        return fut;
    }
#define X(f) vk.f = (PFN_##f)vk.vkGetDeviceProcAddr(d->vk, #f); \
             if (!vk.f){ fprintf(stderr, "Vulkan layer: no %s\n", #f); }
    VK_DEVICE_FNS(X)
#undef X
    vk.vkGetDeviceQueue(d->vk, adapter->family, 0, &d->q);
    obj_init(&d->queue.o, NULL);
    d->queue.dev = d;
    VkCommandPoolCreateInfo pi;
    memset(&pi, 0, sizeof(pi));
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = adapter->family;
    vk.vkCreateCommandPool(d->vk, &pi, NULL, &d->pool);
    VkSemaphoreTypeCreateInfo ti;
    memset(&ti, 0, sizeof(ti));
    ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    ti.semaphoreType = 1;
    VkSemaphoreCreateInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    si.pNext = &ti;
    vk.vkCreateSemaphore(d->vk, &si, NULL, &d->tl);
    adapter->inst->device = d;
    callbackInfo.callback(WGPURequestDeviceStatus_Success, d, sv(""), callbackInfo.userdata1, callbackInfo.userdata2);
    return fut;
}

WGPUQueue wgpuDeviceGetQueue(WGPUDevice device){ return &device->queue; }
void wgpuQueueRelease(WGPUQueue queue){ (void)queue; }
void wgpuDeviceRelease(WGPUDevice device){ if (device) obj_unref(&device->o); }

// ---- command buffers ----
VkCommandBuffer vk_cb_get(WGPUDevice dev){
    VkCommandBuffer cb = NULL;
    if (dev->nfree_cbs) cb = dev->free_cbs[--dev->nfree_cbs];
    else {
        VkCommandBufferAllocateInfo ai;
        memset(&ai, 0, sizeof(ai));
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = dev->pool;
        ai.commandBufferCount = 1;
        if (vk.vkAllocateCommandBuffers(dev->vk, &ai, &cb) != VK_SUCCESS) return NULL;
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cb, &bi);
    vk_barrier(cb);                   // after everything submitted before
    return cb;
}

void vk_cb_put(WGPUDevice dev, VkCommandBuffer cb){
    if (!cb) return;
    if (dev->nfree_cbs == dev->cap_free_cbs){
        int cap = dev->cap_free_cbs ? 2 * dev->cap_free_cbs : 16;
        VkCommandBuffer *v = realloc(dev->free_cbs, (size_t)cap * sizeof(*v));
        if (!v){ vk.vkFreeCommandBuffers(dev->vk, dev->pool, 1, &cb); return; }
        dev->free_cbs = v; dev->cap_free_cbs = cap;
    }
    dev->free_cbs[dev->nfree_cbs++] = cb;
}

void vk_barrier(VkCommandBuffer cb){
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}

void vk_barrier_to_host(VkCommandBuffer cb){
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}

// ---- buffers ----
int vk_alloc_buffer(WGPUDevice dev, VkDeviceSize size, VkFlags usage, VkFlags mem_want, VkBuffer *buf,
                    VkDeviceMemory *mem, void **mapped){
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size ? size : 4;
    bi.usage = usage;
    if (vk.vkCreateBuffer(dev->vk, &bi, NULL, buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    vk.vkGetBufferMemoryRequirements(dev->vk, *buf, &req);
    int type = vk_memory_type(dev->ad, req.memoryTypeBits, mem_want, mem_want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? 0 : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (type < 0 && (mem_want & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
        type = vk_memory_type(dev->ad, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (type < 0){ vk.vkDestroyBuffer(dev->vk, *buf, NULL); return 0; }
    VkMemoryAllocateFlagsInfo fi;
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ? &fi : NULL;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    if (vk.vkAllocateMemory(dev->vk, &ai, NULL, mem) != VK_SUCCESS){ vk.vkDestroyBuffer(dev->vk, *buf, NULL); return 0; }
    vk.vkBindBufferMemory(dev->vk, *buf, *mem, 0);
    if (mapped){
        *mapped = NULL;
        if (vk.vkMapMemory(dev->vk, *mem, 0, VK_WHOLE_SIZE, 0, mapped) != VK_SUCCESS) *mapped = NULL;
    }
    return 1;
}

static void buffer_free(Obj *o){
    WGPUBuffer b = (WGPUBuffer)o;
    if (b->mapped) vk.vkUnmapMemory(b->dev->vk, b->mem);
    if (b->vk) vk.vkDestroyBuffer(b->dev->vk, b->vk, NULL);
    if (b->mem) vk.vkFreeMemory(b->dev->vk, b->mem, NULL);
    free(b);
}

WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, WGPUBufferDescriptor const *d){
    WGPUBuffer b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    obj_init(&b->o, buffer_free);
    b->dev = device;
    b->size = d->size;
    b->usage = d->usage;
    VkFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (d->usage & WGPUBufferUsage_Storage) usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (d->usage & WGPUBufferUsage_Uniform) usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (d->usage & WGPUBufferUsage_Indirect) usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (d->usage & WGPUBufferUsage_Vertex) usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (d->usage & WGPUBufferUsage_Index) usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    int host = (d->usage & (WGPUBufferUsage_MapRead | WGPUBufferUsage_MapWrite)) != 0;
    VkFlags mem = host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                          ((d->usage & WGPUBufferUsage_MapRead) ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0))
                       : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    void *mapped = NULL;
    // (buffer sizes rounded up to 4: vkCmdFillBuffer and the copies work in words)
    if (!vk_alloc_buffer(device, (d->size + 3) & ~3ull, usage, mem, &b->vk, &b->mem, host ? &mapped : NULL)){
        vk_error(device, WGPUErrorType_OutOfMemory, "buffer of %llu bytes: no memory", (unsigned long long)d->size);
        free(b);
        return NULL;
    }
    b->mapped = mapped;
    if (d->mappedAtCreation && b->mapped) b->map_ok = 1;
    return b;
}

void wgpuBufferDestroy(WGPUBuffer buffer){ (void)buffer; }   // freed with its last reference
void wgpuBufferRelease(WGPUBuffer buffer){ if (buffer) obj_unref(&buffer->o); }

// ---- uploads ----
static void staging_free(Obj *o){
    Staging *s = (Staging *)o;
    s->used = 0;
    s->o.refs = 1;                    // back on the device's free list, ready again
    s->next = s->dev->stage_free;
    s->dev->stage_free = s;
}

static Staging *staging_get(WGPUDevice dev, VkDeviceSize need){
    Staging **p = &dev->stage_free;
    for (; *p; p = &(*p)->next)
        if ((*p)->size >= need){ Staging *s = *p; *p = s->next; s->next = NULL; return s; }
    Staging *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    obj_init(&s->o, staging_free);
    s->dev = dev;
    s->size = need > (4u << 20) ? need : (4u << 20);
    void *ptr = NULL;
    if (!vk_alloc_buffer(dev, s->size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &s->buf, &s->mem, &ptr) || !ptr){
        free(s);
        return NULL;
    }
    s->ptr = ptr;
    return s;
}

// room for n bytes in this submission's staging memory: its buffer and offset
static int staging_alloc(WGPUDevice dev, VkDeviceSize n, VkBuffer *buf, VkDeviceSize *off, unsigned char **ptr){
    Staging *s = dev->stage_cur;
    VkDeviceSize at = s ? (s->used + 15) & ~15ull : 0;
    if (!s || at + n > s->size){
        if (s){ refs_add(&dev->up_refs, &s->o); obj_unref(&s->o); }
        s = dev->stage_cur = staging_get(dev, n);
        if (!s) return 0;
        at = 0;
    }
    s->used = at + n;
    *buf = s->buf; *off = at; *ptr = s->ptr + at;
    return 1;
}

static VkCommandBuffer upload_cb(WGPUDevice dev){
    if (!dev->up_open){
        dev->up = vk_cb_get(dev);
        dev->up_open = dev->up != NULL;
        dev->up_dirty = 0;
    }
    if (dev->up_dirty) vk_barrier(dev->up);
    dev->up_dirty = 1;
    return dev->up;
}

void wgpuQueueWriteBuffer(WGPUQueue queue, WGPUBuffer buffer, uint64_t off, void const *data, size_t size){
    WGPUDevice dev = queue->dev;
    if (!buffer || !size) return;
    VkCommandBuffer cb = upload_cb(dev);
    if (!cb) return;
    refs_add(&dev->up_refs, &buffer->o);
    if (size <= 65536 && !(size & 3) && !(off & 3)){
        vk.vkCmdUpdateBuffer(cb, buffer->vk, off, size, data);      // (in the command buffer itself)
        return;
    }
    VkBuffer sb; VkDeviceSize so; unsigned char *p;
    if (!staging_alloc(dev, size, &sb, &so, &p)){ vk_error(dev, WGPUErrorType_OutOfMemory, "no staging memory"); return; }
    memcpy(p, data, size);
    VkBufferCopy c = {so, off, size};
    vk.vkCmdCopyBuffer(cb, sb, buffer->vk, 1, &c);
}

// ---- submissions ----
static int sub_push(WGPUDevice dev, Sub *s){
    if (dev->nsubs == dev->cap_subs){
        int cap = dev->cap_subs ? 2 * dev->cap_subs : 16;
        Sub *v = realloc(dev->subs, (size_t)cap * sizeof(*v));
        if (!v) return 0;
        dev->subs = v; dev->cap_subs = cap;
    }
    dev->subs[dev->nsubs++] = *s;
    return 1;
}

static uint64_t submit(WGPUDevice dev, size_t n, WGPUCommandBuffer const *cbs){
    Sub s;
    memset(&s, 0, sizeof(s));
    if (dev->up_open){
        if (dev->stage_cur){ refs_add(&dev->up_refs, &dev->stage_cur->o); obj_unref(&dev->stage_cur->o); dev->stage_cur = NULL; }
        vk_barrier(dev->up);
        vk.vkEndCommandBuffer(dev->up);
        s.cbs[s.ncb++] = dev->up;
        s.refs = dev->up_refs;
        memset(&dev->up_refs, 0, sizeof(dev->up_refs));
        dev->up_open = 0;
    }
    for (size_t i = 0; i < n && s.ncb < 16; i++){
        WGPUCommandBuffer c = cbs[i];
        if (!c || c->submitted || !c->cb) continue;
        s.cbs[s.ncb++] = c->cb;
        for (int k = 0; k < c->refs.n; k++) refs_add(&s.refs, c->refs.v[k]);
        refs_release(&c->refs);
        c->cb = NULL;
        c->submitted = 1;
    }
    uint64_t value = ++dev->submitted;
    VkTimelineSemaphoreSubmitInfo ts;
    memset(&ts, 0, sizeof(ts));
    ts.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues = &value;
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &ts;
    si.commandBufferCount = (uint32_t)s.ncb;
    si.pCommandBuffers = s.cbs;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &dev->tl;
    VkResult r = vk.vkQueueSubmit(dev->q, 1, &si, 0);
    if (r != VK_SUCCESS){ vk_error(dev, WGPUErrorType_Internal, "vkQueueSubmit failed (%d)", r); dev->lost = 1; }
    s.value = value;
    sub_push(dev, &s);
    return value;
}

void wgpuQueueSubmit(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const *commands){
    submit(queue->dev, commandCount, commands);
}

WGPUSubmissionIndex wgpuQueueSubmitForIndex(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const *commands){
    return (WGPUSubmissionIndex)submit(queue->dev, commandCount, commands);
}

static void pend_push(WGPUDevice dev, Pending *p){
    if (dev->npend == dev->cap_pend){
        int cap = dev->cap_pend ? 2 * dev->cap_pend : 16;
        Pending *v = realloc(dev->pend, (size_t)cap * sizeof(*v));
        if (!v) return;
        dev->pend = v; dev->cap_pend = cap;
    }
    dev->pend[dev->npend++] = *p;
}

// the work up to now (uploads included: submitted if waiting)
static uint64_t current(WGPUDevice dev){
    if (dev->up_open) submit(dev, 0, NULL);
    return dev->submitted;
}

WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo){
    Pending p;
    memset(&p, 0, sizeof(p));
    p.value = current(queue->dev);
    p.kind = 0;
    p.wd = callbackInfo;
    pend_push(queue->dev, &p);
    WGPUFuture f = {3};
    return f;
}

WGPUFuture wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo){
    (void)mode; (void)offset; (void)size;
    WGPUDevice dev = buffer->dev;
    Pending p;
    memset(&p, 0, sizeof(p));
    p.value = current(dev);
    p.kind = 1;
    p.map = callbackInfo;
    p.buf = buffer;
    obj_ref(&buffer->o);
    buffer->map_pending = 1;
    pend_push(dev, &p);
    WGPUFuture f = {4};
    return f;
}

void const *wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size){
    (void)size;
    if (!buffer->mapped || !buffer->map_ok) return NULL;
    return buffer->mapped + offset;
}

void wgpuBufferUnmap(WGPUBuffer buffer){ buffer->map_ok = 0; }

// what is done: command buffers back, references given back, callbacks called
static WGPUBool poll(WGPUDevice dev, int wait, uint64_t until){
    if (!dev || !dev->vk) return 1;
    if (wait && until){
        VkSemaphoreWaitInfo wi;
        memset(&wi, 0, sizeof(wi));
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &dev->tl;
        wi.pValues = &until;
        vk.vkWaitSemaphores(dev->vk, &wi, ~0ull);
    }
    uint64_t done = 0;
    vk.vkGetSemaphoreCounterValue(dev->vk, dev->tl, &done);
    int k = 0;
    for (; k < dev->nsubs && dev->subs[k].value <= done; k++){
        Sub *s = &dev->subs[k];
        for (int i = 0; i < s->ncb; i++) vk_cb_put(dev, s->cbs[i]);
        refs_release(&s->refs);
    }
    if (k){ memmove(dev->subs, dev->subs + k, (size_t)(dev->nsubs - k) * sizeof(Sub)); dev->nsubs -= k; }
    // the callbacks (taken off the list first: one may ask for another)
    for (int i = 0; i < dev->npend;){
        Pending p = dev->pend[i];
        if (p.value > done){ i++; continue; }
        memmove(dev->pend + i, dev->pend + i + 1, (size_t)(dev->npend - i - 1) * sizeof(Pending));
        dev->npend--;
        if (p.kind == 0){
            if (p.wd.callback) p.wd.callback(WGPUQueueWorkDoneStatus_Success, sv(""), p.wd.userdata1, p.wd.userdata2);
        } else {
            p.buf->map_pending = 0;
            p.buf->map_ok = p.buf->mapped != NULL;
            if (p.map.callback) p.map.callback(p.buf->map_ok ? WGPUMapAsyncStatus_Success : WGPUMapAsyncStatus_Error,
                                               sv(p.buf->map_ok ? "" : "not a mappable buffer"), p.map.userdata1, p.map.userdata2);
            obj_unref(&p.buf->o);
        }
    }
    return dev->nsubs == 0;
}

WGPUBool wgpuDevicePoll(WGPUDevice device, WGPUBool wait, WGPUSubmissionIndex const *submissionIndex){
    if (!device) return 1;
    uint64_t until = submissionIndex ? (uint64_t)*submissionIndex : (wait ? current(device) : 0);
    return poll(device, wait, until);
}

void wgpuInstanceProcessEvents(WGPUInstance instance){
    if (instance && instance->device) poll(instance->device, 0, 0);
}
