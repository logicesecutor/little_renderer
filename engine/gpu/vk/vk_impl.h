// vk_impl.h - the Vulkan layer's insides: the part of webgpu.h the engine uses, implemented
// on Vulkan (engine/gpu/vk/*.c), in place of wgpu-native (`make VK=1`).
//
// How WebGPU's model maps here:
// - one queue (graphics + compute); each submit signals a timeline semaphore with the next
//   number: wgpuQueueSubmitForIndex returns it, wgpuDevicePoll waits for it;
// - objects are reference counted; a command buffer holds a reference to everything it uses
//   until its submission is done, so releasing early (as WebGPU allows) is safe;
// - every command is preceded by a full memory barrier (what WebGPU's usage tracking ends up
//   doing for our dispatches, which nearly all depend on the one before): simple and safe,
//   and room to be smarter later;
// - wgpuQueueWriteBuffer goes into an upload command buffer submitted before the next submit;
// - images stay in the GENERAL layout (textureLoad only, no samplers).
#ifndef LR_VK_IMPL_H
#define LR_VK_IMPL_H

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>
#include "gpu/vk/vk.h"
#include "gpu/vk/webgpu_rt.h"

// ---- the Vulkan functions, looked up at run time (vk_core.c) ----
#define VK_GLOBAL_FNS(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceLayerProperties) X(vkEnumerateInstanceExtensionProperties)
#define VK_INSTANCE_FNS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) \
    X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceProperties2)
#define VK_DEVICE_FNS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkQueueWaitIdle) X(vkQueueSubmit) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkFreeDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdDispatch) X(vkCmdDispatchIndirect) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) \
    X(vkCmdPipelineBarrier) X(vkCmdPushConstants) X(vkCreateSemaphore) X(vkDestroySemaphore) \
    X(vkGetSemaphoreCounterValue) X(vkWaitSemaphores) X(vkCreateQueryPool) X(vkDestroyQueryPool) \
    X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkCmdCopyQueryPoolResults) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) \
    X(vkCreateGraphicsPipelines) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdSetViewport) \
    X(vkCmdSetScissor) X(vkCmdDraw) X(vkGetBufferDeviceAddress) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)

// (these may be missing: a machine without a display, a device without a swapchain)
#define VK_SURFACE_FNS(X) \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR) X(vkCreateWin32SurfaceKHR)
#define VK_RT_FNS(X) \
    X(vkCreateAccelerationStructureKHR) X(vkDestroyAccelerationStructureKHR) X(vkGetAccelerationStructureBuildSizesKHR) \
    X(vkCmdBuildAccelerationStructuresKHR) X(vkGetAccelerationStructureDeviceAddressKHR)
#define VK_SWAPCHAIN_FNS(X) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) \
    X(vkQueuePresentKHR)

typedef struct {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#define X(f) PFN_##f f;
    VK_GLOBAL_FNS(X) VK_INSTANCE_FNS(X) VK_DEVICE_FNS(X) VK_SURFACE_FNS(X) VK_SWAPCHAIN_FNS(X) VK_RT_FNS(X)
#undef X
} VkFns;
extern VkFns vk;

// ---- reference counted objects ----
typedef struct Obj { int refs; void (*free)(struct Obj *); } Obj;
void obj_init(Obj *o, void (*free)(Obj *));
void obj_ref(Obj *o);
void obj_unref(Obj *o);
typedef struct { Obj **v; int n, cap; } Refs;
void refs_add(Refs *r, Obj *o);      // takes a reference
void refs_release(Refs *r);          // gives them all back

// staging memory for uploads: chunks of host-visible memory, used once per submission
typedef struct Staging {
    Obj o;
    WGPUDevice dev;
    VkBuffer buf; VkDeviceMemory mem;
    unsigned char *ptr;
    VkDeviceSize size, used;
    struct Staging *next;            // the device's free list
} Staging;

typedef struct { uint64_t value; VkCommandBuffer cbs[16]; int ncb; Refs refs; } Sub;
typedef struct {
    uint64_t value;
    int kind;                        // 0 work done, 1 buffer mapped
    WGPUQueueWorkDoneCallbackInfo wd;
    WGPUBufferMapCallbackInfo map;
    WGPUBuffer buf;
} Pending;

struct WGPUInstanceImpl {
    Obj o;
    void *lib;
    VkInstance vk;
    int validation;
    WGPUDevice device;               // the one device (wgpuInstanceProcessEvents polls it)
};

struct WGPUAdapterImpl {
    Obj o;
    WGPUInstance inst;
    VkPhysicalDevice pd;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    uint32_t family;                 // a queue family with graphics and compute
    uint32_t ts_bits;                // its timestamp bits (0: none)
    int rt;                          // acceleration structures and ray queries (webgpu_rt.h)
    uint32_t as_scratch_align;
};

struct WGPUQueueImpl { Obj o; WGPUDevice dev; };

struct WGPUDeviceImpl {
    Obj o;
    WGPUAdapter ad;
    VkDevice vk;
    VkQueue q;
    struct WGPUQueueImpl queue;
    VkCommandPool pool;
    VkCommandBuffer *free_cbs; int nfree_cbs, cap_free_cbs;
    VkDescriptorPool *dpools; int ndpools;
    VkSemaphore tl;                  // the timeline: submissions done
    uint64_t submitted;
    Sub *subs; int nsubs, cap_subs;  // submitted, not yet known done
    Pending *pend; int npend, cap_pend;
    // uploads (wgpuQueueWriteBuffer) since the last submit
    VkCommandBuffer up; int up_open, up_dirty;
    Refs up_refs;
    Staging *stage_cur, *stage_free;
    WGPUUncapturedErrorCallbackInfo on_error;
    int timestamps;
    int rt;                          // ray queries asked for (WGPUNativeFeature_RayQuery) and on
    int lost;
};

struct WGPUBufferImpl {
    Obj o;
    WGPUDevice dev;
    VkBuffer vk;
    VkDeviceMemory mem;
    uint64_t size;
    WGPUBufferUsage usage;
    unsigned char *mapped;           // host-visible buffers: mapped for good
    int map_pending, map_ok;
};

struct WGPUTextureImpl {
    Obj o;
    WGPUDevice dev;
    VkImage img;
    VkDeviceMemory mem;              // 0: not ours (a swapchain image)
    WGPUTextureFormat format;
    VkFormat vkformat;
    uint32_t texel;                  // bytes per texel
    int swapchain;                   // a window's image: PRESENT layout between frames
    WGPUTextureUsage usage;
    uint32_t w, h, layers, mips;
    VkFlags aspect;
    int ready;                       // in its layout (GENERAL) already
};

struct WGPUTextureViewImpl {
    Obj o;
    WGPUTexture tex;
    VkImageView vk;
};

struct WGPUShaderModuleImpl {
    Obj o;
    WGPUDevice dev;
    VkShaderModule vk;
};

typedef struct { uint32_t binding; VkDescriptorType type; int dynamic; } BglEntry;
struct WGPUBindGroupLayoutImpl {
    Obj o;
    WGPUDevice dev;
    VkDescriptorSetLayout vk;
    BglEntry *e; int n;
    int ndynamic;
};

struct WGPUPipelineLayoutImpl {
    Obj o;
    WGPUDevice dev;
    VkPipelineLayout vk;
    WGPUBindGroupLayout bgl[4]; int n;
};

struct WGPUComputePipelineImpl {
    Obj o;
    WGPUDevice dev;
    VkPipeline vk;
    WGPUPipelineLayout layout;
};

struct WGPUBindGroupImpl {
    Obj o;
    WGPUDevice dev;
    VkDescriptorSet set;
    VkDescriptorPool pool;
    WGPUBindGroupLayout layout;
    Refs held;                       // the buffers and views it binds
};

struct WGPUQuerySetImpl {
    Obj o;
    WGPUDevice dev;
    VkQueryPool vk;
    uint32_t count;
};

struct WGPUCommandEncoderImpl {
    Obj o;
    WGPUDevice dev;
    VkCommandBuffer cb;
    Refs refs;
    VkFlags pend_stage, pend_write;  // what the commands since the last barrier did (vk_before)
    int finished;
};

struct WGPUCommandBufferImpl {
    Obj o;
    WGPUDevice dev;
    VkCommandBuffer cb;
    Refs refs;
    int submitted;
};

struct WGPURenderPipelineImpl {
    Obj o;
    WGPUDevice dev;
    VkPipeline vk;
    WGPUPipelineLayout layout;
};

struct WGPURenderPassEncoderImpl {
    Obj o;
    WGPUCommandEncoder enc;
    WGPURenderPipeline pipe; int pipe_bound;
    WGPUBindGroup bg[4];
    uint32_t dyn[4][8]; uint32_t ndyn[4];
    int bg_dirty;
};

struct WGPUComputePassEncoderImpl {
    Obj o;
    WGPUCommandEncoder enc;
    WGPUComputePipeline pipe; int pipe_bound;
    WGPUBindGroup bg[4];
    uint32_t dyn[4][8]; uint32_t ndyn[4];
    int bg_dirty;
    WGPUQuerySet ts; uint32_t ts_end;
};

// acceleration structures (vk_rt.c, webgpu_rt.h): the structure, its memory, its scratch
typedef struct {
    VkAccelerationStructureKHR as;
    VkBuffer buf; VkDeviceMemory mem;
    VkBuffer scratch; VkDeviceMemory smem; VkDeviceAddress scratch_at;
    VkDeviceAddress addr;
} VkAs;
struct LRBlasImpl {
    Obj o;
    WGPUDevice dev;
    VkAs a;
    VkFlags flags;
    uint32_t ngeo;
    LRGeometry *geo;                 // (holding a reference to each buffer)
};
struct LRTlasImpl {
    Obj o;
    WGPUDevice dev;
    VkAs a;
    uint32_t max;
};

#define VK_SURFACE_IMAGES 8
struct WGPUSurfaceImpl {
    Obj o;
    WGPUInstance inst;
    VkSurfaceKHR vk;
    void *hinstance, *hwnd;
    WGPUDevice dev;                  // configured for it (0: not configured)
    WGPUSurfaceConfiguration cfg;
    VkSwapchainKHR sc;
    int stale;                       // out of date: made again before the next image
    uint32_t n;
    WGPUTexture tex[VK_SURFACE_IMAGES];
    VkSemaphore presentable[VK_SURFACE_IMAGES];
    VkFence acquired;                // an image free to draw on (waited for on the CPU)
    int cur;                         // the image handed out (-1: none)
};

// ---- between the files ----
void vk_error(WGPUDevice dev, WGPUErrorType type, const char *fmt, ...);
VkCommandBuffer vk_cb_get(WGPUDevice dev);                 // a command buffer, begun
void vk_cb_put(WGPUDevice dev, VkCommandBuffer cb);         // back to the free list (not in flight)
void vk_barrier(VkCommandBuffer cb);                        // everything before, before everything after
void vk_barrier_to_host(VkCommandBuffer cb);
int vk_memory_type(WGPUAdapter ad, uint32_t bits, VkFlags want, VkFlags avoid);
int vk_alloc_buffer(WGPUDevice dev, VkDeviceSize size, VkFlags usage, VkFlags mem_want, VkBuffer *buf,
                    VkDeviceMemory *mem, void **mapped);
VkFormat vk_format(WGPUTextureFormat f, uint32_t *texel, VkFlags *aspect);
WGPUTextureFormat vk_wgpu_format(VkFormat f);
void vk_image_layout(VkCommandBuffer cb, VkImage img, VkFlags aspect, uint32_t from, uint32_t to);
void vk_texture_ready(WGPUDevice dev, VkCommandBuffer cb, WGPUTexture t);   // into GENERAL, once
VkCommandBuffer vk_upload_cb(WGPUDevice dev);                 // the upload command buffer (a barrier first)
int vk_staging(WGPUDevice dev, VkDeviceSize n, VkBuffer *buf, VkDeviceSize *off, unsigned char **ptr);
// the kinds of command, for the barriers between them (vk_before)
enum { VK_CMD_DISPATCH, VK_CMD_INDIRECT, VK_CMD_TRANSFER, VK_CMD_BUILD, VK_CMD_RENDER };
void vk_before(WGPUCommandEncoder e, int kind);              // a command of this kind into e: a barrier first
// a submission of cb (ended; given back when done) after the uploads, signalling `signal` too
uint64_t vk_submit_cb(WGPUDevice dev, VkCommandBuffer cb, VkSemaphore signal);
VkDeviceAddress vk_buffer_address(WGPUBuffer b);
// WGSL -> SPIR-V (engine/gpu/wgsl): 0 and the words (free them) or 1 and a message
int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen);

#endif
