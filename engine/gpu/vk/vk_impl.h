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

// ---- the Vulkan functions, looked up at run time (vk_core.c) ----
#define VK_GLOBAL_FNS(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceLayerProperties) X(vkEnumerateInstanceExtensionProperties)
#define VK_INSTANCE_FNS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)
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
    X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkCmdCopyQueryPoolResults)

typedef struct {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#define X(f) PFN_##f f;
    VK_GLOBAL_FNS(X) VK_INSTANCE_FNS(X) VK_DEVICE_FNS(X)
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
    WGPUTextureUsage usage;
    uint32_t w, h;
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
    int dirty;                       // a command recorded since the last barrier
    int finished;
};

struct WGPUCommandBufferImpl {
    Obj o;
    WGPUDevice dev;
    VkCommandBuffer cb;
    Refs refs;
    int submitted;
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

// ---- between the files ----
void vk_error(WGPUDevice dev, WGPUErrorType type, const char *fmt, ...);
VkCommandBuffer vk_cb_get(WGPUDevice dev);                 // a command buffer, begun
void vk_cb_put(WGPUDevice dev, VkCommandBuffer cb);         // back to the free list (not in flight)
void vk_barrier(VkCommandBuffer cb);                        // everything before, before everything after
void vk_barrier_to_host(VkCommandBuffer cb);
int vk_memory_type(WGPUAdapter ad, uint32_t bits, VkFlags want, VkFlags avoid);
int vk_alloc_buffer(WGPUDevice dev, VkDeviceSize size, VkFlags usage, VkFlags mem_want, VkBuffer *buf,
                    VkDeviceMemory *mem, void **mapped);
// WGSL -> SPIR-V (engine/gpu/wgsl): 0 and the words (free them) or 1 and a message
int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen);

#endif
