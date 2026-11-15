// vk.h - the part of the Vulkan API the engine's Vulkan layer uses, declared here (no Vulkan
// SDK or headers needed to build: the loader, vulkan-1.dll / libvulkan.so.1, is opened at run
// time and every function is looked up by name).
//
// The layouts follow the Vulkan specification (1.3, plus the ray tracing extensions);
// tools/vk_check.c compares each struct's size and field offsets with the official headers
// (a development check, built only where the headers are installed).
#ifndef LR_VK_H
#define LR_VK_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define VKAPI_PTR __stdcall
#else
#define VKAPI_PTR
#endif

typedef uint32_t VkBool32;
typedef uint32_t VkFlags;
typedef uint64_t VkFlags64;
typedef uint64_t VkDeviceSize;
typedef uint64_t VkDeviceAddress;
typedef int32_t VkResult;

#define VK_TRUE 1u
#define VK_FALSE 0u
#define VK_WHOLE_SIZE (~0ull)
#define VK_QUEUE_FAMILY_IGNORED (~0u)
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256
#define VK_UUID_SIZE 16
#define VK_MAX_MEMORY_TYPES 32
#define VK_MAX_MEMORY_HEAPS 16
#define VK_MAX_EXTENSION_NAME_SIZE 256
#define VK_MAX_DESCRIPTION_SIZE 256
#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
    ((((uint32_t)(variant)) << 29) | (((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))
#define VK_API_VERSION_1_3 VK_MAKE_API_VERSION(0, 1, 3, 0)

// results
#define VK_SUCCESS 0
#define VK_NOT_READY 1
#define VK_TIMEOUT 2
#define VK_INCOMPLETE 5
#define VK_SUBOPTIMAL_KHR 1000001003
#define VK_ERROR_OUT_OF_DATE_KHR (-1000001004)

// handles: dispatchable ones are pointers, the others 64-bit values
#define VK_HANDLE(name) typedef struct name##_T *name
#define VK_HANDLE64(name) typedef uint64_t name
VK_HANDLE(VkInstance);
VK_HANDLE(VkPhysicalDevice);
VK_HANDLE(VkDevice);
VK_HANDLE(VkQueue);
VK_HANDLE(VkCommandBuffer);
VK_HANDLE64(VkBuffer);
VK_HANDLE64(VkImage);
VK_HANDLE64(VkImageView);
VK_HANDLE64(VkDeviceMemory);
VK_HANDLE64(VkShaderModule);
VK_HANDLE64(VkPipeline);
VK_HANDLE64(VkPipelineLayout);
VK_HANDLE64(VkPipelineCache);
VK_HANDLE64(VkDescriptorSetLayout);
VK_HANDLE64(VkDescriptorPool);
VK_HANDLE64(VkDescriptorSet);
VK_HANDLE64(VkCommandPool);
VK_HANDLE64(VkFence);
VK_HANDLE64(VkSemaphore);
VK_HANDLE64(VkQueryPool);
VK_HANDLE64(VkSampler);
VK_HANDLE64(VkRenderPass);
VK_HANDLE64(VkSurfaceKHR);
VK_HANDLE64(VkSwapchainKHR);
VK_HANDLE64(VkAccelerationStructureKHR);
#define VK_NULL_HANDLE 0

// structure types (VkStructureType)
typedef enum {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9,
    VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14,
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 19,
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 20,
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 22,
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 23,
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 24,
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO = 25,
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 26,
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 27,
    VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44,
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 45,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO = 1000060000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES = 49,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53,
    VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO = 1000207002,
    VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO = 1000207003,
    VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO = 1000207004,
    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO = 1000244001,
    VK_STRUCTURE_TYPE_RENDERING_INFO = 1000044000,
    VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO = 1000044001,
    VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO = 1000044002,
    VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = 1000001000,
    VK_STRUCTURE_TYPE_PRESENT_INFO_KHR = 1000001001,
    VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR = 1000004000,
    VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR = 1000009000,
    VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT = 1000128004,
    VK_STRUCTURE_TYPE_MAX_ENUM = 0x7fffffff
} VkStructureType;

typedef struct VkExtent2D { uint32_t width, height; } VkExtent2D;
typedef struct VkExtent3D { uint32_t width, height, depth; } VkExtent3D;
typedef struct VkOffset2D { int32_t x, y; } VkOffset2D;
typedef struct VkOffset3D { int32_t x, y, z; } VkOffset3D;
typedef struct VkRect2D { VkOffset2D offset; VkExtent2D extent; } VkRect2D;

// ---- instance and devices ----
typedef struct VkApplicationInfo {
    VkStructureType sType; const void *pNext;
    const char *pApplicationName; uint32_t applicationVersion;
    const char *pEngineName; uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkLayerProperties {
    char layerName[VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t specVersion, implementationVersion;
    char description[VK_MAX_DESCRIPTION_SIZE];
} VkLayerProperties;

typedef struct VkExtensionProperties {
    char extensionName[VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t specVersion;
} VkExtensionProperties;

typedef struct VkPhysicalDeviceLimits {
    uint32_t maxImageDimension1D, maxImageDimension2D, maxImageDimension3D, maxImageDimensionCube;
    uint32_t maxImageArrayLayers, maxTexelBufferElements, maxUniformBufferRange, maxStorageBufferRange;
    uint32_t maxPushConstantsSize, maxMemoryAllocationCount, maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity, sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets, maxPerStageDescriptorSamplers, maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers, maxPerStageDescriptorSampledImages;
    uint32_t maxPerStageDescriptorStorageImages, maxPerStageDescriptorInputAttachments, maxPerStageResources;
    uint32_t maxDescriptorSetSamplers, maxDescriptorSetUniformBuffers, maxDescriptorSetUniformBuffersDynamic;
    uint32_t maxDescriptorSetStorageBuffers, maxDescriptorSetStorageBuffersDynamic, maxDescriptorSetSampledImages;
    uint32_t maxDescriptorSetStorageImages, maxDescriptorSetInputAttachments;
    uint32_t maxVertexInputAttributes, maxVertexInputBindings, maxVertexInputAttributeOffset;
    uint32_t maxVertexInputBindingStride, maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel, maxTessellationPatchSize;
    uint32_t maxTessellationControlPerVertexInputComponents, maxTessellationControlPerVertexOutputComponents;
    uint32_t maxTessellationControlPerPatchOutputComponents, maxTessellationControlTotalOutputComponents;
    uint32_t maxTessellationEvaluationInputComponents, maxTessellationEvaluationOutputComponents;
    uint32_t maxGeometryShaderInvocations, maxGeometryInputComponents, maxGeometryOutputComponents;
    uint32_t maxGeometryOutputVertices, maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents, maxFragmentOutputAttachments, maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources, maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3], maxComputeWorkGroupInvocations, maxComputeWorkGroupSize[3];
    uint32_t subPixelPrecisionBits, subTexelPrecisionBits, mipmapPrecisionBits;
    uint32_t maxDrawIndexedIndexValue, maxDrawIndirectCount;
    float maxSamplerLodBias, maxSamplerAnisotropy;
    uint32_t maxViewports, maxViewportDimensions[2];
    float viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment, minUniformBufferOffsetAlignment, minStorageBufferOffsetAlignment;
    int32_t minTexelOffset; uint32_t maxTexelOffset;
    int32_t minTexelGatherOffset; uint32_t maxTexelGatherOffset;
    float minInterpolationOffset, maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits, maxFramebufferWidth, maxFramebufferHeight, maxFramebufferLayers;
    VkFlags framebufferColorSampleCounts, framebufferDepthSampleCounts, framebufferStencilSampleCounts;
    VkFlags framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkFlags sampledImageColorSampleCounts, sampledImageIntegerSampleCounts, sampledImageDepthSampleCounts;
    VkFlags sampledImageStencilSampleCounts, storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    uint32_t maxClipDistances, maxCullDistances, maxCombinedClipAndCullDistances, discreteQueuePriorities;
    float pointSizeRange[2], lineWidthRange[2], pointSizeGranularity, lineWidthGranularity;
    VkBool32 strictLines, standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment, optimalBufferCopyRowPitchAlignment, nonCoherentAtomSize;
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape, residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape, residencyAlignedMipSize, residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;

typedef enum {
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
    VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM = 0x7fffffff
} VkPhysicalDeviceType;

typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion, driverVersion, vendorID, deviceID;
    VkPhysicalDeviceType deviceType;
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t pipelineCacheUUID[VK_UUID_SIZE];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

typedef struct VkPhysicalDeviceProperties2 {
    VkStructureType sType; void *pNext;
    VkPhysicalDeviceProperties properties;
} VkPhysicalDeviceProperties2;

// VkPhysicalDeviceFeatures: 55 VkBool32s in the specification's order; only some are named
typedef struct VkPhysicalDeviceFeatures {
    VkBool32 robustBufferAccess, fullDrawIndexUint32, imageCubeArray, independentBlend, geometryShader;
    VkBool32 tessellationShader, sampleRateShading, dualSrcBlend, logicOp, multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance, depthClamp, depthBiasClamp, fillModeNonSolid, depthBounds;
    VkBool32 wideLines, largePoints, alphaToOne, multiViewport, samplerAnisotropy;
    VkBool32 textureCompressionETC2, textureCompressionASTC_LDR, textureCompressionBC, occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery, vertexPipelineStoresAndAtomics, fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize, shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats, shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat, shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing, shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing, shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance, shaderCullDistance, shaderFloat64, shaderInt64, shaderInt16;
    VkBool32 shaderResourceResidency, shaderResourceMinLod, sparseBinding, sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D, sparseResidencyImage3D, sparseResidency2Samples;
    VkBool32 sparseResidency4Samples, sparseResidency8Samples, sparseResidency16Samples;
    VkBool32 sparseResidencyAliased, variableMultisampleRate, inheritedQueries;
} VkPhysicalDeviceFeatures;

typedef struct VkPhysicalDeviceFeatures2 {
    VkStructureType sType; void *pNext;
    VkPhysicalDeviceFeatures features;
} VkPhysicalDeviceFeatures2;

typedef struct VkPhysicalDeviceVulkan12Features {
    VkStructureType sType; void *pNext;
    VkBool32 samplerMirrorClampToEdge, drawIndirectCount, storageBuffer8BitAccess;
    VkBool32 uniformAndStorageBuffer8BitAccess, storagePushConstant8, shaderBufferInt64Atomics;
    VkBool32 shaderSharedInt64Atomics, shaderFloat16, shaderInt8, descriptorIndexing;
    VkBool32 shaderInputAttachmentArrayDynamicIndexing, shaderUniformTexelBufferArrayDynamicIndexing;
    VkBool32 shaderStorageTexelBufferArrayDynamicIndexing, shaderUniformBufferArrayNonUniformIndexing;
    VkBool32 shaderSampledImageArrayNonUniformIndexing, shaderStorageBufferArrayNonUniformIndexing;
    VkBool32 shaderStorageImageArrayNonUniformIndexing, shaderInputAttachmentArrayNonUniformIndexing;
    VkBool32 shaderUniformTexelBufferArrayNonUniformIndexing, shaderStorageTexelBufferArrayNonUniformIndexing;
    VkBool32 descriptorBindingUniformBufferUpdateAfterBind, descriptorBindingSampledImageUpdateAfterBind;
    VkBool32 descriptorBindingStorageImageUpdateAfterBind, descriptorBindingStorageBufferUpdateAfterBind;
    VkBool32 descriptorBindingUniformTexelBufferUpdateAfterBind, descriptorBindingStorageTexelBufferUpdateAfterBind;
    VkBool32 descriptorBindingUpdateUnusedWhilePending, descriptorBindingPartiallyBound;
    VkBool32 descriptorBindingVariableDescriptorCount, runtimeDescriptorArray, samplerFilterMinmax;
    VkBool32 scalarBlockLayout, imagelessFramebuffer, uniformBufferStandardLayout;
    VkBool32 shaderSubgroupExtendedTypes, separateDepthStencilLayouts, hostQueryReset;
    VkBool32 timelineSemaphore, bufferDeviceAddress, bufferDeviceAddressCaptureReplay;
    VkBool32 bufferDeviceAddressMultiDevice, vulkanMemoryModel, vulkanMemoryModelDeviceScope;
    VkBool32 vulkanMemoryModelAvailabilityVisibilityChains, shaderOutputViewportIndex;
    VkBool32 shaderOutputLayer, subgroupBroadcastDynamicId;
} VkPhysicalDeviceVulkan12Features;

typedef struct VkPhysicalDeviceVulkan13Features {
    VkStructureType sType; void *pNext;
    VkBool32 robustImageAccess, inlineUniformBlock, descriptorBindingInlineUniformBlockUpdateAfterBind;
    VkBool32 pipelineCreationCacheControl, privateData, shaderDemoteToHelperInvocation;
    VkBool32 shaderTerminateInvocation, subgroupSizeControl, computeFullSubgroups, synchronization2;
    VkBool32 textureCompressionASTC_HDR, shaderZeroInitializeWorkgroupMemory, dynamicRendering;
    VkBool32 shaderIntegerDotProduct, maintenance4;
} VkPhysicalDeviceVulkan13Features;

typedef enum {
    VK_QUEUE_GRAPHICS_BIT = 1, VK_QUEUE_COMPUTE_BIT = 2, VK_QUEUE_TRANSFER_BIT = 4
} VkQueueFlagBits;

typedef struct VkQueueFamilyProperties {
    VkFlags queueFlags;
    uint32_t queueCount, timestampValidBits;
    VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef enum {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4, VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 8
} VkMemoryPropertyFlagBits;

typedef struct VkMemoryType { VkFlags propertyFlags; uint32_t heapIndex; } VkMemoryType;
typedef struct VkMemoryHeap { VkDeviceSize size; VkFlags flags; } VkMemoryHeap;
typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryProperties;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t queueFamilyIndex, queueCount;
    const float *pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures *pEnabledFeatures;
} VkDeviceCreateInfo;

// ---- memory and buffers ----
typedef struct VkMemoryRequirements { VkDeviceSize size, alignment; uint32_t memoryTypeBits; } VkMemoryRequirements;

typedef struct VkMemoryAllocateInfo {
    VkStructureType sType; const void *pNext;
    VkDeviceSize allocationSize; uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkMemoryAllocateFlagsInfo {
    VkStructureType sType; const void *pNext;
    VkFlags flags; uint32_t deviceMask;
} VkMemoryAllocateFlagsInfo;
#define VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT 2u

typedef enum {
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x1, VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x2,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20,
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT = 0x40, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x80,
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT = 0x100, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x20000,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR = 0x80000,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR = 0x100000
} VkBufferUsageFlagBits;

typedef struct VkBufferCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkDeviceSize size; VkFlags usage;
    uint32_t sharingMode;           // VK_SHARING_MODE_EXCLUSIVE = 0
    uint32_t queueFamilyIndexCount; const uint32_t *pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkBufferDeviceAddressInfo {
    VkStructureType sType; const void *pNext; VkBuffer buffer;
} VkBufferDeviceAddressInfo;

typedef struct VkBufferCopy { VkDeviceSize srcOffset, dstOffset, size; } VkBufferCopy;

// ---- shaders, descriptors, pipelines ----
typedef struct VkShaderModuleCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    size_t codeSize; const uint32_t *pCode;
} VkShaderModuleCreateInfo;

typedef enum {
    VK_SHADER_STAGE_VERTEX_BIT = 0x1, VK_SHADER_STAGE_FRAGMENT_BIT = 0x10, VK_SHADER_STAGE_COMPUTE_BIT = 0x20
} VkShaderStageFlagBits;

typedef enum {
    VK_DESCRIPTOR_TYPE_SAMPLER = 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1,
    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = 9,
    VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000,
    VK_DESCRIPTOR_TYPE_MAX_ENUM = 0x7fffffff
} VkDescriptorType;

typedef struct VkDescriptorSetLayoutBinding {
    uint32_t binding; VkDescriptorType descriptorType; uint32_t descriptorCount;
    VkFlags stageFlags; const VkSampler *pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t bindingCount; const VkDescriptorSetLayoutBinding *pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkPushConstantRange { VkFlags stageFlags; uint32_t offset, size; } VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t setLayoutCount; const VkDescriptorSetLayout *pSetLayouts;
    uint32_t pushConstantRangeCount; const VkPushConstantRange *pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkDescriptorPoolSize { VkDescriptorType type; uint32_t descriptorCount; } VkDescriptorPoolSize;
#define VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT 1u
typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t maxSets, poolSizeCount; const VkDescriptorPoolSize *pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType sType; const void *pNext;
    VkDescriptorPool descriptorPool; uint32_t descriptorSetCount; const VkDescriptorSetLayout *pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo { VkBuffer buffer; VkDeviceSize offset, range; } VkDescriptorBufferInfo;
typedef struct VkDescriptorImageInfo { VkSampler sampler; VkImageView imageView; uint32_t imageLayout; } VkDescriptorImageInfo;

typedef struct VkWriteDescriptorSet {
    VkStructureType sType; const void *pNext;
    VkDescriptorSet dstSet; uint32_t dstBinding, dstArrayElement, descriptorCount;
    VkDescriptorType descriptorType;
    const VkDescriptorImageInfo *pImageInfo;
    const VkDescriptorBufferInfo *pBufferInfo;
    const void *pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkSpecializationInfo VkSpecializationInfo;
typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkFlags stage; VkShaderModule module; const char *pName;
    const VkSpecializationInfo *pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout; VkPipeline basePipelineHandle; int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

// ---- commands and synchronization ----
typedef struct VkCommandPoolCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags; uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;
#define VK_COMMAND_POOL_CREATE_TRANSIENT_BIT 1u
#define VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 2u

typedef struct VkCommandBufferAllocateInfo {
    VkStructureType sType; const void *pNext;
    VkCommandPool commandPool; uint32_t level; uint32_t commandBufferCount;   // level: 0 primary
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType sType; const void *pNext; VkFlags flags; const void *pInheritanceInfo;
} VkCommandBufferBeginInfo;
#define VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 1u

typedef enum {
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x1, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT = 0x2,
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT = 0x4, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT = 0x8,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT = 0x80, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT = 0x100,
    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT = 0x200, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = 0x400,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800, VK_PIPELINE_STAGE_TRANSFER_BIT = 0x1000,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x2000, VK_PIPELINE_STAGE_HOST_BIT = 0x4000,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x10000,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x2000000
} VkPipelineStageFlagBits;

typedef enum {
    VK_ACCESS_INDIRECT_COMMAND_READ_BIT = 0x1, VK_ACCESS_INDEX_READ_BIT = 0x2,
    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT = 0x4, VK_ACCESS_UNIFORM_READ_BIT = 0x8,
    VK_ACCESS_SHADER_READ_BIT = 0x20, VK_ACCESS_SHADER_WRITE_BIT = 0x40,
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT = 0x80, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT = 0x100,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x200, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x400,
    VK_ACCESS_TRANSFER_READ_BIT = 0x800, VK_ACCESS_TRANSFER_WRITE_BIT = 0x1000,
    VK_ACCESS_HOST_READ_BIT = 0x2000, VK_ACCESS_HOST_WRITE_BIT = 0x4000,
    VK_ACCESS_MEMORY_READ_BIT = 0x8000, VK_ACCESS_MEMORY_WRITE_BIT = 0x10000,
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR = 0x200000,
    VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR = 0x400000
} VkAccessFlagBits;

typedef struct VkMemoryBarrier {
    VkStructureType sType; const void *pNext; VkFlags srcAccessMask, dstAccessMask;
} VkMemoryBarrier;

typedef struct VkImageSubresourceRange {
    VkFlags aspectMask; uint32_t baseMipLevel, levelCount, baseArrayLayer, layerCount;
} VkImageSubresourceRange;

typedef struct VkImageMemoryBarrier {
    VkStructureType sType; const void *pNext;
    VkFlags srcAccessMask, dstAccessMask;
    uint32_t oldLayout, newLayout;
    uint32_t srcQueueFamilyIndex, dstQueueFamilyIndex;
    VkImage image; VkImageSubresourceRange subresourceRange;
} VkImageMemoryBarrier;

typedef struct VkSubmitInfo {
    VkStructureType sType; const void *pNext;
    uint32_t waitSemaphoreCount; const VkSemaphore *pWaitSemaphores; const VkFlags *pWaitDstStageMask;
    uint32_t commandBufferCount; const VkCommandBuffer *pCommandBuffers;
    uint32_t signalSemaphoreCount; const VkSemaphore *pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkSemaphoreCreateInfo { VkStructureType sType; const void *pNext; VkFlags flags; } VkSemaphoreCreateInfo;
typedef struct VkSemaphoreTypeCreateInfo {
    VkStructureType sType; const void *pNext; uint32_t semaphoreType; uint64_t initialValue;   // 1 = timeline
} VkSemaphoreTypeCreateInfo;
typedef struct VkTimelineSemaphoreSubmitInfo {
    VkStructureType sType; const void *pNext;
    uint32_t waitSemaphoreValueCount; const uint64_t *pWaitSemaphoreValues;
    uint32_t signalSemaphoreValueCount; const uint64_t *pSignalSemaphoreValues;
} VkTimelineSemaphoreSubmitInfo;
typedef struct VkSemaphoreWaitInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t semaphoreCount; const VkSemaphore *pSemaphores; const uint64_t *pValues;
} VkSemaphoreWaitInfo;

typedef struct VkQueryPoolCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t queryType; uint32_t queryCount; VkFlags pipelineStatistics;   // queryType: 2 = timestamp
} VkQueryPoolCreateInfo;
#define VK_QUERY_TYPE_TIMESTAMP 2u
#define VK_QUERY_RESULT_64_BIT 1u
#define VK_QUERY_RESULT_WAIT_BIT 2u

#define VK_PIPELINE_BIND_POINT_GRAPHICS 0u
#define VK_PIPELINE_BIND_POINT_COMPUTE 1u

// ---- images ----
typedef enum {
    VK_FORMAT_UNDEFINED = 0, VK_FORMAT_R8G8B8A8_UNORM = 37, VK_FORMAT_R8G8B8A8_SRGB = 43,
    VK_FORMAT_B8G8R8A8_UNORM = 44, VK_FORMAT_B8G8R8A8_SRGB = 50, VK_FORMAT_R16G16B16A16_SFLOAT = 97,
    VK_FORMAT_R32_UINT = 98, VK_FORMAT_R32_SFLOAT = 100, VK_FORMAT_R32G32_SFLOAT = 103,
    VK_FORMAT_R32G32B32A32_SFLOAT = 109, VK_FORMAT_D32_SFLOAT = 126, VK_FORMAT_MAX_ENUM = 0x7fffffff
} VkFormat;

typedef enum {
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x1, VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x2, VK_IMAGE_USAGE_SAMPLED_BIT = 0x4,
    VK_IMAGE_USAGE_STORAGE_BIT = 0x8, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x10,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x20
} VkImageUsageFlagBits;

#define VK_IMAGE_ASPECT_COLOR_BIT 1u
#define VK_IMAGE_ASPECT_DEPTH_BIT 2u
#define VK_IMAGE_LAYOUT_UNDEFINED 0u
#define VK_IMAGE_LAYOUT_GENERAL 1u
#define VK_IMAGE_LAYOUT_PRESENT_SRC_KHR 1000001002u

typedef struct VkImageCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t imageType; VkFormat format; VkExtent3D extent;
    uint32_t mipLevels, arrayLayers, samples, tiling;
    VkFlags usage; uint32_t sharingMode;
    uint32_t queueFamilyIndexCount; const uint32_t *pQueueFamilyIndices;
    uint32_t initialLayout;
} VkImageCreateInfo;

typedef struct VkComponentMapping { uint32_t r, g, b, a; } VkComponentMapping;
typedef struct VkImageViewCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkImage image; uint32_t viewType; VkFormat format;
    VkComponentMapping components; VkImageSubresourceRange subresourceRange;
} VkImageViewCreateInfo;

typedef struct VkImageSubresourceLayers { VkFlags aspectMask; uint32_t mipLevel, baseArrayLayer, layerCount; } VkImageSubresourceLayers;
typedef struct VkBufferImageCopy {
    VkDeviceSize bufferOffset; uint32_t bufferRowLength, bufferImageHeight;
    VkImageSubresourceLayers imageSubresource; VkOffset3D imageOffset; VkExtent3D imageExtent;
} VkBufferImageCopy;

// ---- graphics pipelines and dynamic rendering ----
typedef struct VkPipelineVertexInputStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t vertexBindingDescriptionCount; const void *pVertexBindingDescriptions;
    uint32_t vertexAttributeDescriptionCount; const void *pVertexAttributeDescriptions;
} VkPipelineVertexInputStateCreateInfo;
typedef struct VkPipelineInputAssemblyStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t topology; VkBool32 primitiveRestartEnable;
} VkPipelineInputAssemblyStateCreateInfo;
typedef struct VkViewport { float x, y, width, height, minDepth, maxDepth; } VkViewport;
typedef struct VkPipelineViewportStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t viewportCount; const VkViewport *pViewports;
    uint32_t scissorCount; const VkRect2D *pScissors;
} VkPipelineViewportStateCreateInfo;
typedef struct VkPipelineRasterizationStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkBool32 depthClampEnable, rasterizerDiscardEnable;
    uint32_t polygonMode; VkFlags cullMode; uint32_t frontFace;
    VkBool32 depthBiasEnable; float depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor, lineWidth;
} VkPipelineRasterizationStateCreateInfo;
typedef struct VkPipelineMultisampleStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t rasterizationSamples; VkBool32 sampleShadingEnable; float minSampleShading;
    const uint32_t *pSampleMask; VkBool32 alphaToCoverageEnable, alphaToOneEnable;
} VkPipelineMultisampleStateCreateInfo;
typedef struct VkStencilOpState { uint32_t failOp, passOp, depthFailOp, compareOp, compareMask, writeMask, reference; } VkStencilOpState;
typedef struct VkPipelineDepthStencilStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkBool32 depthTestEnable, depthWriteEnable; uint32_t depthCompareOp;
    VkBool32 depthBoundsTestEnable, stencilTestEnable;
    VkStencilOpState front, back; float minDepthBounds, maxDepthBounds;
} VkPipelineDepthStencilStateCreateInfo;
typedef struct VkPipelineColorBlendAttachmentState {
    VkBool32 blendEnable;
    uint32_t srcColorBlendFactor, dstColorBlendFactor, colorBlendOp;
    uint32_t srcAlphaBlendFactor, dstAlphaBlendFactor, alphaBlendOp;
    VkFlags colorWriteMask;
} VkPipelineColorBlendAttachmentState;
typedef struct VkPipelineColorBlendStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkBool32 logicOpEnable; uint32_t logicOp;
    uint32_t attachmentCount; const VkPipelineColorBlendAttachmentState *pAttachments;
    float blendConstants[4];
} VkPipelineColorBlendStateCreateInfo;
typedef struct VkPipelineDynamicStateCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t dynamicStateCount; const uint32_t *pDynamicStates;
} VkPipelineDynamicStateCreateInfo;
typedef struct VkGraphicsPipelineCreateInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    uint32_t stageCount; const VkPipelineShaderStageCreateInfo *pStages;
    const VkPipelineVertexInputStateCreateInfo *pVertexInputState;
    const VkPipelineInputAssemblyStateCreateInfo *pInputAssemblyState;
    const void *pTessellationState;
    const VkPipelineViewportStateCreateInfo *pViewportState;
    const VkPipelineRasterizationStateCreateInfo *pRasterizationState;
    const VkPipelineMultisampleStateCreateInfo *pMultisampleState;
    const VkPipelineDepthStencilStateCreateInfo *pDepthStencilState;
    const VkPipelineColorBlendStateCreateInfo *pColorBlendState;
    const VkPipelineDynamicStateCreateInfo *pDynamicState;
    VkPipelineLayout layout; VkRenderPass renderPass; uint32_t subpass;
    VkPipeline basePipelineHandle; int32_t basePipelineIndex;
} VkGraphicsPipelineCreateInfo;
typedef struct VkPipelineRenderingCreateInfo {
    VkStructureType sType; const void *pNext;
    uint32_t viewMask, colorAttachmentCount; const VkFormat *pColorAttachmentFormats;
    VkFormat depthAttachmentFormat, stencilAttachmentFormat;
} VkPipelineRenderingCreateInfo;

typedef union VkClearValue { float color[4]; struct { float depth; uint32_t stencil; } depthStencil; } VkClearValue;
typedef struct VkRenderingAttachmentInfo {
    VkStructureType sType; const void *pNext;
    VkImageView imageView; uint32_t imageLayout;
    uint32_t resolveMode; VkImageView resolveImageView; uint32_t resolveImageLayout;
    uint32_t loadOp, storeOp; VkClearValue clearValue;
} VkRenderingAttachmentInfo;
typedef struct VkRenderingInfo {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkRect2D renderArea; uint32_t layerCount, viewMask, colorAttachmentCount;
    const VkRenderingAttachmentInfo *pColorAttachments;
    const VkRenderingAttachmentInfo *pDepthAttachment;
    const VkRenderingAttachmentInfo *pStencilAttachment;
} VkRenderingInfo;

typedef struct VkFenceCreateInfo { VkStructureType sType; const void *pNext; VkFlags flags; } VkFenceCreateInfo;

// ---- window surfaces and swapchains (VK_KHR_surface, VK_KHR_win32_surface, VK_KHR_swapchain) ----
typedef struct VkSurfaceCapabilitiesKHR {
    uint32_t minImageCount, maxImageCount;
    VkExtent2D currentExtent, minImageExtent, maxImageExtent;
    uint32_t maxImageArrayLayers;
    VkFlags supportedTransforms; uint32_t currentTransform;
    VkFlags supportedCompositeAlpha, supportedUsageFlags;
} VkSurfaceCapabilitiesKHR;
typedef struct VkSurfaceFormatKHR { VkFormat format; uint32_t colorSpace; } VkSurfaceFormatKHR;
#define VK_COLOR_SPACE_SRGB_NONLINEAR_KHR 0
#define VK_PRESENT_MODE_IMMEDIATE_KHR 0
#define VK_PRESENT_MODE_MAILBOX_KHR 1
#define VK_PRESENT_MODE_FIFO_KHR 2
#define VK_PRESENT_MODE_FIFO_RELAXED_KHR 3
#define VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR 0x1
#define VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR 0x2
#define VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR 0x4
#define VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR 0x8

typedef struct VkWin32SurfaceCreateInfoKHR {
    VkStructureType sType; const void *pNext; VkFlags flags;
    void *hinstance; void *hwnd;
} VkWin32SurfaceCreateInfoKHR;

typedef struct VkSwapchainCreateInfoKHR {
    VkStructureType sType; const void *pNext; VkFlags flags;
    VkSurfaceKHR surface; uint32_t minImageCount; VkFormat imageFormat; uint32_t imageColorSpace;
    VkExtent2D imageExtent; uint32_t imageArrayLayers; VkFlags imageUsage; uint32_t imageSharingMode;
    uint32_t queueFamilyIndexCount; const uint32_t *pQueueFamilyIndices;
    uint32_t preTransform, compositeAlpha, presentMode; VkBool32 clipped;
    VkSwapchainKHR oldSwapchain;
} VkSwapchainCreateInfoKHR;

typedef struct VkPresentInfoKHR {
    VkStructureType sType; const void *pNext;
    uint32_t waitSemaphoreCount; const VkSemaphore *pWaitSemaphores;
    uint32_t swapchainCount; const VkSwapchainKHR *pSwapchains; const uint32_t *pImageIndices;
    VkResult *pResults;
} VkPresentInfoKHR;

// ---- the functions (pointer types; vk_load.c fills a table of them) ----
typedef void (VKAPI_PTR *PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vkGetInstanceProcAddr)(VkInstance, const char *);
typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vkGetDeviceProcAddr)(VkDevice, const char *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateInstance)(const VkInstanceCreateInfo *, const void *, VkInstance *);
typedef void (VKAPI_PTR *PFN_vkDestroyInstance)(VkInstance, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateInstanceLayerProperties)(uint32_t *, VkLayerProperties *);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateInstanceExtensionProperties)(const char *, uint32_t *, VkExtensionProperties *);
typedef VkResult (VKAPI_PTR *PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t *, VkPhysicalDevice *);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties *);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceFeatures2)(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties *);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice, const char *, uint32_t *, VkExtensionProperties *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo *, const void *, VkDevice *);
typedef void (VKAPI_PTR *PFN_vkDestroyDevice)(VkDevice, const void *);
typedef void (VKAPI_PTR *PFN_vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue *);
typedef VkResult (VKAPI_PTR *PFN_vkDeviceWaitIdle)(VkDevice);
typedef VkResult (VKAPI_PTR *PFN_vkQueueWaitIdle)(VkQueue);
typedef VkResult (VKAPI_PTR *PFN_vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo *, const void *, VkDeviceMemory *);
typedef void (VKAPI_PTR *PFN_vkFreeMemory)(VkDevice, VkDeviceMemory, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void **);
typedef void (VKAPI_PTR *PFN_vkUnmapMemory)(VkDevice, VkDeviceMemory);
typedef VkResult (VKAPI_PTR *PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo *, const void *, VkBuffer *);
typedef void (VKAPI_PTR *PFN_vkDestroyBuffer)(VkDevice, VkBuffer, const void *);
typedef void (VKAPI_PTR *PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements *);
typedef VkResult (VKAPI_PTR *PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
typedef VkDeviceAddress (VKAPI_PTR *PFN_vkGetBufferDeviceAddress)(VkDevice, const VkBufferDeviceAddressInfo *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo *, const void *, VkShaderModule *);
typedef void (VKAPI_PTR *PFN_vkDestroyShaderModule)(VkDevice, VkShaderModule, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, const void *, VkDescriptorSetLayout *);
typedef void (VKAPI_PTR *PFN_vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkCreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo *, const void *, VkPipelineLayout *);
typedef void (VKAPI_PTR *PFN_vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateComputePipelines)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo *, const void *, VkPipeline *);
typedef void (VKAPI_PTR *PFN_vkDestroyPipeline)(VkDevice, VkPipeline, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo *, const void *, VkDescriptorPool *);
typedef void (VKAPI_PTR *PFN_vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo *, VkDescriptorSet *);
typedef VkResult (VKAPI_PTR *PFN_vkFreeDescriptorSets)(VkDevice, VkDescriptorPool, uint32_t, const VkDescriptorSet *);
typedef void (VKAPI_PTR *PFN_vkUpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet *, uint32_t, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo *, const void *, VkCommandPool *);
typedef void (VKAPI_PTR *PFN_vkDestroyCommandPool)(VkDevice, VkCommandPool, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *);
typedef void (VKAPI_PTR *PFN_vkFreeCommandBuffers)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer *);
typedef VkResult (VKAPI_PTR *PFN_vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
typedef VkResult (VKAPI_PTR *PFN_vkEndCommandBuffer)(VkCommandBuffer);
typedef void (VKAPI_PTR *PFN_vkCmdBindPipeline)(VkCommandBuffer, uint32_t, VkPipeline);
typedef void (VKAPI_PTR *PFN_vkCmdBindDescriptorSets)(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet *, uint32_t, const uint32_t *);
typedef void (VKAPI_PTR *PFN_vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef void (VKAPI_PTR *PFN_vkCmdDispatchIndirect)(VkCommandBuffer, VkBuffer, VkDeviceSize);
typedef void (VKAPI_PTR *PFN_vkCmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *);
typedef void (VKAPI_PTR *PFN_vkCmdFillBuffer)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t);
typedef void (VKAPI_PTR *PFN_vkCmdUpdateBuffer)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void *);
typedef void (VKAPI_PTR *PFN_vkCmdPipelineBarrier)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier *,
                                                   uint32_t, const void *, uint32_t, const VkImageMemoryBarrier *);
typedef void (VKAPI_PTR *PFN_vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateSemaphore)(VkDevice, const VkSemaphoreCreateInfo *, const void *, VkSemaphore *);
typedef void (VKAPI_PTR *PFN_vkDestroySemaphore)(VkDevice, VkSemaphore, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkGetSemaphoreCounterValue)(VkDevice, VkSemaphore, uint64_t *);
typedef VkResult (VKAPI_PTR *PFN_vkWaitSemaphores)(VkDevice, const VkSemaphoreWaitInfo *, uint64_t);
typedef VkResult (VKAPI_PTR *PFN_vkCreateQueryPool)(VkDevice, const VkQueryPoolCreateInfo *, const void *, VkQueryPool *);
typedef void (VKAPI_PTR *PFN_vkDestroyQueryPool)(VkDevice, VkQueryPool, const void *);
typedef void (VKAPI_PTR *PFN_vkCmdResetQueryPool)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t);
typedef void (VKAPI_PTR *PFN_vkCmdWriteTimestamp)(VkCommandBuffer, VkFlags, VkQueryPool, uint32_t);
typedef void (VKAPI_PTR *PFN_vkCmdCopyQueryPoolResults)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t, VkBuffer, VkDeviceSize, VkDeviceSize, VkFlags);

typedef VkResult (VKAPI_PTR *PFN_vkCreateImage)(VkDevice, const VkImageCreateInfo *, const void *, VkImage *);
typedef void (VKAPI_PTR *PFN_vkDestroyImage)(VkDevice, VkImage, const void *);
typedef void (VKAPI_PTR *PFN_vkGetImageMemoryRequirements)(VkDevice, VkImage, VkMemoryRequirements *);
typedef VkResult (VKAPI_PTR *PFN_vkBindImageMemory)(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize);
typedef VkResult (VKAPI_PTR *PFN_vkCreateImageView)(VkDevice, const VkImageViewCreateInfo *, const void *, VkImageView *);
typedef void (VKAPI_PTR *PFN_vkDestroyImageView)(VkDevice, VkImageView, const void *);
typedef void (VKAPI_PTR *PFN_vkCmdCopyBufferToImage)(VkCommandBuffer, VkBuffer, VkImage, uint32_t, uint32_t, const VkBufferImageCopy *);
typedef void (VKAPI_PTR *PFN_vkCmdCopyImageToBuffer)(VkCommandBuffer, VkImage, uint32_t, VkBuffer, uint32_t, const VkBufferImageCopy *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateGraphicsPipelines)(VkDevice, VkPipelineCache, uint32_t, const VkGraphicsPipelineCreateInfo *, const void *, VkPipeline *);
typedef void (VKAPI_PTR *PFN_vkCmdBeginRendering)(VkCommandBuffer, const VkRenderingInfo *);
typedef void (VKAPI_PTR *PFN_vkCmdEndRendering)(VkCommandBuffer);
typedef void (VKAPI_PTR *PFN_vkCmdSetViewport)(VkCommandBuffer, uint32_t, uint32_t, const VkViewport *);
typedef void (VKAPI_PTR *PFN_vkCmdSetScissor)(VkCommandBuffer, uint32_t, uint32_t, const VkRect2D *);
typedef void (VKAPI_PTR *PFN_vkCmdDraw)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t);

typedef VkResult (VKAPI_PTR *PFN_vkCreateFence)(VkDevice, const VkFenceCreateInfo *, const void *, VkFence *);
typedef void (VKAPI_PTR *PFN_vkDestroyFence)(VkDevice, VkFence, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkWaitForFences)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t);
typedef VkResult (VKAPI_PTR *PFN_vkResetFences)(VkDevice, uint32_t, const VkFence *);
typedef void (VKAPI_PTR *PFN_vkDestroySurfaceKHR)(VkInstance, VkSurfaceKHR, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceSurfaceSupportKHR)(VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32 *);
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR *);
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, VkSurfaceFormatKHR *);
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, uint32_t *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateWin32SurfaceKHR)(VkInstance, const VkWin32SurfaceCreateInfoKHR *, const void *, VkSurfaceKHR *);
typedef VkResult (VKAPI_PTR *PFN_vkCreateSwapchainKHR)(VkDevice, const VkSwapchainCreateInfoKHR *, const void *, VkSwapchainKHR *);
typedef void (VKAPI_PTR *PFN_vkDestroySwapchainKHR)(VkDevice, VkSwapchainKHR, const void *);
typedef VkResult (VKAPI_PTR *PFN_vkGetSwapchainImagesKHR)(VkDevice, VkSwapchainKHR, uint32_t *, VkImage *);
typedef VkResult (VKAPI_PTR *PFN_vkAcquireNextImageKHR)(VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t *);
typedef VkResult (VKAPI_PTR *PFN_vkQueuePresentKHR)(VkQueue, const VkPresentInfoKHR *);

#endif
