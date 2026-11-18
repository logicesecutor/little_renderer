// webgpu_rt.h - hardware ray tracing: our own extension of webgpu.h, implemented by our Vulkan
// layer only (webgpu/vk/vk_rt.c). Shaped after the WebGPU proposal and
// wgpu's ray queries: bottom-level structures of triangles or boxes, a top level of
// instances, bound like a buffer and queried in WGSL with
//
//     @group(0) @binding(n) var acc: acceleration_structure;
//     var rq: ray_query;
//     rayQueryInitialize(&rq, acc, RayDesc(flags, cull_mask, t_min, t_max, origin, dir));
//     while (rayQueryProceed(&rq)) { ... rayQueryGenerateIntersection(&rq, t) for boxes ... }
//     let hit = rayQueryGetCommittedIntersection(&rq);
//
// The device asks for LRFeatureName_RayQuery (webgpu_lr.h) in its requiredFeatures; the
// adapter has it when wgpuAdapterHasFeature says so.
#ifndef LR_WEBGPU_RT_H
#define LR_WEBGPU_RT_H

#include <webgpu/webgpu.h>
#include "vk/webgpu_lr.h"

typedef struct LRBlasImpl *LRBlas;
typedef struct LRTlasImpl *LRTlas;

typedef enum { LRGeometry_Triangles = 0, LRGeometry_Boxes = 1 } LRGeometryType;

// One geometry of a bottom level. Its buffer needs WGPUBufferUsage_Storage.
// Triangles: 3 vertices each, not indexed, a vertex 3 floats every `stride` bytes (12 if 0);
// boxes: 6 floats (lo x y z, hi x y z) every `stride` bytes (24 if 0). Not opaque: the query
// stops at each candidate for the shader to decide (always so for boxes).
typedef struct {
    LRGeometryType type;
    WGPUBuffer buffer;
    uint64_t offset;
    uint32_t stride;
    uint32_t count;                 // triangles or boxes
    WGPUBool opaque;
} LRGeometry;

typedef struct {
    WGPUStringView label;
    uint32_t geometryCount;
    LRGeometry const *geometries;   // geometry_index in a query is the place in this list
    WGPUBool fastBuild;             // built often (else: traced often)
} LRBlasDescriptor;

// Sized at creation; built by lrCommandEncoderBuildBlas (again whenever the geometry changes,
// with the same counts).
LRBlas lrDeviceCreateBlas(WGPUDevice device, LRBlasDescriptor const *descriptor);
void lrCommandEncoderBuildBlas(WGPUCommandEncoder encoder, LRBlas blas);
uint64_t lrBlasGetAddress(LRBlas blas);           // for LRTlasInstance.blas
void lrBlasRelease(LRBlas blas);

// An instance of the top level, as the GPU reads it (VkAccelerationStructureInstanceKHR):
// 64 bytes, written by the CPU or by a kernel.
typedef struct {
    float transform[3][4];          // object to world, rows
    uint32_t custom_mask;           // instance_custom_data (24 bits) | mask << 24
    uint32_t sbt_flags;             // 0 | flags << 24 (1: no face culling, 4: opaque, 8: not opaque)
    uint64_t blas;                  // lrBlasGetAddress
} LRTlasInstance;

LRTlas lrDeviceCreateTlas(WGPUDevice device, uint32_t maxInstances, WGPUStringView label);
// Built from `count` LRTlasInstance in `instances` (WGPUBufferUsage_Storage) at `offset`, as
// they are when the command runs.
void lrCommandEncoderBuildTlas(WGPUCommandEncoder encoder, LRTlas tlas, WGPUBuffer instances, uint64_t offset, uint32_t count);
// Updated in place from the same number of instances as its last build, moved (a refit: the
// tree's shape kept, its boxes made again; much faster than a build, slower to trace the
// more they moved: build again now and then). Not built yet, or another count: built.
void lrCommandEncoderUpdateTlas(WGPUCommandEncoder encoder, LRTlas tlas, WGPUBuffer instances, uint64_t offset, uint32_t count);
void lrTlasRelease(LRTlas tlas);

// Binding a top level: a bind group layout entry with no other binding type, chaining an
// LRBindGroupLayoutEntryTlas; the bind group's entry chaining an LRBindGroupEntryTlas.
#define LRSType_BindGroupLayoutEntryTlas ((WGPUSType)0x4C520001)
#define LRSType_BindGroupEntryTlas ((WGPUSType)0x4C520002)
typedef struct { WGPUChainedStruct chain; } LRBindGroupLayoutEntryTlas;
typedef struct { WGPUChainedStruct chain; LRTlas tlas; } LRBindGroupEntryTlas;

#endif
