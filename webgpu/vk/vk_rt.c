// vk_rt.c - acceleration structures for ray queries (webgpu_rt.h), on VK_KHR_acceleration_structure.
//
// Each structure has its own memory and its own scratch memory, sized once at creation (by
// vkGetAccelerationStructureBuildSizesKHR for the largest it will hold) and kept, so building
// again (a top level every frame) allocates nothing. A build is a command like any other:
// the barrier before it (vk_before) orders it after what wrote its inputs, and the one
// before the next command orders the queries after it.
#include "vk/vk_impl.h"

#include <stdlib.h>
#include <string.h>

VkDeviceAddress vk_buffer_address(WGPUBuffer b){
    VkBufferDeviceAddressInfo i;
    memset(&i, 0, sizeof(i));
    i.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    i.buffer = b->vk;
    return vk.vkGetBufferDeviceAddress(b->dev->vk, &i);
}

static VkDeviceAddress raw_address(WGPUDevice dev, VkBuffer buf){
    VkBufferDeviceAddressInfo i;
    memset(&i, 0, sizeof(i));
    i.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    i.buffer = buf;
    return vk.vkGetBufferDeviceAddress(dev->vk, &i);
}

static void as_free(WGPUDevice dev, VkAs *a){
    if (a->as) vk.vkDestroyAccelerationStructureKHR(dev->vk, a->as, NULL);
    if (a->buf) vk.vkDestroyBuffer(dev->vk, a->buf, NULL);
    if (a->mem) vk.vkFreeMemory(dev->vk, a->mem, NULL);
    if (a->scratch) vk.vkDestroyBuffer(dev->vk, a->scratch, NULL);
    if (a->smem) vk.vkFreeMemory(dev->vk, a->smem, NULL);
    memset(a, 0, sizeof(*a));
}

// the structure's memory and scratch for this build's sizes; 0 on failure
static int as_make(WGPUDevice dev, VkAs *a, uint32_t type, VkAccelerationStructureBuildGeometryInfoKHR *info,
                   const uint32_t *max_counts){
    VkAccelerationStructureBuildSizesInfoKHR sz;
    memset(&sz, 0, sizeof(sz));
    sz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vk.vkGetAccelerationStructureBuildSizesKHR(dev->vk, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, info, max_counts, &sz);
    if (!vk_alloc_buffer(dev, sz.accelerationStructureSize,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &a->buf, &a->mem, NULL)) return 0;
    uint32_t align = dev->ad->as_scratch_align;
    VkDeviceSize scratch = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
    if (!vk_alloc_buffer(dev, scratch + align,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &a->scratch, &a->smem, NULL)) return 0;
    a->scratch_at = (raw_address(dev, a->scratch) + align - 1) / align * align;
    VkAccelerationStructureCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.buffer = a->buf;
    ci.size = sz.accelerationStructureSize;
    ci.type = type;
    if (vk.vkCreateAccelerationStructureKHR(dev->vk, &ci, NULL, &a->as) != VK_SUCCESS){ a->as = 0; return 0; }
    VkAccelerationStructureDeviceAddressInfoKHR ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    ai.accelerationStructure = a->as;
    a->addr = vk.vkGetAccelerationStructureDeviceAddressKHR(dev->vk, &ai);
    return 1;
}

// ---- bottom level ----
static void geometries(LRBlas b, VkAccelerationStructureGeometryKHR *g, uint32_t *counts){
    for (uint32_t i = 0; i < b->ngeo; i++){
        const LRGeometry *s = &b->geo[i];
        memset(&g[i], 0, sizeof(g[i]));
        g[i].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        VkDeviceAddress at = s->buffer ? vk_buffer_address(s->buffer) + s->offset : 0;
        if (s->type == LRGeometry_Triangles){
            g[i].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            VkAccelerationStructureGeometryTrianglesDataKHR *t = &g[i].geometry.triangles;
            t->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            t->vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            t->vertexData.deviceAddress = at;
            t->vertexStride = s->stride ? s->stride : 12;
            t->maxVertex = s->count ? 3 * s->count - 1 : 0;
            t->indexType = VK_INDEX_TYPE_NONE_KHR;
            g[i].flags = s->opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
        } else {
            g[i].geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
            VkAccelerationStructureGeometryAabbsDataKHR *a = &g[i].geometry.aabbs;
            a->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
            a->data.deviceAddress = at;
            a->stride = s->stride ? s->stride : 24;
        }
        counts[i] = s->count;
    }
}

static void blas_free(Obj *o){
    LRBlas b = (LRBlas)o;
    as_free(b->dev, &b->a);
    for (uint32_t i = 0; i < b->ngeo; i++) if (b->geo[i].buffer) obj_unref(&b->geo[i].buffer->o);
    free(b->geo);
    free(b);
}

LRBlas lrDeviceCreateBlas(WGPUDevice device, LRBlasDescriptor const *d){
    if (!device->rt){ vk_error(device, WGPUErrorType_Validation, "lrDeviceCreateBlas: the device has no ray queries"); return NULL; }
    LRBlas b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    obj_init(&b->o, blas_free);
    b->dev = device;
    b->ngeo = d->geometryCount;
    b->geo = calloc(b->ngeo ? b->ngeo : 1, sizeof(*b->geo));
    VkAccelerationStructureGeometryKHR *g = calloc(b->ngeo ? b->ngeo : 1, sizeof(*g));
    uint32_t *counts = calloc(b->ngeo ? b->ngeo : 1, sizeof(*counts));
    if (!b->geo || !g || !counts){ free(g); free(counts); blas_free(&b->o); return NULL; }
    for (uint32_t i = 0; i < b->ngeo; i++){
        b->geo[i] = d->geometries[i];
        if (b->geo[i].buffer) obj_ref(&b->geo[i].buffer->o);
    }
    b->flags = d->fastBuild ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    geometries(b, g, counts);
    VkAccelerationStructureBuildGeometryInfoKHR info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = b->flags;
    info.geometryCount = b->ngeo;
    info.pGeometries = g;
    int ok = as_make(device, &b->a, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &info, counts);
    free(g); free(counts);
    if (!ok){ vk_error(device, WGPUErrorType_OutOfMemory, "bottom-level acceleration structure: no memory"); blas_free(&b->o); return NULL; }
    return b;
}

void lrCommandEncoderBuildBlas(WGPUCommandEncoder e, LRBlas b){
    if (!b || !b->ngeo) return;
    VkAccelerationStructureGeometryKHR *g = calloc(b->ngeo, sizeof(*g));
    uint32_t *counts = calloc(b->ngeo, sizeof(*counts));
    VkAccelerationStructureBuildRangeInfoKHR *r = calloc(b->ngeo, sizeof(*r));
    if (!g || !counts || !r){ free(g); free(counts); free(r); return; }
    geometries(b, g, counts);
    for (uint32_t i = 0; i < b->ngeo; i++){
        r[i].primitiveCount = counts[i];
        if (b->geo[i].buffer) refs_add(&e->refs, &b->geo[i].buffer->o);
    }
    refs_add(&e->refs, &b->o);
    VkAccelerationStructureBuildGeometryInfoKHR info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = b->flags;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.dstAccelerationStructure = b->a.as;
    info.geometryCount = b->ngeo;
    info.pGeometries = g;
    info.scratchData.deviceAddress = b->a.scratch_at;
    const VkAccelerationStructureBuildRangeInfoKHR *pr = r;
    vk_before(e, VK_CMD_BUILD);
    vk.vkCmdBuildAccelerationStructuresKHR(e->cb, 1, &info, &pr);
    free(g); free(counts); free(r);
}

uint64_t lrBlasGetAddress(LRBlas b){ return b ? b->a.addr : 0; }
void lrBlasRelease(LRBlas b){ if (b) obj_unref(&b->o); }

// ---- top level ----
static void tlas_info(VkAccelerationStructureBuildGeometryInfoKHR *info, VkAccelerationStructureGeometryKHR *g, VkDeviceAddress at){
    memset(g, 0, sizeof(*g));
    g->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    g->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g->geometry.instances.data.deviceAddress = at;
    memset(info, 0, sizeof(*info));
    info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info->type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info->flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info->mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info->geometryCount = 1;
    info->pGeometries = g;
}

static void tlas_free(Obj *o){
    LRTlas t = (LRTlas)o;
    as_free(t->dev, &t->a);
    free(t);
}

LRTlas lrDeviceCreateTlas(WGPUDevice device, uint32_t max, WGPUStringView label){
    (void)label;
    if (!device->rt){ vk_error(device, WGPUErrorType_Validation, "lrDeviceCreateTlas: the device has no ray queries"); return NULL; }
    LRTlas t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    obj_init(&t->o, tlas_free);
    t->dev = device;
    t->max = max ? max : 1;
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureGeometryKHR g;
    tlas_info(&info, &g, 0);
    if (!as_make(device, &t->a, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &info, &t->max)){
        vk_error(device, WGPUErrorType_OutOfMemory, "top-level acceleration structure: no memory");
        tlas_free(&t->o);
        return NULL;
    }
    return t;
}

static void tlas_build(WGPUCommandEncoder e, LRTlas t, WGPUBuffer instances, uint64_t offset, uint32_t count, int update){
    if (!t) return;
    if (count > t->max){ vk_error(e->dev, WGPUErrorType_Validation, "lrCommandEncoderBuildTlas: %u instances, room for %u", count, t->max); count = t->max; }
    if (!instances) count = 0;
    update = update && t->has_built && t->built == count && count > 0;
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureGeometryKHR g;
    tlas_info(&info, &g, instances ? vk_buffer_address(instances) + offset : 0);
    if (update){
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        info.srcAccelerationStructure = t->a.as;
    }
    t->built = count;
    t->has_built = 1;
    info.dstAccelerationStructure = t->a.as;
    info.scratchData.deviceAddress = t->a.scratch_at;
    VkAccelerationStructureBuildRangeInfoKHR r;
    memset(&r, 0, sizeof(r));
    r.primitiveCount = instances ? count : 0;
    const VkAccelerationStructureBuildRangeInfoKHR *pr = &r;
    refs_add(&e->refs, &t->o);
    if (instances) refs_add(&e->refs, &instances->o);
    vk_before(e, VK_CMD_BUILD);
    vk.vkCmdBuildAccelerationStructuresKHR(e->cb, 1, &info, &pr);
}

void lrCommandEncoderBuildTlas(WGPUCommandEncoder e, LRTlas t, WGPUBuffer instances, uint64_t offset, uint32_t count){
    tlas_build(e, t, instances, offset, count, 0);
}
void lrCommandEncoderUpdateTlas(WGPUCommandEncoder e, LRTlas t, WGPUBuffer instances, uint64_t offset, uint32_t count){
    tlas_build(e, t, instances, offset, count, 1);
}

void lrTlasRelease(LRTlas t){ if (t) obj_unref(&t->o); }
