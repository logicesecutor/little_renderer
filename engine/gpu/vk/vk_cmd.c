// vk_cmd.c - command encoders, compute passes, copies, timestamps (vk_impl.h)
#include "gpu/vk/vk_impl.h"

#include <stdlib.h>
#include <string.h>

// ---- encoders and command buffers ----
static void encoder_free(Obj *o){
    WGPUCommandEncoder e = (WGPUCommandEncoder)o;
    if (e->cb && !e->finished){ vk.vkEndCommandBuffer(e->cb); vk_cb_put(e->dev, e->cb); }
    refs_release(&e->refs);
    free(e);
}

WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice device, WGPUCommandEncoderDescriptor const *descriptor){
    (void)descriptor;
    WGPUCommandEncoder e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    obj_init(&e->o, encoder_free);
    e->dev = device;
    e->cb = vk_cb_get(device);
    if (!e->cb){ free(e); return NULL; }
    return e;
}

void wgpuCommandEncoderRelease(WGPUCommandEncoder e){ if (e) obj_unref(&e->o); }

// before a command: the ones before it done (vk_impl.h: a full barrier)
static void before(WGPUCommandEncoder e){
    if (e->dirty) vk_barrier(e->cb);
    e->dirty = 1;
}

static void cmdbuf_free(Obj *o){
    WGPUCommandBuffer c = (WGPUCommandBuffer)o;
    if (c->cb) vk_cb_put(c->dev, c->cb);          // never submitted
    refs_release(&c->refs);
    free(c);
}

WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder e, WGPUCommandBufferDescriptor const *descriptor){
    (void)descriptor;
    WGPUCommandBuffer c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    obj_init(&c->o, cmdbuf_free);
    c->dev = e->dev;
    vk_barrier_to_host(e->cb);                     // (readbacks: the host sees what was written)
    vk.vkEndCommandBuffer(e->cb);
    c->cb = e->cb;
    c->refs = e->refs;
    memset(&e->refs, 0, sizeof(e->refs));
    e->cb = NULL;
    e->finished = 1;
    return c;
}

void wgpuCommandBufferRelease(WGPUCommandBuffer c){ if (c) obj_unref(&c->o); }

void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder e, WGPUBuffer src, uint64_t so, WGPUBuffer dst, uint64_t dof, uint64_t size){
    if (!size) return;
    before(e);
    refs_add(&e->refs, &src->o);
    refs_add(&e->refs, &dst->o);
    VkBufferCopy c = {so, dof, size};
    vk.vkCmdCopyBuffer(e->cb, src->vk, dst->vk, 1, &c);
}

void wgpuCommandEncoderClearBuffer(WGPUCommandEncoder e, WGPUBuffer buffer, uint64_t offset, uint64_t size){
    before(e);
    refs_add(&e->refs, &buffer->o);
    VkDeviceSize n = size == WGPU_WHOLE_SIZE ? VK_WHOLE_SIZE : size;
    if (n == 0) return;
    vk.vkCmdFillBuffer(e->cb, buffer->vk, offset, n, 0);
}

// ---- query sets (timestamps) ----
static void queryset_free(Obj *o){
    WGPUQuerySet q = (WGPUQuerySet)o;
    if (q->vk) vk.vkDestroyQueryPool(q->dev->vk, q->vk, NULL);
    free(q);
}

WGPUQuerySet wgpuDeviceCreateQuerySet(WGPUDevice device, WGPUQuerySetDescriptor const *d){
    WGPUQuerySet q = calloc(1, sizeof(*q));
    if (!q) return NULL;
    obj_init(&q->o, queryset_free);
    q->dev = device;
    q->count = d->count;
    VkQueryPoolCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = d->count;
    if (vk.vkCreateQueryPool(device->vk, &ci, NULL, &q->vk) != VK_SUCCESS){ free(q); return NULL; }
    return q;
}

void wgpuQuerySetDestroy(WGPUQuerySet q){ (void)q; }
void wgpuQuerySetRelease(WGPUQuerySet q){ if (q) obj_unref(&q->o); }

// (Vulkan counts in ticks of timestampPeriod ns: 1 on NVIDIA; other GPUs would need a scale)
void wgpuCommandEncoderResolveQuerySet(WGPUCommandEncoder e, WGPUQuerySet q, uint32_t first, uint32_t count,
                                       WGPUBuffer dst, uint64_t offset){
    before(e);
    refs_add(&e->refs, &q->o);
    refs_add(&e->refs, &dst->o);
    vk.vkCmdCopyQueryPoolResults(e->cb, q->vk, first, count, dst->vk, offset, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

// ---- compute passes ----
static void cpass_free(Obj *o){ free(o); }

WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder e, WGPUComputePassDescriptor const *d){
    WGPUComputePassEncoder p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    obj_init(&p->o, cpass_free);
    p->enc = e;
    if (d && d->timestampWrites && d->timestampWrites->querySet){
        const WGPUPassTimestampWrites *tw = d->timestampWrites;
        p->ts = tw->querySet;
        p->ts_end = tw->endOfPassWriteIndex;
        refs_add(&e->refs, &tw->querySet->o);
        before(e);
        if (tw->beginningOfPassWriteIndex != WGPU_QUERY_SET_INDEX_UNDEFINED){
            vk.vkCmdResetQueryPool(e->cb, tw->querySet->vk, tw->beginningOfPassWriteIndex, 1);
            vk.vkCmdWriteTimestamp(e->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tw->querySet->vk, tw->beginningOfPassWriteIndex);
        }
        if (tw->endOfPassWriteIndex != WGPU_QUERY_SET_INDEX_UNDEFINED)
            vk.vkCmdResetQueryPool(e->cb, tw->querySet->vk, tw->endOfPassWriteIndex, 1);
    }
    return p;
}

void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder p, WGPUComputePipeline pipe){
    if (p->pipe != pipe){ p->pipe = pipe; p->pipe_bound = 0; p->bg_dirty = 1; }
    refs_add(&p->enc->refs, &pipe->o);
}

void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder p, uint32_t index, WGPUBindGroup group,
                                        size_t n, uint32_t const *offsets){
    if (index >= 4) return;
    p->bg[index] = group;
    p->ndyn[index] = (uint32_t)(n < 8 ? n : 8);
    for (uint32_t i = 0; i < p->ndyn[index]; i++) p->dyn[index][i] = offsets[i];
    p->bg_dirty = 1;
    if (group) refs_add(&p->enc->refs, &group->o);
}

// the pipeline and bind groups as set, before a dispatch
static int flush(WGPUComputePassEncoder p){
    if (!p->pipe) return 0;
    VkCommandBuffer cb = p->enc->cb;
    if (!p->pipe_bound){ vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe->vk); p->pipe_bound = 1; }
    if (p->bg_dirty){
        WGPUPipelineLayout pl = p->pipe->layout;
        for (int i = 0; i < 4 && i < pl->n; i++){
            if (!p->bg[i]) continue;
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->vk, (uint32_t)i, 1, &p->bg[i]->set,
                                       p->ndyn[i], p->dyn[i]);
        }
        p->bg_dirty = 0;
    }
    return 1;
}

void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder p, uint32_t x, uint32_t y, uint32_t z){
    if (!x || !y || !z || !flush(p)) return;
    before(p->enc);
    vk.vkCmdDispatch(p->enc->cb, x, y, z);
}

void wgpuComputePassEncoderDispatchWorkgroupsIndirect(WGPUComputePassEncoder p, WGPUBuffer buffer, uint64_t offset){
    if (!flush(p)) return;
    before(p->enc);
    refs_add(&p->enc->refs, &buffer->o);
    vk.vkCmdDispatchIndirect(p->enc->cb, buffer->vk, offset);
}

void wgpuComputePassEncoderEnd(WGPUComputePassEncoder p){
    if (p->ts && p->ts_end != WGPU_QUERY_SET_INDEX_UNDEFINED){
        before(p->enc);
        vk.vkCmdWriteTimestamp(p->enc->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->ts->vk, p->ts_end);
    }
}

void wgpuComputePassEncoderRelease(WGPUComputePassEncoder p){ if (p) obj_unref(&p->o); }
