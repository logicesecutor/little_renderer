// vk_pipe.c - shader modules, bind group layouts, pipeline layouts, bind groups, compute
// pipelines (vk_impl.h)
#include "gpu/vk/vk_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t sv_len(WGPUStringView v){ return v.data ? (v.length == WGPU_STRLEN ? strlen(v.data) : v.length) : 0; }

// ---- shader modules: WGSL through our compiler (engine/gpu/wgsl), or SPIR-V as it is ----
static void module_free(Obj *o){
    WGPUShaderModule m = (WGPUShaderModule)o;
    if (m->vk) vk.vkDestroyShaderModule(m->dev->vk, m->vk, NULL);
    free(m);
}

WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice device, WGPUShaderModuleDescriptor const *d){
    uint32_t *words = NULL;
    size_t nwords = 0;
    const uint32_t *code = NULL;
    for (const WGPUChainedStruct *c = d->nextInChain; c; c = c->next){
        if (c->sType == WGPUSType_ShaderSourceSPIRV){
            const WGPUShaderSourceSPIRV *s = (const WGPUShaderSourceSPIRV *)c;
            code = s->code;
            nwords = s->codeSize;
        } else if (c->sType == WGPUSType_ShaderSourceWGSL){
            const WGPUShaderSourceWGSL *s = (const WGPUShaderSourceWGSL *)c;
            char err[1024];
            // LR_VK_DUMP=dir: each WGSL module and its SPIR-V saved there (to check with spirv-val)
            const char *dump = getenv("LR_VK_DUMP");
            static int k;
            char path[512];
            if (dump && *dump){
                snprintf(path, sizeof(path), "%s/%03d.wgsl", dump, k);
                FILE *f = fopen(path, "wb");
                if (f){ fwrite(s->code.data, 1, sv_len(s->code), f); fclose(f); }
            }
            if (wgsl_to_spirv(s->code.data, sv_len(s->code), &words, &nwords, err, sizeof(err))){
                vk_error(device, WGPUErrorType_Validation, "shader %.*s (%03d): %s", (int)sv_len(d->label), d->label.data ? d->label.data : "", k++, err);
                return NULL;
            }
            code = words;
            if (dump && *dump){
                snprintf(path, sizeof(path), "%s/%03d.spv", dump, k);
                FILE *f = fopen(path, "wb");
                if (f){ fwrite(words, 4, nwords, f); fclose(f); }
            }
            k++;
        }
    }
    if (!code){ vk_error(device, WGPUErrorType_Validation, "shader module without WGSL or SPIR-V"); return NULL; }
    WGPUShaderModule m = calloc(1, sizeof(*m));
    if (!m){ free(words); return NULL; }
    obj_init(&m->o, module_free);
    m->dev = device;
    VkShaderModuleCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = nwords * 4;
    ci.pCode = code;
    VkResult r = vk.vkCreateShaderModule(device->vk, &ci, NULL, &m->vk);
    free(words);
    if (r != VK_SUCCESS){ vk_error(device, WGPUErrorType_Internal, "vkCreateShaderModule failed (%d)", r); free(m); return NULL; }
    return m;
}

void wgpuShaderModuleRelease(WGPUShaderModule m){ if (m) obj_unref(&m->o); }

// ---- bind group layouts ----
static void bgl_free(Obj *o){
    WGPUBindGroupLayout l = (WGPUBindGroupLayout)o;
    if (l->vk) vk.vkDestroyDescriptorSetLayout(l->dev->vk, l->vk, NULL);
    free(l->e);
    free(l);
}

static VkFlags stages(WGPUShaderStage s){
    return ((s & WGPUShaderStage_Vertex) ? VK_SHADER_STAGE_VERTEX_BIT : 0) |
           ((s & WGPUShaderStage_Fragment) ? VK_SHADER_STAGE_FRAGMENT_BIT : 0) |
           ((s & WGPUShaderStage_Compute) ? VK_SHADER_STAGE_COMPUTE_BIT : 0);
}

WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayout(WGPUDevice device, WGPUBindGroupLayoutDescriptor const *d){
    WGPUBindGroupLayout l = calloc(1, sizeof(*l));
    VkDescriptorSetLayoutBinding *b = calloc(d->entryCount ? d->entryCount : 1, sizeof(*b));
    if (!l || !b){ free(l); free(b); return NULL; }
    obj_init(&l->o, bgl_free);
    l->dev = device;
    l->e = calloc(d->entryCount ? d->entryCount : 1, sizeof(*l->e));
    l->n = (int)d->entryCount;
    for (size_t i = 0; i < d->entryCount; i++){
        const WGPUBindGroupLayoutEntry *e = &d->entries[i];
        VkDescriptorType t;
        int dyn = 0;
        if (e->buffer.type != WGPUBufferBindingType_BindingNotUsed){
            dyn = e->buffer.hasDynamicOffset != 0;
            int uniform = e->buffer.type == WGPUBufferBindingType_Uniform || e->buffer.type == WGPUBufferBindingType_Undefined;
            t = uniform ? (dyn ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                        : (dyn ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        } else if (e->texture.sampleType != WGPUTextureSampleType_BindingNotUsed){
            t = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        } else if (e->storageTexture.access != WGPUStorageTextureAccess_BindingNotUsed){
            t = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else {
            t = VK_DESCRIPTOR_TYPE_SAMPLER;
        }
        b[i].binding = e->binding;
        b[i].descriptorType = t;
        b[i].descriptorCount = 1;
        b[i].stageFlags = stages(e->visibility);
        l->e[i].binding = e->binding;
        l->e[i].type = t;
        l->e[i].dynamic = dyn;
        l->ndynamic += dyn;
    }
    VkDescriptorSetLayoutCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = (uint32_t)d->entryCount;
    ci.pBindings = b;
    VkResult r = vk.vkCreateDescriptorSetLayout(device->vk, &ci, NULL, &l->vk);
    free(b);
    if (r != VK_SUCCESS){ vk_error(device, WGPUErrorType_Internal, "vkCreateDescriptorSetLayout failed (%d)", r); bgl_free(&l->o); return NULL; }
    return l;
}

void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout l){ if (l) obj_unref(&l->o); }

// ---- pipeline layouts ----
static void pl_free(Obj *o){
    WGPUPipelineLayout p = (WGPUPipelineLayout)o;
    if (p->vk) vk.vkDestroyPipelineLayout(p->dev->vk, p->vk, NULL);
    for (int i = 0; i < p->n; i++) obj_unref(&p->bgl[i]->o);
    free(p);
}

WGPUPipelineLayout wgpuDeviceCreatePipelineLayout(WGPUDevice device, WGPUPipelineLayoutDescriptor const *d){
    WGPUPipelineLayout p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    obj_init(&p->o, pl_free);
    p->dev = device;
    VkDescriptorSetLayout sets[4];
    for (size_t i = 0; i < d->bindGroupLayoutCount && i < 4; i++){
        p->bgl[p->n] = d->bindGroupLayouts[i];
        obj_ref(&p->bgl[p->n]->o);
        sets[p->n++] = d->bindGroupLayouts[i]->vk;
    }
    VkPipelineLayoutCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = (uint32_t)p->n;
    ci.pSetLayouts = sets;
    if (vk.vkCreatePipelineLayout(device->vk, &ci, NULL, &p->vk) != VK_SUCCESS){ pl_free(&p->o); return NULL; }
    return p;
}

void wgpuPipelineLayoutRelease(WGPUPipelineLayout p){ if (p) obj_unref(&p->o); }

// ---- bind groups: descriptor sets from pools made as needed ----
static VkDescriptorPool new_pool(WGPUDevice dev){
    static const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 512}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256}, {VK_DESCRIPTOR_TYPE_SAMPLER, 64}};
    VkDescriptorPoolCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 1024;
    ci.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    ci.pPoolSizes = sizes;
    VkDescriptorPool p = 0;
    if (vk.vkCreateDescriptorPool(dev->vk, &ci, NULL, &p) != VK_SUCCESS) return 0;
    VkDescriptorPool *v = realloc(dev->dpools, (size_t)(dev->ndpools + 1) * sizeof(*v));
    if (!v){ vk.vkDestroyDescriptorPool(dev->vk, p, NULL); return 0; }
    dev->dpools = v;
    dev->dpools[dev->ndpools++] = p;
    return p;
}

static void bg_free(Obj *o){
    WGPUBindGroup g = (WGPUBindGroup)o;
    if (g->set) vk.vkFreeDescriptorSets(g->dev->vk, g->pool, 1, &g->set);
    refs_release(&g->held);
    if (g->layout) obj_unref(&g->layout->o);
    free(g);
}

WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice device, WGPUBindGroupDescriptor const *d){
    WGPUBindGroup g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    obj_init(&g->o, bg_free);
    g->dev = device;
    g->layout = d->layout;
    obj_ref(&d->layout->o);
    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &d->layout->vk;
    VkResult r = -1;
    for (int i = device->ndpools - 1; i >= 0 && r != VK_SUCCESS; i--){
        ai.descriptorPool = device->dpools[i];
        r = vk.vkAllocateDescriptorSets(device->vk, &ai, &g->set);
        if (r == VK_SUCCESS) g->pool = device->dpools[i];
    }
    if (r != VK_SUCCESS){
        ai.descriptorPool = new_pool(device);
        if (ai.descriptorPool && vk.vkAllocateDescriptorSets(device->vk, &ai, &g->set) == VK_SUCCESS) g->pool = ai.descriptorPool;
        else { g->set = 0; vk_error(device, WGPUErrorType_OutOfMemory, "no descriptor set"); bg_free(&g->o); return NULL; }
    }
    VkWriteDescriptorSet *w = calloc(d->entryCount ? d->entryCount : 1, sizeof(*w));
    VkDescriptorBufferInfo *bi = calloc(d->entryCount ? d->entryCount : 1, sizeof(*bi));
    VkDescriptorImageInfo *ii = calloc(d->entryCount ? d->entryCount : 1, sizeof(*ii));
    int n = 0;
    for (size_t i = 0; w && bi && ii && i < d->entryCount; i++){
        const WGPUBindGroupEntry *e = &d->entries[i];
        VkDescriptorType t = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        for (int k = 0; k < d->layout->n; k++) if (d->layout->e[k].binding == e->binding) t = d->layout->e[k].type;
        if (t == VK_DESCRIPTOR_TYPE_MAX_ENUM) continue;
        w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[n].dstSet = g->set;
        w[n].dstBinding = e->binding;
        w[n].descriptorCount = 1;
        w[n].descriptorType = t;
        if (e->buffer){
            bi[n].buffer = e->buffer->vk;
            bi[n].offset = e->offset;
            bi[n].range = e->size == WGPU_WHOLE_SIZE ? VK_WHOLE_SIZE : e->size;
            w[n].pBufferInfo = &bi[n];
            refs_add(&g->held, &e->buffer->o);
        } else if (e->textureView){
            ii[n].imageView = e->textureView->vk;
            ii[n].imageLayout = 1;                       // VK_IMAGE_LAYOUT_GENERAL
            w[n].pImageInfo = &ii[n];
            refs_add(&g->held, &e->textureView->o);
        } else continue;
        n++;
    }
    if (n) vk.vkUpdateDescriptorSets(device->vk, (uint32_t)n, w, 0, NULL);
    free(w); free(bi); free(ii);
    return g;
}

void wgpuBindGroupRelease(WGPUBindGroup g){ if (g) obj_unref(&g->o); }

// ---- compute pipelines ----
static void cpipe_free(Obj *o){
    WGPUComputePipeline p = (WGPUComputePipeline)o;
    if (p->vk) vk.vkDestroyPipeline(p->dev->vk, p->vk, NULL);
    if (p->layout) obj_unref(&p->layout->o);
    free(p);
}

WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device, WGPUComputePipelineDescriptor const *d){
    if (!d->layout){ vk_error(device, WGPUErrorType_Validation, "compute pipeline without a layout (the automatic layout is not supported)"); return NULL; }
    if (!d->compute.module) return NULL;
    WGPUComputePipeline p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    obj_init(&p->o, cpipe_free);
    p->dev = device;
    p->layout = d->layout;
    obj_ref(&d->layout->o);
    char entry[128];
    size_t n = sv_len(d->compute.entryPoint);
    snprintf(entry, sizeof(entry), "%.*s", (int)(n ? n : 4), n ? d->compute.entryPoint.data : "main");
    VkComputePipelineCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = d->compute.module->vk;
    ci.stage.pName = entry;
    ci.layout = d->layout->vk;
    VkResult r = vk.vkCreateComputePipelines(device->vk, 0, 1, &ci, NULL, &p->vk);
    if (r != VK_SUCCESS){
        vk_error(device, WGPUErrorType_Internal, "compute pipeline %.*s: vkCreateComputePipelines failed (%d)",
                 (int)sv_len(d->label), d->label.data ? d->label.data : "", r);
        p->vk = 0;
        cpipe_free(&p->o);
        return NULL;
    }
    return p;
}

void wgpuComputePipelineRelease(WGPUComputePipeline p){ if (p) obj_unref(&p->o); }
