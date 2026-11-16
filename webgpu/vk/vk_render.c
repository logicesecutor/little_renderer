// vk_render.c - render pipelines and render passes (vk_impl.h), on dynamic rendering
// (Vulkan 1.3: no VkRenderPass or framebuffer objects).
//
// The engine draws without vertex buffers (vertices come from storage buffers by
// vertex_index), so a pipeline has no vertex input. Our WGSL compiler negates a vertex's
// y (Vulkan's clip space points y down, WebGPU's up), which mirrors every triangle as Vulkan
// sees it: a WebGPU counter-clockwise front face is a clockwise one here.
//
// No barrier can go inside a dynamic render pass, so the one before the pass (vk_before)
// covers all of its draws, and the command after the pass waits for the whole pass.
#include "vk/vk_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t sv_len(WGPUStringView v){ return v.data ? (v.length == WGPU_STRLEN ? strlen(v.data) : v.length) : 0; }

static uint32_t compare_op(WGPUCompareFunction f){
    switch (f){
    case WGPUCompareFunction_Never: return 0;
    case WGPUCompareFunction_Less: return 1;
    case WGPUCompareFunction_Equal: return 2;
    case WGPUCompareFunction_LessEqual: return 3;
    case WGPUCompareFunction_Greater: return 4;
    case WGPUCompareFunction_NotEqual: return 5;
    case WGPUCompareFunction_GreaterEqual: return 6;
    default: return 7;                             // Always (and Undefined)
    }
}

static uint32_t blend_factor(WGPUBlendFactor f){
    switch (f){
    case WGPUBlendFactor_One: return 1;
    case WGPUBlendFactor_Src: return 2;
    case WGPUBlendFactor_OneMinusSrc: return 3;
    case WGPUBlendFactor_Dst: return 4;
    case WGPUBlendFactor_OneMinusDst: return 5;
    case WGPUBlendFactor_SrcAlpha: return 6;
    case WGPUBlendFactor_OneMinusSrcAlpha: return 7;
    case WGPUBlendFactor_DstAlpha: return 8;
    case WGPUBlendFactor_OneMinusDstAlpha: return 9;
    case WGPUBlendFactor_Constant: return 10;
    case WGPUBlendFactor_OneMinusConstant: return 11;
    case WGPUBlendFactor_SrcAlphaSaturated: return 14;
    default: return 0;                             // Zero (and Undefined)
    }
}

static uint32_t blend_op(WGPUBlendOperation o){
    switch (o){
    case WGPUBlendOperation_Subtract: return 1;
    case WGPUBlendOperation_ReverseSubtract: return 2;
    case WGPUBlendOperation_Min: return 3;
    case WGPUBlendOperation_Max: return 4;
    default: return 0;                             // Add (and Undefined)
    }
}

// ---- render pipelines ----
static void rpipe_free(Obj *o){
    WGPURenderPipeline p = (WGPURenderPipeline)o;
    if (p->vk) vk.vkDestroyPipeline(p->dev->vk, p->vk, NULL);
    if (p->layout) obj_unref(&p->layout->o);
    free(p);
}

static void entry_name(char *out, size_t n, WGPUStringView v, const char *def){
    size_t k = sv_len(v);
    if (k) snprintf(out, n, "%.*s", (int)k, v.data);
    else snprintf(out, n, "%s", def);
}

WGPURenderPipeline wgpuDeviceCreateRenderPipeline(WGPUDevice device, WGPURenderPipelineDescriptor const *d){
    const char *label = d->label.data ? d->label.data : "";
    int nlabel = (int)sv_len(d->label);
    if (!d->layout){ vk_error(device, WGPUErrorType_Validation, "render pipeline %.*s without a layout (the automatic layout is not supported)", nlabel, label); return NULL; }
    if (!d->vertex.module || (d->fragment && !d->fragment->module)) return NULL;
    if (d->vertex.bufferCount){ vk_error(device, WGPUErrorType_Validation, "render pipeline %.*s: vertex buffers are not supported", nlabel, label); return NULL; }
    WGPURenderPipeline p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    obj_init(&p->o, rpipe_free);
    p->dev = device;
    p->layout = d->layout;
    obj_ref(&d->layout->o);

    char vname[128], fname[128];
    entry_name(vname, sizeof(vname), d->vertex.entryPoint, "main");
    VkPipelineShaderStageCreateInfo st[2];
    memset(st, 0, sizeof(st));
    st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = d->vertex.module->vk;
    st[0].pName = vname;
    uint32_t nst = 1;
    if (d->fragment){
        entry_name(fname, sizeof(fname), d->fragment->entryPoint, "main");
        st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = d->fragment->module->vk;
        st[1].pName = fname;
        nst = 2;
    }

    VkPipelineVertexInputStateCreateInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia;
    memset(&ia, 0, sizeof(ia));
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    switch (d->primitive.topology){
    case WGPUPrimitiveTopology_PointList: ia.topology = 0; break;
    case WGPUPrimitiveTopology_LineList: ia.topology = 1; break;
    case WGPUPrimitiveTopology_LineStrip: ia.topology = 2; break;
    case WGPUPrimitiveTopology_TriangleStrip: ia.topology = 4; break;
    default: ia.topology = 3; break;               // TriangleList
    }
    VkPipelineViewportStateCreateInfo vp;
    memset(&vp, 0, sizeof(vp));
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = vp.scissorCount = 1;        // (dynamic: set at the start of each pass)
    VkPipelineRasterizationStateCreateInfo rs;
    memset(&rs, 0, sizeof(rs));
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = d->primitive.unclippedDepth ? 1 : 0;
    rs.cullMode = d->primitive.cullMode == WGPUCullMode_Front ? 1 : d->primitive.cullMode == WGPUCullMode_Back ? 2 : 0;
    rs.frontFace = d->primitive.frontFace == WGPUFrontFace_CW ? 0 : 1;   // mirrored (see the top)
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms;
    memset(&ms, 0, sizeof(ms));
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = 1;
    uint32_t mask = d->multisample.mask;
    ms.pSampleMask = &mask;
    ms.alphaToCoverageEnable = d->multisample.alphaToCoverageEnabled ? 1 : 0;
    VkPipelineDepthStencilStateCreateInfo ds;
    memset(&ds, 0, sizeof(ds));
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    if (d->depthStencil){
        depth_format = vk_format(d->depthStencil->format, NULL, NULL);
        uint32_t cmp = compare_op(d->depthStencil->depthCompare);
        int write = d->depthStencil->depthWriteEnabled == WGPUOptionalBool_True;
        ds.depthTestEnable = cmp != 7 || write;
        ds.depthWriteEnable = write;
        ds.depthCompareOp = cmp;
        ds.front.compareOp = ds.back.compareOp = 7;
        ds.maxDepthBounds = 1.0f;
        rs.depthBiasEnable = d->depthStencil->depthBias || d->depthStencil->depthBiasSlopeScale != 0.0f;
        rs.depthBiasConstantFactor = (float)d->depthStencil->depthBias;
        rs.depthBiasSlopeFactor = d->depthStencil->depthBiasSlopeScale;
        rs.depthBiasClamp = d->depthStencil->depthBiasClamp;
    }
    VkPipelineColorBlendAttachmentState cba[8];
    VkFormat cfmt[8];
    memset(cba, 0, sizeof(cba));
    uint32_t ntargets = d->fragment ? (uint32_t)d->fragment->targetCount : 0;
    if (ntargets > 8) ntargets = 8;
    for (uint32_t i = 0; i < ntargets; i++){
        const WGPUColorTargetState *t = &d->fragment->targets[i];
        cfmt[i] = vk_format(t->format, NULL, NULL);
        cba[i].colorWriteMask = t->writeMask & 0xF;     // R G B A: the same bits
        if (t->blend){
            cba[i].blendEnable = 1;
            cba[i].srcColorBlendFactor = blend_factor(t->blend->color.srcFactor);
            cba[i].dstColorBlendFactor = blend_factor(t->blend->color.dstFactor);
            cba[i].colorBlendOp = blend_op(t->blend->color.operation);
            cba[i].srcAlphaBlendFactor = blend_factor(t->blend->alpha.srcFactor);
            cba[i].dstAlphaBlendFactor = blend_factor(t->blend->alpha.dstFactor);
            cba[i].alphaBlendOp = blend_op(t->blend->alpha.operation);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb;
    memset(&cb, 0, sizeof(cb));
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = ntargets;
    cb.pAttachments = cba;
    static const uint32_t dyn[] = {0, 1};          // VIEWPORT, SCISSOR
    VkPipelineDynamicStateCreateInfo dy;
    memset(&dy, 0, sizeof(dy));
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri;
    memset(&ri, 0, sizeof(ri));
    ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ri.colorAttachmentCount = ntargets;
    ri.pColorAttachmentFormats = cfmt;
    ri.depthAttachmentFormat = depth_format;

    VkGraphicsPipelineCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext = &ri;
    ci.stageCount = nst;
    ci.pStages = st;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = d->depthStencil ? &ds : NULL;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dy;
    ci.layout = d->layout->vk;
    ci.basePipelineIndex = -1;
    VkResult r = vk.vkCreateGraphicsPipelines(device->vk, 0, 1, &ci, NULL, &p->vk);
    if (r != VK_SUCCESS){
        vk_error(device, WGPUErrorType_Internal, "render pipeline %.*s: vkCreateGraphicsPipelines failed (%d)", nlabel, label, r);
        p->vk = 0;
        rpipe_free(&p->o);
        return NULL;
    }
    return p;
}

void wgpuRenderPipelineRelease(WGPURenderPipeline p){ if (p) obj_unref(&p->o); }

// ---- render passes ----
static void rpass_free(Obj *o){ free(o); }

static uint32_t load_op(WGPULoadOp op){ return op == WGPULoadOp_Clear ? 1 : op == WGPULoadOp_Load ? 0 : 2; }
static uint32_t store_op(WGPUStoreOp op){ return op == WGPUStoreOp_Discard ? 1 : 0; }

WGPURenderPassEncoder wgpuCommandEncoderBeginRenderPass(WGPUCommandEncoder e, WGPURenderPassDescriptor const *d){
    WGPURenderPassEncoder p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    obj_init(&p->o, rpass_free);
    p->enc = e;
    VkRenderingAttachmentInfo ca[8], da;
    memset(ca, 0, sizeof(ca));
    memset(&da, 0, sizeof(da));
    uint32_t n = d->colorAttachmentCount < 8 ? (uint32_t)d->colorAttachmentCount : 8;
    uint32_t w = 0, h = 0;
    for (uint32_t i = 0; i < n; i++){
        const WGPURenderPassColorAttachment *a = &d->colorAttachments[i];
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        if (!a->view) continue;
        refs_add(&e->refs, &a->view->o);
        ca[i].imageView = a->view->vk;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = load_op(a->loadOp);
        ca[i].storeOp = store_op(a->storeOp);
        ca[i].clearValue.color[0] = (float)a->clearValue.r;
        ca[i].clearValue.color[1] = (float)a->clearValue.g;
        ca[i].clearValue.color[2] = (float)a->clearValue.b;
        ca[i].clearValue.color[3] = (float)a->clearValue.a;
        if (!w){ w = a->view->tex->w; h = a->view->tex->h; }
    }
    const WGPURenderPassDepthStencilAttachment *dsa = d->depthStencilAttachment;
    if (dsa && dsa->view){
        refs_add(&e->refs, &dsa->view->o);
        da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        da.imageView = dsa->view->vk;
        da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.loadOp = dsa->depthReadOnly ? 0 : load_op(dsa->depthLoadOp);
        da.storeOp = dsa->depthReadOnly ? 0 : store_op(dsa->depthStoreOp);
        da.clearValue.depthStencil.depth = dsa->depthClearValue;
        if (!w){ w = dsa->view->tex->w; h = dsa->view->tex->h; }
    }
    vk_before(e, VK_CMD_RENDER);
    VkRenderingInfo ri;
    memset(&ri, 0, sizeof(ri));
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent.width = w;
    ri.renderArea.extent.height = h;
    ri.layerCount = 1;
    ri.colorAttachmentCount = n;
    ri.pColorAttachments = ca;
    ri.pDepthAttachment = dsa && dsa->view ? &da : NULL;
    vk.vkCmdBeginRendering(e->cb, &ri);
    VkViewport v = {0, 0, (float)w, (float)h, 0, 1};
    vk.vkCmdSetViewport(e->cb, 0, 1, &v);
    VkRect2D sc = {{0, 0}, {w, h}};
    vk.vkCmdSetScissor(e->cb, 0, 1, &sc);
    return p;
}

void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder p, WGPURenderPipeline pipe){
    if (p->pipe != pipe){ p->pipe = pipe; p->pipe_bound = 0; p->bg_dirty = 1; }
    refs_add(&p->enc->refs, &pipe->o);
}

void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder p, uint32_t index, WGPUBindGroup group,
                                       size_t n, uint32_t const *offsets){
    if (index >= 4) return;
    p->bg[index] = group;
    p->ndyn[index] = (uint32_t)(n < 8 ? n : 8);
    for (uint32_t i = 0; i < p->ndyn[index]; i++) p->dyn[index][i] = offsets[i];
    p->bg_dirty = 1;
    if (group) refs_add(&p->enc->refs, &group->o);
}

static int flush(WGPURenderPassEncoder p){
    if (!p->pipe) return 0;
    VkCommandBuffer cb = p->enc->cb;
    if (!p->pipe_bound){ vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p->pipe->vk); p->pipe_bound = 1; }
    if (p->bg_dirty){
        WGPUPipelineLayout pl = p->pipe->layout;
        for (int i = 0; i < 4 && i < pl->n; i++){
            if (!p->bg[i]) continue;
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl->vk, (uint32_t)i, 1, &p->bg[i]->set,
                                       p->ndyn[i], p->dyn[i]);
        }
        p->bg_dirty = 0;
    }
    return 1;
}

void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder p, uint32_t nverts, uint32_t ninst, uint32_t first_vert, uint32_t first_inst){
    if (!nverts || !ninst || !flush(p)) return;
    vk.vkCmdDraw(p->enc->cb, nverts, ninst, first_vert, first_inst);
}

void wgpuRenderPassEncoderEnd(WGPURenderPassEncoder p){
    vk.vkCmdEndRendering(p->enc->cb);
}

void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder p){ if (p) obj_unref(&p->o); }
