// vk_tex.c - textures, their views, writes into them and copies out of them (vk_impl.h).
// Every image lives in the GENERAL layout: moved there once, at creation, in the upload
// command buffer (so before anything that uses it), and never moved again. textureLoad,
// attachments and copies all work in GENERAL; it costs a little bandwidth on some GPUs, and
// saves tracking layouts. A window's images are the exception (vk_surface.c).
#include "gpu/vk/vk_impl.h"

#include <stdlib.h>
#include <string.h>

VkFormat vk_format(WGPUTextureFormat f, uint32_t *texel, VkFlags *aspect){
    uint32_t t = 4;
    VkFlags a = VK_IMAGE_ASPECT_COLOR_BIT;
    VkFormat v;
    switch (f){
    case WGPUTextureFormat_RGBA8Unorm: v = VK_FORMAT_R8G8B8A8_UNORM; break;
    case WGPUTextureFormat_RGBA8UnormSrgb: v = VK_FORMAT_R8G8B8A8_SRGB; break;
    case WGPUTextureFormat_BGRA8Unorm: v = VK_FORMAT_B8G8R8A8_UNORM; break;
    case WGPUTextureFormat_BGRA8UnormSrgb: v = VK_FORMAT_B8G8R8A8_SRGB; break;
    case WGPUTextureFormat_RGBA16Float: v = VK_FORMAT_R16G16B16A16_SFLOAT; t = 8; break;
    case WGPUTextureFormat_R32Uint: v = VK_FORMAT_R32_UINT; break;
    case WGPUTextureFormat_R32Float: v = VK_FORMAT_R32_SFLOAT; break;
    case WGPUTextureFormat_RG32Float: v = VK_FORMAT_R32G32_SFLOAT; t = 8; break;
    case WGPUTextureFormat_RGBA32Float: v = VK_FORMAT_R32G32B32A32_SFLOAT; t = 16; break;
    case WGPUTextureFormat_Depth32Float: v = VK_FORMAT_D32_SFLOAT; a = VK_IMAGE_ASPECT_DEPTH_BIT; break;
    default: v = VK_FORMAT_UNDEFINED; break;
    }
    if (texel) *texel = t;
    if (aspect) *aspect = a;
    return v;
}

WGPUTextureFormat vk_wgpu_format(VkFormat f){
    switch (f){
    case VK_FORMAT_R8G8B8A8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
    case VK_FORMAT_B8G8R8A8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return WGPUTextureFormat_RGBA16Float;
    default: return WGPUTextureFormat_Undefined;
    }
}

// the image from `from` (UNDEFINED: its contents may go) into `to`, after everything before
void vk_image_layout(VkCommandBuffer cb, VkImage img, VkFlags aspect, uint32_t from, uint32_t to){
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.levelCount = ~0u;          // VK_REMAINING_MIP_LEVELS
    b.subresourceRange.layerCount = ~0u;          // VK_REMAINING_ARRAY_LAYERS
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

void vk_texture_ready(WGPUDevice dev, VkCommandBuffer cb, WGPUTexture t){
    (void)dev;
    if (t->ready) return;
    vk_image_layout(cb, t->img, t->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    t->ready = 1;
}

// ---- textures ----
static void texture_free(Obj *o){
    WGPUTexture t = (WGPUTexture)o;
    if (t->mem){
        if (t->img) vk.vkDestroyImage(t->dev->vk, t->img, NULL);
        vk.vkFreeMemory(t->dev->vk, t->mem, NULL);
    }
    free(t);
}

WGPUTexture wgpuDeviceCreateTexture(WGPUDevice device, WGPUTextureDescriptor const *d){
    uint32_t texel; VkFlags aspect;
    VkFormat f = vk_format(d->format, &texel, &aspect);
    if (f == VK_FORMAT_UNDEFINED){ vk_error(device, WGPUErrorType_Validation, "texture format %d not supported", (int)d->format); return NULL; }
    if (d->dimension != WGPUTextureDimension_2D && d->dimension != WGPUTextureDimension_Undefined){
        vk_error(device, WGPUErrorType_Validation, "only 2D textures are supported");
        return NULL;
    }
    WGPUTexture t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    obj_init(&t->o, texture_free);
    t->dev = device;
    t->format = d->format;
    t->vkformat = f;
    t->texel = texel;
    t->aspect = aspect;
    t->usage = d->usage;
    t->w = d->size.width;
    t->h = d->size.height;
    t->layers = d->size.depthOrArrayLayers ? d->size.depthOrArrayLayers : 1;
    t->mips = d->mipLevelCount ? d->mipLevelCount : 1;
    VkImageCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = 1;                              // VK_IMAGE_TYPE_2D
    ci.format = f;
    ci.extent = (VkExtent3D){t->w, t->h, 1};
    ci.mipLevels = t->mips;
    ci.arrayLayers = t->layers;
    ci.samples = 1;
    ci.tiling = 0;                                 // VK_IMAGE_TILING_OPTIMAL
    VkFlags u = 0;
    if (d->usage & WGPUTextureUsage_CopySrc) u |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (d->usage & WGPUTextureUsage_CopyDst) u |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (d->usage & WGPUTextureUsage_TextureBinding) u |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (d->usage & WGPUTextureUsage_StorageBinding) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (d->usage & WGPUTextureUsage_RenderAttachment)
        u |= aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.usage = u;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vk.vkCreateImage(device->vk, &ci, NULL, &t->img) != VK_SUCCESS){
        vk_error(device, WGPUErrorType_OutOfMemory, "vkCreateImage failed (%ux%u)", t->w, t->h);
        free(t);
        return NULL;
    }
    VkMemoryRequirements req;
    vk.vkGetImageMemoryRequirements(device->vk, t->img, &req);
    int type = vk_memory_type(device->ad, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    if (type < 0 || vk.vkAllocateMemory(device->vk, &ai, NULL, &t->mem) != VK_SUCCESS){
        vk.vkDestroyImage(device->vk, t->img, NULL);
        vk_error(device, WGPUErrorType_OutOfMemory, "texture of %ux%u: no memory", t->w, t->h);
        free(t);
        return NULL;
    }
    vk.vkBindImageMemory(device->vk, t->img, t->mem, 0);
    VkCommandBuffer cb = vk_upload_cb(VK_MAIN(device));
    if (cb) vk_texture_ready(device, cb, t);
    return t;
}

void wgpuTextureAddRef(WGPUTexture t){ if (t) obj_ref(&t->o); }
void wgpuTextureDestroy(WGPUTexture t){ (void)t; }        // freed with its last reference
void wgpuTextureRelease(WGPUTexture t){ if (t) obj_unref(&t->o); }

// ---- views ----
static void view_free(Obj *o){
    WGPUTextureView v = (WGPUTextureView)o;
    if (v->vk) vk.vkDestroyImageView(v->tex->dev->vk, v->vk, NULL);
    obj_unref(&v->tex->o);
    free(v);
}

WGPUTextureView wgpuTextureCreateView(WGPUTexture t, WGPUTextureViewDescriptor const *d){
    if (!t) return NULL;
    WGPUTextureView v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    obj_init(&v->o, view_free);
    v->tex = t;
    obj_ref(&t->o);
    VkImageViewCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image = t->img;
    ci.viewType = 1;                               // VK_IMAGE_VIEW_TYPE_2D
    ci.format = t->vkformat;
    if (d && d->format != WGPUTextureFormat_Undefined){
        VkFormat f = vk_format(d->format, NULL, NULL);
        if (f != VK_FORMAT_UNDEFINED) ci.format = f;
    }
    ci.subresourceRange.aspectMask = t->aspect;
    ci.subresourceRange.baseMipLevel = d ? d->baseMipLevel : 0;
    ci.subresourceRange.levelCount = d && d->mipLevelCount != WGPU_MIP_LEVEL_COUNT_UNDEFINED && d->mipLevelCount ? d->mipLevelCount : t->mips - ci.subresourceRange.baseMipLevel;
    ci.subresourceRange.baseArrayLayer = d ? d->baseArrayLayer : 0;
    ci.subresourceRange.layerCount = 1;
    if (vk.vkCreateImageView(t->dev->vk, &ci, NULL, &v->vk) != VK_SUCCESS){
        vk_error(t->dev, WGPUErrorType_Internal, "vkCreateImageView failed");
        v->vk = 0;
        view_free(&v->o);
        return NULL;
    }
    return v;
}

void wgpuTextureViewAddRef(WGPUTextureView v){ if (v) obj_ref(&v->o); }
void wgpuTextureViewRelease(WGPUTextureView v){ if (v) obj_unref(&v->o); }

// ---- writes and copies ----
static void region(VkBufferImageCopy *r, WGPUTexture t, VkDeviceSize off, uint32_t bytes_per_row, uint32_t rows_per_image,
                   WGPUTexelCopyTextureInfo const *at, WGPUExtent3D const *ext){
    memset(r, 0, sizeof(*r));
    r->bufferOffset = off;
    r->bufferRowLength = bytes_per_row && bytes_per_row != WGPU_COPY_STRIDE_UNDEFINED ? bytes_per_row / t->texel : 0;
    r->bufferImageHeight = rows_per_image != WGPU_COPY_STRIDE_UNDEFINED ? rows_per_image : 0;
    r->imageSubresource.aspectMask = t->aspect;
    r->imageSubresource.mipLevel = at->mipLevel;
    r->imageSubresource.baseArrayLayer = at->origin.z;
    r->imageSubresource.layerCount = ext->depthOrArrayLayers ? ext->depthOrArrayLayers : 1;
    r->imageOffset = (VkOffset3D){(int32_t)at->origin.x, (int32_t)at->origin.y, 0};
    r->imageExtent = (VkExtent3D){ext->width, ext->height, 1};
}

void wgpuQueueWriteTexture(WGPUQueue q, WGPUTexelCopyTextureInfo const *dst, void const *data, size_t size,
                           WGPUTexelCopyBufferLayout const *layout, WGPUExtent3D const *ext){
    WGPUDevice dev = q->dev;
    WGPUTexture t = dst->texture;
    if (!t || !ext->width || !ext->height) return;
    uint32_t bpr = layout->bytesPerRow != WGPU_COPY_STRIDE_UNDEFINED && layout->bytesPerRow ? layout->bytesPerRow : ext->width * t->texel;
    uint32_t rows = layout->rowsPerImage != WGPU_COPY_STRIDE_UNDEFINED && layout->rowsPerImage ? layout->rowsPerImage : ext->height;
    uint32_t layers = ext->depthOrArrayLayers ? ext->depthOrArrayLayers : 1;
    // the bytes the copy reads: whole rows but the last's tail
    size_t need = (size_t)bpr * rows * (layers - 1) + (size_t)bpr * (ext->height - 1) + (size_t)ext->width * t->texel;
    if (layout->offset + need > size){ vk_error(dev, WGPUErrorType_Validation, "wgpuQueueWriteTexture: data too small"); return; }
    if (bpr % t->texel){ vk_error(dev, WGPUErrorType_Validation, "wgpuQueueWriteTexture: bytesPerRow not a whole number of texels"); return; }
    VkCommandBuffer cb = vk_upload_cb(q);
    if (!cb) return;
    VkBuffer sb; VkDeviceSize so; unsigned char *p;
    if (!vk_staging(q, need, &sb, &so, &p)){ vk_error(dev, WGPUErrorType_OutOfMemory, "no staging memory"); return; }
    memcpy(p, (const unsigned char *)data + layout->offset, need);
    refs_add(&q->up_refs, &t->o);
    vk_texture_ready(dev, cb, t);
    VkBufferImageCopy r;
    region(&r, t, so, bpr, rows, dst, ext);
    vk.vkCmdCopyBufferToImage(cb, sb, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

void wgpuCommandEncoderCopyTextureToBuffer(WGPUCommandEncoder e, WGPUTexelCopyTextureInfo const *src,
                                           WGPUTexelCopyBufferInfo const *dst, WGPUExtent3D const *size){
    WGPUTexture t = src->texture;
    if (!t || !size->width || !size->height) return;
    vk_before(e, VK_CMD_TRANSFER);
    refs_add(&e->refs, &t->o);
    refs_add(&e->refs, &dst->buffer->o);
    VkBufferImageCopy r;
    region(&r, t, dst->layout.offset, dst->layout.bytesPerRow, dst->layout.rowsPerImage, src, size);
    vk.vkCmdCopyImageToBuffer(e->cb, t->img, VK_IMAGE_LAYOUT_GENERAL, dst->buffer->vk, 1, &r);
}
