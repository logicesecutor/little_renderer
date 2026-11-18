// vk_surface.c - a window's surface (vk_impl.h): a Win32 surface and its swapchain.
//
// The images keep to the layer's rule while the engine has them: wgpuSurfaceGetCurrentTexture
// acquires one, waits on the CPU until it is free (a fence: so the GPU's queue never stalls
// on the display, and the wait shows where wgpu's does, in getting the image), and moves it
// to GENERAL in the upload command buffer; wgpuSurfacePresent moves it to PRESENT_SRC in a
// small submission of its own that also signals the image's "presentable" semaphore, which
// the present waits for.
#include "vk/vk_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

static void say_once(const char *what){
    static int said;
    if (!said++) fprintf(stderr, "Vulkan layer: %s\n", what);
}

static void release_images(WGPUSurface s){
    for (uint32_t i = 0; i < s->n; i++){
        if (s->tex[i]){ obj_unref(&s->tex[i]->o); s->tex[i] = NULL; }
    }
    s->n = 0;
}

static void destroy_swapchain(WGPUSurface s){
    if (!s->dev) return;
    VkDevice d = s->dev->vk;
    vk.vkDeviceWaitIdle(d);
    lrDevicePoll(s->dev, 0, NULL);           // the finished submissions given back
    release_images(s);
    if (s->sc){ vk.vkDestroySwapchainKHR(d, s->sc, NULL); s->sc = 0; }
    if (s->acquired){ vk.vkDestroyFence(d, s->acquired, NULL); s->acquired = 0; }
    for (int i = 0; i < VK_SURFACE_IMAGES; i++)
        if (s->presentable[i]){ vk.vkDestroySemaphore(d, s->presentable[i], NULL); s->presentable[i] = 0; }
    s->cur = -1;
}

static void surface_free(Obj *o){
    WGPUSurface s = (WGPUSurface)o;
    destroy_swapchain(s);
    if (s->dev) obj_unref(&s->dev->o);
    if (s->vk) vk.vkDestroySurfaceKHR(s->inst->vk, s->vk, NULL);
    obj_unref(&s->inst->o);
    free(s);
}

WGPUSurface wgpuInstanceCreateSurface(WGPUInstance instance, WGPUSurfaceDescriptor const *d){
#ifdef _WIN32
    const WGPUSurfaceSourceWindowsHWND *src = NULL;
    for (const WGPUChainedStruct *c = d->nextInChain; c; c = c->next)
        if (c->sType == WGPUSType_SurfaceSourceWindowsHWND) src = (const WGPUSurfaceSourceWindowsHWND *)c;
    if (!src || !src->hwnd){ say_once("a surface needs a window (WGPUSurfaceSourceWindowsHWND)"); return NULL; }
    if (!vk.vkCreateWin32SurfaceKHR || !vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR){ say_once("no window surfaces (VK_KHR_win32_surface)"); return NULL; }
    WGPUSurface s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    obj_init(&s->o, surface_free);
    s->inst = instance;
    obj_ref(&instance->o);
    s->cur = -1;
    VkWin32SurfaceCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    ci.hinstance = src->hinstance ? src->hinstance : (void *)GetModuleHandleA(NULL);
    ci.hwnd = src->hwnd;
    VkResult r = vk.vkCreateWin32SurfaceKHR(instance->vk, &ci, NULL, &s->vk);
    if (r != VK_SUCCESS){
        fprintf(stderr, "Vulkan layer: vkCreateWin32SurfaceKHR failed (%d)\n", r);
        s->vk = 0;
        surface_free(&s->o);
        return NULL;
    }
    return s;
#else
    (void)instance; (void)d;
    say_once("window surfaces are Windows only for now: drawing offscreen");
    return NULL;
#endif
}

void wgpuSurfaceRelease(WGPUSurface s){ if (s) obj_unref(&s->o); }

// ---- capabilities ----
static WGPUPresentMode wgpu_present_mode(uint32_t m){
    switch (m){
    case VK_PRESENT_MODE_IMMEDIATE_KHR: return WGPUPresentMode_Immediate;
    case VK_PRESENT_MODE_MAILBOX_KHR: return WGPUPresentMode_Mailbox;
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return WGPUPresentMode_FifoRelaxed;
    case VK_PRESENT_MODE_FIFO_KHR: return WGPUPresentMode_Fifo;
    default: return WGPUPresentMode_Undefined;
    }
}

WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface s, WGPUAdapter a, WGPUSurfaceCapabilities *c){
    WGPUChainedStruct *next = c->nextInChain;
    memset(c, 0, sizeof(*c));
    c->nextInChain = next;
    VkBool32 ok = 0;
    if (vk.vkGetPhysicalDeviceSurfaceSupportKHR(a->pd, a->family, s->vk, &ok) != VK_SUCCESS || !ok) return WGPUStatus_Error;
    VkSurfaceCapabilitiesKHR caps;
    if (vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(a->pd, s->vk, &caps) != VK_SUCCESS) return WGPUStatus_Error;
    c->usages = WGPUTextureUsage_RenderAttachment |
                ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? WGPUTextureUsage_CopySrc : 0) |
                ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ? WGPUTextureUsage_CopyDst : 0);
    // formats: the sRGB colour space's, the ones WebGPU names
    uint32_t n = 0;
    vk.vkGetPhysicalDeviceSurfaceFormatsKHR(a->pd, s->vk, &n, NULL);
    VkSurfaceFormatKHR *f = calloc(n ? n : 1, sizeof(*f));
    WGPUTextureFormat *wf = calloc(n ? n : 1, sizeof(*wf));
    size_t nf = 0;
    if (f && wf && vk.vkGetPhysicalDeviceSurfaceFormatsKHR(a->pd, s->vk, &n, f) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++){
            WGPUTextureFormat w = vk_wgpu_format(f[i].format);
            if (f[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && w != WGPUTextureFormat_Undefined) wf[nf++] = w;
        }
    free(f);
    c->formatCount = nf;
    c->formats = wf;
    // present modes
    n = 0;
    vk.vkGetPhysicalDeviceSurfacePresentModesKHR(a->pd, s->vk, &n, NULL);
    uint32_t *m = calloc(n ? n : 1, sizeof(*m));
    WGPUPresentMode *wm = calloc(n ? n : 1, sizeof(*wm));
    size_t nm = 0;
    if (m && wm && vk.vkGetPhysicalDeviceSurfacePresentModesKHR(a->pd, s->vk, &n, m) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++)
            if (wgpu_present_mode(m[i]) != WGPUPresentMode_Undefined) wm[nm++] = wgpu_present_mode(m[i]);
    free(m);
    c->presentModeCount = nm;
    c->presentModes = wm;
    // alpha modes
    WGPUCompositeAlphaMode *am = calloc(4, sizeof(*am));
    size_t na = 0;
    if (am){
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) am[na++] = WGPUCompositeAlphaMode_Opaque;
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) am[na++] = WGPUCompositeAlphaMode_Premultiplied;
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) am[na++] = WGPUCompositeAlphaMode_Unpremultiplied;
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) am[na++] = WGPUCompositeAlphaMode_Inherit;
    }
    c->alphaModeCount = na;
    c->alphaModes = am;
    return WGPUStatus_Success;
}

void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities c){
    free((void *)c.formats);
    free((void *)c.presentModes);
    free((void *)c.alphaModes);
}

// ---- the swapchain ----
// (the image is the swapchain's, not ours: only the record goes)
static void swap_texture_free(Obj *o){ free(o); }

static int make_swapchain(WGPUSurface s){
    WGPUDevice dev = s->dev;
    WGPUAdapter a = dev->ad;
    if (!vk.vkCreateSwapchainKHR){ say_once("the device has no swapchains (VK_KHR_swapchain)"); return 0; }
    VkBool32 ok = 0;
    vk.vkGetPhysicalDeviceSurfaceSupportKHR(a->pd, a->family, s->vk, &ok);
    if (!ok){ say_once("the GPU's queue cannot present to this window"); return 0; }
    VkSurfaceCapabilitiesKHR caps;
    if (vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(a->pd, s->vk, &caps) != VK_SUCCESS) return 0;
    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu){            // the window takes the swapchain's size
        ext.width = s->cfg.width; ext.height = s->cfg.height;
        if (ext.width < caps.minImageExtent.width) ext.width = caps.minImageExtent.width;
        if (ext.width > caps.maxImageExtent.width) ext.width = caps.maxImageExtent.width;
        if (ext.height < caps.minImageExtent.height) ext.height = caps.minImageExtent.height;
        if (ext.height > caps.maxImageExtent.height) ext.height = caps.maxImageExtent.height;
    }
    if (!ext.width || !ext.height) return 0;   // minimized
    uint32_t count = caps.minImageCount + 1;
    if (count < 3) count = 3;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > VK_SURFACE_IMAGES) count = VK_SURFACE_IMAGES;
    uint32_t texel; VkFlags aspect;
    VkFormat fmt = vk_format(s->cfg.format, &texel, &aspect);
    uint32_t mode = VK_PRESENT_MODE_FIFO_KHR;
    if (s->cfg.presentMode == WGPUPresentMode_Immediate) mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else if (s->cfg.presentMode == WGPUPresentMode_Mailbox) mode = VK_PRESENT_MODE_MAILBOX_KHR;
    else if (s->cfg.presentMode == WGPUPresentMode_FifoRelaxed) mode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    if (mode != VK_PRESENT_MODE_FIFO_KHR){       // only if the window has it (FIFO it always has)
        uint32_t n = 0, m[16], found = 0;
        vk.vkGetPhysicalDeviceSurfacePresentModesKHR(a->pd, s->vk, &n, NULL);
        if (n > 16) n = 16;
        vk.vkGetPhysicalDeviceSurfacePresentModesKHR(a->pd, s->vk, &n, m);
        for (uint32_t i = 0; i < n; i++) if (m[i] == mode) found = 1;
        if (!found) mode = VK_PRESENT_MODE_FIFO_KHR;
    }
    uint32_t alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (s->cfg.alphaMode == WGPUCompositeAlphaMode_Premultiplied) alpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    else if (s->cfg.alphaMode == WGPUCompositeAlphaMode_Unpremultiplied) alpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
    else if (s->cfg.alphaMode == WGPUCompositeAlphaMode_Inherit) alpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & alpha))
        for (alpha = 1; alpha <= 8 && !(caps.supportedCompositeAlpha & alpha); alpha <<= 1){}
    VkFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (s->cfg.usage & WGPUTextureUsage_CopySrc) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (s->cfg.usage & WGPUTextureUsage_CopyDst) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    usage &= caps.supportedUsageFlags | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkSwapchainKHR old = s->sc;
    VkSwapchainCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = s->vk;
    ci.minImageCount = count;
    ci.imageFormat = fmt;
    ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = usage;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = alpha;
    ci.presentMode = mode;
    ci.clipped = 1;
    ci.oldSwapchain = old;
    VkSwapchainKHR sc = 0;
    VkResult r = vk.vkCreateSwapchainKHR(dev->vk, &ci, NULL, &sc);
    if (old) destroy_swapchain(s);              // the old one goes (its images with it), whatever happened
    if (r != VK_SUCCESS){ fprintf(stderr, "Vulkan layer: vkCreateSwapchainKHR failed (%d)\n", r); return 0; }
    s->sc = sc;
    VkImage imgs[VK_SURFACE_IMAGES];
    uint32_t n = VK_SURFACE_IMAGES;
    vk.vkGetSwapchainImagesKHR(dev->vk, sc, &n, imgs);
    s->n = n;
    VkSemaphoreCreateInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t i = 0; i < n; i++){
        WGPUTexture t = calloc(1, sizeof(*t));
        if (!t) return 0;
        obj_init(&t->o, swap_texture_free);
        t->dev = dev;
        t->img = imgs[i];
        t->format = s->cfg.format;
        t->vkformat = fmt;
        t->texel = texel;
        t->aspect = aspect;
        t->usage = s->cfg.usage;
        t->w = ext.width; t->h = ext.height;
        t->layers = t->mips = 1;
        s->tex[i] = t;
        vk.vkCreateSemaphore(dev->vk, &si, NULL, &s->presentable[i]);
    }
    VkFenceCreateInfo fi;
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk.vkCreateFence(dev->vk, &fi, NULL, &s->acquired);
    s->stale = 0;
    s->cur = -1;
    return 1;
}

void wgpuSurfaceConfigure(WGPUSurface s, WGPUSurfaceConfiguration const *cfg){
    if (s->dev && s->dev != cfg->device){ destroy_swapchain(s); obj_unref(&s->dev->o); s->dev = NULL; }
    if (!s->dev){ s->dev = cfg->device; obj_ref(&s->dev->o); }
    s->cfg = *cfg;
    s->cfg.nextInChain = NULL;
    s->cfg.viewFormatCount = 0;
    s->cfg.viewFormats = NULL;
    if (vk_format(cfg->format, NULL, NULL) == VK_FORMAT_UNDEFINED){
        vk_error(s->dev, WGPUErrorType_Validation, "surface format %d not supported", (int)cfg->format);
        return;
    }
    if (!make_swapchain(s)) s->stale = 1;
}

void wgpuSurfaceGetCurrentTexture(WGPUSurface s, WGPUSurfaceTexture *st){
    WGPUChainedStruct *next = st->nextInChain;
    memset(st, 0, sizeof(*st));
    st->nextInChain = next;
    st->status = WGPUSurfaceGetCurrentTextureStatus_Error;
    if (!s->dev) return;
    WGPUDevice dev = s->dev;
    if (dev->lost){ st->status = WGPUSurfaceGetCurrentTextureStatus_Lost; return; }
    if (s->cur >= 0){                              // not presented: the same one again
        obj_ref(&s->tex[s->cur]->o);
        st->texture = s->tex[s->cur];
        st->status = WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal;
        return;
    }
    for (int attempt = 0; attempt < 2; attempt++){
        if ((s->stale || !s->sc) && !make_swapchain(s)){ st->status = WGPUSurfaceGetCurrentTextureStatus_Outdated; return; }
        uint32_t idx = 0;
        VkResult r = vk.vkAcquireNextImageKHR(dev->vk, s->sc, ~0ull, 0, s->acquired, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR){ s->stale = 1; continue; }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR){
            vk_check_lost(dev, "vkAcquireNextImageKHR", r);
            st->status = r == VK_TIMEOUT ? WGPUSurfaceGetCurrentTextureStatus_Timeout : WGPUSurfaceGetCurrentTextureStatus_Lost;
            return;
        }
        vk_check_lost(dev, "vkWaitForFences", vk.vkWaitForFences(dev->vk, 1, &s->acquired, 1, ~0ull));
        vk.vkResetFences(dev->vk, 1, &s->acquired);
        // what it held is not kept (the engine clears it): from UNDEFINED, into GENERAL
        WGPUTexture t = s->tex[idx];
        VkCommandBuffer cb = vk_upload_cb(VK_MAIN(dev));
        if (cb) vk_image_layout(cb, t->img, t->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        t->ready = 1;
        s->cur = (int)idx;
        obj_ref(&t->o);
        st->texture = t;
        st->status = r == VK_SUBOPTIMAL_KHR ? WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
                                            : WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal;
        return;
    }
    st->status = WGPUSurfaceGetCurrentTextureStatus_Outdated;
}

WGPUStatus wgpuSurfacePresent(WGPUSurface s){
    if (!s->dev || s->cur < 0) return WGPUStatus_Error;
    WGPUDevice dev = s->dev;
    uint32_t idx = (uint32_t)s->cur;
    s->cur = -1;
    VkCommandBuffer cb = vk_cb_get(VK_MAIN(dev));
    if (!cb) return WGPUStatus_Error;
    vk_image_layout(cb, s->tex[idx]->img, s->tex[idx]->aspect, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vk.vkEndCommandBuffer(cb);
    vk_submit_cb(dev, cb, s->presentable[idx]);
    if (dev->lost) return WGPUStatus_Error;        // (not submitted: nothing will signal the image)
    VkPresentInfoKHR pi;
    memset(&pi, 0, sizeof(pi));
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &s->presentable[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &s->sc;
    pi.pImageIndices = &idx;
    VkResult r = vk.vkQueuePresentKHR(VK_MAIN(dev)->q, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) s->stale = 1;
    else if (r != VK_SUCCESS){ vk_check_lost(dev, "vkQueuePresentKHR", r); return WGPUStatus_Error; }
    return WGPUStatus_Success;
}
