// image.h - the two pixel buffers of the engine.
//   Framebuffer: 8-bit color (0x00RRGGBB) + a depth buffer. The rasterizer draws into it,
//                windows show it, and it saves as PNG.
//   Image:       linear floating-point color, what the path tracer accumulates. Converted
//                to sRGB only when written or shown.
#ifndef LR_IMAGE_H
#define LR_IMAGE_H

#include <stdint.h>
#include "core/lrmath.h"

// ---- Framebuffer --------------------------------------------------------------------
// Pixel (x, y) lives at color[y * w + x]; (0, 0) is the TOP-left corner.

typedef struct {
    int w, h;
    uint32_t *color;   // w*h pixels, 0x00RRGGBB
    real *depth;       // w*h depth values, 0 = near, 1 = far (the z-buffer)
} Framebuffer;

static inline uint32_t rgb8(int r, int g, int b){
    return ((uint32_t)(r & 255) << 16) | ((uint32_t)(g & 255) << 8) | (uint32_t)(b & 255);
}

// colors given in [0, 1] (values outside are clamped), written as they are (no sRGB curve)
static inline uint32_t rgbf(real r, real g, real b){
    r = r < 0 ? 0 : (r > 1 ? 1 : r);
    g = g < 0 ? 0 : (g > 1 ? 1 : g);
    b = b < 0 ? 0 : (b > 1 ? 1 : b);
    return rgb8((int)(r * 255.0f + 0.5f), (int)(g * 255.0f + 0.5f), (int)(b * 255.0f + 0.5f));
}

Framebuffer *fb_create(int w, int h);
void fb_destroy(Framebuffer *fb);
void fb_clear(Framebuffer *fb, uint32_t color);                   // also resets the z-buffer
void fb_set_pixel(Framebuffer *fb, int x, int y, uint32_t color);  // bounds checked
void fb_show_depth(Framebuffer *fb);                               // debug: colors := z-buffer
int fb_save_png(const Framebuffer *fb, const char *path);          // RGB PNG (core/png.h), 0 on success

// ---- Image (linear color) -----------------------------------------------------------

typedef struct {
    int w, h;
    Vec3 *px;   // linear radiance, row 0 = top
} Image;

Image image_create(int w, int h);
void image_free(Image *img);
// clamps to [0,1], applies the sRGB curve and writes an RGB PNG. 0 on success.
int image_write_png(const Image *img, const char *path);

// ---- linear color -> display -------------------------------------------------------

// one linear channel value -> 0..255 sRGB byte (clamped, NaN -> 0). Exact, uses pow().
unsigned char image_srgb8(double linear);

// Fast version for whole frames: a 4096-entry lookup table. Call srgb_lut_init() once
// before using image_pack_srgb (and before starting threads that use it).
void srgb_lut_init(void);
uint32_t image_pack_srgb(Vec3 linear);   // -> 0x00RRGGBB, sRGB-encoded

#endif
