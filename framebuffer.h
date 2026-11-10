// framebuffer.h - the pixel array everything is drawn into.
// Pixel (x, y) lives at color[y * w + x]. (0, 0) is the TOP-left corner.
// Colors are 0x00RRGGBB, which is also what X11 expects on a normal 24-bit display.
#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <stdint.h>

typedef struct {
    int w, h;
    uint32_t *color;   // w*h pixels
    float *depth;      // w*h depth values, 0 = near, 1 = far (the z-buffer)
} Framebuffer;

static inline uint32_t rgb8(int r, int g, int b){
    return ((uint32_t)(r & 255) << 16) | ((uint32_t)(g & 255) << 8) | (uint32_t)(b & 255);
}

// colors given as floats in [0, 1] (values outside are clamped)
static inline uint32_t rgbf(float r, float g, float b){
    r = r < 0 ? 0 : (r > 1 ? 1 : r);
    g = g < 0 ? 0 : (g > 1 ? 1 : g);
    b = b < 0 ? 0 : (b > 1 ? 1 : b);
    return rgb8((int)(r * 255.0f + 0.5f), (int)(g * 255.0f + 0.5f), (int)(b * 255.0f + 0.5f));
}

Framebuffer *fb_create(int w, int h);
void fb_destroy(Framebuffer *fb);
void fb_clear(Framebuffer *fb, uint32_t color);              // also resets the z-buffer
void fb_set_pixel(Framebuffer *fb, int x, int y, uint32_t color);  // bounds checked
void fb_show_depth(Framebuffer *fb);                          // debug: overwrite colors with the z-buffer
int fb_save_ppm(const Framebuffer *fb, const char *path);     // binary PPM (P6), returns 0 on success

#endif
