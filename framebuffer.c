#include "framebuffer.h"

#include <stdio.h>
#include <stdlib.h>

Framebuffer *fb_create(int w, int h){
    Framebuffer *fb = malloc(sizeof(*fb));
    if (!fb) return NULL;
    fb->w = w;
    fb->h = h;
    fb->color = calloc((size_t)w * h, sizeof(uint32_t));
    fb->depth = malloc((size_t)w * h * sizeof(float));
    if (!fb->color || !fb->depth){
        fb_destroy(fb);
        return NULL;
    }
    fb_clear(fb, 0);
    return fb;
}

void fb_destroy(Framebuffer *fb){
    if (!fb) return;
    free(fb->color);
    free(fb->depth);
    free(fb);
}

void fb_clear(Framebuffer *fb, uint32_t color){
    size_t n = (size_t)fb->w * fb->h;
    for (size_t i = 0; i < n; i++){
        fb->color[i] = color;
        fb->depth[i] = 1.0f;   // 1.0 = farthest possible
    }
}

void fb_set_pixel(Framebuffer *fb, int x, int y, uint32_t color){
    if (x < 0 || x >= fb->w || y < 0 || y >= fb->h) return;
    fb->color[(size_t)y * fb->w + x] = color;
}

// Turns the z-buffer into a grayscale image (near = bright). Perspective depth is very
// non-linear (almost everything sits close to 1.0), so we stretch min..max to see something.
void fb_show_depth(Framebuffer *fb){
    size_t n = (size_t)fb->w * fb->h;
    float lo = 1.0f, hi = 0.0f;
    for (size_t i = 0; i < n; i++){
        float d = fb->depth[i];
        if (d >= 1.0f) continue;   // nothing was drawn here
        if (d < lo) lo = d;
        if (d > hi) hi = d;
    }
    float range = hi - lo > 1e-9f ? hi - lo : 1.0f;
    for (size_t i = 0; i < n; i++){
        float d = fb->depth[i];
        if (d >= 1.0f){ fb->color[i] = 0; continue; }
        float t = 1.0f - (d - lo) / range;          // near -> 1, far -> 0
        fb->color[i] = rgbf(0.1f + 0.9f * t, 0.1f + 0.9f * t, 0.1f + 0.9f * t);
    }
}

int fb_save_ppm(const Framebuffer *fb, const char *path){
    FILE *f = fopen(path, "wb");
    if (!f) return 1;

    size_t n = (size_t)fb->w * fb->h;
    unsigned char *bytes = malloc(n * 3);
    if (!bytes){ fclose(f); return 1; }
    for (size_t i = 0; i < n; i++){
        uint32_t c = fb->color[i];
        bytes[3 * i + 0] = (c >> 16) & 255;
        bytes[3 * i + 1] = (c >> 8) & 255;
        bytes[3 * i + 2] = c & 255;
    }

    fprintf(f, "P6\n%d %d\n255\n", fb->w, fb->h);
    size_t written = fwrite(bytes, 3, n, f);
    free(bytes);
    fclose(f);
    return written == n ? 0 : 1;
}
