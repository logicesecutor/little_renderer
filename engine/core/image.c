#include "core/image.h"

#include <stdio.h>
#include <stdlib.h>

// ---- Framebuffer --------------------------------------------------------------------

Framebuffer *fb_create(int w, int h){
    Framebuffer *fb = malloc(sizeof(*fb));
    if (!fb) return NULL;
    fb->w = w;
    fb->h = h;
    fb->color = calloc((size_t)w * h, sizeof(uint32_t));
    fb->depth = malloc((size_t)w * h * sizeof(real));
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
    real lo = 1.0f, hi = 0.0f;
    for (size_t i = 0; i < n; i++){
        real d = fb->depth[i];
        if (d >= 1.0f) continue;   // nothing was drawn here
        if (d < lo) lo = d;
        if (d > hi) hi = d;
    }
    real range = hi - lo > 1e-9f ? hi - lo : 1.0f;
    for (size_t i = 0; i < n; i++){
        real d = fb->depth[i];
        if (d >= 1.0f){ fb->color[i] = 0; continue; }
        real t = 1.0f - (d - lo) / range;          // near -> 1, far -> 0
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

// ---- Image --------------------------------------------------------------------------

Image image_create(int w, int h){
    Image img = {w, h, calloc((size_t)w * h, sizeof(Vec3))};
    if (!img.px){ fprintf(stderr, "out of memory\n"); exit(1); }
    return img;
}

void image_free(Image *img){
    free(img->px);
    img->px = NULL;
}

// Light is computed in linear units; screens expect sRGB-encoded values. Without this
// step the image looks far too dark.
static unsigned char to_srgb8(double c){
    if (!(c > 0)) c = 0;            // also catches NaN
    if (c > 1) c = 1;
    c = c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    return (unsigned char)(c * 255.0 + 0.5);
}

unsigned char image_srgb8(double linear){ return to_srgb8(linear); }

int image_write_ppm(const Image *img, const char *path){
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
    size_t n = (size_t)img->w * img->h;
    for (size_t i = 0; i < n; i++){
        unsigned char rgb[3] = { to_srgb8(img->px[i].x), to_srgb8(img->px[i].y), to_srgb8(img->px[i].z) };
        fwrite(rgb, 1, 3, f);
    }
    return fclose(f) == 0 ? 0 : 1;
}

// ---- fast linear -> sRGB pixel ------------------------------------------------------

static unsigned char srgb_lut[4096];   // linear [0,1] -> sRGB byte
static int srgb_lut_ready;

void srgb_lut_init(void){
    if (srgb_lut_ready) return;
    for (int i = 0; i < 4096; i++) srgb_lut[i] = to_srgb8(i / 4095.0);
    srgb_lut_ready = 1;
}

static inline int lut_index(real c){
    if (!(c > 0)) return 0;
    if (c >= 1) return 4095;
    return (int)(c * 4095.0 + 0.5);
}

uint32_t image_pack_srgb(Vec3 c){
    return ((uint32_t)srgb_lut[lut_index(c.x)] << 16) |
           ((uint32_t)srgb_lut[lut_index(c.y)] << 8) |
            (uint32_t)srgb_lut[lut_index(c.z)];
}
