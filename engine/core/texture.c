#include "core/texture.h"

#include "core/image.h"   // rgb8
#include "core/png.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Texture *texture_load_png(const char *path){
    int w, h, ch;
    unsigned char *px = png_read(path, &w, &h, &ch, 3);
    if (!px) return NULL;
    Texture *t = malloc(sizeof(*t));
    if (t) t->px = malloc((size_t)w * h * sizeof(uint32_t));
    if (!t || !t->px){ free(t); free(px); return NULL; }
    t->w = w;
    t->h = h;
    for (size_t i = 0; i < (size_t)w * h; i++) t->px[i] = rgb8(px[3 * i], px[3 * i + 1], px[3 * i + 2]);
    free(px);
    return t;
}

Texture *texture_checker(int size, int cells){
    Texture *t = malloc(sizeof(*t));
    t->w = t->h = size;
    t->px = malloc((size_t)size * size * sizeof(uint32_t));
    for (int y = 0; y < size; y++){
        for (int x = 0; x < size; x++){
            int on = ((x * cells / size) + (y * cells / size)) & 1;
            t->px[y * size + x] = on ? rgb8(235, 235, 225) : rgb8(200, 60, 50);
        }
    }
    return t;
}

void texture_destroy(Texture *t){
    if (!t) return;
    free(t->px);
    free(t);
}

Vec3 texture_sample(const Texture *t, real u, real v){
    u -= r_floor(u);                       // wrap into [0, 1)
    v -= r_floor(v);
    int x = (int)(u * t->w);
    int y = (int)((1.0f - v) * t->h);     // image row 0 is the top, v = 0 is the bottom
    if (x >= t->w) x = t->w - 1;
    if (y >= t->h) y = t->h - 1;
    if (y < 0) y = 0;
    uint32_t c = t->px[y * t->w + x];
    return vec3(((c >> 16) & 255) / 255.0f, ((c >> 8) & 255) / 255.0f, (c & 255) / 255.0f);
}
