#include "core/texture.h"

#include "core/image.h"   // rgb8
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// next integer in a PPM header / P3 body, skipping whitespace and # comments. -1 on EOF.
static int ppm_int(FILE *f){
    int c;
    for (;;){
        c = fgetc(f);
        if (c == EOF) return -1;
        if (c == '#'){ while (c != '\n' && c != EOF) c = fgetc(f); }
        else if (!isspace(c)) break;
    }
    int n = 0;
    while (isdigit(c)){ n = n * 10 + (c - '0'); c = fgetc(f); }   // eats the one char after the number
    return n;
}

Texture *texture_load_ppm(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int m0 = fgetc(f), m1 = fgetc(f);
    if (m0 != 'P' || (m1 != '3' && m1 != '6')){ fclose(f); return NULL; }
    int w = ppm_int(f), h = ppm_int(f), maxval = ppm_int(f);
    if (w <= 0 || h <= 0 || maxval <= 0 || maxval > 255){ fclose(f); return NULL; }

    Texture *t = malloc(sizeof(*t));
    t->w = w;
    t->h = h;
    t->px = malloc((size_t)w * h * sizeof(uint32_t));
    for (int i = 0; i < w * h; i++){
        int rgb[3];
        for (int k = 0; k < 3; k++) rgb[k] = (m1 == '6') ? fgetc(f) : ppm_int(f);
        if (rgb[0] < 0 || rgb[1] < 0 || rgb[2] < 0){ free(t->px); free(t); fclose(f); return NULL; }
        t->px[i] = rgb8(rgb[0] * 255 / maxval, rgb[1] * 255 / maxval, rgb[2] * 255 / maxval);
    }
    fclose(f);
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
