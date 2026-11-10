#include "image.h"

#include <stdio.h>
#include <stdlib.h>

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
