// image.h - a linear float RGB image and its PPM output.
#ifndef IMAGE_H
#define IMAGE_H

#include "vec3.h"

typedef struct {
    int w, h;
    Vec3 *px;   // linear radiance, row 0 = top
} Image;

Image image_create(int w, int h);
void image_free(Image *img);

// clamps to [0,1], applies the sRGB curve and writes a binary PPM (P6). 0 on success.
int image_write_ppm(const Image *img, const char *path);

// one linear channel value -> 0..255 sRGB byte (clamped, NaN -> 0)
unsigned char image_srgb8(double linear);

#endif
