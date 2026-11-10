// texture.h - a texture is just another pixel array we read from.
#ifndef TEXTURE_H
#define TEXTURE_H

#include "vec.h"
#include <stdint.h>

typedef struct {
    int w, h;
    uint32_t *px;   // 0x00RRGGBB, row 0 = top of the image
} Texture;

Texture *texture_load_ppm(const char *path);   // P3 or P6, NULL on failure
Texture *texture_checker(int size, int cells); // procedural checkerboard
void texture_destroy(Texture *t);

// u, v in [0, 1] with v = 0 at the BOTTOM (OBJ convention). Wraps around outside [0, 1].
// Returns RGB as floats in [0, 1]. Nearest-neighbour sampling.
vec3 texture_sample(const Texture *t, float u, float v);

#endif
