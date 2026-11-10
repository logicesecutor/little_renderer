// raster.h - lines and triangles.
#ifndef RASTER_H
#define RASTER_H

#include "framebuffer.h"

// A vertex that has already been through the whole transform pipeline.
// "vary" holds the per-vertex values (position, normal, uv, light...) that get
// interpolated across the triangle and handed to the fragment shader.
enum { V_PX, V_PY, V_PZ, V_NX, V_NY, V_NZ, V_U, V_V, V_LIGHT, V_COUNT };

typedef struct {
    float x, y;        // screen position in pixels, (0,0) top-left
    float z;           // depth in [0, 1], interpolated linearly in screen space
    float inv_w;       // 1 / clip.w, needed for perspective-correct interpolation
    float vary[V_COUNT];
} ScreenVert;

// A "shader" is just a function that turns interpolated values into a color.
typedef uint32_t (*FragShader)(const float *vary, const void *ctx);

void draw_line(Framebuffer *fb, int x0, int y0, int x1, int y1, uint32_t color);

// Fills one triangle with edge functions + the z-buffer.
// Front faces are counter-clockwise as seen on screen (like OBJ files expect).
// cull_backfaces = 1 skips clockwise triangles; 0 draws both sides.
void draw_triangle(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                   FragShader shader, const void *ctx, int cull_backfaces);

#endif
