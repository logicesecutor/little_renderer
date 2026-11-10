// raster.h - lines and triangles.
#ifndef LR_RASTER_H
#define LR_RASTER_H

#include "core/image.h"

// A vertex that has already been through the whole transform pipeline.
// "vary" holds the per-vertex values (position, normal, uv, light...) that get
// interpolated across the triangle and handed to the fragment shader.
enum { V_PX, V_PY, V_PZ, V_NX, V_NY, V_NZ, V_U, V_V, V_LIGHT, V_COUNT };

typedef struct {
    real x, y;        // screen position in pixels, (0,0) top-left
    real z;           // depth in [0, 1], interpolated linearly in screen space
    real inv_w;       // 1 / clip.w, needed for perspective-correct interpolation
    real vary[V_COUNT];
} ScreenVert;

// A "shader" is just a function that turns interpolated values into a color.
typedef uint32_t (*FragShader)(const real *vary, const void *ctx);

void draw_line(Framebuffer *fb, int x0, int y0, int x1, int y1, uint32_t color);

// Fills one triangle with edge functions + the z-buffer.
// Front faces are counter-clockwise as seen on screen (like OBJ files expect).
// cull_backfaces = 1 skips clockwise triangles; 0 draws both sides.
void draw_triangle(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                   FragShader shader, const void *ctx, int cull_backfaces);

// The same two, but they only write pixels in rows row_lo..row_hi (inclusive). Several
// threads can draw into one framebuffer at once if each owns different rows; the pixels
// come out exactly as if one thread had drawn everything.
void draw_line_rows(Framebuffer *fb, int x0, int y0, int x1, int y1, uint32_t color, int row_lo, int row_hi);
void draw_triangle_rows(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                        FragShader shader, const void *ctx, int cull_backfaces, int row_lo, int row_hi);

#endif
