// raster.h - lines and triangles.
#ifndef LR_RASTER_H
#define LR_RASTER_H

#include "scene/image.h"

// A vertex that has already been through the whole transform pipeline.
// "vary" holds the per-vertex values (position, normal, uv, light...) that get
// interpolated across the triangle and handed to the fragment shader.
// V_LIGHT..B: a colour (render_scene); V_TX..TW: the tangent (PBR5). The fragment shader
// gets V_ALL values: the V_COUNT interpolated ones, then how U and V change to the next
// pixel across and down (the mip level of a texture)
enum { V_PX, V_PY, V_PZ, V_NX, V_NY, V_NZ, V_U, V_V, V_LIGHT, V_LIGHT_G, V_LIGHT_B, V_TX, V_TY, V_TZ, V_TW, V_COUNT,
       V_DUDX = V_COUNT, V_DVDX, V_DUDY, V_DVDY, V_ALL };

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
// The same triangle in one of two passes, to shade each pixel once (render.c): RASTER_DEPTH
// writes the depth and ids[pixel] = id where the triangle is nearer than what is there (no
// colour, no shader); RASTER_SHADE then shades only the pixels whose ids[] is id (the depth
// is already right). The same pixels and values as RASTER_DIRECT (draw_triangle_rows).
enum { RASTER_DIRECT, RASTER_DEPTH, RASTER_SHADE };
// Returns how many pixels RASTER_DEPTH wrote (0: hidden or off screen so far).
int draw_triangle_rows_pass(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                            FragShader shader, const void *ctx, int cull_backfaces, int row_lo, int row_hi,
                            int pass, int *ids, int id);

#endif
