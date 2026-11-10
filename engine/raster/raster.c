#include "raster/raster.h"

#include <math.h>
#include <stdlib.h>

// Bresenham, integer only, works in every octant (this is the version from the first
// milestone, now writing colors into the framebuffer instead of a gray int matrix).
void draw_line(Framebuffer *fb, int x0, int y0, int x1, int y1, uint32_t color){
    draw_line_rows(fb, x0, y0, x1, y1, color, 0, fb->h - 1);
}

void draw_line_rows(Framebuffer *fb, int x0, int y0, int x1, int y1, uint32_t color, int row_lo, int row_hi){
    // Lines that start or end absurdly far off screen would just burn time.
    if (abs(x0) > 100000 || abs(y0) > 100000 || abs(x1) > 100000 || abs(y1) > 100000) return;
    if ((y0 < row_lo && y1 < row_lo) || (y0 > row_hi && y1 > row_hi)) return;   // not in these rows

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; // integer error: no real, no division, handles vertical and all octants

    while (1){
        // set_pixel ignores anything off screen; rows outside [row_lo, row_hi] belong to
        // another thread (render.c), which walks the same line and draws them
        if (y0 >= row_lo && y0 <= row_hi) fb_set_pixel(fb, x0, y0, color);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy){ err += dy; x0 += sx; }
        if (e2 <= dx){ err += dx; y0 += sy; }
    }
}

// Twice the signed area of triangle (a, b, p). Its sign says which side of edge a->b the
// point p is on. Dividing by the area of the whole triangle turns it into a barycentric weight.
static real edge(real ax, real ay, real bx, real by, real px, real py){
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

void draw_triangle(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                   FragShader shader, const void *ctx, int cull_backfaces){
    draw_triangle_rows(fb, a, b, c, shader, ctx, cull_backfaces, 0, fb->h - 1);
}

void draw_triangle_rows(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                        FragShader shader, const void *ctx, int cull_backfaces, int row_lo, int row_hi){
    real area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    if (r_fabs(area) < 1e-8f) return;              // degenerate (edge-on) triangle

    // Screen y points down, so a counter-clockwise triangle has NEGATIVE area here.
    if (area > 0){
        if (cull_backfaces) return;               // back face
        const ScreenVert *t = b; b = c; c = t;    // flip it so the math below stays the same
        area = -area;
    }
    real inv_area = 1.0f / area;

    // Bounding box of the triangle, clamped to the screen. Plain comparisons and int casts
    // instead of fminf / floorf / ceilf: on a default build those are library calls, and
    // this runs for every triangle (on an 871k-triangle model that is measurable).
    real fminx = a->x < b->x ? a->x : b->x, fmaxx = a->x > b->x ? a->x : b->x;
    real fminy = a->y < b->y ? a->y : b->y, fmaxy = a->y > b->y ? a->y : b->y;
    if (c->x < fminx) fminx = c->x;
    if (c->x > fmaxx) fmaxx = c->x;
    if (c->y < fminy) fminy = c->y;
    if (c->y > fmaxy) fmaxy = c->y;
    // entirely off screen (or NaN): no pixel center can be inside
    if (!(fmaxx >= 0 && fmaxy >= 0 && fminx <= fb->w && fminy <= fb->h)) return;
    int minx = fminx < 0 ? 0 : (int)fminx;                 // = floor, since fminx >= 0
    int miny = fminy < 0 ? 0 : (int)fminy;
    int maxx = fmaxx > fb->w - 1 ? fb->w - 1 : (int)fmaxx + ((real)(int)fmaxx < fmaxx);   // = ceil
    int maxy = fmaxy > fb->h - 1 ? fb->h - 1 : (int)fmaxy + ((real)(int)fmaxy < fmaxy);
    if (miny < row_lo) miny = row_lo;             // only the rows we were asked to fill
    if (maxy > row_hi) maxy = row_hi;
    if (miny > maxy) return;

    // Pre-multiply the varyings by 1/w once per vertex: perspective-correct interpolation
    // interpolates (attr / w) and (1 / w) linearly in screen space and divides at the end.
    real pa[V_COUNT], pb[V_COUNT], pc[V_COUNT];
    for (int k = 0; k < V_COUNT; k++){
        pa[k] = a->vary[k] * a->inv_w;
        pb[k] = b->vary[k] * b->inv_w;
        pc[k] = c->vary[k] * c->inv_w;
    }

    for (int y = miny; y <= maxy; y++){
        for (int x = minx; x <= maxx; x++){
            real px = x + 0.5f, py = y + 0.5f;     // sample at the pixel center

            // barycentric weights; all >= 0 means the pixel is inside
            real w0 = edge(b->x, b->y, c->x, c->y, px, py) * inv_area;
            real w1 = edge(c->x, c->y, a->x, a->y, px, py) * inv_area;
            real w2 = edge(a->x, a->y, b->x, b->y, px, py) * inv_area;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            // A pixel exactly on a shared edge can be drawn by both neighbours. With opaque
            // triangles and a z-test that is harmless, so no special fill rule is needed.

            real z = w0 * a->z + w1 * b->z + w2 * c->z;
            size_t idx = (size_t)y * fb->w + x;
            if (z >= fb->depth[idx]) continue;       // something nearer is already here

            real iw = w0 * a->inv_w + w1 * b->inv_w + w2 * c->inv_w;
            real w = 1.0f / iw;
            real vary[V_COUNT];
            for (int k = 0; k < V_COUNT; k++)
                vary[k] = (w0 * pa[k] + w1 * pb[k] + w2 * pc[k]) * w;

            fb->depth[idx] = z;
            fb->color[idx] = shader(vary, ctx);
        }
    }
}
