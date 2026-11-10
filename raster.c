#include "raster.h"

#include <math.h>
#include <stdlib.h>

// Bresenham, integer only, works in every octant (this is the version from the first
// milestone, now writing colors into the framebuffer instead of a gray int matrix).
void draw_line(Framebuffer *fb, int x0, int y0, int x1, int y1, uint32_t color){
    // Lines that start or end absurdly far off screen would just burn time.
    if (abs(x0) > 100000 || abs(y0) > 100000 || abs(x1) > 100000 || abs(y1) > 100000) return;

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; // integer error: no float, no division, handles vertical and all octants

    while (1){
        fb_set_pixel(fb, x0, y0, color);   // set_pixel ignores anything off screen

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy){ err += dy; x0 += sx; }
        if (e2 <= dx){ err += dx; y0 += sy; }
    }
}

// Twice the signed area of triangle (a, b, p). Its sign says which side of edge a->b the
// point p is on. Dividing by the area of the whole triangle turns it into a barycentric weight.
static float edge(float ax, float ay, float bx, float by, float px, float py){
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

void draw_triangle(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                   FragShader shader, const void *ctx, int cull_backfaces){
    float area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    if (fabsf(area) < 1e-8f) return;              // degenerate (edge-on) triangle

    // Screen y points down, so a counter-clockwise triangle has NEGATIVE area here.
    if (area > 0){
        if (cull_backfaces) return;               // back face
        const ScreenVert *t = b; b = c; c = t;    // flip it so the math below stays the same
        area = -area;
    }
    float inv_area = 1.0f / area;

    // Bounding box of the triangle, clamped to the screen.
    float fminx = fminf(a->x, fminf(b->x, c->x)), fmaxx = fmaxf(a->x, fmaxf(b->x, c->x));
    float fminy = fminf(a->y, fminf(b->y, c->y)), fmaxy = fmaxf(a->y, fmaxf(b->y, c->y));
    int minx = fminx < 0 ? 0 : (int)floorf(fminx);
    int miny = fminy < 0 ? 0 : (int)floorf(fminy);
    int maxx = fmaxx > fb->w - 1 ? fb->w - 1 : (int)ceilf(fmaxx);
    int maxy = fmaxy > fb->h - 1 ? fb->h - 1 : (int)ceilf(fmaxy);

    // Pre-multiply the varyings by 1/w once per vertex: perspective-correct interpolation
    // interpolates (attr / w) and (1 / w) linearly in screen space and divides at the end.
    float pa[V_COUNT], pb[V_COUNT], pc[V_COUNT];
    for (int k = 0; k < V_COUNT; k++){
        pa[k] = a->vary[k] * a->inv_w;
        pb[k] = b->vary[k] * b->inv_w;
        pc[k] = c->vary[k] * c->inv_w;
    }

    for (int y = miny; y <= maxy; y++){
        for (int x = minx; x <= maxx; x++){
            float px = x + 0.5f, py = y + 0.5f;     // sample at the pixel center

            // barycentric weights; all >= 0 means the pixel is inside
            float w0 = edge(b->x, b->y, c->x, c->y, px, py) * inv_area;
            float w1 = edge(c->x, c->y, a->x, a->y, px, py) * inv_area;
            float w2 = edge(a->x, a->y, b->x, b->y, px, py) * inv_area;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            // A pixel exactly on a shared edge can be drawn by both neighbours. With opaque
            // triangles and a z-test that is harmless, so no special fill rule is needed.

            float z = w0 * a->z + w1 * b->z + w2 * c->z;
            size_t idx = (size_t)y * fb->w + x;
            if (z >= fb->depth[idx]) continue;       // something nearer is already here

            float iw = w0 * a->inv_w + w1 * b->inv_w + w2 * c->inv_w;
            float w = 1.0f / iw;
            float vary[V_COUNT];
            for (int k = 0; k < V_COUNT; k++)
                vary[k] = (w0 * pa[k] + w1 * pb[k] + w2 * pc[k]) * w;

            fb->depth[idx] = z;
            fb->color[idx] = shader(vary, ctx);
        }
    }
}
