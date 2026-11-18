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
    (void)draw_triangle_rows_pass(fb, a, b, c, shader, ctx, cull_backfaces, row_lo, row_hi, RASTER_DIRECT, NULL, 0);
}

int draw_triangle_rows_pass(Framebuffer *fb, const ScreenVert *a, const ScreenVert *b, const ScreenVert *c,
                            FragShader shader, const void *ctx, int cull_backfaces, int row_lo, int row_hi,
                            int pass, int *ids, int id){
    int written = 0;
    real area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    if (r_fabs(area) < 1e-8f) return 0;              // degenerate (edge-on) triangle

    // Screen y points down, so a counter-clockwise triangle has NEGATIVE area here.
    if (area > 0){
        if (cull_backfaces) return 0;               // back face
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
    if (!(fmaxx >= 0 && fmaxy >= 0 && fminx <= fb->w && fminy <= fb->h)) return 0;
    int minx = fminx < 0 ? 0 : (int)fminx;                 // = floor, since fminx >= 0
    int miny = fminy < 0 ? 0 : (int)fminy;
    int maxx = fmaxx > fb->w - 1 ? fb->w - 1 : (int)fmaxx + ((real)(int)fmaxx < fmaxx);   // = ceil
    int maxy = fmaxy > fb->h - 1 ? fb->h - 1 : (int)fmaxy + ((real)(int)fmaxy < fmaxy);
    if (miny < row_lo) miny = row_lo;             // only the rows we were asked to fill
    if (maxy > row_hi) maxy = row_hi;
    if (miny > maxy) return 0;

    // Pre-multiply the varyings by 1/w once per vertex: perspective-correct interpolation
    // interpolates (attr / w) and (1 / w) linearly in screen space and divides at the end.
    real pa[V_COUNT], pb[V_COUNT], pc[V_COUNT];
    for (int k = 0; k < V_COUNT; k++){
        pa[k] = a->vary[k] * a->inv_w;
        pb[k] = b->vary[k] * b->inv_w;
        pc[k] = c->vary[k] * c->inv_w;
    }
    // How U and V change across the screen: the weights are linear in x and y, so U / w and
    // 1 / w change by constants, and dU = (d(U / w) - U d(1 / w)) w
    real g0x = (b->y - c->y) * inv_area, g1x = (c->y - a->y) * inv_area, g2x = (a->y - b->y) * inv_area;
    real g0y = (c->x - b->x) * inv_area, g1y = (a->x - c->x) * inv_area, g2y = (b->x - a->x) * inv_area;
    real diw_x = g0x * a->inv_w + g1x * b->inv_w + g2x * c->inv_w, diw_y = g0y * a->inv_w + g1y * b->inv_w + g2y * c->inv_w;
    real du_x = g0x * pa[V_U] + g1x * pb[V_U] + g2x * pc[V_U], du_y = g0y * pa[V_U] + g1y * pb[V_U] + g2y * pc[V_U];
    real dv_x = g0x * pa[V_V] + g1x * pb[V_V] + g2x * pc[V_V], dv_y = g0y * pa[V_V] + g1y * pb[V_V] + g2y * pc[V_V];

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
            if (pass == RASTER_SHADE){
                if (ids[idx] != id) continue;        // another triangle is the one seen here
            } else if (z >= fb->depth[idx]) continue;   // something nearer is already here
            if (pass == RASTER_DEPTH){ fb->depth[idx] = z; ids[idx] = id; written++; continue; }

            real iw = w0 * a->inv_w + w1 * b->inv_w + w2 * c->inv_w;
            real w = 1.0f / iw;
            real vary[V_ALL];
            for (int k = 0; k < V_COUNT; k++)
                vary[k] = (w0 * pa[k] + w1 * pb[k] + w2 * pc[k]) * w;
            vary[V_DUDX] = (du_x - vary[V_U] * diw_x) * w;
            vary[V_DVDX] = (dv_x - vary[V_V] * diw_x) * w;
            vary[V_DUDY] = (du_y - vary[V_U] * diw_y) * w;
            vary[V_DVDY] = (dv_y - vary[V_V] * diw_y) * w;

            if (pass == RASTER_DIRECT) fb->depth[idx] = z;   // (RASTER_SHADE: it is there already)
            fb->color[idx] = shader(vary, ctx);
        }
    }
    return written;
}
