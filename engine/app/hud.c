#include "app/hud.h"

#include <stdlib.h>
#include <string.h>

// 5 x 7 glyphs, one byte per row, the low 5 bits from left (bit 4) to right (bit 0)
static const struct { char c; unsigned char rows[7]; } FONT[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}}, {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
    {'(', {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02}}, {')', {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08}},
    {'f', {0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08}}, {'p', {0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10}},
    {'s', {0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E}}, {'m', {0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11}},
};

static const unsigned char *glyph(char c){
    for (size_t i = 0; i < sizeof(FONT) / sizeof(FONT[0]); i++) if (FONT[i].c == c) return FONT[i].rows;
    return NULL;
}

uint32_t *hud_text(const char *text, int scale, int *pw, int *ph){
    if (scale < 1) scale = 1;
    int n = (int)strlen(text), pad = 3 * scale;
    int w = n * 6 * scale - scale + 2 * pad, h = 7 * scale + 2 * pad;
    if (w < 2 * pad) w = 2 * pad;
    uint32_t *px = malloc((size_t)w * h * sizeof(uint32_t));
    if (!px) return NULL;
    for (int i = 0; i < w * h; i++) px[i] = 0xA0101010u;   // dark, 63% opaque
    for (int k = 0; k < n; k++){
        const unsigned char *g = glyph(text[k]);
        if (!g) continue;
        for (int r = 0; r < 7; r++)
            for (int c = 0; c < 5; c++){
                if (!(g[r] >> (4 - c) & 1)) continue;
                for (int sy = 0; sy < scale; sy++)
                    for (int sx = 0; sx < scale; sx++){
                        int x = pad + (k * 6 + c) * scale + sx, y = pad + r * scale + sy;
                        px[y * w + x] = 0xFFFFFFFFu;
                    }
            }
    }
    *pw = w;
    *ph = h;
    return px;
}

void hud_draw(Framebuffer *fb, const uint32_t *px, int w, int h, int x0, int y0, uint32_t **saved){
    *saved = malloc((size_t)w * h * sizeof(uint32_t));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            int fx = x0 + x, fy = y0 + y;
            if (fx < 0 || fy < 0 || fx >= fb->w || fy >= fb->h) continue;
            uint32_t *d = &fb->color[(size_t)fy * fb->w + fx], s = px[y * w + x];
            if (*saved) (*saved)[y * w + x] = *d;
            unsigned a = s >> 24;
            unsigned r = ((s >> 16 & 255) * a + (*d >> 16 & 255) * (255 - a)) / 255;
            unsigned g = ((s >> 8 & 255) * a + (*d >> 8 & 255) * (255 - a)) / 255;
            unsigned b = ((s & 255) * a + (*d & 255) * (255 - a)) / 255;
            *d = r << 16 | g << 8 | b;
        }
}

void hud_restore(Framebuffer *fb, int w, int h, int x0, int y0, uint32_t *saved){
    if (!saved) return;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            int fx = x0 + x, fy = y0 + y;
            if (fx < 0 || fy < 0 || fx >= fb->w || fy >= fb->h) continue;
            fb->color[(size_t)fy * fb->w + fx] = saved[y * w + x];
        }
    free(saved);
}
