// hud.h - a line of text drawn over the image (the viewer's frame rate), with a tiny built-in
// font: 5 x 7 pixel glyphs for digits, a few letters and signs. No font files.
#ifndef LR_HUD_H
#define LR_HUD_H

#include <stdint.h>
#include "core/image.h"

// The text as a bitmap of w x h pixels, 0xAARRGGBB: white glyphs (each pixel of the font
// `scale` x `scale`) on a dark translucent box with a margin. Returns the pixels (malloc'ed,
// *w and *h set), NULL if out of memory. Characters the font lacks are left blank.
uint32_t *hud_text(const char *text, int scale, int *w, int *h);

// Blends such a bitmap into a framebuffer at (x, y), keeping what was under it in *saved
// (malloc'ed; w * h pixels) so hud_restore can put it back after the frame is shown.
void hud_draw(Framebuffer *fb, const uint32_t *px, int w, int h, int x, int y, uint32_t **saved);
void hud_restore(Framebuffer *fb, int w, int h, int x, int y, uint32_t *saved);   // frees saved

#endif
