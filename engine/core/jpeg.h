// jpeg.h - JPEG files read by the engine's own code (no library; ITU T.81): the textures of
// glTF models (docs/PBR.md, PBR6).
//
// Huffman-coded JPEG, 8 bits per sample: baseline and extended sequential (SOF0, SOF1) and
// progressive (SOF2: spectral selection and successive approximation); 1 component (grey)
// or 3 (YCbCr, or RGB when an Adobe marker says so); any sampling factors (4:4:4, 4:2:2,
// 4:2:0, ...); restart markers. Not: arithmetic coding, lossless, 12 bits, CMYK.
// The chroma is upsampled and converted as libjpeg does by default (its "fancy" triangle
// filter for 2:1, its fixed-point YCbCr), so the pixels are libjpeg's within a step or two.
#ifndef LR_JPEG_H
#define LR_JPEG_H

#include <stddef.h>

// As png_read / png_decode (core/png.h): rows from the top, channels interleaved; `want` =
// channels to return (1..4; 0 = the file's own: 1 or 3; 4 adds an opaque alpha). Returns
// malloc'ed pixels and sets *w, *h, *channels; NULL on failure (a message on stderr).
unsigned char *jpeg_read(const char *path, int *w, int *h, int *channels, int want);
unsigned char *jpeg_decode(const unsigned char *data, size_t size, int *w, int *h, int *channels, int want);

#endif
