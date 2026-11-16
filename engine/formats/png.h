// png.h - PNG files, written and read by the engine's own code (no library): the deflate
// compression (zlib, RFC 1950/1951) and the PNG format (W3C PNG specification) from scratch.
//
// Pixels are 8 bits per channel, rows from the top, channels interleaved:
// 1 = grey, 2 = grey + alpha, 3 = RGB, 4 = RGBA.
#ifndef LR_PNG_H
#define LR_PNG_H

#include <stddef.h>

// Writes w x h pixels of `channels` channels. 0 on success.
int png_write(const char *path, int w, int h, int channels, const unsigned char *px);
// The same into memory: a malloc'ed buffer (*size bytes), NULL on failure.
unsigned char *png_encode(int w, int h, int channels, const unsigned char *px, size_t *size);

// Reads any PNG: grey, grey + alpha, RGB, RGBA or palette, 1 to 16 bits, interlaced or not
// (16 bits keep their high byte, fewer are scaled to 0..255, a palette is looked up, a tRNS
// chunk becomes alpha). `want` = channels to return (1..4; 0 = the file's own, after the
// palette). Returns malloc'ed pixels and sets *w, *h and *channels; NULL on failure (a
// message on stderr).
unsigned char *png_read(const char *path, int *w, int *h, int *channels, int want);
unsigned char *png_decode(const unsigned char *data, size_t size, int *w, int *h, int *channels, int want);

// zlib streams on their own (PNG's IDAT, and later other formats). Both return a malloc'ed
// buffer, NULL on failure. zlib_compress: LZ77 with hash chains and fixed Huffman codes.
unsigned char *zlib_compress(const unsigned char *data, size_t n, size_t *out_n);
unsigned char *zlib_decompress(const unsigned char *data, size_t n, size_t *out_n);

#endif
