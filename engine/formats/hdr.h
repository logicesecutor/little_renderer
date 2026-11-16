// hdr.h - Radiance .hdr images (RGBE: an 8-bit mantissa per channel and a shared 8-bit
// exponent), read and written: the HDR skies (scene/env.h) and the light probes. Our own
// code, as PNG's (formats/png.h).
//
// Read: the "32-bit_rle_rgbe" format, the usual "-Y h +X w" orientation (rows top to
// bottom), scanlines flat, in the old run-length form or the new one (each channel run-length
// encoded on its own, what every writer uses today). Written: the new run-length form.
#ifndef LR_HDR_H
#define LR_HDR_H

// The image as linear RGB floats, w * h * 3, rows top to bottom; NULL (and a message on
// stderr) if the file cannot be read. Free with free().
float *hdr_load(const char *path, int *w, int *h);
// The same from memory (a file already read)
float *hdr_decode(const unsigned char *data, long size, int *w, int *h);
// Writes rgb (w * h * 3, rows top to bottom); returns 0 on success
int hdr_save(const char *path, const float *rgb, int w, int h);

#endif
