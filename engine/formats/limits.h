// limits.h - the largest image the decoders (png.c, hdr.c, jpeg.c) read.
#ifndef LR_LIMITS_H
#define LR_LIMITS_H

// At most 2^28 pixels (16384 x 16384), checked before anything is allocated. Their buffers,
// up to 12 bytes a pixel (HDR's RGB floats), then stay below 4 GB: w * h * bytes cannot wrap a
// 32-bit size_t (the WebAssembly build), where it would allocate a short buffer and the
// decoder write past it. Each side alone was bounded, not their product.
#define LR_MAX_PIXELS (1u << 28)

#endif
