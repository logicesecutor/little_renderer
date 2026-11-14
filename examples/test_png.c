// test_png.c - core/png.c against files from other encoders, and against itself.
//
//   make bin/test_png && bin/test_png      (part of make test)
//
// 1. PNG files written by Pillow and by Python's zlib (examples/png_tests.h, made by
//    tools/gen_png_tests.py): every colour type, bit depths 1-16, palettes, transparency,
//    interlacing, every filter, stored / fixed / dynamic deflate blocks. Each must decode
//    to exactly the pixels it holds.
// 2. Round trips: images of 1-4 channels and many sizes written by png_encode and read back
//    unchanged; corrupted files are refused.
// 3. Speed and size on a 1920x1080 picture.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/png.h"
#include "png_tests.h"

static int failures;

static void check(const char *what, int ok){
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void others(void){
    printf("files from other encoders\n");
    for (size_t i = 0; i < sizeof(png_tests) / sizeof(png_tests[0]); i++){
        const PngTest *t = &png_tests[i];
        int w, h, ch;
        unsigned char *px = png_decode(t->png, t->size, &w, &h, &ch, 4);
        int ok = px && w == PNG_TEST_W && h == PNG_TEST_H && ch == 4 && !memcmp(px, t->rgba, (size_t)w * h * 4);
        free(px);
        px = png_decode(t->png, t->size, &w, &h, &ch, 0);
        ok = ok && px && ch == t->channels;
        free(px);
        char what[96];
        snprintf(what, sizeof(what), "%s", t->name);
        check(what, ok);
    }
}

// a picture with smooth parts and noise, as renders have
static unsigned char pattern(int x, int y, int c, unsigned *seed){
    *seed = *seed * 1664525u + 1013904223u;
    int smooth = (x * 3 + y * 2 + c * 40) & 255;
    return (unsigned char)(((x / 16 + y / 16) & 1) ? smooth : smooth ^ (int)(*seed >> 29));
}

static void round_trips(void){
    printf("round trips (png_encode, then png_decode)\n");
    static const int sizes[][2] = {{1, 1}, {2, 1}, {1, 3}, {7, 3}, {31, 17}, {257, 129}, {640, 9}};
    for (int ch = 1; ch <= 4; ch++){
        int ok = 1;
        for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++){
            int w = sizes[s][0], h = sizes[s][1];
            unsigned seed = (unsigned)(w * 977 + h * 131 + ch);
            unsigned char *px = malloc((size_t)w * h * ch);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < ch; c++) px[((size_t)y * w + x) * ch + c] = pattern(x, y, c, &seed);
            size_t n;
            unsigned char *png = png_encode(w, h, ch, px, &n);
            int rw, rh, rc;
            unsigned char *back = png ? png_decode(png, n, &rw, &rh, &rc, 0) : NULL;
            ok = ok && back && rw == w && rh == h && rc == ch && !memcmp(back, px, (size_t)w * h * ch);
            free(back);
            free(png);
            free(px);
        }
        char what[64];
        snprintf(what, sizeof(what), "%d channel(s), 7 sizes from 1x1 to 640x9", ch);
        check(what, ok);
    }
    // zlib on its own: long runs, no repeats, empty
    static unsigned char d[100000];
    for (size_t i = 0; i < sizeof(d); i++) d[i] = i < 50000 ? (unsigned char)(i % 7) : (unsigned char)((i * 2654435761u) >> 24);
    int ok = 1;
    size_t sizes2[] = {0, 1, 3, 1000, sizeof(d)};
    for (size_t k = 0; k < 5; k++){
        size_t zn, bn;
        unsigned char *z = zlib_compress(d, sizes2[k], &zn);
        unsigned char *b = z ? zlib_decompress(z, zn, &bn) : NULL;
        ok = ok && b && bn == sizes2[k] && (bn == 0 || !memcmp(b, d, bn));
        free(z);
        free(b);
    }
    check("zlib streams: 0 to 100000 bytes, runs and noise", ok);
    // a corrupted file is refused, not misread
    unsigned char px[12 * 12 * 3];
    for (size_t i = 0; i < sizeof(px); i++) px[i] = (unsigned char)(i * 7);
    size_t n;
    unsigned char *png = png_encode(12, 12, 3, px, &n);
    int refused = 1;
    for (size_t at = 8; at < n; at += 7){
        png[at] ^= 0x10;
        int w, h, ch;
        unsigned char *b = png_decode(png, n, &w, &h, &ch, 0);
        if (b){ refused = 0; free(b); }
        png[at] ^= 0x10;
    }
    free(png);
    check("a flipped bit anywhere: refused (CRC)", refused);
}

static void speed(void){
    printf("a 1920x1080 RGB picture\n");
    int w = 1920, h = 1080;
    unsigned char *px = malloc((size_t)w * h * 3);
    unsigned seed = 1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            // a soft gradient with a little grain, like a path traced image
            seed = seed * 1664525u + 1013904223u;
            int g = (int)(seed >> 30);
            px[((size_t)y * w + x) * 3 + 0] = (unsigned char)(x * 255 / w + g);
            px[((size_t)y * w + x) * 3 + 1] = (unsigned char)(y * 200 / h + g);
            px[((size_t)y * w + x) * 3 + 2] = (unsigned char)(90 + ((x + y) >> 4 & 31) + g);
        }
    double t0 = now_s();
    size_t n;
    unsigned char *png = png_encode(w, h, 3, px, &n);
    double t1 = now_s();
    int rw, rh, rc;
    unsigned char *back = png ? png_decode(png, n, &rw, &rh, &rc, 3) : NULL;
    double t2 = now_s();
    printf("  %.1f MB -> %.2f MB (%.0f%%), written in %.0f ms, read in %.0f ms\n", w * h * 3 / 1e6, n / 1e6,
           100.0 * n / (w * h * 3), (t1 - t0) * 1e3, (t2 - t1) * 1e3);
    check("read back unchanged", back && !memcmp(back, px, (size_t)w * h * 3));
    free(back);
    free(png);
    free(px);
}

int main(void){
    others();
    round_trips();
    speed();
    if (failures){ printf("%d FAILED\n", failures); return 1; }
    printf("all PNG tests passed\n");
    return 0;
}
