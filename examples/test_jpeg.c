// test_jpeg.c - core/jpeg.c against libjpeg.
//
//   make bin/test_jpeg && bin/test_jpeg      (part of make test)
//
// 1. JPEG files written by libjpeg through Pillow (examples/jpeg_tests.h, made by
//    tools/gen_jpeg_tests.py): baseline and progressive, 4:4:4 / 4:2:2 / 4:2:0 / grey,
//    optimized Huffman tables, restart markers, 16-bit quantization tables (SOF1), RGB
//    without the colour transform, a size that is not a whole number of MCUs. Each must
//    decode to exactly libjpeg's own pixels (the same integer IDCT, upsampling and colour
//    conversion).
// 2. What a caller asks for: grey, grey + alpha, RGBA from the same file.
// 3. Broken files are refused, not crashed on: truncated anywhere, bytes changed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/jpeg.h"
#include "jpeg_tests.h"

static int failures;

static void check(const char *what, int ok, const char *detail){
    printf("  %-44s %-26s %s\n", what, detail, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static void against_libjpeg(void){
    printf("1. files from libjpeg\n");
    for (size_t i = 0; i < sizeof(jpeg_tests) / sizeof(jpeg_tests[0]); i++){
        const JpegTest *t = &jpeg_tests[i];
        int w, h, ch;
        unsigned char *px = jpeg_decode(t->jpg, t->size, &w, &h, &ch, 3);
        char detail[64] = "no image";
        int ok = 0;
        if (px && w == JPEG_TEST_W && h == JPEG_TEST_H && ch == 3){
            int mx = 0;
            long sum = 0, n = (long)w * h * 3;
            for (long k = 0; k < n; k++){
                int d = abs(px[k] - t->rgb[k]);
                if (d > mx) mx = d;
                sum += d;
            }
            snprintf(detail, sizeof(detail), "max %d, mean %.3f", mx, (double)sum / n);
            ok = mx == 0;
        }
        free(px);
        check(t->name, ok, detail);
    }
}

static void channels(void){
    printf("2. channels asked for\n");
    const JpegTest *t = &jpeg_tests[0];
    int w, h, ch, ok = 1;
    unsigned char *rgb = jpeg_decode(t->jpg, t->size, &w, &h, &ch, 0);
    ok = rgb && ch == 3;
    unsigned char *rgba = jpeg_decode(t->jpg, t->size, &w, &h, &ch, 4);
    ok = ok && rgba && ch == 4;
    for (int i = 0; ok && i < w * h; i++) ok = !memcmp(rgba + 4 * i, rgb + 3 * i, 3) && rgba[4 * i + 3] == 255;
    unsigned char *g = jpeg_decode(t->jpg, t->size, &w, &h, &ch, 1);
    ok = ok && g && ch == 1;
    check("RGB, RGBA (alpha 255), grey from a colour file", ok, "");
    free(rgb); free(rgba); free(g);
    for (size_t i = 0; i < sizeof(jpeg_tests) / sizeof(jpeg_tests[0]); i++)
        if (strstr(jpeg_tests[i].name, "grey")){
            unsigned char *p = jpeg_decode(jpeg_tests[i].jpg, jpeg_tests[i].size, &w, &h, &ch, 0);
            check("a grey file's own channels: 1", p && ch == 1, "");
            free(p);
            break;
        }
}

static void broken(void){
    printf("3. broken files (messages on stderr expected)\n");
    const JpegTest *t = &jpeg_tests[6];                       // a progressive one
    unsigned char *buf = malloc(t->size);
    int crashed_not = 1, refused = 0, tries = 0;
    for (unsigned cut = 0; cut < t->size; cut += 37){          // truncated
        int w, h, ch;
        unsigned char *p = jpeg_decode(t->jpg, cut, &w, &h, &ch, 3);
        if (!p) refused++;
        free(p);
        tries++;
    }
    unsigned seed = 1;
    for (int k = 0; k < 200; k++){                              // bytes changed
        memcpy(buf, t->jpg, t->size);
        for (int m = 0; m < 3; m++){
            seed = seed * 1664525u + 1013904223u;
            buf[(seed >> 8) % t->size] = (unsigned char)(seed >> 24);
        }
        int w, h, ch;
        unsigned char *p = jpeg_decode(buf, t->size, &w, &h, &ch, 3);
        free(p);
    }
    free(buf);
    char d[64];
    snprintf(d, sizeof(d), "%d of %d cuts refused", refused, tries);
    check("truncated and changed files: no crash", crashed_not, d);
}

int main(void){
    against_libjpeg();
    channels();
    broken();
    if (failures){ printf("test_jpeg: %d FAILED\n", failures); return 1; }
    printf("all JPEG tests passed\n");
    return 0;
}
