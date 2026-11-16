// hdr.c - Radiance .hdr (RGBE) reading and writing (see hdr.h)
#include "formats/hdr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// one RGBE pixel -> 3 floats: the mantissas times 2^(e - 128 - 8)
static void rgbe_to_float(const unsigned char *p, float *o){
    if (!p[3]){ o[0] = o[1] = o[2] = 0; return; }
    float f = ldexpf(1.0f, (int)p[3] - (128 + 8));
    o[0] = p[0] * f; o[1] = p[1] * f; o[2] = p[2] * f;
}

static void float_to_rgbe(const float *c, unsigned char *p){
    float v = c[0] > c[1] ? c[0] : c[1];
    if (c[2] > v) v = c[2];
    if (!(v > 1e-32f)){ p[0] = p[1] = p[2] = p[3] = 0; return; }
    int e;
    float m = frexpf(v, &e) * 256.0f / v;     // v = m' 2^e, m' in [0.5, 1)
    for (int k = 0; k < 3; k++){
        float x = c[k] > 0 ? c[k] * m : 0;
        p[k] = (unsigned char)(x > 255 ? 255 : x);
    }
    p[3] = (unsigned char)(e + 128);
}

// a scanline of w RGBE pixels into line; returns the bytes read, 0 on a broken file
static long read_scanline(const unsigned char *d, long n, int w, unsigned char *line){
    long at = 0;
    if (w >= 8 && w < 32768 && n >= 4 && d[0] == 2 && d[1] == 2 && !(d[2] & 0x80)){
        // the new form: 4 bytes of header, then each channel run-length encoded on its own
        if (((int)d[2] << 8 | d[3]) != w) return 0;
        at = 4;
        for (int c = 0; c < 4; c++){
            int x = 0;
            while (x < w){
                if (at >= n) return 0;
                int count = d[at++];
                if (count > 128){                      // a run: one byte, count - 128 times
                    count -= 128;
                    if (count > w - x || at >= n) return 0;
                    unsigned char b = d[at++];
                    for (int i = 0; i < count; i++) line[4 * (x++) + c] = b;
                } else {                               // count bytes as they are
                    if (count == 0 || count > w - x || at + count > n) return 0;
                    for (int i = 0; i < count; i++) line[4 * (x++) + c] = d[at++];
                }
            }
        }
        return at;
    }
    // flat pixels, or the old form: a pixel 1, 1, 1, k repeats the last pixel k << shift times
    int x = 0, shift = 0;
    while (x < w){
        if (at + 4 > n) return 0;
        const unsigned char *p = d + at;
        at += 4;
        if (p[0] == 1 && p[1] == 1 && p[2] == 1){
            // (a row is at most 32767 pixels: three repeats in a row, a shift of 16, already
            // cover it; a fourth, 24, is the last Radiance's own reader takes)
            if (x == 0 || shift > 24) return 0;
            int64_t rep = (int64_t)p[3] << shift;
            if (rep > w - x) return 0;
            for (int64_t i = 0; i < rep; i++, x++) memcpy(line + 4 * x, line + 4 * (x - 1), 4);
            shift += 8;
        } else {
            memcpy(line + 4 * (x++), p, 4);
            shift = 0;
        }
    }
    return at;
}

float *hdr_decode(const unsigned char *d, long n, int *pw, int *ph){
    long at = 0;
    if (n < 11 || (memcmp(d, "#?RADIANCE", 10) != 0 && memcmp(d, "#?RGBE", 6) != 0)){
        fprintf(stderr, "hdr: not a Radiance .hdr file\n");
        return NULL;
    }
    // the header: lines up to an empty one
    int format_ok = 1;
    for (;;){
        long end = at;
        while (end < n && d[end] != '\n') end++;
        if (end >= n){ fprintf(stderr, "hdr: the header has no end\n"); return NULL; }
        long len = end - at;
        if (len == 0){ at = end + 1; break; }
        if (len > 7 && memcmp(d + at, "FORMAT=", 7) == 0)
            format_ok = len >= 7 + 15 && memcmp(d + at + 7, "32-bit_rle_rgbe", 15) == 0;
        at = end + 1;
    }
    if (!format_ok){ fprintf(stderr, "hdr: only the 32-bit_rle_rgbe format is read (not XYZE)\n"); return NULL; }
    // the resolution line
    char res[64];
    long end = at;
    while (end < n && d[end] != '\n' && end - at < 63) end++;
    memcpy(res, d + at, (size_t)(end - at));
    res[end - at] = 0;
    at = end + 1;
    int w = 0, h = 0;
    if (sscanf(res, "-Y %d +X %d", &h, &w) != 2 || w <= 0 || h <= 0 || w > 65536 || h > 65536){
        fprintf(stderr, "hdr: unsupported orientation or size \"%s\" (only -Y h +X w)\n", res);
        return NULL;
    }
    float *img = malloc((size_t)w * h * 3 * sizeof(float));
    unsigned char *line = malloc((size_t)w * 4);
    if (!img || !line){ free(img); free(line); fprintf(stderr, "hdr: out of memory\n"); return NULL; }
    for (int y = 0; y < h; y++){
        long used = at < n ? read_scanline(d + at, n - at, w, line) : 0;
        if (!used){
            fprintf(stderr, "hdr: broken scanline %d of %d\n", y, h);
            free(img); free(line);
            return NULL;
        }
        at += used;
        for (int x = 0; x < w; x++) rgbe_to_float(line + 4 * x, img + 3 * ((size_t)y * w + x));
    }
    free(line);
    *pw = w; *ph = h;
    return img;
}

float *hdr_load(const char *path, int *w, int *h){
    FILE *f = fopen(path, "rb");
    if (!f){ fprintf(stderr, "hdr: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *d = n > 0 ? malloc((size_t)n) : NULL;
    if (!d || fread(d, 1, (size_t)n, f) != (size_t)n){
        fclose(f); free(d);
        fprintf(stderr, "hdr: cannot read %s\n", path);
        return NULL;
    }
    fclose(f);
    float *img = hdr_decode(d, n, w, h);
    free(d);
    if (!img) fprintf(stderr, "hdr: (in %s)\n", path);
    return img;
}

// one channel of a scanline, run-length encoded: runs of 3 or more equal bytes as runs,
// the rest as literal stretches (each at most 128)
static void write_channel(FILE *f, const unsigned char *line, int w, int c){
    int x = 0;
    while (x < w){
        // the next run of 3 or more from x on
        int run_at = x, run = 1;
        while (run_at < w){
            run = 1;
            while (run_at + run < w && run < 127 && line[4 * (run_at + run) + c] == line[4 * run_at + c]) run++;
            if (run >= 3) break;
            run_at += run;
        }
        if (run_at >= w) run = 0;
        while (x < run_at){                       // the literal bytes before it
            int k = run_at - x > 128 ? 128 : run_at - x;
            fputc(k, f);
            for (int i = 0; i < k; i++) fputc(line[4 * (x + i) + c], f);
            x += k;
        }
        if (run >= 3){
            fputc(128 + run, f);
            fputc(line[4 * x + c], f);
            x += run;
        }
    }
}

int hdr_save(const char *path, const float *rgb, int w, int h){
    FILE *f = fopen(path, "wb");
    if (!f){ fprintf(stderr, "hdr: cannot write %s\n", path); return 1; }
    fprintf(f, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n", h, w);
    unsigned char *line = malloc((size_t)w * 4);
    if (!line){ fclose(f); return 1; }
    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++) float_to_rgbe(rgb + 3 * ((size_t)y * w + x), line + 4 * x);
        if (w >= 8 && w < 32768){
            fputc(2, f); fputc(2, f); fputc(w >> 8, f); fputc(w & 255, f);
            for (int c = 0; c < 4; c++) write_channel(f, line, w, c);
        } else fwrite(line, 4, (size_t)w, f);
    }
    free(line);
    int bad = ferror(f);
    fclose(f);
    return bad ? 1 : 0;
}
