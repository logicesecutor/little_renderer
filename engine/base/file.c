// file.c - a whole file read into memory (base/file.h).
#include "base/file.h"

#include <stdio.h>
#include <stdlib.h>

unsigned char *file_read_all(const char *path, size_t *n){
    *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, k;
    unsigned char *b = malloc(cap);
    while (b && (k = fread(b + *n, 1, cap - *n, f)) > 0){
        *n += k;
        if (*n == cap){
            unsigned char *q = realloc(b, cap * 2);
            if (!q){ free(b); b = NULL; break; }
            b = q; cap *= 2;
        }
    }
    if (b && ferror(f)){ free(b); b = NULL; }
    fclose(f);
    if (!b) *n = 0;
    return b;
}
