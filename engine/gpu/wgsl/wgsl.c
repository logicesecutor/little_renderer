// wgsl.c - WGSL -> SPIR-V, for the Vulkan layer (engine/gpu/vk): our own compiler of the WGSL
// the engine writes (wgsl_impl.h: what it takes).
#include "gpu/wgsl/wgsl_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen);
int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen){
    Wgsl w;
    memset(&w, 0, sizeof(w));
    w.src = src; w.len = len;
    w.err = err; w.errlen = errlen;
    *words = NULL;
    *nwords = 0;
    int failed = 0;
    if (setjmp(w.fail)) failed = 1;
    else {
        wgsl_lex(&w);
        wgsl_parse(&w);
        wgsl_emit(&w, words, nwords);
    }
    for (int i = 0; i < w.nblocks; i++) free(w.blocks[i]);
    free(w.blocks);
    free(w.toks);
    if (failed){ free(*words); *words = NULL; *nwords = 0; }
    return failed;
}
