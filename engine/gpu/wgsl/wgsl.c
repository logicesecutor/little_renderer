// wgsl.c - WGSL -> SPIR-V, for the Vulkan layer (engine/gpu/vk): not written yet
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen);
int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen){
    (void)src; (void)len;
    *words = NULL;
    *nwords = 0;
    snprintf(err, errlen, "the WGSL compiler is not written yet");
    return 1;
}
