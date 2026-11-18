// wgsl.c - WGSL -> SPIR-V, for the Vulkan layer (webgpu/vk): our own compiler of the WGSL
// the engine writes (wgsl_impl.h: what it takes).
#include "wgsl/wgsl_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// What a module with ray queries can use without declaring it (as in wgpu's WGSL and the
// WebGPU proposal): put before the source, on the same line as its first, so the source's
// line numbers stay.
static const char RAY_PRELUDE[] =
    "struct RayDesc { flags: u32, cull_mask: u32, t_min: f32, t_max: f32, origin: vec3f, dir: vec3f } "
    "struct RayIntersection { kind: u32, t: f32, instance_custom_data: u32, instance_index: u32, sbt_record_offset: u32, "
    "geometry_index: u32, primitive_index: u32, barycentrics: vec2f, front_face: bool, object_to_world: mat4x3f, "
    "world_to_object: mat4x3f } "
    "const RAY_FLAG_NONE = 0u; const RAY_FLAG_FORCE_OPAQUE = 1u; const RAY_FLAG_FORCE_NO_OPAQUE = 2u; "
    "const RAY_FLAG_TERMINATE_ON_FIRST_HIT = 4u; const RAY_FLAG_SKIP_CLOSEST_HIT_SHADER = 8u; "
    "const RAY_FLAG_CULL_BACK_FACING = 16u; const RAY_FLAG_CULL_FRONT_FACING = 32u; const RAY_FLAG_CULL_OPAQUE = 64u; "
    "const RAY_FLAG_CULL_NO_OPAQUE = 128u; const RAY_FLAG_SKIP_TRIANGLES = 256u; const RAY_FLAG_SKIP_AABBS = 512u; "
    "const RAY_QUERY_INTERSECTION_NONE = 0u; const RAY_QUERY_INTERSECTION_TRIANGLE = 1u; "
    "const RAY_QUERY_INTERSECTION_GENERATED = 2u; const RAY_QUERY_INTERSECTION_AABB = 3u; ";

static int mentions(const char *src, size_t len, const char *word){
    size_t n = strlen(word);
    for (size_t i = 0; i + n <= len; i++) if (!memcmp(src + i, word, n)) return 1;
    return 0;
}

int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen);
int wgsl_to_spirv(const char *src, size_t len, uint32_t **words, size_t *nwords, char *err, size_t errlen){
    Wgsl w;
    memset(&w, 0, sizeof(w));
    char *volatile with = NULL;      // (volatile: kept across the longjmp of an error)
    if (mentions(src, len, "ray_query")){
        size_t pn = sizeof(RAY_PRELUDE) - 1;
        with = malloc(pn + len + 1);
        if (!with){ snprintf(err, errlen, "out of memory"); return 1; }
        memcpy(with, RAY_PRELUDE, pn);
        memcpy(with + pn, src, len);
        with[pn + len] = 0;
        src = with;
        len += pn;
    }
    w.src = src; w.len = len;
    w.err = err; w.errlen = errlen;
    *words = NULL;
    *nwords = 0;
    volatile int failed = 0;         // (volatile: set after the longjmp)
    if (setjmp(w.fail)) failed = 1;
    else {
        wgsl_lex(&w);
        wgsl_parse(&w);
        wgsl_emit(&w, words, nwords);
    }
    for (int i = 0; i < w.nblocks; i++) free(w.blocks[i]);
    free(w.blocks);
    free(w.toks);
    free(with);
    if (failed){ free(*words); *words = NULL; *nwords = 0; }
    return failed;
}
