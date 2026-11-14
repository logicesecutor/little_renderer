// brdf_gpu.h - the BSDF of core/brdf.h on the GPU (gpu/brdf_gpu.c): the same functions in
// WGSL, used by the GPU path tracer (and by the rasterizer from PBR3). This header is for
// checking the two against each other (examples/test_gpu_brdf.c).
#ifndef LR_BRDF_GPU_H
#define LR_BRDF_GPU_H

#include "gpu/gpu.h"

// One case, 16 floats: base colour (3), metallic, roughness, ior, transmission, outside
// (0 / 1), v (3), l (3), ul, u1 ... and u2 at [16]: 17 floats in all (GPU_BSDF_IN).
// Its answer, 16 floats (GPU_BSDF_OUT): bsdf_eval(v, l): f (3), pdf; bsdf_sample(v, ul, u1, u2):
// ok, delta, through, pdf, l (3), w (3), and the lobe chances' first two (p[0], p[1]).
// The normal is (0, 0, 1). Runs on the GPU and waits for it: returns 0 if it could not.
#define GPU_BSDF_IN 17
#define GPU_BSDF_OUT 16
int gpu_bsdf_run(Gpu *g, const float *in, int n, float *out);

#endif
