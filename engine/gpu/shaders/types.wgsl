// types.wgsl - used by engine/gpu/view_gpu.c (GPU_WGSL_TYPES)
struct Vol {
    lo: vec4f, hi: vec4f,
    alo: vec4f, ahi: vec4f,
    albedo: vec4f,
    dims: vec4u,
    bdims: vec4u,
    ldims: vec4u,
    p: vec4f,
};
struct Frame {
    right: vec4f, up: vec4f, back: vec4f, eye: vec4f,
    proj: vec4f,
    size: vec4f,
    lamp_pos: vec4f,
    lamp_n: vec4f,
    lamp_L: vec4f,
    amb: vec4f,
    vols: array<Vol, 2>,
};
/* computed on the GPU from the densities (STATS): where the medium is, and its largest density */
struct VStat { alo: vec4f, ahi: vec4f, mx: vec4f, };
struct Sel { k: vec4u };
@group(0) @binding(0) var<uniform> F: Frame;
