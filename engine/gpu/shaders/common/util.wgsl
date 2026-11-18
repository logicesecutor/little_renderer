// util.wgsl - used by engine/gpu/gpu.c (GPU_WGSL_UTIL): the small helpers several shaders
// share, put in front of their modules (gpu_wgsl_join): the path tracer, its resolve pass,
// the denoiser, the BVH builder, the rasterizer, the view's composite, the bodies and the air.

/* a 32-bit integer hash (lowbias32) */
fn hash32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}

/* luminance (Rec. 709) */
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

/* the thread's index in a dispatch of 64-wide workgroups spread over x and y (gpu_dispatch) */
fn tid(gid: vec3u, nw: vec3u) -> u32 { return gid.x + gid.y * nw.x * 64u; }

/* two tangents of a unit normal (base/rng.h onb_from_normal, Duff et al. 2017) */
fn onb(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * c, -s * n.x), vec3f(c, s + n.y * n.y * a, -n.y));
}

/* the sRGB transfer function, both ways */
fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn linear_to_srgb(c0: vec3f) -> vec3f {
    let c = clamp(c0, vec3f(0.0), vec3f(1.0));
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

/* the photon cache's cells (trace.wgsl): a normal's main axis (6), and a cell's key */
fn pc_axis(n: vec3f) -> u32 {
    let a = abs(n);
    if (a.x >= a.y && a.x >= a.z) { return select(1u, 0u, n.x > 0.0); }
    if (a.y >= a.z) { return select(3u, 2u, n.y > 0.0); }
    return select(5u, 4u, n.z > 0.0);
}
fn pc_key(c: vec3i, axis: u32, level: u32) -> u32 {
    return hash32((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u) ^
                  ((axis + 6u * level) * 2654435761u));
}
