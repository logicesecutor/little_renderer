// view_frame_cpu.wgsl - used by engine/gpu/view_gpu.c (FRAME_CPU)
@group(0) @binding(4) var<storage, read> color: array<u32>;
@group(0) @binding(5) var<storage, read> depthbuf: array<f32>;
fn frame_rgb(x: u32, y: u32) -> vec3f {
    let c = color[y * u32(F.size.x) + x];
    return vec3f(f32((c >> 16u) & 255u), f32((c >> 8u) & 255u), f32(c & 255u)) / 255.0;
}
fn frame_depth(x: u32, y: u32) -> f32 { return depthbuf[y * u32(F.size.x) + x]; }
fn traced(x: u32, y: u32) -> vec4f { return vec4f(0.0); }
