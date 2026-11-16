// view_frame_gpu.wgsl - used by engine/gpu/view_gpu.c (FRAME_GPU)
@group(0) @binding(4) var scene_color: texture_2d<f32>;
@group(0) @binding(5) var scene_depth: texture_depth_2d;
fn frame_rgb(x: u32, y: u32) -> vec3f { return textureLoad(scene_color, vec2u(x, y), 0).rgb; }
fn frame_depth(x: u32, y: u32) -> f32 { return textureLoad(scene_depth, vec2u(x, y), 0); }
/* the path tracer's samples so far (rgb = their sum, a = how many), when its image is shown */
@group(0) @binding(7) var<storage, read> accum: array<vec4f>;
fn traced(x: u32, y: u32) -> vec4f {
    if (F.right.w < 0.5) { return vec4f(0.0); }
    return accum[y * u32(F.size.x) + x];
}
