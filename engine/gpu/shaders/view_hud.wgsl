// view_hud.wgsl - used by engine/gpu/view_gpu.c (HUD)
struct HudU { r: vec4f, s: vec4f };   /* x, y, w, h of the box in pixels; the canvas's size */
@group(0) @binding(0) var<uniform> U: HudU;
@group(0) @binding(1) var<storage, read> px: array<u32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    var c = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
    let p = U.r.xy + c[i] * U.r.zw;
    return vec4f(p.x / U.s.x * 2.0 - 1.0, 1.0 - p.y / U.s.y * 2.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let x = u32(pos.x - U.r.x);
    let y = u32(pos.y - U.r.y);
    let c = px[y * u32(U.r.z) + x];
    return vec4f(f32((c >> 16u) & 255u), f32((c >> 8u) & 255u), f32(c & 255u), f32(c >> 24u)) / 255.0;
}
