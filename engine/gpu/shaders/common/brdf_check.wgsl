// brdf_check.wgsl - used by engine/gpu/brdf_gpu.c (CHECK)
struct Tab { t: array<vec4f, GPU_BSDF_TAB_VEC4> };
@group(0) @binding(0) var<uniform> TB: Tab;
@group(0) @binding(1) var<storage, read> cin: array<f32>;
@group(0) @binding(2) var<storage, read_write> cout: array<f32>;
fn bsdf_tab(k: u32) -> f32 { return TB.t[k >> 2u][k & 3u]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i * 17u >= arrayLength(&cin)) { return; }
    let o = i * 17u;
    let n = vec3f(0.0, 0.0, 1.0);
    let v = vec3f(cin[o + 8u], cin[o + 9u], cin[o + 10u]);
    let l = vec3f(cin[o + 11u], cin[o + 12u], cin[o + 13u]);
    var b = bsdf_make(vec3f(cin[o], cin[o + 1u], cin[o + 2u]), cin[o + 3u], cin[o + 4u], cin[o + 5u], cin[o + 6u], cin[o + 7u] > 0.5);
    let fp = bsdf_eval(&b, n, v, l);
    var b2 = bsdf_make(vec3f(cin[o], cin[o + 1u], cin[o + 2u]), cin[o + 3u], cin[o + 4u], cin[o + 5u], cin[o + 6u], cin[o + 7u] > 0.5);
    let s = bsdf_sample(&b2, n, v, cin[o + 14u], vec2f(cin[o + 15u], cin[o + 16u]));
    let r = i * 16u;
    cout[r] = fp.x; cout[r + 1u] = fp.y; cout[r + 2u] = fp.z; cout[r + 3u] = fp.w;
    cout[r + 4u] = select(0.0, 1.0, s.ok); cout[r + 5u] = select(0.0, 1.0, s.delta);
    cout[r + 6u] = select(0.0, 1.0, s.through); cout[r + 7u] = s.pdf;
    cout[r + 8u] = s.l.x; cout[r + 9u] = s.l.y; cout[r + 10u] = s.l.z;
    cout[r + 11u] = s.w.x; cout[r + 12u] = s.w.y; cout[r + 13u] = s.w.z;
    cout[r + 14u] = b2.p.x; cout[r + 15u] = b2.p.y;
}
