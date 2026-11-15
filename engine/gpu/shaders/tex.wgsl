// tex.wgsl - used by engine/gpu/tex_gpu.c (GPU_WGSL_TEX)
/* The includer declares the atlas, var TEXA: texture_2d<f32>, and fn tex_entry(k: u32) -> vec4u,
   the table: entry t (t < 32) is texture t's first rectangle, its levels, srgb, its size
   (w | h << 16); from entry 32 on the rectangles, (x, y, w, h). */
fn tex_srgb(c: f32) -> f32 { return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045); }
/* how coordinates outside 0..1 read (TexMap.wrap_u, wrap_v): 0 repeat, 1 clamp, 2 mirror; a
   mirrored or clamped coordinate is folded into 0..1 first, its texels then clamped */
fn tex_fold(u: f32, mode: u32) -> f32 {
    if (mode == 1u) { return clamp(u, 0.0, 1.0); }
    if (mode == 2u) { let m = u - 2.0 * floor(u * 0.5); return select(m, 2.0 - m, m > 1.0); }
    return u - floor(u);
}
fn tex_index(x: i32, n: i32, mode: u32) -> i32 {
    if (mode == 0u) { return select(select(x, x - n, x >= n), x + n, x < 0); }
    return clamp(x, 0, n - 1);
}
fn tex_texel(t: u32, level: u32, x: i32, y: i32) -> vec4f {
    let e = tex_entry(t);
    let r = tex_entry(e.x + level);
    var c = textureLoad(TEXA, vec2i(i32(r.x) + x, i32(r.y) + y), 0);
    if ((e.z & 1u) != 0u) { c = vec4f(tex_srgb(c.x), tex_srgb(c.y), tex_srgb(c.z), c.w); }
    return c;
}
fn tex_bilinear(t: u32, level: u32, uv: vec2f) -> vec4f {
    let e = tex_entry(t);
    let r = tex_entry(e.x + level);
    let mu = (e.z >> 1u) & 3u;
    let mv = (e.z >> 3u) & 3u;
    let w = i32(r.z);
    let h = i32(r.w);
    let x = tex_fold(uv.x, mu) * f32(w) - 0.5;
    let y = tex_fold(uv.y, mv) * f32(h) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let xa = tex_index(i32(x0), w, mu);
    let xb = tex_index(i32(x0) + 1, w, mu);
    let ya = tex_index(i32(y0), h, mv);
    let yb = tex_index(i32(y0) + 1, h, mv);
    let a = tex_texel(t, level, xa, ya);
    let b = tex_texel(t, level, xb, ya);
    let c = tex_texel(t, level, xa, yb);
    let d = tex_texel(t, level, xb, yb);
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}
/* texture t at uv, level lod (fractional; <= 0: level 0, bilinear) */
fn tex_sample(t: u32, uv0: vec2f, lod: f32) -> vec4f {
    var uv = uv0;
    if (!(abs(uv.x) < 1e6 && abs(uv.y) < 1e6)) { uv = vec2f(0.0); }
    let levels = tex_entry(t).y;
    if (!(lod > 0.0)) { return tex_bilinear(t, 0u, uv); }
    if (lod >= f32(levels - 1u)) { return tex_bilinear(t, levels - 1u, uv); }
    let k = u32(lod);
    let f = lod - f32(k);
    let a = tex_bilinear(t, k, uv);
    if (f <= 0.0) { return a; }
    return mix(a, tex_bilinear(t, k + 1u, uv), f);
}
/* scene.c tex_at: the level for a footprint of 2^uv_lod texture coordinates (scaled by k) */
fn tex_at(t: u32, uv: vec2f, uv_lod: f32, k: f32) -> vec4f {
    var lod = -1000.0;
    if (uv_lod > -500.0) {
        let e = tex_entry(t).w;
        lod = uv_lod + log2(k * f32(max(e & 0xffffu, e >> 16u)));
    }
    return tex_sample(t, uv, lod);
}
/* The material's textures (scene.c material_at, material_normal, material_occlusion), packed
   (gpu_tex_pack): x = base | emissive << 8, y = metal-roughness | occlusion << 8 | its strength
   (0..255) << 16, z = normal | its scale (1/4096) << 8, each 1 + index (0 none); w: the uv
   scale's bits */
struct TexMat { base: vec3f, metallic: f32, rough: f32, emission: vec3f };
fn tex_scale(tex: vec4u) -> f32 { return select(1.0, bitcast<f32>(tex.w), bitcast<f32>(tex.w) > 0.0); }
fn tex_material(tex: vec4u, base: vec3f, metallic: f32, rough: f32, emission: vec3f, uv0: vec2f, uv_lod: f32) -> TexMat {
    let k = tex_scale(tex);
    let uv = uv0 * k;
    var r = TexMat(base, metallic, rough, emission);
    let tb = tex.x & 255u;
    let te = (tex.x >> 8u) & 255u;
    let tm = tex.y & 255u;
    if (tb > 0u) { r.base *= tex_at(tb - 1u, uv, uv_lod, k).rgb; }
    if (tm > 0u) {
        let c = tex_at(tm - 1u, uv, uv_lod, k);
        r.rough *= c.y;
        r.metallic *= c.z;
    }
    if (te > 0u) { r.emission *= tex_at(te - 1u, uv, uv_lod, k).rgb; }
    return r;
}
fn tex_occlusion(tex: vec4u, uv0: vec2f, uv_lod: f32) -> f32 {
    let to = (tex.y >> 8u) & 255u;
    if (to == 0u) { return 1.0; }
    let k = tex_scale(tex);
    let s = f32((tex.y >> 16u) & 255u) / 255.0;
    return 1.0 + s * (tex_at(to - 1u, uv0 * k, uv_lod, k).x - 1.0);
}
fn tex_normal(tex: vec4u, uv0: vec2f, uv_lod: f32, n: vec3f, t: vec4f) -> vec3f {
    let tn = tex.z & 255u;
    if (tn == 0u) { return n; }
    let k = tex_scale(tex);
    let ns = f32(tex.z >> 8u) / 4096.0;
    let c = tex_at(tn - 1u, uv0 * k, uv_lod, k);
    var tt = t.xyz - n * dot(n, t.xyz);
    let l = length(tt);
    if (!(l > 1e-6)) { return n; }
    tt /= l;
    let b = cross(n, tt) * t.w;
    let o = tt * (ns * (2.0 * c.x - 1.0)) + b * (ns * (2.0 * c.y - 1.0)) + n * (2.0 * c.z - 1.0);
    let lo = length(o);
    return select(n, o / lo, lo > 1e-6);
}
/* a direction as 2 numbers (octahedral) and back: the rasterizer's tangents per vertex */
fn oct_decode(e: vec2f) -> vec3f {
    var v = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        let s = vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        let xy = (vec2f(1.0) - abs(v.yx)) * s;
        v = vec3f(xy.x, xy.y, v.z);
    }
    return normalize(v);
}
