// media.wgsl - used by engine/gpu/view_gpu.c (GPU_WGSL_MEDIA)
@group(0) @binding(1) var<storage, read> dens: array<f32>;
@group(0) @binding(2) var<storage, read> bricks: array<f32>;

fn density(k: u32, p: vec3f) -> f32 {
    let v = F.vols[k];
    let n = vec3f(v.dims.xyz);
    let g = (p - v.lo.xyz) / (v.hi.xyz - v.lo.xyz) * n - 0.5;
    if (any(g <= vec3f(-1.0)) || any(g >= n)) { return 0.0; }
    let fl = floor(g);
    let i0 = vec3i(fl);
    let f = g - fl;
    let ni = vec3i(v.dims.xyz);
    var c: array<f32, 8>;
    for (var q = 0; q < 8; q++) {
        let i = i0 + vec3i(q & 1, (q >> 1) & 1, q >> 2);
        var val = 0.0;
        if (all(i >= vec3i(0)) && all(i < ni)) {
            val = dens[v.dims.w + (u32(i.z) * v.dims.y + u32(i.y)) * v.dims.x + u32(i.x)];
        }
        c[q] = val;
    }
    let c00 = mix(c[0], c[1], f.x); let c10 = mix(c[2], c[3], f.x);
    let c01 = mix(c[4], c[5], f.x); let c11 = mix(c[6], c[7], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn brick_majorant(k: u32, p: vec3f) -> f32 {
    let v = F.vols[k];
    let g = (p - v.lo.xyz) / (v.hi.xyz - v.lo.xyz) * vec3f(v.dims.xyz);
    if (any(g < vec3f(0.0))) { return 0.0; }
    let b = vec3u(g / 8.0);
    if (any(b >= v.bdims.xyz)) { return 0.0; }
    return bricks[v.bdims.w + (b.z * v.bdims.y + b.y) * v.bdims.x + b.x];
}

fn ray_box(o: vec3f, inv: vec3f, lo: vec3f, hi: vec3f) -> vec2f {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let a = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let b = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return vec2f(a, b);
}

fn safe_inv(d: vec3f) -> vec3f {
    return vec3f(select(1e30, 1.0 / d.x, abs(d.x) > 1e-12), select(1e30, 1.0 / d.y, abs(d.y) > 1e-12),
                 select(1e30, 1.0 / d.z, abs(d.z) > 1e-12));
}

fn vol_active(k: u32) -> bool { return VS[k].mx.x > 0.0; }
