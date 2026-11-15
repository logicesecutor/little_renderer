// view_light.wgsl - used by engine/gpu/view_gpu.c (LIGHT)
@group(0) @binding(3) var<storage, read_write> light: array<f32>;
@group(0) @binding(4) var<uniform> S: Sel;
@group(0) @binding(5) var<storage, read> VS: array<VStat, 2>;

/* optical depth of all media from p to the lamp (midpoint rule, about a voxel per step) */
fn optical_depth(a: vec3f, b: vec3f) -> f32 {
    let d = b - a;
    let len = length(d);
    if (len < 1e-6) { return 0.0; }
    let dir = d / len;
    let inv = safe_inv(dir);
    var tau = 0.0;
    for (var k = 0u; k < u32(F.size.w); k++) {
        if (!vol_active(k)) { continue; }
        let v = F.vols[k];
        let t = ray_box(a, inv, VS[k].alo.xyz, VS[k].ahi.xyz);
        let t0 = max(t.x, 0.0);
        let t1 = min(t.y, len);
        if (t1 <= t0) { continue; }
        var n = i32(ceil((t1 - t0) / v.p.w));
        n = clamp(n, 1, 256);
        let ds = (t1 - t0) / f32(n);
        var sum = 0.0;
        for (var i = 0; i < n; i++) { sum += density(k, a + dir * (t0 + (f32(i) + 0.5) * ds)); }
        tau += v.albedo.w * sum * ds;
    }
    return tau;
}

@compute @workgroup_size(4, 4, 4)
fn light_grid(@builtin(global_invocation_id) id: vec3u) {
    let k = S.k.x;
    let v = F.vols[k];
    if (any(id >= v.ldims.xyz)) { return; }
    let size = v.hi.xyz - v.lo.xyz;
    let cell = size / vec3f(v.ldims.xyz);
    let c = v.lo.xyz + (vec3f(id) + 0.5) * cell;
    let idx = v.ldims.w + (id.z * v.ldims.y + id.y) * v.ldims.x + id.x;
    if (!vol_active(k) || any(c < VS[k].alo.xyz - cell) || any(c > VS[k].ahi.xyz + cell)) { light[idx] = 0.0; return; }
    light[idx] = optical_depth(c, F.lamp_pos.xyz);
}
