// fluid_forces.wgsl - used by engine/gpu/fluid_gpu.c (K_FORCES)
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read> dens: array<f32>;
/* volume_density: trilinear, 0 outside the grid */
fn vol_density(p: vec3f) -> f32 {
    let n = vec3f(P.dn.xyz);
    let g = (p - P.dlo.xyz) / (P.dhi.xyz - P.dlo.xyz) * n - 0.5;
    if (!(all(g > vec3f(-1.0)) && all(g < n))) { return 0.0; }
    let i0 = vec3i(floor(g));
    let f = g - vec3f(i0);
    let ni = DN();
    var c: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let i = i0 + vec3i(k & 1, (k >> 1) & 1, k >> 2);
        var val = 0.0;
        if (all(i >= vec3i(0)) && all(i < ni)) { val = dens[u32((i.z * ni.y + i.y) * ni.x + i.x)]; }
        c[k] = val;
    }
    let c00 = mix(c[0], c[1], f.x); let c10 = mix(c[2], c[3], f.x);
    let c01 = mix(c[4], c[5], f.x); let c11 = mix(c[6], c[7], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    let a = P.a.z;
    if (q.y < NY() && q.z < NZ()) { let i = iu(q.x, q.y, q.z); u[i] += (P.wind.x - u[i]) * a; }
    if (q.x < NX() && q.z < NZ()) {
        let i = iv(q.x, q.y, q.z);
        var val = v[i];
        val += (P.wind.y - val) * a;
        if (P.dn.w != 0u && P.a.w != 0.0) { val += P.a.w * vol_density(pos_v(q)); }
        v[i] = val;
    }
    if (q.x < NX() && q.y < NY()) { let i = iw(q.x, q.y, q.z); w[i] += (P.wind.z - w[i]) * a; }
}
