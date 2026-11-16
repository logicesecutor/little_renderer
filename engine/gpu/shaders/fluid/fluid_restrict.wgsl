// fluid_restrict.wgsl - used by engine/gpu/fluid_gpu.c (K_RESTRICT)
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read> res: array<f32>;
@group(0) @binding(3) var<storage, read_write> crhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> cp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ccount()) { return; }
    let q = ccell(id);
    var s = 0.0;
    var n = 0;
    for (var k = 0; k < 8; k++) {
        let a = 2 * q + vec3i(k & 1, (k >> 1) & 1, k >> 2);
        if (any(a >= vec3i(L.d.xyz))) { continue; }
        let fi = lidx(a);
        if (links[fi] == 0u) { continue; }
        s += res[fi];
        n++;
    }
    crhs[id] = select(0.0, s / f32(n), n > 0);
    cp[id] = 0.0;
}
