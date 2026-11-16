// fluid_csolid.wgsl - used by engine/gpu/fluid_gpu.c (K_CSOLID)
@group(0) @binding(1) var<storage, read> fine: array<u32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ccount()) { return; }
    let q = ccell(id);
    var all_solid = 1u;
    for (var k = 0; k < 8; k++) {
        let a = 2 * q + vec3i(k & 1, (k >> 1) & 1, k >> 2);
        if (any(a >= vec3i(L.d.xyz))) { continue; }
        if (fine[lidx(a)] == 0u) { all_solid = 0u; }
    }
    coarse[id] = all_solid;
}
