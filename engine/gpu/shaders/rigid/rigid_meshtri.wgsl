// rigid_meshtri.wgsl - used by engine/gpu/rigid_gpu.c (K_MESHTRI)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
@group(0) @binding(7) var<storage, read> T: array<vec2u>;
@group(0) @binding(8) var<storage, read_write> S: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> sc: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let e = tid(gid, nw);
    if (e >= min(atomicLoad(&pc[3]), 64u * P.m.z)) { return; }
    let t = T[e];
    let pr = pairs[2u * P.m.z + t.x];
    nc = 0u;
    MG = margin_of(pr.x, pr.y);
    mesh_tri(pr.x, pr.y, t.y);
    for (var q = 0u; q < nc; q++) {
        let slot = atomicAdd(&sc[t.x], 1u);
        if (slot >= 64u) { break; }
        let o = 2u * (64u * t.x + slot);
        S[o] = vec4f(cp[q], cd[q]);
        S[o + 1u] = vec4f(cn[q], 0.0);
    }
}
