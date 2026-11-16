// rigid_meshflush.wgsl - used by engine/gpu/rigid_gpu.c (K_MESHFLUSH)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;   /* (declared for K_CONVEX's functions, not used here) */
@group(0) @binding(8) var<storage, read> S: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> sc: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = tid(gid, nw);
    if (k >= min(atomicLoad(&pc[2]), P.m.z)) { return; }
    let pr = pairs[2u * P.m.z + k];
    let n = min(atomicLoad(&sc[k]), 64u);
    nc = 0u;
    for (var q = 0u; q < n; q++) {
        let o = 2u * (64u * k + q);
        cand(S[o].xyz, S[o + 1u].xyz, S[o].w);
    }
    flush(pr.x, pr.y);
    atomicStore(&sc[k], 0u);
}
