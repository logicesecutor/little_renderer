// rigid_meshflush.wgsl - used by engine/gpu/rigid_gpu.c (K_MESHFLUSH)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;   /* (declared for K_CONVEX's functions, not used here) */
@group(0) @binding(8) var<storage, read> S: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> sc: array<atomic<u32>>;
/* candidate i of pair k before candidate j */
fn before(k: u32, i: u32, j: u32) -> bool {
    let a = S[2u * (64u * k + i)];
    let b = S[2u * (64u * k + j)];
    if (a.w != b.w) { return a.w > b.w; }
    let ua = bitcast<vec3u>(a.xyz);
    let ub = bitcast<vec3u>(b.xyz);
    if (ua.x != ub.x) { return ua.x < ub.x; }
    if (ua.y != ub.y) { return ua.y < ub.y; }
    if (ua.z != ub.z) { return ua.z < ub.z; }
    let na = bitcast<vec3u>(S[2u * (64u * k + i) + 1u].xyz);
    let nb = bitcast<vec3u>(S[2u * (64u * k + j) + 1u].xyz);
    if (na.x != nb.x) { return na.x < nb.x; }
    if (na.y != nb.y) { return na.y < nb.y; }
    return na.z < nb.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = tid(gid, nw);
    if (k >= min(atomicLoad(&pc[2]), P.m.z)) { return; }
    let pr = pairs[2u * P.m.z + k];
    let n = min(atomicLoad(&sc[k]), 64u);
    /* the candidates in an order of their own (K_MESHTRI's threads stored them in the order they
       came, which changes from run to run, and cand / flush keep the first of equals): the
       deepest first, then by the point's bits */
    var ord: array<u32, 64>;
    for (var q = 0u; q < n; q++) {
        var r = q;
        while (r > 0u && before(k, q, ord[r - 1u])) { ord[r] = ord[r - 1u]; r--; }
        ord[r] = q;
    }
    nc = 0u;
    for (var q = 0u; q < n; q++) {
        let o = 2u * (64u * k + ord[q]);
        cand(S[o].xyz, S[o + 1u].xyz, S[o].w);
    }
    flush(pr.x, pr.y);
    atomicStore(&sc[k], 0u);
}
