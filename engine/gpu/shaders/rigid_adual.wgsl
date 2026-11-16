// rigid_adual.wgsl - used by engine/gpu/rigid_gpu.c (K_ADUAL)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> AB: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> CTL: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c == 0u) { atomicStore(&CTL[0], 0u); }          /* (the next primal passes: colour 0 first) */
    if (c >= min(atomicLoad(&cnt[0]), P.n.y)) { return; }
    /* each row's force, its stiffness up while the force is within its bounds */
    var k = C[c];
    let cs = row_cs(k, false);
    var lam = k.j.xyz;
    var kk = k.imp.xyz;
    for (var r = 0u; r < 3u; r++) {
        k.j = vec4f(lam, k.j.w);                    /* (the tangents' bounds: the new normal force) */
        let lo = row_lo(k, r);
        let hi = row_hi(k, r);
        let l = clamp(kk[r] * cs[r] + lam[r], lo, hi);
        lam[r] = l;
        if (l > lo && l < hi) { kk[r] = min(kk[r] + BETA * k.j.w * abs(cs[r]), KMAX * k.j.w); }
    }
    C[c].j = vec4f(lam, k.j.w);
    C[c].imp = vec4f(kk, k.imp.w);
}
