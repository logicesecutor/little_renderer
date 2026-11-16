// rigid_arows.wgsl - used by engine/gpu/rigid_gpu.c (K_AROWS)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> AB: array<vec4f>;
@group(0) @binding(5) var<storage, read> Cp: array<Contact>;
@group(0) @binding(6) var<storage, read> Lp: array<u32>;
@group(0) @binding(7) var<storage, read> offp: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c >= min(atomicLoad(&cnt[0]), P.n.y)) { return; }
    var k = C[c];
    let ia = bitcast<u32>(k.n.w);
    let ib = bitcast<u32>(k.ra.w);
    let n = k.n.xyz;
    let ra = qmat(AB[8u * ia + 1u]) * k.la.xyz;
    let pa = AB[8u * ia].xyz + ra;
    var rb = vec3f(0.0);
    var pb = k.lb.xyz;
    if (ib != NONE) {
        rb = qmat(AB[8u * ib + 1u]) * k.lb.xyz;
        pb = AB[8u * ib].xyz + rb;
    }
    let t = onb(n);
    let dp = pa - pb;
    if (P.m.w == 0u) {
        /* the step's first substep: friction, the stiffnesses' scale (m / h^2 of the lighter
           moving body), the forces and stiffnesses of the same contact last step (k decayed),
           and the approach speed to bounce from */
        let a = B[ia];
        var mu = sqrt(a.half.w * P.hi.w);
        var ma = select(3.0e38, 1.0 / a.vel.w, a.vel.w > 0.0);
        var mb = 3.0e38;
        var vb = vec3f(0.0);
        if (ib != NONE) {
            let b = B[ib];
            mu = sqrt(a.half.w * b.half.w);
            mb = select(3.0e38, 1.0 / b.vel.w, b.vel.w > 0.0);
            vb = b.vel.xyz + cross(b.ang.xyz, rb);
        }
        let h = P.h.x;
        let m = min(ma, mb);
        let ms = select(1.0, m / (h * h), m < 3.0e38);
        var lam = vec3f(0.0);
        var kk = vec3f(0.0);
        if (ia < P.n.z) {
            for (var q = offp[ia]; q < offp[ia + 1u]; q++) {
                let e2 = Lp[q];
                if ((e2 & 1u) != 0u) { continue; }
                let o = Cp[e2 >> 1u];
                if (bitcast<u32>(o.ra.w) != ib) { continue; }
                let dd = o.p.xyz - k.p.xyz;
                if (dot(dd, dd) > 1.0 || dot(o.n.xyz, n) < 0.95) { continue; }
                lam = o.j.xyz;
                kk = o.imp.xyz;
                break;
            }
        }
        kk = clamp(GAMMA * kk, vec3f(KMIN * ms), vec3f(KMAX * ms));
        k.j = vec4f(lam, ms);
        k.imp = vec4f(kk, mu);
        k.lb.w = dot(a.vel.xyz + cross(a.ang.xyz, ra) - vb, n);
    }
    /* the rows at the substep's start: arms, tangents, C0 (the normal's: its separation, the
       depth when found less how far the points have moved apart since, with the margin) */
    k.ra = vec4f(ra, k.ra.w);
    k.rb = vec4f(rb, -k.la.w + dot(n, dp) + MARGIN);
    k.t1 = vec4f(t[0], dot(t[0], dp));
    k.t2 = vec4f(t[1], dot(t[1], dp));
    C[c] = k;
}
