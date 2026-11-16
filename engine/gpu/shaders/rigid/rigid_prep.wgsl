// rigid_prep.wgsl - used by engine/gpu/rigid_gpu.c (K_PREP)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> off: array<u32>;
@group(0) @binding(5) var<storage, read> Cp: array<Contact>;
@group(0) @binding(6) var<storage, read> Lp: array<u32>;
@group(0) @binding(7) var<storage, read> offp: array<u32>;
@group(0) @binding(8) var<storage, read_write> J0: array<vec4f>;

fn eff(ma: f32, mb: f32, ia: mat3x3f, ib: mat3x3f, ra: vec3f, rb: vec3f, d: vec3f) -> f32 {
    let k = ma + mb + dot(d, cross(ia * cross(ra, d), ra)) + dot(d, cross(ib * cross(rb, d), rb));
    return select(0.0, 1.0 / k, k > 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c >= min(atomicLoad(&cnt[0]), P.n.y)) { return; }
    var k = C[c];
    let ia = bitcast<u32>(k.n.w);
    let ib = bitcast<u32>(k.ra.w);
    let a = B[ia];
    let n = k.n.xyz;
    let p = k.p.xyz;
    let ra = qmat(a.rot) * k.la.xyz;
    var pb = k.lb.xyz;
    let sa = f32(off[ia + 1u] - off[ia]);
    let Ia = inv_inertia(a) * sa;
    let ma = a.vel.w * sa;
    var rb = vec3f(0.0);
    var Ib = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var mb = 0.0;
    var vb = vec3f(0.0);
    var e = max(a.ang.w, P.lo.w);
    var mu = sqrt(a.half.w * P.hi.w);
    if (ib != NONE) {
        let b = B[ib];
        rb = qmat(b.rot) * k.lb.xyz;
        pb = b.pos.xyz + rb;
        let sb = f32(off[ib + 1u] - off[ib]);
        Ib = inv_inertia(b) * sb;
        mb = b.vel.w * sb;
        vb = b.vel.xyz + cross(b.ang.xyz, rb);
        e = max(a.ang.w, b.ang.w);
        mu = sqrt(a.half.w * b.half.w);
    }
    let t = onb(n);
    let vn = dot(a.vel.xyz + cross(a.ang.xyz, ra) - vb, n);
    let depth = k.la.w - dot(a.pos.xyz + ra - pb, n);
    let h = P.h.x;
    /* apart: they may approach until they touch, no closer; touching: the overlap beyond the
       slop is pushed out a little each substep, and they bounce if they approached fast when
       the step began (rigid.c) */
    var j = k.j.xyz;
    if (P.m.w == 0u) {
        /* the step's first substep: the normal speed now, and the same contact last step
           (warm start: its impulses; a bounce it still owed: its speed then) */
        j = vec3f(0.0);
        k.lb.w = vn;
        if (ia < P.n.z) {
            for (var q = offp[ia]; q < offp[ia + 1u]; q++) {
                let e2 = Lp[q];
                if ((e2 & 1u) != 0u) { continue; }
                let o = Cp[e2 >> 1u];
                if (bitcast<u32>(o.ra.w) != ib) { continue; }
                let dd = o.p.xyz - p;
                if (dot(dd, dd) > 1.0 || dot(o.n.xyz, n) < 0.95) { continue; }
                j = o.j.xyz;
                if (o.j.x > 0.0) { k.lb.w = min(vn, o.lb.w); }
                break;
            }
        }
    }
    let bounce = select(0.0, -e * k.lb.w, depth > -P.d.z && k.lb.w < -P.h.w);
    let bias = select(select(min(P.h.z / h * max(depth - P.h.y, 0.0), P.d.w), depth / h, depth < 0.0), bounce, bounce > 0.0);
    if (bounce > 0.0) { j = vec3f(0.0); k.lb.w = 0.0; }
    k.p = vec4f(p, depth);
    k.ra = vec4f(ra, k.ra.w);
    k.rb = vec4f(rb, bias);
    k.t1 = vec4f(t[0], eff(ma, mb, Ia, Ib, ra, rb, n));
    k.t2 = vec4f(t[1], eff(ma, mb, Ia, Ib, ra, rb, t[0]));
    k.j = vec4f(j, eff(ma, mb, Ia, Ib, ra, rb, t[1]));
    k.imp = vec4f(n * j.x + t[0] * j.y + t[1] * j.z, mu);
    C[c] = k;
    J0[c] = k.j;                /* the impulses the iterations start from (K_SOLVE) */
}
