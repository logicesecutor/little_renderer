// rigid_abounce.wgsl - used by engine/gpu/rigid_gpu.c (K_ABOUNCE)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read> C: array<Contact>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read> L: array<u32>;
@group(0) @binding(5) var<storage, read_write> V: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }
    let me = B[i];
    if (me.vel.w == 0.0) { return; }
    /* the bounce: a pushing contact that approached faster than the threshold comes back at
       e times that speed; each contact's impulse from the velocities after the solve, by both
       of its bodies alike (into V; K_AEND takes them) */
    var v = me.vel.xyz;
    var w = me.ang.xyz;
    for (var q = off[i]; q < off[i + 1u]; q++) {
        let e = L[q];
        let k = C[e >> 1u];
        if (k.j.x == 0.0 || k.lb.w >= -P.h.w) { continue; }
        let ia = bitcast<u32>(k.n.w);
        let ib = bitcast<u32>(k.ra.w);
        let a = B[ia];
        let n = k.n.xyz;
        let ra = qmat(a.rot) * k.la.xyz;
        let Ia = inv_inertia(a);
        var den = a.vel.w + dot(n, cross(Ia * cross(ra, n), ra));
        var vb = vec3f(0.0);
        var rb = vec3f(0.0);
        var Ib = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
        var rest = max(a.ang.w, P.lo.w);
        if (ib != NONE) {
            let b = B[ib];
            rb = qmat(b.rot) * k.lb.xyz;
            Ib = inv_inertia(b);
            den += b.vel.w + dot(n, cross(Ib * cross(rb, n), rb));
            vb = b.vel.xyz + cross(b.ang.xyz, rb);
            rest = max(a.ang.w, b.ang.w);
        }
        let vn = dot(a.vel.xyz + cross(a.ang.xyz, ra) - vb, n);
        let want = -rest * k.lb.w;
        if (vn >= want || den <= 0.0) { continue; }
        let j = (want - vn) / den;
        if ((e & 1u) == 0u) { v += n * (j * a.vel.w); w += Ia * cross(ra, n) * j; }
        else { v -= n * (j * me.vel.w); w -= Ib * cross(rb, n) * j; }
    }
    V[2u * i] = vec4f(v, 0.0);
    V[2u * i + 1u] = vec4f(w, 0.0);
}
