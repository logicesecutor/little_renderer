// rigid_solve.wgsl - used by engine/gpu/rigid_gpu.c (K_SOLVE)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read> L: array<u32>;
@group(0) @binding(5) var<storage, read> Vi: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> Vo: array<vec4f>;
@group(0) @binding(7) var<storage, read> Ji: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> Jo: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let lane = lv.x & 7u;
    let i = (wid.x + wid.y * nw.x) * 8u + (lv.x >> 3u);
    let live = i < P.n.x;
    var w = 0.0;
    var lo = 0u;
    var hi = 0u;
    if (live) { w = B[i].vel.w; lo = off[i]; hi = off[i + 1u]; }
    let nch = chunks(lv.x, hi - lo);
    var dv = vec3f(0.0);
    var dw = vec3f(0.0);
    for (var k = 0u; k < nch; k++) {
        var cv = vec3f(0.0);
        var cw = vec3f(0.0);
        let q = lo + 8u * k + lane;
        var e = 0u;
        if (q < hi) { e = L[q]; }
        let second = (e & 1u) != 0u;
        if (q < hi && !(w == 0.0 && second)) {
            let c = e >> 1u;
            /* the 6 of the contact's 10 vec4s it uses (and its impulses so far, Ji) */
            let kn = C[c].n;
            let kra = C[c].ra;
            let krb = C[c].rb;
            let kt1 = C[c].t1;
            let kt2 = C[c].t2;
            let mu = C[c].imp.w;
            let kj = Ji[c];
            let ia = bitcast<u32>(kn.w);
            let ib = bitcast<u32>(kra.w);
            var d = Vi[2u * ia].xyz + cross(Vi[2u * ia + 1u].xyz, kra.xyz);
            if (ib != NONE) { d = d - (Vi[2u * ib].xyz + cross(Vi[2u * ib + 1u].xyz, krb.xyz)); }
            let n = kn.xyz;
            let jn = max(kj.x + kt1.w * (krb.w - dot(d, n)), 0.0);
            let lim = mu * jn;
            let jt1 = clamp(kj.y - kt2.w * dot(d, kt1.xyz), -lim, lim);
            let jt2 = clamp(kj.z - kj.w * dot(d, kt2.xyz), -lim, lim);
            let imp = n * (jn - kj.x) + kt1.xyz * (jt1 - kj.y) + kt2.xyz * (jt2 - kj.z);
            /* both bodies' threads must get the very same impulse (or momentum is not kept,
               and piles never rest): the same instructions for both, the side applied after
               (by an exact sign), not in two branches the compiler may round differently */
            cv = imp * select(1.0, -1.0, second);
            cw = cross(select(kra.xyz, krb.xyz, second), cv);
            if (!second) {
                Jo[c] = vec4f(jn, jt1, jt2, kj.w);
                C[c].j = vec4f(jn, jt1, jt2, kj.w);
            }
        }
        sdv[lv.x] = cv;
        sdw[lv.x] = cw;
        workgroupBarrier();
        if (lane == 0u) { for (var l = 0u; l < 8u; l++) { dv += sdv[lv.x + l]; dw += sdw[lv.x + l]; } }
        workgroupBarrier();
    }
    if (!live || lane != 0u) { return; }
    let v = Vi[2u * i];
    let a = Vi[2u * i + 1u];
    if (w == 0.0) { Vo[2u * i] = v; Vo[2u * i + 1u] = a; return; }
    Vo[2u * i] = vec4f(v.xyz + dv * w, w);
    Vo[2u * i + 1u] = vec4f(a.xyz + inv_inertia(B[i]) * dw, a.w);
}
