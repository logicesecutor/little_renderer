// rigid_apply.wgsl - used by engine/gpu/rigid_gpu.c (K_APPLY)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read> C: array<Contact>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read> L: array<u32>;
@group(0) @binding(5) var<storage, read_write> V: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let lane = lv.x & 7u;
    let i = (wid.x + wid.y * nw.x) * 8u + (lv.x >> 3u);
    let live = i < P.n.x;
    var lo = 0u;
    var hi = 0u;
    if (live && B[i].vel.w != 0.0) { lo = off[i]; hi = off[i + 1u]; }
    let nch = chunks(lv.x, hi - lo);
    var dv = vec3f(0.0);
    var dw = vec3f(0.0);
    for (var k = 0u; k < nch; k++) {
        var cv = vec3f(0.0);
        var cw = vec3f(0.0);
        let q = lo + 8u * k + lane;
        if (q < hi) {
            let e = L[q];
            let ci = e >> 1u;
            var j = C[ci].imp.xyz;
            var r: vec3f;
            if ((e & 1u) != 0u) { j = -j; r = C[ci].rb.xyz; } else { r = C[ci].ra.xyz; }
            cv = j;
            cw = cross(r, j);
        }
        sdv[lv.x] = cv;
        sdw[lv.x] = cw;
        workgroupBarrier();
        if (lane == 0u) { for (var l = 0u; l < 8u; l++) { dv += sdv[lv.x + l]; dw += sdw[lv.x + l]; } }
        workgroupBarrier();
    }
    if (!live || lane != 0u) { return; }
    let b = B[i];
    V[2u * i] = vec4f(b.vel.xyz + dv * b.vel.w, b.vel.w);
    V[2u * i + 1u] = vec4f(b.ang.xyz + inv_inertia(b) * dw, b.ang.w);
}
