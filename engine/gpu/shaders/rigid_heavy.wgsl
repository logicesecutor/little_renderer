// rigid_heavy.wgsl - used by engine/gpu/rigid_gpu.c (K_HEAVY)
var<workgroup> ws: array<f32, 64>;
var<workgroup> wi: array<u32, 64>;
var<workgroup> wj: array<u32, 64>;
var<workgroup> wl: array<vec3f, 64>;
var<workgroup> wsep: u32;
var<workgroup> wsepa: atomic<u32>;
var<workgroup> wcount: u32;
/* The pair's corners and face normals in the world frame, made once for the whole workgroup
   (the tests below read each many times: a mat3 product and a load each time otherwise), when
   both hulls fit: a's at [0, HC) and [0, HF), b's at [HC, 2 HC) and [HF, 2 HF) */
const HC = 64u;
const HF = 128u;
const HE = 192u;
var<workgroup> hv: array<vec3f, 128>;
var<workgroup> hn: array<vec3f, 256>;
var<workgroup> he: array<vec2u, 192>;      /* b's edges (G's a, b, fa, fb), 16 bits each: fa | fb, a | b */
var<workgroup> wcached: u32;
var<private> cached: bool;
/* corner k / face f's normal of a (side 0) or b (side 1) */
fn hcv(side: u32, s: Shape, k: u32) -> vec3f { if (cached) { return hv[side * HC + k]; } return sv(s, k); }
fn hcn(side: u32, s: Shape, f: u32) -> vec3f { if (cached) { return hn[side * HF + f]; } return sn(s, f); }
fn min_along_c(side: u32, s: Shape, d: vec3f) -> f32 {
    var m = 1e30;
    for (var k = 0u; k < s.nv; k++) { m = min(m, dot(hcv(side, s, k), d)); }
    return m;
}

/* the largest ws (and its i, j, axis) into element 0 */
fn reduce_max(lid: u32) {
    for (var st = 32u; st > 0u; st = st >> 1u) {
        workgroupBarrier();
        if (lid < st && ws[lid + st] > ws[lid]) {
            ws[lid] = ws[lid + st];
            wi[lid] = wi[lid + st];
            wj[lid] = wj[lid + st];
            wl[lid] = wl[lid + st];
        }
    }
    workgroupBarrier();
}

/* the pairs K_HULLPRE queued (pc[4]): one workgroup each */
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let lid = lv.x;
    let q = wid.x + wid.y * nw.x;
    if (lid == 0u) { wcount = min(atomicLoad(&pc[4]), P.m.z); wsep = 0u; atomicStore(&wsepa, 0u); }
    let count = workgroupUniformLoad(&wcount);
    if (q >= count) { return; }
    let k = feat[P.m.z + q].x;
    let pr = pairs[P.m.z + k];
    let ia = pr.x;
    let ib = pr.y;
    MG = margin_of(ia, ib);
    let A = shape_of(ia);
    let S = shape_of(ib);
    if (lid == 0u) { wcached = select(0u, 1u, A.nv <= HC && S.nv <= HC && A.nf <= HF && S.nf <= HF && S.ne <= HE); }
    cached = workgroupUniformLoad(&wcached) != 0u;
    if (cached) {
        for (var v = lid; v < A.nv; v += 64u) { hv[v] = sv(A, v); }
        for (var v = lid; v < S.nv; v += 64u) { hv[HC + v] = sv(S, v); }
        for (var f = lid; f < A.nf; f += 64u) { hn[f] = sn(A, f); }
        for (var f = lid; f < S.nf; f += 64u) { hn[HF + f] = sn(S, f); }
        for (var e = lid; e < S.ne; e += 64u) {
            let F = bitcast<vec4u>(G[S.e0 + e]);
            he[e] = vec2u(F.z | (F.w << 16u), F.x | (F.y << 16u));
        }
    }
    workgroupBarrier();
    let key = pair_key(ia, ib);
    let rd = u32(P.h2.w);                   /* the table read this step (the other is written) */
    var trel: vec3f;
    var qrel: vec4f;
    rel_pose(ia, ib, &trel, &qrel);
    /* the full test. a's faces */
    var best = -1e30;
    var bi = 0u;
    for (var f = lid; f < A.nf; f += 64u) {
        let s = min_along_c(1u, S, hcn(0u, A, f)) - sd(A, f);
        if (s > MG) { atomicMax(&wsepa, (1u << 28u) | (f << 12u)); }
        if (s > best) { best = s; bi = f; }
    }
    ws[lid] = best; wi[lid] = bi; wj[lid] = 0u; wl[lid] = vec3f(0.0);
    reduce_max(lid);
    let sa = workgroupUniformLoad(&ws[0]);
    let fa = wi[0];
    if (lid == 0u) { wsep = atomicLoad(&wsepa); }
    if (workgroupUniformLoad(&wsep) != 0u) {
        if (lid == 0u) { memo_put(1u - rd, key, vec3u(1u, (wsep >> 12u) & 0xFFFu, 0u), trel, qrel); }
        return;
    }
    /* b's faces */
    best = -1e30;
    bi = 0u;
    for (var f = lid; f < S.nf; f += 64u) {
        let s = min_along_c(0u, A, hcn(1u, S, f)) - sd(S, f);
        if (s > MG) { atomicMax(&wsepa, (2u << 28u) | (f << 12u)); }
        if (s > best) { best = s; bi = f; }
    }
    workgroupBarrier();
    ws[lid] = best; wi[lid] = bi;
    reduce_max(lid);
    let sb = workgroupUniformLoad(&ws[0]);
    let fb = wi[0];
    if (lid == 0u) { wsep = atomicLoad(&wsepa); }
    if (workgroupUniformLoad(&wsep) != 0u) {
        if (lid == 0u) { memo_put(1u - rd, key, vec3u(2u, (wsep >> 12u) & 0xFFFu, 0u), trel, qrel); }
        return;
    }
    /* the edge pairs: a's edges spread over the threads, each against all of b's */
    best = -1e30;
    bi = NONE;
    var bj = 0u;
    var bl = vec3f(0.0, 1.0, 0.0);
    for (var i = lid; i < A.ne; i += 64u) {
        let E = bitcast<vec4u>(G[A.e0 + i]);
        let a = hcn(0u, A, E.z);
        let b = hcn(0u, A, E.w);
        let pa = hcv(0u, A, E.x);
        let da = hcv(0u, A, E.y) - pa;
        let bxa = cross(b, a);
        for (var j = 0u; j < S.ne; j++) {
            var F: vec4u;
            if (cached) { let w = he[j]; F = vec4u(w.y & 0xFFFFu, w.y >> 16u, w.x & 0xFFFFu, w.x >> 16u); }
            else { F = bitcast<vec4u>(G[S.e0 + j]); }
            let c = -hcn(1u, S, F.z);
            let d = -hcn(1u, S, F.w);
            let cba = dot(c, bxa);
            let dba = dot(d, bxa);
            if (cba * dba >= 0.0) { continue; }
            let dxc = cross(d, c);
            let adc = dot(a, dxc);
            let bdc = dot(b, dxc);
            if (adc * bdc >= 0.0 || cba * bdc <= 0.0) { continue; }
            let pb = hcv(1u, S, F.x);
            let db = hcv(1u, S, F.y) - pb;
            var L = cross(da, db);
            let ll = length(L);
            if (ll < 1e-6 * length(da) * length(db)) { continue; }
            L = L / ll;
            if (dot(L, pa - A.pos) < 0.0) { L = -L; }
            let s = dot(L, pb - pa);
            if (s > MG) { atomicMax(&wsepa, (3u << 28u) | (i << 12u) | j); }
            if (s > best) { best = s; bi = i; bj = j; bl = L; }
        }
    }
    workgroupBarrier();
    ws[lid] = best; wi[lid] = bi; wj[lid] = bj; wl[lid] = bl;
    reduce_max(lid);
    if (lid == 0u) { wsep = atomicLoad(&wsepa); }
    if (workgroupUniformLoad(&wsep) != 0u) {
        if (lid == 0u) { memo_put(1u - rd, key, vec3u(3u, (wsep >> 12u) & 0xFFFu, wsep & 0xFFFu), trel, qrel); }
        return;
    }
    if (lid != 0u) { return; }
    let se = ws[0];
    let ea = wi[0];
    let eb = wj[0];
    /* the feature of least overlap (convex.c's preferences), remembered; its contacts: K_HULLC */
    let use_b = sa < 1.02 * sb - 0.01;
    let sf = select(sa, sb, use_b);
    var f = vec3u(4u, fa, 0u);
    if (ea != NONE && sf < 1.1 * se - 0.1) { f = vec3u(6u, ea, eb); }
    else if (use_b) { f = vec3u(5u, fb, 0u); }
    feat[k] = vec4u(f, 1u);
    memo_put(1u - rd, key, f, trel, qrel);
}
