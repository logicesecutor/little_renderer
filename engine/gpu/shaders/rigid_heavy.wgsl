// rigid_heavy.wgsl - used by engine/gpu/rigid_gpu.c (K_HEAVY)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
/* what decided each pair last step (rigid.c's memos, P3): two tables, last step's (read) and
   this step's (written, cleared before), each `cap` keys then `cap` entries of 12 words: the
   feature (type, i, j), the pose of b in a's frame at the full test (t; q as x, y, z, w) */
@group(0) @binding(7) var<storage, read_write> memo: array<atomic<u32>>;
var<workgroup> ws: array<f32, 64>;
var<workgroup> wi: array<u32, 64>;
var<workgroup> wj: array<u32, 64>;
var<workgroup> wl: array<vec3f, 64>;
var<workgroup> wsep: u32;
var<workgroup> wsepa: atomic<u32>;
var<workgroup> wcount: u32;
var<workgroup> wft: vec4u;                 /* the feature remembered: type, i, j; w = 1 reuse it */

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
fn reduce_max_f(lid: u32) {
    for (var st = 32u; st > 0u; st = st >> 1u) {
        workgroupBarrier();
        if (lid < st && ws[lid + st] > ws[lid]) { ws[lid] = ws[lid + st]; }
    }
    workgroupBarrier();
}

fn qmul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn memo_cap() -> u32 { return 2u * P.m.z; }
fn memo_base(t: u32) -> u32 { return t * 13u * memo_cap(); }
fn memo_hash(key: u32) -> u32 { return (key * 2654435761u) >> 8u; }
/* the entry of key in table t: its index, or NONE */
fn memo_find(t: u32, key: u32) -> u32 {
    let cap = memo_cap();
    let base = memo_base(t);
    var h = memo_hash(key) & (cap - 1u);
    for (var n = 0u; n < 32u; n++) {
        let k = atomicLoad(&memo[base + h]);
        if (k == key) { return h; }
        if (k == 0u) { return NONE; }
        h = (h + 1u) & (cap - 1u);
    }
    return NONE;
}
fn memo_word(t: u32, e: u32, w: u32) -> u32 { return atomicLoad(&memo[memo_base(t) + memo_cap() + e * 12u + w]); }
fn memo_put(t: u32, key: u32, ft: vec3u, tr: vec3f, q: vec4f) {
    let cap = memo_cap();
    let base = memo_base(t);
    var h = memo_hash(key) & (cap - 1u);
    for (var n = 0u; n < 32u; n++) {
        let r = atomicCompareExchangeWeak(&memo[base + h], 0u, key);
        if (r.exchanged || r.old_value == key) {
            let d = base + cap + h * 12u;
            atomicStore(&memo[d], ft.x); atomicStore(&memo[d + 1u], ft.y); atomicStore(&memo[d + 2u], ft.z);
            atomicStore(&memo[d + 4u], bitcast<u32>(tr.x)); atomicStore(&memo[d + 5u], bitcast<u32>(tr.y));
            atomicStore(&memo[d + 6u], bitcast<u32>(tr.z));
            atomicStore(&memo[d + 8u], bitcast<u32>(q.x)); atomicStore(&memo[d + 9u], bitcast<u32>(q.y));
            atomicStore(&memo[d + 10u], bitcast<u32>(q.z)); atomicStore(&memo[d + 11u], bitcast<u32>(q.w));
            return;
        }
        h = (h + 1u) & (cap - 1u);
    }
}

/* the axis of a's edge i and b's edge j, from a to b (0 if they are parallel) */
fn edge_axis(A: Shape, S: Shape, i: u32, j: u32) -> vec3f {
    let E = bitcast<vec4u>(G[A.e0 + i]);
    let F = bitcast<vec4u>(G[S.e0 + j]);
    let pa = sv(A, E.x);
    let da = sv(A, E.y) - pa;
    let db = sv(S, F.y) - sv(S, F.x);
    var L = cross(da, db);
    let ll = length(L);
    if (ll < 1e-6 * length(da) * length(db)) { return vec3f(0.0); }
    L = L / ll;
    return select(L, -L, dot(L, pa - A.pos) < 0.0);
}

/* the contacts of the feature of least overlap (as convex.c's feature_contacts); one thread */
fn feature_contacts(ia: u32, ib: u32, A: Shape, S: Shape, ft: vec3u) {
    nc = 0u;
    if (ft.x == 6u) {                       /* a pair of edges */
        let E = bitcast<vec4u>(G[A.e0 + ft.y]);
        let F = bitcast<vec4u>(G[S.e0 + ft.z]);
        let L = edge_axis(A, S, ft.y, ft.z);
        if (dot(L, L) == 0.0) { return; }
        let s = dot(L, sv(S, F.x) - sv(A, E.x));
        if (s > MG) { return; }
        add_contact(ia, ib, closest_seg(sv(A, E.x), sv(A, E.y), sv(S, F.x), sv(S, F.y)), -L, -s);
        return;
    }
    var rf: array<vec3f, 32>;
    var inc: array<vec3f, 32>;
    if (ft.x == 5u) {                       /* b's face */
        let fb = ft.y;
        let fi = incident_face(A, sn(S, fb));
        let nr = face_loop(S, fb, &rf);
        let ni = face_loop(A, fi, &inc);
        face_contacts(&rf, nr, sn(S, fb), sd(S, fb), &inc, ni, sn(S, fb));
    } else {                                /* a's face */
        let fa = ft.y;
        let fi = incident_face(S, sn(A, fa));
        let nr = face_loop(A, fa, &rf);
        let ni = face_loop(S, fi, &inc);
        face_contacts(&rf, nr, sn(A, fa), sd(A, fa), &inc, ni, -sn(A, fa));
    }
    flush(ia, ib);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let lid = lv.x;
    let k = wid.x + wid.y * nw.x;
    if (lid == 0u) { wcount = min(atomicLoad(&pc[1]), P.m.z); wsep = 0u; atomicStore(&wsepa, 0u); }
    let count = workgroupUniformLoad(&wcount);
    if (k >= count) { return; }
    let pr = pairs[P.m.z + k];
    let ia = pr.x;
    let ib = pr.y;
    MG = margin_of(ia, ib);
    if (ib == NONE) { if (lid == 0u) { hull_walls(ia); } return; }
    let ka = u32(B[ia].pos.w + 0.5);
    let kb = u32(B[ib].pos.w + 0.5);
    if (ka == 0u) { if (lid == 0u) { sphere_convex(ia, ib); } return; }
    if (kb == 0u) { if (lid == 0u) { sphere_convex(ib, ia); } return; }
    let A = shape_of(ia);
    let S = shape_of(ib);
    /* b in a's frame, and what decided the pair last step */
    let key = (ia << 16u | ib) + 1u;
    let rd = u32(P.h2.w);                   /* the table read this step (the other is written) */
    let qa = B[ia].rot;
    let ca = vec4f(-qa.xyz, qa.w);
    let qrel = qmul(ca, B[ib].rot);
    let trel = transpose(qmat(qa)) * (B[ib].pos.xyz - B[ia].pos.xyz);
    var mt = vec3f(0.0);
    var mq = vec4f(0.0, 0.0, 0.0, 1.0);
    if (lid == 0u) {
        wft = vec4u(0u);
        let e = memo_find(rd, key);
        if (e != NONE) {
            let ty = memo_word(rd, e, 0u);
            mt = vec3f(bitcast<f32>(memo_word(rd, e, 4u)), bitcast<f32>(memo_word(rd, e, 5u)), bitcast<f32>(memo_word(rd, e, 6u)));
            mq = vec4f(bitcast<f32>(memo_word(rd, e, 8u)), bitcast<f32>(memo_word(rd, e, 9u)),
                       bitcast<f32>(memo_word(rd, e, 10u)), bitcast<f32>(memo_word(rd, e, 11u)));
            let rmin = min(body_radius(B[ia]), body_radius(B[ib]));
            let still = length(trel - mt) < 0.01 * rmin && abs(dot(qrel, mq)) > 0.99996;
            wft = vec4u(ty, memo_word(rd, e, 1u), memo_word(rd, e, 2u), select(0u, 1u, still));
        }
    }
    let ft = workgroupUniformLoad(&wft);
    /* apart last step: still apart along that axis? (all threads: one reduction) */
    if (ft.x >= 1u && ft.x <= 3u) {
        var lo = -1e30;                     /* the largest of -(b along L) and of (a along L) */
        var hi = -1e30;
        var L = vec3f(0.0);
        var off = 0.0;
        if (ft.x == 1u) { L = sn(A, ft.y); off = sd(A, ft.y); }
        else if (ft.x == 2u) { L = sn(S, ft.y); off = sd(S, ft.y); }
        else { L = edge_axis(A, S, ft.y, ft.z); }
        if (ft.x == 2u) {                   /* a's corners along b's face normal */
            for (var v = lid; v < A.nv; v += 64u) { lo = max(lo, -dot(sv(A, v), L)); }
        } else {
            for (var v = lid; v < S.nv; v += 64u) { lo = max(lo, -dot(sv(S, v), L)); }
        }
        if (ft.x == 3u) { for (var v = lid; v < A.nv; v += 64u) { hi = max(hi, dot(sv(A, v), L)); } }
        ws[lid] = lo;
        reduce_max_f(lid);
        let mn = -workgroupUniformLoad(&ws[0]);
        ws[lid] = hi;
        reduce_max_f(lid);
        let mx = workgroupUniformLoad(&ws[0]);
        var gap = mn - off;
        if (ft.x == 3u) { gap = select(-1e30, mn - mx, dot(L, L) > 0.0); }
        if (gap > MG) {
            if (lid == 0u) { memo_put(1u - rd, key, ft.xyz, mt, mq); }
            return;
        }
    }
    /* touching last step and barely moved since the full test: the same feature */
    if (ft.x >= 4u && ft.w == 1u) {
        if (lid == 0u) {
            feature_contacts(ia, ib, A, S, ft.xyz);
            memo_put(1u - rd, key, ft.xyz, mt, mq);
        }
        return;
    }
    /* the full test. a's faces */
    var best = -1e30;
    var bi = 0u;
    for (var f = lid; f < A.nf; f += 64u) {
        let s = min_along(S, sn(A, f)) - sd(A, f);
        if (s > MG) { atomicMax(&wsepa, 1u << 28u | f << 12u); }
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
        let s = min_along(A, sn(S, f)) - sd(S, f);
        if (s > MG) { atomicMax(&wsepa, 2u << 28u | f << 12u); }
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
        let a = sn(A, E.z);
        let b = sn(A, E.w);
        let pa = sv(A, E.x);
        let da = sv(A, E.y) - pa;
        let bxa = cross(b, a);
        for (var j = 0u; j < S.ne; j++) {
            let F = bitcast<vec4u>(G[S.e0 + j]);
            let c = -sn(S, F.z);
            let d = -sn(S, F.w);
            let cba = dot(c, bxa);
            let dba = dot(d, bxa);
            if (cba * dba >= 0.0) { continue; }
            let dxc = cross(d, c);
            let adc = dot(a, dxc);
            let bdc = dot(b, dxc);
            if (adc * bdc >= 0.0 || cba * bdc <= 0.0) { continue; }
            let pb = sv(S, F.x);
            let db = sv(S, F.y) - pb;
            var L = cross(da, db);
            let ll = length(L);
            if (ll < 1e-6 * length(da) * length(db)) { continue; }
            L = L / ll;
            if (dot(L, pa - A.pos) < 0.0) { L = -L; }
            let s = dot(L, pb - pa);
            if (s > MG) { atomicMax(&wsepa, 3u << 28u | i << 12u | j); }
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
    /* the feature of least overlap (convex.c's preferences), its contacts, remembered */
    let use_b = sa < 1.02 * sb - 0.01;
    let sf = select(sa, sb, use_b);
    var f = vec3u(4u, fa, 0u);
    if (ea != NONE && sf < 1.1 * se - 0.1) { f = vec3u(6u, ea, eb); }
    else if (use_b) { f = vec3u(5u, fb, 0u); }
    feature_contacts(ia, ib, A, S, f);
    memo_put(1u - rd, key, f, trel, qrel);
}
