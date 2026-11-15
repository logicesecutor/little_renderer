// rigid_convex.wgsl - used by engine/gpu/rigid_gpu.c (K_CONVEX)
struct Shape { pos: vec3f, r: mat3x3f, s: vec3f, box: bool, nv: u32, nf: u32, ne: u32, v0: u32, f0: u32, l0: u32, e0: u32 };
const MAXP = 32u;
fn gu(i: u32, c: u32) -> u32 { return bitcast<u32>(G[i][c]); }
fn shape_of(i: u32) -> Shape {
    let b = B[i];
    var sh: Shape;
    sh.pos = b.pos.xyz;
    sh.r = qmat(b.rot);
    sh.box = b.pos.w > 0.5 && b.pos.w < 1.5;
    var gi = bitcast<u32>(b.t.x);
    sh.s = vec3f(1.0);
    if (sh.box) { gi = P.m.x; sh.s = b.half.xyz; }
    sh.nv = gu(2u * gi, 0u); sh.nf = gu(2u * gi, 1u); sh.ne = gu(2u * gi, 2u);
    sh.v0 = gu(2u * gi + 1u, 0u); sh.f0 = gu(2u * gi + 1u, 1u); sh.l0 = gu(2u * gi + 1u, 2u); sh.e0 = gu(2u * gi + 1u, 3u);
    return sh;
}
fn sv(s: Shape, k: u32) -> vec3f { return s.pos + s.r * (G[s.v0 + k].xyz * s.s); }
fn sn(s: Shape, f: u32) -> vec3f { return s.r * G[s.f0 + 2u * f].xyz; }
fn sd(s: Shape, f: u32) -> f32 {
    let nl = G[s.f0 + 2u * f].xyz;
    var d = G[s.f0 + 2u * f].w;
    if (s.box) { d = dot(abs(nl), s.s); }
    return d + dot(s.r * nl, s.pos);
}
fn min_along(s: Shape, d: vec3f) -> f32 {
    var m = 1e30;
    for (var k = 0u; k < s.nv; k++) { m = min(m, dot(sv(s, k), d)); }
    return m;
}

/* up to MAXP candidate contacts of one pair, then the 4 kept (convex_reduce) */
var<private> cp: array<vec3f, 32>;
var<private> cn: array<vec3f, 32>;
var<private> cd: array<f32, 32>;
var<private> nc: u32;
fn cand(p: vec3f, n: vec3f, d: f32) {
    if (nc < MAXP) { cp[nc] = p; cn[nc] = n; cd[nc] = d; nc++; return; }
    /* full: replace the shallowest if this one is deeper */
    var lo = 0u;
    for (var k = 1u; k < MAXP; k++) { if (cd[k] < cd[lo]) { lo = k; } }
    if (d > cd[lo]) { cp[lo] = p; cn[lo] = n; cd[lo] = d; }
}
fn flush(a: u32, b: u32) {
    if (nc == 0u) { return; }
    var pick = array<u32, 4>(0u, 0u, 0u, 0u);
    var np = 1u;
    for (var k = 1u; k < nc; k++) { if (cd[k] > cd[pick[0]]) { pick[0] = k; } }
    if (nc > 1u) {
        var bd = -1.0;
        var i1 = 0u;
        for (var k = 0u; k < nc; k++) { let d = length(cp[k] - cp[pick[0]]); if (d > bd) { bd = d; i1 = k; } }
        if (i1 != pick[0]) { pick[1] = i1; np = 2u; }
    }
    if (np == 2u) {
        var ba = 0.0;
        var i2 = NONE;
        for (var k = 0u; k < nc; k++) {
            let ar = length(cross(cp[k] - cp[pick[0]], cp[k] - cp[pick[1]]));
            if (ar > ba) { ba = ar; i2 = k; }
        }
        if (i2 != NONE) { pick[2] = i2; np = 3u; }
    }
    if (np == 3u) {
        let nn = cross(cp[pick[1]] - cp[pick[0]], cp[pick[2]] - cp[pick[0]]);
        var ba = 0.0;
        var i3 = NONE;
        for (var k = 0u; k < nc; k++) {
            for (var e = 0u; e < 3u; e++) {
                let p = cp[pick[e]];
                let q = cp[pick[(e + 1u) % 3u]];
                let ar = -dot(cross(q - p, cp[k] - p), nn);
                if (ar > ba) { ba = ar; i3 = k; }
            }
        }
        if (i3 != NONE) { pick[3] = i3; np = 4u; }
    }
    for (var k = 0u; k < np; k++) { add_contact(a, b, cp[pick[k]], cn[pick[k]], cd[pick[k]]); }
    nc = 0u;
}

/* the incident polygon clipped to the reference face's sides (loop rf, counter-clockwise
   around nr, plane nr . x = dr): the points below it, with normal ncon (from b to a) */
fn face_contacts(rf: ptr<function, array<vec3f, 32>>, nref: u32, nr: vec3f, dr: f32,
                 inc: ptr<function, array<vec3f, 32>>, ninc: u32, ncon: vec3f) {
    var a = *inc;
    var t: array<vec3f, 32>;
    var n = min(ninc, MAXP);
    for (var k = 0u; k < nref; k++) {
        if (n == 0u) { break; }
        let p = (*rf)[k];
        let q = (*rf)[(k + 1u) % nref];
        let side = cross(q - p, nr);
        let off = dot(side, p);
        var m = 0u;
        for (var e = 0u; e < n; e++) {
            let p0 = a[e];
            let p1 = a[(e + 1u) % n];
            let d0 = dot(p0, side) - off;
            let d1 = dot(p1, side) - off;
            if (d0 <= 0.0 && m < MAXP) { t[m] = p0; m++; }
            if (((d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0)) && m < MAXP) { t[m] = p0 + (p1 - p0) * (d0 / (d0 - d1)); m++; }
        }
        for (var e = 0u; e < m; e++) { a[e] = t[e]; }
        n = m;
    }
    for (var k = 0u; k < n; k++) {
        let depth = dr - dot(nr, a[k]);
        if (depth >= -MG) { cand(a[k] + nr * (0.5 * depth), ncon, depth); }
    }
}

fn face_loop(s: Shape, f: u32, out: ptr<function, array<vec3f, 32>>) -> u32 {
    let first = gu(s.f0 + 2u * f + 1u, 0u);
    let count = min(gu(s.f0 + 2u * f + 1u, 1u), MAXP);
    for (var j = 0u; j < count; j++) { (*out)[j] = sv(s, gu(s.l0 + first + j, 0u)); }
    return count;
}
fn incident_face(s: Shape, n: vec3f) -> u32 {
    var best = 0u;
    var bd = 1e30;
    for (var f = 0u; f < s.nf; f++) { let d = dot(sn(s, f), n); if (d < bd) { bd = d; best = f; } }
    return best;
}
/* the middle of the closest points of segments p0-p1 and q0-q1 */
fn closest_seg(p0: vec3f, p1: vec3f, q0: vec3f, q1: vec3f) -> vec3f {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let c = dot(d1, r);
    let b = dot(d1, d2);
    var s = 0.0;
    let den = a * e - b * b;
    if (den > 1e-12 * a * e) { s = clamp((b * f - c * e) / den, 0.0, 1.0); }
    var t = select(0.0, (b * s + f) / e, e > 0.0);
    if (t < 0.0) { t = 0.0; s = select(0.0, clamp(-c / a, 0.0, 1.0), a > 0.0); }
    else if (t > 1.0) { t = 1.0; s = select(0.0, clamp((b - c) / a, 0.0, 1.0), a > 0.0); }
    return ((p0 + d1 * s) + (q0 + d2 * t)) * 0.5;
}

/* boxes and hulls: the faces of each, and the edge pairs that make a face of the Minkowski
   difference (their arcs cross on the Gauss map: Gregorius 2013) */
fn collide_convex(ia: u32, ib: u32) {
    let A = shape_of(ia);
    let S = shape_of(ib);
    var sa = -1e30;
    var sb = -1e30;
    var se = -1e30;
    var fa = 0u;
    var fb = 0u;
    var ea = NONE;
    var eb = 0u;
    var le = vec3f(0.0, 1.0, 0.0);
    for (var f = 0u; f < A.nf; f++) {
        let s = min_along(S, sn(A, f)) - sd(A, f);
        if (s > MG) { return; }
        if (s > sa) { sa = s; fa = f; }
    }
    for (var f = 0u; f < S.nf; f++) {
        let s = min_along(A, sn(S, f)) - sd(S, f);
        if (s > MG) { return; }
        if (s > sb) { sb = s; fb = f; }
    }
    for (var i = 0u; i < A.ne; i++) {
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
            if (s > MG) { return; }
            if (s > se) { se = s; ea = i; eb = j; le = L; }
        }
    }
    nc = 0u;
    let use_b = sa < 1.02 * sb - 0.01;
    let sf = select(sa, sb, use_b);
    if (ea != NONE && sf < 1.1 * se - 0.1) {
        let E = bitcast<vec4u>(G[A.e0 + ea]);
        let F = bitcast<vec4u>(G[S.e0 + eb]);
        add_contact(ia, ib, closest_seg(sv(A, E.x), sv(A, E.y), sv(S, F.x), sv(S, F.y)), -le, -se);
        return;
    }
    var rf: array<vec3f, 32>;
    var inc: array<vec3f, 32>;
    if (use_b) {
        let fi = incident_face(A, sn(S, fb));
        let nr = face_loop(S, fb, &rf);
        let ni = face_loop(A, fi, &inc);
        face_contacts(&rf, nr, sn(S, fb), sd(S, fb), &inc, ni, sn(S, fb));
    } else {
        let fi = incident_face(S, sn(A, fa));
        let nr = face_loop(A, fa, &rf);
        let ni = face_loop(S, fi, &inc);
        face_contacts(&rf, nr, sn(A, fa), sd(A, fa), &inc, ni, -sn(A, fa));
    }
    flush(ia, ib);
}

/* the point of triangle a b c closest to p (Ericson 5.1.5) */
fn closest_on_triangle(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return a + ab * (d1 / (d1 - d3)); }
    let cq = p - c;
    let d5 = dot(ab, cq);
    let d6 = dot(ac, cq);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return a + ac * (d2 / (d2 - d6)); }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))); }
    let den = 1.0 / (va + vb + vc);
    return a + ab * (vb * den) + ac * (vc * den);
}

/* a sphere (ia) against a box or a hull (ib) */
fn sphere_convex(ia: u32, ib: u32) {
    let c = B[ia].pos.xyz;
    let r = B[ia].half.x;
    let S = shape_of(ib);
    var dmax = -1e30;
    var fm = 0u;
    for (var f = 0u; f < S.nf; f++) { let s = dot(sn(S, f), c) - sd(S, f); if (s > dmax) { dmax = s; fm = f; } }
    if (dmax > r + MG) { return; }
    if (dmax <= 0.0) { add_contact(ia, ib, c - sn(S, fm) * dmax, sn(S, fm), r - dmax); return; }
    var best = 1e30;
    var bq = c;
    for (var f = 0u; f < S.nf; f++) {
        if (dot(sn(S, f), c) - sd(S, f) <= 0.0) { continue; }
        let first = gu(S.f0 + 2u * f + 1u, 0u);
        let count = gu(S.f0 + 2u * f + 1u, 1u);
        let a = sv(S, gu(S.l0 + first, 0u));
        for (var j = 1u; j + 1u < count; j++) {
            let q = closest_on_triangle(c, a, sv(S, gu(S.l0 + first + j, 0u)), sv(S, gu(S.l0 + first + j + 1u, 0u)));
            let d = length(c - q);
            if (d < best) { best = d; bq = q; }
        }
    }
    if (best >= r + MG) { return; }
    add_contact(ia, ib, bq, select(sn(S, fm), (c - bq) / best, best > 1e-5), r - best);
}

/* a hull's corners below each wall, at most 4 per wall */
fn hull_walls(ia: u32) {
    let S = shape_of(ia);
    var ns = array<vec3f, 6>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
                             vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));
    var offs = array<f32, 6>(P.lo.x, -P.hi.x, P.lo.y, -P.hi.y, P.lo.z, -P.hi.z);
    for (var k = 0u; k < 6u; k++) {
        nc = 0u;
        for (var i = 0u; i < S.nv; i++) {
            let v = sv(S, i);
            let d = dot(ns[k], v) - offs[k];
            if (d < MG) { cand(v, ns[k], -d); }
        }
        flush(ia, NONE);
    }
}

/* a body A against triangle t0 t1 t2 (world), two-sided: candidates (cand) */
fn convex_triangle(A: Shape, t0: vec3f, t1: vec3f, t2: vec3f) {
    var tv = array<vec3f, 3>(t0, t1, t2);
    var nt = cross(t1 - t0, t2 - t0);
    let ln = length(nt);
    if (ln <= 0.0) { return; }
    nt = nt / ln;
    let dt = dot(nt, t0);
    let sp = min_along(A, nt) - dt;
    let sm = min_along(A, -nt) + dt;
    let st = max(sp, sm);
    let ns = select(-nt, nt, sp > sm);
    if (st > MG) { return; }
    var sa = -1e30;
    var se = -1e30;
    var fa = 0u;
    var ea = NONE;
    var ek = 0u;
    var le = nt;
    for (var f = 0u; f < A.nf; f++) {
        let n = sn(A, f);
        let s = min(min(dot(t0, n), dot(t1, n)), dot(t2, n)) - sd(A, f);
        if (s > MG) { return; }
        if (s > sa) { sa = s; fa = f; }
    }
    for (var k = 0u; k < 3u; k++) {
        let e = tv[(k + 1u) % 3u] - tv[k];
        for (var i = 0u; i < A.ne; i++) {
            let E = bitcast<vec4u>(G[A.e0 + i]);
            let na = sn(A, E.z);
            let nb = sn(A, E.w);
            if (dot(na, e) * dot(nb, e) >= 0.0) { continue; }
            let pa = sv(A, E.x);
            let h = sv(A, E.y) - pa;
            var L = cross(h, e);
            let ll = length(L);
            if (ll < 1e-6 * length(h) * length(e)) { continue; }
            L = L / ll;
            if (dot(L, na + nb) < 0.0) { L = -L; }
            let s = min(min(dot(t0, L), dot(t1, L)), dot(t2, L)) - dot(L, pa);
            if (s > MG) { return; }
            if (s > se) { se = s; ea = i; ek = k; le = L; }
        }
    }
    let use_t = sa < 1.02 * st - 0.01;
    let sf = select(sa, st, use_t);
    if (ea != NONE && sf < 1.1 * se - 0.1) {
        let E = bitcast<vec4u>(G[A.e0 + ea]);
        cand(closest_seg(sv(A, E.x), sv(A, E.y), tv[ek], tv[(ek + 1u) % 3u]), -le, -se);
        return;
    }
    var rf: array<vec3f, 32>;
    var inc: array<vec3f, 32>;
    if (use_t) {
        rf[0] = t0;
        rf[1] = t1;
        rf[2] = t2;
        if (dot(ns, nt) < 0.0) { rf[1] = t2; rf[2] = t1; }
        let fi = incident_face(A, ns);
        let ni = face_loop(A, fi, &inc);
        face_contacts(&rf, 3u, ns, dot(ns, t0), &inc, ni, ns);
    } else {
        let nr = face_loop(A, fa, &rf);
        inc[0] = t0;
        inc[1] = t1;
        inc[2] = t2;
        face_contacts(&rf, nr, sn(A, fa), sd(A, fa), &inc, 3u, -sn(A, fa));
    }
}

/* a body (pair slot k: body i, mesh m) against triangle q of the mesh (world corners) */
fn mesh_corner(m: Body, tri0: u32, q: u32, c: u32) -> vec3f { return m.pos.xyz + qmat(m.rot) * G[tri0 + 3u * q + c].xyz; }
fn mesh_tri(i: u32, im: u32, q: u32) {
    let a = B[i];
    let m = B[im];
    let gi = bitcast<u32>(m.t.x);
    let tri0 = gu(2u * gi + 1u, 1u);
    let t0 = mesh_corner(m, tri0, q, 0u);
    let t1 = mesh_corner(m, tri0, q, 1u);
    let t2 = mesh_corner(m, tri0, q, 2u);
    if (a.pos.w < 0.5) {
        let c = a.pos.xyz;
        let p = closest_on_triangle(c, t0, t1, t2);
        let d = c - p;
        let l2 = dot(d, d);
        if (l2 < (a.half.x + MG) * (a.half.x + MG)) {
            let l = sqrt(l2);
            var n = normalize(cross(t1 - t0, t2 - t0));
            if (l > 1e-4) { n = d / l; } else if (dot(n, c - t0) < 0.0) { n = -n; }
            cand(p, n, a.half.x - l);
        }
    } else {
        convex_triangle(shape_of(i), t0, t1, t2);
    }
}
