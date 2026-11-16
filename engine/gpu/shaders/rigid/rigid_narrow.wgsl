// rigid_narrow.wgsl - used by engine/gpu/rigid_gpu.c (K_NARROW)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> pairs: array<vec2u>;
@group(0) @binding(5) var<storage, read_write> pc: array<atomic<u32>>;

/* the margin of the pair being tested (set before its tests) */
var<private> MG: f32;
fn margin_of(ia: u32, ib: u32) -> f32 {
    let a = B[ia];
    if (ib == NONE) { return pair_margin(a, a, true); }
    return pair_margin(a, B[ib], false);
}

fn add_contact(a: u32, b: u32, p: vec3f, n: vec3f, depth: f32) {
    let k = atomicAdd(&cnt[0], 1u);
    if (k >= P.n.y) { return; }
    var c: Contact;
    c.p = vec4f(p, depth);
    c.n = vec4f(n, bitcast<f32>(a));
    c.ra = vec4f(0.0, 0.0, 0.0, bitcast<f32>(b));
    let A = B[a];
    c.la = vec4f(transpose(qmat(A.rot)) * (p - A.pos.xyz), depth);
    if (b != NONE) { let Bb = B[b]; c.lb = vec4f(transpose(qmat(Bb.rot)) * (p - Bb.pos.xyz), 0.0); } else { c.lb = vec4f(p, 0.0); }
    C[k] = c;
    atomicAdd(&cnt[1u + a], 1u);
    if (b != NONE) { atomicAdd(&cnt[1u + b], 1u); }
}

fn comp(v: vec3f, k: u32) -> f32 { return select(select(v.z, v.y, k == 1u), v.x, k == 0u); }

fn walls(ia: u32) {
    let b = B[ia];
    var ns = array<vec3f, 6>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
                             vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));
    var offs = array<f32, 6>(P.lo.x, -P.hi.x, P.lo.y, -P.hi.y, P.lo.z, -P.hi.z);
    let r = qmat(b.rot);
    for (var k = 0u; k < 6u; k++) {
        let n = ns[k];
        if (b.pos.w < 0.5) {
            let d = dot(n, b.pos.xyz) - offs[k] - b.half.x;
            if (d < MG) { add_contact(ia, NONE, b.pos.xyz - n * b.half.x, n, -d); }
        } else {
            for (var i = 0u; i < 8u; i++) {
                let l = vec3f(select(-b.half.x, b.half.x, (i & 1u) != 0u), select(-b.half.y, b.half.y, (i & 2u) != 0u),
                              select(-b.half.z, b.half.z, (i & 4u) != 0u));
                let c = b.pos.xyz + r * l;
                let d = dot(n, c) - offs[k];
                if (d < MG) { add_contact(ia, NONE, c, n, -d); }
            }
        }
    }
}

fn sphere_sphere(ia: u32, ib: u32) {
    let a = B[ia];
    let b = B[ib];
    let d = a.pos.xyz - b.pos.xyz;
    let r = a.half.x + b.half.x;
    let l2 = dot(d, d);
    if (l2 >= (r + MG) * (r + MG)) { return; }
    let l = sqrt(l2);
    let n = select(vec3f(0.0, 1.0, 0.0), d / l, l > 1e-6);
    let depth = r - l;
    add_contact(ia, ib, b.pos.xyz + n * (b.half.x - 0.5 * depth), n, depth);
}

/* sphere a against box b */
fn sphere_box(ia: u32, ib: u32) {
    let s = B[ia];
    let b = B[ib];
    let r = qmat(b.rot);
    let c = transpose(r) * (s.pos.xyz - b.pos.xyz);
    let q = clamp(c, -b.half.xyz, b.half.xyz);
    let diff = c - q;
    let l2 = dot(diff, diff);
    let rad = s.half.x;
    var nl = vec3f(0.0);
    var depth = 0.0;
    if (l2 > 1e-12) {
        if (l2 >= (rad + MG) * (rad + MG)) { return; }
        let l = sqrt(l2);
        nl = diff / l;
        depth = rad - l;
    } else {
        /* the centre is inside: out by the nearest face */
        let g = b.half.xyz - abs(c);
        var k = 0u;
        if (g.y < g.x && g.y <= g.z) { k = 1u; } else if (g.z < g.x && g.z < g.y) { k = 2u; }
        let sg = select(1.0, -1.0, comp(c, k) < 0.0);
        nl = vec3f(select(0.0, sg, k == 0u), select(0.0, sg, k == 1u), select(0.0, sg, k == 2u));
        depth = rad + comp(g, k);
    }
    add_contact(ia, ib, b.pos.xyz + r * q, r * nl, depth);
}

/* box - box: separating axes (3 + 3 faces, 9 edge pairs) and face clipping, as rigid.c */
fn face_corners(b: Body, r0: mat3x3f, k: u32, s: f32) -> array<vec3f, 4> {
    var r = r0;
    let i = (k + 1u) % 3u;
    let j = (k + 2u) % 3u;
    let c = b.pos.xyz + r[k] * (s * comp(b.half.xyz, k));
    let u = r[i] * comp(b.half.xyz, i);
    let v = r[j] * comp(b.half.xyz, j);
    return array<vec3f, 4>(c + u + v, c + u - v, c - u - v, c - u + v);
}

fn box_box(ia: u32, ib: u32) {
    let a = B[ia];
    let b = B[ib];
    var ra = qmat(a.rot);
    var rb = qmat(b.rot);
    let t = a.pos.xyz - b.pos.xyz;
    var best = 1e30;
    var depth = 0.0;
    var bn = vec3f(0.0, 1.0, 0.0);
    var axis = -1;
    for (var i = 0; i < 15; i++) {
        var L = vec3f(0.0);
        if (i < 3) { L = ra[i]; } else if (i < 6) { L = rb[i - 3]; } else { L = cross(ra[(i - 6) / 3], rb[(i - 6) % 3]); }
        let ll = dot(L, L);
        if (ll < 1e-6) { continue; }
        L = L / sqrt(ll);
        let pa = a.half.x * abs(dot(ra[0], L)) + a.half.y * abs(dot(ra[1], L)) + a.half.z * abs(dot(ra[2], L));
        let pb = b.half.x * abs(dot(rb[0], L)) + b.half.y * abs(dot(rb[1], L)) + b.half.z * abs(dot(rb[2], L));
        let dist = dot(t, L);
        let overlap = pa + pb - abs(dist);
        if (overlap < -MG) { return; }
        var score = overlap * 1.1 + 0.1;
        if (i < 3) { score = overlap; } else if (i < 6) { score = overlap * 1.02 + 0.01; }
        if (score < best) {
            best = score;
            depth = overlap;
            bn = select(L, -L, dist < 0.0);
            axis = i;
        }
    }
    if (axis < 0) { return; }
    if (axis < 6) {
        let ref_a = axis < 3;
        var R = a;
        var I = b;
        var rr = ra;
        var ri = rb;
        var nref = -bn;
        if (!ref_a) { R = b; I = a; rr = rb; ri = ra; nref = bn; }
        let k = u32(axis % 3);
        /* the incident face: I's most opposed to nref */
        var ki = 0u;
        var most = 0.0;
        for (var q = 0u; q < 3u; q++) {
            let d = dot(ri[q], nref);
            if (abs(d) > abs(most)) { most = d; ki = q; }
        }
        var fc = face_corners(I, ri, ki, select(1.0, -1.0, most > 0.0));
        var poly: array<vec3f, 16>;
        var tmp: array<vec3f, 16>;
        for (var q = 0u; q < 4u; q++) { poly[q] = fc[q]; }
        var n = 4u;
        for (var q = 1u; q <= 2u; q++) {
            let jj = (k + q) % 3u;
            let dj = rr[jj];
            let hj = comp(R.half.xyz, jj);
            let cj = dot(dj, R.pos.xyz);
            for (var side = 0u; side < 2u; side++) {
                let d = select(-dj, dj, side == 0u);
                let off = select(-(cj - hj), cj + hj, side == 0u);
                /* Sutherland-Hodgman: keep dot(p, d) <= off */
                var m = 0u;
                for (var e = 0u; e < n; e++) {
                    let p0 = poly[e];
                    let p1 = poly[(e + 1u) % n];
                    let d0 = dot(p0, d) - off;
                    let d1 = dot(p1, d) - off;
                    if (d0 <= 0.0 && m < 16u) { tmp[m] = p0; m++; }
                    if (((d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0)) && m < 16u) { tmp[m] = p0 + (p1 - p0) * (d0 / (d0 - d1)); m++; }
                }
                for (var e = 0u; e < m; e++) { poly[e] = tmp[e]; }
                n = m;
            }
        }
        let face = dot(nref, R.pos.xyz) + comp(R.half.xyz, k);
        var added = 0u;
        for (var q = 0u; q < n; q++) {
            let sep = dot(nref, poly[q]) - face;
            if (sep > MG) { continue; }
            add_contact(ia, ib, poly[q] - nref * (0.5 * sep), bn, -sep);
            added++;
        }
        if (added > 0u) { return; }
    }
    /* edge against edge: the closest points of the two edges, the ones furthest towards each other */
    var ea = 0u;
    var eb = 0u;
    if (axis >= 6) { ea = u32((axis - 6) / 3); eb = u32((axis - 6) % 3); }
    var pa = a.pos.xyz;
    var pb = b.pos.xyz;
    for (var q = 0u; q < 3u; q++) {
        if (q != ea) { pa = pa + ra[q] * (select(1.0, -1.0, dot(ra[q], bn) > 0.0) * comp(a.half.xyz, q)); }
        if (q != eb) { pb = pb + rb[q] * (select(-1.0, 1.0, dot(rb[q], bn) > 0.0) * comp(b.half.xyz, q)); }
    }
    let d0 = ra[ea] * comp(a.half.xyz, ea);
    let d1 = rb[eb] * comp(b.half.xyz, eb);
    let rr = pa - pb;
    let aa = dot(d0, d0);
    let ee = dot(d1, d1);
    let ff = dot(d1, rr);
    let cc = dot(d0, rr);
    let bb = dot(d0, d1);
    let den = aa * ee - bb * bb;
    var s = 0.0;
    if (den > 1e-8) { s = clamp((bb * ff - cc * ee) / den, -1.0, 1.0); }
    var tt = (bb * s + ff) / ee;
    if (tt < -1.0 || tt > 1.0) {
        tt = clamp(tt, -1.0, 1.0);
        s = clamp((tt * bb - cc) / aa, -1.0, 1.0);
    }
    let c0 = pa + d0 * s;
    let c1 = pb + d1 * tt;
    add_contact(ia, ib, (c0 + c1) * 0.5, bn, depth);
}
