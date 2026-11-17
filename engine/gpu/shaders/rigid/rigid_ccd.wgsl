// rigid_ccd.wgsl - used by engine/gpu/rigid_gpu.c (K_CCD)
/* rigid.c's rigid_ccd, after the step's last substep: each body that moved more than
   CCD_FRACTION of its inner radius since the step began (S: where each body was) has that
   sphere swept along its path, relative to every other body's path (from where both were),
   against the room's walls and the others grown by its radius (a sphere, a box, a hull);
   at the first hit it is put just before it, its velocity kept, and the next step's
   speculative contact stops it at the surface. CCD_STEP 0 finds the places (T: xyz, w = 1
   if moved), CCD_STEP 1 moves the bodies there (two passes: every body's sweep reads the
   others where they ended the step). A fixed mesh is not swept against here. */
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> S: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> T: array<vec4f>;
@group(0) @binding(4) var<storage, read> G: array<vec4f>;

fn gu(i: u32, c: u32) -> u32 { return bitcast<u32>(G[i][c]); }
/* a hull's or box's faces: the shape's index in G (a box: the unit box, scaled) */
fn shape_gi(b: Body) -> u32 { return select(bitcast<u32>(b.t.x), P.m.x, b.pos.w > 0.5 && b.pos.w < 1.5); }

fn inner_radius(b: Body) -> f32 {
    if (b.pos.w < 0.5) { return b.half.x; }
    if (b.pos.w < 1.5) { return min(b.half.x, min(b.half.y, b.half.z)); }
    if (b.pos.w < 2.5) {
        let gi = shape_gi(b);
        let nf = gu(2u * gi, 1u);
        let f0 = gu(2u * gi + 1u, 1u);
        var r = 1e30;
        for (var f = 0u; f < nf; f++) { r = min(r, G[f0 + 2u * f].w); }
        return select(min(b.half.x, min(b.half.y, b.half.z)), r, r > 0.0 && r < 1e30);
    }
    return 0.0;
}

/* p + t d (t < tmax) entering the sphere (c, rad): its t, or tmax */
fn sweep_sphere(p: vec3f, d: vec3f, c: vec3f, rad: f32, tmax: f32) -> f32 {
    let m = p - c;
    let a = dot(d, d);
    let b = dot(m, d);
    let cc = dot(m, m) - rad * rad;
    if (cc <= 0.0 || a <= 0.0 || b >= 0.0) { return tmax; }
    let disc = b * b - a * cc;
    if (disc < 0.0) { return tmax; }
    let t = (-b - sqrt(disc)) / a;
    return select(tmax, t, t >= 0.0 && t < tmax);
}

/* against body o (its place at the step's start, its turn now), grown by rad */
fn sweep_body(o: Body, opos: vec3f, p: vec3f, d: vec3f, rad: f32, tmax: f32) -> f32 {
    if (o.pos.w < 0.5) { return sweep_sphere(p, d, opos, o.half.x + rad, tmax); }
    if (o.pos.w > 2.5) { return tmax; }
    let rt = transpose(qmat(o.rot));
    let lp = rt * (p - opos);
    let ld = rt * d;
    let gi = shape_gi(o);
    let nf = gu(2u * gi, 1u);
    let f0 = gu(2u * gi + 1u, 1u);
    let box = o.pos.w < 1.5;
    var tin = 0.0;
    var tout = 1.0;
    var outside = false;
    for (var f = 0u; f < nf; f++) {
        let n = G[f0 + 2u * f].xyz;
        let off = select(G[f0 + 2u * f].w, dot(abs(n), o.half.xyz), box);
        let dist = dot(n, lp) - off - rad;
        let den = dot(n, ld);
        if (dist > 0.0) { outside = true; }
        if (den == 0.0) { if (dist > 0.0) { return tmax; } continue; }
        let tk = -dist / den;
        if (den < 0.0) { tin = max(tin, tk); } else { tout = min(tout, tk); }
        if (tin > tout) { return tmax; }
    }
    return select(tmax, tin, outside && tin < tmax);
}

/* the others, 64 at a time, shared by the workgroup's threads (each read once per
   workgroup, not once per thread): where each began (xyz) and its bounding radius (w), and
   how far it moved */
var<workgroup> tile_s: array<vec4f, 64>;
var<workgroup> tile_d: array<vec3f, 64>;
var<workgroup> wg_fast: atomic<u32>;
var<workgroup> wg_any: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    let n = P.n.x;
    if (CCD_STEP == 1u) {
        if (i >= n) { return; }
        let t = T[i];
        if (t.w > 0.5) { B[i].pos = vec4f(t.xyz, B[i].pos.w); }
        return;
    }
    /* (every thread goes through the tiles, for the barriers; only the fast bodies sweep) */
    var fast = i < n;
    var rad = 0.0;
    var p = vec3f(0.0);
    var d = vec3f(0.0);
    var len = 0.0;
    if (fast) {
        T[i] = vec4f(0.0);
        let b = B[i];
        rad = inner_radius(b);
        p = S[i].xyz;
        d = b.pos.xyz - p;
        len = length(d);
        fast = b.vel.w != 0.0 && rad > 0.0 && len > P.c.x * rad;
    }
    /* a workgroup without a fast body has nothing to sweep (most of them, most steps) */
    if (lv.x == 0u) { atomicStore(&wg_fast, 0u); }
    workgroupBarrier();
    if (fast) { atomicStore(&wg_fast, 1u); }
    workgroupBarrier();
    if (lv.x == 0u) { wg_any = atomicLoad(&wg_fast); }
    if (workgroupUniformLoad(&wg_any) == 0u) { return; }
    var t = 1.0;
    for (var base = 0u; base < n; base += 64u) {
        let j = base + lv.x;
        if (j < n) {
            let o = B[j];
            tile_s[lv.x] = vec4f(S[j].xyz, body_radius(o));
            tile_d[lv.x] = o.pos.xyz - S[j].xyz;
        }
        workgroupBarrier();
        if (fast) {
            for (var k = 0u; k < 64u && base + k < n; k++) {
                let jj = base + k;
                if (jj == i) { continue; }
                let dr = d - tile_d[k];
                let l2 = dot(dr, dr);
                if (l2 <= 0.0) { continue; }
                let m = tile_s[k].xyz - p;
                let s = clamp(dot(m, dr) / l2, 0.0, 1.0);
                let q = m - dr * s;
                let reach = tile_s[k].w + rad;
                if (dot(q, q) > reach * reach) { continue; }
                t = sweep_body(B[jj], tile_s[k].xyz, p, dr, rad, t);
            }
        }
        workgroupBarrier();
    }
    if (!fast) { return; }
    if (P.n.w != 0u) {          /* the room's walls, each alone */
        var ns = array<vec3f, 6>(vec3f(-1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, -1.0, 0.0), vec3f(0.0, 1.0, 0.0),
                                 vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 1.0));
        var offs = array<f32, 6>(-P.lo.x, P.hi.x, -P.lo.y, P.hi.y, -P.lo.z, P.hi.z);
        for (var k = 0u; k < 6u; k++) {
            let d0 = offs[k] - rad - dot(ns[k], p);
            let dn = dot(ns[k], d);
            if (d0 > 0.0 && dn > 0.0) { t = min(t, d0 / dn); }
        }
    }
    if (t >= 1.0) { return; }
    let back = min(t, (P.d.z + 0.05 * rad) / len);
    T[i] = vec4f(p + d * (t - back), 1.0);
}
