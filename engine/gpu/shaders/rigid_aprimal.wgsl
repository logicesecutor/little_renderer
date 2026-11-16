// rigid_aprimal.wgsl - used by engine/gpu/rigid_gpu.c (K_APRIMAL; FULL: the last pass)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> C: array<Contact>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read> L: array<u32>;
@group(0) @binding(5) var<storage, read> AB: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> CTL: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> ORDER: array<u32>;
/* each thread's sums over its share of its body's contacts: All, Ala, Aaa (9 each), gl, ga */
var<workgroup> part: array<f32, 2112>;

fn outer(u: vec3f, v: vec3f) -> mat3x3f { return mat3x3f(u * v.x, u * v.y, u * v.z); }
/* the inverse of a 3 x 3 matrix (its rows: cross products of its columns over the determinant) */
fn inv3(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]);
    let r1 = cross(m[2], m[0]);
    let r2 = cross(m[0], m[1]);
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}

/* body i to where its inertia (to y) balances its contacts' forces, the others where they
   are now: M/h^2 + sum k J J^T against M/h^2 (x - y) + sum J f (rigid.c's avbd_steps), in
   3 x 3 blocks (moves: ll, turns: aa, both: la), all over m/h^2 (in range for f32). 8
   threads a body (lanes), each through every 8th of its contacts, their sums into `part` */
fn add_contacts(i: u32, lane: u32, li: u32) {
    let b = B[i];
    let sc = P.h.x * P.h.x * b.vel.w;          /* 1 / (m / h^2) */
    var All = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var Ala = All;
    var Aaa = All;
    var gl = vec3f(0.0);
    var ga = vec3f(0.0);
    for (var q = off[i] + lane; q < off[i + 1u]; q += 8u) {
        let e = L[q];
        let k = C[e >> 1u];
        let side_b = (e & 1u) == 1u;
        let sg = select(1.0, -1.0, side_b);
        let arm = select(k.ra.xyz, k.rb.xyz, side_b);
        let cs = row_cs(k, FULL);
        for (var r = 0u; r < 3u; r++) {
            if (r > 0u && k.j.x == 0.0 && !FULL) { continue; }      /* (no push, no friction) */
            let lo = row_lo(k, r);
            let hi = row_hi(k, r);
            let f0 = k.imp[r] * cs[r] + k.j[r];
            let f = clamp(f0, lo, hi);
            if (f == 0.0 && f0 >= hi) { continue; }                 /* (apart: no force) */
            let d = row_dir(k, r);
            let ja = cross(arm, d);
            /* (at a bound the force no longer changes with C: no stiffness) */
            let kk = select(0.0, k.imp[r] * sc, f0 > lo && f0 < hi);
            gl += d * (sg * f * sc);
            ga += ja * (sg * f * sc);
            All += outer(d, d * kk);
            Ala += outer(d, ja * kk);
            Aaa += outer(ja, ja * kk);
        }
    }
    let o = li * 33u;
    for (var c = 0u; c < 3u; c++) {
        for (var r = 0u; r < 3u; r++) {
            part[o + c * 3u + r] = All[c][r];
            part[o + 9u + c * 3u + r] = Ala[c][r];
            part[o + 18u + c * 3u + r] = Aaa[c][r];
        }
        part[o + 27u + c] = gl[c];
        part[o + 30u + c] = ga[c];
    }
}

/* the lanes' sums with the inertia, solved by the turns' Schur complement */
fn solve_body(i: u32, li: u32) {
    let b = B[i];
    let m = 1.0 / b.vel.w;
    let R = qmat(AB[8u * i + 1u]);
    let inr = select(vec3f(0.0), 1.0 / b.iner.xyz, b.iner.xyz > vec3f(0.0));
    let I = R * mat3x3f(vec3f(inr.x, 0.0, 0.0), vec3f(0.0, inr.y, 0.0), vec3f(0.0, 0.0, inr.z)) * transpose(R) * (1.0 / m);
    var All = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    var Ala = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var Aaa = I;
    var gl = b.pos.xyz - AB[8u * i + 2u].xyz;
    var ga = I * rot_delta(b.rot, AB[8u * i + 3u]);
    for (var l = 0u; l < 8u; l++) {
        let o = (li + l) * 33u;
        for (var c = 0u; c < 3u; c++) {
            All[c] += vec3f(part[o + c * 3u], part[o + c * 3u + 1u], part[o + c * 3u + 2u]);
            Ala[c] += vec3f(part[o + 9u + c * 3u], part[o + 10u + c * 3u], part[o + 11u + c * 3u]);
            Aaa[c] += vec3f(part[o + 18u + c * 3u], part[o + 19u + c * 3u], part[o + 20u + c * 3u]);
        }
        gl += vec3f(part[o + 27u], part[o + 28u], part[o + 29u]);
        ga += vec3f(part[o + 30u], part[o + 31u], part[o + 32u]);
    }
    let Ail = inv3(All);
    let Alt = transpose(Ala);
    let S = Aaa - Alt * Ail * Ala;
    if (!(determinant(S) > 0.0) || !(determinant(All) > 0.0)) { return; }
    let xa = inv3(S) * (ga - Alt * (Ail * gl));
    let xl = Ail * (gl - Ala * xa);
    /* (a trust region: the contacts are linear in the turn only while it is small) */
    let s = min(1.0, MAX_TURN / max(length(xa), 1e-30));
    B[i].pos = vec4f(b.pos.xyz - xl * s, b.pos.w);
    B[i].rot = aq_turn(b.rot, -xa * s, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u,
        @builtin(local_invocation_index) li: u32) {
    /* the colour this pass solves (CTL[0]), its bodies' list (K_APLACE: ORDER from the
       counts in CTL[4 + colour]) */
    let c = min(atomicLoad(&CTL[0]), 31u);
    var start = 0u;
    for (var q = 0u; q < c; q++) { start += atomicLoad(&CTL[4u + q]); }
    let t = tid(gid, nw);
    let slot = t / 8u;
    let lane = t % 8u;
    let busy = slot < atomicLoad(&CTL[4u + c]);
    var i = 0u;
    if (busy) { i = ORDER[start + slot]; add_contacts(i, lane, li); }
    workgroupBarrier();
    if (busy && lane == 0u) { solve_body(i, li); }
    /* the last workgroup done moves CTL[0] on to the next colour */
    if (li == 0u) {
        let d = atomicAdd(&CTL[1], 1u);
        if (d + 1u == nw.x * nw.y * nw.z) {
            atomicStore(&CTL[1], 0u);
            atomicAdd(&CTL[0], 1u);
        }
    }
}
