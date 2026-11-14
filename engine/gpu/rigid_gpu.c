#include "gpu/rigid_gpu.h"

#include <stdlib.h>

#ifndef LR_GPU
// ---- built without webgpu.h: the bodies stay on the CPU ------------------------------------
GpuRigid *gpu_rigid_create(Gpu *g, PhysWorld *pw){ (void)g; (void)pw; return NULL; }
void gpu_rigid_set_wait(GpuRigid *gr, int wait){ (void)gr; (void)wait; }
void gpu_rigid_sync(GpuRigid *gr){ (void)gr; }
void gpu_rigid_destroy(GpuRigid *gr){ (void)gr; }
#else

#include <stdio.h>
#include <string.h>
#include "gpu/gpu_internal.h"
#include "phys/rigid.h"

// ---- the kernels ---------------------------------------------------------------------------
//
// Bodies (struct Body, 8 vec4): pos (w: 0 sphere, 1 box, 2 hull, 3 fixed mesh), rot (quaternion x y z w), vel (w:
// 1 / mass), ang (angular velocity; w: restitution), half (w: friction), iner (1 / inertia in
// the body's frame), f (sleeping, rigid.c's: x = how long it has been still, y = 1 asleep, z =
// its 1 / mass while asleep, when vel.w is 0: to the others it is fixed) and t (x: a hull's or
// mesh's shape, its index in G; yzw: the air's force on it when it fell asleep). Contacts (struct Contact, 10 vec4), as rigid.c's: the
// point (w: depth), the normal from b to a (w: a), ra (w: b, NONE = a wall), rb (w: bias), the
// tangents (w: the normal's and the first tangent's effective mass), j = the accumulated
// impulses (w: the second tangent's mass), imp = this iteration's impulse (w: friction), la =
// the point in a's frame (w: the depth when found), lb = in b's (a wall: the world point;
// w: the approach speed it is to bounce from, 0 once it has: rigid.c's vn0).
// Contacts are found once per physics step, with a margin (speculative contacts, as rigid.c),
// and followed through the substeps by la and lb (K_PREP).
// cnt[0] counts the contacts, cnt[1 + i] those of body i; off[] (n + 1) are where each body's
// contacts start in L[], as 2 c (it is a) or 2 c + 1 (it is b).
static const char *RHEAD = WGSL(
struct RP {
    n: vec4u,      /* bodies, room for contacts, bodies in the previous substep, room walls (1) */
    h: vec4f,      /* the substep, slop, Baumgarte, bounce threshold */
    g: vec4f,      /* gravity, linear damping */
    lo: vec4f,     /* the room; w = the walls' restitution */
    hi: vec4f,     /* w = the walls' friction */
    d: vec4f,      /* angular damping, the whole step, the smallest margin (the contacts' margins), the fastest push out */
    h2: vec4f,     /* sleeping: the speed below which a body is still, how long still before it sleeps, the wake speed;
                      w: the memo's table read this step (K_HEAVY) */
    m: vec4u,      /* the unit box's shape in G (boxes against hulls), shapes, room in each pair list,
                      the substep (contacts are found in the first) */
};
@group(0) @binding(0) var<uniform> P: RP;
struct Body { pos: vec4f, rot: vec4f, vel: vec4f, ang: vec4f, half: vec4f, iner: vec4f, f: vec4f, t: vec4f };
struct Contact { p: vec4f, n: vec4f, ra: vec4f, rb: vec4f, t1: vec4f, t2: vec4f, j: vec4f, imp: vec4f, la: vec4f, lb: vec4f };
const NONE: u32 = 0xffffffffu;
fn tid(gid: vec3u, nw: vec3u) -> u32 { return gid.x + gid.y * nw.x * 64u; }
fn body_radius(b: Body) -> f32 { return select(length(b.half.xyz), b.half.x, b.pos.w < 0.5); }
/* how far apart a pair may be and still get a contact (rigid.c's pair_margin): as far as
   they can close in a step, plus a little; at most half their sizes */
fn pair_margin(a: Body, b: Body, wall: bool) -> f32 {
    var closing = length(a.vel.xyz);
    var r = body_radius(a);
    var spin = length(a.ang.xyz) * r;
    if (!wall) {
        let dv = a.vel.xyz - b.vel.xyz;
        let d = a.pos.xyz - b.pos.xyz;
        let l = length(d);
        closing = select(length(dv), max(-dot(dv, d) / l, 0.0), l > 0.0);
        spin += length(b.ang.xyz) * body_radius(b);
        r += body_radius(b);
    }
    return min(P.d.z + P.d.y * (closing + spin), P.d.z + 0.5 * r);
}
fn qmat(q: vec4f) -> mat3x3f {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3f(vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
                   vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
                   vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}
fn inv_inertia(b: Body) -> mat3x3f {
    if (b.vel.w == 0.0) { return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)); }   /* fixed, or asleep */
    let r = qmat(b.rot);
    return r * mat3x3f(vec3f(b.iner.x, 0.0, 0.0), vec3f(0.0, b.iner.y, 0.0), vec3f(0.0, 0.0, b.iner.z)) * transpose(r);
}
/* two tangents of a unit normal (core/rng.h onb_from_normal, Duff et al. 2017) */
fn onb(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y));
}
);

/* velocities from the forces (gravity, the air's, held for the substeps); the contact counts cleared */
static const char *K_VEL = WGSL(
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> F: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> wake: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    let first = P.m.w == 0u;
    if (i == 0u && first) { atomicStore(&cnt[0], 0u); for (var l = 0u; l < 4u; l++) { atomicStore(&pc[l], 0u); } }
    if (i >= P.n.x) { return; }
    if (first) { atomicStore(&cnt[1u + i], 0u); }
    var b = B[i];
    if (first && b.f.y > 0.5 && atomicLoad(&wake[i]) != 0u) {
        /* woken (K_WAKE): its mass back, still at rest */
        b.vel = vec4f(0.0, 0.0, 0.0, b.f.z);
        b.f = vec4f(0.0);
        B[i].f = b.f;
    }
    if (first) { atomicStore(&wake[i], 0u); }
    if (b.vel.w == 0.0) { return; }
    let h = P.h.x;
    var v = b.vel.xyz + (P.g.xyz + F[2u * i].xyz * b.vel.w) * h;
    var w = b.ang.xyz + inv_inertia(b) * (F[2u * i + 1u].xyz * h);
    v = v / (1.0 + P.g.w * h);
    w = w / (1.0 + P.d.x * h);
    B[i].vel = vec4f(v, b.vel.w);
    B[i].ang = vec4f(w, b.ang.w);
}
);

/* contacts, in three steps: every pair of bodies by their bounding spheres (n i + j for the
   pair i < j, then the n walls), into two lists: balls and boxes (these tests), hulls and
   meshes (K_CONVEX); then each list, one thread per entry, launched as many as the lists
   hold (indirect dispatch). */
static const char *K_NARROW = WGSL(
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
);


/* sleeping (rigid.c's): which sleeping bodies wake up this step: a body faster than the wake
   speed near one (one thread per pair, as K_BROAD), or the air's force on it changed by a
   quarter of its weight (one thread per body) */
static const char *K_WAKE = WGSL(
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> wake: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> F: array<vec4f>;
fn asleep(b: Body) -> bool { return b.f.y > 0.5; }
fn moving(b: Body) -> bool {
    return b.vel.w > 0.0 && (length(b.vel.xyz) >= P.h2.z || length(b.ang.xyz) * body_radius(b) >= P.h2.z);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    let n = P.n.x;
    if (id >= n * n) {
        let i = id - n * n;
        if (i < n && asleep(B[i]) && length(F[2u * i].xyz - B[i].t.yzw) > 0.25 * length(P.g.xyz) / B[i].f.z) {
            atomicStore(&wake[i], 1u);
        }
        return;
    }
    let i = id % n;
    let j = id / n;
    if (j <= i) { return; }
    let a = B[i];
    let b = B[j];
    var s = 0u;
    if (asleep(a) && moving(b)) { s = i; } else if (asleep(b) && moving(a)) { s = j; } else { return; }
    let d = a.pos.xyz - b.pos.xyz;
    let reach = body_radius(a) + body_radius(b) + pair_margin(a, b, false);
    if (dot(d, d) <= reach * reach) { atomicStore(&wake[s], 1u); }
}
);

/* the end of a step: how long each body has been still; those still long enough fall asleep:
   velocity 0, and vel.w = 0 (to the others a fixed body) with the mass kept in f.z */
static const char *K_SLEEP = WGSL(
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> F: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }
    let b = B[i];
    if (b.vel.w == 0.0) { return; }
    let still = length(b.vel.xyz) < P.h2.x && length(b.ang.xyz) * body_radius(b) < P.h2.x;
    let t = select(0.0, b.f.x + P.d.y, still);
    if (t < P.h2.y) { B[i].f.x = t; return; }
    B[i].vel = vec4f(0.0);
    B[i].ang = vec4f(0.0, 0.0, 0.0, b.ang.w);
    B[i].f = vec4f(t, 1.0, b.vel.w, 0.0);
    B[i].t.y = F[2u * i].x;     /* (t.x is a shape index's bits: not written back as a float) */
    B[i].t.z = F[2u * i].y;
    B[i].t.w = F[2u * i].z;
}
);

/* the pairs whose bounding spheres meet (and the walls), sorted into the two lists */
static const char *K_BROAD = WGSL(
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> pairs: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> pc: array<atomic<u32>>;
/* list 0: balls and boxes, 1: hulls, 2: a body and a fixed mesh */
fn push(l: u32, i: u32, j: u32) {
    let k = atomicAdd(&pc[l], 1u);
    if (k < P.m.z) { pairs[l * P.m.z + k] = vec2u(i, j); }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    let n = P.n.x;
    if (id >= n * n) {
        let i = id - n * n;
        if (i < n && P.n.w != 0u && B[i].vel.w > 0.0) { push(select(0u, 1u, B[i].pos.w > 1.5), i, NONE); }
        return;
    }
    let i = id % n;
    let j = id / n;
    if (j <= i) { return; }
    let a = B[i];
    let b = B[j];
    if (a.vel.w == 0.0 && b.vel.w == 0.0) { return; }
    let rai = select(length(a.half.xyz), a.half.x, a.pos.w < 0.5);
    let rbj = select(length(b.half.xyz), b.half.x, b.pos.w < 0.5);
    let d = a.pos.xyz - b.pos.xyz;
    let reach = rai + rbj + pair_margin(a, b, false);
    if (dot(d, d) > reach * reach) { return; }
    if (a.pos.w > 2.5 && b.pos.w > 2.5) { return; }
    if (a.pos.w > 2.5) { push(2u, j, i); }
    else if (b.pos.w > 2.5) { push(2u, i, j); }
    else { push(select(0u, 1u, a.pos.w > 1.5 || b.pos.w > 1.5), i, j); }
}
);

/* the two lists' lengths as workgroup counts (indirect dispatch: x, 1, 1 at args[0] and args[4]) */
static const char *K_ARGS = WGSL(
@group(0) @binding(1) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
    for (var l = 0u; l < 3u; l++) {
        let n = min(atomicLoad(&pc[l]), P.m.z);
        args[4u * l] = select((n + 63u) / 64u, min(n, 65535u), l == 1u);   /* hulls: a workgroup each */
        if (l == 2u) {                                                     /* meshes' trees: the same */
            args[16] = min(n, 65535u);
            args[17] = 1u;
            args[18] = 1u;
        }
        args[4u * l + 1u] = 1u;
        args[4u * l + 2u] = 1u;
    }
}
);

/* the triangles' list's length (after K_MESHQ), at args[12] */
static const char *K_ARGS2 = WGSL(
@group(0) @binding(1) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let n = min(atomicLoad(&pc[3]), 64u * P.m.z);
    args[12] = (n + 63u) / 64u;
    args[13] = 1u;
    args[14] = 1u;
}
);

/* the balls' and boxes' list */
static const char *K_SIMPLE = WGSL(
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = tid(gid, nw);
    if (k >= min(atomicLoad(&pc[0]), P.m.z)) { return; }
    let pr = pairs[k];
    let i = pr.x;
    let j = pr.y;
    MG = margin_of(i, j);
    if (j == NONE) { walls(i); return; }
    let sa = B[i].pos.w < 0.5;
    let sb = B[j].pos.w < 0.5;
    if (sa && sb) { sphere_sphere(i, j); }
    else if (sa) { sphere_box(i, j); }
    else if (sb) { sphere_box(j, i); }
    else { box_box(i, j); }
}
);

/* the hulls' list: one workgroup per pair, its 64 threads sharing the separating axis
   test (the faces of each, the pairs of edges), the best axis found by a reduction; then the
   first thread makes the contacts (convex.c's order of preference, so the same choices) */
static const char *K_HEAVY = WGSL(
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
);

/* hulls and fixed meshes (phys/convex.c, the same tests): G holds, per shape, two header
   rows (h0: corners, faces, edges, kind 2; or tree nodes, triangles, 0, kind 3. h1: where
   they start), then the corners, the faces (n, d; first loop index, count), the loops (one
   corner index per row), the edges (a, b, fa, fb); a mesh: its tree's nodes (lo, left; hi,
   right; first, count) and triangles (3 corners each, in the leaves' order). Boxes use the
   unit box (P.m.x) scaled by their half sizes. */
static const char *K_CONVEX = WGSL(
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
);


/* body - fixed mesh pairs, in parallel: each pair walks its mesh's tree and lists the
   triangles near the body (one entry per triangle); each entry is tested by its own thread
   and its contacts gathered per pair (at most 64); then each pair keeps 4 of them */
static const char *K_MESHQ = WGSL(
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> T: array<vec2u>;
fn gq(i: u32, c: u32) -> u32 { return bitcast<u32>(G[i][c]); }
/* one workgroup per pair: the tree is walked level by level, each level's nodes shared by the
   64 threads (a front of at most 1024 nodes; a thread that cannot add to it walks on alone) */
var<workgroup> front: array<u32, 2048>;
var<workgroup> nnext: atomic<u32>;
var<workgroup> wn: u32;
var<workgroup> wcount: u32;
var<workgroup> wlo: vec3f;
var<workgroup> whi: vec3f;

fn emit(k: u32, first: u32, count: u32) {
    for (var q = 0u; q < count; q++) {
        let e = atomicAdd(&pc[3], 1u);
        if (e < 64u * P.m.z) { T[e] = vec2u(k, first + q); }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let lid = lv.x;
    let k = wid.x + wid.y * nw.x;
    if (lid == 0u) { wcount = min(atomicLoad(&pc[2]), P.m.z); }
    let count = workgroupUniformLoad(&wcount);
    if (k >= count) { return; }
    let pr = pairs[2u * P.m.z + k];
    let a = B[pr.x];
    let m = B[pr.y];
    let gi = bitcast<u32>(m.t.x);
    let node0 = gq(2u * gi + 1u, 0u);
    let rt = transpose(qmat(m.rot));
    let mg = pair_margin(a, m, false);
    if (lid == 0u) {
        /* the body's box in the mesh's frame: a ball's, or around a hull's (box's) corners */
        var lo = vec3f(1e30);
        var hi = vec3f(-1e30);
        if (a.pos.w < 0.5) {
            let c = rt * (a.pos.xyz - m.pos.xyz);
            lo = c - vec3f(a.half.x);
            hi = c + vec3f(a.half.x);
        } else {
            let A = shape_of(pr.x);
            for (var v = 0u; v < A.nv; v++) {
                let l = rt * (sv(A, v) - m.pos.xyz);
                lo = min(lo, l);
                hi = max(hi, l);
            }
        }
        wlo = lo - vec3f(mg);
        whi = hi + vec3f(mg);
        front[0] = 0u;
        wn = 1u;
        atomicStore(&nnext, 0u);
    }
    var n = workgroupUniformLoad(&wn);
    let lo = wlo;
    let hi = whi;
    var par = 0u;
    for (var level = 0u; level < 64u; level++) {
        if (n == 0u) { break; }
        for (var idx = lid; idx < n; idx += 64u) {
            let nd = node0 + 3u * front[par * 1024u + idx];
            if (any(G[nd].xyz > hi) || any(G[nd + 1u].xyz < lo)) { continue; }
            let cnt = gq(nd + 2u, 1u);
            if (cnt > 0u) { emit(k, gq(nd + 2u, 0u), cnt); continue; }
            let slot = atomicAdd(&nnext, 2u);
            if (slot + 2u <= 1024u) {
                front[(1u - par) * 1024u + slot] = gq(nd, 3u);
                front[(1u - par) * 1024u + slot + 1u] = gq(nd + 1u, 3u);
                continue;
            }
            /* the front is full: this subtree alone */
            var stack: array<u32, 48>;
            var sp = 2u;
            stack[0] = gq(nd, 3u);
            stack[1] = gq(nd + 1u, 3u);
            while (sp > 0u) {
                sp--;
                let nd2 = node0 + 3u * stack[sp];
                if (any(G[nd2].xyz > hi) || any(G[nd2 + 1u].xyz < lo)) { continue; }
                let c2 = gq(nd2 + 2u, 1u);
                if (c2 > 0u) { emit(k, gq(nd2 + 2u, 0u), c2); continue; }
                if (sp + 2u <= 48u) {
                    stack[sp] = gq(nd2, 3u);
                    sp++;
                    stack[sp] = gq(nd2 + 1u, 3u);
                    sp++;
                }
            }
        }
        workgroupBarrier();
        if (lid == 0u) { wn = min(atomicLoad(&nnext), 1024u); atomicStore(&nnext, 0u); }
        n = workgroupUniformLoad(&wn);
        par = 1u - par;
    }
}
);

static const char *K_MESHTRI = WGSL(
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
@group(0) @binding(7) var<storage, read> T: array<vec2u>;
@group(0) @binding(8) var<storage, read_write> S: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> sc: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let e = tid(gid, nw);
    if (e >= min(atomicLoad(&pc[3]), 64u * P.m.z)) { return; }
    let t = T[e];
    let pr = pairs[2u * P.m.z + t.x];
    nc = 0u;
    MG = margin_of(pr.x, pr.y);
    mesh_tri(pr.x, pr.y, t.y);
    for (var q = 0u; q < nc; q++) {
        let slot = atomicAdd(&sc[t.x], 1u);
        if (slot >= 64u) { break; }
        let o = 2u * (64u * t.x + slot);
        S[o] = vec4f(cp[q], cd[q]);
        S[o + 1u] = vec4f(cn[q], 0.0);
    }
}
);

static const char *K_MESHFLUSH = WGSL(
@group(0) @binding(6) var<storage, read> G: array<vec4f>;   /* (declared for K_CONVEX's functions, not used here) */
@group(0) @binding(8) var<storage, read> S: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> sc: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = tid(gid, nw);
    if (k >= min(atomicLoad(&pc[2]), P.m.z)) { return; }
    let pr = pairs[2u * P.m.z + k];
    let n = min(atomicLoad(&sc[k]), 64u);
    nc = 0u;
    for (var q = 0u; q < n; q++) {
        let o = 2u * (64u * k + q);
        cand(S[o].xyz, S[o + 1u].xyz, S[o].w);
    }
    flush(pr.x, pr.y);
    atomicStore(&sc[k], 0u);
}
);


/* where each body's contacts go (one thread: a few thousand bodies at most) */
static const char *K_SCAN = WGSL(
@group(0) @binding(1) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> off: array<u32>;
@group(0) @binding(3) var<storage, read_write> cur: array<atomic<u32>>;
@compute @workgroup_size(1)
fn main() {
    var s = 0u;
    for (var i = 0u; i < P.n.x; i++) {
        off[i] = s;
        s += atomicLoad(&cnt[1u + i]);
        atomicStore(&cur[i], 0u);
    }
    off[P.n.x] = s;
}
);

/* each contact into its bodies' lists */
static const char *K_FILL = WGSL(
@group(0) @binding(1) var<storage, read> C: array<Contact>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read_write> cur: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> L: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c >= min(atomicLoad(&cnt[0]), P.n.y)) { return; }
    let a = bitcast<u32>(C[c].n.w);
    let b = bitcast<u32>(C[c].ra.w);
    L[off[a] + atomicAdd(&cur[a], 1u)] = 2u * c;
    if (b != NONE) { L[off[b] + atomicAdd(&cur[b], 1u)] = 2u * c + 1u; }
}
);

/* each contact before solving, every substep: where its points are now (la, lb moved with
   their bodies) and so its depth, arms, tangents, effective masses with each body's mass
   split among its contacts, the bias (bounce, push-out, or for a gap how fast it may close),
   and the impulse to start from (warm start), applied first: in the step's first substep
   the same contact's in the step before, then its own of the substep before */
static const char *K_PREP = WGSL(
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> off: array<u32>;
@group(0) @binding(5) var<storage, read> Cp: array<Contact>;
@group(0) @binding(6) var<storage, read> Lp: array<u32>;
@group(0) @binding(7) var<storage, read> offp: array<u32>;

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
}
);

/* one iteration for every contact at once: the impulse that stops it approaching (and the
   friction, at most mu times the normal one), from the bodies' velocities as they are */
static const char *K_SOLVE = WGSL(
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c >= min(atomicLoad(&cnt[0]), P.n.y)) { return; }
    var k = C[c];
    let ia = bitcast<u32>(k.n.w);
    let ib = bitcast<u32>(k.ra.w);
    let a = B[ia];
    var dv = a.vel.xyz + cross(a.ang.xyz, k.ra.xyz);
    if (ib != NONE) { let b = B[ib]; dv = dv - (b.vel.xyz + cross(b.ang.xyz, k.rb.xyz)); }
    let n = k.n.xyz;
    let jn = max(k.j.x + k.t1.w * (k.rb.w - dot(dv, n)), 0.0);
    let lim = k.imp.w * jn;
    let jt1 = clamp(k.j.y - k.t2.w * dot(dv, k.t1.xyz), -lim, lim);
    let jt2 = clamp(k.j.z - k.j.w * dot(dv, k.t2.xyz), -lim, lim);
    k.imp = vec4f(n * (jn - k.j.x) + k.t1.xyz * (jt1 - k.j.y) + k.t2.xyz * (jt2 - k.j.z), k.imp.w);
    k.j = vec4f(jn, jt1, jt2, k.j.w);
    C[c] = k;
}
);


/* each body takes the sum of its contacts' impulses (each with the whole mass: the shares
   average back to it) */
static const char *K_APPLY = WGSL(
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> C: array<Contact>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read> L: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }
    let b = B[i];
    if (b.vel.w == 0.0 || off[i] == off[i + 1u]) { return; }
    var dv = vec3f(0.0);
    var dw = vec3f(0.0);
    for (var q = off[i]; q < off[i + 1u]; q++) {
        let e = L[q];
        let k = C[e >> 1u];
        var j = k.imp.xyz;
        var r = k.ra.xyz;
        if ((e & 1u) != 0u) { j = -j; r = k.rb.xyz; }
        dv += j;
        dw += cross(r, j);
    }
    B[i].vel = vec4f(b.vel.xyz + dv * b.vel.w, b.vel.w);
    B[i].ang = vec4f(b.ang.xyz + inv_inertia(b) * dw, b.ang.w);
}
);

/* velocities -> positions, orientations */
static const char *K_POS = WGSL(
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }
    let b = B[i];
    if (b.vel.w == 0.0) { return; }
    let h = P.h.x;
    let q = b.rot;
    let w = b.ang.xyz;
    /* dq/dt = 0.5 (0, w) q */
    let d = vec4f(w.x * q.w + w.y * q.z - w.z * q.y, -w.x * q.z + w.y * q.w + w.z * q.x,
                  w.x * q.y - w.y * q.x + w.z * q.w, -w.x * q.x - w.y * q.y - w.z * q.z);
    B[i].pos = vec4f(b.pos.xyz + b.vel.xyz * h, b.pos.w);
    B[i].rot = normalize(q + d * (0.5 * h));
}
);

// ---- the C side ----------------------------------------------------------------------------

typedef struct {
    uint32_t n[4];
    float h[4], g[4], lo[4], hi[4], d[4], h2[4];
    uint32_t m[4];
} GRP;

enum { R_VEL, R_BROAD, R_ARGS, R_SIMPLE, R_HEAVY, R_MESHQ, R_ARGS2, R_MESHTRI, R_MESHFLUSH,
       R_SCAN, R_FILL, R_PREP, R_SOLVE, R_APPLY, R_POS, R_WAKE, R_SLEEP, NR };
static const struct { const char *name, *types; const char **body, **more, **more2; } RK[NR] = {
    {"bodies: forces",   "UWWRWW",   &K_VEL, NULL, NULL},
    {"bodies: pairs",    "URWW",     &K_BROAD, NULL, NULL},
    {"bodies: pair counts", "UWW",   &K_ARGS, NULL, NULL},
    {"bodies: contacts", "URWWRW",   &K_NARROW, &K_SIMPLE, NULL},
    {"bodies: contacts (hulls)", "URWWRWRW", &K_NARROW, &K_CONVEX, &K_HEAVY},
    {"bodies: near triangles", "UR--RWRW", &K_NARROW, &K_CONVEX, &K_MESHQ},
    {"bodies: triangle counts", "UWW", &K_ARGS2, NULL, NULL},
    {"bodies: triangles", "UR--RWRRWW", &K_NARROW, &K_CONVEX, &K_MESHTRI},
    {"bodies: contacts (meshes)", "URWWRW--RW", &K_NARROW, &K_CONVEX, &K_MESHFLUSH},
    {"bodies: lists 1",  "UWWW",     &K_SCAN, NULL, NULL},
    {"bodies: lists 2",  "URWRWW",   &K_FILL, NULL, NULL},
    {"bodies: prepare",  "URWWRRRR", &K_PREP, NULL, NULL},
    {"bodies: solve",    "URWW",     &K_SOLVE, NULL, NULL},
    {"bodies: apply",    "UWRRR",    &K_APPLY, NULL, NULL},
    {"bodies: move",     "UW",       &K_POS, NULL, NULL},
    {"bodies: wake",     "URWR",     &K_WAKE, NULL, NULL},
    {"bodies: sleep",    "UWR",      &K_SLEEP, NULL, NULL},
};

#define BODY_BYTES 128
#define CONTACT_BYTES 160
#define MAX_INFLIGHT 2

struct GpuRigid {
    Gpu *g;
    PhysWorld *pw;
    WGPUBindGroupLayout bgl[NR];
    WGPUComputePipeline pipe[NR];
    WGPUBuffer par, bodies, forces, cnt, cursor, wake;
    WGPUBuffer pairs, pc, args;                  // the candidate pairs' lists, their counts, dispatch sizes
    WGPUBuffer memo;                             // hull pairs: what decided them last step (K_HEAVY, P3)
    int memo_rd;                                 // its table read this step (the other is written)
    WGPUBuffer tris, store, scount;              // body - mesh: the triangles to test, their contacts per pair
    size_t cap_pairs;                            // per list
    WGPUBuffer shapes;                           // the hulls' and meshes' shapes (G), and the unit box
    size_t shapes_cap;
    int nshapes;                                 // RigidWorld.geoms in it
    uint32_t unit_box;                           // its shape index in G
    WGPUBuffer contacts[2], lists[2], offs[2];   // this step's and the one before's
    size_t cap_bodies, cap_contacts;             // what the buffers hold
    int nb;                                      // bodies on the GPU
    int prev_nb;                                 // bodies in the last substep (its lists)
    int parity;
    WGPUBindGroup bg[NR][2];
    float *staging;
    // the bodies back: one readback at a time
    WGPUBuffer rb;
    size_t rb_cap;
    int rb_pending, rb_n;
    int inflight, wait;
    GpuTimer *timer;                             // LR_GPU_TIMING=1: GPU time per kernel
};

static const char *const RK_NAMES[NR] = {"forces", "pairs (bounding spheres)", "pair counts", "contacts: balls, boxes",
    "contacts: hulls", "mesh: near triangles", "mesh: triangle counts", "mesh: triangles", "mesh: 4 contacts per pair",
    "contact lists 1", "contact lists 2", "prepare (warm start)", "solve", "apply", "move", "wake", "sleep"};

static GpuRigid *live[4];
static int is_live(const GpuRigid *gr){
    for (int i = 0; i < 4; i++) if (gr && live[i] == gr) return 1;
    return 0;
}

static WGPUBuffer storage(Gpu *g, const char *label, size_t bytes){
    return gpu_buffer(g, label, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
}

static void release(WGPUBuffer *b){ if (*b){ wgpuBufferDestroy(*b); wgpuBufferRelease(*b); *b = NULL; } }

static void release_groups(GpuRigid *gr){
    for (int k = 0; k < NR; k++)
        for (int p = 0; p < 2; p++) if (gr->bg[k][p]){ wgpuBindGroupRelease(gr->bg[k][p]); gr->bg[k][p] = NULL; }
}

static void make_groups(GpuRigid *gr){
    Gpu *g = gr->g;
    release_groups(gr);
    for (int p = 0; p < 2; p++){
        WGPUBuffer C = gr->contacts[p], L = gr->lists[p], O = gr->offs[p];
        WGPUBuffer Cp = gr->contacts[!p], Lp = gr->lists[!p], Op = gr->offs[!p];
#define BG(k, ...) do { WGPUBuffer b_[] = {__VA_ARGS__}; gr->bg[k][p] = gpu_bind(g, gr->bgl[k], b_, (int)(sizeof(b_) / sizeof(b_[0]))); } while (0)
        BG(R_VEL, gr->par, gr->bodies, gr->cnt, gr->forces, gr->pc, gr->wake);
        BG(R_BROAD, gr->par, gr->bodies, gr->pairs, gr->pc);
        BG(R_ARGS, gr->par, gr->pc, gr->args);
        BG(R_SIMPLE, gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc);
        BG(R_HEAVY, gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc, gr->shapes, gr->memo);
        { WGPUBuffer b_[] = {gr->par, gr->bodies, NULL, NULL, gr->pairs, gr->pc, gr->shapes, gr->tris};
          gr->bg[R_MESHQ][p] = gpu_bind(g, gr->bgl[R_MESHQ], b_, 8); }
        BG(R_ARGS2, gr->par, gr->pc, gr->args);
        { WGPUBuffer b_[] = {gr->par, gr->bodies, NULL, NULL, gr->pairs, gr->pc, gr->shapes, gr->tris, gr->store, gr->scount};
          gr->bg[R_MESHTRI][p] = gpu_bind(g, gr->bgl[R_MESHTRI], b_, 10); }
        { WGPUBuffer b_[] = {gr->par, gr->bodies, C, gr->cnt, gr->pairs, gr->pc, NULL, NULL, gr->store, gr->scount};
          gr->bg[R_MESHFLUSH][p] = gpu_bind(g, gr->bgl[R_MESHFLUSH], b_, 10); }
        BG(R_SCAN, gr->par, gr->cnt, O, gr->cursor);
        BG(R_FILL, gr->par, C, gr->cnt, O, gr->cursor, L);
        BG(R_PREP, gr->par, gr->bodies, C, gr->cnt, O, Cp, Lp, Op);
        BG(R_SOLVE, gr->par, gr->bodies, C, gr->cnt);
        BG(R_APPLY, gr->par, gr->bodies, C, O, L);
        BG(R_POS, gr->par, gr->bodies);
        BG(R_WAKE, gr->par, gr->bodies, gr->wake, gr->forces);
        BG(R_SLEEP, gr->par, gr->bodies, gr->forces);
#undef BG
    }
}

// room for nb bodies (and their contacts); 1 = the buffers were made again (their bodies lost)
static int reserve(GpuRigid *gr, int nb){
    size_t want = (size_t)(nb > 0 ? nb : 1);
    if (want <= gr->cap_bodies && gr->bodies) return 0;
    size_t cap = 64;
    while (cap < want) cap *= 2;
    Gpu *g = gr->g;
    WGPUBuffer *all[] = {&gr->bodies, &gr->forces, &gr->cnt, &gr->cursor, &gr->wake, &gr->contacts[0], &gr->contacts[1],
                         &gr->lists[0], &gr->lists[1], &gr->offs[0], &gr->offs[1], &gr->pairs, &gr->pc, &gr->args,
                         &gr->tris, &gr->store, &gr->scount, &gr->memo};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) release(all[i]);
    gr->cap_bodies = cap;
    gr->cap_contacts = 32 * cap;
    gr->bodies = storage(g, "bodies", cap * BODY_BYTES);
    gr->forces = storage(g, "bodies: forces", cap * 32);
    gr->cnt = storage(g, "bodies: contact counts", (cap + 1) * 4);
    gr->cursor = storage(g, "bodies: list cursors", cap * 4);
    gr->wake = storage(g, "bodies: to wake", cap * 4);
    { uint32_t *z = calloc(cap, 4); if (z){ wgpuQueueWriteBuffer(g->queue, gr->wake, 0, z, cap * 4); free(z); } }
    gr->cap_pairs = 32 * cap;
    gr->pairs = storage(g, "bodies: candidate pairs", 3 * gr->cap_pairs * 8);
    gr->pc = storage(g, "bodies: pair counts", 16);
    gr->args = gpu_buffer(g, "bodies: pair dispatches", 80, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst);
    gr->tris = storage(g, "bodies: triangles to test", 64 * gr->cap_pairs * 8);
    gr->store = storage(g, "bodies: mesh contacts", 64 * gr->cap_pairs * 32);
    gr->scount = storage(g, "bodies: mesh contact counts", gr->cap_pairs * 4);
    gr->memo = storage(g, "bodies: hull pairs remembered", 2 * 13 * (2 * gr->cap_pairs) * 4);   // (zeroed: empty)
    gr->memo_rd = 0;
    for (int p = 0; p < 2; p++){
        gr->contacts[p] = storage(g, "contacts", gr->cap_contacts * CONTACT_BYTES);
        gr->lists[p] = storage(g, "contact lists", gr->cap_contacts * 2 * 4);
        gr->offs[p] = storage(g, "contact list starts", (cap + 1) * 4);
    }
    free(gr->staging);
    gr->staging = malloc(cap * BODY_BYTES);
    make_groups(gr);
    gr->prev_nb = 0;          // no lists of the last substep: no warm start this time
    return 1;
}

static void put4(float *d, Vec3 v, float w){ d[0] = (float)v.x; d[1] = (float)v.y; d[2] = (float)v.z; d[3] = w; }

// ---- the shapes (G): see K_CONVEX ----
typedef struct { float *f; size_t n, cap; } Rows;   // vec4 rows
static float *row(Rows *r){
    if (r->n >= r->cap){
        size_t cap = r->cap ? r->cap * 2 : 1024;
        float *nf = realloc(r->f, cap * 16);
        if (!nf) return NULL;
        r->f = nf;
        r->cap = cap;
    }
    float *o = r->f + 4 * r->n++;
    memset(o, 0, 16);
    return o;
}
static void urow(Rows *r, uint32_t a, uint32_t b, uint32_t c, uint32_t d){
    float *o = row(r);
    if (!o) return;
    uint32_t u[4] = {a, b, c, d};
    memcpy(o, u, 16);
}
static void put_hull(Rows *r, const Convex *c, size_t hdr){
    uint32_t v0 = (uint32_t)r->n;
    for (int i = 0; i < c->nv; i++){ float *o = row(r); if (o) put4(o, c->v[i], 0); }
    uint32_t f0 = (uint32_t)r->n;
    for (int f = 0; f < c->nf; f++){
        float *o = row(r);
        if (o) put4(o, c->f[f].n, (float)c->f[f].d);
        urow(r, (uint32_t)c->f[f].first, (uint32_t)c->f[f].count, 0, 0);
    }
    uint32_t l0 = (uint32_t)r->n;
    int nl = 0;
    for (int f = 0; f < c->nf; f++) nl += c->f[f].count;
    for (int k = 0; k < nl; k++) urow(r, (uint32_t)c->fv[k], 0, 0, 0);
    uint32_t e0 = (uint32_t)r->n;
    for (int e = 0; e < c->ne; e++) urow(r, (uint32_t)c->e[e].a, (uint32_t)c->e[e].b, (uint32_t)c->e[e].fa, (uint32_t)c->e[e].fb);
    uint32_t h0[4] = {(uint32_t)c->nv, (uint32_t)c->nf, (uint32_t)c->ne, 2}, h1[4] = {v0, f0, l0, e0};
    memcpy(r->f + 4 * hdr, h0, 16);
    memcpy(r->f + 4 * (hdr + 1), h1, 16);
}
static void put_mesh(Rows *r, const MeshTree *t, size_t hdr){
    uint32_t node0 = (uint32_t)r->n;
    for (int i = 0; i < t->nnodes; i++){
        const MeshNode *nd = &t->nodes[i];
        float *o = row(r);
        if (o){ put4(o, nd->box.lo, 0); uint32_t u = (uint32_t)nd->left; memcpy(o + 3, &u, 4); }
        o = row(r);
        if (o){ put4(o, nd->box.hi, 0); uint32_t u = (uint32_t)nd->right; memcpy(o + 3, &u, 4); }
        urow(r, (uint32_t)nd->first, (uint32_t)nd->count, 0, 0);
    }
    uint32_t tri0 = (uint32_t)r->n;
    for (int q = 0; q < t->ntris; q++){              // in the leaves' order
        const Vec3 *c = t->tri + 3 * (size_t)t->order[q];
        for (int k = 0; k < 3; k++){ float *o = row(r); if (o) put4(o, c[k], 0); }
    }
    uint32_t h0[4] = {(uint32_t)t->nnodes, (uint32_t)t->ntris, 0, 3}, h1[4] = {node0, tri0, 0, 0};
    memcpy(r->f + 4 * hdr, h0, 16);
    memcpy(r->f + 4 * (hdr + 1), h1, 16);
}

static void make_groups(GpuRigid *gr);

// every shape of the world (they never change once made) and the unit box, when new ones came
static void upload_shapes(GpuRigid *gr, const RigidWorld *w){
    if (gr->shapes && gr->nshapes == w->ngeoms) return;
    static Convex unit;
    static int made;
    if (!made){ convex_box(vec3(1, 1, 1), &unit); made = 1; }
    Rows r = {NULL, 0, 0};
    int n = w->ngeoms + 1;
    for (int i = 0; i < 2 * n; i++) row(&r);          // the headers
    for (int i = 0; i < w->ngeoms; i++){
        const RigidGeom *g = w->geoms[i];
        if (g->kind == BODY_HULL) put_hull(&r, &g->hull, 2 * (size_t)i);
        else put_mesh(&r, &g->tree, 2 * (size_t)i);
    }
    put_hull(&r, &unit, 2 * (size_t)w->ngeoms);
    gr->unit_box = (uint32_t)w->ngeoms;
    if (!r.f) return;
    if (gpu_buffer_reserve(gr->g, &gr->shapes, &gr->shapes_cap, r.n * 16,
                           WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "bodies: shapes") && gr->bodies) make_groups(gr);
    wgpuQueueWriteBuffer(gr->g->queue, gr->shapes, 0, r.f, r.n * 16);
    free(r.f);
    gr->nshapes = w->ngeoms;
}

static uint32_t shape_index(const RigidWorld *w, const RigidGeom *g){
    for (int k = 0; k < w->ngeoms; k++) if (w->geoms[k] == g) return (uint32_t)k;
    return 0;
}

static void pack_body(const RigidWorld *w, const Body *b, float *o){
    memset(o, 0, BODY_BYTES);
    float kind = b->shape == BODY_SPHERE ? 0.0f : b->shape == BODY_BOX ? 1.0f : b->shape == BODY_HULL ? 2.0f : 3.0f;
    put4(o, b->pos, kind);
    if (b->g){ uint32_t gi = shape_index(w, b->g); memcpy(o + 28, &gi, 4); }
    o[4] = (float)b->rot.x; o[5] = (float)b->rot.y; o[6] = (float)b->rot.z; o[7] = (float)b->rot.w;
    put4(o + 8, b->vel, (float)b->inv_mass);
    put4(o + 12, b->angvel, (float)b->restitution);
    put4(o + 16, b->half, (float)b->friction);
    put4(o + 20, b->inv_inertia, 0);
}

static void on_bodies(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuRigid *gr = u1;
    if (status != WGPUMapAsyncStatus_Success || !is_live(gr)) return;
    gr->rb_pending = 0;
    RigidWorld *w = &gr->pw->rigid;
    const float *d = wgpuBufferGetConstMappedRange(gr->rb, 0, (size_t)gr->rb_n * BODY_BYTES + 16);
    if (d){
        for (int i = 0; i < gr->rb_n && i < w->nbodies; i++){
            Body *b = &w->bodies[i];
            if (b->dirty) continue;            // changed on the CPU since: the GPU takes it again
            const float *o = d + 32 * (size_t)i;
            b->pos = vec3(o[0], o[1], o[2]);
            b->rot = quat(o[7], o[4], o[5], o[6]);
            b->vel = vec3(o[8], o[9], o[10]);
            b->angvel = vec3(o[12], o[13], o[14]);
            b->sleep_time = o[24];
            b->sleeping = o[25] > 0.5f;
        }
        uint32_t nc;
        memcpy(&nc, d + 32 * (size_t)gr->rb_n, 4);
        w->ncontacts = (int)(nc < gr->cap_contacts ? nc : gr->cap_contacts);
    }
    wgpuBufferUnmap(gr->rb);
}

static void on_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg; (void)u2;
    GpuRigid *gr = u1;
    if (is_live(gr) && gr->inflight > 0) gr->inflight--;
}

static void dispatch1(WGPUComputePassEncoder pass, GpuRigid *gr, int k, size_t threads){
    wgpuComputePassEncoderSetPipeline(pass, gr->pipe[k]);
    wgpuComputePassEncoderSetBindGroup(pass, 0, gr->bg[k][gr->parity], 0, NULL);
    if (k == R_SCAN || k == R_ARGS || k == R_ARGS2) wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    else if (k == R_SIMPLE) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 0);
    else if (k == R_HEAVY) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 16);
    else if (k == R_MESHQ) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 64);
    else if (k == R_MESHFLUSH) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 32);
    else if (k == R_MESHTRI) wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, gr->args, 48);
    else gpu_dispatch(pass, threads);
}

// kernel k in the substep's pass, or (timed) in a pass of its own
static void dispatch(WGPUCommandEncoder enc, WGPUComputePassEncoder *pass, GpuRigid *gr, int k, size_t threads){
    const WGPUPassTimestampWrites *tw = gpu_timer_next(gr->timer, k);
    if (tw){
        if (*pass){ wgpuComputePassEncoderEnd(*pass); wgpuComputePassEncoderRelease(*pass); *pass = NULL; }
        WGPUComputePassDescriptor d;
        memset(&d, 0, sizeof(d));
        d.timestampWrites = tw;
        WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &d);
        dispatch1(p, gr, k, threads);
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        return;
    }
    if (!*pass) *pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    dispatch1(*pass, gr, k, threads);
}

static int rigid_step_gpu(void *user, PhysWorld *pw, real dt, int substeps, const Vec3 *forces){
    GpuRigid *gr = user;
    Gpu *g = gr->g;
    RigidWorld *w = &pw->rigid;
    if (dt <= 0 || gpu_state(g) != GPU_READY) return 0;
    if (gr->inflight >= MAX_INFLIGHT || gr->wait) gpu_wait(g);
    const int nb = w->nbodies;
    // the bodies the CPU changed (or all of them, into new buffers), and the forces
    int fresh = reserve(gr, nb);
    upload_shapes(gr, w);
    for (int i = 0; i < nb; i++){
        Body *b = &w->bodies[i];
        if (!fresh && !b->dirty && i < gr->nb) continue;
        pack_body(w, b, gr->staging);
        wgpuQueueWriteBuffer(g->queue, gr->bodies, (uint64_t)i * BODY_BYTES, gr->staging, BODY_BYTES);
        b->dirty = 0;
        b->force = b->torque = vec3(0, 0, 0);
    }
    gr->nb = nb;
    if (nb == 0) return 1;
    float *f = gr->staging;
    for (int i = 0; i < nb; i++){
        put4(f + 8 * i, forces ? forces[i] : vec3(0, 0, 0), 0);
        put4(f + 8 * i + 4, vec3(0, 0, 0), 0);
    }
    wgpuQueueWriteBuffer(g->queue, gr->forces, 0, f, (size_t)nb * 32);

    int steps = substeps > 0 ? substeps : 1;
    real h = dt / steps;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    {   // the memo's table written this step starts empty (its keys)
        uint64_t cap = 2 * (uint64_t)gr->cap_pairs, wr = (uint64_t)(gr->memo_rd ^ 1);
        wgpuCommandEncoderClearBuffer(enc, gr->memo, wr * 13 * cap * 4, cap * 4);
    }
    for (int s = 0; s < steps; s++){
        GRP P;
        memset(&P, 0, sizeof(P));
        P.n[0] = (uint32_t)nb;
        P.n[1] = (uint32_t)gr->cap_contacts;
        P.n[2] = (uint32_t)gr->prev_nb;
        P.n[3] = (uint32_t)w->has_room;
        P.h[0] = (float)h; P.h[1] = 0.5f; P.h[2] = 0.2f; P.h[3] = 60.0f;   // rigid.c's CONTACT_SLOP, BAUMGARTE, BOUNCE_THRESHOLD
        put4(P.g, w->gravity, (float)w->linear_damping);
        put4(P.lo, w->room.lo, (float)w->wall_restitution);
        put4(P.hi, w->room.hi, (float)w->wall_friction);
        P.d[0] = (float)w->angular_damping;
        P.d[1] = (float)dt;
        P.d[2] = 0.1f;                           // rigid.c's SPECULATIVE_MIN
        P.d[3] = 300.0f;                         // rigid.c's MAX_PUSH
        P.h2[0] = 5.0f; P.h2[1] = 0.5f;          // rigid.c's SLEEP_SPEED, SLEEP_TIME, WAKE_SPEED
        P.h2[2] = 20.0f;
        P.h2[3] = (float)gr->memo_rd;            // K_HEAVY reads this table of the memo, writes the other
        P.m[0] = gr->unit_box;
        P.m[1] = (uint32_t)w->ngeoms;
        P.m[2] = (uint32_t)gr->cap_pairs;
        P.m[3] = (uint32_t)s;
        // the parameters change between substeps only in the substep: written now, read in
        // order (a buffer per substep would do too; one write per pass keeps it simple)
        WGPUBuffer par = gr->par;
        wgpuQueueWriteBuffer(g->queue, par, 0, &P, sizeof(P));
        WGPUComputePassEncoder pass = NULL;
        if (s == 0 && w->sleep) dispatch(enc, &pass, gr, R_WAKE, (size_t)nb * nb + nb);
        dispatch(enc, &pass, gr, R_VEL, (size_t)nb);
        if (s == 0){
            // the contacts, once per step (with margins: they are followed through the substeps)
            dispatch(enc, &pass, gr, R_BROAD, (size_t)nb * nb + nb);
            dispatch(enc, &pass, gr, R_ARGS, 1);
            dispatch(enc, &pass, gr, R_SIMPLE, 0);
            dispatch(enc, &pass, gr, R_HEAVY, 0);
            dispatch(enc, &pass, gr, R_MESHQ, 0);
            dispatch(enc, &pass, gr, R_ARGS2, 1);
            dispatch(enc, &pass, gr, R_MESHTRI, 0);
            dispatch(enc, &pass, gr, R_MESHFLUSH, 0);
            dispatch(enc, &pass, gr, R_SCAN, 1);
            dispatch(enc, &pass, gr, R_FILL, gr->cap_contacts);
        }
        dispatch(enc, &pass, gr, R_PREP, gr->cap_contacts);
        dispatch(enc, &pass, gr, R_APPLY, (size_t)nb);                  // the warm start
        for (int it = 0; it < w->iterations; it++){
            dispatch(enc, &pass, gr, R_SOLVE, gr->cap_contacts);
            dispatch(enc, &pass, gr, R_APPLY, (size_t)nb);
        }
        dispatch(enc, &pass, gr, R_POS, (size_t)nb);
        if (s == steps - 1 && w->sleep){
            // the end of the step: who falls asleep
            dispatch(enc, &pass, gr, R_SLEEP, (size_t)nb);
        }
        if (pass){ wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); }
        // queue writes happen before the command buffer runs: the substeps must go one by one
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g->queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
        enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    }
    gr->prev_nb = nb;          // (the next step's contacts start from this one's)
    gr->parity ^= 1;
    gr->memo_rd ^= 1;          // (what this step wrote is read by the next)
    // the bodies back to the CPU (if the last readback has landed)
    if (!gr->rb_pending){
        size_t bytes = (size_t)nb * BODY_BYTES + 16;
        gpu_buffer_reserve(g, &gr->rb, &gr->rb_cap, bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "bodies back");
        wgpuCommandEncoderCopyBufferToBuffer(enc, gr->bodies, 0, gr->rb, 0, (size_t)nb * BODY_BYTES);
        // the contact count (cnt[0] of the last substep's buffers)
        wgpuCommandEncoderCopyBufferToBuffer(enc, gr->cnt, 0, gr->rb, (size_t)nb * BODY_BYTES, 4);
    }
    gpu_timer_finish(gr->timer, enc);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    gpu_timer_submitted(gr->timer);
    if (!gr->rb_pending){
        gr->rb_pending = 1;
        gr->rb_n = nb;
        WGPUBufferMapCallbackInfo mi;
        memset(&mi, 0, sizeof(mi));
        mi.mode = WGPUCallbackMode_AllowSpontaneous;
        mi.callback = on_bodies;
        mi.userdata1 = gr;
        wgpuBufferMapAsync(gr->rb, WGPUMapMode_Read, 0, (size_t)nb * BODY_BYTES + 16, mi);
    }
    WGPUQueueWorkDoneCallbackInfo wi;
    memset(&wi, 0, sizeof(wi));
    wi.mode = WGPUCallbackMode_AllowSpontaneous;
    wi.callback = on_done;
    wi.userdata1 = gr;
    gr->inflight++;
    wgpuQueueOnSubmittedWorkDone(g->queue, wi);
    if (gr->wait) gpu_rigid_sync(gr);
    return 1;
}

GpuRigid *gpu_rigid_create(Gpu *g, PhysWorld *pw){
    if (!g || gpu_state(g) != GPU_READY || !pw) return NULL;
    GpuRigid *gr = calloc(1, sizeof(*gr));
    if (!gr) return NULL;
    gr->g = g;
    gr->pw = pw;
    size_t cap = strlen(RHEAD) + 4;
    for (int k = 0; k < NR; k++){
        size_t n = cap + strlen(*RK[k].body) + (RK[k].more ? strlen(*RK[k].more) : 0) + (RK[k].more2 ? strlen(*RK[k].more2) : 0) + 2;
        char *src = malloc(n);
        if (!src){ gpu_rigid_destroy(gr); return NULL; }
        snprintf(src, n, "%s\n%s\n%s\n%s", RHEAD, *RK[k].body, RK[k].more ? *RK[k].more : "", RK[k].more2 ? *RK[k].more2 : "");
        gr->bgl[k] = gpu_layout(g, WGPUShaderStage_Compute, RK[k].types);
        gr->pipe[k] = gpu_compute(g, RK[k].name, src, "main", gr->bgl[k]);
        free(src);
    }
    gr->par = gpu_buffer(g, "bodies: parameters", sizeof(GRP), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    upload_shapes(gr, &pw->rigid);
    reserve(gr, pw->rigid.nbodies);
    for (int i = 0; i < pw->rigid.nbodies; i++) pw->rigid.bodies[i].dirty = 1;
    for (int i = 0; i < 4; i++) if (!live[i]){ live[i] = gr; break; }
    pw->rigid_step = rigid_step_gpu;
    pw->rigid_user = gr;
    gr->timer = gpu_timer_create(g, 512, NR);
    return gr;
}

void gpu_rigid_set_wait(GpuRigid *gr, int wait){ if (gr) gr->wait = wait; }

void gpu_rigid_sync(GpuRigid *gr){
    if (!gr) return;
    for (int k = 0; k < 2000 && gr->rb_pending; k++) gpu_wait(gr->g);
}

void gpu_rigid_destroy(GpuRigid *gr){
    if (!gr) return;
    gpu_rigid_sync(gr);
    if (gr->timer){
        char title[64];
        snprintf(title, sizeof(title), "rigid bodies, per physics step (%d bodies)", gr->nb);
        gpu_timer_print(gr->timer, title, RK_NAMES);
        gpu_timer_destroy(gr->timer);
    }
    for (int i = 0; i < 4; i++) if (live[i] == gr) live[i] = NULL;
    if (gr->pw && gr->pw->rigid_user == gr){ gr->pw->rigid_step = NULL; gr->pw->rigid_user = NULL; }
    release_groups(gr);
    WGPUBuffer *all[] = {&gr->par, &gr->shapes, &gr->bodies, &gr->forces, &gr->cnt, &gr->cursor, &gr->wake, &gr->contacts[0], &gr->contacts[1],
                         &gr->lists[0], &gr->lists[1], &gr->offs[0], &gr->offs[1], &gr->rb, &gr->memo};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) release(all[i]);
    for (int k = 0; k < NR; k++){
        if (gr->pipe[k]) wgpuComputePipelineRelease(gr->pipe[k]);
        if (gr->bgl[k]) wgpuBindGroupLayoutRelease(gr->bgl[k]);
    }
    free(gr->staging);
    free(gr);
}

#endif
