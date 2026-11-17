// rigid_head.wgsl - used by engine/gpu/rigid_gpu.c (RHEAD)
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
    q: vec4f,      /* the grid (K_GRID): its cells' size, the radius beyond which a body is on the big list, its slots */
    s: vec4f,      /* settling and sleeping (rigid.c's): slow speed, anchor move, anchor time, settle damping */
    c: vec4f,      /* x: rigid.c's CCD_FRACTION; y: 1 = the step's last substep; zw: unused */
};
@group(0) @binding(0) var<uniform> P: RP;
struct Body { pos: vec4f, rot: vec4f, vel: vec4f, ang: vec4f, half: vec4f, iner: vec4f, f: vec4f, t: vec4f };
struct Contact { p: vec4f, n: vec4f, ra: vec4f, rb: vec4f, t1: vec4f, t2: vec4f, j: vec4f, imp: vec4f, la: vec4f, lb: vec4f };
const NONE: u32 = 0xffffffffu;
fn tid(gid: vec3u, nw: vec3u) -> u32 { return gid.x + gid.y * nw.x * 64u; }
fn body_radius(b: Body) -> f32 { return select(length(b.half.xyz), b.half.x, b.pos.w < 0.5); }
/* how far apart a pair may be and still get a contact (rigid.c's pair_margin): as far as
   they can close in a step (their whole relative speed), plus a little; at most half their
   sizes (faster: K_CCD) */
fn pair_margin(a: Body, b: Body, wall: bool) -> f32 {
    var closing = length(a.vel.xyz);
    var r = body_radius(a);
    var spin = length(a.ang.xyz) * r;
    if (!wall) {
        let dv = a.vel.xyz - b.vel.xyz;
        closing = length(dv);
        spin += length(b.ang.xyz) * body_radius(b);
        r += body_radius(b);
    }
    return min(P.d.z + P.d.y * (closing + spin), P.d.z + 0.5 * r);
}
/* slow: below the speed under which a body resting on something is damped and may settle */
fn body_slow(b: Body) -> bool { return length(b.vel.xyz) < P.s.x && length(b.ang.xyz) * body_radius(b) < P.s.x; }
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
/* two tangents of a unit normal (base/rng.h onb_from_normal, Duff et al. 2017) */
fn onb(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y));
}
