// rigid_avbd.wgsl - used by engine/gpu/rigid_gpu.c (K_AVBD: the AVBD kernels' constants, quaternions)
/* rigid.c's AVBD_*: the stiffnesses relative to the lighter moving body's m / h^2 (the
   smallest, the growth per unit of violation, the largest), what is kept of them from the
   step before, the overlap left, the most a body turns in one of its solves */
const KMIN = 0.003;
const BETA = 0.1;
const KMAX = 10.0;
const GAMMA = 0.99;
const MARGIN = 0.05;
const MAX_TURN = 0.3;
/* COL[i]: a body's colour (K_ACOLOR), or one of these */
const UNCOLOURED = 0xfffffffeu;
const FIXED = 0xffffffffu;
fn aq_mul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
/* q turned by w over t (rigid.c's quat_integrate, as K_POS) */
fn aq_turn(q: vec4f, w: vec3f, t: f32) -> vec4f {
    return normalize(q + aq_mul(vec4f(w, 0.0), q) * (0.5 * t));
}
/* the turn from q0 to q, as a vector (rigid.c's rot_delta) */
fn rot_delta(q: vec4f, q0: vec4f) -> vec3f {
    let d = aq_mul(q, vec4f(-q0.xyz, q0.w));
    return d.xyz * select(2.0, -2.0, d.w < 0.0);
}
/* a row's force bounds: the normal pushes (<= 0); friction within mu |the normal's| */
fn row_lo(k: Contact, r: u32) -> f32 { return select(-k.imp.w * abs(k.j.x), -3.0e38, r == 0u); }
fn row_hi(k: Contact, r: u32) -> f32 { return select(k.imp.w * abs(k.j.x), 0.0, r == 0u); }
fn row_dir(k: Contact, r: u32) -> vec3f { return select(select(k.t2.xyz, k.t1.xyz, r == 1u), k.n.xyz, r == 0u); }
