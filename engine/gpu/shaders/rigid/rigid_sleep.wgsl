// rigid_sleep.wgsl - used by engine/gpu/rigid_gpu.c (K_SLEEP)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> F: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> A: array<vec4f>;
@group(0) @binding(4) var<storage, read> C: array<Contact>;
@group(0) @binding(5) var<storage, read> off: array<u32>;
@group(0) @binding(6) var<storage, read> L: array<u32>;
/* rigid.c's fall_asleep, in two passes. SLEEP_STEP 0: how long each body has been still
   (f.x), and slow near the place and turn it had when it became slow (its anchor, A: place
   and time, turn); f.w = 1 if it is quiet (still, or going nowhere for half the time that
   puts it to sleep), asleep or fixed. SLEEP_STEP 1: those still long enough, or going
   nowhere long enough, fall asleep, unless a body they touch is awake and not quiet. */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }
    let b = B[i];
    if (SLEEP_STEP == 0u) {
        if (b.vel.w == 0.0) { B[i].f.w = 1.0; return; }
        let still = length(b.vel.xyz) < P.h2.x && length(b.ang.xyz) * body_radius(b) < P.h2.x;
        let t = select(0.0, b.f.x + P.d.y, still);
        var a0 = A[2u * i];
        let a1 = A[2u * i + 1u];
        let turn = 2.0 * acos(min(abs(dot(a1, b.rot)), 1.0));
        let near = length(b.pos.xyz - a0.xyz) < P.s.y && turn * body_radius(b) < P.s.y;
        if (body_slow(b) && a0.w > 0.0 && near) { a0.w += P.d.y; }
        else if (body_slow(b)) { a0 = vec4f(b.pos.xyz, P.d.y); A[2u * i + 1u] = b.rot; }
        else { a0.w = 0.0; }
        A[2u * i] = a0;
        B[i].f.x = t;
        B[i].f.w = select(0.0, 1.0, still || a0.w >= 0.5 * P.s.z);
        return;
    }
    if (b.vel.w == 0.0) { return; }
    let t = b.f.x;
    let at = A[2u * i].w;
    if (t < P.h2.y && at < P.s.z) { return; }
    /* (going nowhere three times as long: asleep whatever its neighbours do) */
    for (var q = off[i]; q < off[i + 1u] && at < 3.0 * P.s.z; q++) {
        let k = C[L[q] >> 1u];
        var j = bitcast<u32>(k.n.w);
        if (j == i) { j = bitcast<u32>(k.ra.w); }
        if (j == NONE || (k.p.w <= 0.0 && k.j.x <= 0.0)) { continue; }   /* a wall, or not touching */
        if (B[j].f.w < 0.5) { return; }                                  /* it touches a body still moving */
    }
    A[2u * i].w = 0.0;
    B[i].vel = vec4f(0.0);
    B[i].ang = vec4f(0.0, 0.0, 0.0, b.ang.w);
    B[i].f = vec4f(t, 1.0, b.vel.w, b.f.w);
    B[i].t.y = F[2u * i].x;     /* (t.x is a shape index's bits: not written back as a float) */
    B[i].t.z = F[2u * i].y;
    B[i].t.w = F[2u * i].z;
}
