// rigid_astart.wgsl - used by engine/gpu/rigid_gpu.c (K_ASTART)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> AB: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> COL: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> CTL: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> TENT: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    let first = P.m.w == 0u;
    if (i == 0u) {
        atomicStore(&CTL[0], 0u);
        atomicStore(&CTL[1], 0u);
        if (first) { atomicStore(&CTL[2], 0u); for (var c = 4u; c < 36u; c++) { atomicStore(&CTL[c], 0u); } }
    }
    if (i >= P.n.x) { return; }
    let b = B[i];
    let fixed = b.vel.w == 0.0;
    if (first) { atomicStore(&COL[i], select(UNCOLOURED, FIXED, fixed)); atomicStore(&TENT[i], NONE); }
    /* where it starts, and (velocities from K_VEL: gravity and the forces in) where it would
       go unhindered */
    AB[8u * i] = b.pos;
    AB[8u * i + 1u] = b.rot;
    if (fixed) {
        AB[8u * i + 2u] = b.pos;
        AB[8u * i + 3u] = b.rot;
        AB[8u * i + 4u] = vec4f(0.0);
        AB[8u * i + 5u] = vec4f(0.0);
        return;
    }
    let h = P.h.x;
    let qy = aq_turn(b.rot, b.ang.xyz, h);
    AB[8u * i + 2u] = vec4f(b.pos.xyz + b.vel.xyz * h, 0.0);
    AB[8u * i + 3u] = qy;
    AB[8u * i + 4u] = b.vel;
    AB[8u * i + 5u] = b.ang;
    /* the first guess: as much of gravity as it fell with in the step before (at rest: none,
       so that it does not start pressed into what holds it) */
    let acc = AB[8u * i + 6u].xyz;
    let gg = dot(P.g.xyz, P.g.xyz);
    let wt = select(1.0, clamp(dot(acc, P.g.xyz) / gg, 0.0, 1.0), gg > 0.0);
    B[i].pos = vec4f(b.pos.xyz + b.vel.xyz * h - P.g.xyz * ((1.0 - wt) * h * h), b.pos.w);
    B[i].rot = qy;
}
