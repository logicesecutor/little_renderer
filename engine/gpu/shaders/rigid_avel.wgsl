// rigid_avel.wgsl - used by engine/gpu/rigid_gpu.c (K_AVEL)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> AB: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> CTL: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i == 0u) { atomicStore(&CTL[0], 0u); }          /* (the last pass: colour 0 first) */
    if (i >= P.n.x) { return; }
    let b = B[i];
    if (b.vel.w == 0.0) { return; }
    /* the velocities: the moves over h (from y: exact in free flight), and the acceleration
       the next step's first guess takes gravity from */
    let h = P.h.x;
    let vy = AB[8u * i + 4u].xyz;
    let v = vy + (b.pos.xyz - AB[8u * i + 2u].xyz) / h;
    AB[8u * i + 6u] = vec4f((v - vy) / h + P.g.xyz, 0.0);
    B[i].vel = vec4f(v, b.vel.w);
    B[i].ang = vec4f(AB[8u * i + 5u].xyz + rot_delta(b.rot, AB[8u * i + 3u]) / h, b.ang.w);
}
