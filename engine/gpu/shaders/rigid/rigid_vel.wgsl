// rigid_vel.wgsl - used by engine/gpu/rigid_gpu.c (K_VEL)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> F: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> wake: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    let first = P.m.w == 0u;
    if (i == 0u && first) { atomicStore(&cnt[0], 0u); for (var l = 0u; l < 5u; l++) { atomicStore(&pc[l], 0u); } }
    if (i >= P.n.x) { return; }
    if (first) { atomicStore(&cnt[1u + i], 0u); }
    var b = B[i];
    if (first && b.f.y > 0.5 && atomicLoad(&wake[i]) != 0u) {
        /* woken (K_BROAD, WAKE): its mass back, still at rest */
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
