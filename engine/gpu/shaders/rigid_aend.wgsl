// rigid_aend.wgsl - used by engine/gpu/rigid_gpu.c (K_AEND)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> V: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x || B[i].vel.w == 0.0) { return; }
    B[i].vel = vec4f(V[2u * i].xyz, B[i].vel.w);       /* the bounced velocities (K_ABOUNCE) */
    B[i].ang = vec4f(V[2u * i + 1u].xyz, B[i].ang.w);
}
