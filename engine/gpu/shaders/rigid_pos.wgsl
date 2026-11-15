// rigid_pos.wgsl - used by engine/gpu/rigid_gpu.c (K_POS)
@group(0) @binding(1) var<storage, read_write> B: array<Body>;
@group(0) @binding(2) var<storage, read> V: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }
    let b = B[i];
    if (b.vel.w == 0.0) { return; }
    /* the iterations' velocities, the body's again */
    let vel = V[2u * i];
    let ang = V[2u * i + 1u];
    B[i].vel = vel;
    B[i].ang = ang;
    let h = P.h.x;
    let q = b.rot;
    let w = ang.xyz;
    /* dq/dt = 0.5 (0, w) q */
    let d = vec4f(w.x * q.w + w.y * q.z - w.z * q.y, -w.x * q.z + w.y * q.w + w.z * q.x,
                  w.x * q.y - w.y * q.x + w.z * q.w, -w.x * q.x - w.y * q.y - w.z * q.z);
    B[i].pos = vec4f(b.pos.xyz + vel.xyz * h, b.pos.w);
    B[i].rot = normalize(q + d * (0.5 * h));
}
