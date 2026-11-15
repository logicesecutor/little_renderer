// rigid_sleep.wgsl - used by engine/gpu/rigid_gpu.c (K_SLEEP)
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
