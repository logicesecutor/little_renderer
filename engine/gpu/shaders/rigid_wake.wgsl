// rigid_wake.wgsl - used by engine/gpu/rigid_gpu.c (K_WAKE)
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
