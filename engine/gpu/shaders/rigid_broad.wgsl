// rigid_broad.wgsl - used by engine/gpu/rigid_gpu.c (K_BROAD)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> pairs: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> pc: array<atomic<u32>>;
/* list 0: balls and boxes, 1: hulls, 2: a body and a fixed mesh */
fn push(l: u32, i: u32, j: u32) {
    let k = atomicAdd(&pc[l], 1u);
    if (k < P.m.z) { pairs[l * P.m.z + k] = vec2u(i, j); }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    let n = P.n.x;
    if (id >= n * n) {
        let i = id - n * n;
        if (i < n && P.n.w != 0u && B[i].vel.w > 0.0) { push(select(0u, 1u, B[i].pos.w > 1.5), i, NONE); }
        return;
    }
    let i = id % n;
    let j = id / n;
    if (j <= i) { return; }
    let a = B[i];
    let b = B[j];
    if (a.vel.w == 0.0 && b.vel.w == 0.0) { return; }
    let rai = select(length(a.half.xyz), a.half.x, a.pos.w < 0.5);
    let rbj = select(length(b.half.xyz), b.half.x, b.pos.w < 0.5);
    let d = a.pos.xyz - b.pos.xyz;
    let reach = rai + rbj + pair_margin(a, b, false);
    if (dot(d, d) > reach * reach) { return; }
    if (a.pos.w > 2.5 && b.pos.w > 2.5) { return; }
    if (a.pos.w > 2.5) { push(2u, j, i); }
    else if (b.pos.w > 2.5) { push(2u, i, j); }
    else { push(select(0u, 1u, a.pos.w > 1.5 || b.pos.w > 1.5), i, j); }
}
