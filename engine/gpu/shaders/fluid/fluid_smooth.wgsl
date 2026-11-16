// fluid_smooth.wgsl - used by engine/gpu/fluid_gpu.c (K_SMOOTH)
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    let q = lcell(id);
    if (((u32(q.x + q.y + q.z) + L.c.w) & 1u) != 0u) { return; }
    let m = links[id];
    if (m == 0u) { return; }
    p[id] = (nsum(id, m) - L.h.x * rhs[id]) / ncount(m);
}
