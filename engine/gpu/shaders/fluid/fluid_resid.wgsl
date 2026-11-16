// fluid_resid.wgsl - used by engine/gpu/fluid_gpu.c (K_RESID)
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> res: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    let m = links[id];
    if (m == 0u) { res[id] = 0.0; return; }
    res[id] = rhs[id] - (nsum(id, m) - ncount(m) * p[id]) / L.h.x;
}
