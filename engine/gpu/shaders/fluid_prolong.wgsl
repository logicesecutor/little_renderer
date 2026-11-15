// fluid_prolong.wgsl - used by engine/gpu/fluid_gpu.c (K_PROLONG)
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read> cp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    if (links[id] == 0u) { return; }
    p[id] += cp[cidx(lcell(id) / 2)];
}
