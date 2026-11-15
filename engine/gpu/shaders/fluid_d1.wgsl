// fluid_d1.wgsl - used by engine/gpu/fluid_gpu.c (K_D1)
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> d0: array<f32>;
@group(0) @binding(5) var<storage, read_write> d1: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= nvoxels()) { return; }
    let c = dcenter(voxel_of(id));
    d1[id] = s_d0(dgrid(c - vel_at(c) * P.a.y), DN());
}
