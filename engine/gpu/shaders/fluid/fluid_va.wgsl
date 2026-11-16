// fluid_va.wgsl - used by engine/gpu/fluid_gpu.c (K_VA)
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> cvel: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    cvel[id] = vec4f(0.5 * (u[iu(c.x, c.y, c.z)] + u[iu(c.x + 1, c.y, c.z)]),
                     0.5 * (v[iv(c.x, c.y, c.z)] + v[iv(c.x, c.y + 1, c.z)]),
                     0.5 * (w[iw(c.x, c.y, c.z)] + w[iw(c.x, c.y, c.z + 1)]), 0.0);
}
