// fluid_div.wgsl - used by engine/gpu/fluid_gpu.c (K_DIV)
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> solid: array<u32>;
@group(0) @binding(5) var<storage, read_write> div: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    if (solid[id] != 0u) { div[id] = 0.0; return; }
    let c = cell_of(id);
    let d = (u[iu(c.x + 1, c.y, c.z)] - u[iu(c.x, c.y, c.z)]) + (v[iv(c.x, c.y + 1, c.z)] - v[iv(c.x, c.y, c.z)]) +
            (w[iw(c.x, c.y, c.z + 1)] - w[iw(c.x, c.y, c.z)]);
    div[id] = d * P.a.x - src_div(center(c));
}
