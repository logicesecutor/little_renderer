// fluid_vd.wgsl - used by engine/gpu/fluid_gpu.c (K_VD)
@group(0) @binding(1) var<storage, read> frc: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> w: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    let dt = P.a.y;
    let f = frc[id];
    if (c.x > 0) { u[iu(c.x, c.y, c.z)] += 0.5 * (f.x + frc[cidx(c.x - 1, c.y, c.z)].x) * dt; }
    if (c.y > 0) { v[iv(c.x, c.y, c.z)] += 0.5 * (f.y + frc[cidx(c.x, c.y - 1, c.z)].y) * dt; }
    if (c.z > 0) { w[iw(c.x, c.y, c.z)] += 0.5 * (f.z + frc[cidx(c.x, c.y, c.z - 1)].z) * dt; }
}
