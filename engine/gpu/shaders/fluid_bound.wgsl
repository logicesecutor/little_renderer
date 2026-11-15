// fluid_bound.wgsl - used by engine/gpu/fluid_gpu.c (K_BOUND)
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read> solid: array<u32>;
@group(0) @binding(5) var<storage, read> svel: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    if (q.y < NY() && q.z < NZ()) { let r = fixed_face(0, q); if (r.x > 0.5) { u[iu(q.x, q.y, q.z)] = r.y; } }
    if (q.x < NX() && q.z < NZ()) { let r = fixed_face(1, q); if (r.x > 0.5) { v[iv(q.x, q.y, q.z)] = r.y; } }
    if (q.x < NX() && q.y < NY()) { let r = fixed_face(2, q); if (r.x > 0.5) { w[iw(q.x, q.y, q.z)] = r.y; } }
}
