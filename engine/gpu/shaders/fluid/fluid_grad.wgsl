// fluid_grad.wgsl - used by engine/gpu/fluid_gpu.c (K_GRAD)
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<f32>;
@group(0) @binding(5) var<storage, read> solid: array<u32>;
@group(0) @binding(6) var<storage, read> svel: array<vec4f>;
fn pc(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || y < 0 || z < 0 || x >= NX() || y >= NY() || z >= NZ()) { return 0.0; }
    return p[cidx(x, y, z)];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    let s = P.a.x;
    let here = pc(q.x, q.y, q.z);
    if (q.y < NY() && q.z < NZ() && fixed_face(0, q).x < 0.5) { u[iu(q.x, q.y, q.z)] -= (here - pc(q.x - 1, q.y, q.z)) * s; }
    if (q.x < NX() && q.z < NZ() && fixed_face(1, q).x < 0.5) { v[iv(q.x, q.y, q.z)] -= (here - pc(q.x, q.y - 1, q.z)) * s; }
    if (q.x < NX() && q.y < NY() && fixed_face(2, q).x < 0.5) { w[iw(q.x, q.y, q.z)] -= (here - pc(q.x, q.y, q.z - 1)) * s; }
}
