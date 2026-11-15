// fluid_adv.wgsl - used by engine/gpu/fluid_gpu.c (K_ADV)
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> tu: array<f32>;
@group(0) @binding(5) var<storage, read_write> tv: array<f32>;
@group(0) @binding(6) var<storage, read_write> tw: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    let dt = P.a.y;
    let s = P.a.x;
    if (q.y < NY() && q.z < NZ()) {
        let c = pos_u(q);
        tu[iu(q.x, q.y, q.z)] = s_u((c - vel_at(c) * dt - P.lo.xyz) * s - vec3f(0.0, 0.5, 0.5), SU());
    }
    if (q.x < NX() && q.z < NZ()) {
        let c = pos_v(q);
        tv[iv(q.x, q.y, q.z)] = s_v((c - vel_at(c) * dt - P.lo.xyz) * s - vec3f(0.5, 0.0, 0.5), SV());
    }
    if (q.x < NX() && q.y < NY()) {
        let c = pos_w(q);
        tw[iw(q.x, q.y, q.z)] = s_w((c - vel_at(c) * dt - P.lo.xyz) * s - vec3f(0.5, 0.5, 0.0), SW());
    }
}
