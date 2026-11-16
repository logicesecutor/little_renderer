// fluid_d2.wgsl - used by engine/gpu/fluid_gpu.c (K_D2)
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> d0: array<f32>;
@group(0) @binding(5) var<storage, read> d1: array<f32>;
@group(0) @binding(6) var<storage, read_write> dens: array<f32>;
@group(0) @binding(7) var<storage, read> solid: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= nvoxels()) { return; }
    let c = dcenter(voxel_of(id));
    let fc = vec3i(floor((c - P.lo.xyz) * P.a.x));
    if (solid_at(fc.x, fc.y, fc.z)) { dens[id] = 0.0; return; }
    let dt = P.a.y;
    let vel = vel_at(c);
    let mm = sm_d0(dgrid(c - vel * dt), DN());
    let back = s_d1(dgrid(c + vel * dt), DN());
    var r = d1[id] + 0.5 * (d0[id] - back);
    r = clamp(r, mm.y, mm.z);
    r *= P.dlo.w;
    if (P.dhi.w > 0.5) { r *= exp(-src_div(c) * dt); }
    dens[id] = select(r, 0.0, r < 1e-4);
}
