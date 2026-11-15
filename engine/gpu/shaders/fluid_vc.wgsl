// fluid_vc.wgsl - used by engine/gpu/fluid_gpu.c (K_VC)
@group(0) @binding(1) var<storage, read> omg: array<vec4f>;
@group(0) @binding(2) var<storage, read> solid: array<u32>;
@group(0) @binding(3) var<storage, read_write> frc: array<vec4f>;
fn mag(x: i32, y: i32, z: i32) -> f32 { return omg[cidx(clamp(x, 0, NX() - 1), clamp(y, 0, NY() - 1), clamp(z, 0, NZ() - 1))].w; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    let k = 0.5 / H();
    let g = vec3f(mag(c.x + 1, c.y, c.z) - mag(c.x - 1, c.y, c.z), mag(c.x, c.y + 1, c.z) - mag(c.x, c.y - 1, c.z),
                  mag(c.x, c.y, c.z + 1) - mag(c.x, c.y, c.z - 1)) * k;
    let gl = length(g);
    var force = vec3f(0.0);
    if (gl > 1e-6 && solid[id] == 0u) { force = cross(g / gl, omg[id].xyz) * P.wind.w; }
    frc[id] = vec4f(force, 0.0);
}
