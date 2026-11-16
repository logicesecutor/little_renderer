// fluid_vb.wgsl - used by engine/gpu/fluid_gpu.c (K_VB)
@group(0) @binding(1) var<storage, read> cvel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> omg: array<vec4f>;
fn at(x: i32, y: i32, z: i32) -> vec4f { return cvel[cidx(clamp(x, 0, NX() - 1), clamp(y, 0, NY() - 1), clamp(z, 0, NZ() - 1))]; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    let k = 0.5 / H();
    let dwdy = (at(c.x, c.y + 1, c.z).z - at(c.x, c.y - 1, c.z).z) * k;
    let dvdz = (at(c.x, c.y, c.z + 1).y - at(c.x, c.y, c.z - 1).y) * k;
    let dudz = (at(c.x, c.y, c.z + 1).x - at(c.x, c.y, c.z - 1).x) * k;
    let dwdx = (at(c.x + 1, c.y, c.z).z - at(c.x - 1, c.y, c.z).z) * k;
    let dvdx = (at(c.x + 1, c.y, c.z).y - at(c.x - 1, c.y, c.z).y) * k;
    let dudy = (at(c.x, c.y + 1, c.z).x - at(c.x, c.y - 1, c.z).x) * k;
    let o = vec3f(dwdy - dvdz, dudz - dwdx, dvdx - dudy);
    omg[id] = vec4f(o, length(o));
}
