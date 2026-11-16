// fluid_samp.wgsl - used by engine/gpu/fluid_gpu.c (K_SAMP)
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> around: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= P.m.x) { return; }
    let pos = bodies[6u * id].xyz;
    let r = bodies[6u * id + 5u].w;
    let k = 0.57735027;
    var dirs = array<vec3f, 14>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
        vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0), vec3f(k, k, k), vec3f(k, k, -k), vec3f(k, -k, k), vec3f(k, -k, -k),
        vec3f(-k, k, k), vec3f(-k, k, -k), vec3f(-k, -k, k), vec3f(-k, -k, -k));
    var sum = vec3f(0.0);
    for (var d = 0; d < 14; d++) { sum += vel_at(pos + dirs[d] * r); }
    around[id] = vec4f(sum / 14.0, 0.0);
}
