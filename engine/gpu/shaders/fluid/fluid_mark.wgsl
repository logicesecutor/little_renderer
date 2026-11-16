// fluid_mark.wgsl - used by engine/gpu/fluid_gpu.c (K_MARK)
@group(0) @binding(1) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> solid: array<u32>;
@group(0) @binding(3) var<storage, read_write> svel: array<vec4f>;
@group(0) @binding(4) var<storage, read> occ: array<u32>;
const OCC = 48;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = center(cell_of(id));
    var s = 0u;
    var vel = vec3f(0.0);
    for (var b = 0u; b < P.m.x; b++) {
        let o = 6u * b;
        let pos = bodies[o];
        let d = c - pos.xyz;
        let r = bodies[o + 4u].w;
        if (dot(d, d) > r * r) { continue; }
        var inside = false;
        if (pos.w < 0.5) {
            inside = dot(d, d) <= bodies[o + 1u].w * bodies[o + 1u].w;
        } else if (pos.w > 1.5) {
            let half = vec3f(bodies[o + 1u].w, bodies[o + 2u].w, bodies[o + 3u].w);
            let l = vec3f(dot(bodies[o + 1u].xyz, d), dot(bodies[o + 2u].xyz, d), dot(bodies[o + 3u].xyz, d));
            let q = vec3i(floor((l + half) / (2.0 * half) * f32(OCC)));
            if (all(q >= vec3i(0)) && all(q < vec3i(OCC))) {
                let i = u32((q.z * OCC + q.y) * OCC + q.x) + u32(pos.w - 2.0 + 0.5) * 1024u;   /* in units of 32 words */
                inside = ((occ[i >> 5u] >> (i & 31u)) & 1u) != 0u;
            }
        } else {
            inside = abs(dot(bodies[o + 1u].xyz, d)) <= bodies[o + 1u].w && abs(dot(bodies[o + 2u].xyz, d)) <= bodies[o + 2u].w &&
                     abs(dot(bodies[o + 3u].xyz, d)) <= bodies[o + 3u].w;
        }
        if (inside) { s = 1u; vel = bodies[o + 4u].xyz + cross(bodies[o + 5u].xyz, d); }
    }
    solid[id] = s;
    svel[id] = vec4f(vel, 0.0);
}
