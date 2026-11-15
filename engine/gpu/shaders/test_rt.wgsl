// test_rt.wgsl - used by examples/test_gpu_rt.c (KERNEL)
@group(0) @binding(0) var acc: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> o: array<vec4f>;
const NX = 32u;
const NY = 32u;
fn ray_origin(i: u32) -> vec3f {
    let x = -9.0 + 18.0 * (f32(i % NX) + 0.37) / f32(NX);
    let y = -3.0 + 6.0 * (f32(i / NX) + 0.61) / f32(NY);
    return vec3f(x, y, -10.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= NX * NY) { return; }
    let org = ray_origin(i);
    let dir = vec3f(0.0, 0.0, 1.0);
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(RAY_FLAG_NONE, 0xffu, 0.001, 1000.0, org, dir));
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind != RAY_QUERY_INTERSECTION_AABB) { continue; }
        /* a box stands for the unit sphere, in the instance's own space */
        let oo = c.world_to_object * vec4f(org, 1.0);
        let od = c.world_to_object * vec4f(dir, 0.0);
        let b = dot(oo, od);
        let a = dot(od, od);
        let disc = b * b - a * (dot(oo, oo) - 1.0);
        if (disc < 0.0) { continue; }
        let t = (-b - sqrt(disc)) / a;
        let cm = rayQueryGetCommittedIntersection(&rq);
        let tmax = select(1000.0, cm.t, cm.kind != RAY_QUERY_INTERSECTION_NONE);
        if (t > 0.001 && t < tmax) { rayQueryGenerateIntersection(&rq, t); }
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    var t = -1.0;
    if (h.kind != RAY_QUERY_INTERSECTION_NONE) { t = h.t; }
    o[2u * i] = vec4f(t, f32(h.kind), f32(h.instance_custom_data), f32(h.primitive_index));
    o[2u * i + 1u] = vec4f(f32(h.instance_index), h.barycentrics.x, h.barycentrics.y, select(0.0, 1.0, h.front_face));
}
