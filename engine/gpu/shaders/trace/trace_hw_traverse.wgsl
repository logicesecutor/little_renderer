// trace_hw_traverse.wgsl - used by engine/gpu/trace_gpu.c (HW_TRAVERSE)
@group(0) @binding(20) var ACC: acceleration_structure;
const HW_WTRI: u32 = 0xffffffu;
const HW_WPRIM: u32 = 0xfffffeu;
fn traverse(r: Ray, tmin: f32, tmax: f32, stop_at_first: bool) -> Found {
    var rq: ray_query;
    rayQueryInitialize(&rq, ACC, RayDesc(select(RAY_FLAG_NONE, RAY_FLAG_TERMINATE_ON_FIRST_HIT, stop_at_first), 0xffu,
                                         tmin, tmax, r.o, r.d));
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        let cm = rayQueryGetCommittedIntersection(&rq);
        let tm = select(tmax, cm.t, cm.kind != RAY_QUERY_INTERSECTION_NONE);
        var t = -1.0;
        if (c.instance_custom_data == HW_WPRIM) {
            let pr = prims[c.primitive_index];
            if (pr.q.w < 0.5) { t = hit_sphere(pr, r, tmin, tm); } else { t = hit_quad(pr, r, tmin, tm); }
        } else {
            let i = c.instance_custom_data;
            let o = to_object(i, r);
            if (bitcast<u32>(prims[T.ins.y + 2u * i + 1u].c.z) == UNIT_SPHERE) { t = hit_unit_sphere(o, tmin, tm); }
            else { t = hit_unit_box(o, tmin, tm); }
        }
        if (t > 0.0) {
            rayQueryGenerateIntersection(&rq, t);
            if (stop_at_first) { rayQueryTerminate(&rq); }
        }
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
        let inst = select(h.instance_custom_data, NONE, h.instance_custom_data == HW_WTRI);
        return Found(h.t, T.opt.z + h.sbt_record_offset + h.primitive_index, h.barycentrics, inst);
    }
    if (h.kind == RAY_QUERY_INTERSECTION_GENERATED) {
        if (h.instance_custom_data == HW_WPRIM) { return Found(h.t, h.primitive_index, vec2f(0.0), NONE); }
        let i = h.instance_custom_data;
        return Found(h.t, bitcast<u32>(prims[T.ins.y + 2u * i + 1u].c.z), vec2f(0.0), i);
    }
    return Found(tmax, NONE, vec2f(0.0), NONE);
}
