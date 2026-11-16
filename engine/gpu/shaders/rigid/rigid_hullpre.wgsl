// rigid_hullpre.wgsl - used by engine/gpu/rigid_gpu.c (K_HULLPRE)
/* The hulls' list, one thread per pair: a hull and the walls, a ball and a hull, and what
   decided a pair last step (the memo) when it still holds: apart along the same axis, or
   touching and barely moved since its full test (its feature again, for K_HULLC). The rest
   (new, or come closer, or moved) are queued for K_HEAVY's full test. Most pairs end here:
   mesh_rain's 1500 hulls, per step, 32000 pairs of which 20000 stay apart and 5000 reuse
   their feature; these took a workgroup of 64 each in K_HEAVY. */
/* the separation of a and b along the axis that separated them (type 1 a's face i, 2 b's face
   i, 3 the edges i, j) */
fn gap_along(A: Shape, S: Shape, ty: u32, i: u32, j: u32) -> f32 {
    if (ty == 1u) { return min_along(S, sn(A, i)) - sd(A, i); }
    if (ty == 2u) { return min_along(A, sn(S, i)) - sd(S, i); }
    let L = edge_axis(A, S, i, j);
    if (dot(L, L) == 0.0) { return -1e30; }
    return min_along(S, L) + min_along(A, -L);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = tid(gid, nw);
    if (k >= min(atomicLoad(&pc[1]), P.m.z)) { return; }
    feat[k] = vec4u(0u);
    let pr = pairs[P.m.z + k];
    let ia = pr.x;
    let ib = pr.y;
    MG = margin_of(ia, ib);
    if (ib == NONE) { hull_walls(ia); return; }
    if (B[ia].pos.w < 0.5) { sphere_convex(ia, ib); return; }
    if (B[ib].pos.w < 0.5) { sphere_convex(ib, ia); return; }
    let key = pair_key(ia, ib);
    let rd = u32(P.h2.w);
    let e = memo_find(rd, key);
    if (e != NONE) {
        let ft = vec3u(memo_word(rd, e, 0u), memo_word(rd, e, 1u), memo_word(rd, e, 2u));
        let mt = vec3f(bitcast<f32>(memo_word(rd, e, 4u)), bitcast<f32>(memo_word(rd, e, 5u)), bitcast<f32>(memo_word(rd, e, 6u)));
        let mq = vec4f(bitcast<f32>(memo_word(rd, e, 8u)), bitcast<f32>(memo_word(rd, e, 9u)),
                       bitcast<f32>(memo_word(rd, e, 10u)), bitcast<f32>(memo_word(rd, e, 11u)));
        if (ft.x >= 1u && ft.x <= 3u) {
            if (gap_along(shape_of(ia), shape_of(ib), ft.x, ft.y, ft.z) > MG) { memo_put(1u - rd, key, ft, mt, mq); return; }
        } else if (ft.x >= 4u) {
            var t: vec3f;
            var q: vec4f;
            rel_pose(ia, ib, &t, &q);
            let rmin = min(body_radius(B[ia]), body_radius(B[ib]));
            if (length(t - mt) < 0.01 * rmin && abs(dot(q, mq)) > 0.99996) {
                feat[k] = vec4u(ft, 1u);
                memo_put(1u - rd, key, ft, mt, mq);
                return;
            }
        }
    }
    let n = atomicAdd(&pc[4], 1u);
    if (n < P.m.z) { feat[P.m.z + n] = vec4u(k, 0u, 0u, 0u); }
}
