// rigid_avbdc.wgsl - used by engine/gpu/rigid_gpu.c (K_AVBDC: a contact's rows' violations, K_APRIMAL and K_ADUAL)
/* C of the normal and the tangents now: C0 (the normal's overlap left out unless `full`, the
   tangents' drift only in the last pass) + J dx, dx from the substep's start (rigid.c's avbd_c) */
fn row_cs(k: Contact, full: bool) -> vec3f {
    let ia = bitcast<u32>(k.n.w);
    let ib = bitcast<u32>(k.ra.w);
    var cs = select(vec3f(max(k.rb.w, 0.0), 0.0, 0.0), vec3f(k.rb.w, k.t1.w, k.t2.w), full);
    let dxa = B[ia].pos.xyz - AB[8u * ia].xyz;
    let dra = rot_delta(B[ia].rot, AB[8u * ia + 1u]);
    var dxb = vec3f(0.0);
    var drb = vec3f(0.0);
    if (ib != NONE) {
        dxb = B[ib].pos.xyz - AB[8u * ib].xyz;
        drb = rot_delta(B[ib].rot, AB[8u * ib + 1u]);
    }
    for (var r = 0u; r < 3u; r++) {
        let d = row_dir(k, r);
        cs[r] += dot(d, dxa - dxb) + dot(cross(k.ra.xyz, d), dra) - dot(cross(k.rb.xyz, d), drb);
    }
    return cs;
}
