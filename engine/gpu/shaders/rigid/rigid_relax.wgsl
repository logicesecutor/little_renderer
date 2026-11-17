// rigid_relax.wgsl - used by engine/gpu/rigid_gpu.c (K_RELAX)
/* After the bodies moved (K_MOVE), each contact's target for one more solver pass (rigid.c's
   relax): a gap may still close, a bounce keeps its speed, a touching contact only stops
   approaching. The push out of an overlap, a velocity that has done its work, is taken
   back: it does not stay in the bodies as motion. */
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> C: array<Contact>;
@group(0) @binding(3) var<storage, read> cnt: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c >= min(cnt[0], P.n.y)) { return; }
    let k = C[c];
    let ia = bitcast<u32>(k.n.w);
    let ib = bitcast<u32>(k.ra.w);
    var pb = k.lb.xyz;
    if (ib != NONE) { pb = B[ib].pos.xyz + k.rb.xyz; }
    let depth = k.la.w - dot(B[ia].pos.xyz + k.ra.xyz - pb, k.n.xyz);
    if ((bitcast<u32>(k.imp.w) >> 31u) != 0u) { return; }   /* a bounce (K_PREP): its target kept */
    C[c].rb.w = select(0.0, depth / P.h.x, depth < 0.0);
}
