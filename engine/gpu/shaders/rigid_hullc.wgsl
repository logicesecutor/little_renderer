// rigid_hullc.wgsl - used by engine/gpu/rigid_gpu.c (K_HULLC)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> feat: array<vec4u>;
/* the hull pairs' contacts from the features K_HEAVY chose: one thread per pair (the clipping
   is one thread's work: in K_HEAVY its 63 others waited) */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let k = id.x;
    if (k >= min(min(atomicLoad(&pc[1]), P.m.z), 65535u)) { return; }
    let f = feat[k];
    if (f.w == 0u) { return; }
    let pr = pairs[P.m.z + k];
    MG = margin_of(pr.x, pr.y);
    feature_contacts(pr.x, pr.y, shape_of(pr.x), shape_of(pr.y), f.xyz);
}
