// rigid_acount.wgsl - used by engine/gpu/rigid_gpu.c (K_ACOUNT)
@group(0) @binding(1) var<storage, read> COL: array<u32>;
@group(0) @binding(2) var<storage, read_write> AB: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> CTL: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x || COL[i] == FIXED) { return; }
    /* the colour's pass (those past the passes, and the uncoloured, in the last), its place in it */
    let c = min(COL[i], u32(P.q.w) - 1u);
    AB[8u * i + 7u] = vec4f(bitcast<f32>(c), bitcast<f32>(atomicAdd(&CTL[4u + c], 1u)), 0.0, 0.0);
}
