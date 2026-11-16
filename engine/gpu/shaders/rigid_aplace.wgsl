// rigid_aplace.wgsl - used by engine/gpu/rigid_gpu.c (K_APLACE)
@group(0) @binding(1) var<storage, read> COL: array<u32>;
@group(0) @binding(2) var<storage, read> AB: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> CTL: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ORDER: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x || COL[i] == FIXED) { return; }
    /* the bodies by colour: each after the colours before its own (K_ACOUNT's counts) */
    let c = bitcast<u32>(AB[8u * i + 7u].x);
    var start = 0u;
    for (var q = 0u; q < c; q++) { start += atomicLoad(&CTL[4u + q]); }
    ORDER[start + bitcast<u32>(AB[8u * i + 7u].y)] = i;
}
