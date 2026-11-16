// rigid_fill.wgsl - used by engine/gpu/rigid_gpu.c (K_FILL)
@group(0) @binding(1) var<storage, read> C: array<Contact>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read_write> cur: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> L: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let c = tid(gid, nw);
    if (c >= min(atomicLoad(&cnt[0]), P.n.y)) { return; }
    let a = bitcast<u32>(C[c].n.w);
    let b = bitcast<u32>(C[c].ra.w);
    L[off[a] + atomicAdd(&cur[a], 1u)] = 2u * c;
    if (b != NONE) { L[off[b] + atomicAdd(&cur[b], 1u)] = 2u * c + 1u; }
}
