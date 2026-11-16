// view_stats.wgsl - used by engine/gpu/view_gpu.c (STATS)
@group(0) @binding(1) var<storage, read> dens: array<f32>;
@group(0) @binding(2) var<storage, read_write> bricks: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> S: Sel;

@compute @workgroup_size(64)
fn brick_stats(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = S.k.x;
    let v = F.vols[k];
    let id = gid.x + gid.y * nw.x * 64u;
    if (id >= v.bdims.x * v.bdims.y * v.bdims.z) { return; }
    let b = vec3i(i32(id % v.bdims.x), i32((id / v.bdims.x) % v.bdims.y), i32(id / (v.bdims.x * v.bdims.y)));
    let n = vec3i(v.dims.xyz);
    let lo = max(b * 8 - 1, vec3i(0));
    let hi = min(b * 8 + 8, n - 1);
    let own_lo = b * 8;
    let own_hi = min(b * 8 + 7, n - 1);
    var m = 0.0;
    var own_max = 0.0;
    var amin = n;
    var amax = vec3i(-1);
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let d = dens[v.dims.w + (u32(z) * v.dims.y + u32(y)) * v.dims.x + u32(x)];
                m = max(m, d);
                let q = vec3i(x, y, z);
                if (d > 0.0 && all(q >= own_lo) && all(q <= own_hi)) {
                    amin = min(amin, q);
                    amax = max(amax, q);
                    own_max = max(own_max, d);
                }
            }
        }
    }
    bricks[v.bdims.w + id] = m;
    if (amax.x >= 0) {
        let o = k * 8u;
        atomicMin(&acc[o], u32(amin.x)); atomicMin(&acc[o + 1u], u32(amin.y)); atomicMin(&acc[o + 2u], u32(amin.z));
        atomicMax(&acc[o + 3u], u32(amax.x + 1)); atomicMax(&acc[o + 4u], u32(amax.y + 1)); atomicMax(&acc[o + 5u], u32(amax.z + 1));
        atomicMax(&acc[o + 6u], bitcast<u32>(own_max));   /* >= 0: the bits order like the values */
    }
}
