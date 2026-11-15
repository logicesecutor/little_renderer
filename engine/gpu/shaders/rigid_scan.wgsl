// rigid_scan.wgsl - used by engine/gpu/rigid_gpu.c (K_SCAN)
@group(0) @binding(1) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> off: array<u32>;
@group(0) @binding(3) var<storage, read_write> cur: array<atomic<u32>>;
var<workgroup> part: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lv: vec3u) {
    let t = lv.x;
    let n = P.n.x;
    let per = (n + 255u) / 256u;
    let lo = min(t * per, n);
    let hi = min(lo + per, n);
    var s = 0u;
    for (var i = lo; i < hi; i++) { s += atomicLoad(&cnt[1u + i]); }
    part[t] = s;
    workgroupBarrier();
    for (var d = 1u; d < 256u; d = d << 1u) {
        var v = 0u;
        if (t >= d) { v = part[t - d]; }
        workgroupBarrier();
        part[t] += v;
        workgroupBarrier();
    }
    var base = part[t] - s;
    for (var i = lo; i < hi; i++) {
        off[i] = base;
        base += atomicLoad(&cnt[1u + i]);
        atomicStore(&cur[i], 0u);
    }
    if (t == 255u) { off[n] = part[255]; }
}
