// fluid_coarsest.wgsl - used by engine/gpu/fluid_gpu.c (K_COARSEST)
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = lcount();
    for (var it = 0; it < 40; it++) {
        for (var color = 0u; color < 2u; color++) {
            for (var i = li; i < n; i += 256u) {
                let q = lcell(i);
                if (((u32(q.x + q.y + q.z) + color) & 1u) == 0u) {
                    let m = links[i];
                    if (m != 0u) { p[i] = (nsum(i, m) - L.h.x * rhs[i]) / ncount(m); }
                }
            }
            storageBarrier();
            workgroupBarrier();
        }
    }
}
