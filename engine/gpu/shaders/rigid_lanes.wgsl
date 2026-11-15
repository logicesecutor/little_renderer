// rigid_lanes.wgsl - used by engine/gpu/rigid_gpu.c (K_LANES)
var<workgroup> sdv: array<vec3f, 64>;
var<workgroup> sdw: array<vec3f, 64>;
var<workgroup> nlist: array<u32, 8>;
var<workgroup> nmax: u32;
fn chunks(lv: u32, n: u32) -> u32 {
    if ((lv & 7u) == 0u) { nlist[lv >> 3u] = (n + 7u) / 8u; }
    workgroupBarrier();
    if (lv == 0u) {
        var m = 0u;
        for (var k = 0u; k < 8u; k++) { m = max(m, nlist[k]); }
        nmax = m;
    }
    return workgroupUniformLoad(&nmax);
}
