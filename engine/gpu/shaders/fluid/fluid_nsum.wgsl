// fluid_nsum.wgsl - used by engine/gpu/fluid_gpu.c (NSUM)
fn nsum(i: u32, m: u32) -> f32 {
    let sy = L.d.x;
    let sz = L.d.x * L.d.y;
    var s = 0.0;
    if ((m & 1u) != 0u) { s += p[i - 1u]; }
    if ((m & 2u) != 0u) { s += p[i + 1u]; }
    if ((m & 4u) != 0u) { s += p[i - sy]; }
    if ((m & 8u) != 0u) { s += p[i + sy]; }
    if ((m & 16u) != 0u) { s += p[i - sz]; }
    if ((m & 32u) != 0u) { s += p[i + sz]; }
    return s;
}
fn ncount(m: u32) -> f32 { return f32(countOneBits(m & 63u) + (m >> 6u)); }
