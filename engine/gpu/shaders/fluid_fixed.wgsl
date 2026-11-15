// fluid_fixed.wgsl - used by engine/gpu/fluid_gpu.c (FIXED)
fn fixed_face(ax: i32, q: vec3i) -> vec2f {
    var qq = q;
    var nn = vec3i(NX(), NY(), NZ());
    let c = qq[ax];
    let bit = 1u << u32(2 * ax);
    if (c == 0 && (P.n.w & bit) == 0u) { return vec2f(1.0, 0.0); }
    if (c == nn[ax] && (P.n.w & (bit << 1u)) == 0u) { return vec2f(1.0, 0.0); }
    var o = vec3i(0);
    o[ax] = 1;
    let pa = q - o;
    let a = solid_at(pa.x, pa.y, pa.z);
    let b = solid_at(q.x, q.y, q.z);
    if (!a && !b) { return vec2f(0.0, 0.0); }
    var s = 0.0;
    var k = 0.0;
    if (a) { s += svel[cidx(pa.x, pa.y, pa.z)][ax]; k += 1.0; }
    if (b) { s += svel[cidx(q.x, q.y, q.z)][ax]; k += 1.0; }
    return vec2f(1.0, s / k);
}
