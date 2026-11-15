// fluid_sample_mm.wgsl - used by engine/gpu/fluid_gpu.c (SAMPLE_MM)
fn sm_QN(g0: vec3f, S: vec3i) -> vec3f {
    let g = clamp(g0, vec3f(0.0), vec3f(S - 1));
    let i0 = min(vec3i(g), max(S - 2, vec3i(0)));
    let i1 = min(i0 + 1, S - 1);
    let f = g - vec3f(i0);
    let a = i0.z * S.x * S.y; let b = i1.z * S.x * S.y; let c = i0.y * S.x; let d = i1.y * S.x;
    let c000 = QA[u32(a + c + i0.x)]; let c100 = QA[u32(a + c + i1.x)];
    let c010 = QA[u32(a + d + i0.x)]; let c110 = QA[u32(a + d + i1.x)];
    let c001 = QA[u32(b + c + i0.x)]; let c101 = QA[u32(b + c + i1.x)];
    let c011 = QA[u32(b + d + i0.x)]; let c111 = QA[u32(b + d + i1.x)];
    let lo = min(min(min(c000, c100), min(c010, c110)), min(min(c001, c101), min(c011, c111)));
    let hi = max(max(max(c000, c100), max(c010, c110)), max(max(c001, c101), max(c011, c111)));
    let c00 = mix(c000, c100, f.x); let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x); let c11 = mix(c011, c111, f.x);
    return vec3f(mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z), lo, hi);
}
