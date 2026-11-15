// fluid_sample.wgsl - used by engine/gpu/fluid_gpu.c (SAMPLE)
fn s_QN(g0: vec3f, S: vec3i) -> f32 {
    let g = clamp(g0, vec3f(0.0), vec3f(S - 1));
    let i0 = min(vec3i(g), max(S - 2, vec3i(0)));
    let i1 = min(i0 + 1, S - 1);
    let f = g - vec3f(i0);
    let a = i0.z * S.x * S.y; let b = i1.z * S.x * S.y; let c = i0.y * S.x; let d = i1.y * S.x;
    let c00 = mix(QA[u32(a + c + i0.x)], QA[u32(a + c + i1.x)], f.x);
    let c10 = mix(QA[u32(a + d + i0.x)], QA[u32(a + d + i1.x)], f.x);
    let c01 = mix(QA[u32(b + c + i0.x)], QA[u32(b + c + i1.x)], f.x);
    let c11 = mix(QA[u32(b + d + i0.x)], QA[u32(b + d + i1.x)], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
