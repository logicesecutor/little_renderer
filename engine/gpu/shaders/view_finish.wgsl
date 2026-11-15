// view_finish.wgsl - used by engine/gpu/view_gpu.c (FINISH)
@group(0) @binding(1) var<storage, read> acc: array<u32>;
@group(0) @binding(2) var<storage, read_write> VS: array<VStat, 2>;
@group(0) @binding(3) var<uniform> S: Sel;

@compute @workgroup_size(1)
fn vol_finish() {
    let k = S.k.x;
    let v = F.vols[k];
    let o = k * 8u;
    let mx = bitcast<f32>(acc[o + 6u]);
    if (acc[o + 3u] == 0u || !(mx > 0.0)) {
        VS[k].alo = vec4f(1.0); VS[k].ahi = vec4f(-1.0); VS[k].mx = vec4f(0.0);
        return;
    }
    let vox = (v.hi.xyz - v.lo.xyz) / vec3f(v.dims.xyz);
    let c0 = v.lo.xyz + (vec3f(f32(acc[o]), f32(acc[o + 1u]), f32(acc[o + 2u])) + 0.5) * vox;
    let c1 = v.lo.xyz + (vec3f(f32(acc[o + 3u]), f32(acc[o + 4u]), f32(acc[o + 5u])) - 0.5) * vox;
    /* trilinear interpolation spreads a voxel's density up to one voxel around its center */
    VS[k].alo = vec4f(max(c0 - vox, v.lo.xyz), 0.0);
    VS[k].ahi = vec4f(min(c1 + vox, v.hi.xyz), 0.0);
    VS[k].mx = vec4f(mx, 0.0, 0.0, 0.0);
}
