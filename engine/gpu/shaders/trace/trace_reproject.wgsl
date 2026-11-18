// trace_reproject.wgsl - used by engine/gpu/trace_gpu.c (REPROJECT)
struct RP {
    oo: vec4f, oul: vec4f, oh: vec4f, ov: vec4f,     /* the old camera */
    no: vec4f, nul: vec4f, nh: vec4f, nv: vec4f,     /* the new one */
    dims: vec4u,                                     /* width, height, history */
};
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> old: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cur: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = R.dims.x;
    let H = R.dims.y;
    let N = W * H;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let g2 = cur[2u * N + k];
    let g3 = cur[3u * N + k];
    let x = f32(k % W);
    let y = f32(k / W);
    let d = normalize(R.nul.xyz + R.nh.xyz * ((x + 0.5) / f32(W)) - R.nv.xyz * ((y + 0.5) / f32(H)) - R.no.xyz);
    let far = g2.w > 1e29;
    let p = R.no.xyz + d * select(g2.w, 1e6, far);
    /* into the old camera: the screen is 1 unit in front of the eye, horizontal and vertical
       perpendicular to the view direction c */
    let f = R.oul.xyz - R.oo.xyz;
    let c = f + 0.5 * R.oh.xyz - 0.5 * R.ov.xyz;
    let q0 = p - R.oo.xyz;
    let den = dot(q0, c);
    var acc0 = vec4f(0.0);
    var acc1 = vec4f(0.0);
    var nsum = 0.0;
    var wsum = 0.0;
    if (den > 0.0) {
        let q = q0 * (dot(c, c) / den) - f;
        let fx = dot(q, R.oh.xyz) / dot(R.oh.xyz, R.oh.xyz) * f32(W) - 0.5;
        let fy = -dot(q, R.ov.xyz) / dot(R.ov.xyz, R.ov.xyz) * f32(H) - 0.5;
        let x0 = floor(fx);
        let y0 = floor(fy);
        let dist = length(q0);
        for (var j = 0; j < 4; j++) {
            let tx = x0 + f32(j & 1);
            let ty = y0 + f32(j >> 1);
            if (tx < 0.0 || ty < 0.0 || tx >= f32(W) || ty >= f32(H)) { continue; }
            let o = u32(ty) * W + u32(tx);
            let s = old[o];
            if (s.w == 0.0) { continue; }
            let o3 = old[3u * N + o];
            if (o3.w != g3.w) { continue; }
            let o2 = old[2u * N + o];
            if (!far) {
                let ux = u32(tx);
                let uy = u32(ty);
                let zl = old[2u * N + o - select(0u, 1u, ux > 0u)].w;
                let zr = old[2u * N + o + select(0u, 1u, ux + 1u < W)].w;
                let zu = old[2u * N + o - select(0u, W, uy > 0u)].w;
                let zd = old[2u * N + o + select(0u, W, uy + 1u < H)].w;
                let grad = min(abs(zl - o2.w), abs(zr - o2.w)) + min(abs(zu - o2.w), abs(zd - o2.w));
                if (abs(o2.w - dist) > select(0.02, 0.25, g3.w == 2.0) * dist + grad) { continue; }
            }
            if (g3.w == 0.0 && dot(o2.xyz, g2.xyz) < 0.9) { continue; }
            let w = (1.0 - abs(fx - tx)) * (1.0 - abs(fy - ty));
            acc0 += w * s / s.w;
            acc1 += w * old[N + o] / s.w;
            nsum += w * s.w;
            wsum += w;
        }
    }
    if (wsum < 0.05) {
        cur[k] = vec4f(0.0);
        cur[N + k] = vec4f(0.0);
        return;
    }
    let n = min(nsum / wsum, f32(R.dims.z));
    cur[k] = vec4f(acc0.rgb / wsum * n, n);
    let m = acc1 / wsum * n;
    cur[N + k] = vec4f(m.xy, 0.0, 0.0);      /* (z: adaptive sampling's stop mark, not carried over) */
}
