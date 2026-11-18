// denoise.wgsl - used by engine/gpu/denoise_gpu.c (DENOISE)
struct DP {
    size: vec4u,   /* width, height, step, 1 = the last pass */
    s: vec4f,      /* sigma luminance, -, sigma distance */
};
@group(0) @binding(0) var<uniform> D: DP;
@group(0) @binding(1) var<storage, read> px: array<vec4f>;          /* the path tracer's 4 planes */
@group(0) @binding(2) var<storage, read> src: array<vec2u>;         /* irradiance, variance: 4 halves */
@group(0) @binding(3) var<storage, read_write> dst: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> gd: array<vec2u>;    /* the guide, packed (prep writes it) */
@group(0) @binding(5) var<storage, read_write> outp: array<vec4f>;  /* the image (the last pass) */



/* an image value: irradiance and variance (variance < 0: no sample yet) */
fn put_iv(c: vec3f, v: f32) -> vec2u {
    let h = vec3f(min(c, vec3f(6e4)));
    return vec2u(pack2x16float(h.xy), pack2x16float(vec2f(h.z, clamp(v, -1.0, 6e4))));
}
fn get_iv(p: vec2u) -> vec4f { return vec4f(unpack2x16float(p.x), unpack2x16float(p.y)); }

/* the guide: the normal on an octahedron, the distance with the kind in its sign
   (kind 0: z, kind 2: -(z + 1), kind 1: -1e38) */
fn oct(n: vec3f) -> vec2f {
    let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    let s = select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    return select(p, (1.0 - abs(p.yx)) * s, n.z < 0.0);
}
fn unoct(e: vec2f) -> vec3f {
    var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}
struct G { n: vec3f, z: f32, kind: u32 };
fn get_g(p: vec2u) -> G {
    var g: G;
    let e = bitcast<f32>(p.y);
    g.kind = select(select(2u, 1u, e < -1e37), 0u, e >= 0.0);
    g.z = select(-e - 1.0, e, e >= 0.0);
    g.n = unoct(unpack2x16snorm(p.x));
    return g;
}

fn irradiance(k: u32, N: u32) -> vec3f {
    let a = px[k];
    let g = px[3u * N + k];
    let mean = a.rgb / max(a.w, 1.0);
    return select(mean, mean / max(g.xyz, vec3f(1e-3)), g.w == 0.0);
}

@compute @workgroup_size(64)
fn prep(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = D.size.x;
    let H = D.size.y;
    let N = W * H;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let a = px[k];
    let g = px[3u * N + k];
    let n = px[2u * N + k];
    let e = select(select(-(n.w + 1.0), -2e38, g.w == 1.0), n.w, g.w == 0.0);
    gd[k] = vec2u(pack2x16snorm(oct(n.xyz)), bitcast<u32>(e));
    if (a.w == 0.0) { dst[k] = put_iv(vec3f(0.0), -1.0); return; }
    let irr = irradiance(k, N);
    if (g.w == 1.0) { dst[k] = put_iv(irr, 0.0); return; }
    var v = 0.0;
    if (a.w >= 4.0) {
        let m = px[N + k];
        let el = m.x / a.w;
        v = max(m.y / a.w - el * el, 0.0) / a.w;
    } else {
        /* too few samples of its own: the spread of the neighbours' means, on the same surface */
        let x = i32(k % W);
        let y = i32(k / W);
        var s1 = 0.0;
        var s2 = 0.0;
        var c = 0.0;
        for (var dy = -3; dy <= 3; dy++) {
            for (var dx = -3; dx <= 3; dx++) {
                let qx = x + dx;
                let qy = y + dy;
                if (qx < 0 || qy < 0 || qx >= i32(W) || qy >= i32(H)) { continue; }
                let q = u32(qy) * W + u32(qx);
                if (px[q].w == 0.0 || px[3u * N + q].w != g.w) { continue; }
                if (g.w == 0.0 && dot(px[2u * N + q].xyz, n.xyz) < 0.9) { continue; }
                let l = luma(irradiance(q, N));
                s1 += l;
                s2 += l * l;
                c += 1.0;
            }
        }
        let el = s1 / c;
        v = max(s2 / c - el * el, 0.0);
    }
    dst[k] = put_iv(irr, v);
}

@compute @workgroup_size(64)
fn atrous(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = D.size.x;
    let H = D.size.y;
    let N = W * H;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let last = D.size.w == 1u;
    let c = get_iv(src[k]);
    let gp = get_g(gd[k]);
    if (c.w < 0.0 || gp.kind == 1u) {    /* no sample yet; a lamp, the sky: as they are */
        if (last) { outp[k] = select(vec4f(c.rgb, 1.0), vec4f(0.0), c.w < 0.0); } else { dst[k] = src[k]; }
        return;
    }
    let x = i32(k % W);
    let y = i32(k / W);
    let st = i32(D.size.z);
    let h = array<f32, 5>(1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);

    /* the variance, blurred a little (3 x 3) so that one unlucky pixel does not stand out */
    var v = 0.0;
    var vw = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let qx = clamp(x + dx, 0, i32(W) - 1);
            let qy = clamp(y + dy, 0, i32(H) - 1);
            let s = get_iv(src[u32(qy) * W + u32(qx)]);
            if (s.w < 0.0) { continue; }
            let w = select(0.5, 0.25, dx != 0) * select(0.5, 0.25, dy != 0);
            v += w * s.w;
            vw += w;
        }
    }
    v /= max(vw, 1e-6);
    let lp = luma(c.rgb);

    let surface = gp.kind == 0u;
    /* how fast the distance changes from pixel to pixel here (the smaller side, so that an
       edge next to the pixel does not count) */
    var gx = 0.0;
    var gy = 0.0;
    if (surface) {
        let l = get_g(gd[u32(y) * W + u32(max(x - 1, 0))]).z;
        let r = get_g(gd[u32(y) * W + u32(min(x + 1, i32(W) - 1))]).z;
        let d = get_g(gd[u32(max(y - 1, 0)) * W + u32(x)]).z;
        let u = get_g(gd[u32(min(y + 1, i32(H) - 1)) * W + u32(x)]).z;
        gx = min(abs(gp.z - l), abs(gp.z - r));
        gy = min(abs(gp.z - d), abs(gp.z - u));
    }

    /* the taps st pixels apart, each moved at random within the hole around it: the regular
       holes would print a grid on smooth walls */
    var seed = hash32((k * 0x9e3779b9u) ^ (u32(st) * 0x85ebca6bu));
    let jit = st / 2;
    let jn = u32(2 * jit + 1);
    var sum = vec3f(0.0);
    var vsum = 0.0;
    var wsum = 0.0;
    for (var j = 0; j < 5; j++) {
        for (var i = 0; i < 5; i++) {
            var ox = (i - 2) * st;
            var oy = (j - 2) * st;
            if (jit > 0 && (i != 2 || j != 2)) {
                /* 0 .. 2 jit from 16 random bits each by a product and a shift (a remainder
                   is a division: 48 of them a pixel were a good part of a pass) */
                seed = hash32(seed);
                ox += i32(((seed & 0xffffu) * jn) >> 16u) - jit;
                oy += i32(((seed >> 16u) * jn) >> 16u) - jit;
            }
            let qx = x + ox;
            let qy = y + oy;
            if (qx < 0 || qy < 0 || qx >= i32(W) || qy >= i32(H)) { continue; }
            let q = u32(qy) * W + u32(qx);
            let s = get_iv(src[q]);
            var w = h[i] * h[j];
            if (q != k) {
                let gq = get_g(gd[q]);
                if (s.w < 0.0 || gq.kind != gp.kind) { continue; }
                /* the noise of the noisier of the two: a bright outlier (a large variance) is
                   then spread to its neighbours, as much as they take from it; with the
                   centre's alone it would keep its energy to itself, and the image darken */
                w *= exp(-abs(lp - luma(s.rgb)) / (D.s.x * sqrt(max(v, s.w)) + 1e-6));
                if (surface) {
                    var wn = max(dot(gp.n, gq.n), 0.0);   /* to the power 128 */
                    for (var e = 0; e < 7; e++) { wn *= wn; }
                    w *= wn;
                    let expect = gx * abs(f32(ox)) + gy * abs(f32(oy));
                    w *= exp(-abs(gp.z - gq.z) / (D.s.z * expect + 1e-3 * gp.z + 1e-6));
                }
            }
            sum += w * s.rgb;
            vsum += w * w * s.w;
            wsum += w;
        }
    }
    let o = sum / wsum;
    if (last) {
        outp[k] = vec4f(select(o, o * px[3u * N + k].xyz, surface), 1.0);
    } else {
        dst[k] = put_iv(o, vsum / (wsum * wsum));
    }
}
