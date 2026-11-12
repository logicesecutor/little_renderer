// denoise_gpu.c - a denoiser for the GPU path tracer's image, on the GPU.
//
// An edge-avoiding a-trous wavelet filter (Dammertz et al. 2010) steered by the variance of
// each pixel, as in SVGF (Schied et al. 2017) without its temporal part:
//   prep     the mean of each pixel's samples over its albedo (the irradiance, which is
//            smooth where the colour is not), and the variance of that mean: from the pixel's
//            own samples once it has 4, before that from its neighbours (a 7 x 7 window)
//   a-trous  5 passes of a 5 x 5 B3 spline with holes (1, 2, 4, 8, 16 pixels apart: a
//            65 x 65 footprint), each neighbour weighted by how alike it is: luminance
//            (relative to the noise expected there), normal and distance of the first
//            diffuse surface; the variance is filtered alongside (with the squared weights)
//   last     times the albedo again
// The guide (normal, distance, albedo) is that of the ray through the pixel's centre, through
// mirrors and glass to the first diffuse surface (trace_gpu.c). Lamps and the sky are not
// filtered; pixels seen through a medium are filtered on luminance only. As the samples add
// up the variance falls, the luminance weights tighten, and the filter fades out by itself.
//
// What the passes read is packed, because they are memory bound (25 taps, 5 times): per
// pixel 8 bytes of image (irradiance and variance, half floats) and 8 bytes of guide (the
// normal octahedron-encoded in 2 x 16 bits, the distance as a float whose sign tells the
// kind), against 48 bytes unpacked: about 3 times less to read.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdlib.h>
#include <string.h>

static const char *DENOISE = WGSL(
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

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn hash32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

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
    var sum = vec3f(0.0);
    var vsum = 0.0;
    var wsum = 0.0;
    for (var j = 0; j < 5; j++) {
        for (var i = 0; i < 5; i++) {
            var ox = (i - 2) * st;
            var oy = (j - 2) * st;
            if (jit > 0 && (i != 2 || j != 2)) {
                seed = hash32(seed);
                ox += i32(seed % u32(2 * jit + 1)) - jit;
                oy += i32((seed >> 16u) % u32(2 * jit + 1)) - jit;
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
);

#define PASSES 5

struct GpuDenoise {
    Gpu *g;
    int w, h;
    WGPUBindGroupLayout bgl;
    WGPUComputePipeline prep, atrous;
    WGPUBuffer u[PASSES + 1];      // the parameters of each pass
    WGPUBuffer a, b;               // ping-pong (irradiance and variance, packed)
    WGPUBuffer guide, out;         // the guide, packed; the image
};

GpuDenoise *gpu_denoise_create(Gpu *g, int width, int height){
    if (!g || g->state != GPU_READY) return NULL;
    GpuDenoise *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->g = g;
    d->w = width;
    d->h = height;
    d->bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRWWW");
    d->prep = gpu_compute(g, "denoiser: prep", DENOISE, "prep", d->bgl);
    d->atrous = gpu_compute(g, "denoiser: a-trous", DENOISE, "atrous", d->bgl);
    size_t n = (size_t)width * height;
    d->a = gpu_buffer(g, "denoiser A", n * 8, WGPUBufferUsage_Storage);
    d->b = gpu_buffer(g, "denoiser B", n * 8, WGPUBufferUsage_Storage);
    d->guide = gpu_buffer(g, "denoiser guide", n * 8, WGPUBufferUsage_Storage);
    d->out = gpu_buffer(g, "denoised image", n * 16, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    for (int i = 0; i <= PASSES; i++){
        struct { uint32_t size[4]; float s[4]; } P = {
            {(uint32_t)width, (uint32_t)height, i ? 1u << (i - 1) : 0u, i == PASSES ? 1u : 0u},
            {4.0f, 0.0f, 1.0f, 0.0f}};
        d->u[i] = gpu_buffer(g, "denoiser parameters", sizeof(P), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        wgpuQueueWriteBuffer(g->queue, d->u[i], 0, &P, sizeof(P));
    }
    if (!d->prep || !d->atrous || !d->a || !d->b || !d->guide || !d->out){ gpu_denoise_destroy(d); return NULL; }
    return d;
}

void gpu_denoise_encode(GpuDenoise *d, WGPUCommandEncoder enc, WGPUBuffer pixels){
    Gpu *g = d->g;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    // prep: pixels -> a (and the guide); then a -> b, b -> a, ...; the last pass (the 5th) -> out
    WGPUBindGroup bgs[PASSES + 1];
    for (int i = 0; i <= PASSES; i++){
        WGPUBuffer in = i == 0 ? d->b : (i % 2 ? d->a : d->b);
        WGPUBuffer to = i == 0 ? d->a : (i % 2 ? d->b : d->a);
        WGPUBuffer bufs[6] = {d->u[i], pixels, in, to, d->guide, d->out};
        bgs[i] = gpu_bind(g, d->bgl, bufs, 6);
        wgpuComputePassEncoderSetPipeline(pass, i == 0 ? d->prep : d->atrous);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bgs[i], 0, NULL);
        gpu_dispatch(pass, (size_t)d->w * d->h);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    for (int i = 0; i <= PASSES; i++) wgpuBindGroupRelease(bgs[i]);
}

WGPUBuffer gpu_denoise_output(GpuDenoise *d){ return d->out; }

void gpu_denoise_destroy(GpuDenoise *d){
    if (!d) return;
    WGPUBuffer bufs[PASSES + 5] = {d->a, d->b, d->guide, d->out};
    for (int i = 0; i <= PASSES; i++) bufs[4 + i] = d->u[i];
    for (int i = 0; i < PASSES + 5; i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    if (d->prep) wgpuComputePipelineRelease(d->prep);
    if (d->atrous) wgpuComputePipelineRelease(d->atrous);
    if (d->bgl) wgpuBindGroupLayoutRelease(d->bgl);
    free(d);
}

#else
typedef int denoise_gpu_unused;
#endif
