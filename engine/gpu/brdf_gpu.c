// brdf_gpu.c - the BSDF of core/brdf.h in WGSL (GPU_WGSL_BSDF), for the GPU's path tracer
// (and rasterizer), and a check of it against the C (gpu_bsdf_run, examples/test_gpu_brdf.c).
//
// The WGSL is the C function by function, in f32; core/brdf.h explains the model. A module
// that includes it defines bsdf_tab(k): float k of the tables gpu_bsdf_tables packs
// (core/brdf_tables.h: BRDF_E, BRDF_B, BRDF_E_AVG, BRDF_B_AVG, one after the other).
#include "gpu/brdf_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/brdf.h"
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

const char *const GPU_WGSL_BSDF = WGSL(
const BSDF_SMOOTH: f32 = 0.02;
const BSDF_PI: f32 = 3.14159265;
const BSDF_SPEC: i32 = 0;
const BSDF_DIFF: i32 = 1;
const BSDF_TRANS: i32 = 2;
const BSDF_MS: i32 = 3;
const BSDF_N: u32 = 32u;
const BSDF_TAB_B: u32 = 1024u;
const BSDF_TAB_EAVG: u32 = 2048u;
const BSDF_TAB_BAVG: u32 = 2080u;

struct Bsdf {
    base: vec3f, metallic: f32,
    transmission: f32, rough: f32, alpha: f32, eta: f32,
    e_avg: f32, es_avg: f32, kd_norm: f32, kms_norm: f32,
    ms: vec3f, perfect: u32,
    p: vec4f,
    cv: f32, es_v: f32, e_v: f32, pad: f32,
};
struct BsdfSample { l: vec3f, pdf: f32, w: vec3f, delta: bool, through: bool, ok: bool };

fn bsdf_lum(c: vec3f) -> f32 { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

fn bsdf_table(off: u32, rough: f32, mu: f32) -> f32 {
    let x = rough * f32(BSDF_N - 1u);
    let y = clamp(mu, 0.0, 1.0) * f32(BSDF_N - 1u);
    let j = min(u32(x), BSDF_N - 2u);
    let i = min(u32(y), BSDF_N - 2u);
    let fx = x - f32(j);
    let fy = y - f32(i);
    let r0 = off + j * BSDF_N + i;
    let r1 = r0 + BSDF_N;
    let e0 = bsdf_tab(r0) + fy * (bsdf_tab(r0 + 1u) - bsdf_tab(r0));
    let e1 = bsdf_tab(r1) + fy * (bsdf_tab(r1 + 1u) - bsdf_tab(r1));
    return e0 + fx * (e1 - e0);
}
fn bsdf_avg(off: u32, rough: f32) -> f32 {
    let x = rough * f32(BSDF_N - 1u);
    let j = min(u32(x), BSDF_N - 2u);
    return bsdf_tab(off + j) + (x - f32(j)) * (bsdf_tab(off + j + 1u) - bsdf_tab(off + j));
}

fn bsdf_fresnel_dielectric(cos_i0: f32, eta: f32) -> f32 {
    let cos_i = clamp(cos_i0, 0.0, 1.0);
    let s2 = eta * eta * (1.0 - cos_i * cos_i);
    if (s2 >= 1.0) { return 1.0; }
    let c = select(cos_i, sqrt(1.0 - s2), eta > 1.0);
    var f0 = (1.0 - eta) / (1.0 + eta);
    f0 *= f0;
    let x = 1.0 - c;
    let x2 = x * x;
    return f0 + (1.0 - f0) * x2 * x2 * x;
}
fn bsdf_fresnel_dielectric_avg(eta: f32) -> f32 {
    let e = select(eta, 1.0 / eta, eta > 1.0);
    var f0 = (1.0 - e) / (1.0 + e);
    f0 *= f0;
    let avg = f0 + (1.0 - f0) / 21.0;
    return select(avg, 1.0 - (1.0 - avg) / (eta * eta), eta > 1.0);
}
fn bsdf_fresnel_metal(f0: vec3f, c: f32) -> vec3f {
    let x = 1.0 - clamp(c, 0.0, 1.0);
    let x2 = x * x;
    return f0 + (vec3f(1.0) - f0) * (x2 * x2 * x);
}

fn ggx_d(nh: f32, s2: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = s2 + a2 * nh * nh;
    return a2 / (BSDF_PI * d * d);
}
fn ggx_s2(n: vec3f, h: vec3f) -> f32 { let c = cross(n, h); return dot(c, c); }
fn ggx_g1(c0: f32, a: f32) -> f32 {
    let c = abs(c0);
    return 2.0 * c / (c + sqrt(a * a + (1.0 - a * a) * c * c));
}
fn ggx_vis(cv0: f32, cl0: f32, a: f32) -> f32 {
    let cv = abs(cv0);
    let cl = abs(cl0);
    let a2 = a * a;
    let d = cl * sqrt(cv * cv * (1.0 - a2) + a2) + cv * sqrt(cl * cl * (1.0 - a2) + a2);
    return select(0.0, 0.5 / d, d > 0.0);
}

fn bsdf_onb(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * c, -s * n.x), vec3f(c, s + n.y * n.y * a, -n.y));
}
fn bsdf_cosine(n: vec3f, u: vec2f) -> vec3f {
    let phi = 2.0 * BSDF_PI * u.x;
    let rad = sqrt(u.y);
    let tb = bsdf_onb(n);
    return tb[0] * (rad * cos(phi)) + tb[1] * (rad * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.y));
}
fn bsdf_refract(d: vec3f, n: vec3f, eta: f32) -> vec3f {
    let cos_i = min(dot(-d, n), 1.0);
    let perp = (d + n * cos_i) * eta;
    return perp - n * sqrt(abs(1.0 - dot(perp, perp)));
}
fn ggx_sample_vndf(n: vec3f, v: vec3f, a: f32, u: vec2f) -> vec3f {
    let tb = bsdf_onb(n);
    let vh = normalize(vec3f(a * dot(v, tb[0]), a * dot(v, tb[1]), dot(v, n)));
    let phi = 2.0 * BSDF_PI * u.x;
    let z = (1.0 - u.y) * (1.0 + vh.z) - vh.z;
    let st = sqrt(max(0.0, 1.0 - z * z));
    let hs = vec3f(st * cos(phi) + vh.x, st * sin(phi) + vh.y, z + vh.z);
    let hl = normalize(vec3f(a * hs.x, a * hs.y, max(hs.z, 0.0)));
    return tb[0] * hl.x + tb[1] * hl.y + n * hl.z;
}

fn bsdf_albedo(b: ptr<function, Bsdf>, mu: f32) -> f32 { return bsdf_table(0u, (*b).rough, mu); }
fn bsdf_spec_albedo(b: ptr<function, Bsdf>, mu: f32) -> f32 {
    if ((*b).perfect != 0u || (*b).eta > 1.0) { return bsdf_fresnel_dielectric(mu, (*b).eta); }
    var f0 = (1.0 - (*b).eta) / (1.0 + (*b).eta);
    f0 *= f0;
    let e = bsdf_albedo(b, mu);
    let bb = bsdf_table(BSDF_TAB_B, (*b).rough, mu);
    return f0 * (e - bb) + bb;
}

/* base colour, metallic, roughness, index of refraction, transmission (material_resolved);
   outside: v on the side the geometric normal points to */
fn bsdf_make(base: vec3f, metallic: f32, roughness: f32, ior0: f32, transmission: f32, outside: bool) -> Bsdf {
    var b: Bsdf;
    b.base = base;
    b.metallic = clamp(metallic, 0.0, 1.0);
    b.transmission = clamp(transmission, 0.0, 1.0);
    let r = clamp(roughness, 0.0, 1.0);
    b.perfect = select(0u, 1u, r <= BSDF_SMOOTH);
    b.rough = r;
    b.alpha = max(r * r, 1e-4);
    let ior = select(1.5, ior0, ior0 > 0.0);
    b.eta = select(1.0 / ior, ior, b.transmission > 0.0 && !outside);
    b.e_avg = bsdf_avg(BSDF_TAB_EAVG, r);
    let b_avg = bsdf_avg(BSDF_TAB_BAVG, r);
    var f0 = (1.0 - b.eta) / (1.0 + b.eta);
    f0 *= f0;
    b.es_avg = f0 * (b.e_avg - b_avg) + b_avg;
    var esa = b.es_avg;
    if (b.perfect != 0u || b.eta > 1.0) { esa = bsdf_fresnel_dielectric_avg(b.eta); }
    b.kd_norm = 1.0 / (1.0 - esa);
    b.kms_norm = 1.0 / (BSDF_PI * (1.0 - b.e_avg));
    b.ms = vec3f(0.0);
    if (b.perfect == 0u && b.metallic > 0.0) {
        let e = b.e_avg;
        let fa = b.base + (vec3f(1.0) - b.base) / 21.0;
        b.ms = fa * fa * e / (vec3f(1.0) - fa * (1.0 - e));
    }
    b.cv = -1.0;
    return b;
}

fn bsdf_view(b: ptr<function, Bsdf>, cv: f32) {
    if ((*b).cv == cv) { return; }
    (*b).cv = cv;
    let fd = bsdf_fresnel_dielectric(cv, (*b).eta);
    let fs = select(min(fd, 0.9), fd, (*b).perfect != 0u);
    let m = (*b).metallic;
    let t = (*b).transmission;
    let lb = bsdf_lum((*b).base);
    (*b).es_v = 0.0;
    if ((1.0 - m) * (1.0 - t) > 0.0) { (*b).es_v = bsdf_spec_albedo(b, cv); }
    (*b).e_v = 1.0;
    if ((*b).perfect == 0u && m > 0.0) { (*b).e_v = bsdf_albedo(b, cv); }
    let w = vec4f((1.0 - m) * fs + m * bsdf_lum(bsdf_fresnel_metal((*b).base, cv)),
                  (1.0 - m) * (1.0 - t) * (1.0 - (*b).es_v) * lb,
                  (1.0 - m) * t * (1.0 - fs) * lb,
                  m * bsdf_lum((*b).ms) * (1.0 - (*b).e_v));
    let s = w.x + w.y + w.z + w.w;
    (*b).p = select(vec4f(0.0), w / s, s > 0.0);
}

/* the BSDF without its perfect lobes (rgb) and the density bsdf_sample draws l with (w) */
fn bsdf_eval(b: ptr<function, Bsdf>, n: vec3f, v: vec3f, l: vec3f) -> vec4f {
    var f = vec3f(0.0);
    var pdf = 0.0;
    let cv = dot(v, n);
    let cl = dot(l, n);
    if (cv <= 0.0) { return vec4f(0.0); }
    bsdf_view(b, cv);
    let p = (*b).p;
    let m = (*b).metallic;
    let t = (*b).transmission;
    let a = (*b).alpha;
    if (cl > 0.0) {
        let h = normalize(v + l);
        let vh = clamp(dot(v, h), 0.0, 1.0);
        let nh = dot(n, h);
        let fd = bsdf_fresnel_dielectric(vh, (*b).eta);
        var kd = (1.0 - m) * (1.0 - t);
        if (kd > 0.0) {
            kd *= (1.0 - (*b).es_v) * (1.0 - bsdf_spec_albedo(b, cl)) * (*b).kd_norm;
            f = (*b).base * (kd / BSDF_PI);
            pdf += p.y * cl / BSDF_PI;
        }
        if ((*b).perfect == 0u && m > 0.0) {
            let k = m * (1.0 - (*b).e_v) * (1.0 - bsdf_albedo(b, cl)) * (*b).kms_norm;
            f += (*b).ms * k;
            pdf += p.w * cl / BSDF_PI;
        }
        if ((*b).perfect == 0u && nh > 0.0) {
            let d = ggx_d(nh, ggx_s2(n, h), a);
            let fm = bsdf_fresnel_metal((*b).base, vh);
            let s = d * ggx_vis(cv, cl, a);
            f += (vec3f((1.0 - m) * fd) + m * fm) * s;
            pdf += p.x * ggx_g1(cv, a) * d / (4.0 * cv);
        }
    } else if (cl < 0.0 && (*b).perfect == 0u && t > 0.0 && m < 1.0) {
        let eta = (*b).eta;
        var h = normalize(-(v * eta + l));
        if (dot(h, n) < 0.0) { h = -h; }
        let vh = dot(v, h);
        let lh = dot(l, h);
        let nh = dot(n, h);
        var den = eta * vh + lh;
        den *= den;
        if (vh > 0.0 && lh < 0.0 && nh > 0.0 && den > 1e-12) {
            let fd = bsdf_fresnel_dielectric(vh, eta);
            let d = ggx_d(nh, ggx_s2(n, h), a);
            let g2 = ggx_vis(cv, cl, a) * 4.0 * cv * -cl;
            let ft = vh * -lh * (1.0 - fd) * d * g2 / (cv * -cl * den);
            f = (*b).base * ((1.0 - m) * t * ft);
            pdf = p.z * ggx_g1(cv, a) * vh * d / cv * -lh / den;
        }
    }
    return vec4f(f, pdf);
}

fn bsdf_pick(b: ptr<function, Bsdf>, ul: f32) -> i32 {
    let p = (*b).p;
    var lobe = 0;
    var acc = p[0];
    loop {
        if (lobe >= 3 || !(ul >= acc || p[lobe] <= 0.0)) { break; }
        lobe += 1;
        acc += p[lobe];
    }
    return select(-1, lobe, p[lobe] > 0.0);
}
fn bsdf_lobe_delta(b: ptr<function, Bsdf>, lobe: i32) -> bool { return (*b).perfect != 0u && (lobe == BSDF_SPEC || lobe == BSDF_TRANS); }
fn bsdf_nlobes(b: ptr<function, Bsdf>) -> u32 {
    let p = (*b).p;
    return select(0u, 1u, p.x > 0.0) + select(0u, 1u, p.y > 0.0) + select(0u, 1u, p.z > 0.0) + select(0u, 1u, p.w > 0.0);
}
fn bsdf_rough_part(b: ptr<function, Bsdf>) -> bool {
    return (*b).perfect == 0u || (1.0 - (*b).metallic) * (1.0 - (*b).transmission) > 0.0;
}

fn bsdf_sample_lobe(b: ptr<function, Bsdf>, lobe: i32, n: vec3f, v: vec3f, u: vec2f) -> BsdfSample {
    var s: BsdfSample;
    s.ok = false;
    s.delta = false;
    s.through = false;
    s.pdf = 0.0;
    let cv = dot(v, n);
    if (cv <= 0.0 || lobe < 0) { return s; }
    bsdf_view(b, cv);
    let p = (*b).p;
    if (p[lobe] <= 0.0) { return s; }
    let m = (*b).metallic;
    let t = (*b).transmission;
    if ((*b).perfect != 0u && (lobe == BSDF_SPEC || lobe == BSDF_TRANS)) {
        let fd = bsdf_fresnel_dielectric(cv, (*b).eta);
        if (lobe == BSDF_SPEC) {
            let fm = bsdf_fresnel_metal((*b).base, cv);
            s.l = reflect(-v, n);
            s.w = (vec3f((1.0 - m) * fd) + m * fm) * (1.0 / p[lobe]);
        } else {
            if (fd >= 1.0) { return s; }
            s.l = bsdf_refract(-v, n, (*b).eta);
            s.w = (*b).base * ((1.0 - m) * t * (1.0 - fd) / p[lobe]);
            s.through = true;
        }
        s.delta = true;
        s.ok = true;
        return s;
    }
    var l: vec3f;
    if (lobe == BSDF_DIFF || lobe == BSDF_MS) {
        l = bsdf_cosine(n, u);
    } else {
        let h = ggx_sample_vndf(n, v, (*b).alpha, u);
        let vh = dot(v, h);
        if (vh <= 0.0) { return s; }
        if (lobe == BSDF_SPEC) {
            l = reflect(-v, h);
        } else {
            if ((*b).eta * (*b).eta * (1.0 - vh * vh) >= 1.0) { return s; }
            l = bsdf_refract(-v, h, (*b).eta);
        }
    }
    if ((lobe == BSDF_TRANS) != (dot(l, n) < 0.0)) { return s; }
    let fp = bsdf_eval(b, n, v, l);
    if (!(fp.w > 0.0)) { return s; }
    s.l = l;
    s.w = fp.xyz * (abs(dot(l, n)) / fp.w);
    s.pdf = fp.w;
    s.through = dot(l, n) < 0.0;
    s.ok = true;
    return s;
}
fn bsdf_sample(b: ptr<function, Bsdf>, n: vec3f, v: vec3f, ul: f32, u: vec2f) -> BsdfSample {
    var s: BsdfSample;
    s.ok = false;
    let cv = dot(v, n);
    if (cv <= 0.0) { return s; }
    bsdf_view(b, cv);
    return bsdf_sample_lobe(b, bsdf_pick(b, ul), n, v, u);
}
);

void gpu_bsdf_tables(float *out){
    memcpy(out, BRDF_E, sizeof(BRDF_E));
    memcpy(out + 1024, BRDF_B, sizeof(BRDF_B));
    memcpy(out + 2048, BRDF_E_AVG, sizeof(BRDF_E_AVG));
    memcpy(out + 2080, BRDF_B_AVG, sizeof(BRDF_B_AVG));
}

// ---- the check: the WGSL on cases, against the C ----
static const char *CHECK = WGSL(
struct Tab { t: array<vec4f, GPU_BSDF_TAB_VEC4> };
@group(0) @binding(0) var<uniform> TB: Tab;
@group(0) @binding(1) var<storage, read> cin: array<f32>;
@group(0) @binding(2) var<storage, read_write> cout: array<f32>;
fn bsdf_tab(k: u32) -> f32 { return TB.t[k >> 2u][k & 3u]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i * 17u >= arrayLength(&cin)) { return; }
    let o = i * 17u;
    let n = vec3f(0.0, 0.0, 1.0);
    let v = vec3f(cin[o + 8u], cin[o + 9u], cin[o + 10u]);
    let l = vec3f(cin[o + 11u], cin[o + 12u], cin[o + 13u]);
    var b = bsdf_make(vec3f(cin[o], cin[o + 1u], cin[o + 2u]), cin[o + 3u], cin[o + 4u], cin[o + 5u], cin[o + 6u], cin[o + 7u] > 0.5);
    let fp = bsdf_eval(&b, n, v, l);
    var b2 = bsdf_make(vec3f(cin[o], cin[o + 1u], cin[o + 2u]), cin[o + 3u], cin[o + 4u], cin[o + 5u], cin[o + 6u], cin[o + 7u] > 0.5);
    let s = bsdf_sample(&b2, n, v, cin[o + 14u], vec2f(cin[o + 15u], cin[o + 16u]));
    let r = i * 16u;
    cout[r] = fp.x; cout[r + 1u] = fp.y; cout[r + 2u] = fp.z; cout[r + 3u] = fp.w;
    cout[r + 4u] = select(0.0, 1.0, s.ok); cout[r + 5u] = select(0.0, 1.0, s.delta);
    cout[r + 6u] = select(0.0, 1.0, s.through); cout[r + 7u] = s.pdf;
    cout[r + 8u] = s.l.x; cout[r + 9u] = s.l.y; cout[r + 10u] = s.l.z;
    cout[r + 11u] = s.w.x; cout[r + 12u] = s.w.y; cout[r + 13u] = s.w.z;
    cout[r + 14u] = b2.p.x; cout[r + 15u] = b2.p.y;
}
);

static void on_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    *(int *)u1 = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

int gpu_bsdf_run(Gpu *g, const float *in, int n, float *out){
    if (!g || g->state != GPU_READY || n <= 0) return 0;
    char *src = malloc(strlen(GPU_WGSL_BSDF) + strlen(CHECK) + 64);
    if (!src) return 0;
    sprintf(src, "const GPU_BSDF_TAB_VEC4 = %du;\n", GPU_BSDF_TABLE_FLOATS / 4);   // (the tables' size)
    strcat(src, GPU_WGSL_BSDF);
    strcat(src, "\n");
    strcat(src, CHECK);
    WGPUBindGroupLayout bgl = gpu_layout(g, WGPUShaderStage_Compute, "URW");
    WGPUComputePipeline pipe = gpu_compute(g, "BSDF check", src, "main", bgl);
    free(src);
    float tab[GPU_BSDF_TABLE_FLOATS];
    gpu_bsdf_tables(tab);
    size_t bin = (size_t)n * GPU_BSDF_IN * 4, bout = (size_t)n * GPU_BSDF_OUT * 4;
    WGPUBuffer bt = gpu_buffer(g, "BSDF check: tables", sizeof(tab), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bi = gpu_buffer(g, "BSDF check: cases", bin, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bo = gpu_buffer(g, "BSDF check: answers", bout, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer rb = gpu_buffer(g, "BSDF check: readback", bout, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(g->queue, bt, 0, tab, sizeof(tab));
    wgpuQueueWriteBuffer(g->queue, bi, 0, in, bin);
    WGPUBuffer bufs[3] = {bt, bi, bo};
    WGPUBindGroup bg = gpu_bind(g, bgl, bufs, 3);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)((n + 63) / 64), 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(enc, bo, 0, rb, 0, bout);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    int mapped = 0;
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = on_mapped;
    mi.userdata1 = &mapped;
    wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, bout, mi);
    for (int k = 0; k < 4000 && mapped == 0; k++) gpu_wait(g);
    int ok = mapped == 1;
    if (ok){
        const float *r = wgpuBufferGetConstMappedRange(rb, 0, bout);
        if (r) memcpy(out, r, bout); else ok = 0;
        wgpuBufferUnmap(rb);
    }
    wgpuBindGroupRelease(bg);
    WGPUBuffer all[4] = {bt, bi, bo, rb};
    for (int i = 0; i < 4; i++){ wgpuBufferDestroy(all[i]); wgpuBufferRelease(all[i]); }
    wgpuComputePipelineRelease(pipe);
    wgpuBindGroupLayoutRelease(bgl);
    return ok;
}

#else
int gpu_bsdf_run(Gpu *g, const float *in, int n, float *out){ (void)g; (void)in; (void)n; (void)out; return 0; }
#endif
